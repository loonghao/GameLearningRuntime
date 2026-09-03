//! Versioned, learner-neutral runtime host core.
//!
//! The host owns framing, lifecycle, and episode/step fencing. Engine providers
//! own runtime authorization, main-thread dispatch, game semantics, and
//! authoritative post-state.

use std::collections::BTreeMap;
use std::io::{self, BufRead, Write};
use std::time::{SystemTime, UNIX_EPOCH};

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;
use uuid::Uuid;

pub const HOST_SCHEMA: &str = "glr.host.v1";
pub const DEFAULT_MAX_FRAME_BYTES: usize = 1_048_576;
pub const HARD_MAX_FRAME_BYTES: usize = 1_048_576;
pub const REALTIME_CONTROL_SCHEMA: &str = "glr.realtime-control.v1";
pub const RUNTIME_HEALTH_SCHEMA: &str = "glr.runtime-health.v1";

fn validate_runtime_identifier(value: &str, field: &str) -> Result<(), ProviderError> {
    let mut characters = value.chars();
    let first_is_valid = characters
        .next()
        .is_some_and(|character| character.is_ascii_alphanumeric());
    let rest_is_valid = characters.all(|character| {
        character.is_ascii_alphanumeric() || matches!(character, '_' | '.' | ':' | '-')
    });
    if value.len() > 128 || !first_is_valid || !rest_is_valid {
        return Err(ProviderError::InvalidData(format!(
            "{field} must be a bounded runtime identifier"
        )));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    Bool,
    Uint8,
    Int32,
    Int64,
    Float32,
    Float64,
}

impl DType {
    const fn item_size(self) -> usize {
        match self {
            Self::Bool | Self::Uint8 => 1,
            Self::Int32 | Self::Float32 => 4,
            Self::Int64 | Self::Float64 => 8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WireTensor {
    pub shape: Vec<u64>,
    pub dtype: DType,
    pub data: String,
}

impl WireTensor {
    pub fn scalar_i64(value: i64) -> Self {
        Self {
            shape: vec![1],
            dtype: DType::Int64,
            data: BASE64.encode(value.to_le_bytes()),
        }
    }

    pub fn scalar_f32(value: f32) -> Self {
        Self {
            shape: vec![1],
            dtype: DType::Float32,
            data: BASE64.encode(value.to_le_bytes()),
        }
    }

    pub fn bools(values: &[bool]) -> Self {
        Self {
            shape: vec![values.len() as u64],
            dtype: DType::Bool,
            data: BASE64.encode(
                values
                    .iter()
                    .map(|value| u8::from(*value))
                    .collect::<Vec<_>>(),
            ),
        }
    }

    pub fn validate(&self) -> Result<Vec<u8>, ProviderError> {
        let element_count = self.shape.iter().try_fold(1_u64, |total, dimension| {
            total.checked_mul(*dimension).ok_or_else(|| {
                ProviderError::InvalidData("tensor shape product overflows u64".into())
            })
        })?;
        let expected = usize::try_from(element_count)
            .ok()
            .and_then(|count| count.checked_mul(self.dtype.item_size()))
            .ok_or_else(|| {
                ProviderError::InvalidData("tensor byte length overflows usize".into())
            })?;
        let bytes = BASE64
            .decode(&self.data)
            .map_err(|_| ProviderError::InvalidData("tensor data is not valid base64".into()))?;
        if bytes.len() != expected {
            return Err(ProviderError::InvalidData(format!(
                "tensor has {} bytes; expected {expected}",
                bytes.len()
            )));
        }
        if self.dtype == DType::Bool && bytes.iter().any(|value| *value > 1) {
            return Err(ProviderError::InvalidData(
                "bool tensor contains a value other than 0 or 1".into(),
            ));
        }
        Ok(bytes)
    }

    pub fn scalar_i64_value(&self) -> Result<i64, ProviderError> {
        if self.dtype != DType::Int64 || self.shape != [1] {
            return Err(ProviderError::InvalidAction(
                "choice must be one int64 value".into(),
            ));
        }
        let bytes = self.validate()?;
        let array: [u8; 8] = bytes.try_into().map_err(|_| {
            ProviderError::InvalidAction("choice must contain exactly eight bytes".into())
        })?;
        Ok(i64::from_le_bytes(array))
    }

    fn bool_values(&self) -> Result<Vec<bool>, ProviderError> {
        if self.dtype != DType::Bool {
            return Err(ProviderError::InvalidData(
                "termination tensors must use bool dtype".into(),
            ));
        }
        Ok(self
            .validate()?
            .into_iter()
            .map(|value| value == 1)
            .collect())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpaceKind {
    Continuous,
    Discrete,
    MultiDiscrete,
    Binary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionOutcome {
    Accepted,
    Rejected,
    Unknown,
    NoEffect,
    Partial,
    Blocked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RefusalReasonClass {
    Transient,
    Structural,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RuntimeHealthStatus {
    Starting,
    Ready,
    Draining,
    Unhealthy,
    Stopped,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WireRuntimeIdentity {
    pub runtime_id: String,
    pub runtime_version: String,
}

impl WireRuntimeIdentity {
    fn validate(&self) -> Result<(), ProviderError> {
        validate_runtime_identifier(&self.runtime_id, "runtime_id")?;
        validate_runtime_identifier(&self.runtime_version, "runtime_version")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WireRuntimeLease {
    pub lease_id: String,
    pub owner_id: String,
    pub expires_at_ns: u64,
}

impl WireRuntimeLease {
    fn validate(&self, observed_at_ns: u64) -> Result<(), ProviderError> {
        validate_runtime_identifier(&self.lease_id, "runtime lease_id")?;
        validate_runtime_identifier(&self.owner_id, "runtime owner_id")?;
        if self.expires_at_ns <= observed_at_ns {
            return Err(ProviderError::InvalidData(
                "runtime lease expiry must be after observed_at_ns".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WireRuntimeHealth {
    pub schema_version: String,
    pub identity: WireRuntimeIdentity,
    pub status: RuntimeHealthStatus,
    pub observed_at_ns: u64,
    pub accepting_new_sessions: bool,
    pub active_sessions: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lease: Option<WireRuntimeLease>,
}

impl WireRuntimeHealth {
    fn validate(&self) -> Result<(), ProviderError> {
        if self.schema_version != RUNTIME_HEALTH_SCHEMA {
            return Err(ProviderError::InvalidData(
                "unsupported runtime health schema".into(),
            ));
        }
        self.identity.validate()?;
        if let Some(lease) = &self.lease {
            lease.validate(self.observed_at_ns)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WireTensorSpec {
    pub path: String,
    pub shape: Vec<i64>,
    pub dtype: DType,
    pub kind: SpaceKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minimum: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maximum: Option<f64>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WireEnvironmentDescriptor {
    pub environment_id: String,
    pub protocol_version: String,
    pub observations: Vec<WireTensorSpec>,
    pub actions: Vec<WireTensorSpec>,
    #[serde(default)]
    pub action_masks: Vec<WireTensorSpec>,
    pub reward: WireTensorSpec,
    pub done: WireTensorSpec,
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub realtime_timing: Option<WireRealtimeTimingContract>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_identity: Option<WireRuntimeIdentity>,
}

impl WireEnvironmentDescriptor {
    fn validate(&self) -> Result<(), ProviderError> {
        if let Some(identity) = &self.runtime_identity {
            identity.validate()?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WireRealtimeTimingContract {
    pub schema_version: String,
    pub minimum_hold_ns: u64,
    pub maximum_hold_ns: u64,
    pub settle_deadline_ns: u64,
    pub simulation_quantum_ns: u64,
    pub clock_source: String,
}

impl WireRealtimeTimingContract {
    fn validate(&self) -> Result<(), ProviderError> {
        if self.schema_version != REALTIME_CONTROL_SCHEMA {
            return Err(ProviderError::InvalidData(
                "unsupported realtime control schema".into(),
            ));
        }
        if self.minimum_hold_ns == 0
            || self.maximum_hold_ns == 0
            || self.settle_deadline_ns == 0
            || self.simulation_quantum_ns == 0
            || self.minimum_hold_ns > self.maximum_hold_ns
            || self.maximum_hold_ns > self.settle_deadline_ns
            || self.clock_source.is_empty()
        {
            return Err(ProviderError::InvalidData(
                "invalid realtime timing bounds".into(),
            ));
        }
        Ok(())
    }

    fn validate_step(
        &self,
        deadline_ns: u64,
        quantum_ns: u64,
        hold_ns: Option<u64>,
    ) -> Result<(), ProviderError> {
        if deadline_ns == 0
            || quantum_ns == 0
            || deadline_ns > self.settle_deadline_ns
            || quantum_ns > self.simulation_quantum_ns
            || hold_ns.is_some_and(|value| {
                value == 0 || value < self.minimum_hold_ns || value > self.maximum_hold_ns
            })
        {
            return Err(ProviderError::InvalidData(
                "realtime step timing is outside descriptor bounds".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WireEvent {
    pub name: String,
    pub timestamp_ns: u64,
    #[serde(default)]
    pub payload: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReconciliationOutcome {
    Applied,
    NotApplied,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WireActionReconciliation {
    pub episode_id: String,
    pub expected_step_id: u64,
    pub outcome: ReconciliationOutcome,
    pub authoritative_step_id: u64,
    pub timestamp_ns: u64,
    #[serde(default)]
    pub retryable: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WireResumeResult {
    pub timestep: WireTimeStep,
    pub committed_step_id: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reconciliation: Option<WireActionReconciliation>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WireActionReceipt {
    pub action_id: String,
    pub episode_id: String,
    pub step_id: u64,
    pub outcome: ActionOutcome,
    pub issued_timestamp_ns: u64,
    pub observed_timestamp_ns: u64,
    pub postcondition: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress_delta: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub authoritative_observation_sequence: Option<u64>,
    pub retryable: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub realtime: Option<WireRealtimeActionReceipt>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason_class: Option<RefusalReasonClass>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WireRealtimeActionReceipt {
    pub action_id: String,
    pub status: RealtimeActionStatus,
    pub deadline_ns: u64,
    pub quantum_ns: u64,
    pub issued_at_ns: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub consumed_at_ns: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub settled_at_ns: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cancellation_token: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RealtimeActionStatus {
    Consumed,
    Expired,
    Cancelled,
    Rejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InputLeaseOperation {
    Acquire,
    Renew,
    Release,
    Preempt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InputLeaseStatus {
    Acquired,
    Renewed,
    Released,
    Preempted,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WireInputLeaseToken {
    pub lease_id: String,
    pub session_id: String,
    pub target_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WireInputLeaseRequest {
    pub operation: InputLeaseOperation,
    pub session_id: String,
    pub target_id: String,
    #[serde(default)]
    pub lease_id: Option<String>,
    #[serde(default)]
    pub expires_at_ns: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WireInputLeaseReceipt {
    pub status: InputLeaseStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token: Option<WireInputLeaseToken>,
    pub observed_at_ns: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at_ns: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl WireActionReceipt {
    fn validate_against(&self, timestep: &WireTimeStep) -> Result<(), ProviderError> {
        Uuid::parse_str(&self.episode_id).map_err(|_| {
            ProviderError::InvalidData("action receipt episode_id must be a UUID".into())
        })?;
        if self.action_id.is_empty() || self.action_id.len() > 128 {
            return Err(ProviderError::InvalidData(
                "action receipt action_id must contain 1-128 characters".into(),
            ));
        }
        if self.step_id == 0 {
            return Err(ProviderError::InvalidData(
                "action receipt step_id must be positive".into(),
            ));
        }
        if self.observed_timestamp_ns < self.issued_timestamp_ns {
            return Err(ProviderError::InvalidData(
                "action receipt observed timestamp precedes issued timestamp".into(),
            ));
        }
        if self.postcondition.is_empty() || self.postcondition.len() > 128 {
            return Err(ProviderError::InvalidData(
                "action receipt postcondition must contain 1-128 characters".into(),
            ));
        }
        if self.progress_delta.is_some_and(|value| !value.is_finite()) {
            return Err(ProviderError::InvalidData(
                "action receipt progress_delta must be finite".into(),
            ));
        }
        if let Some(target_id) = &self.target_id
            && (target_id.is_empty() || target_id.len() > 128)
        {
            return Err(ProviderError::InvalidData(
                "action receipt target_id must contain 1-128 characters".into(),
            ));
        }
        if let Some(realtime) = &self.realtime
            && (realtime.action_id.is_empty()
                || realtime.deadline_ns == 0
                || realtime.quantum_ns == 0
                || realtime
                    .consumed_at_ns
                    .is_some_and(|value| value < realtime.issued_at_ns)
                || realtime
                    .settled_at_ns
                    .is_some_and(|value| value < realtime.issued_at_ns)
                || realtime.consumed_at_ns.is_some_and(|value| {
                    value > realtime.issued_at_ns.saturating_add(realtime.deadline_ns)
                })
                || realtime.settled_at_ns.is_some_and(|value| {
                    realtime
                        .consumed_at_ns
                        .is_some_and(|consumed| value < consumed)
                })
                || realtime.action_id != self.action_id)
        {
            return Err(ProviderError::InvalidData(
                "invalid realtime action receipt timing".into(),
            ));
        }
        if self.episode_id != timestep.episode_id || self.step_id != timestep.step_id {
            return Err(ProviderError::InvalidData(
                "action receipt does not match timestep identity".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WireTimeStep {
    pub episode_id: String,
    pub step_id: u64,
    pub timestamp_ns: u64,
    pub observation: BTreeMap<String, WireTensor>,
    pub reward: WireTensor,
    pub terminated: WireTensor,
    pub truncated: WireTensor,
    #[serde(default)]
    pub action_mask: BTreeMap<String, WireTensor>,
    #[serde(default)]
    pub events: Vec<WireEvent>,
    #[serde(default)]
    pub info: BTreeMap<String, Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action_receipt: Option<WireActionReceipt>,
}

impl WireTimeStep {
    fn validate(&self) -> Result<bool, ProviderError> {
        Uuid::parse_str(&self.episode_id)
            .map_err(|_| ProviderError::InvalidData("episode_id must be a UUID".into()))?;
        for tensor in self.observation.values() {
            tensor.validate()?;
        }
        for tensor in self.action_mask.values() {
            tensor.validate()?;
        }
        if let Some(receipt) = &self.action_receipt {
            receipt.validate_against(self)?;
        }
        self.reward.validate()?;
        let terminated = self.terminated.bool_values()?;
        let truncated = self.truncated.bool_values()?;
        if self.terminated.shape != self.truncated.shape {
            return Err(ProviderError::InvalidData(
                "terminated and truncated shapes differ".into(),
            ));
        }
        if terminated
            .iter()
            .zip(&truncated)
            .any(|(is_terminated, is_truncated)| *is_terminated && *is_truncated)
        {
            return Err(ProviderError::InvalidData(
                "one participant cannot be terminated and truncated simultaneously".into(),
            ));
        }
        Ok(terminated
            .iter()
            .zip(truncated)
            .all(|(is_terminated, is_truncated)| *is_terminated || is_truncated))
    }
}

impl WireResumeResult {
    fn validate(&self) -> Result<bool, ProviderError> {
        let done = self.timestep.validate()?;
        if self.committed_step_id != self.timestep.step_id {
            return Err(ProviderError::InvalidData(
                "resume committed_step_id does not match timestep.step_id".into(),
            ));
        }
        if let Some(reconciliation) = &self.reconciliation {
            Uuid::parse_str(&reconciliation.episode_id).map_err(|_| {
                ProviderError::InvalidData("resume reconciliation episode_id must be a UUID".into())
            })?;
            if reconciliation.expected_step_id == 0 {
                return Err(ProviderError::InvalidData(
                    "resume reconciliation expected_step_id must be positive".into(),
                ));
            }
            if reconciliation.episode_id != self.timestep.episode_id
                || reconciliation.authoritative_step_id != self.committed_step_id
            {
                return Err(ProviderError::InvalidData(
                    "resume reconciliation does not match authoritative cursor".into(),
                ));
            }
        }
        Ok(done)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderStepRequest {
    pub episode_id: String,
    pub expected_step_id: u64,
    pub action: BTreeMap<String, WireTensor>,
    pub action_id: Option<String>,
    pub issued_at_ns: Option<u64>,
    pub deadline_ns: Option<u64>,
    pub quantum_ns: Option<u64>,
    pub hold_ns: Option<u64>,
    pub lease: Option<WireInputLeaseToken>,
    pub cancellation_token: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderRefusal {
    pub action_id: Option<String>,
    pub target_id: String,
    pub reason_class: RefusalReasonClass,
    pub retryable: bool,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderResumeRequest {
    pub episode_id: String,
    pub last_committed_step_id: u64,
    pub target_id: Option<String>,
}

#[derive(Debug, Error)]
pub enum ProviderError {
    #[error("unsupported operation: {0}")]
    Unsupported(String),
    #[error("invalid action: {0}")]
    InvalidAction(String),
    #[error("command refused")]
    Refused { refusal: ProviderRefusal },
    #[error("invalid provider data: {0}")]
    InvalidData(String),
    #[error("runtime provider failure: {0}")]
    Runtime(String),
}

pub trait RuntimeProvider: Send {
    fn describe(&self) -> WireEnvironmentDescriptor;

    fn health(&self) -> Result<WireRuntimeHealth, ProviderError> {
        Err(ProviderError::Unsupported(
            "provider does not support runtime-health-v1".into(),
        ))
    }

    fn reset(
        &mut self,
        seed: Option<u64>,
        options: &BTreeMap<String, String>,
    ) -> Result<WireTimeStep, ProviderError>;

    fn attach(&mut self, options: &BTreeMap<String, String>)
    -> Result<WireTimeStep, ProviderError>;

    fn step(&mut self, request: ProviderStepRequest) -> Result<WireTimeStep, ProviderError>;

    fn resume(
        &mut self,
        _request: ProviderResumeRequest,
    ) -> Result<WireResumeResult, ProviderError> {
        Err(ProviderError::Unsupported(
            "provider does not support reconnect-resume-v1".into(),
        ))
    }

    fn lease(
        &mut self,
        _request: WireInputLeaseRequest,
    ) -> Result<WireInputLeaseReceipt, ProviderError> {
        Ok(WireInputLeaseReceipt {
            status: InputLeaseStatus::Rejected,
            token: None,
            observed_at_ns: now_ns(),
            expires_at_ns: None,
            reason: Some("provider does not support input leases".into()),
        })
    }

    fn close(&mut self) -> Result<(), ProviderError>;
}

#[derive(Debug, Error)]
pub enum HostConfigError {
    #[error("max_frame_bytes must be between 1 and {HARD_MAX_FRAME_BYTES}")]
    InvalidMaxFrameBytes,
}

#[derive(Debug, Clone)]
struct SessionCursor {
    episode_id: String,
    step_id: u64,
    done: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestEnvelope {
    schema: String,
    request_id: String,
    operation: String,
    payload: Value,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct EmptyPayload {}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ResetPayload {
    seed: Option<u64>,
    options: BTreeMap<String, String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct AttachPayload {
    options: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StepPayload {
    episode_id: String,
    expected_step_id: u64,
    action: BTreeMap<String, WireTensor>,
    #[serde(default)]
    action_id: Option<String>,
    #[serde(default)]
    issued_at_ns: Option<u64>,
    #[serde(default)]
    deadline_ns: Option<u64>,
    #[serde(default)]
    quantum_ns: Option<u64>,
    #[serde(default)]
    hold_ns: Option<u64>,
    #[serde(default)]
    lease: Option<WireInputLeaseToken>,
    #[serde(default)]
    cancellation_token: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CancelPayload {
    action_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResumePayload {
    episode_id: String,
    last_committed_step_id: u64,
    #[serde(default)]
    target_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct ResponseEnvelope {
    schema: &'static str,
    request_id: Option<String>,
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<ErrorBody>,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    code: &'static str,
    message: String,
    retryable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    refusal: Option<WireCommandRefusal>,
}

#[derive(Debug, Clone, Serialize)]
struct WireCommandRefusal {
    #[serde(skip_serializing_if = "Option::is_none")]
    action_id: Option<String>,
    target_id: String,
    reason_class: RefusalReasonClass,
    retryable: bool,
}

#[derive(Debug)]
struct HostFailure {
    code: &'static str,
    message: String,
    refusal: Option<WireCommandRefusal>,
}

impl HostFailure {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            refusal: None,
        }
    }

    fn refused(message: impl Into<String>, refusal: WireCommandRefusal) -> Self {
        Self {
            code: "command_refused",
            message: message.into(),
            refusal: Some(refusal),
        }
    }
}

pub struct Host {
    provider: Box<dyn RuntimeProvider>,
    current: Option<SessionCursor>,
    previous_episode_id: Option<String>,
    closed: bool,
    max_frame_bytes: usize,
    active_lease: Option<(WireInputLeaseToken, u64)>,
    cancelled_actions: Vec<String>,
    descriptor_identity: Option<WireRuntimeIdentity>,
}

impl Host {
    pub fn new(provider: Box<dyn RuntimeProvider>) -> Self {
        Self::with_max_frame_bytes(provider, DEFAULT_MAX_FRAME_BYTES)
            .expect("the default frame bound is valid")
    }

    pub fn with_max_frame_bytes(
        provider: Box<dyn RuntimeProvider>,
        max_frame_bytes: usize,
    ) -> Result<Self, HostConfigError> {
        if max_frame_bytes == 0 || max_frame_bytes > HARD_MAX_FRAME_BYTES {
            return Err(HostConfigError::InvalidMaxFrameBytes);
        }
        Ok(Self {
            provider,
            current: None,
            previous_episode_id: None,
            closed: false,
            max_frame_bytes,
            active_lease: None,
            cancelled_actions: Vec::new(),
            descriptor_identity: None,
        })
    }

    pub const fn max_frame_bytes(&self) -> usize {
        self.max_frame_bytes
    }

    pub const fn is_closed(&self) -> bool {
        self.closed
    }

    pub fn handle_frame(&mut self, frame: &[u8]) -> Vec<u8> {
        if frame.len() > self.max_frame_bytes {
            return self.error_response(
                None,
                HostFailure::new(
                    "frame_too_large",
                    format!(
                        "request frame has {} bytes; maximum is {}",
                        frame.len(),
                        self.max_frame_bytes
                    ),
                ),
            );
        }
        let request = match serde_json::from_slice::<RequestEnvelope>(frame) {
            Ok(request) => request,
            Err(error) => {
                return self.error_response(
                    None,
                    HostFailure::new("invalid_request", format!("invalid request JSON: {error}")),
                );
            }
        };
        let request_id = Some(request.request_id.clone());
        let result = self.dispatch(request);
        match result {
            Ok(value) => serialize_response(ResponseEnvelope {
                schema: HOST_SCHEMA,
                request_id,
                ok: true,
                result: Some(value),
                error: None,
            }),
            Err(error) => self.error_response(request_id, error),
        }
    }

    pub fn frame_too_large_response(&self) -> Vec<u8> {
        self.error_response(
            None,
            HostFailure::new(
                "frame_too_large",
                format!("request frame exceeds {} bytes", self.max_frame_bytes),
            ),
        )
    }

    fn dispatch(&mut self, request: RequestEnvelope) -> Result<Value, HostFailure> {
        if request.schema != HOST_SCHEMA {
            return Err(HostFailure::new(
                "schema_mismatch",
                format!("expected schema {HOST_SCHEMA}; received {}", request.schema),
            ));
        }
        validate_request_id(&request.request_id)?;
        if self.closed {
            return Err(HostFailure::new("host_closed", "runtime host is closed"));
        }

        match request.operation.as_str() {
            "describe" => {
                parse_payload::<EmptyPayload>(request.payload)?;
                let mut descriptor = self.provider.describe();
                descriptor.validate().map_err(provider_failure)?;
                if let Some(timing) = &descriptor.realtime_timing {
                    timing.validate().map_err(provider_failure)?;
                }
                descriptor.capabilities.push("host-stdio".into());
                descriptor.capabilities.sort();
                descriptor.capabilities.dedup();
                self.descriptor_identity = descriptor.runtime_identity.clone();
                serde_json::to_value(descriptor).map_err(internal_serialization_failure)
            }
            "health" => {
                parse_payload::<EmptyPayload>(request.payload)?;
                let health = self.provider.health().map_err(provider_failure)?;
                health.validate().map_err(provider_failure)?;
                if let Some(identity) = &self.descriptor_identity
                    && identity != &health.identity
                {
                    return Err(HostFailure::new(
                        "provider_contract_violation",
                        "runtime health identity does not match descriptor",
                    ));
                }
                serde_json::to_value(health).map_err(internal_serialization_failure)
            }
            "reset" => {
                let payload = parse_payload::<ResetPayload>(request.payload)?;
                let timestep = self
                    .provider
                    .reset(payload.seed, &payload.options)
                    .map_err(provider_failure)?;
                self.accept_start(&timestep, "reset")?;
                serde_json::to_value(timestep).map_err(internal_serialization_failure)
            }
            "attach" => {
                let payload = parse_payload::<AttachPayload>(request.payload)?;
                let timestep = self
                    .provider
                    .attach(&payload.options)
                    .map_err(provider_failure)?;
                self.accept_start(&timestep, "attach")?;
                serde_json::to_value(timestep).map_err(internal_serialization_failure)
            }
            "step" => {
                let payload = parse_payload::<StepPayload>(request.payload)?;
                let timestep = self.step(payload)?;
                serde_json::to_value(timestep).map_err(internal_serialization_failure)
            }
            "resume" => {
                let payload = parse_payload::<ResumePayload>(request.payload)?;
                let result = self.resume(payload)?;
                serde_json::to_value(result).map_err(internal_serialization_failure)
            }

            "lease" => {
                let payload = parse_payload::<WireInputLeaseRequest>(request.payload)?;
                self.lease(payload)
            }
            "cancel" => {
                let payload = parse_payload::<CancelPayload>(request.payload)?;
                validate_action_id(&payload.action_id)?;
                if !self.cancelled_actions.contains(&payload.action_id) {
                    self.cancelled_actions.push(payload.action_id.clone());
                }
                Ok(json!({"cancelled": true, "action_id": payload.action_id}))
            }
            "close" => {
                parse_payload::<EmptyPayload>(request.payload)?;
                self.provider.close().map_err(provider_failure)?;
                self.current = None;
                self.active_lease = None;
                self.cancelled_actions.clear();
                self.descriptor_identity = None;
                self.closed = true;
                Ok(json!({"closed": true}))
            }
            _ => Err(HostFailure::new(
                "unknown_operation",
                format!("unknown operation: {}", request.operation),
            )),
        }
    }

    fn accept_start(
        &mut self,
        timestep: &WireTimeStep,
        operation: &str,
    ) -> Result<(), HostFailure> {
        let done = timestep.validate().map_err(provider_failure)?;
        if timestep.step_id != 0 {
            return Err(HostFailure::new(
                "provider_contract_violation",
                format!("{operation} returned step {}; expected 0", timestep.step_id),
            ));
        }
        if done {
            return Err(HostFailure::new(
                "provider_contract_violation",
                format!("{operation} returned a terminal time step"),
            ));
        }
        if self.previous_episode_id.as_deref() == Some(timestep.episode_id.as_str()) {
            return Err(HostFailure::new(
                "provider_contract_violation",
                format!("{operation} reused the previous episode_id"),
            ));
        }
        self.previous_episode_id = Some(timestep.episode_id.clone());
        self.active_lease = None;
        self.cancelled_actions.clear();
        self.current = Some(SessionCursor {
            episode_id: timestep.episode_id.clone(),
            step_id: 0,
            done: false,
        });
        Ok(())
    }

    fn step(&mut self, payload: StepPayload) -> Result<WireTimeStep, HostFailure> {
        let current = self.current.as_ref().ok_or_else(|| {
            HostFailure::new("lifecycle_violation", "step requires reset or attach first")
        })?;
        if current.done {
            return Err(HostFailure::new(
                "lifecycle_violation",
                "terminal episode requires a new reset or attach",
            ));
        }
        if payload.episode_id != current.episode_id {
            return Err(HostFailure::new(
                "lifecycle_violation",
                "step episode_id does not match the active episode",
            ));
        }
        let expected = current
            .step_id
            .checked_add(1)
            .ok_or_else(|| HostFailure::new("lifecycle_violation", "step_id overflow"))?;
        if payload.expected_step_id != expected {
            return Err(HostFailure::new(
                "lifecycle_violation",
                format!(
                    "expected_step_id must be {expected}; received {}",
                    payload.expected_step_id
                ),
            ));
        }
        if let Some(action_id) = payload.action_id.as_deref() {
            validate_action_id(action_id)?;
            if self.cancelled_actions.iter().any(|item| item == action_id) {
                return Err(HostFailure::new(
                    "action_cancelled",
                    "realtime action was cancelled before provider dispatch",
                ));
            }
        }
        if payload.deadline_ns.is_some() != payload.quantum_ns.is_some() {
            return Err(HostFailure::new(
                "invalid_request",
                "deadline_ns and quantum_ns must be provided together",
            ));
        }
        if payload.hold_ns.is_some() && payload.deadline_ns.is_none() {
            return Err(HostFailure::new(
                "invalid_request",
                "hold_ns requires deadline_ns and quantum_ns",
            ));
        }
        if let Some(deadline_ns) = payload.deadline_ns {
            let quantum_ns = payload.quantum_ns.expect("presence checked above");
            let issued_at_ns = payload.issued_at_ns.ok_or_else(|| {
                HostFailure::new(
                    "invalid_request",
                    "issued_at_ns is required for a bounded realtime step",
                )
            })?;
            if timestamp_ns() >= issued_at_ns.saturating_add(deadline_ns) {
                return Err(HostFailure::new(
                    "action_expired",
                    "realtime action deadline expired before provider dispatch",
                ));
            }
            let descriptor = self.provider.describe();
            let timing = descriptor.realtime_timing.ok_or_else(|| {
                HostFailure::new(
                    "unsupported_operation",
                    "provider does not advertise realtime timing",
                )
            })?;
            timing
                .validate_step(deadline_ns, quantum_ns, payload.hold_ns)
                .map_err(provider_failure)?;
        }
        if let Some(lease) = payload.lease.as_ref() {
            let now = timestamp_ns();
            let authorized = self
                .active_lease
                .as_ref()
                .is_some_and(|(active, expires)| active == lease && now < *expires);
            if !authorized {
                return Err(HostFailure::new(
                    "lease_violation",
                    "realtime step lease is absent, stale, or bound to another target/session",
                ));
            }
        }
        for tensor in payload.action.values() {
            tensor.validate().map_err(provider_failure)?;
        }

        let timestep = match self.provider.step(ProviderStepRequest {
            episode_id: payload.episode_id,
            expected_step_id: payload.expected_step_id,
            action: payload.action,
            action_id: payload.action_id,
            issued_at_ns: payload.issued_at_ns,
            deadline_ns: payload.deadline_ns,
            quantum_ns: payload.quantum_ns,
            hold_ns: payload.hold_ns,
            lease: payload.lease,
            cancellation_token: payload.cancellation_token,
        }) {
            Ok(timestep) => timestep,
            Err(ProviderError::Refused { refusal }) => {
                return Err(HostFailure::refused(
                    refusal.message.clone(),
                    WireCommandRefusal {
                        action_id: refusal.action_id,
                        target_id: refusal.target_id,
                        reason_class: refusal.reason_class,
                        retryable: refusal.retryable,
                    },
                ));
            }
            Err(error) => return Err(provider_failure(error)),
        };
        let done = timestep.validate().map_err(provider_failure)?;
        if timestep.episode_id != current.episode_id || timestep.step_id != expected {
            return Err(HostFailure::new(
                "provider_contract_violation",
                "provider returned a stale episode_id or step_id",
            ));
        }
        self.current = Some(SessionCursor {
            episode_id: timestep.episode_id.clone(),
            step_id: timestep.step_id,
            done,
        });
        Ok(timestep)
    }

    fn resume(&mut self, payload: ResumePayload) -> Result<WireResumeResult, HostFailure> {
        let active = self.current.clone();
        if let Some(current) = &active {
            if payload.episode_id != current.episode_id {
                return Err(HostFailure::new(
                    "lifecycle_violation",
                    "resume episode_id does not match the active episode",
                ));
            }
            if payload.last_committed_step_id > current.step_id {
                return Err(HostFailure::new(
                    "lifecycle_violation",
                    "resume cursor is ahead of the active authoritative cursor",
                ));
            }
        }
        let result = self
            .provider
            .resume(ProviderResumeRequest {
                episode_id: payload.episode_id.clone(),
                last_committed_step_id: payload.last_committed_step_id,
                target_id: payload.target_id,
            })
            .map_err(provider_failure)?;
        let done = result.validate().map_err(provider_failure)?;
        if result.timestep.episode_id != payload.episode_id
            || result.committed_step_id < payload.last_committed_step_id
            || active
                .as_ref()
                .is_some_and(|current| result.committed_step_id < current.step_id)
        {
            return Err(HostFailure::new(
                "provider_contract_violation",
                "resume returned a stale episode or cursor",
            ));
        }
        self.current = Some(SessionCursor {
            episode_id: result.timestep.episode_id.clone(),
            step_id: result.committed_step_id,
            done,
        });
        if active.is_none() {
            self.previous_episode_id = Some(result.timestep.episode_id.clone());
        }
        Ok(result)
    }

    fn lease(&mut self, payload: WireInputLeaseRequest) -> Result<Value, HostFailure> {
        let now = timestamp_ns();
        if payload.session_id.is_empty() || payload.target_id.is_empty() {
            return Err(HostFailure::new(
                "invalid_request",
                "lease session_id and target_id are required",
            ));
        }
        match payload.operation {
            InputLeaseOperation::Acquire => {
                if self
                    .active_lease
                    .as_ref()
                    .is_some_and(|(_, expires)| now < *expires)
                {
                    return serde_json::to_value(WireInputLeaseReceipt {
                        status: InputLeaseStatus::Rejected,
                        token: self.active_lease.as_ref().map(|(token, _)| token.clone()),
                        observed_at_ns: now,
                        expires_at_ns: self.active_lease.as_ref().map(|(_, expires)| *expires),
                        reason: Some("lease already held".into()),
                    })
                    .map_err(internal_serialization_failure);
                }
                let expires = payload
                    .expires_at_ns
                    .unwrap_or_else(|| now.saturating_add(1));
                if expires <= now {
                    return Err(HostFailure::new(
                        "invalid_request",
                        "lease expiry must be in the future",
                    ));
                }
                let token = WireInputLeaseToken {
                    lease_id: format!("{}.lease", payload.session_id),
                    session_id: payload.session_id,
                    target_id: payload.target_id,
                };
                self.active_lease = Some((token.clone(), expires));
                serde_json::to_value(WireInputLeaseReceipt {
                    status: InputLeaseStatus::Acquired,
                    token: Some(token),
                    observed_at_ns: now,
                    expires_at_ns: Some(expires),
                    reason: None,
                })
                .map_err(internal_serialization_failure)
            }
            operation => {
                let Some((token, current_expiry)) = self.active_lease.clone() else {
                    return serde_json::to_value(WireInputLeaseReceipt {
                        status: InputLeaseStatus::Rejected,
                        token: None,
                        observed_at_ns: now,
                        expires_at_ns: None,
                        reason: Some("lease is absent or expired".into()),
                    })
                    .map_err(internal_serialization_failure);
                };
                if now >= current_expiry
                    || payload.lease_id.as_deref() != Some(token.lease_id.as_str())
                    || payload.session_id != token.session_id
                    || payload.target_id != token.target_id
                {
                    return serde_json::to_value(WireInputLeaseReceipt {
                        status: InputLeaseStatus::Rejected,
                        token: Some(token),
                        observed_at_ns: now,
                        expires_at_ns: Some(current_expiry),
                        reason: Some("lease identity mismatch or expiry".into()),
                    })
                    .map_err(internal_serialization_failure);
                }
                if operation == InputLeaseOperation::Renew {
                    let expires = payload.expires_at_ns.unwrap_or(current_expiry);
                    if expires <= now {
                        return Err(HostFailure::new(
                            "invalid_request",
                            "lease renewal must be in the future",
                        ));
                    }
                    self.active_lease = Some((token.clone(), expires));
                    return serde_json::to_value(WireInputLeaseReceipt {
                        status: InputLeaseStatus::Renewed,
                        token: Some(token),
                        observed_at_ns: now,
                        expires_at_ns: Some(expires),
                        reason: None,
                    })
                    .map_err(internal_serialization_failure);
                }
                self.active_lease = None;
                let status = if operation == InputLeaseOperation::Preempt {
                    InputLeaseStatus::Preempted
                } else {
                    InputLeaseStatus::Released
                };
                serde_json::to_value(WireInputLeaseReceipt {
                    status,
                    token: Some(token),
                    observed_at_ns: now,
                    expires_at_ns: None,
                    reason: None,
                })
                .map_err(internal_serialization_failure)
            }
        }
    }

    fn error_response(&self, request_id: Option<String>, error: HostFailure) -> Vec<u8> {
        serialize_response(ResponseEnvelope {
            schema: HOST_SCHEMA,
            request_id,
            ok: false,
            result: None,
            error: Some(ErrorBody {
                code: error.code,
                message: error.message,
                retryable: error
                    .refusal
                    .as_ref()
                    .is_some_and(|refusal| refusal.retryable),
                refusal: error.refusal,
            }),
        })
    }
}

fn parse_payload<T>(payload: Value) -> Result<T, HostFailure>
where
    T: for<'de> Deserialize<'de>,
{
    serde_json::from_value(payload).map_err(|error| {
        HostFailure::new(
            "invalid_request",
            format!("invalid operation payload: {error}"),
        )
    })
}

fn validate_request_id(request_id: &str) -> Result<(), HostFailure> {
    if request_id.is_empty()
        || request_id.len() > 128
        || !request_id
            .bytes()
            .all(|value| value.is_ascii_alphanumeric() || b"-_.:".contains(&value))
    {
        return Err(HostFailure::new(
            "invalid_request",
            "request_id must contain 1-128 ASCII letters, digits, '-', '_', '.', or ':'",
        ));
    }
    Ok(())
}

fn validate_action_id(action_id: &str) -> Result<(), HostFailure> {
    if action_id.is_empty()
        || action_id.len() > 128
        || !action_id
            .bytes()
            .all(|value| value.is_ascii_alphanumeric() || b"-_.:".contains(&value))
    {
        return Err(HostFailure::new(
            "invalid_request",
            "action_id must contain 1-128 ASCII letters, digits, '-', '_', '.', or ':'",
        ));
    }
    Ok(())
}

fn provider_failure(error: ProviderError) -> HostFailure {
    match error {
        ProviderError::Unsupported(message) => HostFailure::new("unsupported_operation", message),
        ProviderError::InvalidAction(message) => HostFailure::new("provider_rejected", message),
        ProviderError::InvalidData(message) => {
            HostFailure::new("provider_contract_violation", message)
        }
        ProviderError::Runtime(message) => HostFailure::new("provider_error", message),
        ProviderError::Refused { refusal } => HostFailure::refused(
            refusal.message.clone(),
            WireCommandRefusal {
                action_id: refusal.action_id,
                target_id: refusal.target_id,
                reason_class: refusal.reason_class,
                retryable: refusal.retryable,
            },
        ),
    }
}

fn internal_serialization_failure(error: serde_json::Error) -> HostFailure {
    HostFailure::new(
        "internal_error",
        format!("failed to serialize host response: {error}"),
    )
}

fn serialize_response(response: ResponseEnvelope) -> Vec<u8> {
    serde_json::to_vec(&response).unwrap_or_else(|_| {
        br#"{"schema":"glr.host.v1","request_id":null,"ok":false,"error":{"code":"internal_error","message":"response serialization failed","retryable":false}}"#.to_vec()
    })
}

pub struct SyntheticCounterProvider {
    target: i64,
    value: i64,
    episode_id: Option<String>,
}

impl SyntheticCounterProvider {
    pub fn new(target: i64) -> Self {
        Self {
            target: target.max(1),
            value: 0,
            episode_id: None,
        }
    }

    fn timestep(&self, reward: f32, terminated: bool, step_id: u64) -> WireTimeStep {
        let mut observation = BTreeMap::new();
        observation.insert("value".into(), WireTensor::scalar_i64(self.value));
        let mut action_mask = BTreeMap::new();
        action_mask.insert("choice".into(), WireTensor::bools(&[true, true]));
        WireTimeStep {
            episode_id: self
                .episode_id
                .clone()
                .expect("synthetic provider requires an active episode"),
            step_id,
            timestamp_ns: timestamp_ns(),
            observation,
            reward: WireTensor::scalar_f32(reward),
            terminated: WireTensor::bools(&[terminated]),
            truncated: WireTensor::bools(&[false]),
            action_mask,
            events: Vec::new(),
            info: BTreeMap::new(),
            action_receipt: None,
        }
    }
}

impl RuntimeProvider for SyntheticCounterProvider {
    fn describe(&self) -> WireEnvironmentDescriptor {
        WireEnvironmentDescriptor {
            environment_id: "glr.synthetic.counter-v1".into(),
            protocol_version: "1.0".into(),
            observations: vec![WireTensorSpec {
                path: "value".into(),
                shape: vec![1],
                dtype: DType::Int64,
                kind: SpaceKind::Discrete,
                minimum: Some(0.0),
                maximum: Some(self.target as f64),
                description: "Synthetic counter value".into(),
            }],
            actions: vec![WireTensorSpec {
                path: "choice".into(),
                shape: vec![1],
                dtype: DType::Int64,
                kind: SpaceKind::Discrete,
                minimum: Some(0.0),
                maximum: Some(1.0),
                description: "Zero holds; one increments".into(),
            }],
            action_masks: vec![WireTensorSpec {
                path: "choice".into(),
                shape: vec![2],
                dtype: DType::Bool,
                kind: SpaceKind::Binary,
                minimum: None,
                maximum: None,
                description: "Available discrete choices".into(),
            }],
            reward: WireTensorSpec {
                path: "reward".into(),
                shape: vec![1],
                dtype: DType::Float32,
                kind: SpaceKind::Continuous,
                minimum: Some(0.0),
                maximum: Some(1.0),
                description: "One on reaching the target".into(),
            },
            done: WireTensorSpec {
                path: "done".into(),
                shape: vec![1],
                dtype: DType::Bool,
                kind: SpaceKind::Binary,
                minimum: None,
                maximum: None,
                description: "Synthetic episode completion".into(),
            },
            capabilities: vec!["reset".into(), "step".into(), "synthetic-provider".into()],
            metadata: BTreeMap::new(),
            realtime_timing: None,
            runtime_identity: Some(WireRuntimeIdentity {
                runtime_id: "synthetic-counter".into(),
                runtime_version: env!("CARGO_PKG_VERSION").into(),
            }),
        }
    }

    fn health(&self) -> Result<WireRuntimeHealth, ProviderError> {
        Ok(WireRuntimeHealth {
            schema_version: RUNTIME_HEALTH_SCHEMA.into(),
            identity: WireRuntimeIdentity {
                runtime_id: "synthetic-counter".into(),
                runtime_version: env!("CARGO_PKG_VERSION").into(),
            },
            status: RuntimeHealthStatus::Ready,
            observed_at_ns: now_ns(),
            accepting_new_sessions: true,
            active_sessions: if self.episode_id.is_some() { 1 } else { 0 },
            lease: None,
        })
    }

    fn reset(
        &mut self,
        _seed: Option<u64>,
        _options: &BTreeMap<String, String>,
    ) -> Result<WireTimeStep, ProviderError> {
        self.value = 0;
        self.episode_id = Some(Uuid::new_v4().to_string());
        Ok(self.timestep(0.0, false, 0))
    }

    fn attach(
        &mut self,
        _options: &BTreeMap<String, String>,
    ) -> Result<WireTimeStep, ProviderError> {
        Err(ProviderError::Unsupported(
            "synthetic-counter does not support live attach".into(),
        ))
    }

    fn step(&mut self, request: ProviderStepRequest) -> Result<WireTimeStep, ProviderError> {
        if self.episode_id.as_deref() != Some(request.episode_id.as_str()) {
            return Err(ProviderError::InvalidData(
                "provider received a stale episode".into(),
            ));
        }
        if request.action.len() != 1 {
            return Err(ProviderError::InvalidAction(
                "synthetic action requires exactly one choice tensor".into(),
            ));
        }
        let choice = request
            .action
            .get("choice")
            .ok_or_else(|| ProviderError::InvalidAction("missing choice tensor".into()))?
            .scalar_i64_value()?;
        match choice {
            0 => {}
            1 => self.value = (self.value + 1).min(self.target),
            _ => {
                return Err(ProviderError::InvalidAction(
                    "choice must be either zero or one".into(),
                ));
            }
        }
        let terminated = self.value >= self.target;
        Ok(self.timestep(
            if terminated { 1.0 } else { 0.0 },
            terminated,
            request.expected_step_id,
        ))
    }

    fn close(&mut self) -> Result<(), ProviderError> {
        self.episode_id = None;
        Ok(())
    }
}

fn timestamp_ns() -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    u64::try_from(nanos).unwrap_or(u64::MAX)
}

fn now_ns() -> u64 {
    timestamp_ns()
}

enum FrameRead {
    Eof,
    Frame(Vec<u8>),
    Oversized,
}

fn read_bounded_frame<R: BufRead>(reader: &mut R, max_frame_bytes: usize) -> io::Result<FrameRead> {
    let mut frame = Vec::new();
    let mut oversized = false;
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            if frame.is_empty() && !oversized {
                return Ok(FrameRead::Eof);
            }
            return Ok(if oversized {
                FrameRead::Oversized
            } else {
                FrameRead::Frame(frame)
            });
        }
        let newline = available.iter().position(|value| *value == b'\n');
        let consumed = newline.map_or(available.len(), |position| position + 1);
        let content_len = newline.unwrap_or(available.len());
        if !oversized {
            if frame.len().saturating_add(content_len) > max_frame_bytes {
                oversized = true;
                frame.clear();
            } else {
                frame.extend_from_slice(&available[..content_len]);
            }
        }
        reader.consume(consumed);
        if newline.is_some() {
            if frame.last() == Some(&b'\r') {
                frame.pop();
            }
            return Ok(if oversized {
                FrameRead::Oversized
            } else {
                FrameRead::Frame(frame)
            });
        }
    }
}

pub fn serve_json_lines<R: BufRead, W: Write>(
    host: &mut Host,
    reader: &mut R,
    writer: &mut W,
) -> io::Result<()> {
    loop {
        let response = match read_bounded_frame(reader, host.max_frame_bytes())? {
            FrameRead::Eof => break,
            FrameRead::Frame(frame) => host.handle_frame(&frame),
            FrameRead::Oversized => host.frame_too_large_response(),
        };
        writer.write_all(&response)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        if host.is_closed() {
            break;
        }
    }
    Ok(())
}
