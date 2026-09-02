use glr_host::{
    HOST_SCHEMA, Host, ProviderError, ProviderResumeRequest, ProviderStepRequest,
    ReconciliationOutcome, RuntimeProvider, SyntheticCounterProvider, WireActionReconciliation,
    WireEnvironmentDescriptor, WireResumeResult, WireTimeStep,
};
use serde_json::{Value, json};

fn request(request_id: &str, operation: &str, payload: Value) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "schema": HOST_SCHEMA,
        "request_id": request_id,
        "operation": operation,
        "payload": payload,
    }))
    .expect("request should serialize")
}

fn send(host: &mut Host, request_id: &str, operation: &str, payload: Value) -> Value {
    let response = host.handle_frame(&request(request_id, operation, payload));
    serde_json::from_slice(&response).expect("host response should be JSON")
}

#[test]
fn describe_exposes_only_truthful_stdio_and_synthetic_capabilities() {
    let mut host = Host::new(Box::new(SyntheticCounterProvider::new(2)));

    let response = send(&mut host, "describe-1", "describe", json!({}));

    assert_eq!(response["schema"], HOST_SCHEMA);
    assert_eq!(response["request_id"], "describe-1");
    assert_eq!(response["ok"], true);
    assert_eq!(
        response["result"]["environment_id"],
        "glr.synthetic.counter-v1"
    );
    let capabilities = response["result"]["capabilities"]
        .as_array()
        .expect("capabilities should be an array");
    assert!(capabilities.contains(&json!("host-stdio")));
    assert!(capabilities.contains(&json!("reset")));
    assert!(capabilities.contains(&json!("step")));
    assert!(!capabilities.contains(&json!("authenticated")));
    assert!(!capabilities.contains(&json!("target-bound")));
}

#[test]
fn reset_and_step_are_fenced_by_episode_and_expected_step() {
    let mut host = Host::new(Box::new(SyntheticCounterProvider::new(2)));
    let reset = send(
        &mut host,
        "reset-1",
        "reset",
        json!({"seed": 7, "options": {}}),
    );
    let episode_id = reset["result"]["episode_id"]
        .as_str()
        .expect("reset should return an episode id");

    let stale = send(
        &mut host,
        "step-stale",
        "step",
        json!({
            "episode_id": episode_id,
            "expected_step_id": 2,
            "action": {
                "choice": {"shape": [1], "dtype": "int64", "data": "AQAAAAAAAAA="}
            }
        }),
    );
    assert_eq!(stale["ok"], false);
    assert_eq!(stale["error"]["code"], "lifecycle_violation");

    let accepted = send(
        &mut host,
        "step-1",
        "step",
        json!({
            "episode_id": episode_id,
            "expected_step_id": 1,
            "action": {
                "choice": {"shape": [1], "dtype": "int64", "data": "AQAAAAAAAAA="}
            }
        }),
    );
    assert_eq!(accepted["ok"], true);
    assert_eq!(accepted["result"]["step_id"], 1);
}

#[test]
fn terminal_episode_rejects_more_actions() {
    let mut host = Host::new(Box::new(SyntheticCounterProvider::new(1)));
    let reset = send(&mut host, "reset-1", "reset", json!({}));
    let episode_id = reset["result"]["episode_id"]
        .as_str()
        .expect("reset should return an episode id");
    let action = json!({
        "episode_id": episode_id,
        "expected_step_id": 1,
        "action": {
            "choice": {"shape": [1], "dtype": "int64", "data": "AQAAAAAAAAA="}
        }
    });

    let terminal = send(&mut host, "step-1", "step", action);
    assert_eq!(terminal["result"]["terminated"]["data"], "AQ==");

    let after_terminal = send(
        &mut host,
        "step-2",
        "step",
        json!({
            "episode_id": episode_id,
            "expected_step_id": 2,
            "action": {
                "choice": {"shape": [1], "dtype": "int64", "data": "AQAAAAAAAAA="}
            }
        }),
    );
    assert_eq!(after_terminal["ok"], false);
    assert_eq!(after_terminal["error"]["code"], "lifecycle_violation");
}

#[test]
fn malformed_unknown_and_oversized_requests_fail_closed() {
    let mut host = Host::with_max_frame_bytes(Box::new(SyntheticCounterProvider::new(2)), 256)
        .expect("frame bound should be valid");

    let unknown = send(&mut host, "unknown-1", "execute", json!({}));
    assert_eq!(unknown["ok"], false);
    assert_eq!(unknown["error"]["code"], "unknown_operation");

    let malformed: Value = serde_json::from_slice(&host.handle_frame(b"not-json"))
        .expect("malformed response should still be JSON");
    assert_eq!(malformed["ok"], false);
    assert_eq!(malformed["error"]["code"], "invalid_request");

    let oversized: Value = serde_json::from_slice(&host.handle_frame(&vec![b'x'; 257]))
        .expect("oversized response should still be JSON");
    assert_eq!(oversized["ok"], false);
    assert_eq!(oversized["error"]["code"], "frame_too_large");
}

#[test]
fn unsupported_attach_and_post_close_work_are_rejected() {
    let mut host = Host::new(Box::new(SyntheticCounterProvider::new(2)));

    let attach = send(&mut host, "attach-1", "attach", json!({"options": {}}));
    assert_eq!(attach["ok"], false);
    assert_eq!(attach["error"]["code"], "unsupported_operation");

    let closed = send(&mut host, "close-1", "close", json!({}));
    assert_eq!(closed["ok"], true);

    let after_close = send(&mut host, "describe-2", "describe", json!({}));
    assert_eq!(after_close["ok"], false);
    assert_eq!(after_close["error"]["code"], "host_closed");
}

struct ResumableProvider {
    inner: SyntheticCounterProvider,
    current: Option<WireTimeStep>,
}

impl ResumableProvider {
    fn new() -> Self {
        Self {
            inner: SyntheticCounterProvider::new(2),
            current: None,
        }
    }
}

impl RuntimeProvider for ResumableProvider {
    fn describe(&self) -> WireEnvironmentDescriptor {
        let mut descriptor = self.inner.describe();
        descriptor.capabilities.push("reconnect-resume-v1".into());
        descriptor
    }

    fn reset(
        &mut self,
        seed: Option<u64>,
        options: &std::collections::BTreeMap<String, String>,
    ) -> Result<WireTimeStep, ProviderError> {
        let timestep = self.inner.reset(seed, options)?;
        self.current = Some(timestep.clone());
        Ok(timestep)
    }

    fn attach(
        &mut self,
        options: &std::collections::BTreeMap<String, String>,
    ) -> Result<WireTimeStep, ProviderError> {
        let timestep = self.inner.attach(options)?;
        self.current = Some(timestep.clone());
        Ok(timestep)
    }

    fn step(&mut self, request: ProviderStepRequest) -> Result<WireTimeStep, ProviderError> {
        let timestep = self.inner.step(request)?;
        self.current = Some(timestep.clone());
        Ok(timestep)
    }

    fn resume(
        &mut self,
        request: ProviderResumeRequest,
    ) -> Result<WireResumeResult, ProviderError> {
        let timestep = self
            .current
            .clone()
            .ok_or_else(|| ProviderError::Runtime("resume requires an active episode".into()))?;
        Ok(WireResumeResult {
            committed_step_id: timestep.step_id,
            reconciliation: Some(WireActionReconciliation {
                episode_id: request.episode_id,
                expected_step_id: request.last_committed_step_id + 1,
                outcome: ReconciliationOutcome::Unknown,
                authoritative_step_id: timestep.step_id,
                timestamp_ns: 42,
                retryable: false,
            }),
            timestep,
        })
    }

    fn close(&mut self) -> Result<(), ProviderError> {
        self.inner.close()
    }
}

#[test]
fn resumable_provider_returns_authoritative_cursor_and_reconciliation() {
    let mut host = Host::new(Box::new(ResumableProvider::new()));
    let reset = send(&mut host, "reset-1", "reset", json!({}));
    let episode_id = reset["result"]["episode_id"]
        .as_str()
        .expect("reset should return an episode id");

    let resumed = send(
        &mut host,
        "resume-1",
        "resume",
        json!({
            "episode_id": episode_id,
            "last_committed_step_id": 0,
            "target_id": "runtime-1"
        }),
    );

    assert_eq!(resumed["ok"], true);
    assert_eq!(resumed["result"]["committed_step_id"], 0);
    assert_eq!(resumed["result"]["reconciliation"]["outcome"], "unknown");
    assert_eq!(resumed["result"]["reconciliation"]["retryable"], false);
}
