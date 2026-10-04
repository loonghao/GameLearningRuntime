//! Bounded, read-only access to an owner-published local fleet snapshot.
//!
//! Availability means that the supplied file satisfies its closed contract.
//! It does not certify freshness, online machines, or learner improvement.
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{ErrorKind, Read};
use std::path::Path;

use serde::de::{
    DeserializeOwned, Error as _, MapAccess, SeqAccess, Visitor, value::MapAccessDeserializer,
};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Value, json};

use crate::observation::safe_child;

const VIEW_SCHEMA: &str = "glr.fleet.view.v1";
const SNAPSHOT_BYTES: u64 = 1 << 20;
const MAX_COUNTER: u64 = (1 << 53) - 1;

/// Read only the fixed managed file; missing or rejected input remains unknown.
/// Source epochs and all producer, receiver, and publication clocks are retained.
pub fn snapshot(data_dir: &Path) -> Value {
    match read_snapshot(data_dir) {
        Ok(Some(snapshot)) => {
            json!({"schema_version": VIEW_SCHEMA, "status": "available", "snapshot": snapshot})
        }
        Ok(None) => view_without_snapshot("missing"),
        Err(ReadFailure::Invalid) => view_without_snapshot("invalid"),
        Err(ReadFailure::Unavailable) => view_without_snapshot("unavailable"),
    }
}

fn view_without_snapshot(status: &str) -> Value {
    json!({"schema_version": VIEW_SCHEMA, "status": status, "snapshot": null})
}

enum ReadFailure {
    Invalid,
    Unavailable,
}

fn read_snapshot(data_dir: &Path) -> std::result::Result<Option<FrozenSnapshot>, ReadFailure> {
    let relative = Path::new("fleet").join("launcher-snapshot.json");
    let path = safe_child(data_dir, &relative).map_err(|_| ReadFailure::Invalid)?;
    let metadata = match fs::symlink_metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(_) => return Err(ReadFailure::Unavailable),
    };
    if !metadata.is_file() || metadata.len() > SNAPSHOT_BYTES {
        return Err(ReadFailure::Invalid);
    }
    let file = File::open(&path).map_err(|_| ReadFailure::Unavailable)?;
    let opened = file.metadata().map_err(|_| ReadFailure::Unavailable)?;
    if !opened.is_file() || opened.len() > SNAPSHOT_BYTES {
        return Err(ReadFailure::Invalid);
    }
    safe_child(data_dir, &relative).map_err(|_| ReadFailure::Invalid)?;
    let mut bytes = Vec::new();
    file.take(SNAPSHOT_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| ReadFailure::Unavailable)?;
    if bytes.len() as u64 > SNAPSHOT_BYTES {
        return Err(ReadFailure::Invalid);
    }
    let snapshot: JsonObject<FrozenSnapshot> =
        serde_json::from_slice(&bytes).map_err(|_| ReadFailure::Invalid)?;
    let snapshot = snapshot.0;
    if !snapshot.identities_are_consistent() {
        return Err(ReadFailure::Invalid);
    }
    Ok(Some(snapshot))
}

// A custom nullable type keeps each field required. Plain Option<T> would
// silently accept an absent field as None, contrary to the frozen wire shape.
#[derive(Serialize)]
#[serde(transparent)]
struct Nullable<T>(Option<T>);

impl<'de, T: DeserializeOwned> Deserialize<'de> for Nullable<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        if value.is_null() {
            Ok(Self(None))
        } else {
            serde_json::from_value(value)
                .map(|value| Self(Some(value)))
                .map_err(D::Error::custom)
        }
    }
}

// Derived serde structs support sequences as well as maps. The wire contract
// requires objects at the top level and for every record, without losing
// duplicate-key detection by first projecting into serde_json::Value.
#[derive(Serialize)]
#[serde(transparent)]
struct JsonObject<T>(T);

impl<'de, T: Deserialize<'de>> Deserialize<'de> for JsonObject<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct ObjectVisitor<T>(std::marker::PhantomData<T>);

        impl<'de, T: Deserialize<'de>> Visitor<'de> for ObjectVisitor<T> {
            type Value = JsonObject<T>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a fleet JSON object")
            }

            fn visit_map<A: MapAccess<'de>>(
                self,
                map: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                T::deserialize(MapAccessDeserializer::new(map)).map(JsonObject)
            }
        }

        deserializer.deserialize_map(ObjectVisitor::<T>(std::marker::PhantomData))
    }
}

#[derive(Serialize)]
#[serde(transparent)]
struct BoundedVec<T, const MAX: usize>(Vec<T>);

impl<'de, T: Deserialize<'de>, const MAX: usize> Deserialize<'de> for BoundedVec<T, MAX> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct BoundedVisitor<T, const MAX: usize>(std::marker::PhantomData<T>);

        impl<'de, T: Deserialize<'de>, const MAX: usize> Visitor<'de> for BoundedVisitor<T, MAX> {
            type Value = BoundedVec<T, MAX>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "an array of at most {MAX} elements")
            }

            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut sequence: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                if sequence.size_hint().is_some_and(|size| size > MAX) {
                    return Err(A::Error::custom("array exceeds its bound"));
                }
                let mut values = Vec::new();
                while let Some(value) = sequence.next_element()? {
                    if values.len() == MAX {
                        return Err(A::Error::custom("array exceeds its bound"));
                    }
                    values.push(value);
                }
                Ok(BoundedVec(values))
            }
        }

        deserializer.deserialize_seq(BoundedVisitor::<T, MAX>(std::marker::PhantomData))
    }
}

macro_rules! bounded_string {
    ($name:ident, $validator:expr) => {
        #[derive(Serialize)]
        #[serde(transparent)]
        struct $name(String);

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(
                deserializer: D,
            ) -> std::result::Result<Self, D::Error> {
                let value = String::deserialize(deserializer)?;
                if ($validator)(&value) {
                    Ok(Self(value))
                } else {
                    Err(D::Error::custom("invalid bounded fleet value"))
                }
            }
        }
    };
}

fn portable_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 96
        && value.as_bytes()[0].is_ascii_alphanumeric()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
}

fn lowercase_hex(value: &str, width: usize) -> bool {
    value.len() == width
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

bounded_string!(PortableId, portable_id);
bounded_string!(Sha256, |value: &str| lowercase_hex(value, 64));
bounded_string!(SourceCommit, |value: &str| lowercase_hex(value, 40));
bounded_string!(UtcTimestamp, canonical_utc);

fn canonical_utc(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() != 24
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes[10] != b'T'
        || bytes[13] != b':'
        || bytes[16] != b':'
        || bytes[19] != b'.'
        || bytes[23] != b'Z'
    {
        return false;
    }
    let number = |start: usize, end: usize| -> Option<u32> {
        bytes[start..end].iter().try_fold(0, |value, byte| {
            byte.is_ascii_digit()
                .then_some(value * 10 + u32::from(byte.wrapping_sub(b'0')))
        })
    };
    let (Some(year), Some(month), Some(day), Some(hour), Some(minute), Some(second), Some(_)) = (
        number(0, 4),
        number(5, 7),
        number(8, 10),
        number(11, 13),
        number(14, 16),
        number(17, 19),
        number(20, 23),
    ) else {
        return false;
    };
    if year == 0 || !(1..=12).contains(&month) || hour > 23 || minute > 59 || second > 59 {
        return false;
    }
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days = match month {
        2 if leap => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    };
    (1..=days).contains(&day)
}

#[derive(Serialize)]
#[serde(transparent)]
struct Counter(u64);

impl<'de> Deserialize<'de> for Counter {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let value = u64::deserialize(deserializer)?;
        if value > MAX_COUNTER {
            return Err(D::Error::custom("counter exceeds its bound"));
        }
        Ok(Self(value))
    }
}

#[derive(Serialize)]
#[serde(transparent)]
struct HeartbeatTtl(u64);

impl<'de> Deserialize<'de> for HeartbeatTtl {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        match u64::deserialize(deserializer)? {
            120 => Ok(Self(120)),
            _ => Err(D::Error::custom("unsupported heartbeat TTL")),
        }
    }
}

// Deserialize enums from strings only. Externally tagged serde enums would
// also admit object encodings, which are outside this exact JSON contract.
macro_rules! wire_enum {
    ($name:ident { $($variant:ident => $label:literal),+ $(,)? }) => {
        #[derive(Serialize, PartialEq, Eq, Hash)]
        enum $name {
            $(#[serde(rename = $label)] $variant),+
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(
                deserializer: D,
            ) -> std::result::Result<Self, D::Error> {
                let value = String::deserialize(deserializer)?;
                match value.as_str() {
                    $($label => Ok(Self::$variant),)+
                    _ => Err(D::Error::custom("unsupported fleet code")),
                }
            }
        }
    };
}

wire_enum!(SnapshotSchema { V1 => "glr.fleet.snapshot.v1" });
wire_enum!(Scope { LocalTrustedSources => "local_trusted_sources" });
wire_enum!(DeclaredStatus { Unknown => "unknown", Running => "running", Stopped => "stopped" });
wire_enum!(QualityStatus {
    Empty => "empty", Ready => "ready", Quarantine => "quarantine", Revoked => "revoked"
});
wire_enum!(Split { Train => "train", EvaluationHoldout => "evaluation_holdout", Quarantine => "quarantine" });
wire_enum!(ConsumptionStatus { Consumed => "consumed", UnknownEffect => "unknown_effect", Rejected => "rejected" });
wire_enum!(PlanReason {
    Eligible => "eligible",
    Holdout => "holdout",
    Quarantine => "quarantine",
    Revoked => "revoked",
    CompatibilityMismatch => "compatibility_mismatch",
    PolicyMismatch => "policy_mismatch",
    OffPolicyNotAllowed => "off_policy_not_allowed",
    SimulatedNotAllowed => "simulated_not_allowed",
    AlreadyClaimed => "already_claimed",
    NoReadyData => "no_ready_data",
});

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Machine {
    source_id: PortableId,
    source_epoch: PortableId,
    machine_id: PortableId,
    simulated: bool,
    revoked: bool,
    declared_status: DeclaredStatus,
    run_id: PortableId,
    game_id: PortableId,
    environment_id: PortableId,
    source_revision: PortableId,
    runtime_source_commit: SourceCommit,
    adapter_source_sha256: Sha256,
    behavior_policy_sha256: Sha256,
    checkpoint_sha256: Nullable<Sha256>,
    heartbeat_declared_at_utc: Nullable<UtcTimestamp>,
    heartbeat_received_at_utc: Nullable<UtcTimestamp>,
    data_observed_at_utc: Nullable<UtcTimestamp>,
    data_received_at_utc: Nullable<UtcTimestamp>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Dataset {
    source_id: PortableId,
    source_epoch: PortableId,
    compatibility_group_sha256: Sha256,
    assignment_id: PortableId,
    split: Split,
    quality_status: QualityStatus,
    accepted_transition_count: Counter,
    duplicate_transition_count: Counter,
    rejected_shard_count: Counter,
    ready_shard_count: Counter,
    retained_bytes: Counter,
    last_plan_eligible: Nullable<bool>,
    last_plan_reason_codes: BoundedVec<PlanReason, 16>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ConsumerReceipt {
    receipt_id: PortableId,
    plan_id: PortableId,
    learner_id: PortableId,
    source_ids: BoundedVec<PortableId, 64>,
    status: ConsumptionStatus,
    transition_count: Counter,
    callback_completed: bool,
    learner_declared_updates: Nullable<Counter>,
    finished_at_utc: Nullable<UtcTimestamp>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FrozenSnapshot {
    schema_version: SnapshotSchema,
    generated_at_utc: UtcTimestamp,
    scope: Scope,
    heartbeat_ttl_seconds: HeartbeatTtl,
    machines: BoundedVec<JsonObject<Machine>, 64>,
    datasets: BoundedVec<JsonObject<Dataset>, 64>,
    consumer_receipts: BoundedVec<JsonObject<ConsumerReceipt>, 64>,
}

impl FrozenSnapshot {
    fn identities_are_consistent(&self) -> bool {
        let mut epochs = HashSet::new();
        for machine in &self.machines.0 {
            let machine = &machine.0;
            if !epochs.insert((&machine.source_id.0, &machine.source_epoch.0)) {
                return false;
            }
        }
        let mut datasets = HashSet::new();
        for dataset in &self.datasets.0 {
            let dataset = &dataset.0;
            if !epochs.contains(&(&dataset.source_id.0, &dataset.source_epoch.0))
                || !datasets.insert((
                    &dataset.source_id.0,
                    &dataset.source_epoch.0,
                    &dataset.compatibility_group_sha256.0,
                    &dataset.assignment_id.0,
                    &dataset.split,
                ))
            {
                return false;
            }
        }
        let mut receipts = HashSet::new();
        for receipt in &self.consumer_receipts.0 {
            let receipt = &receipt.0;
            let mut receipt_sources = HashSet::new();
            if !receipts.insert(&receipt.receipt_id.0)
                || receipt
                    .source_ids
                    .0
                    .iter()
                    .any(|source| !receipt_sources.insert(&source.0))
            {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn fixture() -> Value {
        json!({
            "schema_version": "glr.fleet.snapshot.v1",
            "generated_at_utc": "2026-01-02T03:04:05.123Z",
            "scope": "local_trusted_sources",
            "heartbeat_ttl_seconds": 120,
            "machines": [{
                "source_id": "source-a", "source_epoch": "epoch-a", "machine_id": "machine-a",
                "simulated": false, "revoked": false, "declared_status": "running",
                "run_id": "run-a", "game_id": "counter", "environment_id": "counter-v1",
                "source_revision": "v1", "runtime_source_commit": "1".repeat(40),
                "adapter_source_sha256": "2".repeat(64),
                "behavior_policy_sha256": "3".repeat(64), "checkpoint_sha256": null,
                "heartbeat_declared_at_utc": "2026-01-02T03:04:00.100Z",
                "heartbeat_received_at_utc": "2026-01-02T03:04:01.200Z",
                "data_observed_at_utc": "2026-01-02T03:03:00.300Z",
                "data_received_at_utc": "2026-01-02T03:03:01.400Z"
            }],
            "datasets": [{
                "source_id": "source-a", "source_epoch": "epoch-a",
                "compatibility_group_sha256": "4".repeat(64), "assignment_id": "train-a",
                "split": "train", "quality_status": "ready", "accepted_transition_count": 3,
                "duplicate_transition_count": 1, "rejected_shard_count": 0,
                "ready_shard_count": 1, "retained_bytes": 256, "last_plan_eligible": true,
                "last_plan_reason_codes": ["eligible"]
            }],
            "consumer_receipts": [{
                "receipt_id": "receipt-a", "plan_id": "plan-a", "learner_id": "learner-a",
                "source_ids": ["source-a"], "status": "consumed", "transition_count": 3,
                "callback_completed": true, "learner_declared_updates": 2,
                "finished_at_utc": "2026-01-02T03:04:04.500Z"
            }]
        })
    }

    fn write_snapshot(root: &Path, value: &Value) -> std::path::PathBuf {
        let fleet = root.join("fleet");
        fs::create_dir_all(&fleet).unwrap();
        let path = fleet.join("launcher-snapshot.json");
        fs::write(&path, serde_json::to_vec(value).unwrap()).unwrap();
        path
    }

    fn assert_invalid(value: &Value) {
        let root = TempDir::new().unwrap();
        let path = write_snapshot(root.path(), value);
        let before = fs::read(&path).unwrap();
        assert_eq!(snapshot(root.path()), view_without_snapshot("invalid"));
        assert_eq!(fs::read(path).unwrap(), before);
    }

    #[test]
    fn missing_storage_does_not_create_directories_or_time() {
        let root = TempDir::new().unwrap();
        let data = root.path().join("absent");
        assert_eq!(snapshot(&data), view_without_snapshot("missing"));
        assert!(!data.exists());
        assert_eq!(fs::read_dir(root.path()).unwrap().count(), 0);
        assert_eq!(snapshot(root.path()), view_without_snapshot("missing"));
        assert!(!root.path().join("fleet").exists());
    }

    #[test]
    fn valid_snapshot_retains_all_clocks_epochs_and_durable_fields_without_writes() {
        let root = TempDir::new().unwrap();
        let value = fixture();
        let path = write_snapshot(root.path(), &value);
        let before = fs::read(&path).unwrap();
        let view = snapshot(root.path());
        assert_eq!(
            view,
            json!({"schema_version":VIEW_SCHEMA,
            "status":"available", "snapshot":value})
        );
        assert_eq!(fs::read(path).unwrap(), before);
        assert_eq!(fs::read_dir(root.path().join("fleet")).unwrap().count(), 1);
    }

    #[test]
    fn stale_future_and_simulated_input_never_gain_health_claims_or_new_clocks() {
        for timestamp in ["1999-01-01T00:00:00.000Z", "2099-01-01T00:00:00.000Z"] {
            let root = TempDir::new().unwrap();
            let mut value = fixture();
            value["generated_at_utc"] = json!(timestamp);
            value["machines"][0]["heartbeat_declared_at_utc"] = json!(timestamp);
            value["machines"][0]["simulated"] = json!(true);
            write_snapshot(root.path(), &value);
            let view = snapshot(root.path());
            assert_eq!(view["snapshot"], value);
            assert_eq!(view.as_object().unwrap().len(), 3);
            assert!(view.get("online").is_none());
            assert!(view.get("snapshot_status").is_none());
        }
    }

    #[test]
    fn required_nullable_fields_accept_null_and_still_require_the_key() {
        let root = TempDir::new().unwrap();
        let mut value = fixture();
        for key in [
            "checkpoint_sha256",
            "heartbeat_declared_at_utc",
            "heartbeat_received_at_utc",
            "data_observed_at_utc",
            "data_received_at_utc",
        ] {
            value["machines"][0][key] = Value::Null;
        }
        value["datasets"][0]["last_plan_eligible"] = Value::Null;
        value["consumer_receipts"][0]["learner_declared_updates"] = Value::Null;
        value["consumer_receipts"][0]["finished_at_utc"] = Value::Null;
        write_snapshot(root.path(), &value);
        assert_eq!(snapshot(root.path())["snapshot"], value);
        for pointer in ["", "/machines/0", "/datasets/0", "/consumer_receipts/0"] {
            let keys: Vec<_> = fixture()
                .pointer(pointer)
                .unwrap()
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect();
            for key in keys {
                let mut missing = fixture();
                missing
                    .pointer_mut(pointer)
                    .unwrap()
                    .as_object_mut()
                    .unwrap()
                    .remove(&key);
                assert_invalid(&missing);
            }
        }
    }

    #[test]
    fn unknown_fields_at_every_level_are_rejected_without_private_error_data() {
        for pointer in ["", "/machines/0", "/datasets/0", "/consumer_receipts/0"] {
            let mut value = fixture();
            value
                .pointer_mut(pointer)
                .unwrap()
                .as_object_mut()
                .unwrap()
                .insert("private_path".into(), json!("synthetic-private-value"));
            assert_invalid(&value);
        }
    }

    #[test]
    fn top_level_and_all_records_reject_sequence_encodings_of_valid_fields() {
        let rows: [(&str, &[&str]); 4] = [
            (
                "",
                &[
                    "schema_version",
                    "generated_at_utc",
                    "scope",
                    "heartbeat_ttl_seconds",
                    "machines",
                    "datasets",
                    "consumer_receipts",
                ],
            ),
            (
                "/machines/0",
                &[
                    "source_id",
                    "source_epoch",
                    "machine_id",
                    "simulated",
                    "revoked",
                    "declared_status",
                    "run_id",
                    "game_id",
                    "environment_id",
                    "source_revision",
                    "runtime_source_commit",
                    "adapter_source_sha256",
                    "behavior_policy_sha256",
                    "checkpoint_sha256",
                    "heartbeat_declared_at_utc",
                    "heartbeat_received_at_utc",
                    "data_observed_at_utc",
                    "data_received_at_utc",
                ],
            ),
            (
                "/datasets/0",
                &[
                    "source_id",
                    "source_epoch",
                    "compatibility_group_sha256",
                    "assignment_id",
                    "split",
                    "quality_status",
                    "accepted_transition_count",
                    "duplicate_transition_count",
                    "rejected_shard_count",
                    "ready_shard_count",
                    "retained_bytes",
                    "last_plan_eligible",
                    "last_plan_reason_codes",
                ],
            ),
            (
                "/consumer_receipts/0",
                &[
                    "receipt_id",
                    "plan_id",
                    "learner_id",
                    "source_ids",
                    "status",
                    "transition_count",
                    "callback_completed",
                    "learner_declared_updates",
                    "finished_at_utc",
                ],
            ),
        ];
        for (pointer, fields) in rows {
            let mut value = fixture();
            let object = value.pointer(pointer).unwrap();
            let ordered = fields.iter().map(|field| object[*field].clone()).collect();
            *value.pointer_mut(pointer).unwrap() = Value::Array(ordered);
            assert_invalid(&value);
        }
    }

    #[test]
    fn identity_duplicates_and_foreign_dataset_epoch_or_sources_are_rejected() {
        for collection in ["machines", "datasets", "consumer_receipts"] {
            let mut value = fixture();
            let first = value[collection][0].clone();
            value[collection].as_array_mut().unwrap().push(first);
            assert_invalid(&value);
        }
        for (pointer, replacement) in [
            ("/datasets/0/source_epoch", json!("other-epoch")),
            ("/datasets/0/source_id", json!("other-source")),
            (
                "/consumer_receipts/0/source_ids",
                json!(["source-a", "source-a"]),
            ),
        ] {
            let mut value = fixture();
            *value.pointer_mut(pointer).unwrap() = replacement;
            assert_invalid(&value);
        }
    }

    #[test]
    fn historical_consumer_receipt_does_not_require_current_machine_membership() {
        let root = TempDir::new().unwrap();
        let mut value = fixture();
        value["consumer_receipts"][0]["source_ids"] = json!(["historical-source"]);
        write_snapshot(root.path(), &value);
        assert_eq!(snapshot(root.path())["snapshot"], value);
        value["machines"] = json!([]);
        value["datasets"] = json!([]);
        write_snapshot(root.path(), &value);
        assert_eq!(snapshot(root.path())["snapshot"], value);
    }

    #[test]
    fn compatibility_cohorts_remain_distinct_and_cannot_be_merged_by_the_reader() {
        let root = TempDir::new().unwrap();
        let mut value = fixture();
        let mut cohort = value["datasets"][0].clone();
        cohort["compatibility_group_sha256"] = json!("5".repeat(64));
        cohort["last_plan_eligible"] = json!(false);
        cohort["last_plan_reason_codes"] = json!(["compatibility_mismatch"]);
        value["datasets"].as_array_mut().unwrap().push(cohort);
        write_snapshot(root.path(), &value);
        assert_eq!(snapshot(root.path())["snapshot"], value);
    }

    #[test]
    fn all_array_bounds_are_enforced_without_truncating_input() {
        for pointer in [
            "/machines",
            "/datasets",
            "/consumer_receipts",
            "/consumer_receipts/0/source_ids",
        ] {
            let mut value = fixture();
            let array = value.pointer_mut(pointer).unwrap().as_array_mut().unwrap();
            *array = vec![array[0].clone(); 65];
            assert_invalid(&value);
        }
        let mut value = fixture();
        value["datasets"][0]["last_plan_reason_codes"] = json!(vec!["eligible"; 17]);
        assert_invalid(&value);
    }

    #[test]
    fn exact_array_limits_and_portable_id_length_are_admitted() {
        let root = TempDir::new().unwrap();
        let mut value = fixture();
        let machine = value["machines"][0].clone();
        let dataset = value["datasets"][0].clone();
        let receipt = value["consumer_receipts"][0].clone();
        value["machines"] = json!([]);
        value["datasets"] = json!([]);
        value["consumer_receipts"] = json!([]);
        let source_ids: Vec<_> = (0..64).map(|index| format!("source-{index}")).collect();
        for (index, source_id) in source_ids.iter().enumerate() {
            let mut machine = machine.clone();
            machine["source_id"] = json!(source_id);
            machine["machine_id"] = json!("a".repeat(96));
            let mut dataset = dataset.clone();
            dataset["source_id"] = json!(source_id);
            dataset["last_plan_reason_codes"] = json!(vec!["eligible"; 16]);
            let mut receipt = receipt.clone();
            receipt["receipt_id"] = json!(format!("receipt-{index}"));
            receipt["source_ids"] = json!(source_ids);
            value["machines"].as_array_mut().unwrap().push(machine);
            value["datasets"].as_array_mut().unwrap().push(dataset);
            value["consumer_receipts"]
                .as_array_mut()
                .unwrap()
                .push(receipt);
        }
        write_snapshot(root.path(), &value);
        assert_eq!(snapshot(root.path())["snapshot"], value);
    }

    #[test]
    fn enums_reject_object_encodings_and_arrays_instead_of_scalar_codes() {
        for (pointer, code) in [
            ("/schema_version", "glr.fleet.snapshot.v1"),
            ("/scope", "local_trusted_sources"),
            ("/machines/0/declared_status", "running"),
            ("/datasets/0/quality_status", "ready"),
            ("/datasets/0/split", "train"),
            ("/consumer_receipts/0/status", "consumed"),
            ("/datasets/0/last_plan_reason_codes/0", "eligible"),
        ] {
            for replacement in [json!({code: null}), json!([code])] {
                let mut value = fixture();
                *value.pointer_mut(pointer).unwrap() = replacement;
                assert_invalid(&value);
            }
        }
    }

    #[test]
    fn counters_booleans_ids_hashes_and_all_enums_use_exact_types_and_bounds() {
        for (pointer, replacement) in [
            ("/heartbeat_ttl_seconds", json!(121)),
            ("/heartbeat_ttl_seconds", json!(120.0)),
            ("/scope", json!("remote")),
            ("/schema_version", json!("glr.fleet.snapshot.v2")),
            ("/machines/0/simulated", json!(1)),
            ("/machines/0/revoked", json!("false")),
            ("/machines/0/declared_status", json!("online")),
            ("/machines/0/source_id", json!("../outside")),
            ("/machines/0/machine_id", json!("a".repeat(97))),
            ("/machines/0/source_revision", json!("version secret")),
            ("/machines/0/run_id", Value::Null),
            ("/machines/0/runtime_source_commit", json!("A".repeat(40))),
            ("/machines/0/adapter_source_sha256", json!("g".repeat(64))),
            ("/machines/0/behavior_policy_sha256", Value::Null),
            ("/machines/0/checkpoint_sha256", json!("a".repeat(63))),
            ("/datasets/0/split", json!("test")),
            ("/datasets/0/quality_status", json!("healthy")),
            ("/datasets/0/accepted_transition_count", json!(true)),
            ("/datasets/0/duplicate_transition_count", json!(-1)),
            ("/datasets/0/rejected_shard_count", json!(1.0)),
            ("/datasets/0/ready_shard_count", json!(MAX_COUNTER + 1)),
            ("/datasets/0/last_plan_eligible", json!(1)),
            (
                "/datasets/0/last_plan_reason_codes",
                json!(["private_reason"]),
            ),
            ("/consumer_receipts/0/status", json!("improved")),
            ("/consumer_receipts/0/callback_completed", json!(1)),
            (
                "/consumer_receipts/0/learner_declared_updates",
                json!(MAX_COUNTER + 1),
            ),
        ] {
            let mut value = fixture();
            *value.pointer_mut(pointer).unwrap() = replacement;
            assert_invalid(&value);
        }
        let root = TempDir::new().unwrap();
        let mut value = fixture();
        value["datasets"][0]["retained_bytes"] = json!(MAX_COUNTER);
        write_snapshot(root.path(), &value);
        assert_eq!(snapshot(root.path())["snapshot"], value);
    }

    #[test]
    fn canonical_utc_requires_real_dates_and_exact_milliseconds_z() {
        for value in [
            "0001-01-01T00:00:00.000Z",
            "2000-02-29T23:59:59.999Z",
            "2024-02-29T00:00:00.001Z",
            "9999-12-31T23:59:59.999Z",
        ] {
            assert!(canonical_utc(value));
        }
        for value in [
            "0000-01-01T00:00:00.000Z",
            "1900-02-29T00:00:00.000Z",
            "2026-02-29T00:00:00.000Z",
            "2026-04-31T00:00:00.000Z",
            "2026-00-01T00:00:00.000Z",
            "2026-13-01T00:00:00.000Z",
            "2026-01-00T00:00:00.000Z",
            "2026-01-01T24:00:00.000Z",
            "2026-01-01T00:60:00.000Z",
            "2026-01-01T00:00:60.000Z",
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:00.00Z",
            "2026-01-01T00:00:00.0000Z",
            "2026-01-01T00:00:00.000+00:00",
            "2026-01-01T00:00:00.000z",
            "2026-01-01T00:00:00.0x0Z",
            "2026-01-01T00:00:00.000Z\n",
            "synthetic-private-timestamp",
        ] {
            assert!(!canonical_utc(value));
            let mut input = fixture();
            input["generated_at_utc"] = json!(value);
            assert_invalid(&input);
        }
        for pointer in [
            "/machines/0/heartbeat_declared_at_utc",
            "/machines/0/heartbeat_received_at_utc",
            "/machines/0/data_observed_at_utc",
            "/machines/0/data_received_at_utc",
            "/consumer_receipts/0/finished_at_utc",
        ] {
            let mut value = fixture();
            *value.pointer_mut(pointer).unwrap() = json!("2026-02-30T00:00:00.000Z");
            assert_invalid(&value);
        }
    }

    #[test]
    fn nonregular_oversized_malformed_and_duplicate_json_are_safe_invalid() {
        let root = TempDir::new().unwrap();
        let path = root.path().join("fleet/launcher-snapshot.json");
        fs::create_dir_all(&path).unwrap();
        assert_eq!(snapshot(root.path()), view_without_snapshot("invalid"));

        let root = TempDir::new().unwrap();
        let path = write_snapshot(root.path(), &fixture());
        let original = serde_json::to_string(&fixture()).unwrap();
        let duplicate_root = format!(
            "{{\"schema_version\":\"glr.fleet.snapshot.v1\",{}",
            &original[1..]
        );
        let duplicate_nested = original.replacen(
            "\"source_id\":\"source-a\"",
            "\"source_id\":\"source-a\",\"source_id\":\"source-a\"",
            1,
        );
        for bytes in [
            vec![b' '; SNAPSHOT_BYTES as usize + 1],
            b"synthetic-private-malformed".to_vec(),
            vec![0xff],
            duplicate_root.into_bytes(),
            duplicate_nested.into_bytes(),
        ] {
            fs::write(&path, &bytes).unwrap();
            assert_eq!(snapshot(root.path()), view_without_snapshot("invalid"));
            assert_eq!(fs::read(&path).unwrap(), bytes);
        }
    }

    #[test]
    #[cfg(any(unix, windows))]
    fn file_and_ancestor_links_cannot_escape_managed_storage() {
        let root = TempDir::new().unwrap();
        let outside = root.path().join("outside");
        fs::create_dir(&outside).unwrap();
        let target = outside.join("target.json");
        fs::write(&target, serde_json::to_vec(&fixture()).unwrap()).unwrap();
        let data = root.path().join("data");
        fs::create_dir_all(data.join("fleet")).unwrap();
        let link = data.join("fleet/launcher-snapshot.json");
        #[cfg(unix)]
        std::os::unix::fs::symlink(&target, &link).unwrap();
        #[cfg(windows)]
        if let Err(error) = std::os::windows::fs::symlink_file(&target, &link) {
            // ERROR_PRIVILEGE_NOT_HELD is not consistently classified as
            // PermissionDenied by std. Do not request privileges for fixtures.
            assert_eq!(error.raw_os_error(), Some(1314));
            eprintln!("Windows link fixture unavailable: ERROR_PRIVILEGE_NOT_HELD");
            return;
        }
        assert_eq!(snapshot(&data), view_without_snapshot("invalid"));
        assert_eq!(
            fs::read(&target).unwrap(),
            serde_json::to_vec(&fixture()).unwrap()
        );

        let ancestor = root.path().join("linked-data");
        #[cfg(unix)]
        std::os::unix::fs::symlink(&data, &ancestor).unwrap();
        #[cfg(windows)]
        std::os::windows::fs::symlink_dir(&data, &ancestor).unwrap();
        assert_eq!(snapshot(&ancestor), view_without_snapshot("invalid"));
    }
}
