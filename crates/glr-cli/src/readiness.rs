//! Bounded startup readiness windows for project roles.
//!
//! A role reports readiness through the published
//! `glr.environment-readiness.v1` mapping, exactly as the Python SDK's
//! `ReadinessResult` does. `run_readiness_window` re-invokes the role while
//! that receipt keeps saying `not_ready`, so a host that is still starting is
//! never recorded as a crash, and a crash is never retried.

use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::error::{Error, Result};

pub const READINESS_SCHEMA_VERSION: &str = "glr.environment-readiness.v1";
const MAX_RECEIPT_BYTES: u64 = 64 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadinessState {
    Ready,
    NotReady,
    Unavailable,
}

impl ReadinessState {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "ready" => Ok(Self::Ready),
            "not_ready" => Ok(Self::NotReady),
            "unavailable" => Ok(Self::Unavailable),
            other => Err(Error::Invalid(format!(
                "readiness receipt state must be ready, not_ready, or unavailable: {other}"
            ))),
        }
    }
}

/// One validated role-published readiness receipt.
#[derive(Debug, Clone, Serialize)]
pub struct ReadinessReceipt {
    pub schema_version: &'static str,
    pub state: ReadinessState,
    #[serde(default)]
    pub reason: String,
    pub checked_at_ns: i64,
}

#[derive(Debug, Deserialize)]
struct RawReceipt {
    schema_version: String,
    state: String,
    #[serde(default)]
    reason: String,
    #[serde(default)]
    checked_at_ns: Option<i64>,
}

/// Read one role receipt, or `None` when the role published nothing usable.
///
/// An absent, oversized, symlinked, unreadable, or off-schema receipt is
/// treated as "no receipt". That keeps the caller fail-closed: only an
/// explicit `not_ready` receipt is retryable.
pub fn read_receipt(path: &Path) -> Result<Option<ReadinessReceipt>> {
    if path.is_symlink() || !path.is_file() {
        return Ok(None);
    }
    if fs::metadata(path)?.len() > MAX_RECEIPT_BYTES {
        return Ok(None);
    }
    let Ok(bytes) = fs::read(path) else {
        return Ok(None);
    };
    let Ok(raw) = serde_json::from_slice::<RawReceipt>(&bytes) else {
        return Ok(None);
    };
    if raw.schema_version != READINESS_SCHEMA_VERSION {
        return Ok(None);
    }
    let Ok(state) = ReadinessState::parse(&raw.state) else {
        return Ok(None);
    };
    if raw.reason.chars().any(char::is_control) || raw.reason.chars().count() > 256 {
        return Ok(None);
    }
    let checked_at_ns = match raw.checked_at_ns {
        None | Some(0) => crate::store::now_ns()?,
        Some(value) if value > 0 => value,
        Some(_) => return Ok(None),
    };
    Ok(Some(ReadinessReceipt {
        schema_version: READINESS_SCHEMA_VERSION,
        state,
        reason: raw.reason,
        checked_at_ns,
    }))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadinessWindowVerdict {
    Succeeded,
    NotReady,
    Unavailable,
    Inconsistent,
    Unreported,
}

/// One role invocation and the readiness receipt it published, if any.
#[derive(Debug, Clone, Serialize)]
pub struct ReadinessAttempt {
    pub index: u32,
    pub exit_code: i32,
    pub readiness: Option<ReadinessReceipt>,
}

impl ReadinessAttempt {
    fn verdict(&self) -> Option<ReadinessWindowVerdict> {
        if self.exit_code == 0 {
            return Some(ReadinessWindowVerdict::Succeeded);
        }
        match &self.readiness {
            None => Some(ReadinessWindowVerdict::Unreported),
            Some(receipt) if receipt.state == ReadinessState::Ready => {
                Some(ReadinessWindowVerdict::Inconsistent)
            }
            Some(receipt) if receipt.state == ReadinessState::Unavailable => {
                Some(ReadinessWindowVerdict::Unavailable)
            }
            Some(_) => None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ReadinessWindowOutcome {
    pub verdict: ReadinessWindowVerdict,
    pub exhausted: bool,
    pub timeout_seconds: f64,
    pub attempts: Vec<ReadinessAttempt>,
}

impl ReadinessWindowOutcome {
    pub fn mapping(&self) -> Value {
        json!({
            "verdict": self.verdict,
            "exhausted": self.exhausted,
            "timeout_seconds": self.timeout_seconds,
            "attempts": self.attempts,
        })
    }
}

/// Re-invoke a role until it is judged ready, terminal, or out of window.
pub fn run_readiness_window<F>(
    timeout_seconds: f64,
    poll_interval_seconds: f64,
    mut attempt: F,
) -> Result<ReadinessWindowOutcome>
where
    F: FnMut(u32) -> Result<ReadinessAttempt>,
{
    if !timeout_seconds.is_finite()
        || timeout_seconds <= 0.0
        || !poll_interval_seconds.is_finite()
        || poll_interval_seconds <= 0.0
    {
        return Err(Error::Invalid(
            "timeout_seconds and poll_interval_seconds must be positive".into(),
        ));
    }
    let deadline = Instant::now() + Duration::from_secs_f64(timeout_seconds);
    let mut attempts: Vec<ReadinessAttempt> = Vec::new();
    loop {
        let record = attempt(attempts.len() as u32 + 1)?;
        let verdict = record.verdict();
        attempts.push(record);
        if let Some(verdict) = verdict {
            return Ok(ReadinessWindowOutcome {
                verdict,
                exhausted: verdict == ReadinessWindowVerdict::NotReady,
                timeout_seconds,
                attempts,
            });
        }
        let now = Instant::now();
        if now >= deadline {
            return Ok(ReadinessWindowOutcome {
                verdict: ReadinessWindowVerdict::NotReady,
                exhausted: true,
                timeout_seconds,
                attempts,
            });
        }
        std::thread::sleep((deadline - now).min(Duration::from_secs_f64(poll_interval_seconds)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn receipt(state: ReadinessState) -> Option<ReadinessReceipt> {
        Some(ReadinessReceipt {
            schema_version: READINESS_SCHEMA_VERSION,
            state,
            reason: "host state".into(),
            checked_at_ns: 1,
        })
    }

    #[test]
    fn window_retries_only_while_the_role_reports_not_ready() {
        let mut observed = Vec::new();
        let outcome = run_readiness_window(30.0, 0.001, |index| {
            observed.push(index);
            Ok(ReadinessAttempt {
                index,
                exit_code: if index < 3 { 63 } else { 0 },
                readiness: receipt(if index < 3 {
                    ReadinessState::NotReady
                } else {
                    ReadinessState::Ready
                }),
            })
        })
        .unwrap();
        assert_eq!(observed, vec![1, 2, 3]);
        assert_eq!(outcome.verdict, ReadinessWindowVerdict::Succeeded);
        assert!(!outcome.exhausted);
        assert_eq!(outcome.mapping()["attempts"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn window_exhausts_without_ever_claiming_a_failure_verdict() {
        let outcome = run_readiness_window(0.05, 0.005, |index| {
            Ok(ReadinessAttempt {
                index,
                exit_code: 63,
                readiness: receipt(ReadinessState::NotReady),
            })
        })
        .unwrap();
        assert_eq!(outcome.verdict, ReadinessWindowVerdict::NotReady);
        assert!(outcome.exhausted);
        assert_eq!(outcome.mapping()["verdict"], "not_ready");
    }

    #[test]
    fn window_is_terminal_on_unreported_unavailable_and_inconsistent_receipts() {
        for (expected, exit_code, state) in [
            (ReadinessWindowVerdict::Unreported, 63, None),
            (
                ReadinessWindowVerdict::Unavailable,
                9,
                Some(ReadinessState::Unavailable),
            ),
            (
                ReadinessWindowVerdict::Inconsistent,
                9,
                Some(ReadinessState::Ready),
            ),
        ] {
            let mut calls = 0;
            let outcome = run_readiness_window(30.0, 0.001, |index| {
                calls += 1;
                Ok(ReadinessAttempt {
                    index,
                    exit_code,
                    readiness: state.map(|value| ReadinessReceipt {
                        schema_version: READINESS_SCHEMA_VERSION,
                        state: value,
                        reason: String::new(),
                        checked_at_ns: 1,
                    }),
                })
            })
            .unwrap();
            assert_eq!(outcome.verdict, expected);
            assert!(!outcome.exhausted);
            assert_eq!(calls, 1, "a terminal receipt must not be retried");
        }
    }

    #[test]
    fn window_rejects_unbounded_or_inverted_bounds() {
        let error = run_readiness_window(0.0, 0.001, |index| {
            Ok(ReadinessAttempt {
                index,
                exit_code: 0,
                readiness: None,
            })
        })
        .unwrap_err();
        assert!(error.to_string().contains("must be positive"));
    }

    #[test]
    fn receipt_reader_fails_closed_on_unknown_schema_and_state() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("runtime-readiness.json");
        fs::write(
            &path,
            br#"{"schema_version":"glr.environment-readiness.v2","state":"ready"}"#,
        )
        .unwrap();
        assert!(read_receipt(&path).unwrap().is_none());
        fs::write(
            &path,
            br#"{"schema_version":"glr.environment-readiness.v1","state":"maybe"}"#,
        )
        .unwrap();
        assert!(read_receipt(&path).unwrap().is_none());
        fs::write(
            &path,
            br#"{"schema_version":"glr.environment-readiness.v1","state":"not_ready","reason":"booting","checked_at_ns":5,"adapter":"extra"}"#,
        )
        .unwrap();
        let receipt = read_receipt(&path).unwrap().unwrap();
        assert_eq!(receipt.state, ReadinessState::NotReady);
        assert_eq!(receipt.reason, "booting");
        assert_eq!(receipt.checked_at_ns, 5);
        fs::write(&path, b"not json").unwrap();
        assert!(read_receipt(&path).unwrap().is_none());
        fs::remove_file(&path).unwrap();
        assert!(read_receipt(&path).unwrap().is_none());
    }
}
