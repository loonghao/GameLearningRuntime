//! Best-effort release notices; cache contents never authorize installation.
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use semver::Version;
use serde::{Deserialize, Serialize};

use crate::update::{BUILD_TARGET, Updater};

const DAY: u64 = 24 * 60 * 60;
const COOLDOWN: u64 = 60 * 60;

#[derive(Default, Serialize, Deserialize)]
struct Cache {
    current: String,
    target: String,
    checked_at: u64,
    retry_at: u64,
    latest: Option<String>,
}

impl Cache {
    fn fresh(&self, now: u64, current: &str) -> bool {
        self.current == current
            && self.target == BUILD_TARGET
            && now >= self.checked_at
            && now < self.retry_at
    }

    fn notice(&self, current: &str) -> Option<String> {
        let latest = Version::parse(self.latest.as_deref()?).ok()?;
        let current = Version::parse(current).ok()?;
        (latest > current).then(|| {
            format!("A new GLR version is available: {current} -> {latest}. Run `glr update` to update GLR and project skills.")
        })
    }
}

fn cache_path() -> Option<PathBuf> {
    std::env::home_dir().map(|home| home.join(".glr/cache/update-check.json"))
}

fn check(path: &Path, now: u64, fetch: impl FnOnce() -> Option<String>) -> Option<String> {
    let current = env!("CARGO_PKG_VERSION");
    let mut cache: Cache = fs::read(path)
        .ok()
        .filter(|bytes| bytes.len() <= 16 * 1024)
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default();
    if !cache.fresh(now, current) {
        let latest = fetch();
        cache = Cache {
            current: current.into(),
            target: BUILD_TARGET.into(),
            checked_at: now,
            retry_at: now.saturating_add(if latest.is_some() { DAY } else { COOLDOWN }),
            latest,
        };
        // Atomic replacement prevents concurrent CLI invocations from exposing partial JSON.
        if let Some(parent) = path.parent()
            && fs::create_dir_all(parent).is_ok()
            && let Ok(mut temporary) = tempfile::NamedTempFile::new_in(parent)
            && serde_json::to_writer(temporary.as_file_mut(), &cache).is_ok()
        {
            let _ = temporary.persist(path);
        }
    }
    cache.notice(current)
}

pub(crate) fn start() -> Option<Receiver<Option<String>>> {
    if std::env::var_os("GLR_NO_UPDATE_CHECK").is_some() || std::env::var_os("CI").is_some() {
        return None;
    }
    let path = cache_path()?;
    let (sender, receiver) = mpsc::channel();
    std::thread::Builder::new()
        .name("glr-update-check".into())
        .spawn(move || {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let notice = check(&path, now, || {
                Updater::notification()
                    .ok()?
                    .check()
                    .ok()
                    .map(|plan| plan.latest_version)
            });
            let _ = sender.send(notice);
        })
        .ok()?;
    Some(receiver)
}

pub(crate) fn finish(receiver: Option<Receiver<Option<String>>>) {
    if let Some(receiver) = receiver
        && let Ok(Some(notice)) = receiver.recv_timeout(Duration::from_secs(1))
    {
        eprintln!("{notice}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caches_success_and_compares_semantic_versions() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("notice.json");
        assert!(
            check(&path, 100, || Some("999.0.0".into()))
                .unwrap()
                .contains("glr update")
        );
        assert!(check(&path, 101, || panic!("fresh cache must not fetch")).is_some());
        assert!(check(&path, 100 + DAY, || Some("0.0.1".into())).is_none());
    }

    #[test]
    fn network_failure_cools_down_and_corrupt_cache_recovers() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("notice.json");
        fs::write(&path, b"broken").unwrap();
        assert!(check(&path, 100, || None).is_none());
        assert!(check(&path, 101, || panic!("failure cooldown must not fetch")).is_none());
        assert!(check(&path, 100 + COOLDOWN, || Some("999.0.0".into())).is_some());
    }

    #[test]
    fn changed_binary_and_clock_rollback_invalidate_cache() {
        let cache = Cache {
            current: "1.0.0".into(),
            target: BUILD_TARGET.into(),
            checked_at: 100,
            retry_at: 200,
            latest: None,
        };
        assert!(cache.fresh(150, "1.0.0"));
        assert!(!cache.fresh(150, "1.0.1"));
        assert!(!cache.fresh(99, "1.0.0"));
    }
}
