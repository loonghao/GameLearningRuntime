//! Identity, durable leases, and enumeration for live workbench servers.
//!
//! The workbench server is already per-project: one `data_dir`, one
//! `environment_id`, job receipts and `dashboard/job.lock` under that
//! project's storage. What it did not have is an identity another process can
//! read back. A caller holding only a port could not name the project it
//! served; a caller holding only a project could not find the port; and with
//! several servers up at once the only available attribution was guessing from
//! `environment_id`.
//!
//! This module supplies the missing layer:
//!
//! * [`Instance`] is the payload -- who serves what, from where, since when.
//! * [`Lease`] publishes it into a user-scoped registry, so enumeration works
//!   without knowing any project path in advance.
//! * [`probe`] answers "is this still live?" by asking the server itself rather
//!   than by trusting a pid, so a reused port cannot be mistaken for the
//!   original instance.
//!
//! The registry directory is passed in rather than read from a global, so unit
//! tests stay hermetic and parallel-safe; [`lease_dir`] is the only place that
//! consults the environment, and only the CLI boundary calls it.
//!
//! Nothing here terminates a process. Retiring a server is an explicit request
//! the server acts on itself; see [`request_stop`] and `crate::observe`.

use std::fs;
use std::net::{Ipv4Addr, SocketAddr, TcpListener};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::error::{Error, Result};
use crate::project::Project;

pub const SCHEMA: &str = "glr.workbench-instance.v1";

/// The literal every GLR release has documented as the dashboard default.
///
/// Kept as the *preferred* address rather than dropped: a single-project host
/// that has been opening `http://127.0.0.1:7432` keeps landing on the same
/// page. It is no longer the only answer -- see [`port_candidates`].
pub const PREFERRED_PORT: u16 = 7432;

/// Width of the per-project band above [`PREFERRED_PORT`].
///
/// 256 slots keeps the derived address in a range a human reads as "GLR" and
/// keeps the arithmetic obvious, at the cost of a birthday collision between
/// two of roughly 19 simultaneously live projects. A collision is not an
/// error: a taken derived slot falls through to the rest of
/// [`port_candidates`].
const BAND: u16 = 256;

const PROBE_TIMEOUT: Duration = Duration::from_millis(500);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Instance {
    pub schema_version: String,
    /// Unique per server process; the key `dashboard stop` accepts.
    pub instance_id: String,
    pub environment_id: String,
    pub project_root: String,
    pub data_dir: String,
    /// Stable digest of the storage path; this is what "the same project"
    /// means when filtering the registry. Kept next to the display path so a
    /// project that moved does not inherit the old server's lease.
    pub data_dir_sha256: String,
    /// The running image, so a binary that cannot be replaced can name the
    /// servers holding it instead of reporting a bare OS error.
    pub executable: String,
    pub pid: u32,
    pub port: u16,
    pub url: String,
    pub read_only: bool,
    pub version: String,
    pub started_at_ms: i64,
    pub started_at: String,
}

impl Instance {
    pub fn new(project: &Project, port: u16, read_only: bool) -> Self {
        let started_at_ms = now_ms();
        Self {
            schema_version: SCHEMA.into(),
            instance_id: uuid::Uuid::new_v4().simple().to_string(),
            environment_id: project.environment_id.clone(),
            project_root: project.root.to_string_lossy().into_owned(),
            data_dir: project.data_dir.to_string_lossy().into_owned(),
            data_dir_sha256: storage_digest(&project.data_dir),
            executable: std::env::current_exe()
                .map(|path| path.to_string_lossy().into_owned())
                .unwrap_or_default(),
            pid: std::process::id(),
            port,
            url: format!("http://127.0.0.1:{port}/"),
            read_only,
            version: env!("CARGO_PKG_VERSION").into(),
            started_at_ms,
            started_at: utc_rfc3339(started_at_ms),
        }
    }

    /// True when this instance serves the storage `project` resolves to.
    pub fn serves(&self, project: &Project) -> bool {
        self.data_dir_sha256 == storage_digest(&project.data_dir)
    }
}

/// A published [`Instance`] that removes itself when the server goes away.
pub struct Lease {
    path: PathBuf,
    armed: bool,
}

impl Lease {
    /// Publish `instance` into `directory`.
    ///
    /// Written to a temporary file and renamed, so a reader never sees a
    /// half-written record even if the writer is killed mid-flight.
    pub fn publish(directory: &Path, instance: Instance) -> Result<Self> {
        fs::create_dir_all(directory)?;
        let path = directory.join(format!("{}.json", instance.instance_id));
        let temporary = directory.join(format!(".{}.json.tmp", instance.instance_id));
        fs::write(&temporary, serde_json::to_vec_pretty(&instance)?)?;
        if let Err(error) = fs::rename(&temporary, &path) {
            let _ = fs::remove_file(&temporary);
            return Err(error.into());
        }
        Ok(Self { path, armed: true })
    }

    /// Remove the lease now. Idempotent; `Drop` covers the error paths.
    pub fn release(mut self) {
        self.armed = false;
        let _ = fs::remove_file(&self.path);
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_file(&self.path);
        }
    }
}

/// Where the user-scoped registry of live servers lives.
///
/// `GLR_STATE_DIR` overrides the whole base, so a test or a CI job never
/// writes into a real user profile.
pub fn lease_dir() -> Result<PathBuf> {
    let base = if let Some(base) = std::env::var_os("GLR_STATE_DIR") {
        PathBuf::from(base)
    } else if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA")
            .or_else(|| std::env::var_os("APPDATA"))
            .map(PathBuf::from)
            .ok_or_else(|| Error::Invalid("no user state directory; set GLR_STATE_DIR".into()))?
    } else {
        std::env::var_os("XDG_STATE_HOME")
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/state"))
            })
            .ok_or_else(|| Error::Invalid("no user state directory; set GLR_STATE_DIR".into()))?
    };
    Ok(base.join("glr").join("workbench").join("instances"))
}

/// The addresses `glr dashboard` binds, in order.
///
/// `preferred` is `None` for "no explicit `--port`" and `Some` for the exact
/// address the operator asked for. An explicit request yields exactly one
/// candidate: silently binding somewhere else would be worse than failing,
/// because the caller is about to paste that number into a browser.
///
/// The implicit list is the historical default, then this project's derived
/// slot, then whatever the operating system hands out. That last entry is why
/// two projects can both start without an argument -- and why every path that
/// binds must report the address it actually got.
pub fn port_candidates(preferred: Option<u16>, data_dir: &Path) -> Vec<u16> {
    match preferred {
        Some(0) => vec![0],
        Some(port) => vec![port],
        None => vec![PREFERRED_PORT, derived_port(data_dir), 0],
    }
}

/// Bind the first usable candidate on loopback.
///
/// Returns the listener and the port actually bound: port `0` means "let the
/// kernel choose", so the address is only knowable after the call, and it is
/// exactly the value the caller must print.
pub fn bind(preferred: Option<u16>, data_dir: &Path) -> Result<(TcpListener, u16)> {
    let mut last: Option<std::io::Error> = None;
    for port in port_candidates(preferred, data_dir) {
        match TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, port))) {
            Ok(listener) => {
                listener.set_nonblocking(true)?;
                let bound = listener.local_addr()?.port();
                return Ok((listener, bound));
            }
            Err(error) => last = Some(error),
        }
    }
    Err(match last {
        Some(error) => Error::Contract(format!(
            "cannot bind a workbench address on loopback ({error}); pass --port 0 to let the \
             operating system choose, or stop the server that holds it \
             (`glr dashboard instances --all`)"
        )),
        None => Error::Contract("no candidate workbench address was attempted".into()),
    })
}

/// This project's slot in the band above [`PREFERRED_PORT`].
fn derived_port(data_dir: &Path) -> u16 {
    let digest = Sha256::digest(storage_key(data_dir).as_bytes());
    let offset = u16::from(digest[0]) | (u16::from(digest[1]) << 8);
    PREFERRED_PORT + 1 + (offset % (BAND - 1))
}

/// The key that decides "same project", stable across the spellings a human
/// reaches for: `--project .`, an absolute path, and either separator.
fn storage_key(data_dir: &Path) -> String {
    data_dir.to_string_lossy().replace('\\', "/")
}

fn storage_digest(data_dir: &Path) -> String {
    format!("{:x}", Sha256::digest(storage_key(data_dir).as_bytes()))
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(i64::MAX as u128) as i64
}

/// Wall clock as `2026-09-16T12:05:03.123Z`.
///
/// Hand-rolled rather than pulled from a date crate: this is the only place
/// GLR formats a timestamp, and the lease is meant to be readable by a human
/// staring at a JSON file.
pub fn utc_rfc3339(milliseconds: i64) -> String {
    let seconds = milliseconds.div_euclid(1000);
    let millis = milliseconds.rem_euclid(1000);
    let days = seconds.div_euclid(86_400);
    let rest = seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{millis:03}Z",
        rest / 3600,
        (rest % 3600) / 60,
        rest % 60
    )
}

/// Howard Hinnant's `civil_from_days`: days since 1970-01-01 to a Gregorian
/// date, valid for every value reachable from a wall clock.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = (day_of_year - (153 * month_prime + 2) / 5 + 1) as u32;
    let month = if month_prime < 10 {
        month_prime + 3
    } else {
        month_prime - 9
    } as u32;
    (if month <= 2 { year + 1 } else { year }, month, day)
}

/// What a probe of a recorded address found.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum State {
    /// The address answers health and names the same instance.
    Live,
    /// Nothing is listening: the recorded server is gone.
    Stale,
    /// Something answers, but it is not this instance -- a recycled port, or a
    /// build old enough to carry no identity. Never treated as live, and never
    /// removed by `--prune`: a responder we cannot identify is not evidence
    /// that the file is wrong.
    Foreign,
}

/// Ask a recorded address who it is.
pub fn probe(instance: &Instance) -> (State, Option<String>) {
    let url = format!("http://127.0.0.1:{}/api/v1/health", instance.port);
    let Ok(client) = reqwest::blocking::Client::builder()
        .timeout(PROBE_TIMEOUT)
        .build()
    else {
        return (State::Foreign, Some("probe client could not start".into()));
    };
    let Ok(response) = client.get(&url).send() else {
        return (State::Stale, Some("nothing is listening".into()));
    };
    if !response.status().is_success() {
        return (
            State::Foreign,
            Some(format!("health returned HTTP {}", response.status())),
        );
    }
    let Ok(body) = response.json::<Value>() else {
        return (State::Foreign, Some("health is not JSON".into()));
    };
    match body
        .get("instance")
        .and_then(|value| value.get("instance_id"))
        .and_then(Value::as_str)
    {
        Some(id) if id == instance.instance_id => (State::Live, None),
        Some(id) => (
            State::Foreign,
            Some(format!("the port is now served by instance {id}")),
        ),
        None => (
            State::Foreign,
            Some("the server exposes no instance identity (older GLR build)".into()),
        ),
    }
}

/// One registry entry, resolved against the live socket by [`probe`].
#[derive(Debug, Clone, Serialize)]
pub struct Report {
    #[serde(flatten)]
    pub instance: Instance,
    pub state: State,
    pub age_seconds: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// Whether `instance` is serving the project the caller asked about.
    pub serves_this_project: bool,
}

/// Every parseable lease, oldest first, plus the files that would not parse.
///
/// A truncated record left by a killed writer is reported rather than fatal:
/// enumeration that dies because one file is corrupt is enumeration nobody can
/// use to clean up the corrupt file.
fn read_leases(directory: &Path) -> Result<(Vec<Leased>, Vec<Value>)> {
    let mut entries = Vec::new();
    let mut unreadable = Vec::new();
    if !directory.is_dir() {
        return Ok((entries, unreadable));
    }
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        match fs::read(&path).and_then(|bytes| {
            serde_json::from_slice::<Instance>(&bytes)
                .map_err(|error| std::io::Error::other(error.to_string()))
        }) {
            Ok(instance) => entries.push((path, instance)),
            Err(error) => unreadable.push(json!({
                "path": path.to_string_lossy(),
                "error": error.to_string(),
            })),
        }
    }
    entries.sort_by_key(|(_, instance)| instance.started_at_ms);
    Ok((entries, unreadable))
}

/// A lease file paired with the record it holds.
type Leased = (PathBuf, Instance);

/// Live servers serving `project`, oldest first.
pub fn live_for(directory: &Path, project: &Project) -> Result<Vec<Instance>> {
    let (entries, _) = read_leases(directory)?;
    Ok(entries
        .into_iter()
        .map(|(_, instance)| instance)
        .filter(|instance| instance.serves(project) && probe(instance).0 == State::Live)
        .collect())
}

/// The lease recorded for `port`.
pub fn find_by_port(directory: &Path, port: u16) -> Result<Instance> {
    let (entries, _) = read_leases(directory)?;
    entries
        .into_iter()
        .find(|(_, instance)| instance.port == port)
        .map(|(_, instance)| instance)
        .ok_or_else(|| {
            Error::Invalid(format!(
                "no workbench server is registered on port {port}; `glr dashboard instances --all` \
                 lists every server this user has"
            ))
        })
}

/// Read every lease in `directory`, probe it, and optionally forget the dead.
///
/// `prune` removes only [`State::Stale`] entries. Anything that still answers
/// is left alone, and this command never terminates anything.
pub fn survey(directory: &Path, project: Option<&Project>, prune: bool) -> Result<Value> {
    let (entries, mut unreadable) = read_leases(directory)?;
    let mut pruned = Vec::new();
    let now = now_ms();
    let mut reports = Vec::new();
    for (path, instance) in entries {
        let (state, detail) = probe(&instance);
        if prune && state == State::Stale {
            if let Err(error) = fs::remove_file(&path) {
                unreadable.push(json!({
                    "path": path.to_string_lossy(),
                    "error": error.to_string(),
                }));
                continue;
            }
            pruned.push(instance.instance_id.clone());
            continue;
        }
        reports.push(Report {
            age_seconds: now.saturating_sub(instance.started_at_ms) / 1000,
            serves_this_project: project.is_some_and(|project| instance.serves(project)),
            state,
            detail,
            instance,
        });
    }
    let scoped: Vec<&Report> = match project {
        Some(_) => reports.iter().filter(|r| r.serves_this_project).collect(),
        None => reports.iter().collect(),
    };
    Ok(json!({
        "schema_version": SCHEMA,
        "scope": if project.is_some() { "project" } else { "user" },
        "registry": directory.to_string_lossy(),
        "instances": scoped,
        "live": scoped.iter().filter(|r| r.state == State::Live).count(),
        "stale": scoped.iter().filter(|r| r.state == State::Stale).count(),
        "unattributed": scoped.iter().filter(|r| r.state == State::Foreign).count(),
        "pruned": pruned,
        "unreadable": unreadable,
        "note": "A lease counts as live only when its recorded address answers /api/v1/health \
                 with the same instance_id; the state is the socket's answer, never a guess from \
                 a pid. --prune forgets addresses that stopped answering and leaves everything \
                 else untouched. No process is ever terminated by this command.",
    }))
}

/// Ask one live instance to stop, after proving the address still belongs to it.
///
/// Fails closed at every step: an unlistening address, or a port that now
/// answers as somebody else, returns an error instead of a shutdown request.
/// The server stops itself; this process only sends the request.
pub fn request_stop(instance: &Instance) -> Result<()> {
    match probe(instance) {
        (State::Live, _) => {}
        (State::Stale, detail) => {
            return Err(Error::Invalid(format!(
                "instance {} is not listening on {}; nothing to stop ({})",
                instance.instance_id,
                instance.url,
                detail.unwrap_or_else(|| "no response".into())
            )));
        }
        (State::Foreign, detail) => {
            return Err(Error::Invalid(format!(
                "refusing to stop {}: {} is no longer served by it ({})",
                instance.instance_id,
                instance.url,
                detail.unwrap_or_else(|| "identity mismatch".into())
            )));
        }
    }
    let origin = instance.url.trim_end_matches('/').to_string();
    let response = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()?
        .post(format!("{origin}/api/v1/control/shutdown"))
        .header("Origin", &origin)
        .json(&json!({}))
        .send()?;
    if !response.status().is_success() {
        return Err(Error::Invalid(format!(
            "{} refused the shutdown request with HTTP {}",
            instance.instance_id,
            response.status()
        )));
    }
    Ok(())
}

/// Load one lease by instance id.
pub fn find(directory: &Path, instance_id: &str) -> Result<Instance> {
    if !instance_id
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
    {
        return Err(Error::Invalid(
            "instance id must be ASCII letters, digits or -".into(),
        ));
    }
    let path = directory.join(format!("{instance_id}.json"));
    let bytes = fs::read(&path).map_err(|_| Error::Missing(path.clone()))?;
    Ok(serde_json::from_slice(&bytes)?)
}

/// Live instances started from the same executable, newest first.
///
/// Used by `glr update` to name what is holding a locked image instead of
/// reporting a bare `os error 5`.
pub fn holders(directory: &Path, executable: &Path) -> Vec<Report> {
    let wanted = storage_key(executable);
    let now = now_ms();
    let mut found = Vec::new();
    let Ok((entries, _)) = read_leases(directory) else {
        return found;
    };
    for (_, instance) in entries {
        if storage_key(Path::new(&instance.executable)) != wanted {
            continue;
        }
        let (state, detail) = probe(&instance);
        if state != State::Live {
            continue;
        }
        found.push(Report {
            age_seconds: now.saturating_sub(instance.started_at_ms) / 1000,
            serves_this_project: false,
            state,
            detail,
            instance,
        });
    }
    found.sort_by_key(|report| std::cmp::Reverse(report.instance.started_at_ms));
    found
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};

    use super::*;

    fn sample(port: u16) -> Instance {
        let data_dir = Path::new("/tmp/sample/.glr");
        Instance {
            schema_version: SCHEMA.into(),
            // A fresh id per call: two samples that share one would write the
            // same lease file and quietly test a registry of size one.
            instance_id: uuid::Uuid::new_v4().simple().to_string(),
            environment_id: "sample.env-v1".into(),
            project_root: "/tmp/sample".into(),
            data_dir: data_dir.to_string_lossy().into_owned(),
            data_dir_sha256: storage_digest(data_dir),
            executable: "/usr/local/bin/glr".into(),
            pid: 4242,
            port,
            url: format!("http://127.0.0.1:{port}/"),
            read_only: false,
            version: "0.20.0".into(),
            started_at_ms: 1_758_000_000_000,
            started_at: utc_rfc3339(1_758_000_000_000),
        }
    }

    /// The same record, but listening where `port` says.
    fn moved(instance: &Instance, port: u16) -> Instance {
        Instance {
            port,
            url: format!("http://127.0.0.1:{port}/"),
            ..instance.clone()
        }
    }

    fn free_port() -> u16 {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        port
    }

    /// A one-shot loopback responder, so a probe can be tested against a real
    /// socket without adding an HTTP server dependency.
    fn responder(body: &str) -> u16 {
        let body = body.to_string();
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for stream in listener.incoming().take(8) {
                let Ok(mut stream) = stream else { continue };
                let mut request = [0_u8; 2048];
                let _ = stream.read(&mut request);
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: \
                     {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes());
            }
        });
        port
    }

    #[test]
    fn wall_clock_formatting_matches_known_instants() {
        for (milliseconds, expected) in [
            (0_i64, "1970-01-01T00:00:00.000Z"),
            (68_169_600_000, "1972-02-29T00:00:00.000Z"),
            (951_782_400_000, "2000-02-29T00:00:00.000Z"),
            (951_868_800_000, "2000-03-01T00:00:00.000Z"),
            (1_758_000_000_000, "2025-09-16T05:20:00.000Z"),
            (1_774_000_000_000, "2026-03-20T09:46:40.000Z"),
            (4_102_444_800_000, "2100-01-01T00:00:00.000Z"),
            (1_758_000_000_123, "2025-09-16T05:20:00.123Z"),
        ] {
            assert_eq!(utc_rfc3339(milliseconds), expected, "{milliseconds}");
        }
    }

    #[test]
    fn an_explicit_port_never_moves_and_an_implicit_one_walks() {
        assert_eq!(port_candidates(Some(7432), Path::new("/tmp/a")), vec![7432]);
        assert_eq!(port_candidates(Some(0), Path::new("/tmp/a")), vec![0]);

        let candidates = port_candidates(None, Path::new("/tmp/a"));
        assert_eq!(candidates.len(), 3);
        assert_eq!(candidates[0], PREFERRED_PORT);
        assert_eq!(candidates[2], 0);
        assert!((PREFERRED_PORT + 1..PREFERRED_PORT + BAND).contains(&candidates[1]));

        // An explicit request is answered exactly, even when the address is
        // already held: moving the server would invalidate the number the
        // operator just wrote down.
        let (held, port) = bind(Some(0), Path::new("/tmp/a")).unwrap();
        assert!(bind(Some(port), Path::new("/tmp/a")).is_err());
        drop(held);

        // The implicit list has somewhere to go. Skipped when this host's
        // default slot belongs to a real server, which is the case the list
        // exists for in the first place.
        let Ok((default_holder, default_port)) = bind(Some(PREFERRED_PORT), Path::new("/tmp/a"))
        else {
            return;
        };
        assert_eq!(default_port, PREFERRED_PORT);
        let (listener, bound) = bind(None, Path::new("/tmp/a")).unwrap();
        assert_ne!(bound, PREFERRED_PORT);
        drop(listener);
        drop(default_holder);
    }

    #[test]
    fn the_derived_slot_is_stable_per_storage() {
        let one = port_candidates(None, Path::new("/tmp/one/.glr"))[1];
        assert_eq!(one, port_candidates(None, Path::new("/tmp/one/.glr"))[1]);
        // The slot must not move because of how the path was spelled.
        assert_eq!(one, port_candidates(None, Path::new(r"\tmp\one\.glr"))[1]);
        let distinct = (0..64)
            .filter(|index| {
                port_candidates(None, &PathBuf::from(format!("/tmp/p{index}/.glr")))[1] != one
            })
            .count();
        assert!(
            distinct > 40,
            "derived slots collapsed: {distinct}/64 distinct"
        );
    }

    #[test]
    fn a_lease_round_trips_and_removes_itself() {
        let directory = tempfile::tempdir().unwrap();
        let mine = sample(free_port());
        let path = directory.path().join(format!("{}.json", mine.instance_id));
        let lease = Lease::publish(directory.path(), mine.clone()).unwrap();
        assert!(path.is_file());

        let listing = survey(directory.path(), None, false).unwrap();
        assert_eq!(listing["instances"].as_array().unwrap().len(), 1);
        assert_eq!(listing["instances"][0]["state"], "stale");
        assert_eq!(listing["instances"][0]["instance_id"], mine.instance_id);
        assert_eq!(
            listing["instances"][0]["started_at"],
            "2025-09-16T05:20:00.000Z"
        );

        lease.release();
        assert!(!path.exists(), "a released lease must not linger");
        assert_eq!(
            survey(directory.path(), None, false).unwrap()["instances"]
                .as_array()
                .unwrap()
                .len(),
            0
        );
    }

    #[test]
    fn an_instance_is_live_only_when_its_own_id_answers() {
        // A port that answers, but as somebody else.
        let directory = tempfile::tempdir().unwrap();
        let stranger =
            responder(r#"{"instance":{"instance_id":"ffffffffffffffffffffffffffffffff"}}"#);
        let _stranger = Lease::publish(directory.path(), sample(stranger)).unwrap();
        let listing = survey(directory.path(), None, false).unwrap();
        assert_eq!(listing["instances"][0]["state"], "foreign");
        assert_eq!(listing["unattributed"], 1);
        assert_eq!(listing["live"], 0);
        assert!(
            listing["instances"][0]["detail"]
                .as_str()
                .unwrap()
                .contains("ffffffffffffffffffffffffffffffff"),
            "{listing}"
        );

        // The same address answering as *this* instance is live.
        let directory = tempfile::tempdir().unwrap();
        let mine = sample(free_port());
        let owned = responder(&format!(
            r#"{{"instance":{{"instance_id":"{}"}}}}"#,
            mine.instance_id
        ));
        let mine = moved(&mine, owned);
        let _mine = Lease::publish(directory.path(), mine.clone()).unwrap();
        let listing = survey(directory.path(), None, false).unwrap();
        assert_eq!(listing["instances"][0]["state"], "live");
        assert_eq!(listing["instances"][0]["instance_id"], mine.instance_id);
        assert_eq!(listing["live"], 1);
    }

    #[test]
    fn pruning_forgets_only_addresses_that_stopped_answering() {
        let directory = tempfile::tempdir().unwrap();
        let gone = sample(free_port());
        let _gone = Lease::publish(directory.path(), gone.clone()).unwrap();
        let answered = sample(responder(
            r#"{"instance":{"instance_id":"ffffffffffffffffffffffffffffffff"}}"#,
        ));
        let _answered = Lease::publish(directory.path(), answered.clone()).unwrap();

        let listing = survey(directory.path(), None, true).unwrap();
        let ids: Vec<&str> = listing["instances"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["instance_id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, vec![answered.instance_id.as_str()], "{listing}");
        assert_eq!(listing["pruned"], json!([gone.instance_id]), "{listing}");
        assert_eq!(listing["stale"], 0);
    }

    #[test]
    fn an_unreadable_lease_does_not_hide_the_readable_ones() {
        let directory = tempfile::tempdir().unwrap();
        let lease = Lease::publish(directory.path(), sample(free_port())).unwrap();
        fs::write(directory.path().join("truncated.json"), b"{").unwrap();
        let listing = survey(directory.path(), None, false).unwrap();
        assert_eq!(listing["instances"].as_array().unwrap().len(), 1);
        assert_eq!(listing["unreadable"].as_array().unwrap().len(), 1);
        lease.release();
    }

    #[test]
    fn stop_refuses_what_it_cannot_prove() {
        let nothing = request_stop(&sample(free_port())).unwrap_err();
        assert!(nothing.to_string().contains("nothing to stop"), "{nothing}");

        let stranger = sample(responder(r#"{"instance":{"instance_id":"beef"}}"#));
        let refused = request_stop(&stranger).unwrap_err();
        assert!(
            refused.to_string().contains("no longer served"),
            "{refused}"
        );
    }

    #[test]
    fn an_id_that_could_escape_the_registry_is_rejected() {
        let directory = tempfile::tempdir().unwrap();
        for hostile in ["../secrets", "a/b", ".hidden", "x.json"] {
            assert!(find(directory.path(), hostile).is_err(), "{hostile}");
        }
        assert!(find(directory.path(), "0123456789abcdef0123456789abcdef").is_err());
    }

    #[test]
    fn only_instances_sharing_the_executable_are_reported_as_holders() {
        let directory = tempfile::tempdir().unwrap();
        let port = responder(r#"{"instance":{}}"#);
        let _lease = Lease::publish(directory.path(), sample(port)).unwrap();
        let mine = Path::new("/usr/local/bin/glr");
        // The responder carries no identity, so nothing is live and nothing is
        // claimed to hold the image.
        assert!(holders(directory.path(), mine).is_empty());
    }

    #[test]
    fn the_registry_directory_honours_the_state_override() {
        let directory = tempfile::tempdir().unwrap();
        // SAFETY: no other test reads this variable; every function that needs
        // a registry takes it as an argument.
        unsafe { std::env::set_var("GLR_STATE_DIR", directory.path()) };
        let resolved = lease_dir().unwrap();
        unsafe { std::env::remove_var("GLR_STATE_DIR") };
        assert_eq!(
            resolved,
            directory
                .path()
                .join("glr")
                .join("workbench")
                .join("instances")
        );
    }
}
