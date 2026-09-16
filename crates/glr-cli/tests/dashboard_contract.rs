use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::time::{Duration, Instant};

use reqwest::blocking::Client;
use rusqlite::{Connection, params};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

fn register_media(project: &Path, run_id: &str, path: &str, role: &str, data: &[u8]) -> String {
    let target = project.join(".glr/runs").join(run_id).join(path);
    fs::create_dir_all(target.parent().unwrap()).unwrap();
    fs::write(&target, data).unwrap();
    let digest = format!("{:x}", Sha256::digest(data));
    let db = Connection::open(project.join(".glr/runs.sqlite3")).unwrap();
    db.execute("INSERT INTO artifacts(run_id,path,role,media_type,sha256,size_bytes,metadata_json) VALUES(?,?,?,'application/octet-stream',?,?,'{}')",params![run_id,path,role,digest,data.len() as i64]).unwrap();
    digest
}

#[test]
fn media_streams_registered_artifacts_and_fences_ranges_paths_and_frame_bindings() {
    let project = project();
    let id = seed(project.path());
    let digest = register_media(
        project.path(),
        &id,
        "recording.mp4",
        "review-video",
        b"0123456789",
    );
    register_media(
        project.path(),
        &id,
        "notes.md",
        "notes",
        b"# Training notes\n\nRecorded result.",
    );
    register_media(
        project.path(),
        &id,
        "unsafe.html",
        "report",
        b"<script>alert(1)</script>",
    );
    let manifest = json!({"schema_version":"glr.capture.v1","environment_id":"test.dashboard","run_id":id,"video":{"path":"recording.mp4","sha256":digest,"size_bytes":10},"index":{"path":"capture-index.jsonl","sha256":"unused","size_bytes":0},"codec":"h264","frame_rate":30.0,"width":100,"height":100,"frames":[{"schema_version":"glr.capture-frame.v1","run_id":id,"episode_id":"ep-one","step_id":4,"frame_index":0,"pts_ns":1_000_000_000u64,"observation_timestamp_ns":1}]});
    register_media(
        project.path(),
        &id,
        "capture.manifest.json",
        "capture-manifest",
        &serde_json::to_vec(&manifest).unwrap(),
    );
    let server = Service::start(project.path(), "observe");
    let url = format!(
        "{}/api/v1/media/file?run={id}&path=recording.mp4",
        server.url
    );
    let head = server.client.head(&url).send().unwrap();
    assert_eq!(head.status(), 200);
    assert_eq!(head.headers()["content-length"], "10");
    assert_eq!(head.headers()["accept-ranges"], "bytes");
    assert_eq!(head.text().unwrap(), "");
    for (range, expected, content_range) in [
        ("bytes=2-5", "2345", "bytes 2-5/10"),
        ("bytes=-3", "789", "bytes 7-9/10"),
        ("bytes=8-", "89", "bytes 8-9/10"),
    ] {
        let response = server
            .client
            .get(&url)
            .header("Range", range)
            .send()
            .unwrap();
        assert_eq!(response.status(), 206);
        assert_eq!(response.headers()["content-range"], content_range);
        assert_eq!(response.headers()["content-type"], "video/mp4");
        assert_eq!(response.text().unwrap(), expected);
    }
    for range in ["bytes=10-", "bytes=6-2", "bytes=0-1,4-5", "bytes=-0"] {
        let response = server
            .client
            .get(&url)
            .header("Range", range)
            .send()
            .unwrap();
        assert_eq!(response.status(), 416);
        assert_eq!(response.headers()["content-range"], "bytes */10");
    }
    let full = server
        .client
        .get(&url)
        .header("Range", "bytes=2-5")
        .header("If-Range", "\"old\"")
        .send()
        .unwrap();
    assert_eq!(full.status(), 200);
    assert_eq!(full.text().unwrap(), "0123456789");
    assert_eq!(
        server
            .client
            .get(&url)
            .header("Origin", "https://foreign.invalid")
            .send()
            .unwrap()
            .status(),
        403
    );
    assert_eq!(server.client.post(&url).send().unwrap().status(), 405);
    for path in [
        "unregistered.png",
        "..%2F..%2Fruns.sqlite3",
        "recording.mp4&path=notes.md",
    ] {
        assert_eq!(
            server
                .get(&format!("/api/v1/media/file?run={id}&path={path}"))
                .status(),
            400
        );
    }
    let html = server.get(&format!("/api/v1/media/file?run={id}&path=unsafe.html"));
    assert_eq!(html.headers()["content-disposition"], "attachment");
    assert_eq!(html.headers()["content-type"], "application/octet-stream");
    let document: Value = server
        .get(&format!("/api/v1/media/document?run={id}&path=notes.md"))
        .json()
        .unwrap();
    assert!(
        document["text"]
            .as_str()
            .unwrap()
            .starts_with("# Training notes")
    );
    let frames_url =
        format!("/api/v1/media/frames?run={id}&manifest=capture.manifest.json&video=recording.mp4");
    let frames: Value = server.get(&frames_url).json().unwrap();
    assert_eq!(frames["frames"][0]["step_id"], 4);
    assert_eq!(frames["frames"][0]["seconds"], 1.0);
    fs::write(
        project
            .path()
            .join(".glr/runs")
            .join(&id)
            .join("capture.manifest.json"),
        serde_json::to_string(&manifest)
            .unwrap()
            .replace("ep-one", "ep-two"),
    )
    .unwrap();
    assert_eq!(server.get(&frames_url).status(), 400);
    let mut wrong_run = manifest.clone();
    wrong_run["run_id"] = "run-other".into();
    register_media(
        project.path(),
        &id,
        "other.manifest.json",
        "capture-manifest",
        &serde_json::to_vec(&wrong_run).unwrap(),
    );
    assert_eq!(
        server
            .get(&format!(
                "/api/v1/media/frames?run={id}&manifest=other.manifest.json&video=recording.mp4"
            ))
            .status(),
        400
    );
    register_media(
        project.path(),
        &id,
        "long.txt",
        "notes",
        &vec![b'x'; 300 * 1024],
    );
    let bounded: Value = server
        .get(&format!("/api/v1/media/document?run={id}&path=long.txt"))
        .json()
        .unwrap();
    assert_eq!(bounded["text"].as_str().unwrap().len(), 256 * 1024);
    assert_eq!(bounded["truncated"], true);
    // A large registered file must stream beyond the JSON response cap.
    let big = vec![b'x'; 9 * 1024 * 1024];
    register_media(project.path(), &id, "large.webm", "review-video", &big);
    let response = server.get(&format!("/api/v1/media/file?run={id}&path=large.webm"));
    assert_eq!(response.status(), 200);
    assert_eq!(response.bytes().unwrap().len(), big.len());
    // Catalog pagination is bounded even for a run with many images.
    for index in 0..105 {
        register_media(
            project.path(),
            &id,
            &format!("frames/{index:03}.png"),
            "frame",
            b"image",
        );
    }
    let page: Value = server
        .get(&format!("/api/v1/media?run={id}"))
        .json()
        .unwrap();
    assert_eq!(page["items"].as_array().unwrap().len(), 100);
    assert!(page["next_after"].is_string());
    let second: Value = server
        .get(&format!(
            "/api/v1/media?run={id}&after={}",
            page["next_after"].as_str().unwrap()
        ))
        .json()
        .unwrap();
    assert!(second["items"].as_array().unwrap().len() < 100);
    let db = Connection::open(project.path().join(".glr/runs.sqlite3")).unwrap();
    db.execute(
        "UPDATE runs SET environment_id='another.environment' WHERE run_id=?",
        [&id],
    )
    .unwrap();
    assert_eq!(server.client.get(&url).send().unwrap().status(), 400);
}

fn binary() -> PathBuf {
    env!("CARGO_BIN_EXE_glr").into()
}
fn project() -> TempDir {
    let dir = tempfile::tempdir().unwrap();
    fs::create_dir(dir.path().join("bridge")).unwrap();
    fs::write(dir.path().join("glr-project.json"),serde_json::to_vec(&json!({
        "schema_version":"glr.project.v1","environment_id":"test.dashboard","environment_family":"test","protocol_version":"1.0",
        "data_dir":".glr","bridge_path":"bridge",
        "runtime":{"argv":[binary(),"--version"]},"trainer":{"argv":[binary(),"--version"]},"player":{"argv":[binary(),"--version"]}
    })).unwrap()).unwrap();
    dir
}
fn run(project: &Path, args: &[&str]) -> Output {
    Command::new(binary())
        .env("GLR_NO_UPDATE_CHECK", "1")
        .args(["--json", "--project"])
        .arg(project)
        .args(args)
        .output()
        .unwrap()
}
fn success(output: Output) -> Value {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}
fn seed(project: &Path) -> String {
    let result = success(run(project, &["train", "--no-observe"]));
    result["data"]["run_id"].as_str().unwrap().to_string()
}
struct Service {
    child: Child,
    url: String,
    client: Client,
}
impl Service {
    fn start(project: &Path, mode: &str) -> Self {
        let mut child = Command::new(binary())
            .env("GLR_NO_UPDATE_CHECK", "1")
            .env(
                "GLR_TELEMETRY_TOKEN",
                "test-bridge-token-01234567890123456789",
            )
            .args(["--json", "--project"])
            .arg(project)
            .args([mode, "--port", "0"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let stdout = child.stdout.take().unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut line = String::new();
            let _ = BufReader::new(stdout).read_line(&mut line);
            let _ = tx.send(line);
        });
        let line = rx
            .recv_timeout(Duration::from_secs(15))
            .expect("service must announce readiness");
        let value: Value = serde_json::from_str(&line).unwrap();
        let url = value["data"]["url"]
            .as_str()
            .unwrap()
            .trim_end_matches('/')
            .to_string();
        Self {
            child,
            url,
            client: Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .unwrap(),
        }
    }
    fn get(&self, path: &str) -> reqwest::blocking::Response {
        self.client
            .get(format!("{}{path}", self.url))
            .send()
            .unwrap()
    }
    fn post(&self, path: &str, body: &Value) -> reqwest::blocking::Response {
        self.client
            .post(format!("{}{path}", self.url))
            .header("Origin", &self.url)
            .json(body)
            .send()
            .unwrap()
    }
}
impl Drop for Service {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn observer_paginates_live_history_logs_and_enforces_read_only_loopback() {
    let project = project();
    let id = seed(project.path());
    let mut db = Connection::open(project.path().join(".glr/runs.sqlite3")).unwrap();
    let tx = db.transaction().unwrap();
    for i in 1..=1005 {
        tx.execute("INSERT INTO events(run_id,sequence_id,timestamp_ns,kind,step_id,payload_json) VALUES(?,?,1,'agent.decision',?,?)",params![id,i,i,json!({"selected_key":"walk","position":[i,2]}).to_string()]).unwrap();
        tx.execute("INSERT INTO metrics(run_id,timestamp_ns,name,value,step_id,metadata_json) VALUES(?,1,'loss',?,?, '{}')",params![id,1.0/i as f64,i]).unwrap();
    }
    tx.commit().unwrap();
    let server = Service::start(project.path(), "observe");
    let mut cursor = 0;
    let mut metrics = 0;
    let mut count = 0;
    loop {
        let page: Value = server
            .get(&format!(
                "/api/v1/snapshot?run={id}&events_after={cursor}&metrics_after={metrics}"
            ))
            .json()
            .unwrap();
        let rows = page["events"].as_array().unwrap();
        if rows.is_empty() {
            break;
        }
        assert_eq!(rows[0]["sequence_id"], cursor + 1);
        count += rows.len();
        cursor = page["cursor"]["events_after"].as_i64().unwrap();
        metrics = page["cursor"]["metrics_after"].as_i64().unwrap();
    }
    assert_eq!(count, 1005);
    assert_eq!(metrics, 1005);
    db.execute("INSERT INTO events(run_id,sequence_id,timestamp_ns,kind,payload_json) VALUES(?,1006,1,'learning.update',?)",params![id,json!({"large":"x".repeat(20000)}).to_string()]).unwrap();
    let page: Value = server
        .get(&format!("/api/v1/snapshot?run={id}&events_after=1005"))
        .json()
        .unwrap();
    assert_eq!(page["events"][0]["payload"]["observation_truncated"], true);
    let log = project
        .path()
        .join(".glr/runs")
        .join(&id)
        .join("capture.log");
    fs::write(&log, b"frame=1\rfps=30\n").unwrap();
    let data: Value = server
        .get(&format!("/api/v1/log?run={id}&path=capture.log&offset=0"))
        .json()
        .unwrap();
    assert_eq!(data["text"], "frame=1\rfps=30\n");
    fs::write(log, b"new").unwrap();
    let data: Value = server
        .get(&format!("/api/v1/log?run={id}&path=capture.log&offset=15"))
        .json()
        .unwrap();
    assert_eq!(data["reset"], true);
    assert_eq!(
        server
            .get(&format!(
                "/api/v1/log?run={id}&path=..%2F..%2Fglr-project.json"
            ))
            .status(),
        400
    );
    assert_eq!(server.get("/api/v1/snapshot?run=..%2Fsecret").status(), 400);
    assert_eq!(
        server
            .post("/api/v1/control/jobs", &json!({"argv":["train"]}))
            .status(),
        405
    );
    assert_eq!(
        server
            .client
            .get(format!("{}/api/v1/runs", server.url))
            .header("Host", "evil.invalid")
            .send()
            .unwrap()
            .status(),
        403
    );
    assert_eq!(
        server
            .client
            .get(format!("{}/api/v1/runs", server.url))
            .header("Origin", "https://evil.invalid")
            .send()
            .unwrap()
            .status(),
        403
    );
    let html = server.get("/").text().unwrap();
    assert!(html.contains("id=\"root\""));
    let health: Value = server.get("/api/v1/health").json().unwrap();
    assert_eq!(
        health["dashboard_source_sha256"].as_str().unwrap().len(),
        64
    );
    let style = html
        .split("href=\"")
        .nth(1)
        .unwrap()
        .split('"')
        .next()
        .unwrap();
    assert!(style.starts_with("/assets/") && style.ends_with(".css"));
    let css = server.get(style);
    assert_eq!(css.status(), 200);
    assert_eq!(css.headers()["content-type"], "text/css; charset=utf-8");
    assert_eq!(server.get("/assets/missing.js").status(), 404);
    let script = html
        .split("src=\"")
        .nth(1)
        .unwrap()
        .split('"')
        .next()
        .unwrap();
    assert!(script.starts_with("/assets/"));
    let response = server.get(script);
    assert_eq!(response.status(), 200);
    assert!(response.headers().contains_key("content-security-policy"));
    let etag = response.headers()["etag"].clone();
    assert_eq!(
        server
            .client
            .get(format!("{}{script}", server.url))
            .header("If-None-Match", etag)
            .send()
            .unwrap()
            .status(),
        304
    );
    let trace = success(run(
        project.path(),
        &["runs", "trace", &id, "--events-after", "1004"],
    ));
    assert_eq!(trace["data"]["events"].as_array().unwrap().len(), 2);
    // Reports must not silently stop at the original 1000-record store limit.
    success(run(project.path(), &["report", "build", &id]));
    let html = fs::read_to_string(
        project
            .path()
            .join(".glr/runs")
            .join(&id)
            .join("report/index.html"),
    )
    .unwrap();
    assert!(html.contains("\"sequence_id\":1005"));
}

#[test]
fn dashboard_launches_one_idempotent_training_job_and_persists_presets() {
    let project = project();
    let server = Service::start(project.path(), "dashboard");
    let catalog: Value = server.get("/api/v1/control/catalog").json().unwrap();
    assert!(
        catalog["data"]["commands"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c["path"] == json!(["package", "import"]))
    );
    let request = json!({"preset":"train.default","request_id":"request-test"});
    let first: Value = server
        .post("/api/v1/control/jobs", &request)
        .json()
        .unwrap();
    let id = first["data"]["id"].as_str().unwrap();
    let second: Value = server
        .post("/api/v1/control/jobs", &request)
        .json()
        .unwrap();
    assert_eq!(second["data"]["id"], id);
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let jobs: Value = server.get("/api/v1/control/jobs").json().unwrap();
        if jobs["data"]["jobs"][0]["status"] == "succeeded" {
            break;
        }
        assert!(Instant::now() < deadline, "job did not finish: {jobs}");
        std::thread::sleep(Duration::from_millis(50));
    }
    let runs: Value = server.get("/api/v1/runs").json().unwrap();
    assert_eq!(runs["runs"].as_array().unwrap().len(), 1);
    assert_eq!(runs["runs"][0]["metadata"]["dashboard_job_id"], id);
    assert_eq!(
        server
            .post(
                "/api/v1/control/jobs",
                &json!({"argv":["doctor"],"request_id":"request-test"})
            )
            .status(),
        400
    );
    assert_eq!(
        server
            .post(
                "/api/v1/control/jobs",
                &json!({"argv":["train","--project","elsewhere"],"request_id":"request-other"})
            )
            .status(),
        400
    );
    assert_eq!(
        server
            .client
            .post(format!("{}/api/v1/control/jobs", server.url))
            .json(&request)
            .send()
            .unwrap()
            .status(),
        403
    );
    assert_eq!(
        server
            .post(
                "/api/v1/control/jobs",
                &json!({"argv":["dashboard"],"request_id":"request-nested"})
            )
            .status(),
        400
    );
    let preset = json!({"schema_version":"glr.training-preset.v1","id":"train.fast","title":"Fast","description":"Explicit recording opt-out","argv":["train","--no-capture"]});
    assert!(
        server
            .post("/api/v1/control/presets", &preset)
            .status()
            .is_success()
    );
    drop(server);
    let list = success(run(project.path(), &["dashboard", "presets"]));
    assert_eq!(list["data"]["data"].as_array().unwrap().len(), 2);
    let list = success(run(project.path(), &["dashboard", "jobs"]));
    assert_eq!(list["data"]["data"]["jobs"][0]["status"], "succeeded");
}

#[test]
fn online_backup_restores_history_without_overwriting_and_detects_tampering() {
    let project = project();
    let id = seed(project.path());
    let db = Connection::open(project.path().join(".glr/runs.sqlite3")).unwrap();
    db.execute_batch("PRAGMA journal_mode=WAL; PRAGMA user_version=2;")
        .unwrap();
    db.execute("INSERT INTO runs(run_id,environment_id,protocol_version,kind,status,started_at_ns,metadata_json) VALUES('run-active','test.dashboard','1.0','training','running',1,'{}')",[]).unwrap();
    db.execute("INSERT INTO events(run_id,sequence_id,timestamp_ns,kind,payload_json) VALUES('run-active',1,1,'learning.update','{}')",[]).unwrap();
    let root = tempfile::tempdir().unwrap();
    let archive = root.path().join("backup");
    let archive_s = archive.to_str().unwrap();
    let result = success(run(
        project.path(),
        &["backup", "create", "--output", archive_s],
    ));
    assert_eq!(
        result["data"]["active_runs_database_only"],
        json!(["run-active"])
    );
    assert!(!archive.join("runs/run-active").exists());
    assert!(archive.join("runs").join(&id).join("trainer.log").is_file());
    assert!(
        success(run(project.path(), &["backup", "verify", archive_s]))["data"]["verified"]
            .as_bool()
            .unwrap()
    );
    let trace = success(run(
        project.path(),
        &["runs", "trace", "run-active", "--archive", archive_s],
    ));
    assert_eq!(trace["data"]["events"][0]["kind"], "learning.update");
    let restored = root.path().join("restored");
    let restored_s = restored.to_str().unwrap();
    success(run(
        project.path(),
        &["backup", "restore", archive_s, "--output", restored_s],
    ));
    assert!(
        !run(
            project.path(),
            &["backup", "restore", archive_s, "--output", restored_s]
        )
        .status
        .success()
    );
    fs::write(restored.join("unregistered.txt"), b"not in manifest").unwrap();
    assert!(
        !run(project.path(), &["backup", "verify", restored_s])
            .status
            .success()
    );
    fs::write(
        archive.join("runs").join(id).join("trainer.log"),
        b"tampered",
    )
    .unwrap();
    assert!(
        !run(project.path(), &["backup", "verify", archive_s])
            .status
            .success()
    );
}

#[test]
fn automatic_observation_and_console_logs_do_not_pollute_json_stdout() {
    let project = project();
    let output = run(project.path(), &["train"]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("GLR observation:"));
    assert!(stderr.contains("[trainer.log]"));
    success(output);
}

#[test]
fn history_cursors_preserve_ties_and_stale_jobs_remain_unverified() {
    let project = project();
    let server = Service::start(project.path(), "dashboard");
    let mut db = Connection::open(project.path().join(".glr/runs.sqlite3")).unwrap();
    let tx = db.transaction().unwrap();
    for index in 0..105 {
        let id = format!("run-{index:03}");
        tx.execute("INSERT INTO runs(run_id,environment_id,protocol_version,kind,status,started_at_ns,metadata_json) VALUES(?,'test.dashboard','1.0','training','succeeded',1,'{}')", [&id]).unwrap();
        let job_id = format!("job-{index:03}");
        let job =
            json!({"id":job_id,"status":"running","created_at_ms":1,"requested_argv":["train"]});
        tx.execute(
            "INSERT INTO dashboard_jobs(id,request_id,created_at_ms,payload) VALUES(?,?,1,?)",
            params![job_id, format!("request-{index:03}"), job.to_string()],
        )
        .unwrap();
    }
    tx.execute("INSERT INTO events(run_id,sequence_id,timestamp_ns,kind,payload_json) VALUES('run-000',0,1,'agent.decision','{}')", []).unwrap();
    tx.commit().unwrap();
    for (endpoint, field, id_field) in [
        ("/api/v1/runs", "runs", "run_id"),
        ("/api/v1/control/jobs", "jobs", "id"),
    ] {
        let mut before = String::new();
        let mut ids = std::collections::HashSet::new();
        loop {
            let value: Value = server.get(&format!("{endpoint}{before}")).json().unwrap();
            let page = value.get("data").unwrap_or(&value);
            for row in page[field].as_array().unwrap() {
                assert!(ids.insert(row[id_field].as_str().unwrap().to_string()));
                if field == "jobs" {
                    assert_eq!(row["observed_status"], "unverified");
                }
            }
            let Some(cursor) = page["next_before"].as_str() else {
                break;
            };
            before = format!("?before={cursor}");
        }
        assert_eq!(ids.len(), 105);
    }
    let trace = success(run(project.path(), &["runs", "trace", "run-000"]));
    assert_eq!(trace["data"]["events"][0]["sequence_id"], 0);
}

#[test]
fn bridge_ingress_is_authenticated_atomic_idempotent_and_shared_with_cli() {
    let project = project();
    let id = seed(project.path());
    let db = Connection::open(project.path().join(".glr/runs.sqlite3")).unwrap();
    db.execute("UPDATE runs SET status='running' WHERE run_id=?", [&id])
        .unwrap();
    let server = Service::start(project.path(), "dashboard");
    let batch = json!({"schema_version":"glr.bridge-telemetry.v1","run_id":id,"source":"bridge.unity","batch_id":"batch-1",
        "events":[{"kind":"bridge.status","step_id":3,"payload":{"state":"ready","message":"Synthetic provider"}},
        {"kind":"navigation.route_sample","episode_id":"episode-1","step_id":3,"payload":{"position":[1,2,3]}}],
        "metrics":[{"name":"bridge.latency_ms","value":2.5,"step_id":3}]});
    let post = |value: &Value| {
        server
            .client
            .post(format!("{}/api/v1/telemetry", server.url))
            .bearer_auth("test-bridge-token-01234567890123456789")
            .json(value)
            .send()
            .unwrap()
    };
    assert_eq!(server.post("/api/v1/telemetry", &batch).status(), 401);
    let first = post(&batch);
    assert!(
        first.status().is_success(),
        "{}",
        first.text().unwrap_or_default()
    );
    let retry: Value = post(&batch).json().unwrap();
    assert_eq!(retry["duplicate"], true);
    let state: Value = server
        .get(&format!("/api/v1/telemetry/state?run={id}"))
        .json()
        .unwrap();
    assert_eq!(state["states"][0]["payload"]["state"], "ready");
    assert_eq!(state["states"][0]["payload"]["authority"], "diagnostic");
    let mut invalid = batch.clone();
    invalid["batch_id"] = "batch-invalid".into();
    invalid["events"][1]["payload"]["position"] = json!([1]);
    assert_eq!(post(&invalid).status(), 400);
    invalid = batch.clone();
    invalid["events"][0]["payload"]["state"] = "changed".into();
    assert_eq!(post(&invalid).status(), 400);
    let trace = success(run(project.path(), &["runs", "trace", &id]));
    assert_eq!(trace["data"]["events"].as_array().unwrap().len(), 2);
    assert_eq!(trace["data"]["metrics"].as_array().unwrap().len(), 1);
    let schema: Value = server.get("/api/v1/telemetry/schema").json().unwrap();
    assert_eq!(
        schema["properties"]["schema_version"]["const"],
        "glr.bridge-telemetry.v1"
    );
    let latest = success(run(project.path(), &["telemetry", "state", &id]));
    assert_eq!(latest["data"]["states"][0]["source"], "bridge.unity");
    let mut oversized = batch.clone();
    oversized["batch_id"] = "batch-big".into();
    oversized["events"][0]["payload"]["large"] = "a".repeat(13000).into();
    assert_eq!(post(&oversized).status(), 400);
    let mut spoof = batch.clone();
    spoof["batch_id"] = "batch-spoof".into();
    spoof["events"][0]["payload"]["authority"] = "authoritative".into();
    assert_eq!(post(&spoof).status(), 400);
    assert_eq!(
        server
            .client
            .post(format!("{}/api/v1/telemetry", server.url))
            .bearer_auth("test-bridge-token-01234567890123456789")
            .header("Origin", "https://foreign.invalid")
            .json(&batch)
            .send()
            .unwrap()
            .status(),
        403
    );
    let file = project.path().join("batch.json");
    fs::write(&file, serde_json::to_vec(&batch).unwrap()).unwrap();
    let replay = success(run(
        project.path(),
        &["telemetry", "ingest", "--file", file.to_str().unwrap()],
    ));
    assert_eq!(replay["data"]["receipts"][0]["duplicate"], true);
    fs::write(&file, format!("{batch}\n{batch}\n")).unwrap();
    let replay = success(run(
        project.path(),
        &[
            "telemetry",
            "ingest",
            "--file",
            file.to_str().unwrap(),
            "--jsonl",
        ],
    ));
    assert_eq!(replay["data"]["receipts"].as_array().unwrap().len(), 2);
    assert_eq!(
        server
            .client
            .post(format!("{}/api/v1/telemetry", server.url))
            .bearer_auth("test-bridge-token-01234567890123456789")
            .header("Content-Type", "application/json; charset=utf-8")
            .body(batch.to_string())
            .send()
            .unwrap()
            .status(),
        200
    );
    let archive_root = tempfile::tempdir().unwrap();
    let archive = archive_root.path().join("archive");
    success(run(
        project.path(),
        &["backup", "create", "--output", archive.to_str().unwrap()],
    ));
    let archive_db = Connection::open(archive.join("runs.sqlite3")).unwrap();
    let archived: i64 = archive_db
        .query_row("SELECT COUNT(*) FROM telemetry_latest", [], |r| r.get(0))
        .unwrap();
    assert_eq!(archived, 1);
    db.execute("UPDATE runs SET status='succeeded' WHERE run_id=?", [&id])
        .unwrap();
    assert!(post(&batch).status().is_success());
    invalid = batch.clone();
    invalid["batch_id"] = "batch-late".into();
    assert_eq!(post(&invalid).status(), 400);
    let observer = Service::start(project.path(), "observe");
    assert_eq!(observer.post("/api/v1/telemetry", &batch).status(), 405);
}

#[test]
fn workbench_views_are_validated_atomically_and_readable_by_agents() {
    let project = project();
    let id = seed(project.path());
    let db = Connection::open(project.path().join(".glr/runs.sqlite3")).unwrap();
    db.execute("UPDATE runs SET status='running' WHERE run_id=?", [&id])
        .unwrap();
    let fixtures: Value =
        serde_json::from_str(include_str!("../../../docs/examples/workbench-views.json")).unwrap();
    let server = Service::start(project.path(), "dashboard");
    let post = |value: &Value| {
        server
            .client
            .post(format!("{}/api/v1/telemetry", server.url))
            .bearer_auth("test-bridge-token-01234567890123456789")
            .json(value)
            .send()
            .unwrap()
    };
    let mut batch = json!({"schema_version":"glr.bridge-telemetry.v1","run_id":id,"source":"bridge.custom","batch_id":"batch-view", "events":[{"kind":"bridge.state","step_id":4,"payload":{"workbench":fixtures["invalid"][0]}}],"metrics":[{"name":"custom.reward","value":1.0}]});
    assert_eq!(post(&batch).status(), 400);
    let before = success(run(project.path(), &["runs", "trace", &id]));
    assert!(before["data"]["events"].as_array().unwrap().is_empty());
    assert!(before["data"]["metrics"].as_array().unwrap().is_empty());
    batch["events"][0]["payload"]["workbench"] = fixtures["valid"][1].clone();
    assert_eq!(post(&batch).status(), 200);
    let latest = success(run(project.path(), &["telemetry", "state", &id]));
    assert_eq!(
        latest["data"]["states"][0]["payload"]["workbench"],
        fixtures["valid"][1]
    );
    let path = project.path().join("view.json");
    batch["batch_id"] = "batch-view-next".into();
    batch["events"][0]["payload"]["workbench"] = fixtures["valid"][2].clone();
    fs::write(&path, serde_json::to_vec(&batch).unwrap()).unwrap();
    success(run(
        project.path(),
        &["telemetry", "ingest", "--file", path.to_str().unwrap()],
    ));
    let latest: Value = server
        .get(&format!("/api/v1/telemetry/state?run={id}"))
        .json()
        .unwrap();
    assert_eq!(
        latest["states"][0]["payload"]["workbench"],
        fixtures["valid"][2]
    );
    let schema: Value = server.get("/api/v1/telemetry/schema").json().unwrap();
    assert_eq!(
        schema["$defs"]["workbench"]["properties"]["schema_version"]["const"],
        "glr.workbench.v1"
    );
}

#[test]
fn log_history_pages_preserve_utf8_and_reassemble_in_both_directions() {
    let project = project();
    let id = seed(project.path());
    let path = project
        .path()
        .join(".glr/runs")
        .join(&id)
        .join("trainer.log");
    let original = format!(
        "{{\"kind\":\"roster\",\"name\":\"{}\"}}\n",
        "汉".repeat(50000)
    );
    fs::write(&path, &original).unwrap();
    let server = Service::start(project.path(), "observe");
    let url = format!("/api/v1/log?run={id}&path=trainer.log");
    let mut cursor = 0;
    let mut forward = String::new();
    loop {
        let page: Value = server
            .get(&format!("{url}&offset={cursor}"))
            .json()
            .unwrap();
        assert_eq!(page["offset"].as_u64().unwrap(), cursor);
        let text = page["text"].as_str().unwrap();
        assert!(!text.contains('\u{fffd}'));
        forward.push_str(text);
        cursor = page["next_offset"].as_u64().unwrap();
        if page["more"] == false {
            break;
        }
    }
    assert_eq!(forward, original);
    let mut page: Value = server.get(&url).json().unwrap();
    assert_eq!(page["partial_start"], true);
    let mut reverse = page["text"].as_str().unwrap().to_owned();
    while page["offset"].as_u64().unwrap() > 0 {
        let before = page["offset"].as_u64().unwrap();
        page = server
            .get(&format!("{url}&before={before}"))
            .json()
            .unwrap();
        assert_eq!(page["next_offset"].as_u64().unwrap(), before);
        reverse.insert_str(0, page["text"].as_str().unwrap());
    }
    assert_eq!(reverse, original);
    assert_eq!(
        server.get(&format!("{url}&before=1&offset=0")).status(),
        400
    );
    fs::write(&path, b"x\xe6\xb1").unwrap();
    let page: Value = server.get(&format!("{url}&offset=0")).json().unwrap();
    assert_eq!(page["text"], "x");
    assert_eq!(page["next_offset"], 1);
    fs::write(&path, "x汉\n").unwrap();
    let page: Value = server.get(&format!("{url}&offset=1")).json().unwrap();
    assert_eq!(page["text"], "汉\n");
    let reset: Value = server.get(&format!("{url}&before=9999")).json().unwrap();
    assert_eq!(reset["reset"], true);
}
