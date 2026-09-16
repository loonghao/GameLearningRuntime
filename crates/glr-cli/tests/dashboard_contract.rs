use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::time::{Duration, Instant};

use reqwest::blocking::Client;
use rusqlite::{Connection, params};
use serde_json::{Value, json};
use tempfile::TempDir;

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
    let response = server.get("/app.js");
    assert!(response.headers().contains_key("content-security-policy"));
    let etag = response.headers()["etag"].clone();
    assert_eq!(
        server
            .client
            .get(format!("{}/app.js", server.url))
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
