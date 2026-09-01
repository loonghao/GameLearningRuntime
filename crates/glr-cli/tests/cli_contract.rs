use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use rusqlite::{Connection, params};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_glr"))
}

fn create_project() -> TempDir {
    let temporary = tempfile::tempdir().unwrap();
    fs::create_dir(temporary.path().join("bridge")).unwrap();
    let executable = binary().to_string_lossy().into_owned();
    let project = json!({
        "schema_version": "glr.project.v1",
        "environment_id": "example.adventure-v1",
        "environment_family": "action-rpg",
        "protocol_version": "1.0",
        "data_dir": ".glr",
        "bridge_path": "bridge",
        "runtime": {"argv": [executable, "--version"]},
        "trainer": {"argv": [executable, "--version"]},
        "player": {"argv": [executable, "--version"]},
        "researcher": null,
        "planner": null,
        "evaluator": null,
        "capture": null
    });
    fs::write(
        temporary.path().join("glr-project.json"),
        serde_json::to_vec_pretty(&project).unwrap(),
    )
    .unwrap();
    temporary
}

fn run(project: &Path, arguments: &[&str]) -> Output {
    Command::new(binary())
        .arg("--project")
        .arg(project)
        .arg("--json")
        .args(arguments)
        .output()
        .unwrap()
}

fn stdout(output: &Output) -> Value {
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn standalone_cli_is_the_project_entrypoint_and_persists_runs() {
    let project = create_project();
    let doctor = stdout(&run(project.path(), &["doctor"]));
    assert_eq!(doctor["schema_version"], "glr.cli-output.v1");
    assert_eq!(doctor["command"], "doctor");
    assert_eq!(doctor["data"]["ready"], true);

    let training = stdout(&run(project.path(), &["train"]));
    assert_eq!(training["command"], "train");
    assert_eq!(training["data"]["status"], "succeeded");
    let run_id = training["data"]["run_id"].as_str().unwrap();
    assert!(
        project
            .path()
            .join(".glr/runs")
            .join(run_id)
            .join("trainer.log")
            .is_file()
    );

    let evidence_connection = Connection::open(project.path().join(".glr/runs.sqlite3")).unwrap();
    let next_sequence: i64 = evidence_connection
        .query_row(
            "SELECT COALESCE(MAX(sequence_id), -1) + 1 FROM events WHERE run_id = ?",
            [run_id],
            |row| row.get(0),
        )
        .unwrap();
    for (offset, kind, payload) in [
        (
            0_i64,
            "navigation.route_sample",
            r#"{"position":[1.0,2.0,3.0],"route_id":"route.demo"}"#,
        ),
        (
            1_i64,
            "progression.item_unlocked",
            r#"{"item_kind":"hero","item_id":"hero.demo","status":"unlocked"}"#,
        ),
        (
            2_i64,
            "match.result",
            r#"{"match_kind":"pvp","outcome":"win","turns":3,"trophy_delta":1}"#,
        ),
    ] {
        evidence_connection
            .execute(
                "INSERT INTO events(run_id, sequence_id, timestamp_ns, kind, episode_id, step_id, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                params![run_id, next_sequence + offset, offset, kind, "episode-demo", offset, payload],
            )
            .unwrap();
    }
    drop(evidence_connection);

    let report = stdout(&run(project.path(), &["report", "build", run_id]));
    assert_eq!(report["command"], "report.build");
    assert_eq!(report["data"]["schema_version"], "glr.run-report.v1");
    let report_path = project
        .path()
        .join(".glr/runs")
        .join(run_id)
        .join("report/index.html");
    let report_html = fs::read_to_string(report_path).unwrap();
    assert!(report_html.contains("GLR run report"));
    assert!(report_html.contains("Event timeline"));
    assert!(report_html.contains("navigation.route_sample"));
    assert!(report_html.contains("progression.item_unlocked"));
    assert!(report_html.contains("match.result"));
    assert!(report_html.contains("Route trace"));
    assert!(report_html.contains("Unlocks and progression"));
    assert!(report_html.contains("Checksummed artifacts"));
    let rebuilt = stdout(&run(project.path(), &["report", "build", run_id]));
    assert_eq!(rebuilt["command"], "report.build");
    let custom_report = stdout(&run(
        project.path(),
        &["report", "build", run_id, "--output", "review/report"],
    ));
    assert_eq!(custom_report["command"], "report.build");
    assert!(
        project
            .path()
            .join(".glr/runs")
            .join(run_id)
            .join("review/report/index.html")
            .is_file()
    );

    let runs = stdout(&run(project.path(), &["runs", "list"]));
    assert_eq!(runs["command"], "runs.list");
    assert_eq!(runs["data"][0]["run_id"], run_id);

    let connection = Connection::open(project.path().join(".glr/runs.sqlite3")).unwrap();
    connection
        .execute(
            "INSERT INTO spatial_entities(environment_id, world_id, entity_id, kind, label, x, y, z, coordinate_frame, authority, confidence, observed_at_ns, source_run_id, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                "example.adventure-v1",
                "forest",
                "shrine.forest-1",
                "shrine",
                "土地庙",
                10.0,
                2.0,
                5.0,
                "world",
                "authoritative",
                0.9,
                1_i64,
                run_id,
                "{}",
            ],
        )
        .unwrap();
    connection
        .execute(
            "INSERT INTO spatial_routes(environment_id, world_id, route_id, name, from_entity_id, to_entity_id, coordinate_frame, confidence, verified_at_ns, source_run_id, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                "example.adventure-v1",
                "forest",
                "route.shrine-1",
                "Path to shrine",
                Option::<String>::None,
                "shrine.forest-1",
                "world",
                0.8,
                1_i64,
                run_id,
                "{}",
            ],
        )
        .unwrap();
    connection
        .execute(
            "INSERT INTO route_waypoints(environment_id, world_id, route_id, waypoint_index, x, y, z, tolerance, label) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                "example.adventure-v1",
                "forest",
                "route.shrine-1",
                0,
                10.0,
                2.0,
                5.0,
                1.0,
                "arrival",
            ],
        )
        .unwrap();
    connection
        .execute(
            "INSERT INTO research_sources(source_id, media_type, accessed_at, source_json) VALUES (?, ?, ?, ?)",
            params![
                "source.guide-1",
                "text-guide",
                "2026-09-01T00:00:00Z",
                r#"{"source_id":"source.guide-1","media_type":"text-guide","url":"https://example.com/guide"}"#,
            ],
        )
        .unwrap();
    connection
        .execute(
            "INSERT INTO research_findings(finding_id, category, status, scope, scope_id, finding_json) VALUES (?, ?, ?, ?, ?, ?)",
            params![
                "finding.navigation-1",
                "strategy",
                "runtime-verified",
                "family",
                "action-rpg",
                r#"{"finding_id":"finding.navigation-1","category":"strategy","status":"runtime-verified","scope":"family","scope_id":"action-rpg","summary":"Follow the marked path.","source_ids":["source.guide-1"],"tags":["navigation"]}"#,
            ],
        )
        .unwrap();
    connection
        .execute(
            "INSERT INTO research_finding_sources(finding_id, source_id, ordinal) VALUES (?, ?, ?)",
            params!["finding.navigation-1", "source.guide-1", 0],
        )
        .unwrap();
    connection
        .execute(
            "INSERT INTO research_finding_tags(finding_id, tag) VALUES (?, ?)",
            params!["finding.navigation-1", "navigation"],
        )
        .unwrap();
    drop(connection);

    let entities = stdout(&run(
        project.path(),
        &["query", "entities", "--world", "forest", "--name", "土地庙"],
    ));
    assert_eq!(entities["data"][0]["entity_id"], "shrine.forest-1");
    let routes = stdout(&run(
        project.path(),
        &[
            "query",
            "routes",
            "--world",
            "forest",
            "--to-entity",
            "shrine.forest-1",
        ],
    ));
    assert_eq!(routes["data"][0]["advisory"], true);
    assert_eq!(routes["data"][0]["waypoints"][0]["label"], "arrival");
    let research = stdout(&run(
        project.path(),
        &[
            "query",
            "research",
            "--tag",
            "navigation",
            "--verified-only",
        ],
    ));
    assert_eq!(research["data"][0]["finding_id"], "finding.navigation-1");
    assert_eq!(research["data"][0]["action_authority"], false);
}

#[test]
fn playback_verifies_the_exact_model_bundle_before_launch() {
    let project = create_project();
    let bundle = project.path().join("bundle");
    fs::create_dir_all(bundle.join("inputs")).unwrap();
    fs::create_dir_all(bundle.join("artifacts")).unwrap();
    fs::write(bundle.join("inputs/training.json"), b"config").unwrap();
    fs::write(bundle.join("artifacts/model.bin"), b"model").unwrap();
    let entry = |path: &Path, relative: &str| {
        let bytes = fs::read(path).unwrap();
        json!({
            "path": relative,
            "sha256": format!("{:x}", Sha256::digest(&bytes)),
            "size_bytes": bytes.len(),
        })
    };
    let manifest = json!({
        "schema_version": "glr.model-bundle.v1",
        "environment_id": "example.adventure-v1",
        "protocol_version": "1.0",
        "algorithm": "bc",
        "framework": "pytorch",
        "framework_version": "2.8.0",
        "seeds": [7],
        "inputs": [entry(&bundle.join("inputs/training.json"), "training.json")],
        "artifacts": [entry(&bundle.join("artifacts/model.bin"), "model.bin")]
    });
    fs::write(
        bundle.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let output = stdout(&run(
        project.path(),
        &["play", "--bundle", bundle.to_str().unwrap()],
    ));
    assert_eq!(output["command"], "play");
    assert_eq!(output["data"]["kind"], "playback");

    fs::write(bundle.join("artifacts/model.bin"), b"tampered").unwrap();
    let rejected = run(
        project.path(),
        &["play", "--bundle", bundle.to_str().unwrap()],
    );
    assert_eq!(rejected.status.code(), Some(2));
    let error: Value = serde_json::from_slice(&rejected.stderr).unwrap();
    assert_eq!(error["command"], "error");
}
