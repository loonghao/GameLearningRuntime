use std::collections::BTreeMap;
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

fn run_raw(project: &Path, arguments: &[&str]) -> Output {
    run(project, arguments)
}

fn checkpoint_contract(reward: &str, action: &str) -> Value {
    json!({
        "schema_version": "glr.checkpoint-contract.v1",
        "protocol_version": "glr.v1",
        "observation_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
        "action_sha256": action,
        "reward_sha256": reward,
        "knowledge_sha256": "3333333333333333333333333333333333333333333333333333333333333333"
    })
}

fn checkpoint_contract_digest(contract: &Value) -> String {
    let fields: BTreeMap<_, _> = contract
        .as_object()
        .unwrap()
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    format!("{:x}", Sha256::digest(serde_json::to_vec(&fields).unwrap()))
}

fn write_checkpoint_fixture(project: &Path, contract: &Value) -> (PathBuf, PathBuf) {
    let root = project.join("checkpoint");
    fs::create_dir_all(&root).unwrap();
    let checkpoint = root.join("policy.ckpt");
    fs::write(&checkpoint, b"weights").unwrap();
    let manifest = root.join("checkpoint.manifest.json");
    let checkpoint_sha256 = format!("{:x}", Sha256::digest(b"weights"));
    fs::write(
        &manifest,
        serde_json::to_vec_pretty(&json!({
            "schema_version": "glr.checkpoint-manifest.v1",
            "checkpoint_path": "policy.ckpt",
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_size_bytes": 7,
            "contract": contract,
            "contract_sha256": checkpoint_contract_digest(contract),
            "metadata": {}
        }))
        .unwrap(),
    )
    .unwrap();
    let current = root.join("current-contract.json");
    fs::write(&current, serde_json::to_vec_pretty(contract).unwrap()).unwrap();
    (manifest, current)
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
    let first_report_sha256 = report["data"]["sha256"].as_str().unwrap();
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
    assert_eq!(rebuilt["data"]["sha256"], first_report_sha256);
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
fn transaction_cli_reports_bounded_structural_abandonment() {
    let project = create_project();
    let training = stdout(&run(project.path(), &["train"]));
    let run_id = training["data"]["run_id"].as_str().unwrap();
    let connection = Connection::open(project.path().join(".glr/runs.sqlite3")).unwrap();
    connection
        .execute(
            "UPDATE runs SET status = 'running' WHERE run_id = ?",
            [run_id],
        )
        .unwrap();
    drop(connection);
    let steps = project.path().join("steps.json");
    fs::write(&steps, br#"[{"action_id":"move-1"}]"#).unwrap();

    let begin = stdout(&run(
        project.path(),
        &[
            "transaction",
            "begin",
            "--run-id",
            run_id,
            "--transaction-id",
            "txn.cli",
            "--steps",
            steps.to_str().unwrap(),
            "--max-resume-attempts",
            "1",
        ],
    ));
    assert_eq!(begin["command"], "transaction.begin");
    assert_eq!(begin["data"]["status"], "pending");

    let refusal = project.path().join("refusal.json");
    fs::write(
        &refusal,
        br#"{"action_id":"move-1","target_id":"card-1","reason_class":"structural","message":"blocked","retryable":false}"#,
    )
    .unwrap();
    let resumed = run(
        project.path(),
        &[
            "transaction",
            "resume",
            "--transaction-id",
            "txn.cli",
            "--refusal",
            refusal.to_str().unwrap(),
        ],
    );
    assert_eq!(resumed.status.code(), Some(77));
    let resumed: Value = serde_json::from_slice(&resumed.stdout).unwrap();
    assert_eq!(resumed["command"], "transaction.resume");
    assert_eq!(resumed["data"]["outcome"], "abandoned");

    let terminal = run(
        project.path(),
        &[
            "transaction",
            "resume",
            "--transaction-id",
            "txn.cli",
            "--refusal",
            refusal.to_str().unwrap(),
        ],
    );
    assert_eq!(terminal.status.code(), Some(0));
    let already_terminal: Value = serde_json::from_slice(&terminal.stdout).unwrap();
    assert_eq!(already_terminal["data"]["outcome"], "already_terminal");
}

#[test]
fn checkpoint_migration_reports_then_applies_explicit_confirmation() {
    let project = create_project();
    let recorded = checkpoint_contract(
        "4444444444444444444444444444444444444444444444444444444444444444",
        "2222222222222222222222222222222222222222222222222222222222222222",
    );
    let (manifest, current) = write_checkpoint_fixture(project.path(), &recorded);
    let changed = checkpoint_contract(
        "5555555555555555555555555555555555555555555555555555555555555555",
        "2222222222222222222222222222222222222222222222222222222222222222",
    );
    fs::write(&current, serde_json::to_vec_pretty(&changed).unwrap()).unwrap();

    let manifest_arg = manifest.to_string_lossy().into_owned();
    let current_arg = current.to_string_lossy().into_owned();
    let dry_run = run_raw(
        project.path(),
        &[
            "checkpoint",
            "migrate",
            "--manifest",
            &manifest_arg,
            "--contract",
            &current_arg,
        ],
    );
    assert_eq!(dry_run.status.code(), Some(3));
    let dry_run_json: Value = serde_json::from_slice(&dry_run.stdout).unwrap();
    assert_eq!(dry_run_json["command"], "checkpoint.migrate");
    assert_eq!(dry_run_json["data"]["requires_confirmation"], true);
    assert_eq!(
        dry_run_json["data"]["mismatches"][0]["field"],
        "reward_sha256"
    );

    let applied = run_raw(
        project.path(),
        &[
            "checkpoint",
            "migrate",
            "--manifest",
            &manifest_arg,
            "--contract",
            &current_arg,
            "--force",
        ],
    );
    assert!(
        applied.status.success(),
        "{}",
        String::from_utf8_lossy(&applied.stderr)
    );
    let applied_json: Value = serde_json::from_slice(&applied.stdout).unwrap();
    assert_eq!(applied_json["data"]["status"], "migrated");
    assert!(manifest.with_extension("json.bak").is_file());
    assert!(manifest.parent().unwrap().join("policy.ckpt.bak").is_file());
    assert_eq!(
        fs::read(manifest.parent().unwrap().join("policy.ckpt")).unwrap(),
        b"weights"
    );
}

#[test]
fn checkpoint_migration_fails_closed_for_action_changes() {
    let project = create_project();
    let recorded = checkpoint_contract(
        "4444444444444444444444444444444444444444444444444444444444444444",
        "2222222222222222222222222222222222222222222222222222222222222222",
    );
    let (manifest, current) = write_checkpoint_fixture(project.path(), &recorded);
    let changed = checkpoint_contract(
        "4444444444444444444444444444444444444444444444444444444444444444",
        "6666666666666666666666666666666666666666666666666666666666666666",
    );
    fs::write(&current, serde_json::to_vec_pretty(&changed).unwrap()).unwrap();
    let output = run_raw(
        project.path(),
        &[
            "checkpoint",
            "migrate",
            "--manifest",
            &manifest.to_string_lossy(),
            "--contract",
            &current.to_string_lossy(),
        ],
    );
    assert_eq!(output.status.code(), Some(4));
    let response: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(response["data"]["status"], "incompatible");
    assert_eq!(response["data"]["mismatches"][0]["field"], "action_sha256");
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

#[test]
fn spatial_graph_import_preserves_negative_evidence_and_filters_frontier_edges() {
    let project = create_project();
    let source = project.path().join("spatial-v2.json");
    let graph = json!({
        "schema_version": "glr.spatial-knowledge.v2",
        "environment_id": "example.adventure-v1",
        "protocol_version": "1.0",
        "exported_at_ns": 100,
        "nodes": [
            {
                "node_id": "node.start",
                "world_id": "forest",
                "position": [0.0, 0.0, 1.0],
                "coordinate_frame": "world",
                "ground_z": 0.0,
                "nav_z": 1.0,
                "observed_at_ns": 100,
                "source_run_id": "run.source",
                "authority": "authoritative",
                "confidence": 1.0,
                "metadata": {}
            },
            {
                "node_id": "node.goal",
                "world_id": "forest",
                "position": [5.0, 0.0, 1.0],
                "coordinate_frame": "world",
                "ground_z": 0.0,
                "nav_z": 1.0,
                "observed_at_ns": 100,
                "source_run_id": "run.source",
                "authority": "authoritative",
                "confidence": 1.0,
                "metadata": {}
            }
        ],
        "edges": [
            {
                "edge_id": "edge.good",
                "world_id": "forest",
                "from_node_id": "node.start",
                "to_node_id": "node.goal",
                "coordinate_frame": "world",
                "source_run_id": "run.source",
                "passability": "traversable",
                "cost": 5.0,
                "success_count": 1,
                "failure_count": 0,
                "last_verified_at_ns": 100,
                "expires_at_ns": 500,
                "ground_projection": [2.5, 0.0, 1.0],
                "nav_projection": [2.5, 0.0, 1.0],
                "vertical_delta": 0.0,
                "slope": 0.0,
                "clearance": 2.0,
                "hazard_reasons": [],
                "negative_evidence": [],
                "transform": null,
                "authority": "authoritative",
                "confidence": 1.0,
                "metadata": {}
            },
            {
                "edge_id": "edge.blocked",
                "world_id": "forest",
                "from_node_id": "node.start",
                "to_node_id": "node.goal",
                "coordinate_frame": "world",
                "source_run_id": "run.source",
                "passability": "blocked",
                "cost": null,
                "success_count": 0,
                "failure_count": 1,
                "last_verified_at_ns": 120,
                "expires_at_ns": null,
                "ground_projection": null,
                "nav_projection": null,
                "vertical_delta": null,
                "slope": 80.0,
                "clearance": 0.5,
                "hazard_reasons": ["steep-slope"],
                "negative_evidence": [
                    {
                        "reason": "steep-slope",
                        "observed_at_ns": 120,
                        "source_run_id": "run.source",
                        "expires_at_ns": null,
                        "detail": "blocked by slope"
                    }
                ],
                "transform": null,
                "authority": "authoritative",
                "confidence": 1.0,
                "metadata": {}
            }
        ],
        "transforms": []
    });
    fs::write(&source, serde_json::to_vec_pretty(&graph).unwrap()).unwrap();

    let imported = stdout(&run(
        project.path(),
        &["knowledge", "import", "--input", source.to_str().unwrap()],
    ));
    assert_eq!(imported["command"], "knowledge.graph-import");
    assert_eq!(imported["data"]["nodes"], 2);
    assert_eq!(imported["data"]["edges"], 2);

    let all_edges = stdout(&run(
        project.path(),
        &[
            "query",
            "edges",
            "--world",
            "forest",
            "--from-node",
            "node.start",
            "--at-ns",
            "200",
        ],
    ));
    assert_eq!(all_edges["data"].as_array().unwrap().len(), 2);
    let blocked = all_edges["data"]
        .as_array()
        .unwrap()
        .iter()
        .find(|edge| edge["edge_id"] == "edge.blocked")
        .unwrap();
    assert_eq!(blocked["advisory"], true);
    assert_eq!(blocked["status"], "blocked");

    let frontier = stdout(&run(
        project.path(),
        &[
            "query",
            "edges",
            "--world",
            "forest",
            "--from-node",
            "node.start",
            "--status",
            "traversable",
            "--at-ns",
            "200",
        ],
    ));
    assert_eq!(frontier["data"].as_array().unwrap().len(), 1);
    assert_eq!(frontier["data"][0]["edge_id"], "edge.good");
}
