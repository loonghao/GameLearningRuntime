use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::contracts::{read_json, sha256_file};
use crate::error::{Error, Result};

const CONTRACT_SCHEMA: &str = "glr.checkpoint-contract.v1";
const MANIFEST_SCHEMA: &str = "glr.checkpoint-manifest.v1";

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CheckpointContract {
    schema_version: String,
    protocol_version: String,
    observation_sha256: String,
    action_sha256: String,
    reward_sha256: String,
    knowledge_sha256: Option<String>,
}

impl CheckpointContract {
    fn validate(&self) -> Result<()> {
        if self.schema_version != CONTRACT_SCHEMA || self.protocol_version.is_empty() {
            return Err(Error::Contract(
                "unsupported or empty checkpoint contract version".into(),
            ));
        }
        for digest in [
            Some(&self.observation_sha256),
            Some(&self.action_sha256),
            Some(&self.reward_sha256),
            self.knowledge_sha256.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            validate_digest(digest)?;
        }
        Ok(())
    }

    fn digest(&self) -> Result<String> {
        // Match Python's sorted-key, compact UTF-8 canonical contract encoding.
        let fields: BTreeMap<String, Value> = serde_json::from_value(serde_json::to_value(self)?)?;
        Ok(format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&fields)?)
        ))
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CheckpointManifest {
    schema_version: String,
    checkpoint_path: String,
    checkpoint_sha256: String,
    checkpoint_size_bytes: u64,
    contract: CheckpointContract,
    contract_sha256: String,
    metadata: Value,
}

#[derive(Debug, Serialize)]
struct Mismatch {
    field: String,
    recorded: Value,
    current: Value,
    migratable: bool,
}

fn validate_digest(value: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(Error::Contract(
            "checkpoint digests must be lowercase SHA-256 values".into(),
        ));
    }
    Ok(())
}

fn checkpoint_file(manifest: &Path, relative: &str) -> Result<PathBuf> {
    let relative_path = Path::new(relative);
    if relative.is_empty()
        || relative.contains(['\\', ':'])
        || relative_path
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(Error::Contract(
            "checkpoint path must be a portable relative path".into(),
        ));
    }
    let root = manifest
        .parent()
        .ok_or_else(|| Error::Invalid("manifest has no parent".into()))?
        .canonicalize()?;
    let requested = root.join(relative_path);
    if requested.is_symlink() || !requested.is_file() {
        return Err(Error::Contract(
            "checkpoint must be an existing regular non-symlink file".into(),
        ));
    }
    let resolved = requested.canonicalize()?;
    if !resolved.starts_with(&root) {
        return Err(Error::Contract(
            "checkpoint path escapes the manifest directory".into(),
        ));
    }
    Ok(resolved)
}

fn verify(path: &Path) -> Result<(CheckpointManifest, PathBuf)> {
    let manifest: CheckpointManifest = read_json(path, "checkpoint manifest")?;
    if manifest.schema_version != MANIFEST_SCHEMA || !manifest.metadata.is_object() {
        return Err(Error::Contract(
            "unsupported checkpoint manifest schema or metadata".into(),
        ));
    }
    manifest.contract.validate()?;
    validate_digest(&manifest.checkpoint_sha256)?;
    validate_digest(&manifest.contract_sha256)?;
    if manifest.contract.digest()? != manifest.contract_sha256 {
        return Err(Error::Contract(
            "checkpoint manifest contract digest is corrupt; restore a verified backup".into(),
        ));
    }
    let checkpoint = checkpoint_file(path, &manifest.checkpoint_path)?;
    let actual_sha256 = sha256_file(&checkpoint)?;
    let actual_size = checkpoint.metadata()?.len();
    if actual_sha256 != manifest.checkpoint_sha256 || actual_size != manifest.checkpoint_size_bytes
    {
        return Err(Error::Contract(format!(
            "checkpoint bytes are corrupt: recorded_sha256={} actual_sha256={} recorded_size={} actual_size={}; restore a verified checkpoint backup",
            manifest.checkpoint_sha256, actual_sha256, manifest.checkpoint_size_bytes, actual_size,
        )));
    }
    Ok((manifest, checkpoint))
}

fn compare(recorded: &CheckpointContract, current: &CheckpointContract) -> Result<Vec<Mismatch>> {
    let recorded = serde_json::to_value(recorded)?;
    let current = serde_json::to_value(current)?;
    let mut mismatches = Vec::new();
    for field in [
        "schema_version",
        "protocol_version",
        "observation_sha256",
        "action_sha256",
        "reward_sha256",
        "knowledge_sha256",
    ] {
        if recorded[field] != current[field] {
            mismatches.push(Mismatch {
                field: field.into(),
                recorded: recorded[field].clone(),
                current: current[field].clone(),
                migratable: matches!(field, "reward_sha256" | "knowledge_sha256"),
            });
        }
    }
    Ok(mismatches)
}

fn backup(path: &Path) -> Result<PathBuf> {
    let name = path
        .file_name()
        .ok_or_else(|| Error::Invalid("backup has no filename".into()))?
        .to_string_lossy();
    for index in 0_u64.. {
        let suffix = if index == 0 {
            ".bak".into()
        } else {
            format!(".bak.{index}")
        };
        let candidate = path.with_file_name(format!("{name}{suffix}"));
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(mut target) => {
                let mut source = fs::File::open(path)?;
                std::io::copy(&mut source, &mut target)?;
                target.sync_all()?;
                return Ok(candidate);
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    unreachable!("backup suffix space cannot be exhausted")
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| Error::Invalid("output has no parent".into()))?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    temporary.write_all(bytes)?;
    temporary.as_file().sync_all()?;
    temporary
        .persist(path)
        .map_err(|error| Error::Io(error.error))?;
    Ok(())
}

pub fn migrate(manifest_path: &Path, contract_path: &Path, confirm: bool) -> Result<(Value, i32)> {
    let (mut manifest, checkpoint) = verify(manifest_path)?;
    let current: CheckpointContract = read_json(contract_path, "live checkpoint contract")?;
    current.validate()?;
    let mismatches = compare(&manifest.contract, &current)?;
    let recorded_digest = manifest.contract_sha256.clone();
    let current_digest = current.digest()?;
    let incompatible = mismatches.iter().any(|item| !item.migratable);
    let mut result = json!({
        "changed": false,
        "requires_confirmation": !mismatches.is_empty() && !incompatible && !confirm,
        "recorded_contract_sha256": recorded_digest,
        "current_contract_sha256": current_digest,
        "mismatches": mismatches,
        "checkpoint_integrity": "verified",
        "status": "unchanged",
        "next_step": "none",
    });
    if mismatches.is_empty() {
        return Ok((result, 0));
    }
    if incompatible {
        result["status"] = json!("incompatible");
        result["next_step"] = json!(
            "retrain with the current observation/action/protocol contract; do not reshape weights"
        );
        return Ok((result, 4));
    }
    result["status"] = json!("contract_changed");
    result["next_step"] = json!(
        "review the stale fields and rerun checkpoint migrate with --force to preserve bytes and update the contract binding"
    );
    if !confirm {
        return Ok((result, 3));
    }

    let manifest_backup = backup(manifest_path)?;
    let checkpoint_backup = backup(&checkpoint)?;
    manifest.contract = current;
    manifest.contract_sha256 = current_digest;
    let migrated = (|| -> Result<()> {
        // The generic CLI is framework-neutral: preserving exact bytes proves zero weight drift.
        // Framework-owned reserialization belongs in the Python saver callback.
        atomic_write(&checkpoint, &fs::read(&checkpoint_backup)?)?;
        atomic_write(manifest_path, &serde_json::to_vec_pretty(&manifest)?)?;
        verify(manifest_path)?;
        Ok(())
    })();
    if let Err(error) = migrated {
        atomic_write(&checkpoint, &fs::read(&checkpoint_backup)?)?;
        atomic_write(manifest_path, &fs::read(&manifest_backup)?)?;
        return Err(error);
    }
    result["changed"] = json!(true);
    result["status"] = json!("migrated");
    result["next_step"] = json!("load the verified checkpoint under the current contract");
    result["manifest_backup"] = json!(manifest_backup);
    result["checkpoint_backup"] = json!(checkpoint_backup);
    Ok((result, 0))
}
