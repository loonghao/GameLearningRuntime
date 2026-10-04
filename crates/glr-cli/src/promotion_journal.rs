//! Recoverable filesystem half of checkpoint promotion. SQLite owns the lock
//! and decides whether a durable intent committed; this module retains bytes.
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::contracts::sha256_file;
use crate::error::{Error, Result};
use crate::project::validate_identifier;
use crate::store::CheckpointPromotionRecord;

const SCHEMA: &str = "glr.checkpoint-promotion-journal.v2";
const MAX_JOURNAL_BYTES: u64 = 32 * 1024;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Binding {
    pub environment_id: String,
    pub protocol_version: String,
    pub target_id: String,
    pub environment_config_sha256: String,
    pub host_fingerprint: String,
    pub host_epoch: String,
    pub goal_id: String,
    pub store_path: String,
    pub live_path: String,
}

impl Binding {
    // Every field is an independent persisted owner binding, kept explicit.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        store: &Path,
        environment_id: &str,
        goal_id: &str,
        live: &Path,
        protocol_version: &str,
        target_id: &str,
        environment_config_sha256: &str,
        host_fingerprint: &str,
        host_epoch: &str,
    ) -> Result<Self> {
        validate_ancestors(store)?;
        validate_identifier(environment_id, "promotion environment_id")?;
        validate_identifier(goal_id, "promotion goal_id")?;
        validate_identifier(target_id, "promotion target_id")?;
        if protocol_version.trim().is_empty() || !digest_valid(environment_config_sha256) {
            return Err(Error::Contract(
                "checkpoint protocol/config binding is absent".into(),
            ));
        }
        if !digest_valid(host_fingerprint)
            || host_epoch.len() != 32
            || !host_epoch.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err(Error::Contract(
                "checkpoint host instance binding is absent".into(),
            ));
        }
        Ok(Self {
            host_fingerprint: host_fingerprint.into(),
            host_epoch: host_epoch.into(),
            protocol_version: protocol_version.into(),
            target_id: target_id.into(),
            environment_config_sha256: environment_config_sha256.into(),
            environment_id: environment_id.into(),
            goal_id: goal_id.into(),
            store_path: path_text(&store.canonicalize()?)?,
            live_path: path_text(&canonical_live_path(live)?)?,
        })
    }

    pub fn validate(&self, store: &Path) -> Result<()> {
        if *self
            != Self::new(
                store,
                &self.environment_id,
                &self.goal_id,
                Path::new(&self.live_path),
                &self.protocol_version,
                &self.target_id,
                &self.environment_config_sha256,
                &self.host_fingerprint,
                &self.host_epoch,
            )?
        {
            return Err(Error::Contract("checkpoint journal binding changed".into()));
        }
        Ok(())
    }

    pub fn directory(&self) -> PathBuf {
        use sha2::{Digest, Sha256};
        // Case aliases must contend for the same owner on Windows, even
        // before the destination exists. A collision only refuses ownership.
        #[cfg(windows)]
        let identity = self.live_path.to_lowercase();
        #[cfg(not(windows))]
        let identity = self.live_path.clone();
        let hash = format!("{:x}", Sha256::digest(identity.as_bytes()));
        Path::new(&self.live_path)
            .parent()
            .expect("validated live parent")
            .join(format!(".glr-promotion-{hash}"))
    }

    pub fn pending(&self) -> PathBuf {
        self.directory().join("pending.json")
    }
    pub fn owner(&self) -> PathBuf {
        let live = Path::new(&self.live_path);
        // The OS arbitrates the leaf name itself, including its native case
        // aliases. Ownership must not depend solely on a string-normalized hash.
        live.with_file_name(format!(
            ".{}.glr-owner.json",
            live.file_name()
                .expect("validated live filename")
                .to_string_lossy()
        ))
    }
    pub fn blob(&self, digest: &str) -> PathBuf {
        self.directory().join(format!("{digest}.checkpoint"))
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Owner {
    schema_version: String,
    binding: Binding,
}

pub(crate) fn verify_owner(binding: &Binding) -> Result<()> {
    let path = binding.owner();
    validate_ancestors(&path)?;
    if !path.is_file() || path.metadata()?.len() > MAX_JOURNAL_BYTES {
        return Err(Error::Contract(
            "checkpoint location owner is missing or invalid; ownership cannot be stolen".into(),
        ));
    }
    let owner: Owner = serde_json::from_slice(&fs::read(path)?)?;
    if owner.schema_version != "glr.checkpoint-promotion-owner.v2" || owner.binding != *binding {
        return Err(Error::Contract("checkpoint location belongs to another store/environment/goal; ownership cannot be stolen".into()));
    }
    Ok(())
}

pub(crate) fn claim_location(binding: &Binding) -> Result<()> {
    validate_ancestors(&binding.owner())?;
    validate_ancestors(&binding.directory())?;
    fs::create_dir_all(binding.directory())?;
    if binding.owner().try_exists()? {
        return verify_owner(binding);
    }
    let mut temporary = tempfile::NamedTempFile::new_in(binding.directory())?;
    write_bounded_json(
        temporary.as_file_mut(),
        &Owner {
            schema_version: "glr.checkpoint-promotion-owner.v2".into(),
            binding: binding.clone(),
        },
    )?;
    temporary.as_file().sync_all()?;
    match temporary.persist_noclobber(binding.owner()) {
        Ok(_) => (),
        Err(error) if error.error.kind() == std::io::ErrorKind::AlreadyExists => {
            return verify_owner(binding);
        }
        Err(error) => return Err(error.error.into()),
    }
    sync_file(&binding.owner())?;
    sync_directory(binding.owner().parent().expect("live owner parent"))?;
    sync_directory(&binding.directory())?;
    sync_directory(binding.directory().parent().expect("owner parent"))?;
    verify_owner(binding)
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Journal {
    schema_version: String,
    promotion_id: String,
    pub authorization_id: String,
    pub binding: Binding,
    pub incumbent_sha256: Option<String>,
    pub previous: Option<CheckpointPromotionRecord>,
    pub proposed: CheckpointPromotionRecord,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Phase {
    Staged,
    Journaled,
    SqlWritten,
    Replaced,
    BeforeCommit,
    Committed,
}

pub(crate) fn canonical_live_path(path: &Path) -> Result<PathBuf> {
    validate_ancestors(path)?;
    let parent = path
        .parent()
        .ok_or_else(|| Error::Invalid("checkpoint has no parent".into()))?;
    for ancestor in parent.ancestors() {
        if ancestor.is_symlink() {
            return Err(Error::Contract("checkpoint ancestor is a symlink".into()));
        }
    }
    if path.is_symlink() || (path.try_exists()? && !path.is_file()) {
        return Err(Error::Contract("checkpoint must be a regular file".into()));
    }
    if path.try_exists()? {
        return Ok(path.canonicalize()?);
    }
    Ok(parent.canonicalize()?.join(
        path.file_name()
            .ok_or_else(|| Error::Invalid("checkpoint has no filename".into()))?,
    ))
}

pub(crate) fn validate_ancestors(path: &Path) -> Result<()> {
    for ancestor in path.ancestors() {
        if is_link(ancestor)? {
            return Err(Error::Contract(
                "checkpoint path contains a symlink/reparse point".into(),
            ));
        }
    }
    Ok(())
}

fn is_link(path: &Path) -> Result<bool> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error.into()),
    };
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        Ok(metadata.file_attributes()
            & 0x400 /* FILE_ATTRIBUTE_REPARSE_POINT */
            != 0)
    }
    #[cfg(not(windows))]
    {
        Ok(metadata.file_type().is_symlink())
    }
}

fn path_text(path: &Path) -> Result<String> {
    path.to_str()
        .map(str::to_owned)
        .ok_or_else(|| Error::Invalid("checkpoint path is not UTF-8".into()))
}

fn digest_valid(digest: &str) -> bool {
    digest.len() == 64
        && digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn verify_blob(binding: &Binding, digest: &str) -> Result<PathBuf> {
    if !digest_valid(digest) {
        return Err(Error::Contract("invalid checkpoint journal digest".into()));
    }
    let path = binding.blob(digest);
    if is_link(&path)? || !path.is_file() || sha256_file(&path)? != digest {
        return Err(Error::Contract(
            "checkpoint journal blob is missing or corrupt; bytes retained".into(),
        ));
    }
    Ok(path)
}

// sync_all protects durable contents; Unix also syncs the directory entry.
// Recovery does not claim protection from storage devices that ignore flushes.
fn sync_directory(path: &Path) -> Result<()> {
    #[cfg(unix)]
    fs::File::open(path)?.sync_all()?;
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

fn sync_file(path: &Path) -> Result<()> {
    // Windows FlushFileBuffers requires a writable handle. These are only
    // generated sidecars/blobs and the replaceable live projection.
    fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)?
        .sync_all()?;
    Ok(())
}

fn write_bounded_json(writer: &mut fs::File, value: &impl Serialize) -> Result<()> {
    use std::io::Write;
    let bytes = serde_json::to_vec(value)?;
    if bytes.len() as u64 > MAX_JOURNAL_BYTES {
        return Err(Error::Contract(
            "checkpoint journal metadata exceeds its bound".into(),
        ));
    }
    writer.write_all(&bytes)?;
    Ok(())
}

fn stage_blob(binding: &Binding, source: &Path) -> Result<String> {
    validate_ancestors(source)?;
    if is_link(source)? || !source.is_file() {
        return Err(Error::Missing(source.to_path_buf()));
    }
    let mut staged = tempfile::NamedTempFile::new_in(binding.directory())?;
    std::io::copy(&mut fs::File::open(source)?, staged.as_file_mut())?;
    staged.as_file().sync_all()?;
    let digest = sha256_file(staged.path())?;
    let destination = binding.blob(&digest);
    if destination.try_exists()? {
        verify_blob(binding, &digest)?;
    } else {
        staged
            .persist_noclobber(&destination)
            .map_err(|error| Error::Io(error.error))?;
        sync_file(&destination)?;
        sync_directory(&binding.directory())?;
    }
    Ok(digest)
}

impl Journal {
    pub fn stage(
        binding: Binding,
        authorization_id: &str,
        candidate: &Path,
        previous: Option<CheckpointPromotionRecord>,
        mut proposed: CheckpointPromotionRecord,
    ) -> Result<Self> {
        verify_owner(&binding)?;
        let directory = binding.directory();
        if is_link(&directory)? || (directory.try_exists()? && !directory.is_dir()) {
            return Err(Error::Contract(
                "checkpoint journal directory must be a regular directory".into(),
            ));
        }
        fs::create_dir_all(&directory)?;
        sync_directory(directory.parent().expect("journal parent"))?;
        if binding.pending().try_exists()? {
            return Err(Error::Contract(
                "checkpoint journal must be reconciled before another promotion".into(),
            ));
        }
        let incumbent_sha256 = if Path::new(&binding.live_path).try_exists()? {
            Some(stage_blob(&binding, Path::new(&binding.live_path))?)
        } else {
            None
        };
        if previous.as_ref().is_some_and(|record| {
            incumbent_sha256.as_deref() != Some(record.checkpoint_sha256.as_str())
        }) {
            return Err(Error::Contract(
                "incumbent bytes disagree with the promotion record".into(),
            ));
        }
        proposed.checkpoint_sha256 = stage_blob(&binding, candidate)?;
        proposed.checkpoint_path = binding.live_path.clone();
        Ok(Self {
            schema_version: SCHEMA.into(),
            authorization_id: authorization_id.into(),
            promotion_id: uuid::Uuid::new_v4().simple().to_string(),
            binding,
            incumbent_sha256,
            previous,
            proposed,
        })
    }

    pub fn persist(&self) -> Result<()> {
        let mut temporary = tempfile::NamedTempFile::new_in(self.binding.directory())?;
        write_bounded_json(temporary.as_file_mut(), self)?;
        temporary.as_file().sync_all()?;
        temporary
            .persist_noclobber(self.binding.pending())
            .map_err(|error| Error::Io(error.error))?;
        sync_file(&self.binding.pending())?;
        sync_directory(&self.binding.directory())
    }

    pub fn load(binding: &Binding) -> Result<Option<Self>> {
        let path = binding.pending();
        validate_ancestors(&path)?;
        if !path.try_exists()? && !path.is_symlink() {
            return Ok(None);
        }
        if binding.directory().is_symlink()
            || path.is_symlink()
            || !path.is_file()
            || path.metadata()?.len() > MAX_JOURNAL_BYTES
        {
            return Err(Error::Contract(
                "checkpoint journal is not a bounded regular file".into(),
            ));
        }
        let journal: Self = serde_json::from_slice(&fs::read(path)?)?;
        if journal.schema_version != SCHEMA
            || journal.binding != *binding
            || journal.promotion_id.len() != 32
            || !journal
                .promotion_id
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(Error::Contract(
                "unknown or mismatched checkpoint journal; bytes retained".into(),
            ));
        }
        for record in journal
            .previous
            .iter()
            .chain(std::iter::once(&journal.proposed))
        {
            if record.goal_id != binding.goal_id
                || !record.best_metric.is_finite()
                || !matches!(record.mode.as_str(), "max" | "min")
                || canonical_live_path(Path::new(&record.checkpoint_path))?
                    != Path::new(&binding.live_path)
            {
                return Err(Error::Contract(
                    "checkpoint journal record binding is invalid".into(),
                ));
            }
            validate_identifier(&record.run_id, "journal run_id")?;
            validate_identifier(&record.trial_id, "journal trial_id")?;
            validate_identifier(&record.metric, "journal metric")?;
            verify_blob(binding, &record.checkpoint_sha256)?;
        }
        if let Some(digest) = &journal.incumbent_sha256 {
            verify_blob(binding, digest)?;
        }
        if journal.previous.as_ref().is_some_and(|record| {
            journal.incumbent_sha256.as_deref() != Some(record.checkpoint_sha256.as_str())
        }) {
            return Err(Error::Contract(
                "checkpoint journal incumbent lineage is invalid".into(),
            ));
        }
        Ok(Some(journal))
    }

    pub fn install(&self, digest: &str) -> Result<()> {
        let blob = verify_blob(&self.binding, digest)?;
        let live = Path::new(&self.binding.live_path);
        canonical_live_path(live)?;
        let mut temporary = tempfile::NamedTempFile::new_in(live.parent().expect("live parent"))?;
        std::io::copy(&mut fs::File::open(blob)?, temporary.as_file_mut())?;
        temporary.as_file().sync_all()?;
        temporary
            .persist(live)
            .map_err(|error| Error::Io(error.error))?;
        sync_file(live)?;
        sync_directory(live.parent().expect("live parent"))
    }

    pub fn reconcile(&self, committed: bool) -> Result<()> {
        let live = Path::new(&self.binding.live_path);
        canonical_live_path(live)?;
        let current = if live.try_exists()? {
            Some(sha256_file(live)?)
        } else {
            None
        };
        if current.as_ref().is_some_and(|digest| {
            Some(digest) != self.incumbent_sha256.as_ref()
                && *digest != self.proposed.checkpoint_sha256
        }) {
            return Err(Error::Contract(
                "unknown live checkpoint bytes; recovery refused and bytes retained".into(),
            ));
        }
        if committed {
            if current.as_deref() != Some(self.proposed.checkpoint_sha256.as_str()) {
                self.install(&self.proposed.checkpoint_sha256)?;
            }
        } else if let Some(digest) = &self.incumbent_sha256 {
            if current.as_deref() != Some(digest.as_str()) {
                self.install(digest)?;
            }
        } else if live.try_exists()? {
            // A first promotion had no incumbent. Keep the abandoned live
            // bytes as well as the immutable candidate; never unlink them.
            archive_no_replace(
                live,
                &self
                    .binding
                    .directory()
                    .join(format!("{}.abandoned", self.promotion_id)),
            )?;
            sync_directory(live.parent().expect("live parent"))?;
        }
        let status = if committed {
            "committed"
        } else {
            "rolled-back"
        };
        archive_no_replace(
            &self.binding.pending(),
            &self
                .binding
                .directory()
                .join(format!("{}.{status}.json", self.promotion_id)),
        )?;
        sync_directory(&self.binding.directory())
    }
}

// Same-volume hard-link creation is atomic and refuses an existing leaf.
// A crash between linking and unlinking leaves both projections; exact bytes
// make repeating the operation safe. Never overwrite a foreign archive.
fn archive_no_replace(source: &Path, destination: &Path) -> Result<()> {
    validate_ancestors(source)?;
    validate_ancestors(destination)?;
    match fs::hard_link(source, destination) {
        Ok(()) => (),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            if sha256_file(source)? != sha256_file(destination)? {
                return Err(Error::Contract(
                    "checkpoint archive conflicts; bytes retained".into(),
                ));
            }
        }
        Err(error) => return Err(error.into()),
    }
    sync_file(destination)?;
    sync_directory(destination.parent().expect("archive parent"))?;
    fs::remove_file(source)?;
    sync_directory(source.parent().expect("source parent"))
}
