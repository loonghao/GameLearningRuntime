use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::thread;
use std::time::Duration;

use reqwest::StatusCode;
use reqwest::blocking::{Client, Response};
use reqwest::header::{ACCEPT, AUTHORIZATION, HeaderMap, HeaderValue, USER_AGENT};
use semver::Version;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

use crate::error::{Error, Result};

const OWNER: &str = "loonghao";
const REPOSITORY: &str = "GameLearningRuntime";
const UPDATE_SCHEMA_VERSION: &str = "glr.release-bundle.v1";
const MAX_RELEASE_JSON_BYTES: usize = 1024 * 1024;
const MAX_CHECKSUM_BYTES: usize = 1024 * 1024;
const MAX_ARCHIVE_BYTES: usize = 256 * 1024 * 1024;
const MAX_ARCHIVE_ENTRIES: usize = 4096;
const MAX_EXTRACTED_BYTES: u64 = 512 * 1024 * 1024;

pub const BUILD_TARGET: &str = env!("GLR_BUILD_TARGET");

#[derive(Debug, Clone, Deserialize)]
struct ReleaseAsset {
    name: String,
    browser_download_url: String,
    size: u64,
}

#[derive(Debug, Clone, Deserialize)]
struct GitHubRelease {
    tag_name: String,
    draft: bool,
    prerelease: bool,
    assets: Vec<ReleaseAsset>,
}

#[derive(Debug, Clone, Serialize)]
pub struct UpdatePlan {
    pub owner: &'static str,
    pub repository: &'static str,
    pub current_version: String,
    pub latest_version: String,
    pub target: &'static str,
    pub version_update_available: bool,
    pub asset: String,
    pub sha256: String,
    pub managed_components: Vec<&'static str>,
    pub integrity: &'static str,
    #[serde(skip)]
    archive_url: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct UpdateResult {
    #[serde(flatten)]
    pub plan: UpdatePlan,
    pub applied: bool,
    pub skills_updated: bool,
    pub host_updated: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReleaseBundleManifest {
    schema_version: String,
    version: String,
    target: String,
    cli: String,
    host: String,
    skills: Vec<SkillEntry>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SkillEntry {
    name: String,
    path: String,
}

pub struct Updater {
    client: Client,
    api_url: String,
}

impl Updater {
    pub fn github() -> Result<Self> {
        let mut headers = HeaderMap::new();
        headers.insert(USER_AGENT, HeaderValue::from_static("glr-self-update"));
        headers.insert(
            ACCEPT,
            HeaderValue::from_static("application/vnd.github+json"),
        );
        if let Ok(token) = std::env::var("GLR_GITHUB_TOKEN") {
            let value = HeaderValue::from_str(&format!("Bearer {token}")).map_err(|_| {
                Error::Invalid("GLR_GITHUB_TOKEN is not a valid header value".into())
            })?;
            headers.insert(AUTHORIZATION, value);
        }
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(30))
            .redirect(reqwest::redirect::Policy::custom(|attempt| {
                if attempt.url().scheme() != "https" {
                    attempt.error("GLR update redirects must remain on HTTPS")
                } else if attempt.previous().len() >= 3 {
                    attempt.stop()
                } else {
                    attempt.follow()
                }
            }))
            .default_headers(headers)
            .build()?;
        Ok(Self {
            client,
            api_url: format!("https://api.github.com/repos/{OWNER}/{REPOSITORY}/releases/latest"),
        })
    }

    #[cfg(test)]
    fn with_api_url(api_url: String) -> Result<Self> {
        let mut headers = HeaderMap::new();
        headers.insert(USER_AGENT, HeaderValue::from_static("glr-self-update-test"));
        let client = Client::builder()
            .timeout(Duration::from_secs(5))
            .redirect(reqwest::redirect::Policy::none())
            .default_headers(headers)
            .build()?;
        Ok(Self { client, api_url })
    }

    pub fn check(&self) -> Result<UpdatePlan> {
        let release_bytes = self.get_limited(&self.api_url, MAX_RELEASE_JSON_BYTES)?;
        let release: GitHubRelease = serde_json::from_slice(&release_bytes)?;
        if release.draft || release.prerelease {
            return Err(Error::Contract(
                "latest release must be a stable published release".into(),
            ));
        }
        let latest_version = release
            .tag_name
            .strip_prefix('v')
            .ok_or_else(|| Error::Contract("release tag must start with v".into()))?;
        let latest = Version::parse(latest_version)?;
        let current = Version::parse(env!("CARGO_PKG_VERSION"))?;
        let asset_name = format!("glr-{latest_version}-{BUILD_TARGET}.zip");
        let archive = release
            .assets
            .iter()
            .find(|asset| asset.name == asset_name)
            .ok_or_else(|| Error::Contract(format!("release is missing {asset_name}")))?;
        if archive.size == 0 || archive.size > MAX_ARCHIVE_BYTES as u64 {
            return Err(Error::Contract(format!(
                "release asset has an invalid size: {}",
                archive.size
            )));
        }
        let checksum_asset = release
            .assets
            .iter()
            .find(|asset| asset.name == "SHA256SUMS")
            .ok_or_else(|| Error::Contract("release is missing SHA256SUMS".into()))?;
        let checksums =
            self.get_limited(&checksum_asset.browser_download_url, MAX_CHECKSUM_BYTES)?;
        let checksums = String::from_utf8(checksums)
            .map_err(|_| Error::Contract("SHA256SUMS is not UTF-8".into()))?;
        let checksum = checksum_for(&checksums, &asset_name)?;
        Ok(UpdatePlan {
            owner: OWNER,
            repository: REPOSITORY,
            current_version: current.to_string(),
            latest_version: latest.to_string(),
            target: BUILD_TARGET,
            version_update_available: latest > current,
            asset: asset_name,
            sha256: checksum,
            managed_components: vec!["cli", "runtime-host", "skills"],
            integrity: "same-release-sha256",
            archive_url: archive.browser_download_url.clone(),
        })
    }

    pub fn apply(&self, plan: UpdatePlan, skills_dir: Option<&Path>) -> Result<UpdateResult> {
        if !plan.version_update_available {
            return Ok(UpdateResult {
                plan,
                applied: false,
                skills_updated: false,
                host_updated: false,
            });
        }
        let archive = self.get_limited(&plan.archive_url, MAX_ARCHIVE_BYTES)?;
        let actual = format!("{:x}", Sha256::digest(&archive));
        if actual != plan.sha256 {
            return Err(Error::Contract(format!(
                "release archive checksum mismatch for {}",
                plan.asset
            )));
        }
        let temporary = tempfile::tempdir()?;
        let archive_path = temporary.path().join("release.zip");
        fs::write(&archive_path, archive)?;
        extract_archive(&archive_path, temporary.path())?;
        let manifest_path = find_manifest(temporary.path())?;
        let manifest: ReleaseBundleManifest = serde_json::from_slice(&fs::read(&manifest_path)?)?;
        validate_manifest(&manifest, &plan)?;
        let root = manifest_path
            .parent()
            .ok_or_else(|| Error::Contract("release manifest has no parent".into()))?;
        let cli = safe_join(root, &manifest.cli)?;
        let host = safe_join(root, &manifest.host)?;
        if !cli.is_file() || !host.is_file() {
            return Err(Error::Contract(
                "release bundle is missing the CLI or Runtime Host".into(),
            ));
        }

        let mut skills_updated = false;
        if let Some(destination) = skills_dir {
            if destination.is_symlink() {
                return Err(Error::Contract(
                    "managed skills directory cannot be a symlink".into(),
                ));
            }
            fs::create_dir_all(destination)?;
            for skill in &manifest.skills {
                validate_skill_name(&skill.name)?;
                let source = safe_join(root, &skill.path)?;
                if !source.is_dir() || !source.join("SKILL.md").is_file() {
                    return Err(Error::Contract(format!(
                        "release skill is invalid: {}",
                        skill.name
                    )));
                }
                replace_directory(&source, &destination.join(&skill.name))?;
            }
            skills_updated = !manifest.skills.is_empty();
        }

        let current_executable = std::env::current_exe()?;
        let executable_dir = current_executable
            .parent()
            .ok_or_else(|| Error::Contract("current executable has no parent".into()))?;
        let host_name = if cfg!(windows) {
            "glr-hostd.exe"
        } else {
            "glr-hostd"
        };
        atomic_copy(&host, &executable_dir.join(host_name))?;
        self_replace::self_replace(&cli)?;
        Ok(UpdateResult {
            plan,
            applied: true,
            skills_updated,
            host_updated: true,
        })
    }

    fn get_limited(&self, url: &str, maximum: usize) -> Result<Vec<u8>> {
        let parsed = reqwest::Url::parse(url)
            .map_err(|_| Error::Contract("update URL is invalid".into()))?;
        if parsed.scheme() != "https" {
            return Err(Error::Contract(
                "update downloads require an HTTPS URL".into(),
            ));
        }
        let mut last_status = None;
        for attempt in 0..3 {
            let response = self.client.get(url).send()?;
            let status = response.status();
            if status.is_success() {
                return read_response(response, maximum);
            }
            last_status = Some(status);
            if status != StatusCode::TOO_MANY_REQUESTS && !status.is_server_error() {
                break;
            }
            thread::sleep(Duration::from_millis(250 * (1 << attempt)));
        }
        Err(Error::Contract(format!(
            "update endpoint returned HTTP {}",
            last_status.map_or(0, |status| status.as_u16())
        )))
    }
}

fn read_response(response: Response, maximum: usize) -> Result<Vec<u8>> {
    if response
        .content_length()
        .is_some_and(|length| length > maximum as u64)
    {
        return Err(Error::Contract(
            "update response exceeds its size limit".into(),
        ));
    }
    let mut output = Vec::new();
    response.take(maximum as u64 + 1).read_to_end(&mut output)?;
    if output.len() > maximum {
        return Err(Error::Contract(
            "update response exceeds its size limit".into(),
        ));
    }
    Ok(output)
}

pub fn checksum_for(checksums: &str, asset: &str) -> Result<String> {
    for line in checksums.lines() {
        let mut fields = line.split_whitespace();
        let Some(checksum) = fields.next() else {
            continue;
        };
        let Some(name) = fields.next() else {
            continue;
        };
        if name.trim_start_matches('*') == asset {
            if checksum.len() == 64
                && checksum
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                return Ok(checksum.into());
            }
            return Err(Error::Contract(format!(
                "SHA256SUMS has an invalid digest for {asset}"
            )));
        }
    }
    Err(Error::Contract(format!(
        "SHA256SUMS does not contain {asset}"
    )))
}

fn extract_archive(archive_path: &Path, destination: &Path) -> Result<()> {
    let file = File::open(archive_path)?;
    let mut archive = zip::ZipArchive::new(file)?;
    if archive.len() > MAX_ARCHIVE_ENTRIES {
        return Err(Error::Contract(
            "release archive contains too many entries".into(),
        ));
    }
    let mut extracted_bytes = 0_u64;
    for index in 0..archive.len() {
        let mut entry = archive.by_index(index)?;
        if entry
            .unix_mode()
            .is_some_and(|mode| mode & 0o170000 == 0o120000)
        {
            return Err(Error::Contract(
                "release archive cannot contain symlinks".into(),
            ));
        }
        extracted_bytes = extracted_bytes
            .checked_add(entry.size())
            .ok_or_else(|| Error::Contract("release archive size overflow".into()))?;
        if extracted_bytes > MAX_EXTRACTED_BYTES {
            return Err(Error::Contract(
                "release archive exceeds the extraction limit".into(),
            ));
        }
        let enclosed = entry
            .enclosed_name()
            .ok_or_else(|| Error::Contract("release archive contains an unsafe path".into()))?;
        let output = destination.join(enclosed);
        if entry.is_dir() {
            fs::create_dir_all(&output)?;
            continue;
        }
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = File::create(&output)?;
        std::io::copy(&mut entry, &mut file)?;
        file.flush()?;
        #[cfg(unix)]
        if let Some(mode) = entry.unix_mode() {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&output, fs::Permissions::from_mode(mode & 0o777))?;
        }
    }
    Ok(())
}

fn find_manifest(root: &Path) -> Result<PathBuf> {
    let direct = root.join("glr-release.json");
    if direct.is_file() {
        return Ok(direct);
    }
    let mut candidates = fs::read_dir(root)?
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path().join("glr-release.json"))
        .filter(|path| path.is_file());
    let first = candidates
        .next()
        .ok_or_else(|| Error::Contract("release archive is missing glr-release.json".into()))?;
    if candidates.next().is_some() {
        return Err(Error::Contract(
            "release archive contains multiple manifests".into(),
        ));
    }
    Ok(first)
}

fn validate_manifest(manifest: &ReleaseBundleManifest, plan: &UpdatePlan) -> Result<()> {
    if manifest.schema_version != UPDATE_SCHEMA_VERSION
        || manifest.version != plan.latest_version
        || manifest.target != BUILD_TARGET
    {
        return Err(Error::Contract(
            "release bundle manifest does not match the selected version and target".into(),
        ));
    }
    safe_relative(&manifest.cli)?;
    safe_relative(&manifest.host)?;
    for skill in &manifest.skills {
        validate_skill_name(&skill.name)?;
        safe_relative(&skill.path)?;
    }
    Ok(())
}

fn validate_skill_name(value: &str) -> Result<()> {
    if !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
    {
        Ok(())
    } else {
        Err(Error::Contract(format!(
            "release skill name is invalid: {value:?}"
        )))
    }
}

fn safe_relative(value: &str) -> Result<PathBuf> {
    let path = PathBuf::from(value);
    if value.is_empty()
        || value.contains('\\')
        || value.contains(':')
        || path.is_absolute()
        || path
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        Err(Error::Contract(format!(
            "release manifest path is unsafe: {value:?}"
        )))
    } else {
        Ok(path)
    }
}

fn safe_join(root: &Path, value: &str) -> Result<PathBuf> {
    Ok(root.join(safe_relative(value)?))
}

fn atomic_copy(source: &Path, destination: &Path) -> Result<()> {
    let parent = destination
        .parent()
        .ok_or_else(|| Error::Contract("managed binary destination has no parent".into()))?;
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(
        ".{}.glr-update",
        destination
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("binary")
    ));
    fs::copy(source, &temporary)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&temporary, fs::Permissions::from_mode(0o755))?;
    }
    let backup_directory = TempDir::new_in(parent)?;
    let backup = backup_directory.path().join("previous");
    if destination.exists() {
        fs::rename(destination, &backup)?;
    }
    if let Err(error) = fs::rename(&temporary, destination) {
        if backup.exists() {
            let _ = fs::rename(&backup, destination);
        }
        return Err(error.into());
    }
    Ok(())
}

fn replace_directory(source: &Path, destination: &Path) -> Result<()> {
    let parent = destination
        .parent()
        .ok_or_else(|| Error::Contract("skill destination has no parent".into()))?;
    fs::create_dir_all(parent)?;
    let temporary = TempDir::new_in(parent)?;
    let staged = temporary.path().join("skill");
    copy_directory(source, &staged)?;
    let backup_directory = TempDir::new_in(parent)?;
    let backup = backup_directory.path().join("previous");
    if destination.exists() {
        fs::rename(destination, &backup)?;
    }
    if let Err(error) = fs::rename(&staged, destination) {
        if backup.exists() {
            let _ = fs::rename(&backup, destination);
        }
        return Err(error.into());
    }
    Ok(())
}

fn copy_directory(source: &Path, destination: &Path) -> Result<()> {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let source_path = entry.path();
        if source_path.is_symlink() {
            return Err(Error::Contract(
                "release skills cannot contain symlinks".into(),
            ));
        }
        let destination_path = destination.join(entry.file_name());
        if source_path.is_dir() {
            copy_directory(&source_path, &destination_path)?;
        } else if source_path.is_file() {
            fs::copy(source_path, destination_path)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{BUILD_TARGET, UpdatePlan, Updater, checksum_for};

    #[test]
    fn checksum_parser_requires_the_exact_asset() {
        let digest = "a".repeat(64);
        let values = format!("{digest}  glr-1.2.3-x86_64-pc-windows-msvc.zip\n");
        assert_eq!(
            checksum_for(&values, "glr-1.2.3-x86_64-pc-windows-msvc.zip").unwrap(),
            digest
        );
        assert!(checksum_for(&values, "glr-1.2.3-x86_64-unknown-linux-gnu.zip").is_err());
    }

    #[test]
    fn test_constructor_keeps_network_override_private_to_tests() {
        assert!(Updater::with_api_url("http://127.0.0.1:9/latest".into()).is_ok());
    }

    #[test]
    fn downloads_reject_non_https_urls_before_network_access() {
        let updater = Updater::with_api_url("http://127.0.0.1:9/latest".into()).unwrap();
        let error = updater
            .get_limited("http://127.0.0.1:9/archive", 1024)
            .unwrap_err();
        assert!(error.to_string().contains("HTTPS"));
    }

    #[test]
    fn applying_an_up_to_date_plan_never_downloads_or_mutates() {
        let updater = Updater::with_api_url("http://127.0.0.1:9/latest".into()).unwrap();
        let result = updater
            .apply(
                UpdatePlan {
                    owner: "loonghao",
                    repository: "GameLearningRuntime",
                    current_version: "1.2.3".into(),
                    latest_version: "1.2.3".into(),
                    target: BUILD_TARGET,
                    version_update_available: false,
                    asset: format!("glr-1.2.3-{BUILD_TARGET}.zip"),
                    sha256: "a".repeat(64),
                    managed_components: vec!["cli", "runtime-host", "skills"],
                    integrity: "same-release-sha256",
                    archive_url: "http://127.0.0.1:9/archive".into(),
                },
                None,
            )
            .unwrap();
        assert!(!result.applied);
        assert!(!result.host_updated);
        assert!(!result.skills_updated);
    }
}
