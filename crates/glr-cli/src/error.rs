use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("contract violation: {0}")]
    Contract(String),
    #[error("invalid configuration: {0}")]
    Invalid(String),
    #[error("missing file or directory: {0}")]
    Missing(PathBuf),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("update request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("invalid semantic version: {0}")]
    Semver(#[from] semver::Error),
    #[error("invalid release archive: {0}")]
    Zip(#[from] zip::result::ZipError),
}

impl Error {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Contract(_) => "ContractViolation",
            Self::Invalid(_) => "ValueError",
            Self::Missing(_) => "FileNotFoundError",
            Self::Io(_) => "IoError",
            Self::Json(_) => "JsonError",
            Self::Sqlite(_) => "SqliteError",
            Self::Http(_) => "UpdateError",
            Self::Semver(_) => "VersionError",
            Self::Zip(_) => "ArchiveError",
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
