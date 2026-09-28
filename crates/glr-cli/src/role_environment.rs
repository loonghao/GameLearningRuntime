//! Declared per-role environment for `glr.project.v1`.
//!
//! A project manifest may declare the environment its roles receive:
//!
//! ```toml
//! [environment]
//! RENDER_DEVICE = "cpu"
//! DATASET_ROOT = "${SYNTHETIC_DATASET_ROOT}"
//!
//! [trainer.environment]
//! RENDER_DEVICE = "cuda"
//! ```
//!
//! Resolution is a pure function of the declaration and the process
//! environment:
//!
//! * a literal value passes through unchanged;
//! * `${NAME}` is interpolated from the **process** environment;
//! * a reference to a variable the process environment does not define fails
//!   closed and names the offending key;
//! * a variable the process environment already defines wins over the declared
//!   table, so an operator override is never shadowed by the manifest.
//!
//! The `GLR_` namespace belongs to the CLI, not to a project: the CLI clears
//! inherited `GLR_*` variables before it spawns a child and then publishes the
//! values it owns. A declared `GLR_*` key is therefore rejected while the
//! manifest is loaded, before any process starts.
//!
//! This mirrors `game_learning_runtime.role_environment` in the Python package.
//! Both sides accept the same tables and reject the same names; a manifest that
//! loads on one side loads on the other.

use std::collections::BTreeMap;
use std::process::Command;

use serde_json::{Value, json};

use crate::error::{Error, Result};

/// Namespace the CLI owns and republishes for every child it spawns.
pub const RESERVED_PREFIX: &str = "GLR_";

/// Substrings that mark a declared name as carrying a credential.
const SECRET_MARKERS: &[&str] = &[
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "APIKEY",
    "ACCESSKEY",
    "PRIVATEKEY",
];

/// Whether `name` may be used as an environment variable name.
///
/// Deliberately hand-rolled rather than pulled from a regular expression crate:
/// the shape is four ASCII rules, and this module is on the manifest load path
/// where an extra dependency is not worth one pattern.
fn is_valid_name(name: &str) -> bool {
    let mut characters = name.chars();
    match characters.next() {
        Some(first) if first.is_ascii_alphabetic() || first == '_' => {}
        _ => return false,
    }
    characters.all(|character| character.is_ascii_alphanumeric() || character == '_')
}

/// Report whether a declared name looks like it carries a credential.
///
/// A conservative lexical test: the manifest does not mark secrets, so a name
/// that reads like a credential is treated as one and its resolved value is
/// kept out of run records and `doctor` output.
pub fn is_secret_name(name: &str) -> bool {
    let upper = name.to_ascii_uppercase();
    if SECRET_MARKERS.iter().any(|marker| upper.contains(marker)) {
        return true;
    }
    upper.split('_').any(|token| token == "KEY")
}

/// Validate one declared `environment` table.
///
/// Whether a referenced variable exists is a property of the process
/// environment, not of the manifest, so it is decided at resolution time.
pub fn validate_table(table: &BTreeMap<String, String>, label: &str) -> Result<()> {
    for (name, raw) in table {
        if !is_valid_name(name) {
            return Err(Error::Invalid(format!(
                "{label} keys must match ^[A-Za-z_][A-Za-z0-9_]*$: {name:?}"
            )));
        }
        if name.starts_with(RESERVED_PREFIX) {
            return Err(Error::Invalid(format!(
                "{label} cannot declare {name:?}: the {RESERVED_PREFIX}* namespace belongs to the CLI"
            )));
        }
        // C0 only, matching `game.environment` and the Python side: `is_control`
        // would also reject U+007F-U+009F, which the other two validators accept.
        if raw.chars().any(|character| (character as u32) < 32) {
            return Err(Error::Invalid(format!(
                "{label}.{name} must be a printable string"
            )));
        }
        validate_references(raw, &format!("{label}.{name}"))?;
    }
    Ok(())
}

/// Reject `${...}` tokens that are not a reference to a valid variable name.
///
/// A `${` with no closing brace is an ordinary literal, matching the Python
/// side, so it is left alone rather than rejected.
fn validate_references(raw: &str, label: &str) -> Result<()> {
    let mut index = 0;
    while let Some(offset) = raw[index..].find("${") {
        let start = index + offset + 2;
        let Some(relative_end) = raw[start..].find('}') else {
            break;
        };
        let end = start + relative_end;
        let name = &raw[start..end];
        if !is_valid_name(name) {
            return Err(Error::Invalid(format!(
                "{label} has a malformed reference ${{{name}}}: write ${{NAME}} with a valid variable name"
            )));
        }
        index = end + 1;
    }
    Ok(())
}

/// The `${NAME}` references a declared value depends on, in order of appearance.
fn references(raw: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut index = 0;
    while let Some(offset) = raw[index..].find("${") {
        let start = index + offset + 2;
        let Some(relative_end) = raw[start..].find('}') else {
            break;
        };
        let end = start + relative_end;
        let name = raw[start..end].to_string();
        if !found.contains(&name) {
            found.push(name);
        }
        index = end + 1;
    }
    found
}

/// Expand every `${NAME}` reference in `raw` from `environ`.
///
/// Callers pass only references they have already confirmed are present, so a
/// missing value is a programming error rather than a user-facing condition.
fn interpolate(raw: &str, environ: &BTreeMap<String, String>) -> String {
    let mut result = String::with_capacity(raw.len());
    let mut index = 0;
    while let Some(offset) = raw[index..].find("${") {
        let start = index + offset + 2;
        let Some(relative_end) = raw[start..].find('}') else {
            break;
        };
        let end = start + relative_end;
        result.push_str(&raw[index..start - 2]);
        let name = &raw[start..end];
        result.push_str(environ.get(name).map(String::as_str).unwrap_or_default());
        index = end + 1;
    }
    result.push_str(&raw[index..]);
    result
}

/// Merge the project-wide table with one role's table; the role wins.
pub fn merge_tables(
    project_table: &BTreeMap<String, String>,
    role_table: Option<&BTreeMap<String, String>>,
) -> BTreeMap<String, String> {
    let mut merged = project_table.clone();
    if let Some(role_table) = role_table {
        merged.extend(
            role_table
                .iter()
                .map(|(name, value)| (name.clone(), value.clone())),
        );
    }
    merged
}

/// Where a resolved value came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// The ambient process environment already defined the name.
    Process,
    /// A `${NAME}` reference was expanded from the process environment.
    Interpolated,
    /// The manifest value passed through unchanged.
    Literal,
}

impl Source {
    fn as_str(self) -> &'static str {
        match self {
            Self::Process => "process",
            Self::Interpolated => "interpolated",
            Self::Literal => "literal",
        }
    }
}

/// One declared variable after resolution.
#[derive(Debug, Clone)]
pub struct ResolvedVariable {
    pub name: String,
    pub value: String,
    pub source: Source,
    pub secret: bool,
}

impl ResolvedVariable {
    /// Report shape shared by `doctor` output and run records.
    ///
    /// The value is present only for non-secret names, so a credential a role
    /// received is reported as received and never as content.
    fn to_value(&self) -> Value {
        if self.secret {
            json!({"name": self.name, "source": self.source.as_str(), "secret": true})
        } else {
            json!({
                "name": self.name,
                "source": self.source.as_str(),
                "secret": false,
                "value": self.value,
            })
        }
    }
}

/// A declared variable that could not be resolved, and why.
#[derive(Debug, Clone)]
pub struct UnresolvedVariable {
    pub name: String,
    pub missing: Vec<String>,
}

impl UnresolvedVariable {
    fn to_value(&self) -> Value {
        json!({"name": self.name, "missing": self.missing})
    }

    fn reason(&self) -> String {
        let names = self
            .missing
            .iter()
            .map(|name| format!("${{{name}}}"))
            .collect::<Vec<_>>()
            .join(", ");
        format!("process environment defines no {names}")
    }
}

/// The declared environment one role receives after resolution.
#[derive(Debug, Clone, Default)]
pub struct RoleEnvironment {
    pub role: Option<String>,
    pub variables: Vec<ResolvedVariable>,
    pub unresolved: Vec<UnresolvedVariable>,
}

impl RoleEnvironment {
    /// True when every declared variable resolved.
    #[must_use]
    pub fn ready(&self) -> bool {
        self.unresolved.is_empty()
    }

    /// Human-readable subject for a failure message.
    fn target(&self) -> String {
        match &self.role {
            Some(role) => format!("role {role:?}"),
            None => "the declared role environment".to_string(),
        }
    }

    /// Fail-closed message naming the first variable that did not resolve.
    #[must_use]
    pub fn refusal(&self) -> String {
        match self.unresolved.first() {
            Some(first) => format!(
                "{} cannot resolve {:?}: {}",
                self.target(),
                first.name,
                first.reason()
            ),
            None => String::new(),
        }
    }

    /// Report shape shared by `doctor` output and run records.
    #[must_use]
    pub fn to_value(&self) -> Value {
        json!({
            "role": self.role,
            "ready": self.ready(),
            "variables": self.variables.iter().map(ResolvedVariable::to_value).collect::<Vec<_>>(),
            "unresolved": self.unresolved.iter().map(UnresolvedVariable::to_value).collect::<Vec<_>>(),
        })
    }

    /// Publish the resolved values on a child process.
    ///
    /// A name the process environment already defines keeps its own value, so
    /// an operator override outranks the declared table here exactly as it does
    /// when the value was resolved.
    pub fn apply(&self, process: &mut Command) {
        for variable in &self.variables {
            if std::env::var_os(&variable.name).is_none() {
                process.env(&variable.name, &variable.value);
            }
        }
    }
}

/// Resolve a declared table against `environ`.
///
/// The real process environment outranks the declaration: a name `environ`
/// already defines keeps its own value, and a declared value that references a
/// missing variable only fails when nothing else supplied that name.
#[must_use]
pub fn resolve_with(
    table: &BTreeMap<String, String>,
    environ: &BTreeMap<String, String>,
    role: Option<&str>,
) -> RoleEnvironment {
    let mut resolved = RoleEnvironment {
        role: role.map(str::to_string),
        ..RoleEnvironment::default()
    };
    for (name, raw) in table {
        let secret = is_secret_name(name);
        if let Some(value) = environ.get(name) {
            resolved.variables.push(ResolvedVariable {
                name: name.clone(),
                value: value.clone(),
                source: Source::Process,
                secret,
            });
            continue;
        }
        let references = references(raw);
        if references.is_empty() {
            resolved.variables.push(ResolvedVariable {
                name: name.clone(),
                value: raw.clone(),
                source: Source::Literal,
                secret,
            });
            continue;
        }
        let missing: Vec<String> = references
            .into_iter()
            .filter(|reference| !environ.contains_key(reference))
            .collect();
        if !missing.is_empty() {
            resolved.unresolved.push(UnresolvedVariable {
                name: name.clone(),
                missing,
            });
            continue;
        }
        resolved.variables.push(ResolvedVariable {
            name: name.clone(),
            value: interpolate(raw, environ),
            source: Source::Interpolated,
            secret,
        });
    }
    resolved
}

/// Resolve a declared table against the current process environment.
#[must_use]
pub fn resolve(table: &BTreeMap<String, String>, role: Option<&str>) -> RoleEnvironment {
    resolve_with(table, &std::env::vars().collect(), role)
}

/// Fail closed, before anything starts, when a declared variable cannot resolve.
pub fn require(table: &BTreeMap<String, String>, role: &str) -> Result<RoleEnvironment> {
    let resolved = resolve(table, Some(role));
    if resolved.ready() {
        Ok(resolved)
    } else {
        Err(Error::Contract(resolved.refusal()))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        RESERVED_PREFIX, Source, is_secret_name, is_valid_name, merge_tables, references,
        resolve_with, validate_references, validate_table,
    };
    use std::collections::BTreeMap;

    fn table(entries: &[(&str, &str)]) -> BTreeMap<String, String> {
        entries
            .iter()
            .map(|(name, value)| ((*name).to_string(), (*value).to_string()))
            .collect()
    }

    /// Resolve without touching the real process environment: concurrent tests
    /// must not observe each other's variables, and edition 2024 makes
    /// mutating it unsafe anyway.
    fn resolve(
        declared: &BTreeMap<String, String>,
        environ: &[(&str, &str)],
        role: Option<&str>,
    ) -> super::RoleEnvironment {
        resolve_with(declared, &table(environ), role)
    }

    #[test]
    fn detects_secret_names_lexically() {
        assert!(is_secret_name("API_TOKEN"));
        assert!(is_secret_name("TRAIN_PASSWORD"));
        assert!(is_secret_name("MY_KEY"));
        assert!(!is_secret_name("DATASET_ROOT"));
        assert!(!is_secret_name("MONKEY_MODE"));
    }

    #[test]
    fn accepts_only_environment_variable_names() {
        assert!(is_valid_name("RENDER_DEVICE"));
        assert!(is_valid_name("_PRIVATE"));
        assert!(!is_valid_name(""));
        assert!(!is_valid_name("9LIVES"));
        assert!(!is_valid_name("NOT A KEY"));
        assert!(!is_valid_name("A-B"));
    }

    #[test]
    fn rejects_reserved_and_malformed_declarations() {
        validate_table(&table(&[("MODE", "synthetic")]), "project.environment")
            .expect("a plain declaration must load");

        let reserved = table(&[("GLR_RUN_ID", "forged")]);
        let error = validate_table(&reserved, "project.environment").expect_err("GLR_ is reserved");
        assert!(error.to_string().contains(RESERVED_PREFIX));

        let malformed = table(&[("NOT A KEY", "value")]);
        assert!(validate_table(&malformed, "project.environment").is_err());

        let control = table(&[("MODE", "value\nwith\tcontrol")]);
        assert!(validate_table(&control, "project.environment").is_err());
        let unit_separator = table(&[("MODE", "value\u{1f}sep")]);
        assert!(validate_table(&unit_separator, "project.environment").is_err());
    }

    #[test]
    fn control_characters_are_rejected_from_c0_only() {
        // C0 is rejected; DEL and C1 are accepted, matching `game.environment`
        // and the Python side. A wider predicate here would reject a
        // declaration the other entry point accepts.
        assert!(validate_table(&table(&[("MODE", "a\nb")]), "project.environment").is_err());
        validate_table(&table(&[("MODE", "a\u{7f}b")]), "project.environment")
            .expect("DEL is outside C0 and must load");
        validate_table(&table(&[("MODE", "a\u{85}b")]), "project.environment")
            .expect("C1 is outside C0 and must load");
    }

    #[test]
    fn rejects_malformed_references_but_keeps_plain_braces() {
        assert!(validate_references("${ROOT}/v1", "project.environment.MODE").is_ok());
        assert!(validate_references("${}", "project.environment.MODE").is_err());
        assert!(validate_references("${BAD-NAME}", "project.environment.MODE").is_err());
        // An unterminated reference is a literal, not an error.
        assert!(validate_references("${UNCLOSED", "project.environment.MODE").is_ok());
        assert_eq!(references("${A}/${B}/${A}"), vec!["A", "B"]);
    }

    #[test]
    fn the_role_table_wins_key_by_key() {
        let project_table = table(&[("RENDER_DEVICE", "cpu"), ("DATASET_ROOT", "datasets")]);
        let role_table = table(&[("RENDER_DEVICE", "cuda")]);

        let merged = merge_tables(&project_table, Some(&role_table));

        assert_eq!(merged["RENDER_DEVICE"], "cuda");
        assert_eq!(merged["DATASET_ROOT"], "datasets");
        assert_eq!(merge_tables(&project_table, None)["RENDER_DEVICE"], "cpu");
    }

    #[test]
    fn literals_and_references_resolve_from_the_process() {
        let declared = table(&[
            ("MODE", "synthetic"),
            ("DATASET_ROOT", "${SYNTHETIC_ROOT}/v1"),
        ]);

        let resolved = resolve(
            &declared,
            &[("SYNTHETIC_ROOT", "/tmp/synthetic")],
            Some("trainer"),
        );

        assert!(resolved.ready());
        let values: BTreeMap<_, _> = resolved
            .variables
            .iter()
            .map(|variable| (variable.name.as_str(), variable.value.as_str()))
            .collect();
        assert_eq!(values["MODE"], "synthetic");
        assert_eq!(values["DATASET_ROOT"], "/tmp/synthetic/v1");
        let sources: Vec<_> = resolved.variables.iter().map(|v| v.source).collect();
        assert_eq!(sources, vec![Source::Interpolated, Source::Literal]);
    }

    #[test]
    fn a_missing_variable_fails_closed_and_names_the_key() {
        let declared = table(&[("DATASET_ROOT", "${SYNTHETIC_MISSING}")]);

        let resolved = resolve(&declared, &[], Some("trainer"));

        assert!(!resolved.ready());
        assert_eq!(resolved.unresolved[0].name, "DATASET_ROOT");
        assert_eq!(resolved.unresolved[0].missing, vec!["SYNTHETIC_MISSING"]);
        assert!(resolved.refusal().contains("DATASET_ROOT"));
        assert!(resolved.refusal().contains("${SYNTHETIC_MISSING}"));
        assert!(resolved.variables.is_empty());
    }

    #[test]
    fn the_process_environment_outranks_the_declared_table() {
        let declared = table(&[("SYNTHETIC_MODE", "${SYNTHETIC_MISSING}")]);

        let resolved = resolve(
            &declared,
            &[("SYNTHETIC_MODE", "from-operator")],
            Some("trainer"),
        );

        assert!(resolved.ready());
        assert_eq!(resolved.variables[0].value, "from-operator");
        assert_eq!(resolved.variables[0].source, Source::Process);
    }

    #[test]
    fn the_process_environment_outranks_a_declared_literal() {
        let declared = table(&[("SYNTHETIC_MODE", "from-manifest")]);

        let resolved = resolve(
            &declared,
            &[("SYNTHETIC_MODE", "from-operator")],
            Some("trainer"),
        );

        assert_eq!(resolved.variables[0].value, "from-operator");
        assert_eq!(resolved.variables[0].source, Source::Process);
    }

    #[test]
    fn a_partial_reference_keeps_its_literal_surroundings() {
        let declared = table(&[("ENDPOINT", "http://${SYNTHETIC_HOST}:8080/v1")]);

        let resolved = resolve(&declared, &[("SYNTHETIC_HOST", "127.0.0.1")], None);

        assert_eq!(resolved.variables[0].value, "http://127.0.0.1:8080/v1");
    }

    #[test]
    fn a_secret_value_is_never_reported_as_content() {
        let declared = table(&[("API_TOKEN", "synthetic-secret-value")]);

        let resolved = resolve(&declared, &[], Some("trainer"));
        let report = resolved.to_value();

        assert!(resolved.variables[0].secret);
        assert_eq!(report["variables"][0]["secret"], true);
        assert!(report["variables"][0].get("value").is_none());
        assert!(!report.to_string().contains("synthetic-secret-value"));
    }
}
