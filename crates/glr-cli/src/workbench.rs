//! Optional, game-neutral presentation contract carried by durable bridge.state events.
use crate::error::{Error, Result};
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashSet;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Workbench {
    schema_version: String,
    title: String,
    agent: Option<String>,
    objective: Option<String>,
    phase: Option<String>,
    sections: Vec<Section>,
}
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase", deny_unknown_fields)]
enum Section {
    Stats {
        id: String,
        title: String,
        fields: Vec<Field>,
    },
    Table {
        id: String,
        title: String,
        columns: Vec<String>,
        rows: Vec<Vec<Value>>,
    },
    Text {
        id: String,
        title: String,
        text: String,
    },
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Field {
    label: String,
    value: Value,
    unit: Option<String>,
}
fn text(s: &str, max: usize) -> bool {
    !s.is_empty() && s.chars().count() <= max
}
fn cell(v: &Value) -> bool {
    v.is_null()
        || v.is_boolean()
        || v.as_f64().is_some_and(f64::is_finite)
        || v.as_str().is_some_and(|s| s.chars().count() <= 240)
}
pub fn validate(value: &Value) -> Result<()> {
    // Optional fields are strings when present; null is not an alias for omission.
    let optional = ["agent", "objective", "phase"];
    if optional
        .iter()
        .any(|k| value.get(k).is_some_and(Value::is_null))
    {
        return Err(Error::Invalid(
            "workbench optional fields must be strings".into(),
        ));
    }
    let view: Workbench = serde_json::from_value(value.clone())?;
    let mut valid = view.schema_version == "glr.workbench.v1"
        && text(&view.title, 120)
        && view.sections.len() <= 8
        && view.agent.as_deref().is_none_or(|v| text(v, 128))
        && view.objective.as_deref().is_none_or(|v| text(v, 2048))
        && view.phase.as_deref().is_none_or(|v| text(v, 80));
    let mut ids = HashSet::new();
    for section in &view.sections {
        let (id, title, content) = match section {
            Section::Stats { id, title, fields } => (
                id,
                title,
                (1..=12).contains(&fields.len())
                    && fields.iter().all(|f| {
                        text(&f.label, 80)
                            && cell(&f.value)
                            && f.unit.as_deref().is_none_or(|s| text(s, 32))
                    }),
            ),
            Section::Table {
                id,
                title,
                columns,
                rows,
            } => (
                id,
                title,
                (1..=8).contains(&columns.len())
                    && columns.iter().all(|c| text(c, 80))
                    && rows.len() <= 40
                    && rows
                        .iter()
                        .all(|r| r.len() == columns.len() && r.iter().all(cell)),
            ),
            Section::Text {
                id,
                title,
                text: value,
            } => (id, title, text(value, 4000)),
        };
        valid &= text(id, 64)
            && id.as_bytes().first().is_some_and(u8::is_ascii_alphabetic)
            && id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b))
            && ids.insert(id)
            && text(title, 120)
            && content;
    }
    // Preserve the same optional-field semantics as the published schema.
    if let Some(sections) = value["sections"].as_array() {
        valid &= sections.iter().all(|s| {
            s.get("fields")
                .and_then(Value::as_array)
                .is_none_or(|fields| {
                    fields
                        .iter()
                        .all(|f| !f.get("unit").is_some_and(Value::is_null))
                })
        });
    }
    if !valid {
        return Err(Error::Invalid(
            "invalid glr.workbench.v1 view or exceeded display limits".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_examples_and_invalid_views_match_the_contract() {
        let fixture: Value =
            serde_json::from_str(include_str!("../../../docs/examples/workbench-views.json"))
                .unwrap();
        for case in fixture["valid"].as_array().unwrap() {
            validate(case).unwrap();
        }
        for case in fixture["invalid"].as_array().unwrap() {
            assert!(validate(case).is_err(), "{case}");
        }
    }
}
