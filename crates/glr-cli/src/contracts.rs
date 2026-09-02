use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read};
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{Error, Result};
use crate::project::validate_identifier;

pub const AGENT_GOAL_SCHEMA_VERSION: &str = "glr.agent-goal.v1";
pub const CAPTURE_FRAME_SCHEMA_VERSION: &str = "glr.capture-frame.v1";
pub const CAPTURE_MANIFEST_SCHEMA_VERSION: &str = "glr.capture.v1";
pub const GOAL_EVIDENCE_SCHEMA_VERSION: &str = "glr.goal-evidence.v1";
pub const MODEL_BUNDLE_SCHEMA_VERSION: &str = "glr.model-bundle.v1";
pub const RESEARCH_BUNDLE_SCHEMA_VERSION: &str = "glr.research-bundle.v1";
pub const SPATIAL_KNOWLEDGE_SCHEMA_VERSION: &str = "glr.spatial-knowledge.v1";
pub const SPATIAL_KNOWLEDGE_V2_SCHEMA_VERSION: &str = "glr.spatial-knowledge.v2";
pub const TRIAL_PLAN_SCHEMA_VERSION: &str = "glr.trial-plan.v1";

pub fn read_json<T: for<'de> Deserialize<'de>>(path: &Path, label: &str) -> Result<T> {
    if path.is_symlink() || !path.is_file() {
        return Err(Error::Missing(path.to_path_buf()));
    }
    let metadata = path.metadata()?;
    if metadata.len() > 8 * 1024 * 1024 {
        return Err(Error::Invalid(format!("{label} exceeds the 8 MiB limit")));
    }
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

pub fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    fs::write(path, bytes)?;
    Ok(())
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn validate_schema(actual: &str, expected: &str, label: &str) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(Error::Invalid(format!(
            "{label}.schema_version must be {expected:?}"
        )))
    }
}

fn validate_text(value: &str, label: &str, maximum: usize) -> Result<()> {
    if value.trim().is_empty()
        || value.chars().count() > maximum
        || value
            .chars()
            .any(|character| character.is_control() && !matches!(character, '\n' | '\t'))
    {
        Err(Error::Invalid(format!(
            "{label} must be non-empty bounded text up to {maximum} characters"
        )))
    } else {
        Ok(())
    }
}

fn validate_confidence(value: f64, label: &str) -> Result<()> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(Error::Invalid(format!("{label} must be between 0 and 1")))
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "kebab-case")]
pub enum GoalOperator {
    Gte,
    Lte,
    Eq,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum PromotionMode {
    Max,
    Min,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointPromotion {
    pub metric: String,
    pub mode: PromotionMode,
}

impl CheckpointPromotion {
    pub fn validate(&self) -> Result<()> {
        validate_identifier(&self.metric, "goal.promotion.metric")
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "kebab-case")]
pub enum ResearchMediaType {
    OfficialRules,
    TextGuide,
    VideoTutorial,
    RuntimeTrace,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "kebab-case")]
pub enum ResearchCategory {
    Mechanic,
    Strategy,
    RewardHypothesis,
    Safety,
    Navigation,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ResearchStatus {
    Unverified,
    RuntimeVerified,
    Rejected,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ResearchScope {
    Environment,
    Family,
    Generic,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum Authority {
    Advisory,
    Authoritative,
}

impl Authority {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Advisory => "advisory",
            Self::Authoritative => "authoritative",
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GoalBudget {
    pub max_trials: u32,
    pub max_training_steps: u64,
    pub max_wall_seconds: u64,
    pub max_research_sources: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SuccessCriterion {
    pub metric: String,
    pub operator: GoalOperator,
    pub target: f64,
    pub source: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AgentGoal {
    pub schema_version: String,
    pub goal_id: String,
    pub objective: String,
    pub environment_family: String,
    pub success_criteria: Vec<SuccessCriterion>,
    pub budget: GoalBudget,
    pub allowed_research_media: Vec<ResearchMediaType>,
    #[serde(default)]
    pub promotion: Option<CheckpointPromotion>,
}

impl AgentGoal {
    pub fn validate(&self) -> Result<()> {
        validate_schema(&self.schema_version, AGENT_GOAL_SCHEMA_VERSION, "goal")?;
        validate_identifier(&self.goal_id, "goal.goal_id")?;
        validate_identifier(&self.environment_family, "goal.environment_family")?;
        validate_text(&self.objective, "goal.objective", 2048)?;
        if self.success_criteria.is_empty() {
            return Err(Error::Invalid("success_criteria cannot be empty".into()));
        }
        if self.budget.max_trials == 0
            || self.budget.max_training_steps == 0
            || self.budget.max_wall_seconds == 0
            || self.budget.max_research_sources == 0
        {
            return Err(Error::Invalid("goal budgets must be positive".into()));
        }
        if self.allowed_research_media.is_empty()
            || self
                .allowed_research_media
                .iter()
                .copied()
                .collect::<HashSet<_>>()
                .len()
                != self.allowed_research_media.len()
        {
            return Err(Error::Invalid(
                "allowed_research_media must be non-empty and unique".into(),
            ));
        }
        let mut keys = HashSet::new();
        for criterion in &self.success_criteria {
            validate_identifier(&criterion.metric, "criterion.metric")?;
            validate_identifier(&criterion.source, "criterion.source")?;
            if !criterion.target.is_finite() {
                return Err(Error::Invalid("criterion.target must be finite".into()));
            }
            if !keys.insert((&criterion.metric, &criterion.source)) {
                return Err(Error::Invalid(
                    "success_criteria contains duplicate metric/source pairs".into(),
                ));
            }
        }
        if let Some(promotion) = &self.promotion {
            promotion.validate()?;
        }
        Ok(())
    }

    pub fn evaluate(&self, evidence: &[GoalEvidence]) -> Result<GoalEvaluation> {
        let mut results = Vec::with_capacity(self.success_criteria.len());
        for criterion in &self.success_criteria {
            let matched = evidence
                .iter()
                .find(|item| item.metric == criterion.metric && item.source == criterion.source);
            let passed = matched.is_some_and(|item| {
                item.authority == Authority::Authoritative
                    && match criterion.operator {
                        GoalOperator::Gte => item.value >= criterion.target,
                        GoalOperator::Lte => item.value <= criterion.target,
                        GoalOperator::Eq => {
                            (item.value - criterion.target).abs()
                                <= 1e-12_f64.max(1e-9 * criterion.target.abs())
                        }
                    }
            });
            results.push(CriterionEvaluation {
                metric: criterion.metric.clone(),
                operator: criterion.operator,
                target: criterion.target,
                source: criterion.source.clone(),
                observed: matched.map(|item| item.value),
                evidence_run_id: matched.map(|item| item.run_id.clone()),
                passed,
            });
        }
        Ok(GoalEvaluation {
            goal_id: self.goal_id.clone(),
            satisfied: results.iter().all(|item| item.passed),
            criteria: results,
        })
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CriterionEvaluation {
    pub metric: String,
    pub operator: GoalOperator,
    pub target: f64,
    pub source: String,
    pub observed: Option<f64>,
    pub evidence_run_id: Option<String>,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct GoalEvaluation {
    pub goal_id: String,
    pub satisfied: bool,
    pub criteria: Vec<CriterionEvaluation>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResearchSource {
    pub source_id: String,
    pub media_type: ResearchMediaType,
    pub url: String,
    pub publisher: String,
    pub title: String,
    pub accessed_at: String,
    pub updated_at: Option<String>,
    pub summary: String,
    pub confidence: f64,
    pub volatility: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResearchFinding {
    pub finding_id: String,
    pub category: ResearchCategory,
    pub status: ResearchStatus,
    pub scope: ResearchScope,
    pub scope_id: Option<String>,
    pub summary: String,
    pub source_ids: Vec<String>,
    pub tags: Vec<String>,
    pub confidence: f64,
    pub locator: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResearchBundle {
    pub schema_version: String,
    pub sources: Vec<ResearchSource>,
    pub findings: Vec<ResearchFinding>,
}

impl ResearchBundle {
    pub fn validate(&self) -> Result<()> {
        validate_schema(
            &self.schema_version,
            RESEARCH_BUNDLE_SCHEMA_VERSION,
            "research bundle",
        )?;
        if self.sources.len() > 256 || self.findings.len() > 2048 {
            return Err(Error::Invalid(
                "research bundle exceeds its bounded source or finding count".into(),
            ));
        }
        let mut source_ids = HashSet::new();
        for source in &self.sources {
            validate_identifier(&source.source_id, "research source.source_id")?;
            if !source.url.starts_with("https://") || source.url.contains('@') {
                return Err(Error::Invalid(
                    "research source.url must be an HTTPS URL without credentials".into(),
                ));
            }
            validate_text(&source.publisher, "research source.publisher", 256)?;
            validate_text(&source.title, "research source.title", 512)?;
            validate_text(&source.summary, "research source.summary", 1024)?;
            validate_confidence(source.confidence, "research source.confidence")?;
            if !source_ids.insert(source.source_id.as_str()) {
                return Err(Error::Invalid(
                    "research bundle contains duplicate source_id values".into(),
                ));
            }
        }
        let mut finding_ids = HashSet::new();
        for finding in &self.findings {
            validate_identifier(&finding.finding_id, "research finding.finding_id")?;
            validate_text(&finding.summary, "research finding.summary", 1024)?;
            validate_confidence(finding.confidence, "research finding.confidence")?;
            if finding.source_ids.is_empty()
                || finding.source_ids.iter().collect::<HashSet<_>>().len()
                    != finding.source_ids.len()
            {
                return Err(Error::Invalid(
                    "research finding.source_ids must be non-empty and unique".into(),
                ));
            }
            if finding
                .source_ids
                .iter()
                .any(|source_id| !source_ids.contains(source_id.as_str()))
            {
                return Err(Error::Invalid(format!(
                    "research finding {:?} references unknown sources",
                    finding.finding_id
                )));
            }
            match finding.scope {
                ResearchScope::Generic if finding.scope_id.is_some() => {
                    return Err(Error::Invalid(
                        "generic research findings cannot declare scope_id".into(),
                    ));
                }
                ResearchScope::Environment | ResearchScope::Family => {
                    validate_identifier(
                        finding.scope_id.as_deref().unwrap_or_default(),
                        "research finding.scope_id",
                    )?;
                }
                ResearchScope::Generic => {}
            }
            if finding.status == ResearchStatus::RuntimeVerified
                && !self.sources.iter().any(|source| {
                    finding.source_ids.contains(&source.source_id)
                        && source.media_type == ResearchMediaType::RuntimeTrace
                })
            {
                return Err(Error::Invalid(
                    "runtime-verified finding requires runtime trace provenance".into(),
                ));
            }
            if !finding_ids.insert(finding.finding_id.as_str()) {
                return Err(Error::Invalid(
                    "research bundle contains duplicate finding_id values".into(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RewardTerm {
    pub name: String,
    pub metric: String,
    pub weight: f64,
    pub rationale: String,
    pub source_finding_ids: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrialPlan {
    pub schema_version: String,
    pub trial_id: String,
    pub goal_id: String,
    pub seed: u64,
    pub max_steps: u64,
    pub reward_terms: Vec<RewardTerm>,
    pub notes: String,
}

impl TrialPlan {
    pub fn validate(&self) -> Result<()> {
        validate_schema(
            &self.schema_version,
            TRIAL_PLAN_SCHEMA_VERSION,
            "trial plan",
        )?;
        validate_identifier(&self.trial_id, "trial plan.trial_id")?;
        validate_identifier(&self.goal_id, "trial plan.goal_id")?;
        validate_text(&self.notes, "trial plan.notes", 4096)?;
        if self.max_steps == 0 {
            return Err(Error::Invalid(
                "trial plan.max_steps must be positive".into(),
            ));
        }
        if self.reward_terms.len() > 128 {
            return Err(Error::Invalid(
                "trial plan cannot contain more than 128 reward terms".into(),
            ));
        }
        let mut names = HashSet::new();
        for term in &self.reward_terms {
            validate_identifier(&term.name, "reward term.name")?;
            validate_identifier(&term.metric, "reward term.metric")?;
            validate_text(&term.rationale, "reward term.rationale", 1024)?;
            if !term.weight.is_finite() || term.weight.abs() > 1000.0 {
                return Err(Error::Invalid(
                    "reward term.weight must be between -1000 and 1000".into(),
                ));
            }
            if term.source_finding_ids.iter().collect::<HashSet<_>>().len()
                != term.source_finding_ids.len()
            {
                return Err(Error::Invalid(
                    "reward term.source_finding_ids must be unique".into(),
                ));
            }
            if !names.insert(term.name.as_str()) {
                return Err(Error::Invalid(
                    "trial plan reward term names must be unique".into(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GoalEvidence {
    pub metric: String,
    pub value: f64,
    pub source: String,
    pub authority: Authority,
    pub run_id: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GoalEvidenceBundle {
    pub schema_version: String,
    pub goal_id: String,
    pub trial_id: String,
    pub evidence: Vec<GoalEvidence>,
}

impl GoalEvidenceBundle {
    pub fn validate(&self) -> Result<()> {
        validate_schema(
            &self.schema_version,
            GOAL_EVIDENCE_SCHEMA_VERSION,
            "goal evidence",
        )?;
        validate_identifier(&self.goal_id, "goal evidence.goal_id")?;
        validate_identifier(&self.trial_id, "goal evidence.trial_id")?;
        let mut keys = HashSet::new();
        for evidence in &self.evidence {
            validate_identifier(&evidence.metric, "evidence.metric")?;
            validate_identifier(&evidence.source, "evidence.source")?;
            validate_identifier(&evidence.run_id, "evidence.run_id")?;
            if !evidence.value.is_finite() {
                return Err(Error::Invalid("evidence.value must be finite".into()));
            }
            if !keys.insert((&evidence.metric, &evidence.source)) {
                return Err(Error::Invalid(
                    "goal evidence contains duplicate metric/source pairs".into(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BundleFile {
    pub path: String,
    pub sha256: String,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelBundleManifest {
    pub schema_version: String,
    pub environment_id: String,
    pub protocol_version: String,
    pub algorithm: String,
    pub framework: String,
    pub framework_version: String,
    pub seeds: Vec<u64>,
    pub inputs: Vec<BundleFile>,
    pub artifacts: Vec<BundleFile>,
}

pub fn verify_model_bundle(root: &Path) -> Result<ModelBundleManifest> {
    let manifest: ModelBundleManifest = read_json(&root.join("manifest.json"), "model bundle")?;
    validate_schema(
        &manifest.schema_version,
        MODEL_BUNDLE_SCHEMA_VERSION,
        "model bundle",
    )?;
    validate_identifier(&manifest.environment_id, "model environment_id")?;
    if manifest.seeds.is_empty() || manifest.inputs.is_empty() || manifest.artifacts.is_empty() {
        return Err(Error::Invalid(
            "model bundle seeds, inputs, and artifacts cannot be empty".into(),
        ));
    }
    for (group, entries) in [
        ("inputs", &manifest.inputs),
        ("artifacts", &manifest.artifacts),
    ] {
        for entry in entries {
            validate_portable_path(&entry.path, "model bundle file")?;
            let path = root.join(group).join(Path::new(&entry.path));
            if path.is_symlink() || !path.is_file() {
                return Err(Error::Missing(path));
            }
            if path.metadata()?.len() != entry.size_bytes || sha256_file(&path)? != entry.sha256 {
                return Err(Error::Contract(format!(
                    "model bundle file failed size or sha256 verification: {group}/{}",
                    entry.path
                )));
            }
        }
    }
    Ok(manifest)
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureFrame {
    pub schema_version: String,
    pub run_id: String,
    pub episode_id: String,
    pub step_id: u64,
    pub frame_index: u64,
    pub pts_ns: u64,
    pub observation_timestamp_ns: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureFile {
    pub path: String,
    pub sha256: String,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureManifest {
    pub schema_version: String,
    pub environment_id: String,
    pub run_id: String,
    pub video: CaptureFile,
    pub index: CaptureFile,
    pub codec: String,
    pub frame_rate: f64,
    pub width: u32,
    pub height: u32,
    pub frames: Vec<CaptureFrame>,
}

pub struct CaptureManifestInput<'a> {
    pub manifest_path: &'a Path,
    pub environment_id: &'a str,
    pub run_id: &'a str,
    pub video_path: &'a Path,
    pub index_path: &'a Path,
    pub codec: &'a str,
    pub frame_rate: f64,
    pub width: u32,
    pub height: u32,
}

pub fn build_capture_manifest(input: CaptureManifestInput<'_>) -> Result<CaptureManifest> {
    let parent = input
        .manifest_path
        .parent()
        .ok_or_else(|| Error::Invalid("capture manifest has no parent".into()))?;
    for path in [input.video_path, input.index_path] {
        if path.is_symlink() || !path.is_file() || !path.starts_with(parent) {
            return Err(Error::Contract(
                "capture files must be regular files inside the manifest directory".into(),
            ));
        }
    }
    let mut frames = Vec::new();
    let reader = BufReader::new(File::open(input.index_path)?);
    let mut previous: Option<(u64, u64)> = None;
    for (line_number, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let frame: CaptureFrame = serde_json::from_str(&line).map_err(|error| {
            Error::Invalid(format!("capture index line {}: {error}", line_number + 1))
        })?;
        validate_schema(
            &frame.schema_version,
            CAPTURE_FRAME_SCHEMA_VERSION,
            "capture frame",
        )?;
        if frame.run_id != input.run_id {
            return Err(Error::Contract(
                "capture index contains a different run_id".into(),
            ));
        }
        if previous.is_some_and(|value| (frame.frame_index, frame.pts_ns) <= value) {
            return Err(Error::Contract(
                "capture frame indexes and timestamps must be monotonic".into(),
            ));
        }
        previous = Some((frame.frame_index, frame.pts_ns));
        frames.push(frame);
    }
    if frames.is_empty() {
        return Err(Error::Contract(
            "capture index must contain at least one frame".into(),
        ));
    }
    let entry = |path: &Path| -> Result<CaptureFile> {
        Ok(CaptureFile {
            path: path
                .strip_prefix(parent)
                .map_err(|_| Error::Invalid("capture file is outside manifest directory".into()))?
                .to_string_lossy()
                .replace('\\', "/"),
            sha256: sha256_file(path)?,
            size_bytes: path.metadata()?.len(),
        })
    };
    let manifest = CaptureManifest {
        schema_version: CAPTURE_MANIFEST_SCHEMA_VERSION.into(),
        environment_id: input.environment_id.into(),
        run_id: input.run_id.into(),
        video: entry(input.video_path)?,
        index: entry(input.index_path)?,
        codec: input.codec.into(),
        frame_rate: input.frame_rate,
        width: input.width,
        height: input.height,
        frames,
    };
    write_json(input.manifest_path, &manifest)?;
    Ok(manifest)
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SpatialEntity {
    pub environment_id: String,
    pub world_id: String,
    pub entity_id: String,
    pub kind: String,
    pub label: String,
    pub position: [f64; 3],
    pub coordinate_frame: String,
    pub authority: Authority,
    pub confidence: f64,
    pub observed_at_ns: u64,
    pub source_run_id: String,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RouteWaypoint {
    pub index: u32,
    pub position: [f64; 3],
    pub tolerance: f64,
    pub label: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SpatialRoute {
    pub environment_id: String,
    pub world_id: String,
    pub route_id: String,
    pub name: String,
    pub from_entity_id: Option<String>,
    pub to_entity_id: Option<String>,
    pub coordinate_frame: String,
    pub confidence: f64,
    pub verified_at_ns: u64,
    pub source_run_id: String,
    pub waypoints: Vec<RouteWaypoint>,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SpatialKnowledgeBundle {
    pub schema_version: String,
    pub environment_id: String,
    pub protocol_version: String,
    pub exported_at_ns: u64,
    pub entities: Vec<SpatialEntity>,
    pub routes: Vec<SpatialRoute>,
}

impl SpatialKnowledgeBundle {
    pub fn validate(&self) -> Result<()> {
        validate_schema(
            &self.schema_version,
            SPATIAL_KNOWLEDGE_SCHEMA_VERSION,
            "spatial knowledge",
        )?;
        if self.protocol_version.is_empty() {
            return Err(Error::Invalid(
                "spatial knowledge protocol_version cannot be empty".into(),
            ));
        }
        if self.entities.len() > 100_000 || self.routes.len() > 100_000 {
            return Err(Error::Invalid(
                "spatial knowledge exceeds the bounded object count".into(),
            ));
        }
        if self
            .entities
            .iter()
            .any(|entity| entity.environment_id != self.environment_id)
            || self
                .routes
                .iter()
                .any(|route| route.environment_id != self.environment_id)
        {
            return Err(Error::Invalid(
                "spatial object environment_id differs from its bundle".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum TraversabilityStatus {
    Unknown,
    Traversable,
    Blocked,
    Stale,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "kebab-case")]
pub enum SpatialHazard {
    DynamicHazard,
    InsufficientClearance,
    NoNavProjection,
    SteepSlope,
    GeometryBlocked,
    TransientFailure,
    UnknownBlocker,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SpatialFrameTransform {
    pub from_frame: String,
    pub to_frame: String,
    pub translation: [f64; 3],
    pub rotation_quaternion: [f64; 4],
    pub scale: [f64; 3],
}

impl SpatialFrameTransform {
    fn validate(&self) -> Result<()> {
        validate_identifier(&self.from_frame, "transform.from_frame")?;
        validate_identifier(&self.to_frame, "transform.to_frame")?;
        validate_position(&self.translation, "transform.translation")?;
        if self
            .rotation_quaternion
            .iter()
            .all(|value| value.abs() <= f64::EPSILON)
        {
            return Err(Error::Invalid(
                "transform.rotation_quaternion cannot be zero".into(),
            ));
        }
        if self
            .rotation_quaternion
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(Error::Invalid(
                "transform.rotation_quaternion must be finite".into(),
            ));
        }
        if self
            .scale
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(Error::Invalid(
                "transform.scale must contain positive finite values".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SpatialGraphNode {
    pub node_id: String,
    pub world_id: String,
    pub position: [f64; 3],
    pub coordinate_frame: String,
    pub ground_z: Option<f64>,
    pub nav_z: Option<f64>,
    pub observed_at_ns: u64,
    pub source_run_id: String,
    pub authority: Authority,
    pub confidence: f64,
    pub metadata: serde_json::Value,
}

impl SpatialGraphNode {
    fn validate(&self) -> Result<()> {
        for (label, value) in [
            ("node.node_id", &self.node_id),
            ("node.world_id", &self.world_id),
            ("node.coordinate_frame", &self.coordinate_frame),
            ("node.source_run_id", &self.source_run_id),
        ] {
            validate_identifier(value, label)?;
        }
        validate_position(&self.position, "node.position")?;
        for (label, value) in [("node.ground_z", self.ground_z), ("node.nav_z", self.nav_z)] {
            if value.is_some_and(|item| !item.is_finite()) {
                return Err(Error::Invalid(format!("{label} must be finite")));
            }
        }
        validate_confidence(self.confidence, "node.confidence")?;
        if !self.metadata.is_object() {
            return Err(Error::Invalid("node.metadata must be an object".into()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NegativeTraversalEvidence {
    pub reason: SpatialHazard,
    pub observed_at_ns: u64,
    pub source_run_id: String,
    pub expires_at_ns: Option<u64>,
    pub detail: Option<String>,
}

impl NegativeTraversalEvidence {
    fn validate(&self) -> Result<()> {
        validate_identifier(&self.source_run_id, "negative evidence.source_run_id")?;
        if self
            .expires_at_ns
            .is_some_and(|expires| expires < self.observed_at_ns)
        {
            return Err(Error::Invalid(
                "negative evidence.expires_at_ns cannot precede observed_at_ns".into(),
            ));
        }
        if self
            .detail
            .as_ref()
            .is_some_and(|detail| detail.trim().is_empty() || detail.chars().count() > 512)
        {
            return Err(Error::Invalid(
                "negative evidence.detail must be bounded text".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SpatialGraphEdge {
    pub edge_id: String,
    pub world_id: String,
    pub from_node_id: String,
    pub to_node_id: String,
    pub coordinate_frame: String,
    pub source_run_id: String,
    pub passability: TraversabilityStatus,
    pub cost: Option<f64>,
    pub success_count: u64,
    pub failure_count: u64,
    pub last_verified_at_ns: u64,
    pub expires_at_ns: Option<u64>,
    pub ground_projection: Option<[f64; 3]>,
    pub nav_projection: Option<[f64; 3]>,
    pub vertical_delta: Option<f64>,
    pub slope: Option<f64>,
    pub clearance: Option<f64>,
    pub hazard_reasons: Vec<SpatialHazard>,
    pub negative_evidence: Vec<NegativeTraversalEvidence>,
    pub transform: Option<SpatialFrameTransform>,
    pub authority: Authority,
    pub confidence: f64,
    pub metadata: serde_json::Value,
}

impl SpatialGraphEdge {
    fn validate(&self) -> Result<()> {
        for (label, value) in [
            ("edge.edge_id", &self.edge_id),
            ("edge.world_id", &self.world_id),
            ("edge.from_node_id", &self.from_node_id),
            ("edge.to_node_id", &self.to_node_id),
            ("edge.coordinate_frame", &self.coordinate_frame),
            ("edge.source_run_id", &self.source_run_id),
        ] {
            validate_identifier(value, label)?;
        }
        if self.from_node_id == self.to_node_id {
            return Err(Error::Invalid(
                "spatial edge cannot connect a node to itself".into(),
            ));
        }
        if self
            .cost
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(Error::Invalid(
                "edge.cost must be non-negative and finite".into(),
            ));
        }
        if self
            .expires_at_ns
            .is_some_and(|expires| expires < self.last_verified_at_ns)
        {
            return Err(Error::Invalid(
                "edge.expires_at_ns cannot precede last_verified_at_ns".into(),
            ));
        }
        for (label, value) in [
            ("edge.ground_projection", self.ground_projection),
            ("edge.nav_projection", self.nav_projection),
        ] {
            if let Some(position) = value {
                validate_position(&position, label)?;
            }
        }
        if self.vertical_delta.is_some_and(|value| !value.is_finite())
            || self
                .slope
                .is_some_and(|value| !value.is_finite() || !(0.0..=90.0).contains(&value))
            || self
                .clearance
                .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(Error::Invalid(
                "edge geometry fields are outside bounded finite ranges".into(),
            ));
        }
        for evidence in &self.negative_evidence {
            evidence.validate()?;
        }
        if let Some(transform) = &self.transform {
            transform.validate()?;
            if transform.from_frame != self.coordinate_frame {
                return Err(Error::Invalid(
                    "edge transform does not match coordinate_frame".into(),
                ));
            }
        }
        validate_confidence(self.confidence, "edge.confidence")?;
        if !self.metadata.is_object() {
            return Err(Error::Invalid("edge.metadata must be an object".into()));
        }
        Ok(())
    }

    pub fn status_at(&self, observed_at_ns: u64) -> TraversabilityStatus {
        if self
            .expires_at_ns
            .is_some_and(|expires| observed_at_ns >= expires)
        {
            return TraversabilityStatus::Stale;
        }
        if self.negative_evidence.iter().any(|evidence| {
            evidence
                .expires_at_ns
                .is_none_or(|expires| observed_at_ns < expires)
        }) {
            return TraversabilityStatus::Blocked;
        }
        if self.passability == TraversabilityStatus::Stale {
            TraversabilityStatus::Stale
        } else {
            self.passability
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SpatialKnowledgeGraph {
    pub schema_version: String,
    pub environment_id: String,
    pub protocol_version: String,
    pub exported_at_ns: u64,
    pub nodes: Vec<SpatialGraphNode>,
    pub edges: Vec<SpatialGraphEdge>,
    pub transforms: Vec<SpatialFrameTransform>,
}

impl SpatialKnowledgeGraph {
    pub fn validate(&self) -> Result<()> {
        validate_schema(
            &self.schema_version,
            SPATIAL_KNOWLEDGE_V2_SCHEMA_VERSION,
            "spatial graph",
        )?;
        validate_identifier(&self.environment_id, "spatial graph.environment_id")?;
        if self.protocol_version.is_empty() {
            return Err(Error::Invalid(
                "spatial graph protocol_version cannot be empty".into(),
            ));
        }
        if self.nodes.len() > 100_000 || self.edges.len() > 100_000 {
            return Err(Error::Invalid(
                "spatial graph exceeds the bounded object count".into(),
            ));
        }
        for node in &self.nodes {
            node.validate()?;
        }
        for transform in &self.transforms {
            transform.validate()?;
        }
        let node_ids = self
            .nodes
            .iter()
            .map(|node| node.node_id.as_str())
            .collect::<HashSet<_>>();
        if node_ids.len() != self.nodes.len() {
            return Err(Error::Invalid(
                "spatial graph.nodes contains duplicate node_id values".into(),
            ));
        }
        let edge_ids = self
            .edges
            .iter()
            .map(|edge| edge.edge_id.as_str())
            .collect::<HashSet<_>>();
        if edge_ids.len() != self.edges.len() {
            return Err(Error::Invalid(
                "spatial graph.edges contains duplicate edge_id values".into(),
            ));
        }
        let node_by_id = self
            .nodes
            .iter()
            .map(|node| (node.node_id.as_str(), node))
            .collect::<std::collections::HashMap<_, _>>();
        for edge in &self.edges {
            edge.validate()?;
            let from = node_by_id.get(edge.from_node_id.as_str()).ok_or_else(|| {
                Error::Invalid("spatial graph edge references an unknown node".into())
            })?;
            let to = node_by_id.get(edge.to_node_id.as_str()).ok_or_else(|| {
                Error::Invalid("spatial graph edge references an unknown node".into())
            })?;
            if from.world_id != edge.world_id || to.world_id != edge.world_id {
                return Err(Error::Invalid(
                    "spatial graph edge crosses world boundaries".into(),
                ));
            }
        }
        let transform_pairs = self
            .transforms
            .iter()
            .map(|transform| (&transform.from_frame, &transform.to_frame))
            .collect::<HashSet<_>>();
        if transform_pairs.len() != self.transforms.len() {
            return Err(Error::Invalid(
                "spatial graph.transforms contains duplicate frame pairs".into(),
            ));
        }
        Ok(())
    }
}

fn validate_position(value: &[f64; 3], label: &str) -> Result<()> {
    if value.iter().any(|item| !item.is_finite()) {
        Err(Error::Invalid(format!(
            "{label} coordinates must be finite"
        )))
    } else {
        Ok(())
    }
}

fn validate_portable_path(value: &str, label: &str) -> Result<PathBuf> {
    let path = PathBuf::from(value);
    if value.is_empty()
        || value.contains('\\')
        || value.contains(':')
        || path.is_absolute()
        || path
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        Err(Error::Invalid(format!(
            "{label} must be a portable relative path"
        )))
    } else {
        Ok(path)
    }
}
