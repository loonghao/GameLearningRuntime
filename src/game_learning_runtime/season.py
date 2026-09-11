"""Declarative season selection and portable, byte-bound role context."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

if TYPE_CHECKING:
    from game_learning_runtime.project import GLRProject

CONTEXT_SCHEMA = "glr.season-context.v1"
ENVIRONMENT_KEYS = (
    "GLR_SEASON_ID",
    "GLR_RULESET_ID",
    "GLR_SEASON_CONFIG_SHA256",
    "GLR_SEASON_CONTEXT_SHA256",
    "GLR_SEASON_CONTEXT",
)
_LIMIT = 1024 * 1024
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


def _identifier(value: object) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError("season identifiers must be portable lowercase identifiers (1..64)")
    return value


def _strict(value: object, required: set[str], optional: set[str] | None = None) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) - required - (optional or set())
        or required - set(value)
    ):
        raise ValueError("season configuration has missing or unknown fields")
    return value


def season_path(root: Path, relative: object) -> Path:
    """Validate every path component, including links and Windows reparse points."""
    if (
        not isinstance(relative, str)
        or len(relative.encode("utf-8")) > 512
        or any(c in relative for c in "\\:\x00")
    ):
        raise ValueError("season config must be a portable root-relative path")
    if any(not part or part in {".", ".."} for part in relative.split("/")):
        raise ValueError("season config must be a portable root-relative path")
    candidate = root.resolve()
    for part in relative.split("/"):
        if any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in part):
            raise ValueError("season config path contains control characters")
        candidate /= part
        if candidate.is_symlink() or (
            candidate.exists() and getattr(candidate.lstat(), "st_file_attributes", 0) & 0x400
        ):
            raise ValueError("season config cannot contain links or reparse points")
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError("season config escapes project root")
    return candidate


def _canonical(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key in frozen season JSON")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class SeasonInput:
    path: str
    sha256: str
    size_bytes: int

    def to_mapping(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "size_bytes": self.size_bytes}


def _read(root: Path, relative: str) -> tuple[SeasonInput, bytes]:
    path = season_path(root, relative)
    if not path.is_file():
        raise ValueError("season config must be an existing regular file")
    with path.open("rb") as stream:
        raw = stream.read(_LIMIT + 1)
    if len(raw) > _LIMIT:
        raise ValueError("season input exceeds the 1 MiB limit")
    return SeasonInput(relative, hashlib.sha256(raw).hexdigest(), len(raw)), raw


def _toml(root: Path, relative: str) -> tuple[SeasonInput, dict[str, Any]]:
    if not relative.endswith(".toml"):
        raise ValueError("season declarations and catalogs must be TOML")
    reference, raw = _read(root, relative)
    return reference, tomllib.loads(raw.decode("utf-8"))


@dataclass(frozen=True, slots=True)
class SeasonContext:
    season_id: str
    ruleset_id: str
    environment_id: str
    protocol_version: str
    status: str
    project: SeasonInput
    catalog: SeasonInput
    declaration: SeasonInput
    extensions: Mapping[str, SeasonInput]

    def to_mapping(self) -> dict[str, Any]:
        value = {
            "schema_version": CONTEXT_SCHEMA,
            "season_id": self.season_id,
            "ruleset_id": self.ruleset_id,
            "environment_id": self.environment_id,
            "protocol_version": self.protocol_version,
            "status": self.status,
            "project": self.project.to_mapping(),
            "catalog": self.catalog.to_mapping(),
            "declaration": self.declaration.to_mapping(),
            "extensions": {key: item.to_mapping() for key, item in self.extensions.items()},
        }
        value["context_sha256"] = hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()
        return value

    def to_json(self) -> str:
        result = _canonical(self.to_mapping())
        if len(result.encode("utf-8")) > 24 * 1024:
            raise ValueError("season context exceeds the 24 KiB role-environment limit")
        return result

    def require_ready(self) -> None:
        if self.status != "ready":
            raise ValueError(
                "selected season is pending; train/goal/play require ready configuration"
            )

    def verify(self, root: Path) -> None:
        for reference in (self.project, self.catalog, self.declaration, *self.extensions.values()):
            if _read(root, reference.path)[0] != reference:
                raise ValueError("frozen season input changed before role execution")

    def environment(self, root: Path) -> dict[str, str]:
        self.verify(root)
        return dict(
            zip(
                ENVIRONMENT_KEYS,
                (
                    self.season_id,
                    self.ruleset_id,
                    self.declaration.sha256,
                    self.to_mapping()["context_sha256"],
                    self.to_json(),
                ),
                strict=True,
            )
        )


def _catalog(project: GLRProject) -> tuple[SeasonInput, list[dict[str, Any]]]:
    if project.seasons is None:
        raise ValueError("project has no [seasons] config reference")
    reference, value = _toml(project.root, project.seasons)
    _strict(value, {"schema_version", "entries"})
    if value["schema_version"] != "glr.seasons.v1" or not isinstance(value["entries"], list):
        raise ValueError("unsupported season catalog schema")
    entries = value["entries"]
    if len(entries) > 256:
        raise ValueError("season catalog exceeds 256 entries")
    pairs: set[tuple[str, str]] = set()
    for entry in entries:
        _strict(entry, {"season_id", "ruleset_id", "config"})
        pair = (_identifier(entry["season_id"]), _identifier(entry["ruleset_id"]))
        if pair in pairs:
            raise ValueError("duplicate season/ruleset selection")
        pairs.add(pair)
        season_path(project.root, entry["config"])
    return reference, entries


def select_season(
    project: GLRProject, season: str | None, ruleset: str | None
) -> SeasonContext | None:
    if (season is None) != (ruleset is None):
        raise ValueError("--season and --ruleset must be supplied together")
    if season is None:
        return None
    _identifier(season)
    _identifier(ruleset)
    catalog, entries = _catalog(project)
    entry = next(
        (item for item in entries if (item["season_id"], item["ruleset_id"]) == (season, ruleset)),
        None,
    )
    if entry is None:
        raise ValueError("unknown season/ruleset selection")
    declaration, value = _toml(project.root, entry["config"])
    _strict(
        value,
        {
            "schema_version",
            "season_id",
            "ruleset_id",
            "environment_id",
            "protocol_version",
            "status",
        },
        {"extensions"},
    )
    if (
        value["schema_version"],
        value["season_id"],
        value["ruleset_id"],
        value["environment_id"],
        value["protocol_version"],
    ) != (
        "glr.season.v1",
        season,
        ruleset,
        project.environment_id,
        project.protocol_version,
    ):
        raise ValueError("season declaration identity does not match project and selection")
    if value["status"] not in {"pending", "ready"}:
        raise ValueError("season status must be pending or ready")
    extensions = value.get("extensions", {})
    if not isinstance(extensions, dict) or len(extensions) > 32:
        raise ValueError("season extensions must be a table with at most 32 entries")
    references = {}
    for namespace, config in extensions.items():
        _identifier(namespace)
        _strict(config, {"config"})
        references[namespace] = _read(project.root, config["config"])[0]
    assert project.manifest_path is not None
    manifest = _read(project.root, project.manifest_path.relative_to(project.root).as_posix())[0]
    if manifest.sha256 != project.manifest_sha256:
        raise ValueError("project manifest changed while selecting season")
    assert ruleset is not None
    return SeasonContext(
        season,
        ruleset,
        project.environment_id,
        project.protocol_version,
        value["status"],
        manifest,
        catalog,
        declaration,
        MappingProxyType(references),
    )


def list_seasons(project: GLRProject) -> list[dict[str, Any]]:
    if project.seasons is None:
        return []
    return [
        select_season(project, item["season_id"], item["ruleset_id"]).to_mapping()  # type: ignore[union-attr]
        for item in _catalog(project)[1]
    ]


def require_selection(project: GLRProject, *, ready: bool) -> None:
    if project.seasons is not None and project.season_context is None:
        raise ValueError("project requires explicit --season and --ruleset before role execution")
    if project.season_context is not None:
        project.season_context.verify(project.root)
        if ready:
            project.season_context.require_ready()


def load_season_context(
    project: GLRProject, environment: Mapping[str, str] | None = None
) -> SeasonContext | None:
    """Revalidate the CLI's frozen bytes, not a project-local active-profile fallback."""
    source = os.environ if environment is None else environment
    present = [key for key in ENVIRONMENT_KEYS if key in source]
    if not present:
        return None
    if len(present) != len(ENVIRONMENT_KEYS) or len(source["GLR_SEASON_CONTEXT"]) > 64 * 1024:
        raise ValueError("incomplete or oversized frozen season environment")
    context = select_season(project, source["GLR_SEASON_ID"], source["GLR_RULESET_ID"])
    assert context is not None
    expected = context.environment(project.root)
    if _canonical(
        json.loads(source["GLR_SEASON_CONTEXT"], object_pairs_hook=_unique_object)
    ) != context.to_json() or any(
        source[key] != expected[key] for key in ENVIRONMENT_KEYS if key != "GLR_SEASON_CONTEXT"
    ):
        raise ValueError("CLI frozen season context does not match current inputs")
    return context


def initialize_season(project: GLRProject, season: str, ruleset: str) -> SeasonContext:
    """Create only pending TOML and register it; never overwrite or execute a hook."""
    season, ruleset = _identifier(season), _identifier(ruleset)
    if project.seasons is None:
        raise ValueError('first declare [seasons] config = "config/seasons.toml" in the project')
    catalog_path = season_path(project.root, project.seasons)
    if not project.seasons.endswith(".toml"):
        raise ValueError("season catalog must be TOML")
    original = _read(project.root, project.seasons)[1] if catalog_path.exists() else None
    entries = _catalog(project)[1] if original is not None else []
    if any((item["season_id"], item["ruleset_id"]) == (season, ruleset) for item in entries):
        raise FileExistsError("season/ruleset already registered")
    if len(entries) >= 256:
        raise ValueError("season catalog exceeds 256 entries")
    relative = f"config/seasons/{season}/{ruleset}.toml"
    if relative == project.seasons:
        raise ValueError("catalog and declaration paths must differ")
    declaration = season_path(project.root, relative)
    if declaration == catalog_path:
        raise ValueError("catalog and declaration paths must differ")
    declaration.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    lock = catalog_path.with_name(catalog_path.name + ".lock")
    lock_stream = lock.open("xb")
    try:
        season_path(project.root, project.seasons)
        current = _read(project.root, project.seasons)[1] if catalog_path.exists() else None
        if current != original:
            raise ValueError("season catalog changed during initialization")
        # Serialize only the validated catalog schema, preserving existing entry order.
        catalog_text = 'schema_version = "glr.seasons.v1"\n'
        for item in [*entries, {"season_id": season, "ruleset_id": ruleset, "config": relative}]:
            catalog_text += "\n[[entries]]\n" + "".join(
                f"{key} = {json.dumps(value, ensure_ascii=False)}\n" for key, value in item.items()
            )
        declaration_text = "".join(
            f"{key} = {json.dumps(value, ensure_ascii=False)}\n"
            for key, value in {
                "schema_version": "glr.season.v1",
                "season_id": season,
                "ruleset_id": ruleset,
                "environment_id": project.environment_id,
                "protocol_version": project.protocol_version,
                "status": "pending",
            }.items()
        )
        season_path(project.root, relative)
        temporary: Path | None = None
        declaration_stream = declaration.open("xb")
        try:
            with declaration_stream as stream:
                stream.write(declaration_text.encode("utf-8"))
            with tempfile.NamedTemporaryFile(dir=catalog_path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(catalog_text.encode("utf-8"))
            os.replace(temporary, catalog_path)
        except BaseException:
            declaration.unlink()
            raise
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    finally:
        lock_stream.close()
        lock.unlink()
    result = select_season(project, season, ruleset)
    assert result is not None
    return result
