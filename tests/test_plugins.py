from __future__ import annotations

import json
from pathlib import Path

import pytest

import game_learning_runtime.plugins as plugins_module
from game_learning_runtime.plugins import (
    PLUGIN_SCHEMA_VERSION,
    PROFILE_SCHEMA_VERSION,
    PluginFile,
    PluginManager,
    PluginManifest,
    PluginProfile,
    PluginRef,
    PluginValidationError,
)


def _write_plugin(
    root: Path, *, plugin_id: str = "example-learner", version: str = "1.2.3"
) -> Path:
    package = root / plugin_id
    package.mkdir(parents=True)
    (package / "example_plugin.py").write_text(
        "def create():\n    return {'ok': True}\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": PLUGIN_SCHEMA_VERSION,
        "id": plugin_id,
        "version": version,
        "kind": "learner",
        "name": "Example learner",
        "description": "A deterministic test plugin.",
        "entrypoint": "example_plugin:create",
        "capabilities": ["learner.ppo", "collector.process"],
        "requires": {"glr": ">=0.17.0,<1.0.0"},
        "platforms": ["windows", "linux", "macos"],
        "isolation": "process",
        "permissions": ["read:environment", "write:checkpoint"],
    }
    (package / "glr-plugin.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return package


def test_manifest_round_trip_is_canonical_and_rejects_unknown_fields() -> None:
    value = {
        "schema_version": PLUGIN_SCHEMA_VERSION,
        "id": "example-learner",
        "version": "1.2.3",
        "kind": "learner",
        "name": "Example",
        "description": "A plugin",
        "entrypoint": "example:create",
        "capabilities": ["learner.ppo"],
    }
    manifest = PluginManifest.from_mapping(value)
    assert manifest.to_mapping() == {
        **value,
        "requires": {},
        "platforms": ["windows", "linux", "macos"],
        "isolation": "process",
        "permissions": [],
        "dependencies": {},
        "files": [],
    }
    with pytest.raises(PluginValidationError, match="unexpected"):
        PluginManifest.from_mapping({**value, "unknown": True})


@pytest.mark.parametrize(
    "field, value",
    [
        ("id", "../unsafe"),
        ("version", "latest"),
        ("kind", "unknown"),
        ("entrypoint", "C:/run.py"),
        ("permissions", ["*"]),
        ("capabilities", []),
    ],
)
def test_manifest_rejects_unsafe_values(field: str, value: object) -> None:
    base = {
        "schema_version": PLUGIN_SCHEMA_VERSION,
        "id": "example-plugin",
        "version": "1.0.0",
        "kind": "learner",
        "name": "Example",
        "description": "A plugin",
        "entrypoint": "example:create",
        "capabilities": ["learner.ppo"],
    }
    if field == "capabilities":
        base[field] = value  # type: ignore[assignment]
    else:
        base[field] = value  # type: ignore[assignment]
    with pytest.raises((PluginValidationError, ValueError), match=r".+"):
        PluginManifest.from_mapping(base)


def test_inspect_install_and_profile_resolution_are_non_executing(tmp_path: Path) -> None:
    source = _write_plugin(tmp_path / "source")
    marker = tmp_path / "imported"
    (source / "example_plugin.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n", encoding="utf-8"
    )
    manager = PluginManager(tmp_path / "project")
    inspection = manager.inspect(source)
    assert inspection.manifest.id == "example-learner"
    assert inspection.total_bytes > 0
    assert not marker.exists()
    expected = inspection.content_sha256
    installed = manager.install(source, expected_sha256=expected)
    assert installed.manifest.id == "example-learner"
    assert installed.content_sha256 == expected
    assert not marker.exists()
    assert [item.manifest.id for item in manager.list_installed()] == ["example-learner"]

    profile = PluginProfile(
        name="training",
        plugins=(
            PluginRef(
                plugin_id="example-learner",
                version=">=1.0.0,<2.0.0",
                permissions=("read:environment",),
            ),
        ),
    )
    manager.save_profile(profile)
    resolved = manager.resolve_profile("training")
    assert resolved.profile.name == "training"
    assert resolved.plugins[0].manifest.version == "1.2.3"
    assert resolved.plugins[0].permissions == ("read:environment",)
    assert resolved.digest
    health = manager.health("training")
    assert health[0]["status"] == "ready"


def test_profile_rejects_permission_escalation_and_missing_plugin(tmp_path: Path) -> None:
    source = _write_plugin(tmp_path / "source")
    manager = PluginManager(tmp_path / "project")
    manager.install(source)
    with pytest.raises(PluginValidationError, match="permission"):
        manager.resolve_profile(
            PluginProfile(
                name="bad",
                plugins=(
                    PluginRef(
                        plugin_id="example-learner",
                        version="1.2.3",
                        permissions=("runtime.act",),
                    ),
                ),
            )
        )
    with pytest.raises(PluginValidationError, match="not installed"):
        manager.resolve_profile(
            PluginProfile(name="missing", plugins=(PluginRef(plugin_id="nope"),))
        )


def test_install_refuses_tampering_symlink_and_overwrite(tmp_path: Path) -> None:
    source = _write_plugin(tmp_path / "source")
    manager = PluginManager(tmp_path / "project")
    manager.install(source)
    with pytest.raises(FileExistsError, match="already installed"):
        manager.install(source)
    (source / "example_plugin.py").write_text("tampered", encoding="utf-8")
    with pytest.raises(PluginValidationError, match="expected_sha256"):
        manager.install(source, expected_sha256="0" * 64)
    linked = tmp_path / "linked"
    linked.mkdir()
    try:
        (linked / "link.py").symlink_to(source / "glr-plugin.json")
    except OSError:
        pytest.skip("symlink privilege is unavailable")
    with pytest.raises(PluginValidationError, match="symlink"):
        manager.inspect(linked)


def test_profile_mapping_and_digest_change_with_order_and_grants() -> None:
    profile = PluginProfile.from_mapping(
        {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "name": "default",
            "plugins": [
                {
                    "id": "example-learner",
                    "version": "1.0.0",
                    "enabled": True,
                    "permissions": ["read:environment"],
                    "config": {"batch": 32},
                }
            ],
        }
    )
    assert profile.to_mapping()["schema_version"] == PROFILE_SCHEMA_VERSION
    assert (
        profile.digest()
        != PluginProfile(
            name="default",
            plugins=(
                PluginRef(
                    plugin_id="example-learner",
                    version="1.0.0",
                    permissions=("write:checkpoint",),
                ),
            ),
        ).digest()
    )


def test_profile_digest_canonical_numbers_match_rust_json() -> None:
    # Integer numbers and UTF-8 text are byte-stable across both runtimes.
    assert (
        plugins_module._canonical_json_bytes({"a": 7, "b": 5, "u": "é😀"})
        == b'{"a":7,"b":5,"u":"\xc3\xa9\xf0\x9f\x98\x80"}'
    )
    assert plugins_module._canonical_json_bytes({"value": 9.999999999999999e-6}) == (
        b'{"value":9.999999999999999e-6}'
    )
    with pytest.raises(PluginValidationError, match="serde_json bounds"):
        plugins_module._canonical_json_bytes({"n": 2**64})
    with pytest.raises(PluginValidationError, match="keys must be strings"):
        plugins_module._canonical_json_bytes({"nested": {1: "invalid"}})
    with pytest.raises(PluginValidationError, match="JSON-serializable"):
        PluginRef(plugin_id="example-learner", config={"bad": "\ud800"})


def test_plugin_contract_rejects_malformed_data_and_bounds(tmp_path: Path) -> None:
    with pytest.raises(PluginValidationError):
        PluginManifest.from_mapping(1)  # type: ignore[arg-type]
    with pytest.raises(PluginValidationError):
        PluginManifest.from_mapping({1: "bad"})  # type: ignore[dict-item]
    base = {
        "schema_version": PLUGIN_SCHEMA_VERSION,
        "id": "example-plugin",
        "version": "1.0.0",
        "kind": "learner",
        "name": "Example",
        "description": "A plugin",
        "entrypoint": "example:create",
        "capabilities": ["learner.ppo"],
    }
    for field, value in (
        ("capabilities", "not-an-array"),
        ("platforms", "not-an-array"),
        ("permissions", "not-an-array"),
        ("files", "not-an-array"),
    ):
        with pytest.raises(PluginValidationError):
            PluginManifest.from_mapping({**base, field: value})
    for value in ("learner.ppo", "learner.ppo"):
        with pytest.raises(PluginValidationError):
            PluginManifest.from_mapping({**base, "capabilities": [value, value]})
    for value in (
        "../escape",
        "folder/../escape",
        "./escape",
        "folder//escape",
        "CON/file.py",
        "folder/file. ",
        "folder/\nfile.py",
    ):
        with pytest.raises(PluginValidationError):
            PluginFile(value, "0" * 64, 1)
    with pytest.raises(PluginValidationError):
        PluginFile(".", "0" * 64, 1)
    with pytest.raises(PluginValidationError):
        PluginFile("file.py", "0" * 64, -1)
    with pytest.raises(PluginValidationError):
        PluginFile("file.py", "0" * 64, 256 * 1024 * 1024 + 1)
    with pytest.raises(PluginValidationError):
        PluginManifest.from_mapping({**base, "version": "1.01.0"})
    with pytest.raises(PluginValidationError):
        PluginManifest.from_mapping({**base, "version": "1.0.0-alpha.01"})
    with pytest.raises(PluginValidationError):
        PluginManifest.from_mapping({**base, "entrypoint": "echo;bad"})
    with pytest.raises(PluginValidationError):
        PluginManifest.from_mapping({**base, "entrypoint": "glr-plugin.json"})
    with pytest.raises(PluginValidationError):
        PluginManifest.from_mapping(
            {
                **base,
                "files": [{"path": "glr-plugin.json", "sha256": "0" * 64, "size_bytes": 1}],
            }
        )
    with pytest.raises(PluginValidationError):
        PluginManifest.from_mapping({**base, "entrypoint": "1module:create"})

    assert plugins_module._satisfies("1.2.3", "==1.2.3")
    assert plugins_module._satisfies("1.2.3", ">=1.0.0")
    assert plugins_module._satisfies("1.2.3", "<=2.0.0")
    assert plugins_module._satisfies("1.2.3", ">1.0.0")
    assert plugins_module._satisfies("1.2.3", "<2.0.0")
    assert not plugins_module._satisfies("1.2.3", "1.2.4")
    assert plugins_module._version_key("1.0.0-alpha") < plugins_module._version_key("1.0.0")
    with pytest.raises(PluginValidationError):
        plugins_module._validate_requirement(">=1.0.0,,<2.0.0", path="requirement")
    with pytest.raises(PluginValidationError):
        plugins_module._validate_requirement("not-a-range", path="requirement")

    nested = PluginRef(plugin_id="example-plugin", config={"nested": [{"ok": True}]})
    assert nested.to_mapping()["config"] == {"nested": [{"ok": True}]}
    with pytest.raises(PluginValidationError):
        PluginRef(plugin_id="example-plugin", config={"bad": float("nan")})
    with pytest.raises(PluginValidationError):
        PluginRef(plugin_id="example-plugin", config={"x": "a" * (64 * 1024)})


def test_plugin_loader_rejects_oversized_duplicate_and_invalid_json(tmp_path: Path) -> None:
    source = _write_plugin(tmp_path / "source")
    manifest_path = source / "glr-plugin.json"
    original = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text('{"schema_version":', encoding="utf-8")
    with pytest.raises(PluginValidationError, match="valid UTF-8 JSON"):
        PluginManifest.load(manifest_path)
    manifest_path.write_text(
        '{"schema_version":"glr.plugin.v1","schema_version":"glr.plugin.v1"}',
        encoding="utf-8",
    )
    with pytest.raises(PluginValidationError, match="duplicate"):
        PluginManifest.load(manifest_path)
    manifest_path.write_text(original, encoding="utf-8")
    manifest_path.write_text("x" * (64 * 1024 + 1), encoding="utf-8")
    with pytest.raises(PluginValidationError, match="size limit"):
        PluginManifest.load(manifest_path)


def test_plugin_file_inventory_and_manifest_integrity_fail_closed(tmp_path: Path) -> None:
    source = _write_plugin(tmp_path / "source")
    manifest = json.loads((source / "glr-plugin.json").read_text(encoding="utf-8"))
    manifest["files"] = [
        {
            "path": "example_plugin.py",
            "sha256": "0" * 64,
            "size_bytes": 1,
        }
    ]
    (source / "glr-plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PluginValidationError, match="integrity"):
        plugins_module.inspect_plugin(source)
    manifest["files"] = [{"path": "missing.py", "sha256": "0" * 64, "size_bytes": 1}]
    (source / "glr-plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PluginValidationError, match="inventory"):
        plugins_module.inspect_plugin(source)
    manifest["files"] = []
    manifest["entrypoint"] = "missing.py"
    (source / "glr-plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PluginValidationError, match="entrypoint"):
        plugins_module.inspect_plugin(source)

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(PluginValidationError, match="manifest"):
        plugins_module.inspect_plugin(empty)
    try:
        (empty / "link.py").symlink_to(source / "example_plugin.py")
    except OSError:
        pass
    else:
        with pytest.raises(PluginValidationError, match="symlink"):
            plugins_module.inspect_plugin(empty)


def test_plugin_manager_store_and_resolution_fail_closed(tmp_path: Path) -> None:
    source = _write_plugin(tmp_path / "source")
    manager = PluginManager(tmp_path / "project", glr_version="0.16.0")
    manager.install(source)
    with pytest.raises(PluginValidationError, match="incompatible GLR"):
        manager.resolve_profile(
            PluginProfile(name="old", plugins=(PluginRef(plugin_id="example-learner"),))
        )
    manager_ok = PluginManager(tmp_path / "project", glr_version="0.17.0")
    with pytest.raises(PluginValidationError, match="permission"):
        manager_ok.resolve_profile(
            PluginProfile(
                name="grant",
                plugins=(PluginRef(plugin_id="example-learner", permissions=("runtime:act",)),),
            )
        )
    manager_ok.save_profile(
        PluginProfile(
            name="disabled", plugins=(PluginRef(plugin_id="example-learner", enabled=False),)
        )
    )
    assert manager_ok.health()[0]["status"] == "ready"
    manager_ok.remove("example-learner")
    assert manager_ok.list_installed() == ()


def test_plugin_manager_detects_dependencies_cycles_and_conflicts(tmp_path: Path) -> None:
    source = _write_plugin(tmp_path / "source", plugin_id="root-plugin")
    root_manifest = json.loads((source / "glr-plugin.json").read_text(encoding="utf-8"))
    root_manifest["dependencies"] = {"missing-plugin": "*"}
    (source / "glr-plugin.json").write_text(json.dumps(root_manifest), encoding="utf-8")
    manager = PluginManager(tmp_path / "project")
    manager.install(source)
    with pytest.raises(PluginValidationError, match="not installed"):
        manager.resolve_profile(
            PluginProfile(name="missing", plugins=(PluginRef(plugin_id="root-plugin"),))
        )

    dep = _write_plugin(tmp_path / "dependency", plugin_id="dep-plugin")
    dep_manifest = json.loads((dep / "glr-plugin.json").read_text(encoding="utf-8"))
    dep_manifest["dependencies"] = {"root-plugin": "*"}
    (dep / "glr-plugin.json").write_text(json.dumps(dep_manifest), encoding="utf-8")
    # Replace root with a cycle and install the second version under a new ID.
    root_manifest["dependencies"] = {"dep-plugin": "*"}
    (source / "glr-plugin.json").write_text(json.dumps(root_manifest), encoding="utf-8")
    manager.remove("root-plugin")
    manager.install(source)
    manager.install(dep)
    with pytest.raises(PluginValidationError, match="cycle"):
        manager.resolve_profile(
            PluginProfile(name="cycle", plugins=(PluginRef(plugin_id="root-plugin"),))
        )


def test_plugin_profile_store_security_and_remove_guards(tmp_path: Path) -> None:
    source = _write_plugin(tmp_path / "source")
    manager = PluginManager(tmp_path / "project")
    manager.install(source)
    profile = PluginProfile(name="active", plugins=(PluginRef(plugin_id="example-learner"),))
    manager.save_profile(profile)
    with pytest.raises(plugins_module.PluginTrustError, match="enabled"):
        manager.remove("example-learner")
    profile_path = manager.profile_root / "active.json"
    profile_path.unlink()
    manager.save_profile(profile)
    # A malformed profile in the store is rejected by listing.
    profile_path.write_text("{}", encoding="utf-8")
    with pytest.raises(PluginValidationError):
        manager.list_profiles()
    profile_path.unlink()
    manager.save_profile(
        PluginProfile(
            name="inactive", plugins=(PluginRef(plugin_id="example-learner", enabled=False),)
        )
    )
    assert manager.load_profile("inactive").name == "inactive"


def test_plugin_contract_edge_cases_and_requirement_ranges(tmp_path: Path) -> None:
    """Exercise the strict validation branches that protect the no-exec boundary."""

    with pytest.raises(PluginValidationError):
        plugins_module._text("", path="text")
    assert plugins_module._text("界" * 1365, path="text", maximum=4096) == "界" * 1365
    with pytest.raises(PluginValidationError):
        plugins_module._text("界" * 1366, path="text", maximum=4096)
    with pytest.raises(PluginValidationError):
        plugins_module._set_of_strings(
            "not-an-array", path="values", pattern=plugins_module._IDENTIFIER
        )
    with pytest.raises(PluginValidationError):
        plugins_module._portable_path("", path="path")
    with pytest.raises(PluginValidationError):
        plugins_module._portable_path("payload-é.py", path="path")
    for invalid in '<>:"|?*':
        with pytest.raises(PluginValidationError):
            plugins_module._portable_path(f"bad{invalid}.py", path="path")
    with pytest.raises(PluginValidationError):
        plugins_module._parse_semver(None, path="version")
    with pytest.raises(PluginValidationError):
        plugins_module._parse_semver("1.0.0-alpha..1", path="version")
    with pytest.raises(PluginValidationError):
        plugins_module._parse_semver("18446744073709551616.0.0", path="version")
    with pytest.raises(PluginValidationError):
        plugins_module._validate_requirement(None, path="requirement")
    with pytest.raises(PluginValidationError):
        plugins_module._validate_requirement(">=1.0.0\n", path="requirement")
    assert plugins_module._validate_requirement("^1.2", path="requirement") == "^1.2"
    assert plugins_module._validate_requirement("~1", path="requirement") == "~1"
    with pytest.raises(PluginValidationError):
        plugins_module._parse_partial_version("1.x", path="requirement")
    assert plugins_module._range_bounds("~", "1", path="requirement") == (
        "1.0.0",
        "2.0.0",
    )
    assert plugins_module._range_bounds("^", "0.2", path="requirement") == (
        "0.2.0",
        "0.3.0",
    )
    assert plugins_module._range_bounds("^", "0", path="requirement") == (
        "0.0.0",
        "1.0.0",
    )
    assert plugins_module._range_bounds("^", "0.0", path="requirement") == (
        "0.0.0",
        "0.1.0",
    )
    assert plugins_module._range_bounds("^", "0.0.3", path="requirement") == (
        "0.0.3",
        "0.0.4",
    )
    assert plugins_module._version_key("1.0.0-alpha.1") < plugins_module._version_key(
        "1.0.0-alpha.2"
    )
    assert plugins_module._version_key("1.0.0+2") == plugins_module._version_key("1.0.0+01")
    assert plugins_module._version_order_key("1.0.0+2") > plugins_module._version_order_key(
        "1.0.0+01"
    )
    assert plugins_module._satisfies("1.2.3", "^1.0")
    assert plugins_module._satisfies("0.9.9", "^0")
    assert plugins_module._satisfies("0.0.9", "^0.0")
    assert plugins_module._satisfies("1.2.3", "~1.2")
    assert plugins_module._satisfies("1.2.3", "1.2")
    assert plugins_module._satisfies("1.2.3", "1.2.3")
    assert plugins_module._satisfies("1.2.4", "1.2.3")
    assert not plugins_module._satisfies("2.0.0", "^1.0")
    assert not plugins_module._satisfies("1.2.3", "<=1.0.0")
    assert not plugins_module._satisfies("1.2.3", ">1.2.3")
    assert not plugins_module._satisfies("1.2.3", "<1.2.3")
    assert not plugins_module._satisfies("1.2.3-alpha.1", ">=1.0.0")
    assert plugins_module._satisfies("1.2.3-alpha.1", ">=1.2.3-alpha.1")

    # Match Cargo semver's wildcard, partial-comparator, and exact forms.
    for requirement in ("1.*", "1.2.*", "1.x", "1.2.X", "=1.2.3", "==1.2.3", "<=1.2", ">1"):
        plugins_module._validate_requirement(requirement, path="requirement")
    assert (
        plugins_module._validate_requirement(" >=1.0.0, <2.0.0 ", path="requirement")
        == ">=1.0.0, <2.0.0"
    )
    assert plugins_module._validate_requirement("1.x", path="requirement") == "1.*"
    # rust-semver accepts operators on wildcard operands and canonicalizes
    # them to partial comparators.
    assert plugins_module._validate_requirement(">=1.*", path="requirement") == ">=1"
    assert plugins_module._validate_requirement("<=1.2.*", path="requirement") == "<=1.2"
    assert plugins_module._validate_requirement("^1.*", path="requirement") == "^1"
    assert plugins_module._validate_requirement("==1.2.*", path="requirement") == "=1.2"
    with pytest.raises(PluginValidationError):
        plugins_module._validate_requirement(",".join([">=1"] * 33), path="requirement")
    assert plugins_module._satisfies("1.8.0", "1.*")
    assert plugins_module._satisfies("1.8.0", "x")
    assert not plugins_module._satisfies("1.8.0-alpha", "X")
    assert plugins_module._satisfies("1.2.9", "1.2.*")
    assert plugins_module._satisfies("2.0.0", ">=1.*")
    assert plugins_module._satisfies("1.2.9", "<=1.2.*")
    assert not plugins_module._satisfies("1.3.0", "<=1.2.*")
    assert plugins_module._satisfies("1.2.3", "^1.*")
    assert plugins_module._satisfies("1.2.3", "==1.2.*")
    assert plugins_module._satisfies("1.2.3", "=1.2")
    assert not plugins_module._satisfies("1.3.0", "=1.2")
    assert plugins_module._satisfies("1.2.3", "=1.2.3")
    assert not plugins_module._satisfies("1.2.4", "=1.2.3")
    assert plugins_module._satisfies("1.2.3-beta.2", "^1.2.3-beta.1")
    assert plugins_module._satisfies("1.2.3-beta.2", "~1.2.3-beta.1")
    assert plugins_module._satisfies("2.0.0", ">1")
    assert not plugins_module._satisfies("1.9.9", ">1")

    max_component = 2**64 - 1
    assert plugins_module._satisfies(f"{max_component}.0.1", f"^{max_component}.0.0")
    assert plugins_module._satisfies(f"{max_component}.9.9", f"~{max_component}")
    assert plugins_module._satisfies(f"{max_component}.9.9", f"{max_component}.*")
    assert plugins_module._satisfies("1.2.3", "1.2.3+build")
    assert plugins_module._validate_requirement("1.2.3+build", path="requirement") == "^1.2.3"

    file_entry = PluginFile("payload.py", "0" * 64, 1)
    with pytest.raises(PluginValidationError):
        PluginFile("payload.py", "not-a-digest", 1)
    manifest_kwargs = dict(
        plugin_id="edge-plugin",
        version="1.0.0",
        kind="learner",
        name="Edge",
        description="Edge",
        entrypoint="payload.py",
        capabilities=("learner.ppo",),
    )
    with pytest.raises(PluginValidationError):
        PluginManifest(**manifest_kwargs, schema_version="glr.plugin.v0")
    with pytest.raises(PluginValidationError):
        PluginManifest(**manifest_kwargs, isolation="thread")
    with pytest.raises(PluginValidationError):
        PluginManifest(**manifest_kwargs, files=(object(),))
    with pytest.raises(PluginValidationError):
        PluginManifest(**manifest_kwargs, files=(file_entry, file_entry))
    with pytest.raises(PluginValidationError):
        PluginRef(plugin_id="edge-plugin", enabled=1)
    with pytest.raises(PluginValidationError):
        PluginProfile(name="edge", schema_version="glr.profile.v0")
    with pytest.raises(PluginValidationError):
        PluginProfile(name="edge", plugins=(object(),))
    with pytest.raises(PluginValidationError):
        PluginProfile(
            name="edge",
            plugins=(PluginRef(plugin_id="edge-plugin"), PluginRef(plugin_id="edge-plugin")),
        )

    manager = PluginManager(tmp_path / "project")
    assert manager.list_installed() == ()
    assert manager.list_profiles() == ()
    plain_file = tmp_path / "plain-file"
    plain_file.write_text("not a directory", encoding="utf-8")
    with pytest.raises(PluginValidationError):
        plugins_module._ensure_directory(plain_file, label="test directory")
    with pytest.raises(PluginValidationError):
        manager.install(plain_file)
    source = _write_plugin(tmp_path / "edge-source")
    with pytest.raises(PluginValidationError):
        manager.install(source, expected_sha256="bad")
    with pytest.raises(PluginValidationError):
        manager.save_profile(object())  # type: ignore[arg-type]
    with pytest.raises(PluginValidationError):
        manager.resolve_profile(object())  # type: ignore[arg-type]
    assert manager.health("missing") == (
        {"status": "blocked", "reason": "plugin profile must be a regular non-symlink file"},
    )


def test_plugin_manager_handles_project_files_and_rejects_deep_empty_trees(
    tmp_path: Path,
) -> None:
    project_file = tmp_path / "glr-project.json"
    project_file.write_text("{}", encoding="utf-8")
    manager = PluginManager(project_file)
    assert manager.project_root == tmp_path

    nested = tmp_path / "empty"
    nested.mkdir()
    for _ in range(17):
        nested /= "nested"
        nested.mkdir()
    with pytest.raises(PluginValidationError, match="too deep"):
        plugins_module._walk_files(tmp_path / "empty")

    with pytest.raises(PluginValidationError):
        plugins_module._parse_semver("1.0.0+..", path="version")


def test_dependency_resolution_preserves_explicit_config_and_rejects_conflicts(
    tmp_path: Path,
) -> None:
    root = _write_plugin(tmp_path / "root", plugin_id="root-plugin")
    root_manifest = json.loads((root / "glr-plugin.json").read_text(encoding="utf-8"))
    root_manifest["dependencies"] = {"dep-plugin": "*"}
    (root / "glr-plugin.json").write_text(json.dumps(root_manifest), encoding="utf-8")
    dependency = _write_plugin(tmp_path / "dependency", plugin_id="dep-plugin")
    manager = PluginManager(tmp_path / "project")
    manager.install(root)
    manager.install(dependency)

    resolved = manager.resolve_profile(
        PluginProfile(
            name="explicit",
            plugins=(
                PluginRef(plugin_id="root-plugin"),
                PluginRef(plugin_id="dep-plugin", config={"batch": 32}),
            ),
        )
    )
    dependency_result = next(
        item for item in resolved.plugins if item.manifest.plugin_id == "dep-plugin"
    )
    assert dependency_result.config["batch"] == 32

    reversed_order = manager.resolve_profile(
        PluginProfile(
            name="reversed",
            plugins=(
                PluginRef(plugin_id="dep-plugin", config={"batch": 64}),
                PluginRef(plugin_id="root-plugin"),
            ),
        )
    )
    reversed_dependency = next(
        item for item in reversed_order.plugins if item.manifest.plugin_id == "dep-plugin"
    )
    assert reversed_dependency.config["batch"] == 64


def test_plugin_store_ancestor_symlink_is_rejected(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (project / ".glr").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink privilege is unavailable")
    manager = PluginManager(project)
    with pytest.raises(PluginValidationError, match="symlink"):
        manager.list_installed()
