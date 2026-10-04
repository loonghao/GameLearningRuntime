"""Offline tests for shared rules and decision-consumption contracts."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, asdict, dataclass, replace
from typing import Any

import pytest

from game_learning_runtime.continuous_learning import HostAuthority, HostRoleCapability
from game_learning_runtime.errors import ContractViolation
from game_learning_runtime.knowledge_evidence import (
    CapabilityGap,
    CapabilityGapKind,
    ConsumptionState,
    DecisionConsumptionReceipt,
    RuleIndexBinding,
)

RULES_SHA = "1" * 64
INDEX_SHA = "2" * 64
FINDING_SHA = "3" * 64
EVIDENCE_SHA = "4" * 64


@dataclass(frozen=True)
class _OverserializingBinding(RuleIndexBinding):
    private_host: Any = None

    def to_mapping(self) -> dict[str, Any]:
        return {"schema_version": "glr.rule-index-binding.v1", **asdict(self)}


class _MapperMustNotRunBinding(RuleIndexBinding):
    def to_mapping(self) -> dict[str, Any]:
        raise AssertionError("nested subclass mapper was invoked")


def _binding(**changes: Any) -> RuleIndexBinding:
    fields = {
        "environment_id": "example.choice-environment",
        "protocol_version": "1.0",
        "rules_version": "build-current-mode-a",
        "rules_sha256": RULES_SHA,
        "index_sha256": INDEX_SHA,
        "index_rules_sha256": RULES_SHA,
    }
    fields.update(changes)
    return RuleIndexBinding(**fields)


def _current_scope(**changes: Any) -> dict[str, Any]:
    fields = {
        "environment_id": "example.choice-environment",
        "protocol_version": "1.0",
        "rules_version": "build-current-mode-a",
        "rules_sha256": RULES_SHA,
    }
    fields.update(changes)
    return fields


def _receipt(**changes: Any) -> DecisionConsumptionReceipt:
    fields: dict[str, Any] = {
        "decision_id": "decision-7",
        "finding_id": "finding-2",
        "finding_sha256": FINDING_SHA,
        "consumer_id": "option-scorer",
        "binding": _binding(),
        "state": ConsumptionState.RETRIEVED,
        "use_evidence_sha256": None,
    }
    fields.update(changes)
    return DecisionConsumptionReceipt(**fields)


def _gap(**changes: Any) -> CapabilityGap:
    fields: dict[str, Any] = {
        "decision_id": "decision-7",
        "kind": CapabilityGapKind.MISSING_ACTION,
        "required_semantic": "revise-options",
        "available_semantics": ("select-option", "refresh-inventory"),
        "evidence_sha256": EVIDENCE_SHA,
    }
    fields.update(changes)
    return CapabilityGap(**fields)


def test_matching_rule_index_can_be_consumed() -> None:
    _binding().assert_current(**_current_scope())


@pytest.mark.parametrize(
    ("field", "foreign"),
    [
        ("environment_id", "example.other-environment"),
        ("protocol_version", "2.0"),
        ("rules_version", "build-previous-mode-a"),
        ("rules_sha256", "a" * 64),
    ],
)
def test_rule_binding_refuses_foreign_or_stale_scope(field: str, foreign: str) -> None:
    with pytest.raises(ContractViolation, match=field):
        _binding().assert_current(**_current_scope(**{field: foreign}))


def test_old_index_cannot_be_relabeled_as_current_rules() -> None:
    with pytest.raises(ContractViolation, match="different rules"):
        _binding(index_rules_sha256="a" * 64)


def test_same_version_label_does_not_hide_changed_rule_contents() -> None:
    modified_rules = _binding(rules_sha256="a" * 64, index_rules_sha256="a" * 64)
    with pytest.raises(ContractViolation, match="rules_sha256"):
        modified_rules.assert_current(**_current_scope())


def test_binding_is_frozen_and_versioned_json_roundtrip_preserves_identity() -> None:
    binding = _binding()
    assert RuleIndexBinding.from_mapping(json.loads(json.dumps(binding.to_mapping()))) == binding
    with pytest.raises(FrozenInstanceError):
        binding.rules_version = "replacement"  # type: ignore[misc]


@pytest.mark.parametrize("digest", ["", "a" * 63, "A" * 64, "z" * 64, None, 42])
def test_invalid_rule_digests_are_rejected(digest: object) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        _binding(index_sha256=digest)


@pytest.mark.parametrize("version", ["", " ", "1\n2", "a" * 257, None, 42])
def test_rule_scope_requires_bounded_explicit_versions(version: object) -> None:
    with pytest.raises(ValueError, match="version"):
        _binding(rules_version=version)


def test_retrieved_finding_is_not_actual_decision_use() -> None:
    receipt = _receipt()
    assert not receipt.is_used
    assert receipt.use_evidence_sha256 is None
    receipt.assert_for_decision(
        decision_id="decision-7",
        finding_id="finding-2",
        finding_sha256=FINDING_SHA,
        consumer_id="option-scorer",
        binding=_binding(),
    )


def test_consumer_use_requires_linked_evidence_and_can_come_from_a_direct_table() -> None:
    receipt = _receipt(state=ConsumptionState.USED, use_evidence_sha256=EVIDENCE_SHA)
    assert receipt.is_used
    assert receipt.state is ConsumptionState.USED
    assert (
        DecisionConsumptionReceipt.from_mapping(json.loads(json.dumps(receipt.to_mapping())))
        == receipt
    )
    # There are no task-outcome or learning claims in a use receipt.
    assert set(receipt.to_mapping()).isdisjoint({"reward", "success", "learned", "authorized"})


@pytest.mark.parametrize("evidence", [None, "", "a" * 63, "A" * 64])
def test_actual_use_without_valid_consumer_evidence_is_refused(evidence: object) -> None:
    with pytest.raises(ValueError, match="use_evidence_sha256"):
        _receipt(state=ConsumptionState.USED, use_evidence_sha256=evidence)


def test_retrieval_cannot_smuggle_a_claim_of_actual_use() -> None:
    with pytest.raises(ValueError, match="retrieved-only"):
        _receipt(use_evidence_sha256=EVIDENCE_SHA)


@pytest.mark.parametrize(
    ("field", "foreign"), [("decision_id", "decision-8"), ("consumer_id", "other-scorer")]
)
def test_use_receipt_cannot_move_between_decisions_or_consumers(field: str, foreign: str) -> None:
    scope: dict[str, Any] = {
        "decision_id": "decision-7",
        "finding_id": "finding-2",
        "finding_sha256": FINDING_SHA,
        "consumer_id": "option-scorer",
        "binding": _binding(),
    }
    scope[field] = foreign
    with pytest.raises(ContractViolation, match="decision or consumer"):
        _receipt().assert_for_decision(**scope)


@pytest.mark.parametrize(
    ("field", "foreign"), [("finding_id", "finding-3"), ("finding_sha256", "a" * 64)]
)
def test_use_receipt_cannot_move_between_findings_or_contents(field: str, foreign: str) -> None:
    scope: dict[str, Any] = {
        "decision_id": "decision-7",
        "finding_id": "finding-2",
        "finding_sha256": FINDING_SHA,
        "consumer_id": "option-scorer",
        "binding": _binding(),
    }
    scope[field] = foreign
    with pytest.raises(ContractViolation, match="finding revision"):
        _receipt().assert_for_decision(**scope)


def test_use_receipt_cannot_move_between_index_revisions_for_the_same_rules() -> None:
    with pytest.raises(ContractViolation, match="rule index binding"):
        _receipt().assert_for_decision(
            decision_id="decision-7",
            finding_id="finding-2",
            finding_sha256=FINDING_SHA,
            consumer_id="option-scorer",
            binding=_binding(index_sha256="a" * 64),
        )


def test_use_receipt_cannot_bypass_current_rules_via_a_direct_table() -> None:
    old = _binding(
        rules_version="build-previous-mode-a",
        rules_sha256="a" * 64,
        index_rules_sha256="a" * 64,
    )
    receipt = _receipt(binding=old, state=ConsumptionState.USED, use_evidence_sha256=EVIDENCE_SHA)
    with pytest.raises(ContractViolation, match="rule index binding"):
        receipt.assert_for_decision(
            decision_id="decision-7",
            finding_id="finding-2",
            finding_sha256=FINDING_SHA,
            consumer_id="option-scorer",
            binding=_binding(),
        )


def test_receipt_is_frozen_and_rejects_untyped_bindings() -> None:
    receipt = _receipt()
    with pytest.raises(FrozenInstanceError):
        receipt.state = ConsumptionState.USED  # type: ignore[misc]
    with pytest.raises(TypeError, match="RuleIndexBinding"):
        _receipt(binding=_binding().to_mapping())
    with pytest.raises(ValueError):
        _receipt(state="learned")


@pytest.mark.parametrize("kind", list(CapabilityGapKind))
def test_capability_gap_is_typed_passive_data(kind: CapabilityGapKind) -> None:
    gap = _gap(kind=kind)
    gap.verify_missing()
    assert gap.available_semantics == ("refresh-inventory", "select-option")
    assert CapabilityGap.from_mapping(json.loads(json.dumps(gap.to_mapping()))) == gap
    assert set(gap.to_mapping()).isdisjoint({"action", "permissions", "execute", "authorized"})
    with pytest.raises(FrozenInstanceError):
        gap.required_semantic = "select-option"  # type: ignore[misc]


def test_existing_semantic_cannot_be_reported_as_missing() -> None:
    with pytest.raises(ContractViolation, match="semantic is present"):
        _gap(required_semantic="select-option")


def test_stale_capability_gap_is_rejected_after_interface_improvement() -> None:
    gap = _gap()
    with pytest.raises(ContractViolation, match="semantic is present"):
        gap.verify_missing(("select-option", "refresh-inventory", "revise-options"))


def test_missing_resource_field_does_not_invent_resource_count_or_action_authority() -> None:
    gap = _gap(
        kind=CapabilityGapKind.MISSING_OBSERVATION,
        required_semantic="remaining-opportunities",
        available_semantics=("option-text", "inventory-refresh-cost"),
    )
    gap.verify_missing()
    assert "remaining_count" not in gap.to_mapping()
    assert "allowed_actions" not in gap.to_mapping()


@pytest.mark.parametrize(
    "surface",
    [
        ("select-option", "select-option"),
        ["select-option"],
        "select-option",
        (42,),
        tuple(f"semantic-{index}" for index in range(257)),
    ],
)
def test_capability_surface_must_be_bounded_unique_immutable_and_typed(surface: object) -> None:
    with pytest.raises(ValueError):
        _gap(available_semantics=surface)


@pytest.mark.parametrize("factory", [_binding, _receipt, _gap])
def test_all_wire_records_reject_unknown_executable_or_authority_fields(factory: Any) -> None:
    record = factory()
    payload = record.to_mapping()
    payload["authorized"] = True
    with pytest.raises(ValueError, match="unexpected fields"):
        type(record).from_mapping(payload)


@pytest.mark.parametrize("factory", [_binding, _receipt, _gap])
def test_all_wire_records_reject_missing_fields_and_unrecognized_versions(factory: Any) -> None:
    record = factory()
    payload = record.to_mapping()
    del payload["schema_version"]
    with pytest.raises(ValueError, match="missing"):
        type(record).from_mapping(payload)
    payload = record.to_mapping()
    payload["schema_version"] = "glr.unreviewed.v9"
    with pytest.raises(ValueError, match="schema_version"):
        type(record).from_mapping(payload)


def test_nested_rule_binding_is_revalidated_on_receipt_decode() -> None:
    payload = _receipt().to_mapping()
    payload["binding"]["index_rules_sha256"] = "a" * 64
    with pytest.raises(ContractViolation, match="different rules"):
        DecisionConsumptionReceipt.from_mapping(payload)


@pytest.mark.parametrize("kind", ["authority", "capability"])
def test_nested_binding_mapper_projects_base_fields_before_privileged_extras_can_flatten(
    kind: str,
) -> None:
    if kind == "authority":
        private_host: Any = HostAuthority(
            "host.synthetic", ("evaluator",), ("supervisor",), ("reviewer",), bytes(range(32))
        )
    else:
        private_host = HostRoleCapability(
            "epoch",
            "1" * 64,
            "campaign.synthetic",
            "trial",
            "token",
            "evaluator",
            "evaluator",
            "2" * 64,
            bytes(range(32)),
        )
    base = _binding()
    constructor = {
        key: value for key, value in base.to_mapping().items() if key != "schema_version"
    }
    extended = _OverserializingBinding(**constructor, private_host=private_host)
    receipt = _receipt(binding=extended)
    exported = receipt.to_mapping()
    # Compare only keys first: a regression must never print synthetic private material.
    observed_fields = frozenset(exported["binding"])
    expected_fields = frozenset(base.to_mapping())
    assert observed_fields == expected_fields
    decoded = DecisionConsumptionReceipt.from_mapping(json.loads(json.dumps(exported)))
    assert decoded == _receipt(binding=base)


def test_nested_binding_export_never_executes_a_subclass_mapper() -> None:
    base = _binding()
    constructor = {
        key: value for key, value in base.to_mapping().items() if key != "schema_version"
    }
    extended = _MapperMustNotRunBinding(**constructor)
    exported = _receipt(binding=extended).to_mapping()
    assert exported["binding"] == base.to_mapping()


def test_gap_decode_requires_an_array_and_supported_kind() -> None:
    payload = _gap().to_mapping()
    payload["available_semantics"] = "select-option"
    with pytest.raises(TypeError, match="array"):
        CapabilityGap.from_mapping(payload)
    with pytest.raises(ValueError):
        replace(_gap(), kind="execute-unlisted-action")  # type: ignore[arg-type]


def test_offline_contract_loop_rejects_stale_rules_then_records_declared_consumer_use() -> None:
    """An entirely synthetic sequence of rules and consumer evidence."""
    old = _binding(
        rules_version="build-previous-mode-a",
        rules_sha256="a" * 64,
        index_rules_sha256="a" * 64,
    )
    with pytest.raises(ContractViolation):
        old.assert_current(**_current_scope())

    gap = _gap()
    gap.verify_missing()
    refreshed = _binding()
    refreshed.assert_current(**_current_scope())
    retrieved = _receipt(binding=refreshed)
    assert not retrieved.is_used
    used = replace(retrieved, state=ConsumptionState.USED, use_evidence_sha256=EVIDENCE_SHA)
    used.assert_for_decision(
        decision_id="decision-7",
        finding_id="finding-2",
        finding_sha256=FINDING_SHA,
        consumer_id="option-scorer",
        binding=refreshed,
    )
    assert used.is_used
    # The loop records evidence; the adapter action vocabulary remains the original tuple.
    assert gap.available_semantics == ("refresh-inventory", "select-option")
