from __future__ import annotations

import json
from typing import Any

import pytest
from yield_rca_core.causal_adversarial import _normalize_challenge_payload
from yield_rca_core.causal_confirmation import evaluate_impact_lot_gate
from yield_rca_core.causal_evidence_gap import build_causal_evidence_gaps
from yield_rca_core.causal_evidence_matrix import build_causal_evidence_matrix
from yield_rca_core.causal_hypothesis import CausalClaimStatus, CausalHypothesis
from yield_rca_core.causal_investigation_models import CandidateSemanticProfile
from yield_rca_core.evidence_builder import EvidenceBuilder
from yield_rca_core.evidence_models import (
    EVIDENCE_SCHEMA_VERSION,
    EntityType,
    Evidence,
    EvidenceEntity,
    EvidenceSourceType,
    EvidenceType,
    ModelValidationError,
)
from yield_rca_core.hypothesis_candidate_generator import (
    HypothesisCandidateProposal,
    _candidate_evidence_closure_assessment,
)
from yield_rca_core.incident_evidence import build_incident_observation_evidence
from yield_rca_core.investigation_decision import assess_action_values
from yield_rca_core.llm_gateway import LLMOutputValidationError
from yield_rca_core.models import AgentKind, RCAJob, ToolInput
from yield_rca_core.repositories import Row
from yield_rca_core.tool_layer import (
    AnalyzeParameterShiftTool,
    SummarizeDefectWatTool,
)

LANE_A = "lane:2500:EQ_A:EQ_A_CH01:RCP_A"
LANE_B = "lane:2500:EQ_A:EQ_A_CH01:RCP_B"


class AtomicProjectionRepository:
    def __init__(self) -> None:
        self.tables: dict[str, list[Row]] = {
            "fdc_feature": [
                {
                    "lot_id": "LOT_BAD",
                    "wafer_id": "LOT_BAD_W01",
                    "operation_no": "2500",
                    "equipment_id": "EQ_A",
                    "chamber_id": "EQ_A_CH01",
                    "recipe_id": "RCP_A",
                    "recipe_version": "V1",
                    "parameter_name": "oxygen_flow",
                    "observed_value": "130",
                    "baseline_value": "100",
                    "delta_percent": "30",
                    "unit": "sccm",
                    "ooc_flag": "true",
                    "severity": "HIGH",
                    "measured_at": "2026-01-01T01:05:00+00:00",
                },
                {
                    "lot_id": "LOT_CONTROL",
                    "wafer_id": "LOT_CONTROL_W01",
                    "operation_no": "2500",
                    "equipment_id": "EQ_A",
                    "chamber_id": "EQ_A_CH01",
                    "recipe_id": "RCP_A",
                    "recipe_version": "V1",
                    "parameter_name": "oxygen_flow",
                    "observed_value": "100",
                    "baseline_value": "100",
                    "delta_percent": "0",
                    "unit": "sccm",
                    "ooc_flag": "false",
                    "severity": "NORMAL",
                    "measured_at": "2026-01-01T01:06:00+00:00",
                },
                {
                    "lot_id": "LOT_OTHER_RECIPE",
                    "wafer_id": "LOT_OTHER_RECIPE_W01",
                    "operation_no": "2500",
                    "equipment_id": "EQ_A",
                    "chamber_id": "EQ_A_CH01",
                    "recipe_id": "RCP_B",
                    "recipe_version": "V1",
                    "parameter_name": "oxygen_flow",
                    "observed_value": "150",
                    "baseline_value": "100",
                    "delta_percent": "50",
                    "unit": "sccm",
                    "ooc_flag": "true",
                    "severity": "HIGH",
                    "measured_at": "2026-01-01T01:07:00+00:00",
                },
            ],
            "defect_summary": [
                {
                    "lot_id": "LOT_BAD",
                    "wafer_id": "LOT_BAD_W01",
                    "defect_type": "R39",
                    "pattern_type": "CENTER",
                    "defect_count": "10",
                    "inspected_at": "2026-01-01T03:00:00+00:00",
                },
                {
                    "lot_id": "LOT_CONTROL",
                    "wafer_id": "LOT_CONTROL_W01",
                    "defect_type": "EDGE_PARTICLE",
                    "pattern_type": "EDGE",
                    "defect_count": "1",
                    "inspected_at": "2026-01-01T03:01:00+00:00",
                },
            ],
            "wat_result": [],
            "metrology_result": [
                {
                    "lot_id": "LOT_BAD",
                    "wafer_id": "LOT_BAD_W01",
                    "measurement_stage": "POST_ETCH",
                    "metric_name": "contact_resistance",
                    "measured_value": "12",
                    "unit": "ohm",
                    "pass_fail": "false",
                    "measured_at": "2026-01-01T03:10:00+00:00",
                },
                {
                    "lot_id": "LOT_CONTROL",
                    "wafer_id": "LOT_CONTROL_W01",
                    "measurement_stage": "POST_ETCH",
                    "metric_name": "contact_resistance",
                    "measured_value": "2",
                    "unit": "ohm",
                    "pass_fail": "true",
                    "measured_at": "2026-01-01T03:11:00+00:00",
                },
            ],
        }

    def rows(self, table_name: str) -> list[Row]:
        return list(self.tables.get(table_name, []))


def _lot_ids(item: Evidence) -> set[str]:
    return {
        entity.entity_id
        for entity in item.entities
        if entity.entity_type == EntityType.LOT.value
    }


def _typed_evidence(
    evidence_id: str,
    evidence_type: str,
    *,
    lot_id: str,
    recipe: str,
    parameter: str | None = None,
    outcome: str | None = None,
    timestamp: str = "2026-01-01T01:10:00+00:00",
    window: bool = False,
    metadata_overrides: dict[str, Any] | None = None,
) -> Evidence:
    entities = [
        EvidenceEntity(EntityType.LOT.value, lot_id),
        EvidenceEntity(EntityType.OPERATION.value, "2500"),
        EvidenceEntity(EntityType.EQUIPMENT.value, "EQ_A"),
        EvidenceEntity(EntityType.CHAMBER.value, "EQ_A_CH01"),
        EvidenceEntity(EntityType.RECIPE.value, recipe),
    ]
    if parameter is not None:
        entities.append(EvidenceEntity(EntityType.PARAMETER.value, parameter))
    if outcome is not None:
        entities.append(EvidenceEntity(EntityType.PARAMETER.value, outcome))
    metadata: dict[str, Any] = {
        "lane_id": LANE_A if recipe == "RCP_A" else LANE_B,
    }
    if parameter is not None:
        metadata["direction"] = "high"
    if window:
        metadata.update(
            {
                "excursion_start": "2026-01-01T01:00:00+00:00",
                "excursion_end": "2026-01-01T02:00:00+00:00",
            }
        )
    if metadata_overrides:
        metadata.update(metadata_overrides)
    return Evidence(
        evidence_id=evidence_id,
        source_type=EvidenceSourceType.ANALYTICS.value,
        source_id=f"source:{evidence_id}",
        summary=evidence_id,
        source_table="contract_fixture",
        source_field=parameter or outcome,
        timestamp=timestamp,
        metadata=metadata,
        evidence_type=evidence_type,
        source_agent=AgentKind.RCA_REASONING.value,
        source_tool="contract_fixture",
        observation=evidence_id,
        entities=entities,
        confidence=1.0,
        evidence_schema_version=EVIDENCE_SCHEMA_VERSION,
    )


def _lane(lane_id: str, recipe: str, lot_id: str) -> dict[str, Any]:
    return {
        "lane_id": lane_id,
        "operation": "2500",
        "equipment": "EQ_A",
        "chamber": "EQ_A_CH01",
        "recipe": recipe,
        "exposed_lot_ids": [lot_id],
    }


def _chamber_profile() -> CandidateSemanticProfile:
    return CandidateSemanticProfile(
        candidate_id="REQ_SCOPE:llm:1",
        scope_relation="shared_effect",
        claimed_scope_kind="chamber",
        claimed_operation="2500",
        claimed_equipment="EQ_A",
        claimed_chamber="EQ_A_CH01",
        claimed_recipe=None,
        claimed_lane_ids=(LANE_A,),
        comparison_lane_ids=(LANE_A, LANE_B),
        mechanism_claim="Oxygen-flow instability leaves residue.",
        primary_mechanism="incomplete removal leaves residue",
        mechanism_relation="reference",
    )


def test_parameter_deviation_is_atomic_and_recipe_filtered() -> None:
    output = AnalyzeParameterShiftTool(AtomicProjectionRepository()).run(
        ToolInput(
            tool_name="analyze_parameter_shift",
            request_id="REQ_ATOMIC_FDC",
            requested_by=AgentKind.FDC.value,
            parameters={
                "lot_ids": ["LOT_BAD", "LOT_CONTROL", "LOT_OTHER_RECIPE"],
                "operation_no": "2500",
                "equipment_id": "EQ_A",
                "chamber_id": "EQ_A_CH01",
                "recipe_id": "RCP_A",
                "lane_id": LANE_A,
            },
        )
    )

    deviation = next(
        item
        for item in output.evidence
        if item.evidence_type == EvidenceType.PARAMETER_DEVIATION.value
    )
    assert _lot_ids(deviation) == {"LOT_BAD"}
    assert {
        entity.entity_id
        for entity in deviation.entities
        if entity.entity_type == EntityType.RECIPE.value
    } == {"RCP_A"}
    assert deviation.metadata["selected_lot_ids"] == (
        "LOT_BAD",
        "LOT_CONTROL",
        "LOT_OTHER_RECIPE",
    )
    assert deviation.metadata["abnormal_lot_ids"] == ("LOT_BAD",)


def test_defect_and_metrology_positive_entities_exclude_controls() -> None:
    output = SummarizeDefectWatTool(AtomicProjectionRepository()).run(
        ToolInput(
            tool_name="summarize_defect_wat",
            request_id="REQ_ATOMIC_QUALITY",
            requested_by=AgentKind.DEFECT_WAT.value,
            parameters={"lot_ids": ["LOT_BAD", "LOT_CONTROL"]},
        )
    )

    defect = next(
        item
        for item in output.evidence
        if item.evidence_type == EvidenceType.DEFECT_SIGNAL.value
    )
    metrology = next(
        item
        for item in output.evidence
        if item.evidence_type == EvidenceType.METROLOGY_DEVIATION.value
    )
    assert _lot_ids(defect) == {"LOT_BAD"}
    assert _lot_ids(metrology) == {"LOT_BAD"}


def test_lane_bound_evidence_rejects_mixed_recipe_identity() -> None:
    with pytest.raises(ModelValidationError, match="Lane-bound Evidence identity"):
        EvidenceBuilder.from_tool(
            tool_input=ToolInput(
                tool_name="contract_tool",
                request_id="REQ_LANE_IDENTITY",
                requested_by=AgentKind.FDC.value,
                parameters={
                    "lane_id": LANE_A,
                    "operation_no": "2500",
                    "equipment_id": "EQ_A",
                    "chamber_id": "EQ_A_CH01",
                    "recipe_id": "RCP_A",
                },
            ),
            evidence_id="EV_MIXED_RECIPE",
            evidence_type=EvidenceType.PARAMETER_DEVIATION,
            source_type=EvidenceSourceType.FDC,
            observation="Mixed Recipe Evidence",
            entities=[
                EvidenceEntity(EntityType.RECIPE.value, "RCP_A"),
                EvidenceEntity(EntityType.RECIPE.value, "RCP_B"),
            ],
            confidence=1.0,
            source_id="mixed",
        )


def test_chamber_shared_effect_requires_process_and_outcome_per_recipe_lane() -> None:
    profile = _chamber_profile()
    items = [
        _typed_evidence(
            "EV_EXPOSURE_A",
            EvidenceType.EQUIPMENT_EXPOSURE.value,
            lot_id="LOT_A",
            recipe="RCP_A",
        ),
        _typed_evidence(
            "EV_PARAMETER_A",
            EvidenceType.PARAMETER_DEVIATION.value,
            lot_id="LOT_A",
            recipe="RCP_A",
            parameter="oxygen_flow",
        ),
        _typed_evidence(
            "EV_WINDOW_A",
            EvidenceType.EXCURSION_WINDOW.value,
            lot_id="LOT_A",
            recipe="RCP_A",
            window=True,
        ),
        _typed_evidence(
            "EV_OUTCOME_A",
            EvidenceType.METROLOGY_DEVIATION.value,
            lot_id="LOT_A",
            recipe="RCP_A",
            outcome="contact_resistance",
        ),
        _typed_evidence(
            "EV_PARAMETER_B",
            EvidenceType.PARAMETER_DEVIATION.value,
            lot_id="LOT_B",
            recipe="RCP_B",
            parameter="oxygen_flow",
        ),
        _typed_evidence(
            "EV_OUTCOME_B",
            EvidenceType.METROLOGY_DEVIATION.value,
            lot_id="LOT_B",
            recipe="RCP_B",
            outcome="contact_resistance",
        ),
    ]
    cited_ids = tuple(
        item.evidence_id
        for item in items
        if item.evidence_id.endswith("_A")
    )
    hypothesis = CausalHypothesis(
        root_cause=(
            "EQ_A_CH01 oxygen_flow instability during operation 2500 causes "
            "high contact resistance"
        ),
        causal_explanation=(
            "High oxygen flow changes removal chemistry, leaves residue, and "
            "raises contact resistance."
        ),
        supporting_evidence_ids=cited_ids,
    )
    proposal = HypothesisCandidateProposal(
        root_cause=hypothesis.root_cause,
        causal_explanation=hypothesis.causal_explanation,
        supporting_evidence_ids=hypothesis.supporting_evidence_ids,
        contradicting_evidence_ids=(),
    )

    matrix = build_causal_evidence_matrix(
        hypothesis,
        items,
        semantic_profile=profile,
        causal_lanes=[
            _lane(LANE_A, "RCP_A", "LOT_A"),
            _lane(LANE_B, "RCP_B", "LOT_B"),
        ],
    )
    closure = _candidate_evidence_closure_assessment(
        (proposal,),
        (profile,),
        evidence_by_id={item.evidence_id: item for item in items},
        causal_lanes=[
            _lane(LANE_A, "RCP_A", "LOT_A"),
            _lane(LANE_B, "RCP_B", "LOT_B"),
        ],
    )

    scope = matrix.claims["scope"]
    assert scope.status == CausalClaimStatus.INCOMPLETE.value
    assert scope.facts["scope_identity_status"] == "grounded"
    assert scope.facts["scope_effect_coverage_status"] == "partial"
    assert scope.facts["scope_process_covered_lane_ids"] == [LANE_A]
    assert scope.facts["scope_outcome_covered_lane_ids"] == [LANE_A]
    assert scope.facts["scope_missing_process_lane_ids"] == [LANE_B]
    assert scope.facts["scope_missing_outcome_lane_ids"] == [LANE_B]
    assert closure[0]["status"] == "incomplete"
    assert closure[0]["matching_claimed_scope_lane_ids"] == [LANE_A, LANE_B]
    assert {
        (gap["lane_id"], gap["evidence_role"])
        for gap in closure[0]["closure_gaps"]
    } == {
        (LANE_B, "parameter"),
        (LANE_B, "outcome"),
    }

    gaps = build_causal_evidence_gaps([matrix])
    coverage_gaps = [item for item in gaps if item.get("gap_origin") == "scope_coverage"]
    assert {item["discriminator_kind"] for item in coverage_gaps} == {
        "parameter_anomaly",
        "product_outcome",
    }
    assert all(item["target_scope"]["lane_id"] == LANE_B for item in coverage_gaps)
    assert not any(
        item["gap_id"] == "candidate_0.scope.incomplete" for item in gaps
    )

    outcome_gap = next(
        item
        for item in coverage_gaps
        if item["discriminator_kind"] == "product_outcome"
    )
    assessments = assess_action_values(
        options=[
            {
                "option_id": "scope-outcome",
                "action_kind": "validate_shared_defect_pattern",
                "source": "defect_wat",
                "scope": outcome_gap["target_scope"],
                "gap": outcome_gap,
            }
        ],
        action_records=[],
        gain_history=[],
        remaining_tool_budget=10,
        evidence=items,
        competition_requirement="mechanism_required",
        competition_axes=["mechanism"],
    )
    assert assessments[0].eligible is True
    assert assessments[0].high_value is True
    assert assessments[0].decision_impact == "confirmation"


def test_complete_broad_scope_lane_coverage_supports_shared_effect() -> None:
    profile = _chamber_profile()
    items = [
        *[
            _typed_evidence(
                f"EV_PARAMETER_{recipe}",
                EvidenceType.PARAMETER_DEVIATION.value,
                lot_id=lot_id,
                recipe=recipe,
                parameter="oxygen_flow",
            )
            for recipe, lot_id in (("RCP_A", "LOT_A"), ("RCP_B", "LOT_B"))
        ],
        *[
            _typed_evidence(
                f"EV_OUTCOME_{recipe}",
                EvidenceType.METROLOGY_DEVIATION.value,
                lot_id=lot_id,
                recipe=recipe,
                outcome="contact_resistance",
            )
            for recipe, lot_id in (("RCP_A", "LOT_A"), ("RCP_B", "LOT_B"))
        ],
    ]
    candidate = CausalHypothesis(
        root_cause=(
            "EQ_A_CH01 oxygen_flow instability during operation 2500 causes "
            "high contact resistance"
        ),
        causal_explanation=(
            "High oxygen flow changes removal chemistry, leaves residue, and "
            "raises contact resistance."
        ),
        supporting_evidence_ids=tuple(item.evidence_id for item in items),
    )

    matrix = build_causal_evidence_matrix(
        candidate,
        items,
        semantic_profile=profile,
        causal_lanes=[
            _lane(LANE_A, "RCP_A", "LOT_A"),
            _lane(LANE_B, "RCP_B", "LOT_B"),
        ],
    )

    scope = matrix.claims["scope"]
    assert scope.status == CausalClaimStatus.SUPPORTED.value
    assert scope.facts["scope_effect_coverage_status"] == "complete"
    assert scope.facts["scope_missing_process_lane_ids"] == []
    assert scope.facts["scope_missing_outcome_lane_ids"] == []


def test_broad_scope_without_lane_inventory_preserves_legacy_compatibility() -> None:
    profile = _chamber_profile()
    item = _typed_evidence(
        "EV_PARAMETER_LEGACY",
        EvidenceType.PARAMETER_DEVIATION.value,
        lot_id="LOT_A",
        recipe="RCP_A",
        parameter="oxygen_flow",
    )
    candidate = CausalHypothesis(
        root_cause="EQ_A_CH01 oxygen_flow instability during operation 2500",
        causal_explanation="Oxygen-flow instability changes removal chemistry.",
        supporting_evidence_ids=(item.evidence_id,),
    )

    matrix = build_causal_evidence_matrix(
        candidate,
        [item],
        semantic_profile=profile,
    )

    assert matrix.claims["scope"].status == CausalClaimStatus.SUPPORTED.value
    assert matrix.claims["scope"].facts["scope_effect_coverage_status"] == (
        "unavailable"
    )


def test_scope_grounded_impact_gate_includes_other_recipe_and_excludes_control() -> None:
    profile = _chamber_profile()
    candidate = CausalHypothesis(
        root_cause=(
            "EQ_A_CH01 high oxygen_flow during operation 2500 causes high "
            "contact resistance"
        ),
        causal_explanation=(
            "High oxygen flow changes removal chemistry and raises contact resistance."
        ),
        supporting_evidence_ids=("EV_PARAMETER_B",),
    )
    items = [
        _typed_evidence(
            "EV_EXPOSURE_B",
            EvidenceType.EQUIPMENT_EXPOSURE.value,
            lot_id="LOT_B",
            recipe="RCP_B",
        ),
        _typed_evidence(
            "EV_PARAMETER_B",
            EvidenceType.PARAMETER_DEVIATION.value,
            lot_id="LOT_B",
            recipe="RCP_B",
            parameter="oxygen_flow",
        ),
        _typed_evidence(
            "EV_WINDOW_B",
            EvidenceType.EXCURSION_WINDOW.value,
            lot_id="LOT_B",
            recipe="RCP_B",
            window=True,
        ),
        _typed_evidence(
            "EV_OUTCOME_B",
            EvidenceType.METROLOGY_DEVIATION.value,
            lot_id="LOT_B",
            recipe="RCP_B",
            outcome="contact_resistance",
            timestamp="2026-01-01T03:00:00+00:00",
        ),
        _typed_evidence(
            "EV_EXPOSURE_CONTROL",
            EvidenceType.EQUIPMENT_EXPOSURE.value,
            lot_id="LOT_CONTROL",
            recipe="RCP_A",
        ),
        _typed_evidence(
            "EV_CONTROL_NORMAL",
            EvidenceType.NEGATIVE_SIGNAL.value,
            lot_id="LOT_CONTROL",
            recipe="RCP_A",
            parameter="oxygen_flow",
        ),
    ]

    result = evaluate_impact_lot_gate(
        source_lot_id="LOT_SOURCE",
        candidate=candidate,
        evidence=items,
        observed_impact_lots=["LOT_B", "LOT_CONTROL"],
        authoritative_conclusion_status="supported",
        semantic_profile=profile,
    )

    assert result["confirmed_impact_lots"] == ["LOT_B"]
    assert result["candidate_impact_lots"] == ["LOT_B"]
    control = next(row for row in result["rows"] if row["lot_id"] == "LOT_CONTROL")
    assert control["included"] is False
    assert control["checks"]["outcome"] is False


def test_broad_scope_projects_parameter_without_relabeling_lot_evidence() -> None:
    profile = _chamber_profile()
    candidate = CausalHypothesis(
        root_cause=(
            "EQ_A_CH01 high oxygen_flow during operation 2500 causes high "
            "contact resistance across recipes"
        ),
        causal_explanation=(
            "A chamber-wide oxygen-flow excursion raises contact resistance."
        ),
        supporting_evidence_ids=("EV_PARAMETER_A", "EV_OUTCOME_B"),
    )
    items = [
        _typed_evidence(
            "EV_WINDOW_A",
            EvidenceType.EXCURSION_WINDOW.value,
            lot_id="LOT_A",
            recipe="RCP_A",
            window=True,
        ),
        _typed_evidence(
            "EV_PARAMETER_A",
            EvidenceType.PARAMETER_DEVIATION.value,
            lot_id="LOT_A",
            recipe="RCP_A",
            parameter="oxygen_flow",
        ),
        _typed_evidence(
            "EV_EXPOSURE_B",
            EvidenceType.EQUIPMENT_EXPOSURE.value,
            lot_id="LOT_B",
            recipe="RCP_B",
            timestamp="2026-01-01T01:30:00+00:00",
        ),
        _typed_evidence(
            "EV_OUTCOME_B",
            EvidenceType.METROLOGY_DEVIATION.value,
            lot_id="LOT_B",
            recipe="RCP_B",
            outcome="contact_resistance",
            timestamp="2026-01-01T03:00:00+00:00",
        ),
        _typed_evidence(
            "EV_EXPOSURE_CONTROL",
            EvidenceType.EQUIPMENT_EXPOSURE.value,
            lot_id="LOT_CONTROL",
            recipe="RCP_A",
            timestamp="2026-01-01T01:35:00+00:00",
        ),
        _typed_evidence(
            "EV_CONTROL_NORMAL",
            EvidenceType.NEGATIVE_SIGNAL.value,
            lot_id="LOT_CONTROL",
            recipe="RCP_A",
            parameter="oxygen_flow",
        ),
    ]

    result = evaluate_impact_lot_gate(
        source_lot_id="LOT_SOURCE",
        candidate=candidate,
        evidence=items,
        observed_impact_lots=["LOT_B", "LOT_CONTROL"],
        authoritative_conclusion_status="supported",
        semantic_profile=profile,
    )

    rows = {row["lot_id"]: row for row in result["rows"]}
    assert result["confirmed_impact_lots"] == ["LOT_B"]
    assert rows["LOT_B"]["parameter_scope_projection_used"] is True
    assert rows["LOT_B"]["checks"]["parameter"] is True
    assert rows["LOT_B"]["checks"]["exposure_temporal"] is True
    assert "EV_PARAMETER_A" in rows["LOT_B"]["supporting_evidence_ids"]
    assert rows["LOT_CONTROL"]["included"] is False
    assert rows["LOT_CONTROL"]["checks"]["outcome"] is False


def test_parameter_projection_requires_broad_claimed_scope() -> None:
    profile = CandidateSemanticProfile(
        candidate_id="REQ_SCOPE:llm:recipe",
        scope_relation="focal_only",
        claimed_scope_kind="recipe",
        claimed_operation="2500",
        claimed_equipment="EQ_A",
        claimed_chamber="EQ_A_CH01",
        claimed_recipe="RCP_B",
        claimed_lane_ids=(LANE_B,),
        comparison_lane_ids=(LANE_A, LANE_B),
        mechanism_claim="Oxygen-flow instability leaves residue.",
        primary_mechanism="incomplete removal leaves residue",
        mechanism_relation="reference",
    )
    candidate = CausalHypothesis(
        root_cause="RCP_B high oxygen_flow causes high contact resistance",
        causal_explanation="High oxygen flow leaves residue.",
        supporting_evidence_ids=("EV_PARAMETER_A", "EV_OUTCOME_B"),
    )
    items = [
        _typed_evidence(
            "EV_WINDOW_A",
            EvidenceType.EXCURSION_WINDOW.value,
            lot_id="LOT_A",
            recipe="RCP_A",
            window=True,
        ),
        _typed_evidence(
            "EV_PARAMETER_A",
            EvidenceType.PARAMETER_DEVIATION.value,
            lot_id="LOT_A",
            recipe="RCP_A",
            parameter="oxygen_flow",
        ),
        _typed_evidence(
            "EV_EXPOSURE_B",
            EvidenceType.EQUIPMENT_EXPOSURE.value,
            lot_id="LOT_B",
            recipe="RCP_B",
            timestamp="2026-01-01T01:30:00+00:00",
        ),
        _typed_evidence(
            "EV_OUTCOME_B",
            EvidenceType.METROLOGY_DEVIATION.value,
            lot_id="LOT_B",
            recipe="RCP_B",
            outcome="contact_resistance",
        ),
    ]

    result = evaluate_impact_lot_gate(
        source_lot_id="LOT_SOURCE",
        candidate=candidate,
        evidence=items,
        observed_impact_lots=["LOT_B"],
        authoritative_conclusion_status="supported",
        semantic_profile=profile,
    )

    row = result["rows"][0]
    assert row["included"] is False
    assert row["parameter_scope_projection_used"] is False
    assert row["checks"]["parameter"] is False


def test_broad_projection_requires_lot_exposure_to_overlap_excursion() -> None:
    profile = _chamber_profile()
    candidate = CausalHypothesis(
        root_cause=(
            "EQ_A_CH01 high oxygen_flow during operation 2500 causes high "
            "contact resistance"
        ),
        causal_explanation="High oxygen flow raises contact resistance.",
        supporting_evidence_ids=("EV_PARAMETER_A", "EV_OUTCOME_B"),
    )
    items = [
        _typed_evidence(
            "EV_WINDOW_A",
            EvidenceType.EXCURSION_WINDOW.value,
            lot_id="LOT_A",
            recipe="RCP_A",
            window=True,
        ),
        _typed_evidence(
            "EV_PARAMETER_A",
            EvidenceType.PARAMETER_DEVIATION.value,
            lot_id="LOT_A",
            recipe="RCP_A",
            parameter="oxygen_flow",
        ),
        _typed_evidence(
            "EV_EXPOSURE_B",
            EvidenceType.EQUIPMENT_EXPOSURE.value,
            lot_id="LOT_B",
            recipe="RCP_B",
            timestamp="2026-01-01T03:00:00+00:00",
            metadata_overrides={
                "time_window": [
                    "2026-01-01T02:30:00+00:00",
                    "2026-01-01T03:00:00+00:00",
                ]
            },
        ),
        _typed_evidence(
            "EV_OUTCOME_B",
            EvidenceType.METROLOGY_DEVIATION.value,
            lot_id="LOT_B",
            recipe="RCP_B",
            outcome="contact_resistance",
        ),
    ]

    result = evaluate_impact_lot_gate(
        source_lot_id="LOT_SOURCE",
        candidate=candidate,
        evidence=items,
        observed_impact_lots=["LOT_B"],
        authoritative_conclusion_status="supported",
        semantic_profile=profile,
    )

    row = result["rows"][0]
    assert row["included"] is False
    assert row["parameter_scope_projection_used"] is True
    assert row["checks"]["exposure_temporal"] is False


def test_excursion_window_cannot_be_an_unexplained_precursor() -> None:
    window = _typed_evidence(
        "EV_WINDOW",
        EvidenceType.EXCURSION_WINDOW.value,
        lot_id="LOT_A",
        recipe="RCP_A",
        window=True,
    )
    payload = {
        "candidate_id": "C1",
        "alternative_candidate_id": None,
        "evidence_probe_lane_id": LANE_A,
        "challenge_kind": "mechanism",
        "mechanism_relation": "unknown",
        "strongest_alternative_lane_id": LANE_A,
        "supporting_evidence_ids": [],
        "contradicting_evidence_ids": [],
        "unexplained_precursor_evidence_ids": ["EV_WINDOW"],
        "distinguishing_gap_ids": [],
        "distinguishing_questions": [],
        "challenge_explanation": "The temporal boundary is not a precursor.",
        "status": "unresolved",
    }

    with pytest.raises(LLMOutputValidationError, match="typed causal/process"):
        _normalize_challenge_payload(
            payload,
            candidate_ids={"C1"},
            lane_ids={LANE_A},
            evidence_ids={"EV_WINDOW"},
            gap_ids=set(),
            evidence_by_id={"EV_WINDOW": window},
        )


def test_old_candidate_semantic_profile_defaults_to_lane_scope() -> None:
    old_payload = {
        "candidate_id": "REQ_OLD:llm:1",
        "scope_relation": "focal_only",
        "claimed_lane_ids": [LANE_A],
        "comparison_lane_ids": [LANE_A, LANE_B],
        "mechanism_claim": "Old State mechanism",
        "primary_mechanism": "Old State mechanism",
        "effect_modifier": None,
        "depends_on_candidate_id": None,
        "mechanism_relation": "reference",
        "distinguishing_predictions": [],
        "source": "qwen",
    }

    profile = CandidateSemanticProfile.from_dict(old_payload)

    assert profile.claimed_scope_kind == "lane"
    assert profile.claimed_operation is None
    assert CandidateSemanticProfile.from_dict(profile.to_dict()) == profile


def test_explicit_incident_inspection_span_is_typed_without_proving_causation() -> None:
    query = (
        "Investigate LOT_A: resistance tail with surface-residue inspection on "
        "matching wafers. The detection point is an observation, not a cause."
    )
    incident = build_incident_observation_evidence(
        RCAJob(
            job_id="JOB_INCIDENT_OBSERVATION",
            user_query=query,
            investigation_mode="lot",
            source_lot_id="LOT_A",
        )
    )

    assert len(incident) == 1
    observation = incident[0]
    provenance = observation.metadata["observation_role_provenance"]
    assert provenance["source_quote"] == "surface-residue"
    assert query[provenance["quote_start"] : provenance["quote_end"]] == (
        provenance["source_quote"]
    )
    assert observation.metadata["causal_status"] == (
        "observed_not_confirmed_cause"
    )

    parameter = _typed_evidence(
        "EV_PARAMETER_INCIDENT",
        EvidenceType.PARAMETER_DEVIATION.value,
        lot_id="LOT_A",
        recipe="RCP_A",
        parameter="oxygen_flow",
    )
    outcome = _typed_evidence(
        "EV_OUTCOME_INCIDENT",
        EvidenceType.METROLOGY_DEVIATION.value,
        lot_id="LOT_A",
        recipe="RCP_A",
        outcome="contact_resistance",
    )
    hypothesis = CausalHypothesis(
        root_cause="High oxygen_flow causes surface residue and contact resistance",
        causal_explanation=(
            "High oxygen flow changes removal chemistry, leaves surface residue, "
            "and raises contact resistance."
        ),
        supporting_evidence_ids=(
            parameter.evidence_id,
            outcome.evidence_id,
            observation.evidence_id,
        ),
    )

    matrix = build_causal_evidence_matrix(
        hypothesis,
        [parameter, outcome, observation],
    )

    assert matrix.mechanism_status == CausalClaimStatus.PLAUSIBLE.value
    assert matrix.mechanism_support_source == "observed_intermediate"
    assert matrix.claims["mechanism"].facts[
        "observed_intermediate_evidence_ids"
    ] == [observation.evidence_id]
    serialized = json.dumps(matrix.to_dict(), ensure_ascii=False)
    assert '"source_quote": "surface-residue"' in serialized


def test_confirmed_typed_mechanism_intermediate_can_support_bridge() -> None:
    parameter = _typed_evidence(
        "EV_PARAMETER_CONFIRMED_INTERMEDIATE",
        EvidenceType.PARAMETER_DEVIATION.value,
        lot_id="LOT_A",
        recipe="RCP_A",
        parameter="oxygen_flow",
    )
    outcome = _typed_evidence(
        "EV_OUTCOME_CONFIRMED_INTERMEDIATE",
        EvidenceType.METROLOGY_DEVIATION.value,
        lot_id="LOT_A",
        recipe="RCP_A",
        outcome="contact_resistance",
    )
    intermediate = _typed_evidence(
        "EV_CONFIRMED_INTERMEDIATE",
        EvidenceType.DEFECT_SIGNAL.value,
        lot_id="LOT_A",
        recipe="RCP_A",
        outcome="surface_residue",
        metadata_overrides={
            "causal_role": "physical_intermediate",
            "causal_status": "confirmed_causal_intermediate",
        },
    )
    hypothesis = CausalHypothesis(
        root_cause="High oxygen_flow causes surface residue and contact resistance",
        causal_explanation=(
            "High oxygen flow changes removal chemistry, leaves surface residue, "
            "and raises contact resistance."
        ),
        supporting_evidence_ids=(
            parameter.evidence_id,
            outcome.evidence_id,
            intermediate.evidence_id,
        ),
    )

    matrix = build_causal_evidence_matrix(
        hypothesis,
        [parameter, outcome, intermediate],
    )

    assert matrix.mechanism_status == CausalClaimStatus.SUPPORTED.value
    assert matrix.mechanism_support_source == "observed_intermediate"


def test_incident_query_without_explicit_observation_does_not_create_evidence() -> None:
    assert build_incident_observation_evidence(
        RCAJob(
            job_id="JOB_NO_INCIDENT_OBSERVATION",
            user_query="Investigate LOT_A and identify a possible cause.",
            investigation_mode="lot",
            source_lot_id="LOT_A",
        )
    ) == []
