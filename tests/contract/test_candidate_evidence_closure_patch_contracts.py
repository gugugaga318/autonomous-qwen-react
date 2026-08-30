from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from yield_rca_core.causal_evidence_matrix import build_causal_evidence_matrix
from yield_rca_core.causal_hypothesis import CausalClaimStatus, CausalHypothesis
from yield_rca_core.causal_investigation_models import CandidateSemanticProfile
from yield_rca_core.evidence_models import (
    EVIDENCE_SCHEMA_VERSION,
    EntityType,
    Evidence,
    EvidenceEntity,
    EvidenceSourceType,
    EvidenceType,
)
from yield_rca_core.evidence_synthesis import compact_evidence_prompt_card
from yield_rca_core.hypothesis_candidate_generator import (
    HypothesisCandidateProposal,
    QwenHypothesisCandidateGenerator,
    _candidate_evidence_closure_assessment,
)
from yield_rca_core.llm_gateway import FakeLLMClient, LLMRequest, LLMResponse
from yield_rca_core.models import AgentFinding, AgentKind, RCAState, ToolInput
from yield_rca_core.repositories import Row
from yield_rca_core.tool_layer import SummarizeDefectWatTool

LANE_A = "lane:2500:EQ_A:EQ_A_CH01:RCP_A"
LANE_B = "lane:2500:EQ_A:EQ_A_CH01:RCP_B"


def evidence(
    evidence_id: str,
    evidence_type: str,
    entity_type: str,
    entity_id: str,
    *,
    lane_id: str | None = None,
    lot_id: str = "LOT_A",
    causal_role: str | None = None,
) -> Evidence:
    metadata: dict[str, Any] = {}
    if lane_id is not None:
        metadata["lane_id"] = lane_id
    if causal_role is not None:
        metadata["causal_role"] = causal_role
        metadata["causal_role_provenance"] = {
            "source_table": "inspection_result",
            "source_fields": ["causal_role"],
            "source_row_count": 1,
        }
    return Evidence(
        evidence_id=evidence_id,
        source_type=EvidenceSourceType.ANALYTICS.value,
        source_id=f"SOURCE_{evidence_id}",
        summary=f"Observed {entity_id}",
        source_table="inspection_result",
        source_field="measured_value",
        evidence_type=evidence_type,
        source_agent=AgentKind.RCA_REASONING.value,
        source_tool="contract_fixture",
        observation=f"Observed {entity_id}",
        entities=[
            EvidenceEntity(EntityType.LOT.value, lot_id),
            EvidenceEntity(entity_type, entity_id),
        ],
        metadata=metadata,
        confidence=0.95,
        evidence_schema_version=EVIDENCE_SCHEMA_VERSION,
    )


def lane(lane_id: str, recipe: str, lot_id: str = "LOT_A") -> dict[str, Any]:
    return {
        "lane_id": lane_id,
        "operation": "2500",
        "equipment": "EQ_A",
        "chamber": "EQ_A_CH01",
        "recipe": recipe,
        "exposed_lot_ids": [lot_id],
        "parameter_scope": ["oxygen_flow_delta"],
    }


def semantic_profile(
    *,
    claimed_lane_ids: list[str],
    comparison_lane_ids: list[str],
) -> dict[str, Any]:
    relation = "focal_only" if len(comparison_lane_ids) > 1 else "unresolved"
    return {
        "candidate_index": 0,
        "claimed_scope": {
            "scope_relation": relation,
            "lane_ids": claimed_lane_ids,
        },
        "comparison_scope": {"lane_ids": comparison_lane_ids},
        "mechanism_claim": "Oxygen-flow instability leaves a physical residue.",
        "primary_mechanism": "incomplete removal leaves residue",
        "effect_modifier": None,
        "depends_on_candidate_index": None,
        "mechanism_relation": "reference",
        "distinguishing_predictions": [
            {
                "discriminator_kind": "parameter_anomaly",
                "lane_ids": comparison_lane_ids,
                "prediction": "The claimed Lane has an oxygen-flow deviation.",
            }
        ],
    }


def candidate(supporting_ids: list[str]) -> dict[str, Any]:
    return {
        "root_cause": (
            "EQ_A_CH01 oxygen-flow instability during operation 2500 leaves "
            "residue and produces HIGH_CONTACT_R"
        ),
        "causal_explanation": (
            "The oxygen-flow deviation causes incomplete removal, leaves residue, "
            "and increases contact resistance."
        ),
        "supporting_evidence_ids": supporting_ids,
        "contradicting_evidence_ids": [],
    }


class ScriptedCandidateClient(FakeLLMClient):
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.requests: list[LLMRequest] = []

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        base = super().complete_json(request)
        return LLMResponse(data=self.responses.pop(0), usage=base.usage)


class PhysicalIntermediateRepository:
    def __init__(self, *, complete_role_metadata: bool) -> None:
        second_role = "physical_intermediate" if complete_role_metadata else ""
        self.tables: dict[str, list[Row]] = {
            "defect_summary": [
                {
                    "lot_id": "LOT_A",
                    "wafer_id": "LOT_A_W01",
                    "defect_type": "polymer_residue",
                    "pattern_type": "contact_opening",
                    "inspected_at": "2026-01-01T02:00:00+00:00",
                    "causal_role": "physical_intermediate",
                },
                {
                    "lot_id": "LOT_A",
                    "wafer_id": "LOT_A_W02",
                    "defect_type": "polymer_residue",
                    "pattern_type": "contact_opening",
                    "inspected_at": "2026-01-01T02:01:00+00:00",
                    "causal_role": second_role,
                },
            ],
            "wat_result": [],
            "metrology_result": [],
        }

    def rows(self, table_name: str) -> list[Row]:
        return [dict(item) for item in self.tables[table_name]]


def finding(items: list[Evidence]) -> AgentFinding:
    return AgentFinding(
        finding_id="FINDING_CLOSURE",
        agent=AgentKind.RCA_REASONING.value,
        summary="Typed Lane Evidence is available.",
        confidence=0.95,
        evidence_ids=[item.evidence_id for item in items],
        evidence=items,
    )


def response(
    supporting_ids: list[str],
    *,
    claimed_lane_ids: list[str] | None = None,
    comparison_lane_ids: list[str] | None = None,
) -> dict[str, Any]:
    claimed = claimed_lane_ids or [LANE_A]
    comparison = comparison_lane_ids or claimed
    return {
        "candidates": [candidate(supporting_ids)],
        "candidate_semantic_profiles": [
            semantic_profile(
                claimed_lane_ids=claimed,
                comparison_lane_ids=comparison,
            )
        ],
        "analysis_summary": "One evidence-bounded candidate is retained.",
    }


def closure_evidence() -> list[Evidence]:
    return [
        evidence(
            "EV_EXPOSURE_A",
            EvidenceType.EQUIPMENT_EXPOSURE.value,
            EntityType.EQUIPMENT.value,
            "EQ_A",
            lane_id=LANE_A,
        ),
        evidence(
            "EV_WINDOW_A",
            EvidenceType.EXCURSION_WINDOW.value,
            EntityType.EXCURSION.value,
            "EXC_A",
            lane_id=LANE_A,
        ),
        evidence(
            "EV_PROCESS_A",
            EvidenceType.PARAMETER_DEVIATION.value,
            EntityType.PARAMETER.value,
            "oxygen_flow_delta",
            lane_id=LANE_A,
        ),
        evidence(
            "EV_OUTCOME_A",
            EvidenceType.ELECTRICAL_FAILURE.value,
            EntityType.WAT_ITEM.value,
            "HIGH_CONTACT_R",
            lane_id=LANE_A,
        ),
    ]


def test_missing_available_lane_endpoints_trigger_one_qwen_repair() -> None:
    items = closure_evidence()
    first_support = ["EV_PROCESS_A"]
    repaired_support = [
        "EV_EXPOSURE_A",
        "EV_WINDOW_A",
        "EV_PROCESS_A",
        "EV_OUTCOME_A",
    ]
    client = ScriptedCandidateClient(
        [response(first_support), response(repaired_support)]
    )

    result = QwenHypothesisCandidateGenerator(client).generate(
        request_id="REQ_CLOSURE_REPAIR",
        findings=[finding(items)],
        causal_lanes=[lane(LANE_A, "RCP_A")],
    )

    assert len(client.requests) == 2
    feedback = client.requests[1].payload["previous_validation_feedback"]
    gaps = feedback["candidate_evidence_closure"][0]["closure_gaps"]
    assert {item["evidence_role"] for item in gaps} == {
        "exposure",
        "temporal",
        "outcome",
    }
    assert result.candidates[0].supporting_evidence_ids == tuple(repaired_support)
    assert result.candidate_evidence_closure[0]["status"] == "complete"
    assert result.candidate_evidence_closure[0][
        "python_mutated_supporting_evidence_ids"
    ] is False
    assert len(result.candidate_evidence_closure_history) == 2


def test_closure_repair_exposes_typed_mechanism_intermediate_with_all_endpoints(
) -> None:
    intermediate = evidence(
        "EV_INTERMEDIATE_A",
        EvidenceType.DEFECT_SIGNAL.value,
        EntityType.DEFECT.value,
        "polymer_residue",
        lane_id=LANE_A,
        causal_role="physical_intermediate",
    )
    items = [*closure_evidence(), intermediate]
    repaired_support = [
        "EV_EXPOSURE_A",
        "EV_WINDOW_A",
        "EV_PROCESS_A",
        "EV_OUTCOME_A",
        "EV_INTERMEDIATE_A",
    ]
    client = ScriptedCandidateClient(
        [response(["EV_PROCESS_A"]), response(repaired_support)]
    )

    result = QwenHypothesisCandidateGenerator(client).generate(
        request_id="REQ_CLOSURE_INTERMEDIATE",
        findings=[finding(items)],
        causal_lanes=[lane(LANE_A, "RCP_A")],
    )

    feedback = client.requests[1].payload["previous_validation_feedback"]
    assessment = feedback["candidate_evidence_closure"][0]
    assert assessment["must_preserve_evidence_ids"] == ["EV_PROCESS_A"]
    assert assessment["candidate_snapshot"]["supporting_evidence_ids"] == [
        "EV_PROCESS_A"
    ]
    assert {item["evidence_role"] for item in assessment["closure_gaps"]} == {
        "exposure",
        "temporal",
        "outcome",
        "mechanism_intermediate",
    }
    assert result.candidates[0].supporting_evidence_ids == tuple(
        repaired_support
    )
    assert result.candidate_evidence_closure[0]["status"] == "complete"


def test_closure_repair_detects_citation_regression_without_python_mutation(
) -> None:
    intermediate = evidence(
        "EV_INTERMEDIATE_A",
        EvidenceType.DEFECT_SIGNAL.value,
        EntityType.DEFECT.value,
        "polymer_residue",
        lane_id=LANE_A,
        causal_role="physical_intermediate",
    )
    items = [*closure_evidence(), intermediate]
    first_support = ["EV_PROCESS_A", "EV_INTERMEDIATE_A"]
    regressed_support = [
        "EV_EXPOSURE_A",
        "EV_WINDOW_A",
        "EV_PROCESS_A",
        "EV_OUTCOME_A",
    ]
    client = ScriptedCandidateClient(
        [response(first_support), response(regressed_support)]
    )

    result = QwenHypothesisCandidateGenerator(client).generate(
        request_id="REQ_CITATION_REGRESSION",
        findings=[finding(items)],
        causal_lanes=[lane(LANE_A, "RCP_A")],
    )

    final_assessment = result.candidate_evidence_closure[0]
    assert final_assessment["status"] == "incomplete"
    assert final_assessment["citation_regression_evidence_ids"] == [
        "EV_INTERMEDIATE_A"
    ]
    assert any(
        item["evidence_role"] == "citation_regression"
        for item in final_assessment["closure_gaps"]
    )
    assert result.candidates[0].supporting_evidence_ids == tuple(
        regressed_support
    )
    assert final_assessment["python_mutated_supporting_evidence_ids"] is False
    assert result.evidence_closure_repair_exhausted is True


def test_semantic_and_closure_failures_share_the_same_bounded_repair() -> None:
    items = closure_evidence()
    invalid_semantics = response(["EV_PROCESS_A"])
    invalid_semantics["candidate_semantic_profiles"][0]["claimed_scope"][
        "scope_relation"
    ] = "shared_effect"
    repaired_support = [
        "EV_EXPOSURE_A",
        "EV_WINDOW_A",
        "EV_PROCESS_A",
        "EV_OUTCOME_A",
    ]
    client = ScriptedCandidateClient(
        [invalid_semantics, response(repaired_support)]
    )

    result = QwenHypothesisCandidateGenerator(client).generate(
        request_id="REQ_JOINT_REPAIR",
        findings=[finding(items)],
        causal_lanes=[lane(LANE_A, "RCP_A")],
    )

    assert len(client.requests) == 2
    feedback = client.requests[1].payload["previous_validation_feedback"]
    assert feedback["candidate_evidence_closure"][0]["status"] == "incomplete"
    assert "semantic profile also requires repair" in feedback["message"]
    first_assessment = result.candidate_evidence_closure_history[0][
        "candidate_assessments"
    ][0]
    assert first_assessment["scope_reference_source"] == (
        "raw_structurally_valid_scope_reference"
    )
    assert first_assessment["status"] == "incomplete"
    assert result.candidate_evidence_closure[0]["status"] == "complete"
    assert result.evidence_closure_repair_attempted is True
    assert result.evidence_closure_repair_exhausted is False


def test_raw_broad_scope_closure_survives_unrelated_prediction_error() -> None:
    window_b = evidence(
        "EV_WINDOW_B",
        EvidenceType.EXCURSION_WINDOW.value,
        EntityType.EXCURSION.value,
        "EXC_B",
        lane_id=LANE_B,
        lot_id="LOT_B",
    )
    items = [*closure_evidence(), window_b]
    first = response(
        ["EV_PROCESS_A"],
        claimed_lane_ids=[LANE_A],
        comparison_lane_ids=[LANE_A, LANE_B],
    )
    repaired = response(
        ["EV_EXPOSURE_A", "EV_PROCESS_A", "EV_OUTCOME_A", "EV_WINDOW_B"],
        claimed_lane_ids=[LANE_A],
        comparison_lane_ids=[LANE_A, LANE_B],
    )
    for payload in (first, repaired):
        claimed_scope = payload["candidate_semantic_profiles"][0][
            "claimed_scope"
        ]
        claimed_scope.update(
            {
                "scope_relation": "shared_effect",
                "scope_kind": "chamber",
                "operation": "2500",
                "equipment": "EQ_A",
                "chamber": "EQ_A_CH01",
                "recipe": None,
            }
        )
    first["candidate_semantic_profiles"][0]["distinguishing_predictions"][0][
        "lane_ids"
    ] = [LANE_A]
    client = ScriptedCandidateClient([first, repaired])

    result = QwenHypothesisCandidateGenerator(client).generate(
        request_id="REQ_RAW_BROAD_SCOPE_CLOSURE",
        findings=[finding(items)],
        causal_lanes=[lane(LANE_A, "RCP_A"), lane(LANE_B, "RCP_B", "LOT_B")],
    )

    first_assessment = result.candidate_evidence_closure_history[0][
        "candidate_assessments"
    ][0]
    assert first_assessment["scope_reference_source"] == (
        "raw_structurally_valid_scope_reference"
    )
    assert first_assessment["claimed_scope_kind"] == "chamber"
    assert first_assessment["matching_claimed_scope_lane_ids"] == [
        LANE_A,
        LANE_B,
    ]
    temporal_gap = next(
        item
        for item in first_assessment["closure_gaps"]
        if item["evidence_role"] == "temporal"
    )
    assert "EV_WINDOW_B" in temporal_gap["eligible_evidence_ids"]
    assert result.candidate_evidence_closure[0]["status"] == "complete"


def test_joint_repair_payload_stays_below_the_governed_target_with_large_evidence() -> None:
    items = closure_evidence()
    items.extend(
        evidence(
            f"EV_PROCESS_EXTRA_{index:03d}",
            EvidenceType.PARAMETER_DEVIATION.value,
            EntityType.PARAMETER.value,
            f"process_parameter_{index:03d}",
            lane_id=LANE_A,
        )
        for index in range(122)
    )
    first = response(["EV_PROCESS_A"])
    first["candidate_semantic_profiles"][0]["claimed_scope"][
        "scope_relation"
    ] = "shared_effect"
    repaired_support = [
        "EV_EXPOSURE_A",
        "EV_WINDOW_A",
        "EV_PROCESS_A",
        "EV_OUTCOME_A",
    ]
    client = ScriptedCandidateClient([first, response(repaired_support)])

    result = QwenHypothesisCandidateGenerator(client).generate(
        request_id="REQ_LARGE_JOINT_REPAIR",
        findings=[finding(items)],
        causal_lanes=[lane(LANE_A, "RCP_A")],
        new_evidence_ids=[item.evidence_id for item in items],
    )

    assert result.candidate_evidence_closure[0]["status"] == "complete"
    assert len(client.requests) == 2
    second_payload = client.requests[1].payload
    second_payload_chars = len(
        json.dumps(
            second_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    )
    assert second_payload_chars <= 52_000
    assert len(second_payload["typed_evidence_register"]) <= 56
    closure_feedback = second_payload["previous_validation_feedback"][
        "candidate_evidence_closure"
    ][0]
    assert set(closure_feedback) == {
        "candidate_index",
        "candidate_id",
        "status",
        "candidate_snapshot",
        "must_preserve_evidence_ids",
        "citation_regression_evidence_ids",
        "closure_gaps",
    }
    assert all(
        set(gap) == {"lane_id", "evidence_role", "eligible_evidence_ids"}
        for gap in closure_feedback["closure_gaps"]
    )
    register_ids = {
        item["evidence_id"] for item in second_payload["typed_evidence_register"]
    }
    feedback = second_payload["previous_validation_feedback"]
    advertised_ids = {
        evidence_id
        for lane_ids in feedback[
            "eligible_supporting_evidence_ids_by_lane"
        ].values()
        for evidence_id in lane_ids
    }
    advertised_ids.update(feedback["source_agent_by_evidence_id"])
    advertised_ids.update(
        evidence_id
        for item in feedback["candidate_evidence_closure"]
        for evidence_id in item["must_preserve_evidence_ids"]
    )
    advertised_ids.update(
        evidence_id
        for item in feedback["candidate_evidence_closure"]
        for gap in item["closure_gaps"]
        for evidence_id in gap["eligible_evidence_ids"]
    )
    assert advertised_ids <= register_ids


def test_repair_feedback_never_advertises_evidence_trimmed_from_register(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "yield_rca_core.hypothesis_candidate_generator._TARGET_PROMPT_PAYLOAD_CHARS",
        9_000,
    )
    items = closure_evidence()
    items.extend(
        evidence(
            f"EV_TRIM_CANDIDATE_{index:03d}",
            EvidenceType.PARAMETER_DEVIATION.value,
            EntityType.PARAMETER.value,
            f"trim_parameter_{index:03d}",
            lane_id=LANE_A,
        )
        for index in range(50)
    )
    first = response(["EV_PROCESS_A"])
    first["candidate_semantic_profiles"][0]["claimed_scope"][
        "scope_relation"
    ] = "shared_effect"
    client = ScriptedCandidateClient(
        [
            first,
            response(
                [
                    "EV_EXPOSURE_A",
                    "EV_WINDOW_A",
                    "EV_PROCESS_A",
                    "EV_OUTCOME_A",
                ]
            ),
        ]
    )

    result = QwenHypothesisCandidateGenerator(client).generate(
        request_id="REQ_TRIMMED_FEEDBACK",
        findings=[finding(items)],
        causal_lanes=[lane(LANE_A, "RCP_A")],
        new_evidence_ids=[item.evidence_id for item in items],
    )

    assert result.candidate_output_invalid is False
    assert len(client.requests) == 2
    second_payload = client.requests[1].payload
    assert result.evidence_synthesis is not None
    assert len(second_payload["typed_evidence_register"]) < len(items)
    register_ids = {
        item["evidence_id"] for item in second_payload["typed_evidence_register"]
    }
    feedback = second_payload["previous_validation_feedback"]
    advertised_ids = {
        evidence_id
        for lane_ids in feedback[
            "eligible_supporting_evidence_ids_by_lane"
        ].values()
        for evidence_id in lane_ids
    }
    advertised_ids.update(feedback["source_agent_by_evidence_id"])
    advertised_ids.update(
        evidence_id
        for item in feedback["candidate_evidence_closure"]
        for evidence_id in item["must_preserve_evidence_ids"]
    )
    advertised_ids.update(
        evidence_id
        for item in feedback["candidate_evidence_closure"]
        for gap in item["closure_gaps"]
        for evidence_id in gap["eligible_evidence_ids"]
    )
    assert advertised_ids <= register_ids


def test_missing_scope_reference_is_not_reported_as_closure_complete() -> None:
    items = closure_evidence()
    without_profile = {
        "candidates": [candidate(["EV_PROCESS_A"])],
        "analysis_summary": "The semantic scope is unavailable.",
    }
    client = ScriptedCandidateClient([without_profile, without_profile])

    result = QwenHypothesisCandidateGenerator(client).generate(
        request_id="REQ_CLOSURE_NOT_EVALUATED",
        findings=[finding(items)],
        causal_lanes=[lane(LANE_A, "RCP_A")],
    )

    assert result.candidate_evidence_closure[0]["status"] == "not_evaluated"
    assert result.candidate_evidence_closure[0]["scope_reference_source"] == (
        "unavailable"
    )
    assert all(
        item["status"] != "complete"
        for item in result.candidate_evidence_closure_history
    )
    assert result.evidence_closure_repair_attempted is False


def test_exhausted_closure_repair_keeps_incomplete_candidate_without_fallback() -> None:
    items = closure_evidence()
    bounded_response = response(["EV_PROCESS_A"])
    client = ScriptedCandidateClient([bounded_response, bounded_response])

    result = QwenHypothesisCandidateGenerator(client).generate(
        request_id="REQ_CLOSURE_EXHAUSTED",
        findings=[finding(items)],
        causal_lanes=[lane(LANE_A, "RCP_A")],
    )

    assert len(result.candidates) == 1
    assert result.candidate_output_invalid is False
    assert result.candidate_evidence_closure[0]["status"] == "incomplete"
    assert result.evidence_closure_repair_exhausted is True
    assert result.evidence_closure_repair_attempted is True
    assert result.evidence_closure_repair_skipped_due_to_budget is False


def test_other_lane_evidence_does_not_create_a_claimed_lane_requirement() -> None:
    items = [
        evidence(
            "EV_PROCESS_A",
            EvidenceType.PARAMETER_DEVIATION.value,
            EntityType.PARAMETER.value,
            "oxygen_flow_delta",
            lane_id=LANE_A,
        ),
        evidence(
            "EV_EXPOSURE_B",
            EvidenceType.EQUIPMENT_EXPOSURE.value,
            EntityType.EQUIPMENT.value,
            "EQ_A",
            lane_id=LANE_B,
            lot_id="LOT_B",
        ),
    ]
    client = ScriptedCandidateClient([response(["EV_PROCESS_A"])])

    result = QwenHypothesisCandidateGenerator(client).generate(
        request_id="REQ_OTHER_LANE",
        findings=[finding(items)],
        causal_lanes=[lane(LANE_A, "RCP_A")],
    )

    assert len(client.requests) == 1
    assert result.candidate_evidence_closure[0]["status"] == "complete"


def test_narrow_claim_does_not_require_comparison_scope_citations() -> None:
    items = closure_evidence() + [
        evidence(
            "EV_EXPOSURE_B",
            EvidenceType.EQUIPMENT_EXPOSURE.value,
            EntityType.EQUIPMENT.value,
            "EQ_A",
            lane_id=LANE_B,
            lot_id="LOT_B",
        )
    ]
    support = [
        "EV_EXPOSURE_A",
        "EV_WINDOW_A",
        "EV_PROCESS_A",
        "EV_OUTCOME_A",
    ]
    bounded_response = response(
        support,
        claimed_lane_ids=[LANE_A],
        comparison_lane_ids=[LANE_A, LANE_B],
    )
    client = ScriptedCandidateClient([bounded_response, bounded_response])

    result = QwenHypothesisCandidateGenerator(client).generate(
        request_id="REQ_NARROW_SCOPE",
        findings=[finding(items)],
        causal_lanes=[lane(LANE_A, "RCP_A"), lane(LANE_B, "RCP_B", "LOT_B")],
    )

    assert result.candidate_evidence_closure[0]["status"] == "complete"
    assert not any(
        "candidate Evidence closure is incomplete" in error
        for error in result.validation_errors
    )
    assert "EV_EXPOSURE_B" not in result.candidates[0].supporting_evidence_ids


def test_explicit_sourced_intermediate_is_visible_and_supports_mechanism() -> None:
    items = closure_evidence()
    intermediate = evidence(
        "EV_INTERMEDIATE",
        EvidenceType.DEFECT_SIGNAL.value,
        EntityType.DEFECT.value,
        "polymer_residue",
        lane_id=LANE_A,
        causal_role="physical_intermediate",
    )
    items.append(intermediate)
    hypothesis = CausalHypothesis(
        root_cause=str(candidate([])["root_cause"]),
        causal_explanation=str(candidate([])["causal_explanation"]),
        supporting_evidence_ids=(
            "EV_EXPOSURE_A",
            "EV_PROCESS_A",
            "EV_OUTCOME_A",
            "EV_INTERMEDIATE",
        ),
    )

    matrix = build_causal_evidence_matrix(hypothesis, items)
    card = compact_evidence_prompt_card(intermediate)

    assert matrix.mechanism_status == CausalClaimStatus.SUPPORTED.value
    assert matrix.mechanism_support_source == "observed_intermediate"
    assert card["typed_details"]["causal_role"] == "physical_intermediate"
    assert card["typed_details"]["causal_role_provenance"]["source_table"] == (
        "inspection_result"
    )


def test_defect_tool_propagates_only_complete_source_declared_role() -> None:
    tool_input = ToolInput(
        tool_name="summarize_defect_wat",
        request_id="REQ_TYPED_INTERMEDIATE",
        parameters={"lot_ids": ["LOT_A"]},
        requested_by=AgentKind.DEFECT_WAT.value,
    )
    complete = SummarizeDefectWatTool(
        PhysicalIntermediateRepository(complete_role_metadata=True)
    ).run(tool_input)
    incomplete = SummarizeDefectWatTool(
        PhysicalIntermediateRepository(complete_role_metadata=False)
    ).run(tool_input)
    complete_defect = next(
        item
        for item in complete.evidence
        if item.evidence_type == EvidenceType.DEFECT_SIGNAL.value
    )
    incomplete_defect = next(
        item
        for item in incomplete.evidence
        if item.evidence_type == EvidenceType.DEFECT_SIGNAL.value
    )

    assert complete_defect.metadata["causal_role"] == "physical_intermediate"
    assert complete_defect.metadata["causal_role_provenance"] == {
        "source_table": "defect_summary",
        "source_fields": ("causal_role",),
        "source_row_count": 2,
    }
    assert "causal_role" not in incomplete_defect.metadata


def test_opaque_defect_and_qwen_bridge_text_remain_only_plausible() -> None:
    items = closure_evidence()
    items.append(
        evidence(
            "EV_OPAQUE_DEFECT",
            EvidenceType.DEFECT_SIGNAL.value,
            EntityType.DEFECT.value,
            "R39",
            lane_id=LANE_A,
        )
    )
    hypothesis = CausalHypothesis(
        root_cause=str(candidate([])["root_cause"]),
        causal_explanation=(
            "Qwen proposes that incomplete removal leaves polymer residue and "
            "raises contact resistance."
        ),
        supporting_evidence_ids=(
            "EV_EXPOSURE_A",
            "EV_PROCESS_A",
            "EV_OUTCOME_A",
            "EV_OPAQUE_DEFECT",
        ),
    )

    matrix = build_causal_evidence_matrix(hypothesis, items)

    assert matrix.mechanism_status == CausalClaimStatus.PLAUSIBLE.value
    assert matrix.mechanism_support_source == "empirical_convergence"
    assert matrix.claims["mechanism"].facts[
        "observed_intermediate_evidence_ids"
    ] == []


def test_formal_009_replay_detects_omitted_exposure_and_window_citations() -> None:
    replay_path = (
        Path(__file__).resolve().parents[2]
        / "outputs"
        / "patch4_mechanism_action_qwen_smoke_FORMAL009_r3"
        / "states"
        / "FORMAL_009.json"
    )
    if not replay_path.exists():
        pytest.skip("FORMAL_009 local replay State is not available")
    state = RCAState.from_dict(json.loads(replay_path.read_text(encoding="utf-8")))
    rca_finding = next(
        finding
        for finding in reversed(state.findings)
        if finding.agent == AgentKind.RCA_REASONING.value
    )
    engine_candidates = rca_finding.details["hypothesis_engine_result"]["candidates"]
    raw_profiles = rca_finding.details["candidate_semantic_profiles"]
    proposals = tuple(
        HypothesisCandidateProposal(
            root_cause=str(item["root_cause"]),
            causal_explanation=str(item["causal_explanation"]),
            supporting_evidence_ids=tuple(item["supporting_evidence_ids"]),
            contradicting_evidence_ids=tuple(item["contradicting_evidence_ids"]),
        )
        for item in engine_candidates
    )
    profiles = tuple(
        CandidateSemanticProfile.from_dict(dict(item)) for item in raw_profiles
    )

    assessments = _candidate_evidence_closure_assessment(
        proposals,
        profiles,
        evidence_by_id={item.evidence_id: item for item in state.evidence},
        causal_lanes=[item.to_dict() for item in state.causal_lanes],
    )

    assert assessments[0]["status"] == "incomplete"
    gaps = assessments[0]["closure_gaps"]
    assert "exposure" in {item["evidence_role"] for item in gaps}
    assert "temporal" in {item["evidence_role"] for item in gaps}
    assert any(
        "EV_FDC_EXCURSION_WINDOW" in item["eligible_evidence_ids"]
        for item in gaps
    )


def test_formal_009_joint_repair_feedback_only_advertises_registered_ids() -> None:
    replay_path = (
        Path(__file__).resolve().parents[2]
        / "outputs"
        / "candidate_evidence_closure_qwen_smoke_FORMAL009_r1"
        / "states"
        / "FORMAL_009.json"
    )
    if not replay_path.exists():
        pytest.skip("FORMAL_009 local replay State is not available")
    state = RCAState.from_dict(json.loads(replay_path.read_text(encoding="utf-8")))
    claimed_lane = next(
        item
        for item in state.causal_lanes
        if item.lane_id.endswith("RCP_800D7305")
    )
    comparison_lane = next(
        item
        for item in state.causal_lanes
        if item.lane_id.endswith("RCP_C5BEFB77")
    )
    process_id = "EV_FDC_ENDPOINT_OVERRUN_DELTA_LANE_5C900B28FA07906D"
    exposure_id = (
        "EV_MES_SHARED_LANE_LOT_1CC80F8B9A_2500_8_LANE_2500_"
        "EQ_44E1AC_EQ_44E1AC_CH03_RCP_800D7305"
    )
    temporal_id = "EV_FDC_EXCURSION_WINDOW"
    outcome_id = "EV_DEFECT_R39_SHARED_EXPOSURE_LANE_5C900B28FA07906D"

    def replay_response(
        supporting_ids: list[str],
        *,
        scope_relation: str,
    ) -> dict[str, Any]:
        return {
            "candidates": [
                {
                    "root_cause": (
                        "EQ_44E1AC_CH03 endpoint-control instability during "
                        "operation 2500 leaves residue and raises contact resistance"
                    ),
                    "causal_explanation": (
                        "Endpoint overrun causes incomplete removal, leaves a "
                        "physical residue, and increases contact resistance."
                    ),
                    "supporting_evidence_ids": supporting_ids,
                    "contradicting_evidence_ids": [],
                }
            ],
            "candidate_semantic_profiles": [
                {
                    "candidate_index": 0,
                    "claimed_scope": {
                        "scope_relation": scope_relation,
                        "lane_ids": [claimed_lane.lane_id],
                    },
                    "comparison_scope": {
                        "lane_ids": [
                            claimed_lane.lane_id,
                            comparison_lane.lane_id,
                        ]
                    },
                    "mechanism_claim": (
                        "Endpoint-control instability leaves physical residue."
                    ),
                    "primary_mechanism": "incomplete removal leaves residue",
                    "effect_modifier": None,
                    "depends_on_candidate_index": None,
                    "mechanism_relation": "reference",
                    "distinguishing_predictions": [
                        {
                            "discriminator_kind": "parameter_anomaly",
                            "lane_ids": [
                                claimed_lane.lane_id,
                                comparison_lane.lane_id,
                            ],
                            "prediction": (
                                "The claimed Lane has the stronger endpoint overrun."
                            ),
                        }
                    ],
                }
            ],
            "analysis_summary": "Retain the evidence-bounded mechanism candidate.",
        }

    client = ScriptedCandidateClient(
        [
            replay_response([process_id], scope_relation="shared_effect"),
            replay_response(
                [exposure_id, temporal_id, process_id, outcome_id],
                scope_relation="focal_only",
            ),
        ]
    )
    result = QwenHypothesisCandidateGenerator(client).generate(
        request_id="REQ_FORMAL_009_JOINT_REPAIR_REPLAY",
        findings=[
            finding
            for finding in state.findings
            if finding.agent != AgentKind.RCA_REASONING.value
        ],
        context_evidence=state.evidence,
        causal_lanes=[item.to_dict() for item in state.causal_lanes],
        new_evidence_ids=[item.evidence_id for item in state.evidence],
    )

    assert result.candidate_output_invalid is False
    assert len(client.requests) == 2
    second_payload = client.requests[1].payload
    register_ids = {
        item["evidence_id"] for item in second_payload["typed_evidence_register"]
    }
    feedback = second_payload["previous_validation_feedback"]
    advertised_ids = {
        evidence_id
        for lane_ids in feedback[
            "eligible_supporting_evidence_ids_by_lane"
        ].values()
        for evidence_id in lane_ids
    }
    advertised_ids.update(feedback["source_agent_by_evidence_id"])
    advertised_ids.update(
        evidence_id
        for item in feedback["candidate_evidence_closure"]
        for evidence_id in item["must_preserve_evidence_ids"]
    )
    advertised_ids.update(
        evidence_id
        for item in feedback["candidate_evidence_closure"]
        for gap in item["closure_gaps"]
        for evidence_id in gap["eligible_evidence_ids"]
    )
    assert advertised_ids <= register_ids
