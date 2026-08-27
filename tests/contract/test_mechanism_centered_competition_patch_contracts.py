from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from yield_rca_core.causal_competition import build_candidate_competition_brief
from yield_rca_core.causal_evidence_matrix import build_causal_evidence_matrix
from yield_rca_core.causal_hypothesis import CausalClaimStatus, CausalHypothesis
from yield_rca_core.causal_investigation_models import (
    CandidateCompetitionAxis,
    CausalLaneRecord,
    CompetitionRequirement,
)
from yield_rca_core.evidence_models import (
    EVIDENCE_SCHEMA_VERSION,
    EntityType,
    Evidence,
    EvidenceEntity,
    EvidenceSourceType,
    EvidenceType,
)
from yield_rca_core.evidence_synthesis import build_lane_first_evidence_synthesis
from yield_rca_core.hypothesis_candidate_generator import (
    QwenHypothesisCandidateGenerator,
)
from yield_rca_core.llm_gateway import FakeLLMClient, LLMRequest, LLMResponse
from yield_rca_core.models import AgentFinding, AgentKind, RCAState


def evidence(
    evidence_id: str,
    evidence_type: str,
    entity_type: str,
    entity_id: str,
    *,
    metadata: dict[str, Any] | None = None,
    observation: str | None = None,
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        source_type=EvidenceSourceType.ANALYTICS.value,
        source_id=f"SOURCE_{evidence_id}",
        summary=observation or evidence_id,
        evidence_type=evidence_type,
        source_agent=AgentKind.RCA_REASONING.value,
        source_tool="contract_fixture",
        observation=observation or evidence_id,
        entities=[
            EvidenceEntity(entity_type=EntityType.LOT.value, entity_id="LOT_A"),
            EvidenceEntity(entity_type=entity_type, entity_id=entity_id),
        ],
        metadata=metadata or {},
        confidence=0.9,
        evidence_schema_version=EVIDENCE_SCHEMA_VERSION,
    )


def matrix_evidence(*, include_intermediate: bool = False) -> list[Evidence]:
    items = [
        evidence(
            "EV_EXPOSURE",
            EvidenceType.IMPACT_SCOPE.value,
            EntityType.EQUIPMENT.value,
            "EQ_A",
        ),
        evidence(
            "EV_PROCESS",
            EvidenceType.PARAMETER_DEVIATION.value,
            EntityType.PARAMETER.value,
            "oxygen_flow_response_delta",
        ),
        evidence(
            "EV_OUTCOME",
            EvidenceType.ELECTRICAL_FAILURE.value,
            EntityType.WAT_ITEM.value,
            "HIGH_CONTACT_R",
        ),
    ]
    if include_intermediate:
        items.append(
            evidence(
                "EV_INTERMEDIATE",
                EvidenceType.DEFECT_SIGNAL.value,
                EntityType.DEFECT.value,
                "polymer_residue",
                metadata={"causal_role": "mechanism_intermediate"},
                observation="Observed polymer residue in the contact opening.",
            )
        )
    return items


def candidate(*, include_intermediate: bool = False) -> CausalHypothesis:
    return CausalHypothesis(
        root_cause=(
            "EQ_A strip/ash oxygen-flow response excursion leaves polymer residue "
            "and produces HIGH_CONTACT_R"
        ),
        causal_explanation=(
            "The oxygen-flow response excursion causes incomplete strip/ash removal, "
            "leaving polymer residue in the contact opening and increasing contact "
            "resistance."
        ),
        supporting_evidence_ids=(
            "EV_EXPOSURE",
            "EV_PROCESS",
            "EV_OUTCOME",
            *(("EV_INTERMEDIATE",) if include_intermediate else ()),
        ),
    )


def test_shared_lot_parameter_outcome_only_makes_mechanism_plausible() -> None:
    matrix = build_causal_evidence_matrix(candidate(), matrix_evidence())

    assert matrix.mechanism_status == CausalClaimStatus.PLAUSIBLE.value
    assert matrix.mechanism_support_source == "empirical_convergence"


def test_typed_physical_intermediate_can_support_open_world_mechanism() -> None:
    matrix = build_causal_evidence_matrix(
        candidate(include_intermediate=True),
        matrix_evidence(include_intermediate=True),
    )

    assert matrix.mechanism_status == CausalClaimStatus.SUPPORTED.value
    assert matrix.mechanism_support_source == "observed_intermediate"


def test_scope_variants_do_not_replace_root_cause_mechanism_competition() -> None:
    lane_a = {
        "lane_id": "LANE_A",
        "operation": "2500",
        "equipment": "EQ_A",
        "chamber": "EQ_A_CH01",
        "recipe": "RCP_A",
        "parameter_scope": ["oxygen_flow_response_delta"],
        "facts": {
            "shared_exposure": [{"evidence_id": "EV_EXP_A"}],
            "process_excursions": [{"evidence_id": "EV_PROC_A"}],
            "outcomes": [
                {
                    "evidence_id": "EV_OUT_A",
                    "evidence_type": "metrology_deviation",
                    "metadata": {
                        "metric_name": "contact resistance delta",
                        "fail_count": 22,
                        "row_count": 120,
                    },
                    "entities": [
                        {"entity_type": "lot", "entity_id": "LOT_A"},
                        {"entity_type": "lot", "entity_id": "LOT_B"},
                    ],
                }
            ],
        },
    }
    lane_b = {
        **lane_a,
        "lane_id": "LANE_B",
        "recipe": "RCP_B",
        "facts": {
            "shared_exposure": [{"evidence_id": "EV_EXP_B"}],
            "process_excursions": [{"evidence_id": "EV_PROC_B"}],
            "outcomes": [
                {
                    "evidence_id": "EV_OUT_B",
                    "evidence_type": "metrology_deviation",
                    "metadata": {
                        "metric_name": "contact resistance delta",
                        "fail_count": 12,
                        "row_count": 72,
                    },
                    "entities": [
                        {"entity_type": "lot", "entity_id": "LOT_A"},
                        {"entity_type": "lot", "entity_id": "LOT_C"},
                    ],
                }
            ],
        },
    }

    brief = build_candidate_competition_brief([lane_a, lane_b])

    assert brief["competition_requirement"] == CompetitionRequirement.MECHANISM_REQUIRED
    assert brief["competition_axes"] == [
        CandidateCompetitionAxis.MECHANISM.value,
        CandidateCompetitionAxis.SCOPE.value,
    ]
    assert brief["scope_assessment_required"] is True
    facts = brief["scope_groups"][0]["normalized_outcome_facts_by_lane"]
    assert facts["LANE_A"][0]["normalized_rate"] == 0.183333
    assert facts["LANE_B"][0]["normalized_rate"] == 0.166667


class RecordingCandidateClient(FakeLLMClient):
    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        base = super().complete_json(request)
        return LLMResponse(
            data={
                "candidates": [],
                "candidate_semantic_profiles": [],
                "analysis_summary": "No bounded candidate.",
            },
            usage=base.usage,
        )


def test_first_round_new_evidence_cannot_bypass_prompt_bound() -> None:
    evidence_items = [
        evidence(
            f"EV_{index:03d}",
            (
                EvidenceType.PARAMETER_DEVIATION.value
                if index % 2 == 0
                else EvidenceType.METROLOGY_DEVIATION.value
            ),
            (
                EntityType.PARAMETER.value
                if index % 2 == 0
                else EntityType.DEFECT.value
            ),
            f"ENTITY_{index:03d}",
        )
        for index in range(126)
    ]
    finding = AgentFinding(
        finding_id="FINDING_BOUND",
        agent=AgentKind.FDC.value,
        summary="Bounded fixture",
        confidence=0.9,
        evidence_ids=[item.evidence_id for item in evidence_items],
        evidence=evidence_items,
    )
    client = RecordingCandidateClient()

    result = QwenHypothesisCandidateGenerator(client).generate(
        request_id="REQ_BOUND",
        findings=[finding],
        new_evidence_ids=[item.evidence_id for item in evidence_items],
    )

    synthesis = result.evidence_synthesis
    assert synthesis is not None
    assert synthesis["prompt_evidence_count"] <= synthesis["prompt_evidence_limit"]
    assert synthesis["prompt_evidence_count"] < len(evidence_items)
    assert len(client.requests[0].payload["new_evidence_ids_since_prior"]) <= 16


def test_lane_round_trip_preserves_engineering_semantics() -> None:
    lane = CausalLaneRecord(
        lane_id="LANE_ASH",
        operation="2500",
        operation_name="Strip and ash",
        module="Dry Etch",
    )

    restored = CausalLaneRecord.from_dict(lane.to_dict())

    assert restored.operation_name == "Strip and ash"
    assert restored.module == "Dry Etch"


def test_formal_009_offline_replay_exposes_mechanism_axis_and_bounded_semantics() -> None:
    replay_path = (
        Path(__file__).resolve().parents[2]
        / "outputs"
        / "scope_competition_progression_qwen_smoke_FORMAL009_r2"
        / "states"
        / "FORMAL_009.json"
    )
    if not replay_path.exists():
        pytest.skip("FORMAL_009 local replay State is not available")
    state = RCAState.from_dict(json.loads(replay_path.read_text(encoding="utf-8")))
    raw_lanes = next(
        (
            finding.details["lane_candidates"]
            for finding in state.findings
            if finding.agent == AgentKind.MES.value
            and isinstance(finding.details.get("lane_candidates"), list)
            and finding.details["lane_candidates"]
        ),
        None,
    )
    if raw_lanes is None:
        pytest.skip("FORMAL_009 replay has no raw MES Lane candidates")

    synthesis = build_lane_first_evidence_synthesis(state.evidence, raw_lanes)
    lanes = synthesis["active_causal_lanes"]
    competition = synthesis["candidate_competition"]

    assert any(lane.get("operation_name") == "Strip and ash" for lane in lanes)
    assert any(lane.get("module") == "Dry Etch" for lane in lanes)
    assert competition["competition_requirement"] == "mechanism_required"
    assert competition["scope_assessment_required"] is True
    assert "mechanism" in competition["competition_axes"]
    assert "scope" in competition["competition_axes"]
    assert synthesis["prompt_evidence_count"] < synthesis["evidence_count"]
