from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core"))

from yield_rca_core.causal_adversarial import (  # noqa: E402
    _challenge_output_contract,
)
from yield_rca_core.causal_evidence_gap import (  # noqa: E402
    build_hypothesis_discrimination_gaps,
)
from yield_rca_core.causal_evidence_matrix import (  # noqa: E402
    CausalClaimResult,
    CausalEvidenceMatrix,
    compact_causal_evidence_matrix_for_prompt,
)
from yield_rca_core.causal_hypothesis import CausalHypothesis  # noqa: E402
from yield_rca_core.causal_investigation_models import (  # noqa: E402
    CandidateCompetitionStatus,
    CandidateDistinguishingPrediction,
    CandidateMechanismRelation,
    CandidateSemanticProfile,
    CompetitionRequirement,
    CompetitionTrace,
    ScopeAssessmentStatus,
)
from yield_rca_core.evidence_models import (  # noqa: E402
    EVIDENCE_SCHEMA_VERSION,
    AgentKind,
    EntityType,
    Evidence,
    EvidenceEntity,
    EvidenceSourceType,
    EvidenceType,
)
from yield_rca_core.hypothesis_candidate_generator import (  # noqa: E402
    HypothesisCandidateProposal,
    QwenHypothesisCandidateGenerator,
    _mechanism_semantics_are_distinct,
    _partition_root_candidates,
)
from yield_rca_core.investigation_decision import (  # noqa: E402
    assess_action_values,
    derive_investigation_gain_history,
)
from yield_rca_core.investigation_finalizer import (  # noqa: E402
    finalize_investigation,
    validate_terminal_investigation_state,
)
from yield_rca_core.investigation_models import ActionKind  # noqa: E402
from yield_rca_core.llm_gateway import FakeLLMClient, LLMRequest  # noqa: E402
from yield_rca_core.models import (  # noqa: E402
    AgentFinding,
    RCAJob,
    RCAState,
    TaskStatus,
)
from yield_rca_core.next_action_planner import (  # noqa: E402
    QwenNextActionPlanner,
    _authoritative_causal_gaps,
)

LANE_A = "lane:2500:EQ_A:EQ_A_CH01:RCP_A"
LANE_B = "lane:2500:EQ_A:EQ_A_CH01:RCP_B"


def _proposal(root: str, evidence_id: str) -> HypothesisCandidateProposal:
    return HypothesisCandidateProposal(
        root_cause=root,
        causal_explanation=f"{root} produces the observed outcome.",
        supporting_evidence_ids=(evidence_id,),
    )


def _profile(
    candidate_id: str,
    *,
    primary: str,
    relation: str,
    dependency: str | None = None,
    modifier: str | None = None,
    prediction: str = "The mechanism produces a distinct parameter signature.",
) -> CandidateSemanticProfile:
    return CandidateSemanticProfile(
        candidate_id=candidate_id,
        scope_relation="focal_only",
        claimed_lane_ids=(LANE_A,),
        comparison_lane_ids=(LANE_A, LANE_B),
        mechanism_claim=primary,
        primary_mechanism=primary,
        effect_modifier=modifier,
        depends_on_candidate_id=dependency,
        mechanism_relation=relation,
        distinguishing_predictions=(
            CandidateDistinguishingPrediction(
                discriminator_kind="parameter_anomaly",
                lane_ids=(LANE_A, LANE_B),
                prediction=prediction,
            ),
        ),
    )


def _matrix(root: str = "Reference mechanism") -> CausalEvidenceMatrix:
    return CausalEvidenceMatrix(
        candidate=CausalHypothesis(
            root_cause=root,
            causal_explanation=f"{root} explains the observed process outcome.",
            supporting_evidence_ids=("EV_PROCESS",),
        ),
        claims={},
    )


def _process_evidence() -> Evidence:
    return Evidence(
        evidence_id="EV_PROCESS",
        source_type=EvidenceSourceType.FDC.value,
        source_id="fdc:process",
        summary="Pressure response deviated from baseline.",
        metadata={
            "operation_no": "2500",
            "equipment_id": "EQ_A",
            "chamber_id": "EQ_A_CH01",
            "recipe_id": "RCP_A",
            "parameter_name": "pressure_response",
            "direction": "high",
            "raw_rows": list(range(100)),
        },
        evidence_type=EvidenceType.PARAMETER_DEVIATION.value,
        source_agent=AgentKind.FDC.value,
        source_tool=ActionKind.INSPECT_FDC_SPC.value,
        observation="Pressure response is above the historical baseline.",
        entities=(
            EvidenceEntity(EntityType.OPERATION.value, "2500"),
            EvidenceEntity(EntityType.EQUIPMENT.value, "EQ_A"),
            EvidenceEntity(EntityType.CHAMBER.value, "EQ_A_CH01"),
            EvidenceEntity(EntityType.RECIPE.value, "RCP_A"),
            EvidenceEntity(EntityType.PARAMETER.value, "pressure_response"),
        ),
        confidence=0.95,
        evidence_schema_version=EVIDENCE_SCHEMA_VERSION,
    )


def test_shared_primary_recipe_modifier_is_not_a_root_candidate_slot() -> None:
    reference = _profile(
        "REQ:llm:1",
        primary="Chamber wall conditioning degradation",
        relation=CandidateMechanismRelation.REFERENCE.value,
    )
    modifier = _profile(
        "REQ:llm:2",
        primary="Chamber wall conditioning degradation",
        relation=CandidateMechanismRelation.SHARED_PRIMARY_WITH_MODIFIER.value,
        dependency=reference.candidate_id,
        modifier="RCP_A has a narrower process window.",
    )

    proposals, profiles, variants = _partition_root_candidates(
        (
            _proposal("Chamber degradation", "EV_A"),
            _proposal("Recipe sensitivity to chamber degradation", "EV_B"),
        ),
        (reference, modifier),
        competition_requirement=CompetitionRequirement.MECHANISM_REQUIRED.value,
    )

    assert len(proposals) == len(profiles) == 1
    assert variants[0]["mechanism_relation"] == (
        CandidateMechanismRelation.SHARED_PRIMARY_WITH_MODIFIER.value
    )
    assert variants[0]["semantic_profile"]["effect_modifier"]


def test_independent_physical_initiator_remains_in_competition() -> None:
    reference = _profile(
        "REQ:llm:1",
        primary="Chamber wall conditioning degradation",
        relation=CandidateMechanismRelation.REFERENCE.value,
        prediction="Wall conditioning drift changes pressure response.",
    )
    independent = _profile(
        "REQ:llm:2",
        primary="RF matching network intermittent fault",
        relation=CandidateMechanismRelation.INDEPENDENT_ALTERNATIVE.value,
        prediction="RF network faults change source-power repeatability.",
    )

    proposals, profiles, variants = _partition_root_candidates(
        (_proposal("Wall degradation", "EV_A"), _proposal("RF fault", "EV_B")),
        (reference, independent),
        competition_requirement=CompetitionRequirement.MECHANISM_REQUIRED.value,
    )

    assert len(proposals) == len(profiles) == 2
    assert variants == ()
    assert _mechanism_semantics_are_distinct(reference, independent)


def test_missing_semantic_profiles_cannot_occupy_root_candidate_slots() -> None:
    proposals, profiles, variants = _partition_root_candidates(
        (_proposal("Wall degradation", "EV_A"), _proposal("RF fault", "EV_B")),
        (),
        competition_requirement=CompetitionRequirement.MECHANISM_REQUIRED.value,
        candidate_ids=("REQ:llm:1", "REQ:llm:2"),
        semantic_validation_errors=("candidate_semantic_profiles is missing",),
    )

    assert [item.root_cause for item in proposals] == ["Wall degradation"]
    assert profiles == ()
    assert variants == (
        {
            "candidate_index": 1,
            "candidate_id": "REQ:llm:2",
            "root_cause": "RF fault",
            "causal_explanation": "RF fault produces the observed outcome.",
            "supporting_evidence_ids": ["EV_B"],
            "contradicting_evidence_ids": [],
            "semantic_profile": None,
            "semantic_profile_status": "missing",
            "semantic_validation_errors": [
                "candidate_semantic_profiles is missing"
            ],
            "mechanism_relation": CandidateMechanismRelation.UNKNOWN.value,
            "depends_on_candidate_id": None,
            "isolation_reason": (
                "candidate has no valid semantic proof of an independent "
                "primary mechanism relative to the reference candidate"
            ),
        },
    )


def test_missing_second_profile_is_isolated_but_reference_survives() -> None:
    reference = _profile(
        "REQ:llm:1",
        primary="Chamber wall conditioning degradation",
        relation=CandidateMechanismRelation.REFERENCE.value,
    )
    proposals, profiles, variants = _partition_root_candidates(
        (_proposal("Wall degradation", "EV_A"), _proposal("RF fault", "EV_B")),
        (reference,),
        competition_requirement=CompetitionRequirement.MECHANISM_REQUIRED.value,
        candidate_ids=("REQ:llm:1", "REQ:llm:2"),
        semantic_validation_errors=(
            "candidate_semantic_profiles missing surviving candidate indexes: [1]",
        ),
    )

    assert len(proposals) == len(profiles) == 1
    assert profiles[0] == reference
    assert variants[0]["semantic_profile_status"] == "missing"
    assert variants[0]["mechanism_relation"] == CandidateMechanismRelation.UNKNOWN.value


def test_mechanism_challenge_contract_excludes_scope_discriminators() -> None:
    candidate_id = "REQ:llm:1"
    mechanism_gap = {
        "gap_id": "candidate_0.hypothesis_discrimination.mechanism_context",
        "gap_type": "hypothesis_discrimination",
        "candidate_id": candidate_id,
        "competition_axis": "mechanism",
        "applicable_lane_ids": [LANE_A],
        "information_gain_by_lane": {LANE_A: 0.45},
    }
    scope_gap = {
        "gap_id": "scope_0.scope_discrimination.parameter_anomaly",
        "gap_type": "hypothesis_discrimination",
        "candidate_ids": [candidate_id],
        "competition_axis": "scope",
        "target_scope": {"lane_id": LANE_A},
        "information_gain_by_lane": {LANE_A: 0.8},
    }

    contract = _challenge_output_contract(
        candidate_ids=[candidate_id],
        active_lane_ids=[LANE_A, LANE_B],
        evidence_gaps=[scope_gap, mechanism_gap],
        candidate_competition={
            "competition_requirement": CompetitionRequirement.MECHANISM_REQUIRED.value,
            "semantic_profiles_complete": False,
        },
    )

    assert contract["required_challenge_kind"] == "mechanism"
    assert contract["required_competition_axis"] == "mechanism"
    assert contract["allowed_gap_ids_by_candidate"][candidate_id] == [
        mechanism_gap["gap_id"]
    ]
    assert contract["highest_information_gain_gap_ids_by_candidate_and_lane"][candidate_id][
        LANE_A
    ] == [mechanism_gap["gap_id"]]


def test_authoritative_gap_projection_preserves_mechanism_action_value_fields() -> None:
    candidate_id = "REQ:llm:1"
    gap_id = "candidate_0.hypothesis_discrimination.mechanism_context"
    expected_types = [
        EvidenceType.ENGINEERING_NOTE.value,
        EvidenceType.HISTORICAL_CASE_MATCH.value,
        EvidenceType.SOP_GUIDANCE.value,
    ]
    finding = AgentFinding(
        finding_id="RCA_MECHANISM_GAP",
        agent=AgentKind.RCA_REASONING.value,
        summary="A mechanism discriminator remains unresolved.",
        confidence=0.5,
        evidence_ids=["EV_PROCESS"],
        details={
            "candidate_challenges": [
                {"distinguishing_gap_ids": [gap_id]}
            ],
            "causal_evidence_gaps": [
                {
                    "gap_id": gap_id,
                    "gap_type": "hypothesis_discrimination",
                    "question_kind": "process_mechanism",
                    "allowed_actions": [
                        ActionKind.VALIDATE_HISTORICAL_CASE.value,
                        ActionKind.RUN_RCA_REASONING.value,
                    ],
                    "candidate_id": candidate_id,
                    "candidate_ids": [candidate_id],
                    "candidate_index": 0,
                    "discriminator_kind": "mechanism_context",
                    "competition_axis": "mechanism",
                    "can_change_ranking": True,
                    "supports_candidate_ids": [candidate_id],
                    "weakens_candidate_ids": [candidate_id],
                    "expected_evidence_types": expected_types,
                    "information_gain": 0.65,
                    "priority": 1,
                    "target_scope": {
                        "lane_id": LANE_A,
                        "operation": "2500",
                        "equipment": "EQ_A",
                        "chamber": "EQ_A_CH01",
                        "recipe": "RCP_A",
                    },
                }
            ],
        },
    )

    projected = _authoritative_causal_gaps([finding], finding.finding_id)

    assert len(projected) == 1
    gap = projected[0]
    assert gap["competition_axis"] == "mechanism"
    assert gap["can_change_ranking"] is True
    assert gap["supports_candidate_ids"] == [candidate_id]
    assert gap["weakens_candidate_ids"] == [candidate_id]
    assert gap["expected_evidence_types"] == expected_types

    assessment = assess_action_values(
        options=[
            {
                "option_id": f"historical:{gap_id}",
                "action_kind": ActionKind.VALIDATE_HISTORICAL_CASE.value,
                "source": AgentKind.KNOWLEDGE.value,
                "scope": gap["target_scope"],
                "gap": gap,
            }
        ],
        action_records=[],
        gain_history=[],
        remaining_tool_budget=10,
        evidence=[_process_evidence()],
        competition_requirement=CompetitionRequirement.MECHANISM_REQUIRED.value,
        competition_axes=("mechanism", "scope"),
    )[0]

    assert assessment.eligible
    assert assessment.high_value
    assert assessment.can_change_ranking
    assert assessment.expected_evidence_types == tuple(expected_types)
    assert assessment.supports_candidate_ids == (candidate_id,)
    assert assessment.weakens_candidate_ids == (candidate_id,)
    assert assessment.already_available_evidence_ids == ()
    assert assessment.rejection_reason is None


def test_formal_009_patch4_replay_closes_root_slots_and_exposes_mechanism_action(
) -> None:
    replay_path = (
        ROOT
        / "outputs"
        / "patch4_qwen_smoke_FORMAL009_r1"
        / "states"
        / "FORMAL_009.json"
    )
    if not replay_path.exists():
        pytest.skip("FORMAL_009 Patch 4 replay State is not available")
    state = RCAState.from_dict(json.loads(replay_path.read_text(encoding="utf-8")))
    finding = state.authoritative_rca_finding
    assert finding is not None
    raw_candidates = finding.details["ranked_candidates"]
    assert len(raw_candidates) == 2
    proposals = tuple(
        HypothesisCandidateProposal(
            root_cause=str(item["root_cause"]),
            causal_explanation=str(item["causal_explanation"]),
            supporting_evidence_ids=tuple(item["supporting_evidence_ids"]),
            contradicting_evidence_ids=tuple(item["contradicting_evidence_ids"]),
        )
        for item in raw_candidates
    )
    generation = finding.details["hypothesis_candidate_generation"]
    roots, profiles, variants = _partition_root_candidates(
        proposals,
        (),
        competition_requirement=CompetitionRequirement.MECHANISM_REQUIRED.value,
        candidate_ids=tuple(str(item["candidate_id"]) for item in raw_candidates),
        semantic_validation_errors=tuple(
            str(item) for item in generation.get("semantic_validation_errors", [])
        ),
    )

    assert len(roots) == 1
    assert profiles == ()
    assert len(variants) == 1
    assert variants[0]["candidate_id"] == raw_candidates[1]["candidate_id"]
    assert variants[0]["semantic_profile_status"] in {"missing", "invalid"}
    assert variants[0]["supporting_evidence_ids"]

    mechanism_gap = next(
        gap
        for gap in finding.details["causal_evidence_gaps"]
        if gap.get("candidate_id") == raw_candidates[0]["candidate_id"]
        and gap.get("discriminator_kind") == "mechanism_context"
    )
    active_lane_ids = [
        lane.lane_id
        for lane in state.causal_lanes
        if lane.lifecycle_status in {"active", "challenged"}
    ]
    contract = _challenge_output_contract(
        candidate_ids=[str(raw_candidates[0]["candidate_id"])],
        active_lane_ids=active_lane_ids,
        evidence_gaps=finding.details["causal_evidence_gaps"],
        candidate_competition={
            "competition_requirement": CompetitionRequirement.MECHANISM_REQUIRED.value,
            "semantic_profiles_complete": False,
        },
    )
    allowed_gap_ids = contract["allowed_gap_ids_by_candidate"][
        raw_candidates[0]["candidate_id"]
    ]
    assert mechanism_gap["gap_id"] in allowed_gap_ids
    assert all("scope_discrimination" not in gap_id for gap_id in allowed_gap_ids)

    challenge = dict(finding.details["candidate_challenges"][0])
    challenge.update(
        {
            "candidate_id": raw_candidates[0]["candidate_id"],
            "alternative_candidate_id": None,
            "challenge_kind": "mechanism",
            "mechanism_relation": "unknown",
            "evidence_probe_lane_id": mechanism_gap["target_scope"]["lane_id"],
            "distinguishing_gap_ids": [mechanism_gap["gap_id"]],
        }
    )
    replay_finding = replace(
        finding,
        details={**finding.details, "candidate_challenges": [challenge]},
    )
    replay_findings = [
        replay_finding if item.finding_id == replay_finding.finding_id else item
        for item in state.findings
    ]
    causal_gaps = _authoritative_causal_gaps(
        replay_findings,
        state.authoritative_rca_finding_id,
    )
    legal = QwenNextActionPlanner(FakeLLMClient())._legal_causal_gap_ids_by_action(
        questions=state.investigation_questions,
        findings=replay_findings,
        action_records=state.action_history,
        causal_gaps=causal_gaps,
        question_evidence_links=state.question_evidence_links,
    )

    assert legal[ActionKind.VALIDATE_HISTORICAL_CASE.value] == [
        mechanism_gap["gap_id"]
    ]

    projected_gap = next(
        gap for gap in causal_gaps if gap["gap_id"] == mechanism_gap["gap_id"]
    )
    assessment = assess_action_values(
        options=[
            {
                "option_id": "PATCH4_REPLAY_MECHANISM",
                "action_kind": ActionKind.VALIDATE_HISTORICAL_CASE.value,
                "source": AgentKind.KNOWLEDGE.value,
                "scope": projected_gap["target_scope"],
                "gap": projected_gap,
            }
        ],
        action_records=state.action_history,
        gain_history=state.investigation_gain_history,
        remaining_tool_budget=10,
        evidence=state.evidence,
        competition_requirement=CompetitionRequirement.MECHANISM_REQUIRED.value,
        competition_axes=("mechanism", "scope"),
    )[0]

    assert assessment.eligible
    assert assessment.high_value
    assert assessment.can_change_ranking
    assert assessment.expected_evidence_types
    assert assessment.already_available_evidence_ids == ()
    assert assessment.rejection_reason is None


def test_formal_009_r2_replay_does_not_stop_before_mechanism_action() -> None:
    replay_path = (
        ROOT
        / "outputs"
        / "patch4_root_slot_mechanism_axis_qwen_smoke_FORMAL009_r2"
        / "states"
        / "FORMAL_009.json"
    )
    if not replay_path.exists():
        pytest.skip("FORMAL_009 Patch 4 R2 replay State is not available")
    state = RCAState.from_dict(json.loads(replay_path.read_text(encoding="utf-8")))
    gain_history = list(
        derive_investigation_gain_history(
            action_records=state.action_history,
            evidence=state.evidence,
            links=state.question_evidence_links,
        )
    )

    outcome = QwenNextActionPlanner(FakeLLMClient()).decide_with_review(
        goal=state.investigation_goal,
        questions=state.investigation_questions,
        findings=state.findings,
        action_records=state.action_history,
        tool_call_count=int(state.execution_metadata.get("tool_call_count", 0)),
        evidence=state.evidence,
        evidence_ids=[item.evidence_id for item in state.evidence],
        question_evidence_links=state.question_evidence_links,
        capability_notices=state.capability_notices,
        hypotheses=state.hypotheses,
        prior_decisions=state.planner_decisions[:-1],
        authoritative_rca_finding_id=state.authoritative_rca_finding_id,
        investigation_gain_history=gain_history,
    )

    decision = outcome.decision
    assert decision.decision_type == "act"
    assert decision.next_action is not None
    assert decision.next_action.kind == ActionKind.VALIDATE_HISTORICAL_CASE.value
    assert decision.next_action.scope["causal_gap_id"] == (
        "candidate_0.hypothesis_discrimination.mechanism_context"
    )
    assert len(outcome.action_value_assessments) == 1
    assessment = outcome.action_value_assessments[0]
    assert assessment.eligible
    assert assessment.high_value
    assert assessment.can_change_ranking
    assert assessment.expected_evidence_types == (
        EvidenceType.ENGINEERING_NOTE.value,
        EvidenceType.HISTORICAL_CASE_MATCH.value,
        EvidenceType.SOP_GUIDANCE.value,
    )
    assert assessment.rejection_reason is None


def test_same_primary_mechanism_or_nested_candidate_cannot_satisfy_mechanism_axis() -> None:
    reference = _profile(
        "REQ:llm:1",
        primary="Chamber wall conditioning degradation",
        relation=CandidateMechanismRelation.REFERENCE.value,
    )
    renamed = _profile(
        "REQ:llm:2",
        primary="Degraded conditioning of the chamber wall",
        relation=CandidateMechanismRelation.INDEPENDENT_ALTERNATIVE.value,
        prediction="A reworded wall mechanism produces another signature.",
    )
    nested = _profile(
        "REQ:llm:2",
        primary="Polymer accumulation caused by wall conditioning degradation",
        relation=CandidateMechanismRelation.NESTED.value,
        dependency=reference.candidate_id,
    )

    assert not _mechanism_semantics_are_distinct(reference, renamed)
    assert not _mechanism_semantics_are_distinct(reference, nested)


def test_mechanism_required_gap_does_not_promote_scope_or_existing_parameter_query() -> None:
    reference = _profile(
        "REQ:llm:1",
        primary="Chamber wall conditioning degradation",
        relation=CandidateMechanismRelation.REFERENCE.value,
    )
    lane = {
        "lane_id": LANE_A,
        "operation": "2500",
        "equipment": "EQ_A",
        "chamber": "EQ_A_CH01",
        "recipe": "RCP_A",
        "parameter_scope": ["pressure_response"],
        "exposed_lot_ids": ["LOT_A", "LOT_B"],
        "time_window": ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"],
        "priority_score": 0.9,
        "lifecycle_status": "active",
    }
    gaps = build_hypothesis_discrimination_gaps(
        [_matrix()],
        causal_lanes=[lane],
        candidate_ids=[reference.candidate_id],
        candidate_semantic_profiles=[reference],
        competition_brief={
            "competition_requirement": CompetitionRequirement.MECHANISM_REQUIRED.value,
            "scope_groups": [],
        },
    )

    assert {gap["discriminator_kind"] for gap in gaps} == {"mechanism_context"}
    assert all(gap["competition_axis"] == "mechanism" for gap in gaps)

    process_gap = {
        "gap_id": "candidate_0.hypothesis_discrimination.parameter_anomaly",
        "gap_type": "hypothesis_discrimination",
        "competition_axis": "mechanism",
        "candidate_id": reference.candidate_id,
        "supports_candidate_ids": [reference.candidate_id],
        "weakens_candidate_ids": [reference.candidate_id],
        "discriminator_kind": "parameter_anomaly",
        "information_gain": 0.9,
        "can_change_ranking": True,
        "expected_evidence_types": [EvidenceType.PARAMETER_DEVIATION.value],
        "target_scope": {
            "lane_id": LANE_A,
            "operation": "2500",
            "equipment": "EQ_A",
            "chamber": "EQ_A_CH01",
            "recipe": "RCP_A",
            "parameters": "pressure_response",
        },
    }
    assessment = assess_action_values(
        options=[
            {
                "option_id": "EXISTING",
                "action_kind": ActionKind.INSPECT_FDC_SPC.value,
                "source": AgentKind.FDC.value,
                "scope": process_gap["target_scope"],
                "gap": process_gap,
            }
        ],
        action_records=[],
        gain_history=[],
        remaining_tool_budget=10,
        evidence=[_process_evidence()],
        competition_requirement=CompetitionRequirement.MECHANISM_REQUIRED.value,
    )[0]

    assert not assessment.eligible
    assert assessment.rejection_reason == "discriminator_evidence_already_available"
    assert assessment.already_available_evidence_ids == ("EV_PROCESS",)


def test_mixed_requirement_without_mechanism_axis_preserves_scope_action_value() -> None:
    assessment = assess_action_values(
        options=[
            {
                "option_id": "SCOPE_ONLY",
                "action_kind": ActionKind.INSPECT_FDC_SPC.value,
                "source": AgentKind.FDC.value,
                "scope": {"lane_id": LANE_B},
                "gap": {
                    "gap_id": "scope_0.parameter_anomaly",
                    "gap_type": "hypothesis_discrimination",
                    "competition_axis": "scope",
                    "discriminator_kind": "parameter_anomaly",
                    "information_gain": 0.8,
                    "can_change_ranking": True,
                },
            }
        ],
        action_records=[],
        gain_history=[],
        remaining_tool_budget=10,
        competition_requirement=CompetitionRequirement.MIXED_REQUIRED.value,
        competition_axes=("direction", "scope"),
    )[0]

    assert assessment.high_value
    assert assessment.rejection_reason is None


def test_matrix_prompt_projection_omits_claim_facts() -> None:
    matrix = CausalEvidenceMatrix(
        candidate=_matrix().candidate,
        claims={
            "parameter": CausalClaimResult(
                claim="parameter",
                status="supported",
                evidence_ids=("EV_PROCESS",),
                reason="The typed parameter signal matches the candidate.",
                facts={"entities": [f"ENTITY_{index}" for index in range(200)]},
            ),
            "mechanism": CausalClaimResult(
                claim="mechanism",
                status="incomplete",
                reason="The physical bridge remains unresolved.",
            ),
        },
    )

    projection = compact_causal_evidence_matrix_for_prompt(matrix)

    assert "facts" not in projection["claims"]["parameter"]
    assert projection["projection_audit"]["omitted_claim_fact_groups"] == 1
    assert len(json.dumps(projection)) < len(json.dumps(matrix.to_dict()))


class _CaptureFakeClient(FakeLLMClient):
    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    def complete_json(self, request: LLMRequest):  # type: ignore[no-untyped-def]
        self.requests.append(request)
        return super().complete_json(request)


def test_candidate_prompt_is_compact_and_audited() -> None:
    evidence = _process_evidence()
    finding = AgentFinding(
        finding_id="F_FDC",
        agent=AgentKind.FDC.value,
        summary="A typed process signal is available.",
        confidence=0.95,
        evidence_ids=[evidence.evidence_id],
        evidence=[evidence],
    )
    client = _CaptureFakeClient()

    result = QwenHypothesisCandidateGenerator(client).generate(
        request_id="REQ_PROMPT_BOUND",
        findings=[finding],
        causal_lanes=[
            {
                "lane_id": LANE_A,
                "operation": "2500",
                "equipment": "EQ_A",
                "chamber": "EQ_A_CH01",
                "recipe": "RCP_A",
                "parameter_scope": ["pressure_response"],
                "priority_score": 0.9,
                "lifecycle_status": "active",
            }
        ],
    )

    assert len(client.requests) == 1
    request = client.requests[0]
    assert len(json.dumps(request.payload, ensure_ascii=False)) <= 64_000
    card = request.payload["typed_evidence_register"][0]
    assert "entities" not in card
    assert "metadata" not in card
    assert card["entity_ids_by_type"]["parameter"] == ["pressure_response"]
    assert result.evidence_synthesis is not None
    assert result.evidence_synthesis["prompt_budget_applied"] is True
    assert result.evidence_synthesis["prompt_payload_char_count"] <= 64_000


def test_finalizer_projects_one_terminal_state_into_authoritative_finding() -> None:
    evidence = _process_evidence()
    finding = AgentFinding(
        finding_id="RCA_AUTH",
        agent=AgentKind.RCA_REASONING.value,
        summary="The root cause remains unresolved.",
        confidence=0.4,
        evidence_ids=[evidence.evidence_id],
        evidence=[evidence],
        details={
            "status": "inconclusive",
            "root_cause": "inconclusive",
            "conclusion_status": "inconclusive",
            "competition_status": "active",
            "scope_assessment_status": "pending",
            "alternative_search_status": "in_progress",
            "ranked_candidates": [
                {
                    "candidate_id": "CANDIDATE_A",
                    "root_cause": "An unresolved mechanism.",
                    "status": "candidate",
                }
            ],
            "confirmation_gate": {"status": "inconclusive"},
            "hypothesis_candidate_generation": {
                "competition_status": "active",
                "competition_assessment": {
                    "competition_status": "active",
                    "scope_assessment_status": "pending",
                },
            },
        },
    )
    state = RCAState(
        job=RCAJob(
            job_id="PATCH4_FINAL",
            user_query="Close the investigation consistently.",
            status=TaskStatus.RUNNING.value,
        ),
        evidence=[evidence],
        findings=[finding],
        authoritative_rca_finding_id=finding.finding_id,
        competition_trace=CompetitionTrace(
            competition_requirement=CompetitionRequirement.MECHANISM_REQUIRED.value,
            competition_status=CandidateCompetitionStatus.ACTIVE.value,
            scope_assessment_status=ScopeAssessmentStatus.PENDING.value,
            alternative_search_status="in_progress",
        ),
    )

    finalized = finalize_investigation(state)
    completed = replace(
        finalized,
        job=replace(finalized.job, status=TaskStatus.COMPLETED.value),
    )
    validate_terminal_investigation_state(completed)

    trace = completed.competition_trace
    assert trace is not None
    details = completed.authoritative_rca_finding.details  # type: ignore[union-attr]
    assert trace.competition_status == CandidateCompetitionStatus.EXHAUSTED.value
    assert trace.scope_assessment_status == ScopeAssessmentStatus.EXHAUSTED.value
    assert details["competition_status"] == trace.competition_status
    assert details["scope_assessment_status"] == trace.scope_assessment_status
    assert details["terminal_investigation_snapshot"]["terminal_reason"] == (
        trace.terminal_reason
    )
    generation = details["hypothesis_candidate_generation"]
    assert generation["competition_status"] == trace.competition_status
    assert generation["competition_assessment"]["scope_assessment_status"] == (
        trace.scope_assessment_status
    )


def test_formal_009_r1_semantics_do_not_pass_as_independent_mechanisms() -> None:
    """Replay the R1 semantic shape without case/device-specific runtime rules."""

    reference = CandidateSemanticProfile.from_dict(
        {
            "candidate_id": "FORMAL_REPLAY:llm:1",
            "scope_relation": "shared_effect",
            "claimed_lane_ids": [LANE_A, LANE_B],
            "comparison_lane_ids": [LANE_A, LANE_B],
            "mechanism_claim": (
                "Chamber degradation causes correlated parameter drift."
            ),
            "distinguishing_predictions": [
                {
                    "discriminator_kind": "parameter_anomaly",
                    "lane_ids": [LANE_A, LANE_B],
                    "prediction": "Both recipes show correlated drift.",
                }
            ],
        }
    )
    sensitivity = CandidateSemanticProfile.from_dict(
        {
            "candidate_id": "FORMAL_REPLAY:llm:2",
            "scope_relation": "differential_sensitivity",
            "claimed_lane_ids": [LANE_A],
            "comparison_lane_ids": [LANE_A, LANE_B],
            "mechanism_claim": (
                "RCP_A is more sensitive to the same chamber degradation."
            ),
            "distinguishing_predictions": [
                {
                    "discriminator_kind": "product_outcome",
                    "lane_ids": [LANE_A, LANE_B],
                    "prediction": "RCP_A has a more severe outcome.",
                }
            ],
        }
    )

    assert reference.mechanism_relation == CandidateMechanismRelation.UNKNOWN.value
    assert sensitivity.mechanism_relation == CandidateMechanismRelation.UNKNOWN.value
    assert not _mechanism_semantics_are_distinct(reference, sensitivity)
    proposals, _profiles, variants = _partition_root_candidates(
        (_proposal("Chamber degradation", "EV_A"), _proposal("Recipe sensitivity", "EV_B")),
        (reference, sensitivity),
        competition_requirement=CompetitionRequirement.MECHANISM_REQUIRED.value,
    )
    assert len(proposals) == 1
    assert len(variants) == 1
