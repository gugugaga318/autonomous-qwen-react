from __future__ import annotations

from yield_rca_core.causal_competition_progression import (
    derive_candidate_resolutions,
    progress_competition,
)
from yield_rca_core.causal_evidence_matrix import build_causal_evidence_matrix
from yield_rca_core.causal_hypothesis import CausalHypothesis
from yield_rca_core.causal_investigation_models import (
    CandidateCompetitionStatus,
    CandidateResolutionStatus,
    CandidateScopeRelation,
    CandidateSemanticProfile,
    CompetitionRequirement,
    CompetitionTrace,
)
from yield_rca_core.evidence_models import (
    EVIDENCE_SCHEMA_VERSION,
    EntityType,
    Evidence,
    EvidenceEntity,
    EvidenceSourceType,
    EvidenceType,
)
from yield_rca_core.hypothesis_engine import HypothesisEngine
from yield_rca_core.models import AgentKind

LANE_A = "lane:2500:EQ_A:EQ_A_CH01:RCP_A"
LANE_B = "lane:2500:EQ_A:EQ_A_CH01:RCP_B"
LANE_OUTSIDE = "lane:9000:EQ_Z:EQ_Z_CH09:RCP_Z"


def _evidence(evidence_id: str, lane_id: str, lot_id: str) -> Evidence:
    recipe = lane_id.rsplit(":", 1)[-1]
    return Evidence(
        evidence_id=evidence_id,
        source_type=EvidenceSourceType.FDC.value,
        source_id=f"SOURCE_{evidence_id}",
        summary=f"Parameter Evidence on {lane_id}",
        evidence_type=EvidenceType.PARAMETER_DEVIATION.value,
        source_agent=AgentKind.FDC.value,
        source_tool="inspect_fdc_spc",
        observation="Measured oxygen-flow response deviation.",
        entities=[
            EvidenceEntity(EntityType.LOT.value, lot_id),
            EvidenceEntity(EntityType.EQUIPMENT.value, lane_id.split(":")[2]),
            EvidenceEntity(EntityType.CHAMBER.value, lane_id.split(":")[3]),
            EvidenceEntity(EntityType.OPERATION.value, lane_id.split(":")[1]),
            EvidenceEntity(EntityType.RECIPE.value, recipe),
            EvidenceEntity(EntityType.PARAMETER.value, "oxygen_flow_response_delta"),
        ],
        metadata={"lane_id": lane_id, "recipe": recipe},
        confidence=0.95,
        evidence_schema_version=EVIDENCE_SCHEMA_VERSION,
    )


def _candidate(*evidence_ids: str) -> CausalHypothesis:
    return CausalHypothesis(
        root_cause="Plasma response instability affects the declared recipe scope.",
        causal_explanation=(
            "Oxygen-flow response instability changes plasma kinetics and can "
            "produce the observed product response."
        ),
        supporting_evidence_ids=tuple(evidence_ids),
    )


def _profile(
    relation: str,
    *,
    claimed: tuple[str, ...],
    comparison: tuple[str, ...],
) -> CandidateSemanticProfile:
    return CandidateSemanticProfile(
        candidate_id="REQUEST:llm:1",
        scope_relation=relation,
        claimed_lane_ids=claimed,
        comparison_lane_ids=comparison,
        mechanism_claim="Plasma response instability changes process outcome.",
    )


def test_shared_effect_accepts_evidence_from_all_claimed_lanes() -> None:
    evidence = [
        _evidence("EV_A", LANE_A, "LOT_A"),
        _evidence("EV_B", LANE_B, "LOT_B"),
    ]
    matrix = build_causal_evidence_matrix(
        _candidate("EV_A", "EV_B"),
        evidence,
        semantic_profile=_profile(
            CandidateScopeRelation.SHARED_EFFECT,
            claimed=(LANE_A, LANE_B),
            comparison=(LANE_A, LANE_B),
        ),
    )

    scope = matrix.claims["scope"]
    assert scope.status == "supported"
    assert scope.facts["evidence_lane_ids"] == [LANE_A, LANE_B]
    assert scope.facts["missing_claimed_lane_ids"] == []
    assert matrix.status != "conflicted"


def test_comparison_lane_evidence_does_not_broaden_focal_or_sensitive_scope() -> None:
    evidence = [
        _evidence("EV_A", LANE_A, "LOT_A"),
        _evidence("EV_B", LANE_B, "LOT_B"),
    ]

    for relation in (
        CandidateScopeRelation.FOCAL_ONLY,
        CandidateScopeRelation.DIFFERENTIAL_SENSITIVITY,
    ):
        matrix = build_causal_evidence_matrix(
            _candidate("EV_A", "EV_B"),
            evidence,
            semantic_profile=_profile(
                relation,
                claimed=(LANE_A,),
                comparison=(LANE_A, LANE_B),
            ),
        )
        scope = matrix.claims["scope"]
        assert scope.status == "supported"
        assert scope.facts["claimed_lane_ids"] == [LANE_A]
        assert scope.facts["evidence_lane_ids"] == [LANE_A, LANE_B]


def test_evidence_outside_claimed_and_comparison_scope_is_conflicted() -> None:
    evidence = [
        _evidence("EV_A", LANE_A, "LOT_A"),
        _evidence("EV_OUTSIDE", LANE_OUTSIDE, "LOT_Z"),
    ]
    matrix = build_causal_evidence_matrix(
        _candidate("EV_A", "EV_OUTSIDE"),
        evidence,
        semantic_profile=_profile(
            CandidateScopeRelation.FOCAL_ONLY,
            claimed=(LANE_A,),
            comparison=(LANE_A, LANE_B),
        ),
    )

    scope = matrix.claims["scope"]
    assert scope.status == "conflicted"
    assert scope.facts["unexpected_evidence_lane_ids"] == [LANE_OUTSIDE]


def test_missing_claimed_lane_evidence_is_incomplete_not_contradicted() -> None:
    evidence = [_evidence("EV_A", LANE_A, "LOT_A")]
    matrix = build_causal_evidence_matrix(
        _candidate("EV_A"),
        evidence,
        semantic_profile=_profile(
            CandidateScopeRelation.SHARED_EFFECT,
            claimed=(LANE_A, LANE_B),
            comparison=(LANE_A, LANE_B),
        ),
    )

    scope = matrix.claims["scope"]
    assert scope.status == "incomplete"
    assert scope.facts["missing_claimed_lane_ids"] == [LANE_B]
    assert matrix.status != "conflicted"


def test_invalid_semantic_profile_keeps_candidate_unresolved() -> None:
    evidence = [_evidence("EV_A", LANE_A, "LOT_A")]
    matrix = build_causal_evidence_matrix(
        _candidate("EV_A"),
        evidence,
        semantic_profile={},
    )
    resolutions = derive_candidate_resolutions(
        {
            "conclusion_status": "inconclusive",
            "ranked_candidates": [
                {
                    "candidate_id": "REQUEST:llm:1",
                    "status": "candidate",
                    "causal_matrix_status": matrix.status,
                    "causal_evidence_matrix": matrix.to_dict(),
                    "supporting_evidence_ids": ["EV_A"],
                }
            ],
            "confirmation_gate": {"status": "inconclusive"},
        }
    )

    assert matrix.claims["scope"].status == "incomplete"
    assert resolutions[0].status == CandidateResolutionStatus.UNRESOLVED
    assert "scope_semantics_incomplete" in resolutions[0].reason_codes


def test_scope_replay_does_not_false_reject_all_multi_recipe_candidates() -> None:
    evidence = [
        _evidence("EV_A", LANE_A, "LOT_A"),
        _evidence("EV_B", LANE_B, "LOT_B"),
    ]
    shared = build_causal_evidence_matrix(
        _candidate("EV_A", "EV_B"),
        evidence,
        semantic_profile=_profile(
            CandidateScopeRelation.SHARED_EFFECT,
            claimed=(LANE_A, LANE_B),
            comparison=(LANE_A, LANE_B),
        ),
    )
    sensitive = build_causal_evidence_matrix(
        _candidate("EV_A", "EV_B"),
        evidence,
        semantic_profile=_profile(
            CandidateScopeRelation.DIFFERENTIAL_SENSITIVITY,
            claimed=(LANE_A,),
            comparison=(LANE_A, LANE_B),
        ),
    )
    result = progress_competition(
        trace=CompetitionTrace(
            competition_requirement=CompetitionRequirement.SCOPE_REQUIRED,
            competition_status=CandidateCompetitionStatus.ACTIVE,
        ),
        authoritative_details={
            "conclusion_status": "inconclusive",
            "ranked_candidates": [
                {
                    "candidate_id": "SHARED",
                    "status": "candidate",
                    "causal_matrix_status": shared.status,
                    "causal_evidence_matrix": shared.to_dict(),
                },
                {
                    "candidate_id": "SENSITIVE",
                    "status": "candidate",
                    "causal_matrix_status": sensitive.status,
                    "causal_evidence_matrix": sensitive.to_dict(),
                },
            ],
            "confirmation_gate": {"status": "inconclusive"},
        },
        action_value_assessments=[],
        gain_history=[],
        force_terminal=True,
    )

    assert result.trace.competition_status == CandidateCompetitionStatus.EXHAUSTED
    assert result.trace.terminal_reason != "all_formal_candidates_contradicted"
    assert all(
        resolution.status == CandidateResolutionStatus.UNRESOLVED
        for resolution in result.trace.candidate_resolutions
    )


def test_hypothesis_engine_passes_candidate_semantics_into_matrix() -> None:
    evidence = [
        _evidence("EV_A", LANE_A, "LOT_A"),
        _evidence("EV_B", LANE_B, "LOT_B"),
    ]
    candidate = _candidate("EV_A", "EV_B")
    profile = _profile(
        CandidateScopeRelation.SHARED_EFFECT,
        claimed=(LANE_A, LANE_B),
        comparison=(LANE_A, LANE_B),
    )

    result = HypothesisEngine().analyze(
        request_id="REQUEST",
        findings=[],
        mode="active",
        external_candidates=[candidate.to_dict()],
        include_deterministic_candidates=False,
        context_evidence=evidence,
        candidate_semantic_profiles=[profile],
        competition_brief={"competition_requirement": "scope_required"},
    )

    scope = result["candidates"][0]["causal_evidence_matrix"]["claims"]["scope"]
    assert scope["status"] == "supported"
    assert scope["facts"]["scope_relation"] == "shared_effect"


def test_legacy_matrix_without_semantic_profile_keeps_old_scope_rule() -> None:
    evidence = [
        _evidence("EV_A", LANE_A, "LOT_A"),
        _evidence("EV_B", LANE_B, "LOT_B"),
    ]

    matrix = build_causal_evidence_matrix(
        _candidate("EV_A", "EV_B"),
        evidence,
    )

    assert matrix.claims["scope"].status == "conflicted"
