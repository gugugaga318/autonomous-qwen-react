from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core"))

from yield_rca_core.causal_investigation_models import (  # noqa: E402
    AlternativeLaneResolution,
    AlternativeLaneResolutionStatus,
    AlternativeSearchStatus,
    CandidateChallenge,
    CandidateCompetitionStatus,
    CandidateCompetitionType,
    CandidateDistinguishingPrediction,
    CandidateScopeRelation,
    CandidateSemanticProfile,
    CausalChainCompleteness,
    CausalLaneRecord,
    ChallengeKind,
    ChallengeStatus,
    CompetitionFailureReason,
    CompetitionGapReason,
    CompetitionRequirement,
    CompetitionTrace,
    InvestigationLaneStatus,
)
from yield_rca_core.evidence_models import Evidence, EvidenceSourceType  # noqa: E402
from yield_rca_core.models import (  # noqa: E402
    ModelValidationError,
    RCAJob,
    RCAState,
)


def make_evidence(evidence_id: str = "EV_001") -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        source_type=EvidenceSourceType.FDC.value,
        source_id="fdc:LOT_001:pressure",
        summary="Backside pressure excursion was observed.",
    )


def make_lane(
    lane_id: str = "LANE_001",
    *,
    status: str = InvestigationLaneStatus.UNINVESTIGATED.value,
    reason: str | None = None,
) -> CausalLaneRecord:
    return CausalLaneRecord(
        lane_id=lane_id,
        operation="OP_4000",
        equipment="EQ_001",
        chamber="CH_01",
        parameter_scope=("backside_pressure",),
        exposed_lot_ids=("LOT_001",),
        time_window=(
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T01:00:00+00:00",
        ),
        initial_evidence_ids=("EV_001",),
        priority_score=0.9,
        investigation_status=status,
        pruned_reason=reason,
    )


def make_state(**kwargs: object) -> RCAState:
    values: dict[str, object] = {
        "job": RCAJob(job_id="JOB_001", user_query="Find the root cause."),
        "evidence": [make_evidence()],
    }
    values.update(kwargs)
    return RCAState(**values)  # type: ignore[arg-type]


def test_lane_round_trip_and_timestamp_contract() -> None:
    lane = make_lane()
    assert CausalLaneRecord.from_dict(lane.to_dict()) == lane

    with pytest.raises(ModelValidationError, match="zero or two timestamps"):
        CausalLaneRecord(lane_id="LANE_BAD", time_window=("2026-01-01T00:00:00+00:00",))
    with pytest.raises(ModelValidationError, match="include a timezone"):
        CausalLaneRecord(
            lane_id="LANE_BAD",
            time_window=("2026-01-01T00:00:00", "2026-01-01T01:00:00"),
        )


def test_lane_status_and_pruning_reason_are_python_validated() -> None:
    with pytest.raises(ModelValidationError, match="investigation_status"):
        make_lane(status="invented")
    with pytest.raises(ModelValidationError, match="require pruned_reason"):
        make_lane(status=InvestigationLaneStatus.ELIMINATED.value)
    with pytest.raises(ModelValidationError, match="require pruned_reason"):
        make_lane(status=InvestigationLaneStatus.BLOCKED.value)


def test_challenge_and_competition_trace_round_trip() -> None:
    challenge = CandidateChallenge(
        candidate_id="CANDIDATE_A",
        alternative_candidate_id="CANDIDATE_B",
        evidence_probe_lane_id="LANE_001",
        challenge_kind=ChallengeKind.SCOPE.value,
        strongest_alternative_lane_id="LANE_001",
        supporting_evidence_ids=("EV_001",),
        contradicting_evidence_ids=(),
        unexplained_precursor_evidence_ids=(),
        distinguishing_gap_ids=("mechanism_missing",),
        challenge_explanation="Pressure lane remains the strongest alternative.",
        status=ChallengeStatus.ALTERNATIVE_IDENTIFIED.value,
    )
    trace = CompetitionTrace(
        active_lane_ids=("LANE_001",),
        overflow_lane_ids=("LANE_002",),
        represented_lane_ids=("LANE_001",),
        unresolved_lane_ids=("LANE_002",),
        alternative_search_status=AlternativeSearchStatus.UNRESOLVED.value,
        competition_requirement=CompetitionRequirement.DIRECTION_REQUIRED.value,
        competition_status=CandidateCompetitionStatus.FAILED.value,
        competition_type=CandidateCompetitionType.CAUSAL_DIRECTION.value,
        competition_failure_reason=(
            CompetitionFailureReason.ALTERNATIVE_DIRECTION_NOT_GENERATED.value
        ),
        candidate_semantic_profiles=(
            CandidateSemanticProfile(
                candidate_id="CANDIDATE_A",
                scope_relation=CandidateScopeRelation.FOCAL_ONLY.value,
                claimed_lane_ids=("LANE_001",),
                comparison_lane_ids=("LANE_001",),
                mechanism_claim="Pressure instability changes wafer clamping.",
                distinguishing_predictions=(
                    CandidateDistinguishingPrediction(
                        discriminator_kind="product_outcome",
                        lane_ids=("LANE_001",),
                        prediction="The focal Lane produces the compatible defect.",
                    ),
                ),
            ),
        ),
        candidate_lineage=(
            {
                "candidate_index": 0,
                "lineage_status": "retained",
            },
        ),
        challenge_round_count=1,
        resolution_evidence_ids=("EV_001",),
        lane_resolutions=(
            AlternativeLaneResolution(
                lane_id="LANE_001",
                status=AlternativeLaneResolutionStatus.RETAINED.value,
                candidate_id="CANDIDATE_A",
                evidence_ids=("EV_001",),
                distinguishing_gap_ids=("mechanism_missing",),
                reason="The Lane remains viable.",
            ),
        ),
    )
    assert CandidateChallenge.from_dict(challenge.to_dict()) == challenge
    assert CompetitionTrace.from_dict(trace.to_dict()) == trace
    with pytest.raises(ModelValidationError, match="active and overflow"):
        CompetitionTrace(active_lane_ids=("LANE_001",), overflow_lane_ids=("LANE_001",))
    with pytest.raises(ModelValidationError, match="unresolved and eliminated"):
        CompetitionTrace(unresolved_lane_ids=("LANE_001",), eliminated_lane_ids=("LANE_001",))
    with pytest.raises(ModelValidationError, match="requires competition_failure_reason"):
        CompetitionTrace(
            competition_status=CandidateCompetitionStatus.FAILED.value,
        )

    pending = CompetitionTrace(
        competition_requirement=CompetitionRequirement.SCOPE_REQUIRED.value,
        competition_status=CandidateCompetitionStatus.PENDING.value,
        competition_gap_reason=CompetitionGapReason.SCOPE_HYPOTHESIS_COLLAPSED.value,
    )
    assert CompetitionTrace.from_dict(pending.to_dict()) == pending


def test_rca_state_causal_fields_round_trip_and_legacy_compatibility() -> None:
    state = make_state(
        causal_lanes=[make_lane()],
        candidate_challenges=[
            CandidateChallenge(
                candidate_id="CANDIDATE_A",
                strongest_alternative_lane_id="LANE_001",
                supporting_evidence_ids=("EV_001",),
            )
        ],
        competition_trace=CompetitionTrace(
            active_lane_ids=("LANE_001",),
            represented_lane_ids=("LANE_001",),
            alternative_search_status=AlternativeSearchStatus.ALTERNATIVES_ELIMINATED.value,
            candidate_semantic_profiles=(
                CandidateSemanticProfile(
                    candidate_id="CANDIDATE_A",
                    scope_relation=CandidateScopeRelation.FOCAL_ONLY.value,
                    claimed_lane_ids=("LANE_001",),
                    comparison_lane_ids=("LANE_001",),
                    mechanism_claim="Pressure instability changes wafer clamping.",
                ),
            ),
            resolution_evidence_ids=("EV_001",),
        ),
        causal_chain_completeness=CausalChainCompleteness.COMPLETE.value,
    )
    restored = RCAState.from_dict(state.to_dict())
    assert restored.to_dict() == state.to_dict()

    legacy = RCAState.from_dict({"job": state.job.to_dict()})
    assert legacy.causal_lanes == []
    assert legacy.candidate_challenges == []
    assert legacy.competition_trace is None
    assert legacy.causal_chain_completeness is None


def test_rca_state_rejects_invalid_causal_references_and_status() -> None:
    with pytest.raises(ModelValidationError, match="duplicate causal lane_id"):
        make_state(causal_lanes=[make_lane("LANE_001"), make_lane("LANE_001")])
    with pytest.raises(ModelValidationError, match="unknown evidence_ids"):
        make_state(
            causal_lanes=[
                CausalLaneRecord(lane_id="LANE_001", initial_evidence_ids=("EV_MISSING",))
            ]
        )
    with pytest.raises(ModelValidationError, match="candidate challenge"):
        make_state(
            causal_lanes=[make_lane()],
            candidate_challenges=[
                CandidateChallenge(
                    candidate_id="CANDIDATE_A",
                    strongest_alternative_lane_id="LANE_001",
                    supporting_evidence_ids=("EV_MISSING",),
                )
            ],
        )
    with pytest.raises(ModelValidationError, match="unknown strongest alternative lane"):
        make_state(
            candidate_challenges=[
                CandidateChallenge(
                    candidate_id="CANDIDATE_A",
                    strongest_alternative_lane_id="LANE_MISSING",
                )
            ]
        )
    with pytest.raises(ModelValidationError, match="unknown causal lanes"):
        make_state(
            causal_lanes=[make_lane()],
            competition_trace=CompetitionTrace(active_lane_ids=("LANE_MISSING",)),
        )
    with pytest.raises(ModelValidationError, match="unknown causal lanes"):
        make_state(
            causal_lanes=[make_lane()],
            competition_trace=CompetitionTrace(
                candidate_semantic_profiles=(
                    CandidateSemanticProfile(
                        candidate_id="CANDIDATE_A",
                        scope_relation=CandidateScopeRelation.FOCAL_ONLY.value,
                        claimed_lane_ids=("LANE_MISSING",),
                        comparison_lane_ids=("LANE_MISSING",),
                        mechanism_claim="Unknown Lane mechanism.",
                    ),
                )
            ),
        )
    with pytest.raises(ModelValidationError, match="causal_chain_completeness"):
        make_state(causal_chain_completeness="not_a_status")
