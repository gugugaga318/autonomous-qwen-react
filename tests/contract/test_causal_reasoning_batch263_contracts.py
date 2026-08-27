from __future__ import annotations

from yield_rca_core.causal_adversarial import (
    _normalize_challenge_payload,
    derive_alternative_lane_resolutions,
    derive_alternative_search_status,
)
from yield_rca_core.causal_investigation_models import (
    AlternativeLaneResolutionStatus,
    AlternativeSearchStatus,
    CandidateChallenge,
    CandidateCompetitionStatus,
    CandidateCompetitionType,
    CausalLaneRecord,
    ChallengeStatus,
    CompetitionFailureReason,
    CompetitionRequirement,
    InvestigationLaneStatus,
)
from yield_rca_core.evidence_models import (
    EVIDENCE_SCHEMA_VERSION,
    EntityType,
    Evidence,
    EvidenceEntity,
    EvidenceSourceType,
    EvidenceType,
)
from yield_rca_core.models import AgentFinding, RCAJob, RCAState
from yield_rca_core.supervisor import _update_competition_state


def _challenge(
    lane_id: str,
    status: str,
    *,
    candidate_id: str = "C1",
    evidence_id: str | None = None,
) -> CandidateChallenge:
    evidence_ids = (evidence_id,) if evidence_id is not None else ()
    return CandidateChallenge(
        candidate_id=candidate_id,
        strongest_alternative_lane_id=lane_id,
        supporting_evidence_ids=evidence_ids,
        distinguishing_gap_ids=(
            ()
            if status in {ChallengeStatus.RESOLVED.value, ChallengeStatus.BLOCKED.value}
            else (f"{candidate_id}.gap.{lane_id}",)
        ),
        challenge_explanation=f"Lane-level result for {lane_id}.",
        status=status,
    )


def _evidence(evidence_id: str, lane_id: str) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        source_type=EvidenceSourceType.FDC.value,
        source_id=f"SRC_{evidence_id}",
        summary=f"Observation for {lane_id}",
        evidence_type=EvidenceType.PARAMETER_DEVIATION.value,
        source_agent="fdc",
        source_tool="inspect_fdc_spc",
        observation=f"Observation for {lane_id}",
        entities=[EvidenceEntity(EntityType.EQUIPMENT.value, lane_id)],
        metadata={"lane_id": lane_id},
        confidence=0.95,
        evidence_schema_version=EVIDENCE_SCHEMA_VERSION,
    )


def test_one_resolved_alternative_does_not_close_another_active_lane() -> None:
    challenges = [
        _challenge("LANE_B", ChallengeStatus.RESOLVED.value, evidence_id="EV_B")
    ]

    resolutions = derive_alternative_lane_resolutions(
        challenges=challenges,
        active_lane_ids=["LANE_A", "LANE_B", "LANE_C"],
    )
    by_lane = {item.lane_id: item.status for item in resolutions}

    assert by_lane == {
        "LANE_A": AlternativeLaneResolutionStatus.UNRESOLVED.value,
        "LANE_B": AlternativeLaneResolutionStatus.ELIMINATED.value,
        "LANE_C": AlternativeLaneResolutionStatus.UNRESOLVED.value,
    }
    assert (
        derive_alternative_search_status(
            challenges=challenges,
            matrices=(),
            active_lane_ids=["LANE_A", "LANE_B", "LANE_C"],
            lane_resolutions=resolutions,
        )
        == AlternativeSearchStatus.ALTERNATIVE_FOUND.value
    )


def test_all_alternatives_must_be_eliminated_before_one_lane_is_retained() -> None:
    challenges = [
        _challenge("LANE_B", ChallengeStatus.RESOLVED.value, evidence_id="EV_B"),
        _challenge(
            "LANE_C",
            ChallengeStatus.RESOLVED.value,
            candidate_id="C2",
            evidence_id="EV_C",
        ),
    ]

    resolutions = derive_alternative_lane_resolutions(
        challenges=challenges,
        active_lane_ids=["LANE_A", "LANE_B", "LANE_C"],
    )
    by_lane = {item.lane_id: item.status for item in resolutions}

    assert by_lane["LANE_A"] == AlternativeLaneResolutionStatus.RETAINED.value
    assert by_lane["LANE_B"] == AlternativeLaneResolutionStatus.ELIMINATED.value
    assert by_lane["LANE_C"] == AlternativeLaneResolutionStatus.ELIMINATED.value
    assert (
        derive_alternative_search_status(
            challenges=challenges,
            matrices=(),
            active_lane_ids=["LANE_A", "LANE_B", "LANE_C"],
            lane_resolutions=resolutions,
        )
        == AlternativeSearchStatus.ALTERNATIVES_ELIMINATED.value
    )


def test_retained_blocked_and_non_discriminative_are_not_elimination() -> None:
    cases = {
        ChallengeStatus.ALTERNATIVE_IDENTIFIED.value: (
            AlternativeLaneResolutionStatus.RETAINED.value,
            AlternativeSearchStatus.ALTERNATIVE_FOUND.value,
        ),
        ChallengeStatus.BLOCKED.value: (
            AlternativeLaneResolutionStatus.BLOCKED.value,
            # One unavailable Lane is unresolved investigation state, not
            # proof that every high-value alternative is blocked. Patch 3's
            # Python Competition Progression Engine owns that global terminal
            # decision after evaluating all remaining Action values.
            AlternativeSearchStatus.UNRESOLVED.value,
        ),
        ChallengeStatus.NON_DISCRIMINATIVE.value: (
            AlternativeLaneResolutionStatus.NON_DISCRIMINATIVE.value,
            AlternativeSearchStatus.ALTERNATIVE_FOUND.value,
        ),
    }
    for challenge_status, (lane_status, search_status) in cases.items():
        challenge = _challenge("LANE_B", challenge_status, evidence_id="EV_B")
        resolutions = derive_alternative_lane_resolutions(
            challenges=[challenge],
            active_lane_ids=["LANE_A", "LANE_B"],
        )
        lane_b = next(item for item in resolutions if item.lane_id == "LANE_B")
        assert lane_b.status == lane_status
        assert (
            derive_alternative_search_status(
                challenges=[challenge],
                matrices=(),
                active_lane_ids=["LANE_A", "LANE_B"],
                lane_resolutions=resolutions,
            )
            == search_status
        )


def test_exhausted_non_discriminative_lane_is_a_valid_audit_result() -> None:
    evidence = _evidence("EV_B", "LANE_B")
    challenge = _normalize_challenge_payload(
        {
            "candidate_id": "C1",
            "strongest_alternative_lane_id": "LANE_B",
            "supporting_evidence_ids": ["EV_B"],
            "contradicting_evidence_ids": [],
            "unexplained_precursor_evidence_ids": [],
            "distinguishing_gap_ids": [],
            "distinguishing_questions": [],
            "challenge_explanation": "Available observations did not distinguish it.",
            "status": "non_discriminative",
        },
        candidate_ids={"C1"},
        lane_ids={"LANE_A", "LANE_B"},
        evidence_ids={"EV_B"},
        gap_ids=set(),
        gap_by_id={},
        evidence_by_id={"EV_B": evidence},
        lane_contexts={"LANE_B": {"lane_id": "LANE_B"}},
    )

    assert challenge.status == ChallengeStatus.NON_DISCRIMINATIVE.value


def test_supervisor_persists_each_lane_result_without_closing_other_lane() -> None:
    state = RCAState(
        job=RCAJob(job_id="JOB_263", user_query="test"),
        evidence=[_evidence("EV_B", "LANE_B"), _evidence("EV_C", "LANE_C")],
        causal_lanes=[
            CausalLaneRecord(lane_id="LANE_A", priority_score=0.9),
            CausalLaneRecord(lane_id="LANE_B", priority_score=0.8),
            CausalLaneRecord(lane_id="LANE_C", priority_score=0.7),
        ],
    )
    challenge = _challenge(
        "LANE_B",
        ChallengeStatus.RESOLVED.value,
        evidence_id="EV_B",
    )
    resolutions = derive_alternative_lane_resolutions(
        challenges=[challenge],
        active_lane_ids=["LANE_A", "LANE_B", "LANE_C"],
    )
    finding = AgentFinding(
        finding_id="F_263_R1",
        agent="rca_reasoning",
        summary="One alternative was eliminated.",
        confidence=0.5,
        evidence_ids=["EV_B"],
        details={
            "adversarial_challenge_generation": {"source": "qwen"},
            "candidate_challenges": [challenge.to_dict()],
            "alternative_lane_resolutions": [
                item.to_dict() for item in resolutions
            ],
        },
    )

    updated = _update_competition_state(state, finding)
    by_lane = {item.lane_id: item for item in updated.causal_lanes}

    assert by_lane["LANE_B"].investigation_status == (
        InvestigationLaneStatus.ELIMINATED.value
    )
    assert by_lane["LANE_C"].investigation_status != (
        InvestigationLaneStatus.ELIMINATED.value
    )
    assert updated.competition_trace is not None
    assert updated.competition_trace.alternative_search_status == (
        AlternativeSearchStatus.ALTERNATIVE_FOUND.value
    )
    assert "LANE_C" in updated.competition_trace.unresolved_lane_ids
    assert {
        item.lane_id: item.status
        for item in updated.competition_trace.lane_resolutions
    }["LANE_C"] == AlternativeLaneResolutionStatus.UNRESOLVED.value

    second_challenge = _challenge(
        "LANE_C",
        ChallengeStatus.RESOLVED.value,
        evidence_id="EV_C",
    )
    second_resolutions = derive_alternative_lane_resolutions(
        challenges=[second_challenge],
        active_lane_ids=["LANE_A", "LANE_C"],
        eliminated_lane_ids=["LANE_B"],
    )
    second_finding = AgentFinding(
        finding_id="F_263_R2",
        agent="rca_reasoning",
        summary="The remaining alternative was eliminated.",
        confidence=0.6,
        evidence_ids=["EV_C"],
        details={
            "adversarial_challenge_generation": {"source": "qwen"},
            "candidate_challenges": [second_challenge.to_dict()],
            "alternative_lane_resolutions": [
                item.to_dict() for item in second_resolutions
            ],
        },
    )

    completed = _update_competition_state(updated, second_finding)

    assert completed.competition_trace is not None
    assert completed.competition_trace.alternative_search_status == (
        AlternativeSearchStatus.ALTERNATIVES_ELIMINATED.value
    )
    assert set(completed.competition_trace.eliminated_lane_ids) == {
        "LANE_B",
        "LANE_C",
    }
    assert set(completed.competition_trace.resolution_evidence_ids) == {
        "EV_B",
        "EV_C",
    }


def test_supervisor_persists_candidate_competition_failure_without_challenge() -> None:
    state = RCAState(
        job=RCAJob(job_id="JOB_265_FAILURE", user_query="test"),
        evidence=[_evidence("EV_A", "LANE_A")],
        causal_lanes=[CausalLaneRecord(lane_id="LANE_A", priority_score=0.9)],
    )
    finding = AgentFinding(
        finding_id="F_265_FAILURE",
        agent="rca_reasoning",
        summary="A second evidence-bounded direction was not proposed.",
        confidence=0.0,
        evidence_ids=["EV_A"],
        details={
            "adversarial_challenge_generation": {"source": "not_requested"},
            "candidate_challenges": [],
            "competition_requirement": (
                CompetitionRequirement.DIRECTION_REQUIRED.value
            ),
            "competition_status": CandidateCompetitionStatus.FAILED.value,
            "competition_type": CandidateCompetitionType.CAUSAL_DIRECTION.value,
            "competition_failure_reason": (
                CompetitionFailureReason.ALTERNATIVE_DIRECTION_NOT_GENERATED.value
            ),
            "candidate_lineage": [
                {
                    "candidate_index": 0,
                    "lineage_status": "retained",
                }
            ],
        },
    )

    updated = _update_competition_state(state, finding)

    assert updated.competition_trace is not None
    assert updated.competition_trace.competition_status == (
        CandidateCompetitionStatus.FAILED.value
    )
    assert updated.competition_trace.competition_failure_reason == (
        CompetitionFailureReason.ALTERNATIVE_DIRECTION_NOT_GENERATED.value
    )
    assert updated.competition_trace.candidate_lineage[0]["lineage_status"] == (
        "retained"
    )
