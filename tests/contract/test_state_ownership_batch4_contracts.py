"""Batch 4 contracts: the Finalizer is the sole production authority writer.

These tests prove that governed terminals populate ``AuthoritativeRCAResult``
and ``ImpactPublicationResult`` from existing confirmed governance results,
that the Impact Gate publication decision is re-evaluated against the terminal
conclusion, and that native controlled and non-competition paths stay
compatible.
"""

from __future__ import annotations

from yield_rca_core.causal_confirmation import derive_impact_publication_result
from yield_rca_core.causal_investigation_models import (
    CandidateCompetitionStatus,
    CandidateCompetitionType,
    CompetitionRequirement,
    CompetitionTrace,
)
from yield_rca_core.evidence_models import Evidence, EvidenceSourceType
from yield_rca_core.investigation_finalizer import (
    finalize_investigation,
    finalize_non_competition_llm_react_terminal,
)
from yield_rca_core.investigation_models import (
    ConclusionLevel,
    GoalStatus,
    StopReason,
)
from yield_rca_core.models import AgentFinding, AgentKind, Hypothesis, RCAJob, RCAState

EVIDENCE_ID = "EV_BATCH4_AUTHORITY"
FINDING_ID = "FINDING_BATCH4_AUTHORITY"
HYPOTHESIS_ID = "HYP_BATCH4_AUTHORITY"


def _evidence() -> Evidence:
    return Evidence(
        evidence_id=EVIDENCE_ID,
        source_type=EvidenceSourceType.SYSTEM.value,
        source_id="batch4:authority",
        summary="Synthetic Evidence grounding the Batch 4 authority contract.",
    )


def _finding(*, conclusion_status: str, extra_details: dict | None = None) -> AgentFinding:
    details: dict = {
        "conclusion_status": conclusion_status,
        "status": conclusion_status,
        "confirmation_gate": {"status": conclusion_status},
        "root_cause_evidence_ids": [EVIDENCE_ID],
        "ranked_candidates": [
            {
                "candidate_id": "cand_1",
                "status": "candidate",
                "supporting_evidence_ids": [EVIDENCE_ID],
                "contradicting_evidence_ids": [],
                "causal_evidence_matrix": {},
            }
        ],
    }
    if conclusion_status == "supported":
        details["root_cause"] = "Chamber process drift caused polymer residue"
    if extra_details:
        details.update(extra_details)
    return AgentFinding(
        finding_id=FINDING_ID,
        agent=AgentKind.RCA_REASONING.value,
        summary="An evidence-bounded RCA result retained for terminal governance.",
        confidence=0.7,
        evidence_ids=[EVIDENCE_ID],
        details=details,
    )


def _hypothesis(*, status: str) -> Hypothesis:
    return Hypothesis(
        hypothesis_id=HYPOTHESIS_ID,
        root_cause="Chamber process drift caused polymer residue",
        confidence=0.7,
        evidence_ids=[EVIDENCE_ID],
        status=status,
    )


def _competition_trace() -> CompetitionTrace:
    return CompetitionTrace(
        competition_requirement=CompetitionRequirement.MECHANISM_REQUIRED.value,
        competition_status=CandidateCompetitionStatus.ACTIVE.value,
        competition_type=CandidateCompetitionType.MECHANISM.value,
    )


def _state(
    *,
    finding: AgentFinding,
    hypothesis: Hypothesis | None = None,
    trace: CompetitionTrace | None = None,
) -> RCAState:
    return RCAState(
        job=RCAJob(job_id="JOB_BATCH4", user_query="Finalize the RCA state."),
        evidence=[_evidence()],
        findings=[finding],
        hypotheses=[hypothesis] if hypothesis is not None else [],
        competition_trace=trace,
        authoritative_rca_finding_id=finding.finding_id,
        authoritative_hypothesis_id=(
            hypothesis.hypothesis_id if hypothesis is not None else None
        ),
        execution_metadata={},
    )


def _impact_gate_details(*, lot_id: str = "LOT_IMPACT_1") -> dict:
    """A reasoning-time Impact Gate snapshot that claims confirmed publication."""

    return {
        "impact_lot_gate": {
            "candidate_impact_lots": [lot_id],
            "rows": [
                {
                    "lot_id": lot_id,
                    "supporting_evidence_ids": [EVIDENCE_ID],
                }
            ],
            "confirmed_impact_lots": [lot_id],
            "publication_status": "confirmed",
        }
    }


def test_finalizer_publishes_supported_authority_and_confirmed_impact() -> None:
    state = _state(
        finding=_finding(
            conclusion_status="supported",
            extra_details=_impact_gate_details(),
        ),
        hypothesis=_hypothesis(status="supported"),
        trace=_competition_trace(),
    )

    finalized = finalize_investigation(state)

    result = finalized.authoritative_rca_result
    assert result is not None
    assert result.result_id == f"RCA_RESULT_{FINDING_ID}"
    assert result.source_finding_id == FINDING_ID
    assert result.source_hypothesis_id == HYPOTHESIS_ID
    assert result.conclusion_status == "supported"
    assert result.root_cause == "Chamber process drift caused polymer residue"
    assert result.root_cause_candidate_id == HYPOTHESIS_ID
    assert result.confirmation_status == "supported"
    assert result.competition_status == CandidateCompetitionStatus.COMPLETE_CONFIRMED.value
    assert result.terminal_reason == "confirmation_gate_supported_unique_candidate"
    assert result.evidence_refs == (EVIDENCE_ID,)

    publication = finalized.impact_publication_result
    assert publication is not None
    assert publication.rca_result_id == result.result_id
    assert publication.publication_status == "confirmed"
    assert publication.confirmed_impact_lots == ("LOT_IMPACT_1",)
    assert publication.evidence_refs == (EVIDENCE_ID,)

    assert finalized.execution_metadata["authoritative_rca_result_writer"] == (
        "python_investigation_finalizer"
    )
    assert finalized.execution_metadata["impact_publication_result_writer"] == (
        "python_impact_gate"
    )
    # Legacy compatibility projections remain intact.
    assert finalized.goal_status == GoalStatus.SATISFIED.value
    assert finalized.conclusion_level == ConclusionLevel.SUPPORTED.value
    assert finalized.stop_reason == StopReason.GOAL_SATISFIED.value
    assert finalized.authoritative_rca_finding.details["conclusion_status"] == "supported"


def test_finalizer_terminal_conclusion_governs_publication_not_reasoning_snapshot() -> None:
    state = _state(
        finding=_finding(
            conclusion_status="inconclusive",
            extra_details=_impact_gate_details(lot_id="LOT_STALE_CLAIM"),
        ),
        hypothesis=_hypothesis(status="candidate"),
        trace=_competition_trace(),
    )

    finalized = finalize_investigation(state)

    result = finalized.authoritative_rca_result
    assert result is not None
    assert result.conclusion_status == "inconclusive"
    assert result.root_cause is None
    assert result.root_cause_candidate_id is None
    assert result.competition_status == CandidateCompetitionStatus.EXHAUSTED.value
    assert result.terminal_reason == "no_high_value_action_remains"
    assert result.evidence_refs == (EVIDENCE_ID,)

    publication = finalized.impact_publication_result
    assert publication is not None
    assert publication.rca_result_id == result.result_id
    # The stale reasoning-time snapshot claimed confirmed publication; the
    # terminal conclusion governs, so confirmed Lots must stay empty.
    assert publication.publication_status == "withheld"
    assert publication.confirmed_impact_lots == ()

    assert finalized.goal_status == GoalStatus.BLOCKED.value
    assert finalized.conclusion_level == ConclusionLevel.INCONCLUSIVE.value
    assert finalized.authoritative_rca_finding.details["root_cause"] == "inconclusive"
    assert finalized.authoritative_hypothesis is not None
    assert finalized.authoritative_hypothesis.root_cause == "inconclusive"


def test_finalizer_not_applicable_terminal_writes_no_authority_objects() -> None:
    state = _state(finding=_finding(conclusion_status="inconclusive"))

    finalized = finalize_investigation(state)

    assert finalized.execution_metadata["investigation_finalizer"] == "not_applicable"
    assert finalized.authoritative_rca_result is None
    assert finalized.impact_publication_result is None
    assert finalized.findings == state.findings
    assert finalized.goal_status == state.goal_status
    assert finalized.conclusion_level == state.conclusion_level
    assert finalized.stop_reason == state.stop_reason


def test_non_competition_finalizer_writes_inconclusive_authority() -> None:
    state = _state(finding=_finding(conclusion_status="insufficient_evidence"))

    finalized = finalize_non_competition_llm_react_terminal(state)

    result = finalized.authoritative_rca_result
    assert result is not None
    assert result.conclusion_status == "insufficient_evidence"
    assert result.root_cause is None
    assert result.competition_status == "not_required"
    assert result.source_finding_id == FINDING_ID
    publication = finalized.impact_publication_result
    assert publication is not None
    assert publication.rca_result_id == result.result_id
    assert publication.publication_status == "not_evaluated"
    assert publication.confirmed_impact_lots == ()
    assert finalized.goal_status == GoalStatus.BLOCKED.value
    assert finalized.conclusion_level == ConclusionLevel.INCONCLUSIVE.value
    assert finalized.stop_reason == StopReason.NO_HIGH_VALUE_ACTION.value


def test_finalize_investigation_is_idempotent() -> None:
    state = _state(
        finding=_finding(
            conclusion_status="supported",
            extra_details=_impact_gate_details(),
        ),
        hypothesis=_hypothesis(status="supported"),
        trace=_competition_trace(),
    )

    first = finalize_investigation(state)
    second = finalize_investigation(first)

    assert second.authoritative_rca_result == first.authoritative_rca_result
    assert second.impact_publication_result == first.impact_publication_result


def test_finalized_state_round_trips_authority_objects() -> None:
    state = _state(
        finding=_finding(
            conclusion_status="supported",
            extra_details=_impact_gate_details(),
        ),
        hypothesis=_hypothesis(status="supported"),
        trace=_competition_trace(),
    )

    finalized = finalize_investigation(state)
    restored = RCAState.from_dict(finalized.to_dict())

    assert restored.authoritative_rca_result == finalized.authoritative_rca_result
    assert restored.impact_publication_result == finalized.impact_publication_result


def test_derive_impact_publication_result_maps_like_the_gate() -> None:
    confirmed = derive_impact_publication_result(
        _impact_gate_details()["impact_lot_gate"],
        rca_result_id="RCA_RESULT_X",
        conclusion_status="supported",
    )
    assert confirmed.publication_status == "confirmed"
    assert confirmed.confirmed_impact_lots == ("LOT_IMPACT_1",)
    assert confirmed.evidence_refs == (EVIDENCE_ID,)

    withheld = derive_impact_publication_result(
        _impact_gate_details()["impact_lot_gate"],
        rca_result_id="RCA_RESULT_X",
        conclusion_status="inconclusive",
    )
    assert withheld.publication_status == "withheld"
    assert withheld.confirmed_impact_lots == ()

    not_evaluated = derive_impact_publication_result(
        {},
        rca_result_id="RCA_RESULT_X",
        conclusion_status="supported",
    )
    assert not_evaluated.publication_status == "not_evaluated"
    assert not_evaluated.confirmed_impact_lots == ()

    unconfirmed = derive_impact_publication_result(
        {
            "candidate_impact_lots": [],
            "rows": [{"lot_id": "LOT_A", "supporting_evidence_ids": [EVIDENCE_ID]}],
        },
        rca_result_id="RCA_RESULT_X",
        conclusion_status="inconclusive",
    )
    assert unconfirmed.publication_status == "unconfirmed"
    assert unconfirmed.confirmed_impact_lots == ()
