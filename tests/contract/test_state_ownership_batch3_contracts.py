from __future__ import annotations

from yield_rca_core.evidence_models import Evidence, EvidenceSourceType
from yield_rca_core.investigation_finalizer import (
    finalize_non_competition_llm_react_terminal,
)
from yield_rca_core.investigation_models import ConclusionLevel, GoalStatus, StopReason
from yield_rca_core.models import AgentFinding, AgentKind, Hypothesis, RCAJob, RCAState


def _state_with_authoritative_finding(*, conclusion_status: str) -> RCAState:
    evidence = Evidence(
        evidence_id="EV_BATCH3_AUTHORITY",
        source_type=EvidenceSourceType.SYSTEM.value,
        source_id="batch3:authority",
        summary="Synthetic Evidence for the Batch 3 ownership contract.",
    )
    finding = AgentFinding(
        finding_id="FINDING_BATCH3_AUTHORITY",
        agent=AgentKind.RCA_REASONING.value,
        summary="An evidence-bounded RCA result retained for terminal governance.",
        confidence=0.7,
        evidence_ids=[evidence.evidence_id],
        details={"conclusion_status": conclusion_status},
    )
    return RCAState(
        job=RCAJob(job_id="JOB_BATCH3_AUTHORITY", user_query="Finalize the RCA state."),
        evidence=[evidence],
        findings=[finding],
        authoritative_rca_finding_id=finding.finding_id,
        goal_status=GoalStatus.SATISFIED.value,
        conclusion_level=ConclusionLevel.SUPPORTED.value,
        stop_reason=StopReason.GOAL_SATISFIED.value,
        execution_metadata={"investigation_finalizer": "not_applicable"},
    )


def _supported_non_competition_state() -> RCAState:
    evidence = Evidence(
        evidence_id="EV_BATCH3_SUPPORTED",
        source_type=EvidenceSourceType.SYSTEM.value,
        source_id="batch3:supported",
        summary="Synthetic Evidence for a supported non-competition terminal.",
    )
    finding = AgentFinding(
        finding_id="FINDING_BATCH3_SUPPORTED",
        agent=AgentKind.RCA_REASONING.value,
        summary="Root cause: supported non-competition result (confidence 80%).",
        confidence=0.8,
        evidence_ids=[evidence.evidence_id],
        details={
            "conclusion_status": "supported",
            "status": "supported",
            "root_cause": "Upstream chamber process drift",
            "confirmation_gate": {"status": "supported"},
            "root_cause_evidence_ids": [evidence.evidence_id],
        },
    )
    return RCAState(
        job=RCAJob(job_id="JOB_BATCH3_SUPPORTED", user_query="Finalize the RCA state."),
        evidence=[evidence],
        findings=[finding],
        hypotheses=[
            Hypothesis(
                hypothesis_id="HYP_BATCH3_SUPPORTED",
                root_cause="Upstream chamber process drift",
                confidence=0.8,
                evidence_ids=[evidence.evidence_id],
                status="supported",
            )
        ],
        authoritative_rca_finding_id=finding.finding_id,
        authoritative_hypothesis_id="HYP_BATCH3_SUPPORTED",
        goal_status=GoalStatus.SATISFIED.value,
        conclusion_level=ConclusionLevel.SUPPORTED.value,
        stop_reason=StopReason.GOAL_SATISFIED.value,
        execution_metadata={"investigation_finalizer": "not_applicable"},
    )


def test_finalizer_owns_non_competition_inconclusive_projection() -> None:
    state = _state_with_authoritative_finding(conclusion_status="inconclusive")

    finalized = finalize_non_competition_llm_react_terminal(state)

    assert finalized.goal_status == GoalStatus.BLOCKED.value
    assert finalized.conclusion_level == ConclusionLevel.INCONCLUSIVE.value
    assert finalized.stop_reason == StopReason.NO_HIGH_VALUE_ACTION.value
    assert finalized.planner_decisions == state.planner_decisions
    assert finalized.execution_metadata["terminal_conclusion_status_source"] == (
        "authoritative_rca_finding"
    )
    assert finalized.execution_metadata["terminal_state_owner"] == (
        "python_investigation_finalizer"
    )


def test_finalizer_does_not_downgrade_supported_non_competition_result() -> None:
    state = _supported_non_competition_state()

    finalized = finalize_non_competition_llm_react_terminal(state)

    assert finalized.goal_status == GoalStatus.SATISFIED.value
    assert finalized.conclusion_level == ConclusionLevel.SUPPORTED.value
    assert finalized.stop_reason == StopReason.GOAL_SATISFIED.value
    assert finalized.findings == state.findings
    assert finalized.planner_decisions == state.planner_decisions
    result = finalized.authoritative_rca_result
    assert result is not None
    assert result.conclusion_status == "supported"
    assert result.root_cause == "Upstream chamber process drift"
    assert result.root_cause_candidate_id == "HYP_BATCH3_SUPPORTED"
    assert result.source_finding_id == "FINDING_BATCH3_SUPPORTED"
    assert result.competition_status == "not_required"
    publication = finalized.impact_publication_result
    assert publication is not None
    assert publication.rca_result_id == result.result_id
    assert publication.confirmed_impact_lots == ()
