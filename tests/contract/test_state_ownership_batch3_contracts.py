from __future__ import annotations

from yield_rca_core.evidence_models import Evidence, EvidenceSourceType
from yield_rca_core.investigation_finalizer import (
    finalize_non_competition_llm_react_terminal,
)
from yield_rca_core.investigation_models import ConclusionLevel, GoalStatus, StopReason
from yield_rca_core.models import AgentFinding, AgentKind, RCAJob, RCAState


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
    state = _state_with_authoritative_finding(conclusion_status="supported")

    assert finalize_non_competition_llm_react_terminal(state) == state
