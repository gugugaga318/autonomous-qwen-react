from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core"))

from yield_rca_core.evidence_models import Evidence, EvidenceSourceType  # noqa: E402
from yield_rca_core.investigation_finalizer import (  # noqa: E402
    gate_planner_conclusion_level,
)
from yield_rca_core.investigation_models import (  # noqa: E402
    ConclusionLevel,
    InvestigationGoal,
    InvestigationIntent,
)
from yield_rca_core.models import (  # noqa: E402
    Hypothesis,
    HypothesisStatus,
    RCAJob,
    RCAState,
)


def _hypothesis(hypothesis_id: str, status: str) -> Hypothesis:
    return Hypothesis(
        hypothesis_id=hypothesis_id,
        root_cause=f"Root cause for {hypothesis_id}",
        confidence=0.8,
        evidence_ids=["EV_GATE"],
        status=status,
    )


def _state(*, historical_status: str, authoritative_status: str) -> RCAState:
    historical = _hypothesis("HYP_HISTORICAL", historical_status)
    authoritative = _hypothesis("HYP_AUTHORITATIVE", authoritative_status)
    return RCAState(
        job=RCAJob(job_id="JOB_GATE", user_query="Investigate the root cause."),
        evidence=[
            Evidence(
                evidence_id="EV_GATE",
                source_type=EvidenceSourceType.MES.value,
                source_id="gate:test",
                summary="One bounded investigation observation.",
            )
        ],
        hypotheses=[historical, authoritative],
        authoritative_hypothesis_id=authoritative.hypothesis_id,
    )


def _goal() -> InvestigationGoal:
    return InvestigationGoal(
        goal_id="GOAL_GATE",
        intent=InvestigationIntent.ROOT_CAUSE.value,
        summary="Investigate the root cause.",
    )


def test_historical_supported_hypothesis_cannot_raise_current_conclusion() -> None:
    state = _state(
        historical_status=HypothesisStatus.SUPPORTED.value,
        authoritative_status=HypothesisStatus.INCONCLUSIVE.value,
    )

    assert gate_planner_conclusion_level(
        ConclusionLevel.SUPPORTED.value,
        state=state,
        goal=_goal(),
    ) == ConclusionLevel.INCONCLUSIVE.value


def test_historical_conflict_cannot_override_current_supported_hypothesis() -> None:
    state = _state(
        historical_status=HypothesisStatus.CONFLICTED.value,
        authoritative_status=HypothesisStatus.SUPPORTED.value,
    )

    assert gate_planner_conclusion_level(
        ConclusionLevel.SUPPORTED.value,
        state=state,
        goal=_goal(),
    ) == ConclusionLevel.SUPPORTED.value
