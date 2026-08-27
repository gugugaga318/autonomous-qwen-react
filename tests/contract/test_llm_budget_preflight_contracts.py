from __future__ import annotations

from yield_rca_core.causal_adversarial import QwenAdversarialChallenger
from yield_rca_core.causal_evidence_matrix import CausalClaimResult, CausalEvidenceMatrix
from yield_rca_core.causal_hypothesis import CausalHypothesis
from yield_rca_core.investigation_models import (
    ConclusionLevel,
    DecisionType,
    GoalStatus,
    InvestigationGoal,
    InvestigationIntent,
    PlannerDecision,
    StopReason,
)
from yield_rca_core.llm_gateway import (
    FakeLLMClient,
    LLMCallError,
    LLMRequest,
    LLMResponse,
    llm_call_budget_available,
    llm_calls_remaining,
)
from yield_rca_core.models import RCAJob, RCAState
from yield_rca_core.supervisor import (
    _align_terminal_planner_stop,
    _rca_reasoning_round_budget_available,
)


class BudgetedInvalidClient:
    provider = "fake"
    model = "fake-json"

    def __init__(self, *, max_calls: int) -> None:
        self.max_calls = max_calls
        self.call_count = 0
        self.limit_exceeded = False
        self.delegate = FakeLLMClient()

    @property
    def remaining_calls(self) -> int:
        return max(0, self.max_calls - self.call_count)

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        if self.call_count >= self.max_calls:
            self.limit_exceeded = True
            raise LLMCallError("cap exceeded", failure_category="call_limit")
        self.call_count += 1
        base = self.delegate.complete_json(request)
        return LLMResponse(data={"invalid": True}, usage=base.usage)


def _matrix() -> CausalEvidenceMatrix:
    return CausalEvidenceMatrix(
        candidate=CausalHypothesis(
            root_cause="A bounded causal candidate.",
            causal_explanation="A physical mechanism may produce the outcome.",
            supporting_evidence_ids=("EV_A",),
        ),
        claims={
            "scope": CausalClaimResult(claim="scope", status="incomplete"),
            "mechanism": CausalClaimResult(
                claim="mechanism",
                status="incomplete",
            ),
        },
    )


def test_budget_capability_is_optional_and_non_consuming() -> None:
    client = BudgetedInvalidClient(max_calls=3)

    assert llm_calls_remaining(client) == 3
    assert llm_call_budget_available(
        client,
        required_calls=1,
        reserve_calls=2,
    )
    assert client.call_count == 0


def test_optional_challenge_repair_does_not_attempt_the_last_reserved_call() -> None:
    client = BudgetedInvalidClient(max_calls=2)

    result = QwenAdversarialChallenger(client).generate(
        request_id="REQ_BUDGET",
        candidates=[
            {
                "candidate_id": "CANDIDATE_A",
                "root_cause": "A bounded causal candidate.",
                "causal_explanation": "A mechanism may produce the outcome.",
            }
        ],
        matrices=[_matrix()],
        evidence_gaps=[],
        evidence_ids=[],
        lane_ids=["LANE_A"],
        active_lane_ids=["LANE_A"],
        lane_contexts=[{"lane_id": "LANE_A"}],
    )

    assert client.call_count == 1
    assert client.limit_exceeded is False
    assert result.attempt_count == 1
    assert result.output_invalid is False
    assert result.repair_skipped_due_to_budget is True


def test_rca_round_requires_generation_challenge_and_post_action_reserve() -> None:
    assert not _rca_reasoning_round_budget_available(
        BudgetedInvalidClient(max_calls=2)
    )
    assert _rca_reasoning_round_budget_available(
        BudgetedInvalidClient(max_calls=3)
    )


def test_terminal_projection_preserves_audit_and_aligns_effective_stop() -> None:
    goal = InvestigationGoal(
        goal_id="GOAL_BUDGET",
        intent=InvestigationIntent.ROOT_CAUSE,
        summary="Investigate the root cause.",
    )
    original = PlannerDecision(
        decision_id="DECISION_ORIGINAL",
        goal_id=goal.goal_id,
        decision_type=DecisionType.STOP,
        reason="The action budget ended.",
        goal_status=GoalStatus.BUDGET_EXHAUSTED,
        proposed_conclusion_level=ConclusionLevel.INCONCLUSIVE,
        stop_reason=StopReason.BUDGET_EXHAUSTED,
    )
    state = RCAState(
        job=RCAJob(job_id="JOB_BUDGET", user_query="Investigate."),
        investigation_goal=goal,
        planner_decisions=[original],
        goal_status=GoalStatus.BLOCKED,
        conclusion_level=ConclusionLevel.INCONCLUSIVE,
        stop_reason=StopReason.NO_HIGH_VALUE_ACTION,
    )

    aligned = _align_terminal_planner_stop(state)

    assert len(aligned.planner_decisions) == 1
    assert aligned.planner_decisions[-1].stop_reason == aligned.stop_reason
    assert aligned.planner_decisions[-1].goal_status == aligned.goal_status
    assert aligned.execution_metadata["planner_stop_proposed_by"] == "python_runtime"
    assert (
        aligned.execution_metadata["superseded_terminal_planner_decision"]
        == original.to_dict()
    )
