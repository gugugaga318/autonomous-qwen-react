"""Shared Planner interface for the unified React orchestration loop.

Batch 5 convergence contract. The loop owns orchestration; planners only
select the next legal action or propose a stop. Two implementations exist:

- ``QwenPlannerAdapter`` wraps ``QwenNextActionPlanner.decide_with_review``.
- ``ControlledPlannerAdapter`` wraps the deterministic ``InvestigationPolicy``.

Ownership matrix for ``PlannerContext`` fields (from the pre-refactor audit;
adapters must consume only these fields, never additional State reads):

=================================  ======  ==========  =============
Field                              Qwen    Controlled  Ownership
=================================  ======  ==========  =============
goal                               yes     yes         shared
findings                           yes     yes         shared
action_records                     yes     yes         shared
tool_call_count                    yes     yes         shared
questions                          yes     no          qwen-only
evidence / evidence_ids            yes     no          qwen-only
question_evidence_links            yes     no          qwen-only
capability_notices                 yes     no          qwen-only
hypotheses                         yes     no          qwen-only
prior_decisions                    yes     no          qwen-only
investigation_gain_history         yes     no          qwen-only
authoritative_rca_finding_id       yes     no          qwen-only
critical_contradictions            yes     no          qwen-only
=================================  ======  ==========  =============

``critical_contradictions`` deserves a pin: the pre-refactor controlled loop
never forwarded it to ``InvestigationPolicy.next_action``, so the controlled
adapter must keep not forwarding it (contract-tested). A conflicted
hypothesis still terminates the controlled policy through the authoritative
RCA Finding status, so semantics are unchanged.

Runtime budget/termination decisions (LLM call budget, RCA round budget) are
NOT planner proposals. They stay on the orchestration/runtime side and must
never be expressed as a ``PlannerProposal``; ``proposed_by`` therefore only
admits ``llm_qwen`` and ``python_controlled``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from yield_rca_core.causal_investigation_models import (
    ActionValueAssessment,
    InvestigationGainRecord,
)
from yield_rca_core.evidence_models import Evidence
from yield_rca_core.investigation_models import (
    ActionRecord,
    CapabilityNotice,
    ConclusionLevel,
    DecisionType,
    GoalStatus,
    InvestigationAction,
    InvestigationGoal,
    InvestigationQuestion,
    PlannerDecision,
    QuestionEvidenceLink,
    QuestionUpdate,
    QuestionUpdateReview,
    StopReason,
)
from yield_rca_core.investigation_policy import InvestigationPolicy, PolicyDecision
from yield_rca_core.models import (
    AgentFinding,
    Hypothesis,
    HypothesisStatus,
    RCAState,
)

PROVENANCE_LLM_QWEN = "llm_qwen"
PROVENANCE_CONTROLLED = "python_controlled"
_ALLOWED_PROVENANCE = {PROVENANCE_LLM_QWEN, PROVENANCE_CONTROLLED}


@dataclass(frozen=True)
class PlannerContext:
    """The full per-iteration observation bundle a planner may consume.

    The loop builds exactly one context per iteration from the shared State.
    Adapters forward only the fields their native planner reads (matrix above).
    """

    goal: InvestigationGoal
    findings: list[AgentFinding]
    action_records: list[ActionRecord]
    tool_call_count: int
    questions: list[InvestigationQuestion] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    question_evidence_links: list[QuestionEvidenceLink] = field(default_factory=list)
    capability_notices: list[CapabilityNotice] = field(default_factory=list)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    prior_decisions: list[PlannerDecision] = field(default_factory=list)
    investigation_gain_history: list[InvestigationGainRecord] = field(
        default_factory=list
    )
    critical_contradictions: list[str] = field(default_factory=list)
    authoritative_rca_finding_id: str | None = None


@dataclass(frozen=True)
class PlannerProposal:
    """One planner outcome expressed in shared loop vocabulary.

    ``native_decision`` preserves the untouched original decision for audit
    and compatibility. The shared loop must branch only on the public fields
    of this proposal, never on the concrete native decision type.
    """

    decision_type: str
    reason: str
    proposed_by: str
    next_action: InvestigationAction | None = None
    proposed_goal_status: str | None = None
    proposed_conclusion_level: str = ConclusionLevel.INCONCLUSIVE.value
    stop_reason: str | None = None
    decision_id: str | None = None
    goal_id: str | None = None
    target_question_ids: list[str] = field(default_factory=list)
    evidence_gaps: list[str] = field(default_factory=list)
    question_updates: list[QuestionUpdate] = field(default_factory=list)
    new_questions: list[InvestigationQuestion] = field(default_factory=list)
    action_value_assessments: list[ActionValueAssessment] = field(default_factory=list)
    question_update_reviews: list[QuestionUpdateReview] = field(default_factory=list)
    # Audit provenance mirrored from the pre-refactor PlannerDecisionOutcome;
    # the shared loop only reads these to rebuild the audit record.
    decision_proposed_by: str | None = None
    question_updates_source: str | None = None
    raw_question_update_count: int = 0
    native_decision: PlannerDecision | PolicyDecision | None = None

    def __post_init__(self) -> None:
        if self.proposed_by not in _ALLOWED_PROVENANCE:
            allowed = ", ".join(sorted(_ALLOWED_PROVENANCE))
            raise ValueError(
                f"PlannerProposal.proposed_by must be one of: {allowed}; "
                "runtime budget/termination is not a planner decision"
            )
        if self.decision_type == DecisionType.ACT.value:
            if self.next_action is None:
                raise ValueError("an act proposal requires next_action")
            if self.stop_reason is not None:
                raise ValueError("an act proposal cannot include stop_reason")
            if self.proposed_goal_status != GoalStatus.IN_PROGRESS.value:
                raise ValueError("an act proposal requires goal_status=in_progress")
        elif self.decision_type == DecisionType.STOP.value:
            if self.next_action is not None:
                raise ValueError("a stop proposal cannot include next_action")
            if self.stop_reason is None:
                raise ValueError("a stop proposal requires stop_reason")
            StopReason(self.stop_reason)
            if self.proposed_goal_status == GoalStatus.IN_PROGRESS.value:
                raise ValueError("a stop proposal cannot leave the goal in progress")
        else:
            raise ValueError(
                f"decision_type must be act or stop, got {self.decision_type!r}"
            )
        if not self.reason.strip():
            raise ValueError("a proposal requires a non-empty reason")


@runtime_checkable
class InvestigationPlanner(Protocol):
    """The single planner seam the unified React loop depends on."""

    def decide(self, context: PlannerContext) -> PlannerProposal:
        """Return one legal action selection or stop proposal."""


class QwenPlannerAdapter:
    """Translate ``QwenNextActionPlanner`` outcomes into the shared contract.

    Planner validation and provider errors pass through untouched so the loop
    keeps the real failure telemetry and fallback semantics.
    """

    def __init__(self, planner: Any) -> None:
        self._planner = planner

    @property
    def llm_client(self) -> Any:
        return self._planner.llm_client

    def decide(self, context: PlannerContext) -> PlannerProposal:
        outcome = self._planner.decide_with_review(
            goal=context.goal,
            questions=context.questions,
            findings=context.findings,
            action_records=context.action_records,
            tool_call_count=context.tool_call_count,
            evidence=context.evidence,
            evidence_ids=context.evidence_ids,
            question_evidence_links=context.question_evidence_links,
            capability_notices=context.capability_notices,
            hypotheses=context.hypotheses,
            prior_decisions=context.prior_decisions,
            critical_contradictions=context.critical_contradictions,
            authoritative_rca_finding_id=context.authoritative_rca_finding_id,
            investigation_gain_history=context.investigation_gain_history,
        )
        decision = outcome.decision
        return PlannerProposal(
            decision_type=decision.decision_type,
            reason=decision.reason,
            proposed_by=PROVENANCE_LLM_QWEN,
            next_action=decision.next_action,
            proposed_goal_status=decision.goal_status,
            proposed_conclusion_level=decision.proposed_conclusion_level,
            stop_reason=decision.stop_reason,
            decision_id=decision.decision_id,
            goal_id=decision.goal_id,
            target_question_ids=list(decision.target_question_ids),
            question_updates=list(decision.question_updates),
            new_questions=list(decision.new_questions),
            action_value_assessments=list(outcome.action_value_assessments),
            question_update_reviews=list(outcome.question_update_reviews),
            decision_proposed_by=outcome.decision_proposed_by,
            question_updates_source=outcome.question_updates_source,
            raw_question_update_count=outcome.raw_question_update_count,
            native_decision=decision,
        )


class ControlledPlannerAdapter:
    """Translate deterministic ``InvestigationPolicy`` results.

    The adapter never swallows or alters policy outcomes, and deliberately
    does not forward ``critical_contradictions`` (ownership matrix pin).
    """

    def __init__(self, policy: InvestigationPolicy | None = None) -> None:
        self._policy = policy if policy is not None else InvestigationPolicy()

    @property
    def policy(self) -> InvestigationPolicy:
        return self._policy

    def decide(self, context: PlannerContext) -> PlannerProposal:
        decision = self._policy.next_action(
            goal=context.goal,
            findings=context.findings,
            action_records=context.action_records,
            tool_call_count=context.tool_call_count,
        )
        is_act = decision.next_action is not None
        return PlannerProposal(
            decision_type=(
                DecisionType.ACT.value if is_act else DecisionType.STOP.value
            ),
            reason=(
                decision.next_action.reason
                if is_act
                else str(decision.stop_reason)
            ),
            proposed_by=PROVENANCE_CONTROLLED,
            next_action=decision.next_action,
            proposed_goal_status=decision.goal_status,
            proposed_conclusion_level=decision.conclusion_level,
            stop_reason=decision.stop_reason,
            evidence_gaps=list(decision.evidence_gaps),
            native_decision=decision,
        )


def build_planner_context(
    *,
    goal: InvestigationGoal,
    state: RCAState,
    tool_call_count: int,
) -> PlannerContext:
    """Build the per-iteration observation bundle from the shared State.

    Kept on the loop side of the boundary so planners never read State
    directly; the field set mirrors the pre-refactor ``decide_with_review``
    call exactly.
    """

    authoritative_hypothesis = getattr(state, "authoritative_hypothesis", None)
    critical_contradictions = (
        [
            f"{authoritative_hypothesis.hypothesis_id}: "
            f"{authoritative_hypothesis.root_cause}"
        ]
        if authoritative_hypothesis is not None
        and authoritative_hypothesis.status == HypothesisStatus.CONFLICTED.value
        else []
    )
    return PlannerContext(
        goal=goal,
        findings=list(state.findings),
        action_records=list(state.action_history),
        tool_call_count=tool_call_count,
        questions=list(state.investigation_questions),
        evidence=list(state.evidence),
        evidence_ids=[item.evidence_id for item in state.evidence],
        question_evidence_links=list(state.question_evidence_links),
        capability_notices=list(state.capability_notices),
        hypotheses=list(state.hypotheses),
        prior_decisions=list(state.planner_decisions),
        investigation_gain_history=list(state.investigation_gain_history),
        critical_contradictions=list(critical_contradictions),
        authoritative_rca_finding_id=state.authoritative_rca_finding_id,
    )


__all__ = [
    "ControlledPlannerAdapter",
    "InvestigationPlanner",
    "PlannerContext",
    "PlannerProposal",
    "PROVENANCE_CONTROLLED",
    "PROVENANCE_LLM_QWEN",
    "QwenPlannerAdapter",
    "build_planner_context",
]
