"""Batch 5 planner interface contracts.

Pins the shared planner seam: faithful adapter translation, the provenance
restriction (runtime termination is not a planner), the controlled adapter's
ownership-matrix pin (no critical_contradictions forwarding), pass-through of
planner errors, and the native-decision opacity requirement.
"""

from __future__ import annotations

import unittest

from yield_rca_core.causal_investigation_models import ActionValueAssessment
from yield_rca_core.evidence_models import Evidence, EvidenceSourceType
from yield_rca_core.investigation_models import (
    ConclusionLevel,
    DecisionType,
    GoalStatus,
    InvestigationAction,
    InvestigationGoal,
    PlannerDecision,
    PlannerDecisionOutcome,
    StopReason,
)
from yield_rca_core.investigation_policy import PolicyDecision
from yield_rca_core.models import AgentFinding, AgentKind, RCAJob, RCAState
from yield_rca_core.next_action_planner import QwenNextActionPlannerError
from yield_rca_core.planner_interface import (
    PROVENANCE_CONTROLLED,
    PROVENANCE_LLM_QWEN,
    ControlledPlannerAdapter,
    PlannerContext,
    PlannerProposal,
    QwenPlannerAdapter,
    build_planner_context,
)


class RecordingPolicy:
    """Spy policy capturing exactly what the adapter forwards."""

    def __init__(self, decision: PolicyDecision) -> None:
        self.decision = decision
        self.calls: list[dict] = []

    def next_action(self, **kwargs: object) -> PolicyDecision:
        self.calls.append(dict(kwargs))
        return self.decision


class FakeQwenPlanner:
    def __init__(self, outcome: PlannerDecisionOutcome | Exception) -> None:
        self.outcome = outcome
        self.llm_client = object()
        self.calls: list[dict] = []

    def decide_with_review(self, **kwargs: object) -> PlannerDecisionOutcome:
        self.calls.append(dict(kwargs))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _goal() -> InvestigationGoal:
    return InvestigationGoal(
        goal_id="GOAL_PLANNER_INTERFACE",
        intent="root_cause",
        summary="Investigate the root cause.",
        max_steps=8,
        max_tool_calls=8,
    )


def _state() -> RCAState:
    return RCAState(
        job=RCAJob(job_id="JOB_PLANNER_INTERFACE", user_query="Investigate."),
    )


def _action() -> InvestigationAction:
    return InvestigationAction(
        action_id="GOAL_PLANNER_INTERFACE:find_shared_exposure",
        kind="find_shared_exposure",
        agent="mes",
        reason="Establish shared exposure first.",
        inputs={},
        required_evidence_ids=[],
    )


def _context() -> PlannerContext:
    return PlannerContext(
        goal=_goal(),
        findings=[],
        action_records=[],
        tool_call_count=0,
    )


class ControlledAdapterContractTest(unittest.TestCase):
    def test_act_proposal_translates_policy_decision_faithfully(self) -> None:
        action = _action()
        decision = PolicyDecision(
            goal_status=GoalStatus.IN_PROGRESS.value,
            conclusion_level=ConclusionLevel.SIGNAL.value,
            next_action=action,
            evidence_gaps=["gap_a"],
        )
        policy = RecordingPolicy(decision)

        proposal = ControlledPlannerAdapter(policy).decide(_context())  # type: ignore[arg-type]

        self.assertEqual(proposal.decision_type, DecisionType.ACT.value)
        self.assertIs(proposal.next_action, action)
        self.assertEqual(proposal.proposed_goal_status, GoalStatus.IN_PROGRESS.value)
        self.assertEqual(
            proposal.proposed_conclusion_level,
            ConclusionLevel.SIGNAL.value,
        )
        self.assertIsNone(proposal.stop_reason)
        self.assertEqual(proposal.evidence_gaps, ["gap_a"])
        self.assertEqual(proposal.proposed_by, PROVENANCE_CONTROLLED)
        self.assertIs(proposal.native_decision, decision)
        self.assertEqual(proposal.reason, action.reason)
        self.assertEqual(len(policy.calls), 1)

    def test_stop_proposal_translates_policy_decision_faithfully(self) -> None:
        decision = PolicyDecision(
            goal_status=GoalStatus.BLOCKED.value,
            conclusion_level=ConclusionLevel.INCONCLUSIVE.value,
            next_action=None,
            evidence_gaps=["gap_b"],
            stop_reason=StopReason.NO_HIGH_VALUE_ACTION.value,
        )
        policy = RecordingPolicy(decision)

        proposal = ControlledPlannerAdapter(policy).decide(_context())  # type: ignore[arg-type]

        self.assertEqual(proposal.decision_type, DecisionType.STOP.value)
        self.assertIsNone(proposal.next_action)
        self.assertEqual(
            proposal.stop_reason,
            StopReason.NO_HIGH_VALUE_ACTION.value,
        )
        self.assertEqual(proposal.proposed_goal_status, GoalStatus.BLOCKED.value)
        self.assertEqual(
            proposal.proposed_conclusion_level,
            ConclusionLevel.INCONCLUSIVE.value,
        )
        self.assertEqual(proposal.proposed_by, PROVENANCE_CONTROLLED)

    def test_controlled_adapter_never_forwards_critical_contradictions(self) -> None:
        decision = PolicyDecision(
            goal_status=GoalStatus.SATISFIED.value,
            conclusion_level=ConclusionLevel.CANDIDATE.value,
            next_action=None,
            evidence_gaps=[],
            stop_reason=StopReason.GOAL_SATISFIED.value,
        )
        policy = RecordingPolicy(decision)
        context = PlannerContext(
            goal=_goal(),
            findings=[],
            action_records=[],
            tool_call_count=0,
            critical_contradictions=["HYP_1: conflicted root cause"],
        )

        ControlledPlannerAdapter(policy).decide(context)  # type: ignore[arg-type]

        # Ownership matrix pin: the pre-refactor controlled loop passed only
        # the shared fields to the policy; the adapter must not silently add
        # critical_contradictions (or any Qwen-only input) to that contract.
        self.assertEqual(
            set(policy.calls[0]),
            {"goal", "findings", "action_records", "tool_call_count"},
        )


class QwenAdapterContractTest(unittest.TestCase):
    def test_proposal_translates_planner_outcome_faithfully(self) -> None:
        decision = PlannerDecision(
            decision_id="DEC_1",
            goal_id="GOAL_PLANNER_INTERFACE",
            decision_type=DecisionType.STOP.value,
            reason="No legal action remains.",
            goal_status=GoalStatus.BLOCKED.value,
            proposed_conclusion_level=ConclusionLevel.SUPPORTED.value,
            stop_reason=StopReason.NO_HIGH_VALUE_ACTION.value,
        )
        assessment = ActionValueAssessment(
            option_id="OPT_1",
            action_kind="find_shared_exposure",
            scope_fingerprint="fp",
            static_information_gain=0.5,
            source_availability="available",
            decision_impact="ranking",
            estimated_tool_cost=1,
            estimated_llm_cost=0,
            remaining_tool_budget=3,
            high_value=False,
            eligible=True,
            value_tier="low",
        )
        outcome = PlannerDecisionOutcome(
            decision=decision,
            decision_proposed_by="qwen",
            action_value_assessments=[assessment],
        )
        planner = FakeQwenPlanner(outcome)

        proposal = QwenPlannerAdapter(planner).decide(_context())  # type: ignore[arg-type]

        self.assertEqual(proposal.decision_type, DecisionType.STOP.value)
        self.assertEqual(proposal.decision_id, "DEC_1")
        self.assertEqual(proposal.reason, "No legal action remains.")
        self.assertEqual(proposal.proposed_by, PROVENANCE_LLM_QWEN)
        self.assertEqual(proposal.proposed_goal_status, GoalStatus.BLOCKED.value)
        self.assertEqual(
            proposal.proposed_conclusion_level,
            ConclusionLevel.SUPPORTED.value,
        )
        self.assertEqual(
            proposal.stop_reason,
            StopReason.NO_HIGH_VALUE_ACTION.value,
        )
        self.assertEqual(proposal.action_value_assessments, [assessment])
        self.assertIs(proposal.native_decision, decision)
        # The context fields are forwarded to decide_with_review untouched.
        self.assertIn("critical_contradictions", planner.calls[0])

    def test_planner_errors_pass_through_untouched(self) -> None:
        error = QwenNextActionPlannerError(
            ["missing decision_type"],
            ["core"],
            goal_id="GOAL_PLANNER_INTERFACE",
            completed_steps=1,
            tool_call_count=1,
        )
        planner = FakeQwenPlanner(error)

        with self.assertRaises(QwenNextActionPlannerError) as raised:
            QwenPlannerAdapter(planner).decide(_context())  # type: ignore[arg-type]

        self.assertIs(raised.exception, error)


class ProposalProvenanceContractTest(unittest.TestCase):
    def test_runtime_termination_is_not_a_planner_proposal(self) -> None:
        with self.assertRaises(ValueError) as raised:
            PlannerProposal(
                decision_type=DecisionType.STOP.value,
                reason="LLM call budget exhausted.",
                proposed_by="python_runtime_budget",
                proposed_goal_status=GoalStatus.BUDGET_EXHAUSTED.value,
                stop_reason=StopReason.BUDGET_EXHAUSTED.value,
            )
        self.assertIn("runtime", str(raised.exception))

    def test_unknown_provenance_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PlannerProposal(
                decision_type=DecisionType.STOP.value,
                reason="Unknown origin.",
                proposed_by="scheduler",
                proposed_goal_status=GoalStatus.BLOCKED.value,
                stop_reason=StopReason.NO_HIGH_VALUE_ACTION.value,
            )

    def test_act_and_stop_shapes_mirror_planner_decision_rules(self) -> None:
        with self.assertRaises(ValueError):
            PlannerProposal(
                decision_type=DecisionType.ACT.value,
                reason="Missing action.",
                proposed_by=PROVENANCE_CONTROLLED,
                proposed_goal_status=GoalStatus.IN_PROGRESS.value,
            )
        with self.assertRaises(ValueError):
            PlannerProposal(
                decision_type=DecisionType.STOP.value,
                reason="Missing stop reason.",
                proposed_by=PROVENANCE_CONTROLLED,
                proposed_goal_status=GoalStatus.BLOCKED.value,
            )
        with self.assertRaises(ValueError):
            PlannerProposal(
                decision_type=DecisionType.STOP.value,
                reason="Stop cannot keep the goal in progress.",
                proposed_by=PROVENANCE_CONTROLLED,
                proposed_goal_status=GoalStatus.IN_PROGRESS.value,
                stop_reason=StopReason.NO_HIGH_VALUE_ACTION.value,
            )

    def test_loop_can_run_on_public_fields_alone(self) -> None:
        # Native-decision opacity: the shared loop must not require the native
        # decision object, so a proposal without one stays fully usable.
        proposal = PlannerProposal(
            decision_type=DecisionType.STOP.value,
            reason="Everything needed is public.",
            proposed_by=PROVENANCE_CONTROLLED,
            proposed_goal_status=GoalStatus.BLOCKED.value,
            stop_reason=StopReason.NO_HIGH_VALUE_ACTION.value,
            native_decision=None,
        )
        self.assertIsNone(proposal.native_decision)


class BuildPlannerContextContractTest(unittest.TestCase):
    def test_context_mirrors_the_pre_refactor_observation_bundle(self) -> None:
        evidence = Evidence(
            evidence_id="EV_CTX",
            source_type=EvidenceSourceType.SYSTEM.value,
            source_id="planner_interface:ctx",
            summary="Synthetic Evidence for the context contract.",
        )
        finding = AgentFinding(
            finding_id="FINDING_CTX",
            agent=AgentKind.MES.value,
            summary="Exposure recorded.",
            confidence=0.5,
            evidence_ids=[evidence.evidence_id],
        )
        state = RCAState(
            job=RCAJob(job_id="JOB_PLANNER_INTERFACE", user_query="Investigate."),
            evidence=[evidence],
            findings=[finding],
            authoritative_rca_finding_id=None,
        )

        context = build_planner_context(
            goal=_goal(),
            state=state,
            tool_call_count=3,
        )

        self.assertEqual(context.goal.goal_id, "GOAL_PLANNER_INTERFACE")
        self.assertEqual(context.findings, state.findings)
        self.assertEqual(context.action_records, state.action_history)
        self.assertEqual(context.tool_call_count, 3)
        self.assertEqual(context.evidence_ids, [evidence.evidence_id])
        self.assertEqual(context.critical_contradictions, [])
        self.assertIsNone(context.authoritative_rca_finding_id)


if __name__ == "__main__":
    unittest.main()
