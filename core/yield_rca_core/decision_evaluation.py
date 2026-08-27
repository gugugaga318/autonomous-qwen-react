"""Deterministic evaluation for completed autonomous Planner traces.

The evaluator deliberately does not ask an LLM to grade another LLM.  It
derives the five public metrics from the typed PlannerDecision, ActionRecord,
Evidence, and terminal-state contracts already present in ``RCAState``.
"""

from __future__ import annotations

from collections.abc import Iterable

from yield_rca_core.investigation_models import (
    ActionKind,
    ConclusionLevel,
    DecisionEvaluation,
    DecisionType,
    EvidenceGapStatus,
    GoalStatus,
    InvestigationIntent,
    OrchestrationMode,
    PlannerDecision,
    QuestionEvidenceRelation,
    RunEvaluation,
    StopReason,
)
from yield_rca_core.investigation_policy import InvestigationPolicy
from yield_rca_core.models import AgentFinding, HypothesisStatus, RCAState, TaskStatus
from yield_rca_core.next_action_planner import LLM_REACT_ACTION_REGISTRY

_FALLBACK_METADATA_KEYS = {
    "orchestration_fallback_reason",
    "orchestration_fallback_stage",
    "orchestration_fallback_after_action_count",
}

_SUCCESS_LEVELS = {
    InvestigationIntent.IMPACT_SCOPE.value: {
        ConclusionLevel.SIGNAL.value,
        ConclusionLevel.CANDIDATE.value,
        ConclusionLevel.SUPPORTED.value,
    },
    InvestigationIntent.SPC_CHECK.value: {
        ConclusionLevel.SIGNAL.value,
        ConclusionLevel.CANDIDATE.value,
        ConclusionLevel.SUPPORTED.value,
    },
    InvestigationIntent.HISTORICAL_LOOKUP.value: {
        ConclusionLevel.CANDIDATE.value,
        ConclusionLevel.SUPPORTED.value,
    },
    InvestigationIntent.ROOT_CAUSE.value: {
        ConclusionLevel.CANDIDATE.value,
        ConclusionLevel.SUPPORTED.value,
    },
    InvestigationIntent.FULL_RCA.value: {
        ConclusionLevel.CANDIDATE.value,
        ConclusionLevel.SUPPORTED.value,
    },
}


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _is_eligible(state: RCAState) -> bool:
    metadata = state.execution_metadata
    requested = metadata.get("orchestration_requested_mode")
    actual = metadata.get("orchestration_mode")
    return (
        requested == OrchestrationMode.LLM_REACT.value
        and actual == OrchestrationMode.LLM_REACT.value
        and not any(key in metadata for key in _FALLBACK_METADATA_KEYS)
        and state.job.status == TaskStatus.COMPLETED.value
        and state.investigation_goal is not None
        and bool(state.planner_decisions)
    )


def _finding_evidence_ids(findings: list[AgentFinding]) -> list[str]:
    return _ordered_unique(
        evidence_id
        for finding in findings
        for evidence_id in finding.evidence_ids
    )


def _evaluate_stop_decision(
    state: RCAState,
    decision: PlannerDecision,
    *,
    decision_index: int,
) -> DecisionEvaluation:
    issues: list[str] = []
    if decision_index != len(state.planner_decisions) - 1:
        issues.append("it is not the final Planner decision")
    if (
        decision.goal_status != state.goal_status
        or decision.stop_reason != state.stop_reason
    ):
        issues.append("its terminal fields do not match the final state")
    committed_action_ids = [
        item.next_action.action_id
        for item in state.planner_decisions
        if item.decision_type == DecisionType.ACT.value
        and item.next_action is not None
    ]
    recorded_action_ids = [
        record.action.action_id for record in state.action_history
    ]
    if committed_action_ids != recorded_action_ids:
        issues.append(
            "the ActionRecord trace does not exactly match the committed act decisions"
        )
    valid = not issues
    return DecisionEvaluation(
        decision_id=decision.decision_id,
        decision_valid=valid,
        evidence_gain=False,
        redundant=False,
        reason=(
            (
                "The stop decision is the final typed Planner decision and "
                "matches the terminal state. Whether it stopped at the right "
                "investigation boundary is evaluated at run level."
            )
            if valid
            else (
                "The stop decision failed deterministic checks: "
                f"{'; '.join(issues)}."
            )
        ),
    )


def _evaluate_action_decision(
    state: RCAState,
    decision: PlannerDecision,
    *,
    previous_record_index: int,
) -> tuple[DecisionEvaluation, int]:
    action = decision.next_action
    if action is None:
        return (
            DecisionEvaluation(
                decision_id=decision.decision_id,
                decision_valid=False,
                evidence_gain=False,
                redundant=False,
                reason="The act decision has no executable action.",
            ),
            previous_record_index,
        )

    matches = [
        (index, record)
        for index, record in enumerate(state.action_history)
        if record.action.action_id == action.action_id
    ]
    issues: list[str] = []
    prior_committed_records = state.action_history[: previous_record_index + 1]
    redundant = action.deduplication_key in {
        record.action.deduplication_key
        for record in prior_committed_records
    }
    if len(matches) != 1:
        issues.append(
            "its action_id does not identify exactly one ActionRecord"
        )
        if redundant:
            issues.append("its Action and Scope were already investigated")
        return (
            DecisionEvaluation(
                decision_id=decision.decision_id,
                decision_valid=False,
                evidence_gain=False,
                redundant=redundant,
                reason=f"The decision failed deterministic checks: {'; '.join(issues)}.",
            ),
            previous_record_index,
        )

    record_index, record = matches[0]
    prior_records = state.action_history[:record_index]
    prior_evidence_ids = {
        evidence_id
        for prior_record in prior_records
        for evidence_id in prior_record.produced_evidence_ids
    }
    prior_keys = {
        prior_record.action.deduplication_key
        for prior_record in prior_records
    }
    redundant = action.deduplication_key in prior_keys

    if record_index <= previous_record_index:
        issues.append("its ActionRecord is out of Planner decision order")
    if record.action != action:
        issues.append("its action does not exactly match the ActionRecord")
    if record.status != "completed":
        issues.append("its ActionRecord is not completed")
    if not action.scope:
        issues.append("its action has no stable autonomous investigation scope")
    if action.max_attempts != 1:
        issues.append("its action exceeds the one-attempt autonomous boundary")

    definition = LLM_REACT_ACTION_REGISTRY.get(action.kind)
    if definition is None:
        issues.append("its action is not in the autonomous action registry")
    elif action.agent != definition.agent:
        issues.append("its Agent does not match the registered action owner")

    findings_by_id = {
        finding.finding_id: finding
        for finding in state.findings
    }
    produced_findings = [
        findings_by_id[finding_id]
        for finding_id in record.produced_finding_ids
        if finding_id in findings_by_id
    ]
    if not record.produced_finding_ids:
        issues.append("its ActionRecord does not reference a Finding")
    elif len(produced_findings) != len(record.produced_finding_ids):
        issues.append("its ActionRecord references an unknown Finding")
    elif any(finding.agent != action.agent for finding in produced_findings):
        issues.append("its Finding was produced by a different Agent")

    final_evidence_ids = {item.evidence_id for item in state.evidence}
    if not set(record.produced_evidence_ids) <= final_evidence_ids:
        issues.append("its ActionRecord references unknown Evidence")
    if produced_findings and (
        record.produced_evidence_ids
        != _finding_evidence_ids(produced_findings)
    ):
        issues.append("its Finding and ActionRecord Evidence references differ")
    if not set(action.required_evidence_ids) <= prior_evidence_ids:
        issues.append("its required Evidence was not available before the action")

    if definition is not None:
        prior_finding_ids = {
            finding_id
            for prior_record in prior_records
            for finding_id in prior_record.produced_finding_ids
        }
        prior_finding_agents = {
            finding.agent
            for finding_id, finding in findings_by_id.items()
            if finding_id in prior_finding_ids
        }
        missing_agents = (
            set(definition.required_finding_agents) - prior_finding_agents
        )
        if missing_agents:
            issues.append(
                "its prerequisite Findings were not available before the action"
            )

    if redundant:
        issues.append("its Action and Scope were already investigated")

    decision_valid = not issues
    new_evidence_ids = (
        [
            evidence_id
            for evidence_id in record.produced_evidence_ids
            if evidence_id in final_evidence_ids
            and evidence_id not in prior_evidence_ids
        ]
        if decision_valid
        else []
    )
    target_question_ids = set(decision.target_question_ids)
    applicable_links = [
        link
        for link in state.question_evidence_links
        if link.action_id == record.action.action_id
        and link.question_id in target_question_ids
        and link.evidence_id in new_evidence_ids
        and link.relation
        in {
            QuestionEvidenceRelation.SUPPORTS.value,
            QuestionEvidenceRelation.CONTRADICTS.value,
            QuestionEvidenceRelation.CONTEXT.value,
            QuestionEvidenceRelation.UNAVAILABLE.value,
        }
    ]
    collecting_action = action.kind != ActionKind.RUN_RCA_REASONING.value
    if state.question_evidence_links:
        relevant_evidence_gain = bool(applicable_links) if collecting_action else False
    else:
        # Legacy snapshots predate QuestionEvidenceLink. Preserve their original
        # Evidence-ID metric while typed llm_react traces use the stricter gate.
        relevant_evidence_gain = bool(new_evidence_ids)

    if issues:
        reason = f"The decision failed deterministic checks: {'; '.join(issues)}."
    elif relevant_evidence_gain:
        reason = (
            f"The allowlisted {action.agent} action matched its completed "
            f"ActionRecord and added {len(applicable_links)} new applicable "
            "QuestionEvidenceLink"
            f"{'s' if len(applicable_links) != 1 else ''} for its target "
            "Question"
            f"{'s' if len(target_question_ids) != 1 else ''}."
        )
    elif new_evidence_ids:
        reason = (
            f"The allowlisted {action.agent} action matched its completed "
            "ActionRecord and added new Evidence IDs, but none produced an "
            "applicable QuestionEvidenceLink for its target Question."
        )
    else:
        reason = (
            f"The allowlisted {action.agent} action matched its completed "
            "ActionRecord. It added no new Evidence IDs or applicable "
            "QuestionEvidenceLinks, but this Action and Scope had not been "
            "investigated before, so it was not redundant."
        )

    return (
        DecisionEvaluation(
            decision_id=decision.decision_id,
            decision_valid=decision_valid,
            evidence_gain=relevant_evidence_gain,
            redundant=redundant,
            reason=reason,
            new_evidence_ids=new_evidence_ids,
        ),
        max(previous_record_index, record_index),
    )


def _evaluate_decisions(state: RCAState) -> list[DecisionEvaluation]:
    evaluations: list[DecisionEvaluation] = []
    previous_record_index = -1
    for decision_index, decision in enumerate(state.planner_decisions):
        if decision.decision_type == DecisionType.STOP.value:
            evaluations.append(
                _evaluate_stop_decision(
                    state,
                    decision,
                    decision_index=decision_index,
                )
            )
            continue
        evaluation, previous_record_index = _evaluate_action_decision(
            state,
            decision,
            previous_record_index=previous_record_index,
        )
        evaluations.append(evaluation)
    return evaluations


def _goal_success(state: RCAState) -> tuple[bool, str]:
    goal = state.investigation_goal
    if goal is None:
        return False, "the run has no typed investigation goal."
    if state.goal_status != GoalStatus.SATISFIED.value:
        return False, f"the final goal status is {state.goal_status or 'missing'}."
    if not state.evidence:
        return False, "the final result contains no Evidence."
    open_questions = [
        question.question_id
        for question in state.investigation_questions
        if question.status == EvidenceGapStatus.OPEN.value
    ]
    if open_questions:
        return False, "one or more investigation questions remain unanswered."
    closed_questions = [
        question.question_id
        for question in state.investigation_questions
        if question.status == EvidenceGapStatus.CLOSED.value
    ]
    if not closed_questions:
        return False, "no investigation question was answered with Evidence."
    if state.evidence_gaps:
        return False, "the final state still records evidence gaps."
    allowed_levels = _SUCCESS_LEVELS[goal.intent]
    if state.conclusion_level not in allowed_levels:
        return (
            False,
            (
                f"the gated conclusion level "
                f"{state.conclusion_level or 'missing'} is below the level "
                f"needed for {goal.intent}."
            ),
        )
    return (
        True,
        (
            f"the {goal.intent} objective is satisfied with at least one "
            "Evidence-backed closed question, no open questions or remaining "
            f"evidence gaps, and an evidence-gated {state.conclusion_level} "
            "conclusion. Explicitly unavailable optional questions do not "
            "invalidate that result."
        ),
    )


def _tool_call_count(state: RCAState) -> int:
    value = state.execution_metadata.get("tool_call_count")
    if type(value) is int and value >= 0:
        return value
    return len(state.action_history)


def _stop_correct(state: RCAState) -> tuple[bool, str]:
    goal = state.investigation_goal
    if goal is None:
        return False, "the run has no typed investigation goal."
    terminal = state.planner_decisions[-1]
    if terminal.decision_type != DecisionType.STOP.value:
        return False, "the Planner trace has no final stop decision."
    if (
        terminal.stop_reason != state.stop_reason
        or terminal.goal_status != state.goal_status
    ):
        return False, "the final stop decision does not match the terminal state."

    authoritative = state.authoritative_hypothesis
    if authoritative is None and len(state.hypotheses) == 1:
        # Preserve unambiguous legacy/synthetic traces that predate explicit
        # authority markers without ever consulting multiple historical rounds.
        authoritative = state.hypotheses[0]
    contradictions = (
        [f"{authoritative.hypothesis_id}: {authoritative.root_cause}"]
        if authoritative is not None
        and authoritative.status == HypothesisStatus.CONFLICTED.value
        else []
    )
    try:
        oracle = InvestigationPolicy().next_action(
            goal=goal,
            findings=state.findings,
            action_records=state.action_history,
            tool_call_count=_tool_call_count(state),
            critical_contradictions=contradictions,
        )
    except (StopIteration, ValueError) as exc:
        return False, f"the deterministic stop oracle could not evaluate the trace: {exc}."

    actual_reason = state.stop_reason
    has_open_questions = any(
        question.status == EvidenceGapStatus.OPEN.value
        for question in state.investigation_questions
    )

    if actual_reason == StopReason.GOAL_SATISFIED.value:
        correct = (
            state.goal_status == GoalStatus.SATISFIED.value
            and not has_open_questions
            and oracle.next_action is None
            and oracle.stop_reason == StopReason.GOAL_SATISFIED.value
        )
        reason = (
            "the deterministic policy also finds the goal satisfied and no "
            "investigation question remains open."
            if correct
            else "a legal next action or an open question remains before goal satisfaction."
        )
        return correct, reason

    if actual_reason == StopReason.BUDGET_EXHAUSTED.value:
        correct = (
            state.goal_status == GoalStatus.BUDGET_EXHAUSTED.value
            and oracle.next_action is None
            and oracle.stop_reason == StopReason.BUDGET_EXHAUSTED.value
        )
        reason = (
            "the recorded step or Tool-call budget reached its hard boundary."
            if correct
            else "the terminal trace does not prove that a hard budget was exhausted."
        )
        return correct, reason

    if actual_reason == StopReason.CRITICAL_CONTRADICTION.value:
        correct = (
            state.goal_status == GoalStatus.BLOCKED.value
            and state.conclusion_level == ConclusionLevel.CONFLICTED.value
            and bool(contradictions)
            and oracle.next_action is None
            and oracle.stop_reason == StopReason.CRITICAL_CONTRADICTION.value
        )
        reason = (
            "a conflicted Hypothesis proves the critical-contradiction boundary."
            if correct
            else "the trace does not contain a conflicted Hypothesis for this stop."
        )
        return correct, reason

    if actual_reason == StopReason.NO_ALLOWED_ACTION.value:
        correct = (
            state.goal_status == GoalStatus.BLOCKED.value
            and oracle.next_action is None
            and oracle.stop_reason == StopReason.NO_ALLOWED_ACTION.value
        )
        reason = (
            "the deterministic policy confirms that no registered action remains."
            if correct
            else "the deterministic policy still has a registered next action."
        )
        return correct, reason

    if actual_reason == StopReason.DATA_UNAVAILABLE.value:
        has_unavailable_question = any(
            question.status == EvidenceGapStatus.UNAVAILABLE.value
            for question in state.investigation_questions
        )
        correct = (
            state.goal_status == GoalStatus.BLOCKED.value
            and not has_open_questions
            and has_unavailable_question
            and oracle.next_action is None
            and oracle.stop_reason
            in {
                StopReason.GOAL_SATISFIED.value,
                StopReason.NO_ALLOWED_ACTION.value,
            }
        )
        reason = (
            "the unanswered questions are explicitly unavailable and no legal "
            "evidence-producing action remains."
            if correct
            else "data is claimed unavailable while a question or legal action remains."
        )
        return correct, reason

    return False, f"the stop reason {actual_reason or 'missing'} is not auditable."


def evaluate(state: RCAState) -> RunEvaluation | None:
    """Evaluate one completed, non-fallback ``llm_react`` run.

    Fixed, controlled, fallback, and incomplete traces return ``None`` because
    they are regression baselines rather than autonomous Planner decisions.
    """

    if not _is_eligible(state):
        return None

    goal = state.investigation_goal
    if goal is None:  # Defensive narrowing for typed callers.
        return None
    decision_evaluations = _evaluate_decisions(state)
    goal_success, goal_reason = _goal_success(state)
    stop_correct, stop_reason = _stop_correct(state)
    ineffective_decisions = [
        evaluation.decision_id
        for evaluation in decision_evaluations
        if not evaluation.decision_valid or evaluation.redundant
    ]
    if ineffective_decisions:
        stop_correct = False
        stop_reason = (
            "the run committed an invalid or redundant Planner decision "
            f"before stopping: {', '.join(ineffective_decisions)}."
        )
    return RunEvaluation(
        goal_id=goal.goal_id,
        goal_success=goal_success,
        stop_correct=stop_correct,
        summary=(
            f"Goal success is {str(goal_success).lower()}: {goal_reason} "
            f"Stop correctness is {str(stop_correct).lower()}: {stop_reason}"
        ),
        decision_evaluations=decision_evaluations,
    )


__all__ = ["evaluate"]
