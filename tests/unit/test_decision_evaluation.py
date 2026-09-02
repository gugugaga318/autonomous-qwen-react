from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core"))

from yield_rca_core.decision_evaluation import evaluate  # noqa: E402
from yield_rca_core.evidence_models import (  # noqa: E402
    Evidence,
    EvidenceSourceType,
)
from yield_rca_core.investigation_models import (  # noqa: E402
    ActionKind,
    ActionRecord,
    ConclusionLevel,
    DecisionType,
    EvidenceGapStatus,
    GoalStatus,
    InvestigationAction,
    InvestigationGoal,
    InvestigationIntent,
    InvestigationQuestion,
    PlannerDecision,
    StopReason,
)
from yield_rca_core.models import (  # noqa: E402
    AgentFinding,
    AgentKind,
    Hypothesis,
    HypothesisStatus,
    RCAJob,
    RCAState,
    TaskStatus,
)

GOAL_ID = "GOAL_EVALUATION"
QUESTION_ID = "QUESTION_EVALUATION"


def make_evidence(evidence_id: str, *, source_type: str = "mes") -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        source_type=source_type,
        source_id=f"source:{evidence_id}",
        summary=f"Observation for {evidence_id}.",
    )


def make_finding(
    finding_id: str,
    agent: str,
    evidence_ids: list[str],
    *,
    details: dict[str, Any] | None = None,
) -> AgentFinding:
    return AgentFinding(
        finding_id=finding_id,
        agent=agent,
        summary=f"{agent} completed its bounded analysis.",
        confidence=0.8,
        evidence_ids=evidence_ids,
        details=details or {},
    )


def make_goal(
    intent: str = InvestigationIntent.IMPACT_SCOPE.value,
    *,
    max_steps: int = 8,
) -> InvestigationGoal:
    return InvestigationGoal(
        goal_id=GOAL_ID,
        intent=intent,
        summary="Evaluate the bounded Lot investigation.",
        known_facts={"lot_id": "LOT_01", "operation": "CU_CMP"},
        max_steps=max_steps,
    )


def make_question(
    *,
    status: str,
    evidence_ids: list[str] | None = None,
) -> InvestigationQuestion:
    terminal: dict[str, Any] = {}
    if status == EvidenceGapStatus.CLOSED.value:
        terminal = {
            "answer": "The recorded Evidence answers the bounded question.",
            "evidence_ids": evidence_ids or [],
        }
    elif status == EvidenceGapStatus.UNAVAILABLE.value:
        terminal = {
            "unavailable_reason": "The bounded source returned no usable data."
        }
    return InvestigationQuestion(
        question_id=QUESTION_ID,
        goal_id=GOAL_ID,
        question="What does the bounded investigation show?",
        rationale="The Planner must answer or explicitly close this Evidence gap.",
        scope={"lot_id": "LOT_01", "operation": "CU_CMP"},
        status=status,
        **terminal,
    )


def make_action(
    action_id: str,
    kind: str,
    agent: str,
    *,
    scope: dict[str, Any] | None = None,
) -> InvestigationAction:
    return InvestigationAction(
        action_id=action_id,
        kind=kind,
        agent=agent,
        reason="Collect the next bounded observation.",
        inputs={"lot_id": "LOT_01"},
        scope=scope or {"lot_id": "LOT_01", "operation": "CU_CMP"},
    )


def make_record(
    action: InvestigationAction,
    finding: AgentFinding,
    *,
    evidence_ids: list[str] | None = None,
    status: str = "completed",
) -> ActionRecord:
    return ActionRecord(
        action=action,
        status=status,
        produced_finding_ids=[finding.finding_id],
        produced_evidence_ids=(
            list(finding.evidence_ids)
            if evidence_ids is None
            else evidence_ids
        ),
        decision_summary=finding.summary,
    )


def make_act_decision(
    decision_id: str,
    action: InvestigationAction,
) -> PlannerDecision:
    return PlannerDecision(
        decision_id=decision_id,
        goal_id=GOAL_ID,
        decision_type=DecisionType.ACT.value,
        reason="This action targets the current bounded question.",
        goal_status=GoalStatus.IN_PROGRESS.value,
        proposed_conclusion_level=ConclusionLevel.SIGNAL.value,
        next_action=action,
        target_question_ids=[QUESTION_ID],
    )


def make_stop_decision(
    *,
    decision_id: str = "DECISION_STOP",
    goal_status: str,
    conclusion_level: str,
    stop_reason: str,
) -> PlannerDecision:
    return PlannerDecision(
        decision_id=decision_id,
        goal_id=GOAL_ID,
        decision_type=DecisionType.STOP.value,
        reason="Stop at the current explicit investigation boundary.",
        goal_status=goal_status,
        proposed_conclusion_level=conclusion_level,
        stop_reason=stop_reason,
    )


def make_state(
    *,
    goal: InvestigationGoal,
    question: InvestigationQuestion,
    decisions: list[PlannerDecision],
    records: list[ActionRecord],
    findings: list[AgentFinding],
    evidence: list[Evidence],
    goal_status: str,
    conclusion_level: str,
    stop_reason: str,
    tool_call_count: int = 0,
    hypotheses: list[Hypothesis] | None = None,
    evidence_gaps: list[str] | None = None,
) -> RCAState:
    return RCAState(
        job=RCAJob(
            job_id="JOB_EVALUATION",
            user_query="Evaluate LOT_01.",
            status=TaskStatus.COMPLETED.value,
        ),
        evidence=evidence,
        findings=findings,
        hypotheses=hypotheses or [],
        execution_metadata={
            "orchestration_requested_mode": "llm_react",
            "orchestration_mode": "llm_react",
            "tool_call_count": tool_call_count,
        },
        investigation_goal=goal,
        investigation_questions=[question],
        action_history=records,
        planner_decisions=decisions,
        goal_status=goal_status,
        conclusion_level=conclusion_level,
        evidence_gaps=evidence_gaps or [],
        stop_reason=stop_reason,
    )


def make_successful_impact_state() -> RCAState:
    goal = make_goal()
    evidence = make_evidence("EV_MES_SCOPE")
    finding = make_finding(
        "FINDING_MES_SCOPE",
        AgentKind.MES.value,
        [evidence.evidence_id],
    )
    action = make_action(
        "ACTION_MES_SCOPE",
        ActionKind.FIND_SHARED_EXPOSURE.value,
        AgentKind.MES.value,
    )
    stop = make_stop_decision(
        goal_status=GoalStatus.SATISFIED.value,
        conclusion_level=ConclusionLevel.SIGNAL.value,
        stop_reason=StopReason.GOAL_SATISFIED.value,
    )
    return make_state(
        goal=goal,
        question=make_question(
            status=EvidenceGapStatus.CLOSED.value,
            evidence_ids=[evidence.evidence_id],
        ),
        decisions=[make_act_decision("DECISION_MES", action), stop],
        records=[make_record(action, finding)],
        findings=[finding],
        evidence=[evidence],
        goal_status=GoalStatus.SATISFIED.value,
        conclusion_level=ConclusionLevel.SIGNAL.value,
        stop_reason=StopReason.GOAL_SATISFIED.value,
        tool_call_count=1,
    )


def test_normal_impact_run_has_valid_gain_success_and_correct_stop() -> None:
    evaluation = evaluate(make_successful_impact_state())

    assert evaluation is not None
    assert evaluation.goal_success is True
    assert evaluation.stop_correct is True
    assert evaluation.decision_evaluations[0].decision_valid is True
    assert evaluation.decision_evaluations[0].evidence_gain is True
    assert evaluation.decision_evaluations[0].redundant is False
    assert evaluation.decision_evaluations[0].new_evidence_ids == [
        "EV_MES_SCOPE"
    ]
    assert evaluation.decision_evaluations[1].decision_valid is True


def test_finalizer_projection_preserves_valid_planner_proposal_and_terminal_stop() -> None:
    original = make_successful_impact_state()
    planner_stop = original.planner_decisions[-1]
    state = replace(
        original,
        execution_metadata={
            **original.execution_metadata,
            "planner_stop_proposed_by": "qwen",
            "terminal_stop_projection_applied": True,
            "terminal_stop_projection_trace": "execution_metadata_only",
            "terminal_stop_projected_by": "python_investigation_finalizer",
            "superseded_terminal_planner_decision": planner_stop.to_dict(),
            "terminal_state_owner": "python_investigation_finalizer",
        },
        goal_status=GoalStatus.BLOCKED.value,
        conclusion_level=ConclusionLevel.INCONCLUSIVE.value,
        stop_reason=StopReason.NO_HIGH_VALUE_ACTION.value,
    )

    evaluation = evaluate(state)

    assert evaluation is not None
    assert evaluation.goal_success is False
    assert evaluation.stop_correct is True
    assert evaluation.decision_evaluations[-1].decision_valid is True
    assert "Finalizer terminal projection" in (
        evaluation.decision_evaluations[-1].reason
    )


def test_unaudited_terminal_mismatch_remains_invalid() -> None:
    original = make_successful_impact_state()
    planner_stop = original.planner_decisions[-1]
    state = replace(
        original,
        execution_metadata={
            **original.execution_metadata,
            "terminal_stop_projection_applied": True,
            "terminal_stop_projection_trace": "execution_metadata_only",
            "terminal_stop_projected_by": "python_investigation_finalizer",
            "terminal_state_owner": "python_investigation_finalizer",
            "superseded_terminal_planner_decision": {
                **planner_stop.to_dict(),
                "decision_id": "DECISION_TAMPERED",
            },
        },
        goal_status=GoalStatus.BLOCKED.value,
        conclusion_level=ConclusionLevel.INCONCLUSIVE.value,
        stop_reason=StopReason.NO_HIGH_VALUE_ACTION.value,
    )

    evaluation = evaluate(state)

    assert evaluation is not None
    assert evaluation.stop_correct is False
    assert evaluation.decision_evaluations[-1].decision_valid is False


def test_optional_unavailable_question_does_not_fail_a_gated_success() -> None:
    state = make_successful_impact_state()
    unavailable = replace(
        make_question(status=EvidenceGapStatus.UNAVAILABLE.value),
        question_id="QUESTION_OPTIONAL_MATERIAL_TRACE",
    )
    state = replace(
        state,
        investigation_questions=[*state.investigation_questions, unavailable],
    )

    evaluation = evaluate(state)

    assert evaluation is not None
    assert evaluation.goal_success is True
    assert "Explicitly unavailable optional questions" in evaluation.summary
    assert evaluation.stop_correct is True


def test_unavailable_questions_cannot_replace_an_evidence_backed_answer() -> None:
    state = make_successful_impact_state()
    state = replace(
        state,
        investigation_questions=[
            make_question(status=EvidenceGapStatus.UNAVAILABLE.value)
        ],
    )

    evaluation = evaluate(state)

    assert evaluation is not None
    assert evaluation.goal_success is False
    assert "no investigation question was answered with Evidence" in evaluation.summary


def test_unavailable_question_does_not_hide_a_remaining_evidence_gap() -> None:
    state = make_successful_impact_state()
    unavailable = replace(
        make_question(status=EvidenceGapStatus.UNAVAILABLE.value),
        question_id="QUESTION_OPTIONAL_METROLOGY",
    )
    state = replace(
        state,
        investigation_questions=[*state.investigation_questions, unavailable],
        evidence_gaps=["Confirm the missing metrology correlation."],
    )

    evaluation = evaluate(state)

    assert evaluation is not None
    assert evaluation.goal_success is False
    assert "final state still records evidence gaps" in evaluation.summary


def test_new_scope_with_no_new_evidence_is_not_automatically_redundant() -> None:
    goal = make_goal(InvestigationIntent.ROOT_CAUSE.value, max_steps=3)
    defect_evidence = make_evidence(
        "EV_DEFECT_NO_GAIN",
        source_type=EvidenceSourceType.DEFECT.value,
    )
    mes_evidence = make_evidence("EV_MES_NO_GAIN")
    defect = make_finding(
        "FINDING_DEFECT_NO_GAIN",
        AgentKind.DEFECT_WAT.value,
        [defect_evidence.evidence_id],
    )
    mes = make_finding(
        "FINDING_MES_NO_GAIN",
        AgentKind.MES.value,
        [mes_evidence.evidence_id],
    )
    comparison = make_finding(
        "FINDING_COMPARISON_NO_GAIN",
        AgentKind.DEFECT_WAT.value,
        [defect_evidence.evidence_id],
    )
    actions = [
        make_action(
            "ACTION_DEFECT_NO_GAIN",
            ActionKind.INSPECT_DEFECT_PATTERN.value,
            AgentKind.DEFECT_WAT.value,
        ),
        make_action(
            "ACTION_MES_NO_GAIN",
            ActionKind.FIND_SHARED_EXPOSURE.value,
            AgentKind.MES.value,
        ),
        make_action(
            "ACTION_COMPARISON_NO_GAIN",
            ActionKind.VALIDATE_SHARED_DEFECT_PATTERN.value,
            AgentKind.DEFECT_WAT.value,
        ),
    ]
    findings = [defect, mes, comparison]
    stop = make_stop_decision(
        goal_status=GoalStatus.BUDGET_EXHAUSTED.value,
        conclusion_level=ConclusionLevel.INCONCLUSIVE.value,
        stop_reason=StopReason.BUDGET_EXHAUSTED.value,
    )
    state = make_state(
        goal=goal,
        question=make_question(status=EvidenceGapStatus.UNAVAILABLE.value),
        decisions=[
            *[
                make_act_decision(f"DECISION_NO_GAIN_{index}", action)
                for index, action in enumerate(actions, start=1)
            ],
            stop,
        ],
        records=[
            make_record(action, finding)
            for action, finding in zip(actions, findings, strict=True)
        ],
        findings=findings,
        evidence=[defect_evidence, mes_evidence],
        goal_status=GoalStatus.BUDGET_EXHAUSTED.value,
        conclusion_level=ConclusionLevel.INCONCLUSIVE.value,
        stop_reason=StopReason.BUDGET_EXHAUSTED.value,
        tool_call_count=3,
        evidence_gaps=[QUESTION_ID],
    )

    evaluation = evaluate(state)

    assert evaluation is not None
    step = evaluation.decision_evaluations[2]
    assert step.decision_valid is True
    assert step.evidence_gain is False
    assert step.redundant is False
    assert evaluation.goal_success is False
    assert evaluation.stop_correct is True


def test_orphan_repeated_scope_is_invalid_redundant_and_invalidates_stop() -> None:
    state = make_successful_impact_state()
    first_action = state.action_history[0].action
    orphan = replace(first_action, action_id="ACTION_MES_SCOPE_REPEAT")
    decisions = [
        state.planner_decisions[0],
        make_act_decision("DECISION_REPEAT", orphan),
        state.planner_decisions[-1],
    ]
    state = replace(state, planner_decisions=decisions)

    evaluation = evaluate(state)

    assert evaluation is not None
    repeated = evaluation.decision_evaluations[1]
    assert repeated.decision_valid is False
    assert repeated.evidence_gain is False
    assert repeated.redundant is True
    assert evaluation.stop_correct is False


def test_rca_reasoning_that_reuses_evidence_is_useful_not_redundant() -> None:
    goal = make_goal(InvestigationIntent.ROOT_CAUSE.value)
    evidence = [
        make_evidence("EV_DEFECT", source_type=EvidenceSourceType.DEFECT.value),
        make_evidence("EV_MES"),
        make_evidence("EV_SHARED", source_type=EvidenceSourceType.DEFECT.value),
        make_evidence("EV_FDC", source_type=EvidenceSourceType.FDC.value),
    ]
    source_defect = make_finding(
        "FINDING_DEFECT",
        AgentKind.DEFECT_WAT.value,
        ["EV_DEFECT"],
        details={
            "evidence_scope": "selected_lots",
            "defect_patterns": {"scratch": 1},
        },
    )
    mes = make_finding("FINDING_MES", AgentKind.MES.value, ["EV_MES"])
    comparison = make_finding(
        "FINDING_SHARED",
        AgentKind.DEFECT_WAT.value,
        ["EV_SHARED"],
        details={
            "evidence_scope": "shared_exposure_comparison",
            "defect_patterns": {"scratch": 1},
        },
    )
    fdc = make_finding("FINDING_FDC", AgentKind.FDC.value, ["EV_FDC"])
    rca = make_finding(
        "FINDING_RCA",
        AgentKind.RCA_REASONING.value,
        ["EV_DEFECT", "EV_MES", "EV_SHARED", "EV_FDC"],
        details={"status": ConclusionLevel.SUPPORTED.value},
    )
    actions = [
        make_action(
            "ACTION_DEFECT",
            ActionKind.INSPECT_DEFECT_PATTERN.value,
            AgentKind.DEFECT_WAT.value,
        ),
        make_action(
            "ACTION_MES",
            ActionKind.FIND_SHARED_EXPOSURE.value,
            AgentKind.MES.value,
        ),
        make_action(
            "ACTION_SHARED",
            ActionKind.VALIDATE_SHARED_DEFECT_PATTERN.value,
            AgentKind.DEFECT_WAT.value,
        ),
        make_action(
            "ACTION_FDC",
            ActionKind.INSPECT_FDC_SPC.value,
            AgentKind.FDC.value,
        ),
        make_action(
            "ACTION_RCA",
            ActionKind.RUN_RCA_REASONING.value,
            AgentKind.RCA_REASONING.value,
        ),
    ]
    findings = [source_defect, mes, comparison, fdc, rca]
    records = [
        make_record(action, finding)
        for action, finding in zip(actions, findings, strict=True)
    ]
    stop = make_stop_decision(
        goal_status=GoalStatus.SATISFIED.value,
        conclusion_level=ConclusionLevel.SUPPORTED.value,
        stop_reason=StopReason.GOAL_SATISFIED.value,
    )
    state = make_state(
        goal=goal,
        question=make_question(
            status=EvidenceGapStatus.CLOSED.value,
            evidence_ids=[item.evidence_id for item in evidence],
        ),
        decisions=[
            *[
                make_act_decision(f"DECISION_{index}", action)
                for index, action in enumerate(actions, start=1)
            ],
            stop,
        ],
        records=records,
        findings=findings,
        evidence=evidence,
        goal_status=GoalStatus.SATISFIED.value,
        conclusion_level=ConclusionLevel.SUPPORTED.value,
        stop_reason=StopReason.GOAL_SATISFIED.value,
        tool_call_count=5,
    )

    evaluation = evaluate(state)

    assert evaluation is not None
    reasoning = evaluation.decision_evaluations[-2]
    assert reasoning.decision_valid is True
    assert reasoning.evidence_gain is False
    assert reasoning.redundant is False
    assert evaluation.goal_success is True
    assert evaluation.stop_correct is True


def test_legal_but_premature_stop_is_valid_and_stop_incorrect() -> None:
    stop = make_stop_decision(
        goal_status=GoalStatus.BLOCKED.value,
        conclusion_level=ConclusionLevel.INCONCLUSIVE.value,
        stop_reason=StopReason.DATA_UNAVAILABLE.value,
    )
    state = make_state(
        goal=make_goal(),
        question=make_question(status=EvidenceGapStatus.OPEN.value),
        decisions=[stop],
        records=[],
        findings=[],
        evidence=[],
        goal_status=GoalStatus.BLOCKED.value,
        conclusion_level=ConclusionLevel.INCONCLUSIVE.value,
        stop_reason=StopReason.DATA_UNAVAILABLE.value,
        evidence_gaps=[QUESTION_ID],
    )

    evaluation = evaluate(state)

    assert evaluation is not None
    assert evaluation.decision_evaluations[0].decision_valid is True
    assert evaluation.goal_success is False
    assert evaluation.stop_correct is False


def test_data_unavailable_stop_is_correct_after_no_legal_action_remains() -> None:
    state = make_successful_impact_state()
    stop = make_stop_decision(
        goal_status=GoalStatus.BLOCKED.value,
        conclusion_level=ConclusionLevel.INCONCLUSIVE.value,
        stop_reason=StopReason.DATA_UNAVAILABLE.value,
    )
    state = replace(
        state,
        investigation_questions=[
            make_question(status=EvidenceGapStatus.UNAVAILABLE.value)
        ],
        planner_decisions=[state.planner_decisions[0], stop],
        goal_status=GoalStatus.BLOCKED.value,
        conclusion_level=ConclusionLevel.INCONCLUSIVE.value,
        evidence_gaps=[QUESTION_ID],
        stop_reason=StopReason.DATA_UNAVAILABLE.value,
    )

    evaluation = evaluate(state)

    assert evaluation is not None
    assert evaluation.goal_success is False
    assert evaluation.stop_correct is True


def test_budget_stop_is_correct_only_when_trace_reaches_the_boundary() -> None:
    goal = make_goal(max_steps=1)
    evidence = make_evidence("EV_BUDGET")
    finding = make_finding(
        "FINDING_BUDGET",
        AgentKind.MES.value,
        [evidence.evidence_id],
    )
    action = make_action(
        "ACTION_BUDGET",
        ActionKind.FIND_SHARED_EXPOSURE.value,
        AgentKind.MES.value,
    )
    stop = make_stop_decision(
        goal_status=GoalStatus.BUDGET_EXHAUSTED.value,
        conclusion_level=ConclusionLevel.INCONCLUSIVE.value,
        stop_reason=StopReason.BUDGET_EXHAUSTED.value,
    )
    state = make_state(
        goal=goal,
        question=make_question(status=EvidenceGapStatus.UNAVAILABLE.value),
        decisions=[make_act_decision("DECISION_BUDGET", action), stop],
        records=[make_record(action, finding)],
        findings=[finding],
        evidence=[evidence],
        goal_status=GoalStatus.BUDGET_EXHAUSTED.value,
        conclusion_level=ConclusionLevel.INCONCLUSIVE.value,
        stop_reason=StopReason.BUDGET_EXHAUSTED.value,
        tool_call_count=1,
        evidence_gaps=[QUESTION_ID],
    )

    evaluation = evaluate(state)

    assert evaluation is not None
    assert evaluation.goal_success is False
    assert evaluation.stop_correct is True

    false_claim = replace(
        state,
        investigation_goal=make_goal(max_steps=2),
    )
    false_evaluation = evaluate(false_claim)
    assert false_evaluation is not None
    assert false_evaluation.stop_correct is False


def test_conflict_is_auditable_but_uncommitted_action_invalidates_stop() -> None:
    evidence = make_evidence("EV_CONFLICT")
    conflict = Hypothesis(
        hypothesis_id="HYPOTHESIS_CONFLICT",
        root_cause="The observed signals support incompatible mechanisms.",
        confidence=0.5,
        evidence_ids=[evidence.evidence_id],
        status=HypothesisStatus.CONFLICTED.value,
    )
    critical_stop = make_stop_decision(
        goal_status=GoalStatus.BLOCKED.value,
        conclusion_level=ConclusionLevel.CONFLICTED.value,
        stop_reason=StopReason.CRITICAL_CONTRADICTION.value,
    )
    critical_state = make_state(
        goal=make_goal(),
        question=make_question(status=EvidenceGapStatus.OPEN.value),
        decisions=[critical_stop],
        records=[],
        findings=[],
        evidence=[evidence],
        hypotheses=[conflict],
        goal_status=GoalStatus.BLOCKED.value,
        conclusion_level=ConclusionLevel.CONFLICTED.value,
        stop_reason=StopReason.CRITICAL_CONTRADICTION.value,
        evidence_gaps=[QUESTION_ID],
    )

    critical_evaluation = evaluate(critical_state)

    assert critical_evaluation is not None
    assert critical_evaluation.goal_success is False
    assert critical_evaluation.stop_correct is True

    attempted = make_action(
        "ACTION_FAILED_DEFECT",
        ActionKind.INSPECT_DEFECT_PATTERN.value,
        AgentKind.DEFECT_WAT.value,
    )
    no_allowed_stop = make_stop_decision(
        goal_status=GoalStatus.BLOCKED.value,
        conclusion_level=ConclusionLevel.INCONCLUSIVE.value,
        stop_reason=StopReason.NO_ALLOWED_ACTION.value,
    )
    no_allowed_state = make_state(
        goal=make_goal(InvestigationIntent.ROOT_CAUSE.value),
        question=make_question(status=EvidenceGapStatus.UNAVAILABLE.value),
        decisions=[no_allowed_stop],
        records=[
            ActionRecord(
                action=attempted,
                status="failed",
                decision_summary="The bounded defect action returned no Finding.",
            )
        ],
        findings=[],
        evidence=[],
        goal_status=GoalStatus.BLOCKED.value,
        conclusion_level=ConclusionLevel.INCONCLUSIVE.value,
        stop_reason=StopReason.NO_ALLOWED_ACTION.value,
        tool_call_count=1,
        evidence_gaps=[QUESTION_ID],
    )

    no_allowed_evaluation = evaluate(no_allowed_state)

    assert no_allowed_evaluation is not None
    assert no_allowed_evaluation.stop_correct is False
    assert (
        no_allowed_evaluation.decision_evaluations[-1].decision_valid
        is False
    )


def test_inconclusive_result_can_stop_correctly_without_claiming_goal_success() -> None:
    state = replace(
        make_successful_impact_state(),
        conclusion_level=ConclusionLevel.INCONCLUSIVE.value,
        planner_decisions=[
            make_successful_impact_state().planner_decisions[0],
            make_stop_decision(
                goal_status=GoalStatus.SATISFIED.value,
                conclusion_level=ConclusionLevel.INCONCLUSIVE.value,
                stop_reason=StopReason.GOAL_SATISFIED.value,
            ),
        ],
    )

    evaluation = evaluate(state)

    assert evaluation is not None
    assert evaluation.goal_success is False
    assert evaluation.stop_correct is True


def test_scope_attempt_and_agent_registry_are_part_of_decision_validity() -> None:
    evidence = make_evidence("EV_WRONG_AGENT", source_type=EvidenceSourceType.FDC.value)
    finding = make_finding(
        "FINDING_WRONG_AGENT",
        AgentKind.FDC.value,
        [evidence.evidence_id],
    )
    action = make_action(
        "ACTION_WRONG_AGENT",
        ActionKind.FIND_SHARED_EXPOSURE.value,
        AgentKind.FDC.value,
    )
    stop = make_stop_decision(
        goal_status=GoalStatus.BLOCKED.value,
        conclusion_level=ConclusionLevel.INCONCLUSIVE.value,
        stop_reason=StopReason.NO_ALLOWED_ACTION.value,
    )
    state = make_state(
        goal=make_goal(),
        question=make_question(
            status=EvidenceGapStatus.CLOSED.value,
            evidence_ids=[evidence.evidence_id],
        ),
        decisions=[make_act_decision("DECISION_WRONG_AGENT", action), stop],
        records=[make_record(action, finding)],
        findings=[finding],
        evidence=[evidence],
        goal_status=GoalStatus.BLOCKED.value,
        conclusion_level=ConclusionLevel.INCONCLUSIVE.value,
        stop_reason=StopReason.NO_ALLOWED_ACTION.value,
        tool_call_count=1,
    )

    evaluation = evaluate(state)

    assert evaluation is not None
    assert evaluation.decision_evaluations[0].decision_valid is False
    assert evaluation.decision_evaluations[0].evidence_gain is False
    assert evaluation.stop_correct is False


def test_non_autonomous_or_fallback_trace_is_not_evaluated() -> None:
    state = make_successful_impact_state()

    assert evaluate(
        replace(
            state,
            execution_metadata={
                **state.execution_metadata,
                "orchestration_mode": "controlled_react",
            },
        )
    ) is None
    assert evaluate(
        replace(
            state,
            execution_metadata={
                **state.execution_metadata,
                "orchestration_fallback_reason": "qwen_output_invalid",
            },
        )
    ) is None
    assert evaluate(replace(state, planner_decisions=[])) is None
    assert evaluate(
        replace(
            state,
            job=replace(state.job, status=TaskStatus.RUNNING.value),
        )
    ) is None
