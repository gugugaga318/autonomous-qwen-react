"""Deterministic final evaluation for the autonomous Qwen ReAct workflow.

The suite intentionally separates three concerns:

* a repeatable Fake-Qwen ``llm_react`` acceptance lane;
* the existing fixed-workflow Step 14 compatibility baseline; and
* an optional real-Qwen smoke status, which is never inferred as passing.

Only the five metrics already defined by ``RunEvaluation`` are aggregated.
Scenario checks such as evidence-gate downgrade and fallback attribution are
acceptance facts, not additional scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from yield_rca_core.evaluation import EvaluationScenario, evaluate_scenarios
from yield_rca_core.evidence_builder import EvidenceBuilder
from yield_rca_core.evidence_models import (
    EntityType,
    EvidenceEntity,
    EvidenceSourceType,
    EvidenceType,
)
from yield_rca_core.investigation_models import (
    ActionKind,
    ActionRecord,
    ConclusionLevel,
    DecisionType,
    EvidenceGapStatus,
    GoalStatus,
    InvestigationAction,
    InvestigationIntent,
    InvestigationQuestion,
    OrchestrationMode,
    PlannerDecision,
    QuestionKind,
    StopReason,
)
from yield_rca_core.llm_gateway import (
    FakeLLMClient,
    LLMClient,
    LLMRequest,
    LLMResponse,
    LLMSettings,
)
from yield_rca_core.models import (
    AgentKind,
    HypothesisStatus,
    InvestigationMode,
    RCAState,
    TaskStatus,
    ToolInput,
)
from yield_rca_core.question_capability import capability_notice_for
from yield_rca_core.question_evidence import QuestionEvidenceResolver
from yield_rca_core.question_update_review import review_question_updates
from yield_rca_core.repositories import FabRepository
from yield_rca_core.workflow import PurePythonRCAWorkflow, build_workflow

LOT_IMPACT_QUERY = "Identify the impact lots for LOT_A_001."
LOT_SPC_QUERY = "Check SPC for LOT_A_001."
LOT_ROOT_QUERY = "Investigate the root cause of LOT_A_001 scratch in Cu CMP."
LOT_HISTORY_QUERY = "Find a similar historical case for LOT_A_001."
PRODUCT_IMPACT_QUERY = (
    "Identify impact lots for 40N_SOC from 2026-07-01 to 2026-07-31."
)
PRODUCT_ROOT_QUERY = (
    "Investigate 40N_SOC yield loss root cause from 2026-07-01 to 2026-07-31."
)

_ROOT_CHAIN = (
    "inspect_defect_pattern",
    "find_shared_exposure",
    "validate_shared_defect_pattern",
    "inspect_fdc_spc",
    "run_rca_reasoning",
)
_PRODUCT_ROOT_CHAIN = (
    "find_shared_exposure",
    "inspect_defect_pattern",
    "validate_shared_defect_pattern",
    "inspect_fdc_spc",
    "run_rca_reasoning",
)
_HISTORY_CHAIN = (
    "inspect_defect_pattern",
    "find_shared_exposure",
    "validate_shared_defect_pattern",
    "inspect_fdc_spc",
    "validate_historical_case",
)


class _RecordingFakeClient:
    provider = FakeLLMClient.provider
    model = FakeLLMClient.model

    def __init__(self) -> None:
        self._delegate = FakeLLMClient()
        self.requests: list[LLMRequest] = []

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return self._delegate.complete_json(request)


class _ImmediateUnsupportedStopClient(_RecordingFakeClient):
    """Propose a supported conclusion before collecting any Evidence."""

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        response = super().complete_json(request)
        if request.prompt_name != "next_action_planner":
            return response
        goal_id = str(request.payload["goal"]["goal_id"])
        question_updates: list[dict[str, Any]] = [
            {
                "question_id": str(question["question_id"]),
                "status": EvidenceGapStatus.UNAVAILABLE.value,
                "answer": None,
                "evidence_ids": [],
                "unavailable_reason": (
                    "The premature-stop fixture claims that no source data is available."
                ),
            }
            for question in request.payload["questions"]
            if question["status"] == EvidenceGapStatus.OPEN.value
        ]
        return LLMResponse(
            data={
                "decision_id": f"{goal_id}:model-stop",
                "goal_id": goal_id,
                "decision_type": DecisionType.STOP.value,
                "reason": "The model cannot obtain the requested source data.",
                "goal_status": GoalStatus.BLOCKED.value,
                "proposed_conclusion_level": ConclusionLevel.SUPPORTED.value,
                "next_action": None,
                "target_question_ids": [],
                "new_questions": [],
                "stop_reason": StopReason.DATA_UNAVAILABLE.value,
                "question_updates": question_updates,
            },
            usage=response.usage,
        )


class _PartialEvidenceUnsupportedStopClient(_RecordingFakeClient):
    """Collect one Specialist observation, then overstate the conclusion."""

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        response = super().complete_json(request)
        if (
            request.prompt_name != "next_action_planner"
            or not request.payload["action_history"]
        ):
            return response
        goal_id = str(request.payload["goal"]["goal_id"])
        question_updates: list[dict[str, Any]] = [
            {
                "question_id": str(question["question_id"]),
                "status": EvidenceGapStatus.UNAVAILABLE.value,
                "answer": None,
                "evidence_ids": [],
                "unavailable_reason": (
                    "The partial-evidence fixture claims no further source is available."
                ),
            }
            for question in request.payload["questions"]
            if question["status"] == EvidenceGapStatus.OPEN.value
        ]
        return LLMResponse(
            data={
                "decision_id": f"{goal_id}:partial-evidence-stop",
                "goal_id": goal_id,
                "decision_type": DecisionType.STOP.value,
                "reason": "Stop early despite having only a defect observation.",
                "goal_status": GoalStatus.BLOCKED.value,
                "proposed_conclusion_level": ConclusionLevel.SUPPORTED.value,
                "next_action": None,
                "target_question_ids": [],
                "new_questions": [],
                "stop_reason": StopReason.DATA_UNAVAILABLE.value,
                "question_updates": question_updates,
            },
            usage=response.usage,
        )


class _InvalidNextActionAfterFirstClient(_RecordingFakeClient):
    """Commit one Qwen action, then fail both next-action output attempts."""

    def __init__(self) -> None:
        super().__init__()
        self.next_action_call_count = 0

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        response = super().complete_json(request)
        if request.prompt_name != "next_action_planner":
            return response
        self.next_action_call_count += 1
        if self.next_action_call_count == 1:
            return response
        return LLMResponse(data={}, usage=response.usage)


class _InvalidIntentClient(_RecordingFakeClient):
    def complete_json(self, request: LLMRequest) -> LLMResponse:
        response = super().complete_json(request)
        if request.prompt_name == "intent_planner":
            return LLMResponse(data={}, usage=response.usage)
        return response


@dataclass(frozen=True)
class _AutonomousScenario:
    scenario_id: str
    title: str
    query: str
    lot_id: str | None
    expected_job_mode: str
    expected_intent: str
    expected_chain: tuple[str, ...]
    expected_conclusion: str
    expected_goal_status: str
    expected_stop_reason: str
    expected_goal_success: bool
    expected_stop_correct: bool
    expected_decision_metrics: tuple[tuple[bool, bool, bool], ...]
    client_kind: str = "recording"
    require_observation_replanning: bool = False


_AUTONOMOUS_SCENARIOS = (
    _AutonomousScenario(
        scenario_id="AUTONOMOUS_LOT_IMPACT",
        title="Lot impact request selects only MES exposure investigation",
        query=LOT_IMPACT_QUERY,
        lot_id="LOT_A_001",
        expected_job_mode=InvestigationMode.LOT.value,
        expected_intent=InvestigationIntent.IMPACT_SCOPE.value,
        expected_chain=("find_shared_exposure",),
        expected_conclusion=ConclusionLevel.SIGNAL.value,
        expected_goal_status=GoalStatus.SATISFIED.value,
        expected_stop_reason=StopReason.GOAL_SATISFIED.value,
        expected_goal_success=True,
        expected_stop_correct=True,
        expected_decision_metrics=((True, True, False), (True, False, False)),
    ),
    _AutonomousScenario(
        scenario_id="AUTONOMOUS_LOT_SPC",
        title="SPC request selects MES scope followed by FDC/SPC",
        query=LOT_SPC_QUERY,
        lot_id="LOT_A_001",
        expected_job_mode=InvestigationMode.LOT.value,
        expected_intent=InvestigationIntent.SPC_CHECK.value,
        expected_chain=("find_shared_exposure", "inspect_fdc_spc"),
        expected_conclusion=ConclusionLevel.SIGNAL.value,
        expected_goal_status=GoalStatus.SATISFIED.value,
        expected_stop_reason=StopReason.GOAL_SATISFIED.value,
        expected_goal_success=True,
        expected_stop_correct=True,
        expected_decision_metrics=(
            (True, True, False),
            (True, True, False),
            (True, False, False),
        ),
    ),
    _AutonomousScenario(
        scenario_id="AUTONOMOUS_SCRATCH_CU_CMP_ROOT_CAUSE",
        title="Scratch and Cu CMP root cause replans after every observation",
        query=LOT_ROOT_QUERY,
        lot_id="LOT_A_001",
        expected_job_mode=InvestigationMode.LOT.value,
        expected_intent=InvestigationIntent.ROOT_CAUSE.value,
        expected_chain=_ROOT_CHAIN,
        expected_conclusion=ConclusionLevel.SUPPORTED.value,
        expected_goal_status=GoalStatus.SATISFIED.value,
        expected_stop_reason=StopReason.GOAL_SATISFIED.value,
        expected_goal_success=True,
        expected_stop_correct=True,
        expected_decision_metrics=(
            (True, True, False),
            (True, True, False),
            (True, True, False),
            (True, True, False),
            (True, False, False),
            (True, False, False),
        ),
        require_observation_replanning=True,
    ),
    _AutonomousScenario(
        scenario_id="AUTONOMOUS_LOT_HISTORY",
        title="Historical lookup reaches Knowledge only after current evidence",
        query=LOT_HISTORY_QUERY,
        lot_id="LOT_A_001",
        expected_job_mode=InvestigationMode.LOT.value,
        expected_intent=InvestigationIntent.HISTORICAL_LOOKUP.value,
        expected_chain=_HISTORY_CHAIN,
        expected_conclusion=ConclusionLevel.CANDIDATE.value,
        expected_goal_status=GoalStatus.SATISFIED.value,
        expected_stop_reason=StopReason.GOAL_SATISFIED.value,
        expected_goal_success=True,
        expected_stop_correct=True,
        expected_decision_metrics=(
            (True, True, False),
            (True, True, False),
            (True, True, False),
            (True, True, False),
            (True, True, False),
            (True, False, False),
        ),
    ),
    _AutonomousScenario(
        scenario_id="AUTONOMOUS_PRODUCT_IMPACT",
        title="Product-window impact request remains a bounded MES investigation",
        query=PRODUCT_IMPACT_QUERY,
        lot_id=None,
        expected_job_mode=InvestigationMode.PRODUCT_WINDOW.value,
        expected_intent=InvestigationIntent.IMPACT_SCOPE.value,
        expected_chain=("find_shared_exposure",),
        expected_conclusion=ConclusionLevel.SIGNAL.value,
        expected_goal_status=GoalStatus.SATISFIED.value,
        expected_stop_reason=StopReason.GOAL_SATISFIED.value,
        expected_goal_success=True,
        expected_stop_correct=True,
        expected_decision_metrics=((True, True, False), (True, False, False)),
    ),
    _AutonomousScenario(
        scenario_id="AUTONOMOUS_PRODUCT_ROOT_CAUSE",
        title="Product-window RCA uses MES to establish Lot scope before Specialists",
        query=PRODUCT_ROOT_QUERY,
        lot_id=None,
        expected_job_mode=InvestigationMode.PRODUCT_WINDOW.value,
        expected_intent=InvestigationIntent.ROOT_CAUSE.value,
        expected_chain=_PRODUCT_ROOT_CHAIN,
        expected_conclusion=ConclusionLevel.SUPPORTED.value,
        expected_goal_status=GoalStatus.SATISFIED.value,
        expected_stop_reason=StopReason.GOAL_SATISFIED.value,
        expected_goal_success=True,
        expected_stop_correct=True,
        expected_decision_metrics=(
            (True, True, False),
            (True, True, False),
            (True, True, False),
            (True, True, False),
            (True, False, False),
            (True, False, False),
        ),
    ),
    _AutonomousScenario(
        scenario_id="AUTONOMOUS_PREMATURE_STOP_GATE",
        title="Premature supported STOP is downgraded by the evidence gate",
        query=LOT_ROOT_QUERY,
        lot_id="LOT_A_001",
        expected_job_mode=InvestigationMode.LOT.value,
        expected_intent=InvestigationIntent.ROOT_CAUSE.value,
        expected_chain=(),
        expected_conclusion=ConclusionLevel.INCONCLUSIVE.value,
        expected_goal_status=GoalStatus.BLOCKED.value,
        expected_stop_reason=StopReason.DATA_UNAVAILABLE.value,
        expected_goal_success=False,
        expected_stop_correct=False,
        expected_decision_metrics=((True, False, False),),
        client_kind="premature_stop",
    ),
    _AutonomousScenario(
        scenario_id="AUTONOMOUS_PARTIAL_EVIDENCE_STOP_GATE",
        title="Partial defect Evidence cannot substitute for a supported Hypothesis",
        query=LOT_ROOT_QUERY,
        lot_id="LOT_A_001",
        expected_job_mode=InvestigationMode.LOT.value,
        expected_intent=InvestigationIntent.ROOT_CAUSE.value,
        expected_chain=("inspect_defect_pattern",),
        expected_conclusion=ConclusionLevel.SIGNAL.value,
        expected_goal_status=GoalStatus.BLOCKED.value,
        expected_stop_reason=StopReason.DATA_UNAVAILABLE.value,
        expected_goal_success=False,
        expected_stop_correct=False,
        expected_decision_metrics=(
            (True, True, False),
            (True, False, False),
        ),
        client_kind="partial_evidence_stop",
    ),
)


def _build_autonomous_workflow(
    repository: FabRepository,
    client: LLMClient,
) -> PurePythonRCAWorkflow:
    return build_workflow(
        repository,
        llm_settings=LLMSettings(agent_mode="fake"),
        llm_client=client,
        orchestration_mode=OrchestrationMode.LLM_REACT.value,
    )


def _trace_integrity_ok(state: RCAState) -> bool:
    act_decisions = [
        decision
        for decision in state.planner_decisions
        if decision.decision_type == DecisionType.ACT.value
    ]
    if len(act_decisions) != len(state.action_history):
        return False
    finding_ids = {finding.finding_id for finding in state.findings}
    evidence_ids = {evidence.evidence_id for evidence in state.evidence}
    for decision, record in zip(act_decisions, state.action_history, strict=True):
        action = decision.next_action
        if (
            action is None
            or action != record.action
            or record.status != "completed"
            or not set(record.produced_finding_ids) <= finding_ids
            or not set(record.produced_evidence_ids) <= evidence_ids
        ):
            return False
    return True


def _audit_fields_ok(state: RCAState) -> bool:
    return all(
        bool(record.action.reason.strip())
        and bool(record.action.inputs)
        and bool(record.action.scope)
        and bool(record.decision_summary.strip())
        and bool(record.produced_finding_ids)
        and bool(record.produced_evidence_ids)
        for record in state.action_history
    )


def _specialist_v2_boundary_ok(state: RCAState) -> bool:
    findings_by_id = {finding.finding_id: finding for finding in state.findings}
    specialist_records = [
        record
        for record in state.action_history
        if record.action.agent
        in {
            AgentKind.MES.value,
            AgentKind.FDC.value,
            AgentKind.DEFECT_WAT.value,
            AgentKind.KNOWLEDGE.value,
        }
    ]
    for record in specialist_records:
        for finding_id in record.produced_finding_ids:
            finding = findings_by_id.get(finding_id)
            if finding is None:
                return False
            trace = finding.details.get("specialist_v2")
            if (
                not isinstance(trace, dict)
                or trace.get("analysis_source") != "qwen"
                or trace.get("fallback_reason")
                or not isinstance(trace.get("tool_call_count"), int)
                or not 1 <= int(trace["tool_call_count"]) <= 2
            ):
                return False
    return True


def _supported_hypothesis_gate_ok(state: RCAState) -> bool:
    final_evidence_ids = {item.evidence_id for item in state.evidence}
    supported = [
        hypothesis
        for hypothesis in state.hypotheses
        if hypothesis.status == HypothesisStatus.SUPPORTED.value
    ]
    return (
        bool(supported)
        and state.conclusion_level == ConclusionLevel.SUPPORTED.value
        and all(hypothesis.evidence_ids for hypothesis in supported)
        and all(
            set(hypothesis.evidence_ids) <= final_evidence_ids
            for hypothesis in supported
        )
    )


def _observation_replanning_ok(
    client: _RecordingFakeClient,
    state: RCAState,
) -> bool:
    requests = [
        request
        for request in client.requests
        if request.prompt_name == "next_action_planner"
    ]
    if len(requests) != len(state.action_history) + 1:
        return False
    for request_index, request in enumerate(requests):
        history = request.payload.get("action_history")
        if not isinstance(history, list) or len(history) != request_index:
            return False
        if request_index == 0:
            if request.payload.get("findings") != []:
                return False
            continue
        previous_record = state.action_history[request_index - 1]
        prompt_record = history[-1]
        if prompt_record.get("action", {}).get("action_id") != previous_record.action.action_id:
            return False
        projected_evidence_ids = set(
            prompt_record.get("produced_evidence_ids", [])
        )
        previous_evidence_ids = set(previous_record.produced_evidence_ids)
        if not projected_evidence_ids <= previous_evidence_ids:
            return False
        if prompt_record.get("produced_evidence_count") != len(
            previous_record.produced_evidence_ids
        ):
            return False
        evidence_ids_truncated = bool(
            prompt_record.get("evidence_ids_truncated", False)
        )
        if evidence_ids_truncated != (
            len(projected_evidence_ids) < len(previous_evidence_ids)
        ):
            return False
        available_evidence_ids = set(request.payload.get("available_evidence_ids", []))
        if not available_evidence_ids <= {
            item.evidence_id for item in state.evidence
        }:
            return False
        if previous_evidence_ids and not (
            previous_evidence_ids & available_evidence_ids
        ):
            return False
        finding_ids = {
            finding.get("finding_id")
            for finding in request.payload.get("findings", [])
            if isinstance(finding, dict)
        }
        if not set(previous_record.produced_finding_ids) <= finding_ids:
            return False
    return True


def _decision_metric_rows(state: RCAState) -> list[dict[str, Any]]:
    evaluation = state.run_evaluation
    if evaluation is None:
        return []
    rows: list[dict[str, Any]] = []
    for decision, item in zip(
        state.planner_decisions,
        evaluation.decision_evaluations,
        strict=True,
    ):
        rows.append(
            {
                "decision_type": decision.decision_type,
                "action_kind": (
                    decision.next_action.kind if decision.next_action is not None else None
                ),
                "decision_valid": item.decision_valid,
                "evidence_gain": item.evidence_gain,
                "redundant": item.redundant,
                "planner_reason": decision.reason,
                "evaluation_reason": item.reason,
                "new_evidence_ids": list(item.new_evidence_ids),
            }
        )
    return rows


def _action_trace_rows(state: RCAState) -> list[dict[str, Any]]:
    return [
        {
            "action_kind": record.action.kind,
            "agent": record.action.agent,
            "execution_reason": record.action.reason,
            "inputs": dict(record.action.inputs),
            "scope": dict(record.action.scope),
            "observation": record.decision_summary,
            "produced_evidence_ids": list(record.produced_evidence_ids),
        }
        for record in state.action_history
    ]


def _evaluate_autonomous_scenario(
    repository: FabRepository,
    scenario: _AutonomousScenario,
) -> tuple[dict[str, Any], RCAState]:
    client: _RecordingFakeClient
    if scenario.client_kind == "premature_stop":
        client = _ImmediateUnsupportedStopClient()
    elif scenario.client_kind == "partial_evidence_stop":
        client = _PartialEvidenceUnsupportedStopClient()
    else:
        client = _RecordingFakeClient()
    state = _build_autonomous_workflow(repository, client).run(
        scenario.query,
        job_id=f"JOB_{scenario.scenario_id}",
        lot_id=scenario.lot_id,
    )
    goal = state.investigation_goal
    evaluation = state.run_evaluation
    action_chain = tuple(record.action.kind for record in state.action_history)
    actual_metrics = tuple(
        (
            item.decision_valid,
            item.evidence_gain,
            item.redundant,
        )
        for item in (evaluation.decision_evaluations if evaluation else [])
    )
    metadata = state.execution_metadata
    tool_call_count = metadata.get("tool_call_count")
    checks = {
        "job_completed": state.job.status == TaskStatus.COMPLETED.value,
        "job_mode": state.job.investigation_mode == scenario.expected_job_mode,
        "intent": goal is not None and goal.intent == scenario.expected_intent,
        "exact_action_chain": action_chain == scenario.expected_chain,
        "terminal_stop": bool(state.planner_decisions)
        and state.planner_decisions[-1].decision_type == DecisionType.STOP.value,
        "decision_count": len(state.planner_decisions)
        == len(state.action_history) + 1,
        "trace_integrity": _trace_integrity_ok(state),
        "audit_fields": _audit_fields_ok(state),
        "specialist_v2_boundary": _specialist_v2_boundary_ok(state),
        "action_budget": goal is not None and len(state.action_history) <= goal.max_steps,
        "tool_budget": goal is not None
        and type(tool_call_count) is int
        and 0 <= tool_call_count <= goal.max_tool_calls,
        "requested_llm_react": metadata.get("orchestration_requested_mode")
        == OrchestrationMode.LLM_REACT.value,
        "actual_llm_react": metadata.get("orchestration_mode")
        == OrchestrationMode.LLM_REACT.value,
        "no_orchestration_fallback": not any(
            key.startswith("orchestration_fallback_") for key in metadata
        ),
        "observation_then_replan": _observation_replanning_ok(client, state),
        "conclusion_level": state.conclusion_level == scenario.expected_conclusion,
        "goal_status": state.goal_status == scenario.expected_goal_status,
        "stop_reason": state.stop_reason == scenario.expected_stop_reason,
        "decision_metrics": actual_metrics == scenario.expected_decision_metrics,
        "goal_success": evaluation is not None
        and evaluation.goal_success == scenario.expected_goal_success,
        "stop_correct": evaluation is not None
        and evaluation.stop_correct == scenario.expected_stop_correct,
        "typed_round_trip": RCAState.from_dict(state.to_dict()) == state,
    }
    if scenario.require_observation_replanning:
        checks["scratch_replanning"] = _observation_replanning_ok(client, state)
    if scenario.expected_conclusion == ConclusionLevel.SUPPORTED.value:
        checks["supported_hypothesis_gate"] = _supported_hypothesis_gate_ok(state)
    if scenario.client_kind == "premature_stop":
        terminal = state.planner_decisions[-1]
        checks["planner_proposed_supported"] = (
            terminal.proposed_conclusion_level == ConclusionLevel.SUPPORTED.value
        )
        checks["evidence_gate_downgraded"] = (
            state.conclusion_level == ConclusionLevel.INCONCLUSIVE.value
            and not state.evidence
        )
    if scenario.client_kind == "partial_evidence_stop":
        terminal = state.planner_decisions[-1]
        checks["planner_proposed_supported"] = (
            terminal.proposed_conclusion_level == ConclusionLevel.SUPPORTED.value
        )
        checks["partial_evidence_preserved"] = bool(state.evidence)
        checks["no_supported_hypothesis"] = not any(
            hypothesis.status == HypothesisStatus.SUPPORTED.value
            for hypothesis in state.hypotheses
        )
        checks["hypothesis_gate_downgraded"] = (
            state.conclusion_level == ConclusionLevel.SIGNAL.value
            and bool(state.evidence)
        )
    terminal_decision = state.planner_decisions[-1]
    result = {
        "scenario_id": scenario.scenario_id,
        "title": scenario.title,
        "passed": all(checks.values()),
        "checks": checks,
        "job_mode": state.job.investigation_mode,
        "intent": goal.intent if goal else None,
        "action_chain": list(action_chain),
        "action_count": len(state.action_history),
        "planner_decision_count": len(state.planner_decisions),
        "tool_call_count": (
            tool_call_count if type(tool_call_count) is int else None
        ),
        "conclusion_level": state.conclusion_level,
        "goal_status": state.goal_status,
        "stop_reason": state.stop_reason,
        "goal_success": evaluation.goal_success if evaluation else None,
        "stop_correct": evaluation.stop_correct if evaluation else None,
        "decision_metrics": _decision_metric_rows(state),
        "action_trace": _action_trace_rows(state),
        "stop_trace": {
            "planner_reason": terminal_decision.reason,
            "goal_status": terminal_decision.goal_status,
            "stop_reason": terminal_decision.stop_reason,
            "final_conclusion_level": state.conclusion_level,
        },
    }
    return result, state


def _evaluate_mid_loop_fallback(repository: FabRepository) -> dict[str, Any]:
    client = _InvalidNextActionAfterFirstClient()
    state = _build_autonomous_workflow(repository, client).run(
        LOT_ROOT_QUERY,
        job_id="JOB_AUTONOMOUS_MID_LOOP_FALLBACK",
        lot_id="LOT_A_001",
    )
    metadata = state.execution_metadata
    action_chain = [record.action.kind for record in state.action_history]
    checks = {
        "job_completed": state.job.status == TaskStatus.COMPLETED.value,
        "requested_llm_react": metadata.get("orchestration_requested_mode")
        == OrchestrationMode.LLM_REACT.value,
        "controlled_tail": metadata.get("orchestration_mode")
        == OrchestrationMode.CONTROLLED_REACT.value,
        "fallback_reason": metadata.get("orchestration_fallback_reason")
        == "qwen_next_action_output_invalid",
        "fallback_stage": metadata.get("orchestration_fallback_stage")
        == "next_action_planning",
        "fallback_after_one_action": metadata.get(
            "orchestration_fallback_after_action_count"
        )
        == 1,
        "qwen_prefix_preserved": len(state.planner_decisions) == 1
        and len(state.action_history) >= 1
        and state.planner_decisions[0].next_action == state.action_history[0].action,
        "complete_controlled_chain": action_chain == list(_ROOT_CHAIN),
        "not_attributed_to_qwen": state.run_evaluation is None,
        "two_invalid_outputs": client.next_action_call_count == 3,
    }
    return {
        "scenario_id": "AUTONOMOUS_MID_LOOP_FALLBACK",
        "title": "Mid-loop invalid Qwen output preserves prefix and uses controlled tail",
        "passed": all(checks.values()),
        "checks": checks,
        "action_chain": action_chain,
        "qwen_planner_decision_count": len(state.planner_decisions),
        "run_evaluation": None,
    }


def _evaluate_intent_fallback(repository: FabRepository) -> dict[str, Any]:
    client = _InvalidIntentClient()
    state = _build_autonomous_workflow(repository, client).run(
        LOT_ROOT_QUERY,
        job_id="JOB_AUTONOMOUS_INTENT_FALLBACK",
        lot_id="LOT_A_001",
    )
    metadata = state.execution_metadata
    intent_requests = [
        request
        for request in client.requests
        if request.prompt_name == "intent_planner"
    ]
    checks = {
        "job_completed": state.job.status == TaskStatus.COMPLETED.value,
        "requested_llm_react": metadata.get("orchestration_requested_mode")
        == OrchestrationMode.LLM_REACT.value,
        "controlled_compatibility_path": metadata.get("orchestration_mode")
        == OrchestrationMode.CONTROLLED_REACT.value,
        "fallback_reason": metadata.get("orchestration_fallback_reason")
        == "qwen_intent_output_invalid",
        "fallback_stage": metadata.get("orchestration_fallback_stage")
        == "intent_planning",
        "fallback_before_action": metadata.get(
            "orchestration_fallback_after_action_count"
        )
        == 0,
        "no_qwen_decision_committed": state.planner_decisions == [],
        "not_attributed_to_qwen": state.run_evaluation is None,
        "two_invalid_outputs": len(intent_requests) == 2,
    }
    return {
        "scenario_id": "AUTONOMOUS_INTENT_FALLBACK",
        "title": "Invalid intent output uses controlled compatibility path",
        "passed": all(checks.values()),
        "checks": checks,
        "action_chain": [record.action.kind for record in state.action_history],
        "qwen_planner_decision_count": 0,
        "run_evaluation": None,
    }


def _evaluate_semantic_negative_case() -> dict[str, Any]:
    """Prove unrelated FDC Evidence cannot close an unsupported material gap."""

    question = InvestigationQuestion(
        question_id="GOAL_MATERIAL:q:material_trace",
        goal_id="GOAL_MATERIAL",
        question="Which material batch is linked to the source Lot?",
        rationale="The user explicitly requested material genealogy.",
        question_kind=QuestionKind.MATERIAL_TRACE.value,
        scope={"lot_id": "LOT_A_001"},
    )
    action = InvestigationAction(
        action_id="ACT_FDC_MATERIAL_NEGATIVE",
        kind=ActionKind.INSPECT_FDC_SPC.value,
        agent=AgentKind.FDC.value,
        reason="Attempted only as a negative semantic-gating fixture.",
        inputs={"lot_id": "LOT_A_001"},
        scope={"lot_id": "LOT_A_001"},
    )
    item = EvidenceBuilder.from_tool(
        tool_input=ToolInput(
            tool_name="perform_basic_spc_analysis",
            request_id="semantic-negative:fdc",
            parameters={"lot_id": "LOT_A_001"},
            requested_by=AgentKind.FDC.value,
        ),
        evidence_id="EV_SEMANTIC_FDC_ONLY",
        evidence_type=EvidenceType.PARAMETER_DEVIATION,
        source_type=EvidenceSourceType.FDC,
        source_id="fdc:LOT_A_001",
        observation="Cu CMP thickness parameter deviation is observed.",
        entities=[EvidenceEntity(entity_type=EntityType.LOT.value, entity_id="LOT_A_001")],
        confidence=0.9,
    )
    record = ActionRecord(
        action=action,
        status="completed",
        produced_finding_ids=["FINDING_SEMANTIC_NEGATIVE"],
        produced_evidence_ids=[item.evidence_id],
        decision_summary="FDC Evidence exists but does not identify material genealogy.",
    )
    links = QuestionEvidenceResolver().resolve(
        questions=[question],
        action_record=record,
        evidence=[item],
    )
    notice = capability_notice_for(QuestionKind.MATERIAL_TRACE)
    decision = PlannerDecision(
        decision_id="DECISION_SEMANTIC_NEGATIVE",
        goal_id=question.goal_id,
        decision_type=DecisionType.STOP.value,
        reason="The requested material source is not configured.",
        goal_status=GoalStatus.BLOCKED.value,
        proposed_conclusion_level=ConclusionLevel.CANDIDATE.value,
        stop_reason=StopReason.DATA_UNAVAILABLE.value,
    )
    review = review_question_updates(
        decision,
        [
            {
                "question_id": question.question_id,
                "status": EvidenceGapStatus.CLOSED.value,
                "answer": "The FDC deviation proves the material batch.",
                "evidence_ids": [item.evidence_id],
                "unavailable_reason": None,
            }
        ],
        questions=[question],
        available_evidence_ids={item.evidence_id},
        question_evidence_links=links,
        capability_notices=[notice],
    )
    checks = {
        "fdc_evidence_has_no_material_link": links == [],
        "unsupported_notice_is_typed": (
            notice.capability == QuestionKind.MATERIAL_TRACE.value
            and not notice.supported
            and notice.request_source == "user"
        ),
        "fabricated_close_is_rejected": (
            review.decision.question_updates == []
            and len(review.question_update_reviews) == 1
            and review.question_update_reviews[0].reason_code
            == "unsupported_capability"
        ),
        "conclusion_not_supported": decision.proposed_conclusion_level
        != ConclusionLevel.SUPPORTED.value,
    }
    return {
        "scenario_id": "SEMANTIC_MATERIAL_TRACE_NEGATIVE",
        "title": "FDC Evidence cannot close an unsupported material-trace Question",
        "passed": all(checks.values()),
        "checks": checks,
        "question_kind": question.question_kind,
        "evidence_ids": [item.evidence_id],
        "link_count": len(links),
        "review_reason_code": review.question_update_reviews[0].reason_code,
    }


def _metric_summary(
    autonomous_states: list[tuple[_AutonomousScenario, RCAState]],
) -> dict[str, Any]:
    evaluations = [
        (scenario, state.run_evaluation)
        for scenario, state in autonomous_states
        if state.run_evaluation is not None
    ]
    decision_evaluations = [
        item
        for _, evaluation in evaluations
        if evaluation is not None
        for item in evaluation.decision_evaluations
    ]
    act_evaluations = [
        evaluation_item
        for _, state in autonomous_states
        if state.run_evaluation is not None
        for decision, evaluation_item in zip(
            state.planner_decisions,
            state.run_evaluation.decision_evaluations,
            strict=True,
        )
        if decision.decision_type == DecisionType.ACT.value
    ]
    positive_runs = [
        (scenario, evaluation)
        for scenario, evaluation in evaluations
        if scenario.expected_goal_success and scenario.expected_stop_correct
    ]

    def rate(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 6) if denominator else 0.0

    valid_count = sum(item.decision_valid for item in decision_evaluations)
    gain_count = sum(item.evidence_gain for item in act_evaluations)
    redundant_count = sum(item.redundant for item in act_evaluations)
    goal_success_count = sum(
        evaluation.goal_success
        for _, evaluation in positive_runs
        if evaluation is not None
    )
    stop_correct_count = sum(
        evaluation.stop_correct
        for _, evaluation in positive_runs
        if evaluation is not None
    )
    return {
        "decision_valid": {
            "valid_count": valid_count,
            "decision_count": len(decision_evaluations),
            "rate": rate(valid_count, len(decision_evaluations)),
        },
        "evidence_gain": {
            "gain_count": gain_count,
            "act_decision_count": len(act_evaluations),
            "rate": rate(gain_count, len(act_evaluations)),
        },
        "redundant": {
            "redundant_count": redundant_count,
            "act_decision_count": len(act_evaluations),
            "rate": rate(redundant_count, len(act_evaluations)),
        },
        "goal_success": {
            "successful_count": goal_success_count,
            "positive_run_count": len(positive_runs),
            "rate": rate(goal_success_count, len(positive_runs)),
        },
        "stop_correct": {
            "correct_count": stop_correct_count,
            "positive_run_count": len(positive_runs),
            "rate": rate(stop_correct_count, len(positive_runs)),
        },
    }


def _fixed_baseline_summary(evaluation: dict[str, Any]) -> dict[str, Any]:
    results = evaluation["results"]
    metrics = evaluation["metrics"]
    acceptance = {
        str(key): bool(value)
        for key, value in evaluation["acceptance"].items()
    }
    checks = {
        "evaluation_passed": bool(evaluation["passed"]),
        "all_acceptance_checks": all(acceptance.values()),
        "ten_scenarios": metrics["scenario_count"] == 10,
        "all_scenarios_passed": metrics["scenario_pass_rate"] == 1.0,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "passed": all(checks.values()),
        "checks": checks,
        "scenario_count": int(metrics["scenario_count"]),
        "scenario_pass_count": sum(bool(item["passed"]) for item in results),
        "acceptance": acceptance,
        "scenarios": [
            {
                "scenario_id": str(item["scenario_id"]),
                "passed": bool(item["passed"]),
            }
            for item in results
        ],
    }


def _controlled_compatibility_summary(repository: FabRepository) -> dict[str, Any]:
    """Run the preserved deterministic controlled-ReAct compatibility path."""

    state = build_workflow(
        repository,
        llm_settings=LLMSettings(agent_mode="deterministic"),
        orchestration_mode=OrchestrationMode.CONTROLLED_REACT.value,
    ).run(
        LOT_ROOT_QUERY,
        job_id="JOB_BATCH_21_2_CONTROLLED_COMPATIBILITY",
        lot_id="LOT_A_001",
    )
    action_chain = [record.action.kind for record in state.action_history]
    expected_chain = list(_ROOT_CHAIN)
    checks = {
        "job_completed": state.job.status == TaskStatus.COMPLETED.value,
        "exact_scratch_chain": action_chain == expected_chain,
        "supported_conclusion": state.conclusion_level == ConclusionLevel.SUPPORTED.value,
        "goal_satisfied": state.goal_status == GoalStatus.SATISFIED.value,
        "terminal_stop": state.stop_reason == StopReason.GOAL_SATISFIED.value,
        "not_attributed_to_qwen": state.run_evaluation is None,
        # Semantic links are additive for compatibility mode; the historical
        # controlled path is allowed to omit the autonomous link projection.
        "semantic_links_are_additive": state.question_evidence_links == [],
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "passed": all(checks.values()),
        "checks": checks,
        "action_chain": action_chain,
        "conclusion_level": state.conclusion_level,
        "goal_status": state.goal_status,
        "stop_reason": state.stop_reason,
    }


def evaluate_autonomous_qwen_react(
    autonomous_repository: FabRepository,
    *,
    fixed_repository: FabRepository,
    fixed_scenarios: list[EvaluationScenario],
    real_qwen_status: str = "SKIPPED",
    real_qwen_reason: str = (
        "DASHSCOPE_API_KEY and RUN_REAL_QWEN_TEST=1 are not configured."
    ),
) -> dict[str, Any]:
    """Run the stable autonomous matrix and the fixed compatibility baseline."""

    if real_qwen_status not in {"PASS", "FAIL", "SKIPPED"}:
        raise ValueError("real_qwen_status must be PASS, FAIL, or SKIPPED")
    if not real_qwen_reason.strip():
        raise ValueError("real_qwen_reason must be non-empty")

    autonomous_results: list[dict[str, Any]] = []
    autonomous_states: list[tuple[_AutonomousScenario, RCAState]] = []
    for scenario in _AUTONOMOUS_SCENARIOS:
        result, state = _evaluate_autonomous_scenario(
            autonomous_repository,
            scenario,
        )
        autonomous_results.append(result)
        autonomous_states.append((scenario, state))

    fallback_results = [
        _evaluate_mid_loop_fallback(autonomous_repository),
        _evaluate_intent_fallback(autonomous_repository),
    ]
    all_fake_results = [*autonomous_results, *fallback_results]
    by_id = {item["scenario_id"]: item for item in autonomous_results}
    cross_scenario_checks = {
        "intent_changes_action_chain": (
            by_id["AUTONOMOUS_LOT_IMPACT"]["action_chain"]
            != by_id["AUTONOMOUS_SCRATCH_CU_CMP_ROOT_CAUSE"]["action_chain"]
        ),
        "product_scope_starts_with_mes": (
            by_id["AUTONOMOUS_PRODUCT_ROOT_CAUSE"]["action_chain"][0]
            == "find_shared_exposure"
        ),
        "scratch_observation_then_action": by_id[
            "AUTONOMOUS_SCRATCH_CU_CMP_ROOT_CAUSE"
        ]["checks"]["scratch_replanning"],
        "evidence_gate_downgrade": by_id[
            "AUTONOMOUS_PREMATURE_STOP_GATE"
        ]["checks"]["evidence_gate_downgraded"],
        "partial_evidence_hypothesis_gate": by_id[
            "AUTONOMOUS_PARTIAL_EVIDENCE_STOP_GATE"
        ]["checks"]["hypothesis_gate_downgraded"],
        "fallback_attribution": all(
            result["checks"]["not_attributed_to_qwen"]
            for result in fallback_results
        ),
    }
    fake_passed = all(item["passed"] for item in all_fake_results) and all(
        cross_scenario_checks.values()
    )

    fixed_evaluation = evaluate_scenarios(fixed_repository, fixed_scenarios)
    fixed_summary = _fixed_baseline_summary(fixed_evaluation)
    controlled_summary = _controlled_compatibility_summary(autonomous_repository)
    semantic_negative = _evaluate_semantic_negative_case()
    deterministic_passed = (
        fake_passed
        and controlled_summary["passed"]
        and fixed_summary["passed"]
        and semantic_negative["passed"]
    )
    return {
        "schema_version": "1.1",
        "suite": "batch_21_2_product_surface_semantic_evaluation",
        "passed": deterministic_passed,
        "lanes": {
            "autonomous_fake": {
                "status": "PASS" if fake_passed else "FAIL",
                "passed": fake_passed,
                "scenario_count": len(all_fake_results),
                "scenario_pass_count": sum(
                    bool(item["passed"]) for item in all_fake_results
                ),
            },
            "fixed_workflow": fixed_summary,
            "controlled_react": controlled_summary,
            "real_qwen_smoke": {
                "status": real_qwen_status,
                "required_for_deterministic_acceptance": False,
                "reason": real_qwen_reason,
            },
        },
        "metrics": _metric_summary(autonomous_states),
        "acceptance": cross_scenario_checks,
        "semantic_evaluation": {
            "status": "PASS" if semantic_negative["passed"] else "FAIL",
            "scenario_count": 1,
            "scenario_pass_count": 1 if semantic_negative["passed"] else 0,
            "scenarios": [semantic_negative],
        },
        "autonomous_scenarios": autonomous_results,
        "fallback_scenarios": fallback_results,
    }


def render_autonomous_qwen_report(evaluation: dict[str, Any]) -> str:
    """Render a stable Markdown report without prompts, secrets, or random IDs."""

    lanes = evaluation["lanes"]
    metrics = evaluation["metrics"]
    lines = [
        "# Batch 21.2 Product Surface + Semantic Evaluation",
        "",
        f"Deterministic acceptance: **{'PASS' if evaluation['passed'] else 'FAIL'}**",
        "",
        "The deterministic result requires the Fake-Qwen autonomous lane, the "
        "controlled compatibility path, and the fixed-workflow baseline. The optional real-Qwen "
        "smoke is reported separately and is not converted into a pass.",
        "",
        "## Verification lanes",
        "",
        "| Lane | Status | Result |",
        "| --- | --- | --- |",
        (
            "| Autonomous Fake | "
            f"{lanes['autonomous_fake']['status']} | "
            f"{lanes['autonomous_fake']['scenario_pass_count']}/"
            f"{lanes['autonomous_fake']['scenario_count']} scenarios |"
        ),
        (
            "| Controlled ReAct | "
            f"{lanes['controlled_react']['status']} | "
            f"{'Scratch + Cu CMP compatibility path'} |"
        ),
        (
            "| Fixed workflow | "
            f"{lanes['fixed_workflow']['status']} | "
            f"{lanes['fixed_workflow']['scenario_pass_count']}/"
            f"{lanes['fixed_workflow']['scenario_count']} scenarios |"
        ),
        (
            "| Real Qwen smoke | "
            f"{lanes['real_qwen_smoke']['status']} | "
            f"{lanes['real_qwen_smoke']['reason']} |"
        ),
        "",
        "## Five contract metrics",
        "",
        "| Metric | Count | Rate |",
        "| --- | --- | --- |",
        (
            "| Decision valid | "
            f"{metrics['decision_valid']['valid_count']}/"
            f"{metrics['decision_valid']['decision_count']} | "
            f"{metrics['decision_valid']['rate']:.1%} |"
        ),
        (
            "| Evidence gain (ACT only) | "
            f"{metrics['evidence_gain']['gain_count']}/"
            f"{metrics['evidence_gain']['act_decision_count']} | "
            f"{metrics['evidence_gain']['rate']:.1%} |"
        ),
        (
            "| Redundant (ACT only) | "
            f"{metrics['redundant']['redundant_count']}/"
            f"{metrics['redundant']['act_decision_count']} | "
            f"{metrics['redundant']['rate']:.1%} |"
        ),
        (
            "| Goal success (positive autonomous runs) | "
            f"{metrics['goal_success']['successful_count']}/"
            f"{metrics['goal_success']['positive_run_count']} | "
            f"{metrics['goal_success']['rate']:.1%} |"
        ),
        (
            "| Stop correct (positive autonomous runs) | "
            f"{metrics['stop_correct']['correct_count']}/"
            f"{metrics['stop_correct']['positive_run_count']} | "
            f"{metrics['stop_correct']['rate']:.1%} |"
        ),
        "",
        "Evidence gain is descriptive rather than a target of 100%: RCA reasoning "
        "correctly adds analysis without inventing new Evidence, so it records "
        "`evidence_gain=false` and `redundant=false`.",
        "",
        "## Semantic negative cases",
        "",
        "The semantic lane is a required acceptance boundary, not a sixth public metric.",
        "",
        "| Scenario | Status | Link count | Review result |",
        "| --- | --- | ---: | --- |",
    ]
    for semantic in evaluation.get("semantic_evaluation", {}).get("scenarios", []):
        lines.append(
            f"| {semantic['scenario_id']} | "
            f"{'PASS' if semantic['passed'] else 'FAIL'} | "
            f"{semantic['link_count']} | `{semantic['review_reason_code']}` |"
        )
    lines.extend(
        [
            "",
        "## Autonomous scenarios",
        "",
        "| Scenario | Status | Intent | Action chain | Conclusion | Goal / Stop |",
        "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for result in evaluation["autonomous_scenarios"]:
        chain = " -> ".join(result["action_chain"]) or "(no action)"
        lines.append(
            f"| {result['scenario_id']} | "
            f"{'PASS' if result['passed'] else 'FAIL'} | "
            f"{result['intent']} | {chain} | "
            f"{result['conclusion_level']} | "
            f"{result['goal_success']} / {result['stop_correct']} |"
        )
    scratch_result = next(
        result
        for result in evaluation["autonomous_scenarios"]
        if result["scenario_id"] == "AUTONOMOUS_SCRATCH_CU_CMP_ROOT_CAUSE"
    )
    lines.extend(
        [
            "",
            "The two premature-stop scenarios pass only when the Planner's "
            "proposed `supported` conclusion is downgraded. With no Evidence the "
            "result is `inconclusive`; with only defect Evidence and no supported "
            "Hypothesis it remains a `signal`. Both keep `goal_success=false` and "
            "`stop_correct=false`.",
            "",
            "## Scratch + Cu CMP action audit",
            "",
            "| Action / Agent | Execution reason | Inputs | Output Evidence | Observation |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for trace in scratch_result["action_trace"]:
        inputs = ", ".join(
            f"{key}={value}"
            for key, value in sorted(trace["inputs"].items())
        )
        evidence_ids = ", ".join(trace["produced_evidence_ids"])
        reason = str(trace["execution_reason"]).replace("|", "\\|")
        observation = str(trace["observation"]).replace("|", "\\|")
        lines.append(
            f"| {trace['action_kind']} / {trace['agent']} | {reason} | "
            f"{inputs} | {evidence_ids} | {observation} |"
        )
    stop_trace = scratch_result["stop_trace"]
    lines.extend(
        [
            "",
            (
                "Terminal STOP: "
                f"{stop_trace['planner_reason']} "
                f"(`{stop_trace['stop_reason']}`, "
                f"final level `{stop_trace['final_conclusion_level']}`)."
            ),
            "",
            "## Compatibility fallback scenarios",
            "",
            "| Scenario | Status | Qwen-attributed evaluation |",
            "| --- | --- | --- |",
        ]
    )
    for result in evaluation["fallback_scenarios"]:
        lines.append(
            f"| {result['scenario_id']} | "
            f"{'PASS' if result['passed'] else 'FAIL'} | null |"
        )
    lines.extend(
        [
            "",
            "Fallback scenarios preserve completed work but deliberately leave "
            "`run_evaluation=null`; the controlled tail is not attributed to Qwen.",
            "",
            "## Fixed-workflow baseline",
            "",
            (
                f"Status: **{lanes['fixed_workflow']['status']}**; "
                f"{lanes['fixed_workflow']['scenario_pass_count']}/"
                f"{lanes['fixed_workflow']['scenario_count']} scenarios passed; "
                "all established acceptance checks remained true."
            ),
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "evaluate_autonomous_qwen_react",
    "render_autonomous_qwen_report",
]
