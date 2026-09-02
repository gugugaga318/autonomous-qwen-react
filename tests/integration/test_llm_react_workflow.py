from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "backend"))

from fastapi.testclient import TestClient  # noqa: E402
from yield_rca_api.app import create_app  # noqa: E402
from yield_rca_core.investigation_models import (  # noqa: E402
    ConclusionLevel,
    DecisionType,
    InvestigationAction,
    InvestigationIntent,
    QuestionUpdateReasonCode,
    StopReason,
)
from yield_rca_core.llm_gateway import (  # noqa: E402
    FakeLLMClient,
    LLMCallError,
    LLMRequest,
    LLMResponse,
    LLMSettings,
)
from yield_rca_core.models import (  # noqa: E402
    AgentFinding,
    AgentKind,
    RCAJob,
    RCAState,
    TaskStatus,
)
from yield_rca_core.specialist_v2 import SpecialistV2Error  # noqa: E402
from yield_rca_core.supervisor import SupervisorExecutionError  # noqa: E402
from yield_rca_core.workflow import build_csv_workflow  # noqa: E402

SEED_DIR = ROOT / "data" / "seeds" / "golden_case"
ROOT_CAUSE_QUERY = (
    "Investigate the root cause of LOT_A_001 scratch in Cu CMP."
)
FULL_RCA_QUERY = (
    "Investigate the scratch found in Cu CMP and identify root cause and impact lots."
)
IMPACT_QUERY = "Identify the impact lots for LOT_A_001."
PRODUCT_IMPACT_QUERY = (
    "Identify impact lots for 40N_SOC from 2026-07-01 to 2026-07-31."
)
PRODUCT_ROOT_QUERY = (
    "Investigate 40N_SOC yield loss root cause from 2026-07-01 to 2026-07-31."
)
PRODUCT_HISTORY_QUERY = (
    "Find a similar historical case for 40N_SOC from 2026-07-01 to 2026-07-31."
)


class RecordingFakeClient(FakeLLMClient):
    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return super().complete_json(request)


class ExposureFirstRootCauseClient(RecordingFakeClient):
    """Choose a different legal first direction for the same root-cause intent."""

    def __init__(self) -> None:
        super().__init__()
        self.next_action_call_count = 0

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        response = super().complete_json(request)
        if request.prompt_name != "next_action_planner":
            return response
        self.next_action_call_count += 1
        if self.next_action_call_count != 1:
            return response
        target_ids = request.payload["legal_target_question_ids_by_action"][
            "find_shared_exposure"
        ]
        known_facts = dict(request.payload["goal"]["known_facts"])
        return LLMResponse(
            data={
                "decision_id": "EXPOSURE_FIRST_DECISION",
                "goal_id": request.payload["goal"]["goal_id"],
                "decision_type": DecisionType.ACT.value,
                "reason": (
                    "The initial symptom is already known; collect the shared "
                    "equipment and recipe exposure before selecting the next domain."
                ),
                "goal_status": "in_progress",
                "proposed_conclusion_level": ConclusionLevel.SIGNAL.value,
                "next_action": {
                    "action_id": "EXPOSURE_FIRST_ACTION",
                    "kind": "find_shared_exposure",
                    "agent": "mes",
                    "reason": "Establish the shared process exposure first.",
                    "inputs": known_facts,
                    "scope": known_facts,
                    "required_evidence_ids": [],
                    "max_attempts": 1,
                },
                "target_question_ids": target_ids,
                "new_questions": [],
                "stop_reason": None,
                "question_updates": [],
            },
            usage=response.usage,
        )


class InvalidNextActionAfterFirstClient(RecordingFakeClient):
    """Run one real observation, then fail both structured-output attempts."""

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


class TransientNextActionCallFailureClient(RecordingFakeClient):
    def __init__(self) -> None:
        super().__init__()
        self.failure_injected = False

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        if request.prompt_name == "next_action_planner" and not self.failure_injected:
            self.requests.append(request)
            self.failure_injected = True
            raise LLMCallError(
                "temporary throttling",
                status_code=429,
                provider_code="Throttling",
                failure_category="provider_http_error",
            )
        return super().complete_json(request)


class PersistentNextActionCallFailureClient(RecordingFakeClient):
    def complete_json(self, request: LLMRequest) -> LLMResponse:
        if request.prompt_name == "next_action_planner":
            self.requests.append(request)
            raise LLMCallError(
                "persistent throttling",
                status_code=429,
                provider_code="Throttling",
                provider_message=(
                    "Authorization: Bearer workflow-secret api_key=workflow-secret"
                ),
                request_id="req-workflow-429",
                failure_category="provider_http_error",
            )
        return super().complete_json(request)


class CandidateGenerationTransportFailureClient(RecordingFakeClient):
    """Fail Candidate generation while keeping Planner/Specialists deterministic."""

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        if request.prompt_name == "hypothesis_candidate_generator":
            self.requests.append(request)
            raise LLMCallError(
                "candidate provider transport failure",
                failure_category="transport_error",
                call_attempt_count=2,
            )
        return super().complete_json(request)


class FallbackThenCandidateTransportFailureClient(
    InvalidNextActionAfterFirstClient
):
    """Exercise Candidate failure after a mid-loop Planner fallback."""

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        if request.prompt_name == "hypothesis_candidate_generator":
            self.requests.append(request)
            raise LLMCallError(
                "candidate provider transport failure after Planner fallback",
                failure_category="transport_error",
                call_attempt_count=2,
            )
        return super().complete_json(request)


class InvalidIntentClient(RecordingFakeClient):
    def complete_json(self, request: LLMRequest) -> LLMResponse:
        response = super().complete_json(request)
        if request.prompt_name == "intent_planner":
            return LLMResponse(data={}, usage=response.usage)
        return response


class IntentPlannerTransportFailureClient(RecordingFakeClient):
    def complete_json(self, request: LLMRequest) -> LLMResponse:
        if request.prompt_name == "intent_planner":
            self.requests.append(request)
            raise LLMCallError(
                "intent planner transport failure",
                failure_category="transport_error",
                call_attempt_count=2,
            )
        return super().complete_json(request)


class ImmediateUnsupportedStopClient(RecordingFakeClient):
    """Model proposes a supported conclusion before collecting any Evidence."""

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        response = super().complete_json(request)
        if request.prompt_name != "next_action_planner":
            return response
        goal_id = str(request.payload["goal"]["goal_id"])
        return LLMResponse(
            data={
                "decision_id": f"{goal_id}:model-stop",
                "goal_id": goal_id,
                "decision_type": DecisionType.STOP.value,
                "reason": "The model cannot obtain the requested source data.",
                "goal_status": "blocked",
                "proposed_conclusion_level": ConclusionLevel.SUPPORTED.value,
                "next_action": None,
                "target_question_ids": [],
                "new_questions": [],
                "stop_reason": StopReason.DATA_UNAVAILABLE.value,
                "question_updates": [],
            },
            usage=response.usage,
        )


class NextActionCallCapClient(RecordingFakeClient):
    def complete_json(self, request: LLMRequest) -> LLMResponse:
        if request.prompt_name == "next_action_planner":
            self.requests.append(request)
            raise LLMCallError(
                "formal blind call cap reached",
                failure_category="formal_blind_call_cap",
            )
        return super().complete_json(request)


class RejectedQuestionUpdateThenStopClient(RecordingFakeClient):
    """Emit one legal Action with an unsafe ancillary QuestionUpdate claim."""

    def __init__(self, *, close_and_target: bool = False) -> None:
        super().__init__()
        self.close_and_target = close_and_target
        self.next_action_call_count = 0

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        response = super().complete_json(request)
        if request.prompt_name != "next_action_planner":
            return response
        self.next_action_call_count += 1
        goal_id = str(request.payload["goal"]["goal_id"])
        if self.next_action_call_count > 1:
            return LLMResponse(
                data={
                    "decision_id": f"{goal_id}:review-test-stop",
                    "goal_id": goal_id,
                    "decision_type": DecisionType.STOP.value,
                    "reason": "Stop after the one bounded review test action.",
                    "goal_status": "blocked",
                    "proposed_conclusion_level": ConclusionLevel.INCONCLUSIVE.value,
                    "next_action": None,
                    "target_question_ids": [],
                    "new_questions": [],
                    "stop_reason": StopReason.NO_ALLOWED_ACTION.value,
                    "question_updates": [],
                },
                usage=response.usage,
            )

        data = dict(response.data)
        question_id = str(data["target_question_ids"][0])
        data["question_updates"] = (
            [
                {
                    "question_id": question_id,
                    "status": "closed",
                    "answer": "The still-targeted Question is already answered.",
                    "evidence_ids": ["EV_NOT_AVAILABLE"],
                    "unavailable_reason": None,
                }
            ]
            if self.close_and_target
            else [
                {
                    "question_id": question_id,
                    "status": "open",
                    "answer": None,
                    "evidence_ids": [],
                    "unavailable_reason": None,
                }
            ]
        )
        return LLMResponse(data=data, usage=response.usage)


class FailingSpecialistExecutor:
    def execute(self, *args: object, **kwargs: object) -> None:
        raise SpecialistV2Error(
            "Injected Specialist failure before a Finding was produced.",
            stage="tool_execution",
            reason="injected_test_failure",
        )


class ContextRecordingSpecialistExecutor:
    def __init__(self, finding: AgentFinding) -> None:
        self.finding = finding
        self.context: dict[str, object] | None = None

    def execute(
        self,
        *args: object,
        context: dict[str, object],
        **kwargs: object,
    ) -> AgentFinding:
        self.context = dict(context)
        return self.finding


def fake_llm_workflow(client: FakeLLMClient):
    return build_csv_workflow(
        SEED_DIR,
        llm_settings=LLMSettings(agent_mode="fake"),
        llm_client=client,
        orchestration_mode="llm_react",
    )


def run_lot(
    client: FakeLLMClient,
    query: str,
    *,
    job_id: str,
) -> RCAState:
    return fake_llm_workflow(client).run(
        query,
        job_id=job_id,
        lot_id="LOT_A_001",
    )


class LLMReactWorkflowIntegrationTest(unittest.TestCase):
    def test_one_transient_next_action_failure_recovers_on_llm_react(self) -> None:
        client = TransientNextActionCallFailureClient()

        state = run_lot(
            client,
            ROOT_CAUSE_QUERY,
            job_id="JOB_LLM_REACT_TRANSIENT_RECOVERY",
        )

        self.assertEqual(state.execution_metadata["orchestration_mode"], "llm_react")
        self.assertNotIn(
            "orchestration_fallback_reason",
            state.execution_metadata,
        )
        next_action_requests = [
            request
            for request in client.requests
            if request.prompt_name == "next_action_planner"
        ]
        self.assertEqual(len(next_action_requests), len(state.action_history) + 2)
        self.assertEqual(
            next_action_requests[0].payload["output_attempt"],
            next_action_requests[1].payload["output_attempt"],
        )

    def test_two_next_action_call_failures_fallback_with_safe_diagnostics(
        self,
    ) -> None:
        state = run_lot(
            PersistentNextActionCallFailureClient(),
            ROOT_CAUSE_QUERY,
            job_id="JOB_LLM_REACT_PROVIDER_FALLBACK",
        )

        metadata = state.execution_metadata
        self.assertEqual(metadata["orchestration_mode"], "controlled_react")
        self.assertEqual(
            metadata["orchestration_fallback_reason"],
            "qwen_next_action_call_failed",
        )
        self.assertEqual(
            metadata["orchestration_fallback_failure_category"],
            "provider_http_error",
        )
        self.assertEqual(metadata["orchestration_fallback_call_attempt_count"], 2)
        self.assertEqual(metadata["orchestration_fallback_status_code"], 429)
        self.assertEqual(
            metadata["orchestration_fallback_provider_code"],
            "Throttling",
        )
        self.assertEqual(
            metadata["orchestration_fallback_request_id"],
            "req-workflow-429",
        )
        self.assertNotIn(
            "workflow-secret",
            metadata["orchestration_fallback_provider_message"],
        )

    def test_candidate_transport_failure_cannot_crash_on_repeated_action_scope(
        self,
    ) -> None:
        client = CandidateGenerationTransportFailureClient()
        workflow = build_csv_workflow(
            SEED_DIR,
            llm_settings=LLMSettings(agent_mode="llm", api_key="test-only-key"),
            llm_client=client,
            orchestration_mode="llm_react",
        )

        state = workflow.run(
            ROOT_CAUSE_QUERY,
            job_id="JOB_CANDIDATE_PROVIDER_GOVERNED_STOP",
            lot_id="LOT_A_001",
        )

        self.assertEqual(state.job.status, TaskStatus.COMPLETED.value)
        self.assertEqual(
            state.execution_metadata["orchestration_mode"],
            "llm_react",
        )
        self.assertNotIn(
            "orchestration_fallback_reason",
            state.execution_metadata,
        )
        self.assertEqual(
            state.execution_metadata["planner_stop_proposed_by"],
            "python_runtime",
        )
        self.assertEqual(
            state.execution_metadata["terminal_question_updates_source"],
            "python_evidence_gate",
        )
        self.assertNotEqual(state.stop_reason, StopReason.GOAL_SATISFIED.value)
        self.assertEqual(state.conclusion_level, ConclusionLevel.INCONCLUSIVE.value)
        self.assertEqual(
            sum(
                record.action.kind == "run_rca_reasoning"
                for record in state.action_history
            ),
            1,
        )
        self.assertIsNotNone(state.competition_trace)
        assert state.competition_trace is not None
        self.assertEqual(state.competition_trace.competition_status, "failed")
        self.assertEqual(
            state.competition_trace.competition_failure_reason,
            "candidate_provider_failed",
        )
        self.assertEqual(
            state.competition_trace.terminal_reason,
            "candidate_competition_processing_failed",
        )
        self.assertEqual(
            state.execution_metadata["investigation_finalizer"],
            "python_competition_progression",
        )
        self.assertTrue(
            any(
                request.prompt_name == "hypothesis_candidate_generator"
                for request in client.requests
            )
        )

    def test_mid_loop_fallback_keeps_qwen_terminal_and_impact_governance(
        self,
    ) -> None:
        client = FallbackThenCandidateTransportFailureClient()
        workflow = build_csv_workflow(
            SEED_DIR,
            llm_settings=LLMSettings(agent_mode="llm", api_key="test-only-key"),
            llm_client=client,
            orchestration_mode="llm_react",
        )

        state = workflow.run(
            ROOT_CAUSE_QUERY,
            job_id="JOB_MID_LOOP_FALLBACK_CANDIDATE_PROVIDER_FAILURE",
            lot_id="LOT_A_001",
        )

        self.assertEqual(
            state.execution_metadata["orchestration_requested_mode"],
            "llm_react",
        )
        self.assertEqual(
            state.execution_metadata["orchestration_mode"],
            "controlled_react",
        )
        mes_finding = next(
            finding for finding in state.findings if finding.agent == AgentKind.MES.value
        )
        self.assertTrue(mes_finding.details["impact_lots"])
        rca_finding = state.authoritative_rca_finding
        self.assertIsNotNone(rca_finding)
        assert rca_finding is not None
        impact_gate = rca_finding.details["impact_lot_gate"]
        self.assertTrue(impact_gate["observed_impact_lots"])
        self.assertEqual(impact_gate["confirmed_impact_lots"], [])
        self.assertIsNotNone(state.competition_trace)
        assert state.competition_trace is not None
        self.assertEqual(state.competition_trace.competition_status, "failed")
        self.assertEqual(
            state.competition_trace.competition_failure_reason,
            "candidate_provider_failed",
        )

        with self.subTest("authoritative impact publication"):
            self.assertEqual(state.impact_lots, [])
            assert state.report is not None
            self.assertIn("- Impact Lot Count: 0", state.report.markdown)
            self.assertIn("- Impact Lots: None identified.", state.report.markdown)
        with self.subTest("Python terminal projection"):
            self.assertEqual(
                state.competition_trace.terminal_reason,
                "candidate_competition_processing_failed",
            )
            self.assertEqual(
                state.execution_metadata["investigation_finalizer"],
                "python_competition_progression",
            )
            self.assertEqual(
                state.execution_metadata["terminal_question_updates_source"],
                "python_evidence_gate",
            )
            self.assertEqual(
                state.execution_metadata["planner_stop_proposed_by"],
                "python_runtime",
            )
            self.assertEqual(state.goal_status, "blocked")
            self.assertEqual(
                state.stop_reason,
                StopReason.NO_HIGH_VALUE_ACTION.value,
            )
            self.assertEqual(state.conclusion_level, "inconclusive")
        self.assertEqual(RCAState.from_dict(state.to_dict()), state)

    def test_llm_call_cap_ends_llm_react_without_controlled_fallback(self) -> None:
        state = run_lot(
            NextActionCallCapClient(),
            ROOT_CAUSE_QUERY,
            job_id="JOB_LLM_REACT_CALL_CAP",
        )

        self.assertEqual(state.job.status, "completed")
        self.assertEqual(state.execution_metadata["orchestration_mode"], "llm_react")
        self.assertNotIn("orchestration_fallback_reason", state.execution_metadata)
        self.assertTrue(state.execution_metadata["llm_budget_exhausted"])
        self.assertEqual(state.execution_metadata["planner_stop_proposed_by"], "python_runtime")
        self.assertEqual(state.stop_reason, "budget_exhausted")

    def test_required_unavailable_source_stops_after_authoritative_rca_gate(self) -> None:
        state = fake_llm_workflow(RecordingFakeClient()).run(
            ROOT_CAUSE_QUERY,
            job_id="JOB_LLM_REACT_REQUIRED_SOURCE_MISSING",
            lot_id="LOT_A_001",
            declared_unavailable_sources=("wafer_rework_and_split_genealogy",),
        )

        self.assertEqual(state.job.status, "completed")
        self.assertEqual(state.execution_metadata["orchestration_mode"], "llm_react")
        self.assertEqual(state.execution_metadata["planner_stop_proposed_by"], "python_runtime")
        self.assertEqual(
            state.execution_metadata["terminal_question_updates_source"],
            "python_evidence_gate",
        )
        self.assertEqual(state.stop_reason, "data_unavailable")
        self.assertIsNotNone(state.authoritative_rca_finding)
        self.assertEqual(
            state.authoritative_rca_finding.details["conclusion_status"],
            "insufficient_evidence",
        )
        self.assertEqual(
            sum(
                record.action.kind == "run_rca_reasoning"
                for record in state.action_history
            ),
            1,
        )

    def test_fake_qwen_intents_produce_different_bounded_action_chains(self) -> None:
        impact = run_lot(
            RecordingFakeClient(),
            IMPACT_QUERY,
            job_id="JOB_LLM_REACT_IMPACT",
        )
        root_cause = run_lot(
            RecordingFakeClient(),
            ROOT_CAUSE_QUERY,
            job_id="JOB_LLM_REACT_ROOT_CAUSE",
        )

        self.assertEqual(impact.investigation_goal.intent, "impact_scope")
        self.assertEqual(
            [record.action.kind for record in impact.action_history],
            ["find_shared_exposure"],
        )
        self.assertEqual(root_cause.investigation_goal.intent, "root_cause")
        self.assertEqual(
            [record.action.kind for record in root_cause.action_history],
            [
                "inspect_defect_pattern",
                "find_shared_exposure",
                "validate_shared_defect_pattern",
                "inspect_fdc_spc",
                "run_rca_reasoning",
            ],
        )
        self.assertNotEqual(
            [record.action.kind for record in impact.action_history],
            [record.action.kind for record in root_cause.action_history],
        )
        impact_evaluation = impact.run_evaluation
        assert impact_evaluation is not None
        self.assertTrue(impact_evaluation.goal_success)
        self.assertTrue(impact_evaluation.stop_correct)
        self.assertEqual(impact.conclusion_level, ConclusionLevel.SIGNAL.value)
        root_evaluation = root_cause.run_evaluation
        assert root_evaluation is not None
        self.assertTrue(root_evaluation.goal_success)
        self.assertTrue(root_evaluation.stop_correct)
        for state in (impact, root_cause):
            diagnostics = state.execution_metadata[
                "intent_planner_attempt_diagnostics"
            ]
            self.assertEqual(len(diagnostics), 1)
            self.assertEqual(diagnostics[0]["stage"], "intent_planning")
            self.assertEqual(diagnostics[0]["outcome"], "success")
            self.assertIsNone(diagnostics[0]["failure_category"])

    def test_lane_only_fdc_action_propagates_recipe_into_tool_parameters(self) -> None:
        workflow = fake_llm_workflow(RecordingFakeClient())
        state = workflow.run(
            ROOT_CAUSE_QUERY,
            job_id="JOB_LANE_RECIPE_PROPAGATION",
            lot_id="LOT_A_001",
        )
        mes_finding = next(
            finding
            for finding in reversed(state.findings)
            if finding.agent == AgentKind.MES.value
        )
        selected_lane = next(
            item
            for item in mes_finding.details["lane_candidates"]
            if item.get("recipe")
        )
        fdc_finding = next(
            finding
            for finding in reversed(state.findings)
            if finding.agent == AgentKind.FDC.value
        )
        recorder = ContextRecordingSpecialistExecutor(fdc_finding)
        supervisor = replace(
            workflow.supervisor,
            specialist_v2_executor=recorder,  # type: ignore[arg-type]
        )
        action = InvestigationAction(
            action_id="ACTION_LANE_RECIPE_PROPAGATION",
            kind="inspect_fdc_spc",
            agent=AgentKind.FDC.value,
            reason="Inspect the selected causal Lane.",
            scope={"lane_id": selected_lane["lane_id"]},
        )

        supervisor._dispatch_llm_react(  # noqa: SLF001
            action,
            state,
            remaining_tool_calls=2,
        )

        assert recorder.context is not None
        assert recorder.context["recipe_id"] == selected_lane["recipe"]
        actual_executor = workflow.supervisor.specialist_v2_executor
        assert actual_executor is not None
        tool_parameters = actual_executor._fdc_parameters(  # noqa: SLF001
            dict(recorder.context)
        )
        assert tool_parameters["recipe_id"] == selected_lane["recipe"]

    def test_same_root_cause_intent_replans_from_an_exposure_first_observation(
        self,
    ) -> None:
        default = run_lot(
            RecordingFakeClient(),
            ROOT_CAUSE_QUERY,
            job_id="JOB_LLM_REACT_DEFAULT_DIRECTION",
        )
        client = ExposureFirstRootCauseClient()
        exposure_first = run_lot(
            client,
            ROOT_CAUSE_QUERY,
            job_id="JOB_LLM_REACT_EXPOSURE_FIRST",
        )

        default_path = [record.action.kind for record in default.action_history]
        exposure_first_path = [
            record.action.kind for record in exposure_first.action_history
        ]
        self.assertEqual(default.investigation_goal.intent, "root_cause")
        self.assertEqual(exposure_first.investigation_goal.intent, "root_cause")
        self.assertEqual(default_path[0], "inspect_defect_pattern")
        self.assertEqual(exposure_first_path[0], "find_shared_exposure")
        self.assertEqual(exposure_first_path[1], "inspect_defect_pattern")
        self.assertNotEqual(default_path, exposure_first_path)
        self.assertEqual(len(exposure_first_path), len(set(exposure_first_path)))
        self.assertEqual(
            exposure_first.execution_metadata["orchestration_mode"],
            "llm_react",
        )
        planner_requests = [
            request
            for request in client.requests
            if request.prompt_name == "next_action_planner"
        ]
        self.assertEqual(
            planner_requests[1].payload["action_history"][0]["action"]["kind"],
            "find_shared_exposure",
        )
        self.assertTrue(
            any(
                finding["agent"] == "mes"
                for finding in planner_requests[1].payload["findings"]
            )
        )

        impact = run_lot(
            RecordingFakeClient(),
            IMPACT_QUERY,
            job_id="JOB_LLM_REACT_EARLY_STOP_BASELINE",
        )
        self.assertEqual(
            [record.action.kind for record in impact.action_history],
            ["find_shared_exposure"],
        )
        self.assertEqual(impact.stop_reason, StopReason.GOAL_SATISFIED.value)

    def test_scratch_cu_cmp_replans_after_observation_and_keeps_auditable_links(
        self,
    ) -> None:
        client = RecordingFakeClient()

        state = run_lot(
            client,
            ROOT_CAUSE_QUERY,
            job_id="JOB_LLM_REACT_REPLAN",
        )

        planner_requests = [
            request
            for request in client.requests
            if request.prompt_name == "next_action_planner"
        ]
        self.assertEqual(len(planner_requests), len(state.action_history) + 1)
        self.assertEqual(planner_requests[0].payload["findings"], [])
        self.assertEqual(planner_requests[0].payload["action_history"], [])
        self.assertEqual(len(planner_requests[1].payload["findings"]), 1)
        self.assertEqual(
            planner_requests[1].payload["findings"][0]["agent"],
            "defect_wat",
        )
        self.assertEqual(len(planner_requests[1].payload["action_history"]), 1)
        self.assertTrue(planner_requests[1].payload["available_evidence_ids"])
        self.assertTrue(
            all(
                "details" not in finding
                for request in planner_requests
                for finding in request.payload["findings"]
            )
        )
        self.assertTrue(
            all(
                "entities" not in evidence
                for request in planner_requests
                for evidence in request.payload["evidence"]
            )
        )
        self.assertTrue(
            all(
                request.payload["deterministic_planner_decision"][
                    "question_updates"
                ]
                == []
                for request in planner_requests
            )
        )

        act_decisions = [
            decision
            for decision in state.planner_decisions
            if decision.decision_type == DecisionType.ACT.value
        ]
        self.assertEqual(len(act_decisions), len(state.action_history))
        finding_ids = {finding.finding_id for finding in state.findings}
        evidence_ids = {evidence.evidence_id for evidence in state.evidence}
        for decision, record in zip(act_decisions, state.action_history, strict=True):
            self.assertIsNotNone(decision.next_action)
            self.assertEqual(
                decision.next_action.action_id,
                record.action.action_id,
            )
            self.assertEqual(
                decision.next_action.kind,
                record.action.kind,
            )
            self.assertTrue(set(record.produced_finding_ids) <= finding_ids)
            self.assertTrue(set(record.produced_evidence_ids) <= evidence_ids)

        self.assertEqual(
            state.planner_decisions[-1].decision_type,
            DecisionType.STOP.value,
        )
        run_evaluation = state.run_evaluation
        assert run_evaluation is not None
        evaluations = run_evaluation.decision_evaluations
        self.assertEqual(len(evaluations), len(state.planner_decisions))
        self.assertTrue(all(item.decision_valid for item in evaluations))
        self.assertTrue(all(not item.redundant for item in evaluations))
        self.assertTrue(all(item.evidence_gain for item in evaluations[:4]))
        self.assertFalse(evaluations[4].evidence_gain)
        self.assertFalse(evaluations[4].redundant)
        self.assertFalse(evaluations[-1].evidence_gain)
        self.assertTrue(run_evaluation.goal_success)
        self.assertTrue(run_evaluation.stop_correct)
        self.assertEqual(RCAState.from_dict(state.to_dict()), state)

    def test_unsupported_data_unavailable_stop_falls_back_at_open_question_boundary(
        self,
    ) -> None:
        state = run_lot(
            ImmediateUnsupportedStopClient(),
            ROOT_CAUSE_QUERY,
            job_id="JOB_LLM_REACT_GATED_STOP",
        )

        self.assertEqual(state.job.status, TaskStatus.COMPLETED.value)
        self.assertTrue(state.action_history)
        self.assertTrue(state.evidence)
        self.assertEqual(
            state.execution_metadata["orchestration_mode"],
            "controlled_react",
        )
        self.assertEqual(
            state.execution_metadata["orchestration_fallback_reason"],
            "qwen_next_action_output_invalid",
        )
        self.assertEqual(state.planner_decisions, [])
        self.assertEqual(state.question_update_reviews, [])
        self.assertIsNone(state.run_evaluation)
        self.assertTrue(state.report is None or state.report.markdown)

    def test_invalid_question_updates_are_rejected_without_losing_the_action(
        self,
    ) -> None:
        cases = {
            "non_terminal": (
                RejectedQuestionUpdateThenStopClient(),
                QuestionUpdateReasonCode.NON_TERMINAL_STATUS.value,
            ),
            "close_and_target": (
                RejectedQuestionUpdateThenStopClient(close_and_target=True),
                QuestionUpdateReasonCode.TARGET_OVERLAP.value,
            ),
        }
        for label, (client, expected_reason) in cases.items():
            with self.subTest(label=label):
                state = run_lot(
                    client,
                    IMPACT_QUERY,
                    job_id=f"JOB_REJECTED_UPDATE_{label.upper()}",
                )

                self.assertEqual(client.next_action_call_count, 2)
                self.assertEqual(
                    state.execution_metadata["orchestration_mode"],
                    "llm_react",
                )
                self.assertNotIn(
                    "orchestration_fallback_reason",
                    state.execution_metadata,
                )
                self.assertEqual(len(state.action_history), 1)
                self.assertEqual(len(state.planner_decisions), 2)
                self.assertEqual(
                    state.planner_decisions[0].question_updates,
                    [],
                )
                self.assertEqual(len(state.question_update_reviews), 1)
                review = state.question_update_reviews[0]
                self.assertEqual(review.reason_code, expected_reason)
                self.assertEqual(review.disposition, "rejected")
                self.assertEqual(
                    state.investigation_questions[0].status,
                    "open",
                )
                second_planner_request = [
                    request
                    for request in client.requests
                    if request.prompt_name == "next_action_planner"
                ][1]
                self.assertEqual(
                    second_planner_request.payload["questions"][0]["status"],
                    "open",
                )
                self.assertEqual(
                    RCAState.from_dict(state.to_dict()),
                    state,
                )

    def test_specialist_failure_commits_no_decision_update_or_review(self) -> None:
        client = RejectedQuestionUpdateThenStopClient()
        workflow = fake_llm_workflow(client)
        assert workflow.intent_planner is not None
        assert workflow.next_action_planner is not None
        intent_plan = workflow.intent_planner.plan(
            IMPACT_QUERY,
            lot_id="LOT_A_001",
        )
        supervisor = replace(
            workflow.supervisor,
            specialist_v2_executor=FailingSpecialistExecutor(),  # type: ignore[arg-type]
        )

        with self.assertRaises(SupervisorExecutionError) as raised:
            supervisor.execute_llm_react(
                RCAJob(
                    job_id="JOB_ATOMIC_REVIEW_FAILURE",
                    user_query=IMPACT_QUERY,
                    investigation_mode="lot",
                    source_lot_id="LOT_A_001",
                ),
                intent_plan,
                workflow.next_action_planner,
                tool_latencies=[],
            )

        failed_state = raised.exception.state
        assert failed_state is not None
        self.assertEqual(failed_state.action_history, [])
        self.assertEqual(failed_state.findings, [])
        self.assertEqual(failed_state.planner_decisions, [])
        self.assertEqual(failed_state.question_update_reviews, [])

    def test_two_invalid_next_actions_fallback_from_current_state(self) -> None:
        client = InvalidNextActionAfterFirstClient()

        state = run_lot(
            client,
            ROOT_CAUSE_QUERY,
            job_id="JOB_LLM_REACT_MID_LOOP_FALLBACK",
        )

        self.assertEqual(client.next_action_call_count, 3)
        self.assertEqual(
            state.execution_metadata["orchestration_requested_mode"],
            "llm_react",
        )
        self.assertEqual(
            state.execution_metadata["orchestration_mode"],
            "controlled_react",
        )
        self.assertEqual(
            state.execution_metadata["orchestration_fallback_reason"],
            "qwen_next_action_output_invalid",
        )
        self.assertEqual(
            state.execution_metadata["orchestration_fallback_stage"],
            "next_action_planning",
        )
        self.assertEqual(
            state.execution_metadata["orchestration_fallback_after_action_count"],
            1,
        )
        self.assertEqual(
            state.execution_metadata["orchestration_fallback_attempt_count"],
            2,
        )
        validation_errors = state.execution_metadata[
            "orchestration_fallback_validation_errors"
        ]
        self.assertEqual(len(validation_errors), 2)
        self.assertTrue(all("decision_id" in error for error in validation_errors))
        self.assertEqual(len(state.planner_decisions), 1)
        self.assertEqual(
            state.planner_decisions[0].next_action.action_id,
            state.action_history[0].action.action_id,
        )
        self.assertTrue(state.action_history[0].produced_evidence_ids)
        self.assertTrue(
            set(state.action_history[0].produced_evidence_ids)
            <= {item.evidence_id for item in state.evidence}
        )
        self.assertEqual(
            [record.action.kind for record in state.action_history],
            [
                "inspect_defect_pattern",
                "find_shared_exposure",
                "validate_shared_defect_pattern",
                "inspect_fdc_spc",
                "run_rca_reasoning",
            ],
        )
        self.assertTrue(state.investigation_questions)
        self.assertTrue(
            all(
                question.status == "closed"
                for question in state.investigation_questions
            )
        )
        self.assertEqual(state.evidence_gaps, [])
        self.assertIsNone(state.run_evaluation)

    def test_two_invalid_intent_outputs_fallback_before_first_action(self) -> None:
        client = InvalidIntentClient()

        state = run_lot(
            client,
            ROOT_CAUSE_QUERY,
            job_id="JOB_LLM_REACT_INTENT_FALLBACK",
        )

        intent_requests = [
            request
            for request in client.requests
            if request.prompt_name == "intent_planner"
        ]
        self.assertEqual(len(intent_requests), 2)
        self.assertEqual(
            state.execution_metadata["orchestration_requested_mode"],
            "llm_react",
        )
        self.assertEqual(
            state.execution_metadata["orchestration_mode"],
            "controlled_react",
        )
        self.assertEqual(
            state.execution_metadata["orchestration_fallback_reason"],
            "qwen_intent_output_invalid",
        )
        self.assertEqual(
            state.execution_metadata["orchestration_fallback_stage"],
            "intent_planning",
        )
        self.assertEqual(
            state.execution_metadata["orchestration_fallback_after_action_count"],
            0,
        )
        self.assertEqual(
            state.execution_metadata["orchestration_fallback_attempt_count"],
            2,
        )
        self.assertEqual(
            len(
                state.execution_metadata[
                    "orchestration_fallback_validation_errors"
                ]
            ),
            2,
        )
        diagnostics = state.execution_metadata[
            "intent_planner_attempt_diagnostics"
        ]
        self.assertEqual(len(diagnostics), 2)
        self.assertTrue(all(item["outcome"] == "failure" for item in diagnostics))
        self.assertTrue(
            all(
                item["failure_category"] == "contract_validation_error"
                for item in diagnostics
            )
        )
        self.assertTrue(all(item["reason_code"] == "malformed_output" for item in diagnostics))
        self.assertTrue(diagnostics[0]["repair_feedback_sent"])
        self.assertFalse(diagnostics[1]["repair_feedback_sent"])
        self.assertEqual(
            state.action_history[0].action.kind,
            "inspect_defect_pattern",
        )
        rca_finding = state.authoritative_rca_finding
        self.assertIsNotNone(rca_finding)
        assert rca_finding is not None
        impact_gate = rca_finding.details["impact_lot_gate"]
        self.assertTrue(impact_gate["observed_impact_lots"])
        self.assertEqual(impact_gate["confirmed_impact_lots"], [])
        self.assertEqual(state.impact_lots, [])
        self.assertIsNotNone(state.competition_trace)
        assert state.competition_trace is not None
        self.assertEqual(
            state.competition_trace.competition_status,
            "not_evaluated",
        )
        self.assertIsNone(state.competition_trace.terminal_reason)
        self.assertEqual(
            state.execution_metadata["investigation_finalizer"],
            "not_applicable",
        )
        self.assertEqual(
            state.execution_metadata["terminal_question_updates_source"],
            "python_evidence_gate",
        )
        self.assertEqual(
            state.execution_metadata["planner_stop_proposed_by"],
            "python_runtime",
        )
        self.assertEqual(
            state.execution_metadata["terminal_conclusion_status_source"],
            "authoritative_rca_finding",
        )
        self.assertEqual(state.planner_decisions, [])
        self.assertEqual(state.goal_status, "blocked")
        self.assertEqual(state.stop_reason, StopReason.NO_HIGH_VALUE_ACTION.value)
        self.assertEqual(state.conclusion_level, ConclusionLevel.INCONCLUSIVE.value)
        assert state.report is not None
        self.assertIn("- Impact Lot Count: 0", state.report.markdown)
        self.assertIn("- Impact Lots: None identified.", state.report.markdown)
        self.assertEqual(RCAState.from_dict(state.to_dict()), state)
        self.assertIsNone(state.run_evaluation)

    def test_invalid_full_rca_intent_preserves_python_baseline_goal(self) -> None:
        state = run_lot(
            InvalidIntentClient(),
            FULL_RCA_QUERY,
            job_id="JOB_LLM_REACT_FULL_RCA_INTENT_FALLBACK",
        )

        self.assertEqual(
            state.execution_metadata["orchestration_mode"],
            "controlled_react",
        )
        self.assertEqual(
            state.execution_metadata["orchestration_fallback_stage"],
            "intent_planning",
        )
        self.assertIsNotNone(state.investigation_goal)
        assert state.investigation_goal is not None
        self.assertEqual(
            state.investigation_goal.intent,
            InvestigationIntent.FULL_RCA.value,
        )
        self.assertEqual(
            state.investigation_goal.required_evidence,
            [
                "defect_signature",
                "shared_exposure",
                "impact_scope",
                "process_mechanism",
                "product_outcome",
            ],
        )

    def test_intent_transport_fallback_keeps_qwen_publication_governance(
        self,
    ) -> None:
        state = run_lot(
            IntentPlannerTransportFailureClient(),
            ROOT_CAUSE_QUERY,
            job_id="JOB_LLM_REACT_INTENT_TRANSPORT_FALLBACK",
        )

        self.assertEqual(
            state.execution_metadata["orchestration_requested_mode"],
            "llm_react",
        )
        self.assertEqual(
            state.execution_metadata["orchestration_mode"],
            "controlled_react",
        )
        self.assertEqual(
            state.execution_metadata["orchestration_fallback_reason"],
            "qwen_intent_call_failed",
        )
        self.assertEqual(
            state.execution_metadata["orchestration_fallback_stage"],
            "intent_planning",
        )
        self.assertEqual(
            state.execution_metadata["investigation_finalizer"],
            "not_applicable",
        )
        self.assertEqual(
            state.execution_metadata["terminal_conclusion_status_source"],
            "authoritative_rca_finding",
        )
        self.assertEqual(state.goal_status, "blocked")
        self.assertEqual(state.conclusion_level, ConclusionLevel.INCONCLUSIVE.value)
        self.assertEqual(state.stop_reason, StopReason.NO_HIGH_VALUE_ACTION.value)
        self.assertEqual(state.impact_lots, [])
        assert state.report is not None
        self.assertIn("- Impact Lot Count: 0", state.report.markdown)
        self.assertEqual(RCAState.from_dict(state.to_dict()), state)

    def test_product_root_cause_and_history_use_mes_selected_lots(self) -> None:
        workflow = fake_llm_workflow(RecordingFakeClient())

        root_cause = workflow.run(
            PRODUCT_ROOT_QUERY,
            job_id="JOB_LLM_REACT_PRODUCT_ROOT",
        )
        history = workflow.run(
            PRODUCT_HISTORY_QUERY,
            job_id="JOB_LLM_REACT_PRODUCT_HISTORY",
        )

        self.assertEqual(root_cause.job.investigation_mode, "product_window")
        self.assertEqual(root_cause.job.source_lot_id, None)
        self.assertEqual(root_cause.action_history[0].action.kind, "find_shared_exposure")
        self.assertIn(
            "inspect_defect_pattern",
            [record.action.kind for record in root_cause.action_history],
        )
        defect = next(
            finding
            for finding in root_cause.findings
            if finding.agent == "defect_wat"
        )
        self.assertTrue(defect.details["lot_ids"])
        self.assertEqual(root_cause.job.status, TaskStatus.COMPLETED.value)

        self.assertEqual(history.job.investigation_mode, "product_window")
        self.assertIn(
            "validate_historical_case",
            [record.action.kind for record in history.action_history],
        )
        self.assertEqual(history.job.status, TaskStatus.COMPLETED.value)


class LLMReactAPIIntegrationTest(unittest.TestCase):
    def test_api_accepts_non_scratch_lot_and_product_window_in_llm_mode(self) -> None:
        app = create_app(
            workflow=fake_llm_workflow(RecordingFakeClient()),
            runtime_dataset="golden_case",
            execute_jobs_inline=True,
        )

        with TestClient(app) as client:
            ready = client.get("/ready")
            lot_created = client.post(
                "/rca/jobs",
                json={
                    "investigation_mode": "lot",
                    "lot_id": "LOT_A_001",
                    "user_query": IMPACT_QUERY,
                },
            )
            product_created = client.post(
                "/rca/jobs",
                json={
                    "investigation_mode": "product_window",
                    "user_query": PRODUCT_IMPACT_QUERY,
                },
            )
            product_root_created = client.post(
                "/rca/jobs",
                json={
                    "investigation_mode": "product_window",
                    "user_query": PRODUCT_ROOT_QUERY,
                },
            )

            self.assertEqual(ready.status_code, 200)
            self.assertEqual(ready.json()["orchestration_mode"], "llm_react")
            self.assertEqual(lot_created.status_code, 201)
            self.assertEqual(product_created.status_code, 201)
            self.assertEqual(product_root_created.status_code, 201)

            for created in (
                lot_created.json(),
                product_created.json(),
                product_root_created.json(),
            ):
                state_response = client.get(created["state_url"])
                self.assertEqual(state_response.status_code, 200)
                metadata = state_response.json()["state"]["execution_metadata"]
                self.assertEqual(
                    metadata["orchestration_requested_mode"],
                    "llm_react",
                )
                self.assertEqual(metadata["orchestration_mode"], "llm_react")
                self.assertNotIn("orchestration_fallback_reason", metadata)
                evaluation = state_response.json()["state"]["run_evaluation"]
                self.assertIsNotNone(evaluation)
                self.assertTrue(evaluation["goal_success"])
                self.assertTrue(evaluation["stop_correct"])

    def test_api_exposes_mid_loop_fallback_metadata(self) -> None:
        app = create_app(
            workflow=fake_llm_workflow(InvalidNextActionAfterFirstClient()),
            runtime_dataset="golden_case",
            execute_jobs_inline=True,
        )

        with TestClient(app) as client:
            created = client.post(
                "/rca/jobs",
                json={
                    "investigation_mode": "lot",
                    "lot_id": "LOT_A_001",
                    "user_query": ROOT_CAUSE_QUERY,
                },
            )
            self.assertEqual(created.status_code, 201)
            state = client.get(created.json()["state_url"]).json()["state"]

        metadata = state["execution_metadata"]
        self.assertEqual(metadata["orchestration_requested_mode"], "llm_react")
        self.assertEqual(metadata["orchestration_mode"], "controlled_react")
        self.assertEqual(
            metadata["orchestration_fallback_reason"],
            "qwen_next_action_output_invalid",
        )
        self.assertEqual(
            metadata["orchestration_fallback_after_action_count"],
            1,
        )
        self.assertEqual(metadata["orchestration_fallback_attempt_count"], 2)
        self.assertEqual(
            len(metadata["orchestration_fallback_validation_errors"]),
            2,
        )
        self.assertTrue(
            all(
                "decision_id" in error
                for error in metadata["orchestration_fallback_validation_errors"]
            )
        )
        self.assertEqual(len(state["planner_decisions"]), 1)
        self.assertTrue(state["evidence"])
        self.assertIsNone(state["run_evaluation"])

    def test_api_exposes_typed_intent_planner_handoff_trace(self) -> None:
        app = create_app(
            workflow=fake_llm_workflow(InvalidIntentClient()),
            runtime_dataset="golden_case",
            execute_jobs_inline=True,
        )

        with TestClient(app) as client:
            created = client.post(
                "/rca/jobs",
                json={
                    "investigation_mode": "lot",
                    "lot_id": "LOT_A_001",
                    "user_query": ROOT_CAUSE_QUERY,
                },
            )
            self.assertEqual(created.status_code, 201)
            state = client.get(created.json()["state_url"]).json()["state"]

        metadata = state["execution_metadata"]
        self.assertEqual(metadata["orchestration_fallback_stage"], "intent_planning")
        self.assertEqual(metadata["orchestration_fallback_attempt_count"], 2)
        diagnostics = metadata["intent_planner_attempt_diagnostics"]
        self.assertEqual(len(diagnostics), 2)
        self.assertEqual(diagnostics[0]["stage"], "intent_planning")
        self.assertEqual(diagnostics[0]["outcome"], "failure")
        self.assertEqual(
            diagnostics[0]["failure_category"],
            "contract_validation_error",
        )
        self.assertEqual(diagnostics[0]["reason_code"], "malformed_output")
        self.assertIn("candidate_summary", diagnostics[0])
        self.assertIn("baseline_diff", diagnostics[0])
        self.assertNotIn("user_query", str(diagnostics))
        self.assertEqual(state["conclusion_level"], "inconclusive")
        self.assertEqual(state["impact_lots"], [])
        self.assertEqual(state["planner_decisions"], [])

    def test_api_exposes_rejected_question_update_without_fallback(self) -> None:
        app = create_app(
            workflow=fake_llm_workflow(RejectedQuestionUpdateThenStopClient()),
            runtime_dataset="golden_case",
            execute_jobs_inline=True,
        )

        with TestClient(app) as client:
            created = client.post(
                "/rca/jobs",
                json={
                    "investigation_mode": "lot",
                    "lot_id": "LOT_A_001",
                    "user_query": IMPACT_QUERY,
                },
            )
            self.assertEqual(created.status_code, 201)
            state = client.get(created.json()["state_url"]).json()["state"]

        self.assertEqual(
            state["execution_metadata"]["orchestration_mode"],
            "llm_react",
        )
        self.assertNotIn(
            "orchestration_fallback_reason",
            state["execution_metadata"],
        )
        self.assertEqual(len(state["action_history"]), 1)
        self.assertEqual(state["planner_decisions"][0]["question_updates"], [])
        self.assertEqual(
            state["question_update_reviews"][0]["reason_code"],
            QuestionUpdateReasonCode.NON_TERMINAL_STATUS.value,
        )
        self.assertEqual(state["investigation_questions"][0]["status"], "open")

    def test_api_invalid_immediate_stop_falls_back_and_never_returns_500(self) -> None:
        app = create_app(
            workflow=fake_llm_workflow(ImmediateUnsupportedStopClient()),
            runtime_dataset="golden_case",
            execute_jobs_inline=True,
        )

        with TestClient(app) as client:
            created = client.post(
                "/rca/jobs",
                json={
                    "investigation_mode": "lot",
                    "lot_id": "LOT_A_001",
                    "user_query": ROOT_CAUSE_QUERY,
                },
            )
            self.assertEqual(created.status_code, 201)
            state_response = client.get(created.json()["state_url"])
            report_response = client.get(created.json()["report_url"])

        self.assertEqual(state_response.status_code, 200)
        state = state_response.json()["state"]
        self.assertEqual(state["job"]["status"], TaskStatus.COMPLETED.value)
        self.assertEqual(
            state["execution_metadata"]["orchestration_mode"],
            "controlled_react",
        )
        self.assertEqual(state["planner_decisions"], [])
        self.assertEqual(state["question_update_reviews"], [])
        self.assertIsNone(state["run_evaluation"])
        self.assertIn(report_response.status_code, {200, 409})


if __name__ == "__main__":
    unittest.main()
