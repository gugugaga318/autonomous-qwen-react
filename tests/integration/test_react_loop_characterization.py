"""Characterization contracts for the pre-Batch-5 React orchestration loops.

These tests pin the CURRENT (pre-refactor) behavior of the native controlled
loop, the next-action fallback, the intent fallback, and the LLM STOP branch:
action sequence, terminal fields, authority objects, execution metadata, event
names and order, and budget counters. They must stay green before, during, and
after the shared ``_react_loop`` extraction; they are semantic contracts, not
byte-for-byte State snapshots.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "backend"))

from yield_rca_core.investigation_models import (  # noqa: E402
    ConclusionLevel,
    DecisionType,
    StopReason,
)
from yield_rca_core.llm_gateway import (  # noqa: E402
    FakeLLMClient,
    LLMCallError,
    LLMResponse,
    LLMSettings,
)
from yield_rca_core.models import TaskStatus  # noqa: E402
from yield_rca_core.workflow import build_csv_workflow  # noqa: E402
from yield_rca_core.workflow_events import (  # noqa: E402
    capture_workflow_events,
)

SEED_DIR = ROOT / "data" / "seeds" / "golden_case"
ROOT_CAUSE_QUERY = "Investigate the root cause of LOT_A_001 scratch in Cu CMP."


class RecordingFakeClient(FakeLLMClient):
    def __init__(self) -> None:
        self.requests: list[Any] = []

    def complete_json(self, request: Any) -> LLMResponse:
        self.requests.append(request)
        return super().complete_json(request)


class PersistentNextActionCallFailureClient(RecordingFakeClient):
    def complete_json(self, request: Any) -> LLMResponse:
        if request.prompt_name == "next_action_planner":
            self.requests.append(request)
            raise LLMCallError(
                "persistent throttling",
                status_code=429,
                provider_code="Throttling",
                request_id="req-characterization-429",
                failure_category="provider_http_error",
            )
        return super().complete_json(request)


class InvalidIntentClient(RecordingFakeClient):
    def complete_json(self, request: Any) -> LLMResponse:
        response = super().complete_json(request)
        if request.prompt_name == "intent_planner":
            return LLMResponse(data={}, usage=response.usage)
        return response


class ImmediateUnsupportedStopClient(RecordingFakeClient):
    def complete_json(self, request: Any) -> LLMResponse:
        response = super().complete_json(request)
        if request.prompt_name != "next_action_planner":
            return response
        goal_id = str(request.payload["goal"]["goal_id"])
        question_updates = [
            {
                "question_id": str(question["question_id"]),
                "status": "unavailable",
                "answer": None,
                "evidence_ids": [],
                "unavailable_reason": "The requested source data is unavailable.",
            }
            for question in request.payload["questions"]
        ]
        return LLMResponse(
            data={
                "decision_id": f"{goal_id}:characterization-stop",
                "goal_id": goal_id,
                "decision_type": DecisionType.STOP.value,
                "reason": "The model cannot obtain the requested source data.",
                "goal_status": "blocked",
                "proposed_conclusion_level": ConclusionLevel.SUPPORTED.value,
                "next_action": None,
                "target_question_ids": [],
                "new_questions": [],
                "stop_reason": StopReason.DATA_UNAVAILABLE.value,
                "question_updates": question_updates,
            },
            usage=response.usage,
        )


class EventRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append((event_type, dict(payload)))

    def types(self) -> list[str]:
        return [event_type for event_type, _ in self.events]


def _run(
    client: FakeLLMClient | None,
    *,
    orchestration_mode: str,
    job_id: str,
) -> tuple[Any, EventRecorder]:
    workflow = build_csv_workflow(
        SEED_DIR,
        llm_settings=LLMSettings(agent_mode="fake"),
        llm_client=client,
        orchestration_mode=orchestration_mode,
    )
    recorder = EventRecorder()
    with capture_workflow_events(recorder):
        state = workflow.run(
            ROOT_CAUSE_QUERY,
            job_id=job_id,
            lot_id="LOT_A_001",
        )
    return state, recorder


class NativeControlledCharacterizationTest(unittest.TestCase):
    def test_native_controlled_terminal_contract(self) -> None:
        state, recorder = _run(
            None,
            orchestration_mode="controlled_react",
            job_id="JOB_CHAR_CONTROLLED_NATIVE",
        )

        self.assertEqual(state.job.status, TaskStatus.COMPLETED.value)
        metadata = state.execution_metadata
        self.assertEqual(metadata["orchestration_mode"], "controlled_react")
        self.assertNotIn("orchestration_requested_mode", metadata)
        self.assertNotIn("orchestration_fallback_reason", metadata)
        self.assertNotIn("planner_stop_proposed_by", metadata)
        # Native controlled terminals are not governed: no Finalizer authority.
        self.assertIsNone(state.authoritative_rca_result)
        self.assertIsNone(state.impact_publication_result)
        # Decision recording stays LLM-only.
        self.assertEqual(state.planner_decisions, [])
        # Terminal event contract.
        self.assertIn("investigation_stopped", recorder.types())
        self.assertNotIn("planner_stopped", recorder.types())
        self.assertEqual(recorder.types()[-1], "investigation_stopped")
        stopped = recorder.events[-1][1]
        self.assertEqual(stopped["mode"], "controlled_react")
        self.assertEqual(stopped["goal_status"], state.goal_status)
        self.assertEqual(stopped["conclusion_level"], state.conclusion_level)
        self.assertEqual(stopped["stop_reason"], state.stop_reason)
        # Action/event ordering contract.
        action_kinds = [record.action.kind for record in state.action_history]
        self.assertEqual(
            action_kinds,
            [kind for kind in action_kinds if kind],
        )
        started = [
            payload["action_kind"]
            for event_type, payload in recorder.events
            if event_type == "action_started"
        ]
        self.assertEqual(started, action_kinds)
        # Terminal state is one of the deterministic policy outcomes.
        self.assertIn(
            state.goal_status,
            {"satisfied", "blocked", "budget_exhausted"},
        )


class NextActionFallbackCharacterizationTest(unittest.TestCase):
    def test_next_action_fallback_terminal_contract(self) -> None:
        state, recorder = _run(
            PersistentNextActionCallFailureClient(),
            orchestration_mode="llm_react",
            job_id="JOB_CHAR_NEXT_ACTION_FALLBACK",
        )

        metadata = state.execution_metadata
        self.assertEqual(metadata["orchestration_requested_mode"], "llm_react")
        self.assertEqual(metadata["orchestration_mode"], "controlled_react")
        self.assertEqual(
            metadata["orchestration_fallback_reason"],
            "qwen_next_action_call_failed",
        )
        self.assertEqual(
            metadata["orchestration_fallback_failure_category"],
            "provider_http_error",
        )
        self.assertEqual(metadata["orchestration_fallback_status_code"], 429)
        self.assertEqual(
            metadata["orchestration_fallback_request_id"],
            "req-characterization-429",
        )
        # No Qwen decision was ever accepted: fallback happened at the first
        # decision, then the controlled profile ran the deterministic sequence.
        self.assertEqual(state.planner_decisions, [])
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
        # Governed fallback terminal carries Batch 4 authority objects.
        self.assertIsNotNone(state.authoritative_rca_result)
        self.assertIsNotNone(state.impact_publication_result)
        result = state.authoritative_rca_result
        assert result is not None and state.impact_publication_result is not None
        self.assertEqual(result.source_finding_id, state.authoritative_rca_finding_id)
        self.assertEqual(
            state.impact_publication_result.rca_result_id,
            result.result_id,
        )
        self.assertIn(
            state.impact_publication_result.publication_status,
            {"withheld", "not_evaluated", "unconfirmed"},
        )
        self.assertNotEqual(result.conclusion_status, "supported")
        # Batch 3 projection provenance: the Finalizer owns the terminal, so
        # the projection records the runtime as the stop proposer.
        self.assertEqual(metadata["planner_stop_proposed_by"], "python_runtime")
        self.assertEqual(
            metadata["terminal_stop_projected_by"],
            "python_investigation_finalizer",
        )
        self.assertEqual(state.goal_status, "blocked")
        self.assertEqual(state.conclusion_level, ConclusionLevel.INCONCLUSIVE.value)
        self.assertEqual(state.stop_reason, StopReason.NO_HIGH_VALUE_ACTION.value)
        # Terminal event contract: controlled profile owns the stop event.
        self.assertEqual(recorder.types()[-1], "investigation_stopped")
        self.assertNotIn("planner_stopped", recorder.types())
        self.assertEqual(state.job.status, TaskStatus.COMPLETED.value)

    def test_budget_continuity_across_planner_and_profile_swap(self) -> None:
        client = PersistentNextActionCallFailureClient()
        state, _ = _run(
            client,
            orchestration_mode="llm_react",
            job_id="JOB_CHAR_BUDGET_CONTINUITY",
        )

        metadata = state.execution_metadata
        # The swap must not re-arm the Qwen budget preflight: no llm budget
        # termination markers may appear on the controlled continuation.
        self.assertNotIn("llm_budget_exhausted", metadata)
        self.assertNotIn("llm_budget_failure_category", metadata)
        # Actions accumulated after the swap stay inside the same goal
        # budget (fallback_after_action_count was 0 at swap time).
        self.assertEqual(
            metadata["orchestration_fallback_after_action_count"],
            0,
        )
        self.assertGreaterEqual(len(state.action_history), 1)
        # One continuous State-level usage ledger spans the whole run: the
        # swap reset neither the LLM usage ledger nor the tool budget.
        self.assertEqual(metadata["llm_call_count"], len(state.llm_usage))
        self.assertGreater(metadata["llm_call_count"], 0)

class IntentFallbackCharacterizationTest(unittest.TestCase):
    def test_intent_fallback_starts_controlled_from_first_iteration(self) -> None:
        client = InvalidIntentClient()
        state, recorder = _run(
            client,
            orchestration_mode="llm_react",
            job_id="JOB_CHAR_INTENT_FALLBACK",
        )

        metadata = state.execution_metadata
        self.assertEqual(metadata["orchestration_requested_mode"], "llm_react")
        self.assertEqual(metadata["orchestration_mode"], "controlled_react")
        self.assertEqual(
            metadata["orchestration_fallback_reason"],
            "qwen_intent_output_invalid",
        )
        self.assertEqual(
            metadata["orchestration_fallback_stage"],
            "intent_planning",
        )
        self.assertEqual(metadata["orchestration_fallback_after_action_count"], 0)
        # No Qwen next-action planning ever happened.
        self.assertEqual(
            [
                request
                for request in client.requests
                if request.prompt_name == "next_action_planner"
            ],
            [],
        )
        # The controlled policy ran the deterministic sequence.
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
        # Governed terminal (requested llm_react) with Batch 4 authority objects.
        self.assertIsNotNone(state.authoritative_rca_result)
        self.assertIsNotNone(state.impact_publication_result)
        self.assertEqual(recorder.types()[-1], "investigation_stopped")
        self.assertNotIn("planner_stopped", recorder.types())
        self.assertEqual(state.planner_decisions, [])
        self.assertEqual(metadata["planner_stop_proposed_by"], "python_runtime")


class LlmStopCharacterizationTest(unittest.TestCase):
    def test_llm_stop_branch_contract(self) -> None:
        state, recorder = _run(
            ImmediateUnsupportedStopClient(),
            orchestration_mode="llm_react",
            job_id="JOB_CHAR_LLM_STOP",
        )

        metadata = state.execution_metadata
        self.assertEqual(metadata["orchestration_mode"], "llm_react")
        self.assertNotIn("orchestration_fallback_reason", metadata)
        self.assertEqual(metadata["planner_stop_proposed_by"], "qwen")
        # The review rejects unsupported question-unavailable claims, so the
        # accepted decision carries no question updates and the terminal
        # metadata keeps the validation source empty.
        self.assertIsNone(metadata.get("terminal_question_updates_validated_by"))
        # The planner decision is recorded, and the conclusion is capped.
        self.assertEqual(len(state.planner_decisions), 1)
        self.assertEqual(
            state.planner_decisions[0].decision_type,
            DecisionType.STOP.value,
        )
        self.assertEqual(
            state.planner_decisions[0].proposed_conclusion_level,
            ConclusionLevel.SUPPORTED.value,
        )
        self.assertEqual(
            state.conclusion_level,
            ConclusionLevel.INCONCLUSIVE.value,
        )
        self.assertEqual(state.goal_status, "blocked")
        self.assertEqual(state.stop_reason, StopReason.DATA_UNAVAILABLE.value)
        self.assertEqual(recorder.types()[-1], "planner_stopped")
        self.assertNotIn("investigation_stopped", recorder.types())
        self.assertEqual(state.job.status, TaskStatus.COMPLETED.value)
        # No RCA reasoning ran, so finalize was not applicable and no
        # authority objects may appear.
        self.assertIsNone(state.authoritative_rca_result)
        self.assertIsNone(state.impact_publication_result)


if __name__ == "__main__":
    unittest.main()
