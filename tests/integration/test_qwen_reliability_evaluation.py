from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT))

from yield_rca_core.llm_gateway import (  # noqa: E402
    FakeLLMClient,
    LLMCallError,
    LLMOutputValidationError,
    LLMRequest,
    LLMResponse,
    LLMSettings,
)
from yield_rca_core.qwen_reliability import (  # noqa: E402
    _is_question_update_validation_error,
    qwen_reliability_failure,
    render_qwen_reliability_report,
)

from scripts.run_qwen_reliability_evaluation import (  # noqa: E402
    run_qwen_reliability,
)

SEED_DIR = ROOT / "data" / "seeds" / "golden_case"


class InvalidQuestionUpdateAfterObservationClient(FakeLLMClient):
    """Emit one invalid ancillary update after an observation, then recover."""

    def __init__(self) -> None:
        self.next_action_calls = 0

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        response = super().complete_json(request)
        if request.prompt_name != "next_action_planner":
            return response
        self.next_action_calls += 1
        if self.next_action_calls != 2:
            return response
        question_id = str(request.payload["questions"][0]["question_id"])
        payload = dict(response.data)
        payload["question_updates"] = [
            {
                "question_id": question_id,
                "status": "open",
                "answer": "The scratch observation provides partial progress.",
                "evidence_ids": list(request.payload["available_evidence_ids"]),
                "unavailable_reason": None,
            }
        ]
        return LLMResponse(data=payload, usage=response.usage)


class CloseAndTargetQuestionClient(FakeLLMClient):
    """Emit one terminal update that conflicts with the next target."""

    def __init__(self) -> None:
        self.next_action_calls = 0

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        response = super().complete_json(request)
        if request.prompt_name != "next_action_planner":
            return response
        self.next_action_calls += 1
        if self.next_action_calls != 2:
            return response
        payload = dict(response.data)
        question_id = str(payload["target_question_ids"][0])
        payload["question_updates"] = [
            {
                "question_id": question_id,
                "status": "unavailable",
                "answer": None,
                "evidence_ids": [],
                "unavailable_reason": "No registered source can answer it.",
            }
        ]
        return LLMResponse(data=payload, usage=response.usage)


class InvalidCoreDecisionAfterObservationClient(FakeLLMClient):
    """Keep ancillary updates out of scope and invalidate the core decision."""

    def __init__(self) -> None:
        self.next_action_calls = 0

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        response = super().complete_json(request)
        if request.prompt_name != "next_action_planner":
            return response
        self.next_action_calls += 1
        if self.next_action_calls == 1:
            return response
        return LLMResponse(data={}, usage=response.usage)


class TransientPlannerCallFailureClient(FakeLLMClient):
    def __init__(self) -> None:
        self.failure_injected = False

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        if request.prompt_name == "next_action_planner" and not self.failure_injected:
            self.failure_injected = True
            raise LLMCallError(
                "temporary throttling",
                status_code=429,
                provider_code="Throttling",
                failure_category="provider_http_error",
            )
        return super().complete_json(request)


class PersistentPlannerCallFailureClient(FakeLLMClient):
    def complete_json(self, request: LLMRequest) -> LLMResponse:
        if request.prompt_name == "next_action_planner":
            raise LLMCallError(
                "persistent throttling",
                status_code=429,
                provider_code="Throttling",
                provider_message="retry later",
                request_id="req-reliability-429",
                failure_category="provider_http_error",
            )
        return super().complete_json(request)


class InvalidJsonPlannerOutputClient(FakeLLMClient):
    def complete_json(self, request: LLMRequest) -> LLMResponse:
        if request.prompt_name == "next_action_planner":
            raise LLMOutputValidationError("model response is not valid JSON")
        return super().complete_json(request)


class QwenReliabilityEvaluationTest(unittest.TestCase):
    def test_three_fake_qwen_runs_pass_the_same_reliability_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            evaluation = run_qwen_reliability(
                settings=LLMSettings(agent_mode="fake"),
                seed_dir=SEED_DIR,
                output_dir=output_dir,
                runs=3,
                max_llm_calls_per_run=20,
            )

            self.assertTrue(evaluation["passed"])
            self.assertEqual(evaluation["required_consecutive_runs"], 3)
            self.assertEqual(evaluation["passed_run_count"], 3)
            self.assertEqual(evaluation["controlled_fallback_count"], 0)
            self.assertEqual(
                evaluation["question_update_validation_error_count"],
                0,
            )
            self.assertGreater(evaluation["question_update_review_count"], 0)
            self.assertEqual(
                evaluation["accepted_question_update_count"],
                sum(run["question_update_count"] for run in evaluation["runs"]),
            )
            self.assertEqual(evaluation["rejected_question_update_count"], 0)
            for run in evaluation["runs"]:
                self.assertEqual(run["actual_mode"], "llm_react")
                self.assertEqual(run["action_chain"][0], "inspect_defect_pattern")
                self.assertGreaterEqual(len(run["action_chain"]), 2)
                self.assertGreater(run["question_update_count"], 0)
                self.assertGreater(run["question_update_review_count"], 0)
                self.assertEqual(
                    run["accepted_question_update_count"],
                    run["question_update_count"],
                )
                self.assertTrue(all(run["checks"].values()))
                self.assertLessEqual(run["paid_llm_call_count"], 20)
            self.assertEqual(
                json.loads(
                    (output_dir / "results.json").read_text(encoding="utf-8")
                ),
                evaluation,
            )
            report = (output_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("Acceptance: **PASS**", report)
            self.assertIn("compact terminal QuestionUpdate deltas", report)

    def test_non_terminal_update_is_rejected_without_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            evaluation = run_qwen_reliability(
                settings=LLMSettings(agent_mode="fake"),
                seed_dir=SEED_DIR,
                output_dir=Path(temporary_directory),
                runs=1,
                max_llm_calls_per_run=20,
                client_factory=lambda _settings: (
                    InvalidQuestionUpdateAfterObservationClient()
                ),
            )

        self.assertTrue(evaluation["passed"])
        self.assertEqual(evaluation["controlled_fallback_count"], 0)
        self.assertEqual(evaluation["question_update_validation_error_count"], 0)
        self.assertEqual(evaluation["rejected_question_update_count"], 1)
        self.assertEqual(
            evaluation["rejection_reason_counts"],
            {"non_terminal_status": 1},
        )
        run = evaluation["runs"][0]
        self.assertEqual(run["actual_mode"], "llm_react")
        self.assertIsNone(run["fallback_reason"])
        self.assertEqual(run["rejected_question_update_count"], 1)
        rejected = next(
            review
            for review in run["question_update_reviews"]
            if review["disposition"] == "rejected"
        )
        self.assertEqual(rejected["reason_code"], "non_terminal_status")
        self.assertEqual(rejected["claimed_status"], "open")
        self.assertEqual(
            set(rejected),
            {
                "decision_id",
                "disposition",
                "reason_code",
                "update_index",
                "question_id",
                "claimed_status",
            },
        )
        self.assertTrue(run["checks"]["rejected_updates_preserved_core_action"])
        report = render_qwen_reliability_report(evaluation)
        self.assertIn("Acceptance: **PASS**", report)
        self.assertIn("Rejected QuestionUpdate audit", report)
        self.assertIn("`non_terminal_status`=1", report)

    def test_close_and_target_update_is_rejected_without_fallback(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            evaluation = run_qwen_reliability(
                settings=LLMSettings(agent_mode="fake"),
                seed_dir=SEED_DIR,
                output_dir=Path(temporary_directory),
                runs=1,
                max_llm_calls_per_run=20,
                client_factory=lambda _settings: CloseAndTargetQuestionClient(),
            )

        self.assertTrue(evaluation["passed"])
        self.assertEqual(evaluation["controlled_fallback_count"], 0)
        self.assertEqual(evaluation["rejected_question_update_count"], 1)
        self.assertEqual(
            evaluation["rejection_reason_counts"],
            {"target_overlap": 1},
        )
        run = evaluation["runs"][0]
        self.assertEqual(run["actual_mode"], "llm_react")
        rejected = next(
            review
            for review in run["question_update_reviews"]
            if review["disposition"] == "rejected"
        )
        self.assertEqual(rejected["reason_code"], "target_overlap")
        self.assertEqual(rejected["claimed_status"], "unavailable")
        self.assertTrue(run["checks"]["rejected_updates_preserved_core_action"])

    def test_invalid_core_decision_still_fails_via_controlled_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            evaluation = run_qwen_reliability(
                settings=LLMSettings(agent_mode="fake"),
                seed_dir=SEED_DIR,
                output_dir=Path(temporary_directory),
                runs=1,
                max_llm_calls_per_run=20,
                client_factory=lambda _settings: (
                    InvalidCoreDecisionAfterObservationClient()
                ),
            )

        self.assertFalse(evaluation["passed"])
        self.assertEqual(evaluation["controlled_fallback_count"], 1)
        self.assertEqual(evaluation["rejected_question_update_count"], 0)
        self.assertEqual(evaluation["core_planner_validation_error_count"], 2)
        run = evaluation["runs"][0]
        self.assertEqual(run["actual_mode"], "controlled_react")
        self.assertEqual(run["fallback_attempt_count"], 2)
        self.assertEqual(len(run["fallback_validation_errors"]), 2)
        self.assertTrue(
            all("decision_id" in error for error in run["fallback_validation_errors"])
        )
        report = render_qwen_reliability_report(evaluation)
        self.assertIn("Acceptance: **FAIL**", report)
        self.assertIn("qwen_next_action_output_invalid", report)

    def test_one_transient_call_failure_is_counted_and_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            evaluation = run_qwen_reliability(
                settings=LLMSettings(agent_mode="fake"),
                seed_dir=SEED_DIR,
                output_dir=Path(temporary_directory),
                runs=1,
                max_llm_calls_per_run=20,
                client_factory=lambda _settings: TransientPlannerCallFailureClient(),
            )

        self.assertTrue(evaluation["passed"])
        self.assertEqual(evaluation["controlled_fallback_count"], 0)
        self.assertEqual(evaluation["planner_call_failure_count"], 1)
        self.assertEqual(evaluation["recovered_planner_call_retry_count"], 1)
        run = evaluation["runs"][0]
        self.assertEqual(run["actual_mode"], "llm_react")
        self.assertEqual(run["planner_call_failure_count"], 1)
        self.assertEqual(run["recovered_planner_call_retry_count"], 1)
        self.assertLessEqual(run["paid_llm_call_count"], 20)

    def test_two_call_failures_fallback_with_provider_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            evaluation = run_qwen_reliability(
                settings=LLMSettings(agent_mode="fake"),
                seed_dir=SEED_DIR,
                output_dir=Path(temporary_directory),
                runs=1,
                max_llm_calls_per_run=20,
                client_factory=lambda _settings: PersistentPlannerCallFailureClient(),
            )

        self.assertFalse(evaluation["passed"])
        self.assertEqual(evaluation["transport_provider_failure_count"], 1)
        self.assertEqual(evaluation["planner_call_failure_count"], 2)
        self.assertEqual(evaluation["recovered_planner_call_retry_count"], 0)
        run = evaluation["runs"][0]
        self.assertEqual(run["fallback_failure_category"], "provider_http_error")
        self.assertEqual(run["fallback_call_attempt_count"], 2)
        self.assertEqual(run["fallback_status_code"], 429)
        self.assertEqual(run["fallback_provider_code"], "Throttling")
        self.assertEqual(run["fallback_request_id"], "req-reliability-429")
        report = render_qwen_reliability_report(evaluation)
        self.assertIn("provider_http_error", report)
        self.assertIn("Throttling", report)

    def test_output_parse_failure_is_separate_from_core_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            evaluation = run_qwen_reliability(
                settings=LLMSettings(agent_mode="fake"),
                seed_dir=SEED_DIR,
                output_dir=Path(temporary_directory),
                runs=1,
                max_llm_calls_per_run=20,
                client_factory=lambda _settings: InvalidJsonPlannerOutputClient(),
            )

        self.assertFalse(evaluation["passed"])
        self.assertEqual(evaluation["output_parse_error_count"], 2)
        self.assertEqual(evaluation["core_planner_validation_error_count"], 0)
        self.assertEqual(evaluation["transport_provider_failure_count"], 0)
        run = evaluation["runs"][0]
        self.assertEqual(len(run["output_parse_validation_errors"]), 2)
        self.assertEqual(run["core_planner_validation_errors"], [])
        self.assertEqual(run["fallback_failure_category"], "planner_output_invalid")

    def test_paid_call_cap_stops_without_an_over_cap_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            evaluation = run_qwen_reliability(
                settings=LLMSettings(agent_mode="fake"),
                seed_dir=SEED_DIR,
                output_dir=Path(temporary_directory),
                runs=1,
                max_llm_calls_per_run=1,
            )

        self.assertFalse(evaluation["passed"])
        run = evaluation["runs"][0]
        self.assertEqual(run["paid_llm_call_count"], 1)
        self.assertFalse(run["call_limit_exceeded"])
        self.assertTrue(run["checks"]["workflow_completed"])
        self.assertEqual(run["actual_mode"], "llm_react")
        self.assertEqual(run["stop_reason"], "budget_exhausted")
        self.assertTrue(run["checks"]["within_llm_call_limit"])
        self.assertIsNone(run["error_type"])

    def test_legacy_close_and_target_error_remains_attributed(self) -> None:
        self.assertTrue(
            _is_question_update_validation_error(
                "an act decision cannot target a question updated to terminal status"
            )
        )
        self.assertFalse(
            _is_question_update_validation_error("Qwen reused an earlier decision_id")
        )

    def test_failure_summary_redacts_a_configured_secret(self) -> None:
        failure = qwen_reliability_failure(
            run_number=1,
            paid_llm_call_count=1,
            max_llm_calls=20,
            call_limit_exceeded=False,
            error=RuntimeError("provider rejected secret-value"),
            redact_values=["secret-value"],
        )

        self.assertNotIn("secret-value", failure["error_message"])
        self.assertIn("[REDACTED]", failure["error_message"])


if __name__ == "__main__":
    unittest.main()
