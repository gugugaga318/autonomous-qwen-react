from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from run_formal_blind_rca import (  # noqa: E402
    _decision_critical_unavailable_evidence_ids,
    _execution_layer,
    _strict_qwen_acceptance_reasons,
)


def clean_result() -> dict[str, object]:
    return {
        "error": None,
        "job_status": "completed",
        "actual_orchestration_mode": "llm_react",
        "fallback_reason": None,
        "hypothesis_candidate_source": "qwen",
        "hypothesis_candidate_count": 1,
        "hypothesis_candidate_fallback_reason": None,
        "provider_failures": [],
        "llm_call_cap_exceeded": False,
        "planner_stop_proposed_by": "qwen",
        "planner_stop_reason": "goal_satisfied",
        "terminal_question_updates_source": "python_evidence_gate",
    }


def test_execution_layer_qwen_candidate_metric_requires_a_candidate() -> None:
    without_candidate = {
        **clean_result(),
        "hypothesis_candidate_count": 0,
        "hypothesis_candidate_fallback_reason": "qwen_candidate_provider_failed",
    }

    metrics = _execution_layer([without_candidate])

    assert metrics["qwen_candidate_count"] == 0
    assert metrics["qwen_candidate_rate"] == 0.0


def test_execution_layer_counts_nonempty_qwen_candidate_output() -> None:
    metrics = _execution_layer([clean_result()])

    assert metrics["qwen_candidate_count"] == 1
    assert metrics["qwen_candidate_rate"] == 1.0


def test_strict_qwen_accepts_a_clean_llm_react_case() -> None:
    assert _strict_qwen_acceptance_reasons(
        clean_result(),
        requested_mode="llm_react",
        agent_mode="llm",
    ) == []


def test_process_completion_does_not_hide_internal_qwen_fallbacks() -> None:
    result = {
        **clean_result(),
        "actual_orchestration_mode": "controlled_react",
        "fallback_reason": "qwen_next_action_output_invalid",
        "hypothesis_candidate_source": "deterministic_fallback",
        "hypothesis_candidate_fallback_reason": (
            "qwen_hypothesis_candidate_generation_failed"
        ),
    }

    reasons = _strict_qwen_acceptance_reasons(
        result,
        requested_mode="llm_react",
        agent_mode="llm",
    )

    assert "orchestration_fallback" in reasons
    assert "orchestration_fallback_reason_present" in reasons
    assert "hypothesis_candidate_not_qwen" in reasons
    assert "hypothesis_candidate_fallback" in reasons


def test_strict_qwen_requires_a_valid_stop_source_and_python_terminal_gate() -> None:
    result = {
        **clean_result(),
        "planner_stop_proposed_by": None,
        "terminal_question_updates_source": "qwen",
    }

    reasons = _strict_qwen_acceptance_reasons(
        result,
        requested_mode="llm_react",
        agent_mode="llm",
    )

    assert "planner_stop_source_invalid" in reasons
    assert "terminal_updates_not_python_evidence_gate" in reasons


def test_strict_qwen_accepts_governed_python_no_gain_stop() -> None:
    result = {
        **clean_result(),
        "planner_stop_proposed_by": "python_runtime",
        "planner_stop_reason": "no_allowed_action",
    }

    assert _strict_qwen_acceptance_reasons(
        result,
        requested_mode="llm_react",
        agent_mode="llm",
    ) == []


def test_strict_qwen_rejects_ungoverned_python_stop() -> None:
    result = {
        **clean_result(),
        "planner_stop_proposed_by": "python_runtime",
        "planner_stop_reason": "goal_satisfied",
    }

    reasons = _strict_qwen_acceptance_reasons(
        result,
        requested_mode="llm_react",
        agent_mode="llm",
    )

    assert "python_runtime_stop_not_governed" in reasons


def test_strict_qwen_accepts_evidence_proven_data_unavailable_stop() -> None:
    result = {
        **clean_result(),
        "planner_stop_proposed_by": "python_runtime",
        "planner_stop_reason": "data_unavailable",
        "conclusion_status": "insufficient_evidence",
        "required_unavailable_evidence_ids": ["EV_REQUIRED_GENEALOGY_MISSING"],
    }

    assert _strict_qwen_acceptance_reasons(
        result,
        requested_mode="llm_react",
        agent_mode="llm",
    ) == []


def test_strict_qwen_rejects_unproven_data_unavailable_stop() -> None:
    result = {
        **clean_result(),
        "planner_stop_proposed_by": "python_runtime",
        "planner_stop_reason": "data_unavailable",
        "conclusion_status": "inconclusive",
        "required_unavailable_evidence_ids": [],
    }

    reasons = _strict_qwen_acceptance_reasons(
        result,
        requested_mode="llm_react",
        agent_mode="llm",
    )

    assert "python_runtime_stop_not_governed" in reasons


def test_strict_qwen_accepts_competition_governed_data_unavailable_stop() -> None:
    result = {
        **clean_result(),
        "planner_stop_proposed_by": "python_runtime",
        "planner_stop_reason": "data_unavailable",
        "conclusion_status": "insufficient_evidence",
        "required_unavailable_evidence_ids": [],
        "decision_critical_unavailable_evidence_ids": ["EV_SCOPED_FDC_MISSING"],
        "competition_status": "blocked_by_missing_data",
        "competition_terminal_reason": (
            "no_high_value_action_after_required_source_unavailable"
        ),
        "competition_lifecycle_valid": True,
        "investigation_decision_accepted": True,
    }

    assert _strict_qwen_acceptance_reasons(
        result,
        requested_mode="llm_react",
        agent_mode="llm",
    ) == []


def test_strict_qwen_rejects_unproven_competition_missing_data_stop() -> None:
    result = {
        **clean_result(),
        "planner_stop_proposed_by": "python_runtime",
        "planner_stop_reason": "data_unavailable",
        "conclusion_status": "insufficient_evidence",
        "required_unavailable_evidence_ids": [],
        "decision_critical_unavailable_evidence_ids": [],
        "competition_status": "blocked_by_missing_data",
        "competition_terminal_reason": (
            "no_high_value_action_after_required_source_unavailable"
        ),
        "competition_lifecycle_valid": True,
        "investigation_decision_accepted": True,
    }

    reasons = _strict_qwen_acceptance_reasons(
        result,
        requested_mode="llm_react",
        agent_mode="llm",
    )

    assert "python_runtime_stop_not_governed" in reasons


def test_decision_critical_missing_projection_requires_typed_blocked_evidence() -> None:
    result = _decision_critical_unavailable_evidence_ids(
        competition_trace={
            "resolution_evidence_ids": ["EV_CONTEXT"],
            "lane_resolutions": [
                {
                    "status": "blocked",
                    "evidence_ids": ["EV_CONTEXT", "EV_SCOPED_FDC_MISSING"],
                },
                {
                    "status": "unresolved",
                    "evidence_ids": ["EV_UNRELATED_MISSING"],
                },
            ],
        },
        rca_details={"blocking_data_missing_evidence_ids": []},
        data_missing_evidence_ids={
            "EV_SCOPED_FDC_MISSING",
            "EV_UNRELATED_MISSING",
        },
    )

    assert result == ["EV_SCOPED_FDC_MISSING"]


def test_strict_qwen_is_not_applied_to_non_real_qwen_configuration() -> None:
    assert _strict_qwen_acceptance_reasons(
        {"error": "failed"},
        requested_mode="controlled_react",
        agent_mode="deterministic",
    ) == []


def test_execution_layer_keeps_completion_and_strict_qwen_separate() -> None:
    clean = {**clean_result(), "workflow_completed": True, "strict_qwen_accepted": True}
    degraded = {
        **clean_result(),
        "workflow_completed": True,
        "strict_qwen_accepted": False,
        "actual_orchestration_mode": "controlled_react",
        "provider_failure": True,
        "hypothesis_candidate_source": "deterministic_fallback",
        "planner_stop_proposed_by": None,
        "terminal_question_updates_source": None,
    }

    layer = _execution_layer([clean, degraded])

    assert layer["workflow_completion_rate"] == 1.0
    assert layer["strict_qwen_acceptance_rate"] == 0.5
    assert layer["llm_react_preservation_rate"] == 0.5
    assert layer["provider_clean_rate"] == 0.5


def test_execution_layer_reports_governed_python_stop_separately() -> None:
    governed = {
        **clean_result(),
        "workflow_completed": True,
        "strict_qwen_accepted": True,
        "planner_stop_proposed_by": "python_runtime",
        "planner_stop_reason": "no_allowed_action",
    }

    layer = _execution_layer([governed])

    assert layer["qwen_stop_proposal_count"] == 0
    assert layer["governed_python_stop_count"] == 1
    assert layer["governed_python_stop_rate"] == 1.0


def test_execution_layer_keeps_candidate_competition_semantics_separate() -> None:
    accepted = {
        **clean_result(),
        "workflow_completed": True,
        "strict_qwen_accepted": True,
        "competition_status": "complete_confirmed",
        "execution_accepted": True,
        "competition_lifecycle_valid": True,
        "investigation_decision_accepted": True,
        "root_cause_confirmed": True,
        "qwen_competition_accepted": True,
    }
    failed = {
        **clean_result(),
        "workflow_completed": True,
        "strict_qwen_accepted": True,
        "competition_status": "failed",
        "competition_failure_reason": "alternative_direction_not_generated",
        "execution_accepted": True,
        "competition_lifecycle_valid": True,
        "investigation_decision_accepted": False,
        "root_cause_confirmed": False,
        "qwen_competition_accepted": False,
    }

    layer = _execution_layer([accepted, failed])

    assert layer["strict_qwen_acceptance_rate"] == 1.0
    assert layer["qwen_competition_evaluated_count"] == 2
    assert layer["qwen_competition_accepted_count"] == 1
    assert layer["qwen_competition_acceptance_rate"] == 0.5
    assert layer["execution_acceptance_rate"] == 1.0
    assert layer["competition_lifecycle_valid_rate"] == 1.0
    assert layer["investigation_decision_acceptance_rate"] == 0.5
    assert layer["root_cause_confirmation_rate"] == 0.5


def test_strict_qwen_rejects_active_terminal_competition() -> None:
    result = {
        **clean_result(),
        "competition_lifecycle_valid": False,
        "investigation_decision_accepted": False,
    }

    reasons = _strict_qwen_acceptance_reasons(
        result,
        requested_mode="llm_react",
        agent_mode="llm",
    )

    assert "competition_lifecycle_invalid" in reasons
    assert "investigation_decision_invalid" in reasons
