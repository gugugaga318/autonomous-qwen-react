"""Patch 6.1 contracts: terminal stop-reason telemetry naming migration.

``terminal_stop_reason`` is the canonical key for the Finalizer-governed
terminal outcome; ``planner_stop_reason`` survives only as a deprecated
compatibility alias for historical run_results.json consumers. The runner
writes both keys with equal values, acceptance reads strictly canonical
(no or-fallback may mask a missing canonical write), and only the explicit
historical read path tolerates the alias. Stop provenance telemetry
(planner_stop_proposed_by / terminal_stop_projected_by / terminal_state_owner)
and all acceptance semantics are unchanged by this migration.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from run_formal_blind_rca import (  # noqa: E402
    _execution_layer,
    _python_terminal_stop_is_governed,
    _stop_reason_telemetry_fields,
    _strict_qwen_acceptance_reasons,
    _terminal_stop_reason,
)


def governed_result(stop_reason: str) -> dict[str, object]:
    """A governed python-runtime stop in the new two-key output format."""

    return {
        "error": None,
        "job_status": "completed",
        "actual_orchestration_mode": "llm_react",
        "fallback_reason": None,
        "hypothesis_candidate_source": "qwen",
        "provider_failures": [],
        "llm_call_cap_exceeded": False,
        "planner_stop_proposed_by": "python_runtime",
        "terminal_stop_projected_by": "python_investigation_finalizer",
        "terminal_stop_reason": stop_reason,
        "planner_stop_reason": stop_reason,
        "terminal_question_updates_source": "python_evidence_gate",
    }


def historical_result(stop_reason: str) -> dict[str, object]:
    """A pre-patch single-key result as stored in older run_results.json."""

    return {
        "planner_stop_proposed_by": "python_runtime",
        "terminal_stop_projected_by": "python_investigation_finalizer",
        "planner_stop_reason": stop_reason,
    }


def test_writer_emits_canonical_and_alias_with_equal_values() -> None:
    fields = _stop_reason_telemetry_fields("no_high_value_action")

    assert fields["terminal_stop_reason"] == "no_high_value_action"
    assert fields["planner_stop_reason"] == "no_high_value_action"
    assert fields["terminal_stop_reason"] == fields["planner_stop_reason"]

    empty = _stop_reason_telemetry_fields(None)

    assert empty["terminal_stop_reason"] is None
    assert empty["planner_stop_reason"] is None


def test_runner_writers_use_the_telemetry_fields_helper() -> None:
    # Writer-contract guard: both case-result construction sites must emit
    # the canonical key through the shared helper, and no site may write the
    # deprecated alias on its own.
    source = (
        (ROOT / "scripts" / "run_formal_blind_rca.py")
        .read_text(encoding="utf-8")
    )
    assert source.count("**_stop_reason_telemetry_fields(") == 2
    assert '"planner_stop_reason": state.stop_reason' not in source
    assert '"planner_stop_reason": None,' not in source


def test_strict_reader_uses_canonical_and_never_falls_back() -> None:
    # Production in-process results always carry the canonical key; the
    # strict reader must follow it even if a malformed alias disagreed.
    assert (
        _terminal_stop_reason(
            {
                "terminal_stop_reason": "goal_satisfied",
                "planner_stop_reason": "no_high_value_action",
            },
            allow_legacy_alias=False,
        )
        == "goal_satisfied"
    )
    # A missing canonical write must surface as None instead of being
    # masked by the deprecated alias.
    assert (
        _terminal_stop_reason(
            {"planner_stop_reason": "no_high_value_action"},
            allow_legacy_alias=False,
        )
        is None
    )


def test_historical_reader_falls_back_to_alias_only_when_allowed() -> None:
    historical = historical_result("no_high_value_action")

    assert (
        _terminal_stop_reason(historical, allow_legacy_alias=True)
        == "no_high_value_action"
    )
    assert (
        _terminal_stop_reason(historical, allow_legacy_alias=False)
        is None
    )


def test_canonical_and_historical_reads_agree_for_equal_values() -> None:
    stop_reason = "no_allowed_action"

    assert (
        _terminal_stop_reason(
            governed_result(stop_reason),
            allow_legacy_alias=False,
        )
        == _terminal_stop_reason(
            historical_result(stop_reason),
            allow_legacy_alias=True,
        )
    )


def test_new_format_acceptance_behavior_is_unchanged() -> None:
    result = governed_result("no_high_value_action")

    assert _python_terminal_stop_is_governed(result) is True
    assert (
        _strict_qwen_acceptance_reasons(
            result,
            requested_mode="llm_react",
            agent_mode="llm",
        )
        == []
    )


def test_alias_value_deviation_cannot_flip_acceptance() -> None:
    # The canonical value governs acceptance; a stale alias cannot flip it.
    result = {
        **governed_result("goal_satisfied"),
        "planner_stop_reason": "no_high_value_action",
    }

    assert _python_terminal_stop_is_governed(result) is False


def test_metrics_are_identical_across_output_formats() -> None:
    stop_reason = "no_allowed_action"
    new_format = governed_result(stop_reason)
    old_format = historical_result(stop_reason)

    # The tolerant historical read yields the same terminal value, so any
    # consumer replaying older run_results.json through the legacy reader
    # observes identical governed-stop semantics.
    assert (
        _terminal_stop_reason(old_format, allow_legacy_alias=True)
        == _terminal_stop_reason(new_format, allow_legacy_alias=False)
    )
    assert _python_terminal_stop_is_governed(new_format) is True

    metrics = _execution_layer([{**new_format, "strict_qwen_accepted": True}])

    assert metrics["governed_python_stop_count"] == 1
    assert metrics["qwen_stop_proposal_count"] == 0
    assert metrics["strict_qwen_accepted_count"] == 1
