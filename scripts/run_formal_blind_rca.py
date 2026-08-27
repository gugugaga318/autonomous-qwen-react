"""Run the formal RCA blind set using only its public data packet.

This runner deliberately has no Ground Truth input.  It reads the public case
catalogue and public CSV tables, writes one RCA state per case, and records the
exact public-file hashes used by the run.  Score the resulting directory only
with ``score_formal_blind_rca.py`` after the run has completed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = ROOT / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from validate_sealed_blind_packet import (  # noqa: E402
    FORMAL_V2_ROLE,
    validate_sealed_public_packet,
)
from yield_rca_core.investigation_models import (  # noqa: E402
    OrchestrationMode,
    StopReason,
)
from yield_rca_core.llm_gateway import (  # noqa: E402
    LLMCallError,
    LLMClient,
    LLMRequest,
    LLMResponse,
    LLMSettings,
    build_llm_client,
)
from yield_rca_core.workflow import build_csv_workflow  # noqa: E402

DEFAULT_PUBLIC_DIR = ROOT / ".blind_evaluation" / "formal_v1_candidate_r2" / "public"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "formal_blind_v1" / "controlled_react_run"
SUPPORTED_CASE_KEYS = {
    "case_id",
    "source_lot_id",
    "query",
    "detected_module",
    "detected_operation",
    "known_observation",
    "candidate_peer_lot_ids",
    "declared_unavailable_sources",
}
DEVELOPMENT_REGRESSION_ROLE = "development_regression"
EVALUATION_ROLES = (DEVELOPMENT_REGRESSION_ROLE, FORMAL_V2_ROLE)


class CappedLLMClient:
    """Apply a per-case ceiling when the runner is configured for a real LLM."""

    def __init__(self, delegate: LLMClient, *, max_calls: int) -> None:
        self.delegate = delegate
        self.max_calls = max_calls
        self.call_count = 0
        self.limit_exceeded = False
        self.last_prompt_name: str | None = None
        self.last_prompt_agent: str | None = None
        self.provider_failures: list[dict[str, Any]] = []
        self.provider = delegate.provider
        self.model = delegate.model

    @property
    def remaining_calls(self) -> int:
        """Expose the non-consuming budget capability used by core preflight."""

        return max(0, self.max_calls - self.call_count)

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        if self.call_count >= self.max_calls:
            self.limit_exceeded = True
            raise LLMCallError(
                "Formal blind case exceeded its configured LLM-call cap",
                failure_category="formal_blind_call_cap",
            )
        self.call_count += 1
        self.last_prompt_name = request.prompt_name
        self.last_prompt_agent = request.agent
        try:
            return self.delegate.complete_json(request)
        except LLMCallError as exc:
            self.provider_failures.append(
                {
                    "prompt_name": request.prompt_name,
                    "agent": request.agent,
                    "failure_category": exc.failure_category,
                    "status_code": exc.status_code,
                    "provider_code": exc.provider_code,
                    "request_id": exc.request_id,
                    "provider_attempt_count": exc.call_attempt_count,
                }
            )
            raise


@dataclass(frozen=True)
class PublicCase:
    case_id: str
    source_lot_id: str
    query: str
    declared_unavailable_sources: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PublicCase:
        unknown = set(payload) - SUPPORTED_CASE_KEYS
        if unknown:
            raise ValueError(f"public case contains unsupported keys: {sorted(unknown)}")
        raw_unavailable = payload.get("declared_unavailable_sources", [])
        if not isinstance(raw_unavailable, list) or any(
            not isinstance(item, str) or not item.strip()
            for item in raw_unavailable
        ):
            raise ValueError(
                "declared_unavailable_sources must be an array of non-empty strings"
            )
        unavailable = tuple(item.strip() for item in raw_unavailable)
        if len(unavailable) != len(set(unavailable)):
            raise ValueError("declared_unavailable_sources must not contain duplicates")
        case = cls(
            case_id=str(payload["case_id"]).strip(),
            source_lot_id=str(payload["source_lot_id"]).strip(),
            query=str(payload["query"]).strip(),
            declared_unavailable_sources=unavailable,
        )
        if not case.case_id or not case.source_lot_id or not case.query:
            raise ValueError("public case_id, source_lot_id, and query are required")
        return case


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _public_files(public_dir: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(public_dir)).replace("\\", "/"),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(public_dir.rglob("*"))
        if path.is_file()
    ]


def _git_command(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _code_snapshot() -> tuple[str | None, bool]:
    completed = _git_command("rev-parse", "HEAD")
    status = _git_command("status", "--porcelain", "--untracked-files=no")
    untracked_runtime = _git_command(
        "ls-files",
        "--others",
        "--exclude-standard",
        "--",
        "core",
        "backend",
        "pyproject.toml",
        "scripts/run_formal_blind_rca.py",
        "scripts/validate_sealed_blind_packet.py",
    )
    value = completed.stdout.strip()
    commit = value if completed.returncode == 0 and value else None
    clean = bool(
        status.returncode == 0
        and untracked_runtime.returncode == 0
        and not status.stdout.strip()
        and not untracked_runtime.stdout.strip()
    )
    return commit, clean


def _validate_sealed_run_selection(*, case_ids: list[str], overwrite: bool) -> None:
    if case_ids:
        raise ValueError("sealed_blind execution must run the complete case catalogue")
    if overwrite:
        raise ValueError("sealed_blind execution cannot overwrite an earlier run")


def load_public_cases(public_dir: Path) -> tuple[str, list[PublicCase]]:
    """Load only the case catalogue exposed to the RCA system."""

    resolved = public_dir.resolve()
    if resolved.name != "public":
        raise ValueError("--public-dir must point to the formal packet's public directory")
    if not (resolved / "fab_data").is_dir():
        raise ValueError("public packet is missing fab_data/")
    payload = json.loads((resolved / "cases.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("public cases.json must contain an object")
    dataset_id = str(payload.get("dataset_id", "")).strip()
    raw_cases = payload.get("cases")
    if not dataset_id or not isinstance(raw_cases, list):
        raise ValueError("public cases.json requires dataset_id and cases")
    cases = [PublicCase.from_dict(dict(item)) for item in raw_cases]
    case_ids = [case.case_id for case in cases]
    if len(cases) != int(payload.get("case_count", 0)) or len(case_ids) != len(set(case_ids)):
        raise ValueError("public case_count or unique case_id contract is invalid")
    return dataset_id, cases


def _select_cases(cases: Iterable[PublicCase], requested: list[str]) -> list[PublicCase]:
    requested_ids = [item.strip() for item in requested if item.strip()]
    if not requested_ids:
        return list(cases)
    by_id = {case.case_id: case for case in cases}
    unknown = sorted(set(requested_ids) - set(by_id))
    if unknown:
        raise ValueError(f"unknown --case-id values: {unknown}")
    return [by_id[case_id] for case_id in requested_ids]


def _prepare_output(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"output directory already exists: {output_dir}; use --overwrite to replace it"
            )
        shutil.rmtree(output_dir)
    (output_dir / "states").mkdir(parents=True)


def _strict_qwen_acceptance_reasons(
    result: dict[str, Any],
    *,
    requested_mode: str,
    agent_mode: str,
) -> list[str]:
    """Separate process completion from a clean, governed real-Qwen path."""

    if requested_mode != OrchestrationMode.LLM_REACT.value or agent_mode != "llm":
        return []
    reasons: list[str] = []
    if result.get("error") is not None or result.get("job_status") != "completed":
        reasons.append("workflow_not_completed")
    if result.get("actual_orchestration_mode") != OrchestrationMode.LLM_REACT.value:
        reasons.append("orchestration_fallback")
    if result.get("fallback_reason"):
        reasons.append("orchestration_fallback_reason_present")
    if result.get("hypothesis_candidate_source") != "qwen":
        reasons.append("hypothesis_candidate_not_qwen")
    if result.get("hypothesis_candidate_fallback_reason"):
        reasons.append("hypothesis_candidate_fallback")
    if result.get("provider_failures"):
        reasons.append("provider_failure")
    if result.get("llm_call_cap_exceeded"):
        reasons.append("llm_call_cap_exceeded")
    stop_proposer = result.get("planner_stop_proposed_by")
    if stop_proposer == "python_runtime":
        governed_stop_reasons = {
            StopReason.NO_ALLOWED_ACTION.value,
            StopReason.NO_HIGH_VALUE_ACTION.value,
            StopReason.BUDGET_EXHAUSTED.value,
        }
        governed_legacy_data_unavailable = (
            result.get("planner_stop_reason") == StopReason.DATA_UNAVAILABLE.value
            and result.get("conclusion_status") == "insufficient_evidence"
            and bool(result.get("required_unavailable_evidence_ids"))
        )
        governed_competition_data_unavailable = (
            result.get("planner_stop_reason") == StopReason.DATA_UNAVAILABLE.value
            and result.get("conclusion_status") == "insufficient_evidence"
            and result.get("competition_status") == "blocked_by_missing_data"
            and result.get("competition_terminal_reason")
            == "no_high_value_action_after_required_source_unavailable"
            and result.get("competition_lifecycle_valid") is True
            and result.get("investigation_decision_accepted") is True
            and bool(result.get("decision_critical_unavailable_evidence_ids"))
        )
        if (
            result.get("planner_stop_reason") not in governed_stop_reasons
            and not governed_legacy_data_unavailable
            and not governed_competition_data_unavailable
        ):
            reasons.append("python_runtime_stop_not_governed")
    elif stop_proposer != "qwen":
        reasons.append("planner_stop_source_invalid")
    if result.get("terminal_question_updates_source") != "python_evidence_gate":
        reasons.append("terminal_updates_not_python_evidence_gate")
    if result.get("competition_lifecycle_valid") is False:
        reasons.append("competition_lifecycle_invalid")
    if result.get("investigation_decision_accepted") is False:
        reasons.append("investigation_decision_invalid")
    return reasons


def _decision_critical_unavailable_evidence_ids(
    *,
    competition_trace: dict[str, Any],
    rca_details: dict[str, Any],
    data_missing_evidence_ids: set[str],
) -> list[str]:
    """Project typed missing Evidence that governed a terminal Competition stop."""

    candidate_ids = [
        *(competition_trace.get("resolution_evidence_ids") or []),
        *(rca_details.get("blocking_data_missing_evidence_ids") or []),
        *(
            evidence_id
            for resolution in (competition_trace.get("lane_resolutions") or [])
            if isinstance(resolution, dict)
            and resolution.get("status") == "blocked"
            for evidence_id in (resolution.get("evidence_ids") or [])
        ),
    ]
    return list(
        dict.fromkeys(
            str(evidence_id)
            for evidence_id in candidate_ids
            if str(evidence_id) in data_missing_evidence_ids
        )
    )


def _execution_layer(results: list[dict[str, Any]]) -> dict[str, Any]:
    case_count = len(results)

    def ratio(count: int) -> float:
        return round(count / case_count, 6) if case_count else 1.0

    completed = sum(bool(item.get("workflow_completed")) for item in results)
    strict = sum(bool(item.get("strict_qwen_accepted")) for item in results)
    llm_react = sum(
        item.get("actual_orchestration_mode") == OrchestrationMode.LLM_REACT.value
        for item in results
    )
    provider_clean = sum(not bool(item.get("provider_failure")) for item in results)
    qwen_candidates = sum(
        item.get("hypothesis_candidate_source") == "qwen" for item in results
    )
    qwen_stop = sum(item.get("planner_stop_proposed_by") == "qwen" for item in results)
    governed_python_stop = sum(
        item.get("planner_stop_proposed_by") == "python_runtime"
        and (
            item.get("planner_stop_reason")
            in {
                StopReason.NO_ALLOWED_ACTION.value,
                StopReason.NO_HIGH_VALUE_ACTION.value,
                StopReason.BUDGET_EXHAUSTED.value,
            }
            or (
                item.get("planner_stop_reason")
                == StopReason.DATA_UNAVAILABLE.value
                and item.get("conclusion_status") == "insufficient_evidence"
                and bool(item.get("required_unavailable_evidence_ids"))
            )
        )
        for item in results
    )
    python_terminal = sum(
        item.get("terminal_question_updates_source") == "python_evidence_gate"
        for item in results
    )
    competition_evaluated = sum(
        item.get("competition_status") not in {None, "not_evaluated"}
        for item in results
    )
    competition_accepted = sum(
        bool(item.get("qwen_competition_accepted")) for item in results
    )
    execution_accepted = sum(
        bool(item.get("execution_accepted")) for item in results
    )
    lifecycle_valid = sum(
        bool(item.get("competition_lifecycle_valid")) for item in results
    )
    investigation_accepted = sum(
        bool(item.get("investigation_decision_accepted")) for item in results
    )
    root_cause_confirmed = sum(
        bool(item.get("root_cause_confirmed")) for item in results
    )
    return {
        "case_count": case_count,
        "workflow_completed_count": completed,
        "workflow_completion_rate": ratio(completed),
        "strict_qwen_accepted_count": strict,
        "strict_qwen_acceptance_rate": ratio(strict),
        "llm_react_preserved_count": llm_react,
        "llm_react_preservation_rate": ratio(llm_react),
        "provider_clean_count": provider_clean,
        "provider_clean_rate": ratio(provider_clean),
        "qwen_candidate_count": qwen_candidates,
        "qwen_candidate_rate": ratio(qwen_candidates),
        "qwen_stop_proposal_count": qwen_stop,
        "qwen_stop_proposal_rate": ratio(qwen_stop),
        "governed_python_stop_count": governed_python_stop,
        "governed_python_stop_rate": ratio(governed_python_stop),
        "python_terminal_gate_count": python_terminal,
        "python_terminal_gate_rate": ratio(python_terminal),
        "qwen_competition_evaluated_count": competition_evaluated,
        "qwen_competition_accepted_count": competition_accepted,
        "qwen_competition_acceptance_rate": ratio(competition_accepted),
        "execution_accepted_count": execution_accepted,
        "execution_acceptance_rate": ratio(execution_accepted),
        "competition_lifecycle_valid_count": lifecycle_valid,
        "competition_lifecycle_valid_rate": ratio(lifecycle_valid),
        "investigation_decision_accepted_count": investigation_accepted,
        "investigation_decision_acceptance_rate": ratio(
            investigation_accepted
        ),
        "root_cause_confirmed_count": root_cause_confirmed,
        "root_cause_confirmation_rate": ratio(root_cause_confirmed),
        "llm_call_count": sum(int(item.get("llm_call_count") or 0) for item in results),
    }


def run_formal_blind(args: argparse.Namespace) -> dict[str, Any]:
    public_dir = args.public_dir.resolve()
    dataset_id, catalog_cases = load_public_cases(public_dir)
    sealed_declaration: dict[str, Any] | None = None
    if args.evaluation_role == FORMAL_V2_ROLE:
        sealed_declaration = validate_sealed_public_packet(public_dir)
        if sealed_declaration["dataset_id"] != dataset_id:
            raise ValueError("sealed declaration and public catalogue dataset_id differ")
        _validate_sealed_run_selection(
            case_ids=args.case_id,
            overwrite=args.overwrite,
        )
    cases = _select_cases(catalog_cases, args.case_id)
    mode = OrchestrationMode(args.orchestration_mode).value
    settings = LLMSettings.from_env()
    if settings.agent_mode == "llm" and not args.confirm_paid_qwen:
        raise ValueError(
            "--confirm-paid-qwen is required when YIELD_RCA_AGENT_MODE=llm"
        )
    code_commit, code_worktree_clean = _code_snapshot()
    if args.evaluation_role == FORMAL_V2_ROLE and (
        code_commit is None or not code_worktree_clean
    ):
        raise ValueError(
            "sealed_blind execution requires a committed, clean tracked worktree"
        )
    _prepare_output(args.output_dir.resolve(), overwrite=args.overwrite)
    output_dir = args.output_dir.resolve()

    public_manifest = _public_files(public_dir)
    manifest = {
        "schema_version": "1.0",
        "dataset_id": dataset_id,
        "run_kind": "formal_rca_blind_execution",
        "evaluation_role": args.evaluation_role,
        "created_at": datetime.now(UTC).isoformat(),
        "code_commit": code_commit,
        "code_worktree_clean": code_worktree_clean,
        "governance": {
            "dataset_generation_independent": (
                bool(sealed_declaration["dataset_generation_independent"])
                if sealed_declaration is not None
                else False
            ),
            "ground_truth_custodian": (
                sealed_declaration["ground_truth_custodian"]
                if sealed_declaration is not None
                else "development_regression"
            ),
            "sealed_before_execution": sealed_declaration is not None,
            "ground_truth_sha256_commitment": (
                sealed_declaration["ground_truth_sha256_commitment"]
                if sealed_declaration is not None
                else None
            ),
            "development_agent_ground_truth_access": (
                sealed_declaration["development_agent_ground_truth_access"]
                if sealed_declaration is not None
                else "previously_exposed"
            ),
        },
        "input_boundary": {
            "mode": "public_only",
            "public_dir": str(public_dir),
            "allowed_files": public_manifest,
            "ground_truth_loaded": False,
        },
        "configuration": {
            "orchestration_mode": mode,
            "agent_mode": settings.agent_mode,
            "provider": settings.provider if settings.agent_mode != "deterministic" else None,
            "model": settings.model if settings.agent_mode != "deterministic" else None,
            "max_llm_calls_per_case": args.max_llm_calls_per_case,
        },
        "selected_case_ids": [case.case_id for case in cases],
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    results: list[dict[str, Any]] = []
    for case in cases:
        client: CappedLLMClient | None = None
        try:
            delegate = build_llm_client(settings)
            if delegate is not None:
                client = CappedLLMClient(
                    delegate,
                    max_calls=args.max_llm_calls_per_case,
                )
            workflow = build_csv_workflow(
                public_dir / "fab_data",
                llm_settings=settings,
                llm_client=client,
                orchestration_mode=mode,
            )
            state = workflow.run(
                case.query,
                job_id=f"FORMAL_BLIND_{case.case_id}",
                lot_id=case.source_lot_id,
                declared_unavailable_sources=case.declared_unavailable_sources,
            )
            state_path = output_dir / "states" / f"{case.case_id}.json"
            state_path.write_text(
                json.dumps(state.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            hypothesis = state.authoritative_hypothesis
            rca_finding = state.authoritative_rca_finding
            rca_details = rca_finding.details if rca_finding is not None else {}
            candidate_generation = dict(
                rca_details.get("hypothesis_candidate_generation", {})
            )
            competition_trace = (
                state.competition_trace.to_dict()
                if state.competition_trace is not None
                else {}
            )
            competition_requirement = competition_trace.get(
                "competition_requirement",
                rca_details.get("competition_requirement"),
            )
            competition_status = competition_trace.get(
                "competition_status",
                rca_details.get("competition_status"),
            )
            competition_failure_reason = competition_trace.get(
                "competition_failure_reason",
                rca_details.get("competition_failure_reason"),
            )
            competition_gap_reason = competition_trace.get(
                "competition_gap_reason",
                rca_details.get("competition_gap_reason"),
            )
            new_terminal_competition_statuses = {
                "complete_confirmed",
                "complete_rejected",
                "exhausted",
                "blocked_by_missing_data",
                "budget_exhausted",
                "failed",
            }
            accepted_terminal_competition_statuses = {
                "not_required",
                "resolved",  # Legacy terminal State compatibility.
                *new_terminal_competition_statuses,
            }
            competition_terminal_reason = competition_trace.get(
                "terminal_reason"
            )
            competition_lifecycle_valid = (
                competition_status in {None, "not_evaluated"}
                or (
                    competition_status in accepted_terminal_competition_statuses
                    and (
                        competition_status not in new_terminal_competition_statuses
                        or bool(competition_terminal_reason)
                    )
                )
            )
            investigation_decision_accepted = (
                competition_lifecycle_valid
                and competition_status != "failed"
                and competition_failure_reason is None
            )
            root_cause_confirmed = (
                competition_status == "complete_confirmed"
                and rca_details.get("conclusion_status") == "supported"
                and hypothesis is not None
                and hypothesis.status == "supported"
            )
            execution_accepted = (
                state.job.status == "completed"
                and state.execution_metadata.get("orchestration_mode") == mode
                and not state.execution_metadata.get(
                    "orchestration_fallback_reason"
                )
                and not (client.provider_failures if client is not None else [])
                and not (client.limit_exceeded if client is not None else False)
            )
            competition_evaluated = competition_status not in {
                None,
                "not_evaluated",
            }
            qwen_competition_accepted = (
                competition_evaluated and investigation_decision_accepted
            )
            ranked_candidates = list(rca_details.get("ranked_candidates", []))
            required_unavailable_evidence_ids = [
                item.evidence_id
                for item in state.evidence
                if item.evidence_type == "data_missing"
                and item.metadata.get("required_for_confirmation") is True
            ]
            data_missing_evidence_ids = {
                item.evidence_id
                for item in state.evidence
                if item.evidence_type == "data_missing"
            }
            decision_critical_unavailable_evidence_ids = (
                _decision_critical_unavailable_evidence_ids(
                    competition_trace=competition_trace,
                    rca_details=rca_details,
                    data_missing_evidence_ids=data_missing_evidence_ids,
                )
            )
            case_result = {
                    "case_id": case.case_id,
                    "source_lot_id": case.source_lot_id,
                    "state_file": str(state_path.relative_to(output_dir)).replace("\\", "/"),
                    "error": None,
                    "job_status": state.job.status,
                    "workflow_completed": state.job.status == "completed",
                    "actual_orchestration_mode": state.execution_metadata.get(
                        "orchestration_mode"
                    ),
                    "fallback_reason": state.execution_metadata.get(
                        "orchestration_fallback_reason"
                    ),
                    "hypothesis_status": hypothesis.status if hypothesis else None,
                    "conclusion_status": rca_details.get("conclusion_status"),
                    "root_cause": hypothesis.root_cause if hypothesis else None,
                    "hypothesis_candidate_source": candidate_generation.get(
                        "source"
                    ),
                    "hypothesis_candidate_count": candidate_generation.get(
                        "candidate_count"
                    ),
                    "hypothesis_candidate_fallback_reason": candidate_generation.get(
                        "fallback_reason"
                    ),
                    "ranked_candidate_count": len(ranked_candidates),
                    "competition_requirement": competition_requirement,
                    "competition_status": competition_status,
                    "competition_type": competition_trace.get(
                        "competition_type",
                        rca_details.get("competition_type"),
                    ),
                    "competition_failure_reason": competition_failure_reason,
                    "competition_gap_reason": competition_gap_reason,
                    "competition_terminal_reason": competition_terminal_reason,
                    "execution_accepted": execution_accepted,
                    "competition_lifecycle_valid": competition_lifecycle_valid,
                    "investigation_decision_accepted": (
                        investigation_decision_accepted
                    ),
                    "root_cause_confirmed": root_cause_confirmed,
                    "qwen_competition_accepted": qwen_competition_accepted,
                    "top_candidate_basis": (
                        ranked_candidates[0].get("basis")
                        if ranked_candidates
                        else None
                    ),
                    "impact_lot_count": len(state.impact_lots),
                    "evidence_count": len(state.evidence),
                    "action_count": len(state.action_history),
                    "llm_call_count": int(
                        state.execution_metadata.get("llm_call_count", 0)
                    ),
                    "last_llm_prompt_name": (
                        client.last_prompt_name if client is not None else None
                    ),
                    "last_llm_prompt_agent": (
                        client.last_prompt_agent if client is not None else None
                    ),
                    "provider_failures": (
                        list(client.provider_failures) if client is not None else []
                    ),
                    "llm_call_cap_exceeded": (
                        client.limit_exceeded if client is not None else False
                    ),
                    "provider_failure": bool(
                        client.provider_failures if client is not None else []
                    ),
                    "planner_stop_proposed_by": state.execution_metadata.get(
                        "planner_stop_proposed_by"
                    ),
                    "planner_stop_reason": state.stop_reason,
                    "terminal_question_updates_source": state.execution_metadata.get(
                        "terminal_question_updates_source"
                    ),
                    "required_unavailable_evidence_ids": (
                        required_unavailable_evidence_ids
                    ),
                    "decision_critical_unavailable_evidence_ids": (
                        decision_critical_unavailable_evidence_ids
                    ),
                }
            acceptance_reasons = _strict_qwen_acceptance_reasons(
                case_result,
                requested_mode=mode,
                agent_mode=settings.agent_mode,
            )
            case_result["strict_qwen_accepted"] = not acceptance_reasons
            case_result["strict_qwen_rejection_reasons"] = acceptance_reasons
            results.append(case_result)
        except Exception as exc:  # noqa: BLE001 - preserve every blind run result
            case_result = {
                    "case_id": case.case_id,
                    "source_lot_id": case.source_lot_id,
                    "state_file": None,
                    "error": f"{type(exc).__name__}: {exc}",
                    "job_status": None,
                    "workflow_completed": False,
                    "actual_orchestration_mode": None,
                    "fallback_reason": None,
                    "hypothesis_status": None,
                    "conclusion_status": None,
                    "root_cause": None,
                    "hypothesis_candidate_source": None,
                    "hypothesis_candidate_count": None,
                    "hypothesis_candidate_fallback_reason": None,
                    "ranked_candidate_count": None,
                    "competition_requirement": None,
                    "competition_status": None,
                    "competition_type": None,
                    "competition_failure_reason": None,
                    "competition_gap_reason": None,
                    "competition_terminal_reason": None,
                    "execution_accepted": False,
                    "competition_lifecycle_valid": False,
                    "investigation_decision_accepted": False,
                    "root_cause_confirmed": False,
                    "qwen_competition_accepted": False,
                    "top_candidate_basis": None,
                    "impact_lot_count": None,
                    "evidence_count": None,
                    "action_count": None,
                    "llm_call_count": client.call_count if client is not None else 0,
                    "last_llm_prompt_name": (
                        client.last_prompt_name if client is not None else None
                    ),
                    "last_llm_prompt_agent": (
                        client.last_prompt_agent if client is not None else None
                    ),
                    "provider_failures": (
                        list(client.provider_failures) if client is not None else []
                    ),
                    "llm_call_cap_exceeded": (
                        client.limit_exceeded if client is not None else False
                    ),
                    "provider_failure": bool(
                        client.provider_failures if client is not None else []
                    ),
                    "planner_stop_proposed_by": None,
                    "planner_stop_reason": None,
                    "terminal_question_updates_source": None,
                    "required_unavailable_evidence_ids": [],
                }
            acceptance_reasons = _strict_qwen_acceptance_reasons(
                case_result,
                requested_mode=mode,
                agent_mode=settings.agent_mode,
            )
            case_result["strict_qwen_accepted"] = not acceptance_reasons
            case_result["strict_qwen_rejection_reasons"] = acceptance_reasons
            results.append(case_result)

    completed = sum(bool(item["workflow_completed"]) for item in results)
    strict_qwen_evaluated = (
        mode == OrchestrationMode.LLM_REACT.value
        and settings.agent_mode == "llm"
    )
    strict_qwen_accepted = sum(
        bool(item["strict_qwen_accepted"]) for item in results
    )
    execution_layer = _execution_layer(results)
    execution_layer["strict_qwen_acceptance_evaluated"] = strict_qwen_evaluated
    payload = {
        "schema_version": "1.0",
        "dataset_id": dataset_id,
        "run_kind": "formal_rca_blind_execution",
        "evaluation_role": args.evaluation_role,
        "case_count": len(results),
        "completed_case_count": completed,
        "failed_case_count": len(results) - completed,
        "strict_qwen_acceptance_evaluated": strict_qwen_evaluated,
        "strict_qwen_accepted_case_count": strict_qwen_accepted,
        "strict_qwen_rejected_case_count": len(results) - strict_qwen_accepted,
        "execution_layer": execution_layer,
        "results": results,
        "notice": "This execution artifact contains no Ground Truth and is not a score.",
    }
    (output_dir / "run_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Formal RCA Blind Execution",
        "",
        f"- Dataset: `{dataset_id}`",
        f"- Evaluation role: `{args.evaluation_role}`",
        f"- Cases completed: {completed}/{len(results)}",
        (
            f"- Strict Qwen accepted: {strict_qwen_accepted}/{len(results)}"
            if strict_qwen_evaluated
            else "- Strict Qwen accepted: not evaluated for this configuration"
        ),
        f"- Requested orchestration: `{mode}`",
        f"- Agent mode: `{settings.agent_mode}`",
        "- Ground Truth loaded: **No**",
        "",
        "Run `score_formal_blind_rca.py` in a separate, explicitly approved step "
        "to score this output.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-dir", type=Path, default=DEFAULT_PUBLIC_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--evaluation-role",
        choices=EVALUATION_ROLES,
        default=DEVELOPMENT_REGRESSION_ROLE,
        help=(
            "V1 and any exposed dataset must use development_regression. "
            "sealed_blind requires an independently supplied sealed manifest."
        ),
    )
    parser.add_argument(
        "--orchestration-mode",
        choices=[item.value for item in OrchestrationMode],
        default=OrchestrationMode.CONTROLLED_REACT.value,
    )
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--max-llm-calls-per-case", type=int, default=20)
    parser.add_argument("--confirm-paid-qwen", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_llm_calls_per_case < 1:
        raise ValueError("--max-llm-calls-per-case must be positive")
    result = run_formal_blind(args)
    print(
        "Formal blind execution: "
        f"completed={result['completed_case_count']}/{result['case_count']}; "
        f"failed={result['failed_case_count']}; "
        + (
            "strict_qwen="
            f"{result['strict_qwen_accepted_case_count']}/{result['case_count']}"
            if result["strict_qwen_acceptance_evaluated"]
            else "strict_qwen=not_evaluated"
        )
    )
    print(f"Results: {args.output_dir.resolve() / 'run_results.json'}")
    execution_failed = result["failed_case_count"] != 0
    strict_qwen_failed = (
        result["strict_qwen_acceptance_evaluated"]
        and result["strict_qwen_rejected_case_count"] != 0
    )
    return 1 if execution_failed or strict_qwen_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
