"""Candidate comparison for evidence-driven RCA.

Python computes the objective comparison signal from the Causal Evidence
Matrices.  Qwen may add a bounded explanation and select one of the already
generated gaps, but it cannot introduce a new candidate or a new Action.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from yield_rca_core.causal_evidence_matrix import (
    CausalEvidenceMatrix,
    compact_causal_evidence_matrix_for_prompt,
)
from yield_rca_core.causal_investigation_models import AlternativeSearchStatus
from yield_rca_core.llm_gateway import LLMClient, LLMOutputValidationError, LLMRequest
from yield_rca_core.models import AgentKind, ModelValidationError

_STATUS_SCORE = {
    "supported": 2,
    "incomplete": 1,
    "unavailable": 0,
    "conflicted": -2,
}
_MAX_COMPARISON_PAYLOAD_CHARS = 48_000


def _matrix_score(matrix: CausalEvidenceMatrix) -> int:
    """Return a transparent, deterministic comparison score."""

    score = sum(_STATUS_SCORE.get(result.status, -2) for result in matrix.claims.values())
    # A mechanism backed by current-Lot empirical convergence is useful, but a
    # rule/approved knowledge source is stronger explanatory context.
    if matrix.mechanism_support_source in {"rule", "approved_knowledge"}:
        score += 1
    if matrix.has_critical_conflict:
        score -= 5
    return score


def _matrix_profile(
    matrix: CausalEvidenceMatrix,
    *,
    candidate_index: int,
) -> dict[str, Any]:
    """Return the Python-owned facts that are safe to use for ranking."""

    statuses = [result.status for result in matrix.claims.values()]
    return {
        "candidate_index": candidate_index,
        "matrix_score": _matrix_score(matrix),
        "matrix_status": matrix.status,
        "supported_claim_count": statuses.count("supported"),
        "incomplete_claim_count": statuses.count("incomplete"),
        "conflicted_claim_count": statuses.count("conflicted"),
        "unavailable_claim_count": statuses.count("unavailable"),
        "has_critical_conflict": matrix.has_critical_conflict,
        "invalid_evidence_ids": list(matrix.invalid_evidence_ids),
    }


def compare_candidate_matrices(
    matrices: Sequence[CausalEvidenceMatrix],
    *,
    evidence_gaps: Sequence[Mapping[str, Any]] = (),
    alternative_search_status: str | None = None,
) -> dict[str, Any]:
    """Compare up to two candidates without making a Qwen-authored claim."""

    if not matrices:
        return {
            "preferred_candidate_index": None,
            "comparison_explanation": "No causal candidates are available.",
            "selected_gap_id": None,
            "scores": [],
            "matrix_profiles": [],
            "unresolved": False,
            "alternative_search_required": False,
            "source": "python",
        }
    matrix_profiles = [
        _matrix_profile(matrix, candidate_index=index)
        for index, matrix in enumerate(matrices)
    ]
    scores = [int(item["matrix_score"]) for item in matrix_profiles]
    best = max(scores)
    winners = [index for index, score in enumerate(scores) if score == best]
    preferred = winners[0] if len(winners) == 1 else None
    if preferred is None:
        explanation = "Candidates are equally strong or cannot be separated by typed Evidence."
    else:
        explanation = (
            f"Candidate {preferred} has the strongest deterministic matrix score "
            f"({scores[preferred]})."
        )
    alternative_search_required = (
        alternative_search_status is not None
        and alternative_search_status
        != AlternativeSearchStatus.ALTERNATIVES_ELIMINATED.value
    )
    return {
        "preferred_candidate_index": preferred,
        "comparison_explanation": explanation,
        "selected_gap_id": None,
        "gap_selection_owner": "adversarial_challenge",
        "scores": scores,
        "matrix_profiles": matrix_profiles,
        "unresolved": (
            (preferred is None and len(matrices) > 1)
            or alternative_search_required
        ),
        "alternative_search_required": alternative_search_required,
        "alternative_search_status": alternative_search_status,
        "source": "python",
    }


@dataclass(frozen=True)
class QwenHypothesisCandidateComparator:
    """Ask Qwen to explain a Python-bounded candidate comparison."""

    llm_client: LLMClient
    prompt_version: str = "v1"

    def __post_init__(self) -> None:
        if self.llm_client is None:
            raise ModelValidationError("candidate comparator requires an LLM client")

    def compare(
        self,
        *,
        request_id: str,
        candidates: Sequence[Mapping[str, Any]],
        matrices: Sequence[CausalEvidenceMatrix],
        evidence_gaps: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if len(candidates) != len(matrices) or not candidates:
            raise LLMOutputValidationError(
                "candidate comparison requires one matrix per candidate"
            )
        python_comparison = compare_candidate_matrices(
            matrices,
            evidence_gaps=evidence_gaps,
        )
        request_payload = {
            "request_id": request_id,
            "candidates": [
                {
                    "root_cause": str(item.get("root_cause", "")),
                    "causal_explanation": str(
                        item.get("causal_explanation", "")
                    ),
                    "causal_evidence_matrix": (
                        compact_causal_evidence_matrix_for_prompt(matrix)
                    ),
                }
                for item, matrix in zip(candidates, matrices, strict=True)
            ],
            "evidence_gaps": [dict(item) for item in evidence_gaps],
            "python_comparison": python_comparison,
        }
        prompt_payload_char_count = len(
            json.dumps(
                request_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        )
        if prompt_payload_char_count > _MAX_COMPARISON_PAYLOAD_CHARS:
            raise LLMOutputValidationError(
                "candidate comparison prompt exceeds the governed payload limit"
            )
        response = self.llm_client.complete_json(
            LLMRequest(
                agent=AgentKind.RCA_REASONING.value,
                prompt_name="causal_candidate_comparator",
                prompt_version=self.prompt_version,
                payload=request_payload,
                temperature=0.0,
            )
        )
        data = response.data
        expected = {
            "preferred_candidate_index",
            "comparison_explanation",
            "selected_gap_id",
        }
        if set(data) != expected:
            raise LLMOutputValidationError(
                "candidate comparison must contain exactly "
                "preferred_candidate_index, comparison_explanation, selected_gap_id"
            )
        index = data.get("preferred_candidate_index")
        if index is not None and (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(candidates)
        ):
            raise LLMOutputValidationError("preferred_candidate_index is out of range")
        explanation = data.get("comparison_explanation")
        if not isinstance(explanation, str) or not explanation.strip():
            raise LLMOutputValidationError("comparison_explanation must be non-empty")
        gap_id = data.get("selected_gap_id")
        if gap_id is not None:
            raise LLMOutputValidationError(
                "selected_gap_id must be null because the adversarial challenge "
                "owns discriminator Gap selection"
            )
        return {
            **python_comparison,
            "preferred_candidate_index": index,
            "comparison_explanation": explanation.strip(),
            "selected_gap_id": None,
            "gap_selection_owner": "adversarial_challenge",
            "source": "qwen",
            "prompt_audit": {
                "prompt_payload_char_count": prompt_payload_char_count,
                "prompt_evidence_count": len(
                    {
                        evidence_id
                        for matrix in matrices
                        for result in matrix.claims.values()
                        for evidence_id in result.evidence_ids
                    }
                ),
                "omitted_entity_count": sum(
                    int(
                        compact_causal_evidence_matrix_for_prompt(matrix)[
                            "projection_audit"
                        ]["omitted_claim_fact_groups"]
                    )
                    for matrix in matrices
                ),
                "omitted_metadata_count": 0,
                "prompt_budget_applied": True,
                "prompt_payload_char_limit": _MAX_COMPARISON_PAYLOAD_CHARS,
            },
        }


__all__ = [
    "QwenHypothesisCandidateComparator",
    "compare_candidate_matrices",
]
