"""Small, framework-free contracts for model-authored causal hypotheses.

Qwen only supplies the four user-facing hypothesis fields.  The classes in
this module deliberately do not contain equipment, parameter, or impact-lot
fields; those facts are derived from typed Evidence by
``causal_evidence_matrix``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from yield_rca_core.models import ModelValidationError


class CausalClaimStatus(StrEnum):
    """Status of one Python-owned claim in a causal evidence matrix."""

    SUPPORTED = "supported"
    PLAUSIBLE = "plausible"
    INCOMPLETE = "incomplete"
    CONFLICTED = "conflicted"
    UNAVAILABLE = "unavailable"


class CausalClaim(StrEnum):
    """Stable keys used by the matrix and its API/UI projections."""

    EQUIPMENT = "equipment"
    CHAMBER = "chamber"
    OPERATION = "operation"
    PARAMETER = "parameter"
    OUTCOME = "outcome"
    MECHANISM = "mechanism"
    CONTROL = "control"
    CONTRADICTION = "contradiction"
    TEMPORAL = "temporal"
    SCOPE = "scope"


class MechanismSupportSource(StrEnum):
    """How Python established that a mechanism is more than prose."""

    RULE = "rule"
    APPROVED_KNOWLEDGE = "approved_knowledge"
    OBSERVED_INTERMEDIATE = "observed_intermediate"
    EMPIRICAL_DISCRIMINATION = "empirical_discrimination"
    EMPIRICAL_CONVERGENCE = "empirical_convergence"
    LLM_EXPLANATION_ONLY = "llm_explanation_only"


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ModelValidationError(f"{field_name} must be an array")
    values = tuple(str(item).strip() for item in value)
    if any(not item for item in values):
        raise ModelValidationError(f"{field_name} must contain non-empty strings")
    if len(values) != len(set(values)):
        raise ModelValidationError(f"{field_name} must not contain duplicates")
    return values


@dataclass(frozen=True)
class CausalHypothesis:
    """The intentionally compact Qwen-authored candidate contract."""

    root_cause: str
    causal_explanation: str
    supporting_evidence_ids: tuple[str, ...]
    contradicting_evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.root_cause, str) or not self.root_cause.strip():
            raise ModelValidationError("causal hypothesis root_cause must be non-empty")
        if not isinstance(self.causal_explanation, str) or not self.causal_explanation.strip():
            raise ModelValidationError(
                "causal hypothesis causal_explanation must be non-empty"
            )
        if not isinstance(self.supporting_evidence_ids, (list, tuple)):
            raise ModelValidationError("supporting_evidence_ids must be an array")
        if not isinstance(self.contradicting_evidence_ids, (list, tuple)):
            raise ModelValidationError("contradicting_evidence_ids must be an array")
        supporting = tuple(str(item).strip() for item in self.supporting_evidence_ids)
        contradicting = tuple(str(item).strip() for item in self.contradicting_evidence_ids)
        object.__setattr__(self, "supporting_evidence_ids", supporting)
        object.__setattr__(self, "contradicting_evidence_ids", contradicting)
        if any(not item for item in supporting + contradicting):
            raise ModelValidationError("Evidence IDs must contain non-empty strings")
        if not supporting:
            raise ModelValidationError("supporting_evidence_ids must not be empty")
        if set(supporting) & set(contradicting):
            raise ModelValidationError(
                "a causal hypothesis cannot cite Evidence as both supporting and contradicting"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CausalHypothesis:
        if not isinstance(value, Mapping):
            raise ModelValidationError("causal hypothesis must be an object")
        return cls(
            root_cause=str(value.get("root_cause", "")).strip(),
            causal_explanation=str(value.get("causal_explanation", "")).strip(),
            supporting_evidence_ids=_string_tuple(
                value.get("supporting_evidence_ids", []),
                "supporting_evidence_ids",
            ),
            contradicting_evidence_ids=_string_tuple(
                value.get("contradicting_evidence_ids", []),
                "contradicting_evidence_ids",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_cause": self.root_cause,
            "causal_explanation": self.causal_explanation,
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "contradicting_evidence_ids": list(self.contradicting_evidence_ids),
        }


__all__ = [
    "CausalClaim",
    "CausalClaimStatus",
    "CausalHypothesis",
    "MechanismSupportSource",
]
