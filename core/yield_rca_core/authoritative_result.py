"""Typed authority contracts for terminal RCA and impact publication.

These value objects are deliberately separate from orchestration and detailed
investigation state. They establish the immutable serialization boundary that
later refactor batches can populate and project without making Report,
Supervisor, or API layers infer an engineering conclusion again.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Self, cast

from yield_rca_core.evidence_models import SCHEMA_VERSION, ModelValidationError

RCAConclusionStatus = Literal["supported", "inconclusive", "insufficient_evidence"]
ImpactPublicationStatus = Literal[
    "confirmed",
    "withheld",
    "unconfirmed",
    "not_evaluated",
]

_RCA_CONCLUSION_STATUSES = {
    "supported",
    "inconclusive",
    "insufficient_evidence",
}
_IMPACT_PUBLICATION_STATUSES = {
    "confirmed",
    "withheld",
    "unconfirmed",
    "not_evaluated",
}
_SUPPORTED_COMPETITION_STATUSES = {
    "complete_confirmed",
    "not_required",
}


def _non_empty(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelValidationError(f"{field_name} must be a non-empty string")
    return value.strip()


def _optional_non_empty(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _non_empty(value, field_name)


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ModelValidationError(f"{field_name} must be a list or tuple")
    normalized = tuple(
        _non_empty(item, f"{field_name}[{index}]") for index, item in enumerate(value)
    )
    if len(normalized) != len(set(normalized)):
        raise ModelValidationError(f"{field_name} must not contain duplicates")
    return normalized


def _status(value: object, allowed: set[str], field_name: str) -> str:
    normalized = _non_empty(value, field_name)
    if normalized not in allowed:
        expected = ", ".join(sorted(allowed))
        raise ModelValidationError(f"{field_name} must be one of: {expected}")
    return normalized


def _schema_version(value: object) -> str:
    normalized = _non_empty(value, "schema_version")
    if normalized != SCHEMA_VERSION:
        raise ModelValidationError(
            f"unsupported schema_version {normalized!r}; expected {SCHEMA_VERSION!r}"
        )
    return normalized


@dataclass(frozen=True)
class AuthoritativeRCAResult:
    """Minimal Python-owned terminal RCA authority.

    Detailed matrices, planner traces, competition traces, and Gate checks stay
    in their owning modules. This object records only the finalized engineering
    conclusion and the IDs required to audit its provenance.
    """

    result_id: str
    source_finding_id: str | None
    source_hypothesis_id: str | None
    conclusion_status: RCAConclusionStatus
    root_cause_candidate_id: str | None
    root_cause: str | None
    confirmation_status: str
    competition_status: str
    terminal_reason: str
    evidence_refs: tuple[str, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "result_id", _non_empty(self.result_id, "result_id"))
        object.__setattr__(
            self,
            "source_finding_id",
            _optional_non_empty(self.source_finding_id, "source_finding_id"),
        )
        object.__setattr__(
            self,
            "source_hypothesis_id",
            _optional_non_empty(self.source_hypothesis_id, "source_hypothesis_id"),
        )
        conclusion_status = _status(
            self.conclusion_status,
            _RCA_CONCLUSION_STATUSES,
            "conclusion_status",
        )
        object.__setattr__(
            self,
            "conclusion_status",
            cast(RCAConclusionStatus, conclusion_status),
        )
        object.__setattr__(
            self,
            "root_cause_candidate_id",
            _optional_non_empty(
                self.root_cause_candidate_id,
                "root_cause_candidate_id",
            ),
        )
        object.__setattr__(
            self,
            "root_cause",
            _optional_non_empty(self.root_cause, "root_cause"),
        )
        object.__setattr__(
            self,
            "confirmation_status",
            _non_empty(self.confirmation_status, "confirmation_status"),
        )
        object.__setattr__(
            self,
            "competition_status",
            _non_empty(self.competition_status, "competition_status"),
        )
        object.__setattr__(
            self,
            "terminal_reason",
            _non_empty(self.terminal_reason, "terminal_reason"),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _string_tuple(self.evidence_refs, "evidence_refs"),
        )
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))

        if self.conclusion_status == "supported":
            if self.root_cause_candidate_id is None:
                raise ModelValidationError(
                    "supported conclusion requires root_cause_candidate_id"
                )
            if self.root_cause is None:
                raise ModelValidationError("supported conclusion requires root_cause")
            if self.confirmation_status != "supported":
                raise ModelValidationError(
                    "supported conclusion requires confirmation_status=supported"
                )
            if self.competition_status not in _SUPPORTED_COMPETITION_STATUSES:
                expected = ", ".join(sorted(_SUPPORTED_COMPETITION_STATUSES))
                raise ModelValidationError(
                    "supported conclusion requires competition_status to be one of: "
                    f"{expected}"
                )
            if not self.evidence_refs:
                raise ModelValidationError("supported conclusion requires evidence_refs")
        elif self.root_cause_candidate_id is not None or self.root_cause is not None:
            raise ModelValidationError(
                "non-supported conclusion must not publish root_cause or "
                "root_cause_candidate_id"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "source_finding_id": self.source_finding_id,
            "source_hypothesis_id": self.source_hypothesis_id,
            "conclusion_status": self.conclusion_status,
            "root_cause_candidate_id": self.root_cause_candidate_id,
            "root_cause": self.root_cause,
            "confirmation_status": self.confirmation_status,
            "competition_status": self.competition_status,
            "terminal_reason": self.terminal_reason,
            "evidence_refs": list(self.evidence_refs),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        if not isinstance(data, Mapping):
            raise ModelValidationError("authoritative RCA result must be a JSON object")
        return cls(
            result_id=data["result_id"],
            source_finding_id=data.get("source_finding_id"),
            source_hypothesis_id=data.get("source_hypothesis_id"),
            conclusion_status=cast(RCAConclusionStatus, data["conclusion_status"]),
            root_cause_candidate_id=data.get("root_cause_candidate_id"),
            root_cause=data.get("root_cause"),
            confirmation_status=data["confirmation_status"],
            competition_status=data["competition_status"],
            terminal_reason=data["terminal_reason"],
            evidence_refs=_string_tuple(data.get("evidence_refs", ()), "evidence_refs"),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class ImpactPublicationResult:
    """Minimal publication decision for the impact scope of one RCA result."""

    rca_result_id: str
    publication_status: ImpactPublicationStatus
    confirmed_impact_lots: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rca_result_id",
            _non_empty(self.rca_result_id, "rca_result_id"),
        )
        publication_status = _status(
            self.publication_status,
            _IMPACT_PUBLICATION_STATUSES,
            "publication_status",
        )
        object.__setattr__(
            self,
            "publication_status",
            cast(ImpactPublicationStatus, publication_status),
        )
        object.__setattr__(
            self,
            "confirmed_impact_lots",
            _string_tuple(self.confirmed_impact_lots, "confirmed_impact_lots"),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _string_tuple(self.evidence_refs, "evidence_refs"),
        )
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))

        if self.publication_status == "confirmed":
            if not self.confirmed_impact_lots:
                raise ModelValidationError(
                    "confirmed publication requires confirmed_impact_lots"
                )
            if not self.evidence_refs:
                raise ModelValidationError("confirmed publication requires evidence_refs")
        elif self.confirmed_impact_lots:
            raise ModelValidationError(
                "non-confirmed publication must not contain confirmed lots"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rca_result_id": self.rca_result_id,
            "publication_status": self.publication_status,
            "confirmed_impact_lots": list(self.confirmed_impact_lots),
            "evidence_refs": list(self.evidence_refs),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        if not isinstance(data, Mapping):
            raise ModelValidationError("impact publication result must be a JSON object")
        return cls(
            rca_result_id=data["rca_result_id"],
            publication_status=cast(ImpactPublicationStatus, data["publication_status"]),
            confirmed_impact_lots=_string_tuple(
                data.get("confirmed_impact_lots", ()),
                "confirmed_impact_lots",
            ),
            evidence_refs=_string_tuple(data.get("evidence_refs", ()), "evidence_refs"),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


__all__ = [
    "AuthoritativeRCAResult",
    "ImpactPublicationResult",
    "ImpactPublicationStatus",
    "RCAConclusionStatus",
]
