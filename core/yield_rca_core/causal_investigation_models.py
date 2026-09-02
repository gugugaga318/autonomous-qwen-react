"""Python-owned models for adversarial causal investigation state.

``causal_scope.CausalLane`` is an existing enum describing broad search
directions.  This module deliberately uses ``CausalLaneRecord`` for one
concrete operation/equipment/chamber investigation path so the existing
scope contract remains backwards compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Self

from yield_rca_core.evidence_models import SCHEMA_VERSION, ModelValidationError


def _non_empty(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelValidationError(f"{field_name} must be a non-empty string")
    return value.strip()


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ModelValidationError(f"{field_name} must be a list or tuple")
    result = tuple(_non_empty(item, f"{field_name}[{index}]") for index, item in enumerate(value))
    if len(result) != len(set(result)):
        raise ModelValidationError(f"{field_name} must not contain duplicates")
    return result


def _optional_string(value: object, field_name: str) -> str | None:
    if value is None or value == "":
        return None
    return _non_empty(value, field_name)


def _time_window(value: object, field_name: str) -> tuple[str, ...]:
    result = _string_tuple(value, field_name)
    if len(result) not in {0, 2}:
        raise ModelValidationError(f"{field_name} must contain zero or two timestamps")
    if not result:
        return result
    try:
        start = datetime.fromisoformat(result[0].replace("Z", "+00:00"))
        end = datetime.fromisoformat(result[1].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ModelValidationError(f"{field_name} must contain ISO-8601 timestamps") from exc
    if start.tzinfo is None or end.tzinfo is None:
        raise ModelValidationError(f"{field_name} timestamps must include a timezone")
    if start > end:
        raise ModelValidationError(f"{field_name} start must not be after end")
    return result


class InvestigationLaneStatus(StrEnum):
    UNINVESTIGATED = "uninvestigated"
    IN_PROGRESS = "in_progress"
    EVIDENCE_COLLECTED = "evidence_collected"
    ELIMINATED = "eliminated"
    BLOCKED = "blocked"


class LaneLifecycleStatus(StrEnum):
    """Python-owned lifecycle of one factual causal Lane.

    This is deliberately separate from ``InvestigationLaneStatus``.  The
    latter describes whether Evidence work is complete, while this enum
    records inventory ownership and whether a Lane may enter the bounded
    active snapshot.
    """

    CREATED = "created"
    ACTIVE = "active"
    DEFERRED = "deferred"
    CHALLENGED = "challenged"
    ELIMINATED = "eliminated"
    MERGED = "merged"
    BLOCKED = "blocked"


class AlternativeSearchStatus(StrEnum):
    NOT_REQUIRED = "not_required"
    NOT_SEARCHED = "not_searched"
    IN_PROGRESS = "in_progress"
    ALTERNATIVE_FOUND = "alternative_found"
    ALTERNATIVES_ELIMINATED = "alternatives_eliminated"
    UNRESOLVED = "unresolved"
    BLOCKED_BY_MISSING_DATA = "blocked_by_missing_data"


class CompetitionRequirement(StrEnum):
    """Python-owned requirement for a causally meaningful candidate set."""

    NOT_EVALUATED = "not_evaluated"
    NOT_REQUIRED = "not_required"
    DIRECTION_REQUIRED = "direction_required"
    MECHANISM_REQUIRED = "mechanism_required"
    SCOPE_REQUIRED = "scope_required"
    MIXED_REQUIRED = "mixed_required"
    ALTERNATIVE_DISCOVERY_REQUIRED = "alternative_discovery_required"


class CandidateCompetitionStatus(StrEnum):
    """Lifecycle state independent from orchestration and RCA conclusion."""

    NOT_EVALUATED = "not_evaluated"
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETE_CONFIRMED = "complete_confirmed"
    COMPLETE_REJECTED = "complete_rejected"
    EXHAUSTED = "exhausted"
    RESOLVED = "resolved"
    FAILED = "failed"
    BLOCKED_BY_MISSING_DATA = "blocked_by_missing_data"
    BUDGET_EXHAUSTED = "budget_exhausted"


class CandidateCompetitionType(StrEnum):
    NOT_EVALUATED = "not_evaluated"
    NONE = "none"
    CAUSAL_DIRECTION = "causal_direction"
    MECHANISM = "mechanism"
    SCOPE = "scope"
    MIXED = "mixed"


class CandidateCompetitionAxis(StrEnum):
    """Independent uncertainty axes; Scope never substitutes for root cause."""

    DIRECTION = "direction"
    MECHANISM = "mechanism"
    SCOPE = "scope"


class ScopeAssessmentStatus(StrEnum):
    """Lifecycle for impact/causal reach assessment, separate from RCA competition."""

    NOT_EVALUATED = "not_evaluated"
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    ACTIVE = "active"
    RESOLVED = "resolved"
    BLOCKED_BY_MISSING_DATA = "blocked_by_missing_data"
    EXHAUSTED = "exhausted"


class CompetitionFailureReason(StrEnum):
    ALTERNATIVE_DIRECTION_NOT_GENERATED = "alternative_direction_not_generated"
    SCOPE_HYPOTHESIS_COLLAPSED = "scope_hypothesis_collapsed"
    SEMANTIC_PROFILE_INVALID = "semantic_profile_invalid"
    CANDIDATE_VALIDATION_EXHAUSTED = "candidate_validation_exhausted"
    CANDIDATE_PROVIDER_FAILED = "candidate_provider_failed"
    CHALLENGE_OUTPUT_INVALID = "challenge_output_invalid"
    NO_DISCRIMINATIVE_ACTION = "no_discriminative_action"


class CompetitionGapReason(StrEnum):
    """Unresolved competition requirement, distinct from processing failure.

    The overlapping values intentionally support old serialized State, where
    these conditions were stored in ``competition_failure_reason``.  New State
    writes use ``competition_gap_reason`` instead.
    """

    ALTERNATIVE_DIRECTION_NOT_GENERATED = "alternative_direction_not_generated"
    MECHANISM_ALTERNATIVE_NOT_GENERATED = "mechanism_alternative_not_generated"
    SCOPE_HYPOTHESIS_COLLAPSED = "scope_hypothesis_collapsed"
    SEMANTIC_PROFILE_INVALID = "semantic_profile_invalid"
    NO_DISCRIMINATIVE_ACTION = "no_discriminative_action"


class CausalChainCompleteness(StrEnum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    CONFLICTING = "conflicting"


class ChallengeStatus(StrEnum):
    OPEN = "open"
    ALTERNATIVE_IDENTIFIED = "alternative_identified"
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    BLOCKED = "blocked"
    NON_DISCRIMINATIVE = "non_discriminative"


class ChallengeKind(StrEnum):
    CANDIDATE_DIRECTION = "candidate_direction"
    MECHANISM = "mechanism"
    SCOPE = "scope"
    LANE_PROBE = "lane_probe"


class CandidateScopeRelation(StrEnum):
    """Qwen-declared causal scope relation, never inferred from Evidence coverage."""

    SHARED_EFFECT = "shared_effect"
    FOCAL_ONLY = "focal_only"
    DIFFERENTIAL_SENSITIVITY = "differential_sensitivity"
    UNRESOLVED = "unresolved"


class CandidateClaimedScopeKind(StrEnum):
    """Granularity of the causal reach explicitly declared by Qwen.

    ``claimed_lane_ids`` remain useful references into the current bounded
    investigation snapshot, but they are not an exhaustive enumeration for a
    broader equipment/chamber/operation claim.
    """

    LANE = "lane"
    RECIPE = "recipe"
    CHAMBER = "chamber"
    EQUIPMENT = "equipment"
    OPERATION = "operation"
    UNRESOLVED = "unresolved"


class CandidateMechanismRelation(StrEnum):
    """Qwen-declared relationship to the first Candidate's primary mechanism.

    The relation is semantic metadata, not Evidence.  Python uses it only to
    decide whether two model-authored explanations form a root-cause
    competition.  Scope and sensitivity modifiers remain auditable without
    occupying an independent Root Cause Candidate slot.
    """

    REFERENCE = "reference"
    INDEPENDENT_ALTERNATIVE = "independent_alternative"
    SHARED_PRIMARY_WITH_MODIFIER = "shared_primary_with_modifier"
    NESTED = "nested"
    SCOPE_VARIANT = "scope_variant"
    UNKNOWN = "unknown"


_DISTINGUISHING_PREDICTION_KINDS = {
    "parameter_anomaly",
    "exposure_commonality",
    "recipe_commonality",
    "product_outcome",
    "temporal_alignment",
    "mechanism_context",
}


class CandidateLaneEffectState(StrEnum):
    """Qwen-declared effect polarity for one Lane-bound prediction."""

    PRESENT = "present"
    ABSENT = "absent"
    INCREASED = "increased"
    DECREASED = "decreased"
    UNCHANGED = "unchanged"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class CandidateLaneEffectExpectation:
    """One structured Lane/effect assertion authored by Qwen."""

    lane_id: str
    effect_key: str
    effect_state: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "lane_id", _non_empty(self.lane_id, "lane_id"))
        object.__setattr__(
            self,
            "effect_key",
            _non_empty(self.effect_key, "effect_key"),
        )
        if not self.normalized_effect_key:
            raise ModelValidationError(
                "effect_key must contain at least one alphanumeric token"
            )
        try:
            effect_state = CandidateLaneEffectState(
                _non_empty(self.effect_state, "effect_state").casefold()
            ).value
        except ValueError as exc:
            allowed = ", ".join(item.value for item in CandidateLaneEffectState)
            raise ModelValidationError(
                f"effect_state must be one of: {allowed}"
            ) from exc
        object.__setattr__(self, "effect_state", effect_state)
        if self.schema_version != SCHEMA_VERSION:
            raise ModelValidationError(
                f"unsupported schema_version {self.schema_version!r}; "
                f"expected {SCHEMA_VERSION!r}"
            )

    @property
    def normalized_effect_key(self) -> str:
        """Return a comparison key without changing the serialized Qwen text."""

        return "".join(
            character
            for character in self.effect_key.casefold()
            if character.isalnum()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane_id": self.lane_id,
            "effect_key": self.effect_key,
            "effect_state": self.effect_state,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            lane_id=data["lane_id"],
            effect_key=data["effect_key"],
            effect_state=data["effect_state"],
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class CandidateDistinguishingPrediction:
    """One falsifiable Qwen claim bound to Python-owned Lane and Gap kinds."""

    discriminator_kind: str
    lane_ids: tuple[str, ...]
    prediction: str
    schema_version: str = SCHEMA_VERSION
    lane_effect_expectations: tuple[CandidateLaneEffectExpectation, ...] = ()

    def __post_init__(self) -> None:
        kind = _non_empty(self.discriminator_kind, "discriminator_kind")
        if kind not in _DISTINGUISHING_PREDICTION_KINDS:
            allowed = ", ".join(sorted(_DISTINGUISHING_PREDICTION_KINDS))
            raise ModelValidationError(
                f"discriminator_kind must be one of: {allowed}"
            )
        object.__setattr__(self, "discriminator_kind", kind)
        object.__setattr__(self, "lane_ids", _string_tuple(self.lane_ids, "lane_ids"))
        if not self.lane_ids:
            raise ModelValidationError("prediction lane_ids must not be empty")
        object.__setattr__(self, "prediction", _non_empty(self.prediction, "prediction"))
        if not isinstance(self.lane_effect_expectations, (list, tuple)) or any(
            not isinstance(item, CandidateLaneEffectExpectation)
            for item in self.lane_effect_expectations
        ):
            raise ModelValidationError(
                "lane_effect_expectations must contain "
                "CandidateLaneEffectExpectation instances"
            )
        expectations = tuple(self.lane_effect_expectations)
        object.__setattr__(self, "lane_effect_expectations", expectations)
        lane_ids = set(self.lane_ids)
        duplicate_keys: set[tuple[str, str]] = set()
        expectation_keys: set[tuple[str, str]] = set()
        for expectation in expectations:
            if expectation.lane_id not in lane_ids:
                raise ModelValidationError(
                    "lane effect expectation lane_id must be within prediction "
                    "lane_ids"
                )
            key = (expectation.lane_id, expectation.normalized_effect_key)
            if key in expectation_keys:
                duplicate_keys.add(key)
            expectation_keys.add(key)
        if duplicate_keys:
            raise ModelValidationError(
                "lane_effect_expectations must not repeat a Lane/effect key"
            )
        if self.schema_version != SCHEMA_VERSION:
            raise ModelValidationError(
                f"unsupported schema_version {self.schema_version!r}; "
                f"expected {SCHEMA_VERSION!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "discriminator_kind": self.discriminator_kind,
            "lane_ids": list(self.lane_ids),
            "prediction": self.prediction,
            "lane_effect_expectations": [
                item.to_dict() for item in self.lane_effect_expectations
            ],
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            discriminator_kind=data["discriminator_kind"],
            lane_ids=tuple(data.get("lane_ids", [])),
            prediction=data["prediction"],
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            lane_effect_expectations=tuple(
                CandidateLaneEffectExpectation.from_dict(item)
                for item in data.get("lane_effect_expectations", [])
            ),
        )


@dataclass(frozen=True)
class CandidateSemanticProfile:
    """Qwen-owned Candidate meaning with Python-validated factual references.

    Evidence coverage is deliberately absent. Python derives coverage from the
    Candidate's cited typed Evidence and must never use it to rewrite these
    declared scope semantics.
    """

    candidate_id: str
    scope_relation: str
    claimed_lane_ids: tuple[str, ...]
    comparison_lane_ids: tuple[str, ...]
    mechanism_claim: str
    claimed_scope_kind: str = CandidateClaimedScopeKind.LANE.value
    claimed_operation: str | None = None
    claimed_equipment: str | None = None
    claimed_chamber: str | None = None
    claimed_recipe: str | None = None
    primary_mechanism: str = ""
    effect_modifier: str | None = None
    depends_on_candidate_id: str | None = None
    mechanism_relation: str = CandidateMechanismRelation.UNKNOWN.value
    distinguishing_predictions: tuple[CandidateDistinguishingPrediction, ...] = ()
    source: str = "qwen"
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _non_empty(self.candidate_id, "candidate_id"))
        try:
            relation = CandidateScopeRelation(self.scope_relation).value
        except ValueError as exc:
            allowed = ", ".join(item.value for item in CandidateScopeRelation)
            raise ModelValidationError(
                f"scope_relation must be one of: {allowed}"
            ) from exc
        object.__setattr__(self, "scope_relation", relation)
        try:
            scope_kind = CandidateClaimedScopeKind(self.claimed_scope_kind).value
        except ValueError as exc:
            allowed = ", ".join(item.value for item in CandidateClaimedScopeKind)
            raise ModelValidationError(
                f"claimed_scope_kind must be one of: {allowed}"
            ) from exc
        object.__setattr__(self, "claimed_scope_kind", scope_kind)
        for field_name in (
            "claimed_operation",
            "claimed_equipment",
            "claimed_chamber",
            "claimed_recipe",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_string(getattr(self, field_name), field_name),
            )
        for field_name in ("claimed_lane_ids", "comparison_lane_ids"):
            object.__setattr__(
                self,
                field_name,
                _string_tuple(getattr(self, field_name), field_name),
            )
        resolved = relation != CandidateScopeRelation.UNRESOLVED.value
        required_identity_fields = {
            CandidateClaimedScopeKind.LANE.value: (),
            CandidateClaimedScopeKind.RECIPE.value: (
                "claimed_operation",
                "claimed_equipment",
                "claimed_chamber",
                "claimed_recipe",
            ),
            CandidateClaimedScopeKind.CHAMBER.value: (
                "claimed_operation",
                "claimed_equipment",
                "claimed_chamber",
            ),
            CandidateClaimedScopeKind.EQUIPMENT.value: (
                "claimed_operation",
                "claimed_equipment",
            ),
            CandidateClaimedScopeKind.OPERATION.value: ("claimed_operation",),
            CandidateClaimedScopeKind.UNRESOLVED.value: (),
        }[scope_kind]
        if resolved and scope_kind == CandidateClaimedScopeKind.UNRESOLVED.value:
            raise ModelValidationError(
                "a resolved scope_relation cannot use claimed_scope_kind=unresolved"
            )
        if resolved and scope_kind == CandidateClaimedScopeKind.LANE.value and not (
            self.claimed_lane_ids
        ):
            raise ModelValidationError(
                "a resolved lane scope requires claimed_lane_ids"
            )
        missing_identity_fields = [
            field_name
            for field_name in required_identity_fields
            if getattr(self, field_name) is None
        ]
        if resolved and missing_identity_fields:
            raise ModelValidationError(
                "resolved claimed scope is missing identity fields: "
                + ", ".join(missing_identity_fields)
            )
        if not set(self.claimed_lane_ids) <= set(self.comparison_lane_ids):
            raise ModelValidationError(
                "comparison_lane_ids must include every claimed_lane_id"
            )
        object.__setattr__(
            self,
            "mechanism_claim",
            _non_empty(self.mechanism_claim, "mechanism_claim"),
        )
        object.__setattr__(
            self,
            "primary_mechanism",
            _non_empty(
                self.primary_mechanism or self.mechanism_claim,
                "primary_mechanism",
            ),
        )
        object.__setattr__(
            self,
            "effect_modifier",
            _optional_string(self.effect_modifier, "effect_modifier"),
        )
        object.__setattr__(
            self,
            "depends_on_candidate_id",
            _optional_string(
                self.depends_on_candidate_id,
                "depends_on_candidate_id",
            ),
        )
        try:
            mechanism_relation = CandidateMechanismRelation(
                self.mechanism_relation
            ).value
        except ValueError as exc:
            allowed = ", ".join(item.value for item in CandidateMechanismRelation)
            raise ModelValidationError(
                f"mechanism_relation must be one of: {allowed}"
            ) from exc
        object.__setattr__(self, "mechanism_relation", mechanism_relation)
        if (
            mechanism_relation
            in {
                CandidateMechanismRelation.REFERENCE.value,
                CandidateMechanismRelation.INDEPENDENT_ALTERNATIVE.value,
            }
            and self.depends_on_candidate_id is not None
        ):
            raise ModelValidationError(
                "reference and independent mechanisms cannot depend on another "
                "candidate"
            )
        if (
            mechanism_relation
            in {
                CandidateMechanismRelation.SHARED_PRIMARY_WITH_MODIFIER.value,
                CandidateMechanismRelation.NESTED.value,
                CandidateMechanismRelation.SCOPE_VARIANT.value,
            }
            and self.depends_on_candidate_id is None
        ):
            raise ModelValidationError(
                "modifier, nested, and scope-variant mechanisms require "
                "depends_on_candidate_id"
            )
        if not isinstance(self.distinguishing_predictions, (list, tuple)) or any(
            not isinstance(item, CandidateDistinguishingPrediction)
            for item in self.distinguishing_predictions
        ):
            raise ModelValidationError(
                "distinguishing_predictions must contain "
                "CandidateDistinguishingPrediction instances"
            )
        comparison = set(self.comparison_lane_ids)
        for prediction in self.distinguishing_predictions:
            if not set(prediction.lane_ids) <= comparison:
                raise ModelValidationError(
                    "prediction lane_ids must be within comparison_lane_ids"
                )
        if self.source != "qwen":
            raise ModelValidationError("candidate semantic profile source must be qwen")
        if self.schema_version != SCHEMA_VERSION:
            raise ModelValidationError(
                f"unsupported schema_version {self.schema_version!r}; "
                f"expected {SCHEMA_VERSION!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "scope_relation": self.scope_relation,
            "claimed_scope_kind": self.claimed_scope_kind,
            "claimed_operation": self.claimed_operation,
            "claimed_equipment": self.claimed_equipment,
            "claimed_chamber": self.claimed_chamber,
            "claimed_recipe": self.claimed_recipe,
            "claimed_lane_ids": list(self.claimed_lane_ids),
            "comparison_lane_ids": list(self.comparison_lane_ids),
            "mechanism_claim": self.mechanism_claim,
            "primary_mechanism": self.primary_mechanism,
            "effect_modifier": self.effect_modifier,
            "depends_on_candidate_id": self.depends_on_candidate_id,
            "mechanism_relation": self.mechanism_relation,
            "distinguishing_predictions": [
                item.to_dict() for item in self.distinguishing_predictions
            ],
            "source": self.source,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            candidate_id=data["candidate_id"],
            scope_relation=data["scope_relation"],
            claimed_scope_kind=data.get(
                "claimed_scope_kind",
                CandidateClaimedScopeKind.LANE.value,
            ),
            claimed_operation=data.get("claimed_operation"),
            claimed_equipment=data.get("claimed_equipment"),
            claimed_chamber=data.get("claimed_chamber"),
            claimed_recipe=data.get("claimed_recipe"),
            claimed_lane_ids=tuple(data.get("claimed_lane_ids", [])),
            comparison_lane_ids=tuple(data.get("comparison_lane_ids", [])),
            mechanism_claim=data["mechanism_claim"],
            primary_mechanism=data.get(
                "primary_mechanism",
                data.get("mechanism_claim", ""),
            ),
            effect_modifier=data.get("effect_modifier"),
            depends_on_candidate_id=data.get("depends_on_candidate_id"),
            mechanism_relation=data.get(
                "mechanism_relation",
                CandidateMechanismRelation.UNKNOWN.value,
            ),
            distinguishing_predictions=tuple(
                CandidateDistinguishingPrediction.from_dict(item)
                for item in data.get("distinguishing_predictions", [])
            ),
            source=data.get("source", "qwen"),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


class AlternativeLaneResolutionStatus(StrEnum):
    """Python-owned outcome for one concrete alternative causal Lane."""

    RETAINED = "retained"
    ELIMINATED = "eliminated"
    UNRESOLVED = "unresolved"
    BLOCKED = "blocked"
    NON_DISCRIMINATIVE = "non_discriminative"


class InvestigationGainType(StrEnum):
    """Python-owned classification of the result of one completed Action."""

    EVIDENCE_GAIN = "evidence_gain"
    STATE_GAIN = "state_gain"
    NO_GAIN = "no_gain"


class InvestigationGainReasonCode(StrEnum):
    """Structured reason for one ex-post investigation result.

    Human-readable ``reason`` remains an audit explanation. Runtime policy must
    use this code instead of parsing that prose.
    """

    SUPPORTING_OR_CONTRADICTING_EVIDENCE = (
        "supporting_or_contradicting_evidence"
    )
    REASONING_STATE_CHANGED = "reasoning_state_changed"
    UNAVAILABLE_SOURCE = "unavailable_source"
    NON_DISCRIMINATIVE = "non_discriminative"
    REPEATED_UNAVAILABLE_SOURCE = "repeated_unavailable_source"
    REPEATED_NON_DISCRIMINATIVE = "repeated_non_discriminative"
    NO_DECISION_CHANGE = "no_decision_change"


def _legacy_investigation_gain_reason_code(data: dict[str, Any]) -> str | None:
    """Recover the structured code for Patch-3 prerelease State payloads."""

    raw = data.get("reason_code")
    if raw is not None:
        return str(raw)
    reason = str(data.get("reason", "")).casefold()
    if "repeated unavailable_source" in reason:
        return InvestigationGainReasonCode.REPEATED_UNAVAILABLE_SOURCE.value
    if "repeated non_discriminative" in reason:
        return InvestigationGainReasonCode.REPEATED_NON_DISCRIMINATIVE.value
    if "unavailable_source" in reason:
        return InvestigationGainReasonCode.UNAVAILABLE_SOURCE.value
    if "non_discriminative" in reason:
        return InvestigationGainReasonCode.NON_DISCRIMINATIVE.value
    if "supporting or contradicting" in reason:
        return (
            InvestigationGainReasonCode.SUPPORTING_OR_CONTRADICTING_EVIDENCE.value
        )
    if "reasoning changed candidate" in reason.casefold():
        return InvestigationGainReasonCode.REASONING_STATE_CHANGED.value
    if reason:
        return InvestigationGainReasonCode.NO_DECISION_CHANGE.value
    return None


class ActionSourceAvailability(StrEnum):
    UNKNOWN = "unknown"
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class ActionDecisionImpact(StrEnum):
    RANKING = "ranking"
    CONFIRMATION = "confirmation"
    QUESTION_CLOSURE = "question_closure"
    REASONING_REFRESH = "reasoning_refresh"
    CONTEXT_ONLY = "context_only"


class ActionValueTier(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class CandidateResolutionStatus(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class InvestigationGainRecord:
    """Auditable ex-post progress produced by one completed Action."""

    action_id: str
    action_kind: str
    scope_fingerprint: str
    gain_type: str
    evidence_ids: tuple[str, ...] = ()
    relation_types: tuple[str, ...] = ()
    candidate_id: str | None = None
    lane_id: str | None = None
    gap_id: str | None = None
    discriminator_kind: str | None = None
    source: str | None = None
    reason_code: str | None = None
    reason: str = ""
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in ("action_id", "action_kind", "scope_fingerprint"):
            object.__setattr__(
                self,
                field_name,
                _non_empty(getattr(self, field_name), field_name),
            )
        try:
            gain_type = InvestigationGainType(self.gain_type).value
        except ValueError as exc:
            allowed = ", ".join(item.value for item in InvestigationGainType)
            raise ModelValidationError(
                f"gain_type must be one of: {allowed}"
            ) from exc
        object.__setattr__(self, "gain_type", gain_type)
        for field_name in ("evidence_ids", "relation_types"):
            object.__setattr__(
                self,
                field_name,
                _string_tuple(getattr(self, field_name), field_name),
            )
        for field_name in (
            "candidate_id",
            "lane_id",
            "gap_id",
            "discriminator_kind",
            "source",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_string(getattr(self, field_name), field_name),
            )
        if self.reason_code is not None:
            try:
                reason_code = InvestigationGainReasonCode(self.reason_code).value
            except ValueError as exc:
                allowed = ", ".join(item.value for item in InvestigationGainReasonCode)
                raise ModelValidationError(
                    f"reason_code must be one of: {allowed}"
                ) from exc
            object.__setattr__(self, "reason_code", reason_code)
        if not isinstance(self.reason, str):
            raise ModelValidationError("reason must be a string")
        object.__setattr__(self, "reason", self.reason.strip())
        if self.schema_version != SCHEMA_VERSION:
            raise ModelValidationError(
                f"unsupported schema_version {self.schema_version!r}; "
                f"expected {SCHEMA_VERSION!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_kind": self.action_kind,
            "scope_fingerprint": self.scope_fingerprint,
            "gain_type": self.gain_type,
            "evidence_ids": list(self.evidence_ids),
            "relation_types": list(self.relation_types),
            "candidate_id": self.candidate_id,
            "lane_id": self.lane_id,
            "gap_id": self.gap_id,
            "discriminator_kind": self.discriminator_kind,
            "source": self.source,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            action_id=data["action_id"],
            action_kind=data["action_kind"],
            scope_fingerprint=data["scope_fingerprint"],
            gain_type=data["gain_type"],
            evidence_ids=tuple(data.get("evidence_ids", [])),
            relation_types=tuple(data.get("relation_types", [])),
            candidate_id=data.get("candidate_id"),
            lane_id=data.get("lane_id"),
            gap_id=data.get("gap_id"),
            discriminator_kind=data.get("discriminator_kind"),
            source=data.get("source"),
            reason_code=_legacy_investigation_gain_reason_code(data),
            reason=data.get("reason", ""),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class ActionValueAssessment:
    """Python-owned ex-ante value of one exact investigation Action option."""

    option_id: str
    action_kind: str
    scope_fingerprint: str
    static_information_gain: float
    source_availability: str
    decision_impact: str
    estimated_tool_cost: int
    estimated_llm_cost: int
    remaining_tool_budget: int
    high_value: bool
    eligible: bool
    value_tier: str
    candidate_id: str | None = None
    lane_id: str | None = None
    gap_id: str | None = None
    discriminator_kind: str | None = None
    previous_attempt_result: str | None = None
    rejection_reason: str | None = None
    can_change_ranking: bool = False
    supports_candidate_ids: tuple[str, ...] = ()
    weakens_candidate_ids: tuple[str, ...] = ()
    expected_evidence_types: tuple[str, ...] = ()
    already_available_evidence_ids: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in ("option_id", "action_kind", "scope_fingerprint"):
            object.__setattr__(
                self,
                field_name,
                _non_empty(getattr(self, field_name), field_name),
            )
        if not isinstance(self.static_information_gain, int | float) or not 0 <= float(
            self.static_information_gain
        ) <= 1:
            raise ModelValidationError(
                "static_information_gain must be between 0 and 1"
            )
        object.__setattr__(
            self,
            "static_information_gain",
            float(self.static_information_gain),
        )
        for enum_type, field_name in (
            (ActionSourceAvailability, "source_availability"),
            (ActionDecisionImpact, "decision_impact"),
            (ActionValueTier, "value_tier"),
        ):
            try:
                normalized = enum_type(getattr(self, field_name)).value
            except ValueError as exc:
                allowed = ", ".join(item.value for item in enum_type)
                raise ModelValidationError(
                    f"{field_name} must be one of: {allowed}"
                ) from exc
            object.__setattr__(self, field_name, normalized)
        for field_name in (
            "estimated_tool_cost",
            "estimated_llm_cost",
            "remaining_tool_budget",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ModelValidationError(
                    f"{field_name} must be a non-negative integer"
                )
        for field_name in ("high_value", "eligible"):
            if not isinstance(getattr(self, field_name), bool):
                raise ModelValidationError(f"{field_name} must be a bool")
        if not isinstance(self.can_change_ranking, bool):
            raise ModelValidationError("can_change_ranking must be a bool")
        for field_name in (
            "supports_candidate_ids",
            "weakens_candidate_ids",
            "expected_evidence_types",
            "already_available_evidence_ids",
        ):
            object.__setattr__(
                self,
                field_name,
                _string_tuple(getattr(self, field_name), field_name),
            )
        for field_name in (
            "candidate_id",
            "lane_id",
            "gap_id",
            "discriminator_kind",
            "previous_attempt_result",
            "rejection_reason",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_string(getattr(self, field_name), field_name),
            )
        if self.high_value and not self.eligible:
            raise ModelValidationError("high_value actions must be eligible")
        if not self.eligible and self.rejection_reason is None:
            raise ModelValidationError(
                "ineligible Action Value assessments require rejection_reason"
            )
        if self.schema_version != SCHEMA_VERSION:
            raise ModelValidationError(
                f"unsupported schema_version {self.schema_version!r}; "
                f"expected {SCHEMA_VERSION!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "option_id": self.option_id,
            "action_kind": self.action_kind,
            "scope_fingerprint": self.scope_fingerprint,
            "static_information_gain": self.static_information_gain,
            "source_availability": self.source_availability,
            "decision_impact": self.decision_impact,
            "estimated_tool_cost": self.estimated_tool_cost,
            "estimated_llm_cost": self.estimated_llm_cost,
            "remaining_tool_budget": self.remaining_tool_budget,
            "high_value": self.high_value,
            "eligible": self.eligible,
            "value_tier": self.value_tier,
            "candidate_id": self.candidate_id,
            "lane_id": self.lane_id,
            "gap_id": self.gap_id,
            "discriminator_kind": self.discriminator_kind,
            "previous_attempt_result": self.previous_attempt_result,
            "rejection_reason": self.rejection_reason,
            "can_change_ranking": self.can_change_ranking,
            "supports_candidate_ids": list(self.supports_candidate_ids),
            "weakens_candidate_ids": list(self.weakens_candidate_ids),
            "expected_evidence_types": list(self.expected_evidence_types),
            "already_available_evidence_ids": list(
                self.already_available_evidence_ids
            ),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            option_id=data["option_id"],
            action_kind=data["action_kind"],
            scope_fingerprint=data["scope_fingerprint"],
            static_information_gain=data.get("static_information_gain", 0.0),
            source_availability=data.get(
                "source_availability",
                ActionSourceAvailability.UNKNOWN.value,
            ),
            decision_impact=data.get(
                "decision_impact",
                ActionDecisionImpact.CONTEXT_ONLY.value,
            ),
            estimated_tool_cost=data.get("estimated_tool_cost", 0),
            estimated_llm_cost=data.get("estimated_llm_cost", 0),
            remaining_tool_budget=data.get("remaining_tool_budget", 0),
            high_value=data.get("high_value", False),
            eligible=data.get("eligible", False),
            value_tier=data.get("value_tier", ActionValueTier.NONE.value),
            candidate_id=data.get("candidate_id"),
            lane_id=data.get("lane_id"),
            gap_id=data.get("gap_id"),
            discriminator_kind=data.get("discriminator_kind"),
            previous_attempt_result=data.get("previous_attempt_result"),
            rejection_reason=data.get("rejection_reason"),
            can_change_ranking=data.get("can_change_ranking", False),
            supports_candidate_ids=tuple(data.get("supports_candidate_ids", [])),
            weakens_candidate_ids=tuple(data.get("weakens_candidate_ids", [])),
            expected_evidence_types=tuple(data.get("expected_evidence_types", [])),
            already_available_evidence_ids=tuple(
                data.get("already_available_evidence_ids", [])
            ),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class CandidateResolution:
    """Python-owned resolution of one candidate causal claim."""

    candidate_id: str
    status: str
    reason_codes: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    unresolved_gap_ids: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            _non_empty(self.candidate_id, "candidate_id"),
        )
        try:
            status = CandidateResolutionStatus(self.status).value
        except ValueError as exc:
            allowed = ", ".join(item.value for item in CandidateResolutionStatus)
            raise ModelValidationError(
                f"candidate resolution status must be one of: {allowed}"
            ) from exc
        object.__setattr__(self, "status", status)
        for field_name in ("reason_codes", "evidence_ids", "unresolved_gap_ids"):
            object.__setattr__(
                self,
                field_name,
                _string_tuple(getattr(self, field_name), field_name),
            )
        if self.schema_version != SCHEMA_VERSION:
            raise ModelValidationError(
                f"unsupported schema_version {self.schema_version!r}; "
                f"expected {SCHEMA_VERSION!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "evidence_ids": list(self.evidence_ids),
            "unresolved_gap_ids": list(self.unresolved_gap_ids),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            candidate_id=data["candidate_id"],
            status=data["status"],
            reason_codes=tuple(data.get("reason_codes", [])),
            evidence_ids=tuple(data.get("evidence_ids", [])),
            unresolved_gap_ids=tuple(data.get("unresolved_gap_ids", [])),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class LaneLifecycleTransition:
    """One auditable Python-owned Lane lifecycle transition."""

    sequence: int
    lane_id: str
    from_status: str | None
    to_status: str
    reason: str
    evidence_ids: tuple[str, ...] = ()
    related_lane_id: str | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.sequence, int) or isinstance(self.sequence, bool):
            raise ModelValidationError("lane lifecycle sequence must be an integer")
        if self.sequence < 1:
            raise ModelValidationError("lane lifecycle sequence must be at least 1")
        object.__setattr__(self, "lane_id", _non_empty(self.lane_id, "lane_id"))
        if self.from_status is not None:
            try:
                from_status = LaneLifecycleStatus(self.from_status).value
            except ValueError as exc:
                raise ModelValidationError("invalid lane lifecycle from_status") from exc
            object.__setattr__(self, "from_status", from_status)
        try:
            to_status = LaneLifecycleStatus(self.to_status).value
        except ValueError as exc:
            raise ModelValidationError("invalid lane lifecycle to_status") from exc
        object.__setattr__(self, "to_status", to_status)
        object.__setattr__(self, "reason", _non_empty(self.reason, "reason"))
        object.__setattr__(
            self,
            "evidence_ids",
            _string_tuple(self.evidence_ids, "evidence_ids"),
        )
        if self.related_lane_id is not None:
            object.__setattr__(
                self,
                "related_lane_id",
                _non_empty(self.related_lane_id, "related_lane_id"),
            )
        if to_status == LaneLifecycleStatus.MERGED.value and self.related_lane_id is None:
            raise ModelValidationError(
                "merged lane lifecycle transitions require related_lane_id"
            )
        if self.schema_version != SCHEMA_VERSION:
            raise ModelValidationError(
                f"unsupported schema_version {self.schema_version!r}; "
                f"expected {SCHEMA_VERSION!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "lane_id": self.lane_id,
            "from_status": self.from_status,
            "to_status": self.to_status,
            "reason": self.reason,
            "evidence_ids": list(self.evidence_ids),
            "related_lane_id": self.related_lane_id,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            sequence=data["sequence"],
            lane_id=data["lane_id"],
            from_status=data.get("from_status"),
            to_status=data["to_status"],
            reason=data["reason"],
            evidence_ids=tuple(data.get("evidence_ids", [])),
            related_lane_id=data.get("related_lane_id"),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class CausalLaneRecord:
    """One concrete, auditable causal investigation path."""

    lane_id: str
    operation: str = ""
    operation_name: str = ""
    module: str = ""
    equipment: str = ""
    chamber: str = ""
    recipe: str = ""
    parameter_scope: tuple[str, ...] = ()
    exposed_lot_ids: tuple[str, ...] = ()
    time_window: tuple[str, ...] = ()
    initial_evidence_ids: tuple[str, ...] = ()
    priority_score: float = 0.0
    investigation_status: str = InvestigationLaneStatus.UNINVESTIGATED.value
    pruned_reason: str | None = None
    lifecycle_status: str = ""
    merged_from_lane_ids: tuple[str, ...] = ()
    merged_into_lane_id: str | None = None
    discovery_count: int = 1
    reactivation_count: int = 0
    last_transition_reason: str | None = None
    lifecycle_history: tuple[LaneLifecycleTransition, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _non_empty(self.lane_id, "lane_id")
        for field_name in (
            "operation",
            "operation_name",
            "module",
            "equipment",
            "chamber",
            "recipe",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise ModelValidationError(f"{field_name} must be a string")
            object.__setattr__(self, field_name, value.strip())
        object.__setattr__(
            self,
            "parameter_scope",
            _string_tuple(self.parameter_scope, "parameter_scope"),
        )
        object.__setattr__(
            self,
            "exposed_lot_ids",
            _string_tuple(self.exposed_lot_ids, "exposed_lot_ids"),
        )
        object.__setattr__(
            self,
            "initial_evidence_ids",
            _string_tuple(self.initial_evidence_ids, "initial_evidence_ids"),
        )
        object.__setattr__(self, "time_window", _time_window(self.time_window, "time_window"))
        try:
            status = InvestigationLaneStatus(self.investigation_status).value
        except ValueError as exc:
            allowed = ", ".join(item.value for item in InvestigationLaneStatus)
            raise ModelValidationError(
                f"investigation_status must be one of: {allowed}"
            ) from exc
        object.__setattr__(self, "investigation_status", status)
        raw_lifecycle_status = self.lifecycle_status
        if not raw_lifecycle_status:
            raw_lifecycle_status = {
                InvestigationLaneStatus.ELIMINATED.value: (
                    LaneLifecycleStatus.ELIMINATED.value
                ),
                InvestigationLaneStatus.BLOCKED.value: LaneLifecycleStatus.BLOCKED.value,
                InvestigationLaneStatus.IN_PROGRESS.value: LaneLifecycleStatus.ACTIVE.value,
                InvestigationLaneStatus.EVIDENCE_COLLECTED.value: (
                    LaneLifecycleStatus.ACTIVE.value
                ),
            }.get(status, LaneLifecycleStatus.CREATED.value)
        try:
            lifecycle_status = LaneLifecycleStatus(raw_lifecycle_status).value
        except ValueError as exc:
            allowed = ", ".join(item.value for item in LaneLifecycleStatus)
            raise ModelValidationError(
                f"lifecycle_status must be one of: {allowed}"
            ) from exc
        object.__setattr__(self, "lifecycle_status", lifecycle_status)
        if not isinstance(self.priority_score, int | float) or not 0 <= float(
            self.priority_score
        ) <= 1:
            raise ModelValidationError("priority_score must be between 0 and 1")
        object.__setattr__(self, "priority_score", float(self.priority_score))
        if self.pruned_reason is not None:
            object.__setattr__(
                self,
                "pruned_reason",
                _non_empty(self.pruned_reason, "pruned_reason"),
            )
        if status in {
            InvestigationLaneStatus.ELIMINATED.value,
            InvestigationLaneStatus.BLOCKED.value,
        } and self.pruned_reason is None:
            raise ModelValidationError(
                "eliminated or blocked lanes require pruned_reason"
            )
        for field_name in ("merged_from_lane_ids",):
            object.__setattr__(
                self,
                field_name,
                _string_tuple(getattr(self, field_name), field_name),
            )
        if self.lane_id in set(self.merged_from_lane_ids):
            raise ModelValidationError("merged_from_lane_ids must not contain lane_id")
        if self.merged_into_lane_id is not None:
            object.__setattr__(
                self,
                "merged_into_lane_id",
                _non_empty(self.merged_into_lane_id, "merged_into_lane_id"),
            )
        if lifecycle_status == LaneLifecycleStatus.MERGED.value:
            if self.merged_into_lane_id is None:
                raise ModelValidationError("merged lanes require merged_into_lane_id")
        elif self.merged_into_lane_id is not None:
            raise ModelValidationError(
                "merged_into_lane_id is valid only when lifecycle_status=merged"
            )
        if status == InvestigationLaneStatus.ELIMINATED.value and lifecycle_status != (
            LaneLifecycleStatus.ELIMINATED.value
        ):
            raise ModelValidationError(
                "eliminated investigation_status requires eliminated lifecycle_status"
            )
        if status == InvestigationLaneStatus.BLOCKED.value and lifecycle_status != (
            LaneLifecycleStatus.BLOCKED.value
        ):
            raise ModelValidationError(
                "blocked investigation_status requires blocked lifecycle_status"
            )
        for field_name in ("discovery_count", "reactivation_count"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ModelValidationError(f"{field_name} must be a non-negative integer")
        if self.discovery_count < 1:
            raise ModelValidationError("discovery_count must be at least 1")
        if self.reactivation_count > self.discovery_count:
            raise ModelValidationError(
                "reactivation_count must not exceed discovery_count"
            )
        if self.last_transition_reason is not None:
            object.__setattr__(
                self,
                "last_transition_reason",
                _non_empty(self.last_transition_reason, "last_transition_reason"),
            )
        if not isinstance(self.lifecycle_history, (list, tuple)) or any(
            not isinstance(item, LaneLifecycleTransition)
            for item in self.lifecycle_history
        ):
            raise ModelValidationError(
                "lifecycle_history must contain LaneLifecycleTransition instances"
            )
        sequence_values = [item.sequence for item in self.lifecycle_history]
        if sequence_values != sorted(sequence_values) or len(sequence_values) != len(
            set(sequence_values)
        ):
            raise ModelValidationError(
                "lifecycle_history sequences must be unique and increasing"
            )
        if self.schema_version != SCHEMA_VERSION:
            raise ModelValidationError(
                f"unsupported schema_version {self.schema_version!r}; "
                f"expected {SCHEMA_VERSION!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane_id": self.lane_id,
            "operation": self.operation,
            "operation_name": self.operation_name,
            "module": self.module,
            "equipment": self.equipment,
            "chamber": self.chamber,
            "recipe": self.recipe,
            "parameter_scope": list(self.parameter_scope),
            "exposed_lot_ids": list(self.exposed_lot_ids),
            "time_window": list(self.time_window),
            "initial_evidence_ids": list(self.initial_evidence_ids),
            "priority_score": self.priority_score,
            "investigation_status": self.investigation_status,
            "pruned_reason": self.pruned_reason,
            "lifecycle_status": self.lifecycle_status,
            "merged_from_lane_ids": list(self.merged_from_lane_ids),
            "merged_into_lane_id": self.merged_into_lane_id,
            "discovery_count": self.discovery_count,
            "reactivation_count": self.reactivation_count,
            "last_transition_reason": self.last_transition_reason,
            "lifecycle_history": [
                item.to_dict() for item in self.lifecycle_history
            ],
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            lane_id=data["lane_id"],
            operation=data.get("operation", ""),
            operation_name=data.get("operation_name", ""),
            module=data.get("module", ""),
            equipment=data.get("equipment", ""),
            chamber=data.get("chamber", ""),
            recipe=data.get("recipe", ""),
            parameter_scope=tuple(data.get("parameter_scope", [])),
            exposed_lot_ids=tuple(data.get("exposed_lot_ids", [])),
            time_window=tuple(data.get("time_window", [])),
            initial_evidence_ids=tuple(data.get("initial_evidence_ids", [])),
            priority_score=float(data.get("priority_score", 0.0)),
            investigation_status=data.get(
                "investigation_status",
                InvestigationLaneStatus.UNINVESTIGATED.value,
            ),
            pruned_reason=data.get("pruned_reason"),
            lifecycle_status=data.get("lifecycle_status", ""),
            merged_from_lane_ids=tuple(data.get("merged_from_lane_ids", [])),
            merged_into_lane_id=data.get("merged_into_lane_id"),
            discovery_count=data.get("discovery_count", 1),
            reactivation_count=data.get("reactivation_count", 0),
            last_transition_reason=data.get("last_transition_reason"),
            lifecycle_history=tuple(
                LaneLifecycleTransition.from_dict(item)
                for item in data.get("lifecycle_history", [])
            ),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class CandidateChallenge:
    """An auditable adversarial challenge for one RCA candidate.

    This is a state record, not the Candidate Generation response contract.
    Qwen may propose the challenge fields, but Python validates IDs and owns
    the resulting AlternativeSearchStatus.
    """

    candidate_id: str
    alternative_candidate_id: str | None = None
    evidence_probe_lane_id: str | None = None
    challenge_kind: str = ChallengeKind.LANE_PROBE.value
    mechanism_relation: str = CandidateMechanismRelation.UNKNOWN.value
    strongest_alternative_lane_id: str | None = None
    # Public API aliases retained alongside the precise Lane-aware names.
    # ``distinguishing_questions`` is explanatory text only; executable work
    # must use Python-generated ``distinguishing_gap_ids``.
    strongest_alternative: str | None = None
    supporting_evidence_ids: tuple[str, ...] = ()
    contradicting_evidence_ids: tuple[str, ...] = ()
    unexplained_precursor_evidence_ids: tuple[str, ...] = ()
    distinguishing_gap_ids: tuple[str, ...] = ()
    distinguishing_questions: tuple[str, ...] = ()
    challenge_explanation: str = ""
    status: str = ChallengeStatus.OPEN.value
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _non_empty(self.candidate_id, "candidate_id")
        object.__setattr__(
            self,
            "alternative_candidate_id",
            _optional_string(
                self.alternative_candidate_id,
                "alternative_candidate_id",
            ),
        )
        if self.alternative_candidate_id == self.candidate_id:
            raise ModelValidationError(
                "alternative_candidate_id must differ from candidate_id"
            )
        object.__setattr__(
            self,
            "evidence_probe_lane_id",
            _optional_string(self.evidence_probe_lane_id, "evidence_probe_lane_id"),
        )
        object.__setattr__(
            self,
            "strongest_alternative_lane_id",
            _optional_string(
                self.strongest_alternative_lane_id,
                "strongest_alternative_lane_id",
            ),
        )
        alias = _optional_string(self.strongest_alternative, "strongest_alternative")
        if (
            self.strongest_alternative_lane_id is not None
            and alias is not None
            and self.strongest_alternative_lane_id != alias
        ):
            raise ModelValidationError(
                "strongest_alternative and strongest_alternative_lane_id must agree"
            )
        object.__setattr__(
            self,
            "strongest_alternative",
            self.strongest_alternative_lane_id or alias,
        )
        if (
            self.evidence_probe_lane_id is not None
            and self.strongest_alternative_lane_id is not None
            and self.evidence_probe_lane_id != self.strongest_alternative_lane_id
        ):
            raise ModelValidationError(
                "evidence_probe_lane_id and strongest_alternative_lane_id must agree"
            )
        object.__setattr__(
            self,
            "evidence_probe_lane_id",
            self.evidence_probe_lane_id or self.strongest_alternative_lane_id,
        )
        try:
            challenge_kind = ChallengeKind(self.challenge_kind).value
        except ValueError as exc:
            allowed = ", ".join(item.value for item in ChallengeKind)
            raise ModelValidationError(
                f"challenge_kind must be one of: {allowed}"
            ) from exc
        object.__setattr__(self, "challenge_kind", challenge_kind)
        try:
            mechanism_relation = CandidateMechanismRelation(
                self.mechanism_relation
            ).value
        except ValueError as exc:
            allowed = ", ".join(item.value for item in CandidateMechanismRelation)
            raise ModelValidationError(
                f"mechanism_relation must be one of: {allowed}"
            ) from exc
        object.__setattr__(self, "mechanism_relation", mechanism_relation)
        for field_name in (
            "supporting_evidence_ids",
            "contradicting_evidence_ids",
            "unexplained_precursor_evidence_ids",
            "distinguishing_gap_ids",
        ):
            object.__setattr__(
                self,
                field_name,
                _string_tuple(getattr(self, field_name), field_name),
            )
        if not isinstance(self.challenge_explanation, str):
            raise ModelValidationError("challenge_explanation must be a string")
        object.__setattr__(self, "challenge_explanation", self.challenge_explanation.strip())
        object.__setattr__(
            self,
            "distinguishing_questions",
            _string_tuple(self.distinguishing_questions, "distinguishing_questions"),
        )
        try:
            status = ChallengeStatus(self.status).value
        except ValueError as exc:
            allowed = ", ".join(item.value for item in ChallengeStatus)
            raise ModelValidationError(f"status must be one of: {allowed}") from exc
        object.__setattr__(self, "status", status)
        if self.schema_version != SCHEMA_VERSION:
            raise ModelValidationError(
                f"unsupported schema_version {self.schema_version!r}; "
                f"expected {SCHEMA_VERSION!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "alternative_candidate_id": self.alternative_candidate_id,
            "evidence_probe_lane_id": self.evidence_probe_lane_id,
            "challenge_kind": self.challenge_kind,
            "mechanism_relation": self.mechanism_relation,
            "strongest_alternative_lane_id": self.strongest_alternative_lane_id,
            "strongest_alternative": self.strongest_alternative,
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "contradicting_evidence_ids": list(self.contradicting_evidence_ids),
            "unexplained_precursor_evidence_ids": list(
                self.unexplained_precursor_evidence_ids
            ),
            "distinguishing_gap_ids": list(self.distinguishing_gap_ids),
            "distinguishing_questions": list(self.distinguishing_questions),
            "challenge_explanation": self.challenge_explanation,
            "status": self.status,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            candidate_id=data["candidate_id"],
            alternative_candidate_id=data.get("alternative_candidate_id"),
            evidence_probe_lane_id=data.get("evidence_probe_lane_id"),
            challenge_kind=data.get(
                "challenge_kind",
                ChallengeKind.LANE_PROBE.value,
            ),
            mechanism_relation=data.get(
                "mechanism_relation",
                CandidateMechanismRelation.UNKNOWN.value,
            ),
            strongest_alternative_lane_id=data.get(
                "strongest_alternative_lane_id",
                data.get("strongest_alternative"),
            ),
            strongest_alternative=data.get("strongest_alternative"),
            supporting_evidence_ids=tuple(data.get("supporting_evidence_ids", [])),
            contradicting_evidence_ids=tuple(data.get("contradicting_evidence_ids", [])),
            unexplained_precursor_evidence_ids=tuple(
                data.get("unexplained_precursor_evidence_ids", [])
            ),
            distinguishing_gap_ids=tuple(data.get("distinguishing_gap_ids", [])),
            distinguishing_questions=tuple(data.get("distinguishing_questions", [])),
            challenge_explanation=data.get("challenge_explanation", ""),
            status=data.get("status", ChallengeStatus.OPEN.value),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class AlternativeLaneResolution:
    """Auditable Python interpretation of one Lane-level challenge result."""

    lane_id: str
    status: str
    candidate_id: str | None = None
    evidence_ids: tuple[str, ...] = ()
    distinguishing_gap_ids: tuple[str, ...] = ()
    reason_code: str | None = None
    reason: str = ""
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _non_empty(self.lane_id, "lane_id")
        object.__setattr__(
            self,
            "candidate_id",
            _optional_string(self.candidate_id, "candidate_id"),
        )
        for field_name in ("evidence_ids", "distinguishing_gap_ids"):
            object.__setattr__(
                self,
                field_name,
                _string_tuple(getattr(self, field_name), field_name),
            )
        try:
            status = AlternativeLaneResolutionStatus(self.status).value
        except ValueError as exc:
            allowed = ", ".join(item.value for item in AlternativeLaneResolutionStatus)
            raise ModelValidationError(
                f"alternative Lane resolution status must be one of: {allowed}"
            ) from exc
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "reason_code",
            _optional_string(self.reason_code, "reason_code"),
        )
        if not isinstance(self.reason, str):
            raise ModelValidationError("reason must be a string")
        object.__setattr__(self, "reason", self.reason.strip())
        if self.schema_version != SCHEMA_VERSION:
            raise ModelValidationError(
                f"unsupported schema_version {self.schema_version!r}; "
                f"expected {SCHEMA_VERSION!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane_id": self.lane_id,
            "status": self.status,
            "candidate_id": self.candidate_id,
            "evidence_ids": list(self.evidence_ids),
            "distinguishing_gap_ids": list(self.distinguishing_gap_ids),
            "reason_code": self.reason_code,
            "reason": self.reason,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            lane_id=data["lane_id"],
            status=data["status"],
            candidate_id=data.get("candidate_id"),
            evidence_ids=tuple(data.get("evidence_ids", [])),
            distinguishing_gap_ids=tuple(
                data.get("distinguishing_gap_ids", [])
            ),
            reason_code=data.get("reason_code"),
            reason=data.get("reason", ""),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class CompetitionTrace:
    """Python-owned state describing whether alternatives were searched."""

    active_lane_ids: tuple[str, ...] = ()
    overflow_lane_ids: tuple[str, ...] = ()
    represented_lane_ids: tuple[str, ...] = ()
    unresolved_lane_ids: tuple[str, ...] = ()
    eliminated_lane_ids: tuple[str, ...] = ()
    blocked_lane_ids: tuple[str, ...] = ()
    lane_resolutions: tuple[AlternativeLaneResolution, ...] = ()
    candidate_resolutions: tuple[CandidateResolution, ...] = ()
    action_value_assessments: tuple[ActionValueAssessment, ...] = ()
    alternative_search_status: str = AlternativeSearchStatus.NOT_SEARCHED.value
    competition_requirement: str = CompetitionRequirement.NOT_EVALUATED.value
    competition_status: str = CandidateCompetitionStatus.NOT_EVALUATED.value
    competition_type: str = CandidateCompetitionType.NOT_EVALUATED.value
    competition_axes: tuple[str, ...] = ()
    scope_assessment_status: str = ScopeAssessmentStatus.NOT_EVALUATED.value
    competition_failure_reason: str | None = None
    competition_gap_reason: str | None = None
    terminal_reason: str | None = None
    candidate_semantic_profiles: tuple[CandidateSemanticProfile, ...] = ()
    candidate_lineage: tuple[dict[str, Any], ...] = ()
    challenge_round_count: int = 0
    resolution_evidence_ids: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "active_lane_ids",
            "overflow_lane_ids",
            "represented_lane_ids",
            "unresolved_lane_ids",
            "eliminated_lane_ids",
            "blocked_lane_ids",
            "resolution_evidence_ids",
        ):
            object.__setattr__(
                self,
                field_name,
                _string_tuple(getattr(self, field_name), field_name),
            )
        if set(self.active_lane_ids) & set(self.overflow_lane_ids):
            raise ModelValidationError("active and overflow Lane IDs must be disjoint")
        if set(self.unresolved_lane_ids) & set(self.eliminated_lane_ids):
            raise ModelValidationError("unresolved and eliminated Lane IDs must be disjoint")
        if set(self.unresolved_lane_ids) & set(self.blocked_lane_ids):
            raise ModelValidationError("unresolved and blocked Lane IDs must be disjoint")
        if set(self.eliminated_lane_ids) & set(self.blocked_lane_ids):
            raise ModelValidationError("eliminated and blocked Lane IDs must be disjoint")
        if not isinstance(self.lane_resolutions, (list, tuple)) or any(
            not isinstance(item, AlternativeLaneResolution)
            for item in self.lane_resolutions
        ):
            raise ModelValidationError(
                "lane_resolutions must contain AlternativeLaneResolution instances"
            )
        resolution_lane_ids = [item.lane_id for item in self.lane_resolutions]
        if len(resolution_lane_ids) != len(set(resolution_lane_ids)):
            raise ModelValidationError(
                "lane_resolutions must contain at most one result per Lane"
            )
        if not isinstance(self.candidate_resolutions, (list, tuple)) or any(
            not isinstance(item, CandidateResolution)
            for item in self.candidate_resolutions
        ):
            raise ModelValidationError(
                "candidate_resolutions must contain CandidateResolution instances"
            )
        candidate_resolution_ids = [
            item.candidate_id for item in self.candidate_resolutions
        ]
        if len(candidate_resolution_ids) != len(set(candidate_resolution_ids)):
            raise ModelValidationError(
                "candidate_resolutions must contain at most one result per Candidate"
            )
        if not isinstance(self.action_value_assessments, (list, tuple)) or any(
            not isinstance(item, ActionValueAssessment)
            for item in self.action_value_assessments
        ):
            raise ModelValidationError(
                "action_value_assessments must contain ActionValueAssessment instances"
            )
        assessment_ids = [item.option_id for item in self.action_value_assessments]
        if len(assessment_ids) != len(set(assessment_ids)):
            raise ModelValidationError(
                "action_value_assessments must contain unique option_id values"
            )
        try:
            status = AlternativeSearchStatus(self.alternative_search_status).value
        except ValueError as exc:
            allowed = ", ".join(item.value for item in AlternativeSearchStatus)
            raise ModelValidationError(
                f"alternative_search_status must be one of: {allowed}"
            ) from exc
        object.__setattr__(self, "alternative_search_status", status)
        try:
            requirement = CompetitionRequirement(self.competition_requirement).value
        except ValueError as exc:
            allowed = ", ".join(item.value for item in CompetitionRequirement)
            raise ModelValidationError(
                f"competition_requirement must be one of: {allowed}"
            ) from exc
        object.__setattr__(self, "competition_requirement", requirement)
        try:
            competition_status = CandidateCompetitionStatus(
                self.competition_status
            ).value
        except ValueError as exc:
            allowed = ", ".join(item.value for item in CandidateCompetitionStatus)
            raise ModelValidationError(
                f"competition_status must be one of: {allowed}"
            ) from exc
        object.__setattr__(self, "competition_status", competition_status)
        try:
            competition_type = CandidateCompetitionType(self.competition_type).value
        except ValueError as exc:
            allowed = ", ".join(item.value for item in CandidateCompetitionType)
            raise ModelValidationError(
                f"competition_type must be one of: {allowed}"
            ) from exc
        object.__setattr__(self, "competition_type", competition_type)
        try:
            competition_axes = tuple(
                dict.fromkeys(
                    CandidateCompetitionAxis(item).value
                    for item in self.competition_axes
                )
            )
        except ValueError as exc:
            allowed = ", ".join(item.value for item in CandidateCompetitionAxis)
            raise ModelValidationError(
                f"competition_axes must contain only: {allowed}"
            ) from exc
        object.__setattr__(self, "competition_axes", competition_axes)
        try:
            scope_assessment_status = ScopeAssessmentStatus(
                self.scope_assessment_status
            ).value
        except ValueError as exc:
            allowed = ", ".join(item.value for item in ScopeAssessmentStatus)
            raise ModelValidationError(
                f"scope_assessment_status must be one of: {allowed}"
            ) from exc
        object.__setattr__(
            self,
            "scope_assessment_status",
            scope_assessment_status,
        )
        if self.competition_failure_reason is not None:
            try:
                failure_reason = CompetitionFailureReason(
                    self.competition_failure_reason
                ).value
            except ValueError as exc:
                allowed = ", ".join(item.value for item in CompetitionFailureReason)
                raise ModelValidationError(
                    f"competition_failure_reason must be one of: {allowed}"
                ) from exc
            object.__setattr__(self, "competition_failure_reason", failure_reason)
        if competition_status == CandidateCompetitionStatus.FAILED.value:
            if self.competition_failure_reason is None:
                raise ModelValidationError(
                    "failed competition_status requires competition_failure_reason"
                )
        elif self.competition_failure_reason is not None:
            raise ModelValidationError(
                "competition_failure_reason is valid only when competition_status=failed"
            )
        if self.competition_gap_reason is not None:
            try:
                gap_reason = CompetitionGapReason(self.competition_gap_reason).value
            except ValueError as exc:
                allowed = ", ".join(item.value for item in CompetitionGapReason)
                raise ModelValidationError(
                    f"competition_gap_reason must be one of: {allowed}"
                ) from exc
            object.__setattr__(self, "competition_gap_reason", gap_reason)
        object.__setattr__(
            self,
            "terminal_reason",
            _optional_string(self.terminal_reason, "terminal_reason"),
        )
        terminal_statuses = {
            CandidateCompetitionStatus.NOT_REQUIRED.value,
            CandidateCompetitionStatus.COMPLETE_CONFIRMED.value,
            CandidateCompetitionStatus.COMPLETE_REJECTED.value,
            CandidateCompetitionStatus.EXHAUSTED.value,
            CandidateCompetitionStatus.RESOLVED.value,
            CandidateCompetitionStatus.FAILED.value,
            CandidateCompetitionStatus.BLOCKED_BY_MISSING_DATA.value,
            CandidateCompetitionStatus.BUDGET_EXHAUSTED.value,
        }
        if competition_status not in terminal_statuses and self.terminal_reason is not None:
            raise ModelValidationError(
                "terminal_reason is valid only for terminal competition states"
            )
        if not isinstance(self.candidate_semantic_profiles, (list, tuple)) or any(
            not isinstance(item, CandidateSemanticProfile)
            for item in self.candidate_semantic_profiles
        ):
            raise ModelValidationError(
                "candidate_semantic_profiles must contain "
                "CandidateSemanticProfile instances"
            )
        semantic_candidate_ids = [
            item.candidate_id for item in self.candidate_semantic_profiles
        ]
        if len(semantic_candidate_ids) != len(set(semantic_candidate_ids)):
            raise ModelValidationError(
                "candidate_semantic_profiles must contain at most one profile "
                "per candidate_id"
            )
        if not isinstance(self.candidate_lineage, (list, tuple)) or any(
            not isinstance(item, dict) for item in self.candidate_lineage
        ):
            raise ModelValidationError("candidate_lineage must contain objects")
        object.__setattr__(
            self,
            "candidate_lineage",
            tuple(dict(item) for item in self.candidate_lineage),
        )
        if not isinstance(self.challenge_round_count, int) or self.challenge_round_count < 0:
            raise ModelValidationError("challenge_round_count must be a non-negative integer")
        if self.schema_version != SCHEMA_VERSION:
            raise ModelValidationError(
                f"unsupported schema_version {self.schema_version!r}; "
                f"expected {SCHEMA_VERSION!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_lane_ids": list(self.active_lane_ids),
            "overflow_lane_ids": list(self.overflow_lane_ids),
            "represented_lane_ids": list(self.represented_lane_ids),
            "unresolved_lane_ids": list(self.unresolved_lane_ids),
            "eliminated_lane_ids": list(self.eliminated_lane_ids),
            "blocked_lane_ids": list(self.blocked_lane_ids),
            "lane_resolutions": [item.to_dict() for item in self.lane_resolutions],
            "candidate_resolutions": [
                item.to_dict() for item in self.candidate_resolutions
            ],
            "action_value_assessments": [
                item.to_dict() for item in self.action_value_assessments
            ],
            "alternative_search_status": self.alternative_search_status,
            "competition_requirement": self.competition_requirement,
            "competition_status": self.competition_status,
            "competition_type": self.competition_type,
            "competition_axes": list(self.competition_axes),
            "scope_assessment_status": self.scope_assessment_status,
            "competition_failure_reason": self.competition_failure_reason,
            "competition_gap_reason": self.competition_gap_reason,
            "terminal_reason": self.terminal_reason,
            "candidate_semantic_profiles": [
                item.to_dict() for item in self.candidate_semantic_profiles
            ],
            "candidate_lineage": [dict(item) for item in self.candidate_lineage],
            "challenge_round_count": self.challenge_round_count,
            "resolution_evidence_ids": list(self.resolution_evidence_ids),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            active_lane_ids=tuple(data.get("active_lane_ids", [])),
            overflow_lane_ids=tuple(data.get("overflow_lane_ids", [])),
            represented_lane_ids=tuple(data.get("represented_lane_ids", [])),
            unresolved_lane_ids=tuple(data.get("unresolved_lane_ids", [])),
            eliminated_lane_ids=tuple(data.get("eliminated_lane_ids", [])),
            blocked_lane_ids=tuple(data.get("blocked_lane_ids", [])),
            lane_resolutions=tuple(
                AlternativeLaneResolution.from_dict(item)
                for item in data.get("lane_resolutions", [])
            ),
            candidate_resolutions=tuple(
                CandidateResolution.from_dict(item)
                for item in data.get("candidate_resolutions", [])
            ),
            action_value_assessments=tuple(
                ActionValueAssessment.from_dict(item)
                for item in data.get("action_value_assessments", [])
            ),
            alternative_search_status=data.get(
                "alternative_search_status",
                AlternativeSearchStatus.NOT_SEARCHED.value,
            ),
            competition_requirement=data.get(
                "competition_requirement",
                CompetitionRequirement.NOT_EVALUATED.value,
            ),
            competition_status=data.get(
                "competition_status",
                CandidateCompetitionStatus.NOT_EVALUATED.value,
            ),
            competition_type=data.get(
                "competition_type",
                CandidateCompetitionType.NOT_EVALUATED.value,
            ),
            competition_axes=tuple(data.get("competition_axes", [])),
            scope_assessment_status=data.get(
                "scope_assessment_status",
                ScopeAssessmentStatus.NOT_EVALUATED.value,
            ),
            competition_failure_reason=data.get("competition_failure_reason"),
            competition_gap_reason=data.get("competition_gap_reason"),
            terminal_reason=data.get("terminal_reason"),
            candidate_semantic_profiles=tuple(
                CandidateSemanticProfile.from_dict(item)
                for item in data.get("candidate_semantic_profiles", [])
            ),
            candidate_lineage=tuple(data.get("candidate_lineage", [])),
            challenge_round_count=int(data.get("challenge_round_count", 0)),
            resolution_evidence_ids=tuple(data.get("resolution_evidence_ids", [])),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


__all__ = [
    "ActionDecisionImpact",
    "ActionSourceAvailability",
    "ActionValueAssessment",
    "ActionValueTier",
    "AlternativeLaneResolution",
    "AlternativeLaneResolutionStatus",
    "AlternativeSearchStatus",
    "CandidateCompetitionAxis",
    "CandidateCompetitionStatus",
    "CandidateCompetitionType",
    "CandidateResolution",
    "CandidateResolutionStatus",
    "CandidateDistinguishingPrediction",
    "CandidateMechanismRelation",
    "CandidateScopeRelation",
    "CandidateSemanticProfile",
    "CandidateChallenge",
    "CausalChainCompleteness",
    "CausalLaneRecord",
    "ChallengeStatus",
    "ChallengeKind",
    "CompetitionTrace",
    "CompetitionFailureReason",
    "CompetitionGapReason",
    "CompetitionRequirement",
    "InvestigationLaneStatus",
    "InvestigationGainRecord",
    "InvestigationGainReasonCode",
    "InvestigationGainType",
    "LaneLifecycleStatus",
    "LaneLifecycleTransition",
    "ScopeAssessmentStatus",
]
