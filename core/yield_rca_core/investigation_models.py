"""Typed contracts shared by controlled and LLM ReAct RCA orchestration.

These contracts deliberately do not execute Tools or Agents.  They describe
the investigation goal, engineering questions, bounded actions, planner
decisions, and compact evaluations exchanged by orchestration components.
Keeping this layer framework-free makes structured LLM output and its audit
trail independently testable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Self

from yield_rca_core.causal_investigation_models import ActionValueAssessment


class InvestigationValidationError(ValueError):
    """Raised when a controlled investigation contract is invalid."""


class OrchestrationMode(StrEnum):
    FIXED = "fixed"
    CONTROLLED_REACT = "controlled_react"
    LLM_REACT = "llm_react"


class InvestigationIntent(StrEnum):
    IMPACT_SCOPE = "impact_scope"
    SPC_CHECK = "spc_check"
    ROOT_CAUSE = "root_cause"
    HISTORICAL_LOOKUP = "historical_lookup"
    FULL_RCA = "full_rca"


class QuestionKind(StrEnum):
    """Bounded semantic type for one investigation evidence gap.

    ``UNSUPPORTED`` is an internal compatibility classification used when a
    legacy snapshot has no stable kind suffix.  It is intentionally not an
    executable capability and cannot be autonomously targeted by Qwen.
    """

    DEFECT_SIGNATURE = "defect_signature"
    IMPACT_SCOPE = "impact_scope"
    SPC_SIGNAL = "spc_signal"
    PROCESS_MECHANISM = "process_mechanism"
    PRODUCT_OUTCOME = "product_outcome"
    HISTORICAL_MATCH = "historical_match"
    TOOL_HISTORY = "tool_history"
    RECIPE_HISTORY = "recipe_history"
    METROLOGY_CORRELATION = "metrology_correlation"
    MATERIAL_TRACE = "material_trace"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class CapabilityNotice:
    """A typed explanation for a requested capability that is unavailable."""

    capability: str
    supported: bool
    reason: str
    available_alternatives: list[str] = field(default_factory=list)
    request_source: str = "user"

    def __post_init__(self) -> None:
        _non_empty(self.capability, "capability")
        _boolean(self.supported, "supported")
        _non_empty(self.reason, "reason")
        _string_list(self.available_alternatives, "available_alternatives")
        if self.request_source not in {"user", "qwen", "system"}:
            raise InvestigationValidationError(
                "request_source must be user, qwen, or system"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "supported": self.supported,
            "reason": self.reason,
            "available_alternatives": list(self.available_alternatives),
            "request_source": self.request_source,
        }

    @classmethod
    def from_dict(cls, data: object) -> Self:
        payload = _strict_object(
            data,
            required={"capability", "supported", "reason"},
            optional={"available_alternatives", "request_source"},
            name="CapabilityNotice",
        )
        return cls(
            capability=payload["capability"],
            supported=payload["supported"],
            reason=payload["reason"],
            available_alternatives=payload.get("available_alternatives", []),
            request_source=payload.get("request_source", "user"),
        )


class ActionKind(StrEnum):
    INSPECT_DEFECT_PATTERN = "inspect_defect_pattern"
    VALIDATE_SHARED_DEFECT_PATTERN = "validate_shared_defect_pattern"
    FIND_SHARED_EXPOSURE = "find_shared_exposure"
    ASSESS_IMPACT_SCOPE = "assess_impact_scope"
    INSPECT_FDC_SPC = "inspect_fdc_spc"
    INSPECT_RECIPE_CHANGE = "inspect_recipe_change"
    VALIDATE_HISTORICAL_CASE = "validate_historical_case"
    RUN_RCA_REASONING = "run_rca_reasoning"
    CONCLUDE_INCONCLUSIVE = "conclude_inconclusive"


def _legacy_question_kind(question_id: str, question: str = "") -> str:
    """Infer a kind only for legacy snapshots that predate ``question_kind``.

    New planner output must carry the field explicitly.  The adapter is kept
    deliberately narrow: stable Intent Planner suffixes are preferred, with a
    small compatibility vocabulary for older fixtures that used ``Q_DEFECT``
    and ``Q_MECHANISM`` identifiers.
    """

    normalized_id = question_id.casefold().replace("-", "_")
    known_suffixes = {
        kind.value: kind.value
        for kind in QuestionKind
        if kind is not QuestionKind.UNSUPPORTED
    }
    for suffix, kind in known_suffixes.items():
        if normalized_id.endswith(suffix) or normalized_id.endswith(f"_{suffix}"):
            return kind
    if normalized_id.endswith("q_defect") or "defect" in normalized_id:
        return QuestionKind.DEFECT_SIGNATURE.value
    if normalized_id.endswith("q_mechanism") or "mechanism" in normalized_id:
        return QuestionKind.PROCESS_MECHANISM.value
    if normalized_id.endswith("q_impact") or "impact" in normalized_id:
        return QuestionKind.IMPACT_SCOPE.value
    if normalized_id.endswith("q_spc") or "spc" in normalized_id:
        return QuestionKind.SPC_SIGNAL.value
    lowered_question = question.casefold()
    if any(token in lowered_question for token in ("mechanism", "root cause", "cause")):
        return QuestionKind.PROCESS_MECHANISM.value
    if any(token in lowered_question for token in ("impact", "affected lot", "share the relevant")):
        return QuestionKind.IMPACT_SCOPE.value
    if "spc" in lowered_question or "control chart" in lowered_question:
        return QuestionKind.SPC_SIGNAL.value
    if any(token in lowered_question for token in ("defect", "scratch", "product outcome")):
        return QuestionKind.DEFECT_SIGNATURE.value
    return QuestionKind.UNSUPPORTED.value


class GoalStatus(StrEnum):
    IN_PROGRESS = "in_progress"
    SATISFIED = "satisfied"
    BLOCKED = "blocked"
    BUDGET_EXHAUSTED = "budget_exhausted"


class ConclusionLevel(StrEnum):
    SIGNAL = "signal"
    CANDIDATE = "candidate"
    SUPPORTED = "supported"
    CONFLICTED = "conflicted"
    INCONCLUSIVE = "inconclusive"


class StopReason(StrEnum):
    GOAL_SATISFIED = "goal_satisfied"
    CRITICAL_CONTRADICTION = "critical_contradiction"
    NO_ALLOWED_ACTION = "no_allowed_action"
    NO_HIGH_VALUE_ACTION = "no_high_value_action"
    BUDGET_EXHAUSTED = "budget_exhausted"
    DATA_UNAVAILABLE = "data_unavailable"


class EvidenceGapStatus(StrEnum):
    OPEN = "open"
    CLOSED = "closed"
    UNAVAILABLE = "unavailable"


class DecisionType(StrEnum):
    ACT = "act"
    STOP = "stop"


class PlannerAttemptStage(StrEnum):
    INTENT_PLANNING = "intent_planning"


class PlannerAttemptOutcome(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"


class PlannerFailureCategory(StrEnum):
    TRANSPORT_PROVIDER_FAILURE = "transport_provider_failure"
    OUTPUT_PARSE_ERROR = "output_parse_error"
    CONTRACT_VALIDATION_ERROR = "contract_validation_error"
    SEMANTIC_VALIDATION_ERROR = "semantic_validation_error"


class IntentPlannerReasonCode(StrEnum):
    GOAL_ID_CHANGED = "goal_id_changed"
    INTENT_INVALID = "intent_invalid"
    BUDGET_CHANGED = "budget_changed"
    KNOWN_FACT_REMOVED = "known_fact_removed"
    KNOWN_FACT_CHANGED = "known_fact_changed"
    FORBIDDEN_KNOWN_FACT_ADDED = "forbidden_known_fact_added"
    UNSUPPORTED_QUESTION_KIND = "unsupported_question_kind"
    UNREQUESTED_MATERIAL_TRACE = "unrequested_material_trace"
    SOURCE_LOT_SCOPE_MISMATCH = "source_lot_scope_mismatch"
    MALFORMED_OUTPUT = "malformed_output"


class QuestionUpdateDisposition(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class QuestionUpdateReasonCode(StrEnum):
    ACCEPTED = "accepted"
    MALFORMED_COLLECTION = "malformed_collection"
    TOO_MANY_UPDATES = "too_many_updates"
    MALFORMED_UPDATE = "malformed_update"
    NON_TERMINAL_STATUS = "non_terminal_status"
    DUPLICATE_QUESTION = "duplicate_question"
    UNKNOWN_QUESTION = "unknown_question"
    NEW_QUESTION_CONFLICT = "new_question_conflict"
    TERMINAL_QUESTION = "terminal_question"
    TARGET_OVERLAP = "target_overlap"
    UNKNOWN_EVIDENCE = "unknown_evidence"
    EVIDENCE_NOT_APPLICABLE = "evidence_not_applicable"
    INSUFFICIENT_EVIDENCE_COVERAGE = "insufficient_evidence_coverage"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    MISSING_UNAVAILABILITY_EVIDENCE = "missing_unavailability_evidence"


class QuestionEvidenceRelation(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    CONTEXT = "context"
    UNAVAILABLE = "unavailable"


MAX_CROSS_DOMAIN_ACTIONS = 8
MAX_INITIAL_QUESTIONS = 5


def _non_empty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise InvestigationValidationError(f"{name} must be a non-empty string")


def _string_list(values: list[str], name: str) -> None:
    valid_values = isinstance(values, list) and all(
        isinstance(value, str) and value.strip() for value in values
    )
    if not valid_values:
        raise InvestigationValidationError(f"{name} must be a list of non-empty strings")
    if len(values) != len(set(values)):
        raise InvestigationValidationError(f"{name} must not contain duplicates")


def _json_object(value: dict[str, Any], name: str) -> None:
    valid_keys = isinstance(value, dict) and all(
        isinstance(key, str) and key.strip() for key in value
    )
    if not valid_keys:
        raise InvestigationValidationError(f"{name} must be a JSON object with non-empty keys")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise InvestigationValidationError(f"{name} must contain JSON-compatible values") from exc


def _boolean(value: bool, name: str) -> None:
    if not isinstance(value, bool):
        raise InvestigationValidationError(f"{name} must be a boolean")


def _strict_object(
    data: object,
    *,
    required: set[str],
    optional: set[str],
    name: str,
) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise InvestigationValidationError(f"{name} must be a JSON object")
    invalid_keys = [key for key in data if not isinstance(key, str) or not key.strip()]
    if invalid_keys:
        raise InvestigationValidationError(f"{name} must contain non-empty string keys")
    actual_keys = set(data)
    missing = required - actual_keys
    if missing:
        raise InvestigationValidationError(f"{name} is missing fields: {sorted(missing)}")
    unknown = actual_keys - required - optional
    if unknown:
        raise InvestigationValidationError(f"{name} has unknown fields: {sorted(unknown)}")
    return data


@dataclass(frozen=True)
class InvestigationGoal:
    """The user outcome and bounded resources for one investigation."""

    goal_id: str
    intent: str
    summary: str
    known_facts: dict[str, Any] = field(default_factory=dict)
    required_evidence: list[str] = field(default_factory=list)
    max_steps: int = MAX_CROSS_DOMAIN_ACTIONS
    max_tool_calls: int = 20

    def __post_init__(self) -> None:
        _non_empty(self.goal_id, "goal_id")
        try:
            InvestigationIntent(self.intent)
        except ValueError as exc:
            raise InvestigationValidationError("intent is invalid") from exc
        _non_empty(self.summary, "summary")
        _json_object(self.known_facts, "known_facts")
        _string_list(self.required_evidence, "required_evidence")
        if (
            type(self.max_steps) is not int
            or self.max_steps < 1
            or self.max_steps > MAX_CROSS_DOMAIN_ACTIONS
        ):
            raise InvestigationValidationError(
                f"max_steps must be between 1 and {MAX_CROSS_DOMAIN_ACTIONS}"
            )
        if type(self.max_tool_calls) is not int or self.max_tool_calls < 1:
            raise InvestigationValidationError("max_tool_calls must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "intent": self.intent,
            "summary": self.summary,
            "known_facts": dict(self.known_facts),
            "required_evidence": list(self.required_evidence),
            "max_steps": self.max_steps,
            "max_tool_calls": self.max_tool_calls,
        }

    @classmethod
    def from_dict(cls, data: object) -> Self:
        payload = _strict_object(
            data,
            required={"goal_id", "intent", "summary"},
            optional={
                "known_facts",
                "required_evidence",
                "max_steps",
                "max_tool_calls",
            },
            name="InvestigationGoal",
        )
        return cls(
            goal_id=payload["goal_id"],
            intent=payload["intent"],
            summary=payload["summary"],
            known_facts=payload.get("known_facts", {}),
            required_evidence=payload.get("required_evidence", []),
            max_steps=payload.get("max_steps", MAX_CROSS_DOMAIN_ACTIONS),
            max_tool_calls=payload.get("max_tool_calls", 20),
        )


@dataclass(frozen=True)
class InvestigationAction:
    """One policy-authorized, auditable next action; never a free-form Tool call."""

    action_id: str
    kind: str
    agent: str
    reason: str
    inputs: dict[str, Any] = field(default_factory=dict)
    scope: dict[str, Any] = field(default_factory=dict)
    required_evidence_ids: list[str] = field(default_factory=list)
    max_attempts: int = 1

    def __post_init__(self) -> None:
        _non_empty(self.action_id, "action_id")
        try:
            ActionKind(self.kind)
        except ValueError as exc:
            raise InvestigationValidationError("action kind is invalid") from exc
        _non_empty(self.agent, "agent")
        _non_empty(self.reason, "reason")
        _json_object(self.inputs, "inputs")
        _json_object(self.scope, "scope")
        _string_list(self.required_evidence_ids, "required_evidence_ids")
        if type(self.max_attempts) is not int or self.max_attempts < 1:
            raise InvestigationValidationError("max_attempts must be a positive integer")

    @property
    def deduplication_key(self) -> tuple[str, str]:
        """Return the stable Action + Scope key used by loop protection."""

        effective_scope = self.scope or self.inputs
        serialized_scope = json.dumps(
            effective_scope,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        return self.kind, serialized_scope

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "kind": self.kind,
            "agent": self.agent,
            "reason": self.reason,
            "inputs": dict(self.inputs),
            "scope": dict(self.scope),
            "required_evidence_ids": list(self.required_evidence_ids),
            "max_attempts": self.max_attempts,
        }

    @classmethod
    def from_dict(cls, data: object) -> Self:
        payload = _strict_object(
            data,
            required={"action_id", "kind", "agent", "reason"},
            optional={"inputs", "scope", "required_evidence_ids", "max_attempts"},
            name="InvestigationAction",
        )
        return cls(
            action_id=payload["action_id"],
            kind=payload["kind"],
            agent=payload["agent"],
            reason=payload["reason"],
            inputs=payload.get("inputs", {}),
            scope=payload.get("scope", {}),
            required_evidence_ids=payload.get("required_evidence_ids", []),
            max_attempts=payload.get("max_attempts", 1),
        )


@dataclass(frozen=True)
class ActionRecord:
    """The immutable audit record for an action selection and its observation."""

    action: InvestigationAction
    status: str
    produced_finding_ids: list[str] = field(default_factory=list)
    produced_evidence_ids: list[str] = field(default_factory=list)
    decision_summary: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.action, InvestigationAction):
            raise InvestigationValidationError("action must be an InvestigationAction")
        if self.status not in {"completed", "skipped", "failed"}:
            raise InvestigationValidationError("status must be completed, skipped, or failed")
        _string_list(self.produced_finding_ids, "produced_finding_ids")
        _string_list(self.produced_evidence_ids, "produced_evidence_ids")
        _non_empty(self.decision_summary, "decision_summary")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "status": self.status,
            "produced_finding_ids": list(self.produced_finding_ids),
            "produced_evidence_ids": list(self.produced_evidence_ids),
            "decision_summary": self.decision_summary,
        }


@dataclass(frozen=True)
class QuestionEvidenceLink:
    """Deterministic relation between one Question and immutable Evidence."""

    question_id: str
    evidence_id: str
    action_id: str
    relation: str
    matched_evidence_group: str
    reason: str

    def __post_init__(self) -> None:
        _non_empty(self.question_id, "question_id")
        _non_empty(self.evidence_id, "evidence_id")
        _non_empty(self.action_id, "action_id")
        try:
            QuestionEvidenceRelation(self.relation)
        except ValueError as exc:
            raise InvestigationValidationError("evidence link relation is invalid") from exc
        _non_empty(self.matched_evidence_group, "matched_evidence_group")
        _non_empty(self.reason, "reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "evidence_id": self.evidence_id,
            "action_id": self.action_id,
            "relation": self.relation,
            "matched_evidence_group": self.matched_evidence_group,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: object) -> Self:
        payload = _strict_object(
            data,
            required={
                "question_id",
                "evidence_id",
                "action_id",
                "relation",
                "matched_evidence_group",
                "reason",
            },
            optional=set(),
            name="QuestionEvidenceLink",
        )
        return cls(
            question_id=payload["question_id"],
            evidence_id=payload["evidence_id"],
            action_id=payload["action_id"],
            relation=payload["relation"],
            matched_evidence_group=payload["matched_evidence_group"],
            reason=payload["reason"],
        )


@dataclass(frozen=True)
class InvestigationQuestion:
    """One evidence gap expressed as a bounded engineering question."""

    question_id: str
    goal_id: str
    question: str
    rationale: str
    question_kind: str | None = None
    scope: dict[str, Any] = field(default_factory=dict)
    status: str = EvidenceGapStatus.OPEN.value
    answer: str | None = None
    evidence_ids: list[str] = field(default_factory=list)
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        _non_empty(self.question_id, "question_id")
        _non_empty(self.goal_id, "goal_id")
        _non_empty(self.question, "question")
        _non_empty(self.rationale, "rationale")
        question_kind = self.question_kind
        if question_kind is None or not str(question_kind).strip():
            question_kind = _legacy_question_kind(self.question_id, self.question)
        try:
            question_kind = QuestionKind(question_kind).value
        except ValueError as exc:
            raise InvestigationValidationError("question_kind is invalid") from exc
        object.__setattr__(self, "question_kind", question_kind)
        _json_object(self.scope, "scope")
        try:
            status = EvidenceGapStatus(self.status)
        except ValueError as exc:
            raise InvestigationValidationError("question status is invalid") from exc
        _string_list(self.evidence_ids, "evidence_ids")
        if status == EvidenceGapStatus.OPEN:
            if self.answer is not None or self.unavailable_reason is not None:
                raise InvestigationValidationError(
                    "an open question cannot have an answer or unavailable_reason"
                )
        elif status == EvidenceGapStatus.CLOSED:
            if self.answer is None:
                raise InvestigationValidationError("a closed question requires an answer")
            _non_empty(self.answer, "answer")
            if not self.evidence_ids:
                raise InvestigationValidationError(
                    "a closed question requires supporting evidence_ids"
                )
            if self.unavailable_reason is not None:
                raise InvestigationValidationError(
                    "a closed question cannot have unavailable_reason"
                )
        else:
            if self.unavailable_reason is None:
                raise InvestigationValidationError(
                    "an unavailable question requires unavailable_reason"
                )
            _non_empty(self.unavailable_reason, "unavailable_reason")
            if self.answer is not None:
                raise InvestigationValidationError(
                    "an unavailable question cannot have an answer"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "goal_id": self.goal_id,
            "question": self.question,
            "rationale": self.rationale,
            "question_kind": self.question_kind,
            "scope": dict(self.scope),
            "status": self.status,
            "answer": self.answer,
            "evidence_ids": list(self.evidence_ids),
            "unavailable_reason": self.unavailable_reason,
        }

    @classmethod
    def from_dict(cls, data: object) -> Self:
        payload = _strict_object(
            data,
            required={"question_id", "goal_id", "question", "rationale"},
            optional={
                "scope",
                "question_kind",
                "status",
                "answer",
                "evidence_ids",
                "unavailable_reason",
            },
            name="InvestigationQuestion",
        )
        return cls(
            question_id=payload["question_id"],
            goal_id=payload["goal_id"],
            question=payload["question"],
            rationale=payload["rationale"],
            question_kind=payload.get("question_kind"),
            scope=payload.get("scope", {}),
            status=payload.get("status", EvidenceGapStatus.OPEN.value),
            answer=payload.get("answer"),
            evidence_ids=payload.get("evidence_ids", []),
            unavailable_reason=payload.get("unavailable_reason"),
        )


@dataclass(frozen=True)
class QuestionUpdate:
    """A terminal-only delta applied to an existing investigation question."""

    question_id: str
    status: str
    answer: str | None = None
    evidence_ids: list[str] = field(default_factory=list)
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        _non_empty(self.question_id, "question_id")
        try:
            status = EvidenceGapStatus(self.status)
        except ValueError as exc:
            raise InvestigationValidationError(
                "status must be closed or unavailable"
            ) from exc
        if status == EvidenceGapStatus.OPEN:
            raise InvestigationValidationError(
                "status must be closed or unavailable"
            )
        _string_list(self.evidence_ids, "evidence_ids")
        if status == EvidenceGapStatus.CLOSED:
            if self.answer is None:
                raise InvestigationValidationError(
                    "answer must be a non-empty string when status is closed"
                )
            _non_empty(self.answer, "answer")
            if not self.evidence_ids:
                raise InvestigationValidationError(
                    "evidence_ids must contain supporting Evidence IDs when status is closed"
                )
            if self.unavailable_reason is not None:
                raise InvestigationValidationError(
                    "unavailable_reason must be null when status is closed"
                )
        else:
            if self.answer is not None:
                raise InvestigationValidationError(
                    "answer must be null when status is unavailable"
                )
            if self.unavailable_reason is None:
                raise InvestigationValidationError(
                    "unavailable_reason must be a non-empty string when status is unavailable"
                )
            _non_empty(self.unavailable_reason, "unavailable_reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "status": self.status,
            "answer": self.answer,
            "evidence_ids": list(self.evidence_ids),
            "unavailable_reason": self.unavailable_reason,
        }

    @classmethod
    def from_dict(cls, data: object) -> Self:
        payload = _strict_object(
            data,
            required={"question_id", "status"},
            optional={"answer", "evidence_ids", "unavailable_reason"},
            name="QuestionUpdate",
        )
        return cls(
            question_id=payload["question_id"],
            status=payload["status"],
            answer=payload.get("answer"),
            evidence_ids=payload.get("evidence_ids", []),
            unavailable_reason=payload.get("unavailable_reason"),
        )


def _raise_with_path(
    *,
    path: str,
    error: InvestigationValidationError,
) -> None:
    message = str(error)
    field_names = (
        "question_id",
        "goal_id",
        "question",
        "rationale",
        "question_kind",
        "scope",
        "status",
        "answer",
        "evidence_ids",
        "unavailable_reason",
    )
    separator = "." if message.startswith(field_names) else ": "
    raise InvestigationValidationError(f"{path}{separator}{message}") from error


def _parse_question_update(
    data: object,
    *,
    goal_id: str,
) -> QuestionUpdate:
    """Read compact updates and project legacy full-Question snapshots."""

    legacy_fields = {"goal_id", "question", "rationale", "scope"}
    if isinstance(data, dict) and legacy_fields & set(data):
        legacy = InvestigationQuestion.from_dict(data)
        if legacy.goal_id != goal_id:
            raise InvestigationValidationError(
                "goal_id must match the planner decision goal_id"
            )
        return QuestionUpdate(
            question_id=legacy.question_id,
            status=legacy.status,
            answer=legacy.answer,
            evidence_ids=list(legacy.evidence_ids),
            unavailable_reason=legacy.unavailable_reason,
        )
    return QuestionUpdate.from_dict(data)


@dataclass(frozen=True)
class IntentPlan:
    """The bounded Goal and initial evidence questions produced from user intent."""

    goal: InvestigationGoal
    questions: list[InvestigationQuestion]
    capability_notices: list[CapabilityNotice] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.goal, InvestigationGoal):
            raise InvestigationValidationError("goal must be an InvestigationGoal")
        if not isinstance(self.questions, list) or not self.questions:
            raise InvestigationValidationError("questions must be a non-empty list")
        if len(self.questions) > MAX_INITIAL_QUESTIONS:
            raise InvestigationValidationError(
                f"questions must not exceed {MAX_INITIAL_QUESTIONS} items"
            )
        question_ids: list[str] = []
        for question in self.questions:
            if not isinstance(question, InvestigationQuestion):
                raise InvestigationValidationError(
                    "questions must contain InvestigationQuestion instances"
                )
            if question.goal_id != self.goal.goal_id:
                raise InvestigationValidationError(
                    "intent questions must reference the planned goal"
                )
            if question.status != EvidenceGapStatus.OPEN.value:
                raise InvestigationValidationError(
                    "initial intent questions must start open"
                )
            question_ids.append(question.question_id)
        if len(question_ids) != len(set(question_ids)):
            raise InvestigationValidationError("questions must not contain duplicate ids")
        if not isinstance(self.capability_notices, list) or any(
            not isinstance(notice, CapabilityNotice)
            for notice in self.capability_notices
        ):
            raise InvestigationValidationError(
                "capability_notices must contain CapabilityNotice instances"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal.to_dict(),
            "questions": [question.to_dict() for question in self.questions],
            "capability_notices": [
                notice.to_dict() for notice in self.capability_notices
            ],
        }

    @classmethod
    def from_dict(cls, data: object) -> Self:
        payload = _strict_object(
            data,
            required={"goal", "questions"},
            optional={"capability_notices"},
            name="IntentPlan",
        )
        raw_questions = payload["questions"]
        if not isinstance(raw_questions, list):
            raise InvestigationValidationError("questions must be a list")
        raw_notices = payload.get("capability_notices", [])
        if not isinstance(raw_notices, list):
            raise InvestigationValidationError("capability_notices must be a list")
        return cls(
            goal=InvestigationGoal.from_dict(payload["goal"]),
            questions=[
                InvestigationQuestion.from_dict(question) for question in raw_questions
            ],
            capability_notices=[
                CapabilityNotice.from_dict(notice) for notice in raw_notices
            ],
        )


@dataclass(frozen=True)
class PlannerAttemptDiagnostic:
    """Bounded, non-sensitive audit data for one Planner output attempt."""

    stage: str
    attempt: int
    prompt_name: str
    prompt_version: str
    outcome: str
    failure_category: str | None = None
    reason_code: str | None = None
    field_path: str | None = None
    message: str | None = None
    repair_feedback_sent: bool = False
    candidate_summary: dict[str, Any] = field(default_factory=dict)
    baseline_diff: dict[str, Any] = field(default_factory=dict)
    provider_request_id: str | None = None

    def __post_init__(self) -> None:
        try:
            PlannerAttemptStage(self.stage)
        except ValueError as exc:
            raise InvestigationValidationError("planner diagnostic stage is invalid") from exc
        if type(self.attempt) is not int or self.attempt < 1:
            raise InvestigationValidationError(
                "planner diagnostic attempt must be a positive integer"
            )
        _non_empty(self.prompt_name, "prompt_name")
        _non_empty(self.prompt_version, "prompt_version")
        try:
            outcome = PlannerAttemptOutcome(self.outcome)
        except ValueError as exc:
            raise InvestigationValidationError(
                "planner diagnostic outcome is invalid"
            ) from exc
        _boolean(self.repair_feedback_sent, "repair_feedback_sent")
        _json_object(self.candidate_summary, "candidate_summary")
        _json_object(self.baseline_diff, "baseline_diff")
        if self.provider_request_id is not None:
            _non_empty(self.provider_request_id, "provider_request_id")
        if outcome == PlannerAttemptOutcome.SUCCESS:
            if any(
                value is not None
                for value in (
                    self.failure_category,
                    self.reason_code,
                    self.field_path,
                    self.message,
                )
            ):
                raise InvestigationValidationError(
                    "a successful planner diagnostic cannot carry failure details"
                )
            if self.repair_feedback_sent:
                raise InvestigationValidationError(
                    "a successful planner attempt cannot send repair feedback"
                )
            return
        try:
            PlannerFailureCategory(self.failure_category or "")
        except ValueError as exc:
            raise InvestigationValidationError(
                "a failed planner diagnostic requires a valid failure_category"
            ) from exc
        try:
            IntentPlannerReasonCode(self.reason_code or "")
        except ValueError as exc:
            raise InvestigationValidationError(
                "a failed planner diagnostic requires a valid reason_code"
            ) from exc
        if self.field_path is not None:
            _non_empty(self.field_path, "field_path")
        if self.message is None:
            raise InvestigationValidationError(
                "a failed planner diagnostic requires a message"
            )
        _non_empty(self.message, "message")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "attempt": self.attempt,
            "prompt_name": self.prompt_name,
            "prompt_version": self.prompt_version,
            "outcome": self.outcome,
            "failure_category": self.failure_category,
            "reason_code": self.reason_code,
            "field_path": self.field_path,
            "message": self.message,
            "repair_feedback_sent": self.repair_feedback_sent,
            "candidate_summary": dict(self.candidate_summary),
            "baseline_diff": dict(self.baseline_diff),
            "provider_request_id": self.provider_request_id,
        }

    @classmethod
    def from_dict(cls, data: object) -> Self:
        payload = _strict_object(
            data,
            required={
                "stage",
                "attempt",
                "prompt_name",
                "prompt_version",
                "outcome",
                "failure_category",
                "reason_code",
                "field_path",
                "message",
                "repair_feedback_sent",
                "candidate_summary",
                "baseline_diff",
                "provider_request_id",
            },
            optional=set(),
            name="PlannerAttemptDiagnostic",
        )
        return cls(
            stage=payload["stage"],
            attempt=payload["attempt"],
            prompt_name=payload["prompt_name"],
            prompt_version=payload["prompt_version"],
            outcome=payload["outcome"],
            failure_category=payload["failure_category"],
            reason_code=payload["reason_code"],
            field_path=payload["field_path"],
            message=payload["message"],
            repair_feedback_sent=payload["repair_feedback_sent"],
            candidate_summary=payload["candidate_summary"],
            baseline_diff=payload["baseline_diff"],
            provider_request_id=payload["provider_request_id"],
        )


@dataclass(frozen=True)
class IntentPlanOutcome:
    """An accepted IntentPlan and the complete bounded attempt audit."""

    plan: IntentPlan
    attempt_diagnostics: list[PlannerAttemptDiagnostic]

    def __post_init__(self) -> None:
        if not isinstance(self.plan, IntentPlan):
            raise InvestigationValidationError("plan must be an IntentPlan")
        if not isinstance(self.attempt_diagnostics, list) or not self.attempt_diagnostics:
            raise InvestigationValidationError(
                "attempt_diagnostics must be a non-empty list"
            )
        for expected_attempt, diagnostic in enumerate(self.attempt_diagnostics, start=1):
            if not isinstance(diagnostic, PlannerAttemptDiagnostic):
                raise InvestigationValidationError(
                    "attempt_diagnostics must contain PlannerAttemptDiagnostic instances"
                )
            if diagnostic.attempt != expected_attempt:
                raise InvestigationValidationError(
                    "attempt_diagnostics must be ordered with contiguous attempt numbers"
                )
        if (
            self.attempt_diagnostics[-1].outcome
            != PlannerAttemptOutcome.SUCCESS.value
        ):
            raise InvestigationValidationError(
                "an IntentPlanOutcome must end with a successful attempt"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "attempt_diagnostics": [
                diagnostic.to_dict() for diagnostic in self.attempt_diagnostics
            ],
        }

    @classmethod
    def from_dict(cls, data: object) -> Self:
        payload = _strict_object(
            data,
            required={"plan", "attempt_diagnostics"},
            optional=set(),
            name="IntentPlanOutcome",
        )
        raw_diagnostics = payload["attempt_diagnostics"]
        if not isinstance(raw_diagnostics, list):
            raise InvestigationValidationError("attempt_diagnostics must be a list")
        return cls(
            plan=IntentPlan.from_dict(payload["plan"]),
            attempt_diagnostics=[
                PlannerAttemptDiagnostic.from_dict(diagnostic)
                for diagnostic in raw_diagnostics
            ],
        )


@dataclass(frozen=True)
class PlannerDecision:
    """One strict planner result: execute one action or stop explicitly."""

    decision_id: str
    goal_id: str
    decision_type: str
    reason: str
    goal_status: str
    proposed_conclusion_level: str
    next_action: InvestigationAction | None = None
    target_question_ids: list[str] = field(default_factory=list)
    new_questions: list[InvestigationQuestion] = field(default_factory=list)
    stop_reason: str | None = None
    question_updates: list[QuestionUpdate] = field(default_factory=list)

    def __post_init__(self) -> None:
        _non_empty(self.decision_id, "decision_id")
        _non_empty(self.goal_id, "goal_id")
        _non_empty(self.reason, "reason")
        try:
            decision_type = DecisionType(self.decision_type)
        except ValueError as exc:
            raise InvestigationValidationError("decision_type is invalid") from exc
        try:
            goal_status = GoalStatus(self.goal_status)
        except ValueError as exc:
            raise InvestigationValidationError("goal_status is invalid") from exc
        try:
            ConclusionLevel(self.proposed_conclusion_level)
        except ValueError as exc:
            raise InvestigationValidationError("proposed_conclusion_level is invalid") from exc
        _string_list(self.target_question_ids, "target_question_ids")
        if not isinstance(self.new_questions, list):
            raise InvestigationValidationError("new_questions must be a list")
        question_ids: list[str] = []
        for question in self.new_questions:
            if not isinstance(question, InvestigationQuestion):
                raise InvestigationValidationError(
                    "new_questions must contain InvestigationQuestion instances"
                )
            if question.goal_id != self.goal_id:
                raise InvestigationValidationError(
                    "a planner decision cannot create a question for another goal"
                )
            if question.status != EvidenceGapStatus.OPEN.value:
                raise InvestigationValidationError("new planner questions must start open")
            question_ids.append(question.question_id)
        if len(question_ids) != len(set(question_ids)):
            raise InvestigationValidationError("new_questions must not contain duplicate ids")
        if not isinstance(self.question_updates, list):
            raise InvestigationValidationError("question_updates must be a list")
        updated_question_ids: list[str] = []
        for update in self.question_updates:
            if not isinstance(update, QuestionUpdate):
                raise InvestigationValidationError(
                    "question_updates must contain QuestionUpdate instances"
                )
            updated_question_ids.append(update.question_id)
        if len(updated_question_ids) != len(set(updated_question_ids)):
            raise InvestigationValidationError(
                "question_updates must not contain duplicate ids"
            )
        if set(updated_question_ids) & set(question_ids):
            raise InvestigationValidationError(
                "a question cannot be both new and updated in one decision"
            )

        if decision_type == DecisionType.ACT:
            if not isinstance(self.next_action, InvestigationAction):
                raise InvestigationValidationError("an act decision requires next_action")
            if self.stop_reason is not None:
                raise InvestigationValidationError(
                    "an act decision cannot include stop_reason"
                )
            if goal_status != GoalStatus.IN_PROGRESS:
                raise InvestigationValidationError(
                    "an act decision requires goal_status=in_progress"
                )
            if not self.target_question_ids:
                raise InvestigationValidationError(
                    "an act decision must target at least one investigation question"
                )
            conflicting_question_ids = sorted(
                set(updated_question_ids) & set(self.target_question_ids)
            )
            if conflicting_question_ids:
                raise InvestigationValidationError(
                    "target_question_ids and question_updates overlap for "
                    f"{conflicting_question_ids}; if the next action still "
                    "investigates this Question, keep it open and omit its "
                    "QuestionUpdate"
                )
        else:
            if self.next_action is not None:
                raise InvestigationValidationError(
                    "a stop decision cannot include next_action"
                )
            if self.stop_reason is None:
                raise InvestigationValidationError("a stop decision requires stop_reason")
            try:
                StopReason(self.stop_reason)
            except ValueError as exc:
                raise InvestigationValidationError("stop_reason is invalid") from exc
            if goal_status == GoalStatus.IN_PROGRESS:
                raise InvestigationValidationError(
                    "a stop decision cannot leave the goal in progress"
                )
            if self.new_questions:
                raise InvestigationValidationError(
                    "a stop decision cannot create new open questions"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "goal_id": self.goal_id,
            "decision_type": self.decision_type,
            "reason": self.reason,
            "goal_status": self.goal_status,
            "proposed_conclusion_level": self.proposed_conclusion_level,
            "next_action": self.next_action.to_dict() if self.next_action else None,
            "target_question_ids": list(self.target_question_ids),
            "new_questions": [question.to_dict() for question in self.new_questions],
            "stop_reason": self.stop_reason,
            "question_updates": [update.to_dict() for update in self.question_updates],
        }

    @classmethod
    def from_dict(
        cls,
        data: object,
        *,
        allow_legacy_question_updates: bool = True,
    ) -> Self:
        payload = _strict_object(
            data,
            required={
                "decision_id",
                "goal_id",
                "decision_type",
                "reason",
                "goal_status",
                "proposed_conclusion_level",
            },
            optional={
                "next_action",
                "target_question_ids",
                "new_questions",
                "stop_reason",
                "question_updates",
            },
            name="PlannerDecision",
        )
        raw_action = payload.get("next_action")
        if raw_action is not None and not isinstance(raw_action, dict):
            raise InvestigationValidationError("next_action must be a JSON object or null")
        raw_questions = payload.get("new_questions", [])
        if not isinstance(raw_questions, list):
            raise InvestigationValidationError("new_questions must be a list")
        raw_question_updates = payload.get("question_updates", [])
        if not isinstance(raw_question_updates, list):
            raise InvestigationValidationError("question_updates must be a list")
        new_questions: list[InvestigationQuestion] = []
        for index, question in enumerate(raw_questions):
            try:
                new_questions.append(InvestigationQuestion.from_dict(question))
            except InvestigationValidationError as exc:
                _raise_with_path(path=f"new_questions[{index}]", error=exc)

        question_updates: list[QuestionUpdate] = []
        for index, question in enumerate(raw_question_updates):
            try:
                if allow_legacy_question_updates:
                    update = _parse_question_update(
                        question,
                        goal_id=payload["goal_id"],
                    )
                else:
                    update = QuestionUpdate.from_dict(question)
                question_updates.append(update)
            except InvestigationValidationError as exc:
                _raise_with_path(path=f"question_updates[{index}]", error=exc)

        return cls(
            decision_id=payload["decision_id"],
            goal_id=payload["goal_id"],
            decision_type=payload["decision_type"],
            reason=payload["reason"],
            goal_status=payload["goal_status"],
            proposed_conclusion_level=payload["proposed_conclusion_level"],
            next_action=(
                InvestigationAction.from_dict(raw_action) if raw_action is not None else None
            ),
            target_question_ids=payload.get("target_question_ids", []),
            new_questions=new_questions,
            stop_reason=payload.get("stop_reason"),
            question_updates=question_updates,
        )


@dataclass(frozen=True)
class QuestionUpdateReview:
    """One bounded audit result for a model-proposed QuestionUpdate."""

    decision_id: str
    disposition: str
    reason_code: str
    reason: str
    update_index: int | None = None
    question_id: str | None = None
    claimed_status: str | None = None

    def __post_init__(self) -> None:
        _non_empty(self.decision_id, "decision_id")
        try:
            disposition = QuestionUpdateDisposition(self.disposition)
        except ValueError as exc:
            raise InvestigationValidationError(
                "question update disposition is invalid"
            ) from exc
        try:
            reason_code = QuestionUpdateReasonCode(self.reason_code)
        except ValueError as exc:
            raise InvestigationValidationError(
                "question update reason_code is invalid"
            ) from exc
        _non_empty(self.reason, "reason")
        if self.update_index is not None and (
            type(self.update_index) is not int or self.update_index < 0
        ):
            raise InvestigationValidationError(
                "update_index must be null or a non-negative integer"
            )
        if self.question_id is not None:
            _non_empty(self.question_id, "question_id")
        if self.claimed_status is not None:
            _non_empty(self.claimed_status, "claimed_status")
        if disposition == QuestionUpdateDisposition.ACCEPTED:
            if reason_code != QuestionUpdateReasonCode.ACCEPTED:
                raise InvestigationValidationError(
                    "an accepted QuestionUpdate requires reason_code=accepted"
                )
            if self.question_id is None or self.update_index is None:
                raise InvestigationValidationError(
                    "an accepted QuestionUpdate review requires an index and question_id"
                )
            if self.claimed_status not in {
                EvidenceGapStatus.CLOSED.value,
                EvidenceGapStatus.UNAVAILABLE.value,
            }:
                raise InvestigationValidationError(
                    "an accepted QuestionUpdate review requires a terminal claimed_status"
                )
        elif reason_code == QuestionUpdateReasonCode.ACCEPTED:
            raise InvestigationValidationError(
                "a rejected QuestionUpdate cannot use reason_code=accepted"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "disposition": self.disposition,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "update_index": self.update_index,
            "question_id": self.question_id,
            "claimed_status": self.claimed_status,
        }

    @classmethod
    def from_dict(cls, data: object) -> Self:
        payload = _strict_object(
            data,
            required={"decision_id", "disposition", "reason_code", "reason"},
            optional={"update_index", "question_id", "claimed_status"},
            name="QuestionUpdateReview",
        )
        return cls(
            decision_id=payload["decision_id"],
            disposition=payload["disposition"],
            reason_code=payload["reason_code"],
            reason=payload["reason"],
            update_index=payload.get("update_index"),
            question_id=payload.get("question_id"),
            claimed_status=payload.get("claimed_status"),
        )


@dataclass(frozen=True)
class PlannerDecisionOutcome:
    """A legal core decision plus independently reviewed QuestionUpdate claims."""

    decision: PlannerDecision
    question_update_reviews: list[QuestionUpdateReview] = field(default_factory=list)
    raw_question_update_count: int = 0
    decision_proposed_by: str = "qwen"
    question_updates_source: str | None = None
    action_value_assessments: list[ActionValueAssessment] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.decision, PlannerDecision):
            raise InvestigationValidationError("decision must be a PlannerDecision")
        if (
            type(self.raw_question_update_count) is not int
            or self.raw_question_update_count < 0
        ):
            raise InvestigationValidationError(
                "raw_question_update_count must be a non-negative integer"
            )
        if self.decision_proposed_by not in {"qwen", "python_runtime"}:
            raise InvestigationValidationError(
                "decision_proposed_by must be qwen or python_runtime"
            )
        if self.question_updates_source not in {
            None,
            "qwen",
            "python_evidence_gate",
        }:
            raise InvestigationValidationError(
                "question_updates_source must be qwen, python_evidence_gate, or null"
            )
        if self.question_updates_source is not None and not self.decision.question_updates:
            raise InvestigationValidationError(
                "question_updates_source requires committed question_updates"
            )
        if not isinstance(self.question_update_reviews, list):
            raise InvestigationValidationError(
                "question_update_reviews must be a list"
            )
        if not isinstance(self.action_value_assessments, list) or any(
            not isinstance(item, ActionValueAssessment)
            for item in self.action_value_assessments
        ):
            raise InvestigationValidationError(
                "action_value_assessments must contain ActionValueAssessment instances"
            )
        accepted_question_ids: list[str] = []
        for review in self.question_update_reviews:
            if not isinstance(review, QuestionUpdateReview):
                raise InvestigationValidationError(
                    "question_update_reviews must contain QuestionUpdateReview instances"
                )
            if review.decision_id != self.decision.decision_id:
                raise InvestigationValidationError(
                    "QuestionUpdate reviews must reference the outcome decision"
                )
            if review.disposition == QuestionUpdateDisposition.ACCEPTED.value:
                assert review.question_id is not None
                accepted_question_ids.append(review.question_id)
        if self.raw_question_update_count > 0 and accepted_question_ids != [
            update.question_id for update in self.decision.question_updates
        ]:
            raise InvestigationValidationError(
                "accepted QuestionUpdate reviews must match committed question_updates"
            )
        if self.raw_question_update_count < len(self.question_update_reviews):
            raise InvestigationValidationError(
                "raw_question_update_count cannot be smaller than the review count"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.to_dict(),
            "question_update_reviews": [
                review.to_dict() for review in self.question_update_reviews
            ],
            "raw_question_update_count": self.raw_question_update_count,
            "decision_proposed_by": self.decision_proposed_by,
            "question_updates_source": self.question_updates_source,
            "action_value_assessments": [
                item.to_dict() for item in self.action_value_assessments
            ],
        }

    @classmethod
    def from_dict(cls, data: object) -> Self:
        payload = _strict_object(
            data,
            required={
                "decision",
                "question_update_reviews",
                "raw_question_update_count",
            },
            optional={
                "decision_proposed_by",
                "question_updates_source",
                "action_value_assessments",
            },
            name="PlannerDecisionOutcome",
        )
        raw_reviews = payload["question_update_reviews"]
        if not isinstance(raw_reviews, list):
            raise InvestigationValidationError(
                "question_update_reviews must be a list"
            )
        return cls(
            decision=PlannerDecision.from_dict(payload["decision"]),
            question_update_reviews=[
                QuestionUpdateReview.from_dict(review) for review in raw_reviews
            ],
            raw_question_update_count=payload["raw_question_update_count"],
            decision_proposed_by=payload.get("decision_proposed_by", "qwen"),
            question_updates_source=payload.get("question_updates_source"),
            action_value_assessments=[
                ActionValueAssessment.from_dict(item)
                for item in payload.get("action_value_assessments", [])
            ],
        )


@dataclass(frozen=True)
class DecisionEvaluation:
    """The three agreed metrics for one planner decision."""

    decision_id: str
    decision_valid: bool
    evidence_gain: bool
    redundant: bool
    reason: str
    new_evidence_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _non_empty(self.decision_id, "decision_id")
        _boolean(self.decision_valid, "decision_valid")
        _boolean(self.evidence_gain, "evidence_gain")
        _boolean(self.redundant, "redundant")
        _non_empty(self.reason, "reason")
        _string_list(self.new_evidence_ids, "new_evidence_ids")
        if self.evidence_gain and not self.new_evidence_ids:
            raise InvestigationValidationError(
                "evidence_gain requires at least one new_evidence_id"
            )
        if self.evidence_gain and self.redundant:
            raise InvestigationValidationError(
                "a decision with evidence gain cannot be redundant"
            )
        if not self.decision_valid and self.evidence_gain:
            raise InvestigationValidationError(
                "an invalid decision cannot claim evidence gain"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "decision_valid": self.decision_valid,
            "evidence_gain": self.evidence_gain,
            "redundant": self.redundant,
            "reason": self.reason,
            "new_evidence_ids": list(self.new_evidence_ids),
        }

    @classmethod
    def from_dict(cls, data: object) -> Self:
        payload = _strict_object(
            data,
            required={
                "decision_id",
                "decision_valid",
                "evidence_gain",
                "redundant",
                "reason",
            },
            optional={"new_evidence_ids"},
            name="DecisionEvaluation",
        )
        return cls(
            decision_id=payload["decision_id"],
            decision_valid=payload["decision_valid"],
            evidence_gain=payload["evidence_gain"],
            redundant=payload["redundant"],
            reason=payload["reason"],
            new_evidence_ids=payload.get("new_evidence_ids", []),
        )


@dataclass(frozen=True)
class RunEvaluation:
    """The two agreed outcome metrics plus the per-decision evaluations."""

    goal_id: str
    goal_success: bool
    stop_correct: bool
    summary: str
    decision_evaluations: list[DecisionEvaluation]

    def __post_init__(self) -> None:
        _non_empty(self.goal_id, "goal_id")
        _boolean(self.goal_success, "goal_success")
        _boolean(self.stop_correct, "stop_correct")
        _non_empty(self.summary, "summary")
        if not isinstance(self.decision_evaluations, list) or not self.decision_evaluations:
            raise InvestigationValidationError(
                "decision_evaluations must be a non-empty list"
            )
        decision_ids: list[str] = []
        for evaluation in self.decision_evaluations:
            if not isinstance(evaluation, DecisionEvaluation):
                raise InvestigationValidationError(
                    "decision_evaluations must contain DecisionEvaluation instances"
                )
            decision_ids.append(evaluation.decision_id)
        if len(decision_ids) != len(set(decision_ids)):
            raise InvestigationValidationError(
                "decision_evaluations must not contain duplicate decision ids"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "goal_success": self.goal_success,
            "stop_correct": self.stop_correct,
            "summary": self.summary,
            "decision_evaluations": [
                evaluation.to_dict() for evaluation in self.decision_evaluations
            ],
        }

    @classmethod
    def from_dict(cls, data: object) -> Self:
        payload = _strict_object(
            data,
            required={
                "goal_id",
                "goal_success",
                "stop_correct",
                "summary",
                "decision_evaluations",
            },
            optional=set(),
            name="RunEvaluation",
        )
        raw_evaluations = payload["decision_evaluations"]
        if not isinstance(raw_evaluations, list):
            raise InvestigationValidationError("decision_evaluations must be a list")
        return cls(
            goal_id=payload["goal_id"],
            goal_success=payload["goal_success"],
            stop_correct=payload["stop_correct"],
            summary=payload["summary"],
            decision_evaluations=[
                DecisionEvaluation.from_dict(evaluation) for evaluation in raw_evaluations
            ],
        )
