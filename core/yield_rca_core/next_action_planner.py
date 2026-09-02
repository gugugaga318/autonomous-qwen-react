"""Qwen-backed next-action planning for bounded autonomous RCA investigations.

The model chooses one registered Agent action or an explicit stop after every
observation.  This module validates runtime safety, but deliberately does not
replace a legal model choice with the deterministic controlled-ReAct policy.
The deterministic policy is supplied only as the Fake Client's no-cost output
and as an explicit fallback hint for the caller.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from yield_rca_core.causal_competition import select_diverse_lane_ids
from yield_rca_core.causal_investigation_models import (
    ActionValueAssessment,
    InvestigationGainReasonCode,
    InvestigationGainRecord,
    InvestigationGainType,
)
from yield_rca_core.causal_lane_lifecycle import lane_is_searchable
from yield_rca_core.evidence_models import Evidence, EvidenceType
from yield_rca_core.investigation_decision import (
    action_scope_fingerprint,
    assess_action_values,
    derive_investigation_gain_history,
    high_value_action_assessments,
)
from yield_rca_core.investigation_models import (
    MAX_CROSS_DOMAIN_ACTIONS,
    MAX_INITIAL_QUESTIONS,
    ActionKind,
    ActionRecord,
    CapabilityNotice,
    ConclusionLevel,
    DecisionType,
    EvidenceGapStatus,
    GoalStatus,
    InvestigationAction,
    InvestigationGoal,
    InvestigationIntent,
    InvestigationQuestion,
    InvestigationValidationError,
    OrchestrationMode,
    PlannerDecision,
    PlannerDecisionOutcome,
    QuestionEvidenceLink,
    QuestionEvidenceRelation,
    QuestionUpdate,
    QuestionUpdateDisposition,
    QuestionUpdateReasonCode,
    QuestionUpdateReview,
    StopReason,
)
from yield_rca_core.investigation_policy import (
    ACTION_REGISTRY,
    ActionDefinition,
    InvestigationPolicy,
)
from yield_rca_core.llm_gateway import (
    LLMCallError,
    LLMClient,
    LLMOutputValidationError,
    LLMRequest,
)
from yield_rca_core.models import (
    AgentFinding,
    AgentKind,
    Hypothesis,
    ModelValidationError,
)
from yield_rca_core.question_capability import (
    QUESTION_CAPABILITY_REGISTRY,
    QuestionCapabilityError,
    action_scope_matches_question,
    capability_for_question,
    validate_action_for_questions,
)
from yield_rca_core.question_update_review import review_qwen_planner_output

_OUTPUT_ATTEMPTS = 2
_CALL_RETRIES = 1
_UNCONDITIONAL_CANDIDATE_GENERATION_ROUNDS = 2
_MAX_CANDIDATE_GENERATION_ROUNDS = 3
_MAX_CONSECUTIVE_NO_GAIN_ACTIONS = 2
_MAX_PROMPT_LANES = 3
_MAX_PROMPT_EVIDENCE_IDS = 24
_MAX_PROMPT_LINK_IDS_PER_RELATION = 6
_MAX_PROMPT_GAP_BASIS_ITEMS = 8
_MAX_PROMPT_AUDIT_ITEMS = 8
_MAX_PROMPT_TEXT_CHARS = 2_000
_OUTPUT_PARSE_ERROR = "output_parse"
_CORE_DECISION_VALIDATION_ERROR = "core_decision_validation"
_LANE_AWARE_ACTION_KINDS = frozenset(
    {
        ActionKind.INSPECT_DEFECT_PATTERN.value,
        ActionKind.VALIDATE_SHARED_DEFECT_PATTERN.value,
        ActionKind.FIND_SHARED_EXPOSURE.value,
        ActionKind.INSPECT_FDC_SPC.value,
        ActionKind.VALIDATE_HISTORICAL_CASE.value,
    }
)
_PLANNER_DECISION_OUTPUT_FIELDS = (
    "decision_id",
    "goal_id",
    "decision_type",
    "reason",
    "goal_status",
    "proposed_conclusion_level",
    "next_action",
    "target_question_ids",
    "new_questions",
    "stop_reason",
    "question_updates",
)
_PLANNER_INPUT_ONLY_FIELDS = (
    "goal_satisfied_stop_contract",
    "deterministic_planner_decision",
    "previous_validation_feedback",
    "legal_target_question_ids_by_action",
    "question_action_capabilities",
    "validator_ready_reference_question_updates",
    "python_terminal_transition_available",
    "python_terminal_question_ids",
    "causal_evidence_gaps",
    "legal_causal_gap_ids_by_action",
)


def _strip_exact_planner_input_echoes(
    output: object,
    *,
    request_payload: Mapping[str, Any],
) -> object:
    """Remove only known prompt scaffolding copied back without modification.

    These fields are Python-owned context rather than model decisions.  Exact
    equality is required so a misspelled, altered, or otherwise unknown field
    still reaches the strict PlannerDecision parser and fails closed.
    """

    if not isinstance(output, dict):
        return output
    sanitized = dict(output)
    for field_name in _PLANNER_INPUT_ONLY_FIELDS:
        if field_name not in sanitized:
            continue
        expected_values: list[Any] = []
        if field_name in request_payload:
            expected_values.append(request_payload[field_name])
        for parent_name in (
            "goal_satisfied_stop_contract",
            "previous_validation_feedback",
        ):
            parent = request_payload.get(parent_name)
            if isinstance(parent, Mapping) and field_name in parent:
                expected_values.append(parent[field_name])
        if not any(sanitized[field_name] == value for value in expected_values):
            continue
        echoed_value = sanitized.pop(field_name)
        if (
            field_name == "validator_ready_reference_question_updates"
            and sanitized.get("decision_type") == DecisionType.STOP.value
            and sanitized.get("goal_status") == GoalStatus.SATISFIED.value
            and sanitized.get("stop_reason") == StopReason.GOAL_SATISFIED.value
        ):
            # Qwen has explicitly selected the goal-satisfied boundary and
            # copied Python's exact, validator-ready transition.  Commit the
            # Python-owned delta under the real contract field instead of
            # asking the model to reproduce the same state twice.
            sanitized["question_updates"] = echoed_value
    return sanitized


def _is_retryable_call_error(error: LLMCallError) -> bool:
    """Retry only transient failures; configuration and call caps fail fast."""

    if error.failure_category == "call_limit":
        return False
    if error.failure_category == "transport_error":
        return True
    if error.status_code in {408, 429}:
        return True
    if error.status_code is not None and error.status_code >= 500:
        return True
    return error.failure_category == "llm_call_error" and error.status_code is None

# Only actions with a Supervisor dispatcher are visible to Qwen. Specialist
# Tool selection for these actions is bounded separately by Specialist V2.
LLM_REACT_EXECUTABLE_ACTION_KINDS = frozenset(
    {
        ActionKind.INSPECT_DEFECT_PATTERN.value,
        ActionKind.VALIDATE_SHARED_DEFECT_PATTERN.value,
        ActionKind.FIND_SHARED_EXPOSURE.value,
        ActionKind.INSPECT_FDC_SPC.value,
        ActionKind.VALIDATE_HISTORICAL_CASE.value,
        ActionKind.RUN_RCA_REASONING.value,
    }
)
LLM_REACT_ACTION_REGISTRY: Mapping[str, ActionDefinition] = MappingProxyType(
    {
        kind: ACTION_REGISTRY[kind]
        for kind in sorted(LLM_REACT_EXECUTABLE_ACTION_KINDS)
    }
)


class QwenNextActionPlannerError(LLMOutputValidationError):
    """Raised after two invalid next-action outputs require controlled fallback."""

    fallback_mode = OrchestrationMode.CONTROLLED_REACT.value

    def __init__(
        self,
        validation_errors: list[str],
        validation_error_categories: list[str],
        *,
        goal_id: str,
        completed_steps: int,
        tool_call_count: int,
    ) -> None:
        self.attempts = len(validation_errors)
        self.validation_errors = tuple(validation_errors)
        self.validation_error_categories = tuple(validation_error_categories)
        self.output_parse_error_count = self.validation_error_categories.count(
            _OUTPUT_PARSE_ERROR
        )
        self.core_validation_error_count = self.validation_error_categories.count(
            _CORE_DECISION_VALIDATION_ERROR
        )
        self.goal_id = goal_id
        self.completed_steps = completed_steps
        self.tool_call_count = tool_call_count
        super().__init__(
            "Qwen Next-action Planner returned invalid output twice; preserve the "
            f"current investigation state and fallback to {self.fallback_mode}"
        )


def _validate_string_list(values: list[str], name: str) -> None:
    if not isinstance(values, list) or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise ModelValidationError(f"{name} must be a list of non-empty strings")
    if len(values) != len(set(values)):
        raise ModelValidationError(f"{name} must not contain duplicates")


def _normalized_lot_id(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().upper()


def _assert_source_lot_boundary(
    payload: dict[str, Any],
    *,
    source_lot_id: str | None,
    label: str,
) -> None:
    """Prevent an impact Lot from silently becoming a new RCA objective."""

    if source_lot_id is None:
        return
    for key, value in payload.items():
        normalized_key = key.casefold()
        if normalized_key == "lot_ids":
            if not isinstance(value, list) or any(
                _normalized_lot_id(item) is None for item in value
            ):
                raise InvestigationValidationError(
                    f"{label}.lot_ids must be a list of non-empty Lot IDs"
                )
            normalized_values = {
                normalized
                for item in value
                if (normalized := _normalized_lot_id(item)) is not None
            }
            if source_lot_id not in normalized_values:
                raise InvestigationValidationError(
                    f"{label}.lot_ids must retain the source Lot {source_lot_id}"
                )
        elif normalized_key == "lot_id" or normalized_key.endswith("_lot_id"):
            normalized_value = _normalized_lot_id(value)
            if normalized_value != source_lot_id:
                raise InvestigationValidationError(
                    f"{label}.{key} cannot replace source Lot {source_lot_id}"
                )


def _missing_groups_for_questions(
    questions: list[InvestigationQuestion],
    links: list[QuestionEvidenceLink],
) -> dict[str, frozenset[str]]:
    """Project the current unsatisfied closure groups from Python-owned links."""

    satisfied_groups: dict[str, set[str]] = {
        question.question_id: set() for question in questions
    }
    for link in links:
        if (
            link.question_id in satisfied_groups
            and link.relation == QuestionEvidenceRelation.SUPPORTS.value
        ):
            satisfied_groups[link.question_id].add(link.matched_evidence_group)
    return {
        question.question_id: frozenset(
            set(capability_for_question(question).closure_evidence_groups)
            - satisfied_groups[question.question_id]
        )
        for question in questions
    }


def _bounded_prompt_text(value: object) -> str:
    text = str(value)
    if len(text) <= _MAX_PROMPT_TEXT_CHARS:
        return text
    return f"{text[:_MAX_PROMPT_TEXT_CHARS]}... [truncated]"


def _prompt_lane_ids_for_gap(
    gap: Mapping[str, Any],
    *,
    active_lane_ids: tuple[str, ...],
) -> tuple[str, ...]:
    applicable = {
        str(item)
        for item in gap.get("applicable_lane_ids", [])
        if isinstance(item, str) and item.strip()
    }
    selected = [lane_id for lane_id in active_lane_ids if lane_id in applicable]
    if not selected:
        selected = list(active_lane_ids)
    return tuple(selected[:_MAX_PROMPT_LANES])


def _compact_causal_gap_for_prompt(
    gap: Mapping[str, Any],
    *,
    active_lane_ids: tuple[str, ...],
) -> dict[str, Any]:
    """Bound Lane inventories without changing Python's authoritative Gap."""

    compact = dict(gap)
    prompt_lane_ids = _prompt_lane_ids_for_gap(
        gap,
        active_lane_ids=active_lane_ids,
    )
    raw_applicable = [
        str(item)
        for item in gap.get("applicable_lane_ids", [])
        if isinstance(item, str) and item.strip()
    ]
    compact["applicable_lane_ids"] = list(prompt_lane_ids)
    compact["applicable_lane_count"] = len(raw_applicable)
    compact["applicable_lane_ids_truncated"] = (
        len(raw_applicable) > len(prompt_lane_ids)
    )
    raw_gain_by_lane = gap.get("information_gain_by_lane", {})
    compact["information_gain_by_lane"] = (
        {
            lane_id: float(raw_gain_by_lane[lane_id])
            for lane_id in prompt_lane_ids
            if lane_id in raw_gain_by_lane
            and isinstance(raw_gain_by_lane[lane_id], int | float)
        }
        if isinstance(raw_gain_by_lane, Mapping)
        else {}
    )
    compact["information_gain_lane_count"] = (
        len(raw_gain_by_lane) if isinstance(raw_gain_by_lane, Mapping) else 0
    )
    basis = [
        _bounded_prompt_text(item)
        for item in gap.get("information_gain_basis", [])
        if isinstance(item, str) and item.strip()
    ]
    compact["information_gain_basis"] = basis[:_MAX_PROMPT_GAP_BASIS_ITEMS]
    compact["information_gain_basis_count"] = len(basis)
    evidence_ids = [
        str(item)
        for item in gap.get("evidence_ids", [])
        if isinstance(item, str) and item.strip()
    ]
    compact["evidence_ids"] = evidence_ids[:_MAX_PROMPT_EVIDENCE_IDS]
    compact["evidence_count"] = len(evidence_ids)
    compact["evidence_ids_truncated"] = (
        len(evidence_ids) > _MAX_PROMPT_EVIDENCE_IDS
    )
    compact["reason"] = _bounded_prompt_text(gap.get("reason", ""))
    return compact


def _compact_audit_mapping_for_prompt(value: Mapping[str, Any]) -> dict[str, Any]:
    """Bound challenge diagnostics while preserving their decision fields."""

    compact: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        if isinstance(raw_value, str):
            compact[key] = _bounded_prompt_text(raw_value)
        elif isinstance(raw_value, list):
            limit = (
                _MAX_PROMPT_EVIDENCE_IDS
                if key.endswith("evidence_ids")
                else _MAX_PROMPT_AUDIT_ITEMS
            )
            compact[key] = raw_value[:limit]
            compact[f"{key}_count"] = len(raw_value)
            compact[f"{key}_truncated"] = len(raw_value) > limit
        else:
            compact[key] = raw_value
    return compact


def _compact_finding(
    finding: AgentFinding,
    *,
    active_lane_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Project only decision-relevant Finding fields into Planner context.

    Full Specialist details remain in RCAState for audit and downstream tools.
    Re-sending them after every observation caused quadratic context growth and
    exposed the Planner to raw domain payloads it is not authorized to edit.
    """

    evidence_ids = list(finding.evidence_ids)
    compact = {
        "finding_id": finding.finding_id,
        "agent": finding.agent,
        "finding_kind": finding.finding_kind,
        "summary": _bounded_prompt_text(finding.summary),
        "confidence": finding.confidence,
        "evidence_ids": evidence_ids[:_MAX_PROMPT_EVIDENCE_IDS],
        "evidence_count": len(evidence_ids),
        "evidence_ids_truncated": len(evidence_ids) > _MAX_PROMPT_EVIDENCE_IDS,
    }
    # RCA diagnostics are already Python-compressed and contain only Evidence
    # IDs, claim statuses, and registry-derived Actions.  Expose this small
    # projection so the next Qwen decision can target a real causal gap without
    # replaying the full specialist payload.
    if finding.agent == AgentKind.RCA_REASONING.value:
        raw_causal_gaps = finding.details.get("causal_evidence_gaps", [])
        compact.update(
            {
                "causal_evidence_gaps": [
                    _compact_causal_gap_for_prompt(
                        item,
                        active_lane_ids=active_lane_ids,
                    )
                    for item in raw_causal_gaps
                    if isinstance(item, Mapping)
                ]
                if isinstance(raw_causal_gaps, list)
                else [],
                "candidate_comparison": dict(
                    finding.details.get("candidate_comparison", {})
                ),
                "alternative_search_status": str(
                    finding.details.get("alternative_search_status", "not_searched")
                ),
                "candidate_challenges": [
                    _compact_audit_mapping_for_prompt(item)
                    for item in finding.details.get("candidate_challenges", [])
                    if isinstance(item, Mapping)
                ][:_MAX_PROMPT_AUDIT_ITEMS],
                "alternative_lane_resolutions": list(
                    finding.details.get("alternative_lane_resolutions", [])
                )[:_MAX_PROMPT_AUDIT_ITEMS],
                "adversarial_challenge_generation": (
                    _compact_audit_mapping_for_prompt(
                        finding.details.get(
                            "adversarial_challenge_generation", {}
                        )
                    )
                    if isinstance(
                        finding.details.get(
                            "adversarial_challenge_generation", {}
                        ),
                        Mapping,
                    )
                    else {}
                ),
                "confirmation_gate": dict(
                    finding.details.get("confirmation_gate", {})
                ),
            }
        )
    if finding.agent == AgentKind.MES.value:
        raw_lanes = finding.details.get("lane_candidates", [])
        if isinstance(raw_lanes, list):
            lanes = [
                dict(item)
                for item in raw_lanes
                if isinstance(item, Mapping) and str(item.get("lane_id", "")).strip()
            ]
            selected_ids = active_lane_ids or select_diverse_lane_ids(
                lanes,
                limit=_MAX_PROMPT_LANES,
            )
            lanes_by_id = {str(item["lane_id"]): item for item in lanes}
            selected = [
                lanes_by_id[lane_id]
                for lane_id in selected_ids
                if lane_id in lanes_by_id
            ]
            compact["causal_lanes"] = selected
            compact["active_lane_ids"] = [
                str(item["lane_id"]) for item in selected
            ]
            compact["overflow_lane_count"] = max(0, len(lanes) - len(selected))
    return compact


def _known_causal_lane_ids(findings: list[AgentFinding]) -> tuple[str, ...]:
    lanes: list[dict[str, Any]] = []
    mes_findings = [
        finding for finding in findings if finding.agent == AgentKind.MES.value
    ]
    authoritative = [
        finding
        for finding in mes_findings
        if finding.details.get("lane_inventory_authoritative") is True
    ]
    source_findings = authoritative[-1:] if authoritative else mes_findings
    for finding in source_findings:
        raw_lanes = finding.details.get("lane_candidates", [])
        if not isinstance(raw_lanes, list):
            continue
        lanes.extend(
            dict(item)
            for item in raw_lanes
            if isinstance(item, Mapping)
            and str(item.get("lane_id", "")).strip()
        )
    ordered = sorted(
        {
            str(item.get("lane_id", "")).strip(): item
            for item in lanes
            if str(item.get("lane_id", "")).strip()
        }.values(),
        key=lambda item: (
            -float(item.get("priority_score", 0.0)),
            str(item.get("lane_id", "")),
        ),
    )
    searchable = [
        item
        for item in ordered
        if lane_is_searchable(item)
    ]
    return (
        select_diverse_lane_ids(searchable, limit=_MAX_PROMPT_LANES)
        if searchable
        else ()
    )


def _compact_action_record(record: ActionRecord) -> dict[str, Any]:
    evidence_ids = list(record.produced_evidence_ids)
    finding_ids = list(record.produced_finding_ids)
    return {
        "action": record.action.to_dict(),
        "status": record.status,
        "produced_finding_ids": finding_ids[:4],
        "produced_finding_count": len(finding_ids),
        "produced_evidence_ids": evidence_ids[:_MAX_PROMPT_EVIDENCE_IDS],
        "produced_evidence_count": len(evidence_ids),
        "evidence_ids_truncated": len(evidence_ids) > _MAX_PROMPT_EVIDENCE_IDS,
        "decision_summary": record.decision_summary,
    }


def _compact_question_context_for_prompt(
    question_context: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for packet in question_context:
        linked = packet.get("linked_evidence", {})
        projected_links: dict[str, list[str]] = {}
        link_counts: dict[str, int] = {}
        for relation in ("supports", "contradicts", "context", "unavailable"):
            values = [str(item) for item in linked.get(relation, [])]
            projected_links[relation] = values[:_MAX_PROMPT_LINK_IDS_PER_RELATION]
            link_counts[relation] = len(values)
        compact.append(
            {
                **packet,
                "linked_evidence": projected_links,
                "linked_evidence_counts": link_counts,
            }
        )
    return compact


def _authoritative_causal_gaps(
    findings: list[AgentFinding],
    authoritative_rca_finding_id: str | None,
) -> list[dict[str, Any]]:
    """Read only registry-bounded gaps from the authoritative RCA Finding."""

    if not authoritative_rca_finding_id:
        return []
    finding = next(
        (
            item
            for item in findings
            if item.finding_id == authoritative_rca_finding_id
            and item.agent == AgentKind.RCA_REASONING.value
        ),
        None,
    )
    if finding is None:
        return []
    raw_gaps = finding.details.get("causal_evidence_gaps", [])
    if not isinstance(raw_gaps, list):
        return []
    raw_challenges = finding.details.get("candidate_challenges", [])
    challenge_selected_gap_ids = {
        str(gap_id).strip()
        for challenge in raw_challenges
        if isinstance(challenge, Mapping)
        for gap_id in (
            challenge.get("distinguishing_gap_ids", [])
            if isinstance(challenge.get("distinguishing_gap_ids", []), list)
            else []
        )
        if isinstance(gap_id, str) and gap_id.strip()
    }
    gaps: list[dict[str, Any]] = []
    for raw in raw_gaps:
        if not isinstance(raw, Mapping):
            continue
        gap_id = str(raw.get("gap_id", "")).strip()
        question_kind = str(raw.get("question_kind", "")).strip()
        capability = QUESTION_CAPABILITY_REGISTRY.get(question_kind)
        raw_actions = raw.get("allowed_actions", [])
        if (
            not gap_id
            or capability is None
            or not capability.supported
            or not isinstance(raw_actions, list)
        ):
            continue
        actions = sorted(
            {
                str(action)
                for action in raw_actions
                if str(action) in capability.allowed_actions
                and str(action) in LLM_REACT_EXECUTABLE_ACTION_KINDS
            }
        )
        if not actions:
            continue
        raw_target_scope = raw.get("target_scope", {})
        target_scope = (
            {
                str(key): str(value)
                for key, value in raw_target_scope.items()
                if isinstance(value, str) and value.strip()
            }
            if isinstance(raw_target_scope, Mapping)
            else {}
        )
        gaps.append(
            {
                "gap_id": gap_id,
                "gap_type": str(raw.get("gap_type", "missing_support")),
                "gap_origin": str(raw.get("gap_origin", "")),
                "scope_group_id": str(raw.get("scope_group_id", "")),
                "discriminator_kind": str(raw.get("discriminator_kind", "")),
                "lane_binding": str(raw.get("lane_binding", "")),
                "priority": int(raw.get("priority", 3)),
                "information_gain": float(raw.get("information_gain", 0.0)),
                "information_gain_by_lane": {
                    str(key): float(value)
                    for key, value in raw.get(
                        "information_gain_by_lane", {}
                    ).items()
                    if isinstance(value, int | float)
                }
                if isinstance(raw.get("information_gain_by_lane", {}), Mapping)
                else {},
                "information_gain_basis": [
                    str(item)
                    for item in raw.get("information_gain_basis", [])
                    if isinstance(item, str) and item.strip()
                ],
                "applicable_lane_ids": [
                    str(item)
                    for item in raw.get("applicable_lane_ids", [])
                    if isinstance(item, str) and item.strip()
                ],
                "candidate_index": raw.get("candidate_index"),
                "candidate_id": str(raw.get("candidate_id", "")),
                "candidate_ids": [
                    str(item)
                    for item in raw.get("candidate_ids", [])
                    if isinstance(item, str) and item.strip()
                ]
                if isinstance(raw.get("candidate_ids", []), list)
                else [],
                "claim": str(raw.get("claim", "")),
                "status": str(raw.get("status", "")),
                "reason": str(raw.get("reason", "")),
                "question_kind": question_kind,
                "allowed_actions": actions,
                "evidence_ids": [
                    str(item)
                    for item in raw.get("evidence_ids", [])
                    if isinstance(item, str) and item.strip()
                ],
                # Preserve the Python-owned competition semantics generated by
                # the authoritative RCA Finding.  These fields are inputs to
                # Action Value assessment, not claims supplied by the Planner.
                # Dropping them makes a valid mechanism discriminator look
                # like a generic context query and can terminate an
                # investigation while a ranking-changing Action still exists.
                "competition_axis": str(raw.get("competition_axis", "")),
                "can_change_ranking": bool(
                    raw.get("can_change_ranking", False)
                ),
                "supports_candidate_ids": [
                    str(item)
                    for item in raw.get("supports_candidate_ids", [])
                    if isinstance(item, str) and item.strip()
                ]
                if isinstance(raw.get("supports_candidate_ids", []), list)
                else [],
                "weakens_candidate_ids": [
                    str(item)
                    for item in raw.get("weakens_candidate_ids", [])
                    if isinstance(item, str) and item.strip()
                ]
                if isinstance(raw.get("weakens_candidate_ids", []), list)
                else [],
                "expected_evidence_types": [
                    str(item)
                    for item in raw.get("expected_evidence_types", [])
                    if isinstance(item, str) and item.strip()
                ]
                if isinstance(raw.get("expected_evidence_types", []), list)
                else [],
                # A competition-wide Scope Gap is created before the Qwen
                # adversarial challenge exists, so its raw flag may still be
                # false.  The authoritative Challenge is the later semantic
                # selection event and must make the referenced discriminator
                # executable for the Planner.
                "challenge_selected": bool(
                    raw.get("challenge_selected", False)
                )
                or gap_id in challenge_selected_gap_ids,
                "decision_impact": str(raw.get("decision_impact", "")),
                "preferred_action": str(raw.get("preferred_action", "")),
                "refresh_action": str(raw.get("refresh_action", "")),
                "required_evidence_groups": [
                    str(item)
                    for item in raw.get("required_evidence_groups", [])
                    if isinstance(item, str) and item.strip()
                ],
                "target_scope": target_scope,
            }
        )
    return sorted(
        gaps,
        key=lambda item: (
            int(item.get("priority", 3)),
            int(item.get("candidate_index", 0)),
            str(item.get("gap_id", "")),
        ),
    )


def _eligible_causal_gaps(
    causal_gaps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return authoritative Gaps that may participate in Action selection."""

    return [
        gap
        for gap in causal_gaps
        if gap.get("gap_type") != "hypothesis_discrimination"
        or gap.get("challenge_selected") is True
    ]


def _active_causal_gaps(
    causal_gaps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the highest-priority authoritative Gap set.

    This raw projection is suitable for audit and bounded-investigation state.
    Action projection must first test every eligible Gap against the current
    registry, prerequisites, Lane binding, and Action history, then prioritize
    only the Gaps that remain executable.  Otherwise a high-priority but
    currently unexecutable Gap can mask a lower-priority discriminator.
    """

    executable_gaps = _eligible_causal_gaps(causal_gaps)
    if not executable_gaps:
        return []
    priority = min(int(gap.get("priority", 3)) for gap in executable_gaps)
    return [
        gap
        for gap in executable_gaps
        if int(gap.get("priority", 3)) == priority
    ]


def _has_executable_hypothesis_discrimination_gap(
    causal_gaps: list[dict[str, Any]],
    causal_gap_ids_by_action: Mapping[str, list[str]],
) -> bool:
    """Return whether a challenge-selected discriminator can still run.

    A confirmation-blocking unavailable source must not terminate the workflow
    while Python has already identified a legal Action that can distinguish an
    unresolved alternative.  The Action mapping is the state-aware projection,
    so exhausted stages, missing prerequisites, closed Questions, and budgeted
    reasoning rounds do not count as executable here.
    """

    executable_gap_ids = {
        gap_id
        for gap_ids in causal_gap_ids_by_action.values()
        for gap_id in gap_ids
    }
    return any(
        gap.get("gap_type") == "hypothesis_discrimination"
        and gap.get("challenge_selected") is True
        and str(gap.get("gap_id", "")) in executable_gap_ids
        for gap in _eligible_causal_gaps(causal_gaps)
    )


def _record_matches_causal_gap_scope(
    record: ActionRecord,
    gap: Mapping[str, Any],
    *,
    action_kind: str | None = None,
) -> bool:
    """Match a prior Action to the same Gap and Python-owned target scope."""

    if record.status != "completed":
        return False
    if action_kind is not None and record.action.kind != action_kind:
        return False
    gap_id = str(gap.get("gap_id", "")).strip()
    if not gap_id or record.action.scope.get("causal_gap_id") != gap_id:
        return False
    target_scope = gap.get("target_scope", {})
    if not isinstance(target_scope, Mapping):
        return True
    return all(
        record.action.scope.get(str(key)) == value
        for key, value in target_scope.items()
    )


def _actions_for_causal_gap_stage(
    gap: Mapping[str, Any],
    action_records: list[ActionRecord],
) -> frozenset[str]:
    """Expose observation first and reasoning refresh only after Evidence gain.

    This prevents a broad process-mechanism Question from re-advertising defect,
    history, and exposure Actions when Python has already identified one exact
    discriminator for an alternative Lane.
    """

    allowed = frozenset(str(item) for item in gap.get("allowed_actions", []))
    preferred = str(gap.get("preferred_action", "")).strip()
    refresh = str(gap.get("refresh_action", "")).strip()
    if not preferred or preferred not in allowed:
        return allowed
    observation_indexes = [
        index
        for index, record in enumerate(action_records)
        if _record_matches_causal_gap_scope(
            record,
            gap,
            action_kind=preferred,
        )
    ]
    if not observation_indexes:
        return frozenset({preferred})
    if refresh and refresh in allowed:
        latest_observation = observation_indexes[-1]
        if any(
            index > latest_observation
            and _record_matches_causal_gap_scope(
                record,
                gap,
                action_kind=refresh,
            )
            for index, record in enumerate(action_records)
        ):
            return frozenset()
        return frozenset({refresh})
    return frozenset()


def _required_unavailable_evidence_ids(
    evidence: list[Evidence],
) -> tuple[str, ...]:
    return tuple(
        item.evidence_id
        for item in evidence
        if item.evidence_type == EvidenceType.DATA_MISSING.value
        and item.metadata.get("required_for_confirmation") is True
    )


def _action_has_new_relevant_evidence(
    record: ActionRecord,
    *,
    earlier_records: list[ActionRecord],
    links: list[QuestionEvidenceLink],
) -> bool:
    earlier_evidence_ids = {
        evidence_id
        for earlier in earlier_records
        for evidence_id in earlier.produced_evidence_ids
    }
    new_ids = set(record.produced_evidence_ids) - earlier_evidence_ids
    if not new_ids:
        return False
    return any(
        link.action_id == record.action.action_id
        and link.evidence_id in new_ids
        and link.relation
        in {
            QuestionEvidenceRelation.SUPPORTS.value,
            QuestionEvidenceRelation.CONTRADICTS.value,
        }
        for link in links
    )


def _reasoning_refresh_has_unconsumed_gap_evidence(
    action: InvestigationAction,
    *,
    target_questions: list[InvestigationQuestion],
    action_records: list[ActionRecord],
    links: list[QuestionEvidenceLink],
) -> bool:
    """Allow one reasoning refresh after a scoped observation adds Evidence.

    Reasoning composes existing Evidence and normally does not create a new raw
    Evidence ID.  Applying the collection-action no-gain rule to it would block
    the exact refresh needed to consume a completed causal-gap observation.
    """

    if action.kind != ActionKind.RUN_RCA_REASONING.value:
        return False
    gap_id = str(action.scope.get("causal_gap_id", "")).strip()
    if not gap_id:
        return False
    observations = [
        (index, record)
        for index, record in enumerate(action_records)
        if record.status == "completed"
        and record.action.kind != ActionKind.RUN_RCA_REASONING.value
        and record.action.scope.get("causal_gap_id") == gap_id
    ]
    if not observations:
        return False
    latest_observation_index, latest_observation = observations[-1]
    if any(
        index > latest_observation_index
        and record.status == "completed"
        and record.action.kind == ActionKind.RUN_RCA_REASONING.value
        and record.action.scope.get("causal_gap_id") == gap_id
        for index, record in enumerate(action_records)
    ):
        return False
    earlier_records = action_records[:latest_observation_index]
    earlier_evidence_ids = {
        evidence_id
        for record in earlier_records
        for evidence_id in record.produced_evidence_ids
    }
    new_ids = set(latest_observation.produced_evidence_ids) - earlier_evidence_ids
    if not new_ids:
        return False
    if not links:
        # A Python-bound causal Gap already establishes relevance for legacy
        # callers that predate persisted QuestionEvidenceLink records.
        return True
    target_question_ids = {question.question_id for question in target_questions}
    return any(
        link.action_id == latest_observation.action.action_id
        and link.question_id in target_question_ids
        and link.evidence_id in new_ids
        and link.relation
        in {
            QuestionEvidenceRelation.SUPPORTS.value,
            QuestionEvidenceRelation.CONTRADICTS.value,
        }
        for link in links
    )


def _reasoning_round_is_allowed(
    action_kind: str,
    *,
    gap: Mapping[str, Any] | None,
    findings: list[AgentFinding],
    action_records: list[ActionRecord],
    links: list[QuestionEvidenceLink],
) -> bool:
    """Permit round three only to consume new discriminative Evidence.

    The first two RCA rounds retain the existing behavior.  A third round is
    exposed only after a completed observation for a Python-generated
    hypothesis-discrimination Gap produced relevant Evidence that no earlier
    RCA Finding could have consumed.  This keeps the extra round bounded and
    prevents a generic ``run_rca_reasoning`` loop.
    """

    if action_kind != ActionKind.RUN_RCA_REASONING.value:
        return True
    reasoning_rounds = sum(
        record.status == "completed"
        and record.action.kind == ActionKind.RUN_RCA_REASONING.value
        for record in action_records
    )
    if reasoning_rounds < _UNCONDITIONAL_CANDIDATE_GENERATION_ROUNDS:
        return True
    if reasoning_rounds >= _MAX_CANDIDATE_GENERATION_ROUNDS:
        return False
    if gap is None or gap.get("gap_type") != "hypothesis_discrimination":
        return False
    if not str(gap.get("gap_id", "")).strip():
        return False
    observations = [
        record
        for record in action_records
        if record.action.kind != ActionKind.RUN_RCA_REASONING.value
        and _record_matches_causal_gap_scope(record, gap)
    ]
    if not observations:
        return False
    latest_observation = observations[-1]
    consumed_evidence_ids = {
        evidence_id
        for finding in findings
        if finding.agent == AgentKind.RCA_REASONING.value
        for evidence_id in finding.evidence_ids
    }
    unconsumed_ids = (
        set(latest_observation.produced_evidence_ids) - consumed_evidence_ids
    )
    if not unconsumed_ids:
        return False
    if not links:
        # The authoritative typed Gap already establishes relevance for legacy
        # states that predate persisted QuestionEvidenceLink records.
        return True
    return any(
        link.action_id == latest_observation.action.action_id
        and link.evidence_id in unconsumed_ids
        and link.relation
        in {
            QuestionEvidenceRelation.SUPPORTS.value,
            QuestionEvidenceRelation.CONTRADICTS.value,
        }
        for link in links
    )


def _conditional_third_round_available(
    *,
    action_records: list[ActionRecord],
    causal_gap_ids_by_action: Mapping[str, list[str]],
) -> bool:
    reasoning_round_count = sum(
        record.status == "completed"
        and record.action.kind == ActionKind.RUN_RCA_REASONING.value
        for record in action_records
    )
    return (
        reasoning_round_count == _UNCONDITIONAL_CANDIDATE_GENERATION_ROUNDS
        and ActionKind.RUN_RCA_REASONING.value in causal_gap_ids_by_action
    )


def _consecutive_no_gain_count(
    action_records: list[ActionRecord],
    links: list[QuestionEvidenceLink],
) -> int:
    completed = [record for record in action_records if record.status == "completed"]
    count = 0
    for index in range(len(completed) - 1, -1, -1):
        if _action_has_new_relevant_evidence(
            completed[index],
            earlier_records=completed[:index],
            links=links,
        ):
            break
        count += 1
    return count


def _compact_evidence(evidence: Evidence) -> dict[str, Any]:
    return {
        "evidence_id": evidence.evidence_id,
        "source_type": evidence.source_type,
        "summary": evidence.summary,
        "evidence_type": evidence.evidence_type,
        "source_agent": evidence.source_agent,
        "observation": evidence.observation,
        "confidence": evidence.confidence,
    }


def _strict_outcome(
    decision: PlannerDecision,
    *,
    decision_proposed_by: str = "qwen",
    question_updates_source: str | None = None,
    action_value_assessments: list[ActionValueAssessment] | None = None,
) -> PlannerDecisionOutcome:
    """Project the legacy strict path into the new outcome contract."""

    reviews = [
        QuestionUpdateReview(
            decision_id=decision.decision_id,
            disposition=QuestionUpdateDisposition.ACCEPTED.value,
            reason_code=QuestionUpdateReasonCode.ACCEPTED.value,
            reason=(
                f"QuestionUpdate {update.question_id} passed the strict "
                "PlannerDecision contract."
            ),
            update_index=index,
            question_id=update.question_id,
            claimed_status=update.status,
        )
        for index, update in enumerate(decision.question_updates)
    ]
    return PlannerDecisionOutcome(
        decision=decision,
        question_update_reviews=reviews,
        raw_question_update_count=len(decision.question_updates),
        decision_proposed_by=decision_proposed_by,
        question_updates_source=(
            question_updates_source
            if decision.question_updates
            else None
        ),
        action_value_assessments=list(action_value_assessments or []),
    )


def _validate_reviewed_stop_boundary(
    outcome: PlannerDecisionOutcome,
    *,
    questions: list[InvestigationQuestion],
) -> None:
    decision = outcome.decision
    if decision.decision_type != DecisionType.STOP.value:
        return
    projected_status = {
        question.question_id: question.status for question in questions
    }
    for update in decision.question_updates:
        projected_status[update.question_id] = update.status
    open_question_ids = sorted(
        question_id
        for question_id, status in projected_status.items()
        if status == EvidenceGapStatus.OPEN.value
    )
    if not open_question_ids:
        return
    if decision.stop_reason == StopReason.GOAL_SATISFIED.value:
        raise InvestigationValidationError(
            "a goal_satisfied stop cannot leave open investigation questions: "
            f"{open_question_ids}"
        )
    if decision.stop_reason == StopReason.DATA_UNAVAILABLE.value:
        # A blocked stop may preserve open Questions when every model-supplied
        # unavailable claim was rejected by the evidence gate. This keeps the
        # stop auditable without converting an unsupported claim into terminal
        # Question state; the run evaluator will mark stop_correct=False until
        # typed DATA_MISSING Evidence or a capability notice is present.
        if outcome.raw_question_update_count > 0 and not decision.question_updates:
            return
        raise InvestigationValidationError(
            "a data_unavailable stop must terminally mark every unavailable "
            f"investigation question: {open_question_ids}"
        )


def _commit_python_goal_satisfied_transition(
    outcome: PlannerDecisionOutcome,
    *,
    open_questions: list[InvestigationQuestion],
    reference_updates: list[QuestionUpdate],
) -> PlannerDecisionOutcome:
    """Commit an Evidence-Gate-owned terminal transition after a Qwen stop.

    Qwen owns the decision to stop. Python owns Question state and may replace
    model-authored deltas only when its deterministic Evidence Gate can
    terminally update every currently open Question. Partial reference updates
    are deliberately ignored so the normal validator still fails closed.
    """

    decision = outcome.decision
    if (
        decision.decision_type != DecisionType.STOP.value
        or decision.goal_status != GoalStatus.SATISFIED.value
        or decision.stop_reason != StopReason.GOAL_SATISFIED.value
    ):
        return outcome
    open_question_ids = {question.question_id for question in open_questions}
    reference_question_ids = {update.question_id for update in reference_updates}
    if not open_question_ids or reference_question_ids != open_question_ids:
        return outcome
    python_owned_decision = replace(
        decision,
        reason=(
            "Qwen selected the goal_satisfied stop boundary. The Python "
            "Evidence Gate committed the terminal Question transitions without "
            f"changing Evidence or conclusion level. Qwen rationale: {decision.reason}"
        ),
        question_updates=list(reference_updates),
    )
    return _strict_outcome(
        python_owned_decision,
        decision_proposed_by="qwen",
        question_updates_source="python_evidence_gate",
    )


def _protect_authoritative_causal_gap_questions(
    outcome: PlannerDecisionOutcome,
    *,
    questions: list[InvestigationQuestion],
    causal_gaps: list[dict[str, Any]],
) -> PlannerDecisionOutcome:
    """Reject Qwen terminal updates for Questions with authoritative Gaps.

    QuestionUpdate review is intentionally isolated from the core Planner
    decision.  A legal Action must not be retried merely because Qwen also tried
    to close a Question whose causal investigation is still Python-governed.
    """

    protected_kinds = {
        str(gap.get("question_kind", ""))
        for gap in causal_gaps
        if str(gap.get("question_kind", "")).strip()
    }
    protected_question_ids = {
        question.question_id
        for question in questions
        if question.question_kind in protected_kinds
    }
    rejected_ids = {
        update.question_id
        for update in outcome.decision.question_updates
        if update.question_id in protected_question_ids
    }
    if not rejected_ids:
        return outcome
    kept_updates = [
        update
        for update in outcome.decision.question_updates
        if update.question_id not in rejected_ids
    ]
    reviews = [
        replace(
            review,
            disposition=QuestionUpdateDisposition.REJECTED.value,
            reason_code=(
                QuestionUpdateReasonCode.INSUFFICIENT_EVIDENCE_COVERAGE.value
            ),
            reason=(
                f"QuestionUpdate {review.question_id} was rejected because the "
                "current authoritative RCA Finding still contains a causal "
                "Evidence Gap for that Question. Python retains the open state."
            ),
        )
        if review.question_id in rejected_ids
        and review.disposition == QuestionUpdateDisposition.ACCEPTED.value
        else review
        for review in outcome.question_update_reviews
    ]
    return replace(
        outcome,
        decision=replace(outcome.decision, question_updates=kept_updates),
        question_update_reviews=reviews,
        question_updates_source="qwen" if kept_updates else None,
    )


@dataclass(frozen=True)
class QwenNextActionPlanner:
    """Select one legal next Agent action or stop after the latest observation."""

    llm_client: LLMClient
    fallback_policy: InvestigationPolicy = field(default_factory=InvestigationPolicy)
    registry: Mapping[str, ActionDefinition] = field(
        default_factory=lambda: dict(LLM_REACT_ACTION_REGISTRY)
    )
    prompt_version: str = "v1"

    def __post_init__(self) -> None:
        if self.llm_client is None:
            raise ModelValidationError("Qwen Next-action Planner requires an LLM client")
        if not isinstance(self.fallback_policy, InvestigationPolicy):
            raise ModelValidationError("fallback_policy must be an InvestigationPolicy")
        if not isinstance(self.prompt_version, str) or not self.prompt_version.strip():
            raise ModelValidationError("prompt_version must be a non-empty string")
        if set(self.registry) != LLM_REACT_EXECUTABLE_ACTION_KINDS:
            raise ModelValidationError(
                "Qwen Next-action Planner registry must contain exactly the executable "
                "Batch 20.9.3 actions"
            )
        for kind, definition in self.registry.items():
            expected = LLM_REACT_ACTION_REGISTRY[kind]
            if not isinstance(definition, ActionDefinition) or definition != expected:
                raise ModelValidationError(
                    f"Qwen Next-action Planner registry definition is invalid: {kind}"
                )
        object.__setattr__(
            self,
            "registry",
            MappingProxyType(dict(self.registry)),
        )

    def decide(
        self,
        *,
        goal: InvestigationGoal,
        questions: list[InvestigationQuestion],
        findings: list[AgentFinding],
        action_records: list[ActionRecord],
        tool_call_count: int,
        evidence: list[Evidence] | None = None,
        evidence_ids: list[str] | None = None,
        question_evidence_links: list[QuestionEvidenceLink] | None = None,
        capability_notices: list[CapabilityNotice] | None = None,
        hypotheses: list[Hypothesis] | None = None,
        prior_decisions: list[PlannerDecision] | None = None,
        critical_contradictions: list[str] | None = None,
        authoritative_rca_finding_id: str | None = None,
        investigation_gain_history: list[InvestigationGainRecord] | None = None,
    ) -> PlannerDecision:
        """Preserve the strict compatibility path until Supervisor integration."""

        return self._decide(
            goal=goal,
            questions=questions,
            findings=findings,
            action_records=action_records,
            tool_call_count=tool_call_count,
            evidence=evidence,
            evidence_ids=evidence_ids,
            question_evidence_links=question_evidence_links,
            capability_notices=capability_notices,
            hypotheses=hypotheses,
            prior_decisions=prior_decisions,
            critical_contradictions=critical_contradictions,
            authoritative_rca_finding_id=authoritative_rca_finding_id,
            investigation_gain_history=investigation_gain_history,
            review_question_updates=False,
        ).decision

    def decide_with_review(
        self,
        *,
        goal: InvestigationGoal,
        questions: list[InvestigationQuestion],
        findings: list[AgentFinding],
        action_records: list[ActionRecord],
        tool_call_count: int,
        evidence: list[Evidence] | None = None,
        evidence_ids: list[str] | None = None,
        question_evidence_links: list[QuestionEvidenceLink] | None = None,
        capability_notices: list[CapabilityNotice] | None = None,
        hypotheses: list[Hypothesis] | None = None,
        prior_decisions: list[PlannerDecision] | None = None,
        critical_contradictions: list[str] | None = None,
        authoritative_rca_finding_id: str | None = None,
        investigation_gain_history: list[InvestigationGainRecord] | None = None,
    ) -> PlannerDecisionOutcome:
        """Return a core decision with independently reviewed update claims."""

        return self._decide(
            goal=goal,
            questions=questions,
            findings=findings,
            action_records=action_records,
            tool_call_count=tool_call_count,
            evidence=evidence,
            evidence_ids=evidence_ids,
            question_evidence_links=question_evidence_links,
            capability_notices=capability_notices,
            hypotheses=hypotheses,
            prior_decisions=prior_decisions,
            critical_contradictions=critical_contradictions,
            authoritative_rca_finding_id=authoritative_rca_finding_id,
            investigation_gain_history=investigation_gain_history,
            review_question_updates=True,
        )

    def _decide(
        self,
        *,
        goal: InvestigationGoal,
        questions: list[InvestigationQuestion],
        findings: list[AgentFinding],
        action_records: list[ActionRecord],
        tool_call_count: int,
        evidence: list[Evidence] | None,
        evidence_ids: list[str] | None,
        question_evidence_links: list[QuestionEvidenceLink] | None,
        capability_notices: list[CapabilityNotice] | None,
        hypotheses: list[Hypothesis] | None,
        prior_decisions: list[PlannerDecision] | None,
        critical_contradictions: list[str] | None,
        authoritative_rca_finding_id: str | None,
        investigation_gain_history: list[InvestigationGainRecord] | None,
        review_question_updates: bool,
    ) -> PlannerDecisionOutcome:
        """Ask Qwen for one core decision, retrying only invalid core output."""

        normalized_evidence = list(evidence or [])
        explicit_evidence_ids = list(evidence_ids or [])
        normalized_hypotheses = list(hypotheses or [])
        links_provided = question_evidence_links is not None
        normalized_question_evidence_links = list(question_evidence_links or [])
        normalized_capability_notices = list(capability_notices or [])
        normalized_prior_decisions = list(prior_decisions or [])
        normalized_gain_history = list(investigation_gain_history or [])
        contradictions = list(critical_contradictions or [])
        causal_gaps = _authoritative_causal_gaps(
            findings,
            authoritative_rca_finding_id,
        )
        authoritative_finding = next(
            (
                item
                for item in findings
                if item.finding_id == authoritative_rca_finding_id
                and item.agent == AgentKind.RCA_REASONING.value
            ),
            None,
        )
        alternative_search_status = str(
            authoritative_finding.details.get("alternative_search_status", "not_searched")
            if authoritative_finding is not None
            else "not_searched"
        )
        competition_requirement = str(
            authoritative_finding.details.get(
                "competition_requirement",
                "not_evaluated",
            )
            if authoritative_finding is not None
            else "not_evaluated"
        )
        competition_axes = list(
            authoritative_finding.details.get("competition_axes", [])
            if authoritative_finding is not None
            and isinstance(
                authoritative_finding.details.get("competition_axes", []),
                list,
            )
            else []
        )
        candidate_challenges = list(
            authoritative_finding.details.get("candidate_challenges", [])
            if authoritative_finding is not None
            and isinstance(authoritative_finding.details.get("candidate_challenges", []), list)
            else []
        )
        candidate_competition_failed = (
            authoritative_finding is not None
            and str(authoritative_finding.details.get("competition_status", ""))
            == "failed"
        )
        self._validate_runtime_inputs(
            goal=goal,
            questions=questions,
            findings=findings,
            action_records=action_records,
            tool_call_count=tool_call_count,
            evidence=normalized_evidence,
            evidence_ids=explicit_evidence_ids,
            hypotheses=normalized_hypotheses,
            prior_decisions=normalized_prior_decisions,
            critical_contradictions=contradictions,
            question_evidence_links=normalized_question_evidence_links,
            capability_notices=normalized_capability_notices,
        )
        available_evidence_ids = self._available_evidence_ids(
            evidence=normalized_evidence,
            explicit_evidence_ids=explicit_evidence_ids,
            findings=findings,
            action_records=action_records,
        )
        open_questions = [
            question
            for question in questions
            if question.status == EvidenceGapStatus.OPEN.value
        ]
        question_context = self._question_context(
            questions=open_questions,
            links=normalized_question_evidence_links,
            action_records=action_records,
        )
        legal_action_targets = self._legal_action_targets(
            questions=open_questions,
            question_context=question_context,
            findings=findings,
            action_records=action_records,
            causal_gaps=causal_gaps,
            question_evidence_links=normalized_question_evidence_links,
        )
        causal_gap_ids_by_action = self._legal_causal_gap_ids_by_action(
            questions=open_questions,
            findings=findings,
            action_records=action_records,
            causal_gaps=causal_gaps,
            question_evidence_links=normalized_question_evidence_links,
        )
        if investigation_gain_history is None:
            normalized_gain_history = list(
                derive_investigation_gain_history(
                    action_records=action_records,
                    evidence=normalized_evidence,
                    links=normalized_question_evidence_links,
                )
            )
        gap_by_id = {
            str(gap.get("gap_id", "")): gap
            for gap in causal_gaps
            if str(gap.get("gap_id", "")).strip()
        }
        action_value_options: list[dict[str, Any]] = []
        for action_kind, gap_ids in causal_gap_ids_by_action.items():
            definition = self.registry[action_kind]
            for gap_id in gap_ids:
                gap = gap_by_id.get(gap_id, {})
                target_scope = gap.get("target_scope", {})
                scope = dict(goal.known_facts or {"goal_id": goal.goal_id})
                if isinstance(target_scope, Mapping):
                    scope.update(dict(target_scope))
                scope["causal_gap_id"] = gap_id
                candidate_id = str(gap.get("candidate_id", "")).strip()
                discriminator_kind = str(
                    gap.get("discriminator_kind", "")
                ).strip()
                if candidate_id:
                    scope["candidate_id"] = candidate_id
                if discriminator_kind:
                    scope["discriminator_kind"] = discriminator_kind
                action_value_options.append(
                    {
                        "option_id": f"{action_kind}:{gap_id}",
                        "action_kind": action_kind,
                        "source": definition.agent,
                        "scope": scope,
                        "gap": gap,
                    }
                )
        action_value_assessments = list(
            assess_action_values(
                options=action_value_options,
                action_records=action_records,
                gain_history=normalized_gain_history,
                remaining_tool_budget=max(0, goal.max_tool_calls - tool_call_count),
                evidence=normalized_evidence,
                competition_requirement=competition_requirement,
                competition_axes=competition_axes,
            )
        )
        high_value_assessments = list(
            high_value_action_assessments(action_value_assessments)
        )
        if high_value_assessments:
            active_priority = min(
                int(gap_by_id[item.gap_id].get("priority", 3))
                for item in high_value_assessments
                if item.gap_id in gap_by_id
            )
            high_value_assessments = [
                item
                for item in high_value_assessments
                if item.gap_id in gap_by_id
                and int(gap_by_id[item.gap_id].get("priority", 3))
                == active_priority
            ]
        decision_critical_unavailable_gain = any(
            gain.gain_type == InvestigationGainType.STATE_GAIN.value
            and gain.reason_code
            == InvestigationGainReasonCode.UNAVAILABLE_SOURCE.value
            and gain.gap_id is not None
            and gain.gap_id in gap_by_id
            for gain in normalized_gain_history
        )
        had_causal_action_options = bool(causal_gap_ids_by_action)
        if had_causal_action_options:
            high_gap_ids_by_action: dict[str, list[str]] = {}
            for assessment in high_value_assessments:
                if assessment.gap_id is not None:
                    high_gap_ids_by_action.setdefault(
                        assessment.action_kind,
                        [],
                    ).append(assessment.gap_id)
            causal_gap_ids_by_action = {
                action_kind: list(dict.fromkeys(gap_ids))
                for action_kind, gap_ids in high_gap_ids_by_action.items()
            }
            legal_action_targets = {
                action_kind: target_ids
                for action_kind, target_ids in legal_action_targets.items()
                if action_kind in causal_gap_ids_by_action
            }
        if (
            candidate_competition_failed
            and not causal_gap_ids_by_action.get(
                ActionKind.RUN_RCA_REASONING.value,
            )
        ):
            # A Candidate-provider/validation failure is not itself new
            # ranking Evidence.  The broad Question capability can otherwise
            # re-advertise generic RCA Reasoning with the exact scope that just
            # failed.  Only a Python-owned, high-value causal Gap may authorize
            # a bounded competition-repair round.
            legal_action_targets = {
                action_kind: target_ids
                for action_kind, target_ids in legal_action_targets.items()
                if action_kind != ActionKind.RUN_RCA_REASONING.value
            }
        executable_discrimination_gap = (
            _has_executable_hypothesis_discrimination_gap(
                causal_gaps,
                causal_gap_ids_by_action,
            )
        )
        required_unavailable_ids = _required_unavailable_evidence_ids(
            normalized_evidence
        )
        if (
            authoritative_finding is not None
            and required_unavailable_ids
            and authoritative_finding.details.get("conclusion_status")
            == "insufficient_evidence"
            and not executable_discrimination_gap
        ):
            return _strict_outcome(
                PlannerDecision(
                    decision_id=self._next_baseline_decision_id(
                        goal=goal,
                        prior_decisions=normalized_prior_decisions,
                    ),
                    goal_id=goal.goal_id,
                    decision_type=DecisionType.STOP.value,
                    reason=(
                        "Python stopped after the authoritative RCA Evidence Gate "
                        "confirmed that a required, non-recoverable source is "
                        "unavailable: "
                        + ", ".join(required_unavailable_ids)
                    ),
                    goal_status=GoalStatus.BLOCKED.value,
                    proposed_conclusion_level=ConclusionLevel.INCONCLUSIVE.value,
                    stop_reason=StopReason.DATA_UNAVAILABLE.value,
                    question_updates=self._terminal_question_updates(
                        open_questions=open_questions,
                        findings=findings,
                        available_evidence_ids=available_evidence_ids,
                        question_evidence_links=normalized_question_evidence_links,
                    ),
                ),
                decision_proposed_by="python_runtime",
                question_updates_source="python_evidence_gate",
                action_value_assessments=action_value_assessments,
            )
        conditional_third_round_available = _conditional_third_round_available(
            action_records=action_records,
            causal_gap_ids_by_action=causal_gap_ids_by_action,
        )
        if (
            (
                len(action_records) >= min(
                    goal.max_steps,
                    MAX_CROSS_DOMAIN_ACTIONS,
                )
                and not conditional_third_round_available
            )
            or tool_call_count >= goal.max_tool_calls
        ):
            open_questions = [
                question
                for question in questions
                if question.status == EvidenceGapStatus.OPEN.value
            ]
            return _strict_outcome(
                PlannerDecision(
                    decision_id=self._next_baseline_decision_id(
                        goal=goal,
                        prior_decisions=normalized_prior_decisions,
                    ),
                    goal_id=goal.goal_id,
                    decision_type=DecisionType.STOP.value,
                    reason=(
                        "The Python runtime budget boundary was reached before "
                        "another Qwen decision could be requested."
                    ),
                    goal_status=GoalStatus.BUDGET_EXHAUSTED.value,
                    proposed_conclusion_level=ConclusionLevel.INCONCLUSIVE.value,
                    stop_reason=StopReason.BUDGET_EXHAUSTED.value,
                    question_updates=self._terminal_question_updates(
                        open_questions=open_questions,
                        findings=findings,
                        available_evidence_ids=available_evidence_ids,
                        question_evidence_links=normalized_question_evidence_links,
                    ),
                ),
                decision_proposed_by="python_runtime",
                question_updates_source="python_evidence_gate",
                action_value_assessments=action_value_assessments,
            )
        if had_causal_action_options and not high_value_assessments:
            relevant_gap_ids = {
                assessment.gap_id
                for assessment in action_value_assessments
                if assessment.gap_id is not None
            }
            unavailable_gain = any(
                gain.gain_type == InvestigationGainType.STATE_GAIN.value
                and gain.reason_code
                == InvestigationGainReasonCode.UNAVAILABLE_SOURCE.value
                and gain.gap_id is not None
                and gain.gap_id in relevant_gap_ids
                for gain in normalized_gain_history
            )
            return _strict_outcome(
                PlannerDecision(
                    decision_id=self._next_baseline_decision_id(
                        goal=goal,
                        prior_decisions=normalized_prior_decisions,
                    ),
                    goal_id=goal.goal_id,
                    decision_type=DecisionType.STOP.value,
                    reason=(
                        "Python found no remaining high-value Action after a "
                        "decision-critical Evidence source became unavailable."
                        if unavailable_gain
                        else "Python found no remaining Action that can change "
                        "Candidate ranking or the Confirmation Gate."
                    ),
                    goal_status=GoalStatus.BLOCKED.value,
                    proposed_conclusion_level=ConclusionLevel.INCONCLUSIVE.value,
                    stop_reason=(
                        StopReason.DATA_UNAVAILABLE.value
                        if unavailable_gain
                        else StopReason.NO_HIGH_VALUE_ACTION.value
                    ),
                    # InvestigationFinalizer owns the terminal Question projection.
                    question_updates=[],
                ),
                decision_proposed_by="python_runtime",
                action_value_assessments=action_value_assessments,
            )
        baseline = self._baseline_decision(
            goal=goal,
            questions=questions,
            findings=findings,
            action_records=action_records,
            tool_call_count=tool_call_count,
            available_evidence_ids=available_evidence_ids,
            question_evidence_links=normalized_question_evidence_links,
            prior_decisions=normalized_prior_decisions,
            critical_contradictions=contradictions,
        )
        active_causal_gaps = _active_causal_gaps(causal_gaps)
        bounded_causal_investigation = any(
            gap.get("challenge_selected") is True
            or bool(gap.get("target_scope"))
            or any(
                record.action.scope.get("causal_gap_id") == gap.get("gap_id")
                for record in action_records
            )
            for gap in active_causal_gaps
        ) or candidate_competition_failed
        if bounded_causal_investigation and not legal_action_targets:
            return _strict_outcome(
                PlannerDecision(
                    decision_id=self._next_baseline_decision_id(
                        goal=goal,
                        prior_decisions=normalized_prior_decisions,
                    ),
                    goal_id=goal.goal_id,
                    decision_type=DecisionType.STOP.value,
                    reason=(
                        "Candidate competition failed without an executable "
                        "Python-bound discriminator; the failure stays isolated "
                        "from Planner/orchestration fallback and the result remains "
                        "inconclusive."
                        if candidate_competition_failed
                        else (
                            "A decision-critical source is unavailable and no "
                            "high-value registered Action remains."
                            if decision_critical_unavailable_gain
                            else "No high-value registered Action remains for "
                            "the authoritative causal Evidence Gaps; the result "
                            "stays inconclusive."
                        )
                    ),
                    goal_status=GoalStatus.BLOCKED.value,
                    proposed_conclusion_level=ConclusionLevel.INCONCLUSIVE.value,
                    stop_reason=(
                        StopReason.DATA_UNAVAILABLE.value
                        if decision_critical_unavailable_gain
                        else StopReason.NO_HIGH_VALUE_ACTION.value
                    ),
                    # InvestigationFinalizer owns the terminal Question projection.
                    question_updates=[],
                ),
                decision_proposed_by="python_runtime",
                action_value_assessments=action_value_assessments,
            )
        if (
            candidate_competition_failed
            and set(legal_action_targets) == {ActionKind.RUN_RCA_REASONING.value}
            and bool(
                causal_gap_ids_by_action.get(
                    ActionKind.RUN_RCA_REASONING.value,
                )
            )
        ):
            action_kind = ActionKind.RUN_RCA_REASONING.value
            definition = self.registry[action_kind]
            decision = PlannerDecision(
                decision_id=self._next_baseline_decision_id(
                    goal=goal,
                    prior_decisions=normalized_prior_decisions,
                ),
                goal_id=goal.goal_id,
                decision_type=DecisionType.ACT.value,
                reason=(
                    "Python selected the only legal bounded action: one final "
                    "RCA Reasoning refresh may repair the isolated candidate "
                    "competition failure without changing orchestration mode."
                ),
                goal_status=GoalStatus.IN_PROGRESS.value,
                proposed_conclusion_level=ConclusionLevel.CANDIDATE.value,
                next_action=InvestigationAction(
                    action_id=(
                        f"{goal.goal_id}:candidate-competition-repair:"
                        f"{len(action_records) + 1}"
                    ),
                    kind=action_kind,
                    agent=definition.agent,
                    reason="Retry the bounded Qwen candidate competition once.",
                    inputs=dict(goal.known_facts),
                    scope=dict(goal.known_facts or {"goal_id": goal.goal_id}),
                ),
                target_question_ids=legal_action_targets[action_kind],
            )
            decision = self._bind_causal_gap_scope(
                decision,
                causal_gap_ids_by_action=causal_gap_ids_by_action,
                causal_gaps=causal_gaps,
            )
            self._validate_candidate(
                decision,
                goal=goal,
                questions=questions,
                findings=findings,
                action_records=action_records,
                tool_call_count=tool_call_count,
                available_evidence_ids=available_evidence_ids,
                question_evidence_links=normalized_question_evidence_links,
                prior_decisions=normalized_prior_decisions,
                legal_action_targets=legal_action_targets,
                causal_gap_ids_by_action=causal_gap_ids_by_action,
                causal_gaps=causal_gaps,
                critical_contradictions=contradictions,
                investigation_gain_history=normalized_gain_history,
            )
            return _strict_outcome(
                decision,
                decision_proposed_by="python_runtime",
                action_value_assessments=action_value_assessments,
            )
        if causal_gap_ids_by_action and len(legal_action_targets) == 1:
            action_kind = next(iter(legal_action_targets))
            gap_ids = causal_gap_ids_by_action.get(action_kind, [])
            if len(gap_ids) == 1:
                target_question_ids = legal_action_targets[action_kind]
                definition = self.registry[action_kind]
                action_index = len(action_records) + 1
                scope = dict(goal.known_facts or {"goal_id": goal.goal_id})
                scope["causal_gap_id"] = gap_ids[0]
                selected_gap = next(
                    (
                        gap
                        for gap in causal_gaps
                        if gap.get("gap_id") == gap_ids[0]
                    ),
                    None,
                )
                if selected_gap is not None:
                    scope.update(dict(selected_gap.get("target_scope", {})))
                    selected_candidate_id = str(
                        selected_gap.get("candidate_id", "")
                    ).strip()
                    selected_discriminator = str(
                        selected_gap.get("discriminator_kind", "")
                    ).strip()
                    if selected_candidate_id:
                        scope["candidate_id"] = selected_candidate_id
                    if selected_discriminator:
                        scope["discriminator_kind"] = selected_discriminator
                decision = PlannerDecision(
                    decision_id=self._next_baseline_decision_id(
                        goal=goal,
                        prior_decisions=normalized_prior_decisions,
                    ),
                    goal_id=goal.goal_id,
                    decision_type=DecisionType.ACT.value,
                    reason=(
                        "Python selected the only legal Action for the current "
                        f"causal Evidence Gap {gap_ids[0]}."
                    ),
                    goal_status=GoalStatus.IN_PROGRESS.value,
                    proposed_conclusion_level=ConclusionLevel.CANDIDATE.value,
                    next_action=InvestigationAction(
                        action_id=f"{goal.goal_id}:causal-gap-action:{action_index}",
                        kind=action_kind,
                        agent=definition.agent,
                        reason=f"Fill deterministic causal Evidence Gap {gap_ids[0]}.",
                        inputs=dict(goal.known_facts),
                        scope=scope,
                    ),
                    target_question_ids=target_question_ids,
                )
                self._validate_candidate(
                    decision,
                    goal=goal,
                    questions=questions,
                    findings=findings,
                    action_records=action_records,
                    tool_call_count=tool_call_count,
                    available_evidence_ids=available_evidence_ids,
                    question_evidence_links=normalized_question_evidence_links,
                    prior_decisions=normalized_prior_decisions,
                    legal_action_targets=legal_action_targets,
                    causal_gap_ids_by_action=causal_gap_ids_by_action,
                    causal_gaps=causal_gaps,
                    critical_contradictions=contradictions,
                    investigation_gain_history=normalized_gain_history,
                )
                return _strict_outcome(
                    decision,
                    decision_proposed_by="python_runtime",
                    action_value_assessments=action_value_assessments,
                )
        advertised_actions = frozenset(legal_action_targets)
        prompt_lane_ids = _known_causal_lane_ids(findings)
        prompt_causal_gaps = [
            _compact_causal_gap_for_prompt(
                gap,
                active_lane_ids=prompt_lane_ids,
            )
            for gap in causal_gaps
        ]
        prompt_candidate_challenges = [
            _compact_audit_mapping_for_prompt(item)
            for item in candidate_challenges
            if isinstance(item, Mapping)
        ][:_MAX_PROMPT_AUDIT_ITEMS]
        prompt_question_context = _compact_question_context_for_prompt(
            question_context
        )
        relevant_evidence_ids = {
            evidence_id
            for packet in prompt_question_context
            for evidence_id in (
                packet["linked_evidence"]["supports"]
                + packet["linked_evidence"]["contradicts"]
                + packet["linked_evidence"]["context"]
                + packet["linked_evidence"]["unavailable"]
            )
        }
        validation_errors: list[str] = []
        validation_error_categories: list[str] = []
        call_retry_count = 0
        failed_provider_call_attempt_count = 0
        open_question_ids = [
            question.question_id for question in open_questions
        ]
        terminal_question_ids = [
            update.question_id for update in baseline.question_updates
        ]
        complete_python_terminal_transition = (
            bool(open_question_ids)
            and set(terminal_question_ids) == set(open_question_ids)
            and not causal_gap_ids_by_action
        )
        goal_satisfied_stop_contract = {
            "currently_open_question_ids": open_question_ids,
            "python_terminal_transition_available": (
                complete_python_terminal_transition
            ),
            "python_terminal_question_ids": (
                terminal_question_ids
                if complete_python_terminal_transition
                else []
            ),
            "executable_causal_gap_ids": (
                sorted(
                    {
                        gap_id
                        for gap_ids in causal_gap_ids_by_action.values()
                        for gap_id in gap_ids
                    }
                )
                if causal_gap_ids_by_action
                else []
            ),
            "boundary": (
                "A goal_satisfied stop is invalid while any listed Question "
                "remains open unless Python has a complete Evidence-Gate-owned "
                "terminal transition, or while an executable causal Evidence Gap "
                "remains. Qwen chooses the stop boundary but must not reproduce "
                "Python-owned Question state."
            ),
        }

        for attempt in range(1, _OUTPUT_ATTEMPTS + 1):
            previous_validation_feedback = None
            if validation_errors:
                last_error = validation_errors[-1]
                goal_satisfied_repair = (
                    "goal_satisfied stop cannot leave open investigation questions"
                    in last_error
                )
                previous_validation_feedback = {
                    "category": validation_error_categories[-1],
                    "message": last_error,
                    "must_repair_before_resubmission": True,
                    "output_fields_exactly": list(
                        _PLANNER_DECISION_OUTPUT_FIELDS
                    ),
                    "input_only_fields_never_copy_to_output": list(
                        _PLANNER_INPUT_ONLY_FIELDS
                    ),
                    "legal_target_question_ids_by_action": legal_action_targets,
                    "known_causal_lane_ids": list(prompt_lane_ids),
                    "lane_aware_action_kinds": sorted(_LANE_AWARE_ACTION_KINDS),
                    "lane_selection_rule": (
                        "When multiple causal lanes are discovered, every lane-aware "
                        "Action must copy exactly one known lane_id into next_action.scope."
                    ),
                    "causal_evidence_gaps": prompt_causal_gaps,
                    "alternative_search_status": alternative_search_status,
                    "candidate_challenges": prompt_candidate_challenges,
                    "legal_causal_gap_ids_by_action": causal_gap_ids_by_action,
                    "action_value_assessments": [
                        item.to_dict() for item in high_value_assessments
                    ],
                    "question_action_capabilities": {
                        question.question_id: [
                            action_kind
                            for action_kind, target_ids in legal_action_targets.items()
                            if question.question_id in target_ids
                        ]
                        for question in open_questions
                    },
                    "must_terminally_update_question_ids": (
                        open_question_ids if goal_satisfied_repair else []
                    ),
                    "python_terminal_transition_available": (
                        complete_python_terminal_transition
                    ),
                    "python_terminal_question_ids": (
                        terminal_question_ids
                        if complete_python_terminal_transition
                        else []
                    ),
                    "repair_instruction": (
                        "Return exactly the fields in output_fields_exactly and never "
                        "copy an input_only_fields_never_copy_to_output field into "
                        "the decision. If the repaired decision keeps a "
                        "goal_satisfied stop, set question_updates=[]; Python will "
                        "commit the Evidence-Gate-owned terminal transition only when "
                        "python_terminal_transition_available is true. Otherwise choose "
                        "a legal action or a different stop boundary. For an act "
                        "decision, choose one Action key from "
                        "legal_target_question_ids_by_action and copy only Question "
                        "IDs listed for that Action."
                    ),
                }
            request = LLMRequest(
                agent=AgentKind.PLANNER.value,
                prompt_name="next_action_planner",
                prompt_version=self.prompt_version,
                payload={
                    "goal": goal.to_dict(),
                    "questions": [question.to_dict() for question in questions],
                    "findings": [
                        _compact_finding(
                            finding,
                            active_lane_ids=prompt_lane_ids,
                        )
                        for finding in findings
                    ],
                    "evidence": [
                        _compact_evidence(item)
                        for item in normalized_evidence
                        if item.evidence_id in relevant_evidence_ids
                    ],
                    "question_context": prompt_question_context,
                    "capability_notices": [
                        notice.to_dict() for notice in normalized_capability_notices
                    ],
                    "available_evidence_ids": sorted(relevant_evidence_ids),
                    "hypotheses": [
                        hypothesis.to_dict() for hypothesis in normalized_hypotheses
                    ],
                    "action_history": [
                        _compact_action_record(record) for record in action_records
                    ],
                    "prior_decision_ids": [
                        decision.decision_id
                        for decision in normalized_prior_decisions
                    ],
                    "critical_contradictions": contradictions,
                    "budget": {
                        "completed_steps": len(action_records),
                        "max_steps": min(
                            goal.max_steps,
                            MAX_CROSS_DOMAIN_ACTIONS,
                        ),
                        "tool_call_count": tool_call_count,
                        "max_tool_calls": goal.max_tool_calls,
                        "evidence_gated_final_reasoning_refresh_available": (
                            conditional_third_round_available
                        ),
                    },
                    "allowed_actions": [
                        {
                            "kind": definition.kind,
                            "agent": definition.agent,
                            "required_finding_agents": list(
                                definition.required_finding_agents
                            ),
                        }
                        for definition in self.registry.values()
                        if definition.kind in advertised_actions
                    ],
                    "question_action_capabilities": {
                        question.question_id: [
                            action_kind
                            for action_kind, target_ids in legal_action_targets.items()
                            if question.question_id in target_ids
                        ]
                        for question in open_questions
                    },
                    "legal_target_question_ids_by_action": legal_action_targets,
                    "known_causal_lane_ids": list(prompt_lane_ids),
                    "lane_aware_action_kinds": sorted(_LANE_AWARE_ACTION_KINDS),
                    "lane_selection_rule": (
                        "When multiple causal lanes are discovered, every lane-aware "
                        "Action must copy exactly one known lane_id into next_action.scope."
                    ),
                    "causal_evidence_gaps": prompt_causal_gaps,
                    "alternative_search_status": alternative_search_status,
                    "candidate_challenges": prompt_candidate_challenges,
                    "legal_causal_gap_ids_by_action": causal_gap_ids_by_action,
                    "action_value_assessments": [
                        item.to_dict() for item in high_value_assessments
                    ],
                    "deterministic_planner_decision": replace(
                        baseline,
                        question_updates=[],
                    ).to_dict(),
                    "goal_satisfied_stop_contract": goal_satisfied_stop_contract,
                    "output_attempt": attempt,
                    "previous_validation_error": (
                        validation_errors[-1] if validation_errors else None
                    ),
                    "previous_validation_feedback": previous_validation_feedback,
                },
                temperature=0.0,
            )
            response = None
            while True:
                try:
                    response = self.llm_client.complete_json(request)
                except LLMCallError as exc:
                    failed_provider_call_attempt_count += exc.call_attempt_count
                    if (
                        call_retry_count < _CALL_RETRIES
                        and _is_retryable_call_error(exc)
                    ):
                        call_retry_count += 1
                        continue
                    raise LLMCallError(
                        "Qwen Next-action Planner call failed after its bounded retry",
                        status_code=exc.status_code,
                        provider_code=exc.provider_code,
                        provider_message=exc.provider_message,
                        request_id=exc.request_id,
                        failure_category=exc.failure_category,
                        call_attempt_count=failed_provider_call_attempt_count,
                    ) from exc
                except LLMOutputValidationError as exc:
                    message = str(exc).strip() or type(exc).__name__
                    validation_errors.append(message)
                    validation_error_categories.append(_OUTPUT_PARSE_ERROR)
                break
            if response is None:
                continue
            try:
                sanitized_response = _strip_exact_planner_input_echoes(
                    response.data,
                    request_payload=request.payload,
                )
                outcome = (
                    review_qwen_planner_output(
                        sanitized_response,
                        questions=questions,
                        available_evidence_ids=available_evidence_ids,
                        question_evidence_links=(
                            normalized_question_evidence_links
                            if links_provided
                            else None
                        ),
                        capability_notices=(
                            normalized_capability_notices
                            if capability_notices is not None
                            else None
                        ),
                    )
                    if review_question_updates
                    else _strict_outcome(
                        PlannerDecision.from_dict(
                            sanitized_response,
                            allow_legacy_question_updates=False,
                        )
                    )
                )
                if review_question_updates:
                    outcome = _protect_authoritative_causal_gap_questions(
                        outcome,
                        questions=questions,
                        causal_gaps=causal_gaps,
                    )
                candidate = outcome.decision
                candidate = self._bind_causal_gap_scope(
                    candidate,
                    causal_gap_ids_by_action=causal_gap_ids_by_action,
                    causal_gaps=causal_gaps,
                )
                if candidate is not outcome.decision:
                    outcome = replace(outcome, decision=candidate)
                if review_question_updates:
                    outcome = _commit_python_goal_satisfied_transition(
                        outcome,
                        open_questions=open_questions,
                        reference_updates=baseline.question_updates,
                    )
                    candidate = outcome.decision
                self._validate_candidate(
                    candidate,
                    goal=goal,
                    questions=questions,
                    findings=findings,
                    action_records=action_records,
                    tool_call_count=tool_call_count,
                    available_evidence_ids=available_evidence_ids,
                    question_evidence_links=normalized_question_evidence_links,
                    prior_decisions=normalized_prior_decisions,
                    legal_action_targets=legal_action_targets,
                    causal_gap_ids_by_action=causal_gap_ids_by_action,
                    causal_gaps=causal_gaps,
                    critical_contradictions=contradictions,
                    investigation_gain_history=normalized_gain_history,
                )
                if review_question_updates:
                    _validate_reviewed_stop_boundary(
                        outcome,
                        questions=questions,
                    )
                return replace(
                    outcome,
                    action_value_assessments=action_value_assessments,
                )
            except (
                InvestigationValidationError,
                LLMOutputValidationError,
                KeyError,
                TypeError,
            ) as exc:
                message = str(exc).strip() or type(exc).__name__
                validation_errors.append(message)
                validation_error_categories.append(
                    _OUTPUT_PARSE_ERROR
                    if isinstance(exc, LLMOutputValidationError)
                    else _CORE_DECISION_VALIDATION_ERROR
                )

        raise QwenNextActionPlannerError(
            validation_errors,
            validation_error_categories,
            goal_id=goal.goal_id,
            completed_steps=len(action_records),
            tool_call_count=tool_call_count,
        )

    def _baseline_decision(
        self,
        *,
        goal: InvestigationGoal,
        questions: list[InvestigationQuestion],
        findings: list[AgentFinding],
        action_records: list[ActionRecord],
        tool_call_count: int,
        available_evidence_ids: set[str],
        question_evidence_links: list[QuestionEvidenceLink],
        prior_decisions: list[PlannerDecision],
        critical_contradictions: list[str],
    ) -> PlannerDecision:
        policy_decision = self.fallback_policy.next_action(
            goal=goal,
            findings=findings,
            action_records=action_records,
            tool_call_count=tool_call_count,
            critical_contradictions=critical_contradictions,
        )
        decision_id = self._next_baseline_decision_id(
            goal=goal,
            prior_decisions=prior_decisions,
        )
        open_questions = [
            question
            for question in questions
            if question.status == EvidenceGapStatus.OPEN.value
        ]
        action = policy_decision.next_action
        if (
            action is not None
            and action.kind in self.registry
            and open_questions
        ):
            scope = dict(action.scope or goal.known_facts or {"goal_id": goal.goal_id})
            if action.kind in {
                ActionKind.INSPECT_DEFECT_PATTERN.value,
                ActionKind.VALIDATE_SHARED_DEFECT_PATTERN.value,
                ActionKind.INSPECT_FDC_SPC.value,
                ActionKind.VALIDATE_HISTORICAL_CASE.value,
            }:
                known_lane_ids = _known_causal_lane_ids(findings)
                if len(known_lane_ids) > 1:
                    scope.setdefault("lane_id", known_lane_ids[0])
            bounded_action = replace(action, scope=scope)
            missing_evidence_groups = _missing_groups_for_questions(
                open_questions,
                question_evidence_links,
            )
            target = None
            for question in open_questions:
                try:
                    validate_action_for_questions(
                        bounded_action,
                        [question],
                        missing_evidence_groups=missing_evidence_groups,
                    )
                except QuestionCapabilityError:
                    continue
                target = question
                break
            if target is None:
                return PlannerDecision(
                    decision_id=decision_id,
                    goal_id=goal.goal_id,
                    decision_type=DecisionType.STOP.value,
                    reason=(
                        "No registered Action can target the remaining typed "
                        "Questions."
                    ),
                    goal_status=GoalStatus.BLOCKED.value,
                    proposed_conclusion_level=ConclusionLevel.INCONCLUSIVE.value,
                    stop_reason=StopReason.NO_ALLOWED_ACTION.value,
                    question_updates=self._terminal_question_updates(
                        open_questions=open_questions,
                        findings=findings,
                        available_evidence_ids=available_evidence_ids,
                        question_evidence_links=question_evidence_links,
                    ),
                )
            return PlannerDecision(
                decision_id=decision_id,
                goal_id=goal.goal_id,
                decision_type=DecisionType.ACT.value,
                reason=action.reason,
                goal_status=GoalStatus.IN_PROGRESS.value,
                proposed_conclusion_level=policy_decision.conclusion_level,
                next_action=bounded_action,
                target_question_ids=[target.question_id],
            )

        if action is not None:
            goal_status = GoalStatus.BLOCKED.value
            stop_reason = StopReason.NO_ALLOWED_ACTION.value
        else:
            goal_status = policy_decision.goal_status
            stop_reason = (
                policy_decision.stop_reason
                or StopReason.NO_ALLOWED_ACTION.value
            )
        if not open_questions and stop_reason != StopReason.BUDGET_EXHAUSTED.value:
            goal_status = GoalStatus.SATISFIED.value
            stop_reason = StopReason.GOAL_SATISFIED.value
        question_updates = self._terminal_question_updates(
            open_questions=open_questions,
            findings=findings,
            available_evidence_ids=available_evidence_ids,
            question_evidence_links=question_evidence_links,
        )
        return PlannerDecision(
            decision_id=decision_id,
            goal_id=goal.goal_id,
            decision_type=DecisionType.STOP.value,
            reason=(
                "The deterministic planner reference reached an explicit "
                f"{stop_reason} boundary."
            ),
            goal_status=goal_status,
            proposed_conclusion_level=policy_decision.conclusion_level,
            stop_reason=stop_reason,
            question_updates=question_updates,
        )

    @staticmethod
    def _terminal_question_updates(
        *,
        open_questions: list[InvestigationQuestion],
        findings: list[AgentFinding],
        available_evidence_ids: set[str],
        question_evidence_links: list[QuestionEvidenceLink] | None = None,
    ) -> list[QuestionUpdate]:
        if question_evidence_links is not None:
            updates: list[QuestionUpdate] = []
            for question in open_questions:
                question_links = [
                    link
                    for link in question_evidence_links
                    if link.question_id == question.question_id
                ]
                applicable_links = [
                    link
                    for link in question_links
                    if link.relation
                    in {
                        QuestionEvidenceRelation.SUPPORTS.value,
                        QuestionEvidenceRelation.CONTRADICTS.value,
                        QuestionEvidenceRelation.CONTEXT.value,
                    }
                ]
                capability = QUESTION_CAPABILITY_REGISTRY.get(
                    str(question.question_kind)
                )
                satisfied_groups = {
                    link.matched_evidence_group
                    for link in applicable_links
                    if link.relation == QuestionEvidenceRelation.SUPPORTS.value
                }
                required_groups = (
                    set(capability.closure_evidence_groups)
                    if capability is not None
                    else set()
                )
                if required_groups <= satisfied_groups and applicable_links:
                    evidence_ids = sorted(
                        {link.evidence_id for link in applicable_links}
                    )
                    answer = " ".join(
                        finding.summary.strip()
                        for finding in findings
                        if finding.summary.strip()
                    ) or (
                        "The applicable Evidence supports this Question: "
                        + ", ".join(evidence_ids)
                    )
                    updates.append(
                        QuestionUpdate(
                            question_id=question.question_id,
                            status=EvidenceGapStatus.CLOSED.value,
                            answer=answer,
                            evidence_ids=evidence_ids,
                            unavailable_reason=None,
                        )
                    )
                else:
                    unavailable = next(
                        (
                            link
                            for link in question_links
                            if link.relation
                            == QuestionEvidenceRelation.UNAVAILABLE.value
                        ),
                        None,
                    )
                    if unavailable is not None:
                        updates.append(
                            QuestionUpdate(
                                question_id=question.question_id,
                                status=EvidenceGapStatus.UNAVAILABLE.value,
                                answer=None,
                                evidence_ids=[unavailable.evidence_id],
                                unavailable_reason=(
                                    "The required Evidence source is unavailable."
                                ),
                            )
                        )
            return updates
        if available_evidence_ids:
            answer = " ".join(
                finding.summary.strip()
                for finding in findings
                if finding.summary.strip()
            )
            if not answer:
                answer = (
                    "The available observations are recorded by Evidence IDs "
                    f"{', '.join(sorted(available_evidence_ids))}."
                )
            return [
                QuestionUpdate(
                    question_id=question.question_id,
                    status=EvidenceGapStatus.CLOSED.value,
                    answer=answer,
                    evidence_ids=sorted(available_evidence_ids),
                    unavailable_reason=None,
                )
                for question in open_questions
            ]
        return [
            QuestionUpdate(
                question_id=question.question_id,
                status=EvidenceGapStatus.UNAVAILABLE.value,
                answer=None,
                evidence_ids=[],
                unavailable_reason=(
                    "The deterministic investigation reached its stop boundary "
                    "without any Evidence that can answer this question."
                ),
            )
            for question in open_questions
        ]

    @staticmethod
    def _next_baseline_decision_id(
        *,
        goal: InvestigationGoal,
        prior_decisions: list[PlannerDecision],
    ) -> str:
        used_ids = {decision.decision_id for decision in prior_decisions}
        index = len(prior_decisions) + 1
        while True:
            candidate = f"{goal.goal_id}:decision:{index}"
            if candidate not in used_ids:
                return candidate
            index += 1

    @staticmethod
    def _available_evidence_ids(
        *,
        evidence: list[Evidence],
        explicit_evidence_ids: list[str],
        findings: list[AgentFinding],
        action_records: list[ActionRecord],
    ) -> set[str]:
        return {
            *explicit_evidence_ids,
            *(item.evidence_id for item in evidence),
            *(
                evidence_id
                for finding in findings
                for evidence_id in finding.evidence_ids
            ),
            *(
                evidence_id
                for record in action_records
                for evidence_id in record.produced_evidence_ids
            ),
        }

    @staticmethod
    def _validate_runtime_inputs(
        *,
        goal: InvestigationGoal,
        questions: list[InvestigationQuestion],
        findings: list[AgentFinding],
        action_records: list[ActionRecord],
        tool_call_count: int,
        evidence: list[Evidence],
        evidence_ids: list[str],
        hypotheses: list[Hypothesis],
        prior_decisions: list[PlannerDecision],
        critical_contradictions: list[str],
        question_evidence_links: list[QuestionEvidenceLink],
        capability_notices: list[CapabilityNotice],
    ) -> None:
        if not isinstance(goal, InvestigationGoal):
            raise ModelValidationError("goal must be an InvestigationGoal")
        if not isinstance(questions, list) or not questions:
            raise ModelValidationError("questions must be a non-empty list")
        question_ids: list[str] = []
        for question in questions:
            if not isinstance(question, InvestigationQuestion):
                raise ModelValidationError(
                    "questions must contain InvestigationQuestion instances"
                )
            if question.goal_id != goal.goal_id:
                raise ModelValidationError("questions must reference the current goal")
            question_ids.append(question.question_id)
        if len(question_ids) != len(set(question_ids)):
            raise ModelValidationError("questions must not contain duplicate ids")
        if len(questions) > MAX_INITIAL_QUESTIONS:
            raise ModelValidationError(
                f"questions must not exceed {MAX_INITIAL_QUESTIONS} total items"
            )
        if not isinstance(findings, list) or any(
            not isinstance(finding, AgentFinding) for finding in findings
        ):
            raise ModelValidationError("findings must contain AgentFinding instances")
        if not isinstance(action_records, list) or any(
            not isinstance(record, ActionRecord) for record in action_records
        ):
            raise ModelValidationError("action_records must contain ActionRecord instances")
        if type(tool_call_count) is not int or tool_call_count < 0:
            raise ModelValidationError("tool_call_count must be a non-negative integer")
        if not isinstance(evidence, list) or any(
            not isinstance(item, Evidence) for item in evidence
        ):
            raise ModelValidationError("evidence must contain Evidence instances")
        _validate_string_list(evidence_ids, "evidence_ids")
        if not isinstance(hypotheses, list) or any(
            not isinstance(hypothesis, Hypothesis) for hypothesis in hypotheses
        ):
            raise ModelValidationError("hypotheses must contain Hypothesis instances")
        if not isinstance(prior_decisions, list) or any(
            not isinstance(decision, PlannerDecision) for decision in prior_decisions
        ):
            raise ModelValidationError(
                "prior_decisions must contain PlannerDecision instances"
            )
        decision_ids = [decision.decision_id for decision in prior_decisions]
        if len(decision_ids) != len(set(decision_ids)):
            raise ModelValidationError("prior_decisions must not contain duplicate ids")
        _validate_string_list(
            critical_contradictions,
            "critical_contradictions",
        )
        if not isinstance(question_evidence_links, list) or any(
            not isinstance(link, QuestionEvidenceLink)
            for link in question_evidence_links
        ):
            raise ModelValidationError(
                "question_evidence_links must contain QuestionEvidenceLink instances"
            )
        if not isinstance(capability_notices, list) or any(
            not isinstance(notice, CapabilityNotice) for notice in capability_notices
        ):
            raise ModelValidationError(
                "capability_notices must contain CapabilityNotice instances"
            )

    def _validate_candidate(
        self,
        candidate: PlannerDecision,
        *,
        goal: InvestigationGoal,
        questions: list[InvestigationQuestion],
        findings: list[AgentFinding],
        action_records: list[ActionRecord],
        tool_call_count: int,
        available_evidence_ids: set[str],
        question_evidence_links: list[QuestionEvidenceLink],
        prior_decisions: list[PlannerDecision],
        legal_action_targets: dict[str, list[str]],
        causal_gap_ids_by_action: dict[str, list[str]],
        causal_gaps: list[dict[str, Any]],
        critical_contradictions: list[str],
        investigation_gain_history: list[InvestigationGainRecord],
    ) -> None:
        if candidate.goal_id != goal.goal_id:
            raise InvestigationValidationError("Qwen changed the active goal_id")
        if candidate.decision_id in {
            decision.decision_id for decision in prior_decisions
        }:
            raise InvestigationValidationError("Qwen reused an earlier decision_id")

        existing_questions = {
            question.question_id: question for question in questions
        }
        new_question_ids = {question.question_id for question in candidate.new_questions}
        if new_question_ids & set(existing_questions):
            raise InvestigationValidationError(
                "new_questions cannot reuse an existing question_id"
            )
        if len(existing_questions) + len(candidate.new_questions) > MAX_INITIAL_QUESTIONS:
            raise InvestigationValidationError(
                f"the investigation cannot exceed {MAX_INITIAL_QUESTIONS} questions"
            )
        source_lot_id = _normalized_lot_id(goal.known_facts.get("lot_id"))
        for question in candidate.new_questions:
            capability = QUESTION_CAPABILITY_REGISTRY.get(str(question.question_kind))
            if capability is None or not capability.supported:
                raise InvestigationValidationError(
                    "unsupported_question_kind: Qwen cannot create an unsupported "
                    f"Question kind {question.question_kind!r}"
                )
            _assert_source_lot_boundary(
                question.scope,
                source_lot_id=source_lot_id,
                label=f"new_questions[{question.question_id}].scope",
            )

        for update in candidate.question_updates:
            current = existing_questions.get(update.question_id)
            if current is None:
                raise InvestigationValidationError(
                    "question_updates can only update an existing question"
                )
            if current.status != EvidenceGapStatus.OPEN.value:
                raise InvestigationValidationError(
                    "question_updates cannot rewrite a terminal question"
                )
            if (
                update.status == EvidenceGapStatus.CLOSED.value
                and not set(update.evidence_ids) <= available_evidence_ids
            ):
                raise InvestigationValidationError(
                    "a closed question references unknown Evidence IDs"
                )
        resulting_questions = {
            **existing_questions,
            **{
                question.question_id: question
                for question in candidate.new_questions
            },
            **{
                update.question_id: replace(
                    existing_questions[update.question_id],
                    status=update.status,
                    answer=update.answer,
                    evidence_ids=list(update.evidence_ids),
                    unavailable_reason=update.unavailable_reason,
                )
                for update in candidate.question_updates
            },
        }
        for target_id in candidate.target_question_ids:
            target = resulting_questions.get(target_id)
            if target is None:
                raise InvestigationValidationError(
                    "target_question_ids must reference a current investigation question"
                )
            if target.status != EvidenceGapStatus.OPEN.value:
                raise InvestigationValidationError(
                    "an action can target only an open investigation question"
                )

        conditional_third_round_available = _conditional_third_round_available(
            action_records=action_records,
            causal_gap_ids_by_action=causal_gap_ids_by_action,
        )
        budget_exhausted = (
            (
                len(action_records) >= min(
                    goal.max_steps,
                    MAX_CROSS_DOMAIN_ACTIONS,
                )
                and not conditional_third_round_available
            )
            or tool_call_count >= goal.max_tool_calls
        )
        if candidate.decision_type == DecisionType.STOP.value:
            if candidate.target_question_ids:
                raise InvestigationValidationError(
                    "a stop decision cannot target an open question"
                )
            expected_goal_status = {
                StopReason.GOAL_SATISFIED.value: GoalStatus.SATISFIED.value,
                StopReason.CRITICAL_CONTRADICTION.value: GoalStatus.BLOCKED.value,
                StopReason.NO_ALLOWED_ACTION.value: GoalStatus.BLOCKED.value,
                StopReason.BUDGET_EXHAUSTED.value: GoalStatus.BUDGET_EXHAUSTED.value,
                StopReason.DATA_UNAVAILABLE.value: GoalStatus.BLOCKED.value,
                StopReason.NO_HIGH_VALUE_ACTION.value: GoalStatus.BLOCKED.value,
            }[str(candidate.stop_reason)]
            if candidate.goal_status != expected_goal_status:
                raise InvestigationValidationError(
                    "stop_reason_goal_status_mismatch: "
                    f"{candidate.stop_reason} requires goal_status="
                    f"{expected_goal_status}"
                )
            if (
                candidate.stop_reason == StopReason.GOAL_SATISFIED.value
                and causal_gaps
                and causal_gap_ids_by_action
            ):
                raise InvestigationValidationError(
                    "goal_satisfied stop cannot bypass executable causal Evidence "
                    "Gaps; select a legal causal-gap Action before stopping"
                )
            if (
                candidate.stop_reason == StopReason.NO_ALLOWED_ACTION.value
                and legal_action_targets
            ):
                raise InvestigationValidationError(
                    "no_allowed_action stop is invalid while Python still exposes "
                    "a legal Action"
                )
            if (
                candidate.stop_reason == StopReason.CRITICAL_CONTRADICTION.value
                and not critical_contradictions
            ):
                raise InvestigationValidationError(
                    "critical_contradiction stop requires a Python-supplied "
                    "critical contradiction"
                )
            if budget_exhausted and (
                candidate.goal_status != GoalStatus.BUDGET_EXHAUSTED.value
                or candidate.stop_reason != StopReason.BUDGET_EXHAUSTED.value
            ):
                raise InvestigationValidationError(
                    "an exhausted runtime budget requires an explicit budget_exhausted stop"
                )
            if (
                not budget_exhausted
                and candidate.stop_reason == StopReason.BUDGET_EXHAUSTED.value
            ):
                raise InvestigationValidationError(
                    "Qwen cannot claim budget_exhausted before the runtime limit"
                )
            return

        if budget_exhausted:
            raise InvestigationValidationError(
                "Qwen cannot select an action after the runtime budget is exhausted"
            )
        action = candidate.next_action
        if action is None:
            raise InvestigationValidationError("an act decision requires next_action")
        known_lane_ids = _known_causal_lane_ids(findings)
        if len(known_lane_ids) > 1 and action.kind in _LANE_AWARE_ACTION_KINDS:
            selected_lane_id = str(
                action.scope.get("lane_id") or action.inputs.get("lane_id") or ""
            ).strip()
            if not selected_lane_id:
                raise InvestigationValidationError(
                    "lane_aware_action_requires_lane_id: select a discovered causal Lane"
                )
            if selected_lane_id not in set(known_lane_ids):
                raise InvestigationValidationError(
                    "lane_aware_action_references_unknown_lane_id"
                )
        targeted_questions = [
            resulting_questions[question_id]
            for question_id in candidate.target_question_ids
            if question_id in resulting_questions
        ]
        missing_evidence_groups = _missing_groups_for_questions(
            targeted_questions,
            question_evidence_links,
        )
        causal_gap_id = str(action.scope.get("causal_gap_id", "")).strip()
        allowed_gap_ids = set(causal_gap_ids_by_action.get(action.kind, []))
        gap_bound = bool(causal_gap_id and causal_gap_id in allowed_gap_ids)
        if len(allowed_gap_ids) > 1 and not causal_gap_id:
            raise InvestigationValidationError(
                "causal_gap_selection_required: the selected Action can fill "
                "multiple legal causal Evidence Gaps, so Qwen must explicitly "
                "select one causal_gap_id"
            )
        if causal_gap_id and not gap_bound:
            raise InvestigationValidationError(
                "next_action.scope.causal_gap_id is not legal for the selected Action"
            )
        if gap_bound:
            selected_gap = next(
                (
                    gap
                    for gap in causal_gaps
                    if gap.get("gap_id") == causal_gap_id
                ),
                None,
            )
            expected_scope = (
                dict(selected_gap.get("target_scope", {}))
                if selected_gap is not None
                else {}
            )
            scope_mismatch = {
                key: value
                for key, value in expected_scope.items()
                if action.scope.get(key) != value
            }
            if scope_mismatch:
                raise InvestigationValidationError(
                    "causal_gap_scope_mismatch: the selected Action must retain "
                    f"Python-owned target scope {scope_mismatch}"
                )
        # This is deliberately atomic: one incompatible target rejects the
        # complete Decision instead of silently dropping that target.
        validate_action_for_questions(
            action,
            targeted_questions,
            missing_evidence_groups=(None if gap_bound else missing_evidence_groups),
        )
        self._validate_no_gain_boundary(
            action=action,
            investigation_gain_history=investigation_gain_history,
        )
        definition = self.registry.get(action.kind)
        if definition is None:
            raise InvestigationValidationError(
                f"action is not in the executable allowlist: {action.kind}"
            )
        if action.agent != definition.agent:
            raise InvestigationValidationError(
                f"action {action.kind} must be executed by Agent {definition.agent}"
            )
        if action.max_attempts != 1:
            raise InvestigationValidationError(
                "a Qwen-selected action must use max_attempts=1"
            )
        finding_agents = {finding.agent for finding in findings}
        missing_agents = set(definition.required_finding_agents) - finding_agents
        if (
            action.kind == ActionKind.INSPECT_DEFECT_PATTERN.value
            and source_lot_id is None
            and AgentKind.MES.value not in finding_agents
        ):
            missing_agents.add(AgentKind.MES.value)
        if missing_agents:
            raise InvestigationValidationError(
                f"action {action.kind} is missing prerequisite Findings from "
                f"{sorted(missing_agents)}"
            )
        legal_targets = set(legal_action_targets.get(action.kind, []))
        if not legal_targets or not set(candidate.target_question_ids) <= legal_targets:
            raise InvestigationValidationError(
                "Qwen selected an Action or Question target outside the current "
                "Python legal-action projection"
            )
        if (
            action.kind == ActionKind.RUN_RCA_REASONING.value
            and goal.intent
            in {
                InvestigationIntent.ROOT_CAUSE.value,
                InvestigationIntent.FULL_RCA.value,
            }
            and not any(
                record.action.kind
                == ActionKind.VALIDATE_SHARED_DEFECT_PATTERN.value
                and record.status == "completed"
                for record in action_records
            )
        ):
            raise InvestigationValidationError(
                "run_rca_reasoning requires a completed "
                "validate_shared_defect_pattern action for root-cause goals"
            )
        if not action.scope:
            raise InvestigationValidationError(
                "a Qwen-selected action requires a non-empty stable scope"
            )
        _assert_source_lot_boundary(
            action.inputs,
            source_lot_id=source_lot_id,
            label="next_action.inputs",
        )
        _assert_source_lot_boundary(
            action.scope,
            source_lot_id=source_lot_id,
            label="next_action.scope",
        )
        if not set(action.required_evidence_ids) <= available_evidence_ids:
            raise InvestigationValidationError(
                "next_action.required_evidence_ids references unknown Evidence"
            )
        prior_action_ids = {record.action.action_id for record in action_records}
        if action.action_id in prior_action_ids:
            raise InvestigationValidationError("Qwen reused an earlier action_id")
        if action.deduplication_key in {
            record.action.deduplication_key for record in action_records
        }:
            raise InvestigationValidationError(
                "Qwen repeated an already attempted Action + Scope"
            )
        if (
            action.kind == ActionKind.FIND_SHARED_EXPOSURE.value
            and not gap_bound
            and any(
                record.action.kind == ActionKind.FIND_SHARED_EXPOSURE.value
                for record in action_records
            )
        ):
            raise InvestigationValidationError(
                "find_shared_exposure is single-use within one bounded investigation"
            )

    def _advertised_actions(
        self,
        questions: list[InvestigationQuestion],
    ) -> frozenset[str]:
        """Advertise only Actions that can target at least one open Question."""

        advertised: set[str] = set()
        for question in questions:
            capability = QUESTION_CAPABILITY_REGISTRY.get(str(question.question_kind))
            if capability is not None and capability.supported:
                advertised.update(
                    action_kind
                    for action_kind in capability.allowed_actions
                    if action_kind in self.registry
                )
        return frozenset(advertised)

    def _legal_action_targets(
        self,
        *,
        questions: list[InvestigationQuestion],
        question_context: list[dict[str, Any]],
        findings: list[AgentFinding],
        action_records: list[ActionRecord],
        causal_gaps: list[dict[str, Any]],
        question_evidence_links: list[QuestionEvidenceLink],
    ) -> dict[str, list[str]]:
        """Project the state-aware Question/Action matrix owned by Python.

        The static capability registry says which Actions may ever answer a
        Question kind.  This projection is narrower: it also removes Actions
        that cannot fill a *currently* missing Evidence group or whose
        Specialist prerequisites are not yet available.  Qwen receives this
        exact matrix on both the first request and any repair request.
        """

        context_by_id = {
            str(packet["question_id"]): packet for packet in question_context
        }
        finding_agents = {finding.agent for finding in findings}
        known_lane_ids = _known_causal_lane_ids(findings)
        completed_kinds = {
            record.action.kind
            for record in action_records
            if record.status == "completed"
        }
        targets_by_action: dict[str, list[str]] = {}
        eligible_gaps = _eligible_causal_gaps(causal_gaps)
        if not eligible_gaps:
            for question in questions:
                capability = capability_for_question(question)
                packet = context_by_id[question.question_id]
                missing = set(packet["missing_evidence_groups"])
                for action_kind in sorted(capability.allowed_actions):
                    definition = self.registry.get(action_kind)
                    if definition is None:
                        continue
                    if not set(definition.required_finding_agents) <= finding_agents:
                        continue
                    if (
                        action_kind == ActionKind.FIND_SHARED_EXPOSURE.value
                        and action_kind in completed_kinds
                    ):
                        continue
                    if not _reasoning_round_is_allowed(
                        action_kind,
                        gap=None,
                        findings=findings,
                        action_records=action_records,
                        links=question_evidence_links,
                    ):
                        continue
                    contribution = capability.contribution_for(action_kind)
                    if missing:
                        if not (missing & contribution):
                            continue
                    elif "hypothesis_synthesis" not in contribution:
                        continue
                    targets_by_action.setdefault(action_kind, []).append(
                        question.question_id
                    )
        gap_action_targets: list[tuple[str, list[str]]] = []
        for gap in eligible_gaps:
            question_kind = str(gap["question_kind"])
            targets = [
                question.question_id
                for question in questions
                if question.question_kind == question_kind
            ]
            if not targets:
                continue
            for action_kind in _actions_for_causal_gap_stage(gap, action_records):
                if (
                    len(known_lane_ids) > 1
                    and action_kind in _LANE_AWARE_ACTION_KINDS
                    and not str(
                        gap.get("target_scope", {}).get("lane_id", "")
                    ).strip()
                ):
                    continue
                definition = self.registry.get(str(action_kind))
                if definition is None:
                    continue
                if not set(definition.required_finding_agents) <= finding_agents:
                    continue
                if not _reasoning_round_is_allowed(
                    action_kind,
                    gap=gap,
                    findings=findings,
                    action_records=action_records,
                    links=question_evidence_links,
                ):
                    continue
                if any(
                    _record_matches_causal_gap_scope(
                        record,
                        gap,
                        action_kind=action_kind,
                    )
                    for record in action_records
                ):
                    continue
                gap_action_targets.append(
                    (str(action_kind), targets)
                )
        for action_kind, targets in gap_action_targets:
            targets_by_action.setdefault(action_kind, []).extend(targets)
        return {
            action_kind: list(dict.fromkeys(target_ids))
            for action_kind, target_ids in sorted(targets_by_action.items())
        }

    def _legal_causal_gap_ids_by_action(
        self,
        *,
        questions: list[InvestigationQuestion],
        findings: list[AgentFinding],
        action_records: list[ActionRecord],
        causal_gaps: list[dict[str, Any]],
        question_evidence_links: list[QuestionEvidenceLink],
    ) -> dict[str, list[str]]:
        finding_agents = {finding.agent for finding in findings}
        known_lane_ids = _known_causal_lane_ids(findings)
        open_kinds = {question.question_kind for question in questions}
        executable_gap_actions: list[tuple[str, str]] = []
        for gap in _eligible_causal_gaps(causal_gaps):
            gap_id = str(gap["gap_id"])
            if str(gap["question_kind"]) not in open_kinds:
                continue
            for raw_action in _actions_for_causal_gap_stage(gap, action_records):
                action_kind = str(raw_action)
                if (
                    len(known_lane_ids) > 1
                    and action_kind in _LANE_AWARE_ACTION_KINDS
                    and not str(
                        gap.get("target_scope", {}).get("lane_id", "")
                    ).strip()
                ):
                    continue
                definition = self.registry.get(action_kind)
                if definition is None:
                    continue
                if not set(definition.required_finding_agents) <= finding_agents:
                    continue
                if not _reasoning_round_is_allowed(
                    action_kind,
                    gap=gap,
                    findings=findings,
                    action_records=action_records,
                    links=question_evidence_links,
                ):
                    continue
                if any(
                    _record_matches_causal_gap_scope(
                        record,
                        gap,
                        action_kind=action_kind,
                    )
                    for record in action_records
                ):
                    continue
                executable_gap_actions.append(
                    (action_kind, gap_id)
                )
        result: dict[str, list[str]] = {}
        for action_kind, gap_id in executable_gap_actions:
            result.setdefault(action_kind, []).append(gap_id)
        return {
            action_kind: list(dict.fromkeys(gap_ids))
            for action_kind, gap_ids in sorted(result.items())
        }

    @staticmethod
    def _bind_causal_gap_scope(
        decision: PlannerDecision,
        *,
        causal_gap_ids_by_action: dict[str, list[str]],
        causal_gaps: list[dict[str, Any]],
    ) -> PlannerDecision:
        if decision.decision_type != DecisionType.ACT.value or decision.next_action is None:
            return decision
        gap_ids = causal_gap_ids_by_action.get(decision.next_action.kind, [])
        if not gap_ids:
            return decision
        scope = dict(decision.next_action.scope)
        proposed_gap_id = str(scope.get("causal_gap_id", "")).strip()
        if len(gap_ids) == 1:
            selected_gap_id = gap_ids[0]
        elif proposed_gap_id in gap_ids:
            selected_gap_id = proposed_gap_id
        else:
            # Multiple legal Gaps represent a real investigation choice owned
            # by Qwen. Preserve a missing or invalid proposal so the strict
            # validator rejects it instead of silently choosing the first.
            return decision
        scope["causal_gap_id"] = selected_gap_id
        selected_gap = next(
            (
                gap
                for gap in causal_gaps
                if gap.get("gap_id") == selected_gap_id
            ),
            None,
        )
        if selected_gap is not None:
            scope.update(dict(selected_gap.get("target_scope", {})))
            candidate_id = str(selected_gap.get("candidate_id", "")).strip()
            discriminator_kind = str(
                selected_gap.get("discriminator_kind", "")
            ).strip()
            if candidate_id:
                scope["candidate_id"] = candidate_id
            if discriminator_kind:
                scope["discriminator_kind"] = discriminator_kind
        return replace(
            decision,
            next_action=replace(decision.next_action, scope=scope),
        )

    @staticmethod
    def _question_context(
        *,
        questions: list[InvestigationQuestion],
        links: list[QuestionEvidenceLink],
        action_records: list[ActionRecord],
    ) -> list[dict[str, Any]]:
        """Build a bounded per-Question projection for Qwen."""

        packets: list[dict[str, Any]] = []
        for question in questions:
            capability = QUESTION_CAPABILITY_REGISTRY[str(question.question_kind)]
            question_links = [
                link for link in links if link.question_id == question.question_id
            ]
            linked_evidence = {
                relation.value: sorted(
                    {
                        link.evidence_id
                        for link in question_links
                        if link.relation == relation.value
                    }
                )
                for relation in QuestionEvidenceRelation
            }
            satisfied_groups = sorted(
                {
                    link.matched_evidence_group
                    for link in question_links
                    if link.relation == QuestionEvidenceRelation.SUPPORTS.value
                }
            )
            missing_groups = sorted(
                set(capability.closure_evidence_groups) - set(satisfied_groups)
            )
            attempted = [
                {
                    "action_id": record.action.action_id,
                    "kind": record.action.kind,
                    "scope": dict(record.action.scope),
                    "status": record.status,
                    "relevant_gain": any(
                        link.action_id == record.action.action_id
                        and link.question_id == question.question_id
                        and link.relation
                        in {
                            QuestionEvidenceRelation.SUPPORTS.value,
                            QuestionEvidenceRelation.CONTRADICTS.value,
                        }
                        for link in links
                    ),
                }
                for record in action_records
                if record.action.kind in capability.allowed_actions
                and action_scope_matches_question(record.action, question)
            ]
            packets.append(
                {
                    "question_id": question.question_id,
                    "question_kind": question.question_kind,
                    "scope": dict(question.scope),
                    "linked_evidence": linked_evidence,
                    "satisfied_evidence_groups": satisfied_groups,
                    "missing_evidence_groups": missing_groups,
                    "compatible_actions": sorted(
                        capability.allowed_actions
                    ),
                    "prior_attempted_actions": attempted,
                }
            )
        return packets

    @staticmethod
    def _validate_no_gain_boundary(
        *,
        action: InvestigationAction,
        investigation_gain_history: list[InvestigationGainRecord] | None = None,
        target_questions: list[InvestigationQuestion] | None = None,
        action_records: list[ActionRecord] | None = None,
        prior_decisions: list[PlannerDecision] | None = None,
        links: list[QuestionEvidenceLink] | None = None,
    ) -> None:
        """Reject repeated work while preserving the pre-Patch-3 private API.

        Production planning always supplies persisted Investigation Gain and
        uses the exact Action fingerprint below.  The older arguments remain
        available for downstream contract callers, but they do not participate
        in the Patch 3 Planner path.
        """

        if investigation_gain_history is None:
            legacy_questions = list(target_questions or [])
            legacy_records = list(action_records or [])
            legacy_decisions = list(prior_decisions or [])
            legacy_links = list(links or [])
            if not legacy_questions or not legacy_records or not legacy_decisions:
                return
            if _reasoning_refresh_has_unconsumed_gap_evidence(
                action,
                target_questions=legacy_questions,
                action_records=legacy_records,
                links=legacy_links,
            ):
                return
            decisions_by_action_id = {
                decision.next_action.action_id: decision
                for decision in legacy_decisions
                if decision.decision_type == DecisionType.ACT.value
                and decision.next_action is not None
            }
            for question in legacy_questions:
                prior_same_direction = [
                    record
                    for record in legacy_records
                    if record.status == "completed"
                    and record.action.kind == action.kind
                    and record.action.deduplication_key != action.deduplication_key
                    and record.action.action_id in decisions_by_action_id
                    and question.question_id
                    in decisions_by_action_id[
                        record.action.action_id
                    ].target_question_ids
                    and action_scope_matches_question(record.action, question)
                ]
                if not prior_same_direction:
                    continue
                latest = prior_same_direction[-1]
                earlier_records = legacy_records[: legacy_records.index(latest)]
                if _action_has_new_relevant_evidence(
                    latest,
                    earlier_records=earlier_records,
                    links=[
                        link
                        for link in legacy_links
                        if link.question_id == question.question_id
                    ],
                ):
                    continue
                raise InvestigationValidationError(
                    "no_expected_evidence_gain: the same Question, Action family, "
                    "and compatible scope produced no relevant Evidence gain on the "
                    "previous attempt; Qwen must switch direction or stop"
                )
            return

        fingerprint = action_scope_fingerprint(
            action.kind,
            action.scope,
            source=action.agent,
        )
        if any(
            gain.action_kind == action.kind
            and gain.scope_fingerprint == fingerprint
            for gain in investigation_gain_history
        ):
            raise InvestigationValidationError(
                "exact_action_scope_already_attempted: Candidate, Lane, Gap, "
                "discriminator, Action, source, and scope must identify a new "
                "investigation option"
            )


__all__ = [
    "LLM_REACT_ACTION_REGISTRY",
    "LLM_REACT_EXECUTABLE_ACTION_KINDS",
    "QwenNextActionPlanner",
    "QwenNextActionPlannerError",
]
