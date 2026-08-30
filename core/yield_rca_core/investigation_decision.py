"""Deterministic investigation progress and next-action value assessment.

The existing causal Gap ``information_gain`` is an ex-ante static discriminator
score.  This module deliberately keeps two different concepts separate:

* ``ActionValueAssessment`` asks whether an exact Action is worth executing now.
* ``InvestigationGainRecord`` records what a completed Action actually changed.

Neither model proposes a root cause or trusts an LLM to declare progress.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from yield_rca_core.causal_investigation_models import (
    ActionDecisionImpact,
    ActionSourceAvailability,
    ActionValueAssessment,
    ActionValueTier,
    CandidateCompetitionAxis,
    CompetitionRequirement,
    InvestigationGainReasonCode,
    InvestigationGainRecord,
    InvestigationGainType,
)
from yield_rca_core.evidence_models import Evidence, EvidenceType
from yield_rca_core.investigation_models import (
    ActionKind,
    ActionRecord,
    QuestionEvidenceLink,
    QuestionEvidenceRelation,
)

_ACTION_TOOL_COSTS: dict[str, int] = {
    ActionKind.INSPECT_DEFECT_PATTERN.value: 1,
    ActionKind.VALIDATE_SHARED_DEFECT_PATTERN.value: 1,
    ActionKind.FIND_SHARED_EXPOSURE.value: 3,
    ActionKind.ASSESS_IMPACT_SCOPE.value: 1,
    ActionKind.INSPECT_FDC_SPC.value: 4,
    ActionKind.INSPECT_RECIPE_CHANGE.value: 1,
    ActionKind.VALIDATE_HISTORICAL_CASE.value: 1,
    ActionKind.RUN_RCA_REASONING.value: 0,
}

_ACTION_LLM_COSTS: dict[str, int] = {
    ActionKind.INSPECT_DEFECT_PATTERN.value: 1,
    ActionKind.VALIDATE_SHARED_DEFECT_PATTERN.value: 1,
    ActionKind.FIND_SHARED_EXPOSURE.value: 1,
    ActionKind.ASSESS_IMPACT_SCOPE.value: 1,
    ActionKind.INSPECT_FDC_SPC.value: 1,
    ActionKind.INSPECT_RECIPE_CHANGE.value: 1,
    ActionKind.VALIDATE_HISTORICAL_CASE.value: 1,
    # Candidate generation and adversarial comparison can each call the model.
    ActionKind.RUN_RCA_REASONING.value: 2,
}

_SCOPE_ENTITY_TYPES = {
    "operation": "operation",
    "equipment": "equipment",
    "chamber": "chamber",
    "recipe": "recipe",
}


def _scope_values_from_evidence(evidence: Evidence) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {key: set() for key in _SCOPE_ENTITY_TYPES}
    values["parameters"] = set()
    entity_key_by_type = {
        entity_type: key for key, entity_type in _SCOPE_ENTITY_TYPES.items()
    }
    entity_key_by_type["parameter"] = "parameters"
    for entity in evidence.entities:
        key = entity_key_by_type.get(str(entity.entity_type))
        if key is not None:
            values[key].add(str(entity.entity_id).strip())
    metadata_keys = {
        "operation": ("operation", "operation_no"),
        "equipment": ("equipment", "equipment_id"),
        "chamber": ("chamber", "chamber_id"),
        "recipe": ("recipe", "recipe_id"),
        "parameters": ("parameter", "parameter_name"),
    }
    for target_key, keys in metadata_keys.items():
        for key in keys:
            value = evidence.metadata.get(key)
            if isinstance(value, str) and value.strip():
                values[target_key].add(value.strip())
    return values


def _matching_existing_evidence_ids(
    *,
    gap: Mapping[str, Any],
    scope: Mapping[str, Any],
    evidence: Sequence[Evidence],
) -> tuple[str, ...]:
    expected_types = {
        str(item).strip()
        for item in gap.get("expected_evidence_types", [])
        if str(item).strip()
    }
    explicit_ids = {
        str(item).strip()
        for item in gap.get("evidence_ids", [])
        if str(item).strip()
    }
    target_parameters = {
        item.strip()
        for item in str(scope.get("parameters", "")).split(",")
        if item.strip()
    }
    matches: list[str] = []
    for item in evidence:
        if expected_types and str(item.evidence_type or "") not in expected_types:
            continue
        if (
            item.evidence_id in explicit_ids
            and str(gap.get("discriminator_kind", "")) == "mechanism_context"
        ):
            matches.append(item.evidence_id)
            continue
        observed = _scope_values_from_evidence(item)
        identity_match = False
        conflicted = False
        for key in _SCOPE_ENTITY_TYPES:
            target = str(scope.get(key, "")).strip()
            if not target or not observed[key]:
                continue
            if target not in observed[key]:
                conflicted = True
                break
            identity_match = True
        if conflicted:
            continue
        if target_parameters and observed["parameters"]:
            if not target_parameters & observed["parameters"]:
                continue
            identity_match = True
        # Do not treat a globally similar Evidence type as proof that this
        # exact discriminator/Lane was already observed.
        if identity_match:
            matches.append(item.evidence_id)
    return tuple(dict.fromkeys(matches))


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def action_scope_fingerprint(
    action_kind: str,
    scope: Mapping[str, Any],
    *,
    source: str | None = None,
) -> str:
    """Return one stable exact Action/source/scope fingerprint."""

    payload = {
        "action_kind": str(action_kind).strip(),
        "source": str(source or "").strip(),
        "scope": _json_value(scope),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
    return f"action-scope:{digest}"


def _record_scope_metadata(record: ActionRecord) -> dict[str, str | None]:
    scope = record.action.scope
    return {
        "candidate_id": str(scope.get("candidate_id", "")).strip() or None,
        "lane_id": str(scope.get("lane_id", "")).strip() or None,
        "gap_id": str(scope.get("causal_gap_id", "")).strip() or None,
        "discriminator_kind": (
            str(scope.get("discriminator_kind", "")).strip() or None
        ),
    }


def classify_investigation_gain(
    record: ActionRecord,
    *,
    earlier_records: Sequence[ActionRecord],
    evidence_by_id: Mapping[str, Evidence],
    links: Sequence[QuestionEvidenceLink],
    earlier_gains: Sequence[InvestigationGainRecord] = (),
    reasoning_state_changed: bool = False,
) -> InvestigationGainRecord:
    """Classify one completed Action without trusting its prose summary."""

    fingerprint = action_scope_fingerprint(
        record.action.kind,
        record.action.scope,
        source=record.action.agent,
    )
    earlier_evidence_ids = {
        evidence_id
        for earlier in earlier_records
        for evidence_id in earlier.produced_evidence_ids
    }
    new_evidence_ids = tuple(
        evidence_id
        for evidence_id in record.produced_evidence_ids
        if evidence_id not in earlier_evidence_ids
    )
    relevant_links = [
        link
        for link in links
        if link.action_id == record.action.action_id
        and link.evidence_id in set(new_evidence_ids)
    ]
    relation_types = tuple(
        dict.fromkeys(link.relation for link in relevant_links)
    )
    evidence_gain_relations = {
        QuestionEvidenceRelation.SUPPORTS.value,
        QuestionEvidenceRelation.CONTRADICTS.value,
    }
    unavailable = any(
        link.relation == QuestionEvidenceRelation.UNAVAILABLE.value
        for link in relevant_links
    ) or any(
        evidence_by_id[evidence_id].evidence_type == EvidenceType.DATA_MISSING.value
        for evidence_id in new_evidence_ids
        if evidence_id in evidence_by_id
    )
    non_discriminative = any(
        str(evidence_by_id[evidence_id].metadata.get("investigation_result", ""))
        == "non_discriminative"
        for evidence_id in new_evidence_ids
        if evidence_id in evidence_by_id
    )
    metadata = _record_scope_metadata(record)

    if any(link.relation in evidence_gain_relations for link in relevant_links):
        gain_type = InvestigationGainType.EVIDENCE_GAIN.value
        reason_code = (
            InvestigationGainReasonCode.SUPPORTING_OR_CONTRADICTING_EVIDENCE.value
        )
        reason = "New supporting or contradicting typed Evidence was recorded."
    elif reasoning_state_changed:
        gain_type = InvestigationGainType.STATE_GAIN.value
        reason_code = InvestigationGainReasonCode.REASONING_STATE_CHANGED.value
        reason = "RCA reasoning changed Candidate or Competition state."
    elif unavailable or non_discriminative:
        state_reason = (
            InvestigationGainReasonCode.UNAVAILABLE_SOURCE.value
            if unavailable
            else InvestigationGainReasonCode.NON_DISCRIMINATIVE.value
        )
        repeated_state = any(
            item.action_kind == record.action.kind
            and item.scope_fingerprint == fingerprint
            and item.gain_type == InvestigationGainType.STATE_GAIN.value
            and item.reason_code == state_reason
            for item in earlier_gains
        )
        if repeated_state:
            gain_type = InvestigationGainType.NO_GAIN.value
            reason_code = (
                InvestigationGainReasonCode.REPEATED_UNAVAILABLE_SOURCE.value
                if unavailable
                else InvestigationGainReasonCode.REPEATED_NON_DISCRIMINATIVE.value
            )
            reason = f"Repeated {state_reason} result for the same exact Action scope."
        else:
            gain_type = InvestigationGainType.STATE_GAIN.value
            reason_code = state_reason
            reason = (
                f"{state_reason} changed the investigation state without "
                "supporting or contradicting the Candidate."
            )
    else:
        gain_type = InvestigationGainType.NO_GAIN.value
        reason_code = InvestigationGainReasonCode.NO_DECISION_CHANGE.value
        reason = "The Action produced no new decision-relevant investigation state."

    return InvestigationGainRecord(
        action_id=record.action.action_id,
        action_kind=record.action.kind,
        scope_fingerprint=fingerprint,
        gain_type=gain_type,
        evidence_ids=new_evidence_ids,
        relation_types=relation_types,
        candidate_id=metadata["candidate_id"],
        lane_id=metadata["lane_id"],
        gap_id=metadata["gap_id"],
        discriminator_kind=metadata["discriminator_kind"],
        source=record.action.agent,
        reason_code=reason_code,
        reason=reason,
    )


def derive_investigation_gain_history(
    *,
    action_records: Sequence[ActionRecord],
    evidence: Sequence[Evidence],
    links: Sequence[QuestionEvidenceLink],
) -> tuple[InvestigationGainRecord, ...]:
    """Build a legacy-safe history when an older State has no persisted gains."""

    evidence_by_id = {item.evidence_id: item for item in evidence}
    gains: list[InvestigationGainRecord] = []
    completed: list[ActionRecord] = []
    for record in action_records:
        if record.status != "completed":
            continue
        gain = classify_investigation_gain(
            record,
            earlier_records=completed,
            evidence_by_id=evidence_by_id,
            links=links,
            earlier_gains=gains,
        )
        gains.append(gain)
        completed.append(record)
    return tuple(gains)


def _decision_impact(action_kind: str, gap: Mapping[str, Any]) -> str:
    if action_kind == ActionKind.RUN_RCA_REASONING.value:
        return ActionDecisionImpact.REASONING_REFRESH.value
    if str(gap.get("gap_type", "")) == "hypothesis_discrimination":
        return ActionDecisionImpact.RANKING.value
    claim = str(gap.get("claim", "")).strip()
    if claim in {
        "exposure",
        "equipment",
        "chamber",
        "operation",
        "parameter",
        "outcome",
        "mechanism",
        "temporal",
        "scope",
        "contradiction",
    }:
        return ActionDecisionImpact.CONFIRMATION.value
    if str(gap.get("question_kind", "")).strip():
        return ActionDecisionImpact.QUESTION_CLOSURE.value
    return ActionDecisionImpact.CONTEXT_ONLY.value


def _latest_reasoning_index(action_records: Sequence[ActionRecord]) -> int:
    indexes = [
        index
        for index, record in enumerate(action_records)
        if record.status == "completed"
        and record.action.kind == ActionKind.RUN_RCA_REASONING.value
    ]
    return indexes[-1] if indexes else -1


def _reasoning_refresh_has_ranking_gain(
    *,
    gap_id: str | None,
    action_records: Sequence[ActionRecord],
    gain_history: Sequence[InvestigationGainRecord],
) -> bool:
    latest_reasoning = _latest_reasoning_index(action_records)
    action_indexes = {
        record.action.action_id: index for index, record in enumerate(action_records)
    }
    return any(
        gain.gain_type == InvestigationGainType.EVIDENCE_GAIN.value
        and (gap_id is None or gain.gap_id == gap_id)
        and action_indexes.get(gain.action_id, -1) > latest_reasoning
        for gain in gain_history
    )


def assess_action_values(
    *,
    options: Sequence[Mapping[str, Any]],
    action_records: Sequence[ActionRecord],
    gain_history: Sequence[InvestigationGainRecord],
    remaining_tool_budget: int,
    evidence: Sequence[Evidence] = (),
    competition_requirement: str = CompetitionRequirement.NOT_EVALUATED.value,
    competition_axes: Sequence[str] = (),
) -> tuple[ActionValueAssessment, ...]:
    """Evaluate exact legal options; legality alone never implies value."""

    assessments: list[ActionValueAssessment] = []
    mechanism_competition_required = (
        competition_requirement == CompetitionRequirement.MECHANISM_REQUIRED.value
        or (
            competition_requirement == CompetitionRequirement.MIXED_REQUIRED.value
            and CandidateCompetitionAxis.MECHANISM.value
            in {str(item) for item in competition_axes}
        )
    )
    for index, option in enumerate(options):
        action_kind = str(option.get("action_kind", "")).strip()
        gap = option.get("gap", {})
        if not isinstance(gap, Mapping):
            gap = {}
        target_scope = gap.get("target_scope", {})
        raw_scope = option.get("scope", target_scope)
        scope = dict(raw_scope) if isinstance(raw_scope, Mapping) else {}
        gap_id = str(gap.get("gap_id", "")).strip() or None
        candidate_id = str(gap.get("candidate_id", "")).strip() or None
        lane_id = str(scope.get("lane_id", "")).strip() or None
        discriminator_kind = (
            str(gap.get("discriminator_kind", "")).strip()
            or str(scope.get("discriminator_kind", "")).strip()
            or None
        )
        scope.update(
            {
                key: value
                for key, value in {
                    "causal_gap_id": gap_id,
                    "candidate_id": candidate_id,
                    "discriminator_kind": discriminator_kind,
                }.items()
                if value is not None
            }
        )
        source = str(option.get("source", "")).strip() or None
        fingerprint = action_scope_fingerprint(action_kind, scope, source=source)
        previous = [
            gain
            for gain in gain_history
            if gain.action_kind == action_kind
            and gain.scope_fingerprint == fingerprint
        ]
        previous_result = previous[-1].gain_type if previous else None
        unavailable = any(
            gain.gain_type == InvestigationGainType.STATE_GAIN.value
            and gain.reason_code
            == InvestigationGainReasonCode.UNAVAILABLE_SOURCE.value
            for gain in previous
        )
        source_availability = (
            ActionSourceAvailability.UNAVAILABLE.value
            if unavailable
            else ActionSourceAvailability.UNKNOWN.value
        )
        tool_cost = _ACTION_TOOL_COSTS.get(action_kind, 1)
        llm_cost = _ACTION_LLM_COSTS.get(action_kind, 1)
        decision_impact = _decision_impact(action_kind, gap)
        competition_axis = str(gap.get("competition_axis", "")).strip()
        supports_candidate_ids = tuple(
            dict.fromkeys(
                str(item).strip()
                for item in gap.get("supports_candidate_ids", [])
                if str(item).strip()
            )
        )
        weakens_candidate_ids = tuple(
            dict.fromkeys(
                str(item).strip()
                for item in gap.get("weakens_candidate_ids", [])
                if str(item).strip()
            )
        )
        expected_evidence_types = tuple(
            dict.fromkeys(
                str(item).strip()
                for item in gap.get("expected_evidence_types", [])
                if str(item).strip()
            )
        )
        already_available_evidence_ids = _matching_existing_evidence_ids(
            gap=gap,
            scope=scope,
            evidence=evidence,
        )
        can_change_ranking = bool(
            gap.get(
                "can_change_ranking",
                decision_impact == ActionDecisionImpact.RANKING.value,
            )
        )
        is_hypothesis_discrimination = (
            str(gap.get("gap_type", "")) == "hypothesis_discrimination"
        )
        if mechanism_competition_required and is_hypothesis_discrimination:
            can_change_ranking = (
                can_change_ranking
                and competition_axis == CandidateCompetitionAxis.MECHANISM.value
            )
        rejection_reason: str | None = None
        eligible = True
        if previous:
            eligible = False
            rejection_reason = "exact_action_scope_already_attempted"
        elif unavailable:
            eligible = False
            rejection_reason = "source_unavailable_for_exact_scope"
        elif tool_cost > remaining_tool_budget:
            eligible = False
            rejection_reason = "insufficient_tool_budget"
        elif decision_impact == ActionDecisionImpact.CONTEXT_ONLY.value:
            eligible = False
            rejection_reason = "action_cannot_change_decision_state"
        elif (
            mechanism_competition_required
            and is_hypothesis_discrimination
            and competition_axis != CandidateCompetitionAxis.MECHANISM.value
        ):
            eligible = False
            rejection_reason = "does_not_discriminate_primary_mechanism"
        elif (
            action_kind != ActionKind.RUN_RCA_REASONING.value
            and str(gap.get("gap_type", "")) == "hypothesis_discrimination"
            and already_available_evidence_ids
        ):
            eligible = False
            rejection_reason = "discriminator_evidence_already_available"
        elif action_kind == ActionKind.RUN_RCA_REASONING.value and not (
            _reasoning_refresh_has_ranking_gain(
                gap_id=gap_id,
                action_records=action_records,
                gain_history=gain_history,
            )
        ):
            eligible = False
            rejection_reason = "no_new_ranking_evidence_for_reasoning_refresh"

        information_gain = float(gap.get("information_gain", 0.0) or 0.0)
        high_value = eligible and decision_impact in {
            ActionDecisionImpact.RANKING.value,
            ActionDecisionImpact.CONFIRMATION.value,
            ActionDecisionImpact.REASONING_REFRESH.value,
        }
        if mechanism_competition_required and is_hypothesis_discrimination:
            high_value = high_value and can_change_ranking
        if high_value and information_gain >= 0.6:
            tier = ActionValueTier.HIGH.value
        elif high_value:
            tier = ActionValueTier.MEDIUM.value
        elif eligible:
            tier = ActionValueTier.LOW.value
        else:
            tier = ActionValueTier.NONE.value
        option_id = str(option.get("option_id", "")).strip() or (
            f"option:{index}:{action_kind}:{gap_id or 'question'}:{fingerprint}"
        )
        assessments.append(
            ActionValueAssessment(
                option_id=option_id,
                action_kind=action_kind,
                scope_fingerprint=fingerprint,
                static_information_gain=max(0.0, min(1.0, information_gain)),
                source_availability=source_availability,
                decision_impact=decision_impact,
                estimated_tool_cost=tool_cost,
                estimated_llm_cost=llm_cost,
                remaining_tool_budget=max(0, remaining_tool_budget),
                high_value=high_value,
                eligible=eligible,
                value_tier=tier,
                candidate_id=candidate_id,
                lane_id=lane_id,
                gap_id=gap_id,
                discriminator_kind=discriminator_kind,
                previous_attempt_result=previous_result,
                rejection_reason=rejection_reason,
                can_change_ranking=can_change_ranking,
                supports_candidate_ids=supports_candidate_ids,
                weakens_candidate_ids=weakens_candidate_ids,
                expected_evidence_types=expected_evidence_types,
                already_available_evidence_ids=already_available_evidence_ids,
            )
        )
    return tuple(assessments)


def high_value_action_assessments(
    assessments: Sequence[ActionValueAssessment],
) -> tuple[ActionValueAssessment, ...]:
    return tuple(item for item in assessments if item.high_value)


__all__ = [
    "action_scope_fingerprint",
    "assess_action_values",
    "classify_investigation_gain",
    "derive_investigation_gain_history",
    "high_value_action_assessments",
]
