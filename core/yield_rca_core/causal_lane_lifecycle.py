"""Python-owned lifecycle governance for factual causal Lane inventory.

Lane identity is an observed process scope (operation/equipment/chamber/recipe),
not an LLM mechanism claim.  The helpers here consolidate repeat discovery,
retain merge/reactivation audit, and keep LLM-active selection separate from
the complete canonical Python inventory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from typing import Any

from yield_rca_core.causal_investigation_models import (
    CausalLaneRecord,
    InvestigationLaneStatus,
    LaneLifecycleStatus,
    LaneLifecycleTransition,
)

_TERMINAL_INVESTIGATION_STATUSES = {
    InvestigationLaneStatus.ELIMINATED.value,
    InvestigationLaneStatus.BLOCKED.value,
}
_TERMINAL_LIFECYCLE_STATUSES = {
    LaneLifecycleStatus.ELIMINATED.value,
    LaneLifecycleStatus.MERGED.value,
    LaneLifecycleStatus.BLOCKED.value,
}


def _text(value: object) -> str:
    return str(value or "").strip()


def _normalized(value: object) -> str:
    return " ".join(_text(value).casefold().split())


def _unique(*values: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            item.strip()
            for value in values
            for item in value
            if isinstance(item, str) and item.strip()
        )
    )


def causal_lane_scope_identity(lane: Mapping[str, Any] | CausalLaneRecord) -> str:
    """Return the stable factual identity of a concrete exposure Lane.

    Parameter observations are intentionally excluded: new FDC parameters are
    Evidence gain on the same process Lane.  Candidate mechanism/scope semantics
    remain Qwen-owned and are never used for Lane merging.
    """

    values: tuple[str, ...]
    if isinstance(lane, CausalLaneRecord):
        values = (lane.operation, lane.equipment, lane.chamber, lane.recipe)
        lane_id = lane.lane_id
    else:
        values = tuple(
            _text(lane.get(field))
            for field in ("operation", "equipment", "chamber", "recipe")
        )
        lane_id = _text(lane.get("lane_id"))
    normalized = tuple(_normalized(value) for value in values)
    if not all(normalized):
        # Incomplete legacy/test records cannot safely be merged by partial
        # entity overlap. Their explicit ID is the only stable identity.
        return f"lane_id:{_normalized(lane_id)}"
    return "scope:" + "|".join(normalized)


def lane_is_searchable(lane: Mapping[str, Any] | CausalLaneRecord) -> bool:
    if isinstance(lane, CausalLaneRecord):
        investigation_status = lane.investigation_status
        lifecycle_status = lane.lifecycle_status
    else:
        investigation_status = _text(lane.get("investigation_status"))
        lifecycle_status = _text(lane.get("lifecycle_status"))
    return (
        investigation_status not in _TERMINAL_INVESTIGATION_STATUSES
        and lifecycle_status not in _TERMINAL_LIFECYCLE_STATUSES
    )


def _next_sequence(lane: CausalLaneRecord) -> int:
    return max((item.sequence for item in lane.lifecycle_history), default=0) + 1


def _transition(
    lane: CausalLaneRecord,
    *,
    to_status: str,
    reason: str,
    evidence_ids: Sequence[str] = (),
) -> CausalLaneRecord:
    if lane.lifecycle_status == to_status:
        return lane
    transition = LaneLifecycleTransition(
        sequence=_next_sequence(lane),
        lane_id=lane.lane_id,
        from_status=lane.lifecycle_status,
        to_status=to_status,
        reason=reason,
        evidence_ids=tuple(evidence_ids),
    )
    return replace(
        lane,
        lifecycle_status=to_status,
        last_transition_reason=reason,
        lifecycle_history=(*lane.lifecycle_history, transition),
    )


def transition_lane(
    lane: CausalLaneRecord,
    *,
    lifecycle_status: str,
    reason: str,
    evidence_ids: Sequence[str] = (),
    investigation_status: str | None = None,
    pruned_reason: str | None = None,
) -> CausalLaneRecord:
    """Apply one lifecycle/investigation transition atomically."""

    if lane.lifecycle_status == lifecycle_status and (
        investigation_status is None
        or lane.investigation_status == investigation_status
    ):
        return lane
    transition = LaneLifecycleTransition(
        sequence=_next_sequence(lane),
        lane_id=lane.lane_id,
        from_status=lane.lifecycle_status,
        to_status=lifecycle_status,
        reason=reason,
        evidence_ids=tuple(evidence_ids),
    )
    return replace(
        lane,
        lifecycle_status=lifecycle_status,
        investigation_status=investigation_status or lane.investigation_status,
        pruned_reason=pruned_reason,
        last_transition_reason=reason,
        lifecycle_history=(*lane.lifecycle_history, transition),
    )


def _record_alias_merge(
    lane: CausalLaneRecord,
    *,
    alias_lane_id: str,
    evidence_ids: Sequence[str],
) -> CausalLaneRecord:
    if alias_lane_id == lane.lane_id or alias_lane_id in lane.merged_from_lane_ids:
        return lane
    reason = (
        f"Lane {alias_lane_id} has the same normalized process scope and was "
        f"merged into canonical Lane {lane.lane_id}."
    )
    transition = LaneLifecycleTransition(
        sequence=_next_sequence(lane),
        lane_id=alias_lane_id,
        from_status=LaneLifecycleStatus.CREATED.value,
        to_status=LaneLifecycleStatus.MERGED.value,
        reason=reason,
        evidence_ids=tuple(evidence_ids),
        related_lane_id=lane.lane_id,
    )
    return replace(
        lane,
        merged_from_lane_ids=(*lane.merged_from_lane_ids, alias_lane_id),
        last_transition_reason=reason,
        lifecycle_history=(*lane.lifecycle_history, transition),
    )


def _merge_time_window(
    left: tuple[str, ...],
    right: tuple[str, ...],
) -> tuple[str, ...]:
    if not left:
        return right
    if not right:
        return left
    starts = (left[0], right[0])
    ends = (left[1], right[1])
    return (
        min(starts, key=lambda item: datetime.fromisoformat(item.replace("Z", "+00:00"))),
        max(ends, key=lambda item: datetime.fromisoformat(item.replace("Z", "+00:00"))),
    )


def merge_lane_discovery(
    existing: CausalLaneRecord,
    incoming: CausalLaneRecord,
) -> CausalLaneRecord:
    """Merge repeat/alias discovery without losing prior typed Evidence."""

    if causal_lane_scope_identity(existing) != causal_lane_scope_identity(incoming):
        raise ValueError("cannot merge causal Lanes with different factual scope")
    gained_evidence = tuple(
        item
        for item in incoming.initial_evidence_ids
        if item not in set(existing.initial_evidence_ids)
    )
    lane = replace(
        existing,
        operation=existing.operation or incoming.operation,
        equipment=existing.equipment or incoming.equipment,
        chamber=existing.chamber or incoming.chamber,
        recipe=existing.recipe or incoming.recipe,
        parameter_scope=_unique(
            existing.parameter_scope,
            incoming.parameter_scope,
        ),
        exposed_lot_ids=_unique(
            existing.exposed_lot_ids,
            incoming.exposed_lot_ids,
        ),
        time_window=_merge_time_window(existing.time_window, incoming.time_window),
        initial_evidence_ids=_unique(
            existing.initial_evidence_ids,
            incoming.initial_evidence_ids,
        ),
        priority_score=max(existing.priority_score, incoming.priority_score),
        discovery_count=existing.discovery_count + 1,
    )
    if existing.investigation_status in _TERMINAL_INVESTIGATION_STATUSES and gained_evidence:
        reason = (
            "Terminal Lane was reactivated because newly collected typed "
            f"Evidence was bound to the same factual scope: {', '.join(gained_evidence)}."
        )
        lane = replace(
            lane,
            investigation_status=InvestigationLaneStatus.EVIDENCE_COLLECTED.value,
            lifecycle_status=existing.lifecycle_status,
            pruned_reason=None,
            reactivation_count=existing.reactivation_count + 1,
        )
        lane = _transition(
            lane,
            to_status=LaneLifecycleStatus.CREATED.value,
            reason=reason,
            evidence_ids=gained_evidence,
        )
    if incoming.lane_id != existing.lane_id:
        lane = _record_alias_merge(
            lane,
            alias_lane_id=incoming.lane_id,
            evidence_ids=incoming.initial_evidence_ids,
        )
    return lane


def reconcile_lane_inventory(
    existing: Sequence[CausalLaneRecord],
    discovered: Sequence[CausalLaneRecord],
) -> tuple[CausalLaneRecord, ...]:
    """Return one canonical record per newly observed stable Lane identity.

    Existing records win canonical-ID selection so Candidate/Lane references do
    not change across rounds. Repeated discovery therefore increases audit
    counters and Evidence coverage, not inventory cardinality.
    """

    records = list(existing)
    index_by_id = {item.lane_id: index for index, item in enumerate(records)}
    index_by_identity: dict[str, int] = {}
    for index, item in enumerate(records):
        index_by_identity.setdefault(causal_lane_scope_identity(item), index)
    for incoming in discovered:
        target_index = index_by_id.get(incoming.lane_id)
        identity = causal_lane_scope_identity(incoming)
        if target_index is None:
            target_index = index_by_identity.get(identity)
        if target_index is None:
            records.append(incoming)
            target_index = len(records) - 1
            index_by_id[incoming.lane_id] = target_index
            index_by_identity[identity] = target_index
            continue
        records[target_index] = merge_lane_discovery(records[target_index], incoming)
        index_by_id[incoming.lane_id] = target_index
    return tuple(records)


def apply_active_lane_snapshot(
    lanes: Sequence[CausalLaneRecord],
    *,
    active_lane_ids: Sequence[str],
) -> tuple[CausalLaneRecord, ...]:
    """Persist active/deferred lifecycle without reopening terminal Lanes."""

    active = set(active_lane_ids)
    updated: list[CausalLaneRecord] = []
    for lane in lanes:
        if not lane_is_searchable(lane):
            updated.append(lane)
            continue
        if lane.lifecycle_status == LaneLifecycleStatus.CHALLENGED.value:
            updated.append(lane)
            continue
        target = (
            LaneLifecycleStatus.ACTIVE.value
            if lane.lane_id in active
            else LaneLifecycleStatus.DEFERRED.value
        )
        reason = (
            "Selected for the bounded active Lane snapshot."
            if target == LaneLifecycleStatus.ACTIVE.value
            else "Retained in Python inventory outside the bounded active snapshot."
        )
        updated.append(_transition(lane, to_status=target, reason=reason))
    return tuple(updated)


def mark_lanes_challenged(
    lanes: Sequence[CausalLaneRecord],
    *,
    challenged_lane_ids: Sequence[str],
    evidence_ids_by_lane: Mapping[str, Sequence[str]] | None = None,
) -> tuple[CausalLaneRecord, ...]:
    """Mark only valid searchable probe Lanes as challenged."""

    challenged = set(challenged_lane_ids)
    evidence = evidence_ids_by_lane or {}
    return tuple(
        _transition(
            lane,
            to_status=LaneLifecycleStatus.CHALLENGED.value,
            reason="Selected by the validated adversarial challenge for investigation.",
            evidence_ids=evidence.get(lane.lane_id, ()),
        )
        if lane.lane_id in challenged and lane_is_searchable(lane)
        else lane
        for lane in lanes
    )


__all__ = [
    "apply_active_lane_snapshot",
    "causal_lane_scope_identity",
    "lane_is_searchable",
    "mark_lanes_challenged",
    "merge_lane_discovery",
    "reconcile_lane_inventory",
    "transition_lane",
]
