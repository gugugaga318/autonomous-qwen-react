"""Compact, traceable synthesis of typed Evidence for RCA reasoning.

The synthesis is deliberately a projection, not an inference layer.  Every
fact in the returned object is copied from an immutable :class:`Evidence`
record and retains its source Evidence ID so a caller can always expand the
summary back to the original observation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from yield_rca_core.causal_competition import (
    build_candidate_competition_brief,
    select_diverse_lane_ids,
)
from yield_rca_core.causal_lane_lifecycle import lane_is_searchable
from yield_rca_core.evidence_models import Evidence, EvidenceType

_EXPOSURE_TYPES = {
    EvidenceType.LOT_CONTEXT.value,
    EvidenceType.PROCESS_EXPOSURE.value,
    EvidenceType.EQUIPMENT_EXPOSURE.value,
    EvidenceType.IMPACT_SCOPE.value,
    EvidenceType.EXCURSION_WINDOW.value,
}
_PROCESS_TYPES = {
    EvidenceType.RECIPE_CHANGE.value,
    EvidenceType.HOLD_EVENT.value,
    EvidenceType.PARAMETER_DEVIATION.value,
    EvidenceType.TREND_DEVIATION.value,
    EvidenceType.OOC_EVENT.value,
    EvidenceType.SPC_VIOLATION.value,
}
_OUTCOME_TYPES = {
    EvidenceType.DEFECT_SIGNAL.value,
    EvidenceType.METROLOGY_DEVIATION.value,
    EvidenceType.ELECTRICAL_FAILURE.value,
}
_KNOWLEDGE_TYPES = {
    EvidenceType.HISTORICAL_CASE_MATCH.value,
    EvidenceType.ENGINEERING_NOTE.value,
}

_MAX_TEXT_LENGTH = 480
_MAX_ENTITIES_PER_EVIDENCE = 16
_MAX_SEQUENCE_ITEMS = 16
_METADATA_ALLOWLIST = frozenset(
    {
        "abnormal_row_count",
        "analysis_cutoff",
        "chamber_id",
        "context_source",
        "declared_source",
        "defect_counts",
        "direction",
        "evidence_scope",
        "equipment_id",
        "excursion_end",
        "excursion_start",
        "fail_count",
        "fail_modes",
        "insufficient_parameters",
        "lane_id",
        "lot_id",
        "lot_ids",
        "magnitude",
        "measurement_stage",
        "metric_name",
        "minimum_baseline_samples",
        "ooc_count",
        "operation_no",
        "operation_name",
        "module",
        "process_module",
        "parameter_name",
        "parameter_names",
        "pattern_counts",
        "processing_window",
        "recipe_id",
        "row_count",
        "wafer_count",
        "wat_fail_count",
        "wat_fail_lot_count",
        "wat_fail_record_count",
        "defect_type",
        "defect_description",
        "fail_mode",
        "outcome_name",
        "engineering_description",
        "required_for_confirmation",
        "source_lot_id",
        "target_row_count",
        "unit",
        "validation_status",
    }
)
_ENTITY_ATTRIBUTE_ALLOWLIST = frozenset(
    {
        "direction",
        "dominant_pattern",
        "end",
        "magnitude",
        "role",
        "start",
        "status",
        "unit",
        "validation_status",
    }
)

_PRIMARY_GROUPS = (
    "shared_exposure",
    "process_excursions",
    "outcomes",
    "controls",
    "knowledge",
    "data_missing",
    "contradictions",
    "other",
)
_LANE_FACT_GROUPS = (
    "shared_exposure",
    "process_excursions",
    "outcomes",
    "controls",
    "approved_knowledge",
    "data_missing",
    "contradictions",
)
_GLOBAL_FACT_GROUPS = (
    "analysis_context",
    "unassigned_shared_exposure",
    "unassigned_process_excursions",
    "outcomes",
    "controls",
    "approved_knowledge",
    "data_missing",
    "contradictions",
)
_MAX_ACTIVE_LANES = 3
_MAX_FACTS_PER_GROUP = 8
_MAX_PROMPT_IDS_PER_ENTITY_TYPE = 4
_PROMPT_FACT_TEXT_LENGTH = 280


def _bounded_text(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    compact = " ".join(value.split())
    if len(compact) <= _MAX_TEXT_LENGTH:
        return compact
    return compact[: _MAX_TEXT_LENGTH - 1].rstrip() + "…"


def _bounded_value(value: Any) -> Any:
    """Keep objective scalar structure while bounding high-cardinality payloads."""

    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return _bounded_text(value)
    if isinstance(value, Mapping):
        return {
            str(key): _bounded_value(item)
            for key, item in list(value.items())[:_MAX_SEQUENCE_ITEMS]
        }
    if isinstance(value, (list, tuple, set)):
        return [_bounded_value(item) for item in list(value)[:_MAX_SEQUENCE_ITEMS]]
    return _bounded_text(str(value))


def compact_evidence_record(item: Evidence) -> dict[str, Any]:
    """Project one immutable Evidence item for an LLM-facing fact register.

    The full Evidence remains in :class:`RCAState` and is still consumed by
    every Python gate.  This projection deliberately excludes arbitrary Tool
    metadata and repeated prose so prompt size cannot grow with raw row width.
    """

    entities = []
    for entity in item.entities[:_MAX_ENTITIES_PER_EVIDENCE]:
        attributes = {
            key: _bounded_value(value)
            for key, value in entity.attributes.items()
            if key in _ENTITY_ATTRIBUTE_ALLOWLIST
        }
        projected: dict[str, Any] = {
            "entity_type": entity.entity_type,
            "entity_id": entity.entity_id,
        }
        if attributes:
            projected["attributes"] = attributes
        entities.append(projected)
    metadata = {
        key: _bounded_value(value)
        for key, value in item.metadata.items()
        if key in _METADATA_ALLOWLIST
    }
    fact = _bounded_text(item.observation) or _bounded_text(item.summary)
    record: dict[str, Any] = {
        "evidence_id": item.evidence_id,
        "evidence_type": item.evidence_type,
        "fact": fact,
        "entities": entities,
    }
    if item.timestamp:
        record["timestamp"] = item.timestamp
    if item.source_agent or item.source_tool or item.source_field:
        record["source"] = {
            "agent": item.source_agent,
            "tool": item.source_tool,
            "field": item.source_field,
        }
    if metadata:
        record["metadata"] = metadata
    if len(item.entities) > len(entities):
        record["entity_count"] = len(item.entities)
        record["entities_truncated"] = True
    return record


def compact_evidence_prompt_card(item: Evidence) -> dict[str, Any]:
    """Return one low-duplication Evidence card for RCA model prompts.

    Python continues to retain the complete immutable Evidence object.  The
    card exposes objective typed fields needed for reasoning without copying
    arbitrary metadata or every entity attribute into multiple prompt blocks.
    """

    ids_by_type: dict[str, list[str]] = {}
    emitted_entity_count = 0
    for entity in item.entities:
        values = ids_by_type.setdefault(str(entity.entity_type), [])
        entity_id = str(entity.entity_id).strip()
        if (
            entity_id
            and entity_id not in values
            and len(values) < _MAX_PROMPT_IDS_PER_ENTITY_TYPE
        ):
            values.append(entity_id)
            emitted_entity_count += 1
    fact = " ".join((item.observation or item.summary).split())
    if len(fact) > _PROMPT_FACT_TEXT_LENGTH:
        fact = fact[: _PROMPT_FACT_TEXT_LENGTH - 1].rstrip() + "…"
    card: dict[str, Any] = {
        "evidence_id": item.evidence_id,
        "evidence_type": item.evidence_type,
        "source": {
            "agent": item.source_agent,
            "tool": item.source_tool,
            "field": item.source_field,
        },
        "fact": fact,
        "entity_ids_by_type": ids_by_type,
    }
    if item.timestamp:
        card["timestamp"] = item.timestamp
    prompt_metadata_keys = (
        "direction",
        "magnitude",
        "excursion_start",
        "excursion_end",
        "processing_window",
        "parameter_name",
        "outcome_name",
        "validation_status",
        "required_for_confirmation",
    )
    details = {
        key: _bounded_value(item.metadata[key])
        for key in prompt_metadata_keys
        if key in item.metadata
    }
    if details:
        card["typed_details"] = details
    card["projection_audit"] = {
        "omitted_entity_count": max(0, len(item.entities) - emitted_entity_count),
        "omitted_metadata_count": max(0, len(item.metadata) - len(details)),
    }
    return card


def compact_lane_first_synthesis_for_prompt(
    synthesis: Mapping[str, Any],
) -> dict[str, Any]:
    """Project Lane synthesis to identities, counts, and traceable IDs only."""

    lane_keys = (
        "lane_id",
        "operation",
        "operation_name",
        "module",
        "equipment",
        "chamber",
        "recipe",
        "parameter_scope",
        "exposed_lot_ids",
        "time_window",
        "priority_score",
        "lifecycle_status",
    )
    compact_lanes: list[dict[str, Any]] = []
    raw_lanes = synthesis.get("active_causal_lanes", [])
    if isinstance(raw_lanes, Sequence) and not isinstance(raw_lanes, str | bytes):
        for raw_lane in raw_lanes:
            if not isinstance(raw_lane, Mapping):
                continue
            raw_facts = raw_lane.get("facts", {})
            evidence_ids_by_group: dict[str, list[str]] = {}
            if isinstance(raw_facts, Mapping):
                for group, facts in raw_facts.items():
                    if not isinstance(facts, Sequence) or isinstance(facts, str | bytes):
                        continue
                    evidence_ids_by_group[str(group)] = [
                        str(fact.get("evidence_id"))
                        for fact in facts
                        if isinstance(fact, Mapping)
                        and str(fact.get("evidence_id", "")).strip()
                    ]
            compact_lanes.append(
                {
                    **{
                        key: _bounded_value(raw_lane[key])
                        for key in lane_keys
                        if key in raw_lane
                    },
                    "evidence_ids_by_group": evidence_ids_by_group,
                    # Compatibility shape retained with ID-only records.  The
                    # full fact card lives once in typed_evidence_register.
                    "facts": {
                        group: [
                            {"evidence_id": evidence_id}
                            for evidence_id in evidence_ids
                        ]
                        for group, evidence_ids in evidence_ids_by_group.items()
                    },
                    "fact_counts": dict(raw_lane.get("fact_counts", {})),
                    "facts_omitted": dict(raw_lane.get("facts_omitted", {})),
                }
            )
    raw_global = synthesis.get("global_facts", {})
    global_evidence_ids_by_group: dict[str, list[str]] = {}
    if isinstance(raw_global, Mapping):
        for group, facts in raw_global.items():
            if not isinstance(facts, Sequence) or isinstance(facts, str | bytes):
                continue
            global_evidence_ids_by_group[str(group)] = [
                str(fact.get("evidence_id"))
                for fact in facts
                if isinstance(fact, Mapping)
                and str(fact.get("evidence_id", "")).strip()
            ]
    return {
        "schema": "lane_first_v1",
        "prompt_projection_version": "compact_v1",
        "evidence_count": synthesis.get("evidence_count", 0),
        "group_counts": dict(synthesis.get("group_counts", {})),
        "active_lane_count": len(compact_lanes),
        "active_causal_lanes": compact_lanes,
        "global_evidence_ids_by_group": global_evidence_ids_by_group,
        "global_facts": {
            group: [
                {"evidence_id": evidence_id}
                for evidence_id in evidence_ids
            ]
            for group, evidence_ids in global_evidence_ids_by_group.items()
        },
        "global_fact_counts": dict(synthesis.get("global_fact_counts", {})),
        "global_facts_omitted": dict(synthesis.get("global_facts_omitted", {})),
        "mechanism_bridge_inputs": dict(
            synthesis.get("mechanism_bridge_inputs", {})
        ),
        "candidate_competition": dict(
            synthesis.get("candidate_competition", {})
        ),
        "prompt_evidence_ids": list(synthesis.get("prompt_evidence_ids", [])),
        "synthesis_note": synthesis.get("synthesis_note"),
    }


def _record(item: Evidence) -> dict[str, Any]:
    """Return only objective, JSON-safe fields from one Evidence item."""

    return compact_evidence_record(item)


def _primary_group(item: Evidence) -> str:
    if item.evidence_type in _EXPOSURE_TYPES:
        return "shared_exposure"
    if item.evidence_type in _PROCESS_TYPES:
        return "process_excursions"
    if item.evidence_type in _OUTCOME_TYPES:
        return "outcomes"
    if item.evidence_type == EvidenceType.NEGATIVE_SIGNAL.value:
        return "controls"
    if item.evidence_type in _KNOWLEDGE_TYPES:
        return "knowledge"
    if item.evidence_type == EvidenceType.DATA_MISSING.value:
        return "data_missing"
    return "other"


def _is_approved_knowledge(item: Evidence) -> bool:
    """Return whether typed Knowledge carries an explicit engineering approval."""

    if item.evidence_type not in _KNOWLEDGE_TYPES or item.source_type != "knowledge":
        return False
    statuses = [
        str(value).upper()
        for key, value in item.metadata.items()
        if str(key).casefold() == "validation_status"
    ]
    statuses.extend(
        str(value).upper()
        for entity in item.entities
        for key, value in entity.attributes.items()
        if str(key).casefold() == "validation_status"
    )
    return bool(statuses) and all(status == "CONFIRMED" for status in statuses)


def _string_values(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return tuple(
            dict.fromkeys(str(item).strip() for item in value if str(item).strip())
        )
    return ()


def _lane_projection(lane: Mapping[str, Any]) -> dict[str, Any]:
    projected: dict[str, Any] = {
        "lane_id": str(lane.get("lane_id", "")).strip(),
        "operation": str(lane.get("operation", "")).strip(),
        "operation_name": str(lane.get("operation_name", "")).strip(),
        "module": str(lane.get("module", "")).strip(),
        "equipment": str(lane.get("equipment", "")).strip(),
        "chamber": str(lane.get("chamber", "")).strip(),
        "recipe": str(lane.get("recipe", "")).strip(),
        "parameter_scope": list(_string_values(lane.get("parameter_scope", ()))),
        "exposed_lot_ids": list(_string_values(lane.get("exposed_lot_ids", ()))),
        "time_window": list(_string_values(lane.get("time_window", ()))),
        "initial_evidence_ids": list(
            _string_values(lane.get("initial_evidence_ids", ()))
        ),
        "priority_score": float(lane.get("priority_score", 0.0) or 0.0),
        "investigation_status": str(lane.get("investigation_status", "")).strip(),
        "lifecycle_status": str(lane.get("lifecycle_status", "")).strip(),
    }
    if len(projected["parameter_scope"]) > _MAX_SEQUENCE_ITEMS:
        projected["parameter_scope_count"] = len(projected["parameter_scope"])
        projected["parameter_scope"] = projected["parameter_scope"][:_MAX_SEQUENCE_ITEMS]
    if len(projected["exposed_lot_ids"]) > _MAX_SEQUENCE_ITEMS:
        projected["exposed_lot_count"] = len(projected["exposed_lot_ids"])
        projected["exposed_lot_ids"] = projected["exposed_lot_ids"][:_MAX_SEQUENCE_ITEMS]
    return {key: value for key, value in projected.items() if value not in ("", [], ())}


def _active_lane_projections(
    causal_lanes: Sequence[Mapping[str, Any]],
    *,
    max_active_lanes: int,
) -> list[dict[str, Any]]:
    unique: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for index, lane in enumerate(causal_lanes):
        lane_id = str(lane.get("lane_id", "")).strip()
        if lane_id:
            unique[lane_id] = (index, lane)
    eligible = [
        lane
        for _, lane in unique.values()
        if lane_is_searchable(lane)
    ]
    selected_ids = select_diverse_lane_ids(
        eligible,
        limit=max_active_lanes,
    )
    by_id = {str(lane.get("lane_id", "")): lane for lane in eligible}
    return [_lane_projection(by_id[lane_id]) for lane_id in selected_ids]


def _entity_scope(item: Evidence) -> dict[str, set[str]]:
    entity_field = {
        "operation": "operation",
        "equipment": "equipment",
        "chamber": "chamber",
        "recipe": "recipe",
    }
    scope: dict[str, set[str]] = {field: set() for field in entity_field}
    for entity in item.entities:
        field = entity_field.get(entity.entity_type)
        if field is not None:
            scope[field].add(entity.entity_id)
    metadata_fields = {
        "operation": ("operation", "operation_no"),
        "equipment": ("equipment", "equipment_id"),
        "chamber": ("chamber", "chamber_id"),
        "recipe": ("recipe", "recipe_id"),
    }
    for field, keys in metadata_fields.items():
        for key in keys:
            scope[field].update(_string_values(item.metadata.get(key)))
    return scope


def _lane_binding(item: Evidence, lane: Mapping[str, Any]) -> str | None:
    """Bind Evidence to a Lane only through explicit IDs or matching typed scope."""

    lane_id = str(lane.get("lane_id", ""))
    explicit_lane_id = str(item.metadata.get("lane_id", "")).strip()
    if explicit_lane_id:
        return "explicit_lane_id" if explicit_lane_id == lane_id else None
    if item.evidence_id in set(_string_values(lane.get("initial_evidence_ids", ()))):
        return "initial_evidence_id"

    evidence_scope = _entity_scope(item)
    matched_dimensions = 0
    for field in ("operation", "equipment", "chamber", "recipe"):
        lane_value = str(lane.get(field, "")).strip()
        observed = evidence_scope[field]
        if not lane_value or not observed:
            continue
        if lane_value not in observed:
            return None
        matched_dimensions += 1
    # One broad equipment or operation match is not enough to assign a fact to
    # a concrete Lane. Two matching scope dimensions are an objective binding.
    return "typed_entity_scope" if matched_dimensions >= 2 else None


def _append_bounded(
    target: dict[str, list[dict[str, Any]]],
    group: str,
    record: dict[str, Any],
) -> bool:
    if len(target[group]) < _MAX_FACTS_PER_GROUP:
        target[group].append(record)
        return True
    return False


def _fact_counts(
    evidence: Sequence[Evidence],
    lane: Mapping[str, Any] | None,
) -> dict[str, int]:
    counts = {group: 0 for group in _LANE_FACT_GROUPS}
    for item in evidence:
        if lane is not None and _lane_binding(item, lane) is None:
            continue
        primary = _primary_group(item)
        if primary == "knowledge":
            if _is_approved_knowledge(item):
                counts["approved_knowledge"] += 1
        elif primary in counts:
            counts[primary] += 1
        if bool(item.metadata.get("contradicts_candidate")):
            counts["contradictions"] += 1
    return counts


def _enrich_lane_semantics_from_evidence(
    causal_lanes: Sequence[Mapping[str, Any]],
    evidence: Sequence[Evidence],
) -> list[dict[str, Any]]:
    """Recover losslessly projected Lane labels for legacy serialized State."""

    semantic_by_lane_id: dict[str, Mapping[str, Any]] = {}
    for item in evidence:
        raw_lane = item.metadata.get("lane")
        if not isinstance(raw_lane, Mapping):
            continue
        lane_id = str(raw_lane.get("lane_id", "")).strip()
        if lane_id:
            semantic_by_lane_id.setdefault(lane_id, raw_lane)
    enriched: list[dict[str, Any]] = []
    for raw_lane in causal_lanes:
        lane = dict(raw_lane)
        semantic = semantic_by_lane_id.get(str(lane.get("lane_id", "")).strip(), {})
        for field in ("operation_name", "module"):
            if not str(lane.get(field, "")).strip() and str(
                semantic.get(field, "")
            ).strip():
                lane[field] = str(semantic[field]).strip()
        enriched.append(lane)
    return enriched


def build_lane_first_evidence_synthesis(
    evidence: Iterable[Evidence],
    causal_lanes: Sequence[Mapping[str, Any]],
    *,
    max_active_lanes: int = _MAX_ACTIVE_LANES,
) -> dict[str, Any]:
    """Build a bounded Lane-first fact map for candidate generation.

    The projection selects active Lanes using Python-owned status and priority,
    then binds facts only through an explicit Lane ID, an initial Evidence ID,
    or at least two matching typed scope dimensions. Unbound product outcomes,
    controls, approved Knowledge, and missing-data facts remain visible as global
    context rather than being falsely attributed to every Lane.
    """

    if max_active_lanes < 1:
        raise ValueError("max_active_lanes must be at least 1")
    unique = {
        item.evidence_id: item
        for item in evidence
        if isinstance(item, Evidence) and item.is_typed
    }
    evidence_items = list(unique.values())
    enriched_causal_lanes = _enrich_lane_semantics_from_evidence(
        causal_lanes,
        evidence_items,
    )
    active_lanes = _active_lane_projections(
        enriched_causal_lanes,
        max_active_lanes=max_active_lanes,
    )
    emitted_ids: set[str] = set()
    lane_summaries: list[dict[str, Any]] = []
    lane_bound_ids: set[str] = set()
    for lane in active_lanes:
        facts: dict[str, list[dict[str, Any]]] = {
            group: [] for group in _LANE_FACT_GROUPS
        }
        bindings: dict[str, str] = {}
        for item in evidence_items:
            binding = _lane_binding(item, lane)
            if binding is None:
                continue
            lane_bound_ids.add(item.evidence_id)
            primary = _primary_group(item)
            target_group = (
                "approved_knowledge"
                if primary == "knowledge" and _is_approved_knowledge(item)
                else primary
            )
            if target_group in facts:
                if _append_bounded(facts, target_group, _record(item)):
                    bindings[item.evidence_id] = binding
                    emitted_ids.add(item.evidence_id)
            if bool(item.metadata.get("contradicts_candidate")):
                if _append_bounded(facts, "contradictions", _record(item)):
                    emitted_ids.add(item.evidence_id)
        counts = _fact_counts(evidence_items, lane)
        lane_summaries.append(
            {
                **lane,
                "facts": facts,
                "fact_counts": counts,
                "facts_omitted": {
                    group: max(0, counts[group] - len(facts[group]))
                    for group in _LANE_FACT_GROUPS
                },
                "evidence_bindings": bindings,
            }
        )

    global_facts: dict[str, list[dict[str, Any]]] = {
        group: [] for group in _GLOBAL_FACT_GROUPS
    }
    global_counts = {group: 0 for group in _GLOBAL_FACT_GROUPS}
    for item in evidence_items:
        if item.evidence_id in lane_bound_ids:
            continue
        primary = _primary_group(item)
        global_target_group: str | None = None
        if item.evidence_type == EvidenceType.LOT_CONTEXT.value:
            global_target_group = "analysis_context"
        elif primary == "shared_exposure" and not active_lanes:
            global_target_group = "unassigned_shared_exposure"
        elif primary == "process_excursions" and not active_lanes:
            global_target_group = "unassigned_process_excursions"
        elif primary in {"outcomes", "controls", "data_missing"}:
            global_target_group = primary
        elif primary == "knowledge" and _is_approved_knowledge(item):
            global_target_group = "approved_knowledge"
        if global_target_group is not None:
            global_counts[global_target_group] += 1
            if _append_bounded(global_facts, global_target_group, _record(item)):
                emitted_ids.add(item.evidence_id)
        if bool(item.metadata.get("contradicts_candidate")):
            global_counts["contradictions"] += 1
            if _append_bounded(global_facts, "contradictions", _record(item)):
                emitted_ids.add(item.evidence_id)

    grouped = build_evidence_synthesis(evidence_items)
    global_outcome_evidence_ids = [
        str(record["evidence_id"])
        for record in global_facts["outcomes"]
    ]
    mechanism_bridge_inputs = {
        "by_lane": [
            {
                "lane_id": str(lane["lane_id"]),
                "parameter_or_process_evidence_ids": [
                    str(record["evidence_id"])
                    for record in lane["facts"]["process_excursions"]
                ],
                "lane_outcome_evidence_ids": [
                    str(record["evidence_id"])
                    for record in lane["facts"]["outcomes"]
                ],
            }
            for lane in lane_summaries
        ],
        "global_outcome_evidence_ids": global_outcome_evidence_ids,
        "approved_knowledge_evidence_ids": list(
            dict.fromkeys(
                [
                    str(record["evidence_id"])
                    for lane in lane_summaries
                    for record in lane["facts"]["approved_knowledge"]
                ]
                + [
                    str(record["evidence_id"])
                    for record in global_facts["approved_knowledge"]
                ]
            )
        ),
        "note": (
            "These are observed inputs for Qwen mechanism reasoning, not a "
            "Python-inferred physical mechanism."
        ),
    }
    candidate_competition = build_candidate_competition_brief(
        lane_summaries,
        global_outcome_evidence_ids=global_outcome_evidence_ids,
    )
    return {
        "schema": "lane_first_v1",
        "evidence_count": len(evidence_items),
        "group_counts": {
            group: len(grouped[group])
            for group in _PRIMARY_GROUPS
        },
        "active_lane_count": len(lane_summaries),
        "active_causal_lanes": lane_summaries,
        "global_facts": global_facts,
        "global_fact_counts": global_counts,
        "global_facts_omitted": {
            group: max(0, global_counts[group] - len(global_facts[group]))
            for group in _GLOBAL_FACT_GROUPS
        },
        "mechanism_bridge_inputs": mechanism_bridge_inputs,
        "candidate_competition": candidate_competition,
        "prompt_evidence_ids": sorted(emitted_ids),
        "prompt_evidence_count": len(emitted_ids),
        "omitted_from_prompt_count": max(0, len(evidence_items) - len(emitted_ids)),
        "synthesis_note": (
            "Facts are copied from typed Evidence and retain Evidence IDs. "
            "Lane binding is organizational only and is not a causal conclusion."
        ),
    }


def build_evidence_synthesis(evidence: Iterable[Evidence]) -> dict[str, Any]:
    """Group typed Evidence into a compact, ID-traceable fact register.

    No causal conclusion is made here.  In particular, a Knowledge record is
    placed in ``knowledge`` and never promoted into a current-Lot process fact.
    """

    unique: dict[str, Evidence] = {}
    for item in evidence:
        if not isinstance(item, Evidence) or not item.is_typed:
            continue
        unique.setdefault(item.evidence_id, item)

    groups: dict[str, list[dict[str, Any]]] = {
        "shared_exposure": [],
        "process_excursions": [],
        "outcomes": [],
        "controls": [],
        "knowledge": [],
        "data_missing": [],
        "contradictions": [],
        "other": [],
    }
    for item in unique.values():
        record = _record(item)
        groups[_primary_group(item)].append(record)

        # A negative signal can be a contradiction only when its typed record
        # explicitly carries a contradiction marker.  We do not infer one from
        # the mere existence of a normal control.
        if bool(item.metadata.get("contradicts_candidate")):
            groups["contradictions"].append(record)

    return {
        "evidence_ids": sorted(unique),
        "evidence_count": len(unique),
        **groups,
    }


__all__ = [
    "build_evidence_synthesis",
    "build_lane_first_evidence_synthesis",
    "compact_evidence_prompt_card",
    "compact_evidence_record",
    "compact_lane_first_synthesis_for_prompt",
]
