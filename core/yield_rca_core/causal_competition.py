"""Python-owned causal-direction and scope competition classification.

This module organizes immutable Lane facts.  It never proposes a root cause or
claims that an observed process condition caused the product outcome.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from yield_rca_core.causal_investigation_models import (
    CandidateCompetitionAxis,
    CandidateCompetitionType,
    CompetitionRequirement,
    ScopeAssessmentStatus,
)
from yield_rca_core.causal_lane_lifecycle import lane_is_searchable


def _text(value: object) -> str:
    return str(value or "").strip()


def _strings(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(
            dict.fromkeys(_text(item) for item in value if _text(item))
        )
    return ()


def causal_direction_identity(lane: Mapping[str, Any]) -> tuple[str, ...]:
    """Return a factual direction identity that intentionally excludes recipe.

    Recipes and time windows are scope dimensions.  Operation, physical asset,
    and parameter family describe the observable causal direction supplied to
    Qwen for mechanism reasoning.
    """

    return (
        _text(lane.get("operation")).casefold(),
        _text(lane.get("equipment")).casefold(),
        _text(lane.get("chamber")).casefold(),
        *sorted(item.casefold() for item in _strings(lane.get("parameter_scope"))),
    )


def select_diverse_lane_ids(
    causal_lanes: Sequence[Mapping[str, Any]],
    *,
    limit: int,
) -> tuple[str, ...]:
    """Select a primary Lane, a different direction, then a scope sibling."""

    if limit < 1:
        raise ValueError("limit must be at least 1")
    unique: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for index, lane in enumerate(causal_lanes):
        lane_id = _text(lane.get("lane_id"))
        if lane_id and lane_is_searchable(lane):
            unique[lane_id] = (index, lane)
    ordered = sorted(
        unique.values(),
        key=lambda item: (
            -float(item[1].get("priority_score", 0.0) or 0.0),
            item[0],
            _text(item[1].get("lane_id")),
        ),
    )
    if not ordered:
        return ()

    selected: list[Mapping[str, Any]] = [ordered[0][1]]
    primary = selected[0]
    primary_direction = causal_direction_identity(primary)
    primary_recipe = _text(primary.get("recipe")).casefold()

    different_direction = next(
        (
            lane
            for _, lane in ordered[1:]
            if causal_direction_identity(lane) != primary_direction
        ),
        None,
    )
    if different_direction is not None and len(selected) < limit:
        selected.append(different_direction)

    scope_sibling = next(
        (
            lane
            for _, lane in ordered[1:]
            if lane not in selected
            and causal_direction_identity(lane) == primary_direction
            and _text(lane.get("recipe")).casefold() != primary_recipe
        ),
        None,
    )
    if scope_sibling is not None and len(selected) < limit:
        selected.append(scope_sibling)

    for _, lane in ordered:
        if len(selected) >= limit:
            break
        if lane not in selected:
            selected.append(lane)
    return tuple(_text(lane.get("lane_id")) for lane in selected)


def _fact_ids(lane: Mapping[str, Any], group: str) -> list[str]:
    raw_facts = lane.get("facts", {})
    if not isinstance(raw_facts, Mapping):
        return []
    facts = raw_facts.get(group, [])
    if not isinstance(facts, list):
        return []
    return [
        _text(item.get("evidence_id"))
        for item in facts
        if isinstance(item, Mapping) and _text(item.get("evidence_id"))
    ]


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _scope_outcome_facts(lane: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Project normalized factual outcome rates without interpreting sensitivity."""

    raw_facts = lane.get("facts", {})
    if not isinstance(raw_facts, Mapping):
        return []
    outcomes = raw_facts.get("outcomes", [])
    if not isinstance(outcomes, list):
        return []
    projected: list[dict[str, Any]] = []
    for fact in outcomes:
        if not isinstance(fact, Mapping):
            continue
        metadata = fact.get("metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        numerator_key = next(
            (
                key
                for key in (
                    "fail_count",
                    "ooc_count",
                    "abnormal_row_count",
                    "wat_fail_record_count",
                )
                if _number(metadata.get(key)) is not None
            ),
            None,
        )
        denominator_key = next(
            (
                key
                for key in ("row_count", "target_row_count", "wafer_count")
                if _number(metadata.get(key)) is not None
            ),
            None,
        )
        numerator = _number(metadata.get(numerator_key)) if numerator_key else None
        denominator = (
            _number(metadata.get(denominator_key)) if denominator_key else None
        )
        entities = fact.get("entities", [])
        lot_ids = sorted(
            {
                _text(entity.get("entity_id"))
                for entity in entities
                if isinstance(entity, Mapping)
                and _text(entity.get("entity_type")) == "lot"
                and _text(entity.get("entity_id"))
            }
        )
        source_lot_id = _text(metadata.get("source_lot_id"))
        independent_lot_ids = [
            lot_id for lot_id in lot_ids if not source_lot_id or lot_id != source_lot_id
        ]
        normalized_rate = (
            round(numerator / denominator, 6)
            if numerator is not None and denominator not in {None, 0.0}
            else None
        )
        projected.append(
            {
                "evidence_id": _text(fact.get("evidence_id")),
                "evidence_type": _text(fact.get("evidence_type")),
                "metric_name": _text(metadata.get("metric_name")),
                "measurement_stage": _text(metadata.get("measurement_stage")),
                "numerator": numerator,
                "numerator_field": numerator_key,
                "denominator": denominator,
                "denominator_field": denominator_key,
                "normalized_rate": normalized_rate,
                "normalization_status": (
                    "available" if normalized_rate is not None else "denominator_unavailable"
                ),
                "lot_ids": lot_ids,
                "independent_lot_ids": independent_lot_ids,
                "independent_lot_count": len(independent_lot_ids),
                "source_lot_overlap": bool(source_lot_id and source_lot_id in lot_ids),
            }
        )
    return projected


def build_candidate_competition_brief(
    active_lanes: Sequence[Mapping[str, Any]],
    *,
    global_outcome_evidence_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Describe root-cause competition and independent Scope assessment.

    Candidate eligibility is deliberately modest: a direction needs bound
    exposure and process observations plus an observed product outcome.  A
    Scope is deliberately not a root-cause Candidate axis.  A same-direction
    recipe sibling creates a separate Scope assessment while Candidate slots
    remain available for different causal directions or physical mechanisms.
    """

    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
    for lane in active_lanes:
        grouped.setdefault(causal_direction_identity(lane), []).append(lane)

    global_outcomes = [
        _text(item) for item in global_outcome_evidence_ids if _text(item)
    ]
    direction_bundles: list[dict[str, Any]] = []
    scope_groups: list[dict[str, Any]] = []
    eligible_direction_ids: list[str] = []
    scope_ready_ids: list[str] = []
    scope_opportunity_ids: list[str] = []

    for index, lanes in enumerate(grouped.values()):
        bundle_id = f"direction_{index}"
        lane_ids = [_text(lane.get("lane_id")) for lane in lanes]
        exposure_ids = list(
            dict.fromkeys(
                evidence_id
                for lane in lanes
                for evidence_id in _fact_ids(lane, "shared_exposure")
            )
        )
        process_ids = list(
            dict.fromkeys(
                evidence_id
                for lane in lanes
                for evidence_id in _fact_ids(lane, "process_excursions")
            )
        )
        lane_outcome_ids = list(
            dict.fromkeys(
                evidence_id
                for lane in lanes
                for evidence_id in _fact_ids(lane, "outcomes")
            )
        )
        outcome_ids = list(dict.fromkeys([*lane_outcome_ids, *global_outcomes]))
        candidate_eligible = bool(exposure_ids and process_ids and outcome_ids)
        if candidate_eligible:
            eligible_direction_ids.append(bundle_id)
        recipes = list(
            dict.fromkeys(_text(lane.get("recipe")) for lane in lanes if _text(lane.get("recipe")))
        )
        process_recipe_ids = list(
            dict.fromkeys(
                _text(lane.get("recipe"))
                for lane in lanes
                if _text(lane.get("recipe"))
                and _fact_ids(lane, "process_excursions")
            )
        )
        focal_recipe_ids = list(
            dict.fromkeys(
                _text(lane.get("recipe"))
                for lane in lanes
                if _text(lane.get("recipe"))
                and _fact_ids(lane, "shared_exposure")
                and _fact_ids(lane, "process_excursions")
                and outcome_ids
            )
        )
        direction_bundles.append(
            {
                "bundle_id": bundle_id,
                "operation": _text(lanes[0].get("operation")),
                "equipment": _text(lanes[0].get("equipment")),
                "chamber": _text(lanes[0].get("chamber")),
                "parameter_scope": list(
                    dict.fromkeys(
                        item
                        for lane in lanes
                        for item in _strings(lane.get("parameter_scope"))
                    )
                ),
                "lane_ids": lane_ids,
                "recipes": recipes,
                "shared_exposure_evidence_ids": exposure_ids,
                "process_evidence_ids": process_ids,
                "outcome_evidence_ids": outcome_ids,
                "candidate_eligible": candidate_eligible,
            }
        )
        if len(recipes) >= 2:
            group_id = f"scope_{index}"
            scope_opportunity = bool(focal_recipe_ids)
            scope_evidence_complete = (
                len(process_recipe_ids) >= 2 and bool(outcome_ids)
            )
            missing_process_recipe_ids = [
                recipe
                for recipe in recipes
                if recipe not in set(process_recipe_ids)
            ]
            missing_process_lane_ids = [
                _text(lane.get("lane_id"))
                for lane in lanes
                if _text(lane.get("lane_id"))
                and _text(lane.get("recipe"))
                in set(missing_process_recipe_ids)
            ]
            if scope_opportunity:
                scope_opportunity_ids.append(group_id)
            if scope_evidence_complete:
                scope_ready_ids.append(group_id)
            scope_groups.append(
                {
                    "scope_group_id": group_id,
                    "direction_bundle_id": bundle_id,
                    "lane_ids": lane_ids,
                    "recipes": recipes,
                    "recipes_with_process_evidence": process_recipe_ids,
                    "focal_recipe_ids": focal_recipe_ids,
                    "missing_process_recipe_ids": missing_process_recipe_ids,
                    "missing_process_lane_ids": missing_process_lane_ids,
                    "scope_competition_opportunity": scope_opportunity,
                    "scope_evidence_complete": scope_evidence_complete,
                    # Compatibility alias retained for existing traces.  It
                    # continues to mean that comparative Evidence is already
                    # available, never that Qwen is allowed to compete.
                    "scope_competition_ready": scope_evidence_complete,
                    "normalized_outcome_facts_by_lane": {
                        _text(lane.get("lane_id")): _scope_outcome_facts(lane)
                        for lane in lanes
                        if _text(lane.get("lane_id"))
                    },
                }
            )

    direction_required = len(eligible_direction_ids) >= 2
    mechanism_required = bool(eligible_direction_ids)
    scope_assessment_required = bool(scope_opportunity_ids)
    competition_axes: list[str] = []
    if len(direction_bundles) >= 2:
        competition_axes.append(CandidateCompetitionAxis.DIRECTION.value)
    if mechanism_required:
        competition_axes.append(CandidateCompetitionAxis.MECHANISM.value)
    if scope_assessment_required:
        competition_axes.append(CandidateCompetitionAxis.SCOPE.value)

    if direction_required:
        requirement = CompetitionRequirement.DIRECTION_REQUIRED.value
        competition_type = CandidateCompetitionType.CAUSAL_DIRECTION.value
    elif mechanism_required:
        requirement = CompetitionRequirement.MECHANISM_REQUIRED.value
        competition_type = CandidateCompetitionType.MECHANISM.value
    elif len(direction_bundles) >= 2:
        requirement = CompetitionRequirement.ALTERNATIVE_DISCOVERY_REQUIRED.value
        competition_type = CandidateCompetitionType.CAUSAL_DIRECTION.value
    elif scope_groups:
        requirement = CompetitionRequirement.ALTERNATIVE_DISCOVERY_REQUIRED.value
        competition_type = (
            CandidateCompetitionType.MIXED.value
            if len(direction_bundles) >= 2 and scope_groups
            else CandidateCompetitionType.CAUSAL_DIRECTION.value
            if len(direction_bundles) >= 2
            else CandidateCompetitionType.SCOPE.value
        )
    else:
        requirement = CompetitionRequirement.NOT_REQUIRED.value
        competition_type = CandidateCompetitionType.NONE.value

    required_candidate_count = (
        2
        if requirement
        in {
            CompetitionRequirement.DIRECTION_REQUIRED.value,
            CompetitionRequirement.MECHANISM_REQUIRED.value,
            CompetitionRequirement.SCOPE_REQUIRED.value,
            CompetitionRequirement.MIXED_REQUIRED.value,
        }
        else 1
    )
    return {
        "competition_requirement": requirement,
        "competition_type": competition_type,
        "competition_axes": competition_axes,
        "required_candidate_count": required_candidate_count,
        "direction_bundles": direction_bundles,
        "scope_groups": scope_groups,
        "eligible_direction_bundle_ids": eligible_direction_ids,
        "unbounded_direction_bundle_ids": [
            str(bundle["bundle_id"])
            for bundle in direction_bundles
            if not bool(bundle["candidate_eligible"])
        ],
        "scope_opportunity_group_ids": scope_opportunity_ids,
        "scope_ready_group_ids": scope_ready_ids,
        "scope_assessment_required": scope_assessment_required,
        "scope_assessment_status": (
            ScopeAssessmentStatus.PENDING.value
            if scope_assessment_required
            else ScopeAssessmentStatus.NOT_REQUIRED.value
        ),
        "boundary_note": (
            "Python supplies factual Evidence bundles, normalized Scope facts, and "
            "root-cause competition requirements; Qwen alone proposes and compares "
            "causal mechanisms. Scope assessment cannot satisfy root-cause competition."
        ),
    }


__all__ = [
    "build_candidate_competition_brief",
    "causal_direction_identity",
    "select_diverse_lane_ids",
]
