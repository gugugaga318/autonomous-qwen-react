"""Deterministic conversion of Matrix claim gaps into legal investigation work."""

from __future__ import annotations

import hashlib
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from yield_rca_core.causal_evidence_matrix import CausalEvidenceMatrix
from yield_rca_core.causal_hypothesis import CausalClaim
from yield_rca_core.causal_investigation_models import (
    AlternativeSearchStatus,
    CandidateChallenge,
    CandidateCompetitionAxis,
    CandidateMechanismRelation,
    CandidateSemanticProfile,
    CausalLaneRecord,
    CompetitionRequirement,
)
from yield_rca_core.causal_lane_lifecycle import lane_is_searchable
from yield_rca_core.investigation_models import ActionKind
from yield_rca_core.question_capability import QUESTION_CAPABILITY_REGISTRY

_GAP_PRIORITY = {
    "data_missing": 0,
    "hypothesis_discrimination": 1,
    "contradiction": 2,
    "missing_support": 3,
}


def _claim_gap_priority(claim: str, *, unavailable: bool) -> int:
    if unavailable:
        return _GAP_PRIORITY["data_missing"]
    if claim == CausalClaim.MECHANISM.value:
        return 1
    if claim == CausalClaim.SCOPE.value:
        return 3
    return 2

_CLAIM_TO_QUESTION = {
    CausalClaim.EQUIPMENT.value: "impact_scope",
    CausalClaim.CHAMBER.value: "impact_scope",
    CausalClaim.OPERATION.value: "impact_scope",
    CausalClaim.PARAMETER.value: "spc_signal",
    CausalClaim.OUTCOME.value: "product_outcome",
    CausalClaim.MECHANISM.value: "process_mechanism",
    CausalClaim.TEMPORAL.value: "spc_signal",
    CausalClaim.SCOPE.value: "impact_scope",
    CausalClaim.CONTRADICTION.value: "process_mechanism",
}
_CLAIM_TO_GROUPS = {
    CausalClaim.EQUIPMENT.value: {"shared_exposure"},
    CausalClaim.CHAMBER.value: {"shared_exposure"},
    CausalClaim.OPERATION.value: {"shared_exposure"},
    CausalClaim.PARAMETER.value: {"process_anomaly"},
    CausalClaim.OUTCOME.value: {"product_signal"},
    CausalClaim.MECHANISM.value: {"process_anomaly", "historical_context"},
    CausalClaim.TEMPORAL.value: {"process_anomaly", "shared_exposure"},
    CausalClaim.SCOPE.value: {"shared_exposure", "impact_scope"},
    CausalClaim.CONTRADICTION.value: {"process_anomaly", "product_signal"},
}


@dataclass(frozen=True)
class _DiscriminatorDefinition:
    """Registry projection for one kind of adversarial Evidence need.

    The definition names an Evidence contribution rather than an Action.  The
    executable Action is derived from ``QUESTION_CAPABILITY_REGISTRY`` below,
    so adding or removing a capability cannot silently leave a second action
    map out of sync.
    """

    kind: str
    required_evidence_group: str
    description: str


_DISCRIMINATOR_DEFINITIONS = (
    _DiscriminatorDefinition(
        kind="parameter_anomaly",
        required_evidence_group="process_anomaly",
        description=(
            "Compare the alternative Lane's parameter direction, magnitude, "
            "and process excursion."
        ),
    ),
    _DiscriminatorDefinition(
        kind="exposure_commonality",
        required_evidence_group="shared_exposure",
        description=(
            "Verify operation, equipment, chamber, and Lot exposure commonality "
            "for the alternative Lane."
        ),
    ),
    _DiscriminatorDefinition(
        kind="recipe_commonality",
        required_evidence_group="shared_exposure",
        description=(
            "Verify whether the alternative recipe assignment is shared by the "
            "affected population."
        ),
    ),
    _DiscriminatorDefinition(
        kind="product_outcome",
        required_evidence_group="shared_product_signal",
        description=(
            "Test whether Lots exposed to the alternative Lane share a compatible "
            "defect, metrology, or electrical outcome."
        ),
    ),
    _DiscriminatorDefinition(
        kind="mechanism_context",
        required_evidence_group="historical_context",
        description=(
            "Look for approved engineering or historical support for the proposed "
            "mechanism without treating that knowledge as proof of this Lot."
        ),
    ),
)

_EVIDENCE_TYPES_BY_GROUP = {
    "process_anomaly": (
        "parameter_deviation",
        "trend_deviation",
        "ooc_event",
        "spc_violation",
        "excursion_window",
    ),
    "shared_exposure": (
        "lot_context",
        "process_exposure",
        "equipment_exposure",
        "impact_scope",
    ),
    "shared_product_signal": (
        "defect_signal",
        "metrology_deviation",
        "electrical_failure",
        "negative_signal",
    ),
    "historical_context": (
        "historical_case_match",
        "engineering_note",
        "sop_guidance",
    ),
}


def _expected_evidence_types(groups: Collection[str]) -> list[str]:
    return sorted(
        {
            evidence_type
            for group in groups
            for evidence_type in _EVIDENCE_TYPES_BY_GROUP.get(group, ())
        }
    )


def _lane_payload(lane: CausalLaneRecord | dict[str, Any]) -> dict[str, Any]:
    if isinstance(lane, CausalLaneRecord):
        return {str(key): value for key, value in lane.to_dict().items()}
    return dict(lane)


def _registered_actions_for_group(group: str) -> list[str]:
    capability = QUESTION_CAPABILITY_REGISTRY["process_mechanism"]
    return sorted(
        action
        for action in capability.allowed_actions
        if group in capability.contribution_for(action)
    )


def _discriminator_definitions_for_lane(
    lane: dict[str, Any],
    matrix: CausalEvidenceMatrix,
    *,
    source_lot_id: str | None = None,
) -> list[_DiscriminatorDefinition]:
    parameter_scope = lane.get("parameter_scope", [])
    exposed_lot_ids = lane.get("exposed_lot_ids", [])
    independent_lot_ids = {
        str(item).strip()
        for item in exposed_lot_ids
        if str(item).strip()
        and (
            source_lot_id is None
            or str(item).strip() != source_lot_id
        )
    }
    selected: list[_DiscriminatorDefinition] = []
    for definition in _DISCRIMINATOR_DEFINITIONS:
        if definition.kind == "parameter_anomaly" and not parameter_scope:
            continue
        if definition.kind == "recipe_commonality" and not lane.get("recipe"):
            continue
        if definition.kind == "product_outcome":
            if source_lot_id is not None and not independent_lot_ids:
                continue
            if source_lot_id is None and not exposed_lot_ids:
                continue
        if definition.kind == "mechanism_context":
            mechanism = matrix.claims.get(CausalClaim.MECHANISM.value)
            if mechanism is not None and mechanism.status == "supported":
                continue
        selected.append(definition)

    raw_window = lane.get("time_window", [])
    if isinstance(raw_window, list | tuple) and len(raw_window) == 2:
        selected.append(
            _DiscriminatorDefinition(
                kind="temporal_alignment",
                required_evidence_group=(
                    "process_anomaly" if parameter_scope else "shared_exposure"
                ),
                description=(
                    "Verify that the distinguishing observation falls inside the "
                    "alternative Lane's processing or excursion window."
                ),
            )
        )
    return selected


def _discriminator_information_gain(
    kind: str,
    lane: dict[str, Any],
    *,
    source_lot_id: str | None,
) -> tuple[float, list[str]]:
    """Estimate how strongly one observation can separate competing Lanes.

    The score uses only Python-owned Lane facts.  It is deliberately generic:
    equipment IDs, operation numbers, recipes, and case labels never affect the
    result.  A higher value means that the observation can add more independent,
    Lane-specific information; it is not a root-cause probability.
    """

    parameters = {
        str(item).strip()
        for item in lane.get("parameter_scope", [])
        if str(item).strip()
    }
    exposed_lots = {
        str(item).strip()
        for item in lane.get("exposed_lot_ids", [])
        if str(item).strip()
    }
    independent_lots = {
        lot_id
        for lot_id in exposed_lots
        if source_lot_id is None or lot_id != source_lot_id
    }
    raw_window = lane.get("time_window", [])
    has_window = isinstance(raw_window, list | tuple) and len(raw_window) == 2
    has_recipe = bool(str(lane.get("recipe", "")).strip())
    basis: list[str] = []

    if kind == "parameter_anomaly":
        score = 0.60 + min(0.20, 0.05 * len(parameters))
        basis.append(f"{len(parameters)} Lane-specific process parameters")
        if has_window:
            score += 0.05
            basis.append("a bounded processing window")
    elif kind == "product_outcome":
        score = 0.55 + min(0.20, 0.10 * len(independent_lots))
        basis.append(f"{len(independent_lots)} independent comparison Lots")
    elif kind == "exposure_commonality":
        score = 0.45 + min(0.15, 0.05 * len(independent_lots))
        basis.append(f"{len(independent_lots)} independently exposed Lots")
        if has_recipe:
            score += 0.05
            basis.append("a concrete recipe assignment")
    elif kind == "recipe_commonality":
        score = 0.40 + min(0.15, 0.05 * len(independent_lots))
        basis.append(f"{len(independent_lots)} independent recipe comparisons")
        if has_recipe:
            score += 0.05
            basis.append("a concrete recipe assignment")
    elif kind == "temporal_alignment":
        score = 0.50
        if has_window:
            score += 0.10
            basis.append("a bounded processing window")
        if parameters:
            score += 0.05
            basis.append("Lane-specific parameter observations")
    else:  # mechanism_context
        score = 0.45
        basis.append("approved engineering context can test plausibility")
    return round(min(0.95, score), 3), basis


def _target_scope(
    lane: dict[str, Any],
    *,
    discriminator_kind: str,
) -> dict[str, str]:
    scope = {
        "lane_id": str(lane.get("lane_id", "")),
        "operation": str(lane.get("operation", "")),
        "equipment": str(lane.get("equipment", "")),
        "chamber": str(lane.get("chamber", "")),
        "recipe": str(lane.get("recipe", "")),
        "discriminator_kind": discriminator_kind,
    }
    parameters = lane.get("parameter_scope", [])
    if isinstance(parameters, list | tuple):
        joined_parameters = ",".join(
            str(item).strip() for item in parameters if str(item).strip()
        )
        if joined_parameters:
            scope["parameters"] = joined_parameters
    raw_window = lane.get("time_window", [])
    if isinstance(raw_window, list | tuple) and len(raw_window) == 2:
        scope["window_start"] = str(raw_window[0])
        scope["window_end"] = str(raw_window[1])
    return {key: value for key, value in scope.items() if value}


def _matrix_evidence_ids_for_discriminator(
    matrix: CausalEvidenceMatrix,
    discriminator_kind: str,
) -> list[str]:
    claims_by_kind = {
        "parameter_anomaly": (CausalClaim.PARAMETER.value,),
        "product_outcome": (CausalClaim.OUTCOME.value,),
        "mechanism_context": (CausalClaim.MECHANISM.value,),
        "temporal_alignment": (CausalClaim.TEMPORAL.value,),
        "exposure_commonality": (
            CausalClaim.EQUIPMENT.value,
            CausalClaim.CHAMBER.value,
            CausalClaim.OPERATION.value,
            CausalClaim.SCOPE.value,
        ),
        "recipe_commonality": (
            CausalClaim.OPERATION.value,
            CausalClaim.SCOPE.value,
        ),
    }
    return list(
        dict.fromkeys(
            evidence_id
            for claim in claims_by_kind.get(discriminator_kind, ())
            if claim in matrix.claims
            for evidence_id in matrix.claims[claim].evidence_ids
        )
    )


def build_causal_evidence_gaps(
    matrices: Sequence[CausalEvidenceMatrix],
) -> list[dict[str, Any]]:
    """Return only gaps that Python can map to registered capabilities."""

    gaps: list[dict[str, Any]] = []
    for candidate_index, matrix in enumerate(matrices):
        for claim, result in matrix.claims.items():
            if claim == CausalClaim.CONTROL.value:
                # Controls are informative, not a mandatory investigation gap.
                continue
            if result.status == "supported":
                continue
            question_kind = _CLAIM_TO_QUESTION.get(claim, "process_mechanism")
            definition = QUESTION_CAPABILITY_REGISTRY[question_kind]
            expected_groups = _CLAIM_TO_GROUPS.get(claim, set())
            actions = sorted(
                action
                for action in definition.allowed_actions
                if not expected_groups
                or expected_groups
                & set(definition.contribution_for(action))
            )
            gaps.append(
                {
                    "gap_id": f"candidate_{candidate_index}.{claim}.{result.status}",
                    "gap_type": (
                        "data_missing"
                        if result.status == "unavailable"
                        else (
                            "contradiction"
                            if claim == CausalClaim.CONTRADICTION.value
                            else "missing_support"
                        )
                    ),
                    "priority": _claim_gap_priority(
                        claim,
                        unavailable=result.status == "unavailable",
                    ),
                    "candidate_index": candidate_index,
                    "claim": claim,
                    "status": result.status,
                    "reason": result.reason,
                    "question_kind": question_kind,
                    "allowed_actions": actions,
                    "evidence_ids": list(result.evidence_ids),
                    "data_missing_evidence_ids": (
                        list(matrix.data_missing_evidence_ids)
                        if result.status == "unavailable"
                        else []
                    ),
                    "unavailable_sources": (
                        [dict(item) for item in matrix.data_missing_sources]
                        if result.status == "unavailable"
                        else []
                    ),
                }
            )
    return sorted(
        gaps,
        key=lambda item: (
            int(item.get("priority", _GAP_PRIORITY["missing_support"])),
            int(item.get("candidate_index", 0)),
            str(item.get("claim", "")),
        ),
    )


def build_hypothesis_discrimination_gaps(
    matrices: Sequence[CausalEvidenceMatrix],
    *,
    alternative_search_status: str = AlternativeSearchStatus.NOT_SEARCHED.value,
    candidate_challenges: Sequence[CandidateChallenge] = (),
    causal_lanes: Sequence[CausalLaneRecord | dict[str, Any]] = (),
    candidate_ids: Sequence[str] = (),
    source_lot_id: str | None = None,
    consumed_discriminators: Collection[tuple[str, str]] = (),
    competition_brief: Mapping[str, Any] | None = None,
    candidate_semantic_profiles: Sequence[
        CandidateSemanticProfile | Mapping[str, Any]
    ] = (),
) -> list[dict[str, Any]]:
    """Create typed, registry-bounded gaps until candidate competition closes.

    One candidate is deliberately enough to create this gap.  The absence of a
    second Qwen proposal is not proof that a second explanation does not exist.
    Python creates per-Lane discriminator kinds and derives their Actions from
    the Question capability registry.  Qwen may select an existing gap, but it
    cannot create an Action or invent a free-form executable question.
    """

    if (
        not matrices
        or alternative_search_status
        == AlternativeSearchStatus.ALTERNATIVES_ELIMINATED.value
    ):
        return []
    challenges_by_id = {
        challenge.candidate_id: challenge for challenge in candidate_challenges
    }
    challenges_by_index = {
        index: challenge for index, challenge in enumerate(candidate_challenges)
    }
    consumed = {
        (str(lane_id), str(discriminator_kind))
        for lane_id, discriminator_kind in consumed_discriminators
        if str(lane_id) and str(discriminator_kind)
    }
    lane_payloads = [
        _lane_payload(lane)
        for lane in causal_lanes
        if lane_is_searchable(_lane_payload(lane))
    ]
    lane_payloads.sort(
        key=lambda item: (
            -float(item.get("priority_score", 0.0)),
            str(item.get("lane_id", "")),
        )
    )
    gaps: list[dict[str, Any]] = []
    lane_payload_by_id = {
        str(lane.get("lane_id", "")): lane
        for lane in lane_payloads
        if str(lane.get("lane_id", ""))
    }
    semantic_profiles: list[CandidateSemanticProfile] = []
    for raw_profile in candidate_semantic_profiles:
        try:
            semantic_profiles.append(
                raw_profile
                if isinstance(raw_profile, CandidateSemanticProfile)
                else CandidateSemanticProfile.from_dict(dict(raw_profile))
            )
        except (KeyError, TypeError, ValueError):
            # Semantic metadata is isolated from Candidate validity.  Invalid
            # profiles are already audited by Candidate generation and cannot
            # prevent a Python-owned factual Scope Gap from being emitted.
            continue

    raw_competition_requirement = str(
        (competition_brief or {}).get(
            "competition_requirement",
            CompetitionRequirement.NOT_EVALUATED.value,
        )
    ).strip()
    competition_axes = {
        str(item).strip()
        for item in (competition_brief or {}).get("competition_axes", [])
        if str(item).strip()
    }
    mechanism_competition_required = (
        raw_competition_requirement
        == CompetitionRequirement.MECHANISM_REQUIRED.value
        or (
            raw_competition_requirement
            == CompetitionRequirement.MIXED_REQUIRED.value
            and CandidateCompetitionAxis.MECHANISM.value in competition_axes
        )
    )
    reference_profiles = [
        profile
        for profile in semantic_profiles
        if profile.mechanism_relation
        == CandidateMechanismRelation.REFERENCE.value
    ]
    independent_profiles = [
        profile
        for profile in semantic_profiles
        if profile.mechanism_relation
        == CandidateMechanismRelation.INDEPENDENT_ALTERNATIVE.value
        and profile.depends_on_candidate_id is None
    ]
    mechanism_profiles = [*reference_profiles[:1], *independent_profiles]
    mechanism_candidate_ids = list(
        dict.fromkeys(profile.candidate_id for profile in mechanism_profiles)
    )
    prediction_kinds_by_candidate = {
        profile.candidate_id: {
            prediction.discriminator_kind
            for prediction in profile.distinguishing_predictions
        }
        for profile in mechanism_profiles
    }
    shared_mechanism_discriminator_kinds: set[str] = set()
    if len(mechanism_profiles) >= 2:
        shared_mechanism_discriminator_kinds = set.intersection(
            *(
                prediction_kinds_by_candidate[profile.candidate_id]
                for profile in mechanism_profiles
            )
        )
    mechanism_allowed_kinds = {
        "mechanism_context",
        *shared_mechanism_discriminator_kinds,
    }

    # Scope remains auditable, but it must not consume the next investigation
    # slot while an independent primary-mechanism competition is still open.
    # Evidence coverage across recipes/Lanes cannot substitute for a distinct
    # physical initiator.
    scope_groups = (competition_brief or {}).get("scope_groups", [])
    for raw_scope_group in scope_groups:
        if not isinstance(raw_scope_group, Mapping):
            continue
        if not bool(raw_scope_group.get("scope_competition_opportunity")):
            continue
        if bool(raw_scope_group.get("scope_evidence_complete")):
            continue
        scope_group_id = str(raw_scope_group.get("scope_group_id", "")).strip()
        scope_lane_ids = {
            str(item).strip()
            for item in raw_scope_group.get("lane_ids", [])
            if str(item).strip()
        }
        related_candidate_ids = [
            profile.candidate_id
            for profile in semantic_profiles
            if set(profile.comparison_lane_ids) & scope_lane_ids
        ]
        for missing_lane_id in raw_scope_group.get(
            "missing_process_lane_ids", []
        ):
            lane_id = str(missing_lane_id).strip()
            lane = lane_payload_by_id.get(lane_id)
            discriminator_kind = "parameter_anomaly"
            if (
                not scope_group_id
                or lane is None
                or (lane_id, discriminator_kind) in consumed
            ):
                continue
            information_gain, information_gain_basis = (
                _discriminator_information_gain(
                    discriminator_kind,
                    lane,
                    source_lot_id=source_lot_id,
                )
            )
            observation_actions = _registered_actions_for_group(
                "process_anomaly"
            )
            if not observation_actions:
                continue
            lane_token = hashlib.sha256(lane_id.encode("utf-8")).hexdigest()[:12]
            gaps.append(
                {
                    "gap_id": (
                        f"{scope_group_id}.scope_discrimination."
                        f"{discriminator_kind}.{lane_token}"
                    ),
                    "gap_type": "hypothesis_discrimination",
                    "gap_origin": "scope_competition",
                    "competition_axis": CandidateCompetitionAxis.SCOPE.value,
                    "scope_group_id": scope_group_id,
                    "discriminator_kind": discriminator_kind,
                    "lane_binding": "scope_competition",
                    "priority": 4 if mechanism_competition_required else 3,
                    "information_gain": information_gain,
                    "information_gain_by_lane": {lane_id: information_gain},
                    "information_gain_basis": [
                        *information_gain_basis,
                        "missing same-direction recipe comparison can change "
                        "Candidate scope ranking",
                    ],
                    "decision_impact": "ranking",
                    "applicable_lane_ids": [lane_id],
                    "candidate_index": 0,
                    # This Gap belongs to the competition, not to one Candidate.
                    # Keep the plural audit field without forging a single owner.
                    "candidate_id": "",
                    "candidate_ids": list(dict.fromkeys(related_candidate_ids)),
                    "supports_candidate_ids": list(
                        dict.fromkeys(related_candidate_ids)
                    ),
                    "weakens_candidate_ids": list(
                        dict.fromkeys(related_candidate_ids)
                    ),
                    "can_change_ranking": True,
                    "claim": "scope_discrimination",
                    "status": "unresolved",
                    "reason": (
                        "A same-direction recipe comparison lacks process "
                        "observations needed to distinguish shared-effect from "
                        "recipe-specific or differential-sensitivity scope."
                    ),
                    "question_kind": "process_mechanism",
                    "allowed_actions": list(
                        dict.fromkeys(
                            [
                                *observation_actions,
                                ActionKind.RUN_RCA_REASONING.value,
                            ]
                        )
                    ),
                    "preferred_action": (
                        observation_actions[0]
                        if len(observation_actions) == 1
                        else ""
                    ),
                    "refresh_action": ActionKind.RUN_RCA_REASONING.value,
                    "required_evidence_groups": ["process_anomaly"],
                    "expected_evidence_types": _expected_evidence_types(
                        ["process_anomaly"]
                    ),
                    "target_scope": _target_scope(
                        lane,
                        discriminator_kind=discriminator_kind,
                    ),
                    "evidence_ids": [],
                    "challenge_selected": False,
                }
            )
    for candidate_index in range(len(matrices)):
        candidate_id = (
            str(candidate_ids[candidate_index])
            if candidate_index < len(candidate_ids)
            else ""
        )
        challenge = (
            challenges_by_id.get(candidate_id)
            if candidate_id
            else challenges_by_index.get(candidate_index)
        )
        selected_gap_ids = (
            set(challenge.distinguishing_gap_ids) if challenge is not None else set()
        )
        selected_lane_id = (
            challenge.strongest_alternative_lane_id
            if challenge is not None
            else None
        )

        candidate_lane_payloads = list(lane_payloads)
        if selected_lane_id:
            candidate_lane_payloads = [
                lane
                for lane in lane_payloads
                if str(lane.get("lane_id", "")) == selected_lane_id
            ]

        definitions_by_kind: dict[
            str,
            list[tuple[_DiscriminatorDefinition, dict[str, Any]]],
        ] = {}
        for lane in candidate_lane_payloads:
            if not str(lane.get("lane_id", "")).strip():
                continue
            lane_definitions = _discriminator_definitions_for_lane(
                lane,
                matrices[candidate_index],
                source_lot_id=source_lot_id,
            )
            if mechanism_competition_required:
                lane_definitions = [
                    definition
                    for definition in lane_definitions
                    if definition.kind in mechanism_allowed_kinds
                ]
                if not any(
                    definition.kind == "mechanism_context"
                    for definition in lane_definitions
                ):
                    lane_definitions.append(
                        next(
                            definition
                            for definition in _DISCRIMINATOR_DEFINITIONS
                            if definition.kind == "mechanism_context"
                        )
                    )
            for definition in lane_definitions:
                definitions_by_kind.setdefault(definition.kind, []).append(
                    (definition, lane)
                )

        for kind, definition_lanes in sorted(definitions_by_kind.items()):
            definition_lanes = [
                (definition, lane)
                for definition, lane in definition_lanes
                if (str(lane.get("lane_id", "")), kind) not in consumed
            ]
            if not definition_lanes:
                continue
            gap_id = (
                f"candidate_{candidate_index}.hypothesis_discrimination.{kind}"
            )
            if (
                selected_gap_ids
                and gap_id not in selected_gap_ids
                and not mechanism_competition_required
            ):
                continue
            required_groups = sorted(
                {
                    definition.required_evidence_group
                    for definition, _lane in definition_lanes
                }
            )
            information_gain_by_lane = {
                str(lane["lane_id"]): _discriminator_information_gain(
                    kind,
                    lane,
                    source_lot_id=source_lot_id,
                )[0]
                for _definition, lane in definition_lanes
            }
            information_gain = max(information_gain_by_lane.values())
            information_gain_basis = list(
                dict.fromkeys(
                    basis
                    for _definition, lane in definition_lanes
                    for basis in _discriminator_information_gain(
                        kind,
                        lane,
                        source_lot_id=source_lot_id,
                    )[1]
                )
            )
            observation_actions = sorted(
                {
                    action
                    for group in required_groups
                    for action in _registered_actions_for_group(group)
                }
            )
            if not observation_actions:
                continue
            preferred_action = (
                observation_actions[0]
                if len(observation_actions) == 1
                else ""
            )
            bound_lane = (
                candidate_lane_payloads[0]
                if selected_lane_id and len(candidate_lane_payloads) == 1
                else None
            )
            gaps.append(
                {
                    "gap_id": gap_id,
                    "gap_type": "hypothesis_discrimination",
                    "competition_axis": (
                        CandidateCompetitionAxis.MECHANISM.value
                        if mechanism_competition_required
                        else CandidateCompetitionAxis.DIRECTION.value
                    ),
                    "discriminator_kind": kind,
                    "lane_binding": (
                        "bound" if bound_lane is not None else "challenge_selected"
                    ),
                    "priority": _GAP_PRIORITY["hypothesis_discrimination"],
                    "information_gain": information_gain,
                    "information_gain_by_lane": information_gain_by_lane,
                    "information_gain_basis": information_gain_basis,
                    "applicable_lane_ids": sorted(information_gain_by_lane),
                    "candidate_index": candidate_index,
                    "candidate_id": candidate_id,
                    "candidate_ids": (
                        mechanism_candidate_ids
                        if mechanism_competition_required
                        else ([candidate_id] if candidate_id else [])
                    ),
                    "supports_candidate_ids": (
                        mechanism_candidate_ids
                        if mechanism_competition_required
                        else ([candidate_id] if candidate_id else [])
                    ),
                    "weakens_candidate_ids": (
                        mechanism_candidate_ids
                        if mechanism_competition_required
                        else ([candidate_id] if candidate_id else [])
                    ),
                    "can_change_ranking": True,
                    "claim": "hypothesis_discrimination",
                    "status": "unresolved",
                    "reason": (
                        "The candidate has not completed an adversarial search "
                        "against the strongest competing explanation. "
                        + definition_lanes[0][0].description
                    ),
                    "question_kind": "process_mechanism",
                    "allowed_actions": list(
                        dict.fromkeys(
                            [
                                *observation_actions,
                                ActionKind.RUN_RCA_REASONING.value,
                            ]
                        )
                    ),
                    "preferred_action": preferred_action,
                    "refresh_action": ActionKind.RUN_RCA_REASONING.value,
                    "required_evidence_groups": required_groups,
                    "expected_evidence_types": _expected_evidence_types(
                        required_groups
                    ),
                    "target_scope": (
                        _target_scope(bound_lane, discriminator_kind=kind)
                        if bound_lane is not None
                        else {}
                    ),
                    "evidence_ids": _matrix_evidence_ids_for_discriminator(
                        matrices[candidate_index],
                        kind,
                    ),
                    "challenge_selected": gap_id in selected_gap_ids,
                }
            )

        if lane_payloads:
            continue

        # Backward compatibility for pre-Batch-25 State without concrete Lane
        # records.  This remains broad and non-auto-selected because Python has
        # no factual target scope from which to derive a typed discriminator.
        gap_id = f"candidate_{candidate_index}.hypothesis_discrimination"
        gaps.append(
            {
                "gap_id": gap_id,
                "gap_type": "hypothesis_discrimination",
                "competition_axis": (
                    CandidateCompetitionAxis.MECHANISM.value
                    if mechanism_competition_required
                    else CandidateCompetitionAxis.DIRECTION.value
                ),
                "discriminator_kind": "legacy_unscoped",
                "priority": _GAP_PRIORITY["hypothesis_discrimination"],
                "information_gain": 0.0,
                "information_gain_by_lane": {},
                "information_gain_basis": [
                    "Legacy State has no concrete Lane facts to score."
                ],
                "applicable_lane_ids": [],
                "candidate_index": candidate_index,
                "candidate_id": candidate_id,
                "candidate_ids": (
                    mechanism_candidate_ids
                    if mechanism_competition_required
                    else ([candidate_id] if candidate_id else [])
                ),
                "supports_candidate_ids": (
                    mechanism_candidate_ids
                    if mechanism_competition_required
                    else ([candidate_id] if candidate_id else [])
                ),
                "weakens_candidate_ids": (
                    mechanism_candidate_ids
                    if mechanism_competition_required
                    else ([candidate_id] if candidate_id else [])
                ),
                "can_change_ranking": False,
                "claim": "hypothesis_discrimination",
                "status": "unresolved",
                "reason": (
                    "No concrete causal Lane is available, so the alternative "
                    "search cannot yet be scoped to a typed discriminator."
                ),
                "question_kind": "process_mechanism",
                "allowed_actions": sorted(
                    QUESTION_CAPABILITY_REGISTRY["process_mechanism"].allowed_actions
                ),
                "preferred_action": "",
                "refresh_action": ActionKind.RUN_RCA_REASONING.value,
                "required_evidence_groups": [],
                "expected_evidence_types": [],
                "target_scope": {},
                "evidence_ids": [],
                "challenge_selected": gap_id in selected_gap_ids,
            }
        )
    return sorted(
        gaps,
        key=lambda item: (
            int(item.get("priority", _GAP_PRIORITY["hypothesis_discrimination"])),
            int(item.get("candidate_index", 0)),
            -float(item.get("information_gain", 0.0)),
            str(item.get("gap_id", "")),
        ),
    )


__all__ = [
    "build_causal_evidence_gaps",
    "build_hypothesis_discrimination_gaps",
]
