"""Python-bounded adversarial challenge for Qwen RCA candidates.

Qwen is allowed to identify the strongest alternative and explain which
Evidence would distinguish the candidates.  Python owns the candidate and
Evidence identifiers, validates the references, and derives the competition
status used by the Confirmation Gate.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from yield_rca_core.causal_evidence_matrix import (
    CausalEvidenceMatrix,
    compact_causal_evidence_matrix_for_prompt,
)
from yield_rca_core.causal_investigation_models import (
    AlternativeLaneResolution,
    AlternativeLaneResolutionStatus,
    AlternativeSearchStatus,
    CandidateChallenge,
    CandidateCompetitionAxis,
    CandidateMechanismRelation,
    ChallengeKind,
    ChallengeStatus,
    CompetitionRequirement,
)
from yield_rca_core.evidence_models import Evidence
from yield_rca_core.evidence_synthesis import compact_evidence_prompt_card
from yield_rca_core.llm_gateway import (
    LLMCallError,
    LLMClient,
    LLMOutputValidationError,
    LLMRequest,
    llm_call_budget_available,
)
from yield_rca_core.models import AgentKind, ModelValidationError

_OUTPUT_ATTEMPTS = 2
_MAX_CHALLENGE_PAYLOAD_CHARS = 64_000


@dataclass(frozen=True)
class AdversarialChallengeGeneration:
    """Result of one bounded Qwen challenge request."""

    challenges: tuple[CandidateChallenge, ...]
    attempt_count: int
    validation_errors: tuple[str, ...] = ()
    output_invalid: bool = False
    repair_skipped_due_to_budget: bool = False
    alternative_search_status: str = AlternativeSearchStatus.NOT_SEARCHED.value
    lane_resolutions: tuple[AlternativeLaneResolution, ...] = ()
    prompt_audit: dict[str, Any] | None = None


def _compact_lane_context_for_prompt(value: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
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
    return {key: value[key] for key in keys if key in value}


def _payload_char_count(payload: Mapping[str, Any]) -> int:
    return len(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    )


def _string_list(value: object, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise LLMOutputValidationError(f"{field_name} must be an array")
    values = tuple(str(item).strip() for item in value)
    if any(not item for item in values):
        raise LLMOutputValidationError(f"{field_name} must contain non-empty strings")
    if len(values) != len(set(values)):
        raise LLMOutputValidationError(f"{field_name} must not contain duplicates")
    return values


def _gap_information_gain_for_lane(
    gap: Mapping[str, Any],
    lane_id: str,
) -> float | None:
    raw_by_lane = gap.get("information_gain_by_lane")
    if isinstance(raw_by_lane, Mapping):
        raw_score = raw_by_lane.get(lane_id)
        if isinstance(raw_score, int | float):
            return float(raw_score)
        return None
    raw_score = gap.get("information_gain")
    if isinstance(raw_score, int | float):
        return float(raw_score)
    return None


def _gap_applies_to_lane(gap: Mapping[str, Any], lane_id: str) -> bool:
    """Return whether a typed Gap can produce an observation on one Lane."""

    raw_scope = gap.get("target_scope", {})
    target_lane_id = (
        str(raw_scope.get("lane_id", "")).strip()
        if isinstance(raw_scope, Mapping)
        else ""
    )
    if target_lane_id:
        return target_lane_id == lane_id
    raw_applicable_lanes = gap.get("applicable_lane_ids", [])
    applicable_lanes = (
        {
            str(item).strip()
            for item in raw_applicable_lanes
            if str(item).strip()
        }
        if isinstance(raw_applicable_lanes, list | tuple)
        else set()
    )
    if applicable_lanes:
        return lane_id in applicable_lanes
    raw_by_lane = gap.get("information_gain_by_lane")
    if isinstance(raw_by_lane, Mapping) and raw_by_lane:
        return lane_id in raw_by_lane
    return True


def _normalize_challenge_payload(
    payload: object,
    *,
    candidate_ids: set[str],
    lane_ids: set[str],
    evidence_ids: set[str],
    gap_ids: set[str],
    gap_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    evidence_by_id: Mapping[str, Evidence] | None = None,
    lane_contexts: Mapping[str, Mapping[str, Any]] | None = None,
) -> CandidateChallenge:
    if not isinstance(payload, Mapping):
        raise LLMOutputValidationError("candidate challenge must be an object")
    candidate_id = str(payload.get("candidate_id", "")).strip()
    if not candidate_id or candidate_id not in candidate_ids:
        raise LLMOutputValidationError(
            f"candidate challenge references unknown candidate_id: {candidate_id!r}"
        )

    alternative_candidate_id = payload.get("alternative_candidate_id")
    normalized_alternative_candidate_id = (
        str(alternative_candidate_id).strip()
        if alternative_candidate_id is not None
        else None
    )
    if normalized_alternative_candidate_id == "":
        normalized_alternative_candidate_id = None
    if (
        normalized_alternative_candidate_id is not None
        and normalized_alternative_candidate_id not in candidate_ids
    ):
        raise LLMOutputValidationError(
            "candidate challenge references unknown alternative_candidate_id"
        )
    if normalized_alternative_candidate_id == candidate_id:
        raise LLMOutputValidationError(
            "alternative_candidate_id must differ from candidate_id"
        )

    alternative = payload.get("evidence_probe_lane_id")
    if alternative is None:
        alternative = payload.get("strongest_alternative_lane_id")
    if alternative is None:
        # The public design wording calls this field strongest_alternative;
        # accept the alias but normalize it into the Python-owned state model.
        alternative = payload.get("strongest_alternative")
    alternative_id = (
        str(alternative).strip() if alternative is not None else None
    )
    if alternative_id == "":
        alternative_id = None
    if alternative_id is not None and alternative_id not in lane_ids:
        raise LLMOutputValidationError(
            "candidate challenge references an unknown strongest alternative Lane"
        )
    raw_challenge_kind = payload.get("challenge_kind")
    if raw_challenge_kind is None:
        challenge_kind = (
            ChallengeKind.CANDIDATE_DIRECTION.value
            if normalized_alternative_candidate_id is not None
            else ChallengeKind.LANE_PROBE.value
        )
    else:
        try:
            challenge_kind = ChallengeKind(str(raw_challenge_kind)).value
        except ValueError as exc:
            allowed = ", ".join(item.value for item in ChallengeKind)
            raise LLMOutputValidationError(
                f"challenge_kind must be one of: {allowed}"
            ) from exc
    if (
        challenge_kind
        in {
            ChallengeKind.CANDIDATE_DIRECTION.value,
            ChallengeKind.SCOPE.value,
        }
        and normalized_alternative_candidate_id is None
    ):
        raise LLMOutputValidationError(
            "candidate-direction and scope challenges require "
            "alternative_candidate_id"
        )
    if (
        challenge_kind == ChallengeKind.MECHANISM.value
        and len(candidate_ids) >= 2
        and normalized_alternative_candidate_id is None
    ):
        raise LLMOutputValidationError(
            "a two-Candidate mechanism challenge requires "
            "alternative_candidate_id"
        )
    if (
        challenge_kind == ChallengeKind.MECHANISM.value
        and normalized_alternative_candidate_id is None
        and alternative_id is None
    ):
        raise LLMOutputValidationError(
            "a one-Candidate mechanism probe requires evidence_probe_lane_id"
        )
    if (
        challenge_kind == ChallengeKind.LANE_PROBE.value
        and normalized_alternative_candidate_id is not None
    ):
        raise LLMOutputValidationError(
            "lane_probe requires alternative_candidate_id=null; a causal Lane "
            "must be supplied only as evidence_probe_lane_id"
        )
    raw_mechanism_relation = payload.get("mechanism_relation")
    if challenge_kind == ChallengeKind.MECHANISM.value:
        if raw_mechanism_relation is None:
            raise LLMOutputValidationError(
                "mechanism challenges require mechanism_relation"
            )
        try:
            mechanism_relation = CandidateMechanismRelation(
                str(raw_mechanism_relation)
            ).value
        except ValueError as exc:
            allowed = ", ".join(item.value for item in CandidateMechanismRelation)
            raise LLMOutputValidationError(
                f"mechanism_relation must be one of: {allowed}"
            ) from exc
        if (
            normalized_alternative_candidate_id is not None
            and mechanism_relation
            != CandidateMechanismRelation.INDEPENDENT_ALTERNATIVE.value
        ):
            raise LLMOutputValidationError(
                "a two-Candidate mechanism challenge must independently verify "
                "mechanism_relation=independent_alternative"
            )
        if (
            normalized_alternative_candidate_id is None
            and mechanism_relation
            != CandidateMechanismRelation.UNKNOWN.value
        ):
            raise LLMOutputValidationError(
                "a one-Candidate mechanism probe must use "
                "mechanism_relation=unknown"
            )
    else:
        mechanism_relation = CandidateMechanismRelation.UNKNOWN.value

    supporting = _string_list(
        payload.get("supporting_evidence_ids", []),
        "supporting_evidence_ids",
    )
    contradicting = _string_list(
        payload.get("contradicting_evidence_ids", []),
        "contradicting_evidence_ids",
    )
    precursor = _string_list(
        payload.get("unexplained_precursor_evidence_ids", []),
        "unexplained_precursor_evidence_ids",
    )
    referenced_evidence = set(supporting + contradicting + precursor)
    unknown_evidence = sorted(referenced_evidence - evidence_ids)
    if unknown_evidence:
        raise LLMOutputValidationError(
            f"candidate challenge references unknown Evidence IDs: {unknown_evidence}"
        )
    if set(supporting) & set(contradicting):
        raise LLMOutputValidationError(
            "candidate challenge Evidence cannot be both supporting and contradicting"
        )
    if evidence_by_id is not None:
        for evidence_id in referenced_evidence:
            evidence = evidence_by_id.get(evidence_id)
            if evidence is None:
                continue
            lane_value = evidence.metadata.get("lane_id")
            if lane_value is not None and str(lane_value).strip() not in lane_ids:
                raise LLMOutputValidationError(
                    "candidate challenge Evidence references an unknown causal Lane"
                )

    gap_values = payload.get("distinguishing_gap_ids", [])
    gaps = _string_list(gap_values, "distinguishing_gap_ids")
    unknown_gaps = sorted(set(gaps) - gap_ids)
    if unknown_gaps:
        raise LLMOutputValidationError(
            f"candidate challenge references unknown Evidence Gaps: {unknown_gaps}"
        )
    if gap_by_id is not None:
        for gap_id in gaps:
            gap = gap_by_id.get(gap_id)
            if gap is None:
                continue
            gap_candidate_id = str(gap.get("candidate_id", "")).strip()
            if gap_candidate_id and gap_candidate_id != candidate_id:
                raise LLMOutputValidationError(
                    "candidate challenge selected an Evidence Gap owned by another "
                    "candidate"
                )
            raw_scope = gap.get("target_scope", {})
            target_lane_id = (
                str(raw_scope.get("lane_id", "")).strip()
                if isinstance(raw_scope, Mapping)
                else ""
            )
            lane_binding = str(gap.get("lane_binding", "")).strip()
            raw_applicable_lanes = gap.get("applicable_lane_ids", [])
            applicable_lanes = (
                {
                    str(item).strip()
                    for item in raw_applicable_lanes
                    if str(item).strip()
                }
                if isinstance(raw_applicable_lanes, list | tuple)
                else set()
            )
            template_binding = (
                lane_binding == "challenge_selected" and not target_lane_id
            )
            if (
                alternative_id is not None
                and target_lane_id != alternative_id
                and not template_binding
            ):
                raise LLMOutputValidationError(
                    "candidate challenge selected a discriminator Gap for a "
                    "different causal Lane"
                )
            if (
                alternative_id is not None
                and applicable_lanes
                and alternative_id not in applicable_lanes
            ):
                raise LLMOutputValidationError(
                    "candidate challenge selected a discriminator Gap that cannot "
                    "produce an independent observation for the strongest "
                    "alternative Lane"
                )
    questions = _string_list(
        payload.get("distinguishing_questions", []),
        "distinguishing_questions",
    )

    explanation = payload.get("challenge_explanation")
    if explanation is None:
        explanation = payload.get("explanation", "")
    if not isinstance(explanation, str) or not explanation.strip():
        raise LLMOutputValidationError("challenge_explanation must be non-empty")

    raw_status = str(payload.get("status", ChallengeStatus.OPEN.value)).strip()
    try:
        status = ChallengeStatus(raw_status).value
    except ValueError as exc:
        allowed = ", ".join(item.value for item in ChallengeStatus)
        raise LLMOutputValidationError(
            f"candidate challenge status must be one of: {allowed}"
        ) from exc
    if status == ChallengeStatus.RESOLVED.value and precursor:
        raise LLMOutputValidationError(
            "resolved challenge cannot retain unexplained precursor Evidence"
        )
    if alternative_id is not None and status == ChallengeStatus.RESOLVED.value:
        if not (supporting or contradicting):
            raise LLMOutputValidationError(
                "an eliminated alternative Lane requires distinguishing Evidence"
            )
    if alternative_id is not None and status not in {
        ChallengeStatus.RESOLVED.value,
        ChallengeStatus.BLOCKED.value,
    }:
        allow_exhausted_non_discriminative = False
        if (
            status == ChallengeStatus.NON_DISCRIMINATIVE.value
            and not gaps
            and gap_by_id is not None
        ):
            applicable_gaps = [
                gap
                for gap in gap_by_id.values()
                if str(gap.get("gap_type", "")) == "hypothesis_discrimination"
                and str(gap.get("candidate_id", "")).strip() in {"", candidate_id}
                and (
                    alternative_id
                    in {
                        str(item).strip()
                        for item in gap.get("applicable_lane_ids", [])
                        if str(item).strip()
                    }
                    or str(gap.get("target_scope", {}).get("lane_id", "")).strip()
                    == alternative_id
                )
            ]
            allow_exhausted_non_discriminative = not applicable_gaps
        if len(gaps) != 1 and not allow_exhausted_non_discriminative:
            raise LLMOutputValidationError(
                "an unresolved named alternative Lane requires exactly one "
                "highest-information-gain Python-generated typed discriminator Gap"
            )
        if gap_by_id is not None and gaps:
            selected_gap_type = str(
                gap_by_id.get(gaps[0], {}).get("gap_type", "")
            ).strip()
            if selected_gap_type != "hypothesis_discrimination":
                raise LLMOutputValidationError(
                    "an unresolved named alternative Lane must select a "
                    "hypothesis_discrimination typed Gap"
                )
            scored_gaps = {
                gap_id: score
                for gap_id, gap in gap_by_id.items()
                if str(gap.get("gap_type", "")) == "hypothesis_discrimination"
                and str(gap.get("candidate_id", "")).strip() in {"", candidate_id}
                and _gap_applies_to_lane(gap, alternative_id)
                and (
                    score := _gap_information_gain_for_lane(gap, alternative_id)
                )
                is not None
            }
            selected_score = scored_gaps.get(gaps[0])
            if scored_gaps and selected_score is not None:
                highest_score = max(scored_gaps.values())
                if selected_score < highest_score:
                    highest_gap_ids = sorted(
                        gap_id
                        for gap_id, score in scored_gaps.items()
                        if score == highest_score
                    )
                    raise LLMOutputValidationError(
                        "selected discriminator is not the highest-information-gain "
                        f"observation for the strongest alternative Lane; choose one "
                        f"of {highest_gap_ids}"
                    )
    if alternative_id is not None and lane_contexts:
        lane = lane_contexts.get(alternative_id)
        if lane is None:
            raise LLMOutputValidationError(
                "strongest alternative Lane has no Python-owned context"
            )
        if gap_by_id is not None:
            for gap_id in gaps:
                gap = gap_by_id.get(gap_id, {})
                kind = str(gap.get("discriminator_kind", ""))
                if kind == "parameter_anomaly" and not lane.get("parameter_scope"):
                    raise LLMOutputValidationError(
                        "parameter_anomaly discriminator is not applicable to the "
                        "selected causal Lane"
                    )
                if kind == "recipe_commonality" and not lane.get("recipe"):
                    raise LLMOutputValidationError(
                        "recipe_commonality discriminator is not applicable to the "
                        "selected causal Lane"
                    )
                if kind == "product_outcome" and not lane.get("exposed_lot_ids"):
                    raise LLMOutputValidationError(
                        "product_outcome discriminator is not applicable to the "
                        "selected causal Lane"
                    )
                raw_window = lane.get("time_window", [])
                if kind == "temporal_alignment" and not (
                    isinstance(raw_window, list | tuple) and len(raw_window) == 2
                ):
                    raise LLMOutputValidationError(
                        "temporal_alignment discriminator is not applicable to the "
                        "selected causal Lane"
                    )
        lane_evidence = [
            evidence_by_id[evidence_id]
            for evidence_id in referenced_evidence
            if evidence_by_id is not None and evidence_id in evidence_by_id
        ]
        if not any(_evidence_matches_lane(item, lane) for item in lane_evidence):
            raise LLMOutputValidationError(
                "challenge does not cite Entity/time-consistent Evidence for the "
                "strongest alternative Lane"
            )
    return CandidateChallenge(
        candidate_id=candidate_id,
        alternative_candidate_id=normalized_alternative_candidate_id,
        evidence_probe_lane_id=alternative_id,
        challenge_kind=challenge_kind,
        mechanism_relation=mechanism_relation,
        strongest_alternative_lane_id=alternative_id,
        strongest_alternative=alternative_id,
        supporting_evidence_ids=supporting,
        contradicting_evidence_ids=contradicting,
        unexplained_precursor_evidence_ids=precursor,
        distinguishing_gap_ids=gaps,
        distinguishing_questions=questions,
        challenge_explanation=explanation.strip(),
        status=status,
    )


def _normalized_entity(value: object) -> str:
    normalized = str(value or "").strip().upper()
    for prefix in ("OP_", "EQ_", "CH_", "RCP_"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
    return normalized


def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        return parsed.astimezone(UTC).replace(tzinfo=None)
    return parsed


def _evidence_matches_lane(
    evidence: Evidence,
    lane: Mapping[str, Any],
) -> bool:
    """Validate a challenge citation against Python-owned Lane facts."""

    lane_id = str(lane.get("lane_id", "")).strip()
    evidence_lane_id = str(evidence.metadata.get("lane_id", "")).strip()
    if evidence_lane_id and evidence_lane_id != lane_id:
        return False

    entity_types = {
        "operation": "operation",
        "equipment": "equipment",
        "chamber": "chamber",
        "recipe": "recipe",
    }
    matched_fact = bool(evidence_lane_id and evidence_lane_id == lane_id)
    for lane_field, entity_type in entity_types.items():
        expected = _normalized_entity(lane.get(lane_field))
        if not expected:
            continue
        observed = {
            _normalized_entity(entity.entity_id)
            for entity in evidence.entities
            if entity.entity_type == entity_type
        }
        if observed and expected not in observed:
            return False
        matched_fact = matched_fact or expected in observed

    raw_window = lane.get("time_window", [])
    if (
        evidence.timestamp
        and isinstance(raw_window, (list, tuple))
        and len(raw_window) == 2
    ):
        observed_at = _parse_time(evidence.timestamp)
        start = _parse_time(raw_window[0])
        end = _parse_time(raw_window[1])
        if observed_at is not None and start is not None and end is not None:
            if not start <= observed_at <= end:
                return False
            matched_fact = True
    return matched_fact


def _candidate_ids(candidates: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    result: list[str] = []
    for index, candidate in enumerate(candidates):
        candidate_id = str(
            candidate.get("candidate_id")
            or candidate.get("hypothesis_id")
            or f"candidate_{index}"
        ).strip()
        if candidate_id and candidate_id not in result:
            result.append(candidate_id)
    return tuple(result)


def _challenge_output_contract(
    *,
    candidate_ids: Sequence[str],
    active_lane_ids: Sequence[str],
    evidence_gaps: Sequence[Mapping[str, Any]] = (),
    candidate_competition: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return explicit ID/kind choices for the current competition phase."""

    normalized_candidate_ids = tuple(
        dict.fromkeys(str(item).strip() for item in candidate_ids if str(item).strip())
    )
    requirement = str(
        (candidate_competition or {}).get("competition_requirement", "")
    )
    discovery_probe = (
        len(candidate_ids) == 1
        and requirement == CompetitionRequirement.ALTERNATIVE_DISCOVERY_REQUIRED.value
    )
    mechanism_probe = (
        len(candidate_ids) == 1
        and requirement == CompetitionRequirement.MECHANISM_REQUIRED.value
    )
    semantic_profiles_complete = bool(
        (candidate_competition or {}).get("semantic_profiles_complete")
    )
    required_challenge_kind: str | None = None
    if discovery_probe:
        required_challenge_kind = ChallengeKind.LANE_PROBE.value
    elif requirement == CompetitionRequirement.MECHANISM_REQUIRED.value and len(
        candidate_ids
    ) >= 1:
        required_challenge_kind = ChallengeKind.MECHANISM.value
    elif (
        requirement == CompetitionRequirement.SCOPE_REQUIRED.value
        and len(candidate_ids) >= 2
        and semantic_profiles_complete
    ):
        required_challenge_kind = ChallengeKind.SCOPE.value
    elif requirement in {
        CompetitionRequirement.DIRECTION_REQUIRED.value,
        CompetitionRequirement.MIXED_REQUIRED.value,
    } and len(candidate_ids) >= 2:
        required_challenge_kind = ChallengeKind.CANDIDATE_DIRECTION.value
    required_competition_axis = (
        CandidateCompetitionAxis.MECHANISM.value
        if required_challenge_kind == ChallengeKind.MECHANISM.value
        else None
    )

    def gap_belongs_to_candidate(
        gap: Mapping[str, Any], candidate_id: str
    ) -> bool:
        return (
            str(gap.get("gap_type", "")).strip()
            == "hypothesis_discrimination"
            and (
                required_competition_axis is None
                or str(gap.get("competition_axis", "")).strip()
                == required_competition_axis
            )
            and (
                str(gap.get("candidate_id", "")).strip() == candidate_id
                or candidate_id
                in {
                    str(item).strip()
                    for item in gap.get("candidate_ids", [])
                    if str(item).strip()
                }
            )
            and bool(str(gap.get("gap_id", "")).strip())
        )

    candidate_gaps = {
        candidate_id: [
            gap
            for gap in evidence_gaps
            if gap_belongs_to_candidate(gap, candidate_id)
        ]
        for candidate_id in normalized_candidate_ids
    }
    allowed_gap_ids_by_candidate = {
        candidate_id: sorted(
            str(gap.get("gap_id", "")).strip() for gap in gaps
        )
        for candidate_id, gaps in candidate_gaps.items()
    }
    scope_gap_lane_ids = tuple(
        dict.fromkeys(
            str(target_scope.get("lane_id", "")).strip()
            for gap in evidence_gaps
            if str(gap.get("gap_origin", "")).strip() == "scope_competition"
            and isinstance(
                target_scope := gap.get("target_scope", {}),
                Mapping,
            )
            and str(target_scope.get("lane_id", "")).strip()
        )
    )
    allowed_probe_lane_ids = (
        scope_gap_lane_ids
        if required_challenge_kind == ChallengeKind.SCOPE.value
        and scope_gap_lane_ids
        else tuple(
            dict.fromkeys(str(item) for item in active_lane_ids if str(item))
        )
    )
    allowed_gap_ids_by_candidate_and_lane: dict[str, dict[str, list[str]]] = {}
    highest_information_gain_gap_ids_by_candidate_and_lane: dict[
        str, dict[str, list[str]]
    ] = {}
    for candidate_id, gaps in candidate_gaps.items():
        allowed_by_lane: dict[str, list[str]] = {}
        highest_by_lane: dict[str, list[str]] = {}
        for lane_id in allowed_probe_lane_ids:
            applicable = [
                gap for gap in gaps if _gap_applies_to_lane(gap, lane_id)
            ]
            allowed_by_lane[lane_id] = sorted(
                str(gap.get("gap_id", "")).strip() for gap in applicable
            )
            scored = [
                (str(gap.get("gap_id", "")).strip(), score)
                for gap in applicable
                if (
                    score := _gap_information_gain_for_lane(gap, lane_id)
                )
                is not None
            ]
            highest_by_lane[lane_id] = (
                sorted(
                    gap_id
                    for gap_id, score in scored
                    if score == max(item[1] for item in scored)
                )
                if scored
                else list(allowed_by_lane[lane_id])
            )
        allowed_gap_ids_by_candidate_and_lane[candidate_id] = allowed_by_lane
        highest_information_gain_gap_ids_by_candidate_and_lane[candidate_id] = (
            highest_by_lane
        )
    return {
        "allowed_candidate_ids": list(normalized_candidate_ids),
        "allowed_alternative_candidate_ids": (
            []
            if discovery_probe or mechanism_probe
            else list(normalized_candidate_ids)
        ),
        "allowed_evidence_probe_lane_ids": list(allowed_probe_lane_ids),
        "required_challenge_kind": required_challenge_kind,
        "required_competition_axis": required_competition_axis,
        "required_mechanism_relation": (
            CandidateMechanismRelation.UNKNOWN.value
            if mechanism_probe
            else CandidateMechanismRelation.INDEPENDENT_ALTERNATIVE.value
            if required_challenge_kind == ChallengeKind.MECHANISM.value
            else None
        ),
        "required_alternative_candidate_id": (
            None if discovery_probe or mechanism_probe else "listed_only"
        ),
        "allowed_gap_ids_by_candidate": allowed_gap_ids_by_candidate,
        "allowed_gap_ids_by_candidate_and_lane": (
            allowed_gap_ids_by_candidate_and_lane
        ),
        "highest_information_gain_gap_ids_by_candidate_and_lane": (
            highest_information_gain_gap_ids_by_candidate_and_lane
        ),
        "boundary": (
            "When required_challenge_kind=lane_probe, set "
            "alternative_candidate_id=null and choose one listed "
            "evidence_probe_lane_id. A Lane is never a candidate ID."
        ),
    }


def derive_alternative_lane_resolutions(
    *,
    challenges: Sequence[CandidateChallenge],
    active_lane_ids: Sequence[str] = (),
    eliminated_lane_ids: Sequence[str] = (),
    blocked_lane_ids: Sequence[str] = (),
) -> tuple[AlternativeLaneResolution, ...]:
    """Derive one conservative, Python-owned result for every searched Lane."""

    active = list(dict.fromkeys(str(item) for item in active_lane_ids if str(item)))
    eliminated = set(str(item) for item in eliminated_lane_ids if str(item))
    blocked = set(str(item) for item in blocked_lane_ids if str(item))
    lane_ids = list(dict.fromkeys([*active, *sorted(eliminated), *sorted(blocked)]))
    challenges_by_lane: dict[str, list[CandidateChallenge]] = {}
    for challenge in challenges:
        lane_id = challenge.strongest_alternative_lane_id
        if lane_id is None:
            continue
        challenges_by_lane.setdefault(lane_id, []).append(challenge)
        if lane_id not in lane_ids:
            lane_ids.append(lane_id)

    resolutions: list[AlternativeLaneResolution] = []
    for lane_id in lane_ids:
        lane_challenges = challenges_by_lane.get(lane_id, [])
        evidence_ids = tuple(
            dict.fromkeys(
                evidence_id
                for challenge in lane_challenges
                for evidence_id in (
                    *challenge.supporting_evidence_ids,
                    *challenge.contradicting_evidence_ids,
                )
            )
        )
        gap_ids = tuple(
            dict.fromkeys(
                gap_id
                for challenge in lane_challenges
                for gap_id in challenge.distinguishing_gap_ids
            )
        )
        candidate_ids = {
            challenge.candidate_id for challenge in lane_challenges
        }
        candidate_id = next(iter(candidate_ids)) if len(candidate_ids) == 1 else None
        statuses = {challenge.status for challenge in lane_challenges}
        if lane_id in blocked or ChallengeStatus.BLOCKED.value in statuses:
            status = AlternativeLaneResolutionStatus.BLOCKED.value
            reason = "The required source for this alternative Lane is blocked."
        elif lane_id in eliminated:
            status = AlternativeLaneResolutionStatus.ELIMINATED.value
            reason = "Python-owned State previously eliminated this alternative Lane."
        elif ChallengeStatus.ALTERNATIVE_IDENTIFIED.value in statuses:
            status = AlternativeLaneResolutionStatus.RETAINED.value
            reason = "Distinguishing Evidence retains this Lane as a live alternative."
        elif ChallengeStatus.NON_DISCRIMINATIVE.value in statuses:
            status = AlternativeLaneResolutionStatus.NON_DISCRIMINATIVE.value
            reason = "The latest observation did not distinguish this Lane."
        elif statuses & {
            ChallengeStatus.OPEN.value,
            ChallengeStatus.UNRESOLVED.value,
        }:
            status = AlternativeLaneResolutionStatus.UNRESOLVED.value
            reason = "This alternative Lane still has an unresolved discriminator."
        elif statuses and statuses <= {ChallengeStatus.RESOLVED.value}:
            status = AlternativeLaneResolutionStatus.ELIMINATED.value
            reason = "Distinguishing Evidence eliminated this alternative Lane."
        else:
            status = AlternativeLaneResolutionStatus.UNRESOLVED.value
            reason = "This Lane has not received a conclusive adversarial result."
        resolutions.append(
            AlternativeLaneResolution(
                lane_id=lane_id,
                status=status,
                candidate_id=candidate_id,
                evidence_ids=evidence_ids,
                distinguishing_gap_ids=gap_ids,
                reason=reason,
            )
        )

    live = [
        item
        for item in resolutions
        if item.status
        not in {
            AlternativeLaneResolutionStatus.ELIMINATED.value,
            AlternativeLaneResolutionStatus.BLOCKED.value,
        }
    ]
    if (
        len(live) == 1
        and live[0].status == AlternativeLaneResolutionStatus.UNRESOLVED.value
        and any(
            item.status == AlternativeLaneResolutionStatus.ELIMINATED.value
            for item in resolutions
        )
    ):
        survivor = live[0]
        resolutions = [
            (
                AlternativeLaneResolution(
                    lane_id=item.lane_id,
                    status=AlternativeLaneResolutionStatus.RETAINED.value,
                    candidate_id=item.candidate_id,
                    evidence_ids=item.evidence_ids,
                    distinguishing_gap_ids=item.distinguishing_gap_ids,
                    reason=(
                        "This is the sole remaining Lane after all investigated "
                        "alternatives were eliminated."
                    ),
                )
                if item.lane_id == survivor.lane_id
                else item
            )
            for item in resolutions
        ]
    return tuple(resolutions)


def derive_alternative_search_status(
    *,
    challenges: Sequence[CandidateChallenge],
    matrices: Sequence[CausalEvidenceMatrix],
    active_lane_ids: Sequence[str] = (),
    eliminated_lane_ids: Sequence[str] = (),
    blocked_lane_ids: Sequence[str] = (),
    lane_resolutions: Sequence[AlternativeLaneResolution] = (),
) -> str:
    """Derive competition status without trusting a Qwen conclusion field.

    A single candidate never implies that alternatives do not exist.  The only
    status that permits a strict Confirmation Gate to treat alternatives as
    eliminated is ``alternatives_eliminated``.  A named alternative or an
    unresolved/blocked challenge remains a live competition.
    """

    resolutions = tuple(lane_resolutions) or derive_alternative_lane_resolutions(
        challenges=challenges,
        active_lane_ids=active_lane_ids,
        eliminated_lane_ids=eliminated_lane_ids,
        blocked_lane_ids=blocked_lane_ids,
    )
    if not challenges:
        if any(
            item.status
            in {
                AlternativeLaneResolutionStatus.UNRESOLVED.value,
                AlternativeLaneResolutionStatus.NON_DISCRIMINATIVE.value,
                AlternativeLaneResolutionStatus.RETAINED.value,
            }
            for item in resolutions
        ):
            return AlternativeSearchStatus.UNRESOLVED.value
        return AlternativeSearchStatus.NOT_SEARCHED.value
    if any(
        item.status == AlternativeLaneResolutionStatus.BLOCKED.value
        for item in resolutions
    ) or any(challenge.status == ChallengeStatus.BLOCKED.value for challenge in challenges):
        # A blocked source changes Lane investigation state but does not by
        # itself terminate the whole competition. Only the Python-owned
        # Competition Progression Engine may emit blocked_by_missing_data after
        # proving that no high-value decision-changing Action remains.
        return AlternativeSearchStatus.UNRESOLVED.value
    if any(challenge.unexplained_precursor_evidence_ids for challenge in challenges):
        return AlternativeSearchStatus.UNRESOLVED.value

    unresolved_resolution_statuses = {
        AlternativeLaneResolutionStatus.UNRESOLVED.value,
        AlternativeLaneResolutionStatus.NON_DISCRIMINATIVE.value,
    }
    if any(item.status in unresolved_resolution_statuses for item in resolutions):
        if any(challenge.strongest_alternative_lane_id for challenge in challenges):
            return AlternativeSearchStatus.ALTERNATIVE_FOUND.value
        return AlternativeSearchStatus.UNRESOLVED.value

    retained = [
        item
        for item in resolutions
        if item.status == AlternativeLaneResolutionStatus.RETAINED.value
    ]
    if len(retained) > 1:
        return AlternativeSearchStatus.ALTERNATIVE_FOUND.value
    if len(retained) == 1 and all(
        item.status
        in {
            AlternativeLaneResolutionStatus.RETAINED.value,
            AlternativeLaneResolutionStatus.ELIMINATED.value,
        }
        for item in resolutions
    ):
        return AlternativeSearchStatus.ALTERNATIVES_ELIMINATED.value
    if resolutions and all(
        item.status == AlternativeLaneResolutionStatus.ELIMINATED.value
        for item in resolutions
    ):
        # Eliminating every Lane is not confirmation of a winner.  The search
        # remains unresolved until one evidence-grounded Lane is retained.
        return AlternativeSearchStatus.UNRESOLVED.value

    # With multiple active Lanes, an empty alternative claim is not enough:
    # every other Lane must have been explicitly eliminated by Python-owned
    # state.  For multiple candidates, a fully conflicted non-winner is also a
    # valid deterministic elimination signal.
    active = set(active_lane_ids)
    eliminated = set(eliminated_lane_ids)
    if active and not active <= eliminated:
        if len(matrices) <= 1:
            return AlternativeSearchStatus.UNRESOLVED.value
        if not all(matrix.has_critical_conflict for matrix in matrices[1:]):
            return AlternativeSearchStatus.UNRESOLVED.value
    return AlternativeSearchStatus.ALTERNATIVES_ELIMINATED.value


@dataclass(frozen=True)
class QwenAdversarialChallenger:
    """Ask Qwen for bounded challenge claims; Python owns the result status."""

    llm_client: LLMClient
    prompt_version: str = "v1"

    def __post_init__(self) -> None:
        if self.llm_client is None:
            raise ModelValidationError("adversarial challenger requires an LLM client")

    def generate(
        self,
        *,
        request_id: str,
        candidates: Sequence[Mapping[str, Any]],
        matrices: Sequence[CausalEvidenceMatrix],
        evidence_gaps: Sequence[Mapping[str, Any]],
        evidence_ids: Sequence[str],
        evidence_by_id: Mapping[str, Evidence] | None = None,
        lane_ids: Sequence[str] = (),
        active_lane_ids: Sequence[str] = (),
        eliminated_lane_ids: Sequence[str] = (),
        blocked_lane_ids: Sequence[str] = (),
        lane_contexts: Sequence[Mapping[str, Any]] = (),
        candidate_competition: Mapping[str, Any] | None = None,
    ) -> AdversarialChallengeGeneration:
        if not candidates or len(candidates) != len(matrices):
            return AdversarialChallengeGeneration(challenges=(), attempt_count=0)
        ids = _candidate_ids(candidates)
        challenge_output_contract = _challenge_output_contract(
            candidate_ids=ids,
            active_lane_ids=active_lane_ids,
            evidence_gaps=evidence_gaps,
            candidate_competition=candidate_competition,
        )
        gap_by_id = {
            str(item.get("gap_id")): item
            for item in evidence_gaps
            if str(item.get("gap_id", "")).strip()
        }
        contract_gap_ids = {
            str(gap_id)
            for gap_ids in challenge_output_contract[
                "allowed_gap_ids_by_candidate"
            ].values()
            for gap_id in gap_ids
        }
        contract_gap_by_id = {
            gap_id: gap
            for gap_id, gap in gap_by_id.items()
            if gap_id in contract_gap_ids
        }
        gap_id_set = set(contract_gap_by_id)
        lane_context_by_id = {
            str(item.get("lane_id", "")).strip(): item
            for item in lane_contexts
            if str(item.get("lane_id", "")).strip()
        }
        if not llm_call_budget_available(
            self.llm_client,
            required_calls=1,
            # A completed RCA Action must leave one call for the Planner to
            # decide whether to investigate further or stop.
            reserve_calls=1,
        ):
            lane_resolutions = derive_alternative_lane_resolutions(
                challenges=(),
                active_lane_ids=active_lane_ids,
                eliminated_lane_ids=eliminated_lane_ids,
                blocked_lane_ids=blocked_lane_ids,
            )
            return AdversarialChallengeGeneration(
                challenges=(),
                attempt_count=0,
                validation_errors=(
                    "adversarial challenge deferred to preserve the governed "
                    "post-Action Planner budget",
                ),
                repair_skipped_due_to_budget=True,
                alternative_search_status=AlternativeSearchStatus.UNRESOLVED.value,
                lane_resolutions=lane_resolutions,
            )
        payload_candidates = []
        for index, (candidate, matrix) in enumerate(zip(candidates, matrices, strict=True)):
            payload_candidates.append(
                {
                    "candidate_id": ids[index],
                    "root_cause": str(candidate.get("root_cause", "")),
                    "causal_explanation": str(candidate.get("causal_explanation", "")),
                    "semantic_profile": (
                        dict(candidate["semantic_profile"])
                        if isinstance(candidate.get("semantic_profile"), Mapping)
                        else None
                    ),
                    "causal_evidence_matrix": (
                        compact_causal_evidence_matrix_for_prompt(matrix)
                    ),
                }
            )
        compact_lane_contexts = [
            _compact_lane_context_for_prompt(item) for item in lane_contexts
        ]
        prompt_evidence_cards: list[dict[str, Any]] = []
        prompt_omitted_entity_count = 0
        prompt_omitted_metadata_count = 0
        for evidence_id in dict.fromkeys(str(item) for item in evidence_ids):
            evidence = (evidence_by_id or {}).get(evidence_id)
            if evidence is None:
                continue
            card = compact_evidence_prompt_card(evidence)
            projection_audit = card.pop("projection_audit", {})
            prompt_omitted_entity_count += int(
                projection_audit.get("omitted_entity_count", 0)
            )
            prompt_omitted_metadata_count += int(
                projection_audit.get("omitted_metadata_count", 0)
            )
            prompt_evidence_cards.append(card)
        validation_errors: list[str] = []
        prompt_audit: dict[str, Any] | None = None
        for attempt in range(1, _OUTPUT_ATTEMPTS + 1):
            request_payload = {
                "request_id": request_id,
                "candidates": payload_candidates,
                "evidence_gaps": [dict(item) for item in evidence_gaps],
                "available_evidence_ids": sorted(set(evidence_ids)),
                "typed_evidence_register": prompt_evidence_cards,
                "causal_lane_ids": list(dict.fromkeys(lane_ids)),
                "active_lane_ids": list(dict.fromkeys(active_lane_ids)),
                "eliminated_lane_ids": list(dict.fromkeys(eliminated_lane_ids)),
                "blocked_lane_ids": list(dict.fromkeys(blocked_lane_ids)),
                "causal_lanes": compact_lane_contexts,
                "candidate_competition": (
                    dict(candidate_competition)
                    if candidate_competition is not None
                    else None
                ),
                "challenge_output_contract": challenge_output_contract,
                "output_attempt": attempt,
                "previous_validation_feedback": (
                    {
                        "category": "challenge_output_validation_error",
                        "message": validation_errors[-1],
                        "must_repair_before_resubmission": True,
                        "unresolved_named_alternative_gap_rule": (
                            "Select exactly one highest-information-gain "
                            "hypothesis_discrimination Gap for each unresolved "
                            "named alternative Lane."
                        ),
                        "allowed_gap_ids": sorted(gap_id_set),
                        "allowed_gap_ids_by_candidate": (
                            challenge_output_contract[
                                "allowed_gap_ids_by_candidate"
                            ]
                        ),
                        "allowed_gap_ids_by_candidate_and_lane": (
                            challenge_output_contract[
                                "allowed_gap_ids_by_candidate_and_lane"
                            ]
                        ),
                        "highest_information_gain_gap_ids_by_candidate_and_lane": (
                            challenge_output_contract[
                                "highest_information_gain_gap_ids_by_candidate_and_lane"
                            ]
                        ),
                        "challenge_output_contract": challenge_output_contract,
                    }
                    if validation_errors
                    else None
                ),
                "deterministic_challenges": [],
            }
            payload_char_count = _payload_char_count(request_payload)
            prompt_audit = {
                "prompt_payload_char_count": payload_char_count,
                "prompt_evidence_count": len(set(evidence_ids)),
                "omitted_entity_count": prompt_omitted_entity_count,
                "omitted_metadata_count": prompt_omitted_metadata_count,
                "omitted_matrix_fact_group_count": sum(
                    int(
                        compact_causal_evidence_matrix_for_prompt(matrix)[
                            "projection_audit"
                        ]["omitted_claim_fact_groups"]
                    )
                    for matrix in matrices
                ),
                "prompt_budget_applied": True,
                "prompt_payload_char_limit": _MAX_CHALLENGE_PAYLOAD_CHARS,
            }
            if payload_char_count > _MAX_CHALLENGE_PAYLOAD_CHARS:
                validation_errors.append(
                    "adversarial challenge prompt exceeds the governed payload limit"
                )
                break
            request = LLMRequest(
                agent=AgentKind.RCA_REASONING.value,
                prompt_name="causal_adversarial_challenge",
                prompt_version=self.prompt_version,
                payload=request_payload,
                temperature=0.0,
            )
            try:
                response = self.llm_client.complete_json(request)
                data = response.data
                if set(data) != {"challenges", "analysis_summary"}:
                    raise LLMOutputValidationError(
                        "adversarial challenge output must contain exactly challenges "
                        "and analysis_summary"
                    )
                raw_challenges = data.get("challenges")
                summary = data.get("analysis_summary")
                if not isinstance(raw_challenges, list):
                    raise LLMOutputValidationError("challenges must be an array")
                if not isinstance(summary, str) or not summary.strip():
                    raise LLMOutputValidationError("analysis_summary must be non-empty")
                parsed: list[CandidateChallenge] = []
                candidate_errors: list[str] = []
                for index, raw in enumerate(raw_challenges):
                    try:
                        parsed.append(
                            _normalize_challenge_payload(
                                raw,
                                candidate_ids=set(ids),
                                lane_ids=set(lane_ids),
                                evidence_ids=set(evidence_ids),
                                gap_ids=gap_id_set,
                                gap_by_id=contract_gap_by_id,
                                evidence_by_id=evidence_by_id,
                                lane_contexts=lane_context_by_id,
                            )
                        )
                    except (LLMOutputValidationError, TypeError, ValueError) as exc:
                        candidate_errors.append(
                            str(exc).strip() or f"challenges[{index}] is invalid"
                        )
                if candidate_errors and (
                    not parsed or attempt < _OUTPUT_ATTEMPTS
                ):
                    raise LLMOutputValidationError("; ".join(candidate_errors))
                required_challenge_kind = challenge_output_contract.get(
                    "required_challenge_kind"
                )
                if required_challenge_kind and any(
                    challenge.challenge_kind != required_challenge_kind
                    for challenge in parsed
                ):
                    raise LLMOutputValidationError(
                        "challenge_kind must match Python-owned "
                        f"required_challenge_kind={required_challenge_kind}"
                    )
                required_mechanism_relation = challenge_output_contract.get(
                    "required_mechanism_relation"
                )
                if required_mechanism_relation and any(
                    challenge.mechanism_relation
                    != required_mechanism_relation
                    for challenge in parsed
                ):
                    raise LLMOutputValidationError(
                        "mechanism_relation must match Python-owned "
                        "required_mechanism_relation="
                        f"{required_mechanism_relation}"
                    )
                validation_errors.extend(candidate_errors)
                lane_resolutions = derive_alternative_lane_resolutions(
                    challenges=parsed,
                    active_lane_ids=active_lane_ids,
                    eliminated_lane_ids=eliminated_lane_ids,
                    blocked_lane_ids=blocked_lane_ids,
                )
                status = derive_alternative_search_status(
                    challenges=parsed,
                    matrices=matrices,
                    active_lane_ids=active_lane_ids,
                    eliminated_lane_ids=eliminated_lane_ids,
                    blocked_lane_ids=blocked_lane_ids,
                    lane_resolutions=lane_resolutions,
                )
                return AdversarialChallengeGeneration(
                    challenges=tuple(parsed),
                    attempt_count=attempt,
                    validation_errors=tuple(validation_errors),
                    alternative_search_status=status,
                    lane_resolutions=lane_resolutions,
                    prompt_audit=prompt_audit,
                )
            except LLMCallError as exc:
                lane_resolutions = derive_alternative_lane_resolutions(
                    challenges=(),
                    active_lane_ids=active_lane_ids,
                    eliminated_lane_ids=eliminated_lane_ids,
                    blocked_lane_ids=blocked_lane_ids,
                )
                return AdversarialChallengeGeneration(
                    challenges=(),
                    attempt_count=attempt,
                    validation_errors=(str(exc).strip() or type(exc).__name__,),
                    alternative_search_status=AlternativeSearchStatus.UNRESOLVED.value,
                    lane_resolutions=lane_resolutions,
                    prompt_audit=prompt_audit,
                )
            except LLMOutputValidationError as exc:
                validation_errors.append(str(exc).strip() or type(exc).__name__)
                if attempt < _OUTPUT_ATTEMPTS and llm_call_budget_available(
                    self.llm_client,
                    required_calls=1,
                    reserve_calls=1,
                ):
                    continue
                lane_resolutions = derive_alternative_lane_resolutions(
                    challenges=(),
                    active_lane_ids=active_lane_ids,
                    eliminated_lane_ids=eliminated_lane_ids,
                    blocked_lane_ids=blocked_lane_ids,
                )
                return AdversarialChallengeGeneration(
                    challenges=(),
                    attempt_count=attempt,
                    validation_errors=tuple(validation_errors),
                    output_invalid=(attempt >= _OUTPUT_ATTEMPTS),
                    repair_skipped_due_to_budget=(attempt < _OUTPUT_ATTEMPTS),
                    alternative_search_status=(
                        AlternativeSearchStatus.UNRESOLVED.value
                    ),
                    lane_resolutions=lane_resolutions,
                    prompt_audit=prompt_audit,
                )
        lane_resolutions = derive_alternative_lane_resolutions(
            challenges=(),
            active_lane_ids=active_lane_ids,
            eliminated_lane_ids=eliminated_lane_ids,
            blocked_lane_ids=blocked_lane_ids,
        )
        return AdversarialChallengeGeneration(
            challenges=(),
            attempt_count=_OUTPUT_ATTEMPTS,
            validation_errors=tuple(validation_errors),
            output_invalid=True,
            alternative_search_status=AlternativeSearchStatus.UNRESOLVED.value,
            lane_resolutions=lane_resolutions,
            prompt_audit=prompt_audit,
        )


__all__ = [
    "AdversarialChallengeGeneration",
    "QwenAdversarialChallenger",
    "derive_alternative_lane_resolutions",
    "derive_alternative_search_status",
]
