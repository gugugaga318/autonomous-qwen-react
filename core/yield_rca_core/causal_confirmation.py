"""Deterministic Confirmation and Impact-Lot gates for RCA results."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

from yield_rca_core.authoritative_result import (
    ImpactPublicationResult,
    ImpactPublicationStatus,
)
from yield_rca_core.causal_chain import assess_causal_chain
from yield_rca_core.causal_evidence_matrix import CausalEvidenceMatrix
from yield_rca_core.causal_hypothesis import CausalClaim, CausalClaimStatus, CausalHypothesis
from yield_rca_core.causal_investigation_models import (
    AlternativeSearchStatus,
    CandidateClaimedScopeKind,
    CandidateScopeRelation,
    CandidateSemanticProfile,
)
from yield_rca_core.evidence_models import (
    EntityType,
    Evidence,
    EvidenceSourceType,
    EvidenceType,
)

CONCLUSION_SUPPORTED = "supported"
CONCLUSION_INCONCLUSIVE = "inconclusive"
CONCLUSION_INSUFFICIENT_EVIDENCE = "insufficient_evidence"

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
_EXPOSURE_TYPES = {
    EvidenceType.LOT_CONTEXT.value,
    EvidenceType.PROCESS_EXPOSURE.value,
    EvidenceType.EQUIPMENT_EXPOSURE.value,
    EvidenceType.IMPACT_SCOPE.value,
    EvidenceType.EXCURSION_WINDOW.value,
}
_IMPACT_SCOPE_SOURCE_TYPES = {
    EvidenceSourceType.MES.value,
    EvidenceSourceType.FDC.value,
    EvidenceSourceType.WAT.value,
    EvidenceSourceType.DEFECT.value,
    EvidenceSourceType.ANALYTICS.value,
}


@dataclass(frozen=True)
class ConfirmationGateResult:
    """Python-owned final status and audit checks for one candidate."""

    status: str
    checks: Mapping[str, bool]
    reasons: tuple[str, ...] = ()
    unresolved_gaps: tuple[str, ...] = ()
    data_missing_evidence_ids: tuple[str, ...] = ()
    blocking_data_missing_evidence_ids: tuple[str, ...] = ()
    non_blocking_data_missing_evidence_ids: tuple[str, ...] = ()
    causal_chain_completeness: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "checks": dict(self.checks),
            "reasons": list(self.reasons),
            "unresolved_gaps": list(self.unresolved_gaps),
            "data_missing_evidence_ids": list(self.data_missing_evidence_ids),
            "blocking_data_missing_evidence_ids": list(
                self.blocking_data_missing_evidence_ids
            ),
            "non_blocking_data_missing_evidence_ids": list(
                self.non_blocking_data_missing_evidence_ids
            ),
            "causal_chain_completeness": self.causal_chain_completeness,
        }


def _status(matrix: CausalEvidenceMatrix, claim: str) -> str:
    result = matrix.claims.get(claim)
    return (
        str(cast(Any, result).status)
        if result is not None
        else CausalClaimStatus.UNAVAILABLE.value
    )


def _confirmation_missing_source_is_blocking(
    matrix: CausalEvidenceMatrix,
    source: Mapping[str, Any],
) -> bool:
    """Classify one unavailable source against still-unavailable causal facts.

    An explicit ``required_for_confirmation`` declaration always wins.  Other
    operational failures block only when the corresponding causal stage has no
    typed substitute Evidence at all.  An incomplete competing explanation is
    therefore not converted into ``insufficient_evidence`` merely because an
    optional SPC baseline or other secondary analysis product is unavailable.
    """

    if source.get("required_for_confirmation") is True:
        return True

    source_type = str(source.get("source_type", "")).casefold()
    source_field = str(source.get("source_field", "")).strip()
    relevant_claims: set[str] = set()
    if source_type in {
        EvidenceSourceType.FDC.value,
        EvidenceSourceType.ANALYTICS.value,
    } or source_field:
        relevant_claims.update(
            {
                CausalClaim.PARAMETER.value,
                CausalClaim.TEMPORAL.value,
            }
        )
    if source_type in {
        EvidenceSourceType.DEFECT.value,
        EvidenceSourceType.WAT.value,
    }:
        relevant_claims.add(CausalClaim.OUTCOME.value)
    if source_type == EvidenceSourceType.MES.value:
        relevant_claims.update(
            {
                CausalClaim.EQUIPMENT.value,
                CausalClaim.CHAMBER.value,
                CausalClaim.OPERATION.value,
                CausalClaim.SCOPE.value,
            }
        )
    return any(
        _status(matrix, claim) == CausalClaimStatus.UNAVAILABLE.value
        for claim in relevant_claims
    )


def confirm_candidate(
    matrix: CausalEvidenceMatrix,
    *,
    alternative_matrices: Sequence[CausalEvidenceMatrix] = (),
    strict: bool = True,
    alternative_search_status: str | None = None,
    require_causal_chain: bool = True,
    unexplained_precursor_evidence_ids: Sequence[str] = (),
) -> ConfirmationGateResult:
    """Apply the final Python confirmation gate.

    ``strict=False`` is retained for legacy deterministic snapshots that were
    created before timestamps and chamber/operation entities were mandatory.
    It still rejects explicit conflicts and missing causal lanes.  The active
    Qwen path uses the strict gate.
    """

    checks: dict[str, bool] = {}
    reasons: list[str] = []
    gaps: list[str] = []
    chain = matrix.causal_chain or assess_causal_chain(
        matrix.claims,
        data_missing_evidence_ids=matrix.data_missing_evidence_ids,
    )
    checks["causal_chain"] = chain.status == "complete"
    if require_causal_chain and not checks["causal_chain"]:
        reasons.append(f"causal chain is {chain.status}.")
        gaps.append(f"causal_chain.{chain.status}")
    data_missing_ids = tuple(matrix.data_missing_evidence_ids)
    blocking_data_missing_ids = tuple(
        dict.fromkeys(
            str(source.get("evidence_id"))
            for source in matrix.data_missing_sources
            if str(source.get("evidence_id", "")).strip()
            and _confirmation_missing_source_is_blocking(matrix, source)
        )
    )
    non_blocking_data_missing_ids = tuple(
        evidence_id
        for evidence_id in data_missing_ids
        if evidence_id not in blocking_data_missing_ids
    )
    checks["data_available"] = not blocking_data_missing_ids
    if blocking_data_missing_ids:
        reasons.append(
            "One or more confirmation-blocking operational sources are unavailable."
        )
        gaps.extend(
            f"data_missing.{evidence_id}"
            for evidence_id in blocking_data_missing_ids
        )

    precursor_ids = tuple(
        dict.fromkeys(
            str(evidence_id).strip()
            for evidence_id in unexplained_precursor_evidence_ids
            if str(evidence_id).strip()
        )
    )
    checks["precursor_explained"] = not precursor_ids
    if precursor_ids:
        reasons.append(
            "The candidate does not explain one or more earlier precursor Evidence items."
        )
        gaps.extend(f"precursor.unexplained.{evidence_id}" for evidence_id in precursor_ids)

    exposure_types = {
        evidence_type
        for claim in (
            CausalClaim.EQUIPMENT.value,
            CausalClaim.CHAMBER.value,
            CausalClaim.OPERATION.value,
        )
        if claim in matrix.claims
        for evidence_type in matrix.claims[claim].facts.get("evidence_types", [])
        if isinstance(evidence_type, str)
    }
    checks["exposure"] = bool(exposure_types & _EXPOSURE_TYPES)
    if not checks["exposure"]:
        reasons.append("no cited typed Evidence establishes current-Lot exposure.")
        gaps.append("exposure.unavailable")

    for claim in (
        CausalClaim.EQUIPMENT.value,
        CausalClaim.CHAMBER.value,
        CausalClaim.OPERATION.value,
    ):
        claim_status = _status(matrix, claim)
        # Equipment/chamber/operation are hard checks only when the candidate
        # explicitly names a structured entity.  A generic mechanism can remain
        # analyzable with legacy evidence that lacks one of these entities.
        explicit = any(
            token.casefold() in matrix.candidate.root_cause.casefold()
            for token in matrix.claims[claim].facts.get("entity_ids_by_type", {}).get(
                claim, []
            )
        ) if claim in matrix.claims else False
        checks[claim] = claim_status != CausalClaimStatus.CONFLICTED.value
        if explicit and claim_status != CausalClaimStatus.SUPPORTED.value:
            checks[claim] = False
        if not checks[claim]:
            reasons.append(f"{claim} claim is conflicted or not grounded.")
            gaps.append(f"{claim}.{claim_status}")

    required_claims = (
        CausalClaim.PARAMETER.value,
        CausalClaim.OUTCOME.value,
        CausalClaim.MECHANISM.value,
        CausalClaim.SCOPE.value,
    )
    for claim in required_claims:
        claim_status = _status(matrix, claim)
        checks[claim] = claim_status == CausalClaimStatus.SUPPORTED.value
        if not checks[claim]:
            reasons.append(f"{claim} claim is {claim_status}.")
            gaps.append(f"{claim}.{claim_status}")

    temporal_status = _status(matrix, CausalClaim.TEMPORAL.value)
    checks["temporal"] = temporal_status != CausalClaimStatus.CONFLICTED.value
    if strict and temporal_status != CausalClaimStatus.SUPPORTED.value:
        checks["temporal"] = False
    if not checks["temporal"]:
        reasons.append(f"temporal claim is {temporal_status}.")
        gaps.append(f"temporal.{temporal_status}")

    contradiction_status = _status(matrix, CausalClaim.CONTRADICTION.value)
    checks["contradiction_free"] = contradiction_status != CausalClaimStatus.CONFLICTED.value
    if not checks["contradiction_free"]:
        reasons.append("a cited or invalid Evidence contradiction remains unresolved.")
        gaps.append("contradiction.conflicted")

    # Control/negative Evidence is informative and intentionally not a hard
    # requirement.  Keep it visible for the report and audit payload.
    checks["control_informational"] = _status(matrix, CausalClaim.CONTROL.value) != "conflicted"

    # Alternative comparison is diagnostic.  It is not a hard gate because the
    # user explicitly chose not to require exclusionary proof for confirmation.
    checks["no_equal_alternative"] = True
    if strict and alternative_search_status is not None:
        checks["no_equal_alternative"] = (
            alternative_search_status
            == AlternativeSearchStatus.ALTERNATIVES_ELIMINATED.value
        )
        if not checks["no_equal_alternative"]:
            reasons.append(
                "Adversarial alternative search is not complete; a single candidate "
                "cannot be treated as the only explanation."
            )
            gaps.append(f"alternative_search.{alternative_search_status}")
    if alternative_matrices:
        current_score = sum(
            result.status == CausalClaimStatus.SUPPORTED.value
            for result in matrix.claims.values()
        )
        if any(
            sum(
                result.status == CausalClaimStatus.SUPPORTED.value
                for result in other.claims.values()
            )
            == current_score
            for other in alternative_matrices
            if other is not matrix
        ):
            checks["no_equal_alternative"] = False

    hard_checks = [
        checks["data_available"],
        checks["exposure"],
        checks["parameter"],
        checks["outcome"],
        checks["mechanism"],
        checks["scope"],
        checks["temporal"],
        checks["contradiction_free"],
        checks["precursor_explained"],
        *([checks["causal_chain"]] if require_causal_chain else []),
        *(
            [checks["no_equal_alternative"]]
            if strict and alternative_search_status is not None
            else []
        ),
        *[
            checks[claim]
            for claim in ("equipment", "chamber", "operation")
            if checks[claim] is False
        ],
    ]
    unresolved_candidate_competition = (
        alternative_search_status == AlternativeSearchStatus.UNRESOLVED.value
        and bool(alternative_matrices)
    )
    if all(hard_checks):
        status = CONCLUSION_SUPPORTED
    elif unresolved_candidate_competition:
        # When two evidence-grounded candidates remain plausible, unavailable
        # discriminator data explains why they cannot be separated; it does not
        # mean that the investigation lacks a causal hypothesis altogether.
        # Preserve the blocking-source audit fields while reporting the more
        # precise public conclusion: unresolved competition.
        status = CONCLUSION_INCONCLUSIVE
    elif blocking_data_missing_ids:
        status = CONCLUSION_INSUFFICIENT_EVIDENCE
    elif any(
        _status(matrix, claim) == CausalClaimStatus.UNAVAILABLE.value
        for claim in required_claims
    ):
        status = CONCLUSION_INSUFFICIENT_EVIDENCE
    else:
        status = CONCLUSION_INCONCLUSIVE
    return ConfirmationGateResult(
        status=status,
        checks=checks,
        reasons=tuple(dict.fromkeys(reasons)),
        unresolved_gaps=tuple(dict.fromkeys(gaps)),
        data_missing_evidence_ids=data_missing_ids,
        blocking_data_missing_evidence_ids=blocking_data_missing_ids,
        non_blocking_data_missing_evidence_ids=non_blocking_data_missing_ids,
        causal_chain_completeness=(
            "incomplete" if precursor_ids and chain.status == "complete" else chain.status
        ),
    )


def _lot_ids(item: Evidence) -> set[str]:
    return {
        entity.entity_id
        for entity in item.entities
        if entity.entity_type == EntityType.LOT.value
    }


def _entity_ids(item: Evidence, entity_type: str) -> set[str]:
    return {
        entity.entity_id
        for entity in item.entities
        if entity.entity_type == entity_type
    }


def _candidate_entity_tokens(candidate: CausalHypothesis, entity_type: str) -> set[str]:
    # Structured IDs are intentionally conservative; arbitrary prose is not
    # converted into an equipment or chamber claim here.
    prefixes = {
        EntityType.EQUIPMENT.value: ("EQ_", "CMP_"),
        EntityType.CHAMBER.value: ("CH_",),
        EntityType.OPERATION.value: ("OP_",),
        EntityType.RECIPE.value: ("RCP_",),
    }
    text = f"{candidate.root_cause} {candidate.causal_explanation}".upper()
    tokens = {
        token.rstrip(".,;:)")
        for token in text.replace("/", " ").split()
        if token.startswith(prefixes.get(entity_type, ()))
    }
    if entity_type == EntityType.CHAMBER.value:
        tokens.update(
            match.group(1)
            for match in re.finditer(
                r"\bCHAMBER\s*[:#_-]?\s*([A-Z0-9_]+)", text
            )
        )
        tokens.update(
            token
            for token in re.findall(r"\b[A-Z][A-Z0-9_]+\b", text)
            if re.search(r"_CH[A-Z0-9]+(?:_|$)", token)
        )
    if entity_type == EntityType.OPERATION.value:
        tokens.update(
            match.group(1)
            for match in re.finditer(
                r"\b(?:OPERATION(?:_NO)?|OP)\s*[:#_-]?\s*([A-Z0-9_]+)",
                text,
            )
        )
    return tokens


def _compatible_with_candidate(
    item: Evidence,
    candidate: CausalHypothesis,
    semantic_profile: CandidateSemanticProfile | None = None,
) -> bool:
    semantic_expected = {
        EntityType.EQUIPMENT.value: (
            {semantic_profile.claimed_equipment}
            if semantic_profile is not None
            and semantic_profile.claimed_equipment is not None
            else set()
        ),
        EntityType.CHAMBER.value: (
            {semantic_profile.claimed_chamber}
            if semantic_profile is not None
            and semantic_profile.claimed_chamber is not None
            else set()
        ),
        EntityType.OPERATION.value: (
            {semantic_profile.claimed_operation}
            if semantic_profile is not None
            and semantic_profile.claimed_operation is not None
            else set()
        ),
        EntityType.RECIPE.value: (
            {semantic_profile.claimed_recipe}
            if semantic_profile is not None
            and semantic_profile.claimed_recipe is not None
            else set()
        ),
    }
    for entity_type in (
        EntityType.EQUIPMENT.value,
        EntityType.CHAMBER.value,
        EntityType.OPERATION.value,
        EntityType.RECIPE.value,
    ):
        expected = semantic_expected[entity_type] or _candidate_entity_tokens(
            candidate, entity_type
        )
        # A validated semantic profile is the authoritative claimed reach. An
        # intentionally null Recipe in a chamber/equipment scope must not be
        # repopulated from prose and accidentally narrow Impact Lots.
        if semantic_profile is not None:
            expected = semantic_expected[entity_type]
        actual = _entity_ids(item, entity_type)
        matches = {
            (expected_id, actual_id)
            for expected_id in expected
            for actual_id in actual
            if _compact(expected_id) == _compact(actual_id)
            or (
                entity_type == EntityType.EQUIPMENT.value
                and _compact(expected_id).startswith(_compact(actual_id))
                and _compact(expected_id)[len(_compact(actual_id)) :].startswith("ch")
            )
            or (
                entity_type == EntityType.CHAMBER.value
                and _compact(expected_id).endswith(_compact(actual_id))
            )
        }
        if expected and actual and not matches:
            return False
    return True


def _compact(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _metadata_values(item: Evidence, keys: set[str]) -> set[str]:
    values: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if str(key).casefold() in keys and isinstance(
                    child, str | int | float
                ):
                    values.add(str(child))
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(item.metadata)
    return values


def _typed_values(item: Evidence, entity_type: str) -> set[str]:
    values = _entity_ids(item, entity_type)
    metadata_keys = {
        EntityType.EQUIPMENT.value: {"equipment", "equipment_id"},
        EntityType.CHAMBER.value: {"chamber", "chamber_id"},
        EntityType.OPERATION.value: {
            "operation",
            "operation_id",
            "operation_no",
            "target_operation_no",
        },
        EntityType.RECIPE.value: {"recipe", "recipe_id"},
        EntityType.PARAMETER.value: {
            "parameter",
            "parameter_id",
            "parameter_name",
            "metric_name",
        },
    }
    values.update(_metadata_values(item, metadata_keys.get(entity_type, set())))
    if entity_type == EntityType.PARAMETER.value and item.source_field:
        values.add(item.source_field)
    return values


def _common_lane_values(
    exposure: Sequence[Evidence],
    process: Sequence[Evidence],
    entity_type: str,
) -> set[str]:
    exposure_values = {
        value for item in exposure for value in _typed_values(item, entity_type)
    }
    process_values = {
        value for item in process for value in _typed_values(item, entity_type)
    }
    if not exposure_values or not process_values:
        return set()
    return {
        left
        for left in exposure_values
        if any(_compact(left) == _compact(right) for right in process_values)
    }


def _candidate_matches_values(
    candidate: CausalHypothesis,
    values: set[str],
) -> bool:
    raw_text = f"{candidate.root_cause} {candidate.causal_explanation}"
    compact_text = _compact(raw_text)
    candidate_tokens = set(re.findall(r"[a-z0-9]+", raw_text.casefold()))

    def compact_variants(value: str) -> set[str]:
        compact_value = _compact(value)
        variants = {compact_value}
        # Typed parameter names commonly append a measurement/statistic suffix
        # that engineers omit in causal prose.  Strip only those suffixes; the
        # physical parameter identity itself must still match the candidate.
        measurement_suffixes = (
            "average",
            "deviation",
            "repeatability",
            "delta",
            "offset",
            "range",
            "shift",
            "sigma",
            "mean",
            "avg",
            "cv",
        )
        for suffix in measurement_suffixes:
            if compact_value.endswith(suffix):
                base = compact_value[: -len(suffix)]
                if len(base) >= 6:
                    variants.add(base)
        return variants

    return any(
        (
            any(
                len(variant) >= 6 and variant in compact_text
                for variant in compact_variants(value)
            )
        )
        or (
            bool(value_tokens := set(re.findall(r"[a-z0-9]+", value.casefold())))
            and value_tokens <= candidate_tokens
        )
        for value in values
    )


def _candidate_direction(candidate: CausalHypothesis) -> str | None:
    aliases = {
        "above": "high",
        "decrease": "low",
        "decreased": "low",
        "drop": "low",
        "high": "high",
        "higher": "high",
        "increase": "high",
        "increased": "high",
        "low": "low",
        "lower": "low",
        "reduced": "low",
    }
    for token in re.findall(
        r"[a-z]+",
        f"{candidate.root_cause} {candidate.causal_explanation}".casefold(),
    ):
        if token in aliases:
            return aliases[token]
    return None


def _evidence_directions(items: Sequence[Evidence]) -> set[str]:
    directions: set[str] = set()
    for item in items:
        raw = _metadata_values(
            item,
            {
                "direction",
                "parameter_direction",
                "same_side_direction",
                "trend_direction",
            },
        )
        for value in raw:
            lowered = value.casefold()
            if lowered in {"high", "higher", "above", "increase", "increased"}:
                directions.add("high")
            elif lowered in {"low", "lower", "below", "decrease", "decreased"}:
                directions.add("low")
        for key in ("delta", "delta_percent", "avg_delta_percent", "mean_z_score"):
            for value in _metadata_values(item, {key}):
                try:
                    numeric = float(value)
                except ValueError:
                    continue
                if numeric > 0:
                    directions.add("high")
                elif numeric < 0:
                    directions.add("low")
    return directions


def _parse_time(value: object) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC).replace(tzinfo=None)
    return parsed


def _evidence_windows(items: Sequence[Evidence]) -> list[tuple[datetime, datetime]]:
    windows: list[tuple[datetime, datetime]] = []

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            normalized = {str(key).casefold(): child for key, child in value.items()}
            raw_time_window = normalized.get("time_window")
            if (
                isinstance(raw_time_window, (list, tuple))
                and len(raw_time_window) == 2
            ):
                start = _parse_time(raw_time_window[0])
                end = _parse_time(raw_time_window[1])
                if start is not None and end is not None and start <= end:
                    windows.append((start, end))
            starts = [
                normalized[key]
                for key in (
                    "excursion_start",
                    "processing_start",
                    "target_window_start",
                    "window_start",
                    "start",
                )
                if key in normalized
            ]
            ends = [
                normalized[key]
                for key in (
                    "excursion_end",
                    "processing_end",
                    "target_window_end",
                    "window_end",
                    "end",
                )
                if key in normalized
            ]
            if starts and ends:
                start = _parse_time(starts[0])
                end = _parse_time(ends[0])
                if start is not None and end is not None and start <= end:
                    windows.append((start, end))
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    for item in items:
        visit(item.metadata)
        for entity in item.entities:
            visit(entity.attributes)
    return list(dict.fromkeys(windows))


def _windows_overlap(
    observed: Sequence[tuple[datetime, datetime]],
    expected: Sequence[tuple[datetime, datetime]],
) -> bool:
    return bool(observed and expected) and any(
        observed_start <= expected_end and observed_end >= expected_start
        for observed_start, observed_end in observed
        for expected_start, expected_end in expected
    )


def _timestamp_inside_window(
    process: Sequence[Evidence],
    windows: Sequence[tuple[datetime, datetime]],
) -> bool:
    if not windows:
        return False
    timestamps = [
        parsed
        for item in process
        if item.timestamp
        if (parsed := _parse_time(item.timestamp)) is not None
    ]
    return bool(timestamps) and all(
        any(start <= timestamp <= end for start, end in windows)
        for timestamp in timestamps
    )


def _compatible_outcomes(
    outcomes: Sequence[Evidence],
    candidate: CausalHypothesis,
) -> list[Evidence]:
    compatible: list[Evidence] = []
    for item in outcomes:
        values = {
            *(_typed_values(item, EntityType.DEFECT.value)),
            *(_typed_values(item, EntityType.WAT_ITEM.value)),
        }
        if item.evidence_type == EvidenceType.METROLOGY_DEVIATION.value:
            values.update(_typed_values(item, EntityType.PARAMETER.value))
        if values and (
            _candidate_matches_values(candidate, values)
            or _candidate_matches_outcome_phrase(candidate, values)
        ):
            compatible.append(item)
    return compatible


def _candidate_matches_outcome_phrase(
    candidate: CausalHypothesis,
    values: set[str],
) -> bool:
    """Match multi-word outcome metrics without treating prose as typed fact.

    Metrology producers represent product outcomes such as ``center seam-void
    density`` as parameter entities.  Candidate prose may express the same
    outcome as ``center seam voids`` or ``insufficient copper fill``.  Exact
    token containment cannot join those equivalent phrases, so require at
    least two normalized informative tokens instead.  The Evidence still has
    to be a typed outcome and carry the Lot entity before this helper is used.
    """

    candidate_tokens = _normalized_outcome_tokens(
        f"{candidate.root_cause} {candidate.causal_explanation}"
    )
    return any(
        len(value_tokens := _normalized_outcome_tokens(value)) >= 2
        and len(candidate_tokens & value_tokens) >= 2
        for value in values
    )


def _normalized_outcome_tokens(value: object) -> set[str]:
    tokens: set[str] = set()
    ignored = {
        "incident",
        "measurement",
        "metric",
        "monitor",
        "observation",
        "post",
        "pre",
        "result",
    }
    for raw_token in re.findall(r"[a-z0-9]+", str(value).casefold()):
        token = raw_token[:-1] if raw_token.endswith("s") and len(raw_token) > 4 else raw_token
        if len(token) >= 3 and token not in ignored:
            tokens.add(token)
    return tokens


def _impact_data_missing_is_blocking(
    item: Evidence,
    checks: Mapping[str, bool],
) -> bool:
    """Return whether unavailable data prevents this Lot's scope decision.

    A missing analysis product is not automatically a missing Impact-Lot fact.
    For example, an unavailable SPC baseline must remain auditable, but it does
    not block scope when direct Lot exposure, parameter, temporal, and outcome
    Evidence already satisfy the gate.  Producers can explicitly mark a source
    as required for Impact scope when no substitute Evidence is acceptable.
    """

    required = item.metadata.get("required_for_impact_scope")
    if required is True:
        return True
    if required is False:
        return False

    entity_types = {entity.entity_type for entity in item.entities}
    if EntityType.PARAMETER.value in entity_types:
        return not (
            checks.get("excursion", False)
            and checks.get("parameter", False)
            and checks.get("parameter_direction", False)
        )
    lane_checks = {
        EntityType.EQUIPMENT.value: "equipment",
        EntityType.CHAMBER.value: "chamber",
        EntityType.OPERATION.value: "operation",
        EntityType.RECIPE.value: "recipe",
    }
    referenced_lane_checks = [
        check for entity_type, check in lane_checks.items() if entity_type in entity_types
    ]
    if referenced_lane_checks:
        return any(not checks.get(check, False) for check in referenced_lane_checks)
    if entity_types & {EntityType.DEFECT.value, EntityType.WAT_ITEM.value}:
        return not checks.get("outcome", False)

    source_tool = str(item.source_tool or "").casefold()
    if item.source_type in {
        EvidenceSourceType.FDC.value,
        EvidenceSourceType.ANALYTICS.value,
    } and any(token in source_tool for token in ("fdc", "spc", "process")):
        return not (
            checks.get("excursion", False)
            and checks.get("parameter", False)
            and checks.get("temporal", False)
        )
    if item.source_type in {
        EvidenceSourceType.DEFECT.value,
        EvidenceSourceType.WAT.value,
    }:
        return not checks.get("outcome", False)
    if item.source_type == EvidenceSourceType.MES.value:
        return not (
            checks.get("exposure", False)
            and checks.get("equipment", False)
            and checks.get("chamber", False)
            and checks.get("operation", False)
        )

    # An unclassified unavailable source blocks only while at least one scope
    # fact is genuinely absent.  Complete substitute Evidence takes precedence.
    return not all(checks.values())


def evaluate_impact_lot_gate(
    *,
    source_lot_id: str | None,
    candidate: CausalHypothesis | Mapping[str, Any],
    evidence: Iterable[Evidence],
    observed_impact_lots: Sequence[str],
    authoritative_conclusion_status: str = CONCLUSION_SUPPORTED,
    semantic_profile: CandidateSemanticProfile | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate each observed Lot using exposure, excursion, and outcome facts.

    This function is intentionally independent from ``RCAState.impact_lots``;
    callers can audit the inclusion/exclusion reasons before deciding whether a
    product surface should consume the gated set.
    """

    normalized = (
        candidate
        if isinstance(candidate, CausalHypothesis)
        else CausalHypothesis(
            root_cause=str(candidate.get("root_cause", "")).strip(),
            # Deterministic legacy candidates predate the compact causal
            # explanation field.  Their root-cause label is still sufficient
            # for the independent Impact Lot scope gate.
            causal_explanation=str(
                candidate.get("causal_explanation")
                or candidate.get("root_cause", "")
            ).strip(),
            supporting_evidence_ids=tuple(candidate.get("supporting_evidence_ids", [])),
            contradicting_evidence_ids=tuple(
                candidate.get("contradicting_evidence_ids", [])
            ),
        )
    )
    normalized_semantic_profile: CandidateSemanticProfile | None = None
    if semantic_profile is not None:
        try:
            normalized_semantic_profile = (
                semantic_profile
                if isinstance(semantic_profile, CandidateSemanticProfile)
                else CandidateSemanticProfile.from_dict(dict(semantic_profile))
            )
        except (KeyError, TypeError, ValueError):
            normalized_semantic_profile = None
    items = list(
        {
            item.evidence_id: item
            for item in evidence
            if item.is_typed
        }.values()
    )
    all_scope_missing = [
        item
        for item in items
        if item.evidence_type == EvidenceType.DATA_MISSING.value
        and item.source_type in _IMPACT_SCOPE_SOURCE_TYPES
        and _compatible_with_candidate(
            item,
            normalized,
            normalized_semantic_profile,
        )
    ]
    broad_shared_scope = bool(
        normalized_semantic_profile is not None
        and normalized_semantic_profile.scope_relation
        == CandidateScopeRelation.SHARED_EFFECT.value
        and normalized_semantic_profile.claimed_scope_kind
        in {
            CandidateClaimedScopeKind.CHAMBER.value,
            CandidateClaimedScopeKind.EQUIPMENT.value,
            CandidateClaimedScopeKind.OPERATION.value,
        }
    )
    declared_scope_process = [
        item
        for item in items
        if item.evidence_type in _PROCESS_TYPES
        and _compatible_with_candidate(
            item,
            normalized,
            normalized_semantic_profile,
        )
    ]
    declared_excursion_windows = _evidence_windows(
        [
            item
            for item in items
            if item.evidence_type == EvidenceType.EXCURSION_WINDOW.value
            and _compatible_with_candidate(
                item,
                normalized,
                normalized_semantic_profile,
            )
        ]
    )
    rows: list[dict[str, Any]] = []
    for raw_lot in observed_impact_lots:
        lot_id = str(raw_lot)
        if source_lot_id and lot_id == source_lot_id:
            rows.append(
                {
                    "lot_id": lot_id,
                    "included": False,
                    "included_reason": None,
                    "excluded_reason": "source_lot_is_not_an_impact_lot",
                    "supporting_evidence_ids": [],
                    "data_missing_evidence_ids": [],
                    "checks": {"source_lot": False},
                }
            )
            continue
        lot_items = [item for item in items if lot_id in _lot_ids(item)]
        scoped_data_missing = [
            item
            for item in all_scope_missing
            if not _lot_ids(item) or lot_id in _lot_ids(item)
        ]
        exposure = [
            item
            for item in lot_items
            if item.evidence_type in _EXPOSURE_TYPES
            and _compatible_with_candidate(
                item,
                normalized,
                normalized_semantic_profile,
            )
        ]
        lot_process = [
            item
            for item in lot_items
            if item.evidence_type in _PROCESS_TYPES
            and _compatible_with_candidate(
                item,
                normalized,
                normalized_semantic_profile,
            )
        ]
        process = (
            list(
                {
                    item.evidence_id: item
                    for item in [*lot_process, *declared_scope_process]
                }.values()
            )
            if broad_shared_scope
            else lot_process
        )
        outcomes = [item for item in lot_items if item.evidence_type in _OUTCOME_TYPES]
        candidate_parameters = {
            value
            for item in process
            for value in _typed_values(item, EntityType.PARAMETER.value)
            if _candidate_matches_values(normalized, {value})
        }
        compatible_outcomes = _compatible_outcomes(outcomes, normalized)
        common_equipment = _common_lane_values(
            exposure, process, EntityType.EQUIPMENT.value
        )
        common_chamber = _common_lane_values(
            exposure, process, EntityType.CHAMBER.value
        )
        common_operation = _common_lane_values(
            exposure, process, EntityType.OPERATION.value
        )
        exposure_recipes = {
            value for item in exposure for value in _typed_values(item, EntityType.RECIPE.value)
        }
        process_recipes = {
            value for item in process for value in _typed_values(item, EntityType.RECIPE.value)
        }
        recipe_consistent = (
            broad_shared_scope
            and normalized_semantic_profile is not None
            and normalized_semantic_profile.claimed_recipe is None
        ) or not (exposure_recipes or process_recipes) or bool(
            {_compact(value) for value in exposure_recipes}
            & {_compact(value) for value in process_recipes}
        )
        windows = declared_excursion_windows or _evidence_windows(
            [*exposure, *process]
        )
        exposure_windows = _evidence_windows(exposure)
        exposure_time_consistent = _windows_overlap(exposure_windows, windows) or any(
            _timestamp_inside_window((item,), windows)
            for item in exposure
        )
        time_consistent = _timestamp_inside_window(process, windows)
        candidate_direction = _candidate_direction(normalized)
        evidence_directions = _evidence_directions(process)
        direction_consistent = (
            candidate_direction is None
            or not evidence_directions
            or candidate_direction in evidence_directions
        )
        supporting = [*exposure, *process, *compatible_outcomes]
        parameter_scope_projection_used = bool(
            broad_shared_scope and not lot_process and process
        )
        scope_checks = {
            "exposure": bool(exposure),
            "excursion": bool(process),
            "equipment": bool(common_equipment),
            "chamber": bool(common_chamber),
            "operation": bool(common_operation),
            "recipe": recipe_consistent,
            "parameter": bool(candidate_parameters),
            "parameter_direction": direction_consistent,
            "excursion_window": bool(windows),
            "exposure_temporal": exposure_time_consistent,
            "temporal": time_consistent,
            "outcome": bool(compatible_outcomes),
        }
        blocking_data_missing = [
            item
            for item in scoped_data_missing
            if _impact_data_missing_is_blocking(item, scope_checks)
        ]
        non_blocking_data_missing = [
            item
            for item in scoped_data_missing
            if item not in blocking_data_missing
        ]
        checks = {
            "data_available": not blocking_data_missing,
            **scope_checks,
        }
        if all(checks.values()):
            rows.append(
                {
                    "lot_id": lot_id,
                    "included": True,
                    "included_reason": (
                        "Lot has matching candidate exposure, process excursion, "
                        "equipment/chamber/operation, parameter, excursion window, "
                        "and compatible outcome Evidence."
                    ),
                    "excluded_reason": None,
                    "supporting_evidence_ids": list(
                        dict.fromkeys(item.evidence_id for item in supporting)
                    ),
                    "data_missing_evidence_ids": [],
                    "non_blocking_data_missing_evidence_ids": [
                        item.evidence_id for item in non_blocking_data_missing
                    ],
                    "parameter_scope_projection_used": (
                        parameter_scope_projection_used
                    ),
                    "checks": checks,
                }
            )
        else:
            missing = [label for label, passed in checks.items() if not passed]
            if blocking_data_missing:
                missing_reason = (
                    "required source unavailable: "
                    + ", ".join(
                        item.evidence_id for item in blocking_data_missing
                    )
                )
            else:
                missing_reason = "missing " + ", ".join(missing) + " Evidence"
            rows.append(
                {
                    "lot_id": lot_id,
                    "included": False,
                    "included_reason": None,
                    "excluded_reason": missing_reason,
                    "supporting_evidence_ids": list(
                        dict.fromkeys(item.evidence_id for item in supporting)
                    ),
                    "data_missing_evidence_ids": [
                        item.evidence_id for item in blocking_data_missing
                    ],
                    "non_blocking_data_missing_evidence_ids": [
                        item.evidence_id for item in non_blocking_data_missing
                    ],
                    "parameter_scope_projection_used": (
                        parameter_scope_projection_used
                    ),
                    "checks": checks,
                }
            )
    candidate_matches = [row for row in rows if row["included"]]
    if not rows:
        scope_status = "not_evaluated"
    elif len(candidate_matches) == len(rows):
        scope_status = "confirmed"
    elif candidate_matches:
        scope_status = "partial"
    elif any(row.get("data_missing_evidence_ids") for row in rows):
        scope_status = "unavailable"
    else:
        scope_status = "unconfirmed"
    publication_allowed = (
        authoritative_conclusion_status == CONCLUSION_SUPPORTED
    )
    confirmed = candidate_matches if publication_allowed else []
    if confirmed:
        publication_status = "confirmed"
    elif candidate_matches and not publication_allowed:
        publication_status = "withheld"
    elif not rows:
        publication_status = "not_evaluated"
    else:
        publication_status = "unconfirmed"
    for row in rows:
        row["candidate_included"] = bool(row["included"])
        row["confirmed"] = bool(row["included"] and publication_allowed)
    blocking_missing_ids = list(
        dict.fromkeys(
            evidence_id
            for row in rows
            for evidence_id in row.get("data_missing_evidence_ids", [])
        )
    )
    non_blocking_missing_ids = list(
        dict.fromkeys(
            evidence_id
            for row in rows
            for evidence_id in row.get(
                "non_blocking_data_missing_evidence_ids", []
            )
        )
    )
    return {
        "source_lot_id": source_lot_id,
        "candidate_root_cause": normalized.root_cause,
        "candidate_semantic_scope": (
            {
                "scope_kind": normalized_semantic_profile.claimed_scope_kind,
                "scope_relation": normalized_semantic_profile.scope_relation,
                "operation": normalized_semantic_profile.claimed_operation,
                "equipment": normalized_semantic_profile.claimed_equipment,
                "chamber": normalized_semantic_profile.claimed_chamber,
                "recipe": normalized_semantic_profile.claimed_recipe,
                "claimed_lane_ids": list(
                    normalized_semantic_profile.claimed_lane_ids
                ),
            }
            if normalized_semantic_profile is not None
            else None
        ),
        "authoritative_conclusion_status": authoritative_conclusion_status,
        "scope_status": scope_status,
        "candidate_scope_status": scope_status,
        "publication_status": publication_status,
        "scope_basis": (
            "candidate exposure ∩ process excursion window ∩ matching operation/recipe "
            "∩ compatible outcome"
        ),
        "observed_impact_lots": list(
            dict.fromkeys(str(item) for item in observed_impact_lots)
        ),
        "candidate_impact_lots": [row["lot_id"] for row in candidate_matches],
        "data_missing_evidence_ids": blocking_missing_ids,
        "non_blocking_data_missing_evidence_ids": non_blocking_missing_ids,
        "confirmed_impact_lots": [row["lot_id"] for row in confirmed],
        "confirmation_blocked_reason": (
            None
            if publication_allowed
            else "authoritative RCA conclusion is not supported"
        ),
        "rows": rows,
    }


def derive_impact_publication_result(
    impact_gate: Mapping[str, Any],
    *,
    rca_result_id: str,
    conclusion_status: str,
) -> ImpactPublicationResult:
    """Re-evaluate a stored Impact Gate assessment against the terminal conclusion.

    The publication rule is identical to :func:`evaluate_impact_lot_gate`.
    This function exists so the terminal writer can evaluate that rule against
    the final authoritative conclusion instead of an earlier reasoning-round
    snapshot, so a downgraded terminal conclusion can never keep publishing
    confirmed impact Lots. It never re-runs sources or changes gate criteria.
    """

    gate = impact_gate if isinstance(impact_gate, Mapping) else {}
    raw_rows = gate.get("rows")
    rows = (
        [row for row in raw_rows if isinstance(row, Mapping)]
        if isinstance(raw_rows, list)
        else []
    )
    raw_matches = gate.get("candidate_impact_lots")
    candidate_matches = tuple(
        dict.fromkeys(
            str(item).strip()
            for item in (raw_matches if isinstance(raw_matches, list) else [])
            if str(item).strip()
        )
    )
    publication_allowed = conclusion_status == CONCLUSION_SUPPORTED
    confirmed = candidate_matches if publication_allowed else ()
    if confirmed:
        publication_status = "confirmed"
    elif candidate_matches:
        publication_status = "withheld"
    elif not rows:
        publication_status = "not_evaluated"
    else:
        publication_status = "unconfirmed"
    confirmed_lots = set(confirmed)
    evidence_refs: list[str] = []
    for row in rows:
        if str(row.get("lot_id", "")).strip() not in confirmed_lots:
            continue
        supporting = row.get("supporting_evidence_ids")
        if not isinstance(supporting, list):
            continue
        for evidence_id in supporting:
            normalized = str(evidence_id).strip()
            if normalized and normalized not in evidence_refs:
                evidence_refs.append(normalized)
    return ImpactPublicationResult(
        rca_result_id=rca_result_id,
        publication_status=cast(ImpactPublicationStatus, publication_status),
        confirmed_impact_lots=confirmed,
        evidence_refs=tuple(evidence_refs),
    )


__all__ = [
    "CONCLUSION_INCONCLUSIVE",
    "CONCLUSION_INSUFFICIENT_EVIDENCE",
    "CONCLUSION_SUPPORTED",
    "ConfirmationGateResult",
    "confirm_candidate",
    "derive_impact_publication_result",
    "evaluate_impact_lot_gate",
]
