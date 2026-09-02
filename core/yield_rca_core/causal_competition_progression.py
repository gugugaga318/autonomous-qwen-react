"""Single Python-owned progression engine for causal Candidate competition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from yield_rca_core.causal_investigation_models import (
    ActionValueAssessment,
    AlternativeLaneResolution,
    AlternativeLaneResolutionStatus,
    AlternativeSearchStatus,
    CandidateCompetitionStatus,
    CandidateResolution,
    CandidateResolutionStatus,
    CompetitionFailureReason,
    CompetitionGapReason,
    CompetitionRequirement,
    CompetitionTrace,
    InvestigationGainReasonCode,
    InvestigationGainRecord,
    InvestigationGainType,
    ScopeAssessmentStatus,
)

CONCLUSION_SUPPORTED = "supported"
CONCLUSION_INCONCLUSIVE = "inconclusive"
CONCLUSION_INSUFFICIENT_EVIDENCE = "insufficient_evidence"
_PROCESSING_FAILURE_REASONS = frozenset(
    {
        CompetitionFailureReason.CANDIDATE_VALIDATION_EXHAUSTED.value,
        CompetitionFailureReason.CANDIDATE_PROVIDER_FAILED.value,
        CompetitionFailureReason.CHALLENGE_OUTPUT_INVALID.value,
    }
)


def competition_processing_failed(trace: CompetitionTrace) -> bool:
    """Return whether Competition stopped because a governed processor failed."""

    return (
        trace.competition_status == CandidateCompetitionStatus.FAILED.value
        and trace.competition_failure_reason in _PROCESSING_FAILURE_REASONS
    )


@dataclass(frozen=True)
class CompetitionProgressionResult:
    trace: CompetitionTrace
    conclusion_status: str
    terminal: bool
    blocking_data_missing_evidence_ids: tuple[str, ...] = ()


def _dict(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: object) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def derive_candidate_resolutions(
    authoritative_details: Mapping[str, Any],
) -> tuple[CandidateResolution, ...]:
    """Project per-Candidate resolution from Python-validated ranking and Gate data."""

    ranked_candidates = [
        item
        for item in _list(authoritative_details.get("ranked_candidates"))
        if isinstance(item, Mapping)
    ]
    confirmation = _dict(authoritative_details.get("confirmation_gate"))
    conclusion_status = str(
        authoritative_details.get("conclusion_status", "")
    ).strip()
    confirmation_status = str(confirmation.get("status", "")).strip()
    unresolved_gate_gaps = tuple(
        dict.fromkeys(
            str(item).strip()
            for item in _list(confirmation.get("unresolved_gaps"))
            if str(item).strip()
        )
    )
    confirmation_reasons = tuple(
        dict.fromkeys(
            str(item).strip()
            for item in _list(confirmation.get("reasons"))
            if str(item).strip()
        )
    )
    results: list[CandidateResolution] = []
    for index, candidate in enumerate(ranked_candidates):
        candidate_id = str(candidate.get("candidate_id", "")).strip() or (
            f"candidate_{index}"
        )
        raw_status = str(candidate.get("status", "candidate")).strip()
        matrix = _dict(candidate.get("causal_evidence_matrix"))
        matrix_status = str(
            candidate.get("causal_matrix_status", matrix.get("status", ""))
        ).strip()
        explicit_contradiction_ids = tuple(
            dict.fromkeys(
                str(item).strip()
                for item in _list(candidate.get("contradicting_evidence_ids"))
                if str(item).strip()
            )
        )
        raw_claims = _dict(matrix.get("claims"))
        conflicted_claims = tuple(
            sorted(
                str(claim)
                for claim, result in raw_claims.items()
                if isinstance(result, Mapping)
                and str(result.get("status", "")).strip() == "conflicted"
            )
        )
        if (
            index == 0
            and conclusion_status == CONCLUSION_SUPPORTED
            and confirmation_status == CONCLUSION_SUPPORTED
        ):
            status = CandidateResolutionStatus.SUPPORTED.value
            reason_codes: tuple[str, ...] = ("confirmation_gate_supported",)
            unresolved_gaps: tuple[str, ...] = ()
        elif explicit_contradiction_ids:
            status = CandidateResolutionStatus.CONTRADICTED.value
            reason_codes = ("explicit_contradiction",)
            unresolved_gaps = ()
        elif raw_status in {"conflicted", "rejected"} or (
            matrix_status == "conflicted" and bool(conflicted_claims)
        ):
            status = CandidateResolutionStatus.CONTRADICTED.value
            reason_codes = (
                "claim_evidence_conflict",
                *(f"conflicted_claim:{claim}" for claim in conflicted_claims),
            )
            unresolved_gaps = ()
        else:
            status = CandidateResolutionStatus.UNRESOLVED.value
            reason_values: list[str] = []
            data_missing_ids = _list(candidate.get("data_missing_evidence_ids"))
            if data_missing_ids:
                reason_values.append("missing_data")
            if any("alternative" in item.casefold() for item in unresolved_gate_gaps):
                reason_values.append("equally_strong_alternative")
            if any("signal" in item.casefold() for item in confirmation_reasons):
                reason_values.append("insufficient_signal")
            scope_claim = _dict(raw_claims.get("scope"))
            scope_facts = _dict(scope_claim.get("facts"))
            if (
                str(scope_claim.get("status", "")).strip() == "incomplete"
                and str(scope_facts.get("semantic_profile_status", "")).strip()
                in {"invalid", "missing"}
            ):
                reason_values.append("scope_semantics_incomplete")
            if not reason_values:
                reason_values.append("need_more_evidence")
            reason_codes = tuple(dict.fromkeys(reason_values))
            unresolved_gaps = unresolved_gate_gaps
        evidence_ids = tuple(
            dict.fromkeys(
                str(item).strip()
                for item in _list(
                    candidate.get(
                        "supporting_evidence_ids",
                        candidate.get("evidence_ids", []),
                    )
                )
                if str(item).strip()
            )
        )
        results.append(
            CandidateResolution(
                candidate_id=candidate_id,
                status=status,
                reason_codes=reason_codes,
                evidence_ids=evidence_ids,
                unresolved_gap_ids=unresolved_gaps,
            )
        )
    return tuple(results)


def _with_missing_source_lane_resolutions(
    trace: CompetitionTrace,
    *,
    gain_history: Sequence[InvestigationGainRecord],
    terminally_blocked: bool,
) -> tuple[AlternativeLaneResolution, ...]:
    resolutions = {item.lane_id: item for item in trace.lane_resolutions}
    for gain in gain_history:
        if (
            gain.gain_type != InvestigationGainType.STATE_GAIN.value
            or gain.reason_code
            != InvestigationGainReasonCode.UNAVAILABLE_SOURCE.value
            or gain.lane_id is None
        ):
            continue
        current = resolutions.get(gain.lane_id)
        if current is not None and current.status == (
            AlternativeLaneResolutionStatus.ELIMINATED.value
        ):
            continue
        if (
            current is not None
            and current.status == AlternativeLaneResolutionStatus.BLOCKED.value
            and not terminally_blocked
        ):
            # A newly discovered high-value Action may reactivate the overall
            # competition, but it does not make an already-proven unavailable
            # source available again. Preserve the exact blocked Lane while
            # other Lanes continue.
            continue
        resolutions[gain.lane_id] = AlternativeLaneResolution(
            lane_id=gain.lane_id,
            status=(
                AlternativeLaneResolutionStatus.BLOCKED.value
                if terminally_blocked
                else AlternativeLaneResolutionStatus.UNRESOLVED.value
            ),
            candidate_id=gain.candidate_id or (current.candidate_id if current else None),
            evidence_ids=tuple(
                dict.fromkeys(
                    [
                        *(current.evidence_ids if current else ()),
                        *gain.evidence_ids,
                    ]
                )
            ),
            distinguishing_gap_ids=tuple(
                dict.fromkeys(
                    [
                        *(current.distinguishing_gap_ids if current else ()),
                        *([gain.gap_id] if gain.gap_id else []),
                    ]
                )
            ),
            reason_code=(
                "blocked_by_missing_data"
                if terminally_blocked
                else "unavailable_source"
            ),
            reason=(
                "All decision-relevant sources for this discriminator are "
                "unavailable and no high-value Action remains."
                if terminally_blocked
                else "The requested source is unavailable; the Lane remains unresolved."
            ),
        )
    return tuple(resolutions.values())


def _decision_critical_unavailable_gains(
    *,
    authoritative_details: Mapping[str, Any],
    trace: CompetitionTrace,
    action_value_assessments: Sequence[ActionValueAssessment],
    gain_history: Sequence[InvestigationGainRecord],
    candidate_resolutions: Sequence[CandidateResolution],
) -> tuple[InvestigationGainRecord, ...]:
    """Return only missing-source results connected to the current decision.

    A missing source on an unrelated historical Lane must not block Candidate
    Competition. The match is intentionally ID/fingerprint based; prose and
    broad entity overlap never establish decision criticality.
    """

    raw_gaps = authoritative_details.get("causal_evidence_gaps", [])
    active_gap_ids = {
        str(item.get("gap_id", "")).strip()
        for item in _list(raw_gaps)
        if isinstance(item, Mapping)
        and str(item.get("gap_id", "")).strip()
        and str(item.get("status", "")).strip()
        not in {"supported", "resolved", "closed", "eliminated"}
    }
    assessment_fingerprints = {
        item.scope_fingerprint for item in action_value_assessments
    }
    assessment_gap_ids = {
        item.gap_id for item in action_value_assessments if item.gap_id is not None
    }
    unresolved_candidate_ids = {
        item.candidate_id
        for item in candidate_resolutions
        if item.status == CandidateResolutionStatus.UNRESOLVED.value
    }
    unresolved_lane_ids = {
        item.lane_id
        for item in trace.lane_resolutions
        if item.status
        in {
            AlternativeLaneResolutionStatus.RETAINED.value,
            AlternativeLaneResolutionStatus.UNRESOLVED.value,
            AlternativeLaneResolutionStatus.NON_DISCRIMINATIVE.value,
        }
    }
    relevant_gap_ids = active_gap_ids | assessment_gap_ids
    return tuple(
        gain
        for gain in gain_history
        if gain.gain_type == InvestigationGainType.STATE_GAIN.value
        and gain.reason_code == InvestigationGainReasonCode.UNAVAILABLE_SOURCE.value
        and (
            gain.scope_fingerprint in assessment_fingerprints
            or (gain.gap_id is not None and gain.gap_id in relevant_gap_ids)
            or (
                gain.gap_id is None
                and gain.candidate_id is not None
                and gain.candidate_id in unresolved_candidate_ids
                and (gain.lane_id is None or gain.lane_id in unresolved_lane_ids)
            )
        )
    )


def _confirmation_blocking_data_missing_evidence_ids(
    authoritative_details: Mapping[str, Any],
) -> tuple[str, ...]:
    """Read decision-blocking missing data from the Python Confirmation Gate.

    Declared unavailable sources can enter the typed Evidence collection before
    any causal-gap Action exists.  They therefore have no Action fingerprint or
    ``InvestigationGainRecord`` to match.  The Confirmation Gate is the Python-
    owned component that classifies whether such typed ``data_missing`` Evidence
    blocks confirmation, so its IDs are the authoritative compatibility path.

    New Gate payloads explicitly provide ``blocking_data_missing_evidence_ids``.
    Older payloads only provided ``data_missing_evidence_ids``; use that field
    only when the blocking field is absent, never when it is present and empty
    (which means all missing sources were non-blocking).
    """

    confirmation = _dict(authoritative_details.get("confirmation_gate"))
    conclusion_status = str(
        authoritative_details.get("conclusion_status", "")
    ).strip()
    gate_status = str(confirmation.get("status", "")).strip()
    if (
        conclusion_status != CONCLUSION_INSUFFICIENT_EVIDENCE
        and gate_status != CONCLUSION_INSUFFICIENT_EVIDENCE
    ):
        return ()

    raw_blocking = confirmation.get(
        "blocking_data_missing_evidence_ids",
        authoritative_details.get("blocking_data_missing_evidence_ids"),
    )
    if raw_blocking is None:
        raw_blocking = confirmation.get(
            "data_missing_evidence_ids",
            authoritative_details.get("data_missing_evidence_ids", []),
        )
    return tuple(
        dict.fromkeys(
            str(item).strip()
            for item in _list(raw_blocking)
            if str(item).strip()
        )
    )


def progress_competition(
    *,
    trace: CompetitionTrace,
    authoritative_details: Mapping[str, Any],
    action_value_assessments: Sequence[ActionValueAssessment],
    gain_history: Sequence[InvestigationGainRecord],
    force_terminal: bool,
    budget_exhausted: bool = False,
) -> CompetitionProgressionResult:
    """Advance Competition using Evidence-derived state and bounded Action value."""

    candidate_resolutions = derive_candidate_resolutions(authoritative_details)
    high_value_actions = [
        item for item in action_value_assessments if item.high_value
    ]
    critical_unavailable_gains = _decision_critical_unavailable_gains(
        authoritative_details=authoritative_details,
        trace=trace,
        action_value_assessments=action_value_assessments,
        gain_history=gain_history,
        candidate_resolutions=candidate_resolutions,
    )
    confirmation_blocking_data_missing_evidence_ids = (
        _confirmation_blocking_data_missing_evidence_ids(
            authoritative_details
        )
    )
    decision_critical_unavailable_evidence_ids = tuple(
        dict.fromkeys(
            evidence_id
            for gain in critical_unavailable_gains
            for evidence_id in gain.evidence_ids
        )
    )
    blocking_data_missing_evidence_ids = tuple(
        dict.fromkeys(
            [
                *confirmation_blocking_data_missing_evidence_ids,
                *decision_critical_unavailable_evidence_ids,
            ]
        )
    )
    requirement = trace.competition_requirement
    current_status = trace.competition_status
    terminal_reason: str | None = None

    processing_failed = competition_processing_failed(trace)
    competition_gap_reason = trace.competition_gap_reason
    legacy_failure_reason = trace.competition_failure_reason
    if (
        current_status == CandidateCompetitionStatus.FAILED.value
        and not processing_failed
        and competition_gap_reason is None
        and legacy_failure_reason is not None
    ):
        try:
            competition_gap_reason = CompetitionGapReason(legacy_failure_reason).value
        except ValueError:
            competition_gap_reason = None

    # Older State may encode an unformed competition (for example a collapsed
    # Scope hypothesis) as ``failed``.  Treat only genuine parsing/provider
    # processing failures as terminal failures; unresolved competition must be
    # closed from Action value, unavailable sources, or budget state below.
    if processing_failed:
        status = current_status
        terminal_reason = "candidate_competition_processing_failed"
    elif any(
        item.status == CandidateResolutionStatus.SUPPORTED.value
        for item in candidate_resolutions
    ):
        status = CandidateCompetitionStatus.COMPLETE_CONFIRMED.value
        terminal_reason = "confirmation_gate_supported_unique_candidate"
    elif candidate_resolutions and all(
        item.status == CandidateResolutionStatus.CONTRADICTED.value
        for item in candidate_resolutions
    ):
        status = CandidateCompetitionStatus.COMPLETE_REJECTED.value
        terminal_reason = "all_formal_candidates_contradicted"
    elif requirement in {
        CompetitionRequirement.NOT_EVALUATED.value,
        CompetitionRequirement.NOT_REQUIRED.value,
    }:
        status = CandidateCompetitionStatus.NOT_REQUIRED.value
        terminal_reason = "candidate_competition_not_required"
    elif budget_exhausted:
        status = CandidateCompetitionStatus.BUDGET_EXHAUSTED.value
        terminal_reason = "budget_exhausted_with_unresolved_competition"
    elif (
        force_terminal
        and not high_value_actions
        and (
            critical_unavailable_gains
            or blocking_data_missing_evidence_ids
        )
    ):
        status = CandidateCompetitionStatus.BLOCKED_BY_MISSING_DATA.value
        terminal_reason = "no_high_value_action_after_required_source_unavailable"
    elif force_terminal and not high_value_actions:
        status = CandidateCompetitionStatus.EXHAUSTED.value
        terminal_reason = "no_high_value_action_remains"
    else:
        status = CandidateCompetitionStatus.ACTIVE.value

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
    terminal = status in terminal_statuses
    scope_assessment_status = trace.scope_assessment_status
    if scope_assessment_status in {
        ScopeAssessmentStatus.PENDING.value,
        ScopeAssessmentStatus.ACTIVE.value,
    }:
        impact_scope = _dict(authoritative_details.get("impact_lot_gate"))
        impact_scope_status = str(
            impact_scope.get(
                "scope_status",
                impact_scope.get("candidate_scope_status", ""),
            )
        ).strip()
        if impact_scope_status in {"supported", "confirmed", "resolved"}:
            scope_assessment_status = ScopeAssessmentStatus.RESOLVED.value
        elif force_terminal and blocking_data_missing_evidence_ids:
            scope_assessment_status = (
                ScopeAssessmentStatus.BLOCKED_BY_MISSING_DATA.value
            )
        elif force_terminal:
            scope_assessment_status = ScopeAssessmentStatus.EXHAUSTED.value
        else:
            scope_assessment_status = ScopeAssessmentStatus.ACTIVE.value
    lane_resolutions = _with_missing_source_lane_resolutions(
        trace,
        gain_history=critical_unavailable_gains,
        terminally_blocked=(
            status == CandidateCompetitionStatus.BLOCKED_BY_MISSING_DATA.value
        ),
    )
    blocked_lane_ids = tuple(
        item.lane_id
        for item in lane_resolutions
        if item.status == AlternativeLaneResolutionStatus.BLOCKED.value
    )
    unresolved_lane_ids = tuple(
        dict.fromkeys(
            [
                *(
                    lane_id
                    for lane_id in trace.unresolved_lane_ids
                    if lane_id not in set(blocked_lane_ids)
                ),
                *(
                    item.lane_id
                    for item in lane_resolutions
                    if item.status
                    in {
                        AlternativeLaneResolutionStatus.RETAINED.value,
                        AlternativeLaneResolutionStatus.UNRESOLVED.value,
                        AlternativeLaneResolutionStatus.NON_DISCRIMINATIVE.value,
                    }
                ),
            ]
        )
    )
    alternative_search_status = trace.alternative_search_status
    if status == CandidateCompetitionStatus.COMPLETE_CONFIRMED.value:
        alternative_search_status = AlternativeSearchStatus.ALTERNATIVES_ELIMINATED.value
    elif status == CandidateCompetitionStatus.BLOCKED_BY_MISSING_DATA.value:
        alternative_search_status = AlternativeSearchStatus.BLOCKED_BY_MISSING_DATA.value
    elif status in {
        CandidateCompetitionStatus.COMPLETE_REJECTED.value,
        CandidateCompetitionStatus.EXHAUSTED.value,
        CandidateCompetitionStatus.BUDGET_EXHAUSTED.value,
    }:
        alternative_search_status = AlternativeSearchStatus.UNRESOLVED.value

    updated_trace = replace(
        trace,
        active_lane_ids=tuple(
            lane_id
            for lane_id in trace.active_lane_ids
            if lane_id not in set(blocked_lane_ids)
        ),
        unresolved_lane_ids=unresolved_lane_ids,
        blocked_lane_ids=tuple(
            dict.fromkeys([*trace.blocked_lane_ids, *blocked_lane_ids])
        ),
        lane_resolutions=lane_resolutions,
        candidate_resolutions=candidate_resolutions,
        action_value_assessments=tuple(action_value_assessments),
        alternative_search_status=alternative_search_status,
        competition_status=status,
        scope_assessment_status=scope_assessment_status,
        competition_failure_reason=(
            trace.competition_failure_reason
            if status == CandidateCompetitionStatus.FAILED.value
            else None
        ),
        competition_gap_reason=(
            None
            if status
            in {
                CandidateCompetitionStatus.NOT_REQUIRED.value,
                CandidateCompetitionStatus.COMPLETE_CONFIRMED.value,
                CandidateCompetitionStatus.COMPLETE_REJECTED.value,
                CandidateCompetitionStatus.RESOLVED.value,
                CandidateCompetitionStatus.FAILED.value,
            }
            else competition_gap_reason
        ),
        terminal_reason=terminal_reason,
        resolution_evidence_ids=tuple(
            dict.fromkeys(
                [
                    *trace.resolution_evidence_ids,
                    *blocking_data_missing_evidence_ids,
                ]
            )
        ),
    )
    if status == CandidateCompetitionStatus.COMPLETE_CONFIRMED.value:
        conclusion_status = CONCLUSION_SUPPORTED
    elif status in {
        CandidateCompetitionStatus.BLOCKED_BY_MISSING_DATA.value,
        CandidateCompetitionStatus.BUDGET_EXHAUSTED.value,
    }:
        conclusion_status = CONCLUSION_INSUFFICIENT_EVIDENCE
    elif status == CandidateCompetitionStatus.NOT_REQUIRED.value:
        raw_conclusion = str(
            authoritative_details.get("conclusion_status", "")
        ).strip()
        conclusion_status = (
            raw_conclusion
            if raw_conclusion
            in {
                CONCLUSION_SUPPORTED,
                CONCLUSION_INCONCLUSIVE,
                CONCLUSION_INSUFFICIENT_EVIDENCE,
            }
            else CONCLUSION_INCONCLUSIVE
        )
    else:
        conclusion_status = CONCLUSION_INCONCLUSIVE
    return CompetitionProgressionResult(
        trace=updated_trace,
        conclusion_status=conclusion_status,
        terminal=terminal,
        blocking_data_missing_evidence_ids=(
            blocking_data_missing_evidence_ids
        ),
    )


__all__ = [
    "CompetitionProgressionResult",
    "competition_processing_failed",
    "derive_candidate_resolutions",
    "progress_competition",
]
