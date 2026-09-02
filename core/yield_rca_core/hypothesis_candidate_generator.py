"""Qwen proposes novel causal hypotheses; Python owns every acceptance gate."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from yield_rca_core.causal_evidence_matrix import (
    build_causal_evidence_matrix,
    is_relevant_mechanism_intermediate,
)
from yield_rca_core.causal_hypothesis import CausalHypothesis
from yield_rca_core.causal_investigation_models import (
    CandidateClaimedScopeKind,
    CandidateCompetitionStatus,
    CandidateCompetitionType,
    CandidateDistinguishingPrediction,
    CandidateLaneEffectExpectation,
    CandidateMechanismRelation,
    CandidateScopeRelation,
    CandidateSemanticProfile,
    CompetitionFailureReason,
    CompetitionGapReason,
    CompetitionRequirement,
)
from yield_rca_core.evidence_models import EntityType, Evidence, EvidenceType
from yield_rca_core.evidence_synthesis import (
    build_lane_first_evidence_synthesis,
    compact_evidence_prompt_card,
    compact_evidence_record,
    compact_lane_first_synthesis_for_prompt,
    project_prompt_evidence_references,
)
from yield_rca_core.llm_gateway import (
    LLMClient,
    LLMOutputValidationError,
    LLMRequest,
    llm_call_budget_available,
)
from yield_rca_core.models import AgentFinding, AgentKind, ModelValidationError

_OUTPUT_ATTEMPTS = 2
_MAX_CANDIDATES = 2
_MAX_PROMPT_EVIDENCE = 56
_MAX_RECENT_EVIDENCE = 16
_MAX_PRIOR_CANDIDATE_EVIDENCE = 12
_MAX_CHALLENGE_EVIDENCE = 8
_TARGET_PROMPT_PAYLOAD_CHARS = 52_000
_MAX_PROMPT_PAYLOAD_CHARS = 64_000
_NON_SUPPORTING_TYPES = {
    EvidenceType.DATA_MISSING.value,
    EvidenceType.NEGATIVE_SIGNAL.value,
    EvidenceType.SOP_GUIDANCE.value,
}
_KNOWLEDGE_MECHANISM_TYPES = {
    EvidenceType.HISTORICAL_CASE_MATCH.value,
    EvidenceType.ENGINEERING_NOTE.value,
}
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
_PRODUCT_TYPES = {
    EvidenceType.DEFECT_SIGNAL.value,
    EvidenceType.METROLOGY_DEVIATION.value,
    EvidenceType.ELECTRICAL_FAILURE.value,
}
_DUPLICATE_EVIDENCE_OVERLAP_THRESHOLD = 0.75
_DUPLICATE_MECHANISM_SIMILARITY_THRESHOLD = 0.65
_MAX_CLOSURE_REPAIR_EVIDENCE = 12
_MAX_CLOSURE_EVIDENCE_PER_ROLE = 2
_CLOSURE_EVIDENCE_TYPES = {
    "exposure": _EXPOSURE_TYPES - {EvidenceType.EXCURSION_WINDOW.value},
    "parameter": _PROCESS_TYPES,
    "temporal": {EvidenceType.EXCURSION_WINDOW.value},
    "outcome": _PRODUCT_TYPES,
    "mechanism_intermediate": _PRODUCT_TYPES,
}


def _compact_scope_identity(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _evidence_bucket(evidence: Evidence) -> str:
    if evidence.evidence_type in _PROCESS_TYPES:
        return "process"
    if evidence.evidence_type in _PRODUCT_TYPES:
        return "product"
    if evidence.evidence_type in _EXPOSURE_TYPES:
        return "exposure"
    if evidence.evidence_type in _KNOWLEDGE_MECHANISM_TYPES:
        return "knowledge"
    if evidence.evidence_type == EvidenceType.DATA_MISSING.value:
        return "data_missing"
    return "other"


def _diverse_bounded_ids(
    evidence_ids: Sequence[str],
    *,
    evidence_by_id: Mapping[str, Evidence],
    limit: int,
) -> list[str]:
    """Select a deterministic typed-Evidence cross-section without inference."""

    buckets: dict[str, list[str]] = {
        key: []
        for key in (
            "process",
            "product",
            "exposure",
            "knowledge",
            "data_missing",
            "other",
        )
    }
    for evidence_id in dict.fromkeys(str(item) for item in evidence_ids):
        evidence = evidence_by_id.get(evidence_id)
        if evidence is not None:
            buckets[_evidence_bucket(evidence)].append(evidence_id)
    selected: list[str] = []
    while len(selected) < limit and any(buckets.values()):
        for bucket in buckets.values():
            if bucket and len(selected) < limit:
                selected.append(bucket.pop(0))
    return selected


def _bounded_prompt_evidence_ids(
    *,
    evidence_by_id: Mapping[str, Evidence],
    synthesis_ids: Sequence[str],
    recent_ids: Sequence[str],
    prior_candidate_ids: Sequence[str],
    challenge_ids: Sequence[str],
) -> tuple[str, ...]:
    """Apply one hard bound to every Candidate Generator prompt round.

    Recent Action Evidence is deliberately reserved space, but a first-round
    ``new_evidence_ids_since_prior`` list can never reopen the full register.
    The remaining slots are filled from Python's already bounded Lane-first
    synthesis and traceable prior Candidate/Challenge references.
    """

    ordered_groups = (
        _diverse_bounded_ids(
            recent_ids,
            evidence_by_id=evidence_by_id,
            limit=_MAX_RECENT_EVIDENCE,
        ),
        _diverse_bounded_ids(
            prior_candidate_ids,
            evidence_by_id=evidence_by_id,
            limit=_MAX_PRIOR_CANDIDATE_EVIDENCE,
        ),
        _diverse_bounded_ids(
            challenge_ids,
            evidence_by_id=evidence_by_id,
            limit=_MAX_CHALLENGE_EVIDENCE,
        ),
        _diverse_bounded_ids(
            synthesis_ids,
            evidence_by_id=evidence_by_id,
            limit=_MAX_PROMPT_EVIDENCE,
        ),
    )
    selected: list[str] = []
    for group in ordered_groups:
        for evidence_id in group:
            if evidence_id not in selected:
                selected.append(evidence_id)
                if len(selected) == _MAX_PROMPT_EVIDENCE:
                    return tuple(selected)
    if selected:
        return tuple(selected)
    return tuple(
        _diverse_bounded_ids(
            list(evidence_by_id),
            evidence_by_id=evidence_by_id,
            limit=_MAX_PROMPT_EVIDENCE,
        )
    )


@dataclass(frozen=True)
class HypothesisCandidateProposal:
    """A model-authored explanation with IDs bound to immutable Evidence."""

    root_cause: str
    causal_explanation: str
    supporting_evidence_ids: tuple[str, ...]
    contradicting_evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.root_cause, str) or not self.root_cause.strip():
            raise ModelValidationError("candidate root_cause must be non-empty")
        if self.root_cause.strip().casefold() == "inconclusive":
            raise ModelValidationError("inconclusive is not a hypothesis candidate")
        if (
            not isinstance(self.causal_explanation, str)
            or not self.causal_explanation.strip()
        ):
            raise ModelValidationError(
                "candidate causal_explanation must be non-empty"
            )
        for field_name, values in (
            ("supporting_evidence_ids", self.supporting_evidence_ids),
            ("contradicting_evidence_ids", self.contradicting_evidence_ids),
        ):
            if not isinstance(values, tuple) or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise ModelValidationError(
                    f"candidate {field_name} must contain non-empty strings"
                )
            if len(values) != len(set(values)):
                raise ModelValidationError(
                    f"candidate {field_name} must not contain duplicates"
                )
        if not self.supporting_evidence_ids:
            raise ModelValidationError(
                "candidate supporting_evidence_ids must not be empty"
            )
        if set(self.supporting_evidence_ids) & set(self.contradicting_evidence_ids):
            raise ModelValidationError(
                "candidate Evidence cannot be both supporting and contradicting"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_cause": self.root_cause,
            "causal_explanation": self.causal_explanation,
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "contradicting_evidence_ids": list(self.contradicting_evidence_ids),
        }


@dataclass(frozen=True)
class HypothesisCandidateGeneration:
    candidates: tuple[HypothesisCandidateProposal, ...]
    attempt_count: int
    validation_errors: tuple[str, ...] = ()
    candidate_output_invalid: bool = False
    analysis_summary: str = ""
    targeted_investigation_results: tuple[dict[str, Any], ...] = ()
    competition_repair_exhausted: bool = False
    competition_repair_skipped_due_to_budget: bool = False
    rejected_candidates: tuple[dict[str, Any], ...] = ()
    competition_requirement: str = CompetitionRequirement.NOT_EVALUATED.value
    competition_status: str = CandidateCompetitionStatus.NOT_EVALUATED.value
    competition_type: str = CandidateCompetitionType.NOT_EVALUATED.value
    competition_failure_reason: str | None = None
    competition_gap_reason: str | None = None
    competition_assessment: dict[str, Any] | None = None
    candidate_semantic_profiles: tuple[CandidateSemanticProfile, ...] = ()
    semantic_validation_errors: tuple[str, ...] = ()
    candidate_lineage: tuple[dict[str, Any], ...] = ()
    evidence_synthesis: dict[str, Any] | None = None
    candidate_evidence_closure: tuple[dict[str, Any], ...] = ()
    candidate_evidence_closure_history: tuple[dict[str, Any], ...] = ()
    evidence_closure_repair_attempted: bool = False
    evidence_closure_repair_exhausted: bool = False
    evidence_closure_repair_skipped_due_to_budget: bool = False


@dataclass(frozen=True)
class _CandidateDedupSignature:
    lane_ids: tuple[str, ...]
    recipes: tuple[str, ...]
    chambers: tuple[str, ...]
    parameter_scope: tuple[str, ...]
    parameter_evidence_ids: tuple[str, ...]
    supporting_evidence_ids: tuple[str, ...]
    evidence_types: tuple[str, ...]
    discriminator_gap_ids: tuple[str, ...]
    mechanism_tokens: frozenset[str]


@dataclass(frozen=True)
class _DuplicateAssessment:
    is_duplicate: bool
    duplicate_score: float
    reason: str
    left_signature: _CandidateDedupSignature
    right_signature: _CandidateDedupSignature


def _evidence_register(
    findings: list[AgentFinding],
    context_evidence: Sequence[Evidence] = (),
    *,
    allowed_evidence_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    evidence_by_id = {
        evidence.evidence_id: evidence
        for finding in findings
        for evidence in finding.evidence
        if evidence.is_typed
    }
    evidence_by_id.update(
        {
            evidence.evidence_id: evidence
            for evidence in context_evidence
            if evidence.is_typed
        }
    )
    return [
        compact_evidence_prompt_card(evidence)
        for evidence_id, evidence in evidence_by_id.items()
        if allowed_evidence_ids is None or evidence_id in allowed_evidence_ids
    ]


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


def _eligible_evidence_ids_by_lane(
    evidence_by_id: dict[str, Evidence],
) -> dict[str, list[str]]:
    """Expose typed repair choices without claiming causal relevance for Qwen."""

    lane_types = {
        "shared_exposure": _EXPOSURE_TYPES,
        "process_anomaly": _PROCESS_TYPES,
        "product_outcome": _PRODUCT_TYPES,
    }
    result = {
        lane: sorted(
            evidence_id
            for evidence_id, evidence in evidence_by_id.items()
            if evidence.evidence_type in evidence_types
            and evidence.evidence_type not in _NON_SUPPORTING_TYPES
        )
        for lane, evidence_types in lane_types.items()
    }
    result["mechanism_support"] = sorted(
        evidence_id
        for evidence_id, evidence in evidence_by_id.items()
        if _is_approved_knowledge_support(evidence)
    )
    return result


def _is_approved_knowledge_support(evidence: Evidence) -> bool:
    """Knowledge may support mechanism only after explicit approval."""

    if (
        evidence.source_type != "knowledge"
        or evidence.evidence_type not in _KNOWLEDGE_MECHANISM_TYPES
    ):
        return False
    statuses = [
        str(value).upper()
        for key, value in evidence.metadata.items()
        if str(key).casefold() == "validation_status"
    ]
    statuses.extend(
        str(value).upper()
        for entity in evidence.entities
        for key, value in entity.attributes.items()
        if str(key).casefold() == "validation_status"
    )
    return bool(statuses) and all(status == "CONFIRMED" for status in statuses)


def _prior_candidate_mechanism_feedback(
    prior_candidates: Sequence[Mapping[str, Any]],
    *,
    evidence_by_id: Mapping[str, Evidence],
) -> list[dict[str, Any]]:
    """Expose Python-owned mechanism gaps during a reasoning refresh."""

    feedback: list[dict[str, Any]] = []
    for index, candidate in enumerate(prior_candidates[:_MAX_CANDIDATES]):
        root_cause = str(candidate.get("root_cause", "")).strip()
        explanation = str(
            candidate.get("causal_explanation", candidate.get("root_cause", ""))
        ).strip()
        supporting = tuple(
            dict.fromkeys(
                str(item)
                for item in candidate.get("supporting_evidence_ids", [])
                if str(item) in evidence_by_id
            )
        )
        contradicting = tuple(
            dict.fromkeys(
                str(item)
                for item in candidate.get("contradicting_evidence_ids", [])
                if str(item) in evidence_by_id and str(item) not in supporting
            )
        )
        if not root_cause or not explanation or not supporting:
            continue
        try:
            matrix = build_causal_evidence_matrix(
                CausalHypothesis(
                    root_cause=root_cause,
                    causal_explanation=explanation,
                    supporting_evidence_ids=supporting,
                    contradicting_evidence_ids=contradicting,
                ),
                evidence_by_id.values(),
            )
        except (TypeError, ValueError):
            continue
        mechanism = matrix.claims["mechanism"]
        feedback.append(
            {
                "candidate_index": index,
                "mechanism_status": mechanism.status,
                "mechanism_support_source": mechanism.support_source,
                "reason": mechanism.reason,
                "evidence_ids": list(mechanism.evidence_ids),
                "proposed_physical_bridge_terms": list(
                    mechanism.facts.get("proposed_physical_bridge_terms", [])
                ),
                "empirical_shared_lot_ids": list(
                    mechanism.facts.get("empirical_shared_lot_ids", [])
                ),
                "causal_chain_status": matrix.causal_chain_completeness,
            }
        )
    return feedback


def _candidate_repair_feedback(
    validation_error: str,
    *,
    evidence_by_id: dict[str, Evidence],
    candidate_competition: Mapping[str, Any] | None = None,
    candidate_evidence_closure: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    eligible_by_lane = _eligible_evidence_ids_by_lane(evidence_by_id)
    compact_closure = [
        {
            "candidate_index": item.get("candidate_index"),
            "candidate_id": item.get("candidate_id"),
            "status": item.get("status"),
            "candidate_snapshot": dict(item.get("candidate_snapshot", {})),
            "must_preserve_evidence_ids": list(
                item.get("must_preserve_evidence_ids", [])
            ),
            "citation_regression_evidence_ids": list(
                item.get("citation_regression_evidence_ids", [])
            ),
            "closure_gaps": [
                {
                    "lane_id": gap.get("lane_id"),
                    "evidence_role": gap.get("evidence_role"),
                    "eligible_evidence_ids": list(
                        gap.get("eligible_evidence_ids", [])
                    ),
                }
                for gap in item.get("closure_gaps", [])
                if isinstance(gap, Mapping)
            ],
        }
        for item in candidate_evidence_closure
    ]
    return {
        "message": validation_error,
        "missing_causal_lanes": [],
        "eligible_supporting_evidence_ids_by_lane": eligible_by_lane,
        "source_agent_by_evidence_id": {
            evidence_id: evidence.source_agent
            for evidence_id, evidence in sorted(evidence_by_id.items())
            if evidence.source_agent is not None
        },
        "repair_instruction": (
            "Repair only the reported schema or Evidence-reference error. Use "
            "causally relevant IDs from eligible_supporting_evidence_ids_by_lane, "
            "but do not attach an irrelevant Evidence ID merely to make a candidate "
            "look complete. An incomplete evidence-bounded candidate is valid: "
            "Python records its missing lanes in CausalEvidenceMatrix and may seek "
            "targeted Evidence later. Return candidates=[] only when no causal "
            "candidate is justified at all."
        ),
        "candidate_evidence_closure": compact_closure,
        "evidence_closure_instruction": (
            "For each reported candidate/Lane/role closure gap, decide whether "
            "one of the listed typed Evidence IDs genuinely supports the "
            "Candidate's claimed scope. Cite it yourself when relevant, narrow "
            "the Qwen-owned claimed_scope when that is the accurate claim, or "
            "return a bounded incomplete/empty candidate when it is not relevant. "
            "Python has not attached or selected Evidence for you. A Closure repair "
            "is cumulative: retain every ID in must_preserve_evidence_ids while "
            "adding only genuinely relevant missing citations. Removing one of "
            "those still-valid IDs is a citation_regression, not a successful "
            "repair. Comparison-scope Lanes that are not claimed are never "
            "citation requirements."
        ),
        "valid_empty_output": {
            "candidates": [],
            "analysis_summary": (
                "No evidence-bounded causal candidate is justified."
            ),
        },
        "candidate_competition": (
            _bounded_candidate_competition_context(
                candidate_competition,
                allowed_evidence_ids=set(evidence_by_id),
            )
            if candidate_competition is not None
            else None
        ),
    }


def _bounded_candidate_competition_context(
    context: Mapping[str, Any],
    *,
    allowed_evidence_ids: set[str],
) -> dict[str, Any]:
    """Project competition context onto the exact typed prompt register."""

    def bounded_ids(values: object) -> list[str]:
        if not isinstance(values, Sequence) or isinstance(values, str | bytes):
            return []
        return list(
            dict.fromkeys(
                str(item)
                for item in values
                if str(item) in allowed_evidence_ids
            )
        )

    challenges: list[dict[str, Any]] = []
    for raw in context.get("prior_candidate_challenges", []):
        if not isinstance(raw, Mapping):
            continue
        challenge = dict(raw)
        for field_name in (
            "supporting_evidence_ids",
            "contradicting_evidence_ids",
            "unexplained_precursor_evidence_ids",
        ):
            challenge[field_name] = bounded_ids(raw.get(field_name, []))
        challenges.append(challenge)

    targeted_results: list[dict[str, Any]] = []
    for raw in context.get("targeted_investigation_results", []):
        if not isinstance(raw, Mapping):
            continue
        result = dict(raw)
        for field_name in (
            "new_evidence_ids",
            "new_supporting_evidence_ids",
            "new_data_missing_evidence_ids",
        ):
            result[field_name] = bounded_ids(raw.get(field_name, []))
        raw_evidence = raw.get("new_evidence", [])
        result["new_evidence"] = [
            dict(item)
            for item in raw_evidence
            if isinstance(item, Mapping)
            and str(item.get("evidence_id", "")) in allowed_evidence_ids
        ] if isinstance(raw_evidence, Sequence) else []
        result["answered"] = bool(result["new_evidence_ids"])
        result["support_observed"] = bool(
            result["new_supporting_evidence_ids"]
        )
        targeted_results.append(result)

    targeted_supporting_ids = bounded_ids(
        context.get("targeted_supporting_evidence_ids", [])
    )
    return {
        "new_evidence_ids_since_prior": bounded_ids(
            context.get("new_evidence_ids_since_prior", [])
        ),
        "prior_candidate_challenges": challenges,
        "targeted_investigation_results": targeted_results,
        "relevant_causal_lanes": [
            dict(item)
            for item in context.get("relevant_causal_lanes", [])
            if isinstance(item, Mapping)
        ],
        "targeted_supporting_evidence_ids": targeted_supporting_ids,
        "requires_distinct_candidate_review": bool(targeted_supporting_ids),
    }


def _normalized_lane_scope(lane: Mapping[str, Any]) -> dict[str, Any]:
    """Return immutable factual scope used only for citation-closure matching."""

    return {
        "lane_id": str(lane.get("lane_id", "")).strip(),
        "operation": str(lane.get("operation", "")).strip().casefold(),
        "equipment": str(lane.get("equipment", "")).strip().casefold(),
        "chamber": str(lane.get("chamber", "")).strip().casefold(),
        "recipe": str(lane.get("recipe", "")).strip().casefold(),
        "exposed_lot_ids": {
            str(item).strip().casefold()
            for item in lane.get("exposed_lot_ids", [])
            if str(item).strip()
        },
    }


def _explicit_evidence_lane_ids(evidence: Evidence) -> set[str]:
    lane_ids = {
        str(value).strip()
        for value in _metadata_scope_values(evidence, "lane_id")
        if str(value).strip()
    }
    raw_lane = evidence.metadata.get("lane")
    if isinstance(raw_lane, Mapping):
        lane_id = str(raw_lane.get("lane_id", "")).strip()
        if lane_id:
            lane_ids.add(lane_id.casefold())
    return {item.casefold() for item in lane_ids}


def _evidence_matches_claimed_lane(
    evidence: Evidence,
    lane: Mapping[str, Any],
) -> bool:
    """Match typed Evidence to one claimed Lane without causal inference."""

    normalized_lane = _normalized_lane_scope(lane)
    lane_id = str(normalized_lane["lane_id"]).casefold()
    explicit_lane_ids = _explicit_evidence_lane_ids(evidence)
    if explicit_lane_ids:
        return lane_id in explicit_lane_ids

    identity_pairs = (
        (
            "operation",
            _entity_scope_values(evidence, EntityType.OPERATION.value)
            | _metadata_scope_values(evidence, "operation", "operation_no"),
        ),
        (
            "equipment",
            _entity_scope_values(evidence, EntityType.EQUIPMENT.value)
            | _metadata_scope_values(evidence, "equipment", "equipment_id"),
        ),
        (
            "chamber",
            _entity_scope_values(evidence, EntityType.CHAMBER.value)
            | _metadata_scope_values(evidence, "chamber", "chamber_id"),
        ),
        (
            "recipe",
            _entity_scope_values(evidence, EntityType.RECIPE.value)
            | _metadata_scope_values(evidence, "recipe", "recipe_id"),
        ),
    )
    observed_identity_count = 0
    matched_identity_count = 0
    for field_name, observed_values in identity_pairs:
        expected = str(normalized_lane[field_name])
        if not observed_values or not expected:
            continue
        observed_identity_count += 1
        if expected not in observed_values:
            return False
        matched_identity_count += 1

    evidence_lots = _entity_scope_values(evidence, EntityType.LOT.value)
    lane_lots = set(normalized_lane["exposed_lot_ids"])
    lot_overlap = bool(evidence_lots & lane_lots)
    if evidence.evidence_type in _PRODUCT_TYPES:
        return lot_overlap or matched_identity_count >= 2
    return matched_identity_count >= 2 or (
        observed_identity_count == 1 and lot_overlap
    )


def _lane_matches_claimed_scope(
    lane: Mapping[str, Any],
    profile: CandidateSemanticProfile,
) -> bool:
    lane_id = str(lane.get("lane_id", "")).strip()
    if profile.claimed_scope_kind == CandidateClaimedScopeKind.LANE.value:
        return lane_id in set(profile.claimed_lane_ids)
    required = {
        "operation": profile.claimed_operation,
        "equipment": profile.claimed_equipment,
        "chamber": profile.claimed_chamber,
        "recipe": profile.claimed_recipe,
    }
    return all(
        expected is None
        or _compact_scope_identity(lane.get(field_name, ""))
        == _compact_scope_identity(expected)
        for field_name, expected in required.items()
    )


def _evidence_matches_claimed_scope(
    evidence: Evidence,
    profile: CandidateSemanticProfile,
    matching_lanes: Sequence[Mapping[str, Any]],
    *,
    allow_broad_identity_fallback: bool = True,
) -> bool:
    if any(
        _evidence_matches_claimed_lane(evidence, lane)
        for lane in matching_lanes
    ):
        return True
    if profile.claimed_scope_kind == CandidateClaimedScopeKind.LANE.value:
        return False
    if not allow_broad_identity_fallback:
        return False
    identity_pairs = (
        (
            profile.claimed_operation,
            _entity_scope_values(evidence, EntityType.OPERATION.value)
            | _metadata_scope_values(evidence, "operation", "operation_no"),
        ),
        (
            profile.claimed_equipment,
            _entity_scope_values(evidence, EntityType.EQUIPMENT.value)
            | _metadata_scope_values(evidence, "equipment", "equipment_id"),
        ),
        (
            profile.claimed_chamber,
            _entity_scope_values(evidence, EntityType.CHAMBER.value)
            | _metadata_scope_values(evidence, "chamber", "chamber_id"),
        ),
        (
            profile.claimed_recipe,
            _entity_scope_values(evidence, EntityType.RECIPE.value)
            | _metadata_scope_values(evidence, "recipe", "recipe_id"),
        ),
    )
    observed = 0
    for expected, actual_values in identity_pairs:
        if expected is None or not actual_values:
            continue
        observed += 1
        if _compact_scope_identity(expected) not in {
            _compact_scope_identity(value) for value in actual_values
        }:
            return False
    return observed > 0


def _candidate_evidence_closure_assessment(
    proposals: Sequence[HypothesisCandidateProposal],
    semantic_profiles: Sequence[CandidateSemanticProfile],
    *,
    evidence_by_id: Mapping[str, Evidence],
    causal_lanes: Sequence[Mapping[str, Any]],
    raw_scope_by_candidate_index: Mapping[int, Mapping[str, Any]] | None = None,
    must_preserve_evidence_ids_by_candidate_index: Mapping[
        int, Sequence[str]
    ] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Detect omitted typed citations without changing a Qwen Candidate."""

    lane_by_id = {
        str(lane.get("lane_id", "")).strip(): lane
        for lane in causal_lanes
        if str(lane.get("lane_id", "")).strip()
    }
    assessments: list[dict[str, Any]] = []
    raw_scopes = raw_scope_by_candidate_index or {}
    preserve_by_index = must_preserve_evidence_ids_by_candidate_index or {}
    for candidate_index, proposal in enumerate(proposals):
        profile = (
            semantic_profiles[candidate_index]
            if candidate_index < len(semantic_profiles)
            else None
        )
        raw_scope = raw_scopes.get(candidate_index, {})
        raw_scope_kind = str(
            raw_scope.get(
                "claimed_scope_kind",
                CandidateClaimedScopeKind.LANE.value,
            )
        )
        raw_scope_relation = str(
            raw_scope.get(
                "scope_relation",
                CandidateScopeRelation.UNRESOLVED.value,
            )
        )
        raw_scope_identity = {
            field_name: raw_scope.get(f"claimed_{field_name}")
            for field_name in ("operation", "equipment", "chamber", "recipe")
        }
        candidate_id = profile.candidate_id if profile is not None else ""
        claimed_lane_ids = (
            tuple(profile.claimed_lane_ids)
            if profile is not None
            else tuple(str(item) for item in raw_scope.get("claimed_lane_ids", []))
        )
        comparison_lane_ids = (
            tuple(profile.comparison_lane_ids)
            if profile is not None
            else tuple(
                str(item) for item in raw_scope.get("comparison_lane_ids", [])
            )
        )
        scope_reference_source = (
            "validated_semantic_profile"
            if profile is not None
            else (
                "raw_structurally_valid_scope_reference"
                if claimed_lane_ids
                else "unavailable"
            )
        )
        supporting_ids = set(proposal.supporting_evidence_ids)
        must_preserve_evidence_ids = tuple(
            dict.fromkeys(
                str(item)
                for item in preserve_by_index.get(
                    candidate_index,
                    proposal.supporting_evidence_ids,
                )
                if str(item) in evidence_by_id
                and evidence_by_id[str(item)].evidence_type
                not in _NON_SUPPORTING_TYPES
            )
        )
        gaps: list[dict[str, Any]] = []
        if profile is not None:
            matching_scope_lanes = [
                lane
                for lane in causal_lanes
                if _lane_matches_claimed_scope(lane, profile)
            ]
        else:
            if raw_scope_kind != CandidateClaimedScopeKind.LANE.value:
                matching_scope_lanes = [
                    lane
                    for lane in causal_lanes
                    if all(
                        expected in (None, "")
                        or _compact_scope_identity(lane.get(field_name, ""))
                        == _compact_scope_identity(expected)
                        for field_name, expected in raw_scope_identity.items()
                    )
                ]
            else:
                matching_scope_lanes = [
                    lane_by_id[lane_id]
                    for lane_id in claimed_lane_ids
                    if lane_id in lane_by_id
                ]
        closure_targets: list[tuple[str | None, Sequence[Mapping[str, Any]]]]
        if (
            profile is not None
            and profile.claimed_scope_kind != CandidateClaimedScopeKind.LANE.value
        ) or (
            profile is None
            and raw_scope_kind != CandidateClaimedScopeKind.LANE.value
        ):
            # Exposure, temporal, and physical-intermediate closure can be
            # grounded at the declared broad identity. A shared-effect claim's
            # process/outcome coverage is checked per Lane below.
            closure_targets = [(None, matching_scope_lanes)]
        else:
            closure_targets = [
                (str(lane.get("lane_id", "")), [lane])
                for lane in matching_scope_lanes
            ]
        candidate_hypothesis = CausalHypothesis(
            root_cause=proposal.root_cause,
            causal_explanation=proposal.causal_explanation,
            supporting_evidence_ids=proposal.supporting_evidence_ids,
            contradicting_evidence_ids=proposal.contradicting_evidence_ids,
        )
        shared_effect_scope = (
            (
                profile.scope_relation
                if profile is not None
                else raw_scope_relation
            )
            == CandidateScopeRelation.SHARED_EFFECT.value
            and (
                profile.claimed_scope_kind
                if profile is not None
                else raw_scope_kind
            )
            != CandidateClaimedScopeKind.LANE.value
        )
        for evidence_role, evidence_types in _CLOSURE_EVIDENCE_TYPES.items():
            role_targets = (
                [
                    (str(lane.get("lane_id", "")), [lane])
                    for lane in matching_scope_lanes
                ]
                if shared_effect_scope
                and evidence_role in {"parameter", "outcome"}
                else closure_targets
            )
            for lane_id, target_lanes in role_targets:
                matching_ids = sorted(
                    evidence_id
                    for evidence_id, evidence in evidence_by_id.items()
                    if evidence.evidence_type in evidence_types
                    and evidence.evidence_type not in _NON_SUPPORTING_TYPES
                    and (
                        evidence_role != "mechanism_intermediate"
                        or is_relevant_mechanism_intermediate(
                            evidence,
                            candidate_hypothesis,
                        )
                    )
                    and (
                        _evidence_matches_claimed_scope(
                            evidence,
                            profile,
                            target_lanes,
                            allow_broad_identity_fallback=not (
                                shared_effect_scope
                                and evidence_role in {"parameter", "outcome"}
                            ),
                        )
                        if profile is not None
                        else any(
                            _evidence_matches_claimed_lane(evidence, lane)
                            for lane in target_lanes
                        )
                    )
                )
                cited_ids = sorted(supporting_ids & set(matching_ids))
                if matching_ids and not cited_ids:
                    gaps.append(
                        {
                            "lane_id": lane_id,
                            "claimed_scope_kind": (
                                profile.claimed_scope_kind
                                if profile is not None
                                else raw_scope_kind
                            ),
                            "evidence_role": evidence_role,
                            "available_evidence_count": len(matching_ids),
                            "eligible_evidence_ids": matching_ids[
                                :_MAX_CLOSURE_EVIDENCE_PER_ROLE
                            ],
                            "cited_matching_evidence_ids": [],
                        }
                    )
        citation_regressions = sorted(
            set(must_preserve_evidence_ids) - supporting_ids
        )
        if citation_regressions:
            gaps.append(
                {
                    "lane_id": None,
                    "claimed_scope_kind": (
                        profile.claimed_scope_kind
                        if profile is not None
                        else raw_scope_kind
                    ),
                    "evidence_role": "citation_regression",
                    "available_evidence_count": len(citation_regressions),
                    "eligible_evidence_ids": citation_regressions,
                    "cited_matching_evidence_ids": [],
                }
            )
        assessments.append(
            {
                "candidate_index": candidate_index,
                "candidate_id": candidate_id or None,
                "status": (
                    "not_evaluated"
                    if profile is None and not claimed_lane_ids
                    else ("incomplete" if gaps else "complete")
                ),
                "scope_reference_source": scope_reference_source,
                "claimed_lane_ids": list(claimed_lane_ids),
                "claimed_scope_kind": (
                    profile.claimed_scope_kind
                    if profile is not None
                    else raw_scope_kind
                ),
                "claimed_scope_identity": (
                    {
                        "operation": profile.claimed_operation,
                        "equipment": profile.claimed_equipment,
                        "chamber": profile.claimed_chamber,
                        "recipe": profile.claimed_recipe,
                    }
                    if profile is not None
                    else raw_scope_identity
                ),
                "matching_claimed_scope_lane_ids": [
                    str(lane.get("lane_id", "")) for lane in matching_scope_lanes
                ],
                "comparison_lane_ids": list(comparison_lane_ids),
                "supporting_evidence_ids": list(proposal.supporting_evidence_ids),
                "must_preserve_evidence_ids": list(
                    must_preserve_evidence_ids
                ),
                "citation_regression_evidence_ids": citation_regressions,
                "candidate_snapshot": {
                    "root_cause": proposal.root_cause,
                    "causal_explanation": proposal.causal_explanation,
                    "supporting_evidence_ids": list(
                        proposal.supporting_evidence_ids
                    ),
                    "contradicting_evidence_ids": list(
                        proposal.contradicting_evidence_ids
                    ),
                    "semantic_profile": (
                        profile.to_dict()
                        if profile is not None
                        else dict(raw_scope)
                    ),
                },
                "closure_gaps": gaps,
                "python_mutated_supporting_evidence_ids": False,
            }
        )
    return tuple(assessments)


def _raw_candidate_scope_references(
    payload: object,
    *,
    surviving_candidate_indexes: Sequence[int],
    known_lane_ids: set[str],
) -> dict[int, dict[str, Any]]:
    """Read a structurally valid Qwen scope for Closure diagnostics only.

    The result is diagnostic input for Evidence Closure only. It is never a
    CandidateSemanticProfile and cannot affect competition, ranking, or Gates.
    """

    if not isinstance(payload, list):
        return {}
    raw_to_surviving = {
        raw_index: surviving_index
        for surviving_index, raw_index in enumerate(surviving_candidate_indexes)
    }
    result: dict[int, dict[str, Any]] = {}
    ambiguous_indexes: set[int] = set()
    for raw in payload:
        if not isinstance(raw, Mapping):
            continue
        raw_candidate_index = raw.get("candidate_index")
        if (
            not isinstance(raw_candidate_index, int)
            or isinstance(raw_candidate_index, bool)
            or raw_candidate_index not in raw_to_surviving
        ):
            continue
        claimed_scope = raw.get("claimed_scope")
        comparison_scope = raw.get("comparison_scope")
        if not isinstance(claimed_scope, Mapping) or not isinstance(
            comparison_scope, Mapping
        ):
            continue
        raw_claimed = claimed_scope.get("lane_ids")
        raw_comparison = comparison_scope.get("lane_ids")
        if (
            not isinstance(raw_claimed, Sequence)
            or isinstance(raw_claimed, str | bytes)
            or not isinstance(raw_comparison, Sequence)
            or isinstance(raw_comparison, str | bytes)
        ):
            continue
        claimed = tuple(
            dict.fromkeys(
                str(item).strip()
                for item in raw_claimed
                if str(item).strip() in known_lane_ids
            )
        )
        comparison = tuple(
            dict.fromkeys(
                str(item).strip()
                for item in raw_comparison
                if str(item).strip() in known_lane_ids
            )
        )
        if not claimed:
            continue
        scope_relation = str(
            claimed_scope.get(
                "scope_relation",
                CandidateScopeRelation.UNRESOLVED.value,
            )
        ).strip()
        if scope_relation not in {
            item.value for item in CandidateScopeRelation
        }:
            continue
        scope_kind = str(
            claimed_scope.get(
                "scope_kind",
                CandidateClaimedScopeKind.LANE.value,
            )
        ).strip()
        if scope_kind not in {
            item.value for item in CandidateClaimedScopeKind
        }:
            continue
        identity: dict[str, str | None] = {}
        invalid_identity = False
        for field_name in ("operation", "equipment", "chamber", "recipe"):
            raw_value = claimed_scope.get(field_name)
            if raw_value is None:
                identity[field_name] = None
            elif isinstance(raw_value, str) and raw_value.strip():
                identity[field_name] = raw_value.strip()
            else:
                invalid_identity = True
                break
        if invalid_identity:
            continue
        required_identity_fields = {
            CandidateClaimedScopeKind.LANE.value: (),
            CandidateClaimedScopeKind.RECIPE.value: (
                "operation",
                "equipment",
                "chamber",
                "recipe",
            ),
            CandidateClaimedScopeKind.CHAMBER.value: (
                "operation",
                "equipment",
                "chamber",
            ),
            CandidateClaimedScopeKind.EQUIPMENT.value: (
                "operation",
                "equipment",
            ),
            CandidateClaimedScopeKind.OPERATION.value: ("operation",),
            CandidateClaimedScopeKind.UNRESOLVED.value: (),
        }[scope_kind]
        if any(identity[field_name] is None for field_name in required_identity_fields):
            continue
        surviving_index = raw_to_surviving[raw_candidate_index]
        if surviving_index in result:
            ambiguous_indexes.add(surviving_index)
            continue
        result[surviving_index] = {
            "scope_relation": scope_relation,
            "claimed_scope_kind": scope_kind,
            "claimed_operation": identity["operation"],
            "claimed_equipment": identity["equipment"],
            "claimed_chamber": identity["chamber"],
            "claimed_recipe": identity["recipe"],
            "claimed_lane_ids": claimed,
            "comparison_lane_ids": comparison,
        }
    for index in ambiguous_indexes:
        result.pop(index, None)
    return result


def _closure_validation_error(
    assessments: Sequence[Mapping[str, Any]],
) -> str:
    fragments = [
        (
            f"candidate[{item.get('candidate_index')}] Lane "
            f"{gap.get('lane_id')} omits available {gap.get('evidence_role')} "
            f"typed Evidence {gap.get('eligible_evidence_ids')}"
        )
        for item in assessments
        for gap in item.get("closure_gaps", [])
        if isinstance(gap, Mapping)
    ]
    return "candidate Evidence closure is incomplete: " + " | ".join(fragments)


def _closure_repair_evidence_ids(
    assessments: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    must_preserve = list(
        dict.fromkeys(
            str(evidence_id)
            for item in assessments
            for evidence_id in item.get("must_preserve_evidence_ids", [])
            if str(evidence_id).strip()
        )
    )
    missing = [
        evidence_id
        for evidence_id in dict.fromkeys(
            str(evidence_id)
            for item in assessments
            for gap in item.get("closure_gaps", [])
            if isinstance(gap, Mapping)
            for evidence_id in gap.get("eligible_evidence_ids", [])
            if str(evidence_id).strip()
        )
        if evidence_id not in set(must_preserve)
    ][:_MAX_CLOSURE_REPAIR_EVIDENCE]
    return tuple([*must_preserve, *missing])


def _candidate_similarity_tokens(value: str) -> set[str]:
    """Return conservative content tokens for near-duplicate isolation."""

    ignored = {
        "abnormal",
        "abnormality",
        "cause",
        "causing",
        "control",
        "degradation",
        "drift",
        "excursion",
        "failure",
        "issue",
        "problem",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.casefold())
        if len(token) >= 2 and token not in ignored
    }


def _token_similarity(left: str, right: str) -> float:
    left_compact = re.sub(r"[^a-z0-9]+", "", left.casefold())
    right_compact = re.sub(r"[^a-z0-9]+", "", right.casefold())
    if left_compact and left_compact == right_compact:
        return 1.0
    left_tokens = _candidate_similarity_tokens(left)
    right_tokens = _candidate_similarity_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    containment = overlap / min(len(left_tokens), len(right_tokens))
    jaccard = overlap / len(left_tokens | right_tokens)
    return max(containment, jaccard)


def _root_causes_near_duplicate(left: str, right: str) -> bool:
    similarity = _token_similarity(left, right)
    return similarity >= 0.85


def _scope_values(value: object) -> set[str]:
    if isinstance(value, str):
        return {
            item.strip().casefold()
            for item in value.split(",")
            if item.strip()
        }
    if isinstance(value, list | tuple | set | frozenset):
        return {
            normalized
            for item in value
            for normalized in _scope_values(item)
        }
    return set()


def _metadata_scope_values(evidence: Evidence, *keys: str) -> set[str]:
    normalized_keys = {key.casefold() for key in keys}
    values = {
        normalized
        for key, value in evidence.metadata.items()
        if str(key).casefold() in normalized_keys
        for normalized in _scope_values(value)
    }
    values.update(
        normalized
        for entity in evidence.entities
        for key, value in entity.attributes.items()
        if str(key).casefold() in normalized_keys
        for normalized in _scope_values(value)
    )
    return values


def _entity_scope_values(evidence: Evidence, entity_type: str) -> set[str]:
    return {
        entity.entity_id.strip().casefold()
        for entity in evidence.entities
        if entity.entity_type == entity_type and entity.entity_id.strip()
    }


def _overlap_coefficient(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def _candidate_consumed_discriminator_gap_ids(
    proposal: HypothesisCandidateProposal,
    competition_context: Mapping[str, Any],
) -> set[str]:
    """Return prior discriminator Gaps whose resulting Evidence this proposal cites.

    These IDs describe investigation history.  They are not a whitelist for the
    next adversarial-challenge round; that whitelist is rebuilt later from the
    current Python-generated causal Gaps after Candidate IDs are assigned.
    """

    supporting_ids = set(proposal.supporting_evidence_ids)
    return {
        str(item.get("gap_id", "")).strip()
        for item in competition_context.get("targeted_investigation_results", [])
        if isinstance(item, Mapping)
        and str(item.get("gap_id", "")).strip()
        and supporting_ids
        & {
            str(evidence_id)
            for evidence_id in item.get("new_supporting_evidence_ids", [])
        }
    }


def _candidate_dedup_signature(
    proposal: HypothesisCandidateProposal,
    *,
    evidence_by_id: Mapping[str, Evidence],
    competition_context: Mapping[str, Any],
) -> _CandidateDedupSignature:
    supporting_evidence = [
        evidence_by_id[evidence_id]
        for evidence_id in proposal.supporting_evidence_ids
        if evidence_id in evidence_by_id
    ]
    lane_ids = {
        value
        for evidence in supporting_evidence
        for value in _metadata_scope_values(evidence, "lane_id")
    }
    recipes = {
        value
        for evidence in supporting_evidence
        for value in (
            _entity_scope_values(evidence, EntityType.RECIPE.value)
            | _metadata_scope_values(evidence, "recipe", "recipe_id")
        )
    }
    chambers = {
        value
        for evidence in supporting_evidence
        for value in (
            _entity_scope_values(evidence, EntityType.CHAMBER.value)
            | _metadata_scope_values(evidence, "chamber", "chamber_id")
        )
    }
    parameter_scope = {
        value
        for evidence in supporting_evidence
        for value in (
            _entity_scope_values(evidence, EntityType.PARAMETER.value)
            | _metadata_scope_values(
                evidence,
                "parameter",
                "parameter_name",
                "parameter_scope",
                "parameters",
            )
        )
    }
    parameter_evidence_ids = {
        evidence.evidence_id
        for evidence in supporting_evidence
        if (
            evidence.evidence_type in _PROCESS_TYPES
            and (
                _entity_scope_values(evidence, EntityType.PARAMETER.value)
                or _metadata_scope_values(
                    evidence,
                    "parameter",
                    "parameter_name",
                    "parameter_scope",
                    "parameters",
                )
            )
        )
    }
    scope_tokens = {
        token
        for value in (
            *lane_ids,
            *recipes,
            *chambers,
            *parameter_scope,
            *[
                entity.entity_id.casefold()
                for evidence in supporting_evidence
                for entity in evidence.entities
            ],
        )
        for token in re.findall(r"[a-z0-9]+", value)
    }
    mechanism_tokens = _candidate_similarity_tokens(
        f"{proposal.root_cause} {proposal.causal_explanation}"
    ) - scope_tokens - {
        "candidate",
        "chamber",
        "equipment",
        "mechanism",
        "operation",
        "process",
        "recipe",
        "root",
    }
    return _CandidateDedupSignature(
        lane_ids=tuple(sorted(lane_ids)),
        recipes=tuple(sorted(recipes)),
        chambers=tuple(sorted(chambers)),
        parameter_scope=tuple(sorted(parameter_scope)),
        parameter_evidence_ids=tuple(sorted(parameter_evidence_ids)),
        supporting_evidence_ids=tuple(sorted(proposal.supporting_evidence_ids)),
        evidence_types=tuple(
            sorted(
                {
                    str(evidence.evidence_type)
                    for evidence in supporting_evidence
                }
            )
        ),
        discriminator_gap_ids=tuple(
            sorted(
                _candidate_consumed_discriminator_gap_ids(
                    proposal,
                    competition_context,
                )
            )
        ),
        mechanism_tokens=frozenset(mechanism_tokens),
    )


def _candidate_duplicate_assessment(
    left: HypothesisCandidateProposal,
    right: HypothesisCandidateProposal,
    *,
    evidence_by_id: Mapping[str, Evidence],
    competition_context: Mapping[str, Any],
) -> _DuplicateAssessment:
    left_signature = _candidate_dedup_signature(
        left,
        evidence_by_id=evidence_by_id,
        competition_context=competition_context,
    )
    right_signature = _candidate_dedup_signature(
        right,
        evidence_by_id=evidence_by_id,
        competition_context=competition_context,
    )
    text_similarity = _token_similarity(left.root_cause, right.root_cause)
    evidence_overlap = _overlap_coefficient(
        set(left_signature.supporting_evidence_ids),
        set(right_signature.supporting_evidence_ids),
    )
    left_mechanism = set(left_signature.mechanism_tokens)
    right_mechanism = set(right_signature.mechanism_tokens)
    mechanism_similarity = (
        1.0
        if not left_mechanism and not right_mechanism
        else _overlap_coefficient(left_mechanism, right_mechanism)
    )
    duplicate_score = round(
        0.35 * text_similarity
        + 0.35 * evidence_overlap
        + 0.30 * mechanism_similarity,
        3,
    )

    differentiators = (
        ("different_lane_id", left_signature.lane_ids, right_signature.lane_ids),
        ("different_recipe", left_signature.recipes, right_signature.recipes),
        ("different_chamber", left_signature.chambers, right_signature.chambers),
        (
            "different_parameter_scope",
            left_signature.parameter_scope,
            right_signature.parameter_scope,
        ),
        (
            "different_parameter_evidence",
            left_signature.parameter_evidence_ids,
            right_signature.parameter_evidence_ids,
        ),
        (
            "different_discriminator_gap",
            left_signature.discriminator_gap_ids,
            right_signature.discriminator_gap_ids,
        ),
    )
    for reason, left_values, right_values in differentiators:
        if (left_values or right_values) and left_values != right_values:
            return _DuplicateAssessment(
                is_duplicate=False,
                duplicate_score=duplicate_score,
                reason=reason,
                left_signature=left_signature,
                right_signature=right_signature,
            )
    if mechanism_similarity < _DUPLICATE_MECHANISM_SIMILARITY_THRESHOLD:
        return _DuplicateAssessment(
            is_duplicate=False,
            duplicate_score=duplicate_score,
            reason="different_causal_mechanism",
            left_signature=left_signature,
            right_signature=right_signature,
        )
    if evidence_overlap < _DUPLICATE_EVIDENCE_OVERLAP_THRESHOLD:
        return _DuplicateAssessment(
            is_duplicate=False,
            duplicate_score=duplicate_score,
            reason="insufficient_supporting_evidence_overlap",
            left_signature=left_signature,
            right_signature=right_signature,
        )
    if not _root_causes_near_duplicate(left.root_cause, right.root_cause):
        return _DuplicateAssessment(
            is_duplicate=False,
            duplicate_score=duplicate_score,
            reason="materially_different_root_cause_text",
            left_signature=left_signature,
            right_signature=right_signature,
        )
    return _DuplicateAssessment(
        is_duplicate=True,
        duplicate_score=duplicate_score,
        reason=(
            "same_lane_identity; supporting_evidence_overlap="
            f"{evidence_overlap:.3f}; same_causal_mechanism_family; "
            f"root_cause_text_similarity={text_similarity:.3f}"
        ),
        left_signature=left_signature,
        right_signature=right_signature,
    )


def _rejected_candidate_audit(
    proposal: HypothesisCandidateProposal,
    *,
    candidate_index: int,
    compared_candidate_index: int,
    output_attempt: int,
    assessment: _DuplicateAssessment,
) -> dict[str, Any]:
    signature = assessment.left_signature
    return {
        "rejected_candidate_index": candidate_index,
        "candidate_root_cause": proposal.root_cause,
        "lane_id": signature.lane_ids[0] if len(signature.lane_ids) == 1 else None,
        "lane_ids": list(signature.lane_ids),
        "recipe": signature.recipes[0] if len(signature.recipes) == 1 else None,
        "recipes": list(signature.recipes),
        "chambers": list(signature.chambers),
        "parameter_scope": list(signature.parameter_scope),
        "evidence_ids": list(proposal.supporting_evidence_ids),
        "evidence_types": list(signature.evidence_types),
        "consumed_discriminator_gap_ids": list(signature.discriminator_gap_ids),
        "causal_mechanism_tokens": sorted(signature.mechanism_tokens),
        "duplicate_score": assessment.duplicate_score,
        "duplicate_reason": assessment.reason,
        "compared_candidate_id": f"candidate_{compared_candidate_index}",
        "output_attempt": output_attempt,
    }


def _normalized_lane_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only immutable Lane facts needed to interpret targeted Evidence."""

    return {
        key: value.get(key)
        for key in (
            "lane_id",
            "operation",
            "equipment",
            "chamber",
            "recipe",
            "parameter_scope",
            "exposed_lot_ids",
            "time_window",
            "investigation_status",
        )
        if value.get(key) not in (None, "", [], ())
    }


def _normalized_prior_challenge(
    value: Mapping[str, Any],
    *,
    evidence_by_id: Mapping[str, Evidence],
) -> dict[str, Any]:
    known_ids = set(evidence_by_id)
    return {
        "candidate_id": str(value.get("candidate_id", "")),
        "alternative_candidate_id": value.get("alternative_candidate_id"),
        "evidence_probe_lane_id": value.get(
            "evidence_probe_lane_id",
            value.get("strongest_alternative_lane_id"),
        ),
        "challenge_kind": str(value.get("challenge_kind", "lane_probe")),
        "strongest_alternative_lane_id": value.get(
            "strongest_alternative_lane_id",
            value.get("strongest_alternative"),
        ),
        "supporting_evidence_ids": [
            str(item)
            for item in value.get("supporting_evidence_ids", [])
            if str(item) in known_ids
        ],
        "contradicting_evidence_ids": [
            str(item)
            for item in value.get("contradicting_evidence_ids", [])
            if str(item) in known_ids
        ],
        "unexplained_precursor_evidence_ids": [
            str(item)
            for item in value.get("unexplained_precursor_evidence_ids", [])
            if str(item) in known_ids
        ],
        "distinguishing_gap_ids": [
            str(item) for item in value.get("distinguishing_gap_ids", [])
        ],
        "challenge_explanation": str(value.get("challenge_explanation", "")),
        "status": str(value.get("status", "")),
    }


def _evidence_matches_lane(evidence: Evidence, lane_id: str) -> bool:
    return bool(lane_id) and str(evidence.metadata.get("lane_id", "")) == lane_id


def _supports_discriminator(evidence: Evidence, discriminator_kind: str) -> bool:
    """Return whether one typed observation can answer the selected Gap kind."""

    if evidence.evidence_type in _NON_SUPPORTING_TYPES:
        return False
    if discriminator_kind == "parameter_anomaly":
        return evidence.evidence_type in _PROCESS_TYPES
    if discriminator_kind in {"exposure_commonality", "recipe_commonality"}:
        return evidence.evidence_type in _EXPOSURE_TYPES | {
            EvidenceType.RECIPE_CHANGE.value
        }
    if discriminator_kind == "product_outcome":
        return evidence.evidence_type in _PRODUCT_TYPES
    if discriminator_kind == "mechanism_context":
        return _is_approved_knowledge_support(evidence)
    if discriminator_kind == "temporal_alignment":
        return evidence.evidence_type in _PROCESS_TYPES | _EXPOSURE_TYPES
    return True


def _candidate_competition_context(
    *,
    evidence_by_id: Mapping[str, Evidence],
    new_evidence_ids: Sequence[str],
    prior_challenges: Sequence[Mapping[str, Any]],
    prior_causal_gaps: Sequence[Mapping[str, Any]],
    causal_lanes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind a prior Challenge to the Evidence collected specifically for it."""

    known_new_ids = [
        evidence_id
        for evidence_id in dict.fromkeys(str(item) for item in new_evidence_ids)
        if evidence_id in evidence_by_id
    ]
    gap_by_id = {
        str(item.get("gap_id", "")): item
        for item in prior_causal_gaps
        if str(item.get("gap_id", "")).strip()
    }
    normalized_challenges = [
        _normalized_prior_challenge(item, evidence_by_id=evidence_by_id)
        for item in prior_challenges
    ]
    targeted_results: list[dict[str, Any]] = []
    relevant_lane_ids: set[str] = set()
    for challenge in normalized_challenges:
        lane_id = str(challenge.get("strongest_alternative_lane_id") or "")
        if lane_id:
            relevant_lane_ids.add(lane_id)
        for gap_id in challenge.get("distinguishing_gap_ids", []):
            gap = gap_by_id.get(str(gap_id), {})
            raw_scope = gap.get("target_scope", {})
            target_scope = dict(raw_scope) if isinstance(raw_scope, Mapping) else {}
            gap_lane_id = str(target_scope.get("lane_id", "") or lane_id)
            discriminator_kind = str(gap.get("discriminator_kind", ""))
            if gap_lane_id:
                relevant_lane_ids.add(gap_lane_id)
            scoped_new_ids = [
                evidence_id
                for evidence_id in known_new_ids
                if _evidence_matches_lane(evidence_by_id[evidence_id], gap_lane_id)
            ]
            supporting_ids = [
                evidence_id
                for evidence_id in scoped_new_ids
                if _supports_discriminator(
                    evidence_by_id[evidence_id], discriminator_kind
                )
            ]
            targeted_results.append(
                {
                    "gap_id": str(gap_id),
                    "discriminator_kind": discriminator_kind,
                    "lane_id": gap_lane_id,
                    "target_scope": target_scope,
                    "new_evidence_ids": scoped_new_ids,
                    "new_supporting_evidence_ids": supporting_ids,
                    "new_data_missing_evidence_ids": [
                        evidence_id
                        for evidence_id in scoped_new_ids
                        if evidence_by_id[evidence_id].evidence_type
                        == EvidenceType.DATA_MISSING.value
                    ],
                    "new_evidence": [
                        compact_evidence_record(evidence_by_id[evidence_id])
                        for evidence_id in scoped_new_ids
                    ],
                    "answered": bool(scoped_new_ids),
                    "support_observed": bool(supporting_ids),
                }
            )
    relevant_lanes = [
        _normalized_lane_payload(item)
        for item in causal_lanes
        if str(item.get("lane_id", "")) in relevant_lane_ids
    ]
    targeted_supporting_ids = list(
        dict.fromkeys(
            evidence_id
            for item in targeted_results
            for evidence_id in item["new_supporting_evidence_ids"]
        )
    )
    return {
        "new_evidence_ids_since_prior": known_new_ids,
        "prior_candidate_challenges": normalized_challenges,
        "targeted_investigation_results": targeted_results,
        "relevant_causal_lanes": relevant_lanes,
        "targeted_supporting_evidence_ids": targeted_supporting_ids,
        "requires_distinct_candidate_review": bool(targeted_supporting_ids),
    }


def _competition_repair_required(
    proposals: Sequence[HypothesisCandidateProposal],
    *,
    prior_candidates: Sequence[Mapping[str, Any]],
    competition_context: Mapping[str, Any],
) -> bool:
    targeted_ids = {
        str(item)
        for item in competition_context.get(
            "targeted_supporting_evidence_ids", []
        )
    }
    prior_roots = {
        str(candidate.get("root_cause", "")).strip()
        for candidate in prior_candidates
        if str(candidate.get("root_cause", "")).strip()
    }
    if not targeted_ids or not prior_roots:
        return False
    targeted_proposals = [
        proposal
        for proposal in proposals
        if targeted_ids & set(proposal.supporting_evidence_ids)
    ]
    if len(proposals) < 2 or not targeted_proposals:
        return True
    # Pairwise Lane-aware Dedup already proved that the surviving proposals
    # differ by causal scope, supporting Evidence, or mechanism.  Requiring two
    # proposals here prevents a broadened rewrite from silently replacing the
    # prior candidate after targeted alternative Evidence was collected.
    return False


def _competition_bundle_evidence_ids(bundle: Mapping[str, Any]) -> set[str]:
    return {
        str(item)
        for field in (
            "shared_exposure_evidence_ids",
            "process_evidence_ids",
        )
        for item in bundle.get(field, [])
        if str(item)
    }


def _candidate_competition_profile(
    proposal: HypothesisCandidateProposal,
    *,
    candidate_index: int,
    semantic_profile: CandidateSemanticProfile | None = None,
    evidence_by_id: Mapping[str, Evidence],
    competition_context: Mapping[str, Any],
    competition_brief: Mapping[str, Any],
) -> dict[str, Any]:
    signature = _candidate_dedup_signature(
        proposal,
        evidence_by_id=evidence_by_id,
        competition_context=competition_context,
    )
    supporting_ids = set(proposal.supporting_evidence_ids)
    direction_bundle_ids = [
        str(bundle.get("bundle_id", ""))
        for bundle in competition_brief.get("direction_bundles", [])
        if isinstance(bundle, Mapping)
        and str(bundle.get("bundle_id", ""))
        and supporting_ids & _competition_bundle_evidence_ids(bundle)
    ]
    return {
        "candidate_index": candidate_index,
        "root_cause": proposal.root_cause,
        "direction_bundle_ids": direction_bundle_ids,
        "lane_ids": list(signature.lane_ids),
        "recipes": list(signature.recipes),
        "parameter_scope": list(signature.parameter_scope),
        "consumed_discriminator_gap_ids": list(signature.discriminator_gap_ids),
        "evidence_coverage": {
            "lane_ids": list(signature.lane_ids),
            "recipes": list(signature.recipes),
            "chambers": list(signature.chambers),
            "parameter_scope": list(signature.parameter_scope),
            "supporting_evidence_ids": list(signature.supporting_evidence_ids),
        },
        "semantic_profile": (
            semantic_profile.to_dict() if semantic_profile is not None else None
        ),
        "scope_relation": (
            semantic_profile.scope_relation
            if semantic_profile is not None
            else CandidateScopeRelation.UNRESOLVED.value
        ),
        "claimed_scope_kind": (
            semantic_profile.claimed_scope_kind
            if semantic_profile is not None
            else CandidateClaimedScopeKind.UNRESOLVED.value
        ),
        "claimed_scope_identity": (
            {
                "operation": semantic_profile.claimed_operation,
                "equipment": semantic_profile.claimed_equipment,
                "chamber": semantic_profile.claimed_chamber,
                "recipe": semantic_profile.claimed_recipe,
            }
            if semantic_profile is not None
            else None
        ),
        "primary_mechanism": (
            semantic_profile.primary_mechanism
            if semantic_profile is not None
            else None
        ),
        "effect_modifier": (
            semantic_profile.effect_modifier
            if semantic_profile is not None
            else None
        ),
        "depends_on_candidate_id": (
            semantic_profile.depends_on_candidate_id
            if semantic_profile is not None
            else None
        ),
        "mechanism_relation": (
            semantic_profile.mechanism_relation
            if semantic_profile is not None
            else CandidateMechanismRelation.UNKNOWN.value
        ),
        "mechanism_tokens": sorted(signature.mechanism_tokens),
    }


def _prediction_semantic_signature(
    profile: CandidateSemanticProfile,
) -> tuple[
    tuple[
        str,
        tuple[str, ...],
        tuple[tuple[str, str, str], ...],
        str,
    ],
    ...,
]:
    return tuple(
        sorted(
            (
                prediction.discriminator_kind,
                tuple(sorted(prediction.lane_ids)),
                tuple(
                    sorted(
                        (
                            expectation.lane_id,
                            expectation.normalized_effect_key,
                            expectation.effect_state,
                        )
                        for expectation in prediction.lane_effect_expectations
                    )
                ),
                re.sub(r"\s+", " ", prediction.prediction.casefold()).strip(),
            )
            for prediction in profile.distinguishing_predictions
        )
    )


def _scope_semantics_are_distinct(
    left: CandidateSemanticProfile,
    right: CandidateSemanticProfile,
) -> bool:
    """Require a different causal claim and a falsifiable prediction.

    Comparison scope may be identical or broader than claimed scope. It is not
    itself evidence that two hypotheses claim the same causal reach. A genuinely
    different mechanism may also preserve competition within one observed
    direction, but Python never treats that mechanism text as proof.
    """

    claimed_scope_distinct = (
        left.scope_relation != right.scope_relation
        or left.claimed_scope_kind != right.claimed_scope_kind
        or left.claimed_operation != right.claimed_operation
        or left.claimed_equipment != right.claimed_equipment
        or left.claimed_chamber != right.claimed_chamber
        or left.claimed_recipe != right.claimed_recipe
        or set(left.claimed_lane_ids) != set(right.claimed_lane_ids)
    )
    predictions_distinct = (
        bool(left.distinguishing_predictions)
        and bool(right.distinguishing_predictions)
        and _prediction_semantic_signature(left)
        != _prediction_semantic_signature(right)
    )
    return claimed_scope_distinct and predictions_distinct


def _mechanism_semantics_are_distinct(
    left: CandidateSemanticProfile,
    right: CandidateSemanticProfile,
) -> bool:
    """Require an independent primary initiator, not a Scope/effect modifier."""

    if (
        left.mechanism_relation
        not in {
            CandidateMechanismRelation.REFERENCE.value,
            CandidateMechanismRelation.INDEPENDENT_ALTERNATIVE.value,
        }
        or right.mechanism_relation
        != CandidateMechanismRelation.INDEPENDENT_ALTERNATIVE.value
        or left.depends_on_candidate_id is not None
        or right.depends_on_candidate_id is not None
    ):
        return False
    mechanism_distinct = (
        _token_similarity(left.primary_mechanism, right.primary_mechanism)
        < _DUPLICATE_MECHANISM_SIMILARITY_THRESHOLD
    )
    predictions_distinct = (
        bool(left.distinguishing_predictions)
        and bool(right.distinguishing_predictions)
        and _prediction_semantic_signature(left)
        != _prediction_semantic_signature(right)
    )
    return mechanism_distinct and predictions_distinct


def _candidate_competition_assessment(
    proposals: Sequence[HypothesisCandidateProposal],
    *,
    evidence_by_id: Mapping[str, Evidence],
    competition_context: Mapping[str, Any],
    competition_brief: Mapping[str, Any],
    semantic_profiles: Sequence[CandidateSemanticProfile] = (),
) -> dict[str, Any]:
    requirement = str(
        competition_brief.get(
            "competition_requirement",
            CompetitionRequirement.NOT_REQUIRED.value,
        )
    )
    competition_type = str(
        competition_brief.get(
            "competition_type",
            CandidateCompetitionType.NONE.value,
        )
    )
    semantic_profiles_by_index: dict[int, CandidateSemanticProfile] = {}
    for profile in semantic_profiles:
        try:
            candidate_number = int(profile.candidate_id.rsplit(":llm:", 1)[1])
        except (IndexError, ValueError):
            continue
        semantic_profiles_by_index[candidate_number - 1] = profile
    profiles = [
        _candidate_competition_profile(
            proposal,
            candidate_index=index,
            semantic_profile=semantic_profiles_by_index.get(index),
            evidence_by_id=evidence_by_id,
            competition_context=competition_context,
            competition_brief=competition_brief,
        )
        for index, proposal in enumerate(proposals)
    ]
    represented_direction_ids = sorted(
        {
            str(bundle_id)
            for profile in profiles
            for bundle_id in profile["direction_bundle_ids"]
            if str(bundle_id)
        }
    )
    semantic_profiles_complete = set(semantic_profiles_by_index) == set(
        range(len(proposals))
    )
    scope_competition_represented = False
    mechanism_competition_represented = False
    pairwise_mechanism_relations: list[dict[str, Any]] = []
    if semantic_profiles_complete:
        for left_index, left in enumerate(profiles):
            left_semantics = semantic_profiles_by_index[left_index]
            for right_index in range(left_index + 1, len(profiles)):
                right = profiles[right_index]
                if not set(left["direction_bundle_ids"]) & set(
                    right["direction_bundle_ids"]
                ):
                    continue
                if _scope_semantics_are_distinct(
                    left_semantics,
                    semantic_profiles_by_index[right_index],
                ):
                    scope_competition_represented = True
                mechanism_distinct = _mechanism_semantics_are_distinct(
                    left_semantics,
                    semantic_profiles_by_index[right_index],
                )
                right_semantics = semantic_profiles_by_index[right_index]
                pairwise_mechanism_relations.append(
                    {
                        "left_candidate_index": left_index,
                        "right_candidate_index": right_index,
                        "declared_relation": right_semantics.mechanism_relation,
                        "depends_on_candidate_id": (
                            right_semantics.depends_on_candidate_id
                        ),
                        "left_primary_mechanism": (
                            left_semantics.primary_mechanism
                        ),
                        "right_primary_mechanism": (
                            right_semantics.primary_mechanism
                        ),
                        "independent_root_competition": mechanism_distinct,
                    }
                )
                if mechanism_distinct:
                    mechanism_competition_represented = True
                if scope_competition_represented and mechanism_competition_represented:
                    break
            if scope_competition_represented:
                if mechanism_competition_represented:
                    break

    # This function assesses whether a valid Candidate set has formed the
    # Python-required competition.  Failure to form that competition is an
    # unresolved investigation gap, not a Candidate-processing failure.
    # Structural/provider failures are assigned by the caller at the boundary
    # where all Candidate output or the adversarial challenge is invalid.
    gap_reason: str | None = None
    if requirement == CompetitionRequirement.NOT_REQUIRED.value:
        status = CandidateCompetitionStatus.NOT_REQUIRED.value
    elif requirement == CompetitionRequirement.ALTERNATIVE_DISCOVERY_REQUIRED.value:
        status = CandidateCompetitionStatus.PENDING.value
    elif requirement in {
        CompetitionRequirement.DIRECTION_REQUIRED.value,
        CompetitionRequirement.MIXED_REQUIRED.value,
    }:
        if len(profiles) >= 2 and len(represented_direction_ids) >= 2:
            status = CandidateCompetitionStatus.ACTIVE.value
        else:
            status = CandidateCompetitionStatus.PENDING.value
            gap_reason = (
                CompetitionGapReason.ALTERNATIVE_DIRECTION_NOT_GENERATED.value
            )
    elif requirement == CompetitionRequirement.MECHANISM_REQUIRED.value:
        if len(profiles) < 2 or not semantic_profiles_complete:
            status = CandidateCompetitionStatus.PENDING.value
            gap_reason = (
                CompetitionGapReason.MECHANISM_ALTERNATIVE_NOT_GENERATED.value
            )
        elif mechanism_competition_represented:
            status = CandidateCompetitionStatus.ACTIVE.value
        else:
            status = CandidateCompetitionStatus.PENDING.value
            gap_reason = (
                CompetitionGapReason.MECHANISM_ALTERNATIVE_NOT_GENERATED.value
            )
    elif requirement == CompetitionRequirement.SCOPE_REQUIRED.value:
        if len(profiles) < 2:
            status = CandidateCompetitionStatus.PENDING.value
            gap_reason = CompetitionGapReason.SCOPE_HYPOTHESIS_COLLAPSED.value
        elif not semantic_profiles_complete:
            # Two valid Candidates remain available. Missing semantic metadata
            # is isolated from Candidate validity and must never trigger a
            # Candidate Generator repair or an orchestration fallback.
            status = CandidateCompetitionStatus.PENDING.value
        elif scope_competition_represented:
            status = CandidateCompetitionStatus.ACTIVE.value
        else:
            status = CandidateCompetitionStatus.PENDING.value
            gap_reason = CompetitionGapReason.SCOPE_HYPOTHESIS_COLLAPSED.value
    else:
        status = CandidateCompetitionStatus.NOT_EVALUATED.value
    return {
        "competition_requirement": requirement,
        "competition_type": competition_type,
        "competition_axes": list(competition_brief.get("competition_axes", [])),
        "competition_status": status,
        "competition_failure_reason": None,
        "competition_gap_reason": gap_reason,
        "required_candidate_count": int(
            competition_brief.get("required_candidate_count", 1)
        ),
        "represented_direction_bundle_ids": represented_direction_ids,
        "scope_competition_represented": scope_competition_represented,
        "mechanism_competition_represented": mechanism_competition_represented,
        "pairwise_mechanism_relations": pairwise_mechanism_relations,
        "scope_assessment_required": bool(
            competition_brief.get("scope_assessment_required")
        ),
        "scope_assessment_status": competition_brief.get(
            "scope_assessment_status",
            "not_required",
        ),
        "semantic_profiles_complete": semantic_profiles_complete,
        "candidate_profiles": profiles,
    }


def _partition_root_candidates(
    proposals: Sequence[HypothesisCandidateProposal],
    semantic_profiles: Sequence[CandidateSemanticProfile],
    *,
    competition_requirement: str,
    candidate_ids: Sequence[str] = (),
    semantic_validation_errors: Sequence[str] = (),
) -> tuple[
    tuple[HypothesisCandidateProposal, ...],
    tuple[CandidateSemanticProfile, ...],
    tuple[dict[str, Any], ...],
]:
    """Keep Scope/modifier explanations out of Root Cause Candidate slots.

    Qwen still authors and audits these explanations. Python only enforces the
    declared semantic relation: a Candidate that depends on the reference
    mechanism is a Scope/effect assessment, not an independent root cause.
    Legacy ``scope_required`` State keeps its historical two-Candidate shape.
    """

    if (
        len(proposals) < 2
        or competition_requirement
        not in {
            CompetitionRequirement.MECHANISM_REQUIRED.value,
            CompetitionRequirement.MIXED_REQUIRED.value,
        }
    ):
        return tuple(proposals), tuple(semantic_profiles), ()

    normalized_candidate_ids = tuple(str(item).strip() for item in candidate_ids)
    profile_by_candidate_id = {
        profile.candidate_id: profile for profile in semantic_profiles
    }

    def profile_for_index(index: int) -> CandidateSemanticProfile | None:
        if len(normalized_candidate_ids) == len(proposals):
            return profile_by_candidate_id.get(normalized_candidate_ids[index])
        expected_suffix = f":llm:{index + 1}"
        suffix_matches = [
            profile
            for profile in semantic_profiles
            if profile.candidate_id.endswith(expected_suffix)
        ]
        if len(suffix_matches) == 1:
            return suffix_matches[0]
        if len(semantic_profiles) == len(proposals):
            return semantic_profiles[index]
        return None

    def profile_status(index: int) -> str:
        indexed_marker = f"candidate_semantic_profiles[{index}]"
        if any(indexed_marker in error for error in semantic_validation_errors):
            return "invalid"
        return "missing"

    reference_profile = profile_for_index(0)
    kept_proposals: list[HypothesisCandidateProposal] = []
    kept_profiles: list[CandidateSemanticProfile] = []
    scope_variants: list[dict[str, Any]] = []
    for index, proposal in enumerate(proposals):
        profile = profile_for_index(index)
        independent_primary_mechanism = bool(
            index > 0
            and reference_profile is not None
            and profile is not None
            and reference_profile.mechanism_relation
            == CandidateMechanismRelation.REFERENCE.value
            and profile.mechanism_relation
            == CandidateMechanismRelation.INDEPENDENT_ALTERNATIVE.value
            and profile.depends_on_candidate_id is None
            and _mechanism_semantics_are_distinct(reference_profile, profile)
        )
        is_root_candidate = index == 0 or independent_primary_mechanism
        if is_root_candidate:
            kept_proposals.append(proposal)
            if profile is not None:
                kept_profiles.append(profile)
            continue
        semantic_profile_status = (
            "valid_non_root" if profile is not None else profile_status(index)
        )
        mechanism_relation = (
            profile.mechanism_relation
            if profile is not None
            else CandidateMechanismRelation.UNKNOWN.value
        )
        depends_on_candidate_id = (
            profile.depends_on_candidate_id if profile is not None else None
        )
        scope_variants.append(
            {
                "candidate_index": index,
                "candidate_id": (
                    normalized_candidate_ids[index]
                    if len(normalized_candidate_ids) == len(proposals)
                    else None
                ),
                "root_cause": proposal.root_cause,
                "causal_explanation": proposal.causal_explanation,
                "supporting_evidence_ids": list(proposal.supporting_evidence_ids),
                "contradicting_evidence_ids": list(
                    proposal.contradicting_evidence_ids
                ),
                "semantic_profile": profile.to_dict() if profile is not None else None,
                "semantic_profile_status": semantic_profile_status,
                "semantic_validation_errors": list(semantic_validation_errors),
                "mechanism_relation": mechanism_relation,
                "depends_on_candidate_id": depends_on_candidate_id,
                "isolation_reason": (
                    "candidate has no valid semantic proof of an independent "
                    "primary mechanism relative to the reference candidate"
                ),
            }
        )
    return (
        tuple(kept_proposals),
        tuple(kept_profiles),
        tuple(scope_variants),
    )


def _prior_proposals(
    prior_candidates: Sequence[Mapping[str, Any]],
    *,
    evidence_by_id: Mapping[str, Evidence],
) -> list[HypothesisCandidateProposal]:
    proposals: list[HypothesisCandidateProposal] = []
    for candidate in prior_candidates[:_MAX_CANDIDATES]:
        supporting = tuple(
            dict.fromkeys(
                str(item)
                for item in candidate.get("supporting_evidence_ids", [])
                if str(item) in evidence_by_id
            )
        )
        contradicting = tuple(
            dict.fromkeys(
                str(item)
                for item in candidate.get("contradicting_evidence_ids", [])
                if str(item) in evidence_by_id and str(item) not in supporting
            )
        )
        try:
            proposals.append(
                HypothesisCandidateProposal(
                    root_cause=str(candidate.get("root_cause", "")),
                    causal_explanation=str(
                        candidate.get(
                            "causal_explanation",
                            candidate.get("root_cause", ""),
                        )
                    ),
                    supporting_evidence_ids=supporting,
                    contradicting_evidence_ids=contradicting,
                )
            )
        except (TypeError, ValueError, ModelValidationError):
            continue
    return proposals


def _candidate_lineage(
    proposals: Sequence[HypothesisCandidateProposal],
    *,
    prior_candidates: Sequence[Mapping[str, Any]],
    semantic_profiles: Sequence[CandidateSemanticProfile] = (),
    prior_semantic_profiles: Sequence[Mapping[str, Any]] = (),
    evidence_by_id: Mapping[str, Evidence],
    competition_context: Mapping[str, Any],
    competition_brief: Mapping[str, Any],
) -> list[dict[str, Any]]:
    priors = _prior_proposals(prior_candidates, evidence_by_id=evidence_by_id)
    if not priors:
        return [
            {
                "candidate_index": index,
                "prior_candidate_index": None,
                "lineage_status": "new",
                "root_cause": proposal.root_cause,
            }
            for index, proposal in enumerate(proposals)
        ]
    current_semantics_by_index: dict[int, CandidateSemanticProfile] = {}
    for semantic_profile in semantic_profiles:
        try:
            candidate_number = int(
                semantic_profile.candidate_id.rsplit(":llm:", 1)[1]
            )
        except (IndexError, ValueError):
            continue
        current_semantics_by_index[candidate_number - 1] = semantic_profile
    prior_semantics_by_candidate_id: dict[str, CandidateSemanticProfile] = {}
    for raw_profile in prior_semantic_profiles:
        try:
            semantic_profile = CandidateSemanticProfile.from_dict(dict(raw_profile))
        except (TypeError, ValueError, ModelValidationError):
            continue
        prior_semantics_by_candidate_id[semantic_profile.candidate_id] = (
            semantic_profile
        )
    prior_semantics_by_index = {
        index: prior_semantics_by_candidate_id[candidate_id]
        for index, candidate in enumerate(prior_candidates[:_MAX_CANDIDATES])
        if (
            candidate_id := str(candidate.get("candidate_id", "")).strip()
        ) in prior_semantics_by_candidate_id
    }
    current_profiles = [
        _candidate_competition_profile(
            proposal,
            candidate_index=index,
            semantic_profile=current_semantics_by_index.get(index),
            evidence_by_id=evidence_by_id,
            competition_context=competition_context,
            competition_brief=competition_brief,
        )
        for index, proposal in enumerate(proposals)
    ]
    prior_profiles = [
        _candidate_competition_profile(
            proposal,
            candidate_index=index,
            semantic_profile=prior_semantics_by_index.get(index),
            evidence_by_id=evidence_by_id,
            competition_context=competition_context,
            competition_brief=competition_brief,
        )
        for index, proposal in enumerate(priors)
    ]
    lineage: list[dict[str, Any]] = []
    matched_prior_indexes: set[int] = set()
    for index, (proposal, profile) in enumerate(zip(proposals, current_profiles, strict=True)):
        match_index: int | None = None
        for prior_index, prior_profile in enumerate(prior_profiles):
            if set(profile["direction_bundle_ids"]) & set(
                prior_profile["direction_bundle_ids"]
            ):
                match_index = prior_index
                break
            if _root_causes_near_duplicate(
                proposal.root_cause,
                priors[prior_index].root_cause,
            ):
                match_index = prior_index
                break
        if match_index is None:
            status = "new_direction"
        else:
            matched_prior_indexes.add(match_index)
            prior_profile = prior_profiles[match_index]
            current_semantics = current_semantics_by_index.get(index)
            prior_semantics = prior_semantics_by_index.get(match_index)
            if (
                current_semantics is not None
                and prior_semantics is not None
                and set(current_semantics.claimed_lane_ids)
                > set(prior_semantics.claimed_lane_ids)
                and set(profile["direction_bundle_ids"])
                == set(prior_profile["direction_bundle_ids"])
            ):
                status = "scope_expanded"
            elif _root_causes_near_duplicate(
                proposal.root_cause,
                priors[match_index].root_cause,
            ):
                status = "retained"
            else:
                status = "revised"
        lineage.append(
            {
                "candidate_index": index,
                "prior_candidate_index": match_index,
                "lineage_status": status,
                "root_cause": proposal.root_cause,
                "prior_root_cause": (
                    priors[match_index].root_cause
                    if match_index is not None
                    else None
                ),
                "direction_bundle_ids": list(profile["direction_bundle_ids"]),
                "recipes": list(profile["recipes"]),
                "claimed_lane_ids": (
                    list(current_semantics_by_index[index].claimed_lane_ids)
                    if index in current_semantics_by_index
                    else []
                ),
            }
        )
    lineage.extend(
        {
            "candidate_index": None,
            "prior_candidate_index": prior_index,
            "lineage_status": "omitted",
            "root_cause": None,
            "prior_root_cause": prior.root_cause,
            "direction_bundle_ids": list(
                prior_profiles[prior_index]["direction_bundle_ids"]
            ),
            "recipes": list(prior_profiles[prior_index]["recipes"]),
            "claimed_lane_ids": (
                list(prior_semantics_by_index[prior_index].claimed_lane_ids)
                if prior_index in prior_semantics_by_index
                else []
            ),
        }
        for prior_index, prior in enumerate(priors)
        if prior_index not in matched_prior_indexes
    )
    return lineage


def _parse_candidate(
    payload: object,
    *,
    index: int,
    evidence_by_id: dict[str, Evidence],
) -> HypothesisCandidateProposal:
    if not isinstance(payload, dict):
        raise LLMOutputValidationError(f"candidates[{index}] must be an object")
    expected = {
        "root_cause",
        "causal_explanation",
        "supporting_evidence_ids",
        "contradicting_evidence_ids",
    }
    if set(payload) != expected:
        raise LLMOutputValidationError(
            f"candidates[{index}] must contain exactly {sorted(expected)}"
        )
    if not isinstance(payload.get("root_cause"), str) or not isinstance(
        payload.get("causal_explanation"), str
    ):
        raise LLMOutputValidationError(
            f"candidates[{index}] root_cause and causal_explanation must be strings"
        )
    supporting = payload.get("supporting_evidence_ids")
    contradicting = payload.get("contradicting_evidence_ids")
    if not isinstance(supporting, list) or not isinstance(contradicting, list):
        raise LLMOutputValidationError(
            f"candidates[{index}] Evidence IDs must be arrays"
        )
    try:
        proposal = HypothesisCandidateProposal(
            root_cause=payload["root_cause"].strip(),
            causal_explanation=payload["causal_explanation"].strip(),
            supporting_evidence_ids=tuple(supporting),
            contradicting_evidence_ids=tuple(contradicting),
        )
    except ModelValidationError as exc:
        raise LLMOutputValidationError(str(exc)) from exc
    referenced = set(proposal.supporting_evidence_ids) | set(
        proposal.contradicting_evidence_ids
    )
    unknown = sorted(referenced - set(evidence_by_id))
    if unknown:
        raise LLMOutputValidationError(
            f"candidates[{index}] references unknown Evidence IDs: {unknown}"
        )
    invalid_support = sorted(
        evidence_id
        for evidence_id in proposal.supporting_evidence_ids
        if (
            evidence_by_id[evidence_id].evidence_type in _NON_SUPPORTING_TYPES
            or (
                evidence_by_id[evidence_id].evidence_type
                in _KNOWLEDGE_MECHANISM_TYPES
                and not _is_approved_knowledge_support(evidence_by_id[evidence_id])
            )
        )
    )
    if invalid_support:
        raise LLMOutputValidationError(
            f"candidates[{index}] uses non-supporting Evidence as support: "
            f"{invalid_support}"
        )
    return proposal


def _parse_candidate_semantic_profiles(
    payload: object,
    *,
    request_id: str,
    surviving_candidate_indexes: Sequence[int],
    known_lane_ids: set[str],
    known_lane_contexts: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[tuple[CandidateSemanticProfile, ...], tuple[str, ...]]:
    """Parse Qwen scope meaning without changing Candidate validity.

    Profiles are deliberately isolated from the four-field Candidate contract.
    A malformed or missing profile blocks semantic competition assessment but
    never rejects an otherwise valid Evidence-bounded Candidate.
    """

    if payload is None:
        return (), ("candidate_semantic_profiles is missing",)
    if not isinstance(payload, list):
        return (), ("candidate_semantic_profiles must be an array",)
    raw_to_surviving = {
        raw_index: surviving_index
        for surviving_index, raw_index in enumerate(surviving_candidate_indexes)
    }
    profiles_by_surviving_index: dict[int, CandidateSemanticProfile] = {}
    errors: list[str] = []
    expected_profile_fields = {
        "candidate_index",
        "claimed_scope",
        "comparison_scope",
        "mechanism_claim",
        "primary_mechanism",
        "effect_modifier",
        "depends_on_candidate_index",
        "mechanism_relation",
        "distinguishing_predictions",
    }
    legacy_scope_fields = {"scope_relation", "lane_ids"}
    expected_scope_fields = {
        "scope_relation",
        "scope_kind",
        "operation",
        "equipment",
        "chamber",
        "recipe",
        "lane_ids",
    }
    expected_comparison_fields = {"lane_ids"}
    required_prediction_fields = {
        "discriminator_kind",
        "lane_ids",
        "prediction",
    }
    product_outcome_prediction_fields = {
        *required_prediction_fields,
        "lane_effect_expectations",
    }
    expected_effect_expectation_fields = {
        "lane_id",
        "effect_key",
        "effect_state",
    }
    for profile_index, raw in enumerate(payload):
        try:
            if not isinstance(raw, dict):
                raise LLMOutputValidationError(
                    f"candidate_semantic_profiles[{profile_index}] must be an object"
                )
            if set(raw) != expected_profile_fields:
                raise LLMOutputValidationError(
                    f"candidate_semantic_profiles[{profile_index}] must contain "
                    f"exactly {sorted(expected_profile_fields)}"
                )
            raw_candidate_index = raw.get("candidate_index")
            if not isinstance(raw_candidate_index, int) or isinstance(
                raw_candidate_index, bool
            ):
                raise LLMOutputValidationError(
                    f"candidate_semantic_profiles[{profile_index}].candidate_index "
                    "must be an integer"
                )
            if raw_candidate_index not in raw_to_surviving:
                raise LLMOutputValidationError(
                    f"candidate_semantic_profiles[{profile_index}] references an "
                    "unknown or isolated candidate_index"
                )
            surviving_index = raw_to_surviving[raw_candidate_index]
            if surviving_index in profiles_by_surviving_index:
                raise LLMOutputValidationError(
                    f"candidate_semantic_profiles contains duplicate candidate_index "
                    f"{raw_candidate_index}"
                )
            claimed_scope = raw.get("claimed_scope")
            comparison_scope = raw.get("comparison_scope")
            if not isinstance(claimed_scope, dict) or frozenset(claimed_scope) not in {
                frozenset(legacy_scope_fields),
                frozenset(expected_scope_fields),
            }:
                raise LLMOutputValidationError(
                    f"candidate_semantic_profiles[{profile_index}].claimed_scope "
                    "must use either the legacy Lane-only fields "
                    f"{sorted(legacy_scope_fields)} or exactly "
                    f"{sorted(expected_scope_fields)}"
                )
            if not isinstance(comparison_scope, dict) or set(
                comparison_scope
            ) != expected_comparison_fields:
                raise LLMOutputValidationError(
                    f"candidate_semantic_profiles[{profile_index}].comparison_scope "
                    f"must contain exactly {sorted(expected_comparison_fields)}"
                )
            raw_predictions = raw.get("distinguishing_predictions")
            if not isinstance(raw_predictions, list):
                raise LLMOutputValidationError(
                    f"candidate_semantic_profiles[{profile_index}]."
                    "distinguishing_predictions must be an array"
                )
            predictions: list[CandidateDistinguishingPrediction] = []
            for prediction_index, prediction in enumerate(raw_predictions):
                if not isinstance(prediction, dict):
                    raise LLMOutputValidationError(
                        f"candidate_semantic_profiles[{profile_index}]."
                        f"distinguishing_predictions[{prediction_index}] must "
                        "be an object"
                    )
                discriminator_kind = prediction.get("discriminator_kind")
                if not isinstance(discriminator_kind, str):
                    raise LLMOutputValidationError(
                        f"candidate_semantic_profiles[{profile_index}]."
                        f"distinguishing_predictions[{prediction_index}]."
                        "discriminator_kind must be a string"
                    )
                actual_prediction_fields = set(prediction)
                allowed_prediction_fields = (
                    {frozenset(product_outcome_prediction_fields)}
                    if discriminator_kind == "product_outcome"
                    else {
                        frozenset(required_prediction_fields),
                        frozenset(product_outcome_prediction_fields),
                    }
                )
                if frozenset(actual_prediction_fields) not in allowed_prediction_fields:
                    raise LLMOutputValidationError(
                        f"candidate_semantic_profiles[{profile_index}]."
                        f"distinguishing_predictions[{prediction_index}] must "
                        "contain the required prediction fields and only the "
                        "optional lane_effect_expectations field"
                    )
                raw_effect_expectations = prediction.get(
                    "lane_effect_expectations",
                    [],
                )
                if not isinstance(raw_effect_expectations, list):
                    raise LLMOutputValidationError(
                        f"candidate_semantic_profiles[{profile_index}]."
                        f"distinguishing_predictions[{prediction_index}]."
                        "lane_effect_expectations must be an array"
                    )
                if (
                    discriminator_kind != "product_outcome"
                    and raw_effect_expectations
                ):
                    raise LLMOutputValidationError(
                        f"candidate_semantic_profiles[{profile_index}]."
                        f"distinguishing_predictions[{prediction_index}] non-"
                        "product predictions cannot declare Lane effects"
                    )
                lane_effect_expectations: list[
                    CandidateLaneEffectExpectation
                ] = []
                for expectation_index, expectation in enumerate(
                    raw_effect_expectations
                ):
                    if not isinstance(expectation, dict) or set(
                        expectation
                    ) != expected_effect_expectation_fields:
                        raise LLMOutputValidationError(
                            f"candidate_semantic_profiles[{profile_index}]."
                            f"distinguishing_predictions[{prediction_index}]."
                            "lane_effect_expectations"
                            f"[{expectation_index}] must contain exactly "
                            f"{sorted(expected_effect_expectation_fields)}"
                        )
                    lane_effect_expectations.append(
                        CandidateLaneEffectExpectation(
                            lane_id=expectation["lane_id"],
                            effect_key=expectation["effect_key"],
                            effect_state=expectation["effect_state"],
                        )
                    )
                predictions.append(
                    CandidateDistinguishingPrediction(
                        discriminator_kind=discriminator_kind,
                        lane_ids=tuple(prediction.get("lane_ids", [])),
                        prediction=prediction["prediction"],
                        lane_effect_expectations=tuple(
                            lane_effect_expectations
                        ),
                    )
                )
            raw_dependency_index = raw.get("depends_on_candidate_index")
            if raw_dependency_index is not None and (
                not isinstance(raw_dependency_index, int)
                or isinstance(raw_dependency_index, bool)
                or raw_dependency_index not in raw_to_surviving
                or raw_dependency_index == raw_candidate_index
            ):
                raise LLMOutputValidationError(
                    f"candidate_semantic_profiles[{profile_index}]."
                    "depends_on_candidate_index must be null or reference a "
                    "different surviving candidate"
                )
            depends_on_candidate_id = (
                f"{request_id}:llm:{raw_to_surviving[raw_dependency_index] + 1}"
                if raw_dependency_index is not None
                else None
            )
            profile = CandidateSemanticProfile(
                candidate_id=f"{request_id}:llm:{surviving_index + 1}",
                scope_relation=claimed_scope["scope_relation"],
                claimed_scope_kind=claimed_scope.get(
                    "scope_kind",
                    CandidateClaimedScopeKind.LANE.value,
                ),
                claimed_operation=claimed_scope.get("operation"),
                claimed_equipment=claimed_scope.get("equipment"),
                claimed_chamber=claimed_scope.get("chamber"),
                claimed_recipe=claimed_scope.get("recipe"),
                claimed_lane_ids=tuple(claimed_scope.get("lane_ids", [])),
                comparison_lane_ids=tuple(comparison_scope.get("lane_ids", [])),
                mechanism_claim=raw["mechanism_claim"],
                primary_mechanism=raw["primary_mechanism"],
                effect_modifier=raw.get("effect_modifier"),
                depends_on_candidate_id=depends_on_candidate_id,
                mechanism_relation=raw["mechanism_relation"],
                distinguishing_predictions=tuple(predictions),
            )
            if surviving_index == 0 and (
                profile.mechanism_relation
                != CandidateMechanismRelation.REFERENCE.value
            ):
                raise LLMOutputValidationError(
                    "semantic coherence: candidate_semantic_profiles[0] must "
                    "use mechanism_relation=reference"
                )
            if surviving_index > 0 and (
                profile.mechanism_relation
                == CandidateMechanismRelation.REFERENCE.value
            ):
                raise LLMOutputValidationError(
                    "semantic coherence: only the first Candidate may use "
                    "mechanism_relation=reference"
                )
            referenced_lane_ids = {
                *profile.claimed_lane_ids,
                *profile.comparison_lane_ids,
                *(
                    lane_id
                    for prediction in profile.distinguishing_predictions
                    for lane_id in prediction.lane_ids
                ),
            }
            unknown_lane_ids = sorted(referenced_lane_ids - known_lane_ids)
            if unknown_lane_ids:
                raise LLMOutputValidationError(
                    f"candidate_semantic_profiles[{profile_index}] references "
                    f"unknown Lane IDs: {unknown_lane_ids}"
                )
            if known_lane_contexts is not None:
                scope_identity = {
                    "operation": profile.claimed_operation,
                    "equipment": profile.claimed_equipment,
                    "chamber": profile.claimed_chamber,
                    "recipe": profile.claimed_recipe,
                }
                mismatched_claimed_lanes = sorted(
                    lane_id
                    for lane_id in profile.claimed_lane_ids
                    if (
                        lane := known_lane_contexts.get(lane_id)
                    ) is not None
                    and any(
                        expected is not None
                        and _compact_scope_identity(expected)
                        != _compact_scope_identity(lane.get(field_name, ""))
                        for field_name, expected in scope_identity.items()
                    )
                )
                if mismatched_claimed_lanes:
                    raise LLMOutputValidationError(
                        f"candidate_semantic_profiles[{profile_index}] claimed "
                        "Lane identity conflicts with claimed_scope: "
                        f"{mismatched_claimed_lanes}"
                    )
            claimed_lane_ids = set(profile.claimed_lane_ids)
            comparison_lane_ids = set(profile.comparison_lane_ids)
            prediction_lane_ids = {
                lane_id
                for prediction in profile.distinguishing_predictions
                for lane_id in prediction.lane_ids
            }
            product_outcome_predictions = [
                prediction
                for prediction in profile.distinguishing_predictions
                if prediction.discriminator_kind == "product_outcome"
            ]
            for prediction in product_outcome_predictions:
                expectation_lane_ids = {
                    expectation.lane_id
                    for expectation in prediction.lane_effect_expectations
                }
                if expectation_lane_ids != set(prediction.lane_ids):
                    raise LLMOutputValidationError(
                        "semantic coherence: "
                        f"candidate_semantic_profiles[{profile_index}] each "
                        "product_outcome prediction must provide Lane effect "
                        "expectations for exactly its prediction Lanes"
                    )
            if (
                profile.scope_relation
                == CandidateScopeRelation.SHARED_EFFECT.value
                and profile.claimed_scope_kind
                == CandidateClaimedScopeKind.LANE.value
                and len(claimed_lane_ids) < 2
            ):
                raise LLMOutputValidationError(
                    "semantic coherence: "
                    f"candidate_semantic_profiles[{profile_index}] shared_effect "
                    "must claim at least two Lanes"
                )
            if (
                profile.scope_relation == CandidateScopeRelation.FOCAL_ONLY.value
                and (
                    profile.claimed_scope_kind
                    not in {
                        CandidateClaimedScopeKind.LANE.value,
                        CandidateClaimedScopeKind.RECIPE.value,
                    }
                    or len(claimed_lane_ids) != 1
                    or not claimed_lane_ids < comparison_lane_ids
                )
            ):
                raise LLMOutputValidationError(
                    "semantic coherence: "
                    f"candidate_semantic_profiles[{profile_index}] focal_only "
                    "must claim one Lane within a larger comparison scope"
                )
            if (
                profile.scope_relation
                == CandidateScopeRelation.DIFFERENTIAL_SENSITIVITY.value
                and (
                    profile.claimed_scope_kind
                    not in {
                        CandidateClaimedScopeKind.LANE.value,
                        CandidateClaimedScopeKind.RECIPE.value,
                    }
                    or not claimed_lane_ids < comparison_lane_ids
                )
            ):
                raise LLMOutputValidationError(
                    "semantic coherence: "
                    f"candidate_semantic_profiles[{profile_index}] "
                    "differential_sensitivity must claim a strict subset of "
                    "the comparison Lanes"
                )
            if (
                profile.scope_relation != CandidateScopeRelation.UNRESOLVED.value
                and prediction_lane_ids != comparison_lane_ids
            ):
                raise LLMOutputValidationError(
                    "semantic coherence: "
                    f"candidate_semantic_profiles[{profile_index}] prediction "
                    "Lane coverage must match the resolved comparison scope"
                )
            if (
                profile.scope_relation
                == CandidateScopeRelation.SHARED_EFFECT.value
                and product_outcome_predictions
            ):
                effects_by_lane: dict[str, set[tuple[str, str]]] = {
                    lane_id: set() for lane_id in comparison_lane_ids
                }
                for prediction in product_outcome_predictions:
                    for expectation in prediction.lane_effect_expectations:
                        effects_by_lane.setdefault(expectation.lane_id, set()).add(
                            (
                                expectation.normalized_effect_key,
                                expectation.effect_state,
                            )
                        )
                claimed_effect_sets = {
                    frozenset(effects_by_lane.get(lane_id, set()))
                    for lane_id in claimed_lane_ids
                }
                if len(claimed_effect_sets) != 1:
                    raise LLMOutputValidationError(
                        "semantic coherence: "
                        f"candidate_semantic_profiles[{profile_index}] "
                        "shared_effect claimed Lanes must predict the same "
                        "product effect"
                    )
            profiles_by_surviving_index[surviving_index] = profile
        except (LLMOutputValidationError, ModelValidationError, TypeError, ValueError) as exc:
            errors.append(
                str(exc).strip()
                or f"candidate_semantic_profiles[{profile_index}] is invalid"
            )
    missing_indexes = [
        index
        for index in range(len(surviving_candidate_indexes))
        if index not in profiles_by_surviving_index
    ]
    if missing_indexes:
        errors.append(
            "candidate_semantic_profiles missing surviving candidate indexes: "
            f"{missing_indexes}"
        )
    return (
        tuple(
            profiles_by_surviving_index[index]
            for index in sorted(profiles_by_surviving_index)
        ),
        tuple(errors),
    )


@dataclass(frozen=True)
class QwenHypothesisCandidateGenerator:
    """Generate at most two proposals without deciding the RCA conclusion."""

    llm_client: LLMClient
    prompt_version: str = "v1"

    def __post_init__(self) -> None:
        if self.llm_client is None:
            raise ModelValidationError(
                "Qwen Hypothesis Candidate Generator requires an LLM client"
            )

    def generate(
        self,
        *,
        request_id: str,
        findings: list[AgentFinding],
        context_evidence: Sequence[Evidence] = (),
        prior_candidates: Sequence[Mapping[str, Any]] = (),
        prior_semantic_profiles: Sequence[Mapping[str, Any]] = (),
        prior_challenges: Sequence[Mapping[str, Any]] = (),
        prior_causal_gaps: Sequence[Mapping[str, Any]] = (),
        causal_lanes: Sequence[Mapping[str, Any]] = (),
        new_evidence_ids: Sequence[str] = (),
    ) -> HypothesisCandidateGeneration:
        evidence_by_id = {
            evidence.evidence_id: evidence
            for finding in findings
            for evidence in finding.evidence
            if evidence.is_typed
        }
        evidence_by_id.update(
            {
                evidence.evidence_id: evidence
                for evidence in context_evidence
                if evidence.is_typed
            }
        )
        if not evidence_by_id:
            return HypothesisCandidateGeneration(candidates=(), attempt_count=0)
        competition_context = _candidate_competition_context(
            evidence_by_id=evidence_by_id,
            new_evidence_ids=new_evidence_ids,
            prior_challenges=prior_challenges,
            prior_causal_gaps=prior_causal_gaps,
            causal_lanes=causal_lanes,
        )
        evidence_synthesis = build_lane_first_evidence_synthesis(
            evidence_by_id.values(),
            causal_lanes,
        )
        competition_brief = dict(
            evidence_synthesis.get("candidate_competition", {})
        )
        prior_candidate_evidence_ids = [
            evidence_id
            for candidate in prior_candidates[:_MAX_CANDIDATES]
            for field in ("supporting_evidence_ids", "contradicting_evidence_ids")
            for evidence_id in (str(item) for item in candidate.get(field, []))
            if evidence_id in evidence_by_id
        ]
        challenge_evidence_ids = [
            str(item)
            for challenge in competition_context["prior_candidate_challenges"]
            for field in (
                "supporting_evidence_ids",
                "contradicting_evidence_ids",
                "unexplained_precursor_evidence_ids",
            )
            for item in challenge.get(field, [])
            if str(item) in evidence_by_id
        ]
        recent_evidence_ids = [
            str(item)
            for item in competition_context["new_evidence_ids_since_prior"]
            if str(item) in evidence_by_id
        ]
        prompt_evidence_ids = _bounded_prompt_evidence_ids(
            evidence_by_id=evidence_by_id,
            synthesis_ids=[
                str(item)
                for item in evidence_synthesis.get("prompt_evidence_ids", [])
                if str(item) in evidence_by_id
            ],
            recent_ids=recent_evidence_ids,
            prior_candidate_ids=prior_candidate_evidence_ids,
            challenge_ids=challenge_evidence_ids,
        )
        evidence_synthesis["prompt_evidence_ids"] = list(prompt_evidence_ids)
        evidence_synthesis["prompt_evidence_count"] = len(prompt_evidence_ids)
        evidence_synthesis["prompt_evidence_limit"] = _MAX_PROMPT_EVIDENCE
        evidence_synthesis["recent_evidence_available_count"] = len(
            recent_evidence_ids
        )
        evidence_synthesis["recent_evidence_emitted_count"] = len(
            set(recent_evidence_ids) & set(prompt_evidence_ids)
        )
        evidence_synthesis["omitted_from_prompt_count"] = max(
            0,
            len(evidence_by_id) - len(prompt_evidence_ids),
        )
        evidence_register = _evidence_register(
            findings,
            context_evidence,
            allowed_evidence_ids=set(prompt_evidence_ids),
        )
        projection_audit_by_id: dict[str, dict[str, int]] = {}
        for record in evidence_register:
            raw_audit = record.pop("projection_audit", {})
            projection_audit_by_id[str(record.get("evidence_id", ""))] = {
                "omitted_entity_count": int(
                    raw_audit.get("omitted_entity_count", 0)
                ),
                "omitted_metadata_count": int(
                    raw_audit.get("omitted_metadata_count", 0)
                ),
            }
        # Protect only the bounded reserved subsets.  Protecting the complete
        # first-round recent-Evidence inventory made every emitted card
        # untrimable and allowed repair prompts to hover near the hard 64K
        # boundary even though only 16 recent IDs receive reserved space.
        protected_prompt_evidence_ids = {
            *_diverse_bounded_ids(
                recent_evidence_ids,
                evidence_by_id=evidence_by_id,
                limit=_MAX_RECENT_EVIDENCE,
            ),
            *_diverse_bounded_ids(
                prior_candidate_evidence_ids,
                evidence_by_id=evidence_by_id,
                limit=_MAX_PRIOR_CANDIDATE_EVIDENCE,
            ),
            *_diverse_bounded_ids(
                challenge_evidence_ids,
                evidence_by_id=evidence_by_id,
                limit=_MAX_CHALLENGE_EVIDENCE,
            ),
        }
        prompt_evidence_by_id = {
            evidence_id: evidence_by_id[evidence_id]
            for evidence_id in prompt_evidence_ids
        }
        prior_mechanism_feedback = _prior_candidate_mechanism_feedback(
            prior_candidates,
            evidence_by_id=evidence_by_id,
        )

        validation_errors: list[str] = []
        rejected_candidates: list[dict[str, Any]] = []
        competition_repair_skipped_due_to_budget = False
        pending_closure_feedback: tuple[dict[str, Any], ...] = ()
        pending_closure_proposals: tuple[HypothesisCandidateProposal, ...] = ()
        closure_history: list[dict[str, Any]] = []
        final_closure_assessment: tuple[dict[str, Any], ...] = ()
        evidence_closure_repair_attempted = False
        evidence_closure_repair_exhausted = False
        evidence_closure_repair_skipped_due_to_budget = False
        for attempt in range(1, _OUTPUT_ATTEMPTS + 1):
            # ``evidence_register`` may change between output attempts when a
            # Closure repair protects newly surfaced typed Evidence.  Build
            # every repair projection from the register that this exact
            # request will expose, never from the previous attempt's mapping.
            current_prompt_evidence_ids = {
                str(record.get("evidence_id", ""))
                for record in evidence_register
                if str(record.get("evidence_id", "")).strip()
            }
            prompt_synthesis = compact_lane_first_synthesis_for_prompt(
                evidence_synthesis,
                allowed_evidence_ids=current_prompt_evidence_ids,
                known_evidence_ids=set(evidence_by_id),
            )
            prompt_evidence_by_id = {
                evidence_id: evidence_by_id[evidence_id]
                for evidence_id in current_prompt_evidence_ids
                if evidence_id in evidence_by_id
            }
            request_competition_context = _bounded_candidate_competition_context(
                competition_context,
                allowed_evidence_ids=set(prompt_evidence_by_id),
            )
            request = LLMRequest(
                agent=AgentKind.RCA_REASONING.value,
                prompt_name="hypothesis_candidate_generator",
                prompt_version=self.prompt_version,
                payload={
                    "request_id": request_id,
                    "specialist_findings": [
                        {
                            "finding_id": finding.finding_id,
                            "agent": finding.agent,
                            "summary": finding.summary,
                            "confidence": finding.confidence,
                            "evidence_count": len(finding.evidence_ids),
                        }
                        for finding in findings
                    ],
                    "typed_evidence_register": evidence_register,
                    "evidence_synthesis": prompt_synthesis,
                    "candidate_competition_requirement": competition_brief,
                    "prior_authoritative_candidates": [
                        {
                            "candidate_id": str(
                                candidate.get("candidate_id", "")
                            ),
                            "root_cause": str(candidate.get("root_cause", "")),
                            "causal_explanation": str(
                                candidate.get(
                                    "causal_explanation",
                                    candidate.get("root_cause", ""),
                                )
                            ),
                            "supporting_evidence_ids": [
                                str(item)
                                for item in candidate.get(
                                    "supporting_evidence_ids", []
                                )
                                if str(item) in evidence_by_id
                            ],
                            "contradicting_evidence_ids": [
                                str(item)
                                for item in candidate.get(
                                    "contradicting_evidence_ids", []
                                )
                                if str(item) in evidence_by_id
                            ],
                        }
                        for candidate in prior_candidates[:_MAX_CANDIDATES]
                        if str(candidate.get("root_cause", "")).strip()
                    ],
                    "prior_candidate_semantic_profiles": [
                        dict(item)
                        for item in prior_semantic_profiles[:_MAX_CANDIDATES]
                        if isinstance(item, Mapping)
                    ],
                    "prior_candidate_mechanism_feedback": prior_mechanism_feedback,
                    "new_evidence_ids_since_prior": request_competition_context[
                        "new_evidence_ids_since_prior"
                    ],
                    "prior_candidate_challenges": request_competition_context[
                        "prior_candidate_challenges"
                    ],
                    "targeted_investigation_results": request_competition_context[
                        "targeted_investigation_results"
                    ],
                    "relevant_causal_lanes": request_competition_context[
                        "relevant_causal_lanes"
                    ],
                    "max_candidates": _MAX_CANDIDATES,
                    "output_attempt": attempt,
                    "previous_validation_feedback": (
                        _candidate_repair_feedback(
                            validation_errors[-1],
                            evidence_by_id=prompt_evidence_by_id,
                            candidate_competition=request_competition_context,
                            candidate_evidence_closure=(
                                pending_closure_feedback
                            ),
                        )
                        if validation_errors
                        else None
                    ),
                    "deterministic_candidate_proposals": [],
                },
                temperature=0.0,
            )
            projected_payload = project_prompt_evidence_references(
                request.payload,
                allowed_evidence_ids=current_prompt_evidence_ids,
                known_evidence_ids=set(evidence_by_id),
            )
            request.payload.clear()
            request.payload.update(projected_payload)
            payload_char_count = _payload_char_count(request.payload)
            while payload_char_count > _TARGET_PROMPT_PAYLOAD_CHARS:
                removable_index = next(
                    (
                        index
                        for index in range(len(evidence_register) - 1, -1, -1)
                        if str(evidence_register[index].get("evidence_id", ""))
                        not in protected_prompt_evidence_ids
                    ),
                    None,
                )
                if removable_index is None:
                    break
                evidence_register.pop(removable_index)
                current_ids = [
                    str(record.get("evidence_id", ""))
                    for record in evidence_register
                    if str(record.get("evidence_id", "")).strip()
                ]
                request.payload["new_evidence_ids_since_prior"] = [
                    evidence_id
                    for evidence_id in request.payload[
                        "new_evidence_ids_since_prior"
                    ]
                    if evidence_id in set(current_ids)
                ]
                current_evidence_by_id = {
                    evidence_id: evidence_by_id[evidence_id]
                    for evidence_id in current_ids
                    if evidence_id in evidence_by_id
                }
                current_competition_context = (
                    _bounded_candidate_competition_context(
                        competition_context,
                        allowed_evidence_ids=set(current_evidence_by_id),
                    )
                )
                request.payload["new_evidence_ids_since_prior"] = (
                    current_competition_context["new_evidence_ids_since_prior"]
                )
                request.payload["prior_candidate_challenges"] = (
                    current_competition_context["prior_candidate_challenges"]
                )
                request.payload["targeted_investigation_results"] = (
                    current_competition_context[
                        "targeted_investigation_results"
                    ]
                )
                request.payload["relevant_causal_lanes"] = (
                    current_competition_context["relevant_causal_lanes"]
                )
                if validation_errors:
                    # Keep the repair contract atomic with the final typed
                    # register.  Qwen must never be advised to cite an ID that
                    # Python just removed to satisfy the prompt budget.
                    request.payload["previous_validation_feedback"] = (
                        _candidate_repair_feedback(
                            validation_errors[-1],
                            evidence_by_id=current_evidence_by_id,
                            candidate_competition=current_competition_context,
                            candidate_evidence_closure=(
                                pending_closure_feedback
                            ),
                        )
                    )
                prompt_synthesis = compact_lane_first_synthesis_for_prompt(
                    evidence_synthesis,
                    allowed_evidence_ids=set(current_ids),
                    known_evidence_ids=set(evidence_by_id),
                )
                request.payload["evidence_synthesis"] = prompt_synthesis
                projected_payload = project_prompt_evidence_references(
                    request.payload,
                    allowed_evidence_ids=set(current_ids),
                    known_evidence_ids=set(evidence_by_id),
                )
                request.payload.clear()
                request.payload.update(projected_payload)
                payload_char_count = _payload_char_count(request.payload)
            current_prompt_evidence_ids = {
                str(record.get("evidence_id", ""))
                for record in evidence_register
                if str(record.get("evidence_id", "")).strip()
            }
            prompt_evidence_by_id = {
                evidence_id: evidence_by_id[evidence_id]
                for evidence_id in current_prompt_evidence_ids
                if evidence_id in evidence_by_id
            }
            current_prompt_evidence_id_list = [
                str(record.get("evidence_id", ""))
                for record in evidence_register
                if str(record.get("evidence_id", "")).strip()
            ]
            evidence_synthesis["prompt_evidence_ids"] = (
                current_prompt_evidence_id_list
            )
            evidence_synthesis["prompt_evidence_count"] = len(
                current_prompt_evidence_id_list
            )
            evidence_synthesis["omitted_from_prompt_count"] = max(
                0,
                len(evidence_by_id) - len(current_prompt_evidence_id_list),
            )
            omitted_entity_count = 0
            omitted_metadata_count = 0
            for evidence_id in prompt_evidence_ids:
                item = evidence_by_id[evidence_id]
                if evidence_id in current_prompt_evidence_ids:
                    audit = projection_audit_by_id.get(evidence_id, {})
                    omitted_entity_count += int(
                        audit.get("omitted_entity_count", 0)
                    )
                    omitted_metadata_count += int(
                        audit.get("omitted_metadata_count", 0)
                    )
                else:
                    omitted_entity_count += len(item.entities)
                    omitted_metadata_count += len(item.metadata)
            prompt_audit = {
                "prompt_payload_char_count": payload_char_count,
                "prompt_evidence_count": len(current_prompt_evidence_ids),
                "omitted_entity_count": omitted_entity_count,
                "omitted_metadata_count": omitted_metadata_count,
                "prompt_budget_applied": True,
                "prompt_payload_char_limit": _MAX_PROMPT_PAYLOAD_CHARS,
                "trimmed_evidence_count": (
                    len(prompt_evidence_ids) - len(current_prompt_evidence_ids)
                ),
            }
            evidence_synthesis.update(prompt_audit)
            prompt_synthesis.update(prompt_audit)
            request_prompt_synthesis = request.payload.get(
                "evidence_synthesis"
            )
            if isinstance(request_prompt_synthesis, dict):
                request_prompt_synthesis.update(prompt_audit)
            payload_char_count = _payload_char_count(request.payload)
            evidence_synthesis["prompt_payload_char_count"] = payload_char_count
            prompt_synthesis["prompt_payload_char_count"] = payload_char_count
            try:
                if payload_char_count > _MAX_PROMPT_PAYLOAD_CHARS:
                    raise LLMOutputValidationError(
                        "candidate generator prompt exceeds the governed payload limit"
                    )
                response = self.llm_client.complete_json(request)
            except LLMOutputValidationError as exc:
                validation_errors.append(str(exc).strip() or type(exc).__name__)
                continue
            try:
                allowed_output_fields = {
                    "candidates",
                    "analysis_summary",
                    "candidate_semantic_profiles",
                }
                required_output_fields = {"candidates", "analysis_summary"}
                if not required_output_fields <= set(response.data) or not set(
                    response.data
                ) <= allowed_output_fields:
                    raise LLMOutputValidationError(
                        "candidate output must contain candidates and "
                        "analysis_summary, with optional isolated "
                        "candidate_semantic_profiles"
                    )
                raw_candidates = response.data.get("candidates")
                summary = response.data.get("analysis_summary")
                if not isinstance(raw_candidates, list) or len(raw_candidates) > _MAX_CANDIDATES:
                    raise LLMOutputValidationError(
                        f"candidates must be an array with at most {_MAX_CANDIDATES} items"
                    )
                if not isinstance(summary, str) or not summary.strip():
                    raise LLMOutputValidationError(
                        "analysis_summary must be a non-empty string"
                    )
                proposals_list: list[HypothesisCandidateProposal] = []
                candidate_errors: list[str] = []
                for index, candidate in enumerate(raw_candidates):
                    try:
                        proposals_list.append(
                            _parse_candidate(
                                candidate,
                                index=index,
                                evidence_by_id=prompt_evidence_by_id,
                            )
                        )
                    except (LLMOutputValidationError, TypeError, ValueError) as exc:
                        candidate_errors.append(
                            str(exc).strip() or f"candidates[{index}] is invalid"
                        )
                if candidate_errors and proposals_list:
                    validation_errors.extend(candidate_errors)
                if candidate_errors and not proposals_list:
                    validation_errors.extend(candidate_errors)
                    if attempt < _OUTPUT_ATTEMPTS and llm_call_budget_available(
                        self.llm_client,
                        required_calls=1,
                        # Preserve one call for the post-Action Planner stop.
                        reserve_calls=1,
                    ):
                        continue
                    return HypothesisCandidateGeneration(
                        candidates=(),
                        attempt_count=attempt,
                        validation_errors=tuple(validation_errors),
                        candidate_output_invalid=True,
                        analysis_summary=summary.strip(),
                        targeted_investigation_results=tuple(
                            competition_context["targeted_investigation_results"]
                        ),
                        rejected_candidates=tuple(rejected_candidates),
                        competition_requirement=str(
                            competition_brief.get(
                                "competition_requirement",
                                CompetitionRequirement.NOT_REQUIRED.value,
                            )
                        ),
                        competition_status=(
                            CandidateCompetitionStatus.FAILED.value
                        ),
                        competition_type=str(
                            competition_brief.get(
                                "competition_type",
                                CandidateCompetitionType.NONE.value,
                            )
                        ),
                        competition_failure_reason=(
                            CompetitionFailureReason.CANDIDATE_VALIDATION_EXHAUSTED.value
                        ),
                        evidence_synthesis=evidence_synthesis,
                    )
                distinct: list[HypothesisCandidateProposal] = []
                distinct_candidate_indexes: list[int] = []
                for index, proposal in enumerate(proposals_list):
                    duplicate_match: tuple[int, _DuplicateAssessment] | None = None
                    for distinct_index, existing in zip(
                        distinct_candidate_indexes,
                        distinct,
                        strict=True,
                    ):
                        assessment = _candidate_duplicate_assessment(
                            proposal,
                            existing,
                            evidence_by_id=evidence_by_id,
                            competition_context=competition_context,
                        )
                        if assessment.is_duplicate:
                            duplicate_match = (distinct_index, assessment)
                            break
                    if duplicate_match is not None:
                        compared_index, assessment = duplicate_match
                        validation_errors.append(
                            f"candidates[{index}] is a near-duplicate of an earlier "
                            f"candidate ({assessment.reason}) and was isolated"
                        )
                        rejected_candidates.append(
                            _rejected_candidate_audit(
                                proposal,
                                candidate_index=index,
                                compared_candidate_index=compared_index,
                                output_attempt=attempt,
                                assessment=assessment,
                            )
                        )
                        continue
                    distinct.append(proposal)
                    distinct_candidate_indexes.append(index)
                proposals = tuple(distinct)
                known_lane_ids = {
                    str(item.get("lane_id", "")).strip()
                    for item in causal_lanes
                    if str(item.get("lane_id", "")).strip()
                }
                raw_scope_by_distinct_index = _raw_candidate_scope_references(
                    response.data.get("candidate_semantic_profiles"),
                    surviving_candidate_indexes=distinct_candidate_indexes,
                    known_lane_ids=known_lane_ids,
                )
                raw_scope_by_proposal = {
                    proposal: raw_scope_by_distinct_index[index]
                    for index, proposal in enumerate(proposals)
                    if index in raw_scope_by_distinct_index
                }
                semantic_profiles: tuple[CandidateSemanticProfile, ...] = ()
                semantic_validation_errors: tuple[str, ...] = ()
                if proposals:
                    semantic_profiles, semantic_validation_errors = (
                        _parse_candidate_semantic_profiles(
                            response.data.get("candidate_semantic_profiles"),
                            request_id=request_id,
                            surviving_candidate_indexes=distinct_candidate_indexes,
                            known_lane_ids=known_lane_ids,
                            known_lane_contexts={
                                str(item.get("lane_id", "")).strip(): item
                                for item in causal_lanes
                                if str(item.get("lane_id", "")).strip()
                            },
                        )
                    )
                scope_assessment_variants: tuple[dict[str, Any], ...] = ()
                if proposals:
                    proposals, semantic_profiles, scope_assessment_variants = (
                        _partition_root_candidates(
                            proposals,
                            semantic_profiles,
                            competition_requirement=str(
                                competition_brief.get(
                                    "competition_requirement",
                                    CompetitionRequirement.NOT_REQUIRED.value,
                                )
                            ),
                            candidate_ids=tuple(
                                f"{request_id}:llm:{index + 1}"
                                for index in range(len(proposals))
                            ),
                            semantic_validation_errors=semantic_validation_errors,
                        )
                    )
                    for variant in scope_assessment_variants:
                        validation_errors.append(
                            "candidate semantic relation does not form an "
                            "independent primary-mechanism alternative: "
                            f"candidate_index={variant['candidate_index']}, "
                            f"relation={variant['mechanism_relation']}"
                        )
                final_closure_assessment = (
                    _candidate_evidence_closure_assessment(
                        proposals,
                        semantic_profiles,
                        evidence_by_id=evidence_by_id,
                        causal_lanes=causal_lanes,
                        raw_scope_by_candidate_index={
                            index: raw_scope_by_proposal[proposal]
                            for index, proposal in enumerate(proposals)
                            if proposal in raw_scope_by_proposal
                        },
                        must_preserve_evidence_ids_by_candidate_index={
                            index: previous.supporting_evidence_ids
                            for index, (proposal, previous) in enumerate(
                                zip(
                                    proposals,
                                    pending_closure_proposals,
                                    strict=False,
                                )
                            )
                            if _token_similarity(
                                (
                                    f"{proposal.root_cause} "
                                    f"{proposal.causal_explanation}"
                                ),
                                (
                                    f"{previous.root_cause} "
                                    f"{previous.causal_explanation}"
                                ),
                            )
                            >= 0.65
                        },
                    )
                    if proposals
                    else ()
                )
                closure_history.append(
                    {
                        "output_attempt": attempt,
                        "status": next(
                            (
                                status
                                for status in (
                                    "incomplete",
                                    "not_evaluated",
                                    "complete",
                                )
                                if any(
                                    item.get("status") == status
                                    for item in final_closure_assessment
                                )
                            ),
                            "not_evaluated",
                        ),
                        "candidate_assessments": [
                            dict(item) for item in final_closure_assessment
                        ],
                    }
                )
                closure_repair_required = any(
                    item.get("status") == "incomplete"
                    for item in final_closure_assessment
                )
                if closure_repair_required and attempt < _OUTPUT_ATTEMPTS:
                    closure_error = _closure_validation_error(
                        final_closure_assessment
                    )
                    joint_feedback = [closure_error]
                    if semantic_validation_errors:
                        joint_feedback.append(
                            "candidate semantic profile also requires repair: "
                            + " | ".join(semantic_validation_errors)
                        )
                    validation_errors.append(" | ".join(joint_feedback))
                    pending_closure_feedback = final_closure_assessment
                    pending_closure_proposals = proposals
                    if llm_call_budget_available(
                        self.llm_client,
                        required_calls=1,
                        # Retain one governed post-Action Planner call. Closure
                        # shares the existing Candidate output-attempt budget.
                        reserve_calls=1,
                    ):
                        repair_ids = _closure_repair_evidence_ids(
                            final_closure_assessment
                        )
                        for evidence_id in repair_ids:
                            if evidence_id in {
                                str(card.get("evidence_id", ""))
                                for card in evidence_register
                            }:
                                protected_prompt_evidence_ids.add(evidence_id)
                                continue
                            while len(evidence_register) >= _MAX_PROMPT_EVIDENCE:
                                removable_index = next(
                                    (
                                        index
                                        for index in range(
                                            len(evidence_register) - 1,
                                            -1,
                                            -1,
                                        )
                                        if str(
                                            evidence_register[index].get(
                                                "evidence_id", ""
                                            )
                                        )
                                        not in protected_prompt_evidence_ids
                                    ),
                                    None,
                                )
                                if removable_index is None:
                                    break
                                evidence_register.pop(removable_index)
                            if len(evidence_register) >= _MAX_PROMPT_EVIDENCE:
                                continue
                            card = compact_evidence_prompt_card(
                                evidence_by_id[evidence_id]
                            )
                            raw_audit = card.pop("projection_audit", {})
                            projection_audit_by_id[evidence_id] = {
                                "omitted_entity_count": int(
                                    raw_audit.get("omitted_entity_count", 0)
                                ),
                                "omitted_metadata_count": int(
                                    raw_audit.get("omitted_metadata_count", 0)
                                ),
                            }
                            evidence_register.append(card)
                            protected_prompt_evidence_ids.add(evidence_id)
                        prompt_evidence_ids = tuple(
                            str(card.get("evidence_id", ""))
                            for card in evidence_register
                            if str(card.get("evidence_id", "")).strip()
                        )
                        prompt_synthesis["prompt_evidence_ids"] = list(
                            prompt_evidence_ids
                        )
                        evidence_closure_repair_attempted = True
                        continue
                    evidence_closure_repair_skipped_due_to_budget = True
                elif closure_repair_required:
                    evidence_closure_repair_exhausted = (
                        evidence_closure_repair_attempted
                    )
                competition_assessment = _candidate_competition_assessment(
                    proposals,
                    evidence_by_id=evidence_by_id,
                    competition_context=competition_context,
                    competition_brief=competition_brief,
                    semantic_profiles=semantic_profiles,
                )
                competition_assessment["semantic_validation_errors"] = list(
                    semantic_validation_errors
                )
                competition_assessment["scope_assessment_variants"] = [
                    dict(item) for item in scope_assessment_variants
                ]
                candidate_lineage = _candidate_lineage(
                    proposals,
                    prior_candidates=prior_candidates,
                    semantic_profiles=semantic_profiles,
                    prior_semantic_profiles=prior_semantic_profiles,
                    evidence_by_id=evidence_by_id,
                    competition_context=competition_context,
                    competition_brief=competition_brief,
                )
                targeted_repair_required = _competition_repair_required(
                    proposals,
                    prior_candidates=prior_candidates,
                    competition_context=competition_context,
                )
                competition_gap_repair_required = bool(
                    competition_assessment.get("competition_gap_reason")
                )
                semantic_coherence_repair_required = any(
                    error.startswith("semantic coherence:")
                    for error in semantic_validation_errors
                )
                competition_repair_required = (
                    targeted_repair_required
                    or competition_gap_repair_required
                    or semantic_coherence_repair_required
                )
                if competition_repair_required and attempt < _OUTPUT_ATTEMPTS:
                    failure_reason = str(
                        competition_assessment.get("competition_gap_reason")
                        or CompetitionGapReason.MECHANISM_ALTERNATIVE_NOT_GENERATED.value
                    )
                    if semantic_coherence_repair_required:
                        validation_errors.append(
                            "candidate semantic profile is internally inconsistent: "
                            + " | ".join(semantic_validation_errors)
                        )
                    else:
                        validation_errors.append(
                            "candidate competition is incomplete: "
                            f"{failure_reason}; preserve the prior candidate and "
                            "represent the evidence-bounded competing causal "
                            "direction or a materially different physical mechanism "
                            "independently; a Scope-only variant does not satisfy "
                            "root-cause competition"
                        )
                    # Competition repair is optional once at least one valid
                    # Candidate survives.  Keep enough global budget for one
                    # Challenge (when concrete Lanes exist) and the post-Action
                    # Planner decision instead of attempting a cap-exceeding
                    # repair call.
                    downstream_reserve = 1 + (1 if causal_lanes else 0)
                    if llm_call_budget_available(
                        self.llm_client,
                        required_calls=1,
                        reserve_calls=downstream_reserve,
                    ):
                        continue
                    competition_repair_skipped_due_to_budget = True
                final_competition_status = str(
                    competition_assessment["competition_status"]
                )
                final_failure_reason = competition_assessment.get(
                    "competition_failure_reason"
                )
                final_gap_reason = competition_assessment.get(
                    "competition_gap_reason"
                )
                if competition_repair_skipped_due_to_budget:
                    final_competition_status = (
                        CandidateCompetitionStatus.PENDING.value
                        if str(competition_assessment["competition_requirement"])
                        != CompetitionRequirement.NOT_REQUIRED.value
                        else CandidateCompetitionStatus.NOT_REQUIRED.value
                    )
                    final_failure_reason = None
                elif targeted_repair_required:
                    final_competition_status = CandidateCompetitionStatus.PENDING.value
                    final_failure_reason = None
                    if (
                        competition_assessment["competition_requirement"]
                        == CompetitionRequirement.MECHANISM_REQUIRED.value
                    ):
                        final_gap_reason = (
                            CompetitionGapReason.MECHANISM_ALTERNATIVE_NOT_GENERATED.value
                        )
                    else:
                        final_gap_reason = (
                            CompetitionGapReason.SCOPE_HYPOTHESIS_COLLAPSED.value
                            if any(
                                item.get("lineage_status") == "scope_expanded"
                                for item in candidate_lineage
                            )
                            else CompetitionGapReason.ALTERNATIVE_DIRECTION_NOT_GENERATED.value
                        )
                competition_assessment = {
                    **competition_assessment,
                    "competition_status": final_competition_status,
                    "competition_failure_reason": final_failure_reason,
                    "competition_gap_reason": final_gap_reason,
                    "targeted_competition_repair_required": (
                        targeted_repair_required
                    ),
                    "competition_repair_skipped_due_to_budget": (
                        competition_repair_skipped_due_to_budget
                    ),
                }
                return HypothesisCandidateGeneration(
                    candidates=proposals,
                    attempt_count=attempt,
                    validation_errors=tuple(validation_errors),
                    analysis_summary=summary.strip(),
                    targeted_investigation_results=tuple(
                        competition_context["targeted_investigation_results"]
                    ),
                    competition_repair_exhausted=(
                        competition_repair_required
                        and not competition_repair_skipped_due_to_budget
                    ),
                    competition_repair_skipped_due_to_budget=(
                        competition_repair_skipped_due_to_budget
                    ),
                    rejected_candidates=tuple(rejected_candidates),
                    competition_requirement=str(
                        competition_assessment["competition_requirement"]
                    ),
                    competition_status=final_competition_status,
                    competition_type=str(
                        competition_assessment["competition_type"]
                    ),
                    competition_failure_reason=(
                        str(final_failure_reason)
                        if final_failure_reason is not None
                        else None
                    ),
                    competition_gap_reason=(
                        str(final_gap_reason)
                        if final_gap_reason is not None
                        else None
                    ),
                    competition_assessment=competition_assessment,
                    candidate_semantic_profiles=semantic_profiles,
                    semantic_validation_errors=semantic_validation_errors,
                    candidate_lineage=tuple(candidate_lineage),
                    evidence_synthesis=evidence_synthesis,
                    candidate_evidence_closure=final_closure_assessment,
                    candidate_evidence_closure_history=tuple(closure_history),
                    evidence_closure_repair_attempted=(
                        evidence_closure_repair_attempted
                    ),
                    evidence_closure_repair_exhausted=(
                        evidence_closure_repair_exhausted
                    ),
                    evidence_closure_repair_skipped_due_to_budget=(
                        evidence_closure_repair_skipped_due_to_budget
                    ),
                )
            except (LLMOutputValidationError, TypeError, ValueError) as exc:
                validation_errors.append(str(exc).strip() or type(exc).__name__)
                if attempt < _OUTPUT_ATTEMPTS and llm_call_budget_available(
                    self.llm_client,
                    required_calls=1,
                    # A wholly invalid result has no downstream Candidate work,
                    # but the Supervisor still needs a governed terminal call.
                    reserve_calls=1,
                ):
                    continue
                break

        raise LLMOutputValidationError(
            "Qwen Hypothesis Candidate Generator returned invalid output twice: "
            + " | ".join(validation_errors)
        )


__all__ = [
    "HypothesisCandidateGeneration",
    "HypothesisCandidateProposal",
    "QwenHypothesisCandidateGenerator",
]
