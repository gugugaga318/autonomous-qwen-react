"""Hypothesis Engine RCA reasoning.

Batch 19 removes the retired Legacy scoring engine.  Historical RCA snapshots
remain readable through the domain DTO compatibility contracts; new work always
uses the deterministic, evidence-bounded Hypothesis Engine.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from yield_rca_core.causal_adversarial import (
    QwenAdversarialChallenger,
)
from yield_rca_core.causal_candidate_comparison import (
    QwenHypothesisCandidateComparator,
)
from yield_rca_core.causal_competition import select_diverse_lane_ids
from yield_rca_core.causal_confirmation import evaluate_impact_lot_gate
from yield_rca_core.causal_evidence_gap import (
    build_causal_evidence_gaps,
    build_hypothesis_discrimination_gaps,
)
from yield_rca_core.causal_evidence_matrix import build_causal_evidence_matrix
from yield_rca_core.causal_hypothesis import CausalHypothesis
from yield_rca_core.causal_investigation_models import (
    AlternativeLaneResolution,
    AlternativeSearchStatus,
    CandidateChallenge,
    CandidateCompetitionStatus,
    CandidateCompetitionType,
    CausalLaneRecord,
    CompetitionFailureReason,
    CompetitionRequirement,
)
from yield_rca_core.causal_lane_lifecycle import lane_is_searchable
from yield_rca_core.evidence_models import EntityType, Evidence, EvidenceType
from yield_rca_core.evidence_synthesis import build_evidence_synthesis
from yield_rca_core.hypothesis_candidate_generator import (
    QwenHypothesisCandidateGenerator,
)
from yield_rca_core.hypothesis_engine import HypothesisEngine
from yield_rca_core.llm_gateway import (
    LLMCallError,
    LLMClient,
    LLMOutputValidationError,
    llm_call_budget_available,
)
from yield_rca_core.models import (
    AgentFinding,
    AgentKind,
    AgentMode,
    FindingKind,
    Hypothesis,
    HypothesisStatus,
    ModelValidationError,
    Warning,
)

SPECIALIST_AGENTS = frozenset(
    {
        AgentKind.MES.value,
        AgentKind.FDC.value,
        AgentKind.DEFECT_WAT.value,
        AgentKind.KNOWLEDGE.value,
    }
)


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _next_reasoning_round(prior_finding: AgentFinding | None) -> int:
    if prior_finding is None:
        return 1
    raw_round = prior_finding.details.get("reasoning_round", 1)
    try:
        prior_round = int(raw_round)
    except (TypeError, ValueError):
        prior_round = 1
    return max(1, prior_round) + 1


def _lane_context(
    findings: list[AgentFinding],
    causal_lanes: Sequence[CausalLaneRecord] = (),
) -> tuple[
    list[str],
    list[str],
    list[str],
    list[str],
    list[dict[str, Any]],
]:
    """Return a full inventory plus one bounded, diverse active snapshot.

    The full inventory stays available to Python-owned state.  Every LLM-facing
    RCA stage receives only the same direction/scope-diverse active snapshot so
    a later component cannot silently revert to the top-N-by-score behaviour.
    """

    raw_lanes: list[dict[str, Any]] = []
    if causal_lanes:
        # The reconciled Python inventory is authoritative after discovery.
        # Historical raw MES payloads must not reintroduce aliases or terminal
        # Lanes into later Candidate/Challenge prompts.
        raw_lanes.extend(record.to_dict() for record in causal_lanes)
    else:
        for finding in findings:
            if finding.agent != AgentKind.MES.value:
                continue
            raw = finding.details.get("lane_candidates", [])
            if isinstance(raw, list):
                raw_lanes.extend(item for item in raw if isinstance(item, dict))
    ordered = sorted(
        {
            str(item.get("lane_id", "")).strip(): item
            for item in raw_lanes
            if str(item.get("lane_id", "")).strip()
        }.values(),
        key=lambda item: (
            -float(item.get("priority_score", 0.0)),
            str(item.get("lane_id", "")),
        ),
    )
    all_ids = [str(item["lane_id"]) for item in ordered]
    searchable = [
        item
        for item in ordered
        if lane_is_searchable(item)
    ]
    active_ids = list(select_diverse_lane_ids(searchable, limit=3))
    active_id_set = set(active_ids)
    active_contexts = [
        item for item in ordered if str(item["lane_id"]) in active_id_set
    ]
    active_contexts.sort(key=lambda item: active_ids.index(str(item["lane_id"])))
    eliminated_ids = [
        str(item["lane_id"])
        for item in ordered
        if str(item.get("investigation_status", "")) == "eliminated"
    ]
    blocked_ids = [
        str(item["lane_id"])
        for item in ordered
        if str(item.get("investigation_status", "")) == "blocked"
    ]
    return all_ids, active_ids, eliminated_ids, blocked_ids, active_contexts


def _bounded_challenge_evidence_ids(
    *,
    candidates: Sequence[dict[str, Any]],
    evidence_gaps: Sequence[dict[str, Any]],
    evidence_by_id: dict[str, Evidence],
    active_lane_ids: Sequence[str],
) -> list[str]:
    """Keep challenge citations relevant to candidates or active Lane probes."""

    active = set(active_lane_ids)
    selected: set[str] = {
        str(evidence_id)
        for candidate in candidates
        for field in ("supporting_evidence_ids", "contradicting_evidence_ids")
        for evidence_id in candidate.get(field, [])
        if str(evidence_id) in evidence_by_id
    }
    selected.update(
        str(evidence_id)
        for gap in evidence_gaps
        for evidence_id in gap.get("evidence_ids", [])
        if str(evidence_id) in evidence_by_id
    )
    selected.update(
        evidence_id
        for evidence_id, evidence in evidence_by_id.items()
        if str(evidence.metadata.get("lane_id", "")).strip() in active
    )
    return sorted(selected)


def _merge_evidence_payload(
    findings: list[AgentFinding],
    context_evidence: Sequence[Evidence] = (),
) -> list[dict[str, Any]]:
    evidence_by_id: dict[str, dict[str, Any]] = {}
    for finding in findings:
        for item in finding.details.get("evidence", []):
            evidence_by_id[str(item["evidence_id"])] = dict(item)
    for item in context_evidence:
        evidence_by_id[item.evidence_id] = item.to_dict()
    return list(evidence_by_id.values())


def _typed_evidence(
    findings: Sequence[AgentFinding],
    context_evidence: Sequence[Evidence],
) -> list[Evidence]:
    return list(
        {
            item.evidence_id: item
            for item in (
                *[
                    evidence
                    for finding in findings
                    for evidence in finding.evidence
                ],
                *context_evidence,
            )
            if item.is_typed
        }.values()
    )


def _observed_impact_lot_ids(
    findings: Sequence[AgentFinding],
    evidence: Sequence[Evidence],
    *,
    source_lot_id: str | None,
) -> list[str]:
    """Recover the candidate Lot universe from findings and typed Evidence."""

    values: list[str] = []
    for finding in findings:
        raw = finding.details.get("impact_lots")
        if isinstance(raw, list):
            values.extend(str(item) for item in raw)
    for item in evidence:
        if item.evidence_type != EvidenceType.IMPACT_SCOPE.value:
            continue
        raw = item.metadata.get("impact_lots", [])
        if isinstance(raw, (list, tuple)):
            values.extend(str(lot_id) for lot_id in raw)
        values.extend(
            entity.entity_id
            for entity in item.entities
            if entity.entity_type == EntityType.LOT.value
        )
    return list(
        dict.fromkeys(
            lot_id.strip()
            for lot_id in values
            if lot_id.strip() and lot_id.strip() != source_lot_id
        )
    )


def _merge_warnings(findings: list[AgentFinding]) -> list[Warning]:
    return list(
        {
            warning.warning_id: warning
            for finding in findings
            for warning in finding.warnings
        }.values()
    )


def _recommended_actions(findings: list[AgentFinding]) -> list[dict[str, Any]]:
    discovery = next(
        (
            finding
            for finding in findings
            if finding.finding_kind == FindingKind.KNOWLEDGE_DISCOVERY.value
        ),
        None,
    )
    if discovery is None:
        return []
    top_case = discovery.details.get("top_case", {})
    solution = str(top_case.get("solution", "")).strip()
    if not solution:
        return []
    evidence_ids = list(discovery.evidence_ids)
    return [
        {"action": action.strip(), "evidence_ids": evidence_ids}
        for action in solution.split(",")
        if action.strip()
    ]


def _unsupported_source_warning(
    findings: list[AgentFinding],
    *,
    supported: bool,
    ranked_candidates: list[dict[str, Any]],
    blocking_data_missing_evidence_ids: Sequence[str] = (),
) -> Warning | None:
    """Make an evidence gap explicit without letting Knowledge fill it.

    A typed missing-FDC observation is already a hard signal that a required
    source is unavailable.  A second, narrower boundary covers an unresolved
    slurry/material hypothesis when every current-Lot slurry trace is normal:
    the deployment has no material-batch genealogy capability, so historical
    similarity cannot silently stand in for that missing source.
    """

    blocking_ids = {
        str(evidence_id)
        for evidence_id in blocking_data_missing_evidence_ids
        if str(evidence_id)
    }
    missing_ids = _unique(
        [
            evidence.evidence_id
            for finding in findings
            for evidence in finding.evidence
            if evidence.evidence_type == "data_missing"
            and evidence.source_type in {"fdc", "wat", "mes", "analytics"}
            and (
                evidence.evidence_id in blocking_ids
                or evidence.metadata.get("required_for_confirmation") is True
            )
        ]
    )
    if missing_ids:
        return Warning(
            warning_id="WARN_UNSUPPORTED_DATA_SOURCE",
            message=(
                "A required operational data source is unavailable; the RCA remains "
                "bounded by typed DATA_MISSING Evidence."
            ),
            evidence_ids=missing_ids,
        )
    if supported:
        return None

    fdc = next((item for item in findings if item.agent == AgentKind.FDC.value), None)
    if fdc is None:
        return None
    parameters = {
        str(item.get("parameter_name", "")).casefold(): abs(
            float(item.get("avg_delta_percent", 0.0))
        )
        for item in fdc.details.get("parameter_summary", [])
        if isinstance(item, dict)
    }
    material_terms = ("slurry", "chemical", "material", "consumable")
    normal_material_parameters = {
        name
        for name, delta in parameters.items()
        if any(term in name for term in material_terms) and delta < 5.0
    }
    abnormal_material_parameters = {
        name
        for name, delta in parameters.items()
        if any(term in name for term in material_terms) and delta >= 5.0
    }
    candidate_needs_material_trace = any(
        any(term in str(item.get("root_cause", "")).casefold() for term in material_terms)
        for item in ranked_candidates
    )
    if (
        candidate_needs_material_trace
        and normal_material_parameters
        and not abnormal_material_parameters
    ):
        evidence_ids = _unique(
            [
                evidence.evidence_id
                for evidence in fdc.evidence
                if evidence.source_field
                and str(evidence.source_field).casefold() in normal_material_parameters
            ]
        )
        return Warning(
            warning_id="WARN_UNSUPPORTED_DATA_SOURCE",
            message=(
                "Material or chemical batch genealogy is not configured. Current "
                "equipment traces are insufficient to confirm the remaining "
                "material-related hypothesis."
            ),
            evidence_ids=evidence_ids,
        )
    return None


@dataclass(frozen=True)
class RCAReasoningAgent:
    """Generate optional Qwen candidates, then use the Python Evidence Gate."""

    hypothesis_engine: HypothesisEngine = HypothesisEngine()
    llm_client: LLMClient | None = None
    agent_mode: str = AgentMode.DETERMINISTIC.value
    prompt_version: str = "v1"

    def analyze(
        self,
        *,
        request_id: str,
        findings: list[AgentFinding],
        context_evidence: Sequence[Evidence] = (),
        causal_lanes: Sequence[CausalLaneRecord] = (),
        prior_rca_finding: AgentFinding | None = None,
    ) -> AgentFinding:
        if not findings:
            raise ModelValidationError("RCA reasoning requires Specialist findings")
        unsupported = {finding.agent for finding in findings} - SPECIALIST_AGENTS
        if unsupported:
            raise ModelValidationError(
                f"RCA reasoning only accepts Specialist findings, got {sorted(unsupported)}"
            )
        evidence_ids = _unique(
            [
                *[
                    evidence_id
                    for finding in findings
                    for evidence_id in finding.evidence_ids
                ],
                *[item.evidence_id for item in context_evidence],
            ]
        )
        if not evidence_ids:
            raise ModelValidationError("Specialist findings must reference evidence_ids")
        source_lot_id = next(
            (
                str(finding.details.get("source_lot_id"))
                for finding in findings
                if finding.details.get("source_lot_id")
            ),
            None,
        )
        prior_candidates = (
            [
                dict(candidate)
                for candidate in prior_rca_finding.details.get(
                    "ranked_candidates", []
                )
                if isinstance(candidate, dict)
            ]
            if prior_rca_finding is not None
            and isinstance(
                prior_rca_finding.details.get("ranked_candidates", []),
                list,
            )
            else []
        )
        prior_candidate_semantic_profiles = (
            [
                dict(profile)
                for profile in prior_rca_finding.details.get(
                    "candidate_semantic_profiles", []
                )
                if isinstance(profile, dict)
            ]
            if prior_rca_finding is not None
            and isinstance(
                prior_rca_finding.details.get("candidate_semantic_profiles", []),
                list,
            )
            else []
        )
        prior_evidence_ids = (
            set(prior_rca_finding.evidence_ids)
            if prior_rca_finding is not None
            else set()
        )
        new_evidence_ids_since_prior = [
            evidence_id
            for evidence_id in evidence_ids
            if evidence_id not in prior_evidence_ids
        ]
        prior_challenges = (
            [
                dict(item)
                for item in prior_rca_finding.details.get("candidate_challenges", [])
                if isinstance(item, dict)
            ]
            if prior_rca_finding is not None
            and isinstance(
                prior_rca_finding.details.get("candidate_challenges", []),
                list,
            )
            else []
        )
        prior_causal_gaps = (
            [
                dict(item)
                for item in prior_rca_finding.details.get("causal_evidence_gaps", [])
                if isinstance(item, dict)
            ]
            if prior_rca_finding is not None
            and isinstance(
                prior_rca_finding.details.get("causal_evidence_gaps", []),
                list,
            )
            else []
        )
        (
            all_lane_ids,
            active_lane_ids,
            eliminated_lane_ids,
            blocked_lane_ids,
            lane_contexts,
        ) = _lane_context(findings, causal_lanes)

        warnings = _merge_warnings(findings)
        present_agents = {finding.agent for finding in findings}
        missing_agents = sorted(SPECIALIST_AGENTS - present_agents)
        if missing_agents:
            warnings.append(
                Warning(
                    warning_id="WARN_RCA_MISSING_FINDINGS",
                    message=f"Missing Specialist findings: {', '.join(missing_agents)}.",
                )
            )

        candidate_generation: dict[str, Any] = {
            "source": "deterministic_only",
            "candidate_count": 0,
            "attempt_count": 0,
            "validation_errors": [],
            "rejected_candidates": [],
            "fallback_reason": None,
            "candidate_output_invalid": False,
        }
        external_candidates: list[dict[str, Any]] = []
        deterministic_candidates_enabled = self.agent_mode != AgentMode.LLM.value
        candidate_comparison: dict[str, Any] | None = None
        candidate_challenges: list[CandidateChallenge] = []
        alternative_lane_resolutions: list[AlternativeLaneResolution] = []
        consumed_discriminators: set[tuple[str, str]] = set()
        alternative_search_status = AlternativeSearchStatus.NOT_SEARCHED.value
        competition_requirement = CompetitionRequirement.NOT_EVALUATED.value
        competition_status = CandidateCompetitionStatus.NOT_EVALUATED.value
        competition_type = CandidateCompetitionType.NOT_EVALUATED.value
        competition_axes: list[str] = []
        scope_assessment_status = "not_evaluated"
        competition_failure_reason: str | None = None
        competition_gap_reason: str | None = None
        candidate_semantic_profiles: list[dict[str, Any]] = []
        candidate_lineage: list[dict[str, Any]] = []
        challenge_generation: dict[str, Any] = {
            "source": "not_requested",
            "attempt_count": 0,
            "validation_errors": [],
            "output_invalid": False,
        }
        lane_first_evidence_synthesis: dict[str, Any] | None = None
        if self.agent_mode == AgentMode.LLM.value and self.llm_client is not None:
            try:
                generated = QwenHypothesisCandidateGenerator(
                    self.llm_client,
                    prompt_version=self.prompt_version,
                ).generate(
                    request_id=request_id,
                    findings=findings,
                    context_evidence=context_evidence,
                    prior_candidates=prior_candidates,
                    prior_semantic_profiles=(
                        prior_candidate_semantic_profiles
                    ),
                    prior_challenges=prior_challenges,
                    prior_causal_gaps=prior_causal_gaps,
                    causal_lanes=lane_contexts,
                    new_evidence_ids=new_evidence_ids_since_prior,
                )
                lane_first_evidence_synthesis = (
                    dict(generated.evidence_synthesis)
                    if generated.evidence_synthesis is not None
                    else None
                )
                targeted_investigation_results = [
                    dict(item) for item in generated.targeted_investigation_results
                ]
                consumed_discriminators = {
                    (
                        str(item.get("lane_id", "")),
                        str(item.get("discriminator_kind", "")),
                    )
                    for item in targeted_investigation_results
                    if item.get("answered")
                    and str(item.get("lane_id", ""))
                    and str(item.get("discriminator_kind", ""))
                }
                external_candidates = [
                    candidate.to_dict() for candidate in generated.candidates
                ]
                candidate_generation = {
                    "source": "qwen",
                    "candidate_count": len(external_candidates),
                    "attempt_count": generated.attempt_count,
                    "validation_errors": list(generated.validation_errors),
                    "rejected_candidates": [
                        dict(item) for item in generated.rejected_candidates
                    ],
                    "fallback_reason": None,
                    "candidate_output_invalid": generated.candidate_output_invalid,
                    "prior_candidate_count": len(prior_candidates),
                    "analysis_summary": generated.analysis_summary,
                    "competition_repair_exhausted": (
                        generated.competition_repair_exhausted
                    ),
                    "competition_repair_skipped_due_to_budget": (
                        generated.competition_repair_skipped_due_to_budget
                    ),
                    "targeted_investigation_count": len(
                        targeted_investigation_results
                    ),
                    "targeted_investigation_results": (
                        targeted_investigation_results
                    ),
                    "targeted_supporting_evidence_ids": list(
                        dict.fromkeys(
                            evidence_id
                            for item in targeted_investigation_results
                            for evidence_id in item.get(
                                "new_supporting_evidence_ids", []
                            )
                        )
                    ),
                    "consumed_discriminators": [
                        {
                            "lane_id": lane_id,
                            "discriminator_kind": discriminator_kind,
                        }
                        for lane_id, discriminator_kind in sorted(
                            consumed_discriminators
                        )
                    ],
                    "competition_requirement": generated.competition_requirement,
                    "competition_status": generated.competition_status,
                    "competition_type": generated.competition_type,
                    "competition_failure_reason": (
                        generated.competition_failure_reason
                    ),
                    "competition_gap_reason": generated.competition_gap_reason,
                    "competition_assessment": (
                        dict(generated.competition_assessment)
                        if generated.competition_assessment is not None
                        else None
                    ),
                    "candidate_semantic_profiles": [
                        item.to_dict()
                        for item in generated.candidate_semantic_profiles
                    ],
                    "semantic_validation_errors": list(
                        generated.semantic_validation_errors
                    ),
                    "candidate_lineage": [
                        dict(item) for item in generated.candidate_lineage
                    ],
                }
                competition_requirement = generated.competition_requirement
                competition_status = generated.competition_status
                competition_type = generated.competition_type
                competition_axes = list(
                    (generated.competition_assessment or {}).get(
                        "competition_axes", []
                    )
                )
                scope_assessment_status = str(
                    (generated.competition_assessment or {}).get(
                        "scope_assessment_status", "not_evaluated"
                    )
                )
                competition_failure_reason = (
                    generated.competition_failure_reason
                )
                competition_gap_reason = generated.competition_gap_reason
                candidate_semantic_profiles = [
                    item.to_dict()
                    for item in generated.candidate_semantic_profiles
                ]
                candidate_lineage = [
                    dict(item) for item in generated.candidate_lineage
                ]
                if generated.candidate_output_invalid:
                    candidate_generation["fallback_reason"] = "qwen_candidate_output_invalid"
                    warnings.append(
                        Warning(
                            warning_id="WARN_RCA_QWEN_CANDIDATE_INVALID",
                            message=(
                                "Qwen returned no valid causal candidate after the "
                                "bounded validation attempts."
                            ),
                            evidence_ids=evidence_ids,
                        )
                    )
                if external_candidates:
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
                    matrices = []
                    semantic_profiles_by_candidate_id = {
                        str(profile.get("candidate_id", "")).strip(): profile
                        for profile in candidate_semantic_profiles
                        if str(profile.get("candidate_id", "")).strip()
                    }
                    semantic_profiles_expected = bool(
                        candidate_semantic_profiles
                    ) or competition_requirement not in {
                        CompetitionRequirement.NOT_EVALUATED.value,
                        CompetitionRequirement.NOT_REQUIRED.value,
                    }
                    for index, candidate in enumerate(external_candidates):
                        candidate_id = f"{request_id}:llm:{index + 1}"
                        matrices.append(
                            build_causal_evidence_matrix(
                                CausalHypothesis(
                                    root_cause=str(candidate["root_cause"]),
                                    causal_explanation=str(candidate["causal_explanation"]),
                                    supporting_evidence_ids=tuple(
                                        candidate["supporting_evidence_ids"]
                                    ),
                                    contradicting_evidence_ids=tuple(
                                        candidate["contradicting_evidence_ids"]
                                    ),
                                ),
                                evidence_by_id.values(),
                                semantic_profile=(
                                    semantic_profiles_by_candidate_id.get(
                                        candidate_id,
                                        {} if semantic_profiles_expected else None,
                                    )
                                ),
                            )
                        )
                    challenge_candidates = [
                        {
                            **candidate,
                            "candidate_id": f"{request_id}:llm:{index + 1}",
                            "semantic_profile": next(
                                (
                                    profile
                                    for profile in candidate_semantic_profiles
                                    if profile.get("candidate_id")
                                    == f"{request_id}:llm:{index + 1}"
                                ),
                                None,
                            ),
                        }
                        for index, candidate in enumerate(external_candidates)
                    ]
                    gaps = build_causal_evidence_gaps(matrices)
                    gaps.extend(
                        build_hypothesis_discrimination_gaps(
                            matrices,
                            causal_lanes=lane_contexts,
                            candidate_ids=[
                                str(candidate["candidate_id"])
                                for candidate in challenge_candidates
                            ],
                            source_lot_id=source_lot_id,
                            consumed_discriminators=consumed_discriminators,
                            competition_brief=(
                                lane_first_evidence_synthesis or {}
                            ).get("candidate_competition", {}),
                            candidate_semantic_profiles=(
                                candidate_semantic_profiles
                            ),
                        )
                    )
                    challenge_evidence_ids = _bounded_challenge_evidence_ids(
                        candidates=challenge_candidates,
                        evidence_gaps=gaps,
                        evidence_by_id=evidence_by_id,
                        active_lane_ids=active_lane_ids,
                    )
                    challenge_evidence_by_id = {
                        evidence_id: evidence_by_id[evidence_id]
                        for evidence_id in challenge_evidence_ids
                    }
                    if len(external_candidates) >= 2:
                        if not llm_call_budget_available(
                            self.llm_client,
                            required_calls=1,
                            # Challenge and the post-Action Planner are both
                            # decision-critical; comparison prose is optional.
                            reserve_calls=1 + (1 if all_lane_ids else 0),
                        ):
                            candidate_comparison = {
                                "source": "python",
                                "comparison_skipped_due_to_budget": True,
                            }
                        else:
                            try:
                                candidate_comparison = (
                                    QwenHypothesisCandidateComparator(
                                        self.llm_client,
                                        prompt_version=self.prompt_version,
                                    ).compare(
                                        request_id=request_id,
                                        candidates=challenge_candidates,
                                        matrices=matrices,
                                        evidence_gaps=gaps,
                                    )
                                )
                            except (LLMCallError, LLMOutputValidationError) as exc:
                                candidate_comparison = {
                                    "source": "python",
                                    "comparison_error": str(exc),
                                }
                    if not all_lane_ids:
                        # Pre-Batch-25 snapshots may not contain concrete Lane
                        # records. There is no alternative Lane to investigate,
                        # so preserve the legacy contract without making an
                        # extra model call.
                        candidate_challenges = [
                            CandidateChallenge(
                                candidate_id=str(candidate["candidate_id"]),
                                supporting_evidence_ids=tuple(
                                    str(item)
                                    for item in candidate.get(
                                        "supporting_evidence_ids", []
                                    )
                                ),
                                challenge_explanation=(
                                    "Legacy evidence snapshot has no concrete Lane "
                                    "inventory; no additional Lane can be compared."
                                ),
                                status="resolved",
                            )
                            for candidate in challenge_candidates
                        ]
                        alternative_search_status = (
                            AlternativeSearchStatus.ALTERNATIVES_ELIMINATED.value
                        )
                        challenge_generation = {
                            "source": "python_legacy_compatibility",
                            "attempt_count": 0,
                            "challenge_count": len(candidate_challenges),
                            "validation_errors": [],
                            "output_invalid": False,
                            "alternative_search_status": alternative_search_status,
                        }
                    else:
                        challenge_result = QwenAdversarialChallenger(
                            self.llm_client,
                            prompt_version=self.prompt_version,
                        ).generate(
                            request_id=request_id,
                            candidates=challenge_candidates,
                            matrices=matrices,
                            evidence_gaps=gaps,
                            evidence_ids=challenge_evidence_ids,
                            evidence_by_id=challenge_evidence_by_id,
                            lane_ids=active_lane_ids,
                            active_lane_ids=active_lane_ids,
                            eliminated_lane_ids=[
                                lane_id
                                for lane_id in eliminated_lane_ids
                                if lane_id in set(active_lane_ids)
                            ],
                            blocked_lane_ids=[
                                lane_id
                                for lane_id in blocked_lane_ids
                                if lane_id in set(active_lane_ids)
                            ],
                            lane_contexts=lane_contexts,
                            candidate_competition=(
                                generated.competition_assessment
                            ),
                        )
                        candidate_challenges = list(challenge_result.challenges)
                        alternative_lane_resolutions = list(
                            challenge_result.lane_resolutions
                        )
                        alternative_search_status = (
                            challenge_result.alternative_search_status
                        )
                        challenge_generation = {
                            "source": "qwen",
                            "attempt_count": challenge_result.attempt_count,
                            "challenge_count": len(candidate_challenges),
                            "validation_errors": list(
                                challenge_result.validation_errors
                            ),
                            "output_invalid": challenge_result.output_invalid,
                            "repair_skipped_due_to_budget": (
                                challenge_result.repair_skipped_due_to_budget
                            ),
                            "alternative_search_status": alternative_search_status,
                            "alternative_lane_resolutions": [
                                item.to_dict()
                                for item in alternative_lane_resolutions
                            ],
                            "prompt_audit": (
                                dict(challenge_result.prompt_audit)
                                if challenge_result.prompt_audit is not None
                                else None
                            ),
                        }
                    if challenge_generation.get("output_invalid"):
                        competition_status = CandidateCompetitionStatus.FAILED.value
                        competition_failure_reason = (
                            CompetitionFailureReason.CHALLENGE_OUTPUT_INVALID.value
                        )
                        warnings.append(
                            Warning(
                                warning_id="WARN_RCA_QWEN_CHALLENGE_INVALID",
                                message=(
                                    "Qwen adversarial challenge output was invalid; "
                                    "the candidate remains unconfirmed."
                                ),
                                evidence_ids=evidence_ids,
                            )
                        )
            except (LLMCallError, LLMOutputValidationError) as exc:
                candidate_generation = {
                    "source": "qwen",
                    "candidate_count": 0,
                    "attempt_count": (
                        2 if isinstance(exc, LLMOutputValidationError) else 1
                    ),
                    "validation_errors": [str(exc)],
                    "fallback_reason": (
                        "qwen_candidate_output_invalid"
                        if isinstance(exc, LLMOutputValidationError)
                        else "qwen_candidate_provider_failed"
                    ),
                    "candidate_output_invalid": True,
                    "competition_requirement": competition_requirement,
                    "competition_status": CandidateCompetitionStatus.FAILED.value,
                    "competition_type": competition_type,
                    "competition_failure_reason": (
                        CompetitionFailureReason.CANDIDATE_VALIDATION_EXHAUSTED.value
                    ),
                    "candidate_lineage": [],
                }
                competition_status = CandidateCompetitionStatus.FAILED.value
                competition_failure_reason = (
                    CompetitionFailureReason.CANDIDATE_VALIDATION_EXHAUSTED.value
                )
                warnings.append(
                    Warning(
                        warning_id=(
                            "WARN_RCA_QWEN_CANDIDATE_INVALID"
                            if isinstance(exc, LLMOutputValidationError)
                            else "WARN_RCA_LLM_CANDIDATE_FALLBACK"
                        ),
                        message=(
                            "Qwen hypothesis candidate output was invalid; the "
                            "workflow returned an inconclusive result without "
                            "inventing a deterministic replacement candidate."
                            if isinstance(exc, LLMOutputValidationError)
                            else "Qwen hypothesis candidate generation failed; the "
                            "deterministic Evidence Gate continued without model "
                            "candidates."
                        ),
                        evidence_ids=evidence_ids,
                    )
                )
        if candidate_generation.get("candidate_output_invalid"):
            warnings.append(
                Warning(
                    warning_id="WARN_RCA_CANDIDATE_VALIDATION_BOUNDARY",
                    message=(
                        "At least one Qwen candidate was rejected by the Python "
                        "candidate contract or no valid Qwen candidate remained."
                    ),
                    evidence_ids=evidence_ids,
                )
            )

        if (
            competition_status == CandidateCompetitionStatus.ACTIVE.value
            and alternative_search_status
            == AlternativeSearchStatus.ALTERNATIVES_ELIMINATED.value
        ):
            competition_status = CandidateCompetitionStatus.RESOLVED.value
            competition_failure_reason = None
        elif (
            competition_status == CandidateCompetitionStatus.PENDING.value
            and competition_requirement
            == CompetitionRequirement.ALTERNATIVE_DISCOVERY_REQUIRED.value
            and alternative_search_status
            in {
                AlternativeSearchStatus.ALTERNATIVE_FOUND.value,
                AlternativeSearchStatus.IN_PROGRESS.value,
                AlternativeSearchStatus.UNRESOLVED.value,
            }
        ):
            competition_status = CandidateCompetitionStatus.ACTIVE.value

        if competition_status == CandidateCompetitionStatus.FAILED.value:
            alternative_search_status = AlternativeSearchStatus.UNRESOLVED.value
            warnings.append(
                Warning(
                    warning_id="WARN_RCA_CANDIDATE_COMPETITION_FAILED",
                    message=(
                        "Qwen did not complete the Python-required candidate "
                        "competition; confirmation remains blocked."
                    ),
                    evidence_ids=evidence_ids,
                )
            )
        candidate_generation["competition_requirement"] = competition_requirement
        candidate_generation["competition_status"] = competition_status
        candidate_generation["competition_type"] = competition_type
        candidate_generation["competition_axes"] = competition_axes
        candidate_generation["scope_assessment_status"] = (
            scope_assessment_status
        )
        candidate_generation["competition_failure_reason"] = (
            competition_failure_reason
        )
        candidate_generation["competition_gap_reason"] = competition_gap_reason
        candidate_generation["candidate_lineage"] = candidate_lineage
        candidate_generation["candidate_semantic_profiles"] = (
            candidate_semantic_profiles
        )

        engine_result = self.hypothesis_engine.analyze(
            request_id=request_id,
            findings=findings,
            mode="active",
            external_candidates=external_candidates,
            include_deterministic_candidates=deterministic_candidates_enabled,
            candidate_comparison=candidate_comparison,
            # Every active Qwen result uses the strict Confirmation Gate.
            # Legacy deterministic/controlled snapshots retain the non-strict
            # compatibility path, but missing Qwen temporal Evidence is a real
            # gap rather than permission to relax the gate.
            strict_confirmation=self.agent_mode == AgentMode.LLM.value,
            alternative_search_status=alternative_search_status,
            candidate_challenges=candidate_challenges,
            context_evidence=context_evidence,
            causal_lanes=tuple(
                lane for lane in causal_lanes if lane.lane_id in set(active_lane_ids)
            ),
            consumed_discriminators=consumed_discriminators,
            competition_brief=(lane_first_evidence_synthesis or {}).get(
                "candidate_competition", {}
            ),
            candidate_semantic_profiles=candidate_semantic_profiles,
        )
        decision = engine_result["decision_gate"]
        root_cause = str(decision["root_cause"])
        status = str(decision["status"])
        conclusion_status = str(decision.get("conclusion_status", status))
        confidence = float(decision["confidence"])
        supported = status == HypothesisStatus.SUPPORTED.value
        if (
            self.agent_mode == AgentMode.LLM.value
            and external_candidates
            and alternative_search_status
            != AlternativeSearchStatus.ALTERNATIVES_ELIMINATED.value
        ):
            warnings.append(
                Warning(
                    warning_id="WARN_RCA_ALTERNATIVE_SEARCH_INCOMPLETE",
                    message=(
                        "The Qwen candidate has not completed a Python-validated "
                        "adversarial alternative search; confirmation is blocked."
                    ),
                    evidence_ids=evidence_ids,
                )
            )
        active_candidate = next(
            (
                candidate
                for candidate in engine_result["candidates"]
                if candidate["root_cause"] == root_cause
            ),
            None,
        )
        root_cause_evidence_ids = (
            list(active_candidate["evidence_ids"]) if active_candidate is not None else []
        ) or list(evidence_ids)
        rationale = (
            "The Hypothesis Engine decision gate accepted the highest-ranked "
            "evidence-bounded hypothesis."
            if supported
            else "The Hypothesis Engine decision gate is inconclusive."
        )
        actions = _recommended_actions(findings) if supported else []
        if supported and not actions:
            warnings.append(
                Warning(
                    warning_id="WARN_RCA_NO_ACTION_EVIDENCE",
                    message="No evidence-backed recommended actions are available.",
                    evidence_ids=root_cause_evidence_ids,
                )
            )
        if bool(decision.get("conflicting_physics", False)):
            warnings.append(
                Warning(
                    warning_id="WARN_RCA_CONFLICTING_EVIDENCE",
                    message=(
                        "Independent operational Evidence conflicts with the candidate "
                        "mechanism, so the root cause cannot be confirmed."
                    ),
                    evidence_ids=root_cause_evidence_ids,
                )
            )
        if not supported:
            warnings.append(
                Warning(
                    warning_id="WARN_RCA_INCONCLUSIVE",
                    message=rationale,
                    evidence_ids=root_cause_evidence_ids,
                )
            )

        hypothesis = Hypothesis(
            hypothesis_id=f"{request_id}:hypothesis",
            root_cause=root_cause,
            confidence=confidence,
            evidence_ids=root_cause_evidence_ids,
            status=status,
            rationale=rationale,
            supporting_evidence_ids=(
                list(active_candidate["supporting_evidence_ids"])
                if active_candidate is not None
                else []
            ),
            contradicting_evidence_ids=(
                list(active_candidate["contradicting_evidence_ids"])
                if active_candidate is not None
                else []
            ),
            neutral_evidence_ids=(
                list(active_candidate["neutral_evidence_ids"])
                if active_candidate is not None
                else []
            ),
            validation_results=(
                [dict(item) for item in active_candidate["validation_results"]]
                if active_candidate is not None
                else []
            ),
            rank=(int(active_candidate["rank"]) if active_candidate is not None else None),
            rejection_reasons=(
                list(active_candidate["rejection_reasons"])
                if active_candidate is not None
                else []
            ),
        )
        ranked_candidates = [
            {
                "candidate_id": candidate["hypothesis_id"],
                "root_cause": candidate["root_cause"],
                "causal_explanation": candidate.get(
                    "causal_explanation", candidate["root_cause"]
                ),
                "score": candidate["confidence"],
                "basis": candidate["basis"],
                "status": candidate["status"],
                "evidence_ids": list(candidate["evidence_ids"]),
                "supporting_evidence_ids": list(candidate["supporting_evidence_ids"]),
                "contradicting_evidence_ids": list(
                    candidate["contradicting_evidence_ids"]
                ),
                "rejection_reasons": list(candidate["rejection_reasons"]),
                "causal_matrix_status": candidate.get("causal_matrix_status"),
                "causal_chain_completeness": candidate.get(
                    "causal_evidence_matrix", {}
                ).get("causal_chain_completeness"),
                "data_missing_evidence_ids": list(
                    candidate.get("causal_evidence_matrix", {}).get(
                        "data_missing_evidence_ids", []
                    )
                ),
                "mechanism_support_source": candidate.get("mechanism_support_source"),
                "causal_evidence_matrix": dict(
                    candidate.get("causal_evidence_matrix", {})
                ),
            }
            for candidate in engine_result["candidates"]
        ]
        unsupported_source = _unsupported_source_warning(
            findings,
            supported=supported,
            ranked_candidates=ranked_candidates,
            blocking_data_missing_evidence_ids=decision.get(
                "blocking_data_missing_evidence_ids", []
            ),
        )
        if unsupported_source is not None:
            warnings.append(unsupported_source)
        all_typed_evidence = _typed_evidence(findings, context_evidence)
        observed_impact_lots = _observed_impact_lot_ids(
            findings,
            all_typed_evidence,
            source_lot_id=source_lot_id,
        )
        candidate_impact_scopes: list[dict[str, Any]] = []
        selected_impact_scope: dict[str, Any] | None = None
        for candidate_index, candidate in enumerate(engine_result["candidates"]):
            candidate_is_authoritative = (
                supported and candidate["root_cause"] == root_cause
            )
            candidate_scope = evaluate_impact_lot_gate(
                source_lot_id=source_lot_id,
                candidate=candidate,
                evidence=all_typed_evidence,
                observed_impact_lots=observed_impact_lots,
                authoritative_conclusion_status=(
                    conclusion_status
                    if candidate_is_authoritative
                    else "inconclusive"
                ),
            )
            candidate_scope = {
                **candidate_scope,
                "candidate_index": candidate_index,
                "candidate_rank": candidate.get("rank"),
            }
            candidate_impact_scopes.append(candidate_scope)
            if candidate_is_authoritative:
                selected_impact_scope = candidate_scope
        if selected_impact_scope is None and candidate_impact_scopes:
            selected_impact_scope = candidate_impact_scopes[0]
        impact_lot_gate = (
            {
                **selected_impact_scope,
                "candidate_scopes": candidate_impact_scopes,
            }
            if selected_impact_scope is not None
            else {
                "source_lot_id": source_lot_id,
                "candidate_root_cause": None,
                "authoritative_conclusion_status": conclusion_status,
                "observed_impact_lots": observed_impact_lots,
                "candidate_impact_lots": [],
                "confirmed_impact_lots": [],
                "confirmation_blocked_reason": (
                    "no causal candidate is available"
                ),
                "scope_status": "not_evaluated",
                "candidate_scope_status": "not_evaluated",
                "publication_status": "not_evaluated",
                "scope_basis": (
                    "Impact scope was not evaluated because no causal candidate "
                    "is available."
                ),
                "data_missing_evidence_ids": [
                    item.evidence_id
                    for item in context_evidence
                    if item.evidence_type == "data_missing"
                ],
                "non_blocking_data_missing_evidence_ids": [],
                "candidate_scopes": [],
                "rows": [],
            }
        )
        return AgentFinding(
            finding_id=f"{request_id}:rca",
            agent=AgentKind.RCA_REASONING.value,
            summary=(
                f"Root cause: {root_cause} (confidence {confidence:.0%})."
                if supported
                else f"RCA result is inconclusive (confidence {confidence:.0%})."
            ),
            confidence=confidence,
            evidence_ids=evidence_ids,
            details={
                "root_cause": root_cause,
                "reasoning_round": _next_reasoning_round(prior_rca_finding),
                "prior_authoritative_rca_finding_id": (
                    prior_rca_finding.finding_id
                    if prior_rca_finding is not None
                    else None
                ),
                "new_evidence_ids_since_prior": new_evidence_ids_since_prior,
                "root_cause_evidence_ids": root_cause_evidence_ids,
                "status": status,
                "hypothesis": hypothesis.to_dict(),
                "evidence_chain": [
                    {
                        "stage": finding.agent,
                        "claim": finding.summary,
                        "confidence": finding.confidence,
                        "evidence_ids": list(finding.evidence_ids),
                    }
                    for finding in findings
                ],
                "recommended_actions": actions,
                "ranked_candidates": ranked_candidates,
                "reasoning_engine": "hypothesis_v1",
                "hypothesis_candidate_generation": candidate_generation,
                "competition_requirement": competition_requirement,
                "competition_status": competition_status,
                "competition_type": competition_type,
                "competition_axes": competition_axes,
                "scope_assessment_status": scope_assessment_status,
                "competition_failure_reason": competition_failure_reason,
                "competition_gap_reason": competition_gap_reason,
                "candidate_semantic_profiles": candidate_semantic_profiles,
                "candidate_lineage": candidate_lineage,
                "adversarial_challenge_generation": challenge_generation,
                "candidate_challenges": [
                    challenge.to_dict() for challenge in candidate_challenges
                ],
                "alternative_lane_resolutions": [
                    item.to_dict() for item in alternative_lane_resolutions
                ],
                "alternative_search_status": alternative_search_status,
                "hypothesis_engine_result": engine_result,
                "conclusion_status": conclusion_status,
                "causal_chain_completeness": decision.get(
                    "causal_chain_completeness"
                ),
                "data_missing_evidence_ids": list(
                    decision.get("data_missing_evidence_ids", [])
                ),
                "blocking_data_missing_evidence_ids": list(
                    decision.get("blocking_data_missing_evidence_ids", [])
                ),
                "non_blocking_data_missing_evidence_ids": list(
                    decision.get("non_blocking_data_missing_evidence_ids", [])
                ),
                "evidence_synthesis": (
                    lane_first_evidence_synthesis
                    or engine_result.get(
                        "evidence_synthesis",
                        build_evidence_synthesis(
                            [
                                *[
                                    item
                                    for finding in findings
                                    for item in finding.evidence
                                ],
                                *context_evidence,
                            ]
                        ),
                    )
                ),
                "causal_evidence_gaps": list(
                    engine_result.get("causal_evidence_gaps", [])
                ),
                "candidate_comparison": dict(
                    engine_result.get("candidate_comparison", candidate_comparison or {})
                ),
                "confirmation_gate": dict(
                    decision.get("confirmation_gate", {})
                ),
                "impact_lot_gate": impact_lot_gate,
                "evidence": _merge_evidence_payload(findings, context_evidence),
            },
            warnings=list({warning.warning_id: warning for warning in warnings}.values()),
        )
