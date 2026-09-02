"""Supervisor orchestration for the pure Python Yield RCA workflow."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from yield_rca_core.causal_adversarial import (
    derive_alternative_lane_resolutions,
    derive_alternative_search_status,
)
from yield_rca_core.causal_chain import build_declared_unavailable_evidence
from yield_rca_core.causal_competition import select_diverse_lane_ids
from yield_rca_core.causal_investigation_models import (
    ActionValueAssessment,
    AlternativeLaneResolution,
    AlternativeLaneResolutionStatus,
    AlternativeSearchStatus,
    CandidateChallenge,
    CandidateCompetitionStatus,
    CandidateCompetitionType,
    CandidateSemanticProfile,
    CausalChainCompleteness,
    CausalLaneRecord,
    CompetitionRequirement,
    CompetitionTrace,
    InvestigationLaneStatus,
    LaneLifecycleStatus,
    LaneLifecycleTransition,
)
from yield_rca_core.causal_lane_lifecycle import (
    apply_active_lane_snapshot,
    lane_is_searchable,
    mark_lanes_challenged,
    reconcile_lane_inventory,
    transition_lane,
)
from yield_rca_core.causal_scope import explicit_module_limit_requested
from yield_rca_core.evidence_collection import EvidenceCollection
from yield_rca_core.evidence_models import Evidence
from yield_rca_core.improvement_agent import ImprovementAgent
from yield_rca_core.incident_evidence import build_incident_observation_evidence
from yield_rca_core.investigation_decision import (
    classify_investigation_gain,
    derive_investigation_gain_history,
)
from yield_rca_core.investigation_finalizer import (
    finalize_investigation,
    finalize_non_competition_llm_react_terminal,
    gate_planner_conclusion_level,
    validate_terminal_investigation_state,
)
from yield_rca_core.investigation_models import (
    ActionRecord,
    ConclusionLevel,
    DecisionType,
    EvidenceGapStatus,
    GoalStatus,
    IntentPlan,
    InvestigationAction,
    InvestigationGoal,
    InvestigationQuestion,
    PlannerDecision,
    PlannerDecisionOutcome,
    StopReason,
)
from yield_rca_core.investigation_policy import ACTION_REGISTRY, InvestigationPolicy
from yield_rca_core.llm_gateway import (
    LLMCallError,
    LLMClient,
    LLMOutputValidationError,
    LLMRequest,
    llm_calls_remaining,
)
from yield_rca_core.models import (
    AgentFinding,
    AgentKind,
    AgentMode,
    AgentTask,
    FindingKind,
    Hypothesis,
    InvestigationMode,
    LotDrivenRCAError,
    ModelValidationError,
    RCAJob,
    RCAState,
    TaskPlan,
    TaskStatus,
    Warning,
)
from yield_rca_core.next_action_planner import (
    LLM_REACT_EXECUTABLE_ACTION_KINDS,
    QwenNextActionPlanner,
    QwenNextActionPlannerError,
)
from yield_rca_core.planner_interface import (
    PROVENANCE_LLM_QWEN,
    ControlledPlannerAdapter,
    PlannerProposal,
    QwenPlannerAdapter,
    build_planner_context,
)
from yield_rca_core.question_evidence import QuestionEvidenceResolver
from yield_rca_core.rca_reasoning_agent import RCAReasoningAgent
from yield_rca_core.report_generator import ReportGenerator
from yield_rca_core.specialist_agents import DefectWATAgent, FDCAgent, KnowledgeAgent, MESAgent
from yield_rca_core.specialist_v2 import SpecialistV2Error, SpecialistV2Executor
from yield_rca_core.warning_policy import reconcile_current_warnings
from yield_rca_core.workflow_events import emit_workflow_event

SUPERVISOR_EXECUTABLE_AGENTS = frozenset(
    {
        AgentKind.MES.value,
        AgentKind.FDC.value,
        AgentKind.DEFECT_WAT.value,
        AgentKind.KNOWLEDGE.value,
        AgentKind.RCA_REASONING.value,
        AgentKind.IMPROVEMENT.value,
    }
)
_RCA_REASONING_MIN_REMAINING_LLM_CALLS = 3


def _llm_react_governance_requested(state: RCAState) -> bool:
    """Preserve Qwen-path publication governance after executor fallback."""

    requested_mode = state.execution_metadata.get("orchestration_requested_mode")
    if requested_mode is None:
        requested_mode = state.execution_metadata.get("orchestration_mode")
    return str(requested_mode or "").strip() == "llm_react"


def _record_finalizer_terminal_projection(state: RCAState) -> RCAState:
    """Record a Finalizer override without rewriting Planner-owned history."""

    last = state.planner_decisions[-1] if state.planner_decisions else None
    if (
        last is not None
        and last.decision_type == DecisionType.STOP.value
        and last.stop_reason == state.stop_reason
        and last.goal_status == state.goal_status
    ):
        return state
    superseded_stop = (
        last.to_dict()
        if last is not None and last.decision_type == DecisionType.STOP.value
        else None
    )
    metadata = {
        **state.execution_metadata,
        "terminal_stop_projection_applied": True,
        "terminal_stop_projection_trace": "execution_metadata_only",
        "terminal_stop_projected_by": "python_investigation_finalizer",
        "terminal_state_owner": "python_investigation_finalizer",
        **(
            {"superseded_terminal_planner_decision": superseded_stop}
            if superseded_stop is not None
            else {}
        ),
    }
    if last is None or last.decision_type != DecisionType.STOP.value:
        metadata.setdefault("planner_stop_proposed_by", "python_runtime")
    return replace(
        state,
        execution_metadata=metadata,
    )


def _rca_reasoning_round_budget_available(llm_client: LLMClient) -> bool:
    """Reserve Candidate generation, Challenge, and governed post-Action planning."""

    remaining = llm_calls_remaining(llm_client)
    return (
        remaining is None
        or remaining >= _RCA_REASONING_MIN_REMAINING_LLM_CALLS
    )

SPECIALIST_TOOL_ALLOWLISTS = {
    AgentKind.MES.value: [
        "find_affected_lots",
        "get_lot_context",
        "find_impact_lots",
        "analyze_lot_genealogy",
    ],
    AgentKind.FDC.value: [
        "analyze_parameter_shift",
        "find_ooc_events",
        "perform_basic_spc_analysis",
        "analyze_spc_evidence",
    ],
    AgentKind.DEFECT_WAT.value: ["summarize_defect_wat"],
    AgentKind.KNOWLEDGE.value: ["retrieve_similar_case"],
}


def _initial_context_evidence(
    job: RCAJob,
    *,
    include_incident_observations: bool = False,
) -> list[Evidence]:
    if job.declared_unavailable_sources and job.source_lot_id is None:
        raise ModelValidationError(
            "declared unavailable sources require a source Lot"
        )
    evidence = (
        list(
            build_declared_unavailable_evidence(
                job.declared_unavailable_sources,
                source_lot_id=job.source_lot_id,
            )
        )
        if job.declared_unavailable_sources and job.source_lot_id is not None
        else []
    )
    if include_incident_observations:
        evidence.extend(build_incident_observation_evidence(job))
    return evidence


def _is_llm_budget_exhaustion(error: LLMCallError) -> bool:
    """Recognize runtime call ceilings without treating them as provider faults."""

    return error.failure_category in {
        "call_limit",
        "evaluation_call_cap",
        "formal_blind_call_cap",
    }


def _persist_authoritative_lane_inventory(
    findings: list[AgentFinding],
    *,
    lanes: list[CausalLaneRecord],
    discovery_finding_id: str | None = None,
    raw_lane_candidate_count: int | None = None,
) -> list[AgentFinding]:
    """Project canonical Python Lane state into one Planner-visible MES Finding."""

    target_id = discovery_finding_id
    if target_id is None:
        target_id = next(
            (
                item.finding_id
                for item in reversed(findings)
                if item.agent == AgentKind.MES.value
                and item.details.get("lane_inventory_authoritative") is True
            ),
            None,
        )
    if target_id is None:
        return findings
    updated: list[AgentFinding] = []
    for item in findings:
        if item.finding_id != target_id:
            updated.append(item)
            continue
        details = {
            **item.details,
            "lane_candidates": [lane.to_dict() for lane in lanes],
            "lane_inventory_authoritative": True,
            "canonical_lane_count": len(lanes),
        }
        if raw_lane_candidate_count is not None:
            details["raw_lane_candidate_count"] = raw_lane_candidate_count
        updated.append(replace(item, details=details))
    return updated


def _update_causal_lane_state(state: RCAState, finding: AgentFinding) -> RCAState:
    """Persist bounded concrete Lane discovery from a MES Finding.

    Lane ranking is deterministic and owned by Python.  A Finding may expose
    factual ``lane_candidates`` produced by a Tool, but it cannot mark an
    alternative eliminated or make a conclusion supported.
    """

    raw_candidates = finding.details.get("lane_candidates")
    if finding.agent != AgentKind.MES.value or not isinstance(raw_candidates, list):
        return state
    known_evidence_ids = set(item.evidence_id for item in state.evidence)
    discovered: list[CausalLaneRecord] = []
    for raw in raw_candidates:
        if not isinstance(raw, dict):
            continue
        lane_id = str(raw.get("lane_id", "")).strip()
        if not lane_id:
            continue
        raw_evidence_ids = raw.get("evidence_ids", [])
        evidence_ids = (
            tuple(
                dict.fromkeys(
                    str(item).strip()
                    for item in raw_evidence_ids
                    if isinstance(item, str)
                    and item.strip()
                    and item.strip() in known_evidence_ids
                )
            )
            if isinstance(raw_evidence_ids, list | tuple)
            else ()
        )
        time_window = raw.get("time_window", [])
        try:
            record = CausalLaneRecord(
                lane_id=lane_id,
                operation=str(raw.get("operation", "")),
                operation_name=str(raw.get("operation_name", "")),
                module=str(raw.get("module", "")),
                equipment=str(raw.get("equipment", "")),
                chamber=str(raw.get("chamber", "")),
                recipe=str(raw.get("recipe", "")),
                parameter_scope=tuple(
                    str(item)
                    for item in raw.get("parameter_scope", [])
                    if str(item).strip()
                ),
                exposed_lot_ids=tuple(
                    str(item)
                    for item in raw.get("exposed_lot_ids", [])
                    if str(item).strip()
                ),
                time_window=(
                    tuple(str(item) for item in time_window)
                    if isinstance(time_window, list | tuple)
                    else ()
                ),
                initial_evidence_ids=evidence_ids,
                priority_score=float(raw.get("priority_score", 0.0)),
                investigation_status=InvestigationLaneStatus.EVIDENCE_COLLECTED.value,
                lifecycle_status=LaneLifecycleStatus.CREATED.value,
                last_transition_reason="Discovered from typed MES exposure Evidence.",
                lifecycle_history=(
                    LaneLifecycleTransition(
                        sequence=1,
                        lane_id=lane_id,
                        from_status=None,
                        to_status=LaneLifecycleStatus.CREATED.value,
                        reason="Discovered from typed MES exposure Evidence.",
                        evidence_ids=evidence_ids,
                    ),
                ),
            )
        except (TypeError, ValueError, ModelValidationError):
            # Malformed one-Lane payloads are isolated; other factual lanes
            # remain usable and this is not a Planner/orchestration failure.
            continue
        discovered.append(record)
    if not discovered:
        return state

    reconciled = reconcile_lane_inventory(state.causal_lanes, discovered)
    ordered = sorted(
        reconciled,
        key=lambda item: (-item.priority_score, item.lane_id),
    )
    searchable = [
        item
        for item in ordered
        if lane_is_searchable(item)
    ]
    selected_ids = select_diverse_lane_ids(
        [item.to_dict() for item in searchable],
        limit=3,
    )
    selected_id_set = set(selected_ids)
    searchable_by_id = {item.lane_id: item for item in searchable}
    active = [searchable_by_id[lane_id] for lane_id in selected_ids]
    overflow = [item for item in searchable if item.lane_id not in selected_id_set]
    active_ids = tuple(item.lane_id for item in active)
    overflow_ids = tuple(item.lane_id for item in overflow)
    terminal_eliminated = tuple(
        item.lane_id
        for item in ordered
        if item.investigation_status == InvestigationLaneStatus.ELIMINATED.value
    )
    terminal_blocked = tuple(
        item.lane_id
        for item in ordered
        if item.investigation_status == InvestigationLaneStatus.BLOCKED.value
    )
    unresolved_ids: tuple[str, ...] = ()
    ordered = list(
        apply_active_lane_snapshot(
            ordered,
            active_lane_ids=active_ids,
        )
    )
    trace = CompetitionTrace(
        active_lane_ids=active_ids,
        overflow_lane_ids=overflow_ids,
        represented_lane_ids=active_ids,
        unresolved_lane_ids=unresolved_ids,
        eliminated_lane_ids=terminal_eliminated,
        blocked_lane_ids=terminal_blocked,
        alternative_search_status=AlternativeSearchStatus.NOT_SEARCHED.value,
        competition_requirement=(
            state.competition_trace.competition_requirement
            if state.competition_trace is not None
            else CompetitionRequirement.NOT_EVALUATED.value
        ),
        competition_status=(
            state.competition_trace.competition_status
            if state.competition_trace is not None
            else CandidateCompetitionStatus.NOT_EVALUATED.value
        ),
        competition_type=(
            state.competition_trace.competition_type
            if state.competition_trace is not None
            else CandidateCompetitionType.NOT_EVALUATED.value
        ),
        competition_failure_reason=(
            state.competition_trace.competition_failure_reason
            if state.competition_trace is not None
            else None
        ),
        competition_gap_reason=(
            state.competition_trace.competition_gap_reason
            if state.competition_trace is not None
            else None
        ),
        candidate_semantic_profiles=(
            state.competition_trace.candidate_semantic_profiles
            if state.competition_trace is not None
            else ()
        ),
        candidate_lineage=(
            state.competition_trace.candidate_lineage
            if state.competition_trace is not None
            else ()
        ),
        challenge_round_count=(
            state.competition_trace.challenge_round_count
            if state.competition_trace is not None
            else 0
        ),
    )
    return replace(
        state,
        causal_lanes=ordered,
        competition_trace=trace,
        findings=_persist_authoritative_lane_inventory(
            state.findings,
            lanes=ordered,
            discovery_finding_id=finding.finding_id,
            raw_lane_candidate_count=len(raw_candidates),
        ),
    )


def _selected_lane(
    finding: AgentFinding,
    lane_id: object,
) -> dict[str, Any] | None:
    normalized_lane_id = str(lane_id or "").strip()
    if not normalized_lane_id:
        return None
    raw_lanes = finding.details.get("lane_candidates", [])
    if not isinstance(raw_lanes, list):
        return None
    return next(
        (
            dict(item)
            for item in raw_lanes
            if isinstance(item, dict)
            and str(item.get("lane_id", "")).strip() == normalized_lane_id
        ),
        None,
    )


def _source_anchored_lot_scope(
    source_lot_id: object,
    comparison_lot_ids: object,
) -> list[str]:
    """Keep the immutable source Lot while adding bounded comparison Lots."""

    source = str(source_lot_id or "").strip().upper()
    comparisons = (
        [
            str(item).strip().upper()
            for item in comparison_lot_ids
            if str(item).strip()
        ]
        if isinstance(comparison_lot_ids, list | tuple)
        else []
    )
    return list(
        dict.fromkeys(
            [
                *([source] if source else []),
                *comparisons,
            ]
        )
    )


def _update_competition_state(state: RCAState, finding: AgentFinding) -> RCAState:
    """Persist Python-validated adversarial challenge state from RCA Finding."""

    if finding.agent != AgentKind.RCA_REASONING.value:
        return state
    raw_generation = finding.details.get("adversarial_challenge_generation", {})
    raw_challenges = finding.details.get("candidate_challenges", [])
    if not isinstance(raw_generation, dict) or not isinstance(raw_challenges, list):
        return state
    raw_candidate_generation = finding.details.get(
        "hypothesis_candidate_generation",
        {},
    )
    if not isinstance(raw_candidate_generation, dict):
        raw_candidate_generation = {}
    candidate_competition_status = str(
        finding.details.get(
            "competition_status",
            raw_candidate_generation.get(
                "competition_status",
                CandidateCompetitionStatus.NOT_EVALUATED.value,
            ),
        )
    )
    if (
        raw_generation.get("source") == "not_requested"
        and candidate_competition_status
        == CandidateCompetitionStatus.NOT_EVALUATED.value
    ):
        return state
    challenges: list[CandidateChallenge] = []
    for raw in raw_challenges:
        if not isinstance(raw, dict):
            continue
        try:
            challenges.append(CandidateChallenge.from_dict(raw))
        except (TypeError, ValueError, ModelValidationError):
            # RCA-level validation already isolated malformed Qwen output; keep
            # the persisted state safe if a legacy Finding is replayed.
            continue
    previous_trace = state.competition_trace
    raw_resolutions = finding.details.get("alternative_lane_resolutions", [])
    lane_resolutions: list[AlternativeLaneResolution] = []
    if isinstance(raw_resolutions, list):
        for raw in raw_resolutions:
            if not isinstance(raw, dict):
                continue
            try:
                lane_resolutions.append(AlternativeLaneResolution.from_dict(raw))
            except (TypeError, ValueError, ModelValidationError):
                continue
    searchable_before = [
        lane
        for lane in state.causal_lanes
        if lane_is_searchable(lane)
    ]
    active_before = list(
        select_diverse_lane_ids(
            [lane.to_dict() for lane in searchable_before],
            limit=3,
        )
    )
    eliminated_before = [
        lane.lane_id
        for lane in state.causal_lanes
        if lane.investigation_status == InvestigationLaneStatus.ELIMINATED.value
    ]
    blocked_before = [
        lane.lane_id
        for lane in state.causal_lanes
        if lane.investigation_status == InvestigationLaneStatus.BLOCKED.value
    ]
    if not lane_resolutions:
        lane_resolutions = list(
            derive_alternative_lane_resolutions(
                challenges=challenges,
                active_lane_ids=active_before,
                eliminated_lane_ids=eliminated_before,
                blocked_lane_ids=blocked_before,
            )
        )
    previous_resolutions = {
        item.lane_id: item
        for item in (
            previous_trace.lane_resolutions if previous_trace is not None else ()
        )
    }
    merged_resolutions: list[AlternativeLaneResolution] = []
    current_resolution_ids = {item.lane_id for item in lane_resolutions}
    for item in lane_resolutions:
        previous = previous_resolutions.get(item.lane_id)
        if previous is not None and not item.evidence_ids:
            item = replace(
                item,
                candidate_id=item.candidate_id or previous.candidate_id,
                evidence_ids=previous.evidence_ids,
                distinguishing_gap_ids=(
                    item.distinguishing_gap_ids
                    or previous.distinguishing_gap_ids
                ),
            )
        merged_resolutions.append(item)
    merged_resolutions.extend(
        item
        for lane_id, item in previous_resolutions.items()
        if lane_id not in current_resolution_ids
    )
    lane_resolutions = merged_resolutions
    resolutions_by_lane = {item.lane_id: item for item in lane_resolutions}
    causal_lanes: list[CausalLaneRecord] = []
    for lane in state.causal_lanes:
        resolution = resolutions_by_lane.get(lane.lane_id)
        if resolution is None or resolution.status != (
            AlternativeLaneResolutionStatus.ELIMINATED.value
        ):
            causal_lanes.append(lane)
            continue
        reason = resolution.reason or "Python resolved this alternative Lane."
        causal_lanes.append(
            transition_lane(
                lane,
                lifecycle_status=LaneLifecycleStatus.ELIMINATED.value,
                investigation_status=InvestigationLaneStatus.ELIMINATED.value,
                reason=reason,
                pruned_reason=reason,
                evidence_ids=resolution.evidence_ids,
            )
        )
    challenge_evidence_by_lane: dict[str, tuple[str, ...]] = {}
    challenged_lane_ids: list[str] = []
    for challenge in challenges:
        lane_id = challenge.evidence_probe_lane_id
        if lane_id is None:
            continue
        challenged_lane_ids.append(lane_id)
        challenge_evidence_by_lane[lane_id] = tuple(
            dict.fromkeys(
                [
                    *challenge.supporting_evidence_ids,
                    *challenge.contradicting_evidence_ids,
                    *challenge.unexplained_precursor_evidence_ids,
                ]
            )
        )
    causal_lanes = list(
        mark_lanes_challenged(
            causal_lanes,
            challenged_lane_ids=challenged_lane_ids,
            evidence_ids_by_lane=challenge_evidence_by_lane,
        )
    )
    searchable_after = [
        lane
        for lane in causal_lanes
        if lane_is_searchable(lane)
    ]
    active_ids = select_diverse_lane_ids(
        [lane.to_dict() for lane in searchable_after],
        limit=3,
    )
    known_ids = {lane.lane_id for lane in causal_lanes}
    overflow_ids = tuple(
        lane.lane_id
        for lane in causal_lanes
        if lane.lane_id not in set(active_ids)
        and lane_is_searchable(lane)
    )
    alternative_ids = tuple(
        challenge.strongest_alternative_lane_id
        for challenge in challenges
        if challenge.strongest_alternative_lane_id in known_ids
    )
    represented_ids = tuple(
        dict.fromkeys(
            [
                *active_ids,
                *alternative_ids,
                *[item.lane_id for item in lane_resolutions],
            ]
        )
    )
    resolution_ids = tuple(
        dict.fromkeys(
            evidence_id
            for resolution in lane_resolutions
            for evidence_id in resolution.evidence_ids
        )
    )
    previous_count = previous_trace.challenge_round_count if previous_trace else 0
    eliminated_ids = tuple(
        lane.lane_id
        for lane in causal_lanes
        if lane.investigation_status == InvestigationLaneStatus.ELIMINATED.value
    )
    blocked_ids = tuple(
        lane.lane_id
        for lane in causal_lanes
        if lane.investigation_status == InvestigationLaneStatus.BLOCKED.value
    )
    causal_lanes = list(
        apply_active_lane_snapshot(
            causal_lanes,
            active_lane_ids=active_ids,
        )
    )
    alternative_search_status = derive_alternative_search_status(
        challenges=challenges,
        matrices=(),
        active_lane_ids=active_before,
        eliminated_lane_ids=eliminated_before,
        blocked_lane_ids=blocked_before,
        lane_resolutions=lane_resolutions,
    )
    unresolved_resolution_statuses = {
        AlternativeLaneResolutionStatus.RETAINED.value,
        AlternativeLaneResolutionStatus.UNRESOLVED.value,
        AlternativeLaneResolutionStatus.NON_DISCRIMINATIVE.value,
    }
    unresolved_ids = tuple(
        dict.fromkeys(
            [
                *(
                    []
                    if alternative_search_status
                    == AlternativeSearchStatus.ALTERNATIVES_ELIMINATED.value
                    else [
                        item.lane_id
                        for item in lane_resolutions
                        if item.status in unresolved_resolution_statuses
                    ]
                ),
            ]
        )
    )
    competition_requirement = str(
        finding.details.get(
            "competition_requirement",
            raw_candidate_generation.get(
                "competition_requirement",
                previous_trace.competition_requirement
                if previous_trace is not None
                else CompetitionRequirement.NOT_EVALUATED.value,
            ),
        )
    )
    competition_status = str(
        finding.details.get(
            "competition_status",
            raw_candidate_generation.get(
                "competition_status",
                previous_trace.competition_status
                if previous_trace is not None
                else CandidateCompetitionStatus.NOT_EVALUATED.value,
            ),
        )
    )
    competition_type = str(
        finding.details.get(
            "competition_type",
            raw_candidate_generation.get(
                "competition_type",
                previous_trace.competition_type
                if previous_trace is not None
                else CandidateCompetitionType.NOT_EVALUATED.value,
            ),
        )
    )
    raw_competition_axes = finding.details.get(
        "competition_axes",
        raw_candidate_generation.get(
            "competition_axes",
            list(previous_trace.competition_axes)
            if previous_trace is not None
            else [],
        ),
    )
    competition_axes = (
        tuple(str(item) for item in raw_competition_axes if str(item))
        if isinstance(raw_competition_axes, list | tuple)
        else ()
    )
    scope_assessment_status = str(
        finding.details.get(
            "scope_assessment_status",
            raw_candidate_generation.get(
                "scope_assessment_status",
                previous_trace.scope_assessment_status
                if previous_trace is not None
                else "not_evaluated",
            ),
        )
    )
    raw_failure_reason = finding.details.get(
        "competition_failure_reason",
        raw_candidate_generation.get("competition_failure_reason"),
    )
    competition_failure_reason = (
        str(raw_failure_reason) if raw_failure_reason is not None else None
    )
    raw_gap_reason = finding.details.get(
        "competition_gap_reason",
        raw_candidate_generation.get(
            "competition_gap_reason",
            previous_trace.competition_gap_reason
            if previous_trace is not None
            else None,
        ),
    )
    competition_gap_reason = (
        str(raw_gap_reason) if raw_gap_reason is not None else None
    )
    raw_semantic_profiles = finding.details.get(
        "candidate_semantic_profiles",
        raw_candidate_generation.get("candidate_semantic_profiles", []),
    )
    candidate_semantic_profiles: list[CandidateSemanticProfile] = []
    if isinstance(raw_semantic_profiles, list):
        for raw_profile in raw_semantic_profiles:
            if not isinstance(raw_profile, dict):
                continue
            try:
                candidate_semantic_profiles.append(
                    CandidateSemanticProfile.from_dict(raw_profile)
                )
            except (TypeError, ValueError, ModelValidationError):
                # Semantic metadata is candidate-adjacent audit state. An
                # invalid profile cannot invalidate the Candidate or State.
                continue
    raw_lineage = finding.details.get(
        "candidate_lineage",
        raw_candidate_generation.get("candidate_lineage", []),
    )
    current_lineage = (
        [dict(item) for item in raw_lineage if isinstance(item, dict)]
        if isinstance(raw_lineage, list)
        else []
    )
    candidate_lineage = tuple(
        [
            *(previous_trace.candidate_lineage if previous_trace is not None else ()),
            *current_lineage,
        ]
    )
    trace = CompetitionTrace(
        active_lane_ids=active_ids,
        overflow_lane_ids=overflow_ids,
        represented_lane_ids=represented_ids,
        unresolved_lane_ids=unresolved_ids,
        eliminated_lane_ids=eliminated_ids,
        blocked_lane_ids=blocked_ids,
        lane_resolutions=tuple(lane_resolutions),
        alternative_search_status=alternative_search_status,
        competition_requirement=competition_requirement,
        competition_status=competition_status,
        competition_type=competition_type,
        competition_axes=competition_axes,
        scope_assessment_status=scope_assessment_status,
        competition_failure_reason=competition_failure_reason,
        competition_gap_reason=competition_gap_reason,
        candidate_semantic_profiles=tuple(candidate_semantic_profiles),
        candidate_lineage=candidate_lineage,
        challenge_round_count=previous_count + 1,
        resolution_evidence_ids=resolution_ids,
    )
    return replace(
        state,
        causal_lanes=causal_lanes,
        candidate_challenges=[*state.candidate_challenges, *challenges],
        competition_trace=trace,
        findings=_persist_authoritative_lane_inventory(
            state.findings,
            lanes=causal_lanes,
        ),
    )


def _update_causal_chain_state(state: RCAState, finding: AgentFinding) -> RCAState:
    """Persist the RCA Finding's Python-derived causal-chain status."""

    if finding.agent != AgentKind.RCA_REASONING.value:
        return state
    raw_status = finding.details.get("causal_chain_completeness")
    if raw_status is None:
        raw_gate = finding.details.get("confirmation_gate", {})
        if isinstance(raw_gate, dict):
            raw_status = raw_gate.get("causal_chain_completeness")
    if raw_status is None:
        return state
    try:
        status = CausalChainCompleteness(str(raw_status)).value
    except ValueError:
        # A malformed diagnostic must not make a valid RCA Finding
        # unpersistable; the Confirmation Gate remains the source of truth.
        return state
    return replace(state, causal_chain_completeness=status)


def _reasoning_decision_signature(finding: AgentFinding | None) -> str | None:
    """Return a stable signature of decision state, excluding audit counters."""

    if finding is None or finding.agent != AgentKind.RCA_REASONING.value:
        return None
    details = finding.details
    payload = {
        "ranked_candidates": details.get("ranked_candidates", []),
        "conclusion_status": details.get("conclusion_status"),
        "competition_requirement": details.get("competition_requirement"),
        "competition_status": details.get("competition_status"),
        "competition_type": details.get("competition_type"),
        "competition_failure_reason": details.get(
            "competition_failure_reason"
        ),
        "confirmation_gate": details.get("confirmation_gate", {}),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def _llm_call_fallback_diagnostics(error: LLMCallError) -> dict[str, Any]:
    """Expose only bounded, credential-free provider failure facts."""

    diagnostics: dict[str, Any] = {
        "orchestration_fallback_failure_category": error.failure_category,
        "orchestration_fallback_call_attempt_count": error.call_attempt_count,
    }
    optional_values = {
        "orchestration_fallback_status_code": error.status_code,
        "orchestration_fallback_provider_code": error.provider_code,
        "orchestration_fallback_provider_message": error.provider_message,
        "orchestration_fallback_request_id": error.request_id,
    }
    diagnostics.update(
        {key: value for key, value in optional_values.items() if value is not None}
    )
    return diagnostics


def _open_question_gaps(questions: list[InvestigationQuestion]) -> list[str]:
    return [
        question.question_id
        for question in questions
        if question.status == EvidenceGapStatus.OPEN.value
    ]


def _reconcile_satisfied_questions(state: RCAState) -> list[InvestigationQuestion]:
    if not state.investigation_questions or not state.evidence:
        return list(state.investigation_questions)
    evidence_ids = sorted(item.evidence_id for item in state.evidence)
    answer = " ".join(
        finding.summary.strip()
        for finding in state.findings
        if finding.summary.strip()
    )
    if not answer:
        answer = (
            "The controlled fallback completed the investigation with "
            f"Evidence IDs {', '.join(evidence_ids)}."
        )
    return [
        (
            replace(
                question,
                status=EvidenceGapStatus.CLOSED.value,
                answer=answer,
                evidence_ids=evidence_ids,
                unavailable_reason=None,
            )
            if question.status == EvidenceGapStatus.OPEN.value
            else question
        )
        for question in state.investigation_questions
    ]


@dataclass(frozen=True)
class ReactStopBundle:
    """Terminal input for the unified React loop.

    ``origin`` distinguishes a planner-proposed stop from a runtime
    termination; both flow through the same terminal governance. Runtime
    terminations carry the pre-existing audit outcome so ``planner_decisions``
    and telemetry keep their pre-refactor content; they are never expressed
    as planner proposals.
    """

    origin: str  # "planner" | "runtime"
    reason: str
    goal_status: str
    proposed_conclusion_level: str
    stop_reason: str | None
    evidence_gaps: list[str]
    decision_id: str | None = None
    decision_proposed_by: str | None = None
    question_updates_source: str | None = None
    has_question_updates: bool = False
    action_value_assessments: list[ActionValueAssessment] | None = None
    audit_outcome: PlannerDecisionOutcome | None = None

    @classmethod
    def from_proposal(cls, proposal: PlannerProposal) -> ReactStopBundle:
        return cls(
            origin="planner",
            reason=proposal.reason,
            goal_status=(
                proposal.proposed_goal_status
                if proposal.proposed_goal_status is not None
                else GoalStatus.BLOCKED.value
            ),
            proposed_conclusion_level=proposal.proposed_conclusion_level,
            stop_reason=proposal.stop_reason,
            evidence_gaps=list(proposal.evidence_gaps),
            decision_id=proposal.decision_id,
            decision_proposed_by=proposal.decision_proposed_by,
            question_updates_source=proposal.question_updates_source,
            has_question_updates=bool(proposal.question_updates),
            # Governed finalize receives the planner's assessments; controlled
            # proposals keep the finalize default (State's latest assessments)
            # exactly as the pre-refactor controlled terminal did.
            action_value_assessments=(
                list(proposal.action_value_assessments)
                if proposal.proposed_by == PROVENANCE_LLM_QWEN
                else None
            ),
            # Rebuild the pre-refactor audit outcome only for planner
            # outcomes that carry one; controlled proposals never inject
            # PlannerDecision records into planner_decisions.
            audit_outcome=(
                Supervisor._planner_audit_outcome(proposal)
                if proposal.decision_proposed_by is not None
                else None
            ),
        )


@dataclass(frozen=True)
class ReactExecutionProfile:
    """Explicit per-mode behavior for the unified React loop.

    The shared loop never infers mode from planner provenance or from the
    concrete native decision type; every mode difference is an explicit
    profile field selected at entry and swapped explicitly at fallback.
    """

    name: str
    stop_event: str
    dispatch: Callable[..., AgentFinding]
    stop_event_payload: Callable[[RCAState, ReactStopBundle], dict[str, Any]]
    preflight_termination: Callable[[RCAState], PlannerDecisionOutcome | None] | None
    act_gate: Callable[..., ReactStopBundle | PlannerProposal | None] | None
    stop_provenance_metadata: (
        Callable[[RCAState, ReactStopBundle], dict[str, Any]] | None
    )
    records_planner_outcome: bool
    emits_decision_events: bool
    caps_conclusion: bool
    reconciles_satisfied_questions: bool
    pre_finalize_trace_metadata: bool
    finalize_mode: str  # "always" | "governance_gated"


class SupervisorExecutionError(RuntimeError):
    """Raised when a TaskPlan cannot be executed to completion."""

    def __init__(
        self,
        message: str,
        *,
        state: RCAState | None = None,
        error_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.state = state
        self.error_code = error_code


def _replace_task_status(plan: TaskPlan, task_id: str, status: str) -> TaskPlan:
    tasks = [
        replace(task, status=status) if task.task_id == task_id else task for task in plan.tasks
    ]
    return TaskPlan(
        plan_id=plan.plan_id,
        objective=plan.objective,
        tasks=tasks,
        schema_version=plan.schema_version,
    )


def _finding_for_task(
    state: RCAState,
    task_id: str,
    *,
    expected_agent: str | None = None,
) -> AgentFinding:
    finding = state.finding_for_task(task_id)
    if finding is None:
        raise SupervisorExecutionError(
            f"expected a finding for completed task {task_id!r}",
            state=state,
        )
    if expected_agent is not None and finding.agent != expected_agent:
        raise SupervisorExecutionError(
            f"task {task_id!r} produced {finding.agent}, expected {expected_agent}",
            state=state,
        )
    return finding


def _input_findings(state: RCAState, task: AgentTask) -> list[AgentFinding]:
    raw_task_ids = task.inputs.get("finding_task_ids")
    if raw_task_ids is None:
        task_ids = list(task.depends_on)
    elif isinstance(raw_task_ids, list) and all(
        isinstance(item, str) and item.strip() for item in raw_task_ids
    ):
        task_ids = list(raw_task_ids)
    else:
        raise SupervisorExecutionError(
            f"task {task.task_id!r} has invalid finding_task_ids",
            state=state,
        )
    if len(task_ids) != len(set(task_ids)):
        raise SupervisorExecutionError(
            f"task {task.task_id!r} has duplicate finding_task_ids",
            state=state,
        )
    return [_finding_for_task(state, task_id) for task_id in task_ids]


def _finding_for_agent(
    findings: list[AgentFinding],
    agent: str,
    *,
    state: RCAState,
) -> AgentFinding:
    matches = [finding for finding in findings if finding.agent == agent]
    if len(matches) != 1:
        raise SupervisorExecutionError(
            f"expected exactly one selected {agent} finding, found {len(matches)}",
            state=state,
        )
    return matches[0]


def _latest_finding_for_agent(
    findings: list[AgentFinding],
    agent: str,
    *,
    state: RCAState,
) -> AgentFinding:
    matches = [finding for finding in findings if finding.agent == agent]
    if not matches:
        raise SupervisorExecutionError(
            f"expected at least one selected {agent} finding",
            state=state,
        )
    return matches[-1]


def _merge_warnings(
    existing: list[Warning],
    incoming: list[Warning],
    *,
    current_findings: list[AgentFinding] | None = None,
) -> list[Warning]:
    if current_findings is None:
        warnings_by_id = {item.warning_id: item for item in existing}
        for item in incoming:
            warnings_by_id[item.warning_id] = item
        return list(warnings_by_id.values())
    return reconcile_current_warnings(
        existing,
        incoming,
        current_findings=current_findings,
    )


def _time_window(inputs: dict[str, Any], job: RCAJob) -> tuple[str | None, str | None]:
    raw_window = inputs.get("time_window", job.time_window)
    if not isinstance(raw_window, dict):
        return None, None
    start = raw_window.get("start") or raw_window.get("start_date")
    end = raw_window.get("end") or raw_window.get("end_date")
    return (str(start) if start else None, str(end) if end else None)


def _knowledge_query(
    state: RCAState,
    findings: list[AgentFinding],
    lane_scope: dict[str, Any] | None = None,
) -> tuple[str, str, str]:
    mes_finding = _finding_for_agent(findings, AgentKind.MES.value, state=state)
    fdc_finding = _finding_for_agent(findings, AgentKind.FDC.value, state=state)
    defect_finding = _finding_for_agent(
        findings,
        AgentKind.DEFECT_WAT.value,
        state=state,
    )

    bounded_lane_scope = lane_scope or {}
    target_operation = str(
        bounded_lane_scope.get("operation")
        or bounded_lane_scope.get("operation_no")
        or mes_finding.details.get("target_operation_no", "")
    )
    raw_operation_rows = mes_finding.details.get("operation_commonality", [])
    operation_rows = (
        [item for item in raw_operation_rows if isinstance(item, dict)]
        if isinstance(raw_operation_rows, list)
        else []
    )
    operation: dict[str, Any] = next(
        (item for item in operation_rows if str(item.get("operation_no", "")) == target_operation),
        {},
    )
    module = str(operation.get("module", "")).strip()
    raw_commonality = mes_finding.details.get("target_commonality", {})
    commonality = raw_commonality if isinstance(raw_commonality, dict) else {}
    equipment_id = str(
        bounded_lane_scope.get("equipment")
        or bounded_lane_scope.get("equipment_id")
        or commonality.get("equipment_id", "")
    )
    equipment_type = equipment_id.split("_", maxsplit=1)[0] if equipment_id else ""

    terms: list[str] = [module, target_operation, equipment_id]
    raw_parameters = bounded_lane_scope.get("parameters", "")
    if isinstance(raw_parameters, str):
        terms.extend(
            item.strip().replace("_", " ")
            for item in raw_parameters.split(",")
            if item.strip()
        )
    terms.extend(
        str(item.get("parameter_name", "")).replace("_", " ")
        for item in fdc_finding.details.get("parameter_summary", [])
    )
    terms.extend(
        str(item).replace("_", " ") for item in defect_finding.details.get("defect_counts", {})
    )
    terms.extend(
        str(item).replace("_", " ") for item in defect_finding.details.get("wat_fail_modes", {})
    )
    terms.extend(
        str(item.get("metric_name", "")).replace("_", " ")
        for item in defect_finding.details.get("metrology_summaries", [])
    )
    terms.extend(
        (
            f"{item.get('source_recipe_id', '')} "
            f"{item.get('source_recipe_version', '')} recipe change"
        )
        for item in mes_finding.details.get("recipe_changes", [])
    )
    query = " ".join(item for item in terms if item).strip() or state.job.user_query
    return query, module, equipment_type


def _knowledge_observation_context(
    state: RCAState,
    findings: list[AgentFinding],
) -> dict[str, Any]:
    """Project trusted operational findings into non-causal observation facts."""

    mes_finding = _finding_for_agent(findings, AgentKind.MES.value, state=state)
    defect_finding = _finding_for_agent(
        findings,
        AgentKind.DEFECT_WAT.value,
        state=state,
    )
    raw_commonality = mes_finding.details.get("target_commonality", {})
    commonality = raw_commonality if isinstance(raw_commonality, dict) else {}
    defect_counts = defect_finding.details.get("defect_counts", {})
    symptom_types = (
        tuple(str(item) for item in defect_counts)
        if isinstance(defect_counts, dict)
        else ()
    )
    detected_at = max(
        (
            item.timestamp
            for item in defect_finding.evidence
            if item.timestamp is not None
        ),
        default="",
    )
    return {
        "source_lot_id": state.job.source_lot_id or "",
        "product_id": state.job.product_id or "",
        "detected_operation": str(
            mes_finding.details.get("target_operation_no", "")
        ),
        "detected_equipment_id": str(commonality.get("equipment_id", "")),
        "detected_at": detected_at,
        "symptom_types": symptom_types,
    }


def _knowledge_explicit_module_limit(
    state: RCAState,
    module: str,
    action_inputs: dict[str, Any] | None = None,
) -> bool:
    """Validate, rather than trust, a Planner-proposed Module restriction."""

    if action_inputs is not None and not isinstance(
        action_inputs.get("explicit_module_limit", False),
        bool,
    ):
        return False
    return explicit_module_limit_requested(
        state.job.user_query,
        module,
    )


def _legacy_preliminary_candidates(
    findings: list[AgentFinding],
    *,
    state: RCAState,
) -> list[dict[str, Any]]:
    mes_finding = _finding_for_agent(findings, AgentKind.MES.value, state=state)
    fdc_finding = _finding_for_agent(findings, AgentKind.FDC.value, state=state)
    defect_finding = _finding_for_agent(findings, AgentKind.DEFECT_WAT.value, state=state)
    candidates: dict[str, dict[str, Any]] = {}

    def add(root_cause: str, basis: str, evidence_ids: list[str]) -> None:
        normalized = root_cause.strip()
        if not normalized:
            return
        candidates.setdefault(
            normalized,
            {
                "root_cause": normalized,
                "basis": basis,
                "evidence_ids": list(dict.fromkeys(evidence_ids)),
            },
        )

    raw_commonality = mes_finding.details.get("target_commonality", {})
    commonality = raw_commonality if isinstance(raw_commonality, dict) else {}
    chamber_id = str(commonality.get("chamber_id", "")).strip()
    parameter_summary = {
        str(item.get("parameter_name", "")): item
        for item in fdc_finding.details.get("parameter_summary", [])
        if isinstance(item, dict)
    }
    signature_rules = (
        ("slurry_flow", "slurry delivery degradation"),
        ("carrier_pressure", "carrier pressure instability"),
        ("wf6_flow", "WF6 delivery degradation"),
        ("deposition_rate", "deposition rate excursion"),
    )
    for parameter_name, failure_mode in signature_rules:
        delta = float(parameter_summary.get(parameter_name, {}).get("avg_delta_percent", 0.0))
        if chamber_id and delta <= -5.0:
            add(
                f"{chamber_id} {failure_mode}",
                "legacy_fdc_signature",
                list(mes_finding.evidence_ids) + list(fdc_finding.evidence_ids),
            )
            break

    recipe_changes = [
        item for item in mes_finding.details.get("recipe_changes", []) if isinstance(item, dict)
    ]
    if recipe_changes:
        change = recipe_changes[0]
        recipe_id = str(change.get("source_recipe_id", "")).strip()
        recipe_version = str(change.get("source_recipe_version", "")).strip()
        if recipe_id and recipe_version:
            add(
                f"{recipe_id} {recipe_version} recipe version change",
                "legacy_recipe_change",
                list(mes_finding.evidence_ids) + list(defect_finding.evidence_ids),
            )

    return list(candidates.values())[:3]


def _review_specialist_finding(
    finding: AgentFinding,
    *,
    llm_client: LLMClient | None,
    agent_mode: str,
    prompt_version: str,
) -> AgentFinding:
    if agent_mode == AgentMode.DETERMINISTIC.value:
        return finding
    if llm_client is None:
        raise SupervisorExecutionError("LLM Specialist review requires an LLM client")
    try:
        response = llm_client.complete_json(
            LLMRequest(
                agent=finding.agent,
                prompt_name="specialist",
                prompt_version=prompt_version,
                payload={
                    "agent": finding.agent,
                    "allowed_tools": SPECIALIST_TOOL_ALLOWLISTS[finding.agent],
                    "deterministic_finding": finding.to_dict(),
                },
            )
        )
    except LLMCallError as exc:
        if exc.failure_category in {
            "call_limit",
            "formal_blind_call_cap",
        }:
            raise
        fallback_details: dict[str, Any] = {
            "source": "deterministic_fallback",
            "fallback_reason": "llm_call_failed",
            "failure_category": exc.failure_category,
        }
        if exc.status_code is not None:
            fallback_details["status_code"] = exc.status_code
        if exc.provider_code is not None:
            fallback_details["provider_code"] = exc.provider_code
        if exc.request_id is not None:
            fallback_details["provider_request_id"] = exc.request_id
        warning = Warning(
            warning_id=(
                f"WARN_{finding.agent.upper()}_LLM_REVIEW_FALLBACK"
            ),
            message=(
                f"The optional {finding.agent} LLM review was unavailable; "
                "the deterministic Tool-derived Finding was preserved."
            ),
            evidence_ids=list(finding.evidence_ids),
        )
        return replace(
            finding,
            details={
                **finding.details,
                "agent_mode": agent_mode,
                "llm_prompt_version": prompt_version,
                "engineering_interpretation": finding.summary,
                "specialist_review": fallback_details,
            },
            warnings=[*finding.warnings, warning],
        )
    try:
        summary = str(response.data["summary"]).strip()
        confidence = float(response.data["confidence"])
        evidence_ids = [str(item) for item in response.data["evidence_ids"]]
        interpretation = str(response.data["engineering_interpretation"]).strip()
    except (KeyError, TypeError, ValueError) as exc:
        raise LLMOutputValidationError(
            "Specialist returned an invalid AgentFinding review"
        ) from exc
    if not summary or not interpretation:
        raise LLMOutputValidationError("Specialist summary and interpretation must not be empty")
    if set(evidence_ids) != set(finding.evidence_ids):
        raise LLMOutputValidationError(
            "Specialist response must preserve exactly the Tool evidence_ids"
        )
    return AgentFinding(
        finding_id=finding.finding_id,
        task_id=finding.task_id,
        agent=finding.agent,
        finding_kind=finding.finding_kind,
        summary=summary,
        confidence=confidence,
        evidence_ids=list(finding.evidence_ids),
        evidence=list(finding.evidence),
        details={
            **finding.details,
            "agent_mode": agent_mode,
            "llm_prompt_version": prompt_version,
            "engineering_interpretation": interpretation,
        },
        warnings=list(finding.warnings),
    )


@dataclass(frozen=True)
class Supervisor:
    """Execute an existing TaskPlan and maintain an immutable RCAState."""

    mes_agent: MESAgent
    fdc_agent: FDCAgent
    defect_wat_agent: DefectWATAgent
    knowledge_agent: KnowledgeAgent
    rca_reasoning_agent: RCAReasoningAgent
    improvement_agent: ImprovementAgent
    report_generator: ReportGenerator
    llm_client: LLMClient | None = None
    agent_mode: str = AgentMode.DETERMINISTIC.value
    specialist_prompt_version: str = "v1"
    specialist_v2_executor: SpecialistV2Executor | None = None

    def execute_controlled(
        self,
        job: RCAJob,
        goal: InvestigationGoal,
        *,
        policy: InvestigationPolicy | None = None,
        tool_latencies: list[dict[str, str | float]] | None = None,
        orchestration_requested_mode: str | None = None,
    ) -> RCAState:
        """Run a bounded observation-action loop without changing fixed-plan execution."""
        state = RCAState(
            job=replace(job, status=TaskStatus.RUNNING.value),
            investigation_goal=goal,
            evidence=_initial_context_evidence(job),
            execution_metadata=(
                {"orchestration_requested_mode": orchestration_requested_mode}
                if orchestration_requested_mode is not None
                else {}
            ),
        )
        return self._react_loop(
            state,
            goal,
            profile=self._controlled_react_profile(goal=goal),
            planner=ControlledPlannerAdapter(policy),
            fallback_policy=None,
            tool_latencies=tool_latencies,
        )

    def _react_loop(
        self,
        state: RCAState,
        goal: InvestigationGoal,
        *,
        profile: ReactExecutionProfile,
        planner: Any,
        fallback_policy: InvestigationPolicy | None,
        tool_latencies: list[dict[str, str | float]] | None,
    ) -> RCAState:
        """The single shared ReAct orchestration loop.

        The loop owns orchestration only: it builds the planner context,
        asks the active planner for one proposal, routes ACT/STOP, records
        findings through the shared recorder, and routes terminals through
        the shared Finalizer governance. Mode differences (planner
        preflight, dispatch strategy, budget preflight, stop-event naming,
        decision recording) live in the explicit execution profile; fallback
        swaps both the planner and the profile explicitly inside this loop
        and never jumps to a second orchestration loop.
        """

        observed_tool_latencies = tool_latencies if tool_latencies is not None else []
        while True:
            runtime_outcome = (
                profile.preflight_termination(state)
                if profile.preflight_termination is not None
                else None
            )
            stop: ReactStopBundle | None = None
            proposal: PlannerProposal | None = None
            if runtime_outcome is not None:
                stop = self._runtime_stop_bundle(runtime_outcome)
            else:
                try:
                    proposal = planner.decide(
                        build_planner_context(
                            goal=goal,
                            state=state,
                            tool_call_count=len(observed_tool_latencies),
                        )
                    )
                except (QwenNextActionPlannerError, LLMCallError) as exc:
                    if isinstance(exc, LLMCallError) and _is_llm_budget_exhaustion(exc):
                        return self._react_llm_budget_terminal(state, exc)
                    reason = (
                        "qwen_next_action_output_invalid"
                        if isinstance(exc, QwenNextActionPlannerError)
                        else "qwen_next_action_call_failed"
                    )
                    validation_diagnostics = (
                        {
                            "orchestration_fallback_failure_category": (
                                "planner_output_invalid"
                            ),
                            "orchestration_fallback_attempt_count": exc.attempts,
                            "orchestration_fallback_validation_errors": list(
                                exc.validation_errors
                            ),
                            "orchestration_fallback_validation_error_categories": list(
                                exc.validation_error_categories
                            ),
                            "orchestration_fallback_output_parse_error_count": (
                                exc.output_parse_error_count
                            ),
                            "orchestration_fallback_core_validation_error_count": (
                                exc.core_validation_error_count
                            ),
                        }
                        if isinstance(exc, QwenNextActionPlannerError)
                        else _llm_call_fallback_diagnostics(exc)
                    )
                    state = replace(
                        state,
                        execution_metadata={
                            **state.execution_metadata,
                            "orchestration_requested_mode": "llm_react",
                            "orchestration_mode": "controlled_react",
                            "orchestration_fallback_reason": reason,
                            "orchestration_fallback_stage": "next_action_planning",
                            "orchestration_fallback_after_action_count": len(
                                state.action_history
                            ),
                            **validation_diagnostics,
                        },
                    )
                    # Explicit planner and execution-profile swap: fallback
                    # keeps the same State and this same loop, mirroring the
                    # pre-refactor controlled continuation. Neither swap is
                    # implicit in the other.
                    planner = ControlledPlannerAdapter(
                        fallback_policy or InvestigationPolicy()
                    )
                    profile = self._controlled_react_profile(goal=goal)
                    continue
            if stop is None and proposal is not None:
                if proposal.decision_type == DecisionType.STOP.value:
                    stop = ReactStopBundle.from_proposal(proposal)
                elif profile.act_gate is not None:
                    gated = profile.act_gate(
                        state,
                        planner,
                        proposal,
                        max(
                            0,
                            goal.max_tool_calls - len(observed_tool_latencies),
                        ),
                    )
                    if isinstance(gated, ReactStopBundle):
                        stop = gated
                    elif isinstance(gated, PlannerProposal):
                        proposal = gated
                        if proposal.decision_type == DecisionType.STOP.value:
                            stop = ReactStopBundle.from_proposal(proposal)
            if stop is not None:
                return self._react_terminal(
                    state,
                    stop=stop,
                    profile=profile,
                    goal=goal,
                )
            assert proposal is not None
            action = proposal.next_action
            if action is None:
                raise SupervisorExecutionError(
                    "React act proposal lost its next_action",
                    state=state,
                )
            if profile.emits_decision_events:
                emit_workflow_event(
                    "planner_decision",
                    {
                        "decision_id": proposal.decision_id,
                        "decision_type": proposal.decision_type,
                        "reason": proposal.reason,
                        "action_id": action.action_id,
                        "action_kind": action.kind,
                        "agent": action.agent,
                        "target_question_ids": list(
                            proposal.target_question_ids
                        ),
                    },
                )
            emit_workflow_event(
                "action_started",
                {
                    "action_id": action.action_id,
                    "action_kind": action.kind,
                    "agent": action.agent,
                    "reason": action.reason,
                },
            )
            finding = profile.dispatch(
                state,
                action,
                max(0, goal.max_tool_calls - len(observed_tool_latencies)),
            )
            state_with_finding = self._record_controlled_finding(
                state,
                action,
                finding,
            )
            if profile.records_planner_outcome:
                state = self._record_planner_outcome(
                    state_with_finding,
                    self._planner_audit_outcome(proposal),
                )
            else:
                state = state_with_finding
            emit_workflow_event(
                "action_completed",
                {
                    "action_id": action.action_id,
                    "action_kind": action.kind,
                    "agent": finding.agent,
                    "finding_id": finding.finding_id,
                    "summary": finding.summary,
                    "evidence_ids": list(finding.evidence_ids),
                    "confidence": finding.confidence,
                },
            )

    @staticmethod
    def _runtime_stop_bundle(
        outcome: PlannerDecisionOutcome,
    ) -> ReactStopBundle:
        decision = outcome.decision
        return ReactStopBundle(
            origin="runtime",
            reason=decision.reason,
            goal_status=decision.goal_status,
            proposed_conclusion_level=decision.proposed_conclusion_level,
            stop_reason=decision.stop_reason,
            evidence_gaps=[],
            decision_id=decision.decision_id,
            decision_proposed_by=outcome.decision_proposed_by,
            question_updates_source=outcome.question_updates_source,
            has_question_updates=bool(decision.question_updates),
            action_value_assessments=list(outcome.action_value_assessments),
            audit_outcome=outcome,
        )

    @staticmethod
    def _planner_audit_outcome(
        proposal: PlannerProposal,
    ) -> PlannerDecisionOutcome:
        """Rebuild the pre-refactor audit outcome from public fields.

        The shared loop never reads the native decision object; the audit
        record is reconstructed from the same public proposal contract the
        loop consumes, so ``planner_decisions`` content is unchanged.
        """

        decision = PlannerDecision(
            decision_id=proposal.decision_id or "",
            goal_id=proposal.goal_id or "",
            decision_type=proposal.decision_type,
            reason=proposal.reason,
            goal_status=(
                proposal.proposed_goal_status or GoalStatus.IN_PROGRESS.value
            ),
            proposed_conclusion_level=proposal.proposed_conclusion_level,
            next_action=proposal.next_action,
            target_question_ids=list(proposal.target_question_ids),
            new_questions=list(proposal.new_questions),
            stop_reason=proposal.stop_reason,
            question_updates=list(proposal.question_updates),
        )
        return PlannerDecisionOutcome(
            decision=decision,
            question_update_reviews=list(proposal.question_update_reviews),
            raw_question_update_count=proposal.raw_question_update_count,
            decision_proposed_by=proposal.decision_proposed_by or "qwen",
            question_updates_source=proposal.question_updates_source,
            action_value_assessments=list(proposal.action_value_assessments),
        )

    def _react_terminal(
        self,
        state: RCAState,
        *,
        stop: ReactStopBundle,
        profile: ReactExecutionProfile,
        goal: InvestigationGoal,
    ) -> RCAState:
        """Shared terminal governance for planner and runtime stops."""

        working = state
        if profile.records_planner_outcome and stop.audit_outcome is not None:
            working = self._record_planner_outcome(working, stop.audit_outcome)
        if profile.stop_provenance_metadata is not None:
            working = replace(
                working,
                execution_metadata={
                    **working.execution_metadata,
                    **profile.stop_provenance_metadata(working, stop),
                },
            )
        conclusion_level = (
            gate_planner_conclusion_level(
                stop.proposed_conclusion_level,
                state=working,
                goal=goal,
            )
            if profile.caps_conclusion
            else stop.proposed_conclusion_level
        )
        questions = working.investigation_questions
        if (
            profile.reconciles_satisfied_questions
            and stop.goal_status == GoalStatus.SATISFIED.value
        ):
            questions = _reconcile_satisfied_questions(working)
        evidence_gaps = list(
            dict.fromkeys(
                [
                    *stop.evidence_gaps,
                    *_open_question_gaps(questions),
                ]
            )
        )
        terminal = replace(
            working,
            job=replace(working.job, status=TaskStatus.COMPLETED.value),
            investigation_questions=questions,
            goal_status=stop.goal_status,
            conclusion_level=conclusion_level,
            evidence_gaps=evidence_gaps,
            stop_reason=stop.stop_reason,
        )
        if profile.pre_finalize_trace_metadata and (
            terminal.competition_trace is not None
        ):
            terminal = replace(
                terminal,
                execution_metadata={
                    **terminal.execution_metadata,
                    "terminal_question_updates_source": "python_evidence_gate",
                    "terminal_question_updates_validated_by": (
                        "python_evidence_gate"
                    ),
                },
            )
        if profile.finalize_mode == "always":
            terminal = finalize_investigation(
                terminal,
                action_value_assessments=stop.action_value_assessments,
                budget_exhausted=(
                    stop.stop_reason == StopReason.BUDGET_EXHAUSTED.value
                ),
            )
            terminal = _record_finalizer_terminal_projection(terminal)
            validate_terminal_investigation_state(terminal)
        elif _llm_react_governance_requested(terminal) and (
            terminal.authoritative_rca_finding is not None
        ):
            terminal = finalize_investigation(terminal)
            if terminal.execution_metadata.get("investigation_finalizer") == (
                "python_competition_progression"
            ):
                terminal = replace(
                    terminal,
                    execution_metadata={
                        **terminal.execution_metadata,
                        "terminal_question_updates_source": (
                            "python_evidence_gate"
                        ),
                        "terminal_question_updates_validated_by": (
                            "python_evidence_gate"
                        ),
                    },
                )
                terminal = _record_finalizer_terminal_projection(terminal)
                validate_terminal_investigation_state(terminal)
            elif (
                terminal.execution_metadata.get("investigation_finalizer")
                == "not_applicable"
            ):
                terminal = finalize_non_competition_llm_react_terminal(terminal)
                terminal = _record_finalizer_terminal_projection(terminal)
                validate_terminal_investigation_state(terminal)
        emit_workflow_event(
            profile.stop_event,
            profile.stop_event_payload(terminal, stop),
        )
        if not terminal.evidence:
            return terminal
        report = self.report_generator.generate(terminal)
        return replace(terminal, report=report)

    def _react_llm_budget_terminal(
        self,
        state: RCAState,
        exc: LLMCallError,
    ) -> RCAState:
        """Runtime termination for exhausted LLM budget (not a planner stop)."""

        terminal_state = replace(
            state,
            goal_status=GoalStatus.BUDGET_EXHAUSTED.value,
            conclusion_level=ConclusionLevel.INCONCLUSIVE.value,
            evidence_gaps=_open_question_gaps(state.investigation_questions),
            stop_reason=StopReason.BUDGET_EXHAUSTED.value,
            execution_metadata={
                **state.execution_metadata,
                "orchestration_requested_mode": "llm_react",
                "orchestration_mode": "llm_react",
                "planner_stop_proposed_by": "python_runtime",
                "terminal_question_updates_source": "python_evidence_gate",
                "llm_budget_exhausted": True,
                "llm_budget_failure_category": exc.failure_category,
            },
        )
        terminal_state = finalize_investigation(
            terminal_state,
            budget_exhausted=True,
        )
        terminal_state = _record_finalizer_terminal_projection(terminal_state)
        terminal = replace(
            terminal_state,
            job=replace(
                terminal_state.job,
                status=TaskStatus.COMPLETED.value,
            ),
        )
        validate_terminal_investigation_state(terminal)
        if not terminal.evidence:
            return terminal
        report = self.report_generator.generate(terminal)
        return replace(terminal, report=report)

    def _llm_react_profile(
        self,
        *,
        goal: InvestigationGoal,
        qwen_planner: QwenNextActionPlanner,
    ) -> ReactExecutionProfile:
        """llm_react execution profile: Qwen planner + Specialist V2 dispatch."""

        def preflight_termination(
            state: RCAState,
        ) -> PlannerDecisionOutcome | None:
            remaining_llm_calls = llm_calls_remaining(qwen_planner.llm_client)
            if remaining_llm_calls is not None and remaining_llm_calls <= 1:
                return PlannerDecisionOutcome(
                    decision=PlannerDecision(
                        decision_id=(
                            f"{goal.goal_id}:llm-budget-stop:"
                            f"{len(state.planner_decisions) + 1}"
                        ),
                        goal_id=goal.goal_id,
                        decision_type=DecisionType.STOP.value,
                        reason=(
                            "Python stopped before another provider call because "
                            "the remaining global LLM-call budget cannot safely "
                            "cover both planning and a possible selected action."
                        ),
                        goal_status=GoalStatus.BUDGET_EXHAUSTED.value,
                        proposed_conclusion_level=(
                            ConclusionLevel.INCONCLUSIVE.value
                        ),
                        stop_reason=StopReason.BUDGET_EXHAUSTED.value,
                    ),
                    decision_proposed_by="python_runtime",
                    action_value_assessments=list(
                        state.latest_action_value_assessments
                    ),
                )
            return None

        def act_gate(
            state: RCAState,
            planner: Any,
            proposal: PlannerProposal,
            remaining_tool_calls: int,
        ) -> ReactStopBundle | None:
            action = proposal.next_action
            if (
                action is not None
                and action.kind == "run_rca_reasoning"
                and not _rca_reasoning_round_budget_available(
                    qwen_planner.llm_client
                )
            ):
                decision = PlannerDecision(
                    decision_id=f"{proposal.decision_id}:rca-budget-stop",
                    goal_id=goal.goal_id,
                    decision_type=DecisionType.STOP.value,
                    reason=(
                        "Python did not start another RCA reasoning round because "
                        "the remaining LLM-call budget cannot cover Candidate "
                        "generation, adversarial Challenge, and the governed "
                        "post-Action Planner decision."
                    ),
                    goal_status=GoalStatus.BUDGET_EXHAUSTED.value,
                    proposed_conclusion_level=ConclusionLevel.INCONCLUSIVE.value,
                    stop_reason=StopReason.BUDGET_EXHAUSTED.value,
                )
                return self._runtime_stop_bundle(
                    PlannerDecisionOutcome(
                        decision=decision,
                        decision_proposed_by="python_runtime",
                        action_value_assessments=list(
                            proposal.action_value_assessments
                        ),
                    )
                )
            return None

        def dispatch(
            state: RCAState,
            action: InvestigationAction,
            remaining_tool_calls: int,
        ) -> AgentFinding:
            try:
                return self._dispatch_llm_react(
                    action,
                    state,
                    remaining_tool_calls=remaining_tool_calls,
                )
            except SpecialistV2Error as exc:
                raise SupervisorExecutionError(str(exc), state=state) from exc

        def stop_event_payload(
            terminal: RCAState,
            stop: ReactStopBundle,
        ) -> dict[str, Any]:
            return {
                "decision_id": stop.decision_id,
                "reason": stop.reason,
                "goal_status": terminal.goal_status,
                "conclusion_level": terminal.conclusion_level,
                "stop_reason": terminal.stop_reason,
                "evidence_gaps": list(terminal.evidence_gaps),
            }

        def stop_provenance_metadata(
            state: RCAState,
            stop: ReactStopBundle,
        ) -> dict[str, Any]:
            return {
                "planner_stop_proposed_by": stop.decision_proposed_by,
                "terminal_question_updates_source": (
                    stop.question_updates_source
                    or (
                        "python_evidence_gate"
                        if stop.decision_proposed_by == "python_runtime"
                        else None
                    )
                ),
                "terminal_question_updates_validated_by": (
                    "python_evidence_gate"
                    if stop.has_question_updates
                    or state.competition_trace is not None
                    else None
                ),
            }

        return ReactExecutionProfile(
            name="llm_react",
            stop_event="planner_stopped",
            preflight_termination=preflight_termination,
            act_gate=act_gate,
            dispatch=dispatch,
            stop_event_payload=stop_event_payload,
            stop_provenance_metadata=stop_provenance_metadata,
            records_planner_outcome=True,
            emits_decision_events=True,
            caps_conclusion=True,
            reconciles_satisfied_questions=False,
            pre_finalize_trace_metadata=True,
            finalize_mode="always",
        )

    def _controlled_react_profile(
        self,
        *,
        goal: InvestigationGoal,
    ) -> ReactExecutionProfile:
        """controlled_react profile: deterministic policy + registry dispatch."""

        def act_gate(
            state: RCAState,
            planner: Any,
            proposal: PlannerProposal,
            remaining_tool_calls: int,
        ) -> PlannerProposal | None:
            action = proposal.next_action
            if action is None:
                return None
            required_tool_calls = self._controlled_action_tool_cost(action, state)
            if required_tool_calls <= remaining_tool_calls:
                return None
            # Ask the policy for its normal budget terminal so fallback
            # execution cannot cross the global Tool-call boundary.
            return planner.decide(
                build_planner_context(
                    goal=goal,
                    state=state,
                    tool_call_count=goal.max_tool_calls,
                )
            )

        def dispatch(
            state: RCAState,
            action: InvestigationAction,
            remaining_tool_calls: int,
        ) -> AgentFinding:
            return self._dispatch_controlled(action, state)

        def stop_event_payload(
            terminal: RCAState,
            stop: ReactStopBundle,
        ) -> dict[str, Any]:
            return {
                "mode": "controlled_react",
                "goal_status": terminal.goal_status,
                "conclusion_level": terminal.conclusion_level,
                "stop_reason": terminal.stop_reason,
                "evidence_gaps": list(terminal.evidence_gaps),
            }

        return ReactExecutionProfile(
            name="controlled_react",
            stop_event="investigation_stopped",
            preflight_termination=None,
            act_gate=act_gate,
            dispatch=dispatch,
            stop_event_payload=stop_event_payload,
            stop_provenance_metadata=None,
            records_planner_outcome=False,
            emits_decision_events=False,
            caps_conclusion=False,
            reconciles_satisfied_questions=True,
            pre_finalize_trace_metadata=False,
            finalize_mode="governance_gated",
        )

    def _controlled_action_tool_cost(
        self,
        action: InvestigationAction,
        state: RCAState,
    ) -> int:
        """Return the conservative V1 Tool cost used for budget preflight."""

        if action.kind in {
            "inspect_defect_pattern",
            "validate_shared_defect_pattern",
            "validate_historical_case",
        }:
            return 1
        if action.kind == "find_shared_exposure":
            lot_id = str(
                action.inputs.get("lot_id") or state.job.source_lot_id or ""
            ).strip()
            return 3 if lot_id else 2
        if action.kind == "inspect_fdc_spc":
            return 4 if self.fdc_agent.analyze_spc_evidence_tool is not None else 3
        return 0

    def execute_llm_react(
        self,
        job: RCAJob,
        intent_plan: IntentPlan,
        planner: QwenNextActionPlanner,
        *,
        fallback_policy: InvestigationPolicy | None = None,
        tool_latencies: list[dict[str, str | float]] | None = None,
    ) -> RCAState:
        """Let Qwen choose one registered Agent action after every observation."""

        state = RCAState(
            job=replace(job, status=TaskStatus.RUNNING.value),
            investigation_goal=intent_plan.goal,
            evidence=_initial_context_evidence(
                job,
                include_incident_observations=True,
            ),
            capability_notices=list(intent_plan.capability_notices),
            investigation_questions=list(intent_plan.questions),
            execution_metadata={
                "orchestration_requested_mode": "llm_react",
                "orchestration_mode": "llm_react",
            },
        )
        unsupported_kinds = {
            notice.capability
            for notice in intent_plan.capability_notices
            if not notice.supported
        }
        if unsupported_kinds:
            questions = [
                replace(
                    question,
                    status=EvidenceGapStatus.UNAVAILABLE.value,
                    answer=None,
                    evidence_ids=[],
                    unavailable_reason=next(
                        (
                            notice.reason
                            for notice in intent_plan.capability_notices
                            if notice.capability == question.question_kind
                        ),
                        "The requested capability is not configured.",
                    ),
                )
                if question.question_kind in unsupported_kinds
                and question.status == EvidenceGapStatus.OPEN.value
                else question
                for question in state.investigation_questions
            ]
            state = replace(state, investigation_questions=questions)
            if not any(
                question.status == EvidenceGapStatus.OPEN.value
                for question in questions
            ):
                terminal = replace(
                    state,
                    job=replace(state.job, status=TaskStatus.COMPLETED.value),
                    goal_status=GoalStatus.BLOCKED.value,
                    conclusion_level=ConclusionLevel.INCONCLUSIVE.value,
                    evidence_gaps=[],
                    stop_reason=StopReason.DATA_UNAVAILABLE.value,
                )
                # No investigation Evidence exists for a pure unsupported
                # request, so a traceable RCA report would be misleading.
                return terminal
        return self._react_loop(
            state,
            intent_plan.goal,
            profile=self._llm_react_profile(
                goal=intent_plan.goal,
                qwen_planner=planner,
            ),
            planner=QwenPlannerAdapter(planner),
            fallback_policy=fallback_policy,
            tool_latencies=tool_latencies,
        )

    @staticmethod
    def _record_planner_outcome(
        state: RCAState,
        outcome: PlannerDecisionOutcome,
    ) -> RCAState:
        decision = outcome.decision
        questions_by_id = {
            question.question_id: question
            for question in state.investigation_questions
        }
        for question in getattr(decision, "question_updates", []):
            current = questions_by_id.get(question.question_id)
            if current is None:
                raise SupervisorExecutionError(
                    "Planner question update references an unknown question",
                    state=state,
                )
            if current.status != EvidenceGapStatus.OPEN.value:
                raise SupervisorExecutionError(
                    "Planner question update references a terminal question",
                    state=state,
                )
            questions_by_id[question.question_id] = replace(
                current,
                status=question.status,
                answer=question.answer,
                evidence_ids=list(question.evidence_ids),
                unavailable_reason=question.unavailable_reason,
            )
        for question in decision.new_questions:
            questions_by_id[question.question_id] = question
        return replace(
            state,
            investigation_questions=list(questions_by_id.values()),
            planner_decisions=[*state.planner_decisions, decision],
            question_update_reviews=[
                *state.question_update_reviews,
                *outcome.question_update_reviews,
            ],
            latest_action_value_assessments=list(
                outcome.action_value_assessments
            ),
        )

    def _dispatch_llm_react(
        self,
        action: InvestigationAction,
        state: RCAState,
        *,
        remaining_tool_calls: int,
    ) -> AgentFinding:
        """Dispatch one Qwen action through Specialist V2 or the RCA evidence gate."""

        definition = ACTION_REGISTRY.get(action.kind)
        if (
            definition is None
            or action.kind not in LLM_REACT_EXECUTABLE_ACTION_KINDS
            or action.agent != definition.agent
        ):
            raise SupervisorExecutionError(
                f"LLM action {action.kind!r} is not executable by {action.agent!r}",
                state=state,
            )
        if action.kind == "run_rca_reasoning":
            return self._dispatch_controlled(action, state)
        if self.specialist_v2_executor is None:
            raise SupervisorExecutionError(
                "llm_react Specialist V2 executor is not configured",
                state=state,
            )
        if remaining_tool_calls <= 0:
            raise SupervisorExecutionError(
                "the global Tool budget is exhausted before Specialist execution",
                state=state,
            )

        facts = action.inputs
        context: dict[str, Any] = {
            "investigation_intent": (
                state.investigation_goal.intent
                if state.investigation_goal is not None
                else ""
            ),
            "source_lot_id": state.job.source_lot_id,
            "user_query": state.job.user_query,
        }
        lane_scope = action.scope
        if isinstance(lane_scope.get("lane_id"), str) and lane_scope["lane_id"].strip():
            context["lane_id"] = lane_scope["lane_id"].strip()
        for scope_key, context_key in (
            ("operation", "operation_no"),
            ("operation_no", "operation_no"),
            ("equipment", "equipment_id"),
            ("equipment_id", "equipment_id"),
            ("chamber", "chamber_id"),
            ("chamber_id", "chamber_id"),
            ("recipe", "recipe_id"),
            ("recipe_id", "recipe_id"),
        ):
            value = lane_scope.get(scope_key)
            if isinstance(value, str) and value.strip():
                context[context_key] = value.strip()
        raw_parameters = lane_scope.get("parameters", "")
        if isinstance(raw_parameters, str) and raw_parameters.strip():
            context["parameter_names"] = [
                item.strip()
                for item in raw_parameters.split(",")
                if item.strip()
            ]
        if isinstance(lane_scope.get("window_start"), str):
            context["window_start"] = lane_scope["window_start"]
        if isinstance(lane_scope.get("window_end"), str):
            context["window_end"] = lane_scope["window_end"]
        if action.kind in {
            "inspect_defect_pattern",
            "validate_shared_defect_pattern",
        }:
            source_lot_id = str(
                facts.get("lot_id") or state.job.source_lot_id or ""
            ).strip()
            lot_ids = [source_lot_id] if source_lot_id else []
            if action.kind == "validate_shared_defect_pattern" or not lot_ids:
                mes = _latest_finding_for_agent(
                    state.findings,
                    AgentKind.MES.value,
                    state=state,
                )
                selected_lane = _selected_lane(mes, context.get("lane_id"))
                raw_lot_ids = (
                    selected_lane.get("exposed_lot_ids", [])
                    if selected_lane is not None
                    else mes.details.get("affected_lots", [])
                )
                authorized_lot_ids = (
                    raw_lot_ids if isinstance(raw_lot_ids, list) else []
                )
                selected = [
                    str(item).strip()
                    for item in authorized_lot_ids
                    if str(item).strip()
                ]
                lot_ids = list(
                    dict.fromkeys(
                        [
                            *([source_lot_id] if source_lot_id else []),
                            *selected,
                        ]
                    )
                )
            if not lot_ids:
                raise SupervisorExecutionError(
                    "defect-pattern action requires a source or MES-selected Lot scope",
                    state=state,
                )
            context.update(
                {
                    "lot_ids": lot_ids,
                    "evidence_scope": (
                        "shared_exposure_comparison"
                        if action.kind == "validate_shared_defect_pattern"
                        else "selected_lots"
                    ),
                }
            )
        elif action.kind == "find_shared_exposure":
            start_date, end_date = _time_window(facts, state.job)
            context.update(
                {
                    "lot_id": str(
                        facts.get("lot_id") or state.job.source_lot_id or ""
                    ).strip()
                    or None,
                    "product_id": str(
                        facts.get("product_id") or state.job.product_id or ""
                    ).strip()
                    or None,
                    "start_date": start_date,
                    "end_date": end_date,
                    "target_operation_no": (
                        str(
                            facts.get("target_operation_no")
                            or lane_scope.get("operation_no")
                            or lane_scope.get("operation")
                        ).strip()
                        if (
                            facts.get("target_operation_no")
                            or lane_scope.get("operation_no")
                            or lane_scope.get("operation")
                        )
                        else None
                    ),
                }
            )
        elif action.kind == "inspect_fdc_spc":
            mes = _latest_finding_for_agent(
                state.findings,
                AgentKind.MES.value,
                state=state,
            )
            raw_commonality = mes.details.get("target_commonality", {})
            commonality = (
                raw_commonality if isinstance(raw_commonality, dict) else {}
            )
            selected_lane = _selected_lane(mes, context.get("lane_id"))
            if selected_lane is not None:
                commonality = {
                    **commonality,
                    "equipment_id": selected_lane.get("equipment", ""),
                    "chamber_id": selected_lane.get("chamber", ""),
                    "recipe_id": selected_lane.get("recipe", ""),
                }
            raw_lot_ids = (
                selected_lane.get("exposed_lot_ids", [])
                if selected_lane is not None
                else mes.details.get("affected_lots", [])
            )
            lot_ids = _source_anchored_lot_scope(
                state.job.source_lot_id,
                raw_lot_ids,
            )
            context.update(
                {
                    "lot_ids": lot_ids,
                    "equipment_id": str(commonality.get("equipment_id", "")),
                    "chamber_id": str(commonality.get("chamber_id", "")),
                    "recipe_id": str(commonality.get("recipe_id", "")),
                    "operation_no": str(
                        selected_lane.get("operation", "")
                        if selected_lane is not None
                        else mes.details.get("target_operation_no", "")
                    ),
                }
            )
        elif action.kind == "validate_historical_case":
            selected_findings = [
                _latest_finding_for_agent(
                    state.findings,
                    agent,
                    state=state,
                )
                for agent in (
                    AgentKind.MES.value,
                    AgentKind.FDC.value,
                    AgentKind.DEFECT_WAT.value,
                )
            ]
            query, module, equipment_type = _knowledge_query(
                state,
                selected_findings,
                lane_scope,
            )
            observation = _knowledge_observation_context(state, selected_findings)
            context.update(
                {
                    "query": query,
                    "module": module,
                    "equipment_type": equipment_type,
                    **observation,
                    "explicit_module_limit": _knowledge_explicit_module_limit(
                        state,
                        module,
                        action.inputs,
                    ),
                }
            )
        else:
            raise SupervisorExecutionError(
                f"LLM Specialist action {action.kind!r} has no V2 dispatcher",
                state=state,
            )

        return self.specialist_v2_executor.execute(
            action,
            request_id=f"{state.job.job_id}:{action.action_id}",
            context=context,
            max_tool_calls=min(2, remaining_tool_calls),
        )

    def _dispatch_controlled(
        self,
        action: InvestigationAction,
        state: RCAState,
    ) -> AgentFinding:
        definition = ACTION_REGISTRY.get(action.kind)
        if (
            definition is None
            or action.kind not in LLM_REACT_EXECUTABLE_ACTION_KINDS
            or action.agent != definition.agent
        ):
            raise SupervisorExecutionError(
                f"controlled action {action.kind!r} is not executable by {action.agent!r}",
                state=state,
            )
        request_id = f"{state.job.job_id}:{action.action_id}"
        facts = action.inputs
        if action.kind in {"inspect_defect_pattern", "validate_shared_defect_pattern"}:
            lot_id = str(facts.get("lot_id") or state.job.source_lot_id or "")
            lot_ids = [lot_id] if lot_id else []
            if action.kind == "validate_shared_defect_pattern" or not lot_ids:
                mes = _latest_finding_for_agent(
                    state.findings,
                    AgentKind.MES.value,
                    state=state,
                )
                raw_mes_lot_ids = mes.details.get("affected_lots", [])
                mes_lot_ids = [
                    str(item)
                    for item in (
                        raw_mes_lot_ids
                        if isinstance(raw_mes_lot_ids, list)
                        else []
                    )
                    if str(item).strip()
                ]
                if action.kind == "validate_shared_defect_pattern" or not lot_ids:
                    lot_ids = mes_lot_ids or lot_ids
            if not lot_ids:
                raise SupervisorExecutionError(
                    "defect-pattern action requires a source or MES-selected Lot scope",
                    state=state,
                )
            finding = self.defect_wat_agent.analyze(
                request_id=request_id,
                lot_ids=lot_ids,
                evidence_scope=(
                    "shared_exposure_comparison"
                    if action.kind == "validate_shared_defect_pattern"
                    else "selected_lots"
                ),
            )
            return _review_specialist_finding(
                finding,
                llm_client=self.llm_client,
                agent_mode=self.agent_mode,
                prompt_version=self.specialist_prompt_version,
            )
        if action.kind == "find_shared_exposure":
            lot_id = str(facts.get("lot_id") or state.job.source_lot_id or "")
            if lot_id:
                finding = self.mes_agent.analyze_lot(request_id=request_id, lot_id=lot_id)
            else:
                product_id = str(facts.get("product_id") or state.job.product_id or "")
                if not product_id:
                    raise SupervisorExecutionError(
                        "shared-exposure action requires a lot_id or product_id", state=state
                    )
                finding = self.mes_agent.analyze(
                    request_id=request_id,
                    product_id=product_id,
                    start_date=str(facts.get("start_date") or "") or None,
                    end_date=str(facts.get("end_date") or "") or None,
                )
            return _review_specialist_finding(
                finding,
                llm_client=self.llm_client,
                agent_mode=self.agent_mode,
                prompt_version=self.specialist_prompt_version,
            )
        if action.kind == "inspect_fdc_spc":
            mes = _latest_finding_for_agent(
                state.findings,
                AgentKind.MES.value,
                state=state,
            )
            commonality = mes.details["target_commonality"]
            finding = self.fdc_agent.analyze(
                request_id=request_id,
                lot_ids=list(mes.details["affected_lots"]),
                equipment_id=str(commonality["equipment_id"]),
                chamber_id=str(commonality["chamber_id"]),
                operation_no=str(mes.details["target_operation_no"]),
            )
            return _review_specialist_finding(
                finding,
                llm_client=self.llm_client,
                agent_mode=self.agent_mode,
                prompt_version=self.specialist_prompt_version,
            )
        if action.kind == "validate_historical_case":
            selected_findings = [
                _latest_finding_for_agent(
                    state.findings,
                    agent,
                    state=state,
                )
                for agent in (
                    AgentKind.MES.value,
                    AgentKind.FDC.value,
                    AgentKind.DEFECT_WAT.value,
                )
            ]
            query, module, equipment_type = _knowledge_query(
                state,
                selected_findings,
            )
            observation = _knowledge_observation_context(state, selected_findings)
            finding = self.knowledge_agent.analyze(
                request_id=request_id,
                query=query,
                module=module,
                equipment_type=equipment_type,
                source_lot_id=str(observation["source_lot_id"]),
                product_id=str(observation["product_id"]),
                detected_operation=str(observation["detected_operation"]),
                detected_equipment_id=str(observation["detected_equipment_id"]),
                detected_at=str(observation["detected_at"]),
                symptom_types=tuple(observation["symptom_types"]),
                explicit_module_limit=_knowledge_explicit_module_limit(
                    state,
                    module,
                    action.inputs,
                ),
            )
            return _review_specialist_finding(
                finding,
                llm_client=self.llm_client,
                agent_mode=self.agent_mode,
                prompt_version=self.specialist_prompt_version,
            )
        if action.kind == "run_rca_reasoning":
            specialists = [
                finding
                for finding in state.findings
                if finding.agent
                in {
                    AgentKind.MES.value,
                    AgentKind.FDC.value,
                    AgentKind.DEFECT_WAT.value,
                    AgentKind.KNOWLEDGE.value,
                }
            ]
            rca_finding: AgentFinding = self.rca_reasoning_agent.analyze(
                request_id=request_id,
                findings=specialists,
                context_evidence=state.evidence,
                causal_lanes=state.causal_lanes,
                prior_rca_finding=state.authoritative_rca_finding,
            )
            return rca_finding
        raise SupervisorExecutionError(
            f"controlled action {action.kind!r} has no dispatcher", state=state
        )

    def _record_controlled_finding(
        self,
        state: RCAState,
        action: InvestigationAction,
        finding: AgentFinding,
    ) -> RCAState:
        prior_reasoning_signature = _reasoning_decision_signature(
            state.authoritative_rca_finding
        )
        if finding.agent != action.agent:
            raise SupervisorExecutionError(
                f"action {action.action_id} expected {action.agent} finding, "
                f"got {finding.agent}",
                state=state,
            )
        available_before_action = {item.evidence_id for item in state.evidence}
        missing_required = set(action.required_evidence_ids) - available_before_action
        if missing_required:
            raise SupervisorExecutionError(
                f"action requires unavailable evidence: {sorted(missing_required)}",
                state=state,
            )
        if any(item.finding_id == finding.finding_id for item in state.findings):
            raise SupervisorExecutionError(
                f"finding_id is already recorded: {finding.finding_id}",
                state=state,
            )
        try:
            evidence = EvidenceCollection(state.evidence).merge(finding.evidence).to_list()
        except ModelValidationError as exc:
            raise SupervisorExecutionError(str(exc), state=state) from exc
        known_evidence_ids = {item.evidence_id for item in evidence}
        missing = set(finding.evidence_ids) - known_evidence_ids
        if missing:
            raise SupervisorExecutionError(
                f"controlled finding references evidence without payload: {sorted(missing)}",
                state=state,
            )
        affected_lots = list(state.affected_lots)
        impact_lots = list(state.impact_lots)
        affected_wafers = list(state.affected_wafers)
        impact_wafers = list(state.impact_wafers)
        scope_level = state.scope_level
        impact_criteria = dict(state.impact_criteria)
        updated_job = state.job
        if finding.agent == AgentKind.MES.value:
            affected_lots = list(finding.details.get("affected_lots", []))
            impact_lots = list(finding.details.get("impact_lots", []))
            affected_wafers = list(finding.details.get("affected_wafers", []))
            impact_wafers = list(finding.details.get("impact_wafers", []))
            scope_level = str(finding.details.get("scope_level", scope_level))
            impact_criteria = dict(finding.details.get("impact_criteria", {}))
            resolved_product = finding.details.get("product_id")
            if resolved_product and not updated_job.product_id:
                updated_job = replace(updated_job, product_id=str(resolved_product))
        hypotheses = list(state.hypotheses)
        authoritative_rca_finding_id = state.authoritative_rca_finding_id
        authoritative_hypothesis_id = state.authoritative_hypothesis_id
        if finding.agent == AgentKind.RCA_REASONING.value:
            payload = finding.details.get("hypothesis")
            if not isinstance(payload, dict):
                raise SupervisorExecutionError("RCA finding must include a hypothesis", state=state)
            hypothesis = Hypothesis.from_dict(payload)
            hypotheses.append(hypothesis)
            authoritative_rca_finding_id = finding.finding_id
            authoritative_hypothesis_id = hypothesis.hypothesis_id
            if _llm_react_governance_requested(state):
                raw_impact_gate = finding.details.get("impact_lot_gate", {})
                confirmed = (
                    raw_impact_gate.get("confirmed_impact_lots", [])
                    if isinstance(raw_impact_gate, dict)
                    else []
                )
                if isinstance(confirmed, list) and all(
                    isinstance(item, str) and item.strip() for item in confirmed
                ):
                    impact_lots = list(
                        dict.fromkeys(
                            item
                            for item in confirmed
                            if item != state.job.source_lot_id
                        )
                    )
        record = ActionRecord(
            action=action,
            status="completed",
            produced_finding_ids=[finding.finding_id],
            produced_evidence_ids=list(finding.evidence_ids),
            decision_summary=finding.summary,
        )
        links = QuestionEvidenceResolver().resolve(
            questions=state.investigation_questions,
            action_record=record,
            evidence=evidence,
        )
        recorded_state = replace(
            state,
            job=updated_job,
            evidence=evidence,
            findings=[*state.findings, finding],
            hypotheses=hypotheses,
            affected_lots=affected_lots,
            impact_lots=impact_lots,
            affected_wafers=affected_wafers,
            impact_wafers=impact_wafers,
            scope_level=scope_level,
            impact_criteria=impact_criteria,
            action_history=[*state.action_history, record],
            question_evidence_links=[
                *state.question_evidence_links,
                *links,
            ],
            authoritative_rca_finding_id=authoritative_rca_finding_id,
            authoritative_hypothesis_id=authoritative_hypothesis_id,
            warnings=_merge_warnings(
                state.warnings,
                finding.warnings,
                current_findings=[*state.findings, finding],
            ),
        )
        lane_state = _update_causal_lane_state(recorded_state, finding)
        competition_state = _update_competition_state(lane_state, finding)
        updated_state = _update_causal_chain_state(competition_state, finding)
        prior_gains = list(state.investigation_gain_history)
        if not prior_gains and state.action_history:
            prior_gains = list(
                derive_investigation_gain_history(
                    action_records=state.action_history,
                    evidence=state.evidence,
                    links=state.question_evidence_links,
                )
            )
        gain = classify_investigation_gain(
            record,
            earlier_records=state.action_history,
            evidence_by_id=updated_state.evidence_by_id,
            links=updated_state.question_evidence_links,
            earlier_gains=prior_gains,
            reasoning_state_changed=(
                finding.agent == AgentKind.RCA_REASONING.value
                and _reasoning_decision_signature(finding)
                != prior_reasoning_signature
            ),
        )
        return replace(
            updated_state,
            investigation_gain_history=[*prior_gains, gain],
        )

    def execute(self, job: RCAJob, task_plan: TaskPlan) -> RCAState:
        plan_agents = {task.agent for task in task_plan.tasks}
        unsupported = plan_agents - SUPERVISOR_EXECUTABLE_AGENTS
        if unsupported:
            raise SupervisorExecutionError(
                f"TaskPlan references Agents not registered in Supervisor: {sorted(unsupported)}"
            )

        running_job = replace(job, status=TaskStatus.RUNNING.value)
        state = RCAState(
            job=running_job,
            task_plan=task_plan,
            evidence=_initial_context_evidence(job),
        )
        remaining = {task.task_id for task in task_plan.tasks}

        while remaining:
            ready = [
                task
                for task in task_plan.tasks
                if task.task_id in remaining
                and set(task.depends_on) <= set(state.completed_task_ids)
            ]
            if not ready:
                raise SupervisorExecutionError(
                    "TaskPlan has no executable task; dependencies cannot be satisfied",
                    state=state,
                )

            for task in ready:
                emit_workflow_event(
                    "agent_started",
                    {
                        "task_id": task.task_id,
                        "agent": task.agent,
                        "objective": task.objective,
                    },
                )
                state = self._execute_task(state, task)
                finding = state.finding_for_task(task.task_id)
                emit_workflow_event(
                    "agent_completed",
                    {
                        "task_id": task.task_id,
                        "agent": task.agent,
                        "finding_id": finding.finding_id if finding is not None else None,
                        "summary": finding.summary if finding is not None else "",
                        "evidence_ids": (
                            list(finding.evidence_ids) if finding is not None else []
                        ),
                        "confidence": finding.confidence if finding is not None else None,
                    },
                )
                remaining.remove(task.task_id)

        if state.task_plan is None:
            raise SupervisorExecutionError("RCAState lost its TaskPlan", state=state)
        state = replace(state, current_task_id=None)
        report = self.report_generator.generate(state)
        completed_job = replace(state.job, status=TaskStatus.COMPLETED.value)
        emit_workflow_event(
            "investigation_stopped",
            {
                "mode": "fixed",
                "goal_status": "satisfied",
                "conclusion_level": (
                    state.authoritative_hypothesis.status
                    if state.authoritative_hypothesis is not None
                    else "inconclusive"
                ),
                "stop_reason": "workflow_completed",
                "evidence_gaps": [],
            },
        )
        return replace(state, job=completed_job, report=report)

    def _execute_task(self, state: RCAState, task: AgentTask) -> RCAState:
        if state.task_plan is None:
            raise SupervisorExecutionError("RCAState requires a TaskPlan", state=state)
        running_plan = _replace_task_status(
            state.task_plan,
            task.task_id,
            TaskStatus.RUNNING.value,
        )
        running_state = replace(
            state,
            task_plan=running_plan,
            current_task_id=task.task_id,
        )
        try:
            finding = self._dispatch(task, running_state)
            return self._record_finding(running_state, task, finding)
        except Exception as exc:
            failed_plan = _replace_task_status(
                running_plan,
                task.task_id,
                TaskStatus.FAILED.value,
            )
            failed_state = replace(
                running_state,
                job=replace(running_state.job, status=TaskStatus.FAILED.value),
                task_plan=failed_plan,
                current_task_id=None,
            )
            if isinstance(exc, SupervisorExecutionError):
                raise SupervisorExecutionError(
                    str(exc),
                    state=failed_state,
                    error_code=exc.error_code,
                ) from exc
            error_code = exc.error_code if isinstance(exc, LotDrivenRCAError) else None
            raise SupervisorExecutionError(
                f"task {task.task_id} failed: {exc}",
                state=failed_state,
                error_code=error_code,
            ) from exc

    def _dispatch(self, task: AgentTask, state: RCAState) -> AgentFinding:
        request_id = f"{state.job.job_id}:{task.task_id}"
        if task.agent == AgentKind.MES.value:
            if state.job.investigation_mode == InvestigationMode.LOT.value:
                lot_id = str(task.inputs.get("lot_id") or state.job.source_lot_id or "")
                if not lot_id:
                    raise SupervisorExecutionError(
                        "Lot-driven MES task requires lot_id",
                        state=state,
                        error_code="LOT_ID_REQUIRED",
                    )
                requested_operation = task.inputs.get("target_operation_no")
                finding = self.mes_agent.analyze_lot(
                    request_id=request_id,
                    lot_id=lot_id,
                    target_operation_no=(str(requested_operation) if requested_operation else None),
                )
                return _review_specialist_finding(
                    finding,
                    llm_client=self.llm_client,
                    agent_mode=self.agent_mode,
                    prompt_version=self.specialist_prompt_version,
                )
            product_id = str(task.inputs.get("product_id") or state.job.product_id or "")
            if not product_id:
                raise SupervisorExecutionError("MES task requires product_id", state=state)
            start_date, end_date = _time_window(task.inputs, state.job)
            finding = self.mes_agent.analyze(
                request_id=request_id,
                product_id=product_id,
                start_date=start_date,
                end_date=end_date,
                target_operation_no=(
                    str(task.inputs["target_operation_no"])
                    if task.inputs.get("target_operation_no")
                    else None
                ),
            )
            return _review_specialist_finding(
                finding,
                llm_client=self.llm_client,
                agent_mode=self.agent_mode,
                prompt_version=self.specialist_prompt_version,
            )

        if task.agent == AgentKind.FDC.value:
            mes_finding = _finding_for_agent(
                _input_findings(state, task),
                AgentKind.MES.value,
                state=state,
            )
            commonality = mes_finding.details["target_commonality"]
            finding = self.fdc_agent.analyze(
                request_id=request_id,
                lot_ids=list(mes_finding.details["affected_lots"]),
                equipment_id=str(commonality["equipment_id"]),
                chamber_id=str(commonality["chamber_id"]),
                operation_no=str(mes_finding.details["target_operation_no"]),
            )
            return _review_specialist_finding(
                finding,
                llm_client=self.llm_client,
                agent_mode=self.agent_mode,
                prompt_version=self.specialist_prompt_version,
            )

        if task.agent == AgentKind.DEFECT_WAT.value:
            mes_finding = _finding_for_agent(
                _input_findings(state, task),
                AgentKind.MES.value,
                state=state,
            )
            finding = self.defect_wat_agent.analyze(
                request_id=request_id,
                lot_ids=list(mes_finding.details["affected_lots"]),
            )
            return _review_specialist_finding(
                finding,
                llm_client=self.llm_client,
                agent_mode=self.agent_mode,
                prompt_version=self.specialist_prompt_version,
            )

        if task.agent == AgentKind.KNOWLEDGE.value:
            input_findings = _input_findings(state, task)
            query, module, equipment_type = _knowledge_query(state, input_findings)
            observation = _knowledge_observation_context(state, input_findings)
            if task.finding_kind == FindingKind.KNOWLEDGE_VALIDATION.value:
                finding = self.knowledge_agent.validate_preliminary_candidates(
                    request_id=request_id,
                    preliminary_candidates=_legacy_preliminary_candidates(
                        input_findings,
                        state=state,
                    ),
                    module=module,
                    equipment_type=equipment_type,
                    source_lot_id=str(observation["source_lot_id"]),
                    product_id=str(observation["product_id"]),
                    detected_operation=str(observation["detected_operation"]),
                    detected_equipment_id=str(observation["detected_equipment_id"]),
                    detected_at=str(observation["detected_at"]),
                    explicit_module_limit=_knowledge_explicit_module_limit(
                        state,
                        module,
                    ),
                )
            else:
                finding = self.knowledge_agent.analyze(
                    request_id=request_id,
                    query=query,
                    module=module,
                    equipment_type=equipment_type,
                    source_lot_id=str(observation["source_lot_id"]),
                    product_id=str(observation["product_id"]),
                    detected_operation=str(observation["detected_operation"]),
                    detected_equipment_id=str(observation["detected_equipment_id"]),
                    detected_at=str(observation["detected_at"]),
                    symptom_types=tuple(observation["symptom_types"]),
                    explicit_module_limit=_knowledge_explicit_module_limit(
                        state,
                        module,
                    ),
                )
            return _review_specialist_finding(
                finding,
                llm_client=self.llm_client,
                agent_mode=self.agent_mode,
                prompt_version=self.specialist_prompt_version,
            )

        if task.agent == AgentKind.RCA_REASONING.value:
            specialist_findings = _input_findings(state, task)
            rca_finding: AgentFinding = self.rca_reasoning_agent.analyze(
                request_id=request_id,
                findings=specialist_findings,
                context_evidence=state.evidence,
                causal_lanes=state.causal_lanes,
                prior_rca_finding=state.authoritative_rca_finding,
            )
            return rca_finding

        if task.agent == AgentKind.IMPROVEMENT.value:
            improvement_inputs = _input_findings(state, task)
            improvement_finding: AgentFinding = self.improvement_agent.analyze(
                request_id=request_id,
                findings=improvement_inputs,
            )
            return improvement_finding

        raise SupervisorExecutionError(
            f"no dispatch implementation for Agent {task.agent}",
            state=state,
        )

    def _record_finding(
        self,
        state: RCAState,
        task: AgentTask,
        finding: AgentFinding,
    ) -> RCAState:
        if finding.agent != task.agent:
            raise SupervisorExecutionError(
                f"task {task.task_id} expected {task.agent} finding, got {finding.agent}",
                state=state,
            )
        if finding.task_id is not None and finding.task_id != task.task_id:
            raise SupervisorExecutionError(
                f"task {task.task_id} received finding for task {finding.task_id}",
                state=state,
            )
        if state.finding_for_task(task.task_id) is not None:
            raise SupervisorExecutionError(
                f"task {task.task_id} already has a recorded finding",
                state=state,
            )
        finding = replace(
            finding,
            task_id=task.task_id,
            finding_kind=task.finding_kind,
        )
        try:
            evidence = EvidenceCollection(state.evidence).merge(finding.evidence).to_list()
        except ModelValidationError as exc:
            raise SupervisorExecutionError(str(exc), state=state) from exc
        known_evidence_ids = {item.evidence_id for item in evidence}
        missing_evidence = set(finding.evidence_ids) - known_evidence_ids
        if missing_evidence:
            raise SupervisorExecutionError(
                f"finding references evidence without payload: {sorted(missing_evidence)}",
                state=state,
            )

        affected_lots = list(state.affected_lots)
        impact_lots = list(state.impact_lots)
        affected_wafers = list(state.affected_wafers)
        impact_wafers = list(state.impact_wafers)
        scope_level = state.scope_level
        impact_criteria = dict(state.impact_criteria)
        updated_job = state.job
        if finding.agent == AgentKind.MES.value:
            affected_lots = list(finding.details.get("affected_lots", []))
            impact_lots = list(finding.details.get("impact_lots", []))
            affected_wafers = list(finding.details.get("affected_wafers", []))
            impact_wafers = list(finding.details.get("impact_wafers", []))
            scope_level = str(finding.details.get("scope_level", scope_level))
            impact_criteria = dict(finding.details.get("impact_criteria", {}))
            resolved_product = finding.details.get("product_id")
            if resolved_product and not updated_job.product_id:
                updated_job = replace(updated_job, product_id=str(resolved_product))

        hypotheses = list(state.hypotheses)
        authoritative_rca_finding_id = state.authoritative_rca_finding_id
        authoritative_hypothesis_id = state.authoritative_hypothesis_id
        if finding.agent == AgentKind.RCA_REASONING.value:
            hypothesis_payload = finding.details.get("hypothesis")
            if not isinstance(hypothesis_payload, dict):
                raise SupervisorExecutionError(
                    "RCA Reasoning finding must include a hypothesis",
                    state=state,
                )
            hypothesis = Hypothesis.from_dict(hypothesis_payload)
            hypotheses.append(hypothesis)
            authoritative_rca_finding_id = finding.finding_id
            authoritative_hypothesis_id = hypothesis.hypothesis_id

        if state.task_plan is None:
            raise SupervisorExecutionError("RCAState requires a TaskPlan", state=state)
        completed_plan = _replace_task_status(
            state.task_plan,
            task.task_id,
            TaskStatus.COMPLETED.value,
        )
        recorded_state = replace(
            state,
            job=updated_job,
            task_plan=completed_plan,
            current_task_id=None,
            completed_task_ids=[*state.completed_task_ids, task.task_id],
            affected_lots=affected_lots,
            impact_lots=impact_lots,
            affected_wafers=affected_wafers,
            impact_wafers=impact_wafers,
            scope_level=scope_level,
            impact_criteria=impact_criteria,
            evidence=evidence,
            findings=[*state.findings, finding],
            hypotheses=hypotheses,
            authoritative_rca_finding_id=authoritative_rca_finding_id,
            authoritative_hypothesis_id=authoritative_hypothesis_id,
            warnings=_merge_warnings(
                state.warnings,
                finding.warnings,
                current_findings=[*state.findings, finding],
            ),
        )
        lane_state = _update_causal_lane_state(recorded_state, finding)
        competition_state = _update_competition_state(lane_state, finding)
        return _update_causal_chain_state(competition_state, finding)
