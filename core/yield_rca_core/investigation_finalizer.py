"""Python-owned terminal projection for evidence-first RCA investigations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from yield_rca_core.causal_competition_progression import (
    competition_processing_failed,
    progress_competition,
)
from yield_rca_core.causal_investigation_models import (
    ActionValueAssessment,
    CandidateCompetitionStatus,
    CompetitionRequirement,
    InvestigationGainReasonCode,
    InvestigationGainType,
)
from yield_rca_core.investigation_models import (
    ConclusionLevel,
    EvidenceGapStatus,
    GoalStatus,
    InvestigationGoal,
    InvestigationIntent,
    InvestigationQuestion,
    QuestionEvidenceRelation,
    StopReason,
)
from yield_rca_core.models import (
    AgentFinding,
    AgentKind,
    Hypothesis,
    HypothesisStatus,
    ModelValidationError,
    RCAState,
    TaskStatus,
)
from yield_rca_core.question_capability import QUESTION_CAPABILITY_REGISTRY


class InvestigationFinalizationError(RuntimeError):
    """Raised when a Job attempts to finish with non-terminal investigation state."""


def gate_planner_conclusion_level(
    proposed_level: str,
    *,
    state: RCAState,
    goal: InvestigationGoal,
) -> str:
    """Bound a Planner proposal without giving Supervisor conclusion authority."""

    authoritative = state.authoritative_hypothesis
    authoritative_status = (
        authoritative.status if authoritative is not None else None
    )
    if authoritative_status == HypothesisStatus.CONFLICTED.value:
        cap = ConclusionLevel.CONFLICTED.value
    elif authoritative_status == HypothesisStatus.SUPPORTED.value:
        cap = ConclusionLevel.SUPPORTED.value
    elif authoritative_status == HypothesisStatus.CANDIDATE.value:
        cap = ConclusionLevel.CANDIDATE.value
    elif authoritative_status in {
        HypothesisStatus.INCONCLUSIVE.value,
        HypothesisStatus.REJECTED.value,
    } or not state.evidence:
        cap = ConclusionLevel.INCONCLUSIVE.value
    else:
        finding_agents = {finding.agent for finding in state.findings}
        if (
            goal.intent == InvestigationIntent.HISTORICAL_LOOKUP.value
            and AgentKind.KNOWLEDGE.value in finding_agents
        ):
            cap = ConclusionLevel.CANDIDATE.value
        elif goal.intent in {
            InvestigationIntent.ROOT_CAUSE.value,
            InvestigationIntent.FULL_RCA.value,
        } and {
            AgentKind.MES.value,
            AgentKind.FDC.value,
            AgentKind.DEFECT_WAT.value,
        } <= finding_agents:
            cap = ConclusionLevel.CANDIDATE.value
        else:
            cap = ConclusionLevel.SIGNAL.value

    if cap == ConclusionLevel.CONFLICTED.value:
        return cap
    if proposed_level == ConclusionLevel.CONFLICTED.value:
        return ConclusionLevel.INCONCLUSIVE.value
    if proposed_level == ConclusionLevel.INCONCLUSIVE.value:
        return proposed_level
    if cap == ConclusionLevel.INCONCLUSIVE.value:
        return cap

    ordered = [
        ConclusionLevel.SIGNAL.value,
        ConclusionLevel.CANDIDATE.value,
        ConclusionLevel.SUPPORTED.value,
    ]
    return ordered[min(ordered.index(proposed_level), ordered.index(cap))]


def finalize_non_competition_llm_react_terminal(state: RCAState) -> RCAState:
    """Withhold unsupported Qwen publication when Competition was not applicable."""

    authoritative = state.authoritative_rca_finding
    if authoritative is None:
        return state
    conclusion_status = str(
        authoritative.details.get("conclusion_status", "")
    ).strip()
    if conclusion_status == "supported":
        return state
    return replace(
        state,
        goal_status=GoalStatus.BLOCKED.value,
        conclusion_level=ConclusionLevel.INCONCLUSIVE.value,
        stop_reason=StopReason.NO_HIGH_VALUE_ACTION.value,
        execution_metadata={
            **state.execution_metadata,
            "terminal_question_updates_source": "python_evidence_gate",
            "terminal_question_updates_validated_by": "python_evidence_gate",
            "terminal_conclusion_status_source": "authoritative_rca_finding",
            "terminal_state_owner": "python_investigation_finalizer",
        },
    )


def _formal_competition_is_applicable(state: RCAState) -> bool:
    """Return whether Patch 3 owns this RCA terminal projection.

    Lane discovery also supports impact-scope and other non-RCA workflows, so
    the mere presence of ``competition_trace`` is not evidence that formal
    Candidate Competition started.  Patch 3 may take terminal ownership only
    after RCA Reasoning produced an authoritative Finding and Python evaluated
    the competition requirement.
    """

    return (
        state.authoritative_rca_finding is not None
        and state.competition_trace is not None
        and (
            state.competition_trace.competition_requirement
            != CompetitionRequirement.NOT_EVALUATED.value
            or competition_processing_failed(state.competition_trace)
        )
    )


def _terminal_question_reason(competition_status: str) -> str:
    if competition_status == CandidateCompetitionStatus.BLOCKED_BY_MISSING_DATA.value:
        return (
            "The remaining decision-critical Evidence source is unavailable and "
            "no high-value registered investigation Action remains."
        )
    if competition_status == CandidateCompetitionStatus.BUDGET_EXHAUSTED.value:
        return "The investigation budget ended before the remaining uncertainty could close."
    if competition_status == CandidateCompetitionStatus.COMPLETE_REJECTED.value:
        return "The formal Candidates were contradicted and no supported root cause remains."
    return "No remaining registered Action has enough value to change the RCA decision."


def _reconcile_questions_with_complete_evidence_groups(
    state: RCAState,
) -> list[InvestigationQuestion]:
    """Close open Questions only from complete Python-validated Evidence groups."""

    available_evidence_ids = {item.evidence_id for item in state.evidence}
    result: list[InvestigationQuestion] = []
    for question in state.investigation_questions:
        if question.status != EvidenceGapStatus.OPEN.value:
            result.append(question)
            continue
        definition = QUESTION_CAPABILITY_REGISTRY.get(str(question.question_kind))
        required_groups = (
            set(definition.closure_evidence_groups)
            if definition is not None
            else set()
        )
        supporting_links = [
            link
            for link in state.question_evidence_links
            if (
                link.question_id == question.question_id
                and link.relation == QuestionEvidenceRelation.SUPPORTS.value
                and link.evidence_id in available_evidence_ids
                and link.matched_evidence_group in required_groups
            )
        ]
        satisfied_groups = {
            link.matched_evidence_group for link in supporting_links
        }
        if not required_groups or not required_groups <= satisfied_groups:
            result.append(question)
            continue
        evidence_ids = list(
            dict.fromkeys(link.evidence_id for link in supporting_links)
        )
        result.append(
            replace(
                question,
                status=EvidenceGapStatus.CLOSED.value,
                answer=(
                    "Python-validated Evidence links satisfy the required groups "
                    f"{', '.join(sorted(required_groups))}."
                ),
                evidence_ids=evidence_ids,
                unavailable_reason=None,
            )
        )
    return result


def _update_authoritative_finding(
    state: RCAState,
    *,
    competition_status: str,
    conclusion_status: str,
    terminal_reason: str | None,
    blocking_data_missing_evidence_ids: Sequence[str] = (),
) -> list[AgentFinding]:
    authoritative = state.authoritative_rca_finding
    if authoritative is None:
        return list(state.findings)
    authoritative_id = authoritative.finding_id
    result: list[AgentFinding] = []
    for finding in state.findings:
        if finding.finding_id != authoritative_id:
            result.append(finding)
            continue
        details = {
            **finding.details,
            "competition_status": competition_status,
            "scope_assessment_status": (
                state.competition_trace.scope_assessment_status
                if state.competition_trace is not None
                else "not_evaluated"
            ),
            "alternative_search_status": (
                state.competition_trace.alternative_search_status
                if state.competition_trace is not None
                else "not_searched"
            ),
            "competition_failure_reason": (
                state.competition_trace.competition_failure_reason
                if state.competition_trace is not None
                else None
            ),
            "competition_gap_reason": (
                state.competition_trace.competition_gap_reason
                if state.competition_trace is not None
                else None
            ),
            "competition_terminal_reason": terminal_reason,
            "conclusion_status": conclusion_status,
            "blocking_data_missing_evidence_ids": list(
                blocking_data_missing_evidence_ids
            ),
            "candidate_resolutions": (
                [item.to_dict() for item in state.competition_trace.candidate_resolutions]
                if state.competition_trace is not None
                else []
            ),
            "action_value_assessments": [
                item.to_dict() for item in state.latest_action_value_assessments
            ],
            "terminal_investigation_snapshot": {
                "owner": "python_investigation_finalizer",
                "competition_status": competition_status,
                "scope_assessment_status": (
                    state.competition_trace.scope_assessment_status
                    if state.competition_trace is not None
                    else "not_evaluated"
                ),
                "alternative_search_status": (
                    state.competition_trace.alternative_search_status
                    if state.competition_trace is not None
                    else "not_searched"
                ),
                "conclusion_status": conclusion_status,
                "terminal_reason": terminal_reason,
                "confirmation_gate_unchanged": True,
            },
        }
        raw_candidate_generation = finding.details.get(
            "hypothesis_candidate_generation"
        )
        if isinstance(raw_candidate_generation, dict):
            updated_competition_assessment = raw_candidate_generation.get(
                "competition_assessment"
            )
            details["hypothesis_candidate_generation"] = {
                **raw_candidate_generation,
                "competition_status": competition_status,
                "scope_assessment_status": details[
                    "scope_assessment_status"
                ],
                "alternative_search_status": details[
                    "alternative_search_status"
                ],
                "competition_terminal_reason": terminal_reason,
                "competition_assessment": (
                    {
                        **updated_competition_assessment,
                        "competition_status": competition_status,
                        "scope_assessment_status": details[
                            "scope_assessment_status"
                        ],
                        "terminal_reason": terminal_reason,
                    }
                    if isinstance(updated_competition_assessment, dict)
                    else updated_competition_assessment
                ),
            }
        if conclusion_status != "supported":
            details["status"] = HypothesisStatus.INCONCLUSIVE.value
            details["root_cause"] = "inconclusive"
        result.append(replace(finding, details=details))
    return result


def _update_authoritative_hypothesis(
    state: RCAState,
    *,
    conclusion_status: str,
) -> list[Hypothesis]:
    """Keep the final Hypothesis aligned with the Python-owned terminal Gate."""

    authoritative = state.authoritative_hypothesis
    if authoritative is None or conclusion_status == "supported":
        return list(state.hypotheses)
    authoritative_id = authoritative.hypothesis_id
    return [
        (
            replace(
                hypothesis,
                root_cause="inconclusive",
                confidence=0.0,
                status=HypothesisStatus.INCONCLUSIVE.value,
                rationale=(
                    "The Python Investigation Decision/Confirmation Gate did not "
                    "confirm a unique root cause. Candidate details remain in the "
                    "authoritative RCA Finding for audit."
                ),
                supporting_evidence_ids=[],
                rank=None,
            )
            if hypothesis.hypothesis_id == authoritative_id
            else hypothesis
        )
        for hypothesis in state.hypotheses
    ]


def finalize_investigation(
    state: RCAState,
    *,
    action_value_assessments: Sequence[ActionValueAssessment] | None = None,
    budget_exhausted: bool = False,
) -> RCAState:
    """Close Candidate/Lane/Competition/Question state before Job completion."""

    trace = state.competition_trace
    if not _formal_competition_is_applicable(state) or trace is None:
        return replace(
            state,
            execution_metadata={
                **state.execution_metadata,
                "investigation_finalizer": "not_applicable",
            },
        )
    assessments = list(
        action_value_assessments
        if action_value_assessments is not None
        else state.latest_action_value_assessments
    )
    authoritative = state.authoritative_rca_finding
    details = authoritative.details if authoritative is not None else {}
    progression = progress_competition(
        trace=trace,
        authoritative_details=details,
        action_value_assessments=assessments,
        gain_history=state.investigation_gain_history,
        force_terminal=True,
        budget_exhausted=budget_exhausted,
    )
    if not progression.terminal:
        raise InvestigationFinalizationError(
            "Job completion is illegal while Candidate Competition remains active"
        )
    updated = replace(
        state,
        competition_trace=progression.trace,
        latest_action_value_assessments=assessments,
    )
    missing_evidence_ids = list(
        dict.fromkeys(
            [
                *progression.blocking_data_missing_evidence_ids,
                *(
                    evidence_id
                    for gain in updated.investigation_gain_history
                    if gain.gain_type == InvestigationGainType.STATE_GAIN.value
                    and gain.reason_code
                    == InvestigationGainReasonCode.UNAVAILABLE_SOURCE.value
                    for evidence_id in gain.evidence_ids
                ),
            ]
        )
    )
    unavailable_evidence_by_question: dict[str, list[str]] = {}
    missing_evidence_id_set = set(missing_evidence_ids)
    for link in updated.question_evidence_links:
        if (
            link.relation == QuestionEvidenceRelation.UNAVAILABLE.value
            and link.evidence_id in missing_evidence_id_set
        ):
            unavailable_evidence_by_question.setdefault(
                link.question_id,
                [],
            ).append(link.evidence_id)
    questions = _reconcile_questions_with_complete_evidence_groups(updated)
    if progression.trace.competition_status == (
        CandidateCompetitionStatus.BLOCKED_BY_MISSING_DATA.value
    ):
        questions = [
            (
                replace(
                    question,
                    status=EvidenceGapStatus.UNAVAILABLE.value,
                    answer=None,
                    evidence_ids=list(
                        dict.fromkeys(
                            unavailable_evidence_by_question.get(
                                question.question_id,
                                [],
                            )
                        )
                    ),
                    unavailable_reason=_terminal_question_reason(
                        progression.trace.competition_status
                    ),
                )
                if question.status == EvidenceGapStatus.OPEN.value
                else question
            )
            for question in questions
        ]
    if progression.conclusion_status == "supported":
        goal_status = GoalStatus.SATISFIED.value
        conclusion_level = ConclusionLevel.SUPPORTED.value
        stop_reason = StopReason.GOAL_SATISFIED.value
    elif progression.trace.competition_status == (
        CandidateCompetitionStatus.BUDGET_EXHAUSTED.value
    ):
        goal_status = GoalStatus.BUDGET_EXHAUSTED.value
        conclusion_level = ConclusionLevel.INCONCLUSIVE.value
        stop_reason = StopReason.BUDGET_EXHAUSTED.value
    elif progression.trace.competition_status == (
        CandidateCompetitionStatus.BLOCKED_BY_MISSING_DATA.value
    ) or (
        progression.conclusion_status == "insufficient_evidence"
        and bool(missing_evidence_ids)
    ):
        goal_status = GoalStatus.BLOCKED.value
        conclusion_level = ConclusionLevel.INCONCLUSIVE.value
        stop_reason = StopReason.DATA_UNAVAILABLE.value
    else:
        goal_status = GoalStatus.BLOCKED.value
        conclusion_level = ConclusionLevel.INCONCLUSIVE.value
        stop_reason = StopReason.NO_HIGH_VALUE_ACTION.value
    updated = replace(
        updated,
        investigation_questions=questions,
        goal_status=goal_status,
        conclusion_level=conclusion_level,
        evidence_gaps=[
            question.question
            for question in questions
            if question.status == EvidenceGapStatus.OPEN.value
        ],
        stop_reason=stop_reason,
        execution_metadata={
            **updated.execution_metadata,
            "investigation_finalizer": "python_competition_progression",
            "competition_lifecycle_valid": True,
            "investigation_decision_accepted": (
                progression.trace.competition_status
                != CandidateCompetitionStatus.FAILED.value
            ),
            "root_cause_confirmed": (
                progression.trace.competition_status
                == CandidateCompetitionStatus.COMPLETE_CONFIRMED.value
            ),
        },
    )
    findings = _update_authoritative_finding(
        updated,
        competition_status=progression.trace.competition_status,
        conclusion_status=progression.conclusion_status,
        terminal_reason=progression.trace.terminal_reason,
        blocking_data_missing_evidence_ids=(
            progression.blocking_data_missing_evidence_ids
        ),
    )
    hypotheses = _update_authoritative_hypothesis(
        updated,
        conclusion_status=progression.conclusion_status,
    )
    return replace(updated, findings=findings, hypotheses=hypotheses)


def validate_terminal_investigation_state(state: RCAState) -> None:
    """Validate new terminal writes without rejecting legacy State deserialization."""

    if state.job.status != TaskStatus.COMPLETED.value:
        return
    trace = state.competition_trace
    if not _formal_competition_is_applicable(state) or trace is None:
        return
    if trace.competition_status in {
        CandidateCompetitionStatus.PENDING.value,
        CandidateCompetitionStatus.ACTIVE.value,
    }:
        raise ModelValidationError(
            "completed RCA Job cannot retain pending or active Candidate Competition"
        )
    if trace.competition_status in {
        CandidateCompetitionStatus.COMPLETE_CONFIRMED.value,
        CandidateCompetitionStatus.COMPLETE_REJECTED.value,
        CandidateCompetitionStatus.EXHAUSTED.value,
        CandidateCompetitionStatus.BLOCKED_BY_MISSING_DATA.value,
        CandidateCompetitionStatus.BUDGET_EXHAUSTED.value,
        CandidateCompetitionStatus.FAILED.value,
    } and trace.terminal_reason is None:
        raise ModelValidationError(
            "new terminal Candidate Competition states require terminal_reason"
        )
    authoritative = state.authoritative_rca_finding
    if authoritative is None:
        raise ModelValidationError(
            "terminal Candidate Competition requires an authoritative RCA Finding"
        )
    details = authoritative.details
    if str(details.get("competition_status", "")) != trace.competition_status:
        raise ModelValidationError(
            "authoritative RCA Finding competition_status must match Competition trace"
        )
    if str(details.get("scope_assessment_status", "")) != (
        trace.scope_assessment_status
    ):
        raise ModelValidationError(
            "authoritative RCA Finding scope_assessment_status must match Competition trace"
        )
    if str(details.get("alternative_search_status", "")) != (
        trace.alternative_search_status
    ):
        raise ModelValidationError(
            "authoritative RCA Finding alternative_search_status must match Competition trace"
        )
    snapshot = details.get("terminal_investigation_snapshot")
    if not isinstance(snapshot, dict):
        raise ModelValidationError(
            "terminal authoritative RCA Finding requires Python terminal snapshot"
        )
    for key, expected in (
        ("competition_status", trace.competition_status),
        ("scope_assessment_status", trace.scope_assessment_status),
        ("alternative_search_status", trace.alternative_search_status),
        ("terminal_reason", trace.terminal_reason),
        ("conclusion_status", details.get("conclusion_status")),
    ):
        if snapshot.get(key) != expected:
            raise ModelValidationError(
                f"terminal investigation snapshot {key} is inconsistent"
            )


__all__ = [
    "InvestigationFinalizationError",
    "finalize_non_competition_llm_react_terminal",
    "finalize_investigation",
    "gate_planner_conclusion_level",
    "validate_terminal_investigation_state",
]
