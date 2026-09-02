from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core"))

from yield_rca_core.causal_competition_progression import (  # noqa: E402
    progress_competition,
)
from yield_rca_core.causal_investigation_models import (  # noqa: E402
    AlternativeLaneResolution,
    AlternativeLaneResolutionStatus,
    CandidateCompetitionStatus,
    CandidateResolutionStatus,
    CompetitionRequirement,
    CompetitionTrace,
    InvestigationGainReasonCode,
    InvestigationGainType,
)
from yield_rca_core.evidence_models import (  # noqa: E402
    EVIDENCE_SCHEMA_VERSION,
    EntityType,
    Evidence,
    EvidenceEntity,
    EvidenceSourceType,
    EvidenceType,
)
from yield_rca_core.investigation_decision import (  # noqa: E402
    assess_action_values,
    classify_investigation_gain,
    derive_investigation_gain_history,
    high_value_action_assessments,
)
from yield_rca_core.investigation_finalizer import (  # noqa: E402
    finalize_investigation,
    validate_terminal_investigation_state,
)
from yield_rca_core.investigation_models import (  # noqa: E402
    ActionKind,
    ActionRecord,
    EvidenceGapStatus,
    InvestigationAction,
    InvestigationGoal,
    InvestigationIntent,
    InvestigationQuestion,
    QuestionEvidenceLink,
    QuestionEvidenceRelation,
    QuestionKind,
    StopReason,
)
from yield_rca_core.models import (  # noqa: E402
    AgentFinding,
    AgentKind,
    ModelValidationError,
    RCAJob,
    RCAState,
    TaskStatus,
)
from yield_rca_core.next_action_planner import QwenNextActionPlanner  # noqa: E402


def _action(
    action_id: str,
    *,
    lane_id: str = "LANE_ALT",
    gap_id: str = "candidate_0.hypothesis_discrimination.parameter_anomaly",
) -> ActionRecord:
    return ActionRecord(
        action=InvestigationAction(
            action_id=action_id,
            kind=ActionKind.INSPECT_FDC_SPC.value,
            agent="fdc",
            reason="Inspect the exact Lane discriminator.",
            inputs={"lot_id": "LOT_01"},
            scope={
                "lot_id": "LOT_01",
                "candidate_id": "CANDIDATE_0",
                "lane_id": lane_id,
                "causal_gap_id": gap_id,
                "discriminator_kind": "parameter_anomaly",
            },
        ),
        status="completed",
        produced_evidence_ids=[f"EV_{action_id}"],
        decision_summary="The scoped observation completed.",
    )


def _evidence(evidence_id: str, *, missing: bool = False) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        source_type=EvidenceSourceType.FDC.value,
        source_id=f"fdc:{evidence_id}",
        summary=(
            "The requested FDC source is unavailable."
            if missing
            else "A typed process observation was collected."
        ),
        evidence_type=(
            EvidenceType.DATA_MISSING.value
            if missing
            else EvidenceType.PARAMETER_DEVIATION.value
        ),
        source_agent="fdc",
        source_tool="inspect_fdc_spc",
        observation=(
            "The requested FDC source is unavailable."
            if missing
            else "The process parameter differs from baseline."
        ),
        entities=(EvidenceEntity(EntityType.LOT.value, "LOT_01"),),
        confidence=1.0,
        evidence_schema_version=EVIDENCE_SCHEMA_VERSION,
    )


def _link(
    record: ActionRecord,
    *,
    relation: str,
) -> QuestionEvidenceLink:
    return QuestionEvidenceLink(
        question_id="Q_MECHANISM",
        evidence_id=record.produced_evidence_ids[0],
        action_id=record.action.action_id,
        relation=relation,
        matched_evidence_group="process_anomaly",
        reason="The typed result updates the scoped investigation.",
    )


@pytest.mark.parametrize(
    "relation",
    [
        QuestionEvidenceRelation.SUPPORTS.value,
        QuestionEvidenceRelation.CONTRADICTS.value,
    ],
)
def test_support_or_contradiction_is_evidence_gain(relation: str) -> None:
    record = _action("GAIN")
    evidence = _evidence(record.produced_evidence_ids[0])

    gain = classify_investigation_gain(
        record,
        earlier_records=[],
        evidence_by_id={evidence.evidence_id: evidence},
        links=[_link(record, relation=relation)],
    )

    assert gain.gain_type == InvestigationGainType.EVIDENCE_GAIN.value
    assert gain.reason_code == (
        InvestigationGainReasonCode.SUPPORTING_OR_CONTRADICTING_EVIDENCE.value
    )


def test_first_missing_is_state_gain_and_exact_repeat_is_no_gain() -> None:
    first = _action("MISSING_FIRST")
    first_evidence = _evidence(first.produced_evidence_ids[0], missing=True)
    first_gain = classify_investigation_gain(
        first,
        earlier_records=[],
        evidence_by_id={first_evidence.evidence_id: first_evidence},
        links=[
            _link(first, relation=QuestionEvidenceRelation.UNAVAILABLE.value)
        ],
    )
    assert first_gain.gain_type == InvestigationGainType.STATE_GAIN.value
    assert first_gain.reason_code == InvestigationGainReasonCode.UNAVAILABLE_SOURCE.value

    repeated = _action("MISSING_REPEAT")
    repeated_evidence = _evidence(
        repeated.produced_evidence_ids[0],
        missing=True,
    )
    repeated_gain = classify_investigation_gain(
        repeated,
        earlier_records=[first],
        evidence_by_id={repeated_evidence.evidence_id: repeated_evidence},
        links=[
            _link(repeated, relation=QuestionEvidenceRelation.UNAVAILABLE.value)
        ],
        earlier_gains=[first_gain],
    )
    assert repeated_gain.gain_type == InvestigationGainType.NO_GAIN.value
    assert repeated_gain.reason_code == (
        InvestigationGainReasonCode.REPEATED_UNAVAILABLE_SOURCE.value
    )


def test_exact_missing_scope_does_not_poison_another_lane() -> None:
    attempted = _action("MISSING_LANE_A", lane_id="LANE_A")
    attempted_evidence = _evidence(
        attempted.produced_evidence_ids[0],
        missing=True,
    )
    gain = classify_investigation_gain(
        attempted,
        earlier_records=[],
        evidence_by_id={attempted_evidence.evidence_id: attempted_evidence},
        links=[
            _link(attempted, relation=QuestionEvidenceRelation.UNAVAILABLE.value)
        ],
    )
    options = [
        {
            "option_id": "LANE_A",
            "action_kind": ActionKind.INSPECT_FDC_SPC.value,
            "source": "fdc",
            "scope": attempted.action.scope,
            "gap": {
                "gap_id": attempted.action.scope["causal_gap_id"],
                "gap_type": "hypothesis_discrimination",
                "candidate_id": "CANDIDATE_0",
                "discriminator_kind": "parameter_anomaly",
                "information_gain": 0.9,
            },
        },
        {
            "option_id": "LANE_B",
            "action_kind": ActionKind.INSPECT_FDC_SPC.value,
            "source": "fdc",
            "scope": {
                **attempted.action.scope,
                "lane_id": "LANE_B",
            },
            "gap": {
                "gap_id": attempted.action.scope["causal_gap_id"],
                "gap_type": "hypothesis_discrimination",
                "candidate_id": "CANDIDATE_0",
                "discriminator_kind": "parameter_anomaly",
                "information_gain": 0.9,
            },
        },
    ]

    assessments = assess_action_values(
        options=options,
        action_records=[attempted],
        gain_history=[gain],
        remaining_tool_budget=10,
    )

    assert not assessments[0].eligible
    assert assessments[1].high_value
    assert [item.option_id for item in high_value_action_assessments(assessments)] == [
        "LANE_B"
    ]


def test_missing_data_blocks_only_when_no_high_value_action_remains() -> None:
    record = _action("MISSING_BLOCK")
    missing = _evidence(record.produced_evidence_ids[0], missing=True)
    gain = classify_investigation_gain(
        record,
        earlier_records=[],
        evidence_by_id={missing.evidence_id: missing},
        links=[_link(record, relation=QuestionEvidenceRelation.UNAVAILABLE.value)],
    )
    trace = CompetitionTrace(
        active_lane_ids=("LANE_ALT",),
        represented_lane_ids=("LANE_ALT",),
        unresolved_lane_ids=("LANE_ALT",),
        lane_resolutions=(
            AlternativeLaneResolution(
                lane_id="LANE_ALT",
                status=AlternativeLaneResolutionStatus.UNRESOLVED.value,
                candidate_id="CANDIDATE_0",
                distinguishing_gap_ids=(gain.gap_id,),
            ),
        ),
        competition_requirement=(
            CompetitionRequirement.ALTERNATIVE_DISCOVERY_REQUIRED.value
        ),
        competition_status=CandidateCompetitionStatus.ACTIVE.value,
    )
    details = {
        "conclusion_status": "inconclusive",
        "ranked_candidates": [
            {
                "candidate_id": "CANDIDATE_0",
                "root_cause": "A still-unresolved process mechanism.",
                "status": "candidate",
            }
        ],
        "causal_evidence_gaps": [
            {
                "gap_id": gain.gap_id,
                "status": "unresolved",
            }
        ],
        "confirmation_gate": {
            "status": "inconclusive",
            "unresolved_gaps": [gain.gap_id],
        },
    }

    result = progress_competition(
        trace=trace,
        authoritative_details=details,
        action_value_assessments=[],
        gain_history=[gain],
        force_terminal=True,
    )

    assert result.trace.competition_status == (
        CandidateCompetitionStatus.BLOCKED_BY_MISSING_DATA.value
    )
    assert result.trace.candidate_resolutions[0].status == (
        CandidateResolutionStatus.UNRESOLVED.value
    )
    assert result.conclusion_status == "insufficient_evidence"
    assert result.blocking_data_missing_evidence_ids == (missing.evidence_id,)
    assert missing.evidence_id in result.trace.resolution_evidence_ids


def test_historical_unavailable_gain_does_not_relabel_current_gap_on_same_lane() -> None:
    old_gap_id = "candidate_0.hypothesis_discrimination.parameter_anomaly.old"
    current_gap_id = "candidate_0.hypothesis_discrimination.parameter_anomaly.current"
    historical = _action(
        "MISSING_OLD_GAP",
        lane_id="LANE_ALT",
        gap_id=old_gap_id,
    )
    missing = _evidence(historical.produced_evidence_ids[0], missing=True)
    historical_gain = classify_investigation_gain(
        historical,
        earlier_records=[],
        evidence_by_id={missing.evidence_id: missing},
        links=[
            _link(
                historical,
                relation=QuestionEvidenceRelation.UNAVAILABLE.value,
            )
        ],
    )
    trace = CompetitionTrace(
        active_lane_ids=("LANE_ALT",),
        represented_lane_ids=("LANE_ALT",),
        unresolved_lane_ids=("LANE_ALT",),
        lane_resolutions=(
            AlternativeLaneResolution(
                lane_id="LANE_ALT",
                status=AlternativeLaneResolutionStatus.UNRESOLVED.value,
                candidate_id="CANDIDATE_0",
                distinguishing_gap_ids=(current_gap_id,),
            ),
        ),
        competition_requirement=(
            CompetitionRequirement.ALTERNATIVE_DISCOVERY_REQUIRED.value
        ),
        competition_status=CandidateCompetitionStatus.ACTIVE.value,
    )

    result = progress_competition(
        trace=trace,
        authoritative_details={
            "conclusion_status": "inconclusive",
            "ranked_candidates": [
                {
                    "candidate_id": "CANDIDATE_0",
                    "root_cause": "The current candidate remains unresolved.",
                    "status": "candidate",
                }
            ],
            "causal_evidence_gaps": [
                {"gap_id": current_gap_id, "status": "unresolved"}
            ],
            "confirmation_gate": {
                "status": "inconclusive",
                "unresolved_gaps": [current_gap_id],
            },
        },
        action_value_assessments=[],
        gain_history=[historical_gain],
        force_terminal=False,
    )

    resolution = result.trace.lane_resolutions[0]
    assert resolution.status == AlternativeLaneResolutionStatus.UNRESOLVED.value
    assert resolution.reason_code is None
    assert resolution.distinguishing_gap_ids == (current_gap_id,)


def _formal_trace() -> CompetitionTrace:
    return CompetitionTrace(
        competition_requirement=(
            CompetitionRequirement.ALTERNATIVE_DISCOVERY_REQUIRED.value
        ),
        competition_status=CandidateCompetitionStatus.ACTIVE.value,
    )


def _formal_state_with_question_group(matched_group: str) -> RCAState:
    evidence = Evidence(
        evidence_id="EV_PRODUCT_SIGNAL",
        source_type=EvidenceSourceType.DEFECT.value,
        source_id="defect:EV_PRODUCT_SIGNAL",
        summary="A typed product signal was collected.",
        evidence_type=EvidenceType.DEFECT_SIGNAL.value,
        source_agent=AgentKind.DEFECT_WAT.value,
        source_tool="inspect_defect_pattern",
        observation="The product signal differs from the passing control.",
        entities=(EvidenceEntity(EntityType.LOT.value, "LOT_01"),),
        confidence=1.0,
        evidence_schema_version=EVIDENCE_SCHEMA_VERSION,
    )
    question = InvestigationQuestion(
        question_id="Q_PRODUCT_SIGNAL",
        goal_id="GOAL_RCA",
        question="What product signature is present?",
        rationale="The product signal is required for RCA closure.",
        question_kind=QuestionKind.DEFECT_SIGNATURE.value,
    )
    action = ActionRecord(
        action=InvestigationAction(
            action_id="ACTION_PRODUCT_SIGNAL",
            kind=ActionKind.INSPECT_DEFECT_PATTERN.value,
            agent=AgentKind.DEFECT_WAT.value,
            reason="Collect the typed product signal.",
            inputs={"lot_id": "LOT_01"},
            scope={"lot_id": "LOT_01"},
        ),
        status="completed",
        produced_evidence_ids=[evidence.evidence_id],
        decision_summary="The product observation completed.",
    )
    link = QuestionEvidenceLink(
        question_id=question.question_id,
        evidence_id=evidence.evidence_id,
        action_id=action.action.action_id,
        relation=QuestionEvidenceRelation.SUPPORTS.value,
        matched_evidence_group=matched_group,
        reason="The typed Evidence addresses the investigation question.",
    )
    finding = AgentFinding(
        finding_id="RCA_AUTH_QUESTION_RECONCILIATION",
        agent=AgentKind.RCA_REASONING.value,
        summary="The root cause remains unresolved.",
        confidence=0.4,
        evidence_ids=[evidence.evidence_id],
        evidence=[evidence],
        details={
            "status": "inconclusive",
            "root_cause": "inconclusive",
            "conclusion_status": "inconclusive",
            "ranked_candidates": [
                {
                    "candidate_id": "CANDIDATE_0",
                    "root_cause": "An unresolved mechanism.",
                    "status": "candidate",
                }
            ],
            "confirmation_gate": {"status": "inconclusive"},
        },
    )
    return RCAState(
        job=RCAJob(
            job_id="QUESTION_RECONCILIATION",
            user_query="Complete the evidence-bounded RCA.",
            status=TaskStatus.RUNNING.value,
        ),
        evidence=[evidence],
        findings=[finding],
        authoritative_rca_finding_id=finding.finding_id,
        investigation_goal=InvestigationGoal(
            goal_id="GOAL_RCA",
            intent=InvestigationIntent.FULL_RCA.value,
            summary="Complete the evidence-bounded RCA.",
        ),
        action_history=[action],
        investigation_questions=[question],
        question_evidence_links=[link],
        evidence_gaps=[question.question],
        competition_trace=_formal_trace(),
    )


def test_finalizer_closes_question_when_all_required_groups_are_supported() -> None:
    finalized = finalize_investigation(
        _formal_state_with_question_group("product_signal")
    )

    assert finalized.investigation_questions[0].status == (
        EvidenceGapStatus.CLOSED.value
    )
    assert finalized.investigation_questions[0].evidence_ids == [
        "EV_PRODUCT_SIGNAL"
    ]
    assert finalized.evidence_gaps == []


def test_finalizer_keeps_question_open_when_required_group_is_missing() -> None:
    finalized = finalize_investigation(
        _formal_state_with_question_group("context")
    )

    assert finalized.investigation_questions[0].status == (
        EvidenceGapStatus.OPEN.value
    )
    assert finalized.evidence_gaps == ["What product signature is present?"]


def test_confirmation_gate_support_completes_competition() -> None:
    result = progress_competition(
        trace=_formal_trace(),
        authoritative_details={
            "conclusion_status": "supported",
            "ranked_candidates": [
                {
                    "candidate_id": "CANDIDATE_0",
                    "root_cause": "A confirmed typed-Evidence root cause.",
                    "status": "candidate",
                }
            ],
            "confirmation_gate": {"status": "supported"},
        },
        action_value_assessments=[],
        gain_history=[],
        force_terminal=True,
    )

    assert result.trace.competition_status == (
        CandidateCompetitionStatus.COMPLETE_CONFIRMED.value
    )
    assert result.trace.terminal_reason == (
        "confirmation_gate_supported_unique_candidate"
    )
    assert result.conclusion_status == "supported"


def test_all_contradicted_candidates_complete_rejected() -> None:
    result = progress_competition(
        trace=_formal_trace(),
        authoritative_details={
            "conclusion_status": "inconclusive",
            "ranked_candidates": [
                {
                    "candidate_id": "CANDIDATE_0",
                    "root_cause": "A contradicted candidate.",
                    "status": "conflicted",
                },
                {
                    "candidate_id": "CANDIDATE_1",
                    "root_cause": "Another contradicted candidate.",
                    "status": "rejected",
                },
            ],
            "confirmation_gate": {"status": "inconclusive"},
        },
        action_value_assessments=[],
        gain_history=[],
        force_terminal=True,
    )

    assert result.trace.competition_status == (
        CandidateCompetitionStatus.COMPLETE_REJECTED.value
    )
    assert all(
        item.status == CandidateResolutionStatus.CONTRADICTED.value
        for item in result.trace.candidate_resolutions
    )
    assert result.conclusion_status == "inconclusive"


def test_no_high_value_action_without_missing_data_is_exhausted() -> None:
    result = progress_competition(
        trace=_formal_trace(),
        authoritative_details={
            "conclusion_status": "inconclusive",
            "ranked_candidates": [
                {
                    "candidate_id": "CANDIDATE_0",
                    "root_cause": "An unresolved candidate.",
                    "status": "candidate",
                }
            ],
            "confirmation_gate": {"status": "inconclusive"},
        },
        action_value_assessments=[],
        gain_history=[],
        force_terminal=True,
    )

    assert result.trace.competition_status == (
        CandidateCompetitionStatus.EXHAUSTED.value
    )
    assert result.conclusion_status == "inconclusive"


def test_legacy_unformed_competition_is_not_a_processing_failure() -> None:
    record = _action("LEGACY_SCOPE_MISSING")
    missing = _evidence(record.produced_evidence_ids[0], missing=True)
    gain = classify_investigation_gain(
        record,
        earlier_records=[],
        evidence_by_id={missing.evidence_id: missing},
        links=[_link(record, relation=QuestionEvidenceRelation.UNAVAILABLE.value)],
    )
    trace = CompetitionTrace(
        active_lane_ids=("LANE_ALT",),
        represented_lane_ids=("LANE_ALT",),
        unresolved_lane_ids=("LANE_ALT",),
        competition_requirement=CompetitionRequirement.SCOPE_REQUIRED.value,
        # Legacy State encoded an unformed, but otherwise valid, Candidate
        # competition as a processing failure.
        competition_status=CandidateCompetitionStatus.FAILED.value,
        competition_failure_reason="scope_hypothesis_collapsed",
    )

    result = progress_competition(
        trace=trace,
        authoritative_details={
            "conclusion_status": "insufficient_evidence",
            "ranked_candidates": [
                {
                    "candidate_id": "CANDIDATE_0",
                    "root_cause": "A valid but unresolved scope hypothesis.",
                    "status": "candidate",
                }
            ],
            "confirmation_gate": {
                "status": "insufficient_evidence",
                "blocking_data_missing_evidence_ids": [missing.evidence_id],
            },
        },
        action_value_assessments=[],
        gain_history=[gain],
        force_terminal=True,
    )

    assert result.trace.competition_status == (
        CandidateCompetitionStatus.BLOCKED_BY_MISSING_DATA.value
    )
    assert result.trace.competition_failure_reason is None
    assert result.trace.competition_gap_reason == "scope_hypothesis_collapsed"
    assert result.trace.terminal_reason == (
        "no_high_value_action_after_required_source_unavailable"
    )
    assert result.conclusion_status == "insufficient_evidence"


def test_candidate_validation_exhaustion_remains_a_processing_failure() -> None:
    trace = CompetitionTrace(
        competition_requirement=CompetitionRequirement.SCOPE_REQUIRED.value,
        competition_status=CandidateCompetitionStatus.FAILED.value,
        competition_failure_reason="candidate_validation_exhausted",
    )

    result = progress_competition(
        trace=trace,
        authoritative_details={
            "conclusion_status": "inconclusive",
            "ranked_candidates": [],
            "confirmation_gate": {"status": "inconclusive"},
        },
        action_value_assessments=[],
        gain_history=[],
        force_terminal=True,
    )

    assert result.trace.competition_status == CandidateCompetitionStatus.FAILED.value
    assert result.trace.terminal_reason == "candidate_competition_processing_failed"


def test_candidate_provider_failure_before_requirement_is_processing_failure() -> None:
    trace = CompetitionTrace(
        competition_requirement=CompetitionRequirement.NOT_EVALUATED.value,
        competition_status=CandidateCompetitionStatus.FAILED.value,
        competition_failure_reason="candidate_provider_failed",
    )

    result = progress_competition(
        trace=trace,
        authoritative_details={
            "conclusion_status": "insufficient_evidence",
            "ranked_candidates": [],
            "confirmation_gate": {"status": "insufficient_evidence"},
        },
        action_value_assessments=[],
        gain_history=[],
        force_terminal=True,
    )

    assert result.trace.competition_status == CandidateCompetitionStatus.FAILED.value
    assert result.trace.terminal_reason == "candidate_competition_processing_failed"
    assert result.conclusion_status == "inconclusive"


def test_budget_exhaustion_is_distinct_from_no_gain_exhaustion() -> None:
    result = progress_competition(
        trace=_formal_trace(),
        authoritative_details={
            "conclusion_status": "inconclusive",
            "ranked_candidates": [
                {
                    "candidate_id": "CANDIDATE_0",
                    "root_cause": "An unresolved candidate.",
                    "status": "candidate",
                }
            ],
            "confirmation_gate": {"status": "inconclusive"},
        },
        action_value_assessments=[],
        gain_history=[],
        force_terminal=True,
        budget_exhausted=True,
    )

    assert result.trace.competition_status == (
        CandidateCompetitionStatus.BUDGET_EXHAUSTED.value
    )
    assert result.conclusion_status == "insufficient_evidence"


def test_confirmation_gate_blocking_missing_data_needs_no_action_gain() -> None:
    result = progress_competition(
        trace=_formal_trace(),
        authoritative_details={
            "conclusion_status": "insufficient_evidence",
            "ranked_candidates": [
                {
                    "candidate_id": "CANDIDATE_0",
                    "root_cause": "An unresolved candidate.",
                    "status": "candidate",
                }
            ],
            "confirmation_gate": {
                "status": "insufficient_evidence",
                "data_missing_evidence_ids": ["EV_OPTIONAL_MISSING", "EV_REQUIRED"],
                "blocking_data_missing_evidence_ids": ["EV_REQUIRED"],
            },
        },
        action_value_assessments=[],
        gain_history=[],
        force_terminal=True,
    )

    assert result.trace.competition_status == (
        CandidateCompetitionStatus.BLOCKED_BY_MISSING_DATA.value
    )
    assert result.blocking_data_missing_evidence_ids == ("EV_REQUIRED",)


def test_explicitly_non_blocking_missing_data_does_not_block_competition() -> None:
    result = progress_competition(
        trace=_formal_trace(),
        authoritative_details={
            "conclusion_status": "insufficient_evidence",
            "ranked_candidates": [
                {
                    "candidate_id": "CANDIDATE_0",
                    "root_cause": "An unresolved candidate.",
                    "status": "candidate",
                }
            ],
            "confirmation_gate": {
                "status": "insufficient_evidence",
                "data_missing_evidence_ids": ["EV_OPTIONAL_MISSING"],
                "blocking_data_missing_evidence_ids": [],
            },
        },
        action_value_assessments=[],
        gain_history=[],
        force_terminal=True,
    )

    assert result.trace.competition_status == (
        CandidateCompetitionStatus.EXHAUSTED.value
    )
    assert result.blocking_data_missing_evidence_ids == ()


def test_finalizer_does_not_take_over_lane_only_non_rca_state() -> None:
    state = RCAState(
        job=RCAJob(
            job_id="IMPACT_ONLY",
            user_query="Identify impact lots.",
            status=TaskStatus.RUNNING.value,
        ),
        competition_trace=CompetitionTrace(
            competition_requirement=CompetitionRequirement.NOT_EVALUATED.value,
            competition_status=CandidateCompetitionStatus.ACTIVE.value,
        ),
        goal_status="satisfied",
        conclusion_level="signal",
        stop_reason=StopReason.GOAL_SATISFIED.value,
    )

    finalized = finalize_investigation(state)

    assert finalized.goal_status == state.goal_status
    assert finalized.conclusion_level == state.conclusion_level
    assert finalized.stop_reason == state.stop_reason
    assert finalized.execution_metadata["investigation_finalizer"] == (
        "not_applicable"
    )

    completed = replace(
        finalized,
        job=replace(finalized.job, status=TaskStatus.COMPLETED.value),
    )
    validate_terminal_investigation_state(completed)


def test_legacy_state_without_patch3_fields_still_deserializes() -> None:
    legacy = RCAState.from_dict(
        {
            "job": RCAJob(
                job_id="LEGACY",
                user_query="Read an older State.",
                status=TaskStatus.COMPLETED.value,
            ).to_dict()
        }
    )

    assert legacy.competition_trace is None
    assert legacy.investigation_gain_history == []
    assert legacy.latest_action_value_assessments == []
    validate_terminal_investigation_state(legacy)


def test_completed_job_rejects_active_competition() -> None:
    replay_path = (
        ROOT
        / "outputs"
        / "patch12_qwen_smoke_FORMAL009_r1"
        / "states"
        / "FORMAL_009.json"
    )
    if not replay_path.exists():
        pytest.skip("FORMAL_009 local replay State is not available")
    state = RCAState.from_dict(json.loads(replay_path.read_text(encoding="utf-8")))
    assert state.job.status == TaskStatus.COMPLETED.value
    assert state.competition_trace is not None
    assert state.competition_trace.competition_status == (
        CandidateCompetitionStatus.ACTIVE.value
    )
    with pytest.raises(ModelValidationError, match="cannot retain"):
        validate_terminal_investigation_state(state)


class _NoPlannerCallClient:
    provider = "offline"
    model = "no-call"

    def __init__(self) -> None:
        self.call_count = 0

    def complete_json(self, request: object) -> object:
        self.call_count += 1
        raise AssertionError("FORMAL_009 replay must terminate before a Qwen call")


def test_formal_009_offline_replay_closes_missing_data_without_qwen() -> None:
    replay_path = (
        ROOT
        / "outputs"
        / "patch12_qwen_smoke_FORMAL009_r1"
        / "states"
        / "FORMAL_009.json"
    )
    if not replay_path.exists():
        pytest.skip("FORMAL_009 local replay State is not available")
    original = RCAState.from_dict(
        json.loads(replay_path.read_text(encoding="utf-8"))
    )
    last_decision = original.planner_decisions[-1]
    terminal_question_ids = {
        item.question_id for item in last_decision.question_updates
    }
    questions = [
        (
            replace(
                question,
                status=EvidenceGapStatus.OPEN.value,
                answer=None,
                evidence_ids=[],
                unavailable_reason=None,
            )
            if question.question_id in terminal_question_ids
            else question
        )
        for question in original.investigation_questions
    ]
    prior_decision_ids = {
        decision.decision_id for decision in original.planner_decisions[:-1]
    }
    replay = replace(
        original,
        job=replace(original.job, status=TaskStatus.RUNNING.value),
        report=None,
        investigation_questions=questions,
        planner_decisions=original.planner_decisions[:-1],
        question_update_reviews=[
            review
            for review in original.question_update_reviews
            if review.decision_id in prior_decision_ids
        ],
        goal_status=None,
        conclusion_level=None,
        evidence_gaps=[],
        stop_reason=None,
        run_evaluation=None,
        investigation_gain_history=list(
            derive_investigation_gain_history(
                action_records=original.action_history,
                evidence=original.evidence,
                links=original.question_evidence_links,
            )
        ),
        latest_action_value_assessments=[],
    )
    client = _NoPlannerCallClient()
    planner = QwenNextActionPlanner(client)
    outcome = planner.decide_with_review(
        goal=replay.investigation_goal,
        questions=replay.investigation_questions,
        findings=replay.findings,
        action_records=replay.action_history,
        tool_call_count=int(replay.execution_metadata.get("tool_call_count", 0)),
        evidence=replay.evidence,
        evidence_ids=[item.evidence_id for item in replay.evidence],
        question_evidence_links=replay.question_evidence_links,
        capability_notices=replay.capability_notices,
        hypotheses=replay.hypotheses,
        prior_decisions=replay.planner_decisions,
        authoritative_rca_finding_id=replay.authoritative_rca_finding_id,
        investigation_gain_history=replay.investigation_gain_history,
    )

    assert client.call_count == 0
    assert outcome.decision.stop_reason == StopReason.DATA_UNAVAILABLE.value
    finalized = finalize_investigation(
        replace(
            replay,
            latest_action_value_assessments=outcome.action_value_assessments,
        ),
        action_value_assessments=outcome.action_value_assessments,
    )
    terminal = replace(
        finalized,
        job=replace(finalized.job, status=TaskStatus.COMPLETED.value),
    )
    validate_terminal_investigation_state(terminal)

    assert terminal.competition_trace is not None
    assert terminal.competition_trace.competition_status == (
        CandidateCompetitionStatus.BLOCKED_BY_MISSING_DATA.value
    )
    assert terminal.stop_reason == StopReason.DATA_UNAVAILABLE.value
    assert terminal.authoritative_rca_finding is not None
    assert terminal.authoritative_rca_finding.details["conclusion_status"] == (
        "insufficient_evidence"
    )
    target_lane = next(
        lane
        for lane in terminal.causal_lanes
        if lane.lane_id
        == "lane:1000:EQ_4D35CA:EQ_4D35CA_CH02:RCP_CF41B53F"
    )
    assert target_lane.lifecycle_status != "blocked"
