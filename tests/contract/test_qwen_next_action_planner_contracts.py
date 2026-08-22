from __future__ import annotations

import copy
import inspect
import sys
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core"))

from yield_rca_core import (  # noqa: E402
    LLM_REACT_EXECUTABLE_ACTION_KINDS,
    EvidenceGapStatus,
    InvestigationAction,
    InvestigationGoal,
    InvestigationIntent,
    InvestigationQuestion,
    PlannerDecision,
    QuestionEvidenceLink,
    QuestionEvidenceRelation,
    QuestionUpdateDisposition,
    QuestionUpdateReasonCode,
    QwenNextActionPlanner,
    QwenNextActionPlannerError,
)
from yield_rca_core.evidence_models import (  # noqa: E402
    EVIDENCE_SCHEMA_VERSION,
    EntityType,
    Evidence,
    EvidenceEntity,
    EvidenceSourceType,
    EvidenceType,
)
from yield_rca_core.investigation_models import (  # noqa: E402
    ActionKind,
    ActionRecord,
    ConclusionLevel,
    DecisionType,
    GoalStatus,
    StopReason,
)
from yield_rca_core.llm_gateway import (  # noqa: E402
    FakeLLMClient,
    LLMCallError,
    LLMRequest,
    LLMResponse,
    capture_llm_usage,
    load_prompt,
)
from yield_rca_core.models import AgentFinding, AgentKind  # noqa: E402


def goal(*, max_steps: int = 8) -> InvestigationGoal:
    return InvestigationGoal(
        goal_id="GOAL_LOT_01",
        intent=InvestigationIntent.ROOT_CAUSE.value,
        summary="Investigate the Cu CMP scratch on LOT_01.",
        known_facts={
            "lot_id": "LOT_01",
            "module": "CU_CMP",
            "defect": "scratch",
        },
        required_evidence=["defect_signature", "process_mechanism"],
        max_steps=max_steps,
    )


def questions() -> list[InvestigationQuestion]:
    return [
        InvestigationQuestion(
            question_id="Q_DEFECT",
            goal_id="GOAL_LOT_01",
            question="What is the source Lot scratch signature?",
            rationale="The symptom must be characterized before mechanism analysis.",
            scope={"lot_id": "LOT_01", "module": "CU_CMP"},
        ),
        InvestigationQuestion(
            question_id="Q_MECHANISM",
            goal_id="GOAL_LOT_01",
            question="Which Cu CMP mechanism explains the scratch?",
            rationale="The requested outcome is an evidence-backed root cause.",
            scope={"lot_id": "LOT_01", "module": "CU_CMP"},
        ),
    ]


def finding(agent: str, *, evidence_id: str | None = None) -> AgentFinding:
    normalized_evidence_id = evidence_id or f"EV_{agent.upper()}"
    return AgentFinding(
        finding_id=f"FINDING_{agent.upper()}",
        agent=agent,
        summary=f"{agent} observation",
        confidence=0.8,
        evidence_ids=[normalized_evidence_id],
        details={"observation": f"{agent} evidence is available"},
    )


def action_record(
    *,
    kind: str,
    agent: str,
    scope: dict[str, Any],
    action_id: str | None = None,
) -> ActionRecord:
    return ActionRecord(
        action=InvestigationAction(
            action_id=action_id or f"PRIOR_{kind}",
            kind=kind,
            agent=agent,
            reason="Earlier investigation action.",
            inputs={"lot_id": "LOT_01"},
            scope=scope,
        ),
        status="completed",
        decision_summary="Earlier observation recorded.",
    )


def model_act_payload(
    request: LLMRequest,
    *,
    kind: str = ActionKind.FIND_SHARED_EXPOSURE.value,
    agent: str = AgentKind.MES.value,
) -> dict[str, Any]:
    return {
        "decision_id": f"MODEL_DECISION_{request.payload['output_attempt']}",
        "goal_id": request.payload["goal"]["goal_id"],
        "decision_type": DecisionType.ACT.value,
        "reason": "The model selected a useful registered action.",
        "goal_status": GoalStatus.IN_PROGRESS.value,
        "proposed_conclusion_level": ConclusionLevel.SIGNAL.value,
        "next_action": {
            "action_id": f"MODEL_ACTION_{request.payload['output_attempt']}",
            "kind": kind,
            "agent": agent,
            "reason": "Collect the observation needed by the open question.",
            "inputs": {"lot_id": "LOT_01"},
            "scope": {"lot_id": "LOT_01", "module": "CU_CMP"},
            "required_evidence_ids": [],
            "max_attempts": 1,
        },
        "target_question_ids": ["Q_MECHANISM"],
        "new_questions": [],
        "stop_reason": None,
        "question_updates": [],
    }


Mutation = Callable[[dict[str, Any], LLMRequest], None]


class RecordingNextActionClient(FakeLLMClient):
    def __init__(self, mutation: Mutation | None = None) -> None:
        self.requests: list[LLMRequest] = []
        self.mutation = mutation

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        response = super().complete_json(request)
        payload = copy.deepcopy(response.data)
        if self.mutation is not None:
            self.mutation(payload, request)
        return LLMResponse(data=payload, usage=response.usage)


class ModelSelectedMESClient(RecordingNextActionClient):
    def complete_json(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        response = FakeLLMClient.complete_json(self, request)
        return LLMResponse(
            data=model_act_payload(request),
            usage=response.usage,
        )


class InvalidThenValidClient(RecordingNextActionClient):
    def complete_json(self, request: LLMRequest) -> LLMResponse:
        response = super().complete_json(request)
        if len(self.requests) == 1:
            payload = model_act_payload(
                request,
                kind=ActionKind.INSPECT_FDC_SPC.value,
                agent=AgentKind.FDC.value,
            )
            return LLMResponse(data=payload, usage=response.usage)
        return response


class TransientCallFailureClient(RecordingNextActionClient):
    def complete_json(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            raise LLMCallError(
                "temporary provider failure",
                status_code=429,
                provider_code="Throttling",
                failure_category="provider_http_error",
            )
        return FakeLLMClient.complete_json(self, request)


class PersistentCallFailureClient(RecordingNextActionClient):
    def complete_json(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        raise LLMCallError(
            "provider unavailable",
            status_code=429,
            provider_code="Throttling",
            provider_message=(
                "Authorization: Bearer planner-secret api_key=planner-secret"
            ),
            request_id="req-429",
            failure_category="provider_http_error",
        )


class QwenNextActionPlannerContractTest(unittest.TestCase):
    @staticmethod
    def _causal_gap_runtime() -> tuple[
        InvestigationQuestion,
        list[AgentFinding],
        list[ActionRecord],
        list[QuestionEvidenceLink],
        str,
    ]:
        mechanism = questions()[1]
        gap_id = "candidate_0.mechanism.incomplete"
        rca = AgentFinding(
            finding_id="FINDING_RCA_AUTHORITATIVE",
            agent=AgentKind.RCA_REASONING.value,
            summary="The current candidate still lacks mechanism support.",
            confidence=0.5,
            evidence_ids=["EV_RCA_TRACE"],
            details={
                "causal_evidence_gaps": [
                    {
                        "gap_id": gap_id,
                        "candidate_index": 0,
                        "claim": "mechanism",
                        "status": "incomplete",
                        "reason": "Approved mechanism Evidence is missing.",
                        "question_kind": "process_mechanism",
                        "allowed_actions": [
                            ActionKind.VALIDATE_HISTORICAL_CASE.value
                        ],
                        "evidence_ids": ["EV_RCA_TRACE"],
                    }
                ]
            },
        )
        findings = [
            finding(AgentKind.MES.value),
            finding(AgentKind.FDC.value),
            finding(AgentKind.DEFECT_WAT.value),
            rca,
        ]
        records: list[ActionRecord] = []
        links: list[QuestionEvidenceLink] = []
        for index, group in enumerate(
            (
                "process_anomaly",
                "product_signal",
                "shared_exposure",
                "shared_product_signal",
            ),
            start=1,
        ):
            action_id = f"RCA_ROUND_{min(index, 2)}"
            evidence_id = f"EV_GAIN_{index}"
            links.append(
                QuestionEvidenceLink(
                    question_id=mechanism.question_id,
                    evidence_id=evidence_id,
                    action_id=action_id,
                    relation=QuestionEvidenceRelation.SUPPORTS.value,
                    matched_evidence_group=group,
                    reason=f"The observation fills {group}.",
                )
            )
        for index in (1, 2):
            records.append(
                ActionRecord(
                    action=InvestigationAction(
                        action_id=f"RCA_ROUND_{index}",
                        kind=ActionKind.RUN_RCA_REASONING.value,
                        agent=AgentKind.RCA_REASONING.value,
                        reason="Compare the current candidates.",
                        inputs={"lot_id": "LOT_01"},
                        scope={"lot_id": "LOT_01", "round": index},
                    ),
                    status="completed",
                    produced_evidence_ids=[
                        f"EV_GAIN_{2 * index - 1}",
                        f"EV_GAIN_{2 * index}",
                    ],
                    decision_summary="Candidate comparison completed.",
                )
            )
        return mechanism, findings, records, links, gap_id

    def test_authoritative_gap_selects_the_only_legal_action_without_qwen(self) -> None:
        mechanism, findings, records, links, gap_id = self._causal_gap_runtime()
        client = RecordingNextActionClient()

        outcome = QwenNextActionPlanner(client).decide_with_review(
            goal=goal(),
            questions=[mechanism],
            findings=findings,
            action_records=records,
            tool_call_count=2,
            evidence_ids=[link.evidence_id for link in links],
            question_evidence_links=links,
            authoritative_rca_finding_id="FINDING_RCA_AUTHORITATIVE",
        )

        self.assertEqual(client.requests, [])
        self.assertEqual(outcome.decision_proposed_by, "python_runtime")
        self.assertEqual(
            outcome.decision.next_action.kind,
            ActionKind.VALIDATE_HISTORICAL_CASE.value,
        )
        self.assertEqual(
            outcome.decision.next_action.scope["causal_gap_id"],
            gap_id,
        )

    def test_typed_product_discriminator_observes_lane_then_refreshes_reasoning(self) -> None:
        mechanism = questions()[1]
        gap_id = "candidate_0.hypothesis_discrimination.product_outcome.lane_runtime"
        target_scope = {
            "lane_id": "LANE_RUNTIME",
            "operation": "OP_RUNTIME",
            "equipment": "TOOL_RUNTIME",
            "chamber": "CH_RUNTIME",
            "recipe": "RCP_RUNTIME",
            "discriminator_kind": "product_outcome",
        }
        mes = finding(AgentKind.MES.value)
        mes.details["lane_candidates"] = [
            {
                **target_scope,
                "parameter_scope": [],
                "exposed_lot_ids": ["LOT_01"],
            }
        ]
        rca = AgentFinding(
            finding_id="FINDING_TYPED_RCA",
            agent=AgentKind.RCA_REASONING.value,
            summary="The competing Lane requires product-outcome discrimination.",
            confidence=0.5,
            evidence_ids=["EV_RCA_TRACE"],
            details={
                "causal_evidence_gaps": [
                    {
                        "gap_id": gap_id,
                        "gap_type": "hypothesis_discrimination",
                        "discriminator_kind": "product_outcome",
                        "candidate_index": 0,
                        "candidate_id": "C_RUNTIME",
                        "claim": "hypothesis_discrimination",
                        "status": "unresolved",
                        "reason": "Compare the product outcome on the alternative Lane.",
                        "question_kind": "process_mechanism",
                        "allowed_actions": [
                            ActionKind.VALIDATE_SHARED_DEFECT_PATTERN.value,
                            ActionKind.RUN_RCA_REASONING.value,
                        ],
                        "preferred_action": (
                            ActionKind.VALIDATE_SHARED_DEFECT_PATTERN.value
                        ),
                        "refresh_action": ActionKind.RUN_RCA_REASONING.value,
                        "required_evidence_groups": ["shared_product_signal"],
                        "target_scope": target_scope,
                        "challenge_selected": True,
                    }
                ]
            },
        )
        findings = [
            mes,
            finding(AgentKind.FDC.value),
            finding(AgentKind.DEFECT_WAT.value),
            rca,
        ]
        planner = QwenNextActionPlanner(RecordingNextActionClient())

        observation = planner.decide_with_review(
            goal=goal(),
            questions=[mechanism],
            findings=findings,
            action_records=[],
            tool_call_count=0,
            evidence_ids=["EV_RCA_TRACE"],
            question_evidence_links=[],
            authoritative_rca_finding_id="FINDING_TYPED_RCA",
        )

        self.assertEqual(
            observation.decision.next_action.kind,
            ActionKind.VALIDATE_SHARED_DEFECT_PATTERN.value,
        )
        self.assertEqual(
            observation.decision.next_action.scope["causal_gap_id"],
            gap_id,
        )
        for key, value in target_scope.items():
            self.assertEqual(observation.decision.next_action.scope[key], value)

        records = [
            ActionRecord(
                action=InvestigationAction(
                    action_id="FIRST_RCA_REASONING",
                    kind=ActionKind.RUN_RCA_REASONING.value,
                    agent=AgentKind.RCA_REASONING.value,
                    reason="Create the first evidence-bounded candidate.",
                    inputs={"lot_id": "LOT_01"},
                    scope={"lot_id": "LOT_01"},
                ),
                status="completed",
                produced_evidence_ids=["EV_RCA_TRACE"],
                decision_summary="The first RCA candidate was generated.",
            ),
            ActionRecord(
                action=observation.decision.next_action,
                status="completed",
                produced_evidence_ids=["EV_PRODUCT_GAIN"],
                decision_summary="The Lane-scoped product outcome was observed.",
            )
        ]
        prior_decisions = [
            PlannerDecision(
                decision_id="FIRST_RCA_DECISION",
                goal_id=goal().goal_id,
                decision_type=DecisionType.ACT.value,
                reason="Generate the first candidate.",
                goal_status=GoalStatus.IN_PROGRESS.value,
                proposed_conclusion_level=ConclusionLevel.CANDIDATE.value,
                next_action=records[0].action,
                target_question_ids=[mechanism.question_id],
            ),
            observation.decision,
        ]
        gain_link = QuestionEvidenceLink(
            question_id=mechanism.question_id,
            evidence_id="EV_PRODUCT_GAIN",
            action_id=observation.decision.next_action.action_id,
            relation=QuestionEvidenceRelation.SUPPORTS.value,
            matched_evidence_group="shared_product_signal",
            reason="The scoped observation adds product Evidence for the challenge.",
        )
        refreshed = planner.decide_with_review(
            goal=goal(),
            questions=[mechanism],
            findings=findings,
            action_records=records,
            tool_call_count=1,
            evidence_ids=["EV_RCA_TRACE", "EV_PRODUCT_GAIN"],
            question_evidence_links=[gain_link],
            prior_decisions=prior_decisions,
            authoritative_rca_finding_id="FINDING_TYPED_RCA",
        )

        self.assertEqual(
            refreshed.decision.next_action.kind,
            ActionKind.RUN_RCA_REASONING.value,
        )
        self.assertEqual(refreshed.decision.next_action.scope["causal_gap_id"], gap_id)

        consumed_records = [
            *records,
            ActionRecord(
                action=refreshed.decision.next_action,
                status="completed",
                produced_evidence_ids=["EV_RCA_TRACE", "EV_PRODUCT_GAIN"],
                decision_summary="The new Evidence was compared in the second round.",
            ),
        ]
        consumed = planner.decide_with_review(
            goal=goal(),
            questions=[mechanism],
            findings=findings,
            action_records=consumed_records,
            tool_call_count=3,
            evidence_ids=["EV_RCA_TRACE", "EV_PRODUCT_GAIN"],
            question_evidence_links=[gain_link],
            prior_decisions=[*prior_decisions, refreshed.decision],
            authoritative_rca_finding_id="FINDING_TYPED_RCA",
        )

        self.assertEqual(consumed.decision.decision_type, DecisionType.STOP.value)
        self.assertEqual(consumed.decision.stop_reason, StopReason.NO_ALLOWED_ACTION.value)

    def test_third_reasoning_round_requires_unconsumed_discriminative_evidence(
        self,
    ) -> None:
        mechanism = questions()[1]
        gap_id = "candidate_0.hypothesis_discrimination.product_outcome.lane_round_3"
        target_scope = {
            "lane_id": "LANE_ROUND_3",
            "operation": "OP_ROUND_3",
            "equipment": "TOOL_ROUND_3",
            "chamber": "CH_ROUND_3",
            "recipe": "RCP_ROUND_3",
            "discriminator_kind": "product_outcome",
        }
        mes = finding(AgentKind.MES.value)
        mes.details["lane_candidates"] = [
            {
                **target_scope,
                "parameter_scope": ["pressure_cv"],
                "exposed_lot_ids": ["LOT_01"],
            }
        ]
        rca_round_1 = AgentFinding(
            finding_id="FINDING_RCA_ROUND_1",
            agent=AgentKind.RCA_REASONING.value,
            summary="The first candidate comparison is incomplete.",
            confidence=0.4,
            evidence_ids=["EV_RCA_ROUND_1"],
        )
        rca_round_2 = AgentFinding(
            finding_id="FINDING_RCA_ROUND_2",
            agent=AgentKind.RCA_REASONING.value,
            summary="A second comparison selected one more discriminator.",
            confidence=0.5,
            evidence_ids=["EV_RCA_ROUND_1", "EV_RCA_ROUND_2"],
            details={
                "causal_evidence_gaps": [
                    {
                        "gap_id": gap_id,
                        "gap_type": "hypothesis_discrimination",
                        "discriminator_kind": "product_outcome",
                        "candidate_index": 0,
                        "candidate_id": "C_ROUND_3",
                        "claim": "hypothesis_discrimination",
                        "status": "unresolved",
                        "reason": "Compare the alternative Lane product outcome.",
                        "question_kind": "process_mechanism",
                        "allowed_actions": [
                            ActionKind.VALIDATE_SHARED_DEFECT_PATTERN.value,
                            ActionKind.RUN_RCA_REASONING.value,
                        ],
                        "preferred_action": (
                            ActionKind.VALIDATE_SHARED_DEFECT_PATTERN.value
                        ),
                        "refresh_action": ActionKind.RUN_RCA_REASONING.value,
                        "required_evidence_groups": ["shared_product_signal"],
                        "target_scope": target_scope,
                        "challenge_selected": True,
                    }
                ]
            },
        )
        first_reasoning = ActionRecord(
            action=InvestigationAction(
                action_id="RCA_ACTION_ROUND_1",
                kind=ActionKind.RUN_RCA_REASONING.value,
                agent=AgentKind.RCA_REASONING.value,
                reason="Generate the first candidates.",
                inputs={"lot_id": "LOT_01"},
                scope={"lot_id": "LOT_01"},
            ),
            status="completed",
            produced_evidence_ids=["EV_RCA_ROUND_1"],
            decision_summary="First RCA round completed.",
        )
        second_reasoning = ActionRecord(
            action=InvestigationAction(
                action_id="RCA_ACTION_ROUND_2",
                kind=ActionKind.RUN_RCA_REASONING.value,
                agent=AgentKind.RCA_REASONING.value,
                reason="Compare candidates after the first discriminator.",
                inputs={"lot_id": "LOT_01"},
                scope={"lot_id": "LOT_01", "causal_gap_id": "PRIOR_GAP"},
            ),
            status="completed",
            produced_evidence_ids=["EV_RCA_ROUND_1", "EV_RCA_ROUND_2"],
            decision_summary="Second RCA round completed.",
        )
        observation_action = InvestigationAction(
            action_id="ROUND_3_DISCRIMINATIVE_OBSERVATION",
            kind=ActionKind.VALIDATE_SHARED_DEFECT_PATTERN.value,
            agent=AgentKind.DEFECT_WAT.value,
            reason="Collect the final discriminative product observation.",
            inputs={"lot_id": "LOT_01"},
            scope={
                "lot_id": "LOT_01",
                "causal_gap_id": gap_id,
                **target_scope,
            },
        )
        observation = ActionRecord(
            action=observation_action,
            status="completed",
            produced_evidence_ids=["EV_NEW_DISCRIMINATOR"],
            decision_summary="A new Lane-scoped product observation was recorded.",
        )
        gain_link = QuestionEvidenceLink(
            question_id=mechanism.question_id,
            evidence_id="EV_NEW_DISCRIMINATOR",
            action_id=observation_action.action_id,
            relation=QuestionEvidenceRelation.SUPPORTS.value,
            matched_evidence_group="shared_product_signal",
            reason="The observation distinguishes the competing causal Lane.",
        )
        findings = [
            mes,
            finding(AgentKind.FDC.value),
            finding(AgentKind.DEFECT_WAT.value),
            rca_round_1,
            rca_round_2,
        ]
        setup_records = [
            ActionRecord(
                action=InvestigationAction(
                    action_id=f"SETUP_ACTION_{index}",
                    kind=ActionKind.INSPECT_DEFECT_PATTERN.value,
                    agent=AgentKind.DEFECT_WAT.value,
                    reason="Collect a bounded setup observation.",
                    inputs={"lot_id": "LOT_01"},
                    scope={"lot_id": "LOT_01", "setup_step": index},
                ),
                status="completed",
                produced_evidence_ids=[f"EV_SETUP_{index}"],
                decision_summary="A setup observation was recorded.",
            )
            for index in range(5)
        ]
        records = [
            *setup_records,
            first_reasoning,
            second_reasoning,
            observation,
        ]
        self.assertEqual(len(records), 8)
        planner = QwenNextActionPlanner(RecordingNextActionClient())

        third = planner.decide_with_review(
            goal=goal(),
            questions=[mechanism],
            findings=findings,
            action_records=records,
            tool_call_count=3,
            evidence_ids=[
                "EV_RCA_ROUND_1",
                "EV_RCA_ROUND_2",
                "EV_NEW_DISCRIMINATOR",
            ],
            question_evidence_links=[gain_link],
            authoritative_rca_finding_id=rca_round_2.finding_id,
        )

        self.assertEqual(third.decision.decision_type, DecisionType.ACT.value)
        self.assertEqual(
            third.decision.next_action.kind,
            ActionKind.RUN_RCA_REASONING.value,
        )
        self.assertEqual(third.decision.next_action.scope["causal_gap_id"], gap_id)

        third_record = ActionRecord(
            action=third.decision.next_action,
            status="completed",
            produced_evidence_ids=[
                "EV_RCA_ROUND_1",
                "EV_RCA_ROUND_2",
                "EV_NEW_DISCRIMINATOR",
            ],
            decision_summary="The third and final RCA round consumed the Evidence.",
        )
        exhausted = planner.decide_with_review(
            goal=goal(),
            questions=[mechanism],
            findings=findings,
            action_records=[*records, third_record],
            tool_call_count=4,
            evidence_ids=[
                "EV_RCA_ROUND_1",
                "EV_RCA_ROUND_2",
                "EV_NEW_DISCRIMINATOR",
            ],
            question_evidence_links=[gain_link],
            prior_decisions=[third.decision],
            authoritative_rca_finding_id=rca_round_2.finding_id,
        )

        self.assertEqual(exhausted.decision.decision_type, DecisionType.STOP.value)
        self.assertEqual(
            exhausted.decision.stop_reason,
            StopReason.BUDGET_EXHAUSTED.value,
        )

        consumed_round_2 = AgentFinding(
            finding_id=rca_round_2.finding_id,
            agent=rca_round_2.agent,
            summary=rca_round_2.summary,
            confidence=rca_round_2.confidence,
            evidence_ids=[*rca_round_2.evidence_ids, "EV_NEW_DISCRIMINATOR"],
            details=rca_round_2.details,
        )
        already_consumed = planner.decide_with_review(
            goal=goal(),
            questions=[mechanism],
            findings=[*findings[:-1], consumed_round_2],
            action_records=records,
            tool_call_count=3,
            evidence_ids=[
                "EV_RCA_ROUND_1",
                "EV_RCA_ROUND_2",
                "EV_NEW_DISCRIMINATOR",
            ],
            question_evidence_links=[gain_link],
            authoritative_rca_finding_id=consumed_round_2.finding_id,
        )

        self.assertEqual(
            already_consumed.decision.decision_type,
            DecisionType.STOP.value,
        )
        self.assertEqual(
            already_consumed.decision.stop_reason,
            StopReason.BUDGET_EXHAUSTED.value,
        )

    def test_required_missing_waits_for_executable_discrimination_gap(self) -> None:
        mechanism = questions()[1]
        gap_id = "candidate_0.hypothesis_discrimination.parameter_anomaly"
        target_scope = {
            "lane_id": "lane:1000:EQ_ALT:EQ_ALT_CH01:RCP_ALT",
            "operation": "1000",
            "equipment": "EQ_ALT",
            "chamber": "EQ_ALT_CH01",
            "recipe": "RCP_ALT",
            "discriminator_kind": "parameter_anomaly",
        }
        mes = finding(AgentKind.MES.value)
        mes.details["lane_candidates"] = [
            {
                **target_scope,
                "parameter_scope": ["pressure_cv"],
                "exposed_lot_ids": ["LOT_01"],
            },
            {
                "lane_id": "lane:2000:EQ_PRIMARY:EQ_PRIMARY_CH01:RCP_PRIMARY",
                "operation": "2000",
                "equipment": "EQ_PRIMARY",
                "chamber": "EQ_PRIMARY_CH01",
                "recipe": "RCP_PRIMARY",
                "parameter_scope": ["temperature_delta"],
                "exposed_lot_ids": ["LOT_01"],
            },
        ]
        rca = AgentFinding(
            finding_id="FINDING_REQUIRED_MISSING_WITH_DISCRIMINATOR",
            agent=AgentKind.RCA_REASONING.value,
            summary="A required source is unavailable, but an alternative is testable.",
            confidence=0.5,
            evidence_ids=["EV_RCA_TRACE", "EV_REQUIRED_MISSING"],
            details={
                "conclusion_status": "insufficient_evidence",
                "causal_evidence_gaps": [
                    {
                        "gap_id": gap_id,
                        "gap_type": "hypothesis_discrimination",
                        "discriminator_kind": "parameter_anomaly",
                        "candidate_index": 0,
                        "candidate_id": "C_PRIMARY",
                        "claim": "hypothesis_discrimination",
                        "status": "unresolved",
                        "reason": "Measure the alternative Lane parameter excursion.",
                        "question_kind": "process_mechanism",
                        "allowed_actions": [
                            ActionKind.INSPECT_FDC_SPC.value,
                            ActionKind.RUN_RCA_REASONING.value,
                        ],
                        "preferred_action": ActionKind.INSPECT_FDC_SPC.value,
                        "refresh_action": ActionKind.RUN_RCA_REASONING.value,
                        "required_evidence_groups": ["process_anomaly"],
                        "target_scope": target_scope,
                        "challenge_selected": True,
                    }
                ],
            },
        )
        first_reasoning = action_record(
            kind=ActionKind.RUN_RCA_REASONING.value,
            agent=AgentKind.RCA_REASONING.value,
            scope={"lot_id": "LOT_01"},
        )
        required_missing = Evidence(
            evidence_id="EV_REQUIRED_MISSING",
            source_type=EvidenceSourceType.ANALYTICS.value,
            source_id="FORMAL_CASE_CONTEXT",
            summary="A confirmation source is unavailable.",
            source_field="downstream_confirmation_metric",
            evidence_type=EvidenceType.DATA_MISSING.value,
            source_agent=AgentKind.PLANNER.value,
            source_tool="formal_case_context",
            observation="The confirmation source is unavailable.",
            entities=[EvidenceEntity(EntityType.LOT.value, "LOT_01")],
            confidence=1.0,
            metadata={"required_for_confirmation": True},
            evidence_schema_version=EVIDENCE_SCHEMA_VERSION,
        )
        client = RecordingNextActionClient()

        outcome = QwenNextActionPlanner(client).decide_with_review(
            goal=goal(),
            questions=[mechanism],
            findings=[
                mes,
                finding(AgentKind.FDC.value),
                finding(AgentKind.DEFECT_WAT.value),
                rca,
            ],
            action_records=[first_reasoning],
            tool_call_count=1,
            evidence=[required_missing],
            evidence_ids=["EV_RCA_TRACE", required_missing.evidence_id],
            question_evidence_links=[],
            authoritative_rca_finding_id=rca.finding_id,
        )

        self.assertEqual(client.requests, [])
        self.assertEqual(outcome.decision.decision_type, DecisionType.ACT.value)
        self.assertEqual(
            outcome.decision.next_action.kind,
            ActionKind.INSPECT_FDC_SPC.value,
        )
        self.assertEqual(outcome.decision.next_action.scope["causal_gap_id"], gap_id)
        self.assertEqual(
            outcome.decision.next_action.scope["lane_id"],
            target_scope["lane_id"],
        )

    def test_unexecutable_priority_zero_gap_does_not_mask_discriminator(self) -> None:
        mechanism = questions()[1]
        discriminator_gap_id = (
            "candidate_0.hypothesis_discrimination.product_outcome"
        )
        target_scope = {
            "lane_id": "lane:2000:EQ_ALT:EQ_ALT_CH02:RCP_ALT",
            "operation": "2000",
            "equipment": "EQ_ALT",
            "chamber": "EQ_ALT_CH02",
            "recipe": "RCP_ALT",
            "discriminator_kind": "product_outcome",
        }
        prior_scope = {
            "lane_id": "lane:1000:EQ_PRIMARY:EQ_PRIMARY_CH01:RCP_PRIMARY",
            "operation": "1000",
            "equipment": "EQ_PRIMARY",
            "chamber": "EQ_PRIMARY_CH01",
            "recipe": "RCP_PRIMARY",
            "discriminator_kind": "product_outcome",
        }
        mes = finding(AgentKind.MES.value)
        mes.details["lane_candidates"] = [
            {
                **prior_scope,
                "parameter_scope": ["pressure_cv"],
                "exposed_lot_ids": ["LOT_01"],
            },
            {
                **target_scope,
                "parameter_scope": ["temperature_delta"],
                "exposed_lot_ids": ["LOT_01"],
            },
        ]
        rca = AgentFinding(
            finding_id="FINDING_PRIORITY_FALLTHROUGH_RCA",
            agent=AgentKind.RCA_REASONING.value,
            summary="One unavailable outcome must not hide a testable alternative.",
            confidence=0.5,
            evidence_ids=["EV_RCA_TRACE"],
            details={
                "causal_evidence_gaps": [
                    {
                        "gap_id": "candidate_1.outcome.unavailable",
                        "gap_type": "data_missing",
                        "candidate_index": 1,
                        "claim": "outcome",
                        "status": "unavailable",
                        "priority": 0,
                        "reason": "The outcome source is unavailable.",
                        "question_kind": "process_mechanism",
                        "allowed_actions": [
                            ActionKind.VALIDATE_SHARED_DEFECT_PATTERN.value,
                            ActionKind.RUN_RCA_REASONING.value,
                        ],
                        "preferred_action": (
                            ActionKind.VALIDATE_SHARED_DEFECT_PATTERN.value
                        ),
                        "refresh_action": ActionKind.RUN_RCA_REASONING.value,
                        "target_scope": {},
                    },
                    {
                        "gap_id": discriminator_gap_id,
                        "gap_type": "hypothesis_discrimination",
                        "discriminator_kind": "product_outcome",
                        "candidate_index": 0,
                        "candidate_id": "C_ALT",
                        "claim": "hypothesis_discrimination",
                        "status": "unresolved",
                        "priority": 1,
                        "reason": "Test the alternative Lane product outcome.",
                        "question_kind": "process_mechanism",
                        "allowed_actions": [
                            ActionKind.VALIDATE_SHARED_DEFECT_PATTERN.value,
                            ActionKind.RUN_RCA_REASONING.value,
                        ],
                        "preferred_action": (
                            ActionKind.VALIDATE_SHARED_DEFECT_PATTERN.value
                        ),
                        "refresh_action": ActionKind.RUN_RCA_REASONING.value,
                        "required_evidence_groups": ["shared_product_signal"],
                        "target_scope": target_scope,
                        "challenge_selected": True,
                    },
                ]
            },
        )
        records: list[ActionRecord] = []
        links: list[QuestionEvidenceLink] = []
        prior_actions = (
            (
                ActionKind.VALIDATE_SHARED_DEFECT_PATTERN.value,
                AgentKind.DEFECT_WAT.value,
            ),
            (ActionKind.RUN_RCA_REASONING.value, AgentKind.RCA_REASONING.value),
        )
        for index, (action_kind, agent) in enumerate(prior_actions, start=1):
            action_id = f"PRIOR_LANE_GAP_ACTION_{index}"
            evidence_id = f"EV_PRIORITY_GAIN_{index}"
            records.append(
                ActionRecord(
                    action=InvestigationAction(
                        action_id=action_id,
                        kind=action_kind,
                        agent=agent,
                        reason="Investigate the same Gap on the prior Lane.",
                        inputs={"lot_id": "LOT_01"},
                        scope={
                            "lot_id": "LOT_01",
                            "causal_gap_id": discriminator_gap_id,
                            **prior_scope,
                        },
                    ),
                    status="completed",
                    produced_evidence_ids=[evidence_id],
                    decision_summary="The prior Lane recorded new Evidence.",
                )
            )
            links.append(
                QuestionEvidenceLink(
                    question_id=mechanism.question_id,
                    evidence_id=evidence_id,
                    action_id=action_id,
                    relation=QuestionEvidenceRelation.SUPPORTS.value,
                    matched_evidence_group="process_anomaly",
                    reason="The reasoning round recorded relevant Evidence.",
                )
            )
        client = RecordingNextActionClient()

        outcome = QwenNextActionPlanner(client).decide_with_review(
            goal=goal(),
            questions=[mechanism],
            findings=[
                mes,
                finding(AgentKind.FDC.value),
                finding(AgentKind.DEFECT_WAT.value),
                rca,
            ],
            action_records=records,
            tool_call_count=2,
            evidence_ids=["EV_RCA_TRACE", *[link.evidence_id for link in links]],
            question_evidence_links=links,
            authoritative_rca_finding_id=rca.finding_id,
        )

        self.assertEqual(client.requests, [])
        self.assertEqual(
            outcome.decision.decision_type,
            DecisionType.ACT.value,
            msg=outcome.decision.reason,
        )
        self.assertEqual(
            outcome.decision.next_action.kind,
            ActionKind.VALIDATE_SHARED_DEFECT_PATTERN.value,
        )
        self.assertEqual(
            outcome.decision.next_action.scope["causal_gap_id"],
            discriminator_gap_id,
        )
        self.assertEqual(
            outcome.decision.next_action.scope["lane_id"],
            target_scope["lane_id"],
        )

    def test_goal_satisfied_is_repaired_while_causal_gap_action_remains(self) -> None:
        mechanism, findings, records, links, first_gap_id = (
            self._causal_gap_runtime()
        )
        second_gap_id = "candidate_1.parameter.incomplete"
        authoritative = next(
            item for item in findings if item.finding_id == "FINDING_RCA_AUTHORITATIVE"
        )
        authoritative.details["causal_evidence_gaps"].append(
            {
                "gap_id": second_gap_id,
                "candidate_index": 1,
                "claim": "parameter",
                "status": "incomplete",
                "reason": "The competing candidate needs parameter Evidence.",
                "question_kind": "process_mechanism",
                "allowed_actions": [ActionKind.INSPECT_FDC_SPC.value],
                "evidence_ids": ["EV_RCA_TRACE"],
            }
        )

        def stop_then_repair(
            payload: dict[str, Any],
            request: LLMRequest,
        ) -> None:
            if request.payload["output_attempt"] == 1:
                payload.clear()
                payload.update(
                    {
                        "decision_id": "PREMATURE_STOP",
                        "goal_id": request.payload["goal"]["goal_id"],
                        "decision_type": DecisionType.STOP.value,
                        "reason": "All Questions appear closed.",
                        "goal_status": GoalStatus.SATISFIED.value,
                        "proposed_conclusion_level": ConclusionLevel.SUPPORTED.value,
                        "next_action": None,
                        "target_question_ids": [],
                        "new_questions": [],
                        "stop_reason": StopReason.GOAL_SATISFIED.value,
                        "question_updates": [],
                    }
                )
                return
            payload.clear()
            payload.update(
                model_act_payload(
                    request,
                    kind=ActionKind.INSPECT_FDC_SPC.value,
                    agent=AgentKind.FDC.value,
                )
            )

        client = RecordingNextActionClient(stop_then_repair)
        outcome = QwenNextActionPlanner(client).decide_with_review(
            goal=goal(),
            questions=[mechanism],
            findings=findings,
            action_records=records,
            tool_call_count=2,
            evidence_ids=[link.evidence_id for link in links],
            question_evidence_links=links,
            authoritative_rca_finding_id="FINDING_RCA_AUTHORITATIVE",
        )

        self.assertEqual(len(client.requests), 2)
        contract = client.requests[0].payload["goal_satisfied_stop_contract"]
        self.assertFalse(contract["python_terminal_transition_available"])
        self.assertEqual(
            contract["executable_causal_gap_ids"],
            [first_gap_id, second_gap_id],
        )
        feedback = client.requests[1].payload["previous_validation_feedback"]
        self.assertIn("cannot bypass executable causal Evidence Gaps", feedback["message"])
        self.assertEqual(
            outcome.decision.next_action.kind,
            ActionKind.INSPECT_FDC_SPC.value,
        )
        self.assertEqual(
            outcome.decision.next_action.scope["causal_gap_id"],
            second_gap_id,
        )

    def test_same_candidate_gap_action_is_single_use_and_stops_without_fallback(self) -> None:
        mechanism, findings, records, links, gap_id = self._causal_gap_runtime()
        records.append(
            ActionRecord(
                action=InvestigationAction(
                    action_id="GAP_KNOWLEDGE_1",
                    kind=ActionKind.VALIDATE_HISTORICAL_CASE.value,
                    agent=AgentKind.KNOWLEDGE.value,
                    reason="Validate mechanism Knowledge.",
                    inputs={"lot_id": "LOT_01"},
                    scope={"lot_id": "LOT_01", "causal_gap_id": gap_id},
                ),
                status="completed",
                produced_evidence_ids=["EV_KNOWLEDGE_GAIN"],
                decision_summary="Knowledge validation completed.",
            )
        )
        links.append(
            QuestionEvidenceLink(
                question_id=mechanism.question_id,
                evidence_id="EV_KNOWLEDGE_GAIN",
                action_id="GAP_KNOWLEDGE_1",
                relation=QuestionEvidenceRelation.SUPPORTS.value,
                matched_evidence_group="historical_context",
                reason="Approved Knowledge was evaluated.",
            )
        )
        client = RecordingNextActionClient()

        outcome = QwenNextActionPlanner(client).decide_with_review(
            goal=goal(),
            questions=[mechanism],
            findings=findings,
            action_records=records,
            tool_call_count=3,
            evidence_ids=[link.evidence_id for link in links],
            question_evidence_links=links,
            authoritative_rca_finding_id="FINDING_RCA_AUTHORITATIVE",
        )

        self.assertEqual(client.requests, [])
        self.assertEqual(outcome.decision.decision_type, DecisionType.STOP.value)
        self.assertEqual(outcome.decision.stop_reason, StopReason.NO_ALLOWED_ACTION.value)

    def test_only_gap_action_after_no_gain_stops_instead_of_raising(self) -> None:
        mechanism, findings, records, links, _ = self._causal_gap_runtime()
        prior_action = InvestigationAction(
            action_id="OLD_GAP_KNOWLEDGE",
            kind=ActionKind.VALIDATE_HISTORICAL_CASE.value,
            agent=AgentKind.KNOWLEDGE.value,
            reason="Validate a prior candidate gap.",
            inputs={"lot_id": "LOT_01"},
            scope={"lot_id": "LOT_01", "causal_gap_id": "candidate_1.mechanism"},
        )
        records.append(
            ActionRecord(
                action=prior_action,
                status="completed",
                produced_evidence_ids=["EV_CONTEXT_ONLY"],
                decision_summary="The search returned context only.",
            )
        )
        links.append(
            QuestionEvidenceLink(
                question_id=mechanism.question_id,
                evidence_id="EV_CONTEXT_ONLY",
                action_id=prior_action.action_id,
                relation=QuestionEvidenceRelation.CONTEXT.value,
                matched_evidence_group="historical_context",
                reason="The result did not support or contradict the candidate.",
            )
        )
        prior_decision = PlannerDecision(
            decision_id="OLD_GAP_DECISION",
            goal_id=goal().goal_id,
            decision_type=DecisionType.ACT.value,
            reason="Search the prior gap.",
            goal_status=GoalStatus.IN_PROGRESS.value,
            proposed_conclusion_level=ConclusionLevel.CANDIDATE.value,
            next_action=prior_action,
            target_question_ids=[mechanism.question_id],
        )
        client = RecordingNextActionClient()

        outcome = QwenNextActionPlanner(client).decide_with_review(
            goal=goal(),
            questions=[mechanism],
            findings=findings,
            action_records=records,
            tool_call_count=3,
            evidence_ids=[link.evidence_id for link in links],
            question_evidence_links=links,
            prior_decisions=[prior_decision],
            authoritative_rca_finding_id="FINDING_RCA_AUTHORITATIVE",
        )

        self.assertEqual(client.requests, [])
        self.assertEqual(outcome.decision.decision_type, DecisionType.STOP.value)
        self.assertIn("no new relevant Evidence", outcome.decision.reason)

    def test_two_context_only_actions_force_a_python_no_gain_stop(self) -> None:
        mechanism = questions()[1]
        records = [
            ActionRecord(
                action=InvestigationAction(
                    action_id=f"CONTEXT_{index}",
                    kind=ActionKind.INSPECT_FDC_SPC.value,
                    agent=AgentKind.FDC.value,
                    reason="Inspect another signal.",
                    inputs={"lot_id": "LOT_01"},
                    scope={"lot_id": "LOT_01", "attempt": index},
                ),
                status="completed",
                produced_evidence_ids=[f"EV_CONTEXT_{index}"],
                decision_summary="Only contextual Evidence was found.",
            )
            for index in (1, 2)
        ]
        links = [
            QuestionEvidenceLink(
                question_id=mechanism.question_id,
                evidence_id=f"EV_CONTEXT_{index}",
                action_id=f"CONTEXT_{index}",
                relation=QuestionEvidenceRelation.CONTEXT.value,
                matched_evidence_group="context",
                reason="This Evidence is contextual only.",
            )
            for index in (1, 2)
        ]
        client = RecordingNextActionClient()

        outcome = QwenNextActionPlanner(client).decide_with_review(
            goal=goal(),
            questions=[mechanism],
            findings=[finding(AgentKind.MES.value)],
            action_records=records,
            tool_call_count=2,
            evidence_ids=[link.evidence_id for link in links],
            question_evidence_links=links,
        )

        self.assertEqual(client.requests, [])
        self.assertEqual(outcome.decision.decision_type, DecisionType.STOP.value)
        self.assertIn("no new supporting", outcome.decision.reason)

    def test_fake_client_uses_a_registered_deterministic_baseline(self) -> None:
        client = RecordingNextActionClient()

        with capture_llm_usage() as usage:
            decision = QwenNextActionPlanner(client).decide(
                goal=goal(),
                questions=questions(),
                findings=[],
                action_records=[],
                tool_call_count=0,
            )

        self.assertEqual(decision.decision_type, DecisionType.ACT.value)
        self.assertEqual(
            decision.next_action.kind,
            ActionKind.INSPECT_DEFECT_PATTERN.value,
        )
        self.assertTrue(decision.next_action.scope)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(client.requests[0].prompt_name, "next_action_planner")
        self.assertEqual(
            {
                item["kind"]
                for item in client.requests[0].payload["allowed_actions"]
            },
            {
                ActionKind.INSPECT_DEFECT_PATTERN.value,
                ActionKind.FIND_SHARED_EXPOSURE.value,
            },
        )
        self.assertEqual(
            client.requests[0].payload["legal_target_question_ids_by_action"],
            {
                ActionKind.FIND_SHARED_EXPOSURE.value: [
                    "Q_MECHANISM",
                ],
                ActionKind.INSPECT_DEFECT_PATTERN.value: [
                    "Q_DEFECT",
                    "Q_MECHANISM",
                ],
            },
        )
        self.assertNotIn(
            ActionKind.ASSESS_IMPACT_SCOPE.value,
            LLM_REACT_EXECUTABLE_ACTION_KINDS,
        )
        self.assertNotIn(
            ActionKind.INSPECT_RECIPE_CHANGE.value,
            LLM_REACT_EXECUTABLE_ACTION_KINDS,
        )
        self.assertNotIn(
            ActionKind.CONCLUDE_INCONCLUSIVE.value,
            LLM_REACT_EXECUTABLE_ACTION_KINDS,
        )
        self.assertIn(
            ActionKind.VALIDATE_HISTORICAL_CASE.value,
            LLM_REACT_EXECUTABLE_ACTION_KINDS,
        )
        self.assertEqual(len(usage), 1)
        self.assertEqual(usage[0].provider, "fake")

    def test_legal_model_action_is_not_overridden_by_deterministic_policy(self) -> None:
        client = ModelSelectedMESClient()

        decision = QwenNextActionPlanner(client).decide(
            goal=goal(),
            questions=questions(),
            findings=[],
            action_records=[],
            tool_call_count=0,
        )

        self.assertEqual(
            client.requests[0].payload["deterministic_planner_decision"][
                "next_action"
            ]["kind"],
            ActionKind.INSPECT_DEFECT_PATTERN.value,
        )
        self.assertEqual(
            decision.next_action.kind,
            ActionKind.FIND_SHARED_EXPOSURE.value,
        )
        self.assertEqual(decision.decision_id, "MODEL_DECISION_1")
        self.assertEqual(len(client.requests), 1)

    def test_action_that_only_repeats_a_satisfied_group_is_retried_atomically(
        self,
    ) -> None:
        mechanism = questions()[1]
        spc_question = InvestigationQuestion(
            question_id="Q_SPC",
            goal_id=mechanism.goal_id,
            question="Which SPC signal is abnormal?",
            rationale="The process signal must be inspected.",
            question_kind="spc_signal",
            scope={"lot_id": "LOT_01", "module": "CU_CMP"},
        )
        prior_record = action_record(
            kind=ActionKind.INSPECT_FDC_SPC.value,
            agent=AgentKind.FDC.value,
            scope={
                "lot_id": "LOT_01",
                "module": "CU_CMP",
                "parameter": "THK",
            },
            action_id="PRIOR_FDC_PROCESS_SIGNAL",
        )
        prior_record = ActionRecord(
            action=prior_record.action,
            status=prior_record.status,
            produced_finding_ids=list(prior_record.produced_finding_ids),
            produced_evidence_ids=["EV_PROCESS_SIGNAL"],
            decision_summary=prior_record.decision_summary,
        )
        prior_decision = PlannerDecision(
            decision_id="PRIOR_FDC_DECISION",
            goal_id=mechanism.goal_id,
            decision_type=DecisionType.ACT.value,
            reason="The first FDC Action filled process_anomaly.",
            goal_status=GoalStatus.IN_PROGRESS.value,
            proposed_conclusion_level=ConclusionLevel.SIGNAL.value,
            next_action=prior_record.action,
            target_question_ids=[mechanism.question_id],
        )
        process_link = QuestionEvidenceLink(
            question_id=mechanism.question_id,
            evidence_id="EV_PROCESS_SIGNAL",
            action_id=prior_record.action.action_id,
            relation=QuestionEvidenceRelation.SUPPORTS.value,
            matched_evidence_group="process_anomaly",
            reason="The first FDC observation filled process_anomaly.",
        )

        def repeat_fdc_for_both_questions(
            payload: dict[str, Any],
            request: LLMRequest,
        ) -> None:
            payload.clear()
            payload.update(
                model_act_payload(
                    request,
                    kind=ActionKind.INSPECT_FDC_SPC.value,
                    agent=AgentKind.FDC.value,
                )
            )
            payload["next_action"]["scope"]["parameter"] = "PRESSURE"
            payload["target_question_ids"] = [
                spc_question.question_id,
                mechanism.question_id,
            ]

        client = RecordingNextActionClient(repeat_fdc_for_both_questions)
        with self.assertRaises(QwenNextActionPlannerError) as caught:
            QwenNextActionPlanner(client).decide(
                goal=goal(),
                questions=[mechanism, spc_question],
                findings=[finding(AgentKind.MES.value)],
                action_records=[prior_record],
                tool_call_count=1,
                evidence_ids=["EV_PROCESS_SIGNAL"],
                question_evidence_links=[process_link],
                prior_decisions=[prior_decision],
            )

        self.assertEqual(len(client.requests), 2)
        self.assertTrue(
            all(
                "no_expected_evidence_gain" in error
                and mechanism.question_id in error
                for error in caught.exception.validation_errors
            )
        )
        self.assertTrue(
            all(
                request.payload["previous_validation_error"] is None
                if index == 0
                else "no_expected_evidence_gain"
                in str(request.payload["previous_validation_error"])
                for index, request in enumerate(client.requests)
            )
        )
        retry_feedback = client.requests[1].payload[
            "previous_validation_feedback"
        ]
        self.assertEqual(
            retry_feedback["legal_target_question_ids_by_action"][
                ActionKind.INSPECT_FDC_SPC.value
            ],
            [spc_question.question_id],
        )
        self.assertNotIn(
            ActionKind.INSPECT_FDC_SPC.value,
            retry_feedback["question_action_capabilities"][
                mechanism.question_id
            ],
        )

    def test_product_defect_inspection_requires_mes_selected_lots(self) -> None:
        product_goal = InvestigationGoal(
            goal_id="GOAL_PRODUCT",
            intent=InvestigationIntent.ROOT_CAUSE.value,
            summary="Investigate the product-window yield loss.",
            known_facts={"product_id": "40N_SOC"},
            required_evidence=["defect_signature"],
        )
        product_questions = [
            InvestigationQuestion(
                question_id="Q_PRODUCT_DEFECT",
                goal_id=product_goal.goal_id,
                question="Which affected Lots share the defect signature?",
                rationale="MES must first select the bounded product-window Lots.",
                scope={"product_id": "40N_SOC"},
            )
        ]

        def choose_defect(payload: dict[str, Any], request: LLMRequest) -> None:
            payload.clear()
            payload.update(
                model_act_payload(
                    request,
                    kind=ActionKind.INSPECT_DEFECT_PATTERN.value,
                    agent=AgentKind.DEFECT_WAT.value,
                )
            )
            payload["next_action"]["inputs"] = {"product_id": "40N_SOC"}
            payload["next_action"]["scope"] = {"product_id": "40N_SOC"}
            payload["target_question_ids"] = ["Q_PRODUCT_DEFECT"]

        without_mes = RecordingNextActionClient(choose_defect)
        with self.assertRaises(QwenNextActionPlannerError):
            QwenNextActionPlanner(without_mes).decide(
                goal=product_goal,
                questions=product_questions,
                findings=[],
                action_records=[],
                tool_call_count=0,
            )
        self.assertEqual(len(without_mes.requests), 2)

        decision = QwenNextActionPlanner(
            RecordingNextActionClient(choose_defect)
        ).decide(
            goal=product_goal,
            questions=product_questions,
            findings=[finding(AgentKind.MES.value)],
            action_records=[],
            tool_call_count=1,
        )
        self.assertEqual(
            decision.next_action.kind,
            ActionKind.INSPECT_DEFECT_PATTERN.value,
        )

    def test_legal_model_stop_and_proposed_level_are_not_policy_overridden(self) -> None:
        def choose_stop(payload: dict[str, Any], request: LLMRequest) -> None:
            payload.clear()
            payload.update(
                {
                    "decision_id": "MODEL_STOP",
                    "goal_id": request.payload["goal"]["goal_id"],
                    "decision_type": DecisionType.STOP.value,
                    "reason": "The model considers the requested boundary complete.",
                    "goal_status": GoalStatus.SATISFIED.value,
                    "proposed_conclusion_level": ConclusionLevel.SUPPORTED.value,
                    "next_action": None,
                    "target_question_ids": [],
                    "new_questions": [],
                    "stop_reason": StopReason.GOAL_SATISFIED.value,
                    "question_updates": [],
                }
            )

        decision = QwenNextActionPlanner(
            RecordingNextActionClient(choose_stop)
        ).decide(
            goal=goal(),
            questions=questions(),
            findings=[],
            action_records=[],
            tool_call_count=0,
        )

        self.assertEqual(decision.decision_type, DecisionType.STOP.value)
        self.assertEqual(
            decision.proposed_conclusion_level,
            ConclusionLevel.SUPPORTED.value,
        )

    def test_reviewed_goal_satisfied_stop_cannot_hide_rejected_open_updates(
        self,
    ) -> None:
        def stop_with_open_update(
            payload: dict[str, Any],
            request: LLMRequest,
        ) -> None:
            payload.clear()
            payload.update(
                {
                    "decision_id": f"STOP_{request.payload['output_attempt']}",
                    "goal_id": request.payload["goal"]["goal_id"],
                    "decision_type": DecisionType.STOP.value,
                    "reason": "Incorrectly claim completion with an open update.",
                    "goal_status": GoalStatus.SATISFIED.value,
                    "proposed_conclusion_level": ConclusionLevel.SUPPORTED.value,
                    "next_action": None,
                    "target_question_ids": [],
                    "new_questions": [],
                    "stop_reason": StopReason.GOAL_SATISFIED.value,
                    "question_updates": [
                        {
                            "question_id": "Q_DEFECT",
                            "status": EvidenceGapStatus.OPEN.value,
                            "answer": "Partial progress is not terminal.",
                            "evidence_ids": ["EV_DEFECT"],
                            "unavailable_reason": None,
                        }
                    ],
                }
            )

        client = RecordingNextActionClient(stop_with_open_update)
        with self.assertRaises(QwenNextActionPlannerError) as captured:
            QwenNextActionPlanner(client).decide_with_review(
                goal=goal(),
                questions=questions(),
                findings=[],
                action_records=[],
                tool_call_count=0,
                evidence_ids=["EV_DEFECT"],
            )

        self.assertEqual(len(client.requests), 2)
        self.assertTrue(
            all(
                "goal_satisfied stop cannot leave open investigation questions"
                in error
                for error in captured.exception.validation_errors
            )
        )
        stop_contract = client.requests[0].payload[
            "goal_satisfied_stop_contract"
        ]
        self.assertEqual(
            stop_contract["currently_open_question_ids"],
            ["Q_DEFECT", "Q_MECHANISM"],
        )
        self.assertFalse(
            stop_contract["python_terminal_transition_available"]
        )
        self.assertEqual(stop_contract["python_terminal_question_ids"], [])
        retry_feedback = client.requests[1].payload[
            "previous_validation_feedback"
        ]
        self.assertEqual(
            retry_feedback["must_terminally_update_question_ids"],
            ["Q_DEFECT", "Q_MECHANISM"],
        )
        self.assertIn(
            "python_terminal_transition_available",
            retry_feedback["repair_instruction"],
        )

    def test_reviewed_stop_accepts_supported_updates_for_every_open_question(
        self,
    ) -> None:
        def close_questions_and_stop(
            payload: dict[str, Any],
            request: LLMRequest,
        ) -> None:
            payload.clear()
            payload.update(
                {
                    "decision_id": "STOP_SUPPORTED",
                    "goal_id": request.payload["goal"]["goal_id"],
                    "decision_type": DecisionType.STOP.value,
                    "reason": "Both requested Questions are evidence-backed.",
                    "goal_status": GoalStatus.SATISFIED.value,
                    "proposed_conclusion_level": ConclusionLevel.SUPPORTED.value,
                    "next_action": None,
                    "target_question_ids": [],
                    "new_questions": [],
                    "stop_reason": StopReason.GOAL_SATISFIED.value,
                    "question_updates": [
                        {
                            "question_id": "Q_DEFECT",
                            "status": EvidenceGapStatus.CLOSED.value,
                            "answer": "The source Lot has an edge scratch.",
                            "evidence_ids": ["EV_DEFECT"],
                            "unavailable_reason": None,
                        },
                        {
                            "question_id": "Q_MECHANISM",
                            "status": EvidenceGapStatus.CLOSED.value,
                            "answer": "The process Evidence supports the mechanism.",
                            "evidence_ids": ["EV_MECHANISM"],
                            "unavailable_reason": None,
                        },
                    ],
                }
            )

        client = RecordingNextActionClient(close_questions_and_stop)
        outcome = QwenNextActionPlanner(client).decide_with_review(
            goal=goal(),
            questions=questions(),
            findings=[],
            action_records=[],
            tool_call_count=0,
            evidence_ids=["EV_DEFECT", "EV_MECHANISM"],
        )

        self.assertEqual(len(client.requests), 1)
        self.assertEqual(outcome.decision.decision_type, DecisionType.STOP.value)
        self.assertEqual(len(outcome.decision.question_updates), 2)
        self.assertTrue(
            all(
                review.disposition == QuestionUpdateDisposition.ACCEPTED.value
                for review in outcome.question_update_reviews
            )
        )

    def test_qwen_goal_satisfied_stop_uses_complete_python_terminal_transition(
        self,
    ) -> None:
        current_questions = questions()
        links = [
            QuestionEvidenceLink(
                question_id="Q_DEFECT",
                evidence_id="EV_DEFECT",
                action_id="ACT_DEFECT",
                relation=QuestionEvidenceRelation.SUPPORTS.value,
                matched_evidence_group="product_signal",
                reason="The defect observation fills the product-signal group.",
            ),
            QuestionEvidenceLink(
                question_id="Q_MECHANISM",
                evidence_id="EV_PROCESS",
                action_id="ACT_PROCESS",
                relation=QuestionEvidenceRelation.SUPPORTS.value,
                matched_evidence_group="process_anomaly",
                reason="The FDC observation fills the process-anomaly group.",
            ),
            QuestionEvidenceLink(
                question_id="Q_MECHANISM",
                evidence_id="EV_DEFECT",
                action_id="ACT_DEFECT",
                relation=QuestionEvidenceRelation.SUPPORTS.value,
                matched_evidence_group="product_signal",
                reason="The defect observation fills the product-signal group.",
            ),
            QuestionEvidenceLink(
                question_id="Q_MECHANISM",
                evidence_id="EV_EXPOSURE",
                action_id="ACT_EXPOSURE",
                relation=QuestionEvidenceRelation.SUPPORTS.value,
                matched_evidence_group="shared_exposure",
                reason="The MES observation fills the shared-exposure group.",
            ),
            QuestionEvidenceLink(
                question_id="Q_MECHANISM",
                evidence_id="EV_SHARED_DEFECT",
                action_id="ACT_SHARED_DEFECT",
                relation=QuestionEvidenceRelation.SUPPORTS.value,
                matched_evidence_group="shared_product_signal",
                reason="The cross-Lot observation fills the shared-product group.",
            ),
        ]

        def stop_without_updates(
            payload: dict[str, Any],
            request: LLMRequest,
        ) -> None:
            payload.clear()
            payload.update(
                {
                    "decision_id": "QWEN_STOP",
                    "goal_id": request.payload["goal"]["goal_id"],
                    "decision_type": DecisionType.STOP.value,
                    "reason": "The investigation goal is satisfied.",
                    "goal_status": GoalStatus.SATISFIED.value,
                    "proposed_conclusion_level": ConclusionLevel.SUPPORTED.value,
                    "next_action": None,
                    "target_question_ids": [],
                    "new_questions": [],
                    "stop_reason": StopReason.GOAL_SATISFIED.value,
                    "question_updates": [],
                }
            )

        client = RecordingNextActionClient(stop_without_updates)
        outcome = QwenNextActionPlanner(client).decide_with_review(
            goal=goal(),
            questions=current_questions,
            findings=[
                finding(AgentKind.DEFECT_WAT.value, evidence_id="EV_DEFECT"),
                finding(AgentKind.FDC.value, evidence_id="EV_PROCESS"),
                finding(AgentKind.MES.value, evidence_id="EV_EXPOSURE"),
            ],
            action_records=[],
            tool_call_count=3,
            evidence_ids=[
                "EV_DEFECT",
                "EV_PROCESS",
                "EV_EXPOSURE",
                "EV_SHARED_DEFECT",
            ],
            question_evidence_links=links,
        )

        self.assertEqual(len(client.requests), 1)
        self.assertEqual(outcome.decision.decision_id, "QWEN_STOP")
        self.assertEqual(
            {update.question_id for update in outcome.decision.question_updates},
            {"Q_DEFECT", "Q_MECHANISM"},
        )
        self.assertEqual(outcome.decision_proposed_by, "qwen")
        self.assertEqual(
            outcome.question_updates_source,
            "python_evidence_gate",
        )
        self.assertIn(
            "Python Evidence Gate committed the terminal Question transitions",
            outcome.decision.reason,
        )
        self.assertTrue(
            all(
                review.disposition == QuestionUpdateDisposition.ACCEPTED.value
                for review in outcome.question_update_reviews
            )
        )

    def test_modified_input_echo_retry_also_repairs_goal_satisfied_stop(self) -> None:
        defect_question = questions()[0]
        defect_link = QuestionEvidenceLink(
            question_id=defect_question.question_id,
            evidence_id="EV_DEFECT",
            action_id="ACT_DEFECT",
            relation=QuestionEvidenceRelation.SUPPORTS.value,
            matched_evidence_group="product_signal",
            reason="The defect observation fills the product-signal group.",
        )

        def echo_then_repair(
            payload: dict[str, Any],
            request: LLMRequest,
        ) -> None:
            payload.clear()
            payload.update(
                {
                    "decision_id": f"STOP_{request.payload['output_attempt']}",
                    "goal_id": request.payload["goal"]["goal_id"],
                    "decision_type": DecisionType.STOP.value,
                    "reason": "The requested defect signature is evidence-backed.",
                    "goal_status": GoalStatus.SATISFIED.value,
                    "proposed_conclusion_level": ConclusionLevel.SIGNAL.value,
                    "next_action": None,
                    "target_question_ids": [],
                    "new_questions": [],
                    "stop_reason": StopReason.GOAL_SATISFIED.value,
                    "question_updates": [],
                }
            )
            if request.payload["output_attempt"] == 1:
                modified_contract = copy.deepcopy(
                    request.payload["goal_satisfied_stop_contract"]
                )
                modified_contract["unexpected_model_field"] = True
                payload["goal_satisfied_stop_contract"] = modified_contract

        client = RecordingNextActionClient(echo_then_repair)
        outcome = QwenNextActionPlanner(client).decide_with_review(
            goal=goal(),
            questions=[defect_question],
            findings=[
                finding(AgentKind.DEFECT_WAT.value, evidence_id="EV_DEFECT")
            ],
            action_records=[],
            tool_call_count=1,
            evidence_ids=["EV_DEFECT"],
            question_evidence_links=[defect_link],
        )

        self.assertEqual(len(client.requests), 2)
        feedback = client.requests[1].payload["previous_validation_feedback"]
        self.assertIn(
            "goal_satisfied_stop_contract",
            feedback["input_only_fields_never_copy_to_output"],
        )
        self.assertEqual(
            feedback["python_terminal_question_ids"],
            client.requests[1].payload["goal_satisfied_stop_contract"][
                "python_terminal_question_ids"
            ],
        )
        self.assertEqual(outcome.decision.decision_type, DecisionType.STOP.value)
        self.assertEqual(len(outcome.decision.question_updates), 1)

    def test_exact_validator_ready_echo_becomes_python_owned_question_updates(
        self,
    ) -> None:
        defect_question = questions()[0]
        defect_link = QuestionEvidenceLink(
            question_id=defect_question.question_id,
            evidence_id="EV_DEFECT",
            action_id="ACT_DEFECT",
            relation=QuestionEvidenceRelation.SUPPORTS.value,
            matched_evidence_group="product_signal",
            reason="The defect observation fills the product-signal group.",
        )

        def echo_reference_at_top_level(
            payload: dict[str, Any],
            request: LLMRequest,
        ) -> None:
            terminal_question_ids = copy.deepcopy(
                request.payload["goal_satisfied_stop_contract"][
                    "python_terminal_question_ids"
                ]
            )
            payload.clear()
            payload.update(
                {
                    "decision_id": "STOP_WITH_REFERENCE_ECHO",
                    "goal_id": request.payload["goal"]["goal_id"],
                    "decision_type": DecisionType.STOP.value,
                    "reason": "Use the evidence-backed Python stop transition.",
                    "goal_status": GoalStatus.SATISFIED.value,
                    "proposed_conclusion_level": ConclusionLevel.SIGNAL.value,
                    "next_action": None,
                    "target_question_ids": [],
                    "new_questions": [],
                    "stop_reason": StopReason.GOAL_SATISFIED.value,
                    "question_updates": [],
                    "python_terminal_question_ids": terminal_question_ids,
                    "python_terminal_transition_available": True,
                }
            )

        client = RecordingNextActionClient(echo_reference_at_top_level)
        outcome = QwenNextActionPlanner(client).decide_with_review(
            goal=goal(),
            questions=[defect_question],
            findings=[
                finding(AgentKind.DEFECT_WAT.value, evidence_id="EV_DEFECT")
            ],
            action_records=[],
            tool_call_count=1,
            evidence_ids=["EV_DEFECT"],
            question_evidence_links=[defect_link],
        )

        self.assertEqual(len(client.requests), 1)
        self.assertEqual(outcome.decision.decision_type, DecisionType.STOP.value)
        self.assertEqual(len(outcome.decision.question_updates), 1)
        self.assertEqual(
            outcome.decision.question_updates[0].question_id,
            defect_question.question_id,
        )

    def test_modified_validator_ready_echo_cannot_change_python_transition(
        self,
    ) -> None:
        def echo_modified_reference(
            payload: dict[str, Any],
            request: LLMRequest,
        ) -> None:
            payload["validator_ready_reference_question_updates"] = [
                {
                    "question_id": "Q_INVENTED",
                    "status": EvidenceGapStatus.CLOSED.value,
                    "answer": "Invented transition.",
                    "evidence_ids": ["EV_INVENTED"],
                    "unavailable_reason": None,
                }
            ]

        client = RecordingNextActionClient(echo_modified_reference)
        with self.assertRaises(QwenNextActionPlannerError) as captured:
            QwenNextActionPlanner(client).decide(
                goal=goal(),
                questions=questions(),
                findings=[],
                action_records=[],
                tool_call_count=0,
            )

        self.assertEqual(len(client.requests), 2)
        self.assertTrue(
            all(
                "validator_ready_reference_question_updates" in error
                and "unknown fields" in error
                for error in captured.exception.validation_errors
            )
        )

    def test_invalid_output_is_retried_once_with_validation_feedback(self) -> None:
        client = InvalidThenValidClient()

        decision = QwenNextActionPlanner(client).decide(
            goal=goal(),
            questions=questions(),
            findings=[],
            action_records=[],
            tool_call_count=0,
        )

        self.assertEqual(
            decision.next_action.kind,
            ActionKind.INSPECT_DEFECT_PATTERN.value,
        )
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(client.requests[1].payload["output_attempt"], 2)
        self.assertIn(
            "prerequisite",
            str(client.requests[1].payload["previous_validation_error"]),
        )
        feedback = client.requests[1].payload["previous_validation_feedback"]
        self.assertEqual(feedback["category"], "core_decision_validation")
        self.assertTrue(feedback["must_repair_before_resubmission"])
        self.assertIn("prerequisite", feedback["message"])

    def test_exact_python_owned_context_echo_is_removed_before_strict_parse(
        self,
    ) -> None:
        def echo_context(
            payload: dict[str, Any],
            request: LLMRequest,
        ) -> None:
            payload["question_action_capabilities"] = copy.deepcopy(
                request.payload["question_action_capabilities"]
            )

        client = RecordingNextActionClient(echo_context)
        decision = QwenNextActionPlanner(client).decide(
            goal=goal(),
            questions=questions(),
            findings=[],
            action_records=[],
            tool_call_count=0,
        )

        self.assertEqual(len(client.requests), 1)
        self.assertEqual(decision.decision_type, DecisionType.ACT.value)

    def test_modified_python_owned_context_echo_still_fails_closed(self) -> None:
        def modify_context(
            payload: dict[str, Any],
            request: LLMRequest,
        ) -> None:
            payload["question_action_capabilities"] = {"Q_INVENTED": []}

        client = RecordingNextActionClient(modify_context)
        with self.assertRaises(QwenNextActionPlannerError) as captured:
            QwenNextActionPlanner(client).decide(
                goal=goal(),
                questions=questions(),
                findings=[],
                action_records=[],
                tool_call_count=0,
            )

        self.assertEqual(len(client.requests), 2)
        self.assertTrue(
            all(
                "question_action_capabilities" in error
                and "unknown fields" in error
                for error in captured.exception.validation_errors
            )
        )

    def test_one_transient_call_failure_is_retried_without_advancing_output_attempt(
        self,
    ) -> None:
        client = TransientCallFailureClient()

        decision = QwenNextActionPlanner(client).decide(
            goal=goal(),
            questions=questions(),
            findings=[],
            action_records=[],
            tool_call_count=0,
        )

        self.assertEqual(
            decision.next_action.kind,
            ActionKind.INSPECT_DEFECT_PATTERN.value,
        )
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(
            [request.payload["output_attempt"] for request in client.requests],
            [1, 1],
        )

    def test_two_call_failures_exhaust_retry_with_sanitized_diagnostics(self) -> None:
        client = PersistentCallFailureClient()

        with self.assertRaises(LLMCallError) as captured:
            QwenNextActionPlanner(client).decide(
                goal=goal(),
                questions=questions(),
                findings=[],
                action_records=[],
                tool_call_count=0,
            )

        error = captured.exception
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(error.call_attempt_count, 2)
        self.assertEqual(error.failure_category, "provider_http_error")
        self.assertEqual(error.status_code, 429)
        self.assertEqual(error.provider_code, "Throttling")
        self.assertEqual(error.request_id, "req-429")
        self.assertNotIn("planner-secret", error.provider_message or "")

    def test_review_path_still_falls_back_for_an_invalid_core_action(self) -> None:
        def wrong_agent(payload: dict[str, Any], request: LLMRequest) -> None:
            payload.clear()
            payload.update(
                model_act_payload(
                    request,
                    kind=ActionKind.FIND_SHARED_EXPOSURE.value,
                    agent=AgentKind.FDC.value,
                )
            )

        client = RecordingNextActionClient(wrong_agent)
        with self.assertRaises(QwenNextActionPlannerError) as captured:
            QwenNextActionPlanner(client).decide_with_review(
                goal=goal(),
                questions=questions(),
                findings=[],
                action_records=[],
                tool_call_count=0,
            )

        self.assertEqual(len(client.requests), 2)
        self.assertTrue(
            all(
                "must be executed by Agent mes" in error
                for error in captured.exception.validation_errors
            )
        )
        self.assertEqual(captured.exception.core_validation_error_count, 2)
        self.assertEqual(captured.exception.output_parse_error_count, 0)

    def test_strict_compatibility_path_still_retries_close_and_target(self) -> None:
        def close_and_target(
            payload: dict[str, Any],
            request: LLMRequest,
        ) -> None:
            payload.clear()
            payload.update(model_act_payload(request))
            payload["question_updates"] = [
                {
                    "question_id": "Q_MECHANISM",
                    "status": EvidenceGapStatus.UNAVAILABLE.value,
                    "answer": None,
                    "evidence_ids": [],
                    "unavailable_reason": "No registered source can answer it.",
                }
            ]

        client = RecordingNextActionClient(close_and_target)
        with self.assertRaises(QwenNextActionPlannerError) as captured:
            QwenNextActionPlanner(client).decide(
                goal=goal(),
                questions=questions(),
                findings=[],
                action_records=[],
                tool_call_count=0,
            )

        self.assertEqual(len(client.requests), 2)
        retry_error = str(
            client.requests[1].payload["previous_validation_error"]
        )
        self.assertIn(
            "target_question_ids and question_updates overlap for ['Q_MECHANISM']",
            retry_error,
        )
        self.assertIn(
            "keep it open and omit its QuestionUpdate",
            retry_error,
        )
        self.assertEqual(len(captured.exception.validation_errors), 2)

    def test_review_path_rejects_overlap_without_retrying_or_changing_action(
        self,
    ) -> None:
        def close_and_target(
            payload: dict[str, Any],
            request: LLMRequest,
        ) -> None:
            payload.clear()
            payload.update(
                model_act_payload(
                    request,
                    kind=ActionKind.INSPECT_DEFECT_PATTERN.value,
                    agent=AgentKind.DEFECT_WAT.value,
                )
            )
            payload["target_question_ids"] = ["Q_DEFECT"]
            payload["question_updates"] = [
                {
                    "question_id": "Q_DEFECT",
                    "status": EvidenceGapStatus.CLOSED.value,
                    "answer": "The selected Lot has an edge-dominant scratch.",
                    "evidence_ids": ["EV_DEFECT"],
                    "unavailable_reason": None,
                }
            ]

        client = RecordingNextActionClient(close_and_target)
        outcome = QwenNextActionPlanner(client).decide_with_review(
            goal=goal(),
            questions=questions(),
            findings=[
                finding(AgentKind.DEFECT_WAT.value, evidence_id="EV_DEFECT")
            ],
            action_records=[],
            tool_call_count=1,
        )

        self.assertEqual(len(client.requests), 1)
        self.assertEqual(outcome.decision.target_question_ids, ["Q_DEFECT"])
        self.assertEqual(outcome.decision.question_updates, [])
        self.assertEqual(
            outcome.question_update_reviews[0].disposition,
            QuestionUpdateDisposition.REJECTED.value,
        )
        self.assertEqual(
            outcome.question_update_reviews[0].reason_code,
            QuestionUpdateReasonCode.TARGET_OVERLAP.value,
        )

    def test_review_path_rejects_open_status_without_retrying_core_action(
        self,
    ) -> None:
        def report_partial_progress(
            payload: dict[str, Any],
            request: LLMRequest,
        ) -> None:
            payload.clear()
            payload.update(model_act_payload(request))
            payload["question_updates"] = [
                {
                    "question_id": "Q_DEFECT",
                    "status": EvidenceGapStatus.OPEN.value,
                    "answer": "The first observation provides partial progress.",
                    "evidence_ids": ["EV_DEFECT"],
                    "unavailable_reason": None,
                }
            ]

        client = RecordingNextActionClient(report_partial_progress)
        outcome = QwenNextActionPlanner(client).decide_with_review(
            goal=goal(),
            questions=questions(),
            findings=[
                finding(AgentKind.DEFECT_WAT.value, evidence_id="EV_DEFECT")
            ],
            action_records=[],
            tool_call_count=1,
        )

        self.assertEqual(len(client.requests), 1)
        self.assertEqual(
            outcome.decision.next_action.kind,
            ActionKind.FIND_SHARED_EXPOSURE.value,
        )
        self.assertEqual(outcome.decision.question_updates, [])
        self.assertEqual(
            outcome.question_update_reviews[0].reason_code,
            QuestionUpdateReasonCode.NON_TERMINAL_STATUS.value,
        )

    def test_valid_question_update_requires_existing_evidence(self) -> None:
        def close_defect_question(
            payload: dict[str, Any],
            request: LLMRequest,
        ) -> None:
            payload.clear()
            payload.update(model_act_payload(request))
            payload["question_updates"] = [
                {
                    "question_id": "Q_DEFECT",
                    "status": EvidenceGapStatus.CLOSED.value,
                    "answer": "The selected Lot has a scratch signature.",
                    "evidence_ids": ["EV_DEFECT"],
                    "unavailable_reason": None,
                }
            ]

        decision = QwenNextActionPlanner(
            RecordingNextActionClient(close_defect_question)
        ).decide(
            goal=goal(),
            questions=questions(),
            findings=[
                finding(AgentKind.DEFECT_WAT.value, evidence_id="EV_DEFECT")
            ],
            action_records=[],
            tool_call_count=1,
        )

        self.assertEqual(len(decision.question_updates), 1)
        self.assertEqual(
            decision.question_updates[0].status,
            EvidenceGapStatus.CLOSED.value,
        )
        self.assertEqual(
            PlannerDecision.from_dict(decision.to_dict()),
            decision,
        )

    def test_open_question_update_is_repaired_after_indexed_validation_feedback(
        self,
    ) -> None:
        def repair_open_update(
            payload: dict[str, Any],
            request: LLMRequest,
        ) -> None:
            payload.clear()
            payload.update(model_act_payload(request))
            if request.payload["output_attempt"] == 1:
                payload["question_updates"] = [
                    {
                        "question_id": "Q_DEFECT",
                        "status": EvidenceGapStatus.OPEN.value,
                        "answer": "The scratch signature is partially characterized.",
                        "evidence_ids": ["EV_DEFECT"],
                        "unavailable_reason": None,
                    }
                ]
                return
            payload["question_updates"] = [
                {
                    "question_id": "Q_DEFECT",
                    "status": EvidenceGapStatus.CLOSED.value,
                    "answer": "The selected Lot has a scratch signature.",
                    "evidence_ids": ["EV_DEFECT"],
                    "unavailable_reason": None,
                }
            ]

        client = RecordingNextActionClient(repair_open_update)
        decision = QwenNextActionPlanner(client).decide(
            goal=goal(),
            questions=questions(),
            findings=[
                finding(AgentKind.DEFECT_WAT.value, evidence_id="EV_DEFECT")
            ],
            action_records=[],
            tool_call_count=1,
        )

        self.assertEqual(len(client.requests), 2)
        self.assertIn(
            "question_updates[0].status must be closed or unavailable",
            str(client.requests[1].payload["previous_validation_error"]),
        )
        self.assertEqual(
            decision.question_updates[0].status,
            EvidenceGapStatus.CLOSED.value,
        )

    def test_qwen_cannot_send_a_legacy_full_question_update(self) -> None:
        def copy_full_question(
            payload: dict[str, Any],
            request: LLMRequest,
        ) -> None:
            payload.clear()
            payload.update(model_act_payload(request))
            payload["question_updates"] = [
                {
                    **request.payload["questions"][0],
                    "status": EvidenceGapStatus.CLOSED.value,
                    "answer": "The scratch signature is characterized.",
                    "evidence_ids": ["EV_DEFECT"],
                    "unavailable_reason": None,
                }
            ]

        with self.assertRaises(QwenNextActionPlannerError) as captured:
            QwenNextActionPlanner(
                RecordingNextActionClient(copy_full_question)
            ).decide(
                goal=goal(),
                questions=questions(),
                findings=[
                    finding(AgentKind.DEFECT_WAT.value, evidence_id="EV_DEFECT")
                ],
                action_records=[],
                tool_call_count=1,
            )

        self.assertTrue(
            all(
                "question_updates[0]" in error and "unknown fields" in error
                for error in captured.exception.validation_errors
            )
        )

    def test_invalid_safety_boundaries_fail_twice_with_typed_fallback(self) -> None:
        cases: list[
            tuple[
                str,
                Mutation,
                list[AgentFinding],
                list[ActionRecord],
                int,
            ]
        ] = []

        def unsupported(payload: dict[str, Any], request: LLMRequest) -> None:
            payload.clear()
            payload.update(
                model_act_payload(
                    request,
                    kind=ActionKind.ASSESS_IMPACT_SCOPE.value,
                    agent=AgentKind.MES.value,
                )
            )

        cases.append(("allowlist", unsupported, [], [], 0))

        def wrong_agent(payload: dict[str, Any], request: LLMRequest) -> None:
            payload.clear()
            payload.update(
                model_act_payload(
                    request,
                    kind=ActionKind.INSPECT_DEFECT_PATTERN.value,
                    agent=AgentKind.MES.value,
                )
            )

        cases.append(("kind_agent", wrong_agent, [], [], 0))

        def missing_prerequisite(
            payload: dict[str, Any],
            request: LLMRequest,
        ) -> None:
            payload.clear()
            payload.update(
                model_act_payload(
                    request,
                    kind=ActionKind.INSPECT_FDC_SPC.value,
                    agent=AgentKind.FDC.value,
                )
            )

        cases.append(("precondition", missing_prerequisite, [], [], 0))

        def empty_scope(payload: dict[str, Any], request: LLMRequest) -> None:
            payload.clear()
            payload.update(model_act_payload(request))
            payload["next_action"]["scope"] = {}

        cases.append(("scope", empty_scope, [], [], 0))

        def changed_lot(payload: dict[str, Any], request: LLMRequest) -> None:
            payload.clear()
            payload.update(model_act_payload(request))
            payload["next_action"]["scope"]["lot_id"] = "LOT_99"

        cases.append(("lot_boundary", changed_lot, [], [], 0))

        def unknown_evidence(payload: dict[str, Any], request: LLMRequest) -> None:
            payload.clear()
            payload.update(model_act_payload(request))
            payload["next_action"]["required_evidence_ids"] = ["EV_UNKNOWN"]

        cases.append(("evidence", unknown_evidence, [], [], 0))

        def multiple_attempts(payload: dict[str, Any], request: LLMRequest) -> None:
            payload.clear()
            payload.update(model_act_payload(request))
            payload["next_action"]["max_attempts"] = 2

        cases.append(("max_attempts", multiple_attempts, [], [], 0))

        def unknown_question(payload: dict[str, Any], request: LLMRequest) -> None:
            payload.clear()
            payload.update(model_act_payload(request))
            payload["target_question_ids"] = ["Q_UNREQUESTED"]

        cases.append(("question", unknown_question, [], [], 0))

        def unknown_question_update(
            payload: dict[str, Any],
            request: LLMRequest,
        ) -> None:
            payload.clear()
            payload.update(model_act_payload(request))
            payload["question_updates"] = [
                {
                    "question_id": "Q_UNKNOWN",
                    "status": EvidenceGapStatus.UNAVAILABLE.value,
                    "answer": None,
                    "evidence_ids": [],
                    "unavailable_reason": "No registered source can answer it.",
                }
            ]

        cases.append(("question_update", unknown_question_update, [], [], 0))

        def unsupported_question_answer(
            payload: dict[str, Any],
            request: LLMRequest,
        ) -> None:
            payload.clear()
            payload.update(model_act_payload(request))
            payload["question_updates"] = [
                {
                    "question_id": "Q_DEFECT",
                    "status": EvidenceGapStatus.CLOSED.value,
                    "answer": "An unsupported answer.",
                    "evidence_ids": ["EV_NOT_OBSERVED"],
                    "unavailable_reason": None,
                }
            ]

        cases.append(
            ("question_evidence", unsupported_question_answer, [], [], 0)
        )

        prior_mes = action_record(
            kind=ActionKind.FIND_SHARED_EXPOSURE.value,
            agent=AgentKind.MES.value,
            scope={"lot_id": "LOT_01", "module": "DIFFUSION"},
        )

        def second_mes(payload: dict[str, Any], request: LLMRequest) -> None:
            payload.clear()
            payload.update(model_act_payload(request))

        cases.append(
            (
                "single_use_mes",
                second_mes,
                [],
                [prior_mes],
                1,
            )
        )

        prior_same_scope = action_record(
            kind=ActionKind.INSPECT_DEFECT_PATTERN.value,
            agent=AgentKind.DEFECT_WAT.value,
            scope={"lot_id": "LOT_01", "module": "CU_CMP"},
        )

        def duplicate_scope(payload: dict[str, Any], request: LLMRequest) -> None:
            payload.clear()
            payload.update(
                model_act_payload(
                    request,
                    kind=ActionKind.INSPECT_DEFECT_PATTERN.value,
                    agent=AgentKind.DEFECT_WAT.value,
                )
            )

        cases.append(("dedup", duplicate_scope, [], [prior_same_scope], 1))

        def premature_rca(payload: dict[str, Any], request: LLMRequest) -> None:
            payload.clear()
            payload.update(
                model_act_payload(
                    request,
                    kind=ActionKind.RUN_RCA_REASONING.value,
                    agent=AgentKind.RCA_REASONING.value,
                )
            )

        cases.append(
            (
                "shared_pattern_precondition",
                premature_rca,
                [
                    finding(AgentKind.MES.value),
                    finding(AgentKind.FDC.value),
                    finding(AgentKind.DEFECT_WAT.value),
                ],
                [],
                3,
            )
        )

        for name, mutation, current_findings, records, tool_calls in cases:
            with self.subTest(boundary=name):
                client = RecordingNextActionClient(mutation)
                with self.assertRaises(QwenNextActionPlannerError) as context:
                    QwenNextActionPlanner(client).decide(
                        goal=goal(),
                        questions=questions(),
                        findings=current_findings,
                        action_records=records,
                        tool_call_count=tool_calls,
                    )
                error = context.exception
                self.assertEqual(error.attempts, 2)
                self.assertEqual(error.fallback_mode, "controlled_react")
                self.assertEqual(error.goal_id, "GOAL_LOT_01")
                self.assertEqual(error.completed_steps, len(records))
                self.assertEqual(len(error.validation_errors), 2)
                self.assertEqual(len(client.requests), 2)

    def test_runtime_budget_forces_python_stop_without_an_llm_call(self) -> None:
        eight_records = [
            action_record(
                kind=ActionKind.INSPECT_DEFECT_PATTERN.value,
                agent=AgentKind.DEFECT_WAT.value,
                scope={"lot_id": "LOT_01", "step": index},
                action_id=f"PRIOR_{index}",
            )
            for index in range(8)
        ]

        for boundary, records, tool_calls in (
            ("max_steps", eight_records, 8),
            ("max_tool_calls", [], goal().max_tool_calls),
        ):
            with self.subTest(boundary=boundary):
                client = RecordingNextActionClient()
                decision = QwenNextActionPlanner(client).decide(
                    goal=goal(),
                    questions=questions(),
                    findings=[],
                    action_records=records,
                    tool_call_count=tool_calls,
                )

                self.assertEqual(
                    decision.decision_type,
                    DecisionType.STOP.value,
                )
                self.assertEqual(
                    decision.goal_status,
                    GoalStatus.BUDGET_EXHAUSTED.value,
                )
                self.assertEqual(
                    decision.stop_reason,
                    StopReason.BUDGET_EXHAUSTED.value,
                )
                self.assertEqual(len(client.requests), 0)
                self.assertTrue(
                    all(
                        update.status == EvidenceGapStatus.UNAVAILABLE.value
                        for update in decision.question_updates
                    )
                )

    def test_prompt_and_runtime_keep_tool_dispatch_outside_planner(self) -> None:
        import yield_rca_core.next_action_planner as next_action_planner

        prompt = load_prompt("next_action_planner", "v1").lower()
        self.assertIn("choose exactly one entry from allowed_actions", prompt)
        self.assertIn("executable_causal_gap_ids is empty", prompt)
        self.assertIn("impact lot is a result", prompt)
        self.assertIn("does not answer this specific question", prompt)
        source = inspect.getsource(next_action_planner).lower()
        self.assertNotIn("yield_rca_core.repositories", source)
        self.assertNotIn("yield_rca_core.tool_layer", source)


if __name__ == "__main__":
    unittest.main()
