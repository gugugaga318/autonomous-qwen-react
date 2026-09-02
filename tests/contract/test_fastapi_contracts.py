from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "backend"))

from yield_rca_api.app import _state_api_payload, create_app  # noqa: E402
from yield_rca_api.schemas import (  # noqa: E402
    CreateRCAJobRequest,
    DecisionEvaluationResponse,
    ExecutionMetadataResponse,
    PlannerAttemptDiagnosticResponse,
    QuestionUpdateReviewResponse,
    RcaDiagnosisResponse,
    RCAJobStateResponse,
    RunEvaluationResponse,
)
from yield_rca_core.authoritative_result import (  # noqa: E402
    AuthoritativeRCAResult,
    ImpactPublicationResult,
)
from yield_rca_core.evidence_models import Evidence, EvidenceSourceType  # noqa: E402
from yield_rca_core.models import (  # noqa: E402
    AgentFinding,
    AgentKind,
    RCAJob,
    RCAState,
)


class FastAPIContractTest(unittest.TestCase):
    def test_required_rca_routes_are_registered(self) -> None:
        app = create_app()
        routes = {
            (method, route.path)
            for route in app.routes
            for method in getattr(route, "methods", set())
        }

        self.assertIn(("POST", "/rca/jobs"), routes)
        self.assertIn(("GET", "/rca/jobs/{job_id}"), routes)
        self.assertIn(("GET", "/rca/jobs/{job_id}/report"), routes)
        self.assertIn(("POST", "/rca/jobs/{job_id}/cancel"), routes)
        self.assertIn(("GET", "/rca/jobs/{job_id}/events"), routes)
        self.assertIn(("GET", "/ready"), routes)

    def test_create_job_openapi_uses_async_acceptance_response(self) -> None:
        operation = create_app().openapi()["paths"]["/rca/jobs"]["post"]
        self.assertIn("202", operation["responses"])
        self.assertNotIn("201", operation["responses"])
        properties = create_app().openapi()["components"]["schemas"][
            "CreateRCAJobResponse"
        ]["properties"]
        self.assertIn("events_url", properties)
        self.assertIn("cancel_url", properties)
        self.assertIn("idempotency_key", properties)
        cancel_operation = create_app().openapi()["paths"][
            "/rca/jobs/{job_id}/cancel"
        ]["post"]
        self.assertIn("202", cancel_operation["responses"])
        events_operation = create_app().openapi()["paths"][
            "/rca/jobs/{job_id}/events"
        ]["get"]
        self.assertIn("text/event-stream", events_operation["responses"]["200"]["content"])

    def test_create_request_normalizes_query_and_rejects_unknown_fields(self) -> None:
        request = CreateRCAJobRequest(user_query="  Analyze the July yield drop.  ")
        self.assertEqual(request.user_query, "Analyze the July yield drop.")

        with self.assertRaises(ValueError):
            CreateRCAJobRequest(user_query="Analyze.", generate_synthetic_data=True)

    def test_api_has_no_synthetic_generator_dependency(self) -> None:
        import yield_rca_api.app as api_app

        source = inspect.getsource(api_app).lower()
        forbidden = (
            "generate_synthetic_fab_data",
            "scripts.generate_synthetic",
            "subprocess",
            "seed_database",
        )
        for dependency in forbidden:
            self.assertNotIn(dependency, source)

    def test_job_response_openapi_describes_typed_evidence(self) -> None:
        app = create_app()
        schema = app.openapi()
        components = schema["components"]["schemas"]

        evidence_schema = components["EvidenceResponse"]["properties"]
        self.assertIn("evidence_type", evidence_schema)
        self.assertIn("source_agent", evidence_schema)
        self.assertIn("source_tool", evidence_schema)
        self.assertIn("observation", evidence_schema)
        self.assertIn("entities", evidence_schema)
        self.assertIn("confidence", evidence_schema)
        self.assertIn("evidence_schema_version", evidence_schema)

    def test_execution_metadata_exposes_typed_intent_planner_diagnostics(self) -> None:
        diagnostic = PlannerAttemptDiagnosticResponse(
            stage="intent_planning",
            attempt=1,
            prompt_name="intent_planner",
            prompt_version="v1",
            outcome="failure",
            failure_category="semantic_validation_error",
            reason_code="known_fact_changed",
            field_path="$.goal.known_facts.defect",
            message="Qwen changed an explicit known fact.",
            repair_feedback_sent=True,
            candidate_summary={"intent": "root_cause"},
            baseline_diff={"known_fact_keys_changed": ["defect"]},
            provider_request_id=None,
        )
        metadata = ExecutionMetadataResponse.model_validate(
            {
                "agent_mode": "llm",
                "intent_planner_attempt_diagnostics": [diagnostic.model_dump()],
            }
        )
        state = RCAJobStateResponse(
            job={"job_id": "JOB_INTENT_DIAGNOSTIC"},
            execution_metadata=metadata,
        )
        payload = state.model_dump()["execution_metadata"]

        self.assertEqual(payload["agent_mode"], "llm")
        self.assertEqual(
            payload["intent_planner_attempt_diagnostics"][0]["reason_code"],
            "known_fact_changed",
        )
        components = create_app().openapi()["components"]["schemas"]
        metadata_properties = components["ExecutionMetadataResponse"]["properties"]
        diagnostic_properties = components["PlannerAttemptDiagnosticResponse"][
            "properties"
        ]
        self.assertIn("intent_planner_attempt_diagnostics", metadata_properties)
        self.assertEqual(
            set(diagnostic_properties),
            {
                "stage",
                "attempt",
                "prompt_name",
                "prompt_version",
                "outcome",
                "failure_category",
                "reason_code",
                "field_path",
                "message",
                "repair_feedback_sent",
                "candidate_summary",
                "baseline_diff",
                "provider_request_id",
            },
        )
        with self.assertRaises(ValueError):
            PlannerAttemptDiagnosticResponse(
                **{
                    **diagnostic.model_dump(),
                    "outcome": "success",
                    "repair_feedback_sent": False,
                }
            )

    def test_job_state_accepts_typed_run_evaluation_and_legacy_omission(
        self,
    ) -> None:
        evaluation = RunEvaluationResponse(
            goal_id="GOAL_LOT_01",
            goal_success=True,
            stop_correct=True,
            summary="The impact-scope goal was answered before stopping.",
            decision_evaluations=[
                DecisionEvaluationResponse(
                    decision_id="DECISION_MES_01",
                    decision_valid=True,
                    evidence_gain=True,
                    redundant=False,
                    reason="The MES action added new impact-lot Evidence.",
                    new_evidence_ids=["EV_MES_IMPACT_LOTS"],
                )
            ],
        )

        state = RCAJobStateResponse(
            job={"job_id": "JOB_EVALUATED"},
            run_evaluation=evaluation,
        )
        legacy_state = RCAJobStateResponse(job={"job_id": "JOB_LEGACY"})

        run_evaluation = state.run_evaluation
        assert run_evaluation is not None
        self.assertIsInstance(run_evaluation, RunEvaluationResponse)
        self.assertEqual(run_evaluation.goal_id, "GOAL_LOT_01")
        self.assertEqual(
            run_evaluation.decision_evaluations[0].new_evidence_ids,
            ["EV_MES_IMPACT_LOTS"],
        )
        self.assertIsNone(legacy_state.run_evaluation)

    def test_run_evaluation_response_rejects_unagreed_score_fields(self) -> None:
        with self.assertRaises(ValueError):
            RunEvaluationResponse(
                goal_id="GOAL_LOT_01",
                goal_success=True,
                stop_correct=True,
                summary="Only the five agreed boolean metrics are exposed.",
                decision_evaluations=[
                    DecisionEvaluationResponse(
                        decision_id="DECISION_STOP",
                        decision_valid=True,
                        evidence_gain=False,
                        redundant=False,
                        reason="The typed stop decision passed runtime checks.",
                    )
                ],
                weighted_score=0.95,  # type: ignore[call-arg]
            )

    def test_openapi_exposes_nested_run_evaluation_contracts(self) -> None:
        components = create_app().openapi()["components"]["schemas"]

        state_properties = components["RCAJobStateResponse"]["properties"]
        run_properties = components["RunEvaluationResponse"]["properties"]
        decision_properties = components["DecisionEvaluationResponse"]["properties"]

        self.assertIn("run_evaluation", state_properties)
        self.assertEqual(
            set(run_properties),
            {
                "goal_id",
                "goal_success",
                "stop_correct",
                "summary",
                "decision_evaluations",
            },
        )
        self.assertEqual(
            set(decision_properties),
            {
                "decision_id",
                "decision_valid",
                "evidence_gain",
                "redundant",
                "reason",
                "new_evidence_ids",
            },
        )

    def test_job_state_exposes_question_update_review_contract(self) -> None:
        review = QuestionUpdateReviewResponse(
            decision_id="DECISION_01",
            disposition="rejected",
            reason_code="non_terminal_status",
            reason="QuestionUpdate status=open was rejected.",
            update_index=0,
            question_id="Q_ROOT_CAUSE",
            claimed_status="open",
        )

        state = RCAJobStateResponse(
            job={"job_id": "JOB_REVIEWED"},
            question_update_reviews=[review],
        )
        legacy_state = RCAJobStateResponse(job={"job_id": "JOB_LEGACY"})
        components = create_app().openapi()["components"]["schemas"]

        self.assertEqual(state.question_update_reviews, [review])
        self.assertEqual(legacy_state.question_update_reviews, [])
        self.assertIn(
            "question_update_reviews",
            components["RCAJobStateResponse"]["properties"],
        )
        self.assertEqual(
            set(components["QuestionUpdateReviewResponse"]["properties"]),
            {
                "decision_id",
                "disposition",
                "reason_code",
                "reason",
                "update_index",
                "question_id",
                "claimed_status",
            },
        )

    def test_job_state_exposes_rca_diagnosis_projection(self) -> None:
        diagnosis = RcaDiagnosisResponse(
            finding_id="RCA_AUTH",
            conclusion_status="inconclusive",
            root_cause="Pressure excursion",
            ranked_candidates=[
                {
                    "root_cause": "Pressure excursion",
                    "causal_evidence_matrix": {
                        "status": "incomplete",
                        "claims": {},
                    },
                }
            ],
            causal_evidence_gaps=[
                {
                    "gap_id": "candidate_0.mechanism.incomplete",
                    "candidate_index": 0,
                    "claim": "mechanism",
                    "status": "incomplete",
                    "reason": "Mechanism Evidence is incomplete.",
                    "question_kind": "process_mechanism",
                    "allowed_actions": ["retrieve_knowledge"],
                    "evidence_ids": [],
                }
            ],
            confirmation_gate={"status": "inconclusive", "checks": {"mechanism": False}},
            impact_lot_gate={"confirmed_impact_lots": ["LOT_IMPACT"], "rows": []},
        )
        state = RCAJobStateResponse(job={"job_id": "JOB_DIAGNOSIS"}, rca_diagnosis=diagnosis)
        self.assertEqual(state.rca_diagnosis.finding_id, "RCA_AUTH")
        components = create_app().openapi()["components"]["schemas"]
        self.assertIn("rca_diagnosis", components["RCAJobStateResponse"]["properties"])
        self.assertIn(
            "RcaDiagnosisResponse",
            str(components["RCAJobStateResponse"]["properties"]["rca_diagnosis"]),
        )

    def test_job_state_projects_populated_typed_authority_without_legacy_finding(
        self,
    ) -> None:
        evidence = Evidence(
            evidence_id="EV_API_AUTHORITY",
            source_type=EvidenceSourceType.ANALYTICS.value,
            source_id="authority:api",
            summary="Evidence for the typed API authority projection.",
        )
        authority = AuthoritativeRCAResult(
            result_id="RCA_RESULT_API",
            source_finding_id=None,
            source_hypothesis_id=None,
            conclusion_status="supported",
            root_cause_candidate_id="CANDIDATE_API",
            root_cause="Typed API root cause",
            confirmation_status="supported",
            competition_status="not_required",
            terminal_reason="candidate_competition_not_required",
            evidence_refs=(evidence.evidence_id,),
        )
        publication = ImpactPublicationResult(
            rca_result_id=authority.result_id,
            publication_status="confirmed",
            confirmed_impact_lots=("LOT_IMPACT_API",),
            evidence_refs=(evidence.evidence_id,),
        )
        state = RCAState(
            job=RCAJob(job_id="JOB_API_AUTHORITY", user_query="Project typed authority."),
            evidence=[evidence],
            authoritative_rca_result=authority,
            impact_publication_result=publication,
        )

        response = RCAJobStateResponse.model_validate(_state_api_payload(state))

        self.assertEqual(response.authoritative_rca_result.result_id, authority.result_id)
        self.assertEqual(
            response.impact_publication_result.confirmed_impact_lots,
            ["LOT_IMPACT_API"],
        )
        self.assertIsNotNone(response.rca_diagnosis)
        self.assertIsNone(response.rca_diagnosis.finding_id)
        self.assertEqual(response.rca_diagnosis.result_id, authority.result_id)
        self.assertEqual(response.rca_diagnosis.root_cause, authority.root_cause)
        self.assertEqual(response.rca_diagnosis.conclusion_status, "supported")
        self.assertEqual(response.rca_diagnosis.publication_status, "confirmed")
        self.assertEqual(response.rca_diagnosis.confirmed_impact_lots, ["LOT_IMPACT_API"])

        components = create_app().openapi()["components"]["schemas"]
        state_properties = components["RCAJobStateResponse"]["properties"]
        self.assertIn("authoritative_rca_result", state_properties)
        self.assertIn("impact_publication_result", state_properties)

    def test_typed_authority_without_source_does_not_project_stale_finding_details(
        self,
    ) -> None:
        evidence = Evidence(
            evidence_id="EV_API_STALE_FINDING",
            source_type=EvidenceSourceType.ANALYTICS.value,
            source_id="authority:api-stale-finding",
            summary="Evidence retained by a stale legacy Finding.",
        )
        stale_finding = AgentFinding(
            finding_id="FINDING_API_STALE",
            agent=AgentKind.RCA_REASONING.value,
            summary="Stale Finding retained for audit history.",
            confidence=0.8,
            evidence_ids=[evidence.evidence_id],
            details={
                "root_cause": "Stale API root cause",
                "conclusion_status": "supported",
                "ranked_candidates": [{"root_cause": "Stale API candidate"}],
                "confirmation_gate": {"status": "supported"},
                "impact_lot_gate": {
                    "publication_status": "confirmed",
                    "confirmed_impact_lots": ["LOT_STALE_API"],
                },
            },
        )
        authority = AuthoritativeRCAResult(
            result_id="RCA_RESULT_API_WITHOUT_SOURCE",
            source_finding_id=None,
            source_hypothesis_id=None,
            conclusion_status="inconclusive",
            root_cause_candidate_id=None,
            root_cause=None,
            confirmation_status="inconclusive",
            competition_status="exhausted",
            terminal_reason="no_high_value_action_remains",
            evidence_refs=(evidence.evidence_id,),
        )
        state = RCAState(
            job=RCAJob(job_id="JOB_API_STALE_FINDING", user_query="Project typed authority."),
            evidence=[evidence],
            findings=[stale_finding],
            authoritative_rca_finding_id=stale_finding.finding_id,
            authoritative_rca_result=authority,
        )

        response = RCAJobStateResponse.model_validate(_state_api_payload(state))

        self.assertIsNotNone(response.rca_diagnosis)
        self.assertIsNone(response.rca_diagnosis.finding_id)
        self.assertIsNone(response.rca_diagnosis.root_cause)
        self.assertEqual(response.rca_diagnosis.ranked_candidates, [])
        self.assertEqual(response.rca_diagnosis.confirmation_gate, {})
        self.assertEqual(response.rca_diagnosis.impact_lot_gate, {})
        self.assertEqual(response.rca_diagnosis.confirmed_impact_lots, [])

    def test_legacy_api_state_keeps_new_authority_fields_optional(self) -> None:
        response = RCAJobStateResponse.model_validate(
            RCAState(
                job=RCAJob(job_id="JOB_API_LEGACY", user_query="Project legacy State.")
            ).to_dict()
        )

        self.assertIsNone(response.authoritative_rca_result)
        self.assertIsNone(response.impact_publication_result)


if __name__ == "__main__":
    unittest.main()
