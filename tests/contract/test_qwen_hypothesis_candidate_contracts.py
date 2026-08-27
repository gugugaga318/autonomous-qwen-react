from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core"))

from yield_rca_core.causal_candidate_comparison import (  # noqa: E402
    QwenHypothesisCandidateComparator,
)
from yield_rca_core.causal_evidence_gap import (  # noqa: E402
    build_causal_evidence_gaps,
    build_hypothesis_discrimination_gaps,
)
from yield_rca_core.causal_evidence_matrix import (  # noqa: E402
    build_causal_evidence_matrix,
)
from yield_rca_core.causal_hypothesis import CausalClaimStatus, CausalHypothesis  # noqa: E402
from yield_rca_core.causal_investigation_models import (  # noqa: E402
    CandidateCompetitionStatus,
    CandidateDistinguishingPrediction,
    CandidateSemanticProfile,
    CausalLaneRecord,
    CompetitionGapReason,
    CompetitionRequirement,
)
from yield_rca_core.evidence_models import (  # noqa: E402
    EVIDENCE_SCHEMA_VERSION,
    EntityType,
    Evidence,
    EvidenceEntity,
    EvidenceSourceType,
    EvidenceType,
)
from yield_rca_core.hypothesis_candidate_generator import (  # noqa: E402
    QwenHypothesisCandidateGenerator,
    _mechanism_semantics_are_distinct,
    _scope_semantics_are_distinct,
)
from yield_rca_core.hypothesis_engine import HypothesisEngine  # noqa: E402
from yield_rca_core.llm_gateway import (  # noqa: E402
    FakeLLMClient,
    LLMCallError,
    LLMOutputValidationError,
    LLMRequest,
    LLMResponse,
)
from yield_rca_core.models import AgentFinding, AgentKind  # noqa: E402
from yield_rca_core.rca_reasoning_agent import RCAReasoningAgent  # noqa: E402


def typed_evidence(
    *,
    evidence_id: str,
    evidence_type: str,
    agent: str,
    entity_type: str,
    entity_id: str,
    metadata: dict[str, object] | None = None,
    timestamp: str | None = None,
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        source_type={
            AgentKind.MES.value: EvidenceSourceType.ANALYTICS.value,
            AgentKind.FDC.value: EvidenceSourceType.FDC.value,
            AgentKind.DEFECT_WAT.value: EvidenceSourceType.DEFECT.value,
        }[agent],
        source_id=f"SOURCE_{evidence_id}",
        summary=f"Observation for {evidence_id}",
        evidence_type=evidence_type,
        source_agent=agent,
        source_tool=f"{agent}_tool",
        observation=f"Measured observation for {evidence_id}",
        entities=[
            EvidenceEntity(entity_type=EntityType.LOT.value, entity_id="LOT_01"),
            EvidenceEntity(entity_type=entity_type, entity_id=entity_id),
        ],
        metadata=metadata or {},
        timestamp=timestamp,
        confidence=0.95,
        evidence_schema_version=EVIDENCE_SCHEMA_VERSION,
    )


def lane_process_evidence(
    *,
    evidence_id: str,
    lane_id: str,
    recipe: str,
    parameter: str,
    equipment: str = "EQ_01",
    chamber: str = "EQ_01_CH01",
    operation: str = "4000",
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        source_type=EvidenceSourceType.FDC.value,
        source_id=f"SOURCE_{evidence_id}",
        summary=f"{parameter} deviation on {lane_id}",
        evidence_type=EvidenceType.PARAMETER_DEVIATION.value,
        source_agent=AgentKind.FDC.value,
        source_tool="inspect_fdc_spc",
        observation=f"Measured {parameter} deviation for {recipe}",
        entities=[
            EvidenceEntity(EntityType.LOT.value, "LOT_01"),
            EvidenceEntity(EntityType.EQUIPMENT.value, equipment),
            EvidenceEntity(EntityType.CHAMBER.value, chamber),
            EvidenceEntity(EntityType.OPERATION.value, operation),
            EvidenceEntity(EntityType.RECIPE.value, recipe),
            EvidenceEntity(EntityType.PARAMETER.value, parameter),
        ],
        metadata={
            "lane_id": lane_id,
            "recipe": recipe,
            "parameter_name": parameter,
        },
        confidence=0.95,
        evidence_schema_version=EVIDENCE_SCHEMA_VERSION,
    )


def findings_with_process_evidence(*items: Evidence) -> list[AgentFinding]:
    findings = causal_findings()
    findings[1] = AgentFinding(
        finding_id="FINDING_FDC_LANES",
        agent=AgentKind.FDC.value,
        summary="Lane-specific FDC Evidence is available.",
        confidence=0.95,
        evidence_ids=[item.evidence_id for item in items],
        evidence=list(items),
    )
    return findings


def approved_knowledge_evidence() -> Evidence:
    return Evidence(
        evidence_id="EV_KNOWLEDGE_MECHANISM",
        source_type=EvidenceSourceType.KNOWLEDGE.value,
        source_id="CASE_01",
        summary="Confirmed engineering mechanism for temperature drift and edge void.",
        evidence_type=EvidenceType.HISTORICAL_CASE_MATCH.value,
        source_agent=AgentKind.KNOWLEDGE.value,
        source_tool="retrieve_similar_case",
        observation="Confirmed case documents temperature drift causing edge void.",
        entities=[
            EvidenceEntity(
                entity_type=EntityType.KNOWLEDGE_ASSET.value,
                entity_id="CASE_01",
                attributes={"validation_status": "CONFIRMED"},
            )
        ],
        metadata={"validation_status": "CONFIRMED"},
        confidence=0.9,
        evidence_schema_version=EVIDENCE_SCHEMA_VERSION,
    )


def causal_findings() -> list[AgentFinding]:
    evidence = [
        typed_evidence(
            evidence_id="EV_EXPOSURE",
            evidence_type=EvidenceType.IMPACT_SCOPE.value,
            agent=AgentKind.MES.value,
            entity_type=EntityType.EQUIPMENT.value,
            entity_id="EQ_01",
        ),
        typed_evidence(
            evidence_id="EV_PROCESS",
            evidence_type=EvidenceType.PARAMETER_DEVIATION.value,
            agent=AgentKind.FDC.value,
            entity_type=EntityType.PARAMETER.value,
            entity_id="chamber_temperature_range",
            metadata={
                "processing_window": {
                    "start": "2026-01-01T00:00:00",
                    "end": "2026-01-01T01:00:00",
                },
                "causal_support_kind": "matched_comparison",
                "validation_status": "CONFIRMED",
            },
            timestamp="2026-01-01T00:30:00",
        ),
        typed_evidence(
            evidence_id="EV_PRODUCT",
            evidence_type=EvidenceType.DEFECT_SIGNAL.value,
            agent=AgentKind.DEFECT_WAT.value,
            entity_type=EntityType.DEFECT.value,
            entity_id="edge_void",
        ),
    ]
    return [
        AgentFinding(
            finding_id=f"FINDING_{item.source_agent}",
            agent=str(item.source_agent),
            summary=item.summary,
            confidence=0.9,
            evidence_ids=[item.evidence_id],
            evidence=[item],
        )
        for item in evidence
    ]


def proposal(*, supporting: list[str] | None = None) -> dict[str, object]:
    return {
        "root_cause": "EQ_01 chamber temperature control drift",
        "causal_explanation": (
            "The chamber temperature deviation destabilizes surface-reaction "
            "kinetics and creates non-uniform film nucleation, producing the "
            "observed edge void signature."
        ),
        "supporting_evidence_ids": supporting
        or ["EV_EXPOSURE", "EV_PROCESS", "EV_PRODUCT"],
        "contradicting_evidence_ids": [],
    }


def semantic_profile(
    candidate_index: int,
    *,
    scope_relation: str,
    claimed_lane_ids: list[str],
    comparison_lane_ids: list[str],
    prediction: str,
    discriminator_kind: str = "product_outcome",
    primary_mechanism: str = "Plasma-control drift",
    effect_modifier: str | None = None,
    mechanism_relation: str | None = None,
    depends_on_candidate_index: int | None = None,
) -> dict[str, object]:
    return {
        "candidate_index": candidate_index,
        "claimed_scope": {
            "scope_relation": scope_relation,
            "lane_ids": claimed_lane_ids,
        },
        "comparison_scope": {"lane_ids": comparison_lane_ids},
        "mechanism_claim": "Plasma-control drift changes contact etch response.",
        "primary_mechanism": primary_mechanism,
        "effect_modifier": effect_modifier,
        "depends_on_candidate_index": depends_on_candidate_index,
        "mechanism_relation": (
            mechanism_relation
            or ("reference" if candidate_index == 0 else "independent_alternative")
        ),
        "distinguishing_predictions": [
            {
                "discriminator_kind": discriminator_kind,
                "lane_ids": comparison_lane_ids,
                "prediction": prediction,
            }
        ],
    }


def comparison_proposals() -> tuple[dict[str, object], dict[str, object]]:
    """Return equal-confidence candidates with different mechanism quality."""

    weak = {
        **proposal(),
        "root_cause": "EQ_01 chamber_temperature_range drift correlation A",
        "causal_explanation": (
            "The chamber_temperature_range deviation causes the observed "
            "edge_void outcome."
        ),
    }
    strong = {
        **proposal(),
        "root_cause": "EQ_01 chamber_temperature_range drift mechanism B",
    }
    return weak, strong


class CandidateClient(FakeLLMClient):
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = responses
        self.requests: list[LLMRequest] = []

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        base = super().complete_json(request)
        response = self.responses[min(len(self.requests) - 1, len(self.responses) - 1)]
        return LLMResponse(data=dict(response), usage=base.usage)


class CandidateProviderFailureClient(FakeLLMClient):
    def complete_json(self, request: LLMRequest) -> LLMResponse:
        raise LLMCallError(
            "candidate provider unavailable",
            status_code=503,
            failure_category="provider_http_error",
        )


class QwenHypothesisCandidateContractTest(unittest.TestCase):
    def test_scope_collapse_requires_same_declared_scope_and_prediction(self) -> None:
        shared = CandidateSemanticProfile(
            candidate_id="REQ:llm:1",
            scope_relation="shared_effect",
            claimed_lane_ids=("LANE_A", "LANE_B"),
            comparison_lane_ids=("LANE_A", "LANE_B"),
            mechanism_claim="Shared chamber drift changes plasma response.",
            primary_mechanism="Chamber wall conditioning drift",
            mechanism_relation="reference",
            distinguishing_predictions=(
                CandidateDistinguishingPrediction(
                    discriminator_kind="product_outcome",
                    lane_ids=("LANE_A", "LANE_B"),
                    prediction="Both Lanes produce compatible outcomes.",
                ),
            ),
        )
        paraphrase = CandidateSemanticProfile.from_dict(
            {**shared.to_dict(), "candidate_id": "REQ:llm:2"}
        )
        sensitive = CandidateSemanticProfile(
            candidate_id="REQ:llm:2",
            scope_relation="differential_sensitivity",
            claimed_lane_ids=("LANE_A",),
            comparison_lane_ids=("LANE_A", "LANE_B"),
            mechanism_claim="RCP_A amplifies the shared chamber drift.",
            distinguishing_predictions=(
                CandidateDistinguishingPrediction(
                    discriminator_kind="product_outcome",
                    lane_ids=("LANE_A", "LANE_B"),
                    prediction="LANE_A produces a more severe outcome than LANE_B.",
                ),
            ),
        )
        mechanism_alternative = CandidateSemanticProfile.from_dict(
            {
                **shared.to_dict(),
                "candidate_id": "REQ:llm:2",
                "mechanism_claim": (
                    "Recipe timing creates endpoint overshoot independently "
                    "of chamber conditioning."
                ),
                "primary_mechanism": "Recipe timing control fault",
                "mechanism_relation": "independent_alternative",
                "distinguishing_predictions": [
                    {
                        "discriminator_kind": "temporal_alignment",
                        "lane_ids": ["LANE_A", "LANE_B"],
                        "prediction": (
                            "The anomaly follows recipe timing rather than "
                            "shared chamber exposure."
                        ),
                        "schema_version": "1.0",
                    }
                ],
            }
        )

        self.assertFalse(_scope_semantics_are_distinct(shared, paraphrase))
        self.assertTrue(_scope_semantics_are_distinct(shared, sensitive))
        self.assertFalse(
            _scope_semantics_are_distinct(shared, mechanism_alternative)
        )
        self.assertTrue(
            _mechanism_semantics_are_distinct(shared, mechanism_alternative)
        )

    def test_comparator_cannot_own_discriminator_gap_selection(self) -> None:
        candidate = proposal()
        matrix = build_causal_evidence_matrix(
            CausalHypothesis.from_mapping(candidate),
            [item for finding in causal_findings() for item in finding.evidence],
        )
        client = CandidateClient(
            [
                {
                    "preferred_candidate_index": 0,
                    "comparison_explanation": "Candidate zero is stronger.",
                    "selected_gap_id": "GAP_1",
                }
            ]
        )

        with self.assertRaisesRegex(
            LLMOutputValidationError,
            "adversarial challenge owns discriminator Gap selection",
        ):
            QwenHypothesisCandidateComparator(client).compare(
                request_id="REQ_SINGLE_GAP_OWNER",
                candidates=[candidate],
                matrices=[matrix],
                evidence_gaps=[{"gap_id": "GAP_1", "candidate_index": 0}],
            )

    def test_first_candidate_request_receives_traceable_lane_first_synthesis(
        self,
    ) -> None:
        lane_process = lane_process_evidence(
            evidence_id="EV_LANE_PROCESS",
            lane_id="LANE_4000_A",
            recipe="RCP_A",
            parameter="backside_pressure_cv",
        )
        global_outcome = typed_evidence(
            evidence_id="EV_GLOBAL_OUTCOME",
            evidence_type=EvidenceType.DEFECT_SIGNAL.value,
            agent=AgentKind.DEFECT_WAT.value,
            entity_type=EntityType.DEFECT.value,
            entity_id="center_seam_void",
        )
        findings = findings_with_process_evidence(lane_process)
        findings[2] = AgentFinding(
            finding_id="FINDING_GLOBAL_OUTCOME",
            agent=AgentKind.DEFECT_WAT.value,
            summary=global_outcome.summary,
            confidence=0.95,
            evidence_ids=[global_outcome.evidence_id],
            evidence=[global_outcome],
        )
        client = CandidateClient(
            [{"candidates": [], "analysis_summary": "No bounded candidate."}]
        )

        QwenHypothesisCandidateGenerator(client).generate(
            request_id="REQ_LANE_FIRST",
            findings=findings,
            causal_lanes=[
                {
                    "lane_id": "LANE_4000_A",
                    "operation": "4000",
                    "equipment": "EQ_01",
                    "chamber": "EQ_01_CH01",
                    "recipe": "RCP_A",
                    "parameter_scope": ["backside_pressure_cv"],
                    "exposed_lot_ids": ["LOT_01", "LOT_02"],
                    "time_window": [
                        "2026-01-01T00:00:00Z",
                        "2026-01-01T01:00:00Z",
                    ],
                    "initial_evidence_ids": ["EV_EXPOSURE"],
                    "priority_score": 0.9,
                    "investigation_status": "evidence_collected",
                }
            ],
        )

        payload = client.requests[0].payload
        synthesis = payload["evidence_synthesis"]
        self.assertEqual(synthesis["schema"], "lane_first_v1")
        self.assertEqual(synthesis["active_lane_count"], 1)
        lane = synthesis["active_causal_lanes"][0]
        self.assertEqual(lane["lane_id"], "LANE_4000_A")
        self.assertEqual(lane["operation"], "4000")
        self.assertEqual(lane["recipe"], "RCP_A")
        self.assertEqual(
            lane["facts"]["process_excursions"][0]["evidence_id"],
            "EV_LANE_PROCESS",
        )
        self.assertEqual(
            synthesis["global_facts"]["outcomes"][0]["evidence_id"],
            "EV_GLOBAL_OUTCOME",
        )
        mechanism_inputs = synthesis["mechanism_bridge_inputs"]
        self.assertEqual(
            mechanism_inputs["by_lane"][0][
                "parameter_or_process_evidence_ids"
            ],
            ["EV_LANE_PROCESS"],
        )
        self.assertEqual(
            mechanism_inputs["global_outcome_evidence_ids"],
            ["EV_GLOBAL_OUTCOME"],
        )
        self.assertEqual(mechanism_inputs["approved_knowledge_evidence_ids"], [])
        register_ids = {
            item["evidence_id"] for item in payload["typed_evidence_register"]
        }
        self.assertIn("EV_LANE_PROCESS", register_ids)
        self.assertIn("EV_GLOBAL_OUTCOME", register_ids)
        self.assertEqual(payload["prior_authoritative_candidates"], [])

    def test_lane_first_synthesis_is_bounded_and_excludes_terminal_lanes(self) -> None:
        process = lane_process_evidence(
            evidence_id="EV_ACTIVE_PROCESS",
            lane_id="LANE_HIGH",
            recipe="RCP_HIGH",
            parameter="pressure_cv",
        )
        inactive_process = lane_process_evidence(
            evidence_id="EV_INACTIVE_PROCESS",
            lane_id="LANE_LOW",
            recipe="RCP_LOW",
            parameter="temperature_range",
        )
        client = CandidateClient(
            [{"candidates": [], "analysis_summary": "No bounded candidate."}]
        )
        lanes = [
            {
                "lane_id": "LANE_LOW",
                "priority_score": 0.1,
                "investigation_status": "uninvestigated",
            },
            {
                "lane_id": "LANE_BLOCKED",
                "priority_score": 1.0,
                "investigation_status": "blocked",
            },
            {
                "lane_id": "LANE_HIGH",
                "priority_score": 0.9,
                "investigation_status": "evidence_collected",
            },
            {
                "lane_id": "LANE_MID_A",
                "priority_score": 0.7,
                "investigation_status": "uninvestigated",
            },
            {
                "lane_id": "LANE_MID_B",
                "priority_score": 0.6,
                "investigation_status": "uninvestigated",
            },
        ]

        QwenHypothesisCandidateGenerator(client).generate(
            request_id="REQ_BOUNDED_LANES",
            findings=findings_with_process_evidence(process, inactive_process),
            causal_lanes=lanes,
        )

        payload = client.requests[0].payload
        active_ids = [
            item["lane_id"]
            for item in payload["evidence_synthesis"]["active_causal_lanes"]
        ]
        self.assertEqual(active_ids, ["LANE_HIGH", "LANE_MID_A", "LANE_MID_B"])
        self.assertNotIn("LANE_BLOCKED", active_ids)
        self.assertNotIn("LANE_LOW", active_ids)
        register_ids = {
            item["evidence_id"] for item in payload["typed_evidence_register"]
        }
        self.assertIn("EV_ACTIVE_PROCESS", register_ids)
        self.assertNotIn("EV_INACTIVE_PROCESS", register_ids)

    def test_candidate_prompt_uses_bounded_nonduplicated_evidence_projection(
        self,
    ) -> None:
        findings = causal_findings()
        process = findings[1].evidence[0]
        bloated = Evidence.from_dict(
            {
                **process.to_dict(),
                "summary": "S" * 10_000,
                "observation": "O" * 10_000,
                "metadata": {
                    "lane_id": "LANE_4000",
                    "direction": "high",
                    "raw_rows": ["RAW" * 5_000 for _ in range(20)],
                },
            }
        )
        findings[1] = AgentFinding(
            finding_id="FINDING_FDC_BLOATED",
            agent=AgentKind.FDC.value,
            summary="FDC typed Evidence is available.",
            confidence=0.9,
            evidence_ids=[bloated.evidence_id],
            evidence=[bloated],
        )
        client = CandidateClient(
            [
                {
                    "candidates": [],
                    "analysis_summary": "No bounded candidate is required.",
                }
            ]
        )

        QwenHypothesisCandidateGenerator(client).generate(
            request_id="REQ_COMPACT_PROMPT",
            findings=findings,
        )

        payload = client.requests[0].payload
        serialized = json.dumps(payload)
        process_row = next(
            item
            for item in payload["typed_evidence_register"]
            if item["evidence_id"] == "EV_PROCESS"
        )
        self.assertNotIn("raw_rows", serialized)
        self.assertLessEqual(len(process_row["fact"]), 480)
        self.assertEqual(payload["evidence_synthesis"]["evidence_count"], 3)
        self.assertIn("group_counts", payload["evidence_synthesis"])
        self.assertNotIn("process_excursions", payload["evidence_synthesis"])
        self.assertLess(len(serialized), 20_000)

    def test_approved_knowledge_can_support_mechanism_without_becoming_a_causal_lane(
        self,
    ) -> None:
        findings = causal_findings()
        knowledge = approved_knowledge_evidence()
        findings.append(
            AgentFinding(
                finding_id="FINDING_KNOWLEDGE",
                agent=AgentKind.KNOWLEDGE.value,
                summary=knowledge.summary,
                confidence=0.9,
                evidence_ids=[knowledge.evidence_id],
                evidence=[knowledge],
            )
        )
        candidate = proposal(
            supporting=["EV_EXPOSURE", "EV_PROCESS", "EV_PRODUCT", knowledge.evidence_id]
        )
        client = CandidateClient(
            [
                {
                    "candidates": [candidate],
                    "analysis_summary": "Current evidence plus approved mechanism.",
                }
            ]
        )

        generated = QwenHypothesisCandidateGenerator(client).generate(
            request_id="REQ_APPROVED_KNOWLEDGE",
            findings=findings,
        )

        self.assertEqual(len(generated.candidates), 1)
        result = HypothesisEngine().analyze(
            request_id="REQ_APPROVED_KNOWLEDGE",
            findings=findings,
            mode="active",
            external_candidates=[candidate],
        )
        selected = result["candidates"][0]
        self.assertEqual(selected["mechanism_support_source"], "approved_knowledge")
        self.assertEqual(set(selected["causal_lanes"]), {
            "shared_exposure",
            "process_anomaly",
            "product_outcome",
        })

    def test_unapproved_knowledge_is_not_accepted_as_mechanism_support(self) -> None:
        knowledge = approved_knowledge_evidence()
        unapproved = Evidence.from_dict(
            {
                **knowledge.to_dict(),
                "metadata": {"validation_status": "DRAFT"},
                "entities": [
                    {
                        "entity_type": EntityType.KNOWLEDGE_ASSET.value,
                        "entity_id": "CASE_01",
                        "attributes": {"validation_status": "DRAFT"},
                    }
                ],
            }
        )
        findings = causal_findings()
        findings.append(
            AgentFinding(
                finding_id="FINDING_KNOWLEDGE_DRAFT",
                agent=AgentKind.KNOWLEDGE.value,
                summary=unapproved.summary,
                confidence=0.9,
                evidence_ids=[unapproved.evidence_id],
                evidence=[unapproved],
            )
        )
        candidate = proposal(
            supporting=["EV_EXPOSURE", "EV_PROCESS", "EV_PRODUCT", unapproved.evidence_id]
        )
        client = CandidateClient(
            [
                {
                    "candidates": [candidate],
                    "analysis_summary": "Draft knowledge is not causal proof.",
                }
            ]
        )

        generated = QwenHypothesisCandidateGenerator(client).generate(
            request_id="REQ_DRAFT_KNOWLEDGE",
            findings=findings,
        )

        self.assertEqual(generated.candidates, ())
        self.assertTrue(generated.candidate_output_invalid)

    def test_parameter_direction_mismatch_is_a_matrix_conflict_and_facts_are_exposed(
        self,
    ) -> None:
        findings = causal_findings()
        process = typed_evidence(
            evidence_id="EV_PROCESS",
            evidence_type=EvidenceType.PARAMETER_DEVIATION.value,
            agent=AgentKind.FDC.value,
            entity_type=EntityType.PARAMETER.value,
            entity_id="chamber_temperature_range",
            metadata={
                "direction": "high",
                "magnitude": 2.4,
                "processing_window": {
                    "start": "2026-01-01T00:00:00",
                    "end": "2026-01-01T01:00:00",
                },
            },
            timestamp="2026-01-01T00:30:00",
        )
        findings[1] = AgentFinding(
            finding_id="FINDING_FDC_DIRECTION",
            agent=AgentKind.FDC.value,
            summary=process.summary,
            confidence=0.9,
            evidence_ids=[process.evidence_id],
            evidence=[process],
        )
        candidate = CausalHypothesis(
            root_cause="EQ_01 chamber temperature low control drift",
            causal_explanation="Low temperature deviation produces the observed edge void.",
            supporting_evidence_ids=("EV_EXPOSURE", "EV_PROCESS", "EV_PRODUCT"),
        )
        matrix = build_causal_evidence_matrix(
            candidate,
            [item for finding in findings for item in finding.evidence],
        )

        self.assertEqual(
            matrix.claims["parameter"].status,
            CausalClaimStatus.CONFLICTED.value,
        )
        self.assertEqual(
            matrix.claims["parameter"].facts["parameter_directions"][0]["directions"],
            ["high"],
        )
        self.assertEqual(
            matrix.claims["parameter"].facts["parameter_magnitudes"][0]["magnitudes"],
            [2.4],
        )
        self.assertEqual(
            matrix.claims["parameter"].facts["processing_windows"][0]["windows"][0]["start"],
            "2026-01-01T00:00:00",
        )
    def test_invalid_candidate_isolated_when_another_candidate_is_valid(self) -> None:
        client = CandidateClient(
            [
                {
                    "candidates": [
                        proposal(supporting=["EV_UNKNOWN"]),
                        proposal(),
                    ],
                    "analysis_summary": "One invalid and one valid candidate.",
                }
            ]
        )

        result = QwenHypothesisCandidateGenerator(client).generate(
            request_id="REQ_CANDIDATE_ISOLATION",
            findings=causal_findings(),
        )

        self.assertFalse(result.candidate_output_invalid)
        self.assertEqual(len(result.candidates), 1)
        self.assertTrue(any("unknown Evidence" in item for item in result.validation_errors))
        self.assertEqual(len(client.requests), 1)

    def test_matrix_rejects_mismatched_equipment_claim(self) -> None:
        candidate = CausalHypothesis(
            root_cause="EQ_99 chamber temperature control drift",
            causal_explanation="EQ_99 temperature deviation produces the observed edge void.",
            supporting_evidence_ids=("EV_EXPOSURE", "EV_PROCESS", "EV_PRODUCT"),
        )

        matrix = build_causal_evidence_matrix(
            candidate,
            [item for finding in causal_findings() for item in finding.evidence],
        )

        self.assertEqual(
            matrix.claims["equipment"].status,
            CausalClaimStatus.CONFLICTED.value,
        )
        self.assertTrue(matrix.has_critical_conflict)

    def test_entity_prefix_does_not_turn_eq_010_into_eq_01_support(self) -> None:
        candidate = CausalHypothesis(
            root_cause="EQ_010 chamber temperature control drift",
            causal_explanation="EQ_010 temperature deviation produces edge void.",
            supporting_evidence_ids=("EV_EXPOSURE", "EV_PROCESS", "EV_PRODUCT"),
        )

        matrix = build_causal_evidence_matrix(
            candidate,
            [item for finding in causal_findings() for item in finding.evidence],
        )

        self.assertEqual(matrix.claims["equipment"].status, "conflicted")

    def test_matrix_marks_validated_matched_comparison_as_mechanism_support(self) -> None:
        candidate = CausalHypothesis(
            root_cause="EQ_01 chamber temperature control drift",
            causal_explanation=(
                "Temperature deviation destabilizes surface-reaction kinetics "
                "and creates non-uniform film nucleation, producing the observed "
                "edge void."
            ),
            supporting_evidence_ids=("EV_EXPOSURE", "EV_PROCESS", "EV_PRODUCT"),
        )

        matrix = build_causal_evidence_matrix(
            candidate,
            [item for finding in causal_findings() for item in finding.evidence],
        )

        self.assertEqual(matrix.mechanism_status, CausalClaimStatus.SUPPORTED.value)
        self.assertEqual(
            matrix.mechanism_support_source,
            "empirical_discrimination",
        )
        self.assertTrue(
            matrix.claims["mechanism"].facts["proposed_physical_bridge_terms"]
        )
        self.assertEqual(
            matrix.claims["mechanism"].facts["empirical_shared_lot_ids"],
            ["LOT_01"],
        )

    def test_parameter_to_outcome_restatement_is_not_a_supported_mechanism(
        self,
    ) -> None:
        candidate = CausalHypothesis(
            root_cause="EQ_01 chamber temperature control drift",
            causal_explanation="Temperature deviation produces the observed edge void.",
            supporting_evidence_ids=("EV_EXPOSURE", "EV_PROCESS", "EV_PRODUCT"),
        )

        matrix = build_causal_evidence_matrix(
            candidate,
            [item for finding in causal_findings() for item in finding.evidence],
        )

        mechanism = matrix.claims["mechanism"]
        self.assertEqual(mechanism.status, CausalClaimStatus.INCOMPLETE.value)
        self.assertEqual(mechanism.support_source, "llm_explanation_only")
        self.assertFalse(mechanism.facts["mechanism_expression_present"])
        self.assertIn("intervening physical process", mechanism.reason)
        gaps = build_causal_evidence_gaps([matrix])
        mechanism_gap = next(item for item in gaps if item["claim"] == "mechanism")
        self.assertEqual(mechanism_gap["question_kind"], "process_mechanism")
        self.assertIn("validate_historical_case", mechanism_gap["allowed_actions"])

    def test_empirical_mechanism_requires_shared_current_lot_scope(self) -> None:
        findings = causal_findings()
        product = findings[2].evidence[0]
        mismatched_product = Evidence.from_dict(
            {
                **product.to_dict(),
                "entities": [
                    {
                        **entity.to_dict(),
                        "entity_id": (
                            "LOT_02"
                            if entity.entity_type == EntityType.LOT.value
                            else entity.entity_id
                        ),
                    }
                    for entity in product.entities
                ],
            }
        )

        matrix = build_causal_evidence_matrix(
            CausalHypothesis.from_mapping(proposal()),
            [
                findings[0].evidence[0],
                findings[1].evidence[0],
                mismatched_product,
            ],
        )

        mechanism = matrix.claims["mechanism"]
        self.assertEqual(mechanism.status, CausalClaimStatus.INCOMPLETE.value)
        self.assertEqual(mechanism.facts["empirical_shared_lot_ids"], [])

    def test_generator_retries_unknown_evidence_and_accepts_repaired_candidate(
        self,
    ) -> None:
        client = CandidateClient(
            [
                {
                    "candidates": [proposal(supporting=["EV_UNKNOWN"])],
                    "analysis_summary": "First attempt uses an unknown ID.",
                },
                {
                    "candidates": [proposal()],
                    "analysis_summary": "Repaired evidence-bounded proposal.",
                },
            ]
        )

        result = QwenHypothesisCandidateGenerator(client).generate(
            request_id="REQ_CANDIDATE",
            findings=causal_findings(),
        )

        self.assertEqual(result.attempt_count, 2)
        self.assertEqual(len(result.validation_errors), 1)
        self.assertIn("unknown Evidence", result.validation_errors[0])
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(
            client.requests[1].payload["previous_validation_feedback"]["message"],
            result.validation_errors[0],
        )

    def test_generator_rejects_negative_evidence_as_causal_support(self) -> None:
        findings = causal_findings()
        negative = typed_evidence(
            evidence_id="EV_NORMAL",
            evidence_type=EvidenceType.NEGATIVE_SIGNAL.value,
            agent=AgentKind.FDC.value,
            entity_type=EntityType.PARAMETER.value,
            entity_id="normal_pressure",
        )
        findings[1] = AgentFinding(
            finding_id="FINDING_FDC_WITH_NORMAL",
            agent=AgentKind.FDC.value,
            summary="FDC contains process and normal control Evidence.",
            confidence=0.9,
            evidence_ids=["EV_PROCESS", "EV_NORMAL"],
            evidence=[findings[1].evidence[0], negative],
        )
        client = CandidateClient(
            [
                {
                    "candidates": [proposal(supporting=["EV_NORMAL"])],
                    "analysis_summary": "Invalid use of a normal control.",
                }
            ]
        )

        result = QwenHypothesisCandidateGenerator(client).generate(
            request_id="REQ_NEGATIVE",
            findings=findings,
        )

        self.assertEqual(len(client.requests), 2)
        self.assertEqual(result.candidates, ())
        self.assertTrue(result.candidate_output_invalid)
        self.assertTrue(any("non-supporting Evidence" in item for item in result.validation_errors))

    def test_generator_accepts_candidate_missing_shared_exposure_for_matrix_gap(self) -> None:
        client = CandidateClient(
            [
                {
                    "candidates": [
                        proposal(supporting=["EV_PROCESS", "EV_PRODUCT"])
                    ],
                    "analysis_summary": "The first proposal omitted MES exposure.",
                },
                {
                    "candidates": [proposal()],
                    "analysis_summary": "The repaired proposal joins all lanes.",
                },
            ]
        )

        result = QwenHypothesisCandidateGenerator(client).generate(
            request_id="REQ_MISSING_EXPOSURE",
            findings=causal_findings(),
        )

        self.assertEqual(result.attempt_count, 1)
        self.assertEqual(result.validation_errors, ())
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(
            result.candidates[0].supporting_evidence_ids,
            ("EV_PROCESS", "EV_PRODUCT"),
        )

    def test_generator_does_not_replace_an_incomplete_candidate_with_empty_answer(
        self,
    ) -> None:
        client = CandidateClient(
            [
                {
                    "candidates": [
                        proposal(supporting=["EV_PROCESS", "EV_PRODUCT"])
                    ],
                    "analysis_summary": "The first proposal omitted MES exposure.",
                },
                {
                    "candidates": [],
                    "analysis_summary": (
                        "No complete three-lane causal mechanism is justified."
                    ),
                },
            ]
        )

        result = QwenHypothesisCandidateGenerator(client).generate(
            request_id="REQ_BOUNDED_EMPTY",
            findings=causal_findings(),
        )

        self.assertEqual(result.attempt_count, 1)
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(len(client.requests), 1)

    def test_near_duplicate_second_candidate_is_isolated(self) -> None:
        process = lane_process_evidence(
            evidence_id="EV_PROCESS_LANE_A",
            lane_id="LANE_A",
            recipe="RCP_A",
            parameter="chamber_temperature_range",
        )
        findings = findings_with_process_evidence(process)
        first = proposal(
            supporting=["EV_EXPOSURE", "EV_PROCESS_LANE_A", "EV_PRODUCT"]
        )
        second = dict(first)
        second["root_cause"] = "Temperature drift failure at EQ_01 chamber"
        client = CandidateClient(
            [
                {
                    "candidates": [first, second],
                    "analysis_summary": "Two paraphrases of one mechanism.",
                }
            ]
        )

        result = QwenHypothesisCandidateGenerator(client).generate(
            request_id="REQ_NEAR_DUPLICATE",
            findings=findings,
        )

        self.assertEqual(len(result.candidates), 1)
        self.assertTrue(any("near-duplicate" in item for item in result.validation_errors))
        self.assertEqual(len(result.rejected_candidates), 1)
        rejected = result.rejected_candidates[0]
        self.assertEqual(rejected["rejected_candidate_index"], 1)
        self.assertEqual(rejected["candidate_root_cause"], second["root_cause"])
        self.assertEqual(rejected["lane_id"], "lane_a")
        self.assertEqual(
            rejected["evidence_ids"],
            ["EV_EXPOSURE", "EV_PROCESS_LANE_A", "EV_PRODUCT"],
        )
        self.assertGreaterEqual(rejected["duplicate_score"], 0.75)
        self.assertIn("same_lane_identity", rejected["duplicate_reason"])
        self.assertEqual(rejected["compared_candidate_id"], "candidate_0")

        finding = RCAReasoningAgent(
            llm_client=CandidateClient(
                [
                    {
                        "candidates": [first, second],
                        "analysis_summary": "Two paraphrases of one mechanism.",
                    }
                ]
            ),
            agent_mode="llm",
        ).analyze(request_id="REQ_NEAR_DUPLICATE_AUDIT", findings=findings)
        finding_rejected = finding.details["hypothesis_candidate_generation"][
            "rejected_candidates"
        ]
        self.assertEqual(len(finding_rejected), 1)
        self.assertEqual(finding_rejected[0]["lane_id"], "lane_a")

    def test_same_equipment_operation_with_distinct_recipe_evidence_is_preserved(
        self,
    ) -> None:
        process_a = lane_process_evidence(
            evidence_id="EV_PROCESS_RECIPE_A",
            lane_id="LANE_SHARED",
            recipe="RCP_A",
            parameter="backside_pressure_cv",
        )
        process_b = lane_process_evidence(
            evidence_id="EV_PROCESS_RECIPE_B",
            lane_id="LANE_SHARED",
            recipe="RCP_B",
            parameter="backside_pressure_cv",
        )
        findings = findings_with_process_evidence(process_a, process_b)
        candidate_a = proposal(
            supporting=["EV_EXPOSURE", "EV_PROCESS_RECIPE_A", "EV_PRODUCT"]
        )
        candidate_b = proposal(
            supporting=["EV_EXPOSURE", "EV_PROCESS_RECIPE_B", "EV_PRODUCT"]
        )
        client = CandidateClient(
            [
                {
                    "candidates": [candidate_a, candidate_b],
                    "analysis_summary": "Two recipes remain evidence-bounded.",
                }
            ]
        )

        result = QwenHypothesisCandidateGenerator(client).generate(
            request_id="REQ_DIFFERENT_RECIPE",
            findings=findings,
        )

        self.assertEqual(len(result.candidates), 2)
        self.assertEqual(result.rejected_candidates, ())

    def test_different_lanes_survive_even_with_identical_candidate_text(self) -> None:
        process_a = lane_process_evidence(
            evidence_id="EV_PROCESS_LANE_A",
            lane_id="LANE_A",
            recipe="RCP_SHARED",
            parameter="backside_pressure_cv",
        )
        process_b = lane_process_evidence(
            evidence_id="EV_PROCESS_LANE_B",
            lane_id="LANE_B",
            recipe="RCP_SHARED",
            parameter="backside_pressure_cv",
        )
        findings = findings_with_process_evidence(process_a, process_b)
        candidate_a = proposal(
            supporting=["EV_EXPOSURE", "EV_PROCESS_LANE_A", "EV_PRODUCT"]
        )
        candidate_b = proposal(
            supporting=["EV_EXPOSURE", "EV_PROCESS_LANE_B", "EV_PRODUCT"]
        )

        result = QwenHypothesisCandidateGenerator(
            CandidateClient(
                [
                    {
                        "candidates": [candidate_a, candidate_b],
                        "analysis_summary": "Identical wording refers to two Lanes.",
                    }
                ]
            )
        ).generate(request_id="REQ_DIFFERENT_LANE", findings=findings)

        self.assertEqual(len(result.candidates), 2)
        self.assertEqual(result.rejected_candidates, ())

    def test_formal_002_lane_competition_structure_replay_preserves_candidate_b(
        self,
    ) -> None:
        original_process = lane_process_evidence(
            evidence_id="EV_PROCESS_ORIGINAL_RECIPE",
            lane_id="LANE_ORIGINAL_RECIPE",
            recipe="RCP_A",
            parameter="deposition_rate_delta",
        )
        alternative_process = lane_process_evidence(
            evidence_id="EV_PROCESS_ALTERNATIVE_RECIPE",
            lane_id="LANE_ALTERNATIVE_RECIPE",
            recipe="RCP_B",
            parameter="backside_pressure_cv",
        )
        findings = findings_with_process_evidence(original_process)
        candidate_a = proposal(
            supporting=[
                "EV_EXPOSURE",
                "EV_PROCESS_ORIGINAL_RECIPE",
                "EV_PRODUCT",
            ]
        )
        candidate_b = {
            **candidate_a,
            "root_cause": "Temperature drift failure at EQ_01 chamber",
            "supporting_evidence_ids": [
                "EV_EXPOSURE",
                "EV_PROCESS_ALTERNATIVE_RECIPE",
                "EV_PRODUCT",
            ],
        }
        gap_id = "candidate_0.hypothesis_discrimination.parameter_anomaly"
        client = CandidateClient(
            [
                {
                    "candidates": [candidate_a, candidate_b],
                    "analysis_summary": (
                        "The newly investigated recipe remains a distinct "
                        "evidence-bounded competitor."
                    ),
                }
            ]
        )

        result = QwenHypothesisCandidateGenerator(client).generate(
            request_id="REQ_FORMAL_002_STRUCTURE_REPLAY",
            findings=findings,
            context_evidence=[alternative_process],
            prior_candidates=[candidate_a],
            prior_challenges=[
                {
                    "candidate_id": "prior_candidate_0",
                    "strongest_alternative_lane_id": "LANE_ALTERNATIVE_RECIPE",
                    "distinguishing_gap_ids": [gap_id],
                    "status": "alternative_identified",
                }
            ],
            prior_causal_gaps=[
                {
                    "gap_id": gap_id,
                    "discriminator_kind": "parameter_anomaly",
                    "target_scope": {
                        "lane_id": "LANE_ALTERNATIVE_RECIPE",
                        "operation": "4000",
                    },
                }
            ],
            causal_lanes=[
                {
                    "lane_id": "LANE_ALTERNATIVE_RECIPE",
                    "operation": "4000",
                    "equipment": "EQ_01",
                    "chamber": "EQ_01_CH01",
                    "recipe": "RCP_B",
                    "parameter_scope": ["backside_pressure_cv"],
                    "investigation_status": "evidence_collected",
                }
            ],
            new_evidence_ids=["EV_PROCESS_ALTERNATIVE_RECIPE"],
        )

        self.assertEqual(result.attempt_count, 1)
        self.assertEqual(len(result.candidates), 2)
        self.assertFalse(result.competition_repair_exhausted)
        self.assertEqual(result.rejected_candidates, ())
        self.assertEqual(
            result.targeted_investigation_results[0][
                "new_supporting_evidence_ids"
            ],
            ["EV_PROCESS_ALTERNATIVE_RECIPE"],
        )

    def test_unstated_operation_is_incomplete_not_conflicted(self) -> None:
        findings = causal_findings()
        exposure = Evidence.from_dict(
            {
                **findings[0].evidence[0].to_dict(),
                "entities": [
                    EvidenceEntity(EntityType.LOT.value, "LOT_01").to_dict(),
                    EvidenceEntity(EntityType.EQUIPMENT.value, "EQ_01").to_dict(),
                    EvidenceEntity(EntityType.OPERATION.value, "4000").to_dict(),
                ],
            }
        )
        candidate = CausalHypothesis(
            root_cause="EQ_01 chamber temperature control drift",
            causal_explanation="Temperature drift produces the observed edge void.",
            supporting_evidence_ids=("EV_EXPOSURE", "EV_PROCESS", "EV_PRODUCT"),
        )

        matrix = build_causal_evidence_matrix(
            candidate,
            [exposure, findings[1].evidence[0], findings[2].evidence[0]],
        )

        self.assertEqual(matrix.claims["operation"].status, "incomplete")

    def test_equipment_tokens_do_not_create_parameter_support(self) -> None:
        candidate = CausalHypothesis(
            root_cause="EQ_01 chamber malfunction",
            causal_explanation="The chamber can produce the observed edge void.",
            supporting_evidence_ids=("EV_EXPOSURE", "EV_PROCESS", "EV_PRODUCT"),
        )

        matrix = build_causal_evidence_matrix(
            candidate,
            [item for finding in causal_findings() for item in finding.evidence],
        )

        self.assertEqual(matrix.claims["parameter"].status, "incomplete")

    def test_explicit_wrong_parameter_is_conflicted(self) -> None:
        candidate = CausalHypothesis(
            root_cause="EQ_01 chamber pressure drift",
            causal_explanation="High pressure produces the observed edge void.",
            supporting_evidence_ids=("EV_EXPOSURE", "EV_PROCESS", "EV_PRODUCT"),
        )

        matrix = build_causal_evidence_matrix(
            candidate,
            [item for finding in causal_findings() for item in finding.evidence],
        )

        self.assertEqual(matrix.claims["parameter"].status, "conflicted")

    def test_timestamps_without_a_window_are_temporally_incomplete(self) -> None:
        findings = causal_findings()
        timed = [
            Evidence.from_dict(
                {
                    **item.to_dict(),
                    "timestamp": "2026-01-01T00:30:00",
                    "metadata": {},
                }
            )
            for finding in findings
            for item in finding.evidence
        ]
        matrix = build_causal_evidence_matrix(
            CausalHypothesis.from_mapping(proposal()),
            timed,
        )

        self.assertEqual(matrix.claims["temporal"].status, "incomplete")

    def test_unrelated_approved_knowledge_does_not_support_mechanism(self) -> None:
        unrelated = Evidence.from_dict(
            {
                **approved_knowledge_evidence().to_dict(),
                "evidence_id": "EV_UNRELATED_KNOWLEDGE",
                "summary": "Confirmed slurry flow mechanism for scratch defects.",
                "observation": "Low slurry flow can cause wafer scratches.",
            }
        )
        candidate = CausalHypothesis(
            root_cause="EQ_01 temperature drift",
            causal_explanation="Temperature drift may produce edge voids.",
            supporting_evidence_ids=("EV_UNRELATED_KNOWLEDGE",),
        )

        matrix = build_causal_evidence_matrix(candidate, [unrelated])

        self.assertEqual(matrix.claims["mechanism"].status, "incomplete")
        self.assertEqual(matrix.claims["mechanism"].support_source, "llm_explanation_only")

    def test_python_gate_can_support_a_three_lane_llm_candidate(self) -> None:
        result = HypothesisEngine().analyze(
            request_id="REQ_ENGINE",
            findings=causal_findings(),
            mode="active",
            external_candidates=[proposal()],
        )

        selected = result["candidates"][0]
        self.assertEqual(selected["basis"], "llm_evidence_composition")
        self.assertTrue(selected["llm_gate_passed"])
        self.assertEqual(
            set(selected["causal_lanes"]),
            {"shared_exposure", "process_anomaly", "product_outcome"},
        )
        self.assertEqual(result["decision_gate"]["status"], "supported")
        self.assertEqual(
            result["decision_gate"]["root_cause"],
            "EQ_01 chamber temperature control drift",
        )
        self.assertEqual(
            result["candidate_comparison"]["ranking_source"],
            "legacy_gate_confidence",
        )

    def test_python_gate_rejects_a_candidate_missing_product_lane(self) -> None:
        result = HypothesisEngine().analyze(
            request_id="REQ_MISSING_LANE",
            findings=causal_findings(),
            mode="active",
            external_candidates=[
                proposal(supporting=["EV_EXPOSURE", "EV_PROCESS"])
            ],
        )

        selected = result["candidates"][0]
        self.assertFalse(selected["llm_gate_passed"])
        self.assertEqual(result["decision_gate"]["status"], "inconclusive")
        self.assertTrue(
            any("three" in reason for reason in selected["rejection_reasons"])
        )

    def test_python_gate_outranks_qwen_preference_for_a_conflicted_candidate(
        self,
    ) -> None:
        supported = proposal()
        conflicted = {
            **proposal(),
            "root_cause": "EQ_02 chamber temperature control drift",
        }

        result = HypothesisEngine().analyze(
            request_id="REQ_GATE_BEFORE_QWEN_PREFERENCE",
            findings=causal_findings(),
            mode="active",
            external_candidates=[supported, conflicted],
            include_deterministic_candidates=False,
            candidate_comparison={
                "preferred_candidate_index": 1,
                "comparison_summary": "Qwen preferred the conflicted candidate.",
            },
            strict_confirmation=True,
        )

        self.assertEqual(
            result["candidates"][0]["root_cause"],
            supported["root_cause"],
        )
        self.assertTrue(result["candidates"][0]["decision_gate_passed"])
        self.assertFalse(result["candidates"][1]["decision_gate_passed"])
        self.assertEqual(result["decision_gate"]["status"], "supported")
        self.assertEqual(
            result["decision_gate"]["root_cause"],
            supported["root_cause"],
        )
        self.assertFalse(
            result["candidate_comparison"]["comparator_preference_accepted"]
        )
        self.assertEqual(
            result["candidate_comparison"][
                "comparator_preference_rejection_reason"
            ],
            "preferred_candidate_has_critical_conflict",
        )

    def test_python_matrix_promotes_stronger_unconfirmed_candidate(self) -> None:
        weak, strong = comparison_proposals()

        result = HypothesisEngine().analyze(
            request_id="REQ_COMPARISON_PROPAGATION",
            findings=causal_findings(),
            mode="active",
            external_candidates=[weak, strong],
            include_deterministic_candidates=False,
            candidate_comparison={
                "preferred_candidate_index": 1,
                "comparison_summary": "Qwen agrees with the stronger matrix.",
                "source": "qwen",
            },
            strict_confirmation=True,
            alternative_search_status="alternative_found",
        )

        selected = result["candidates"][0]
        comparison = result["candidate_comparison"]
        self.assertEqual(selected["root_cause"], strong["root_cause"])
        self.assertEqual(selected["pre_comparison_rank"], 2)
        self.assertEqual(selected["final_rank"], 1)
        self.assertGreater(
            selected["python_matrix_score"],
            result["candidates"][1]["python_matrix_score"],
        )
        self.assertTrue(comparison["comparator_preference_accepted"])
        self.assertEqual(
            comparison["ranking_source"],
            "python_matrix_qwen_consistent",
        )
        self.assertEqual(result["decision_gate"]["status"], "inconclusive")

    def test_python_matrix_rejects_qwen_preference_for_weaker_candidate(self) -> None:
        weak, strong = comparison_proposals()

        result = HypothesisEngine().analyze(
            request_id="REQ_WEAKER_QWEN_PREFERENCE",
            findings=causal_findings(),
            mode="active",
            external_candidates=[weak, strong],
            include_deterministic_candidates=False,
            candidate_comparison={
                "preferred_candidate_index": 0,
                "comparison_summary": "Qwen preferred the weaker candidate.",
                "source": "qwen",
            },
            strict_confirmation=True,
            alternative_search_status="alternative_found",
        )

        comparison = result["candidate_comparison"]
        self.assertEqual(result["candidates"][0]["root_cause"], strong["root_cause"])
        self.assertFalse(comparison["comparator_preference_accepted"])
        self.assertEqual(
            comparison["comparator_preference_rejection_reason"],
            "preferred_candidate_has_lower_python_matrix_score",
        )
        self.assertEqual(comparison["ranking_source"], "python_matrix_qwen_rejected")
        self.assertEqual(result["decision_gate"]["status"], "inconclusive")

    def test_equal_matrix_without_preference_keeps_stable_order(self) -> None:
        first = proposal()
        second = {
            **proposal(),
            "root_cause": "EQ_01 chamber temperature control drift Z",
        }

        result = HypothesisEngine().analyze(
            request_id="REQ_STABLE_COMPARISON_TIE",
            findings=causal_findings(),
            mode="active",
            external_candidates=[second, first],
            include_deterministic_candidates=False,
            candidate_comparison={
                "preferred_candidate_index": None,
                "comparison_summary": "The matrices cannot be separated.",
                "source": "qwen",
            },
            strict_confirmation=True,
            alternative_search_status="alternative_found",
        )

        self.assertEqual(result["candidates"][0]["root_cause"], first["root_cause"])
        self.assertEqual(
            result["candidate_comparison"]["ranking_source"],
            "python_matrix",
        )
        self.assertIsNone(
            result["candidate_comparison"]["comparator_preference_accepted"]
        )

    def test_unknown_evidence_candidate_cannot_be_promoted_by_qwen(self) -> None:
        unsafe = {
            **proposal(),
            "root_cause": "EQ_01 chamber temperature control drift A",
            "supporting_evidence_ids": [
                "EV_EXPOSURE",
                "EV_PROCESS",
                "EV_PRODUCT",
                "EV_UNKNOWN",
            ],
        }
        safe = {
            **proposal(),
            "root_cause": "EQ_01 chamber temperature control drift B",
        }

        result = HypothesisEngine().analyze(
            request_id="REQ_UNKNOWN_EVIDENCE_PREFERENCE",
            findings=causal_findings(),
            mode="active",
            external_candidates=[unsafe, safe],
            include_deterministic_candidates=False,
            candidate_comparison={
                "preferred_candidate_index": 0,
                "comparison_summary": "Qwen cited an unknown Evidence ID.",
                "source": "qwen",
            },
            strict_confirmation=True,
            alternative_search_status="alternative_found",
        )

        comparison = result["candidate_comparison"]
        self.assertEqual(result["candidates"][0]["root_cause"], safe["root_cause"])
        self.assertFalse(comparison["comparator_preference_accepted"])
        self.assertEqual(
            comparison["comparator_preference_rejection_reason"],
            "preferred_candidate_has_unknown_evidence",
        )
        self.assertEqual(result["decision_gate"]["status"], "inconclusive")

    def test_unrelated_normal_parameter_is_not_a_causal_contradiction(self) -> None:
        findings = causal_findings()
        unrelated_normal = typed_evidence(
            evidence_id="EV_UNRELATED_NORMAL",
            evidence_type=EvidenceType.NEGATIVE_SIGNAL.value,
            agent=AgentKind.FDC.value,
            entity_type=EntityType.PARAMETER.value,
            entity_id="unrelated_gas_flow",
        )
        findings[1] = AgentFinding(
            finding_id="FINDING_FDC_WITH_UNRELATED_CONTROL",
            agent=AgentKind.FDC.value,
            summary="FDC includes the anomaly and an unrelated normal control.",
            confidence=0.9,
            evidence_ids=["EV_PROCESS", "EV_UNRELATED_NORMAL"],
            evidence=[findings[1].evidence[0], unrelated_normal],
        )

        result = HypothesisEngine().analyze(
            request_id="REQ_UNRELATED_NORMAL",
            findings=findings,
            mode="active",
            external_candidates=[proposal()],
        )

        selected = result["candidates"][0]
        self.assertTrue(selected["llm_gate_passed"])
        self.assertNotIn(
            "EV_UNRELATED_NORMAL",
            selected["contradicting_evidence_ids"],
        )
        self.assertEqual(result["decision_gate"]["status"], "supported")

    def test_rca_agent_uses_qwen_proposal_but_python_gate_owns_result(self) -> None:
        client = CandidateClient(
            [
                {
                    "candidates": [proposal()],
                    "analysis_summary": "One evidence-bounded mechanism.",
                }
            ]
        )

        result = RCAReasoningAgent(
            llm_client=client,
            agent_mode="llm",
        ).analyze(
            request_id="REQ_RCA_AGENT",
            findings=causal_findings(),
        )

        generation = result.details["hypothesis_candidate_generation"]
        self.assertEqual(generation["source"], "qwen")
        self.assertEqual(generation["candidate_count"], 1)
        self.assertEqual(result.details["status"], "supported")
        self.assertEqual(
            result.details["hypothesis_engine_result"]["candidates"][0]["basis"],
            "llm_evidence_composition",
        )
        self.assertEqual(len(client.requests), 1)

    def test_rca_impact_scope_recovers_candidate_lots_from_typed_evidence(self) -> None:
        findings = causal_findings()
        original_exposure = findings[0].evidence[0]
        typed_impact_scope = Evidence.from_dict(
            {
                **original_exposure.to_dict(),
                "evidence_id": "EV_TYPED_IMPACT_SCOPE",
                "entities": [
                    EvidenceEntity(EntityType.LOT.value, "LOT_01").to_dict(),
                    EvidenceEntity(EntityType.LOT.value, "LOT_IMPACT").to_dict(),
                    EvidenceEntity(EntityType.EQUIPMENT.value, "EQ_01").to_dict(),
                ],
                "metadata": {"impact_lots": ["LOT_IMPACT"]},
            }
        )
        findings[0] = AgentFinding(
            finding_id=findings[0].finding_id,
            agent=findings[0].agent,
            summary=findings[0].summary,
            confidence=findings[0].confidence,
            evidence_ids=[original_exposure.evidence_id, typed_impact_scope.evidence_id],
            evidence=[original_exposure, typed_impact_scope],
            details={"source_lot_id": "LOT_01"},
        )
        client = CandidateClient(
            [
                {
                    "candidates": [proposal()],
                    "analysis_summary": "One evidence-bounded mechanism.",
                }
            ]
        )

        result = RCAReasoningAgent(
            llm_client=client,
            agent_mode="llm",
        ).analyze(
            request_id="REQ_TYPED_IMPACT_SCOPE",
            findings=findings,
        )

        impact_gate = result.details["impact_lot_gate"]
        self.assertEqual(impact_gate["observed_impact_lots"], ["LOT_IMPACT"])
        self.assertEqual([row["lot_id"] for row in impact_gate["rows"]], ["LOT_IMPACT"])
        self.assertEqual(len(impact_gate["candidate_scopes"]), 1)

    def test_rca_reasoning_refresh_receives_prior_candidate_and_new_evidence(self) -> None:
        first_client = CandidateClient(
            [
                {
                    "candidates": [proposal()],
                    "analysis_summary": "First evidence-bounded mechanism.",
                }
            ]
        )
        agent = RCAReasoningAgent(llm_client=first_client, agent_mode="llm")
        findings = causal_findings()
        first = agent.analyze(request_id="REQ_RCA_ROUND_1", findings=findings)
        new_product = typed_evidence(
            evidence_id="EV_PRODUCT_NEW",
            evidence_type=EvidenceType.METROLOGY_DEVIATION.value,
            agent=AgentKind.DEFECT_WAT.value,
            entity_type=EntityType.DEFECT.value,
            entity_id="edge_void",
        )
        second_client = CandidateClient(
            [
                {
                    "candidates": [proposal()],
                    "analysis_summary": "Prior mechanism rechecked with new Evidence.",
                }
            ]
        )

        second = RCAReasoningAgent(
            llm_client=second_client,
            agent_mode="llm",
        ).analyze(
            request_id="REQ_RCA_ROUND_2",
            findings=findings,
            context_evidence=[new_product],
            prior_rca_finding=first,
        )

        self.assertEqual(second.details["reasoning_round"], 2)
        self.assertEqual(
            second.details["prior_authoritative_rca_finding_id"],
            first.finding_id,
        )
        self.assertEqual(
            second.details["new_evidence_ids_since_prior"],
            ["EV_PRODUCT_NEW"],
        )
        self.assertEqual(
            second.details["hypothesis_candidate_generation"][
                "prior_candidate_count"
            ],
            1,
        )
        prior_payload = second_client.requests[0].payload[
            "prior_authoritative_candidates"
        ]
        self.assertEqual(len(prior_payload), 1)
        self.assertEqual(
            prior_payload[0]["root_cause"],
            first.details["ranked_candidates"][0]["root_cause"],
        )
        self.assertEqual(
            second_client.requests[0].payload["new_evidence_ids_since_prior"],
            ["EV_PRODUCT_NEW"],
        )
        self.assertEqual(
            second_client.requests[0].payload["prior_candidate_challenges"][0][
                "status"
            ],
            "resolved",
        )
        self.assertEqual(
            second_client.requests[0].payload["targeted_investigation_results"],
            [],
        )
        mechanism_feedback = second_client.requests[0].payload[
            "prior_candidate_mechanism_feedback"
        ]
        self.assertEqual(len(mechanism_feedback), 1)
        self.assertEqual(mechanism_feedback[0]["mechanism_status"], "supported")
        self.assertEqual(
            mechanism_feedback[0]["mechanism_support_source"],
            "empirical_discrimination",
        )
        self.assertTrue(mechanism_feedback[0]["proposed_physical_bridge_terms"])
        self.assertNotEqual(second.finding_id, first.finding_id)

        third_product = typed_evidence(
            evidence_id="EV_PRODUCT_ROUND_3",
            evidence_type=EvidenceType.METROLOGY_DEVIATION.value,
            agent=AgentKind.DEFECT_WAT.value,
            entity_type=EntityType.DEFECT.value,
            entity_id="edge_void",
        )
        third_client = CandidateClient(
            [
                {
                    "candidates": [proposal()],
                    "analysis_summary": "Third bounded Evidence refresh.",
                }
            ]
        )
        third = RCAReasoningAgent(
            llm_client=third_client,
            agent_mode="llm",
        ).analyze(
            request_id="REQ_RCA_ROUND_3",
            findings=findings,
            context_evidence=[new_product, third_product],
            prior_rca_finding=second,
        )

        self.assertEqual(third.details["reasoning_round"], 3)
        self.assertEqual(
            third.details["prior_authoritative_rca_finding_id"],
            second.finding_id,
        )
        self.assertEqual(
            third.details["new_evidence_ids_since_prior"],
            ["EV_PRODUCT_ROUND_3"],
        )

    def test_reasoning_refresh_exposes_incomplete_prior_mechanism_feedback(self) -> None:
        weak_prior = {
            **proposal(),
            "causal_explanation": (
                "Temperature deviation produces the observed edge void."
            ),
        }
        client = CandidateClient(
            [{"candidates": [], "analysis_summary": "No repaired mechanism."}]
        )

        QwenHypothesisCandidateGenerator(client).generate(
            request_id="REQ_MECHANISM_FEEDBACK",
            findings=causal_findings(),
            prior_candidates=[weak_prior],
        )

        feedback = client.requests[0].payload[
            "prior_candidate_mechanism_feedback"
        ]
        self.assertEqual(len(feedback), 1)
        self.assertEqual(feedback[0]["mechanism_status"], "incomplete")
        self.assertEqual(
            feedback[0]["mechanism_support_source"],
            "llm_explanation_only",
        )
        self.assertEqual(feedback[0]["proposed_physical_bridge_terms"], [])

    def test_two_evidence_bounded_directions_force_candidate_repair(self) -> None:
        lane_a = "lane:2500:EQ_A:EQ_A_CH01:RCP_A"
        lane_b = "lane:6000:EQ_B:EQ_B_CH02:RCP_B"
        exposure_a = typed_evidence(
            evidence_id="EV_EXP_DIRECTION_A",
            evidence_type=EvidenceType.PROCESS_EXPOSURE.value,
            agent=AgentKind.MES.value,
            entity_type=EntityType.EQUIPMENT.value,
            entity_id="EQ_A",
            metadata={"lane_id": lane_a, "recipe": "RCP_A"},
        )
        exposure_b = typed_evidence(
            evidence_id="EV_EXP_DIRECTION_B",
            evidence_type=EvidenceType.PROCESS_EXPOSURE.value,
            agent=AgentKind.MES.value,
            entity_type=EntityType.EQUIPMENT.value,
            entity_id="EQ_B",
            metadata={"lane_id": lane_b, "recipe": "RCP_B"},
        )
        process_a = lane_process_evidence(
            evidence_id="EV_PROC_DIRECTION_A",
            lane_id=lane_a,
            recipe="RCP_A",
            parameter="oxygen_flow",
            equipment="EQ_A",
            chamber="EQ_A_CH01",
            operation="2500",
        )
        process_b = lane_process_evidence(
            evidence_id="EV_PROC_DIRECTION_B",
            lane_id=lane_b,
            recipe="RCP_B",
            parameter="clean_time",
            equipment="EQ_B",
            chamber="EQ_B_CH02",
            operation="6000",
        )
        candidate_a = {
            "root_cause": "EQ_A plasma oxygen-flow instability",
            "causal_explanation": (
                "Oxygen-flow instability changes plasma reaction kinetics and "
                "produces the observed edge void."
            ),
            "supporting_evidence_ids": [
                "EV_EXP_DIRECTION_A",
                "EV_PROC_DIRECTION_A",
                "EV_PRODUCT",
            ],
            "contradicting_evidence_ids": [],
        }
        response = {
            "candidates": [candidate_a],
            "analysis_summary": "Only the first direction was proposed.",
        }
        client = CandidateClient([response, response])

        result = QwenHypothesisCandidateGenerator(client).generate(
            request_id="REQ_DIRECTION_REQUIRED",
            findings=causal_findings(),
            context_evidence=[exposure_a, exposure_b, process_a, process_b],
            causal_lanes=[
                {
                    "lane_id": lane_a,
                    "operation": "2500",
                    "equipment": "EQ_A",
                    "chamber": "EQ_A_CH01",
                    "recipe": "RCP_A",
                    "parameter_scope": ["oxygen_flow"],
                    "priority_score": 0.9,
                    "investigation_status": "evidence_collected",
                },
                {
                    "lane_id": lane_b,
                    "operation": "6000",
                    "equipment": "EQ_B",
                    "chamber": "EQ_B_CH02",
                    "recipe": "RCP_B",
                    "parameter_scope": ["clean_time"],
                    "priority_score": 0.8,
                    "investigation_status": "evidence_collected",
                },
            ],
        )

        self.assertEqual(result.attempt_count, 2)
        self.assertEqual(
            result.competition_requirement,
            CompetitionRequirement.DIRECTION_REQUIRED.value,
        )
        self.assertEqual(
            result.competition_status,
            CandidateCompetitionStatus.PENDING.value,
        )
        self.assertIsNone(result.competition_failure_reason)
        self.assertEqual(
            result.competition_gap_reason,
            CompetitionGapReason.ALTERNATIVE_DIRECTION_NOT_GENERATED.value,
        )
        self.assertTrue(result.competition_repair_exhausted)
        feedback = client.requests[1].payload["previous_validation_feedback"]
        self.assertIn("alternative_direction_not_generated", feedback["message"])

    def test_scope_competition_preserves_recipe_and_chamber_hypotheses(self) -> None:
        lane_a = "lane:2500:EQ_A:EQ_A_CH01:RCP_A"
        lane_b = "lane:2500:EQ_A:EQ_A_CH01:RCP_B"
        exposure_a = typed_evidence(
            evidence_id="EV_EXP_SCOPE_A",
            evidence_type=EvidenceType.PROCESS_EXPOSURE.value,
            agent=AgentKind.MES.value,
            entity_type=EntityType.EQUIPMENT.value,
            entity_id="EQ_A",
            metadata={"lane_id": lane_a, "recipe": "RCP_A"},
        )
        exposure_b = typed_evidence(
            evidence_id="EV_EXP_SCOPE_B",
            evidence_type=EvidenceType.PROCESS_EXPOSURE.value,
            agent=AgentKind.MES.value,
            entity_type=EntityType.EQUIPMENT.value,
            entity_id="EQ_A",
            metadata={"lane_id": lane_b, "recipe": "RCP_B"},
        )
        process_a = lane_process_evidence(
            evidence_id="EV_PROC_SCOPE_A",
            lane_id=lane_a,
            recipe="RCP_A",
            parameter="oxygen_flow",
            equipment="EQ_A",
            chamber="EQ_A_CH01",
            operation="2500",
        )
        process_b = lane_process_evidence(
            evidence_id="EV_PROC_SCOPE_B",
            lane_id=lane_b,
            recipe="RCP_B",
            parameter="oxygen_flow",
            equipment="EQ_A",
            chamber="EQ_A_CH01",
            operation="2500",
        )
        recipe_candidate = {
            "root_cause": "RCP_A-specific oxygen-flow control instability",
            "causal_explanation": (
                "The anomaly is bounded to recipe RCP_A and changes plasma "
                "reaction kinetics, producing the observed edge void."
            ),
            "supporting_evidence_ids": [
                "EV_EXP_SCOPE_A",
                "EV_PROC_SCOPE_A",
                "EV_PRODUCT",
            ],
            "contradicting_evidence_ids": [],
        }
        chamber_candidate = {
            "root_cause": "EQ_A_CH01 chamber-wide oxygen-flow instability",
            "causal_explanation": (
                "The shared anomaly spans recipes RCP_A and RCP_B, predicting a "
                "chamber-wide plasma-control scope for the observed edge void."
            ),
            "supporting_evidence_ids": [
                "EV_EXP_SCOPE_A",
                "EV_PROC_SCOPE_A",
                "EV_EXP_SCOPE_B",
                "EV_PROC_SCOPE_B",
                "EV_PRODUCT",
            ],
            "contradicting_evidence_ids": [],
        }
        client = CandidateClient(
            [
                {
                    "candidates": [recipe_candidate, chamber_candidate],
                    "candidate_semantic_profiles": [
                        semantic_profile(
                            0,
                            scope_relation="focal_only",
                            claimed_lane_ids=[lane_a],
                            comparison_lane_ids=[lane_a, lane_b],
                            prediction="Only RCP_A produces the product outcome.",
                        ),
                        semantic_profile(
                            1,
                            scope_relation="shared_effect",
                            claimed_lane_ids=[lane_a, lane_b],
                            comparison_lane_ids=[lane_a, lane_b],
                            prediction="Both recipes produce compatible outcomes.",
                        ),
                    ],
                    "analysis_summary": (
                        "Recipe-specific and chamber-wide scope hypotheses compete."
                    ),
                }
            ]
        )

        result = QwenHypothesisCandidateGenerator(client).generate(
            request_id="REQ_SCOPE_REQUIRED",
            findings=causal_findings(),
            context_evidence=[exposure_a, exposure_b, process_a, process_b],
            causal_lanes=[
                {
                    "lane_id": lane_a,
                    "operation": "2500",
                    "equipment": "EQ_A",
                    "chamber": "EQ_A_CH01",
                    "recipe": "RCP_A",
                    "parameter_scope": ["oxygen_flow"],
                    "priority_score": 0.9,
                    "investigation_status": "evidence_collected",
                },
                {
                    "lane_id": lane_b,
                    "operation": "2500",
                    "equipment": "EQ_A",
                    "chamber": "EQ_A_CH01",
                    "recipe": "RCP_B",
                    "parameter_scope": ["oxygen_flow"],
                    "priority_score": 0.8,
                    "investigation_status": "evidence_collected",
                },
            ],
        )

        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(
            result.competition_requirement,
            CompetitionRequirement.MECHANISM_REQUIRED.value,
        )
        self.assertEqual(
            result.competition_status,
            CandidateCompetitionStatus.PENDING.value,
        )
        self.assertIsNone(result.competition_failure_reason)
        self.assertFalse(
            result.competition_assessment["scope_competition_represented"]
        )
        self.assertFalse(
            result.competition_assessment["mechanism_competition_represented"]
        )
        self.assertEqual(
            result.competition_gap_reason,
            CompetitionGapReason.MECHANISM_ALTERNATIVE_NOT_GENERATED.value,
        )
        variants = result.competition_assessment["scope_assessment_variants"]
        self.assertEqual(len(variants), 1)
        self.assertEqual(variants[0]["semantic_profile_status"], "valid_non_root")
        self.assertEqual(
            variants[0]["semantic_profile"]["scope_relation"],
            "shared_effect",
        )

    def test_incomplete_recipe_comparison_still_requires_scope_candidates(
        self,
    ) -> None:
        lane_a = "lane:2500:EQ_A:EQ_A_CH01:RCP_A"
        lane_b = "lane:2500:EQ_A:EQ_A_CH01:RCP_B"
        exposure_a = typed_evidence(
            evidence_id="EV_SCOPE_OPPORTUNITY_EXP_A",
            evidence_type=EvidenceType.PROCESS_EXPOSURE.value,
            agent=AgentKind.MES.value,
            entity_type=EntityType.EQUIPMENT.value,
            entity_id="EQ_A",
            metadata={"lane_id": lane_a, "recipe": "RCP_A"},
        )
        exposure_b = typed_evidence(
            evidence_id="EV_SCOPE_OPPORTUNITY_EXP_B",
            evidence_type=EvidenceType.PROCESS_EXPOSURE.value,
            agent=AgentKind.MES.value,
            entity_type=EntityType.EQUIPMENT.value,
            entity_id="EQ_A",
            metadata={"lane_id": lane_b, "recipe": "RCP_B"},
        )
        process_a = lane_process_evidence(
            evidence_id="EV_SCOPE_OPPORTUNITY_PROC_A",
            lane_id=lane_a,
            recipe="RCP_A",
            parameter="oxygen_flow",
            equipment="EQ_A",
            chamber="EQ_A_CH01",
            operation="2500",
        )
        focal = {
            **proposal(
                supporting=[
                    exposure_a.evidence_id,
                    process_a.evidence_id,
                    "EV_PRODUCT",
                ]
            ),
            "root_cause": "RCP_A-specific oxygen-flow sensitivity",
        }
        shared = {
            **proposal(
                supporting=[
                    exposure_a.evidence_id,
                    exposure_b.evidence_id,
                    process_a.evidence_id,
                    "EV_PRODUCT",
                ]
            ),
            "root_cause": "EQ_A_CH01 chamber-wide oxygen-flow instability",
        }
        response = {
            "candidates": [focal, shared],
            "candidate_semantic_profiles": [
                semantic_profile(
                    0,
                    scope_relation="focal_only",
                    claimed_lane_ids=[lane_a],
                    comparison_lane_ids=[lane_a, lane_b],
                    prediction=(
                        "Only RCP_A shows the parameter anomaly and product response."
                    ),
                    discriminator_kind="parameter_anomaly",
                ),
                semantic_profile(
                    1,
                    scope_relation="shared_effect",
                    claimed_lane_ids=[lane_a, lane_b],
                    comparison_lane_ids=[lane_a, lane_b],
                    prediction=(
                        "Both recipes show a compatible parameter anomaly and response."
                    ),
                    discriminator_kind="parameter_anomaly",
                ),
            ],
            "analysis_summary": "Focal and shared Scope hypotheses compete.",
        }
        client = CandidateClient([response])

        result = QwenHypothesisCandidateGenerator(client).generate(
            request_id="REQ_SCOPE_OPPORTUNITY",
            findings=causal_findings(),
            context_evidence=[exposure_a, exposure_b, process_a],
            causal_lanes=[
                {
                    "lane_id": lane_a,
                    "operation": "2500",
                    "equipment": "EQ_A",
                    "chamber": "EQ_A_CH01",
                    "recipe": "RCP_A",
                    "parameter_scope": ["oxygen_flow"],
                    "priority_score": 0.9,
                    "investigation_status": "evidence_collected",
                },
                {
                    "lane_id": lane_b,
                    "operation": "2500",
                    "equipment": "EQ_A",
                    "chamber": "EQ_A_CH01",
                    "recipe": "RCP_B",
                    "parameter_scope": ["oxygen_flow"],
                    "priority_score": 0.8,
                    "investigation_status": "evidence_collected",
                },
            ],
        )

        self.assertEqual(result.attempt_count, 2)
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(
            result.competition_requirement,
            CompetitionRequirement.MECHANISM_REQUIRED.value,
        )
        self.assertEqual(
            result.competition_status,
            CandidateCompetitionStatus.PENDING.value,
        )
        competition = result.evidence_synthesis["candidate_competition"]
        scope_group = competition["scope_groups"][0]
        self.assertTrue(scope_group["scope_competition_opportunity"])
        self.assertFalse(scope_group["scope_evidence_complete"])
        self.assertEqual(scope_group["missing_process_recipe_ids"], ["RCP_B"])
        variants = result.competition_assessment["scope_assessment_variants"]
        self.assertEqual(len(variants), 1)
        self.assertEqual(variants[0]["semantic_profile_status"], "valid_non_root")

    def test_formal_009_scope_semantics_replay_does_not_infer_claim_from_coverage(
        self,
    ) -> None:
        lane_a = "lane:2500:EQ_A:EQ_A_CH01:RCP_A"
        lane_b = "lane:2500:EQ_A:EQ_A_CH01:RCP_B"
        exposure_a = typed_evidence(
            evidence_id="EV_SCOPE_COVERAGE_EXP_A",
            evidence_type=EvidenceType.PROCESS_EXPOSURE.value,
            agent=AgentKind.MES.value,
            entity_type=EntityType.EQUIPMENT.value,
            entity_id="EQ_A",
            metadata={"lane_id": lane_a, "recipe": "RCP_A"},
        )
        exposure_b = typed_evidence(
            evidence_id="EV_SCOPE_COVERAGE_EXP_B",
            evidence_type=EvidenceType.PROCESS_EXPOSURE.value,
            agent=AgentKind.MES.value,
            entity_type=EntityType.EQUIPMENT.value,
            entity_id="EQ_A",
            metadata={"lane_id": lane_b, "recipe": "RCP_B"},
        )
        process_a = lane_process_evidence(
            evidence_id="EV_SCOPE_COVERAGE_PROC_A",
            lane_id=lane_a,
            recipe="RCP_A",
            parameter="oxygen_flow",
            equipment="EQ_A",
            chamber="EQ_A_CH01",
            operation="2500",
        )
        process_b = lane_process_evidence(
            evidence_id="EV_SCOPE_COVERAGE_PROC_B",
            lane_id=lane_b,
            recipe="RCP_B",
            parameter="oxygen_flow",
            equipment="EQ_A",
            chamber="EQ_A_CH01",
            operation="2500",
        )
        shared_support = [
            "EV_SCOPE_COVERAGE_EXP_A",
            "EV_SCOPE_COVERAGE_PROC_A",
            "EV_SCOPE_COVERAGE_EXP_B",
            "EV_SCOPE_COVERAGE_PROC_B",
            "EV_PRODUCT",
        ]
        chamber_wide = {
            "root_cause": "Chamber-wide plasma-control drift affects both recipes",
            "causal_explanation": (
                "A shared chamber condition changes plasma kinetics in both "
                "recipes and produces compatible product outcomes."
            ),
            "supporting_evidence_ids": shared_support,
            "contradicting_evidence_ids": [],
        }
        recipe_sensitive = {
            "root_cause": "RCP_A amplifies sensitivity to the chamber drift",
            "causal_explanation": (
                "Both recipes are compared, but RCP_A is predicted to amplify "
                "the same chamber condition into a more severe outcome."
            ),
            "supporting_evidence_ids": shared_support,
            "contradicting_evidence_ids": [],
        }
        client = CandidateClient(
            [
                {
                    "candidates": [chamber_wide, recipe_sensitive],
                    "candidate_semantic_profiles": [
                        semantic_profile(
                            0,
                            scope_relation="shared_effect",
                            claimed_lane_ids=[lane_a, lane_b],
                            comparison_lane_ids=[lane_a, lane_b],
                            prediction="Both recipe Lanes show compatible outcomes.",
                        ),
                        semantic_profile(
                            1,
                            scope_relation="differential_sensitivity",
                            claimed_lane_ids=[lane_a],
                            comparison_lane_ids=[lane_a, lane_b],
                            prediction="RCP_A shows a materially more severe outcome.",
                        ),
                    ],
                    "analysis_summary": "Shared-effect and sensitivity scopes compete.",
                }
            ]
        )

        result = QwenHypothesisCandidateGenerator(client).generate(
            request_id="REQ_SCOPE_COVERAGE_NE_CLAIM",
            findings=causal_findings(),
            context_evidence=[exposure_a, exposure_b, process_a, process_b],
            causal_lanes=[
                {
                    "lane_id": lane_a,
                    "operation": "2500",
                    "equipment": "EQ_A",
                    "chamber": "EQ_A_CH01",
                    "recipe": "RCP_A",
                    "parameter_scope": ["oxygen_flow"],
                    "priority_score": 0.9,
                    "investigation_status": "evidence_collected",
                },
                {
                    "lane_id": lane_b,
                    "operation": "2500",
                    "equipment": "EQ_A",
                    "chamber": "EQ_A_CH01",
                    "recipe": "RCP_B",
                    "parameter_scope": ["oxygen_flow"],
                    "priority_score": 0.8,
                    "investigation_status": "evidence_collected",
                },
            ],
        )

        self.assertEqual(result.attempt_count, 2)
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(
            result.competition_status,
            CandidateCompetitionStatus.PENDING.value,
        )
        profiles = result.competition_assessment["candidate_profiles"]
        self.assertTrue(
            all("consumed_discriminator_gap_ids" in profile for profile in profiles)
        )
        self.assertTrue(
            all("discriminator_gap_ids" not in profile for profile in profiles)
        )
        self.assertEqual(
            {profile["scope_relation"] for profile in profiles},
            {"shared_effect"},
        )
        self.assertTrue(
            all(
                set(profile["evidence_coverage"]["recipes"])
                == {"rcp_a", "rcp_b"}
                for profile in profiles
            )
        )
        variants = result.competition_assessment["scope_assessment_variants"]
        self.assertEqual(len(variants), 1)
        sensitivity = variants[0]["semantic_profile"]
        self.assertEqual(
            sensitivity["claimed_lane_ids"],
            [lane_a],
        )
        self.assertEqual(
            set(sensitivity["comparison_lane_ids"]),
            {lane_a, lane_b},
        )

    def test_invalid_scope_semantics_keeps_candidates_pending_without_repair(self) -> None:
        lane_a = "lane:2500:EQ_A:EQ_A_CH01:RCP_A"
        lane_b = "lane:2500:EQ_A:EQ_A_CH01:RCP_B"
        exposure_a = typed_evidence(
            evidence_id="EV_PENDING_EXP_A",
            evidence_type=EvidenceType.PROCESS_EXPOSURE.value,
            agent=AgentKind.MES.value,
            entity_type=EntityType.EQUIPMENT.value,
            entity_id="EQ_A",
            metadata={"lane_id": lane_a, "recipe": "RCP_A"},
        )
        exposure_b = typed_evidence(
            evidence_id="EV_PENDING_EXP_B",
            evidence_type=EvidenceType.PROCESS_EXPOSURE.value,
            agent=AgentKind.MES.value,
            entity_type=EntityType.EQUIPMENT.value,
            entity_id="EQ_A",
            metadata={"lane_id": lane_b, "recipe": "RCP_B"},
        )
        process_a = lane_process_evidence(
            evidence_id="EV_PENDING_PROC_A",
            lane_id=lane_a,
            recipe="RCP_A",
            parameter="oxygen_flow",
            equipment="EQ_A",
            chamber="EQ_A_CH01",
            operation="2500",
        )
        process_b = lane_process_evidence(
            evidence_id="EV_PENDING_PROC_B",
            lane_id=lane_b,
            recipe="RCP_B",
            parameter="oxygen_flow",
            equipment="EQ_A",
            chamber="EQ_A_CH01",
            operation="2500",
        )
        support_a = ["EV_PENDING_EXP_A", "EV_PENDING_PROC_A", "EV_PRODUCT"]
        support_b = ["EV_PENDING_EXP_B", "EV_PENDING_PROC_B", "EV_PRODUCT"]
        client = CandidateClient(
            [
                {
                    "candidates": [
                        {
                            **proposal(supporting=support_a),
                            "root_cause": "RCP_A plasma response instability",
                        },
                        {
                            **proposal(supporting=support_b),
                            "root_cause": "RCP_B endpoint response instability",
                        },
                    ],
                    "candidate_semantic_profiles": [
                        semantic_profile(
                            0,
                            scope_relation="focal_only",
                            claimed_lane_ids=["LANE_UNKNOWN"],
                            comparison_lane_ids=["LANE_UNKNOWN"],
                            prediction="The unknown Lane produces the outcome.",
                        )
                    ],
                    "analysis_summary": "Two valid Candidates lack scope metadata.",
                }
            ]
        )

        result = QwenHypothesisCandidateGenerator(client).generate(
            request_id="REQ_SCOPE_PROFILE_PENDING",
            findings=causal_findings(),
            context_evidence=[exposure_a, exposure_b, process_a, process_b],
            causal_lanes=[
                {
                    "lane_id": lane_a,
                    "operation": "2500",
                    "equipment": "EQ_A",
                    "chamber": "EQ_A_CH01",
                    "recipe": "RCP_A",
                    "parameter_scope": ["oxygen_flow"],
                    "priority_score": 0.9,
                    "investigation_status": "evidence_collected",
                },
                {
                    "lane_id": lane_b,
                    "operation": "2500",
                    "equipment": "EQ_A",
                    "chamber": "EQ_A_CH01",
                    "recipe": "RCP_B",
                    "parameter_scope": ["oxygen_flow"],
                    "priority_score": 0.8,
                    "investigation_status": "evidence_collected",
                },
            ],
        )

        self.assertEqual(len(client.requests), 2)
        self.assertEqual(result.attempt_count, 2)
        self.assertEqual(len(result.candidates), 1)
        self.assertFalse(result.candidate_output_invalid)
        self.assertEqual(
            result.competition_status,
            CandidateCompetitionStatus.PENDING.value,
        )
        self.assertIsNone(result.competition_failure_reason)
        self.assertTrue(result.competition_repair_exhausted)
        self.assertTrue(result.semantic_validation_errors)
        self.assertTrue(
            any(
                "unknown Lane IDs" in error
                for error in result.semantic_validation_errors
            )
        )
        variants = result.competition_assessment["scope_assessment_variants"]
        self.assertEqual(len(variants), 1)
        self.assertEqual(variants[0]["candidate_index"], 1)
        self.assertEqual(variants[0]["semantic_profile_status"], "missing")
        self.assertEqual(variants[0]["mechanism_relation"], "unknown")
        self.assertEqual(
            variants[0]["supporting_evidence_ids"],
            support_b,
        )

    def test_invalid_shared_scope_profile_does_not_fail_valid_candidate(self) -> None:
        lane_a = "lane:2500:EQ_A:EQ_A_CH01:RCP_A"
        lane_b = "lane:2500:EQ_A:EQ_A_CH01:RCP_B"
        exposure_a = typed_evidence(
            evidence_id="EV_SHARED_INVALID_EXP_A",
            evidence_type=EvidenceType.PROCESS_EXPOSURE.value,
            agent=AgentKind.MES.value,
            entity_type=EntityType.EQUIPMENT.value,
            entity_id="EQ_A",
            metadata={"lane_id": lane_a, "recipe": "RCP_A"},
        )
        exposure_b = typed_evidence(
            evidence_id="EV_SHARED_INVALID_EXP_B",
            evidence_type=EvidenceType.PROCESS_EXPOSURE.value,
            agent=AgentKind.MES.value,
            entity_type=EntityType.EQUIPMENT.value,
            entity_id="EQ_A",
            metadata={"lane_id": lane_b, "recipe": "RCP_B"},
        )
        process_a = lane_process_evidence(
            evidence_id="EV_SHARED_INVALID_PROC_A",
            lane_id=lane_a,
            recipe="RCP_A",
            parameter="oxygen_flow",
            equipment="EQ_A",
            chamber="EQ_A_CH01",
            operation="2500",
        )
        process_b = lane_process_evidence(
            evidence_id="EV_SHARED_INVALID_PROC_B",
            lane_id=lane_b,
            recipe="RCP_B",
            parameter="oxygen_flow",
            equipment="EQ_A",
            chamber="EQ_A_CH01",
            operation="2500",
        )
        response = {
            "candidates": [
                {
                    **proposal(
                        supporting=[
                            "EV_SHARED_INVALID_EXP_A",
                            "EV_SHARED_INVALID_PROC_A",
                            "EV_PRODUCT",
                        ]
                    ),
                    "root_cause": "RCP_A plasma response instability",
                }
            ],
            "candidate_semantic_profiles": [
                semantic_profile(
                    0,
                    scope_relation="shared_effect",
                    # A shared-effect hypothesis must claim both Lanes.  This
                    # malformed adjacent profile must not invalidate Candidate A.
                    claimed_lane_ids=[lane_a],
                    comparison_lane_ids=[lane_a, lane_b],
                    prediction="Both recipes should exhibit the same excursion.",
                )
            ],
            "analysis_summary": "One valid Candidate with invalid scope metadata.",
        }
        client = CandidateClient([response, response])

        result = QwenHypothesisCandidateGenerator(client).generate(
            request_id="REQ_INVALID_SHARED_SCOPE",
            findings=causal_findings(),
            context_evidence=[exposure_a, exposure_b, process_a, process_b],
            causal_lanes=[
                {
                    "lane_id": lane_a,
                    "operation": "2500",
                    "equipment": "EQ_A",
                    "chamber": "EQ_A_CH01",
                    "recipe": "RCP_A",
                    "parameter_scope": ["oxygen_flow"],
                    "priority_score": 0.9,
                    "investigation_status": "evidence_collected",
                },
                {
                    "lane_id": lane_b,
                    "operation": "2500",
                    "equipment": "EQ_A",
                    "chamber": "EQ_A_CH01",
                    "recipe": "RCP_B",
                    "parameter_scope": ["oxygen_flow"],
                    "priority_score": 0.8,
                    "investigation_status": "evidence_collected",
                },
            ],
        )

        self.assertEqual(result.attempt_count, 2)
        self.assertEqual(len(result.candidates), 1)
        self.assertFalse(result.candidate_output_invalid)
        self.assertEqual(
            result.competition_status,
            CandidateCompetitionStatus.PENDING.value,
        )
        self.assertIsNone(result.competition_failure_reason)
        self.assertEqual(
            result.competition_gap_reason,
            CompetitionGapReason.MECHANISM_ALTERNATIVE_NOT_GENERATED.value,
        )
        self.assertTrue(result.competition_repair_exhausted)
        self.assertTrue(
            any("shared_effect" in error for error in result.semantic_validation_errors)
        )

    def test_scope_expansion_cannot_silently_replace_prior_candidate(self) -> None:
        lane_a = "lane:2500:EQ_A:EQ_A_CH01:RCP_A"
        lane_b = "lane:2500:EQ_A:EQ_A_CH01:RCP_B"
        exposure_a = typed_evidence(
            evidence_id="EV_SCOPE_LINEAGE_EXP_A",
            evidence_type=EvidenceType.PROCESS_EXPOSURE.value,
            agent=AgentKind.MES.value,
            entity_type=EntityType.EQUIPMENT.value,
            entity_id="EQ_A",
            metadata={"lane_id": lane_a, "recipe": "RCP_A"},
        )
        exposure_b = typed_evidence(
            evidence_id="EV_SCOPE_LINEAGE_EXP_B",
            evidence_type=EvidenceType.PROCESS_EXPOSURE.value,
            agent=AgentKind.MES.value,
            entity_type=EntityType.EQUIPMENT.value,
            entity_id="EQ_A",
            metadata={"lane_id": lane_b, "recipe": "RCP_B"},
        )
        process_a = lane_process_evidence(
            evidence_id="EV_SCOPE_LINEAGE_PROC_A",
            lane_id=lane_a,
            recipe="RCP_A",
            parameter="oxygen_flow",
            equipment="EQ_A",
            chamber="EQ_A_CH01",
            operation="2500",
        )
        process_b = lane_process_evidence(
            evidence_id="EV_SCOPE_LINEAGE_PROC_B",
            lane_id=lane_b,
            recipe="RCP_B",
            parameter="oxygen_flow",
            equipment="EQ_A",
            chamber="EQ_A_CH01",
            operation="2500",
        )
        prior = {
            "candidate_id": "PRIOR:llm:1",
            "root_cause": "RCP_A-specific oxygen-flow instability",
            "causal_explanation": (
                "Recipe RCP_A oxygen-flow instability changes plasma kinetics "
                "and produces the observed edge void."
            ),
            "supporting_evidence_ids": [
                "EV_SCOPE_LINEAGE_EXP_A",
                "EV_SCOPE_LINEAGE_PROC_A",
                "EV_PRODUCT",
            ],
            "contradicting_evidence_ids": [],
        }
        broadened = {
            "root_cause": "EQ_A_CH01 chamber-wide oxygen-flow instability",
            "causal_explanation": (
                "The anomaly now spans RCP_A and RCP_B and is rewritten as one "
                "chamber-wide plasma-control cause."
            ),
            "supporting_evidence_ids": [
                "EV_SCOPE_LINEAGE_EXP_A",
                "EV_SCOPE_LINEAGE_PROC_A",
                "EV_SCOPE_LINEAGE_EXP_B",
                "EV_SCOPE_LINEAGE_PROC_B",
                "EV_PRODUCT",
            ],
            "contradicting_evidence_ids": [],
        }
        response = {
            "candidates": [broadened],
            "candidate_semantic_profiles": [
                semantic_profile(
                    0,
                    scope_relation="shared_effect",
                    claimed_lane_ids=[lane_a, lane_b],
                    comparison_lane_ids=[lane_a, lane_b],
                    prediction="Both recipe Lanes produce the compatible outcome.",
                )
            ],
            "analysis_summary": "The prior candidate was broadened in place.",
        }
        client = CandidateClient([response, response])
        gap_id = "candidate_0.hypothesis_discrimination.parameter_anomaly"

        result = QwenHypothesisCandidateGenerator(client).generate(
            request_id="REQ_SCOPE_COLLAPSE",
            findings=causal_findings(),
            context_evidence=[exposure_a, exposure_b, process_a, process_b],
            prior_candidates=[prior],
            prior_semantic_profiles=[
                {
                    "candidate_id": "PRIOR:llm:1",
                    "scope_relation": "focal_only",
                    "claimed_lane_ids": [lane_a],
                    "comparison_lane_ids": [lane_a, lane_b],
                    "mechanism_claim": "RCP_A amplifies chamber drift.",
                    "distinguishing_predictions": [
                        {
                            "discriminator_kind": "product_outcome",
                            "lane_ids": [lane_a, lane_b],
                            "prediction": "Only RCP_A produces the product outcome.",
                            "schema_version": "1.0",
                        }
                    ],
                    "source": "qwen",
                    "schema_version": "1.0",
                }
            ],
            prior_challenges=[
                {
                    "candidate_id": "PRIOR:llm:1",
                    "strongest_alternative_lane_id": lane_b,
                    "distinguishing_gap_ids": [gap_id],
                    "status": "alternative_identified",
                }
            ],
            prior_causal_gaps=[
                {
                    "gap_id": gap_id,
                    "discriminator_kind": "parameter_anomaly",
                    "target_scope": {"lane_id": lane_b},
                }
            ],
            causal_lanes=[
                {
                    "lane_id": lane_a,
                    "operation": "2500",
                    "equipment": "EQ_A",
                    "chamber": "EQ_A_CH01",
                    "recipe": "RCP_A",
                    "parameter_scope": ["oxygen_flow"],
                    "priority_score": 0.9,
                    "investigation_status": "evidence_collected",
                },
                {
                    "lane_id": lane_b,
                    "operation": "2500",
                    "equipment": "EQ_A",
                    "chamber": "EQ_A_CH01",
                    "recipe": "RCP_B",
                    "parameter_scope": ["oxygen_flow"],
                    "priority_score": 0.8,
                    "investigation_status": "evidence_collected",
                },
            ],
            new_evidence_ids=["EV_SCOPE_LINEAGE_PROC_B"],
        )

        self.assertEqual(result.attempt_count, 2)
        self.assertEqual(
            result.competition_status,
            CandidateCompetitionStatus.PENDING.value,
        )
        self.assertIsNone(result.competition_failure_reason)
        self.assertEqual(
            result.competition_gap_reason,
            CompetitionGapReason.MECHANISM_ALTERNATIVE_NOT_GENERATED.value,
        )
        self.assertTrue(
            any(
                item["lineage_status"] == "scope_expanded"
                for item in result.candidate_lineage
            )
        )

    def test_targeted_alternative_evidence_repairs_candidate_competition(self) -> None:
        findings = causal_findings()
        alternative = typed_evidence(
            evidence_id="EV_ALT_PROCESS",
            evidence_type=EvidenceType.PARAMETER_DEVIATION.value,
            agent=AgentKind.FDC.value,
            entity_type=EntityType.PARAMETER.value,
            entity_id="backside_pressure_range",
            metadata={"lane_id": "LANE_ALT", "direction": "high"},
        )
        gap_id = "candidate_0.hypothesis_discrimination.parameter_anomaly"
        alternative_candidate = {
            "root_cause": "EQ_02 backside pressure regulation instability",
            "causal_explanation": (
                "The alternative Lane contains an independent backside pressure "
                "deviation that can explain a different process failure mechanism."
            ),
            "supporting_evidence_ids": ["EV_ALT_PROCESS"],
            "contradicting_evidence_ids": [],
        }
        client = CandidateClient(
            [
                {
                    "candidates": [proposal()],
                    "analysis_summary": "The prior mechanism remains plausible.",
                },
                {
                    "candidates": [proposal(), alternative_candidate],
                    "analysis_summary": "Two distinct mechanisms now compete.",
                },
            ]
        )

        result = QwenHypothesisCandidateGenerator(client).generate(
            request_id="REQ_COMPETITION_REPAIR",
            findings=findings,
            context_evidence=[alternative],
            prior_candidates=[proposal()],
            prior_challenges=[
                {
                    "candidate_id": "REQ_PRIOR:llm:1",
                    "strongest_alternative_lane_id": "LANE_ALT",
                    "distinguishing_gap_ids": [gap_id],
                    "challenge_explanation": "An alternative process Lane remains.",
                    "status": "alternative_identified",
                }
            ],
            prior_causal_gaps=[
                {
                    "gap_id": gap_id,
                    "discriminator_kind": "parameter_anomaly",
                    "target_scope": {
                        "lane_id": "LANE_ALT",
                        "operation": "OP_ALT",
                    },
                }
            ],
            causal_lanes=[
                {
                    "lane_id": "LANE_ALT",
                    "operation": "OP_ALT",
                    "equipment": "EQ_02",
                    "chamber": "CH_02",
                    "parameter_scope": ["backside_pressure_range"],
                    "investigation_status": "evidence_collected",
                }
            ],
            new_evidence_ids=["EV_ALT_PROCESS"],
        )

        self.assertEqual(result.attempt_count, 2)
        self.assertEqual(len(result.candidates), 2)
        self.assertFalse(result.competition_repair_exhausted)
        self.assertEqual(
            result.targeted_investigation_results[0][
                "new_supporting_evidence_ids"
            ],
            ["EV_ALT_PROCESS"],
        )
        first_payload = client.requests[0].payload
        self.assertEqual(
            first_payload["new_evidence_ids_since_prior"],
            ["EV_ALT_PROCESS"],
        )
        self.assertEqual(
            first_payload["prior_candidate_challenges"][0][
                "strongest_alternative_lane_id"
            ],
            "LANE_ALT",
        )
        self.assertEqual(
            first_payload["targeted_investigation_results"][0]["lane_id"],
            "LANE_ALT",
        )
        feedback = client.requests[1].payload["previous_validation_feedback"]
        self.assertIn("candidate competition", feedback["message"])
        self.assertTrue(
            feedback["candidate_competition"][
                "requires_distinct_candidate_review"
            ]
        )

    def test_competition_repair_does_not_invent_an_alternative_candidate(self) -> None:
        findings = causal_findings()
        alternative = typed_evidence(
            evidence_id="EV_ALT_PROCESS",
            evidence_type=EvidenceType.PARAMETER_DEVIATION.value,
            agent=AgentKind.FDC.value,
            entity_type=EntityType.PARAMETER.value,
            entity_id="backside_pressure_range",
            metadata={"lane_id": "LANE_ALT"},
        )
        gap_id = "candidate_0.hypothesis_discrimination.parameter_anomaly"
        response = {
            "candidates": [proposal()],
            "analysis_summary": "No independent alternative is justified.",
        }
        client = CandidateClient([response, response])

        result = QwenHypothesisCandidateGenerator(client).generate(
            request_id="REQ_COMPETITION_EXHAUSTED",
            findings=findings,
            context_evidence=[alternative],
            prior_candidates=[proposal()],
            prior_challenges=[
                {
                    "candidate_id": "REQ_PRIOR:llm:1",
                    "strongest_alternative_lane_id": "LANE_ALT",
                    "distinguishing_gap_ids": [gap_id],
                    "status": "alternative_identified",
                }
            ],
            prior_causal_gaps=[
                {
                    "gap_id": gap_id,
                    "discriminator_kind": "parameter_anomaly",
                    "target_scope": {"lane_id": "LANE_ALT"},
                }
            ],
            causal_lanes=[{"lane_id": "LANE_ALT"}],
            new_evidence_ids=["EV_ALT_PROCESS"],
        )

        self.assertEqual(len(result.candidates), 1)
        self.assertTrue(result.competition_repair_exhausted)
        self.assertFalse(result.candidate_output_invalid)

    def test_non_discriminating_evidence_does_not_force_an_alternative(self) -> None:
        findings = causal_findings()
        unavailable = typed_evidence(
            evidence_id="EV_ALT_MISSING",
            evidence_type=EvidenceType.DATA_MISSING.value,
            agent=AgentKind.FDC.value,
            entity_type=EntityType.PARAMETER.value,
            entity_id="backside_pressure_range",
            metadata={"lane_id": "LANE_ALT"},
        )
        unrelated_product = typed_evidence(
            evidence_id="EV_ALT_PRODUCT",
            evidence_type=EvidenceType.DEFECT_SIGNAL.value,
            agent=AgentKind.DEFECT_WAT.value,
            entity_type=EntityType.DEFECT.value,
            entity_id="edge_void",
            metadata={"lane_id": "LANE_ALT"},
        )
        gap_id = "candidate_0.hypothesis_discrimination.parameter_anomaly"
        client = CandidateClient(
            [
                {
                    "candidates": [proposal()],
                    "analysis_summary": "The alternative source is unavailable.",
                }
            ]
        )

        result = QwenHypothesisCandidateGenerator(client).generate(
            request_id="REQ_COMPETITION_MISSING",
            findings=findings,
            context_evidence=[unavailable, unrelated_product],
            prior_candidates=[proposal()],
            prior_challenges=[
                {
                    "candidate_id": "REQ_PRIOR:llm:1",
                    "strongest_alternative_lane_id": "LANE_ALT",
                    "distinguishing_gap_ids": [gap_id],
                    "status": "blocked",
                }
            ],
            prior_causal_gaps=[
                {
                    "gap_id": gap_id,
                    "discriminator_kind": "parameter_anomaly",
                    "target_scope": {"lane_id": "LANE_ALT"},
                }
            ],
            causal_lanes=[{"lane_id": "LANE_ALT"}],
            new_evidence_ids=["EV_ALT_MISSING", "EV_ALT_PRODUCT"],
        )

        self.assertEqual(result.attempt_count, 1)
        self.assertEqual(len(result.candidates), 1)
        self.assertFalse(result.competition_repair_exhausted)
        self.assertFalse(
            result.targeted_investigation_results[0]["support_observed"]
        )

    def test_consumed_lane_discriminator_is_not_generated_again(self) -> None:
        findings = causal_findings()
        evidence = [item for finding in findings for item in finding.evidence]
        matrix = build_causal_evidence_matrix(
            CausalHypothesis(
                root_cause=str(proposal()["root_cause"]),
                causal_explanation=str(proposal()["causal_explanation"]),
                supporting_evidence_ids=tuple(
                    proposal()["supporting_evidence_ids"]
                ),
            ),
            evidence,
        )
        lane = CausalLaneRecord(
            lane_id="LANE_ALT",
            operation="OP_ALT",
            equipment="EQ_02",
            chamber="CH_02",
            parameter_scope=("backside_pressure_range",),
            exposed_lot_ids=("LOT_01", "LOT_02"),
        )

        gaps = build_hypothesis_discrimination_gaps(
            [matrix],
            causal_lanes=[lane],
            source_lot_id="LOT_01",
            consumed_discriminators={("LANE_ALT", "parameter_anomaly")},
        )

        kinds = {str(item["discriminator_kind"]) for item in gaps}
        self.assertNotIn("parameter_anomaly", kinds)
        self.assertIn("product_outcome", kinds)

    def test_rca_agent_falls_back_safely_after_invalid_qwen_candidates(self) -> None:
        invalid = {
            "candidates": [proposal(supporting=["EV_UNKNOWN"])],
            "analysis_summary": "Invalid unknown Evidence.",
        }
        client = CandidateClient([invalid])

        result = RCAReasoningAgent(
            llm_client=client,
            agent_mode="llm",
        ).analyze(
            request_id="REQ_RCA_FALLBACK",
            findings=causal_findings(),
        )

        generation = result.details["hypothesis_candidate_generation"]
        self.assertEqual(generation["source"], "qwen")
        self.assertEqual(
            generation["fallback_reason"],
            "qwen_candidate_output_invalid",
        )
        self.assertTrue(generation["candidate_output_invalid"])
        self.assertEqual(result.details["status"], "inconclusive")
        self.assertIn(
            "WARN_RCA_QWEN_CANDIDATE_INVALID",
            {warning.warning_id for warning in result.warnings},
        )
        self.assertEqual(result.details["hypothesis_engine_result"]["candidates"], [])

    def test_rca_finding_persists_lane_first_competition_synthesis(self) -> None:
        client = CandidateClient(
            [
                {
                    "candidates": [proposal()],
                    "analysis_summary": "One evidence-bounded candidate remains.",
                }
            ]
        )

        result = RCAReasoningAgent(
            llm_client=client,
            agent_mode="llm",
        ).analyze(
            request_id="REQ_RCA_LANE_FIRST_PERSISTENCE",
            findings=causal_findings(),
        )

        synthesis = result.details["evidence_synthesis"]
        self.assertEqual(synthesis["schema"], "lane_first_v1")
        self.assertIn("candidate_competition", synthesis)
        self.assertLessEqual(len(synthesis["active_causal_lanes"]), 3)
        self.assertIn("prompt_evidence_ids", synthesis)

    def test_valid_empty_qwen_candidates_do_not_enable_deterministic_root_cause(
        self,
    ) -> None:
        client = CandidateClient(
            [
                {
                    "candidates": [],
                    "analysis_summary": "No evidence-bounded causal candidate exists.",
                }
            ]
        )

        result = RCAReasoningAgent(
            llm_client=client,
            agent_mode="llm",
        ).analyze(
            request_id="REQ_RCA_EMPTY",
            findings=causal_findings(),
        )

        generation = result.details["hypothesis_candidate_generation"]
        self.assertFalse(generation["candidate_output_invalid"])
        self.assertEqual(generation["candidate_count"], 0)
        self.assertEqual(result.details["status"], "inconclusive")
        self.assertEqual(result.details["hypothesis_engine_result"]["candidates"], [])

    def test_candidate_provider_failure_does_not_invent_a_deterministic_candidate(
        self,
    ) -> None:
        result = RCAReasoningAgent(
            llm_client=CandidateProviderFailureClient(),
            agent_mode="llm",
        ).analyze(
            request_id="REQ_RCA_PROVIDER_FAILURE",
            findings=causal_findings(),
        )

        generation = result.details["hypothesis_candidate_generation"]
        self.assertEqual(
            generation["fallback_reason"],
            "qwen_candidate_provider_failed",
        )
        self.assertEqual(generation["candidate_count"], 0)
        self.assertEqual(result.details["status"], "inconclusive")
        self.assertEqual(result.details["hypothesis_engine_result"]["candidates"], [])
        self.assertIn(
            "WARN_RCA_LLM_CANDIDATE_FALLBACK",
            {warning.warning_id for warning in result.warnings},
        )


if __name__ == "__main__":
    unittest.main()
