from __future__ import annotations

import inspect
import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core"))

from yield_rca_core.authoritative_result import AuthoritativeRCAResult  # noqa: E402
from yield_rca_core.causal_investigation_models import (  # noqa: E402
    AlternativeSearchStatus,
    CandidateCompetitionStatus,
    CandidateCompetitionType,
    CompetitionFailureReason,
    CompetitionRequirement,
    CompetitionTrace,
)
from yield_rca_core.models import (  # noqa: E402
    AgentFinding,
    AgentKind,
    Evidence,
    EvidenceEntity,
    EvidenceSourceType,
    EvidenceType,
    FindingKind,
    Hypothesis,
    RCAJob,
    RCAState,
    Report,
    Warning,
)
from yield_rca_core.rca_reasoning_agent import RCAReasoningAgent  # noqa: E402
from yield_rca_core.report_generator import (  # noqa: E402
    ReportGenerationError,
    ReportGenerator,
)
from yield_rca_core.repositories import CsvFabRepository  # noqa: E402
from yield_rca_core.specialist_agents import (  # noqa: E402
    DefectWATAgent,
    FDCAgent,
    KnowledgeAgent,
    MESAgent,
)
from yield_rca_core.tool_layer import (  # noqa: E402
    AnalyzeLotGenealogyTool,
    AnalyzeParameterShiftTool,
    FindAffectedLotsTool,
    FindOocEventsTool,
    PerformBasicSpcAnalysisTool,
    RetrieveSimilarCaseTool,
    SummarizeDefectWatTool,
)

SEED_DIR = ROOT / "data" / "seeds" / "golden_case"
AFFECTED_LOTS = [f"LOT_A_{index:03d}" for index in range(1, 21)]
EXPECTED_ROOT_CAUSE = "CMP_CU03_CH02 slurry delivery degradation"


def golden_specialist_findings() -> list[AgentFinding]:
    repository = CsvFabRepository(SEED_DIR)
    return [
        MESAgent(
            FindAffectedLotsTool(repository),
            AnalyzeLotGenealogyTool(repository),
        ).analyze(
            request_id="REQ_REPORT_MES",
            product_id="40N_SOC",
            start_date="2026-07-01",
            end_date="2026-07-31",
        ),
        FDCAgent(
            AnalyzeParameterShiftTool(repository),
            FindOocEventsTool(repository),
            PerformBasicSpcAnalysisTool(repository),
        ).analyze(
            request_id="REQ_REPORT_FDC",
            lot_ids=AFFECTED_LOTS,
            operation_no="6400",
            equipment_id="CMP_CU03",
            chamber_id="CMP_CU03_CH02",
        ),
        DefectWATAgent(SummarizeDefectWatTool(repository)).analyze(
            request_id="REQ_REPORT_DEFECT_WAT",
            lot_ids=AFFECTED_LOTS,
        ),
        KnowledgeAgent(RetrieveSimilarCaseTool(repository)).analyze(
            request_id="REQ_REPORT_KNOWLEDGE",
            query="Cu CMP slurry flow scratch leakage",
            module="Cu CMP",
            equipment_type="CMP",
        ),
    ]


def golden_rca_state() -> RCAState:
    specialist_findings = golden_specialist_findings()
    rca_finding = RCAReasoningAgent().analyze(
        request_id="REQ_REPORT_RCA",
        findings=specialist_findings,
    )
    evidence_by_id: dict[str, Evidence] = {}
    for finding in specialist_findings:
        for payload in finding.details["evidence"]:
            evidence = Evidence.from_dict(payload)
            evidence_by_id[evidence.evidence_id] = evidence

    return RCAState(
        job=RCAJob(
            job_id="JOB_GOLDEN_REPORT",
            user_query="Analyze the July yield drop for 40N_SOC.",
            product_id="40N_SOC",
            time_window={"start": "2026-07-01", "end": "2026-07-31"},
        ),
        affected_lots=AFFECTED_LOTS,
        evidence=list(evidence_by_id.values()),
        findings=[*specialist_findings, rca_finding],
        hypotheses=[Hypothesis.from_dict(rca_finding.details["hypothesis"])],
        warnings=list(rca_finding.warnings),
    )


class ReportGeneratorContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.state = golden_rca_state()
        cls.report = ReportGenerator().generate(
            cls.state,
            report_id="REPORT_GOLDEN",
        )

    def test_report_contains_all_required_sections(self) -> None:
        required_sections = (
            "## Problem Summary",
            "## Affected Lots",
            "## Evidence Chain",
            "## Root Cause",
            "## Confidence",
            "## Recommended Actions",
            "## Warnings",
            "## Typed Evidence Register",
        )
        for section in required_sections:
            self.assertIn(section, self.report.markdown)

    def test_golden_report_contains_only_state_backed_rca_values(self) -> None:
        self.assertIn(EXPECTED_ROOT_CAUSE, self.report.markdown)
        self.assertIn("**95.0%**", self.report.markdown)
        self.assertIn("Count: 20", self.report.markdown)
        self.assertIn("`LOT_A_001`", self.report.markdown)
        self.assertIn("`LOT_A_020`", self.report.markdown)
        self.assertIn("Inspect slurry pump", self.report.markdown)
        self.assertIn("WARN_SPC_BASELINE_INSUFFICIENT", self.report.markdown)
        self.assertIn("## Minimal SPC Analysis", self.report.markdown)

    def test_report_does_not_resurrect_historical_missing_specialist_warning(self) -> None:
        stale = Warning(
            warning_id="WARN_RCA_MISSING_FINDINGS",
            message="Missing Specialist findings: knowledge.",
        )
        historical_rca = replace(
            self.state.findings[-1],
            warnings=[*self.state.findings[-1].warnings, stale],
        )
        state = replace(
            self.state,
            findings=[*self.state.findings[:-1], historical_rca],
            warnings=[*self.state.warnings, stale],
        )

        report = ReportGenerator().generate(state)

        self.assertNotIn("WARN_RCA_MISSING_FINDINGS", report.markdown)
        self.assertIn("WARN_SPC_BASELINE_INSUFFICIENT", report.markdown)

    def test_report_citations_resolve_to_rca_state_evidence(self) -> None:
        state_evidence_ids = {item.evidence_id for item in self.state.evidence}

        self.assertTrue(self.report.cited_evidence_ids)
        self.assertTrue(set(self.report.cited_evidence_ids) <= state_evidence_ids)
        for evidence_id in self.report.cited_evidence_ids:
            self.assertIn(f"`{evidence_id}`", self.report.markdown)

    def test_typed_evidence_register_uses_actual_typed_evidence_fields(self) -> None:
        evidence = next(
            item for item in self.state.evidence if item.evidence_id == "EV_MES_COMMON_CHAMBER"
        )

        self.assertIn(evidence.evidence_type, self.report.markdown)
        self.assertIn(evidence.observation, self.report.markdown)
        self.assertIn(f"{evidence.source_agent}/{evidence.source_tool}", self.report.markdown)
        self.assertNotIn("unknown_table", self.report.markdown)

    def test_report_round_trip_preserves_markdown_and_citations(self) -> None:
        restored = Report.from_dict(self.report.to_dict())

        self.assertEqual(restored, self.report)

    def test_report_prefers_hypothesis_ranking_finding_kind(self) -> None:
        evidence = Evidence(
            evidence_id="EV_TYPED_ROOT",
            source_type=EvidenceSourceType.ANALYTICS.value,
            source_id="rca:selected",
            source_table="rca_results",
            summary="Selected RCA result is supported.",
            evidence_type=EvidenceType.EQUIPMENT_EXPOSURE.value,
            source_agent=AgentKind.RCA_REASONING.value,
            source_tool="rank_hypotheses",
            observation="Selected RCA result points to chamber drift.",
            entities=[EvidenceEntity(entity_type="equipment", entity_id="CMP_CU03")],
            confidence=0.91,
            evidence_schema_version="1.0",
        )
        draft = AgentFinding(
            finding_id="FINDING_DRAFT_RCA",
            agent=AgentKind.RCA_REASONING.value,
            finding_kind=FindingKind.HYPOTHESIS_GENERATION.value,
            summary="Draft RCA finding.",
            confidence=0.2,
            evidence_ids=[evidence.evidence_id],
            details={
                "root_cause": "Draft unsupported cause",
                "status": "candidate",
                "root_cause_evidence_ids": [evidence.evidence_id],
            },
        )
        selected = AgentFinding(
            finding_id="FINDING_SELECTED_RCA",
            agent=AgentKind.RCA_REASONING.value,
            finding_kind=FindingKind.HYPOTHESIS_RANKING.value,
            summary="Selected RCA finding.",
            confidence=0.91,
            evidence_ids=[evidence.evidence_id],
            details={
                "root_cause": "Chamber drift",
                "status": "supported",
                "root_cause_evidence_ids": [evidence.evidence_id],
                "recommended_actions": [
                    {"action": "Inspect chamber controls.", "evidence_ids": [evidence.evidence_id]}
                ],
                "evidence_chain": [
                    {
                        "stage": "rca_reasoning",
                        "claim": "Typed finding selected by finding_kind.",
                        "confidence": 0.91,
                        "evidence_ids": [evidence.evidence_id],
                    }
                ],
            },
        )
        state = RCAState(
            job=RCAJob(job_id="JOB_TYPED_REPORT", user_query="Test typed report."),
            affected_lots=["LOT_A_001"],
            evidence=[evidence],
            findings=[draft, selected],
        )

        report = ReportGenerator().generate(state)

        self.assertIn("Chamber drift", report.markdown)
        self.assertNotIn("Draft unsupported cause", report.markdown)
        self.assertIn("## Typed Evidence Register", report.markdown)
        self.assertIn("equipment: CMP_CU03", report.markdown)
        self.assertIn("Selected RCA result points to chamber drift.", report.markdown)

    def test_report_uses_explicit_authority_when_reasoning_is_iterative(self) -> None:
        historical = self.state.findings[-1]
        current = replace(
            historical,
            finding_id="FINDING_CURRENT_RCA",
            summary="Current RCA finding supersedes the historical reasoning pass.",
            details={
                **historical.details,
                "root_cause": "Current chamber pressure drift",
                "status": "supported",
            },
        )
        historical_hypothesis = self.state.hypotheses[0]
        current_hypothesis = replace(
            historical_hypothesis,
            hypothesis_id="HYP_CURRENT_RCA",
            root_cause="Current chamber pressure drift",
        )
        state = replace(
            self.state,
            findings=[*self.state.findings[:-1], historical, current],
            hypotheses=[historical_hypothesis, current_hypothesis],
            authoritative_rca_finding_id=current.finding_id,
            authoritative_hypothesis_id=current_hypothesis.hypothesis_id,
        )

        report = ReportGenerator().generate(state)

        self.assertIn("Current chamber pressure drift", report.markdown)
        self.assertNotIn(EXPECTED_ROOT_CAUSE, report.markdown)
        self.assertEqual(state.authoritative_rca_finding, current)
        self.assertEqual(state.authoritative_hypothesis, current_hypothesis)

    def test_report_prefers_typed_authority_over_legacy_finding_conclusion(self) -> None:
        evidence = Evidence(
            evidence_id="EV_TYPED_AUTHORITY_REPORT",
            source_type=EvidenceSourceType.ANALYTICS.value,
            source_id="authority:report",
            summary="Typed authority evidence for the final report.",
        )
        finding = AgentFinding(
            finding_id="FINDING_LEGACY_AUTHORITY_REPORT",
            agent=AgentKind.RCA_REASONING.value,
            summary="Legacy finding retained for audit details.",
            confidence=0.83,
            evidence_ids=[evidence.evidence_id],
            details={
                "root_cause": "Legacy finding root cause",
                "status": "supported",
                "conclusion_status": "supported",
                "root_cause_evidence_ids": [evidence.evidence_id],
                "recommended_actions": [],
                "evidence_chain": [],
            },
        )
        authority = AuthoritativeRCAResult(
            result_id="RCA_RESULT_REPORT",
            source_finding_id=finding.finding_id,
            source_hypothesis_id=None,
            conclusion_status="supported",
            root_cause_candidate_id="CANDIDATE_TYPED_REPORT",
            root_cause="Typed authoritative root cause",
            confirmation_status="supported",
            competition_status="complete_confirmed",
            terminal_reason="confirmation_gate_supported_unique_candidate",
            evidence_refs=(evidence.evidence_id,),
        )
        state = RCAState(
            job=RCAJob(job_id="JOB_TYPED_AUTHORITY_REPORT", user_query="Render authority."),
            affected_lots=["LOT_A_001"],
            evidence=[evidence],
            findings=[finding],
            authoritative_rca_finding_id=finding.finding_id,
            authoritative_rca_result=authority,
        )

        report = ReportGenerator().generate(state)

        self.assertIn("- Root Cause: **Typed authoritative root cause**", report.markdown)
        self.assertNotIn("- Root Cause: **Legacy finding root cause**", report.markdown)
        self.assertIn("- Confirmation Status: `supported`", report.markdown)
        self.assertIn("- Competition Status: `complete_confirmed`", report.markdown)
        self.assertIn(
            "- Terminal Reason: `confirmation_gate_supported_unique_candidate`",
            report.markdown,
        )

    def test_report_does_not_publish_legacy_root_for_inconclusive_authority(self) -> None:
        evidence = Evidence(
            evidence_id="EV_INCONCLUSIVE_AUTHORITY_REPORT",
            source_type=EvidenceSourceType.ANALYTICS.value,
            source_id="authority:inconclusive-report",
            summary="Evidence remains auditable without a publishable root cause.",
        )
        finding = AgentFinding(
            finding_id="FINDING_STALE_SUPPORTED_REPORT",
            agent=AgentKind.RCA_REASONING.value,
            summary="Historical finding predating terminal authority.",
            confidence=0.79,
            evidence_ids=[evidence.evidence_id],
            details={
                "root_cause": "Stale supported root cause",
                "status": "supported",
                "conclusion_status": "supported",
                "root_cause_evidence_ids": [evidence.evidence_id],
                "recommended_actions": [
                    {
                        "action": "Apply stale legacy recommendation.",
                        "evidence_ids": [evidence.evidence_id],
                    }
                ],
                "evidence_chain": [
                    {
                        "stage": "rca_reasoning",
                        "claim": "Stale legacy evidence chain.",
                        "confidence": 0.79,
                        "evidence_ids": [evidence.evidence_id],
                    }
                ],
            },
        )
        authority = AuthoritativeRCAResult(
            result_id="RCA_RESULT_INCONCLUSIVE_REPORT",
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
            job=RCAJob(job_id="JOB_INCONCLUSIVE_REPORT", user_query="Withhold root cause."),
            affected_lots=["LOT_A_001"],
            evidence=[evidence],
            findings=[finding],
            authoritative_rca_finding_id=finding.finding_id,
            authoritative_rca_result=authority,
        )

        report = ReportGenerator().generate(state)

        self.assertIn("- Conclusion Status: `inconclusive`", report.markdown)
        self.assertIn("- Root Cause: Not published by authoritative result.", report.markdown)
        self.assertNotIn("- Root Cause: **Stale supported root cause**", report.markdown)
        self.assertNotIn("Apply stale legacy recommendation.", report.markdown)
        self.assertNotIn("Stale legacy evidence chain.", report.markdown)

    def test_report_exposes_candidate_competition_state_and_lineage(self) -> None:
        state = replace(
            self.state,
            competition_trace=CompetitionTrace(
                alternative_search_status=AlternativeSearchStatus.UNRESOLVED.value,
                competition_requirement=CompetitionRequirement.SCOPE_REQUIRED.value,
                competition_status=CandidateCompetitionStatus.FAILED.value,
                competition_type=CandidateCompetitionType.SCOPE.value,
                competition_failure_reason=(
                    CompetitionFailureReason.SCOPE_HYPOTHESIS_COLLAPSED.value
                ),
                candidate_lineage=(
                    {
                        "candidate_id": "candidate_0",
                        "lineage_status": "retained",
                        "root_cause": "Recipe-specific chamber drift",
                    },
                    {
                        "candidate_id": "candidate_1",
                        "lineage_status": "omitted",
                        "prior_root_cause": "Chamber-wide drift",
                    },
                ),
            ),
        )

        report = ReportGenerator().generate(state)

        self.assertIn("## Candidate Competition", report.markdown)
        self.assertIn("`scope_required`", report.markdown)
        self.assertIn("`failed`", report.markdown)
        self.assertIn("`scope_hypothesis_collapsed`", report.markdown)
        self.assertIn("`retained`: Recipe-specific chamber drift", report.markdown)
        self.assertIn("`omitted`: Chamber-wide drift", report.markdown)

    def test_missing_state_data_is_marked_not_invented(self) -> None:
        evidence = Evidence(
            evidence_id="EV_ONLY_RECORD",
            source_type=EvidenceSourceType.MES.value,
            source_id="process_history:only-record",
            source_table="process_history",
            summary="One manufacturing record is available.",
        )
        state = RCAState(
            job=RCAJob(job_id="JOB_INCOMPLETE", user_query="Investigate available data."),
            evidence=[evidence],
        )

        report = ReportGenerator().generate(state)

        self.assertIn("Root Cause: Not available in RCAState.", report.markdown)
        self.assertIn("Confidence: Not available in RCAState.", report.markdown)
        self.assertIn("Affected lots are not available in RCAState.", report.markdown)
        self.assertIn("RCA reasoning result is not available in RCAState.", report.markdown)
        self.assertNotIn(EXPECTED_ROOT_CAUSE, report.markdown)
        self.assertNotIn("Inspect slurry pump", report.markdown)

    def test_unknown_nested_evidence_reference_is_rejected(self) -> None:
        evidence = Evidence(
            evidence_id="EV_KNOWN",
            source_type=EvidenceSourceType.MES.value,
            source_id="process_history:known",
            summary="Known evidence.",
        )
        rca_finding = AgentFinding(
            finding_id="FINDING_BAD_REPORT_RCA",
            agent=AgentKind.RCA_REASONING.value,
            summary="Unsupported nested citation.",
            confidence=0.8,
            evidence_ids=[evidence.evidence_id],
            details={
                "root_cause": "A state-provided conclusion",
                "status": "supported",
                "root_cause_evidence_ids": ["EV_NOT_IN_STATE"],
                "recommended_actions": [],
                "evidence_chain": [],
            },
        )
        state = RCAState(
            job=RCAJob(job_id="JOB_BAD_CITATION", user_query="Test citations."),
            evidence=[evidence],
            findings=[rca_finding],
        )

        with self.assertRaises(ReportGenerationError):
            ReportGenerator().generate(state)

    def test_report_without_evidence_is_rejected(self) -> None:
        state = RCAState(
            job=RCAJob(job_id="JOB_NO_EVIDENCE", user_query="No evidence available."),
        )

        with self.assertRaises(ReportGenerationError):
            ReportGenerator().generate(state)

    def test_report_generator_only_depends_on_domain_state(self) -> None:
        import yield_rca_core.report_generator as report_generator

        source = inspect.getsource(report_generator).lower()
        forbidden_dependencies = (
            "yield_rca_core.repositories",
            "yield_rca_core.tool_layer",
            "yield_rca_core.rca_reasoning_agent",
            "csvfabrepository",
            "postgresfabrepository",
            "psycopg",
            "sqlalchemy",
            "toolinput",
            "tooloutput",
        )
        for dependency in forbidden_dependencies:
            self.assertNotIn(dependency, source)


if __name__ == "__main__":
    unittest.main()
