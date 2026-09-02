"""Integration: Finalizer authority objects drive Report and API projections.

Proves the Batch 4 write path end to end: governed terminal -> authority
objects -> Report generation and API projection agree without re-deriving the
conclusion, while legacy compatibility fields stay intact.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "backend"))

from yield_rca_api.app import _state_api_payload  # noqa: E402
from yield_rca_core.causal_investigation_models import (  # noqa: E402
    CandidateCompetitionStatus,
    CandidateCompetitionType,
    CompetitionRequirement,
    CompetitionTrace,
)
from yield_rca_core.evidence_models import Evidence, EvidenceSourceType  # noqa: E402
from yield_rca_core.investigation_finalizer import (  # noqa: E402
    finalize_investigation,
)
from yield_rca_core.models import (  # noqa: E402
    AgentFinding,
    AgentKind,
    Hypothesis,
    RCAJob,
    RCAState,
)
from yield_rca_core.report_generator import ReportGenerator  # noqa: E402

EVIDENCE_ID = "EV_BATCH4_INTEGRATION"
FINDING_ID = "FINDING_BATCH4_INTEGRATION"
HYPOTHESIS_ID = "HYP_BATCH4_INTEGRATION"


def _finding(*, conclusion_status: str) -> AgentFinding:
    details: dict = {
        "conclusion_status": conclusion_status,
        "status": conclusion_status,
        "confirmation_gate": {"status": conclusion_status},
        "root_cause_evidence_ids": [EVIDENCE_ID],
        "ranked_candidates": [
            {
                "candidate_id": "cand_1",
                "status": "candidate",
                "supporting_evidence_ids": [EVIDENCE_ID],
                "contradicting_evidence_ids": [],
                "causal_evidence_matrix": {},
            }
        ],
    }
    if conclusion_status == "supported":
        details["root_cause"] = "Chamber process drift caused polymer residue"
    details["impact_lot_gate"] = {
        "candidate_impact_lots": ["LOT_INTEGRATION_IMPACT"],
        "rows": [
            {
                "lot_id": "LOT_INTEGRATION_IMPACT",
                "supporting_evidence_ids": [EVIDENCE_ID],
            }
        ],
        "confirmed_impact_lots": ["LOT_INTEGRATION_IMPACT"],
        "publication_status": "confirmed",
    }
    return AgentFinding(
        finding_id=FINDING_ID,
        agent=AgentKind.RCA_REASONING.value,
        summary="An evidence-bounded RCA result retained for terminal governance.",
        confidence=0.7,
        evidence_ids=[EVIDENCE_ID],
        details=details,
    )


def _state(*, conclusion_status: str) -> RCAState:
    evidence = Evidence(
        evidence_id=EVIDENCE_ID,
        source_type=EvidenceSourceType.SYSTEM.value,
        source_id="batch4:integration",
        summary="Synthetic Evidence for the Batch 4 integration contract.",
    )
    return RCAState(
        job=RCAJob(job_id="JOB_BATCH4_INTEGRATION", user_query="Finalize the RCA state."),
        evidence=[evidence],
        findings=[
            _finding(conclusion_status=conclusion_status),
        ],
        hypotheses=[
            Hypothesis(
                hypothesis_id=HYPOTHESIS_ID,
                root_cause="Chamber process drift caused polymer residue",
                confidence=0.7,
                evidence_ids=[EVIDENCE_ID],
                status=conclusion_status,
            )
        ],
        competition_trace=CompetitionTrace(
            competition_requirement=CompetitionRequirement.MECHANISM_REQUIRED.value,
            competition_status=CandidateCompetitionStatus.ACTIVE.value,
            competition_type=CandidateCompetitionType.MECHANISM.value,
        ),
        authoritative_rca_finding_id=FINDING_ID,
        authoritative_hypothesis_id=HYPOTHESIS_ID,
        execution_metadata={},
    )


def test_supported_terminal_agrees_across_authority_report_and_api() -> None:
    finalized = finalize_investigation(_state(conclusion_status="supported"))
    result = finalized.authoritative_rca_result
    assert result is not None
    assert result.conclusion_status == "supported"

    report = ReportGenerator().generate(finalized)
    report_text = report.to_dict()["markdown"]
    assert f"RCA Result ID: `{result.result_id}`" in report_text
    assert f"Conclusion Status: `{result.conclusion_status}`" in report_text
    assert f"Root Cause: **{result.root_cause}**" in report_text

    payload = _state_api_payload(finalized)
    diagnosis = payload["rca_diagnosis"]
    assert diagnosis["result_id"] == result.result_id
    assert diagnosis["finding_id"] == FINDING_ID
    assert diagnosis["conclusion_status"] == result.conclusion_status
    assert diagnosis["root_cause"] == result.root_cause
    assert diagnosis["publication_status"] == "confirmed"
    assert diagnosis["confirmed_impact_lots"] == ["LOT_INTEGRATION_IMPACT"]


def test_inconclusive_terminal_publishes_no_impact_across_surfaces() -> None:
    finalized = finalize_investigation(_state(conclusion_status="inconclusive"))
    result = finalized.authoritative_rca_result
    assert result is not None
    assert result.conclusion_status == "inconclusive"
    assert result.root_cause is None
    publication = finalized.impact_publication_result
    assert publication is not None
    assert publication.publication_status == "withheld"
    assert publication.confirmed_impact_lots == ()

    report = ReportGenerator().generate(finalized)
    report_text = report.to_dict()["markdown"]
    assert f"Conclusion Status: `{result.conclusion_status}`" in report_text
    assert "Root Cause: Not published by authoritative result." in report_text

    payload = _state_api_payload(finalized)
    diagnosis = payload["rca_diagnosis"]
    assert diagnosis["result_id"] == result.result_id
    assert diagnosis["conclusion_status"] == "inconclusive"
    assert diagnosis["root_cause"] is None
    assert diagnosis["confirmed_impact_lots"] == []
