from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_formal_blind_rca import PublicCase  # noqa: E402
from yield_rca_core.causal_adversarial import (  # noqa: E402
    _challenge_output_contract,
    _normalize_challenge_payload,
    derive_alternative_search_status,
)
from yield_rca_core.causal_confirmation import confirm_candidate  # noqa: E402
from yield_rca_core.causal_evidence_matrix import build_causal_evidence_matrix  # noqa: E402
from yield_rca_core.causal_hypothesis import CausalHypothesis  # noqa: E402
from yield_rca_core.causal_investigation_models import (  # noqa: E402
    AlternativeSearchStatus,
    CandidateChallenge,
    CausalLaneRecord,
    ChallengeStatus,
    InvestigationLaneStatus,
)
from yield_rca_core.evidence_models import (  # noqa: E402
    EVIDENCE_SCHEMA_VERSION,
    EntityType,
    Evidence,
    EvidenceEntity,
    EvidenceSourceType,
    EvidenceType,
)
from yield_rca_core.llm_gateway import LLMOutputValidationError  # noqa: E402
from yield_rca_core.models import AgentFinding, RCAJob, RCAState  # noqa: E402
from yield_rca_core.rca_reasoning_agent import (  # noqa: E402
    _unsupported_source_warning,
)
from yield_rca_core.supervisor import (  # noqa: E402
    _source_anchored_lot_scope,
    _update_competition_state,
)
from yield_rca_core.workflow import build_csv_workflow  # noqa: E402


def _evidence(
    evidence_id: str,
    evidence_type: str,
    entities: list[EvidenceEntity],
    *,
    metadata: dict[str, object] | None = None,
    source_field: str | None = None,
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        source_type=EvidenceSourceType.FDC.value,
        source_id=f"SRC_{evidence_id}",
        summary=evidence_id,
        source_field=source_field,
        timestamp="2026-01-01T00:30:00+00:00",
        metadata=metadata or {},
        evidence_type=evidence_type,
        source_agent="fdc",
        source_tool="test_tool",
        observation=evidence_id,
        entities=entities,
        confidence=0.95,
        evidence_schema_version=EVIDENCE_SCHEMA_VERSION,
    )


def _complete_matrix(
    *,
    required_missing: bool = False,
    optional_missing: bool = False,
):
    lot = EvidenceEntity(EntityType.LOT.value, "LOT_01")
    evidence = [
        _evidence(
            "EV_EXPOSURE",
            EvidenceType.IMPACT_SCOPE.value,
            [
                lot,
                EvidenceEntity(EntityType.EQUIPMENT.value, "EQ_01"),
                EvidenceEntity(EntityType.CHAMBER.value, "CH_01"),
                EvidenceEntity(EntityType.OPERATION.value, "OP_4000"),
            ],
        ),
        _evidence(
            "EV_PROCESS",
            EvidenceType.PARAMETER_DEVIATION.value,
            [
                lot,
                EvidenceEntity(EntityType.EQUIPMENT.value, "EQ_01"),
                EvidenceEntity(EntityType.CHAMBER.value, "CH_01"),
                EvidenceEntity(EntityType.OPERATION.value, "OP_4000"),
                EvidenceEntity(EntityType.PARAMETER.value, "pressure"),
            ],
            source_field="pressure",
            metadata={
                "direction": "high",
                "excursion_start": "2026-01-01T00:00:00+00:00",
                "excursion_end": "2026-01-01T01:00:00+00:00",
            },
        ),
        _evidence(
            "EV_OUTCOME",
            EvidenceType.DEFECT_SIGNAL.value,
            [lot, EvidenceEntity(EntityType.DEFECT.value, "center_void")],
        ),
    ]
    if required_missing:
        evidence.append(
            _evidence(
                "EV_REQUIRED_MISSING",
                EvidenceType.DATA_MISSING.value,
                [lot],
                source_field="split_genealogy",
                metadata={"required_for_confirmation": True},
            )
        )
    if optional_missing:
        evidence.append(
            _evidence(
                "EV_OPTIONAL_SPC_MISSING",
                EvidenceType.DATA_MISSING.value,
                [
                    lot,
                    EvidenceEntity(EntityType.PARAMETER.value, "pressure"),
                ],
                source_field="pressure",
                metadata={"required_for_confirmation": False},
            )
        )
    candidate = CausalHypothesis(
        root_cause="EQ_01 CH_01 OP_4000 pressure high excursion",
        causal_explanation="High pressure produces center void.",
        supporting_evidence_ids=("EV_EXPOSURE", "EV_PROCESS", "EV_OUTCOME"),
    )
    return build_causal_evidence_matrix(candidate, evidence)


def test_validated_scope_semantics_require_scope_challenge_kind() -> None:
    contract = _challenge_output_contract(
        candidate_ids=["REQ:llm:1", "REQ:llm:2"],
        active_lane_ids=["LANE_A", "LANE_B"],
        candidate_competition={
            "competition_requirement": "scope_required",
            "semantic_profiles_complete": True,
        },
    )
    assert contract["required_challenge_kind"] == "scope"

    pending_contract = _challenge_output_contract(
        candidate_ids=["REQ:llm:1", "REQ:llm:2"],
        active_lane_ids=["LANE_A", "LANE_B"],
        candidate_competition={
            "competition_requirement": "scope_required",
            "semantic_profiles_complete": False,
        },
    )
    assert pending_contract["required_challenge_kind"] is None


def test_lane_comparison_scope_keeps_the_fixed_source_lot() -> None:
    assert _source_anchored_lot_scope(
        "lot-source",
        ["lot-compare-01", "LOT-SOURCE", "lot-compare-02"],
    ) == ["LOT-SOURCE", "LOT-COMPARE-01", "LOT-COMPARE-02"]


def test_formal_case_preserves_declared_unavailable_sources() -> None:
    case = PublicCase.from_dict(
        {
            "case_id": "FORMAL_TEST",
            "source_lot_id": "LOT_01",
            "query": "Investigate LOT_01",
            "declared_unavailable_sources": ["split_genealogy"],
        }
    )
    assert case.declared_unavailable_sources == ("split_genealogy",)


def test_workflow_context_becomes_confirmation_blocking_typed_evidence() -> None:
    workflow = build_csv_workflow(ROOT / "data" / "seeds" / "golden_case")
    state = workflow.run(
        "Investigate LOT_A_001",
        job_id="JOB_BATCH25_CONTEXT",
        lot_id="LOT_A_001",
        declared_unavailable_sources=("split_genealogy",),
    )
    context = [
        item
        for item in state.evidence
        if item.metadata.get("context_source") == "formal_case_context"
    ]
    assert state.job.declared_unavailable_sources == ["split_genealogy"]
    assert len(context) == 1
    assert context[0].evidence_type == EvidenceType.DATA_MISSING.value
    assert context[0].metadata.get("required_for_confirmation") is True
    finding = state.authoritative_rca_finding
    assert finding is not None
    assert context[0].evidence_id in finding.details["data_missing_evidence_ids"]
    assert finding.details["conclusion_status"] == "insufficient_evidence"


def test_required_missing_blocks_even_a_complete_candidate() -> None:
    result = confirm_candidate(
        _complete_matrix(required_missing=True),
        strict=True,
        alternative_search_status=AlternativeSearchStatus.ALTERNATIVES_ELIMINATED.value,
    )
    assert result.status == "insufficient_evidence"
    assert result.checks["data_available"] is False
    assert result.data_missing_evidence_ids == ("EV_REQUIRED_MISSING",)
    assert result.blocking_data_missing_evidence_ids == ("EV_REQUIRED_MISSING",)
    assert result.non_blocking_data_missing_evidence_ids == ()


def test_unresolved_candidate_competition_precedes_missing_discriminator_status() -> None:
    result = confirm_candidate(
        _complete_matrix(required_missing=True),
        alternative_matrices=(_complete_matrix(),),
        strict=True,
        alternative_search_status=AlternativeSearchStatus.UNRESOLVED.value,
    )

    assert result.status == "inconclusive"
    assert result.checks["data_available"] is False
    assert result.blocking_data_missing_evidence_ids == ("EV_REQUIRED_MISSING",)
    assert "alternative_search.unresolved" in result.unresolved_gaps


def test_optional_spc_missing_does_not_reclassify_competition_as_insufficient() -> None:
    result = confirm_candidate(
        _complete_matrix(optional_missing=True),
        strict=True,
        alternative_search_status=AlternativeSearchStatus.UNRESOLVED.value,
    )

    assert result.status == "inconclusive"
    assert result.checks["data_available"] is True
    assert result.data_missing_evidence_ids == ("EV_OPTIONAL_SPC_MISSING",)
    assert result.blocking_data_missing_evidence_ids == ()
    assert result.non_blocking_data_missing_evidence_ids == (
        "EV_OPTIONAL_SPC_MISSING",
    )
    assert "data_missing.EV_OPTIONAL_SPC_MISSING" not in result.unresolved_gaps


def test_unavailable_source_warning_uses_only_confirmation_blocking_ids() -> None:
    missing = _evidence(
        "EV_OPTIONAL_SPC_MISSING",
        EvidenceType.DATA_MISSING.value,
        [
            EvidenceEntity(EntityType.LOT.value, "LOT_01"),
            EvidenceEntity(EntityType.PARAMETER.value, "pressure"),
        ],
        source_field="pressure",
        metadata={"required_for_confirmation": False},
    )
    finding = AgentFinding(
        finding_id="F_OPTIONAL_MISSING",
        agent="fdc",
        summary="Optional SPC baseline is unavailable.",
        confidence=1.0,
        evidence_ids=[missing.evidence_id],
        evidence=[missing],
    )

    assert _unsupported_source_warning(
        [finding],
        supported=False,
        ranked_candidates=[],
        blocking_data_missing_evidence_ids=[],
    ) is None
    warning = _unsupported_source_warning(
        [finding],
        supported=False,
        ranked_candidates=[],
        blocking_data_missing_evidence_ids=[missing.evidence_id],
    )
    assert warning is not None
    assert warning.evidence_ids == [missing.evidence_id]


def test_unexplained_precursor_blocks_supported_and_downgrades_chain() -> None:
    result = confirm_candidate(
        _complete_matrix(),
        strict=True,
        alternative_search_status=AlternativeSearchStatus.ALTERNATIVES_ELIMINATED.value,
        unexplained_precursor_evidence_ids=("EV_PRECURSOR",),
    )
    assert result.status == "inconclusive"
    assert result.checks["precursor_explained"] is False
    assert result.causal_chain_completeness == "incomplete"


def test_mixed_concrete_lane_support_is_scope_conflicted() -> None:
    base = _complete_matrix()
    evidence = []
    for index, item in enumerate(
        [
            _evidence(
                "EV_EXPOSURE",
                EvidenceType.IMPACT_SCOPE.value,
                [
                    EvidenceEntity(EntityType.LOT.value, "LOT_01"),
                    EvidenceEntity(EntityType.EQUIPMENT.value, "EQ_01"),
                ],
                metadata={"lane_id": "LANE_A"},
            ),
            _evidence(
                "EV_PROCESS",
                EvidenceType.PARAMETER_DEVIATION.value,
                [
                    EvidenceEntity(EntityType.LOT.value, "LOT_01"),
                    EvidenceEntity(EntityType.PARAMETER.value, "pressure"),
                ],
                metadata={
                    "lane_id": "LANE_B",
                    "direction": "high",
                    "excursion_start": "2026-01-01T00:00:00+00:00",
                    "excursion_end": "2026-01-01T01:00:00+00:00",
                },
                source_field="pressure",
            ),
            _evidence(
                "EV_OUTCOME",
                EvidenceType.DEFECT_SIGNAL.value,
                [
                    EvidenceEntity(EntityType.LOT.value, "LOT_01"),
                    EvidenceEntity(EntityType.DEFECT.value, "center_void"),
                ],
            ),
        ]
    ):
        evidence.append(replace(item, confidence=0.95 - index * 0.01))
    matrix = build_causal_evidence_matrix(base.candidate, evidence)
    assert matrix.claims["scope"].status == "conflicted"
    assert "multiple concrete causal Lanes" in matrix.claims["scope"].reason


def test_resolved_alternative_closes_and_persists_lane_elimination() -> None:
    challenge = CandidateChallenge(
        candidate_id="C1",
        strongest_alternative_lane_id="LANE_B",
        supporting_evidence_ids=("EV_B",),
        challenge_explanation="Lane B was contradicted by its own evidence.",
        status=ChallengeStatus.RESOLVED.value,
    )
    assert derive_alternative_search_status(
        challenges=[challenge],
        matrices=[_complete_matrix()],
        active_lane_ids=["LANE_A", "LANE_B"],
    ) == AlternativeSearchStatus.ALTERNATIVES_ELIMINATED.value

    state = RCAState(
        job=RCAJob(job_id="JOB_1", user_query="test"),
        evidence=[
            _evidence(
                "EV_B",
                EvidenceType.PARAMETER_DEVIATION.value,
                [EvidenceEntity(EntityType.LOT.value, "LOT_01")],
            )
        ],
        causal_lanes=[
            CausalLaneRecord(lane_id="LANE_A", priority_score=0.9),
            CausalLaneRecord(lane_id="LANE_B", priority_score=0.8),
        ],
    )
    finding = AgentFinding(
        finding_id="F_RCA",
        agent="rca_reasoning",
        summary="challenge",
        confidence=0.5,
        evidence_ids=["EV_B"],
        details={
            "adversarial_challenge_generation": {"source": "qwen"},
            "candidate_challenges": [challenge.to_dict()],
            "alternative_search_status": "alternatives_eliminated",
        },
    )
    updated = _update_competition_state(state, finding)
    lane_b = next(item for item in updated.causal_lanes if item.lane_id == "LANE_B")
    assert lane_b.investigation_status == InvestigationLaneStatus.ELIMINATED.value
    assert lane_b.lifecycle_status == "eliminated"
    assert lane_b.lifecycle_history[-1].to_status == "eliminated"
    assert lane_b.lifecycle_history[-1].evidence_ids == ("EV_B",)
    assert updated.competition_trace is not None
    assert "LANE_B" in updated.competition_trace.eliminated_lane_ids


def test_resolved_challenge_rejects_unexplained_precursor_and_wrong_lane() -> None:
    lane_a_evidence = _evidence(
        "EV_LANE_A",
        EvidenceType.PARAMETER_DEVIATION.value,
        [
            EvidenceEntity(EntityType.LOT.value, "LOT_01"),
            EvidenceEntity(EntityType.EQUIPMENT.value, "EQ_A"),
        ],
        metadata={"lane_id": "LANE_A"},
    )
    payload = {
        "candidate_id": "C1",
        "strongest_alternative_lane_id": "LANE_B",
        "supporting_evidence_ids": ["EV_LANE_A"],
        "contradicting_evidence_ids": [],
        "unexplained_precursor_evidence_ids": ["EV_LANE_A"],
        "distinguishing_gap_ids": ["G1"],
        "distinguishing_questions": ["Can Lane B explain the precursor?"],
        "challenge_explanation": "Lane B is resolved.",
        "status": "resolved",
    }
    with pytest.raises(
        LLMOutputValidationError,
        match="cannot retain unexplained precursor",
    ):
        _normalize_challenge_payload(
            payload,
            candidate_ids={"C1"},
            lane_ids={"LANE_A", "LANE_B"},
            evidence_ids={"EV_LANE_A"},
            gap_ids={"G1"},
            evidence_by_id={"EV_LANE_A": lane_a_evidence},
            lane_contexts={
                "LANE_A": {"lane_id": "LANE_A", "equipment": "EQ_A"},
                "LANE_B": {"lane_id": "LANE_B", "equipment": "EQ_B"},
            },
        )

    payload["unexplained_precursor_evidence_ids"] = []
    with pytest.raises(
        LLMOutputValidationError,
        match="Entity/time-consistent Evidence",
    ):
        _normalize_challenge_payload(
            payload,
            candidate_ids={"C1"},
            lane_ids={"LANE_A", "LANE_B"},
            evidence_ids={"EV_LANE_A"},
            gap_ids={"G1"},
            evidence_by_id={"EV_LANE_A": lane_a_evidence},
            lane_contexts={
                "LANE_A": {"lane_id": "LANE_A", "equipment": "EQ_A"},
                "LANE_B": {"lane_id": "LANE_B", "equipment": "EQ_B"},
            },
        )
