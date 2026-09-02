from __future__ import annotations

import json
from pathlib import Path

import pytest
from yield_rca_core.causal_confirmation import confirm_candidate, evaluate_impact_lot_gate
from yield_rca_core.causal_evidence_matrix import build_causal_evidence_matrix
from yield_rca_core.causal_hypothesis import CausalClaimStatus, CausalHypothesis
from yield_rca_core.causal_investigation_models import (
    AlternativeSearchStatus,
    CandidateSemanticProfile,
)
from yield_rca_core.evidence_models import EntityType, Evidence, EvidenceType
from yield_rca_core.incident_evidence import build_incident_observation_evidence
from yield_rca_core.models import AgentKind, RCAState, ToolInput
from yield_rca_core.repositories import CsvFabRepository
from yield_rca_core.tool_layer import (
    AnalyzeParameterShiftTool,
    FindImpactLotsTool,
    SummarizeDefectWatTool,
)

ROOT = Path(__file__).resolve().parents[2]
REPLAY_STATE = (
    ROOT
    / "outputs"
    / "candidate_evidence_closure_atomic_projection_qwen_smoke_FORMAL009_r4"
    / "states"
    / "FORMAL_009.json"
)
REPLAY_FAB_DATA = (
    ROOT
    / ".blind_evaluation"
    / "formal_v1_candidate_r2"
    / "public"
    / "fab_data"
)


def _entity_ids(item: Evidence, entity_type: str) -> set[str]:
    return {
        entity.entity_id
        for entity in item.entities
        if entity.entity_type == entity_type
    }


def test_formal_009_atomic_scope_confirmation_offline_replay() -> None:
    if not REPLAY_STATE.exists() or not REPLAY_FAB_DATA.exists():
        pytest.skip("FORMAL_009 local replay inputs are not available")

    prior_state = RCAState.from_dict(
        json.loads(REPLAY_STATE.read_text(encoding="utf-8"))
    )
    repository = CsvFabRepository(REPLAY_FAB_DATA)
    source_lot_id = str(prior_state.job.source_lot_id)
    impact_output = FindImpactLotsTool(repository).run(
        ToolInput(
            tool_name="find_impact_lots",
            request_id="REPLAY_ATOMIC_IMPACT_SCOPE",
            requested_by=AgentKind.MES.value,
            parameters={"lot_id": source_lot_id},
        )
    )
    observed_lots = [str(item) for item in impact_output.data["impact_lots"]]
    selected_lots = [source_lot_id, *observed_lots]
    relevant_recipes = ("RCP_800D7305", "RCP_C5BEFB77", "RCP_B4CC2C8B")
    parameter_evidence: list[Evidence] = []
    for recipe in relevant_recipes:
        lane_id = f"lane:2500:EQ_44E1AC:EQ_44E1AC_CH03:{recipe}"
        parameter_evidence.extend(
            AnalyzeParameterShiftTool(repository).run(
                ToolInput(
                    tool_name="analyze_parameter_shift",
                    request_id=f"REPLAY_ATOMIC_FDC_{recipe}",
                    requested_by=AgentKind.FDC.value,
                    parameters={
                        "lot_ids": selected_lots,
                        "operation_no": "2500",
                        "equipment_id": "EQ_44E1AC",
                        "chamber_id": "EQ_44E1AC_CH03",
                        "recipe_id": recipe,
                        "lane_id": lane_id,
                    },
                )
            ).evidence
        )
    quality_evidence = SummarizeDefectWatTool(repository).run(
        ToolInput(
            tool_name="summarize_defect_wat",
            request_id="REPLAY_ATOMIC_QUALITY",
            requested_by=AgentKind.DEFECT_WAT.value,
            parameters={"lot_ids": selected_lots},
        )
    ).evidence
    incident_evidence = build_incident_observation_evidence(prior_state.job)
    evidence = list(
        {
            item.evidence_id: item
            for item in [
                *impact_output.evidence,
                *parameter_evidence,
                *quality_evidence,
                *incident_evidence,
            ]
        }.values()
    )

    relevant_exposure_ids = [
        item.evidence_id
        for item in impact_output.evidence
        if item.evidence_type
        in {
            EvidenceType.EQUIPMENT_EXPOSURE.value,
            EvidenceType.EXCURSION_WINDOW.value,
            EvidenceType.IMPACT_SCOPE.value,
        }
        and (
            not _entity_ids(item, EntityType.OPERATION.value)
            or _entity_ids(item, EntityType.OPERATION.value) == {"2500"}
        )
        and (
            not _entity_ids(item, EntityType.EQUIPMENT.value)
            or _entity_ids(item, EntityType.EQUIPMENT.value) == {"EQ_44E1AC"}
        )
        and (
            not _entity_ids(item, EntityType.CHAMBER.value)
            or _entity_ids(item, EntityType.CHAMBER.value) == {"EQ_44E1AC_CH03"}
        )
    ]
    parameter_ids = [
        item.evidence_id
        for item in parameter_evidence
        if item.evidence_type == EvidenceType.PARAMETER_DEVIATION.value
    ]
    outcome_ids = [
        item.evidence_id
        for item in quality_evidence
        if item.evidence_id
        in {
            "EV_DEFECT_R39",
            "EV_WAT_HIGH_CONTACT_R",
            "EV_METROLOGY_INCIDENT_OBSERVATION_CONTACT_RESISTANCE_DELTA",
        }
    ]
    candidate = CausalHypothesis(
        root_cause=(
            "EQ_44E1AC_CH03 oxygen-flow response and endpoint-overrun excursion "
            "during operation 2500"
        ),
        causal_explanation=(
            "The chamber-wide strip/ash excursion caused incomplete removal, left "
            "polymer residue in contact openings, and raised contact resistance."
        ),
        supporting_evidence_ids=tuple(
            dict.fromkeys(
                [
                    *relevant_exposure_ids,
                    *parameter_ids,
                    *outcome_ids,
                    *(item.evidence_id for item in incident_evidence),
                ]
            )
        ),
    )
    profile = CandidateSemanticProfile(
        candidate_id="REPLAY_FORMAL_009:llm:1",
        scope_relation="shared_effect",
        claimed_scope_kind="chamber",
        claimed_operation="2500",
        claimed_equipment="EQ_44E1AC",
        claimed_chamber="EQ_44E1AC_CH03",
        claimed_recipe=None,
        claimed_lane_ids=(
            "lane:2500:EQ_44E1AC:EQ_44E1AC_CH03:RCP_800D7305",
        ),
        comparison_lane_ids=tuple(
            f"lane:2500:EQ_44E1AC:EQ_44E1AC_CH03:{recipe}"
            for recipe in relevant_recipes
        ),
        mechanism_claim="Incomplete strip/ash removal leaves polymer residue.",
        primary_mechanism="incomplete removal leaves polymer residue",
        mechanism_relation="reference",
    )

    matrix = build_causal_evidence_matrix(
        candidate,
        evidence,
        semantic_profile=profile,
        causal_lanes=tuple(
            lane
            for lane in prior_state.causal_lanes
            if lane.operation == "2500"
            and lane.equipment == "EQ_44E1AC"
            and lane.chamber == "EQ_44E1AC_CH03"
            and lane.recipe in relevant_recipes
        ),
    )
    confirmation = confirm_candidate(
        matrix,
        alternative_search_status=(
            AlternativeSearchStatus.ALTERNATIVES_ELIMINATED.value
        ),
    )
    impact_gate = evaluate_impact_lot_gate(
        source_lot_id=source_lot_id,
        candidate=candidate,
        evidence=evidence,
        observed_impact_lots=observed_lots,
        authoritative_conclusion_status=confirmation.status,
        semantic_profile=profile,
    )

    assert matrix.mechanism_status == CausalClaimStatus.PLAUSIBLE.value
    assert matrix.mechanism_support_source == "observed_intermediate"
    assert matrix.claims["mechanism"].facts[
        "occurrence_only_intermediate_evidence_ids"
    ] == [item.evidence_id for item in incident_evidence]
    assert matrix.claims["scope"].status == CausalClaimStatus.SUPPORTED.value
    assert confirmation.status == "inconclusive"
    assert impact_gate["candidate_impact_lots"] == [
        "LOT_016F55DC01",
        "LOT_2EE4F67090",
        "LOT_706579D262",
        "LOT_E694A57AFB",
    ], impact_gate["rows"]
    assert impact_gate["confirmed_impact_lots"] == []
    assert impact_gate["publication_status"] == "withheld"
    rows = {item["lot_id"]: item for item in impact_gate["rows"]}
    assert rows["LOT_06EDD88B80"]["checks"]["outcome"] is False
    assert rows["LOT_39FBF686C1"]["checks"]["outcome"] is False
    assert all(
        item.evidence_type != EvidenceType.EXCURSION_WINDOW.value
        for item in incident_evidence
    )
