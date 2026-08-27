from __future__ import annotations

from typing import Any

from yield_rca_core.causal_competition import (
    build_candidate_competition_brief,
    causal_direction_identity,
    select_diverse_lane_ids,
)
from yield_rca_core.causal_investigation_models import (
    CandidateCompetitionType,
    CompetitionRequirement,
)
from yield_rca_core.models import AgentFinding, AgentKind
from yield_rca_core.next_action_planner import _known_causal_lane_ids
from yield_rca_core.rca_reasoning_agent import _lane_context


def lane(
    lane_id: str,
    *,
    operation: str,
    equipment: str,
    chamber: str,
    recipe: str,
    parameter: str,
    priority: float,
    exposure: bool = True,
    process: bool = True,
) -> dict[str, Any]:
    facts = {
        "shared_exposure": (
            [{"evidence_id": f"EV_EXP_{lane_id}"}] if exposure else []
        ),
        "process_excursions": (
            [{"evidence_id": f"EV_PROC_{lane_id}"}] if process else []
        ),
        "outcomes": [],
    }
    return {
        "lane_id": lane_id,
        "operation": operation,
        "equipment": equipment,
        "chamber": chamber,
        "recipe": recipe,
        "parameter_scope": [parameter],
        "priority_score": priority,
        "investigation_status": "evidence_collected",
        "facts": facts,
    }


def test_lane_selection_preserves_direction_and_scope_diversity() -> None:
    primary = lane(
        "LANE_PRIMARY",
        operation="2500",
        equipment="EQ_A",
        chamber="EQ_A_CH01",
        recipe="RCP_A",
        parameter="oxygen_flow",
        priority=0.9,
    )
    same_direction = lane(
        "LANE_SCOPE",
        operation="2500",
        equipment="EQ_A",
        chamber="EQ_A_CH01",
        recipe="RCP_B",
        parameter="oxygen_flow",
        priority=0.8,
    )
    different_direction = lane(
        "LANE_DIRECTION_B",
        operation="6000",
        equipment="EQ_B",
        chamber="EQ_B_CH02",
        recipe="RCP_C",
        parameter="clean_time",
        priority=0.7,
    )

    selected = select_diverse_lane_ids(
        [primary, same_direction, different_direction],
        limit=3,
    )

    assert selected == ("LANE_PRIMARY", "LANE_DIRECTION_B", "LANE_SCOPE")
    assert causal_direction_identity(primary) == causal_direction_identity(
        same_direction
    )
    assert causal_direction_identity(primary) != causal_direction_identity(
        different_direction
    )


def test_two_evidence_bounded_directions_require_two_candidates() -> None:
    brief = build_candidate_competition_brief(
        [
            lane(
                "LANE_A",
                operation="2500",
                equipment="EQ_A",
                chamber="EQ_A_CH01",
                recipe="RCP_A",
                parameter="oxygen_flow",
                priority=0.9,
            ),
            lane(
                "LANE_B",
                operation="6000",
                equipment="EQ_B",
                chamber="EQ_B_CH02",
                recipe="RCP_B",
                parameter="clean_time",
                priority=0.8,
            ),
        ],
        global_outcome_evidence_ids=["EV_OUTCOME"],
    )

    assert brief["competition_requirement"] == (
        CompetitionRequirement.DIRECTION_REQUIRED.value
    )
    assert brief["competition_type"] == (
        CandidateCompetitionType.CAUSAL_DIRECTION.value
    )
    assert brief["required_candidate_count"] == 2
    assert len(brief["eligible_direction_bundle_ids"]) == 2


def test_same_direction_two_recipe_observations_keep_root_mechanism_competition() -> None:
    brief = build_candidate_competition_brief(
        [
            lane(
                "LANE_RCP_A",
                operation="2500",
                equipment="EQ_A",
                chamber="EQ_A_CH01",
                recipe="RCP_A",
                parameter="oxygen_flow",
                priority=0.9,
            ),
            lane(
                "LANE_RCP_B",
                operation="2500",
                equipment="EQ_A",
                chamber="EQ_A_CH01",
                recipe="RCP_B",
                parameter="oxygen_flow",
                priority=0.8,
            ),
        ],
        global_outcome_evidence_ids=["EV_OUTCOME"],
    )

    assert brief["competition_requirement"] == (
        CompetitionRequirement.MECHANISM_REQUIRED.value
    )
    assert brief["competition_type"] == CandidateCompetitionType.MECHANISM.value
    assert brief["scope_assessment_required"] is True
    assert brief["scope_ready_group_ids"] == ["scope_0"]


def test_unobserved_alternative_does_not_displace_bounded_mechanism_competition() -> None:
    brief = build_candidate_competition_brief(
        [
            lane(
                "LANE_A",
                operation="2500",
                equipment="EQ_A",
                chamber="EQ_A_CH01",
                recipe="RCP_A",
                parameter="oxygen_flow",
                priority=0.9,
            ),
            lane(
                "LANE_B",
                operation="6000",
                equipment="EQ_B",
                chamber="EQ_B_CH02",
                recipe="RCP_B",
                parameter="clean_time",
                priority=0.8,
                process=False,
            ),
        ],
        global_outcome_evidence_ids=["EV_OUTCOME"],
    )

    assert brief["competition_requirement"] == (
        CompetitionRequirement.MECHANISM_REQUIRED.value
    )
    assert brief["required_candidate_count"] == 2
    assert brief["unbounded_direction_bundle_ids"] == ["direction_1"]


def test_formal_009_structure_replay_treats_recipes_as_scope_not_directions() -> None:
    recipe_lanes = [
        lane(
            f"lane:2500:EQ_44E1AC:EQ_44E1AC_CH03:{recipe}",
            operation="2500",
            equipment="EQ_44E1AC",
            chamber="EQ_44E1AC_CH03",
            recipe=recipe,
            parameter="oxygen_flow_response_delta",
            priority=0.9 - index * 0.05,
        )
        for index, recipe in enumerate(
            ("RCP_800D7305", "RCP_C5BEFB77", "RCP_B4CC2C8B")
        )
    ]
    distinct_direction = lane(
        "lane:6000:EQ_OTHER:EQ_OTHER_CH01:RCP_OTHER",
        operation="6000",
        equipment="EQ_OTHER",
        chamber="EQ_OTHER_CH01",
        recipe="RCP_OTHER",
        parameter="clean_time_delta",
        priority=0.7,
    )

    brief = build_candidate_competition_brief(
        [*recipe_lanes, distinct_direction],
        global_outcome_evidence_ids=["EV_CONTACT_RESISTANCE"],
    )

    assert len(brief["direction_bundles"]) == 2
    primary_scope = next(
        item
        for item in brief["scope_groups"]
        if "RCP_800D7305" in item["recipes"]
    )
    assert set(primary_scope["recipes"]) == {
        "RCP_800D7305",
        "RCP_C5BEFB77",
        "RCP_B4CC2C8B",
    }
    assert primary_scope["scope_competition_ready"] is True


def test_rca_runtime_uses_same_diverse_active_lane_snapshot() -> None:
    eliminated = lane(
        "lane:eliminated:EQ_BLOCKED:CH_BLOCKED:RCP_BLOCKED",
        operation="9999",
        equipment="EQ_BLOCKED",
        chamber="CH_BLOCKED",
        recipe="RCP_BLOCKED",
        parameter="blocked_parameter",
        priority=1.0,
    )
    eliminated["investigation_status"] = "eliminated"
    recipe_lanes = [
        lane(
            f"lane:2500:EQ_44E1AC:EQ_44E1AC_CH03:{recipe}",
            operation="2500",
            equipment="EQ_44E1AC",
            chamber="EQ_44E1AC_CH03",
            recipe=recipe,
            parameter="oxygen_flow_response_delta",
            priority=0.9 - index * 0.05,
        )
        for index, recipe in enumerate(
            ("RCP_800D7305", "RCP_C5BEFB77", "RCP_B4CC2C8B")
        )
    ]
    different_direction = lane(
        "lane:1000:EQ_4D35CA:EQ_4D35CA_CH02:RCP_CF41B53F",
        operation="1000",
        equipment="EQ_4D35CA",
        chamber="EQ_4D35CA_CH02",
        recipe="RCP_CF41B53F",
        parameter="hotplate_settle_time_delta",
        priority=0.4,
    )
    overflow = [
        lane(
            f"lane:{3000 + index}:EQ_{index}:CH_{index}:RCP_{index}",
            operation=str(3000 + index),
            equipment=f"EQ_{index}",
            chamber=f"CH_{index}",
            recipe=f"RCP_{index}",
            parameter=f"parameter_{index}",
            priority=0.1,
        )
        for index in range(95)
    ]
    finding = AgentFinding(
        finding_id="FINDING_FORMAL_009_LANES",
        agent=AgentKind.MES.value,
        summary="FORMAL_009-like Lane inventory.",
        confidence=0.8,
        evidence_ids=["EV_FORMAL_009_LANES"],
        details={
            "lane_candidates": [
                eliminated,
                *recipe_lanes,
                different_direction,
                *overflow,
            ]
        },
    )

    all_ids, active_ids, eliminated_ids, _, active_contexts = _lane_context([finding])

    assert len(all_ids) == 100
    assert eliminated_ids == [eliminated["lane_id"]]
    assert active_ids == [
        recipe_lanes[0]["lane_id"],
        different_direction["lane_id"],
        recipe_lanes[1]["lane_id"],
    ]
    assert [item["lane_id"] for item in active_contexts] == active_ids
    assert len(active_contexts) == 3
    assert list(_known_causal_lane_ids([finding])) == active_ids
