from __future__ import annotations

from typing import Any

from yield_rca_core.causal_adversarial import _challenge_output_contract
from yield_rca_core.causal_competition import build_candidate_competition_brief
from yield_rca_core.causal_competition_progression import progress_competition
from yield_rca_core.causal_evidence_gap import build_hypothesis_discrimination_gaps
from yield_rca_core.causal_evidence_matrix import (
    CausalClaimResult,
    CausalEvidenceMatrix,
)
from yield_rca_core.causal_hypothesis import CausalClaimStatus, CausalHypothesis
from yield_rca_core.causal_investigation_models import (
    ActionDecisionImpact,
    ActionSourceAvailability,
    ActionValueAssessment,
    ActionValueTier,
    AlternativeLaneResolution,
    AlternativeLaneResolutionStatus,
    CandidateChallenge,
    CandidateCompetitionStatus,
    CandidateCompetitionType,
    CandidateScopeRelation,
    CompetitionRequirement,
    CompetitionTrace,
    InvestigationGainReasonCode,
    InvestigationGainRecord,
    InvestigationGainType,
)
from yield_rca_core.hypothesis_candidate_generator import (
    _parse_candidate_semantic_profiles,
)


def lane(
    lane_id: str,
    *,
    recipe: str,
    process: bool,
    exposure: bool = True,
    parameter: str = "oxygen_flow_response_delta",
) -> dict[str, Any]:
    return {
        "lane_id": lane_id,
        "operation": "2500",
        "equipment": "EQ_A",
        "chamber": "EQ_A_CH01",
        "recipe": recipe,
        "parameter_scope": [parameter],
        "priority_score": 0.9,
        "investigation_status": "evidence_collected",
        "facts": {
            "shared_exposure": (
                [{"evidence_id": f"EV_EXP_{recipe}"}] if exposure else []
            ),
            "process_excursions": (
                [{"evidence_id": f"EV_PROC_{recipe}"}] if process else []
            ),
            "outcomes": [],
        },
    }


def matrix() -> CausalEvidenceMatrix:
    return CausalEvidenceMatrix(
        candidate=CausalHypothesis(
            root_cause="EQ_A_CH01 plasma response instability",
            causal_explanation=(
                "Observed oxygen-flow response instability changes plasma kinetics "
                "and can produce the product outcome."
            ),
            supporting_evidence_ids=("EV_PROC_RCP_A",),
        ),
        claims={
            "mechanism": CausalClaimResult(
                claim="mechanism",
                status=CausalClaimStatus.INCOMPLETE.value,
            )
        },
    )


def semantic_profile(
    *,
    relation: str,
    claimed_lane_ids: list[str],
    comparison_lane_ids: list[str],
    prediction_lane_ids: list[str],
) -> dict[str, object]:
    return {
        "candidate_index": 0,
        "claimed_scope": {
            "scope_relation": relation,
            "lane_ids": claimed_lane_ids,
        },
        "comparison_scope": {"lane_ids": comparison_lane_ids},
        "mechanism_claim": "Recipe response changes plasma kinetics.",
        "primary_mechanism": "Recipe response changes plasma kinetics.",
        "effect_modifier": None,
        "depends_on_candidate_index": None,
        "mechanism_relation": "reference",
        "distinguishing_predictions": [
            {
                "discriminator_kind": "parameter_anomaly",
                "lane_ids": prediction_lane_ids,
                "prediction": "The compared recipes have different parameter response.",
            }
        ],
    }


def test_focal_process_evidence_opens_scope_competition_before_comparison_is_complete(
) -> None:
    focal = lane("LANE_RCP_A", recipe="RCP_A", process=True)
    comparison = lane("LANE_RCP_B", recipe="RCP_B", process=False)

    brief = build_candidate_competition_brief(
        [focal, comparison],
        global_outcome_evidence_ids=["EV_OUTCOME"],
    )

    scope = brief["scope_groups"][0]
    assert brief["competition_requirement"] == CompetitionRequirement.MECHANISM_REQUIRED
    assert brief["competition_type"] == CandidateCompetitionType.MECHANISM
    assert brief["required_candidate_count"] == 2
    assert brief["scope_assessment_required"] is True
    assert brief["scope_assessment_status"] == "pending"
    assert brief["scope_opportunity_group_ids"] == ["scope_0"]
    assert brief["scope_ready_group_ids"] == []
    assert scope["scope_competition_opportunity"] is True
    assert scope["scope_evidence_complete"] is False
    assert scope["missing_process_recipe_ids"] == ["RCP_B"]
    assert scope["missing_process_lane_ids"] == ["LANE_RCP_B"]


def test_two_unobserved_recipes_do_not_force_low_quality_scope_candidates() -> None:
    brief = build_candidate_competition_brief(
        [
            lane("LANE_RCP_A", recipe="RCP_A", process=False),
            lane("LANE_RCP_B", recipe="RCP_B", process=False),
        ],
        global_outcome_evidence_ids=["EV_OUTCOME"],
    )

    scope = brief["scope_groups"][0]
    assert scope["scope_competition_opportunity"] is False
    assert brief["scope_opportunity_group_ids"] == []
    assert brief["required_candidate_count"] == 1
    assert brief["competition_requirement"] == (
        CompetitionRequirement.ALTERNATIVE_DISCOVERY_REQUIRED
    )


def test_shared_effect_requires_multiple_claimed_lanes() -> None:
    profiles, errors = _parse_candidate_semantic_profiles(
        [
            semantic_profile(
                relation=CandidateScopeRelation.SHARED_EFFECT,
                claimed_lane_ids=["LANE_RCP_A"],
                comparison_lane_ids=["LANE_RCP_A", "LANE_RCP_B"],
                prediction_lane_ids=["LANE_RCP_A", "LANE_RCP_B"],
            )
        ],
        request_id="REQ_SCOPE_SEMANTICS",
        surviving_candidate_indexes=[0],
        known_lane_ids={"LANE_RCP_A", "LANE_RCP_B"},
    )

    assert profiles == ()
    assert any("shared_effect" in error for error in errors)


def test_non_outcome_prediction_allows_empty_lane_effect_expectations() -> None:
    profile = semantic_profile(
        relation=CandidateScopeRelation.FOCAL_ONLY,
        claimed_lane_ids=["LANE_RCP_A"],
        comparison_lane_ids=["LANE_RCP_A", "LANE_RCP_B"],
        prediction_lane_ids=["LANE_RCP_A", "LANE_RCP_B"],
    )
    profile["distinguishing_predictions"][0]["lane_effect_expectations"] = []

    profiles, errors = _parse_candidate_semantic_profiles(
        [profile],
        request_id="REQ_NON_OUTCOME_EMPTY_EFFECTS",
        surviving_candidate_indexes=[0],
        known_lane_ids={"LANE_RCP_A", "LANE_RCP_B"},
    )

    assert errors == ()
    assert len(profiles) == 1


def test_qwen_product_outcome_requires_lane_effect_expectations() -> None:
    profile = semantic_profile(
        relation=CandidateScopeRelation.FOCAL_ONLY,
        claimed_lane_ids=["LANE_RCP_A"],
        comparison_lane_ids=["LANE_RCP_A", "LANE_RCP_B"],
        prediction_lane_ids=["LANE_RCP_A", "LANE_RCP_B"],
    )
    profile["distinguishing_predictions"] = [
        {
            "discriminator_kind": "product_outcome",
            "lane_ids": ["LANE_RCP_A", "LANE_RCP_B"],
            "prediction": "Only RCP_A produces the product outcome.",
        }
    ]

    profiles, errors = _parse_candidate_semantic_profiles(
        [profile],
        request_id="REQ_PRODUCT_OUTCOME_REQUIRES_EFFECTS",
        surviving_candidate_indexes=[0],
        known_lane_ids={"LANE_RCP_A", "LANE_RCP_B"},
    )

    assert profiles == ()
    assert any("lane_effect_expectations" in error for error in errors)


def test_shared_effect_rejects_lane_split_product_outcome_predictions() -> None:
    profile = semantic_profile(
        relation=CandidateScopeRelation.SHARED_EFFECT,
        claimed_lane_ids=["LANE_RCP_A", "LANE_RCP_B"],
        comparison_lane_ids=["LANE_RCP_A", "LANE_RCP_B"],
        prediction_lane_ids=["LANE_RCP_A", "LANE_RCP_B"],
    )
    profile["distinguishing_predictions"] = [
        {
            "discriminator_kind": "product_outcome",
            "lane_ids": ["LANE_RCP_A"],
            "prediction": "RCP_A produces the product failure.",
            "lane_effect_expectations": [
                {
                    "lane_id": "LANE_RCP_A",
                    "effect_key": "HIGH_CONTACT_R",
                    "effect_state": "present",
                }
            ],
        },
        {
            "discriminator_kind": "product_outcome",
            "lane_ids": ["LANE_RCP_B"],
            "prediction": "RCP_B does not produce the product failure.",
            "lane_effect_expectations": [
                {
                    "lane_id": "LANE_RCP_B",
                    "effect_key": "HIGH_CONTACT_R",
                    "effect_state": "absent",
                }
            ],
        },
    ]

    profiles, errors = _parse_candidate_semantic_profiles(
        [profile],
        request_id="REQ_SHARED_EFFECT_OUTCOME_CONFLICT",
        surviving_candidate_indexes=[0],
        known_lane_ids={"LANE_RCP_A", "LANE_RCP_B"},
    )

    assert profiles == ()
    assert any(
        "shared_effect claimed Lanes must predict the same product effect"
        in error
        for error in errors
    )


def test_shared_effect_rejects_conflicting_claimed_lane_effects() -> None:
    profile = semantic_profile(
        relation=CandidateScopeRelation.SHARED_EFFECT,
        claimed_lane_ids=["LANE_RCP_A", "LANE_RCP_B"],
        comparison_lane_ids=["LANE_RCP_A", "LANE_RCP_B"],
        prediction_lane_ids=["LANE_RCP_A", "LANE_RCP_B"],
    )
    profile["distinguishing_predictions"] = [
        {
            "discriminator_kind": "product_outcome",
            "lane_ids": ["LANE_RCP_A", "LANE_RCP_B"],
            "prediction": "The compared Lanes have different product outcomes.",
            "lane_effect_expectations": [
                {
                    "lane_id": "LANE_RCP_A",
                    "effect_key": "HIGH_CONTACT_R",
                    "effect_state": "present",
                },
                {
                    "lane_id": "LANE_RCP_B",
                    "effect_key": "HIGH_CONTACT_R",
                    "effect_state": "absent",
                },
            ],
        }
    ]

    profiles, errors = _parse_candidate_semantic_profiles(
        [profile],
        request_id="REQ_SHARED_EFFECT_CONFLICTING_EFFECTS",
        surviving_candidate_indexes=[0],
        known_lane_ids={"LANE_RCP_A", "LANE_RCP_B"},
    )

    assert profiles == ()
    assert any(
        "shared_effect claimed Lanes must predict the same product effect"
        in error
        for error in errors
    )


def test_shared_effect_allows_different_effect_for_extra_comparison_control() -> None:
    profile = semantic_profile(
        relation=CandidateScopeRelation.SHARED_EFFECT,
        claimed_lane_ids=["LANE_RCP_A", "LANE_RCP_B"],
        comparison_lane_ids=["LANE_RCP_A", "LANE_RCP_B", "LANE_CONTROL"],
        prediction_lane_ids=["LANE_RCP_A", "LANE_RCP_B", "LANE_CONTROL"],
    )
    profile["distinguishing_predictions"] = [
        {
            "discriminator_kind": "product_outcome",
            "lane_ids": ["LANE_RCP_A", "LANE_RCP_B", "LANE_CONTROL"],
            "prediction": "The claimed Lanes fail while the control remains normal.",
            "lane_effect_expectations": [
                {
                    "lane_id": "LANE_RCP_A",
                    "effect_key": "HIGH_CONTACT_R",
                    "effect_state": "present",
                },
                {
                    "lane_id": "LANE_RCP_B",
                    "effect_key": "high-contact-r",
                    "effect_state": "present",
                },
                {
                    "lane_id": "LANE_CONTROL",
                    "effect_key": "HIGH_CONTACT_R",
                    "effect_state": "absent",
                },
            ],
        }
    ]

    profiles, errors = _parse_candidate_semantic_profiles(
        [profile],
        request_id="REQ_SHARED_EFFECT_COMPARISON_CONTROL",
        surviving_candidate_indexes=[0],
        known_lane_ids={"LANE_RCP_A", "LANE_RCP_B", "LANE_CONTROL"},
    )

    assert errors == ()
    assert len(profiles) == 1


def test_differential_sensitivity_allows_lane_split_product_outcomes() -> None:
    profile = semantic_profile(
        relation=CandidateScopeRelation.DIFFERENTIAL_SENSITIVITY,
        claimed_lane_ids=["LANE_RCP_A"],
        comparison_lane_ids=["LANE_RCP_A", "LANE_RCP_B"],
        prediction_lane_ids=["LANE_RCP_A", "LANE_RCP_B"],
    )
    profile["distinguishing_predictions"] = [
        {
            "discriminator_kind": "product_outcome",
            "lane_ids": ["LANE_RCP_A"],
            "prediction": "RCP_A produces the more severe product failure.",
            "lane_effect_expectations": [
                {
                    "lane_id": "LANE_RCP_A",
                    "effect_key": "HIGH_CONTACT_R",
                    "effect_state": "increased",
                }
            ],
        },
        {
            "discriminator_kind": "product_outcome",
            "lane_ids": ["LANE_RCP_B"],
            "prediction": "RCP_B remains within the product limit.",
            "lane_effect_expectations": [
                {
                    "lane_id": "LANE_RCP_B",
                    "effect_key": "HIGH_CONTACT_R",
                    "effect_state": "unchanged",
                }
            ],
        },
    ]

    profiles, errors = _parse_candidate_semantic_profiles(
        [profile],
        request_id="REQ_DIFFERENTIAL_OUTCOME",
        surviving_candidate_indexes=[0],
        known_lane_ids={"LANE_RCP_A", "LANE_RCP_B"},
    )

    assert errors == ()
    assert len(profiles) == 1
    assert profiles[0].scope_relation == (
        CandidateScopeRelation.DIFFERENTIAL_SENSITIVITY
    )


def test_differential_sensitivity_accepts_claimed_subset_and_full_comparison() -> None:
    profiles, errors = _parse_candidate_semantic_profiles(
        [
            semantic_profile(
                relation=CandidateScopeRelation.DIFFERENTIAL_SENSITIVITY,
                claimed_lane_ids=["LANE_RCP_A"],
                comparison_lane_ids=["LANE_RCP_A", "LANE_RCP_B"],
                prediction_lane_ids=["LANE_RCP_A", "LANE_RCP_B"],
            )
        ],
        request_id="REQ_SCOPE_SEMANTICS",
        surviving_candidate_indexes=[0],
        known_lane_ids={"LANE_RCP_A", "LANE_RCP_B"},
    )

    assert errors == ()
    assert len(profiles) == 1
    assert profiles[0].scope_relation == (
        CandidateScopeRelation.DIFFERENTIAL_SENSITIVITY
    )


def test_resolved_scope_prediction_must_cover_every_comparison_lane() -> None:
    profiles, errors = _parse_candidate_semantic_profiles(
        [
            semantic_profile(
                relation=CandidateScopeRelation.FOCAL_ONLY,
                claimed_lane_ids=["LANE_RCP_A"],
                comparison_lane_ids=["LANE_RCP_A", "LANE_RCP_B"],
                prediction_lane_ids=["LANE_RCP_A"],
            )
        ],
        request_id="REQ_SCOPE_SEMANTICS",
        surviving_candidate_indexes=[0],
        known_lane_ids={"LANE_RCP_A", "LANE_RCP_B"},
    )

    assert profiles == ()
    assert any("prediction Lane coverage" in error for error in errors)


def test_scope_comparison_gap_survives_an_unrelated_challenge_lane_selection() -> None:
    focal = lane("LANE_RCP_A", recipe="RCP_A", process=True)
    comparison = lane("LANE_RCP_B", recipe="RCP_B", process=False)
    unrelated = {
        **lane("LANE_OTHER", recipe="RCP_OTHER", process=False),
        "operation": "1000",
        "equipment": "EQ_B",
        "chamber": "EQ_B_CH02",
        "parameter_scope": ["hotplate_settle_time_delta"],
    }
    brief = build_candidate_competition_brief(
        [focal, comparison, unrelated],
        global_outcome_evidence_ids=["EV_OUTCOME"],
    )
    challenge = CandidateChallenge(
        candidate_id="CANDIDATE_A",
        strongest_alternative_lane_id="LANE_OTHER",
        distinguishing_gap_ids=(
            "candidate_0.hypothesis_discrimination.parameter_anomaly",
        ),
        challenge_explanation="Probe a separate direction.",
        status="unresolved",
    )

    gaps = build_hypothesis_discrimination_gaps(
        [matrix()],
        candidate_challenges=[challenge],
        causal_lanes=[focal, comparison, unrelated],
        candidate_ids=["CANDIDATE_A"],
        competition_brief=brief,
    )

    scope_gap = next(
        gap for gap in gaps if gap.get("gap_origin") == "scope_competition"
    )
    assert scope_gap["target_scope"]["lane_id"] == "LANE_RCP_B"
    assert scope_gap["discriminator_kind"] == "parameter_anomaly"
    assert scope_gap["decision_impact"] == "ranking"
    assert scope_gap["information_gain"] >= 0.6
    assert scope_gap["challenge_selected"] is False


def test_scope_challenge_contract_excludes_unrelated_direction_lanes() -> None:
    scope_gap = {
        "gap_id": "scope_0.scope_discrimination.parameter_anomaly.LANE_B",
        "gap_type": "hypothesis_discrimination",
        "gap_origin": "scope_competition",
        "candidate_id": "",
        "candidate_ids": ["CANDIDATE_A", "CANDIDATE_B"],
        "target_scope": {"lane_id": "LANE_RCP_B"},
    }

    contract = _challenge_output_contract(
        candidate_ids=["CANDIDATE_A", "CANDIDATE_B"],
        active_lane_ids=["LANE_RCP_A", "LANE_RCP_B", "LANE_OTHER_DIRECTION"],
        evidence_gaps=[scope_gap],
        candidate_competition={
            "competition_requirement": "scope_required",
            "semantic_profiles_complete": True,
        },
    )

    assert contract["required_challenge_kind"] == "scope"
    assert contract["allowed_evidence_probe_lane_ids"] == ["LANE_RCP_B"]
    assert contract["allowed_gap_ids_by_candidate"] == {
        "CANDIDATE_A": [scope_gap["gap_id"]],
        "CANDIDATE_B": [scope_gap["gap_id"]],
    }
    assert contract["allowed_gap_ids_by_candidate_and_lane"] == {
        "CANDIDATE_A": {"LANE_RCP_B": [scope_gap["gap_id"]]},
        "CANDIDATE_B": {"LANE_RCP_B": [scope_gap["gap_id"]]},
    }


def test_challenge_contract_never_recommends_a_gap_for_an_inapplicable_lane(
) -> None:
    parameter_gap = {
        "gap_id": "candidate_0.hypothesis_discrimination.parameter_anomaly",
        "gap_type": "hypothesis_discrimination",
        "candidate_id": "CANDIDATE_A",
        "lane_binding": "challenge_selected",
        "applicable_lane_ids": ["LANE_RCP_A"],
        "information_gain_by_lane": {"LANE_RCP_A": 0.8},
    }
    outcome_gap = {
        "gap_id": "candidate_0.hypothesis_discrimination.product_outcome",
        "gap_type": "hypothesis_discrimination",
        "candidate_id": "CANDIDATE_A",
        "lane_binding": "challenge_selected",
        "applicable_lane_ids": ["LANE_RCP_A", "LANE_RCP_B"],
        "information_gain_by_lane": {
            "LANE_RCP_A": 0.75,
            "LANE_RCP_B": 0.75,
        },
    }

    contract = _challenge_output_contract(
        candidate_ids=["CANDIDATE_A"],
        active_lane_ids=["LANE_RCP_A", "LANE_RCP_B"],
        evidence_gaps=[parameter_gap, outcome_gap],
        candidate_competition={
            "competition_requirement": "scope_required",
            "semantic_profiles_complete": False,
        },
    )

    allowed = contract["allowed_gap_ids_by_candidate_and_lane"]["CANDIDATE_A"]
    highest = contract[
        "highest_information_gain_gap_ids_by_candidate_and_lane"
    ]["CANDIDATE_A"]
    assert allowed["LANE_RCP_A"] == [
        parameter_gap["gap_id"],
        outcome_gap["gap_id"],
    ]
    assert allowed["LANE_RCP_B"] == [outcome_gap["gap_id"]]
    assert highest["LANE_RCP_A"] == [parameter_gap["gap_id"]]
    assert highest["LANE_RCP_B"] == [outcome_gap["gap_id"]]


def test_blocked_unrelated_lane_does_not_hide_remaining_scope_action() -> None:
    blocked_lane = "LANE_OTHER_DIRECTION"
    scope_lane = "LANE_RCP_B"
    missing_gain = InvestigationGainRecord(
        action_id="ACTION_OTHER_MISSING",
        action_kind="inspect_fdc_spc",
        scope_fingerprint="action-scope:other-missing",
        gain_type=InvestigationGainType.STATE_GAIN,
        evidence_ids=("EV_OTHER_MISSING",),
        candidate_id="CANDIDATE_A",
        lane_id=blocked_lane,
        gap_id="candidate_0.hypothesis_discrimination.parameter_anomaly",
        discriminator_kind="parameter_anomaly",
        source="fdc",
        reason_code=InvestigationGainReasonCode.UNAVAILABLE_SOURCE,
    )
    scope_assessment = ActionValueAssessment(
        option_id="inspect_fdc_spc:scope_0",
        action_kind="inspect_fdc_spc",
        scope_fingerprint="action-scope:scope-comparison",
        static_information_gain=0.8,
        source_availability=ActionSourceAvailability.UNKNOWN,
        decision_impact=ActionDecisionImpact.RANKING,
        estimated_tool_cost=4,
        estimated_llm_cost=1,
        remaining_tool_budget=12,
        high_value=True,
        eligible=True,
        value_tier=ActionValueTier.HIGH,
        lane_id=scope_lane,
        gap_id="scope_0.scope_discrimination.parameter_anomaly",
        discriminator_kind="parameter_anomaly",
    )
    trace = CompetitionTrace(
        active_lane_ids=(scope_lane,),
        represented_lane_ids=(blocked_lane, scope_lane),
        unresolved_lane_ids=(scope_lane,),
        blocked_lane_ids=(blocked_lane,),
        lane_resolutions=(
            AlternativeLaneResolution(
                lane_id=blocked_lane,
                status=AlternativeLaneResolutionStatus.BLOCKED,
                candidate_id="CANDIDATE_A",
                evidence_ids=("EV_OTHER_MISSING",),
                reason_code="blocked_by_missing_data",
            ),
            AlternativeLaneResolution(
                lane_id=scope_lane,
                status=AlternativeLaneResolutionStatus.UNRESOLVED,
            ),
        ),
        competition_requirement=CompetitionRequirement.SCOPE_REQUIRED,
        competition_status=CandidateCompetitionStatus.BLOCKED_BY_MISSING_DATA,
        terminal_reason=(
            "no_high_value_action_after_required_source_unavailable"
        ),
    )

    result = progress_competition(
        trace=trace,
        authoritative_details={
            "conclusion_status": "inconclusive",
            "ranked_candidates": [
                {
                    "candidate_id": "CANDIDATE_A",
                    "root_cause": "An unresolved focal Scope hypothesis.",
                    "status": "candidate",
                }
            ],
            "causal_evidence_gaps": [
                {
                    "gap_id": scope_assessment.gap_id,
                    "status": "unresolved",
                }
            ],
            "confirmation_gate": {"status": "inconclusive"},
        },
        action_value_assessments=[scope_assessment],
        gain_history=[missing_gain],
        force_terminal=True,
    )

    assert result.trace.competition_status == CandidateCompetitionStatus.ACTIVE
    assert result.terminal is False
    assert result.trace.blocked_lane_ids == (blocked_lane,)
    assert result.trace.unresolved_lane_ids == (scope_lane,)
