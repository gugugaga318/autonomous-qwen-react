from __future__ import annotations

from dataclasses import replace

from yield_rca_core.causal_investigation_models import (
    CandidateChallenge,
    CausalLaneRecord,
    ChallengeStatus,
    InvestigationLaneStatus,
    LaneLifecycleStatus,
)
from yield_rca_core.causal_lane_lifecycle import (
    apply_active_lane_snapshot,
    causal_lane_scope_identity,
    lane_is_searchable,
    mark_lanes_challenged,
    reconcile_lane_inventory,
)
from yield_rca_core.evidence_models import Evidence, EvidenceSourceType
from yield_rca_core.models import AgentFinding, AgentKind, RCAJob, RCAState
from yield_rca_core.next_action_planner import _known_causal_lane_ids
from yield_rca_core.supervisor import (
    _update_causal_lane_state,
    _update_competition_state,
)


def _lane(
    lane_id: str = "lane:4000:EQ_A:EQ_A_CH01:RCP_A",
    *,
    recipe: str = "RCP_A",
    parameter_scope: tuple[str, ...] = ("pressure",),
    evidence_ids: tuple[str, ...] = ("EV_1",),
    status: str = InvestigationLaneStatus.EVIDENCE_COLLECTED.value,
    lifecycle_status: str = LaneLifecycleStatus.CREATED.value,
    reason: str | None = None,
) -> CausalLaneRecord:
    return CausalLaneRecord(
        lane_id=lane_id,
        operation="4000",
        equipment="EQ_A",
        chamber="EQ_A_CH01",
        recipe=recipe,
        parameter_scope=parameter_scope,
        exposed_lot_ids=("LOT_1",),
        initial_evidence_ids=evidence_ids,
        priority_score=0.9,
        investigation_status=status,
        lifecycle_status=lifecycle_status,
        pruned_reason=reason,
    )


def _evidence(evidence_id: str) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        source_type=EvidenceSourceType.MES.value,
        source_id=f"mes:{evidence_id}",
        summary=f"Typed observation {evidence_id}.",
    )


def test_fifty_repeat_discoveries_keep_one_canonical_lane_and_all_evidence() -> None:
    inventory: tuple[CausalLaneRecord, ...] = ()
    for index in range(50):
        inventory = reconcile_lane_inventory(
            inventory,
            [
                _lane(
                    parameter_scope=("pressure", f"signal_{index % 3}"),
                    evidence_ids=(f"EV_{index}",),
                )
            ],
        )

    assert len(inventory) == 1
    assert inventory[0].discovery_count == 50
    assert len(inventory[0].initial_evidence_ids) == 50
    assert set(inventory[0].parameter_scope) == {
        "pressure",
        "signal_0",
        "signal_1",
        "signal_2",
    }


def test_alias_with_same_scope_merges_but_different_recipe_remains_distinct() -> None:
    canonical = _lane()
    alias = _lane(
        "MES_ALIAS_4000_A",
        parameter_scope=("temperature",),
        evidence_ids=("EV_2",),
    )
    other_recipe = _lane(
        "lane:4000:EQ_A:EQ_A_CH01:RCP_B",
        recipe="RCP_B",
        evidence_ids=("EV_3",),
    )

    inventory = reconcile_lane_inventory([canonical], [alias, other_recipe])

    assert len(inventory) == 2
    merged = inventory[0]
    assert merged.lane_id == canonical.lane_id
    assert merged.merged_from_lane_ids == (alias.lane_id,)
    assert set(merged.initial_evidence_ids) == {"EV_1", "EV_2"}
    assert merged.lifecycle_history[-1].lane_id == alias.lane_id
    assert merged.lifecycle_history[-1].to_status == LaneLifecycleStatus.MERGED.value
    assert causal_lane_scope_identity(merged) != causal_lane_scope_identity(
        other_recipe
    )


def test_terminal_lane_reopens_only_for_new_typed_evidence() -> None:
    eliminated = _lane(
        status=InvestigationLaneStatus.ELIMINATED.value,
        lifecycle_status=LaneLifecycleStatus.ELIMINATED.value,
        reason="Alternative was contradicted.",
    )

    unchanged = reconcile_lane_inventory([eliminated], [_lane()])[0]
    assert unchanged.investigation_status == InvestigationLaneStatus.ELIMINATED.value
    assert unchanged.reactivation_count == 0

    reopened = reconcile_lane_inventory(
        [unchanged],
        [_lane(evidence_ids=("EV_1", "EV_2"))],
    )[0]
    assert reopened.investigation_status == (
        InvestigationLaneStatus.EVIDENCE_COLLECTED.value
    )
    assert reopened.lifecycle_status == LaneLifecycleStatus.CREATED.value
    assert reopened.reactivation_count == 1
    assert reopened.pruned_reason is None
    assert reopened.lifecycle_history[-1].evidence_ids == ("EV_2",)


def test_active_snapshot_and_challenge_do_not_reactivate_terminal_lanes() -> None:
    active = _lane("LANE_ACTIVE")
    deferred = _lane("LANE_DEFERRED", recipe="RCP_B")
    blocked = _lane(
        "LANE_BLOCKED",
        recipe="RCP_C",
        status=InvestigationLaneStatus.BLOCKED.value,
        lifecycle_status=LaneLifecycleStatus.BLOCKED.value,
        reason="Source unavailable.",
    )
    selected = apply_active_lane_snapshot(
        [active, deferred, blocked],
        active_lane_ids=[active.lane_id, blocked.lane_id],
    )
    by_id = {item.lane_id: item for item in selected}
    assert by_id[active.lane_id].lifecycle_status == LaneLifecycleStatus.ACTIVE.value
    assert by_id[deferred.lane_id].lifecycle_status == (
        LaneLifecycleStatus.DEFERRED.value
    )
    assert by_id[blocked.lane_id].lifecycle_status == (
        LaneLifecycleStatus.BLOCKED.value
    )

    challenged = mark_lanes_challenged(
        selected,
        challenged_lane_ids=[active.lane_id, blocked.lane_id],
    )
    by_id = {item.lane_id: item for item in challenged}
    assert by_id[active.lane_id].lifecycle_status == (
        LaneLifecycleStatus.CHALLENGED.value
    )
    assert by_id[blocked.lane_id].lifecycle_status == (
        LaneLifecycleStatus.BLOCKED.value
    )
    assert lane_is_searchable(by_id[active.lane_id]) is True
    assert lane_is_searchable(by_id[blocked.lane_id]) is False


def test_legacy_lane_deserialization_derives_lifecycle_without_new_fields() -> None:
    legacy = _lane().to_dict()
    for field in (
        "lifecycle_status",
        "merged_from_lane_ids",
        "merged_into_lane_id",
        "discovery_count",
        "reactivation_count",
        "last_transition_reason",
        "lifecycle_history",
    ):
        legacy.pop(field)

    restored = CausalLaneRecord.from_dict(legacy)

    assert restored.lifecycle_status == LaneLifecycleStatus.ACTIVE.value
    assert restored.discovery_count == 1
    assert restored.lifecycle_history == ()


def test_supervisor_persists_canonical_inventory_for_planner() -> None:
    lane_a = _lane().to_dict()
    lane_alias = _lane(
        "MES_ALIAS_4000_A",
        parameter_scope=("temperature",),
        evidence_ids=("EV_2",),
    ).to_dict()
    finding = AgentFinding(
        finding_id="F_MES_LANES",
        agent=AgentKind.MES.value,
        summary="Discovered exposure Lanes.",
        confidence=0.9,
        evidence_ids=["EV_1", "EV_2"],
        details={"lane_candidates": [lane_a, lane_alias]},
    )
    state = RCAState(
        job=RCAJob(job_id="JOB_LANE_LIFECYCLE", user_query="Find root cause."),
        evidence=[_evidence("EV_1"), _evidence("EV_2")],
        findings=[finding],
    )

    updated = _update_causal_lane_state(state, finding)

    assert len(updated.causal_lanes) == 1
    authoritative = updated.findings[0]
    assert authoritative.details["lane_inventory_authoritative"] is True
    assert authoritative.details["raw_lane_candidate_count"] == 2
    assert authoritative.details["canonical_lane_count"] == 1
    assert _known_causal_lane_ids(updated.findings) == (
        updated.causal_lanes[0].lane_id,
    )


def test_validated_challenge_updates_lane_and_authoritative_planner_inventory() -> None:
    lane_payload = _lane().to_dict()
    mes_finding = AgentFinding(
        finding_id="F_MES_CHALLENGE",
        agent=AgentKind.MES.value,
        summary="Discovered exposure Lane.",
        confidence=0.9,
        evidence_ids=["EV_1"],
        details={"lane_candidates": [lane_payload]},
    )
    state = RCAState(
        job=RCAJob(job_id="JOB_CHALLENGE", user_query="Find root cause."),
        evidence=[_evidence("EV_1")],
        findings=[mes_finding],
    )
    state = _update_causal_lane_state(state, mes_finding)
    lane_id = state.causal_lanes[0].lane_id
    challenge = CandidateChallenge(
        candidate_id="CANDIDATE_A",
        evidence_probe_lane_id=lane_id,
        strongest_alternative_lane_id=lane_id,
        supporting_evidence_ids=("EV_1",),
        challenge_explanation="Probe the active factual Lane.",
        status=ChallengeStatus.ALTERNATIVE_IDENTIFIED.value,
    )
    rca_finding = AgentFinding(
        finding_id="F_RCA_CHALLENGE",
        agent=AgentKind.RCA_REASONING.value,
        summary="Adversarial challenge selected.",
        confidence=0.7,
        evidence_ids=["EV_1"],
        details={
            "adversarial_challenge_generation": {"source": "qwen"},
            "candidate_challenges": [challenge.to_dict()],
        },
    )

    updated = _update_competition_state(state, rca_finding)

    assert updated.causal_lanes[0].lifecycle_status == (
        LaneLifecycleStatus.CHALLENGED.value
    )
    assert updated.causal_lanes[0].lifecycle_history[-1].to_status == (
        LaneLifecycleStatus.CHALLENGED.value
    )
    authoritative_mes = next(
        item
        for item in updated.findings
        if item.details.get("lane_inventory_authoritative") is True
    )
    assert authoritative_mes.details["lane_candidates"][0]["lifecycle_status"] == (
        LaneLifecycleStatus.CHALLENGED.value
    )


def test_authoritative_inventory_hides_historical_terminal_lane_from_planner() -> None:
    historical = AgentFinding(
        finding_id="F_MES_OLD",
        agent=AgentKind.MES.value,
        summary="Old raw inventory.",
        confidence=0.8,
        evidence_ids=["EV_OLD"],
        details={"lane_candidates": [_lane("LANE_OLD").to_dict()]},
    )
    active = replace(
        _lane("LANE_NEW"),
        lifecycle_status=LaneLifecycleStatus.ACTIVE.value,
    )
    eliminated = _lane(
        "LANE_TERMINAL",
        recipe="RCP_B",
        status=InvestigationLaneStatus.ELIMINATED.value,
        lifecycle_status=LaneLifecycleStatus.ELIMINATED.value,
        reason="Contradicted.",
    )
    authoritative = AgentFinding(
        finding_id="F_MES_CURRENT",
        agent=AgentKind.MES.value,
        summary="Canonical inventory.",
        confidence=0.8,
        evidence_ids=["EV_CURRENT"],
        details={
            "lane_inventory_authoritative": True,
            "lane_candidates": [active.to_dict(), eliminated.to_dict()],
        },
    )

    assert _known_causal_lane_ids([historical, authoritative]) == (active.lane_id,)
