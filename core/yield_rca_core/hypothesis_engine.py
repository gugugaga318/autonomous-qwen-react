"""Deterministic, evidence-bounded hypothesis engine used in shadow mode.

This module deliberately has no repository, Tool, LLM, report, or workflow
dependency. It is the sole production RCA decision engine after Batch 19.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from yield_rca_core.causal_candidate_comparison import compare_candidate_matrices
from yield_rca_core.causal_confirmation import confirm_candidate
from yield_rca_core.causal_evidence_gap import (
    build_causal_evidence_gaps,
    build_hypothesis_discrimination_gaps,
)
from yield_rca_core.causal_evidence_matrix import (
    CausalEvidenceMatrix,
    build_causal_evidence_matrix,
)
from yield_rca_core.causal_hypothesis import CausalHypothesis
from yield_rca_core.causal_investigation_models import (
    AlternativeSearchStatus,
    CandidateChallenge,
    CandidateSemanticProfile,
    CausalLaneRecord,
)
from yield_rca_core.evidence_models import Evidence
from yield_rca_core.evidence_synthesis import build_evidence_synthesis
from yield_rca_core.models import (
    AgentFinding,
    AgentKind,
    FindingKind,
    ModelValidationError,
)

_EXPOSURE_EVIDENCE_TYPES = {
    "lot_context",
    "process_exposure",
    "equipment_exposure",
    "impact_scope",
    "excursion_window",
}
_PROCESS_EVIDENCE_TYPES = {
    "recipe_change",
    "hold_event",
    "parameter_deviation",
    "trend_deviation",
    "ooc_event",
    "spc_violation",
}
_PRODUCT_EVIDENCE_TYPES = {
    "defect_signal",
    "metrology_deviation",
    "electrical_failure",
}
_LLM_NON_SUPPORTING_EVIDENCE_TYPES = {
    "data_missing",
    "negative_signal",
    "sop_guidance",
}
_KNOWLEDGE_MECHANISM_EVIDENCE_TYPES = {
    "historical_case_match",
    "engineering_note",
}
_CAUSAL_GROUNDING_IGNORED_TOKENS = {
    "abnormal",
    "cause",
    "chamber",
    "control",
    "degradation",
    "deviation",
    "drift",
    "equipment",
    "excursion",
    "failure",
    "instability",
    "issue",
    "lot",
    "mechanism",
    "observed",
    "process",
    "range",
    "shared",
    "tool",
    "wafer",
}

_SIGNATURE_RULES = (
    ("slurry_flow", "slurry delivery degradation"),
    ("carrier_pressure", "carrier pressure instability"),
    ("wf6_flow", "WF6 delivery degradation"),
    ("deposition_rate", "deposition rate excursion"),
)


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _findings_by_kind(findings: list[AgentFinding]) -> dict[str, AgentFinding]:
    result: dict[str, AgentFinding] = {}
    for finding in findings:
        key = finding.finding_kind
        # MES, FDC, and Defect/WAT intentionally share the generic Specialist
        # observation kind.  Only stage-specific findings are unique here.
        if key == FindingKind.SPECIALIST_OBSERVATION.value:
            continue
        if key in result:
            raise ModelValidationError(f"duplicate finding_kind for HypothesisEngine: {key}")
        result[key] = finding
    return result


def _specialist(findings: list[AgentFinding], agent: str) -> AgentFinding | None:
    return next((finding for finding in findings if finding.agent == agent), None)


def _compact(value: str) -> str:
    return "".join(re.findall(r"[a-z0-9]+", value.lower()))


def _parameter_tokens(parameter: str) -> set[str]:
    ignored = {
        "bias",
        "drop",
        "error",
        "index",
        "motor",
        "parameter",
        "proxy",
        "rate",
        "signal",
        "speed",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", parameter.lower())
        if len(token) >= 3 and token not in ignored
    }


def _is_approved_knowledge_support(evidence: Any) -> bool:
    """Only confirmed RCA/engineering knowledge may support a mechanism."""

    if (
        evidence.source_type != "knowledge"
        or evidence.evidence_type not in _KNOWLEDGE_MECHANISM_EVIDENCE_TYPES
    ):
        return False
    statuses = [
        str(value).upper()
        for key, value in evidence.metadata.items()
        if str(key).casefold() == "validation_status"
    ]
    statuses.extend(
        str(value).upper()
        for entity in evidence.entities
        for key, value in entity.attributes.items()
        if str(key).casefold() == "validation_status"
    )
    return bool(statuses) and all(status == "CONFIRMED" for status in statuses)


def _operationally_aligned_knowledge_candidate(
    mes: AgentFinding | None,
    fdc: AgentFinding | None,
    discovery: AgentFinding | None,
) -> str | None:
    """Bind a Knowledge hypothesis to current-Lot equipment and mechanism Evidence."""

    if mes is None or fdc is None or discovery is None:
        return None
    equipment_id = str(
        mes.details.get("target_commonality", {}).get("equipment_id", "")
    ).strip()
    if not equipment_id:
        return None
    mechanism_tokens = {
        token
        for item in fdc.details.get("parameter_summary", [])
        if isinstance(item, dict)
        and abs(float(item.get("avg_delta_percent", 0.0))) >= 5.0
        for token in _parameter_tokens(str(item.get("parameter_name", "")))
    }
    if not mechanism_tokens:
        return None

    candidates: list[tuple[int, float, str]] = []
    for case in discovery.details.get("cases", []):
        if not isinstance(case, dict):
            continue
        root_cause = str(case.get("root_cause", "")).strip()
        if not root_cause or _compact(equipment_id) not in _compact(root_cause):
            continue
        overlap = mechanism_tokens & _mechanism_tokens(root_cause)
        if not overlap:
            continue
        candidates.append(
            (len(overlap), float(case.get("similarity", 0.0)), root_cause)
        )
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1], item[2]))[2]


def _signature_candidate(
    mes: AgentFinding | None,
    fdc: AgentFinding | None,
    discovery: AgentFinding | None,
) -> str | None:
    if mes is None or fdc is None:
        return None
    chamber = str(mes.details.get("target_commonality", {}).get("chamber_id", "")).strip()
    parameters = {
        str(item.get("parameter_name", "")): float(item.get("avg_delta_percent", 0.0))
        for item in fdc.details.get("parameter_summary", [])
        if isinstance(item, dict)
    }
    for parameter, failure_mode in _SIGNATURE_RULES:
        if chamber and parameters.get(parameter, 0.0) <= -5.0:
            return f"{chamber} {failure_mode}"
    return _operationally_aligned_knowledge_candidate(mes, fdc, discovery)


def _recipe_candidate(mes: AgentFinding | None) -> str | None:
    if mes is None or "EV_MES_RECIPE_CHANGE" not in mes.evidence_ids:
        return None
    changes = mes.details.get("recipe_changes", [])
    if not changes or not isinstance(changes[0], dict):
        return None
    recipe_id = str(changes[0].get("source_recipe_id", "")).strip()
    version = str(changes[0].get("source_recipe_version", "")).strip()
    return f"{recipe_id} {version} recipe version change" if recipe_id and version else None


def _mechanism_tokens(root_cause: str) -> set[str]:
    ignored = {"delivery", "reduced", "flow", "on", "the", "chamber", "degradation"}
    return {
        token
        for token in re.findall(r"[a-z0-9]+", root_cause.lower())
        if len(token) >= 3 and token not in ignored
    }


def _causal_grounding_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.casefold())
        if len(token) >= 3 and token not in _CAUSAL_GROUNDING_IGNORED_TOKENS
    }


def _evidence_grounding_tokens(evidence: Any) -> set[str]:
    values = [
        str(evidence.observation or ""),
        *(entity.entity_id for entity in evidence.entities),
    ]
    return {
        token for value in values for token in _causal_grounding_tokens(value)
    }


def _parameter_entity_tokens(evidence: Any) -> set[str]:
    return {
        token
        for entity in evidence.entities
        if entity.entity_type == "parameter"
        for token in _causal_grounding_tokens(entity.entity_id)
    }


def _grounding_overlap(left: set[str], right: set[str]) -> list[str]:
    """Match exact or stable prefix variants such as temp/temperature."""

    matched: set[str] = set()
    for left_token in left:
        for right_token in right:
            if (
                left_token == right_token
                or (
                    min(len(left_token), len(right_token)) >= 4
                    and (
                        left_token.startswith(right_token)
                        or right_token.startswith(left_token)
                    )
                )
            ):
                matched.add(left_token)
    return sorted(matched)


def _canonical_root_cause(root_cause: str, signature_root_cause: str | None) -> str:
    """Prefer the current equipment signature over a coarser historical wording."""
    if signature_root_cause is None:
        return root_cause
    shared = _mechanism_tokens(root_cause) & _mechanism_tokens(signature_root_cause)
    return signature_root_cause if len(shared) >= 2 else root_cause


def _mes_strength(finding: AgentFinding | None) -> float:
    if finding is None:
        return 0.0
    commonality = finding.details.get("target_commonality", {})
    affected_lots = finding.details.get("affected_lots", [])
    return float(commonality.get("coverage", 0.0)) if affected_lots else 0.0


def _fdc_strength(finding: AgentFinding | None) -> float:
    if finding is None:
        return 0.0
    signals = [
        min(1.0, abs(float(item.get("avg_delta_percent", 0.0))) / 10.0)
        for item in finding.details.get("parameter_summary", [])
        if isinstance(item, dict)
    ]
    # Independent normal traces are useful exclusion Evidence; they must not
    # dilute a separate, strongly abnormal causal parameter into a false miss.
    parameter_signal = max(signals, default=0.0)
    ooc_signal = 1.0 if int(finding.details.get("event_count", 0)) > 0 else 0.0
    return 0.8 * parameter_signal + 0.2 * ooc_signal


def _defect_wat_strength(finding: AgentFinding | None) -> float:
    if finding is None:
        return 0.0
    defect_counts = finding.details.get("defect_counts", {})
    fail_modes = finding.details.get("wat_fail_modes", {})
    defect_signal = any(int(value) > 0 for value in defect_counts.values())
    wat_signal = any(int(value) > 0 for value in fail_modes.values())
    metrology_signal = int(finding.details.get("metrology_fail_count", 0)) > 0
    if metrology_signal:
        return 1.0
    return (float(defect_signal) + float(wat_signal) + float(defect_signal and wat_signal)) / 3


def _has_conflicting_physics(
    mes: AgentFinding | None,
    fdc: AgentFinding | None,
) -> bool:
    if fdc is None:
        return False
    parameters = {
        str(item.get("parameter_name", "")): float(item.get("avg_delta_percent", 0.0))
        for item in fdc.details.get("parameter_summary", [])
        if isinstance(item, dict)
    }
    if (
        parameters.get("slurry_flow", 0.0) <= -5.0
        and parameters.get("estimated_removal_rate", 0.0) >= 5.0
    ):
        return True
    abnormal_tokens = {
        token
        for parameter, delta in parameters.items()
        if abs(delta) >= 5.0
        for token in _parameter_tokens(parameter)
    }
    normal_tokens: set[str] = set()
    if mes is not None:
        normal_tokens = {
            token
            for evidence in mes.evidence
            if evidence.evidence_type == "negative_signal"
            # Later recovery and passing-control observations describe containment
            # after the excursion.  They support the time-bounded causal story and
            # must not be treated as simultaneous physics that contradicts it.
            and not evidence.evidence_id.startswith(
                ("EV_MES_RECOVERY_CONTROLS", "EV_WAT_PASSING_CONTROLS")
            )
            for entity in evidence.entities
            if entity.entity_type == "parameter"
            for token in _parameter_tokens(entity.entity_id)
        }
    return bool(abnormal_tokens & normal_tokens)


def _candidate_payload(
    *,
    hypothesis_id: str,
    root_cause: str,
    basis: str,
    base_score: float,
    core_evidence_ids: list[str],
    non_supporting_evidence_ids: list[str],
    discovery: AgentFinding | None,
    validation: AgentFinding | None,
    signature_root_cause: str | None,
) -> dict[str, Any]:
    supporting = (
        [
            evidence_id
            for evidence_id in core_evidence_ids
            if evidence_id not in non_supporting_evidence_ids
        ]
        if root_cause == signature_root_cause
        else []
    )
    if basis == "recipe_change":
        supporting.extend(core_evidence_ids)
    neutral: list[str] = (
        list(non_supporting_evidence_ids)
        if root_cause == signature_root_cause
        else []
    )
    contradicting: list[str] = []
    validation_results = []

    if discovery is not None:
        for case in discovery.details.get("cases", []):
            if isinstance(case, dict) and str(case.get("root_cause", "")).strip() == root_cause:
                supporting.extend(discovery.evidence_ids)
                base_score = max(base_score, float(case.get("similarity", 0.0)))

    if validation is not None:
        for result in validation.details.get("validation_results", []):
            if (
                not isinstance(result, dict)
                or str(result.get("root_cause", "")).strip() != root_cause
            ):
                continue
            result_evidence = [str(item) for item in result.get("evidence_ids", [])]
            outcome = str(result.get("validation", "data_missing"))
            validation_results.append(
                {
                    "outcome": outcome,
                    "knowledge_case_id": str(result.get("knowledge_case_id", "")),
                    "evidence_ids": result_evidence,
                }
            )
            if outcome == "supporting":
                supporting.extend(result_evidence)
                base_score = min(1.0, base_score + 0.05)
            elif outcome == "contradicting":
                contradicting.extend(result_evidence)
                base_score = max(0.0, base_score - 0.20)
            else:
                neutral.extend(result_evidence)

    supporting = _unique(supporting)
    contradicting = _unique(contradicting)
    neutral = _unique(neutral)
    evidence_ids = _unique(supporting + contradicting + neutral)
    confidence = round(min(0.95, max(0.0, base_score)), 3)
    status = "conflicted" if contradicting else ("supported" if confidence >= 0.75 else "candidate")
    return {
        "hypothesis_id": hypothesis_id,
        "root_cause": root_cause,
        "basis": basis,
        "supporting_evidence_ids": supporting,
        "contradicting_evidence_ids": contradicting,
        "neutral_evidence_ids": neutral,
        "evidence_ids": evidence_ids,
        "validation_results": validation_results,
        "confidence": confidence,
        "status": status,
        "rejection_reasons": [
            "Explicit normal/exclusion evidence contradicts this hypothesis."
        ]
        if contradicting
        else [],
    }


def _llm_candidate_payload(
    *,
    hypothesis_id: str,
    proposal: dict[str, Any],
    evidence_by_id: dict[str, Any],
) -> dict[str, Any]:
    """Apply deterministic causal-lane and scope gates to one Qwen proposal."""

    root_cause = str(proposal.get("root_cause", "")).strip()
    explanation = str(proposal.get("causal_explanation", "")).strip()
    supporting = _unique(
        [str(item) for item in proposal.get("supporting_evidence_ids", [])]
    )
    contradicting = _unique(
        [str(item) for item in proposal.get("contradicting_evidence_ids", [])]
    )
    unknown = sorted((set(supporting) | set(contradicting)) - set(evidence_by_id))
    invalid_support = sorted(
        evidence_id
        for evidence_id in supporting
        if evidence_id in evidence_by_id
        if (
            evidence_by_id[evidence_id].evidence_type
            in _LLM_NON_SUPPORTING_EVIDENCE_TYPES
            or (
                evidence_by_id[evidence_id].evidence_type
                in _KNOWLEDGE_MECHANISM_EVIDENCE_TYPES
                and not _is_approved_knowledge_support(evidence_by_id[evidence_id])
            )
        )
    )
    known_supporting = [
        evidence_id
        for evidence_id in supporting
        if evidence_id in evidence_by_id and evidence_id not in invalid_support
    ]
    known_contradicting = [
        evidence_id
        for evidence_id in contradicting
        if evidence_id in evidence_by_id
    ]
    supporting_evidence = [
        evidence_by_id[evidence_id]
        for evidence_id in known_supporting
    ]
    lane_ids = {
        "shared_exposure": [
            item.evidence_id
            for item in supporting_evidence
            if item.evidence_type in _EXPOSURE_EVIDENCE_TYPES
        ],
        "process_anomaly": [
            item.evidence_id
            for item in supporting_evidence
            if item.evidence_type in _PROCESS_EVIDENCE_TYPES
        ],
        "product_outcome": [
            item.evidence_id
            for item in supporting_evidence
            if item.evidence_type in _PRODUCT_EVIDENCE_TYPES
        ],
    }
    complete_lanes = {lane for lane, ids in lane_ids.items() if ids}
    lane_lot_ids: dict[str, set[str]] = {}
    for lane, ids in lane_ids.items():
        lane_lot_ids[lane] = {
            entity.entity_id
            for evidence_id in ids
            for entity in evidence_by_id[evidence_id].entities
            if entity.entity_type == "lot"
        }
    populated_lot_sets = [values for values in lane_lot_ids.values() if values]
    shared_lot_ids = (
        set.intersection(*populated_lot_sets) if populated_lot_sets else set()
    )
    source_agents = {
        str(item.source_agent)
        for item in supporting_evidence
        if item.source_agent is not None and item.source_type != "knowledge"
    }
    root_tokens = _causal_grounding_tokens(root_cause)
    explanation_tokens = _causal_grounding_tokens(explanation)
    process_grounding_tokens = {
        token
        for item in supporting_evidence
        if item.evidence_type in _PROCESS_EVIDENCE_TYPES
        for token in _evidence_grounding_tokens(item)
    }
    product_grounding_tokens = {
        token
        for item in supporting_evidence
        if item.evidence_type in _PRODUCT_EVIDENCE_TYPES
        for token in _evidence_grounding_tokens(item)
    }
    root_process_overlap = _grounding_overlap(
        root_tokens,
        process_grounding_tokens,
    )
    explanation_product_overlap = _grounding_overlap(
        explanation_tokens,
        product_grounding_tokens,
    )
    causal_parameter_tokens = {
        token
        for item in supporting_evidence
        if item.evidence_type in _PROCESS_EVIDENCE_TYPES
        for token in _parameter_entity_tokens(item)
    }
    supporting_lot_ids = {
        entity.entity_id
        for item in supporting_evidence
        for entity in item.entities
        if entity.entity_type == "lot"
    }
    automatically_contradicting: list[str] = []
    for evidence in evidence_by_id.values():
        if evidence.evidence_type != "negative_signal" or evidence.evidence_id.startswith(
            ("EV_MES_RECOVERY_CONTROLS", "EV_WAT_PASSING_CONTROLS")
        ):
            continue
        negative_parameter_tokens = _parameter_entity_tokens(evidence)
        negative_lot_ids = {
            entity.entity_id
            for entity in evidence.entities
            if entity.entity_type == "lot"
        }
        if (
            causal_parameter_tokens
            and _grounding_overlap(
                causal_parameter_tokens,
                negative_parameter_tokens,
            )
            and (
                not supporting_lot_ids
                or not negative_lot_ids
                or bool(supporting_lot_ids & negative_lot_ids)
            )
        ):
            automatically_contradicting.append(evidence.evidence_id)
    known_contradicting = _unique(
        [*known_contradicting, *automatically_contradicting]
    )
    matrix = None
    matrix_error: str | None = None
    if root_cause and explanation and known_supporting:
        try:
            matrix = build_causal_evidence_matrix(
                CausalHypothesis(
                    root_cause=root_cause,
                    causal_explanation=explanation,
                    supporting_evidence_ids=tuple(known_supporting),
                    contradicting_evidence_ids=tuple(known_contradicting),
                ),
                evidence_by_id.values(),
            )
        except (TypeError, ValueError) as exc:
            matrix_error = str(exc)
    grounded_entity_ids = {
        entity.entity_id
        for item in supporting_evidence
        for entity in item.entities
    }
    named_structured_entities = set(
        re.findall(r"\b(?:LOT|EQ|RCP)_[A-Z0-9_]+\b", root_cause.upper())
    )
    ungrounded_entities = sorted(named_structured_entities - grounded_entity_ids)
    gate_checks = {
        "known_evidence": not unknown,
        "supporting_evidence_types": not invalid_support,
        "three_causal_lanes": len(complete_lanes) == 3,
        "shared_lot_scope": bool(shared_lot_ids),
        "independent_source_agents": len(source_agents) >= 3,
        "root_cause_process_grounding": bool(root_process_overlap),
        "explanation_product_grounding": bool(explanation_product_overlap),
        "grounded_structured_entities": not ungrounded_entities,
        "causal_explanation_present": bool(explanation),
        "claim_evidence_consistency": (
            matrix_error is None and (matrix is None or not matrix.has_critical_conflict)
        ),
    }
    rejection_reasons: list[str] = []
    if unknown:
        rejection_reasons.append(f"Unknown Evidence IDs: {unknown}.")
    if invalid_support:
        rejection_reasons.append(
            "Non-causal or missing Evidence was cited as support: "
            f"{invalid_support}."
        )
    if len(complete_lanes) < 3:
        rejection_reasons.append(
            "The proposal does not join shared exposure, process anomaly, and "
            "product outcome Evidence."
        )
    if not shared_lot_ids:
        rejection_reasons.append(
            "The causal Evidence lanes do not share a grounded Lot scope."
        )
    if len(source_agents) < 3:
        rejection_reasons.append(
            "The proposal lacks three independent Specialist Evidence sources."
        )
    if not root_process_overlap:
        rejection_reasons.append(
            "The root-cause wording is not grounded in an abnormal process "
            "parameter or process Evidence entity."
        )
    if not explanation_product_overlap:
        rejection_reasons.append(
            "The causal explanation does not connect to the observed product "
            "Evidence."
        )
    if ungrounded_entities:
        rejection_reasons.append(
            f"The root cause names ungrounded structured entities: {ungrounded_entities}."
        )
    if not explanation:
        rejection_reasons.append("The proposal lacks a causal explanation.")
    if matrix_error:
        rejection_reasons.append(f"Causal Evidence Matrix could not be built: {matrix_error}.")
    elif matrix is not None and matrix.has_critical_conflict:
        rejection_reasons.append(
            "The Causal Evidence Matrix found a critical claim/Evidence conflict."
        )

    confidence = 0.45 + 0.10 * len(complete_lanes)
    if len(source_agents) >= 3:
        confidence += 0.05
    if shared_lot_ids:
        confidence += 0.05
    confidence = round(min(0.90, confidence), 3)
    gate_passed = all(gate_checks.values()) and not known_contradicting
    status = "conflicted" if known_contradicting else (
        "supported" if gate_passed else "candidate"
    )
    if known_contradicting:
        rejection_reasons.append(
            "The proposal explicitly cites contradicting operational Evidence."
        )
    return {
        "hypothesis_id": hypothesis_id,
        "root_cause": root_cause,
        "basis": "llm_evidence_composition",
        "causal_explanation": explanation,
        "supporting_evidence_ids": known_supporting,
        "contradicting_evidence_ids": known_contradicting,
        "neutral_evidence_ids": [],
        "evidence_ids": _unique(known_supporting + known_contradicting),
        "validation_results": [
            {
                "outcome": "passed" if passed else "failed",
                "gate": name,
                "evidence_ids": _unique(
                    [evidence_id for ids in lane_ids.values() for evidence_id in ids]
                ),
            }
            for name, passed in gate_checks.items()
        ],
        "confidence": confidence,
        "status": status,
        "rejection_reasons": rejection_reasons,
        "llm_gate_passed": gate_passed,
        "causal_lanes": lane_ids,
        "shared_lot_ids": sorted(shared_lot_ids),
        "source_agents": sorted(source_agents),
        "root_process_grounding_tokens": root_process_overlap,
        "explanation_product_grounding_tokens": explanation_product_overlap,
        "causal_evidence_matrix": matrix.to_dict() if matrix is not None else {
            "status": "unavailable",
            "claims": {},
            "invalid_evidence_ids": sorted(set(unknown)),
            "mechanism_support_source": None,
        },
        "causal_matrix_status": matrix.status if matrix is not None else "unavailable",
        "mechanism_support_source": (
            matrix.mechanism_support_source if matrix is not None else None
        ),
    }


_MATRIX_REJECTION_PREFIXES = (
    "The Causal Evidence Matrix found a critical claim/Evidence conflict.",
    "Causal Evidence Matrix could not be built:",
)


def _synchronize_candidate_matrix_validation(
    candidate: dict[str, Any],
    matrix: CausalEvidenceMatrix,
) -> None:
    """Keep candidate validation fields consistent with the final Matrix."""

    matrix_consistent = not matrix.has_critical_conflict
    validation_results = [
        dict(item)
        for item in candidate.get("validation_results", [])
        if isinstance(item, Mapping)
    ]
    for result in validation_results:
        if result.get("gate") == "claim_evidence_consistency":
            result["outcome"] = "passed" if matrix_consistent else "failed"
    candidate["validation_results"] = validation_results

    rejection_reasons = [
        str(reason)
        for reason in candidate.get("rejection_reasons", [])
        if not str(reason).startswith(_MATRIX_REJECTION_PREFIXES)
    ]
    if not matrix_consistent:
        rejection_reasons.append(_MATRIX_REJECTION_PREFIXES[0])
    candidate["rejection_reasons"] = list(dict.fromkeys(rejection_reasons))

    gate_passed = (
        bool(validation_results)
        and all(
            result.get("outcome") == "passed"
            for result in validation_results
        )
        and not candidate.get("contradicting_evidence_ids", [])
    )
    candidate["llm_gate_passed"] = gate_passed
    if candidate.get("status") != "conflicted" and not gate_passed:
        candidate["status"] = "candidate"


@dataclass(frozen=True)
class HypothesisEngine:
    """Generate, validate, rank, and gate deterministic RCA hypotheses."""

    supported_threshold: float = 0.75
    maximum_confidence: float = 0.95

    def __post_init__(self) -> None:
        if not 0.0 <= self.supported_threshold <= 1.0:
            raise ModelValidationError(
                "HypothesisEngine supported_threshold must be between 0 and 1"
            )
        if not self.supported_threshold <= self.maximum_confidence <= 1.0:
            raise ModelValidationError("HypothesisEngine maximum_confidence is invalid")

    def analyze(
        self,
        *,
        request_id: str,
        findings: list[AgentFinding],
        mode: str = "shadow",
        external_candidates: list[dict[str, Any]] | None = None,
        include_deterministic_candidates: bool = True,
        candidate_comparison: dict[str, Any] | None = None,
        strict_confirmation: bool = False,
        alternative_search_status: str | None = None,
        candidate_challenges: list[CandidateChallenge] | None = None,
        context_evidence: Sequence[Evidence] = (),
        causal_lanes: Sequence[CausalLaneRecord] = (),
        consumed_discriminators: Collection[tuple[str, str]] = (),
        competition_brief: Mapping[str, Any] | None = None,
        candidate_semantic_profiles: Sequence[
            CandidateSemanticProfile | Mapping[str, Any]
        ] = (),
    ) -> dict[str, Any]:
        """Return a JSON-safe deterministic hypothesis decision result."""
        by_kind = _findings_by_kind(findings)
        mes = _specialist(findings, AgentKind.MES.value)
        fdc = _specialist(findings, AgentKind.FDC.value)
        defect_wat = _specialist(findings, AgentKind.DEFECT_WAT.value)
        discovery = by_kind.get(FindingKind.KNOWLEDGE_DISCOVERY.value)
        validation = by_kind.get(FindingKind.KNOWLEDGE_VALIDATION.value)
        signature_root_cause = _signature_candidate(mes, fdc, discovery)
        core_evidence_ids = _unique(
            [
                evidence_id
                for finding in (mes, fdc, defect_wat)
                if finding is not None
                for evidence_id in finding.evidence_ids
            ]
        )
        non_supporting_evidence_ids = _unique(
            [
                evidence.evidence_id
                for finding in (mes, fdc, defect_wat)
                if finding is not None
                for evidence in finding.evidence
                if evidence.evidence_type in {"data_missing", "negative_signal"}
            ]
        )
        evidence_by_id = {
            evidence.evidence_id: evidence
            for finding in findings
            for evidence in finding.evidence
        }
        evidence_by_id.update(
            {evidence.evidence_id: evidence for evidence in context_evidence}
        )
        source_lot_id = next(
            (
                str(finding.details.get("source_lot_id"))
                for finding in findings
                if finding.details.get("source_lot_id")
            ),
            None,
        )
        non_supporting_evidence_ids = _unique(
            [
                *non_supporting_evidence_ids,
                *[
                    evidence.evidence_id
                    for evidence in context_evidence
                    if evidence.evidence_type in {"data_missing", "negative_signal"}
                ],
            ]
        )
        candidates: dict[str, dict[str, Any]] = {}

        def add(root_cause: str | None, basis: str, score: float) -> None:
            if not root_cause:
                return
            candidate = _candidate_payload(
                hypothesis_id=f"{request_id}:shadow:{len(candidates) + 1}",
                root_cause=root_cause,
                basis=basis,
                base_score=score,
                core_evidence_ids=core_evidence_ids,
                non_supporting_evidence_ids=non_supporting_evidence_ids,
                discovery=discovery,
                validation=validation,
                signature_root_cause=signature_root_cause,
            )
            existing = candidates.get(root_cause)
            if existing is None or candidate["confidence"] > existing["confidence"]:
                candidates[root_cause] = candidate

        if include_deterministic_candidates:
            add(signature_root_cause, "equipment_signature", 0.95)
            add(_recipe_candidate(mes), "recipe_change", 0.80)
            if discovery is not None:
                for case in discovery.details.get("cases", []):
                    if isinstance(case, dict):
                        raw_root_cause = str(case.get("root_cause", "")).strip()
                        add(
                            _canonical_root_cause(raw_root_cause, signature_root_cause) or None,
                            "knowledge_discovery",
                            float(case.get("similarity", 0.0)),
                        )
            if validation is not None:
                for candidate in validation.details.get("preliminary_candidates", []):
                    if isinstance(candidate, dict):
                        raw_root_cause = str(candidate.get("root_cause", "")).strip()
                        add(
                            _canonical_root_cause(raw_root_cause, signature_root_cause) or None,
                            "legacy_preliminary_candidate",
                            float(candidate.get("score", 0.0)),
                        )

        for proposal in external_candidates or []:
            if not isinstance(proposal, dict):
                raise ModelValidationError(
                    "external hypothesis candidates must be JSON objects"
                )
            candidate = _llm_candidate_payload(
                hypothesis_id=f"{request_id}:llm:{len(candidates) + 1}",
                proposal=proposal,
                evidence_by_id=evidence_by_id,
            )
            root_cause = str(candidate["root_cause"])
            existing = candidates.get(root_cause)
            if existing is None or candidate["confidence"] > existing["confidence"]:
                candidates[root_cause] = candidate

        semantic_profiles_by_candidate_id: dict[
            str, CandidateSemanticProfile | Mapping[str, Any]
        ] = {}
        for raw_profile in candidate_semantic_profiles:
            candidate_id = (
                raw_profile.candidate_id
                if isinstance(raw_profile, CandidateSemanticProfile)
                else str(raw_profile.get("candidate_id", "")).strip()
            )
            if candidate_id:
                semantic_profiles_by_candidate_id[candidate_id] = raw_profile
        semantic_profiles_expected = bool(candidate_semantic_profiles) or str(
            (competition_brief or {}).get("competition_requirement", "")
        ).strip() in {
            "alternative_discovery_required",
            "direction_required",
            "mechanism_required",
            "scope_required",
            "mixed_required",
        }

        matrices_by_root: dict[str, CausalEvidenceMatrix] = {}
        for candidate in candidates.values():
            hypothesis_id = str(candidate.get("hypothesis_id", "")).strip()
            semantic_profile: CandidateSemanticProfile | Mapping[str, Any] | None
            if candidate.get("basis") == "llm_evidence_composition":
                # An explicit empty mapping means the LLM Candidate was expected
                # to have scope semantics but none survived validation.  Matrix
                # must keep that scope unresolved instead of falling back to the
                # legacy Evidence-coverage inference used by controlled paths.
                semantic_profile = semantic_profiles_by_candidate_id.get(
                    hypothesis_id,
                    {} if semantic_profiles_expected else None,
                )
            else:
                semantic_profile = None
            try:
                matrix = build_causal_evidence_matrix(
                    CausalHypothesis(
                        root_cause=str(candidate["root_cause"]),
                        causal_explanation=str(
                            candidate.get("causal_explanation", candidate["root_cause"])
                        ),
                        supporting_evidence_ids=tuple(
                            candidate.get("supporting_evidence_ids", [])
                        ),
                        contradicting_evidence_ids=tuple(
                            candidate.get("contradicting_evidence_ids", [])
                        ),
                    ),
                    evidence_by_id.values(),
                    semantic_profile=semantic_profile,
                    causal_lanes=causal_lanes,
                )
            except (TypeError, ValueError):
                continue
            matrices_by_root[str(candidate["root_cause"])] = matrix
            candidate["causal_evidence_matrix"] = matrix.to_dict()
            candidate["causal_matrix_status"] = matrix.status
            candidate["mechanism_support_source"] = matrix.mechanism_support_source
            if candidate.get("basis") == "llm_evidence_composition":
                _synchronize_candidate_matrix_validation(candidate, matrix)

        matrices = list(matrices_by_root.values())
        matrix_candidate_ids = [
            str(candidates[matrix.candidate.root_cause]["hypothesis_id"])
            for matrix in matrices
        ]
        competition_status = str(
            alternative_search_status or AlternativeSearchStatus.NOT_SEARCHED.value
        )
        competition_status_supplied = alternative_search_status is not None
        challenges = list(candidate_challenges or [])
        evidence_gaps = build_causal_evidence_gaps(matrices)
        evidence_gaps.extend(
            build_hypothesis_discrimination_gaps(
                matrices,
                alternative_search_status=competition_status,
                candidate_challenges=challenges,
                causal_lanes=causal_lanes,
                candidate_ids=matrix_candidate_ids,
                source_lot_id=source_lot_id,
                consumed_discriminators=consumed_discriminators,
                competition_brief=competition_brief,
                candidate_semantic_profiles=candidate_semantic_profiles,
            )
        )
        evidence_gaps.sort(
            key=lambda item: (
                int(item.get("priority", 3)),
                int(item.get("candidate_index", 0)),
                str(item.get("claim", "")),
            )
        )
        python_comparison = compare_candidate_matrices(
            matrices,
            evidence_gaps=evidence_gaps,
            alternative_search_status=competition_status,
        )
        effective_comparison = dict(python_comparison)
        effective_comparison["python_comparison_explanation"] = python_comparison.get(
            "comparison_explanation"
        )
        comparator_preferred_index: int | None = None
        if candidate_comparison is not None:
            preferred = candidate_comparison.get("preferred_candidate_index")
            if preferred is None or (
                isinstance(preferred, int)
                and not isinstance(preferred, bool)
                and 0 <= preferred < len(matrices)
            ):
                comparator_preferred_index = preferred
                for key in (
                    "comparison_explanation",
                    "comparison_summary",
                    "selected_gap_id",
                    "source",
                ):
                    if key in candidate_comparison:
                        effective_comparison[key] = candidate_comparison[key]
                effective_comparison["comparator_comparison_explanation"] = (
                    candidate_comparison.get(
                        "comparison_explanation",
                        candidate_comparison.get("comparison_summary"),
                    )
                )
        challenge_selected_gap_ids = list(
            dict.fromkeys(
                gap_id
                for challenge in challenges
                for gap_id in challenge.distinguishing_gap_ids
            )
        )
        effective_comparison["comparator_selected_gap_id"] = (
            candidate_comparison.get("selected_gap_id")
            if candidate_comparison is not None
            else None
        )
        effective_comparison["selected_gap_id"] = (
            challenge_selected_gap_ids[0]
            if len(challenge_selected_gap_ids) == 1
            else None
        )
        effective_comparison["selected_gap_ids"] = challenge_selected_gap_ids
        effective_comparison["gap_selection_owner"] = "adversarial_challenge"

        mes_strength = _mes_strength(mes)
        fdc_strength = _fdc_strength(fdc)
        defect_wat_strength = _defect_wat_strength(defect_wat)
        conflicting_physics = _has_conflicting_physics(mes, fdc)

        def candidate_challenge(candidate: dict[str, Any]) -> CandidateChallenge | None:
            hypothesis_id = str(candidate.get("hypothesis_id", ""))
            return next(
                (
                    challenge
                    for challenge in challenges
                    if challenge.candidate_id == hypothesis_id
                ),
                None,
            )

        def passes_decision_gate(candidate: dict[str, Any]) -> bool:
            if candidate["status"] != "supported" or conflicting_physics:
                return False
            if candidate["basis"] == "llm_evidence_composition":
                if not bool(candidate.get("llm_gate_passed", False)):
                    return False
                if (
                    strict_confirmation
                    and competition_status_supplied
                    and competition_status
                    != AlternativeSearchStatus.ALTERNATIVES_ELIMINATED.value
                ):
                    return False
                if strict_confirmation:
                    matrix = matrices_by_root.get(str(candidate["root_cause"]))
                    challenge = candidate_challenge(candidate)
                    return bool(
                        matrix is not None
                        and confirm_candidate(
                            matrix,
                            strict=True,
                            alternative_search_status=(
                                competition_status if competition_status_supplied else None
                            ),
                            unexplained_precursor_evidence_ids=(
                                challenge.unexplained_precursor_evidence_ids
                                if challenge is not None
                                else ()
                            ),
                        ).status
                        == "supported"
                    )
                return True
            if candidate["basis"] == "recipe_change":
                return mes_strength >= 0.8 and defect_wat_strength >= 0.6
            return (
                candidate["root_cause"] == signature_root_cause
                and mes_strength >= 0.8
                and fdc_strength >= 0.6
                and defect_wat_strength >= 0.6
            )

        matrix_index_by_root = {
            matrix.candidate.root_cause: index
            for index, matrix in enumerate(matrices)
        }
        matrix_scores = [int(score) for score in python_comparison.get("scores", [])]

        def candidate_matrix(candidate: dict[str, Any]) -> CausalEvidenceMatrix | None:
            return matrices_by_root.get(str(candidate["root_cause"]))

        def candidate_matrix_score(candidate: dict[str, Any]) -> int:
            index = matrix_index_by_root.get(str(candidate["root_cause"]))
            if index is None or not 0 <= index < len(matrix_scores):
                return -10_000
            return matrix_scores[index]

        def candidate_has_unknown_evidence(candidate: dict[str, Any]) -> bool:
            return any(
                result.get("gate") == "known_evidence"
                and result.get("outcome") == "failed"
                for result in candidate.get("validation_results", [])
                if isinstance(result, dict)
            )

        def matrix_safety_rank(candidate: dict[str, Any]) -> int:
            matrix = candidate_matrix(candidate)
            if matrix is None or matrix.status == "unavailable":
                return 3
            if matrix.has_critical_conflict:
                return 2
            if matrix.invalid_evidence_ids or candidate_has_unknown_evidence(candidate):
                return 1
            return 0

        legacy_ranked = sorted(
            candidates.values(),
            key=lambda item: (
                not passes_decision_gate(item),
                -float(item["confidence"]),
                str(item["root_cause"]),
            ),
        )
        pre_comparison_rank_by_root = {
            str(candidate["root_cause"]): rank
            for rank, candidate in enumerate(legacy_ranked, start=1)
        }

        preference_accepted: bool | None = None
        preference_rejection_reason: str | None = None
        qwen_tiebreak_root: str | None = None
        if comparator_preferred_index is not None:
            preferred_matrix = matrices[comparator_preferred_index]
            preferred_root = preferred_matrix.candidate.root_cause
            preferred_candidate = candidates[preferred_root]
            preferred_safety_rank = matrix_safety_rank(preferred_candidate)
            preferred_gate_status = passes_decision_gate(preferred_candidate)
            comparable_candidates = [
                candidate
                for candidate in candidates.values()
                if passes_decision_gate(candidate) == preferred_gate_status
                and matrix_safety_rank(candidate) == 0
            ]
            best_comparable_score = max(
                (candidate_matrix_score(candidate) for candidate in comparable_candidates),
                default=-10_000,
            )
            if preferred_matrix.has_critical_conflict:
                preference_accepted = False
                preference_rejection_reason = "preferred_candidate_has_critical_conflict"
            elif (
                preferred_matrix.invalid_evidence_ids
                or candidate_has_unknown_evidence(preferred_candidate)
            ):
                preference_accepted = False
                preference_rejection_reason = "preferred_candidate_has_unknown_evidence"
            elif preferred_matrix.status == "unavailable":
                preference_accepted = False
                preference_rejection_reason = "preferred_candidate_matrix_unavailable"
            elif any(
                passes_decision_gate(candidate)
                for candidate in candidates.values()
            ) and not preferred_gate_status:
                preference_accepted = False
                preference_rejection_reason = "preferred_candidate_is_below_decision_gate_tier"
            elif candidate_matrix_score(preferred_candidate) < best_comparable_score:
                preference_accepted = False
                preference_rejection_reason = "preferred_candidate_has_lower_python_matrix_score"
            elif preferred_safety_rank != 0:
                preference_accepted = False
                preference_rejection_reason = "preferred_candidate_is_not_safe_for_ranking"
            else:
                preference_accepted = True
                qwen_tiebreak_root = preferred_root

        comparison_ranking_enabled = candidate_comparison is not None
        if comparison_ranking_enabled:
            ranked_all = sorted(
                candidates.values(),
                key=lambda item: (
                    not passes_decision_gate(item),
                    matrix_safety_rank(item),
                    -candidate_matrix_score(item),
                    (
                        0
                        if qwen_tiebreak_root is not None
                        and str(item["root_cause"]) == qwen_tiebreak_root
                        else 1
                    ),
                    -float(item["confidence"]),
                    str(item["root_cause"]),
                ),
            )
        else:
            # Controlled and legacy callers retain the pre-comparison order.
            ranked_all = legacy_ranked

        ranked = ranked_all[:3]
        for rank, candidate in enumerate(ranked, start=1):
            candidate["rank"] = rank
            candidate["final_rank"] = rank
            candidate["pre_comparison_rank"] = pre_comparison_rank_by_root[
                str(candidate["root_cause"])
            ]
            candidate["python_matrix_score"] = candidate_matrix_score(candidate)
            candidate["decision_gate_passed"] = passes_decision_gate(candidate)

        python_preferred_index = python_comparison.get("preferred_candidate_index")
        final_preferred_index = (
            matrix_index_by_root.get(str(ranked_all[0]["root_cause"]))
            if comparison_ranking_enabled and ranked_all
            else python_preferred_index
        )
        if comparison_ranking_enabled:
            if preference_accepted and comparator_preferred_index != python_preferred_index:
                ranking_source = "python_matrix_with_qwen_tiebreak"
            elif preference_accepted:
                ranking_source = "python_matrix_qwen_consistent"
            elif comparator_preferred_index is not None:
                ranking_source = "python_matrix_qwen_rejected"
            else:
                ranking_source = "python_matrix"
        else:
            ranking_source = "legacy_gate_confidence"
        effective_comparison.update(
            {
                "preferred_candidate_index": final_preferred_index,
                "python_preferred_candidate_index": python_preferred_index,
                "comparator_preferred_candidate_index": comparator_preferred_index,
                "comparator_preference_accepted": preference_accepted,
                "comparator_preference_rejection_reason": (
                    preference_rejection_reason
                ),
                "ranking_source": ranking_source,
                "pre_comparison_ranks": [
                    {
                        "candidate_index": matrix_index_by_root.get(
                            str(candidate["root_cause"])
                        ),
                        "root_cause": str(candidate["root_cause"]),
                        "rank": pre_comparison_rank_by_root[str(candidate["root_cause"])],
                    }
                    for candidate in legacy_ranked
                ],
                "final_ranks": [
                    {
                        "candidate_index": matrix_index_by_root.get(
                            str(candidate["root_cause"])
                        ),
                        "root_cause": str(candidate["root_cause"]),
                        "rank": rank,
                        "python_matrix_score": candidate_matrix_score(candidate),
                    }
                    for rank, candidate in enumerate(ranked_all, start=1)
                ],
            }
        )
        selected = ranked[0] if ranked else None
        supported = (
            selected is not None
            and bool(selected["decision_gate_passed"])
        )
        selected_matrix = (
            matrices_by_root.get(str(selected["root_cause"])) if selected is not None else None
        )
        selected_challenge = (
            candidate_challenge(selected) if selected is not None else None
        )
        confirmation = (
            confirm_candidate(
                selected_matrix,
                alternative_matrices=[
                    matrix
                    for matrix in matrices
                    if selected_matrix is None or matrix is not selected_matrix
                ],
                strict=strict_confirmation,
                alternative_search_status=(
                    competition_status if competition_status_supplied else None
                ),
                require_causal_chain=strict_confirmation,
                unexplained_precursor_evidence_ids=(
                    selected_challenge.unexplained_precursor_evidence_ids
                    if selected_challenge is not None
                    else ()
                ),
            )
            if selected_matrix is not None
            else None
        )
        if strict_confirmation and confirmation is not None:
            supported = confirmation.status == "supported"
        decision = {
            "status": "supported" if supported else "inconclusive",
            "root_cause": (
                selected["root_cause"] if supported and selected is not None else "inconclusive"
            ),
            "confidence": selected["confidence"] if supported and selected is not None else 0.0,
            "reasons": (
                ["Top hypothesis passed the deterministic evidence and contradiction gates."]
                if supported
                else ["No ranked hypothesis passed the deterministic decision gate."]
            ),
            "conflicting_physics": conflicting_physics,
            "alternative_search_status": competition_status,
            "candidate_challenges": [item.to_dict() for item in challenges],
            "causal_chain_completeness": (
                confirmation.causal_chain_completeness
                if confirmation is not None
                else None
            ),
            "data_missing_evidence_ids": (
                list(confirmation.data_missing_evidence_ids)
                if confirmation is not None
                else []
            ),
            "blocking_data_missing_evidence_ids": (
                list(confirmation.blocking_data_missing_evidence_ids)
                if confirmation is not None
                else []
            ),
            "non_blocking_data_missing_evidence_ids": (
                list(confirmation.non_blocking_data_missing_evidence_ids)
                if confirmation is not None
                else []
            ),
            "conclusion_status": (
                confirmation.status
                if confirmation is not None
                else "insufficient_evidence"
            ),
            "confirmation_gate": (
                confirmation.to_dict()
                if confirmation is not None
                else {
                    "status": "insufficient_evidence",
                    "checks": {},
                    "reasons": ["No candidate matrix is available."],
                    "unresolved_gaps": [],
                }
            ),
        }
        return {
            "engine": "hypothesis_v1",
            "mode": mode,
            "candidates": ranked,
            "decision_gate": decision,
            "input": {
                "finding_kinds": [finding.finding_kind for finding in findings],
                "typed_evidence_ids": _unique(
                    [
                        *[
                            evidence_id
                            for finding in findings
                            for evidence_id in finding.evidence_ids
                        ],
                        *[evidence.evidence_id for evidence in context_evidence],
                    ]
                ),
                "knowledge_validation_present": validation is not None,
                "external_candidate_count": len(external_candidates or []),
                "deterministic_candidates_enabled": include_deterministic_candidates,
                "strict_confirmation": strict_confirmation,
                "alternative_search_status": competition_status,
                "candidate_challenge_count": len(challenges),
            },
            "evidence_synthesis": build_evidence_synthesis(evidence_by_id.values()),
            "causal_evidence_gaps": evidence_gaps,
            "candidate_comparison": effective_comparison,
        }
