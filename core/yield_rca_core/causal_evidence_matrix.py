"""Python-owned claim/evidence validation for Qwen causal candidates.

The matrix is deliberately additive: Qwen still returns only the compact four
field candidate contract.  This module reads typed Evidence and derives the
objective entity, scope, temporal, mechanism, and contradiction claims.  It
never treats a causal explanation string as proof by itself.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from yield_rca_core.causal_chain import (
    CausalChainAssessment,
    assess_causal_chain,
    collect_data_missing_sources,
)
from yield_rca_core.causal_hypothesis import (
    CausalClaim,
    CausalClaimStatus,
    CausalHypothesis,
    MechanismSupportSource,
)
from yield_rca_core.causal_investigation_models import (
    CandidateClaimedScopeKind,
    CandidateScopeRelation,
    CandidateSemanticProfile,
    CausalLaneRecord,
)
from yield_rca_core.evidence_models import EntityType, Evidence, EvidenceType

_EXPOSURE_TYPES = {
    EvidenceType.LOT_CONTEXT.value,
    EvidenceType.PROCESS_EXPOSURE.value,
    EvidenceType.EQUIPMENT_EXPOSURE.value,
    EvidenceType.IMPACT_SCOPE.value,
    EvidenceType.EXCURSION_WINDOW.value,
}
_PROCESS_TYPES = {
    EvidenceType.RECIPE_CHANGE.value,
    EvidenceType.HOLD_EVENT.value,
    EvidenceType.PARAMETER_DEVIATION.value,
    EvidenceType.TREND_DEVIATION.value,
    EvidenceType.OOC_EVENT.value,
    EvidenceType.SPC_VIOLATION.value,
}
_OUTCOME_TYPES = {
    EvidenceType.DEFECT_SIGNAL.value,
    EvidenceType.METROLOGY_DEVIATION.value,
    EvidenceType.ELECTRICAL_FAILURE.value,
}
_CONTROL_TYPES = {EvidenceType.NEGATIVE_SIGNAL.value}
_KNOWLEDGE_TYPES = {
    EvidenceType.HISTORICAL_CASE_MATCH.value,
    EvidenceType.ENGINEERING_NOTE.value,
}
_CRITICAL_CLAIMS = {
    CausalClaim.EQUIPMENT.value,
    CausalClaim.CHAMBER.value,
    CausalClaim.OPERATION.value,
    CausalClaim.PARAMETER.value,
    CausalClaim.OUTCOME.value,
    CausalClaim.MECHANISM.value,
    CausalClaim.CONTRADICTION.value,
    CausalClaim.TEMPORAL.value,
    CausalClaim.SCOPE.value,
}
_ENTITY_CLAIMS = {
    CausalClaim.EQUIPMENT.value: EntityType.EQUIPMENT.value,
    CausalClaim.CHAMBER.value: EntityType.CHAMBER.value,
    CausalClaim.OPERATION.value: EntityType.OPERATION.value,
}
_STRUCTURED_TOKEN = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")
_EQUIPMENT_PREFIXES = (
    "ald_",
    "cmp_",
    "cvd_",
    "diff_",
    "etch_",
    "eq_",
    "implant_",
    "litho_",
    "pvd_",
    "wet_",
)
_PARAMETER_HINTS = {
    "bias",
    "current",
    "deposition",
    "flow",
    "force",
    "pressure",
    "rate",
    "removal",
    "resistance",
    "rf",
    "slurry",
    "speed",
    "temperature",
    "thickness",
    "uniformity",
    "vacuum",
    "voltage",
}
_OUTCOME_HINTS = {
    "bridge",
    "crack",
    "defect",
    "erosion",
    "fail",
    "failure",
    "leakage",
    "nonuniformity",
    "open",
    "scratch",
    "short",
    "void",
}
_SEMANTIC_STOP_WORDS = {
    "abnormal",
    "candidate",
    "causal",
    "cause",
    "caused",
    "causing",
    "chamber",
    "control",
    "create",
    "created",
    "creates",
    "creating",
    "current",
    "equipment",
    "evidence",
    "failure",
    "lot",
    "observed",
    "operation",
    "process",
    "signal",
}
_GENERIC_CAUSAL_LINK_TOKENS = {
    "abnormal",
    "abnormality",
    "across",
    "affect",
    "affected",
    "affecting",
    "all",
    "among",
    "and",
    "anomaly",
    "another",
    "are",
    "arising",
    "both",
    "can",
    "cause",
    "caused",
    "causes",
    "causing",
    "change",
    "changed",
    "compatible",
    "concurrent",
    "consistent",
    "control",
    "critical",
    "data",
    "delta",
    "defect",
    "defects",
    "deviation",
    "deviations",
    "direct",
    "directly",
    "disrupt",
    "disrupted",
    "disrupting",
    "disrupts",
    "drift",
    "due",
    "elevated",
    "evidence",
    "excursion",
    "exposure",
    "explain",
    "explained",
    "explains",
    "failure",
    "failures",
    "for",
    "high",
    "increase",
    "increased",
    "indicate",
    "indicated",
    "indicates",
    "instability",
    "into",
    "lead",
    "leading",
    "leads",
    "likely",
    "low",
    "lots",
    "may",
    "multiple",
    "non",
    "outcome",
    "pattern",
    "patterns",
    "primary",
    "produce",
    "produced",
    "produces",
    "producing",
    "reduced",
    "reported",
    "record",
    "records",
    "result",
    "resulting",
    "results",
    "shared",
    "signature",
    "same",
    "than",
    "that",
    "the",
    "then",
    "this",
    "through",
    "variation",
    "values",
    "wafers",
    "when",
    "which",
    "while",
    "with",
    "would",
}


def _compact(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _tokens(value: object) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(value).casefold())
        if len(token) >= 3
    }


def _candidate_text(candidate: CausalHypothesis) -> str:
    return f"{candidate.root_cause} {candidate.causal_explanation}"


def _candidate_structured_tokens(candidate: CausalHypothesis) -> list[str]:
    return [item.casefold() for item in _STRUCTURED_TOKEN.findall(_candidate_text(candidate))]


def _explicit_entity_tokens(
    candidate: CausalHypothesis,
    entity_type: str,
) -> set[str]:
    """Extract only explicit claims for one entity type.

    A chamber token must not make an unrelated Operation look conflicted, and
    an Operation number is considered explicit only when it is labelled as an
    operation in the candidate text.
    """

    text = _candidate_text(candidate).casefold()
    structured = set(_candidate_structured_tokens(candidate))
    if entity_type == EntityType.EQUIPMENT.value:
        return {
            token
            for token in structured
            if token.startswith(_EQUIPMENT_PREFIXES)
        }
    if entity_type == EntityType.CHAMBER.value:
        explicit = {
            token
            for token in structured
            if token.startswith("ch_") or re.search(r"_ch[0-9a-z]+(?:_|$)", token)
        }
        explicit.update(
            match.group(1)
            for match in re.finditer(
                r"\bchamber\s*[:#_-]?\s*([a-z0-9_]+)", text
            )
        )
        return explicit
    if entity_type == EntityType.OPERATION.value:
        explicit = {token for token in structured if token.startswith("op_")}
        explicit.update(
            match.group(1)
            for match in re.finditer(
                r"\b(?:operation(?:_no)?|op)\s*[:#_-]?\s*([a-z0-9_]+)",
                text,
            )
        )
        return explicit
    return set()


def _label_tokens(value: object) -> set[str]:
    return {
        token
        for token in _tokens(value)
        if token not in _SEMANTIC_STOP_WORDS
        and not token.startswith(("ev", "eq", "ch", "op", "lot"))
    }


def _candidate_matches_label(candidate: CausalHypothesis, label: str) -> bool:
    text = _candidate_text(candidate)
    compact_label = _compact(label)
    if len(compact_label) >= 6 and compact_label in _compact(text):
        return True
    label_tokens = _label_tokens(label)
    candidate_tokens = _label_tokens(text)
    overlap = candidate_tokens & label_tokens
    return bool(overlap) and (
        overlap == label_tokens
        or len(overlap) >= 2
        or any(len(token) >= 5 for token in overlap)
    )


def _metadata_labels(evidence: Evidence, keys: set[str]) -> set[str]:
    return {
        str(value)
        for key, value in _metadata_and_entity_values(evidence)
        if key in keys and isinstance(value, str | int | float)
    }


def _evidence_text(evidence: Evidence) -> str:
    metadata = " ".join(
        str(value)
        for value in evidence.metadata.values()
        if isinstance(value, str | int | float | bool)
    )
    return " ".join(
        value
        for value in (
            evidence.summary,
            evidence.observation or "",
            evidence.source_field or "",
            metadata,
            *(entity.entity_id for entity in evidence.entities),
        )
        if value
    )


def _explicit_mechanism_intermediate_role(evidence: Evidence) -> str | None:
    """Return a source-traceable physical-intermediate role, if present."""

    if not evidence.is_typed:
        return None
    role = str(
        evidence.metadata.get(
            "causal_role",
            evidence.metadata.get("mechanism_role", ""),
        )
    ).strip().casefold()
    if role in {"mechanism_intermediate", "physical_intermediate"}:
        return role
    if evidence.metadata.get("mechanism_intermediate") is True:
        return "mechanism_intermediate"
    observation_role = str(
        evidence.metadata.get("observation_role", "")
    ).strip().casefold()
    observation_provenance = evidence.metadata.get(
        "observation_role_provenance"
    )
    if (
        observation_role == "physical_inspection_observation"
        and evidence.source_type == "user"
        and evidence.source_tool == "incident_observation_extractor"
        and isinstance(observation_provenance, Mapping)
        and str(observation_provenance.get("source_quote", "")).strip()
        and any(
            entity.entity_type == EntityType.DEFECT.value
            for entity in evidence.entities
        )
    ):
        return observation_role
    return None


def is_relevant_mechanism_intermediate(
    evidence: Evidence,
    candidate: CausalHypothesis,
) -> bool:
    """Check typed intermediate provenance and semantic relevance to a Candidate."""

    if _explicit_mechanism_intermediate_role(evidence) is None:
        return False
    candidate_tokens = _label_tokens(_candidate_text(candidate))
    evidence_tokens = _label_tokens(_evidence_text(evidence))
    return len(candidate_tokens & evidence_tokens) >= 2 or any(
        _candidate_matches_label(candidate, entity.entity_id)
        for entity in evidence.entities
        if entity.entity_type
        in {
            EntityType.PARAMETER.value,
            EntityType.DEFECT.value,
            EntityType.WAT_ITEM.value,
        }
    )


_DIRECTION_ALIASES = {
    "high": "high",
    "higher": "high",
    "increase": "high",
    "increased": "high",
    "increasing": "high",
    "rise": "high",
    "rising": "high",
    "elevated": "high",
    "above": "high",
    "positive": "high",
    "up": "high",
    "low": "low",
    "lower": "low",
    "decrease": "low",
    "decreased": "low",
    "decreasing": "low",
    "drop": "low",
    "dropped": "low",
    "fall": "low",
    "falling": "low",
    "reduced": "low",
    "below": "low",
    "negative": "low",
    "down": "low",
}
_DIRECTION_KEYS = {
    "direction",
    "parameter_direction",
    "trend_direction",
    "same_side_direction",
    "change_direction",
    "delta_direction",
}
_MAGNITUDE_KEYS = {
    "magnitude",
    "delta",
    "delta_value",
    "delta_percent",
    "avg_delta_percent",
    "relative_change",
    "difference",
    "deviation",
    "observed_value",
    "avg_observed",
    "avg_baseline",
    "target_mean",
    "center_line",
    "sigma",
    "lower_control_limit",
    "upper_control_limit",
    "mean_z_score",
    "z_score",
}
_WINDOW_START_KEYS = {
    "start",
    "start_date",
    "window_start",
    "target_window_start",
    "excursion_start",
    "processing_start",
}
_WINDOW_END_KEYS = {
    "end",
    "end_date",
    "window_end",
    "target_window_end",
    "excursion_end",
    "processing_end",
}


def _json_safe(value: object) -> object:
    """Convert immutable Evidence metadata into JSON-safe diagnostic values."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _metadata_and_entity_values(evidence: Evidence) -> list[tuple[str, object]]:
    values: list[tuple[str, object]] = [
        (str(key).casefold(), value) for key, value in evidence.metadata.items()
    ]
    for entity in evidence.entities:
        values.extend(
            (str(key).casefold(), value) for key, value in entity.attributes.items()
        )
    return values


def _normalise_direction(value: object) -> str | None:
    token = str(value).casefold().strip().replace("-", "_")
    if token in _DIRECTION_ALIASES:
        return _DIRECTION_ALIASES[token]
    tokens = re.findall(r"[a-z]+", token)
    for item in tokens:
        if item in _DIRECTION_ALIASES:
            return _DIRECTION_ALIASES[item]
    return None


def _directions(evidence: Evidence) -> list[str]:
    directions: list[str] = []
    for key, value in _metadata_and_entity_values(evidence):
        if key in _DIRECTION_KEYS:
            direction = _normalise_direction(value)
            if direction is not None:
                directions.append(direction)
        elif key in {
            "delta",
            "delta_value",
            "delta_percent",
            "avg_delta_percent",
            "relative_change",
        }:
            try:
                numeric = float(str(value))
            except (TypeError, ValueError):
                continue
            if numeric > 0:
                directions.append("high")
            elif numeric < 0:
                directions.append("low")
        elif key in {"point_violations", "violations"} and isinstance(value, (list, tuple)):
            for violation in value:
                if isinstance(violation, Mapping):
                    direction = _normalise_direction(violation.get("direction"))
                    if direction is not None:
                        directions.append(direction)
    # Observations are a fallback for typed tools that put the direction only
    # in prose.  Structured metadata always takes precedence when available.
    if not directions:
        for token in re.findall(r"[a-z]+", _evidence_text(evidence).casefold()):
            direction = _DIRECTION_ALIASES.get(token)
            if direction is not None:
                directions.append(direction)
    return list(dict.fromkeys(directions))


def _magnitudes(evidence: Evidence) -> list[object]:
    values: list[object] = []
    for key, value in _metadata_and_entity_values(evidence):
        if key in _MAGNITUDE_KEYS and isinstance(value, (int, float, str)):
            values.append(_json_safe(value))
    return list(dict.fromkeys(values))


def _processing_windows(evidence: Evidence) -> list[dict[str, object]]:
    windows: list[dict[str, object]] = []

    def visit(value: object, path: str = "") -> None:
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
            return
        if not isinstance(value, Mapping):
            return
        normalized = {str(key).casefold(): item for key, item in value.items()}
        starts = {
            key: item
            for key, item in normalized.items()
            if key in _WINDOW_START_KEYS and isinstance(item, str | int | float)
        }
        ends = {
            key: item
            for key, item in normalized.items()
            if key in _WINDOW_END_KEYS and isinstance(item, str | int | float)
        }
        if starts and ends:
            start = next(iter(starts.values()))
            end = next(iter(ends.values()))
            windows.append(
                {
                    "start": _json_safe(start),
                    "end": _json_safe(end),
                    "path": path or "metadata",
                }
            )
        for key, item in value.items():
            visit(item, f"{path}.{key}" if path else str(key))

    visit(evidence.metadata)
    for entity in evidence.entities:
        visit(entity.attributes, f"entity:{entity.entity_id}")
    # A timestamp without a declared interval remains useful, but is not a
    # fabricated processing window.
    return windows


def _outcome_facts(evidence: Evidence) -> dict[str, object]:
    return {
        "evidence_id": evidence.evidence_id,
        "evidence_type": evidence.evidence_type,
        "entities": [
            {"entity_type": entity.entity_type, "entity_id": entity.entity_id}
            for entity in evidence.entities
            if entity.entity_type in {
                EntityType.DEFECT.value,
                EntityType.WAT_ITEM.value,
                EntityType.PRODUCT.value,
            }
        ],
        "observation": evidence.observation,
    }


def _is_approved_knowledge(evidence: Evidence) -> bool:
    """Return whether a knowledge record is explicitly engineer-approved."""

    if evidence.source_type != "knowledge" or evidence.evidence_type not in _KNOWLEDGE_TYPES:
        return False
    statuses = [
        str(value).upper()
        for key, value in _metadata_and_entity_values(evidence)
        if key == "validation_status"
    ]
    return bool(statuses) and all(status == "CONFIRMED" for status in statuses)


def _facts(evidence: Iterable[Evidence]) -> dict[str, Any]:
    items = list(evidence)
    entities: dict[str, list[str]] = {}
    for item in items:
        for entity in item.entities:
            entities.setdefault(entity.entity_type, []).append(entity.entity_id)
    parameter_direction_facts = [
        {
            "evidence_id": item.evidence_id,
            "parameters": [
                entity.entity_id
                for entity in item.entities
                if entity.entity_type == EntityType.PARAMETER.value
            ],
            "directions": _directions(item),
        }
        for item in items
        if _directions(item)
    ]
    parameter_magnitude_facts = [
        {
            "evidence_id": item.evidence_id,
            "parameters": [
                entity.entity_id
                for entity in item.entities
                if entity.entity_type == EntityType.PARAMETER.value
            ],
            "magnitudes": _magnitudes(item),
        }
        for item in items
        if _magnitudes(item)
    ]
    processing_window_facts = [
        {
            "evidence_id": item.evidence_id,
            "windows": _processing_windows(item),
        }
        for item in items
        if _processing_windows(item)
    ]
    outcome_facts = [
        _outcome_facts(item)
        for item in items
        if item.evidence_type in _OUTCOME_TYPES
    ]
    knowledge_ids = [item.evidence_id for item in items if _is_approved_knowledge(item)]
    negative_ids = [
        item.evidence_id for item in items if item.evidence_type in _CONTROL_TYPES
    ]
    return {
        "evidence_ids": [item.evidence_id for item in items],
        "evidence_types": sorted({str(item.evidence_type) for item in items if item.evidence_type}),
        "source_agents": sorted({str(item.source_agent) for item in items if item.source_agent}),
        "entity_ids_by_type": {
            key: list(dict.fromkeys(values)) for key, values in sorted(entities.items())
        },
        "lots": sorted(
            {
                entity.entity_id
                for item in items
                for entity in item.entities
                if entity.entity_type == EntityType.LOT.value
            }
        ),
        "timestamps": sorted({item.timestamp for item in items if item.timestamp}),
        "parameter_fields": sorted(
            {
                str(item.source_field)
                for item in items
                if item.source_field and item.evidence_type in _PROCESS_TYPES
            }
        ),
        "parameter_directions": parameter_direction_facts,
        "parameter_magnitudes": parameter_magnitude_facts,
        "processing_windows": processing_window_facts,
        "outcomes": outcome_facts,
        "knowledge_mechanism_support": knowledge_ids,
        "normal_controls": negative_ids,
        "negative_signals": negative_ids,
        # This key is populated when _facts is used for the contradiction lane;
        # it intentionally remains Evidence-ID based and never infers physics.
        "contradictions": negative_ids,
    }


@dataclass(frozen=True)
class CausalClaimResult:
    """One deterministic claim result in a candidate matrix."""

    claim: str
    status: str
    evidence_ids: tuple[str, ...] = ()
    reason: str = ""
    facts: Mapping[str, Any] = field(default_factory=dict)
    support_source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        serialized_facts = _json_safe(self.facts or {})
        if not isinstance(serialized_facts, dict):
            raise TypeError("causal claim facts must serialize as an object")
        return {
            "claim": self.claim,
            "status": self.status,
            "evidence_ids": list(self.evidence_ids),
            "reason": self.reason,
            "facts": serialized_facts,
            "support_source": self.support_source,
        }


@dataclass(frozen=True)
class CausalEvidenceMatrix:
    """Evidence matrix for one candidate; all fields are Python-derived."""

    candidate: CausalHypothesis
    claims: Mapping[str, CausalClaimResult]
    invalid_evidence_ids: tuple[str, ...] = ()
    data_missing_evidence_ids: tuple[str, ...] = ()
    data_missing_sources: tuple[Mapping[str, Any], ...] = ()
    causal_chain: CausalChainAssessment | None = None

    @property
    def status(self) -> str:
        statuses = {item.status for item in self.claims.values()}
        if any(
            self.claims.get(claim) is not None
            and self.claims[claim].status == CausalClaimStatus.CONFLICTED.value
            for claim in _CRITICAL_CLAIMS
        ):
            return CausalClaimStatus.CONFLICTED.value
        if not self.claims or statuses <= {CausalClaimStatus.UNAVAILABLE.value}:
            return CausalClaimStatus.UNAVAILABLE.value
        if all(
            self.claims.get(claim) is not None
            and self.claims[claim].status == CausalClaimStatus.SUPPORTED.value
            for claim in _CRITICAL_CLAIMS
        ):
            return CausalClaimStatus.SUPPORTED.value
        return CausalClaimStatus.INCOMPLETE.value

    @property
    def has_critical_conflict(self) -> bool:
        return self.status == CausalClaimStatus.CONFLICTED.value

    @property
    def mechanism_status(self) -> str:
        return self.claims[CausalClaim.MECHANISM.value].status

    @property
    def mechanism_support_source(self) -> str | None:
        return self.claims[CausalClaim.MECHANISM.value].support_source

    @property
    def causal_chain_completeness(self) -> str:
        """Return the Python-derived chain status for this candidate."""

        if self.causal_chain is None:
            return assess_causal_chain(
                self.claims,
                data_missing_evidence_ids=self.data_missing_evidence_ids,
            ).status
        return self.causal_chain.status

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_cause": self.candidate.root_cause,
            "claims": {
                claim: result.to_dict() for claim, result in self.claims.items()
            },
            "status": self.status,
            "invalid_evidence_ids": list(self.invalid_evidence_ids),
            "data_missing_evidence_ids": list(self.data_missing_evidence_ids),
            "data_missing_sources": [
                _json_safe(item) for item in self.data_missing_sources
            ],
            "causal_chain": (
                self.causal_chain.to_dict()
                if self.causal_chain is not None
                else assess_causal_chain(
                    self.claims,
                    data_missing_evidence_ids=self.data_missing_evidence_ids,
                ).to_dict()
            ),
            "causal_chain_completeness": self.causal_chain_completeness,
            "mechanism_support_source": self.mechanism_support_source,
        }


def compact_causal_evidence_matrix_for_prompt(
    matrix: CausalEvidenceMatrix,
) -> dict[str, Any]:
    """Return a bounded Matrix projection for model comparison/challenge.

    Full claim facts remain Python-owned in ``RCAState``.  Qwen needs the
    validation outcome, reasons, support provenance, and Evidence IDs—not the
    repeated entity/fact arrays already available through compact Evidence
    cards.
    """

    chain = (
        matrix.causal_chain
        if matrix.causal_chain is not None
        else assess_causal_chain(
            matrix.claims,
            data_missing_evidence_ids=matrix.data_missing_evidence_ids,
        )
    )
    return {
        "root_cause": matrix.candidate.root_cause,
        "status": matrix.status,
        "claims": {
            claim: {
                "status": result.status,
                "reason": result.reason,
                "evidence_ids": list(result.evidence_ids),
                "support_source": result.support_source,
            }
            for claim, result in matrix.claims.items()
        },
        "invalid_evidence_ids": list(matrix.invalid_evidence_ids),
        "data_missing_evidence_ids": list(matrix.data_missing_evidence_ids),
        "causal_chain": {
            "status": chain.status,
            "stages": dict(chain.stages),
            "missing_stages": list(chain.missing_stages),
            "conflicting_stages": list(chain.conflicting_stages),
            "data_missing_evidence_ids": list(chain.data_missing_evidence_ids),
            "reason": chain.reason,
        },
        "causal_chain_completeness": matrix.causal_chain_completeness,
        "mechanism_support_source": (
            matrix.claims[CausalClaim.MECHANISM.value].support_source
            if CausalClaim.MECHANISM.value in matrix.claims
            else None
        ),
        "projection_audit": {
            "omitted_claim_fact_groups": sum(
                1 for result in matrix.claims.values() if result.facts
            ),
            "full_matrix_retained_by_python": True,
        },
    }


def _result(
    claim: CausalClaim | str,
    status: CausalClaimStatus | str,
    evidence: Iterable[Evidence],
    reason: str,
    *,
    support_source: MechanismSupportSource | str | None = None,
    fact_overrides: Mapping[str, Any] | None = None,
) -> CausalClaimResult:
    items = list(evidence)
    facts = _facts(items)
    if fact_overrides:
        facts.update(fact_overrides)
    if str(claim) == CausalClaim.CONTRADICTION.value:
        facts["contradictions"] = [item.evidence_id for item in items]
    return CausalClaimResult(
        claim=str(claim),
        status=str(status),
        evidence_ids=tuple(item.evidence_id for item in items),
        reason=reason,
        facts=facts,
        support_source=str(support_source) if support_source is not None else None,
    )


def _matching_entity_claim(
    *,
    claim: str,
    entity_type: str,
    candidate: CausalHypothesis,
    evidence: list[Evidence],
) -> CausalClaimResult:
    matches = [
        item
        for item in evidence
        if any(entity.entity_type == entity_type for entity in item.entities)
    ]
    if not matches:
        return _result(
            claim,
            CausalClaimStatus.UNAVAILABLE,
            (),
            "No typed Evidence supplies this entity.",
        )
    ids = [
        entity.entity_id
        for item in matches
        for entity in item.entities
        if entity.entity_type == entity_type
    ]
    explicit_tokens = _explicit_entity_tokens(candidate, entity_type)
    matching_ids = [
        entity_id
        for entity_id in ids
        if any(
            _compact(token) == _compact(entity_id)
            or (
                entity_type == EntityType.EQUIPMENT.value
                and _compact(token).startswith(_compact(entity_id))
                and _compact(token)[len(_compact(entity_id)) :].startswith("ch")
            )
            or (
                entity_type == EntityType.CHAMBER.value
                and _compact(token).endswith(_compact(entity_id))
            )
            for token in explicit_tokens
        )
    ]
    if matching_ids:
        return _result(
            claim,
            CausalClaimStatus.SUPPORTED,
            matches,
            f"Candidate names typed {entity_type} Evidence: {sorted(set(matching_ids))}.",
        )
    if explicit_tokens:
        return _result(
            claim,
            CausalClaimStatus.CONFLICTED,
            matches,
            f"Candidate entity claims do not match typed {entity_type} IDs: {sorted(set(ids))}.",
        )
    return _result(
        claim,
        CausalClaimStatus.INCOMPLETE,
        matches,
        f"Typed {entity_type} Evidence exists but the candidate does not identify it explicitly.",
    )


def _parameter_claim(candidate: CausalHypothesis, evidence: list[Evidence]) -> CausalClaimResult:
    process = [item for item in evidence if item.evidence_type in _PROCESS_TYPES]
    if not process:
        return _result(
            CausalClaim.PARAMETER,
            CausalClaimStatus.UNAVAILABLE,
            (),
            "No process Evidence is available.",
        )
    parameter_items = [
        item
        for item in process
        if any(entity.entity_type == EntityType.PARAMETER.value for entity in item.entities)
        or item.source_field
    ]
    if not parameter_items:
        return _result(
            CausalClaim.PARAMETER,
            CausalClaimStatus.UNAVAILABLE,
            process,
            "Process Evidence exists but supplies no typed parameter or source field.",
        )
    labels_by_id: dict[str, set[str]] = {}
    for item in parameter_items:
        labels = {
            entity.entity_id
            for entity in item.entities
            if entity.entity_type == EntityType.PARAMETER.value
        }
        if item.source_field:
            labels.add(item.source_field)
        labels.update(
            _metadata_labels(
                item,
                {"parameter", "parameter_id", "parameter_name", "metric_name"},
            )
        )
        labels_by_id[item.evidence_id] = labels
    aligned_items = [
        item
        for item in parameter_items
        if any(
            _candidate_matches_label(candidate, label)
            for label in labels_by_id[item.evidence_id]
        )
    ]
    candidate_directions = {
        direction
        for token in re.findall(r"[a-z]+", _candidate_text(candidate).casefold())
        if (direction := _DIRECTION_ALIASES.get(token)) is not None
    }
    evidence_directions = {
        direction for item in aligned_items for direction in _directions(item)
    }
    if candidate_directions and evidence_directions and not (
        candidate_directions & evidence_directions
    ):
        return _result(
            CausalClaim.PARAMETER,
            CausalClaimStatus.CONFLICTED,
            aligned_items,
            (
                "Candidate parameter direction conflicts with typed process facts: "
                f"candidate={sorted(candidate_directions)}, "
                f"evidence={sorted(evidence_directions)}."
            ),
        )
    if aligned_items:
        matched_labels = sorted(
            {
                label
                for item in aligned_items
                for label in labels_by_id[item.evidence_id]
                if _candidate_matches_label(candidate, label)
            }
        )
        return _result(
            CausalClaim.PARAMETER,
            CausalClaimStatus.SUPPORTED,
            aligned_items,
            f"Candidate identifies typed process parameter facts: {matched_labels}.",
        )
    candidate_parameter_hints = sorted(_label_tokens(_candidate_text(candidate)) & _PARAMETER_HINTS)
    if candidate_parameter_hints:
        return _result(
            CausalClaim.PARAMETER,
            CausalClaimStatus.CONFLICTED,
            parameter_items,
            "Candidate explicitly names a parameter that does not match typed "
            f"process facts: {candidate_parameter_hints}.",
        )
    return _result(
        CausalClaim.PARAMETER,
        CausalClaimStatus.INCOMPLETE,
        parameter_items,
        "Process anomaly Evidence exists, but the candidate parameter is not aligned to it.",
    )


def _outcome_claim(candidate: CausalHypothesis, evidence: list[Evidence]) -> CausalClaimResult:
    outcomes = [item for item in evidence if item.evidence_type in _OUTCOME_TYPES]
    if not outcomes:
        return _result(
            CausalClaim.OUTCOME,
            CausalClaimStatus.UNAVAILABLE,
            (),
            "No typed product outcome Evidence is available.",
        )
    labels_by_id: dict[str, set[str]] = {}
    for item in outcomes:
        labels = {
            entity.entity_id
            for entity in item.entities
            if entity.entity_type in {EntityType.DEFECT.value, EntityType.WAT_ITEM.value}
        }
        if item.source_field:
            labels.add(item.source_field)
        labels.update(
            _metadata_labels(
                item,
                {"defect", "defect_type", "fail_mode", "metric_name", "wat_item"},
            )
        )
        labels_by_id[item.evidence_id] = labels
    aligned = [
        item
        for item in outcomes
        if any(
            _candidate_matches_label(candidate, label)
            for label in labels_by_id[item.evidence_id]
        )
    ]
    if aligned:
        matched_labels = sorted(
            {
                label
                for item in aligned
                for label in labels_by_id[item.evidence_id]
                if _candidate_matches_label(candidate, label)
            }
        )
        return _result(
            CausalClaim.OUTCOME,
            CausalClaimStatus.SUPPORTED,
            aligned,
            f"Candidate identifies typed product outcome facts: {matched_labels}.",
        )
    candidate_outcome_hints = sorted(_label_tokens(_candidate_text(candidate)) & _OUTCOME_HINTS)
    if candidate_outcome_hints and any(labels_by_id.values()):
        return _result(
            CausalClaim.OUTCOME,
            CausalClaimStatus.CONFLICTED,
            outcomes,
            "Candidate explicitly names an outcome that does not match typed "
            f"outcome facts: {candidate_outcome_hints}.",
        )
    return _result(
        CausalClaim.OUTCOME,
        CausalClaimStatus.INCOMPLETE,
        outcomes,
        "Product outcome Evidence exists but the candidate explanation does not name it.",
    )


def _mechanism_claim(
    candidate: CausalHypothesis,
    evidence: list[Evidence],
    *,
    parameter: CausalClaimResult,
    outcome: CausalClaimResult,
) -> CausalClaimResult:
    process = [item for item in evidence if item.evidence_type in _PROCESS_TYPES]
    knowledge = [item for item in evidence if _is_approved_knowledge(item)]
    rule = [
        item
        for item in evidence
        if item.metadata.get("mechanism_rule")
        or str(item.source_tool or "").casefold().startswith("mechanism_rule")
    ]
    def relevant(items: list[Evidence]) -> list[Evidence]:
        candidate_tokens = _label_tokens(_candidate_text(candidate))
        return [
            item
            for item in items
            if len(candidate_tokens & _label_tokens(_evidence_text(item))) >= 2
            or any(
                _candidate_matches_label(candidate, entity.entity_id)
                for entity in item.entities
                if entity.entity_type
                in {
                    EntityType.PARAMETER.value,
                    EntityType.DEFECT.value,
                    EntityType.WAT_ITEM.value,
                }
            )
        ]

    relevant_rules = relevant(rule)
    relevant_knowledge = relevant(knowledge)
    explicit_intermediates = [
        item
        for item in evidence
        if _explicit_mechanism_intermediate_role(item) is not None
    ]
    relevant_intermediates = [
        item
        for item in explicit_intermediates
        if is_relevant_mechanism_intermediate(item, candidate)
    ]
    empirical_discrimination = [
        item
        for item in evidence
        if str(item.metadata.get("causal_support_kind", "")).casefold()
        in {"intervention", "recovery", "dose_response", "matched_comparison"}
        and str(item.metadata.get("validation_status", "")).casefold()
        in {"confirmed", "approved"}
    ]
    relevant_empirical_discrimination = relevant(empirical_discrimination)
    claim_entity_tokens = {
        token
        for claim_result in (parameter, outcome)
        for values in claim_result.facts.get("entity_ids_by_type", {}).values()
        if isinstance(values, list)
        for value in values
        for token in _label_tokens(value)
    }
    claim_entity_tokens.update(
        token
        for claim_result in (parameter, outcome)
        for value in claim_result.facts.get("parameter_fields", [])
        if isinstance(value, str)
        for token in _label_tokens(value)
    )
    proposed_bridge_terms = sorted(
        token
        for token in _label_tokens(candidate.causal_explanation)
        if token not in claim_entity_tokens
        and token not in _GENERIC_CAUSAL_LINK_TOKENS
        and not any(character.isdigit() for character in token)
    )
    parameter_ids = set(parameter.evidence_ids)
    outcome_ids = set(outcome.evidence_ids)
    parameter_items = [item for item in evidence if item.evidence_id in parameter_ids]
    outcome_items = [item for item in evidence if item.evidence_id in outcome_ids]
    parameter_lots = {
        entity.entity_id
        for item in parameter_items
        for entity in item.entities
        if entity.entity_type == EntityType.LOT.value
    }
    outcome_lots = {
        entity.entity_id
        for item in outcome_items
        for entity in item.entities
        if entity.entity_type == EntityType.LOT.value
    }
    shared_empirical_lots = sorted(parameter_lots & outcome_lots)
    mechanism_facts = {
        "proposed_physical_bridge_terms": proposed_bridge_terms,
        "mechanism_expression_present": bool(proposed_bridge_terms),
        "parameter_evidence_ids": sorted(parameter_ids),
        "outcome_evidence_ids": sorted(outcome_ids),
        "empirical_shared_lot_ids": shared_empirical_lots,
        "observed_intermediate_evidence_ids": [
            item.evidence_id for item in relevant_intermediates
        ],
        "observed_intermediate_provenance": {
            item.evidence_id: {
                "causal_role": _explicit_mechanism_intermediate_role(item),
                "source_type": item.source_type,
                "source_id": item.source_id,
                "source_table": item.source_table,
                "source_field": item.source_field,
                "source_agent": item.source_agent,
                "source_tool": item.source_tool,
                "causal_role_provenance": _json_safe(
                    item.metadata.get("causal_role_provenance")
                ),
                "observation_role_provenance": _json_safe(
                    item.metadata.get("observation_role_provenance")
                ),
            }
            for item in relevant_intermediates
        },
        "empirical_discrimination_evidence_ids": [
            item.evidence_id for item in relevant_empirical_discrimination
        ],
    }
    if not proposed_bridge_terms:
        return _result(
            CausalClaim.MECHANISM,
            CausalClaimStatus.INCOMPLETE,
            [*parameter_items, *outcome_items, *relevant_knowledge, *relevant_rules],
            (
                "The candidate restates an abnormal parameter and outcome but does "
                "not identify an intervening physical process."
            ),
            support_source=MechanismSupportSource.LLM_EXPLANATION_ONLY,
            fact_overrides=mechanism_facts,
        )
    if relevant_rules:
        return _result(
            CausalClaim.MECHANISM,
            CausalClaimStatus.SUPPORTED,
            relevant_rules,
            "An explicit Python-registered mechanism rule supports the relationship.",
            support_source=MechanismSupportSource.RULE,
            fact_overrides=mechanism_facts,
        )
    if relevant_knowledge:
        return _result(
            CausalClaim.MECHANISM,
            CausalClaimStatus.SUPPORTED,
            relevant_knowledge,
            (
                "Approved Knowledge supports the engineering mechanism; it does not "
                "prove current-Lot occurrence alone."
            ),
            support_source=MechanismSupportSource.APPROVED_KNOWLEDGE,
            fact_overrides=mechanism_facts,
        )
    if relevant_intermediates and shared_empirical_lots:
        return _result(
            CausalClaim.MECHANISM,
            CausalClaimStatus.SUPPORTED,
            relevant_intermediates,
            (
                "Typed current-Lot Evidence explicitly records a physical "
                "intermediate relevant to the proposed mechanism."
            ),
            support_source=MechanismSupportSource.OBSERVED_INTERMEDIATE,
            fact_overrides=mechanism_facts,
        )
    if relevant_empirical_discrimination and shared_empirical_lots:
        return _result(
            CausalClaim.MECHANISM,
            CausalClaimStatus.SUPPORTED,
            relevant_empirical_discrimination,
            (
                "Validated intervention, recovery, dose-response, or matched "
                "comparison Evidence discriminates the proposed mechanism."
            ),
            support_source=MechanismSupportSource.EMPIRICAL_DISCRIMINATION,
            fact_overrides=mechanism_facts,
        )
    if (
        process
        and parameter.status == CausalClaimStatus.SUPPORTED.value
        and outcome.status == CausalClaimStatus.SUPPORTED.value
        and shared_empirical_lots
    ):
        return _result(
            CausalClaim.MECHANISM,
            CausalClaimStatus.PLAUSIBLE,
            [*parameter_items, *outcome_items],
            (
                "The candidate states an explicit physical bridge and aligned "
                "parameter/outcome Evidence converges on a shared current-Lot "
                "scope. This makes the bridge plausible, but co-occurrence does "
                "not prove the intervening physical process."
            ),
            support_source=MechanismSupportSource.EMPIRICAL_CONVERGENCE,
            fact_overrides=mechanism_facts,
        )
    if process or knowledge or rule or outcome.status != CausalClaimStatus.UNAVAILABLE.value:
        return _result(
            CausalClaim.MECHANISM,
            CausalClaimStatus.INCOMPLETE,
            [*process, *knowledge, *rule],
            (
                "The causal explanation remains plausible, but current Evidence does "
                "not converge on the same mechanism. Approved Knowledge or rule "
                "Evidence, when present, is not relevant to this candidate."
            ),
            support_source=MechanismSupportSource.LLM_EXPLANATION_ONLY,
            fact_overrides=mechanism_facts,
        )
    return _result(
        CausalClaim.MECHANISM,
        CausalClaimStatus.UNAVAILABLE,
        (),
        "No current-Lot process or outcome Evidence can test the mechanism.",
        support_source=MechanismSupportSource.LLM_EXPLANATION_ONLY,
        fact_overrides=mechanism_facts,
    )


def build_causal_evidence_matrix(
    candidate: CausalHypothesis | Mapping[str, Any],
    evidence: Iterable[Evidence],
    *,
    semantic_profile: CandidateSemanticProfile | Mapping[str, Any] | None = None,
    causal_lanes: Sequence[CausalLaneRecord | Mapping[str, Any]] = (),
) -> CausalEvidenceMatrix:
    """Build a deterministic matrix for one candidate and typed Evidence set.

    ``semantic_profile`` is Qwen's declared Candidate meaning, not Evidence.
    Python validates cited Evidence against that declaration without deriving or
    broadening the claimed scope from Evidence coverage.  Omitting the profile
    preserves the legacy Matrix contract used by controlled and old-State paths.
    """

    normalized = (
        candidate
        if isinstance(candidate, CausalHypothesis)
        else CausalHypothesis.from_mapping(candidate)
    )
    evidence_items = list(evidence)
    normalized_semantic_profile: CandidateSemanticProfile | None = None
    semantic_profile_error: str | None = None
    if semantic_profile is not None:
        try:
            normalized_semantic_profile = (
                semantic_profile
                if isinstance(semantic_profile, CandidateSemanticProfile)
                else CandidateSemanticProfile.from_dict(dict(semantic_profile))
            )
        except (KeyError, TypeError, ValueError) as exc:
            semantic_profile_error = str(exc).strip() or type(exc).__name__
    evidence_by_id = {item.evidence_id: item for item in evidence_items}
    referenced_ids = set(normalized.supporting_evidence_ids) | set(
        normalized.contradicting_evidence_ids
    )
    invalid_ids = tuple(sorted(referenced_ids - set(evidence_by_id)))
    non_supporting_types = {
        EvidenceType.DATA_MISSING.value,
        EvidenceType.NEGATIVE_SIGNAL.value,
        EvidenceType.SOP_GUIDANCE.value,
    }
    non_supporting_ids = tuple(
        sorted(
            evidence_id
            for evidence_id in normalized.supporting_evidence_ids
            if evidence_id in evidence_by_id
            and evidence_by_id[evidence_id].evidence_type in non_supporting_types
        )
    )
    supporting = [
        evidence_by_id[item]
        for item in normalized.supporting_evidence_ids
        if item in evidence_by_id
        and evidence_by_id[item].evidence_type not in non_supporting_types
    ]
    contradicting = [
        evidence_by_id[item]
        for item in normalized.contradicting_evidence_ids
        if item in evidence_by_id
    ]
    claims: dict[str, CausalClaimResult] = {}
    for claim, entity_type in _ENTITY_CLAIMS.items():
        claims[claim] = _matching_entity_claim(
            claim=claim,
            entity_type=entity_type,
            candidate=normalized,
            evidence=supporting,
        )
    claims[CausalClaim.PARAMETER.value] = _parameter_claim(normalized, supporting)
    claims[CausalClaim.OUTCOME.value] = _outcome_claim(normalized, supporting)
    claims[CausalClaim.SCOPE.value] = _scope_claim(
        supporting,
        semantic_profile=normalized_semantic_profile,
        semantic_profile_error=semantic_profile_error,
        semantic_profile_supplied=semantic_profile is not None,
        causal_lanes=causal_lanes,
    )
    claims[CausalClaim.TEMPORAL.value] = _temporal_claim(supporting)
    claims[CausalClaim.CONTRADICTION.value] = (
        _result(
            CausalClaim.CONTRADICTION,
            CausalClaimStatus.CONFLICTED,
            contradicting,
            "Candidate explicitly cites contradicting Evidence.",
        )
        if contradicting or invalid_ids
        else _result(
            CausalClaim.CONTRADICTION,
            CausalClaimStatus.SUPPORTED,
            (),
            "No cited contradiction is present.",
        )
    )
    claims[CausalClaim.CONTROL.value] = _control_claim(normalized, evidence_by_id.values())
    claims[CausalClaim.MECHANISM.value] = _mechanism_claim(
        normalized,
        supporting,
        parameter=claims[CausalClaim.PARAMETER.value],
        outcome=claims[CausalClaim.OUTCOME.value],
    )
    missing_sources = collect_data_missing_sources(evidence_items)
    missing_ids = tuple(source.evidence_id for source in missing_sources)
    chain = assess_causal_chain(claims, data_missing_evidence_ids=missing_ids)
    return CausalEvidenceMatrix(
        candidate=normalized,
        claims=claims,
        invalid_evidence_ids=tuple(dict.fromkeys([*invalid_ids, *non_supporting_ids])),
        data_missing_evidence_ids=missing_ids,
        data_missing_sources=tuple(source.to_dict() for source in missing_sources),
        causal_chain=chain,
    )


def _scope_lane_payload(
    lane: CausalLaneRecord | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(lane, CausalLaneRecord):
        return lane.to_dict()
    return dict(lane)


def _scope_lane_matches_identity(
    lane: Mapping[str, Any],
    identity: Mapping[str, object],
) -> bool:
    return all(
        expected is None
        or _compact(lane.get(field_name, "")) == _compact(expected)
        for field_name, expected in identity.items()
    )


def _evidence_scope_values(
    evidence: Evidence,
    entity_type: str,
    *metadata_keys: str,
) -> set[str]:
    values = {
        _compact(entity.entity_id)
        for entity in evidence.entities
        if entity.entity_type == entity_type
    }
    values.update(
        _compact(value)
        for key in metadata_keys
        for value in (
            evidence.metadata.get(key),
        )
        if isinstance(value, str | int | float) and str(value).strip()
    )
    return values


def _evidence_covers_scope_lane(
    evidence: Evidence,
    lane: Mapping[str, Any],
) -> bool:
    lane_id = str(lane.get("lane_id", "")).strip()
    explicit_lane_id = str(evidence.metadata.get("lane_id", "")).strip()
    if explicit_lane_id:
        return explicit_lane_id == lane_id

    entity_mapping = {
        "operation": (EntityType.OPERATION.value, ("operation", "operation_no")),
        "equipment": (EntityType.EQUIPMENT.value, ("equipment", "equipment_id")),
        "chamber": (EntityType.CHAMBER.value, ("chamber", "chamber_id")),
        "recipe": (EntityType.RECIPE.value, ("recipe", "recipe_id")),
    }
    matched_identity_count = 0
    for field_name, (entity_type, metadata_keys) in entity_mapping.items():
        expected = _compact(lane.get(field_name, ""))
        observed = _evidence_scope_values(
            evidence,
            entity_type,
            *metadata_keys,
        )
        if not expected or not observed:
            continue
        if expected not in observed:
            return False
        matched_identity_count += 1

    lane_lots = {
        str(item).strip()
        for item in lane.get("exposed_lot_ids", [])
        if str(item).strip()
    }
    evidence_lots = {
        entity.entity_id
        for entity in evidence.entities
        if entity.entity_type == EntityType.LOT.value
    }
    lot_overlap = bool(lane_lots & evidence_lots)
    if evidence.evidence_type in _OUTCOME_TYPES:
        return lot_overlap or matched_identity_count >= 2
    return matched_identity_count >= 2 or lot_overlap


def _scope_claim(
    evidence: list[Evidence],
    *,
    semantic_profile: CandidateSemanticProfile | None = None,
    semantic_profile_error: str | None = None,
    semantic_profile_supplied: bool = False,
    causal_lanes: Sequence[CausalLaneRecord | Mapping[str, Any]] = (),
) -> CausalClaimResult:
    concrete_lane_ids = {
        str(item.metadata.get("lane_id", "")).strip()
        for item in evidence
        if str(item.metadata.get("lane_id", "")).strip()
    }
    if semantic_profile_supplied:
        claimed_scope_identity = {
            "operation": (
                semantic_profile.claimed_operation
                if semantic_profile is not None
                else None
            ),
            "equipment": (
                semantic_profile.claimed_equipment
                if semantic_profile is not None
                else None
            ),
            "chamber": (
                semantic_profile.claimed_chamber
                if semantic_profile is not None
                else None
            ),
            "recipe": (
                semantic_profile.claimed_recipe
                if semantic_profile is not None
                else None
            ),
        }
        fact_overrides: dict[str, Any] = {
            "semantic_profile_status": (
                "invalid" if semantic_profile is None else "validated"
            ),
            "scope_relation": (
                semantic_profile.scope_relation
                if semantic_profile is not None
                else CandidateScopeRelation.UNRESOLVED.value
            ),
            "claimed_scope_kind": (
                semantic_profile.claimed_scope_kind
                if semantic_profile is not None
                else CandidateClaimedScopeKind.UNRESOLVED.value
            ),
            "claimed_scope_identity": claimed_scope_identity,
            "claimed_lane_ids": (
                list(semantic_profile.claimed_lane_ids)
                if semantic_profile is not None
                else []
            ),
            "comparison_lane_ids": (
                list(semantic_profile.comparison_lane_ids)
                if semantic_profile is not None
                else []
            ),
            "evidence_lane_ids": sorted(concrete_lane_ids),
            "missing_claimed_lane_ids": [],
            "unexpected_evidence_lane_ids": [],
        }
        if semantic_profile is None:
            if semantic_profile_error:
                fact_overrides["semantic_profile_error"] = semantic_profile_error
            return _result(
                CausalClaim.SCOPE,
                CausalClaimStatus.INCOMPLETE,
                evidence,
                (
                    "Candidate scope semantics are unavailable or invalid; "
                    "Evidence coverage cannot be promoted into claimed scope."
                ),
                fact_overrides=fact_overrides,
            )

        if (
            semantic_profile.scope_relation
            == CandidateScopeRelation.UNRESOLVED.value
        ):
            return _result(
                CausalClaim.SCOPE,
                CausalClaimStatus.INCOMPLETE,
                evidence,
                (
                    "The Candidate explicitly leaves claimed scope unresolved; "
                    "Evidence coverage is retained only as comparison context."
                ),
                fact_overrides=fact_overrides,
            )

        if (
            semantic_profile.claimed_scope_kind
            != CandidateClaimedScopeKind.LANE.value
        ):
            entity_mapping = {
                "operation": EntityType.OPERATION.value,
                "equipment": EntityType.EQUIPMENT.value,
                "chamber": EntityType.CHAMBER.value,
                "recipe": EntityType.RECIPE.value,
            }
            observed_by_field: dict[str, set[str]] = {
                field_name: {
                    _compact(entity.entity_id)
                    for item in evidence
                    for entity in item.entities
                    if entity.entity_type == entity_type
                }
                for field_name, entity_type in entity_mapping.items()
            }
            required_fields = {
                field_name: str(value)
                for field_name, value in claimed_scope_identity.items()
                if value is not None
            }
            conflicting_fields = {
                field_name: sorted(values)
                for field_name, expected in required_fields.items()
                if (values := observed_by_field[field_name])
                and _compact(expected) not in values
            }
            missing_fields = sorted(
                field_name
                for field_name, expected in required_fields.items()
                if _compact(expected) not in observed_by_field[field_name]
            )
            scoped_operational_lots = {
                entity.entity_id
                for item in evidence
                if item.evidence_type in (_EXPOSURE_TYPES | _PROCESS_TYPES)
                and all(
                    not (actual := {
                        _compact(entity.entity_id)
                        for entity in item.entities
                        if entity.entity_type == entity_mapping[field_name]
                    })
                    or _compact(expected) in actual
                    for field_name, expected in required_fields.items()
                )
                for entity in item.entities
                if entity.entity_type == EntityType.LOT.value
            }
            outcome_lots = {
                entity.entity_id
                for item in evidence
                if item.evidence_type in _OUTCOME_TYPES
                for entity in item.entities
                if entity.entity_type == EntityType.LOT.value
            }
            lane_payloads = [
                _scope_lane_payload(lane) for lane in causal_lanes
            ]
            matching_scope_lanes = [
                lane
                for lane in lane_payloads
                if str(lane.get("lifecycle_status", "")).strip() != "merged"
                and _scope_lane_matches_identity(
                    lane,
                    claimed_scope_identity,
                )
            ]
            scope_lane_contexts = {
                str(lane.get("lane_id", "")): {
                    key: value
                    for key, value in {
                        "lane_id": str(lane.get("lane_id", "")),
                        "operation": str(lane.get("operation", "")),
                        "equipment": str(lane.get("equipment", "")),
                        "chamber": str(lane.get("chamber", "")),
                        "recipe": str(lane.get("recipe", "")),
                        "exposed_lot_ids": [
                            str(item).strip()
                            for item in lane.get("exposed_lot_ids", [])
                            if str(item).strip()
                        ],
                        "parameters": ",".join(
                            str(item).strip()
                            for item in lane.get("parameter_scope", [])
                            if str(item).strip()
                        ),
                        "window_start": (
                            str(lane.get("time_window", [""])[0])
                            if isinstance(lane.get("time_window"), list | tuple)
                            and len(lane.get("time_window", [])) == 2
                            else ""
                        ),
                        "window_end": (
                            str(lane.get("time_window", ["", ""])[1])
                            if isinstance(lane.get("time_window"), list | tuple)
                            and len(lane.get("time_window", [])) == 2
                            else ""
                        ),
                    }.items()
                    if value not in ("", [], ())
                }
                for lane in matching_scope_lanes
                if str(lane.get("lane_id", "")).strip()
            }
            process_covered_lane_ids = sorted(
                lane_id
                for lane_id, lane_context in scope_lane_contexts.items()
                if any(
                    item.evidence_type in _PROCESS_TYPES
                    and _evidence_covers_scope_lane(item, lane_context)
                    for item in evidence
                )
            )
            outcome_covered_lane_ids = sorted(
                lane_id
                for lane_id, lane_context in scope_lane_contexts.items()
                if any(
                    item.evidence_type in _OUTCOME_TYPES
                    and _evidence_covers_scope_lane(item, lane_context)
                    for item in evidence
                )
            )
            effect_lane_ids = sorted(scope_lane_contexts)
            missing_process_lane_ids = sorted(
                set(effect_lane_ids) - set(process_covered_lane_ids)
            )
            missing_outcome_lane_ids = sorted(
                set(effect_lane_ids) - set(outcome_covered_lane_ids)
            )
            shared_effect_coverage_required = (
                semantic_profile.scope_relation
                == CandidateScopeRelation.SHARED_EFFECT.value
            )
            if not shared_effect_coverage_required:
                effect_coverage_status = "not_required"
            elif not causal_lanes or not effect_lane_ids:
                effect_coverage_status = "unavailable"
            elif missing_process_lane_ids or missing_outcome_lane_ids:
                effect_coverage_status = "partial"
            else:
                effect_coverage_status = "complete"
            fact_overrides.update(
                {
                    "scope_identity_status": "grounded",
                    "scope_effect_coverage_status": effect_coverage_status,
                    "scope_effect_lane_ids": effect_lane_ids,
                    "scope_process_covered_lane_ids": process_covered_lane_ids,
                    "scope_outcome_covered_lane_ids": outcome_covered_lane_ids,
                    "scope_missing_process_lane_ids": missing_process_lane_ids,
                    "scope_missing_outcome_lane_ids": missing_outcome_lane_ids,
                    "scope_lane_contexts": scope_lane_contexts,
                    "scope_identity_observed_values": {
                        key: sorted(values)
                        for key, values in observed_by_field.items()
                    },
                    "scope_identity_missing_fields": missing_fields,
                    "scope_identity_conflicting_fields": conflicting_fields,
                    "scoped_operational_lot_ids": sorted(
                        scoped_operational_lots
                    ),
                    "scope_outcome_lot_ids": sorted(outcome_lots),
                    "scope_shared_lot_ids": sorted(
                        scoped_operational_lots & outcome_lots
                    ),
                }
            )
            if conflicting_fields:
                fact_overrides["scope_identity_status"] = "conflicted"
                return _result(
                    CausalClaim.SCOPE,
                    CausalClaimStatus.CONFLICTED,
                    evidence,
                    (
                        "Cited operational Evidence conflicts with the Candidate's "
                        f"declared scope identity: {conflicting_fields}."
                    ),
                    fact_overrides=fact_overrides,
                )
            if missing_fields:
                fact_overrides["scope_identity_status"] = "incomplete"
                return _result(
                    CausalClaim.SCOPE,
                    CausalClaimStatus.INCOMPLETE,
                    evidence,
                    (
                        "Cited typed Evidence does not ground every declared scope "
                        f"identity field: {missing_fields}."
                    ),
                    fact_overrides=fact_overrides,
                )
            if (
                shared_effect_coverage_required
                and causal_lanes
                and effect_coverage_status != "complete"
            ):
                return _result(
                    CausalClaim.SCOPE,
                    CausalClaimStatus.INCOMPLETE,
                    evidence,
                    (
                        "The broad shared-effect identity is grounded, but cited "
                        "typed Evidence does not close process and outcome "
                        "coverage for every matching causal Lane: "
                        f"missing_process={missing_process_lane_ids}, "
                        f"missing_outcome={missing_outcome_lane_ids}."
                    ),
                    fact_overrides=fact_overrides,
                )
            if outcome_lots and not (scoped_operational_lots & outcome_lots):
                return _result(
                    CausalClaim.SCOPE,
                    CausalClaimStatus.CONFLICTED,
                    evidence,
                    (
                        "Operational and outcome Evidence do not converge on a Lot "
                        "inside the declared scope."
                    ),
                    fact_overrides=fact_overrides,
                )
            return _result(
                CausalClaim.SCOPE,
                CausalClaimStatus.SUPPORTED,
                evidence,
                (
                    "Typed Evidence grounds the Candidate's declared "
                    f"{semantic_profile.claimed_scope_kind} scope identity. "
                    "Comparison Evidence does not broaden that identity."
                ),
                fact_overrides=fact_overrides,
            )

        claimed_lane_ids = set(semantic_profile.claimed_lane_ids)
        comparison_lane_ids = set(semantic_profile.comparison_lane_ids)
        unexpected_lane_ids = concrete_lane_ids - comparison_lane_ids
        missing_lane_ids = claimed_lane_ids - concrete_lane_ids
        fact_overrides.update(
            {
                "missing_claimed_lane_ids": sorted(missing_lane_ids),
                "unexpected_evidence_lane_ids": sorted(unexpected_lane_ids),
            }
        )
        if unexpected_lane_ids:
            return _result(
                CausalClaim.SCOPE,
                CausalClaimStatus.CONFLICTED,
                evidence,
                (
                    "Cited supporting Evidence includes causal Lanes outside the "
                    "Candidate's declared claimed/comparison scope: "
                    f"{sorted(unexpected_lane_ids)}."
                ),
                fact_overrides=fact_overrides,
            )
        if missing_lane_ids:
            return _result(
                CausalClaim.SCOPE,
                CausalClaimStatus.INCOMPLETE,
                evidence,
                (
                    "Cited supporting Evidence does not yet cover every declared "
                    f"claimed Lane: {sorted(missing_lane_ids)}."
                ),
                fact_overrides=fact_overrides,
            )
        return _result(
            CausalClaim.SCOPE,
            CausalClaimStatus.SUPPORTED,
            evidence,
            (
                "Typed Evidence covers the Candidate's declared "
                f"{semantic_profile.scope_relation} scope. Evidence from other "
                "declared comparison Lanes remains comparison context and does "
                "not broaden the claim."
            ),
            fact_overrides=fact_overrides,
        )

    # Legacy controlled/old-State compatibility: without an explicit semantic
    # profile, preserve the original Evidence-only scope projection.
    if len(concrete_lane_ids) > 1:
        return _result(
            CausalClaim.SCOPE,
            CausalClaimStatus.CONFLICTED,
            evidence,
            "Cited supporting Evidence mixes multiple concrete causal Lanes: "
            f"{sorted(concrete_lane_ids)}.",
        )
    lot_sets = [
        {
            entity.entity_id
            for entity in item.entities
            if entity.entity_type == EntityType.LOT.value
        }
        for item in evidence
    ]
    populated = [values for values in lot_sets if values]
    if not populated:
        return _result(
            CausalClaim.SCOPE,
            CausalClaimStatus.INCOMPLETE,
            evidence,
            "No Lot scope is attached to the cited Evidence.",
        )
    intersection = set.intersection(*populated)
    if not intersection:
        return _result(
            CausalClaim.SCOPE,
            CausalClaimStatus.CONFLICTED,
            evidence,
            "Cited Evidence lanes have no common Lot scope.",
        )
    return _result(
        CausalClaim.SCOPE,
        CausalClaimStatus.SUPPORTED,
        evidence,
        f"Cited Evidence converges on Lot scope: {sorted(intersection)}.",
    )


def _temporal_claim(evidence: list[Evidence]) -> CausalClaimResult:
    timestamps = {item.timestamp for item in evidence if item.timestamp}
    if not timestamps:
        return _result(
            CausalClaim.TEMPORAL,
            CausalClaimStatus.UNAVAILABLE,
            evidence,
            "Cited Evidence has no timestamps to align.",
        )
    global_processing_windows = [
        window
        for item in evidence
        if item.evidence_type == EvidenceType.EXCURSION_WINDOW.value
        for window in _processing_windows(item)
    ]
    if not global_processing_windows and not any(
        _processing_windows(item) for item in evidence if item.evidence_type in _PROCESS_TYPES
    ):
        return _result(
            CausalClaim.TEMPORAL,
            CausalClaimStatus.INCOMPLETE,
            evidence,
            "Cited Evidence has timestamps but no processing/excursion window to verify.",
        )
    out_of_window: list[str] = []
    aligned_to_window: list[str] = []
    unverifiable: list[str] = []
    for item in evidence:
        if item.evidence_type not in _PROCESS_TYPES:
            continue
        if not item.timestamp:
            unverifiable.append(item.evidence_id)
            continue
        timestamp = str(item.timestamp).replace("Z", "+00:00")
        try:
            observed_at = datetime.fromisoformat(timestamp)
            if observed_at.tzinfo is not None:
                observed_at = observed_at.astimezone(UTC).replace(tzinfo=None)
        except ValueError:
            # Keep legacy/non-ISO timestamps usable; a hard temporal conflict
            # requires an explicitly parseable interval.
            unverifiable.append(item.evidence_id)
            continue
        item_windows = _processing_windows(item)
        if not item_windows and item.evidence_type in _PROCESS_TYPES:
            item_windows = global_processing_windows
        parsed_windows: list[tuple[datetime, datetime]] = []
        for window in item_windows:
            try:
                start = datetime.fromisoformat(
                    str(window["start"]).replace("Z", "+00:00")
                )
                end = datetime.fromisoformat(
                    str(window["end"]).replace("Z", "+00:00")
                )
                if start.tzinfo is not None:
                    start = start.astimezone(UTC).replace(tzinfo=None)
                if end.tzinfo is not None:
                    end = end.astimezone(UTC).replace(tzinfo=None)
            except (TypeError, ValueError):
                continue
            parsed_windows.append((start, end))
        if parsed_windows and not any(
            start <= observed_at <= end for start, end in parsed_windows
        ):
            out_of_window.append(item.evidence_id)
        elif parsed_windows:
            aligned_to_window.append(item.evidence_id)
        else:
            unverifiable.append(item.evidence_id)
    if out_of_window:
        return _result(
            CausalClaim.TEMPORAL,
            CausalClaimStatus.CONFLICTED,
            evidence,
            "Evidence timestamp falls outside its declared processing/excursion window: "
            f"{sorted(set(out_of_window))}.",
        )
    if not aligned_to_window:
        return _result(
            CausalClaim.TEMPORAL,
            CausalClaimStatus.INCOMPLETE,
            evidence,
            "No cited process Evidence timestamp can be verified against a declared window"
            + (f": {sorted(set(unverifiable))}." if unverifiable else "."),
        )
    return _result(
        CausalClaim.TEMPORAL,
        CausalClaimStatus.SUPPORTED,
        evidence,
        "Cited process Evidence timestamps are consistent with declared windows: "
        f"{sorted(set(aligned_to_window))}.",
    )


def _control_claim(candidate: CausalHypothesis, evidence: Iterable[Evidence]) -> CausalClaimResult:
    controls = [item for item in evidence if item.evidence_type in _CONTROL_TYPES]
    if controls:
        return _result(
            CausalClaim.CONTROL,
            CausalClaimStatus.SUPPORTED,
            controls,
            "Normal/negative control Evidence is available as supporting context.",
        )
    return _result(
        CausalClaim.CONTROL,
        CausalClaimStatus.UNAVAILABLE,
        (),
        "No normal or exclusion control Evidence is available; this is not a hard gate.",
    )


validate_causal_candidate = build_causal_evidence_matrix


__all__ = [
    "CausalClaimResult",
    "CausalEvidenceMatrix",
    "build_causal_evidence_matrix",
    "validate_causal_candidate",
]
