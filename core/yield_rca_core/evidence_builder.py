"""Factories and compatibility adapters for typed Evidence."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

from yield_rca_core.evidence_models import (
    EVIDENCE_SCHEMA_VERSION,
    EntityType,
    Evidence,
    EvidenceEntity,
    EvidenceSourceType,
    EvidenceType,
    ModelValidationError,
)
from yield_rca_core.models import ToolInput


def _enum_string(value: EvidenceType | EvidenceSourceType | str) -> str:
    return value.value if isinstance(value, EvidenceType | EvidenceSourceType) else value


class EvidenceBuilder:
    """Build complete typed Evidence from a validated Tool request."""

    @staticmethod
    def scoped_evidence_id(tool_input: ToolInput, evidence_id: str) -> str:
        """Return a stable Evidence identity for a Lane-scoped Tool call.

        Batch 25 can execute the same Tool once per causal Lane.  The payloads
        are intentionally different because ``lane_id`` is part of their
        metadata, so they must not share the legacy global Evidence ID.  Calls
        without a Lane retain their original IDs for controlled/legacy
        compatibility.
        """

        if not isinstance(tool_input, ToolInput):
            raise ModelValidationError("tool_input must be a ToolInput instance")
        lane_id = tool_input.parameters.get("lane_id")
        if not isinstance(lane_id, str) or not lane_id.strip():
            return evidence_id
        lane_digest = sha256(lane_id.strip().encode("utf-8")).hexdigest()[:16].upper()
        suffix = f"_LANE_{lane_digest}"
        return evidence_id if evidence_id.endswith(suffix) else f"{evidence_id}{suffix}"

    @classmethod
    def from_tool(
        cls,
        *,
        tool_input: ToolInput,
        evidence_id: str,
        evidence_type: EvidenceType | str,
        source_type: EvidenceSourceType | str,
        observation: str,
        entities: list[EvidenceEntity],
        confidence: float,
        source_id: str,
        source_table: str | None = None,
        source_field: str | None = None,
        timestamp: str | None = None,
        metadata: dict[str, Any] | None = None,
        summary: str | None = None,
    ) -> Evidence:
        if not isinstance(tool_input, ToolInput):
            raise ModelValidationError("tool_input must be a ToolInput instance")
        resolved_evidence_id = cls.scoped_evidence_id(tool_input, evidence_id)
        resolved_metadata = dict(metadata or {})
        lane_id = tool_input.parameters.get("lane_id")
        if isinstance(lane_id, str) and lane_id.strip():
            resolved_metadata.setdefault("lane_id", lane_id.strip())
            expected_identity = {
                EntityType.OPERATION.value: tool_input.parameters.get(
                    "operation_no",
                    tool_input.parameters.get("operation"),
                ),
                EntityType.EQUIPMENT.value: tool_input.parameters.get(
                    "equipment_id",
                    tool_input.parameters.get("equipment"),
                ),
                EntityType.CHAMBER.value: tool_input.parameters.get(
                    "chamber_id",
                    tool_input.parameters.get("chamber"),
                ),
                EntityType.RECIPE.value: tool_input.parameters.get(
                    "recipe_id",
                    tool_input.parameters.get("recipe"),
                ),
            }
            for entity_type, raw_expected in expected_identity.items():
                expected = str(raw_expected or "").strip().casefold()
                actual = {
                    entity.entity_id.strip().casefold()
                    for entity in entities
                    if entity.entity_type == entity_type
                }
                if expected and actual and actual != {expected}:
                    raise ModelValidationError(
                        "Lane-bound Evidence identity conflicts with Tool scope: "
                        f"lane_id={lane_id!r}, entity_type={entity_type!r}, "
                        f"expected={raw_expected!r}, actual={sorted(actual)!r}"
                    )
        return Evidence(
            evidence_id=resolved_evidence_id,
            source_type=_enum_string(source_type),
            source_id=source_id,
            summary=summary if summary is not None else observation,
            source_table=source_table,
            source_field=source_field,
            timestamp=timestamp,
            metadata=resolved_metadata,
            evidence_type=_enum_string(evidence_type),
            source_agent=tool_input.requested_by,
            source_tool=tool_input.tool_name,
            observation=observation,
            entities=list(entities),
            confidence=confidence,
            evidence_schema_version=EVIDENCE_SCHEMA_VERSION,
        )


class LegacyEvidenceAdapter:
    """Upgrade one legacy Evidence record using explicit industrial semantics."""

    @classmethod
    def to_typed(
        cls,
        evidence: Evidence,
        *,
        evidence_type: EvidenceType | str,
        source_agent: str,
        source_tool: str,
        entities: list[EvidenceEntity],
        confidence: float,
        observation: str | None = None,
    ) -> Evidence:
        if not isinstance(evidence, Evidence):
            raise ModelValidationError("evidence must be an Evidence instance")
        if evidence.is_typed:
            raise ModelValidationError("LegacyEvidenceAdapter requires legacy Evidence")
        return Evidence(
            evidence_id=evidence.evidence_id,
            source_type=evidence.source_type,
            source_id=evidence.source_id,
            summary=evidence.summary,
            source_table=evidence.source_table,
            source_field=evidence.source_field,
            timestamp=evidence.timestamp,
            metadata=dict(evidence.metadata),
            schema_version=evidence.schema_version,
            evidence_type=_enum_string(evidence_type),
            source_agent=source_agent,
            source_tool=source_tool,
            observation=observation if observation is not None else evidence.summary,
            entities=list(entities),
            confidence=confidence,
            evidence_schema_version=EVIDENCE_SCHEMA_VERSION,
        )
