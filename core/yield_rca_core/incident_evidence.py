"""Typed, source-traceable observations reported in an incident request."""

from __future__ import annotations

import hashlib
import re

from yield_rca_core.evidence_models import (
    EVIDENCE_SCHEMA_VERSION,
    EntityType,
    Evidence,
    EvidenceEntity,
    EvidenceSourceType,
    EvidenceType,
)
from yield_rca_core.models import AgentKind, RCAJob

_EXPLICIT_INSPECTION_PATTERNS = (
    re.compile(
        r"\bwith\s+(?P<observation>[a-z0-9][a-z0-9 _/\-]{1,64}?)\s+"
        r"(?:inspection|observation)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:inspection|observation)\s+(?:showed|shows|found|recorded)\s+"
        r"(?P<observation>[a-z0-9][a-z0-9 _/\-]{1,64}?)"
        r"(?=[,.;:]|\s+(?:on|in|for|and)\b|$)",
        re.IGNORECASE,
    ),
)


def _explicit_inspection_spans(user_query: str) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    occupied: set[tuple[int, int]] = set()
    for pattern in _EXPLICIT_INSPECTION_PATTERNS:
        for match in pattern.finditer(user_query):
            start, end = match.span("observation")
            observation = user_query[start:end].strip(" -_/\t\r\n")
            if len(observation) < 2 or (start, end) in occupied:
                continue
            spans.append((observation, start, end))
            occupied.add((start, end))
    return spans


def build_incident_observation_evidence(job: RCAJob) -> list[Evidence]:
    """Project explicit user-reported inspection facts without causal inference."""

    result: list[Evidence] = []
    for observation, start, end in _explicit_inspection_spans(job.user_query):
        digest = hashlib.sha256(
            f"{job.source_lot_id or ''}:{start}:{end}:{observation}".encode()
        ).hexdigest()[:16].upper()
        entities = [
            *(
                [EvidenceEntity(EntityType.LOT.value, job.source_lot_id)]
                if job.source_lot_id is not None
                else []
            ),
            EvidenceEntity(
                EntityType.DEFECT.value,
                observation,
                attributes={"reported_as": "physical_inspection_observation"},
            ),
        ]
        result.append(
            Evidence(
                evidence_id=f"EV_INCIDENT_OBSERVATION_{digest}",
                source_type=EvidenceSourceType.USER.value,
                source_id=f"user_query:{job.job_id}",
                summary=f"Incident report explicitly records {observation}.",
                source_table="rca_job",
                source_field="user_query",
                metadata={
                    "observation_role": "physical_inspection_observation",
                    "observation_role_provenance": {
                        "source_type": "user_query",
                        "source_field": "user_query",
                        "source_quote": observation,
                        "quote_start": start,
                        "quote_end": end,
                        "extractor": "explicit_inspection_span_v1",
                    },
                    "causal_status": "observed_not_confirmed_cause",
                },
                evidence_type=EvidenceType.DEFECT_SIGNAL.value,
                source_agent=AgentKind.SUPERVISOR.value,
                source_tool="incident_observation_extractor",
                observation=(
                    f"User-reported inspection observation: {observation}. "
                    "This records occurrence only and does not assert causation."
                ),
                entities=entities,
                confidence=1.0,
                evidence_schema_version=EVIDENCE_SCHEMA_VERSION,
            )
        )
    return result


__all__ = ["build_incident_observation_evidence"]
