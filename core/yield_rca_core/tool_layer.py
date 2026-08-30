"""Structured Tool Layer for the Yield RCA MVP.

Tools are the only execution surface intended for Agents. They translate
repository data into structured outputs and traceable Evidence objects.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import date, datetime
from functools import wraps
from itertools import pairwise
from statistics import mean, stdev
from time import perf_counter
from typing import Any, cast

from yield_rca_core.evidence_builder import EvidenceBuilder
from yield_rca_core.evidence_models import EntityType, EvidenceEntity, EvidenceType
from yield_rca_core.knowledge_retrieval import (
    KeywordRetriever,
    KnowledgeAssetRepository,
    RetrievalQuery,
    Retriever,
)
from yield_rca_core.models import (
    AgentKind,
    Evidence,
    EvidenceSourceType,
    LotDrivenRCAError,
    LotNotFoundError,
    ModelValidationError,
    ToolInput,
    ToolOutput,
    Warning,
)
from yield_rca_core.repositories import FabRepository, Row, filter_rows
from yield_rca_core.spc_engine import calculate_imr, calculate_p_chart, calculate_xbar
from yield_rca_core.spc_models import SpcChartType, SpcSample

ToolLatencyRecord = dict[str, str | float]
ToolRun = Callable[[Any, ToolInput], ToolOutput]
_TOOL_LATENCY_SINK: ContextVar[list[ToolLatencyRecord] | None] = ContextVar(
    "yield_rca_tool_latency_sink",
    default=None,
)


@contextmanager
def capture_tool_latencies() -> Iterator[list[ToolLatencyRecord]]:
    """Collect exact Tool execution latency without changing Tool DTOs."""

    records: list[ToolLatencyRecord] = []
    parent = _TOOL_LATENCY_SINK.get()
    token = _TOOL_LATENCY_SINK.set(records)
    try:
        yield records
    finally:
        _TOOL_LATENCY_SINK.reset(token)
        if parent is not None:
            parent.extend(records)


def _measure_tool_latency(function: ToolRun) -> ToolRun:
    @wraps(function)
    def measured(self: Any, tool_input: ToolInput) -> ToolOutput:
        started = perf_counter()
        outcome = "failed"
        try:
            output = function(self, tool_input)
            outcome = "success" if output.success else "failed"
            return output
        finally:
            sink = _TOOL_LATENCY_SINK.get()
            if sink is not None:
                sink.append(
                    {
                        "tool_name": str(tool_input.tool_name),
                        "tool_request_id": str(tool_input.request_id),
                        "agent": str(tool_input.requested_by),
                        "outcome": outcome,
                        "duration_ms": round((perf_counter() - started) * 1000.0, 3),
                    }
                )

    return cast(ToolRun, measured)


def _date_in_window(timestamp: str, start_date: str | None, end_date: str | None) -> bool:
    if not timestamp:
        return True
    timestamp_date = date.fromisoformat(timestamp[:10])
    if start_date and timestamp_date < date.fromisoformat(start_date):
        return False
    if end_date and timestamp_date > date.fromisoformat(end_date):
        return False
    return True


def _float(value: str) -> float:
    return float(value) if value else 0.0


def _evidence_id(prefix: str, value: str) -> str:
    normalized = re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")
    return f"{prefix}_{normalized or 'UNKNOWN'}"


_MECHANISM_INTERMEDIATE_ROLES = {
    "mechanism_intermediate",
    "physical_intermediate",
}


def _explicit_mechanism_intermediate_metadata(
    rows: list[Row],
    *,
    source_table: str,
) -> dict[str, Any]:
    """Propagate only source-declared physical intermediates.

    A defect code, metric name, pattern, or model-authored explanation is never
    interpreted here.  Aggregated Evidence receives a causal role only when
    every contributing source row explicitly carries the same accepted role.
    """

    if not rows:
        return {}
    roles: list[str] = []
    source_fields: set[str] = set()
    for row in rows:
        role = ""
        for field_name in ("causal_role", "mechanism_role"):
            raw_value = str(row.get(field_name, "")).strip().casefold()
            if raw_value:
                role = raw_value
                source_fields.add(field_name)
                break
        if not role and str(row.get("mechanism_intermediate", "")).strip().casefold() in {
            "1",
            "true",
            "yes",
        }:
            role = "mechanism_intermediate"
            source_fields.add("mechanism_intermediate")
        if role not in _MECHANISM_INTERMEDIATE_ROLES:
            return {}
        roles.append(role)
    if len(set(roles)) != 1:
        return {}
    return {
        "causal_role": roles[0],
        "causal_role_provenance": {
            "source_table": source_table,
            "source_fields": sorted(source_fields),
            "source_row_count": len(rows),
        },
    }


def _shared_exposure_lanes(
    process_rows: list[Row],
    lot_ids: list[str],
    *,
    target_operation_no: str | None = None,
    fdc_rows: list[Row] | None = None,
    evidence_scope: str = "scope",
) -> list[dict[str, Any]]:
    """Discover concrete operation/resource lanes without selecting a cause.

    A lane is a factual process assignment, not a root-cause conclusion.  The
    helper intentionally returns every observed assignment so the caller can
    keep a bounded active set while retaining an auditable overflow inventory.
    """

    selected_lots = {str(item).strip() for item in lot_ids if str(item).strip()}
    if not selected_lots:
        return []
    grouped: dict[tuple[str, str, str, str], list[Row]] = defaultdict(list)
    for row in process_rows:
        key = (
            str(row.get("operation_no", "")),
            str(row.get("equipment_id", "")),
            str(row.get("chamber_id", "")),
            str(row.get("recipe_id", "")),
        )
        if all(key) and row.get("lot_id") in selected_lots:
            grouped[key].append(row)
    signal_rows = fdc_rows or []
    lanes: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        operation_no, equipment_id, chamber_id, recipe_id = key
        lane_lots = {str(row["lot_id"]) for row in rows}
        lane_wafers = {str(row["wafer_id"]) for row in rows}
        matching_signals = [
            row
            for row in signal_rows
            if row.get("operation_no") == operation_no
            and row.get("equipment_id") == equipment_id
            and row.get("chamber_id") == chamber_id
        ]
        abnormal_signal_count = sum(
            row.get("severity") != "NORMAL" or row.get("ooc_flag") == "true"
            for row in matching_signals
        )
        coverage = len(lane_lots) / max(1, len(selected_lots))
        target_bonus = 0.2 if target_operation_no and operation_no == target_operation_no else 0.0
        signal_bonus = min(0.1, abnormal_signal_count / max(1, len(matching_signals)) * 0.1)
        priority_score = round(min(1.0, 0.65 * coverage + target_bonus + signal_bonus), 4)
        lane_id = ":".join(("lane", operation_no, equipment_id, chamber_id, recipe_id))
        timestamp_rows = [
            row
            for row in rows
            if row.get("started_at") and row.get("ended_at")
        ]
        parameter_scope = sorted(
            {
                str(row.get("parameter_name", ""))
                for row in matching_signals
                if str(row.get("parameter_name", "")).strip()
            }
        )
        lanes.append(
            {
                "lane_id": lane_id,
                "operation": operation_no,
                "operation_name": str(rows[0].get("operation_name", "")),
                "module": str(rows[0].get("module", "")),
                "equipment": equipment_id,
                "chamber": chamber_id,
                "recipe": recipe_id,
                "parameter_scope": parameter_scope,
                "exposed_lot_ids": sorted(lane_lots),
                "exposed_wafer_ids": sorted(lane_wafers),
                "time_window": (
                    [
                        min(str(row["started_at"]) for row in timestamp_rows),
                        max(str(row["ended_at"]) for row in timestamp_rows),
                    ]
                    if timestamp_rows
                    else []
                ),
                "priority_score": priority_score,
                "abnormal_signal_count": abnormal_signal_count,
                "coverage": round(coverage, 4),
                "evidence_scope": evidence_scope,
            }
        )
    lanes.sort(
        key=lambda item: (
            -float(item["priority_score"]),
            -float(item["coverage"]),
            -len(item["exposed_lot_ids"]),
            str(item["operation"]),
            str(item["equipment"]),
            str(item["chamber"]),
        )
    )
    return lanes


def _evidence_payload(evidence: list[Evidence]) -> tuple[list[str], list[dict[str, Any]]]:
    return [item.evidence_id for item in evidence], [item.to_dict() for item in evidence]


def _tool_output(
    tool_input: ToolInput,
    data: dict[str, Any],
    evidence: list[Evidence],
    warnings: list[Warning] | None = None,
) -> ToolOutput:
    evidence_ids, evidence_dicts = _evidence_payload(evidence)
    return ToolOutput(
        tool_name=tool_input.tool_name,
        request_id=tool_input.request_id,
        success=True,
        data={**data, "evidence": evidence_dicts},
        evidence_ids=evidence_ids,
        evidence=list(evidence),
        warnings=list(warnings or []),
    )


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _overlaps(started_at: str, ended_at: str, window_start: str, window_end: str) -> bool:
    return _timestamp(started_at) <= _timestamp(window_end) and _timestamp(ended_at) >= _timestamp(
        window_start
    )


@dataclass(frozen=True)
class BaseTool:
    repository: FabRepository
    tool_name: str
    owner_agent: str

    def _validate_tool_name(self, tool_input: ToolInput) -> None:
        if tool_input.tool_name != self.tool_name:
            raise ValueError(f"expected tool_name={self.tool_name}, got {tool_input.tool_name}")
        if tool_input.requested_by != self.owner_agent:
            raise ModelValidationError(
                f"tool {self.tool_name} belongs to agent {self.owner_agent}; "
                f"requested_by was {tool_input.requested_by}"
            )


class FindAffectedLotsTool(BaseTool):
    """Find lots with WAT yield loss symptoms for a product and time window."""

    def __init__(self, repository: FabRepository) -> None:
        super().__init__(
            repository=repository,
            tool_name="find_affected_lots",
            owner_agent=AgentKind.MES.value,
        )

    @_measure_tool_latency
    def run(self, tool_input: ToolInput) -> ToolOutput:
        self._validate_tool_name(tool_input)
        product_id = str(tool_input.parameters["product_id"])
        start_date = tool_input.parameters.get("start_date")
        end_date = tool_input.parameters.get("end_date")

        lots = [
            row
            for row in self.repository.rows("lot_master")
            if row["product_id"] == product_id
            and _date_in_window(row["started_at"], start_date, end_date)
        ]
        lot_ids = {row["lot_id"] for row in lots}
        wat_rows = [row for row in self.repository.rows("wat_result") if row["lot_id"] in lot_ids]
        wat_rows_by_lot: dict[str, list[Row]] = defaultdict(list)
        for row in wat_rows:
            wat_rows_by_lot[row["lot_id"]].append(row)
        affected_lot_set = {
            lot_id
            for lot_id, rows in wat_rows_by_lot.items()
            if any(row["pass_fail"] == "false" for row in rows)
        }
        passing_lot_set = {
            lot_id
            for lot_id, rows in wat_rows_by_lot.items()
            if rows and all(row["pass_fail"] == "true" for row in rows)
        }
        untested_lot_set = lot_ids - affected_lot_set - passing_lot_set
        affected_lots = sorted(affected_lot_set)
        suspect_lots = sorted(
            {
                row["lot_id"]
                for row in self.repository.rows("fdc_feature")
                if row["lot_id"] in lot_ids and row["severity"] != "NORMAL"
            }
        )
        suspect_lot_set = set(suspect_lots)
        normal_lots = sorted(passing_lot_set - suspect_lot_set)
        passing_suspect_lots = sorted(passing_lot_set & suspect_lot_set)
        untested_lots = sorted(untested_lot_set)
        fail_modes = Counter(row["fail_mode"] for row in wat_rows if row["fail_mode"])
        wat_by_date: dict[str, list[Row]] = defaultdict(list)
        for row in wat_rows:
            wat_by_date[row["tested_at"][:10]].append(row)
        yield_trend = []
        for test_date, date_rows in sorted(wat_by_date.items()):
            pass_by_lot: dict[str, bool] = defaultdict(lambda: True)
            for row in date_rows:
                pass_by_lot[row["lot_id"]] = (
                    pass_by_lot[row["lot_id"]] and row["pass_fail"] == "true"
                )
            pass_count = sum(pass_by_lot.values())
            lot_count = len(pass_by_lot)
            yield_trend.append(
                {
                    "date": test_date,
                    "lot_count": lot_count,
                    "pass_count": pass_count,
                    "fail_count": lot_count - pass_count,
                    "pass_rate": round(100 * pass_count / lot_count, 1),
                }
            )

        affected_summary = (
            f"No WAT pass/fail records are available for {product_id} in the selected window."
            if not wat_rows
            else (
                f"Detected {len(affected_lots)} affected lots and {len(normal_lots)} "
                f"normal lots for {product_id} from WAT pass/fail data; "
                f"{len(untested_lots)} lots have no conclusive WAT result."
            )
        )
        lot_classifications = {
            **{lot_id: "normal" for lot_id in normal_lots},
            **{lot_id: "suspect_passing" for lot_id in passing_suspect_lots},
            **{lot_id: "affected" for lot_id in affected_lots},
            **{lot_id: "untested" for lot_id in untested_lots},
        }
        evidence = [
            EvidenceBuilder.from_tool(
                tool_input=tool_input,
                evidence_id="EV_ANALYTICS_AFFECTED_LOTS",
                evidence_type=(
                    EvidenceType.DATA_MISSING
                    if not wat_rows
                    else (
                        EvidenceType.IMPACT_SCOPE if affected_lots else EvidenceType.NEGATIVE_SIGNAL
                    )
                ),
                source_type=EvidenceSourceType.ANALYTICS,
                observation=affected_summary,
                entities=[
                    EvidenceEntity(
                        entity_type=EntityType.PRODUCT.value,
                        entity_id=product_id,
                    ),
                    *[
                        EvidenceEntity(
                            entity_type=EntityType.LOT.value,
                            entity_id=lot_id,
                            attributes={"classification": classification},
                        )
                        for lot_id, classification in sorted(lot_classifications.items())
                    ],
                ],
                confidence=1.0,
                source_id=f"{product_id}:{start_date or '*'}:{end_date or '*'}",
                source_table="wat_result",
                metadata={
                    "affected_lots": affected_lots,
                    "normal_lots": normal_lots,
                    "suspect_lots": suspect_lots,
                    "passing_suspect_lots": passing_suspect_lots,
                    "untested_lots": untested_lots,
                    "fail_modes": dict(fail_modes),
                    "yield_trend": yield_trend,
                },
            )
        ]
        if wat_rows and untested_lots:
            evidence.append(
                EvidenceBuilder.from_tool(
                    tool_input=tool_input,
                    evidence_id="EV_ANALYTICS_WAT_DATA_MISSING",
                    evidence_type=EvidenceType.DATA_MISSING,
                    source_type=EvidenceSourceType.WAT,
                    observation=(
                        f"{len(untested_lots)} selected Lots have no conclusive WAT "
                        "pass/fail records and are not classified as normal."
                    ),
                    entities=[
                        EvidenceEntity(
                            entity_type=EntityType.PRODUCT.value,
                            entity_id=product_id,
                        ),
                        *[
                            EvidenceEntity(
                                entity_type=EntityType.LOT.value,
                                entity_id=lot_id,
                                attributes={"classification": "untested"},
                            )
                            for lot_id in untested_lots
                        ],
                    ],
                    confidence=1.0,
                    source_id=f"wat_result:{product_id}:missing",
                    source_table="wat_result",
                    source_field="pass_fail",
                    metadata={"untested_lots": untested_lots},
                )
            )
        return _tool_output(
            tool_input,
            {
                "product_id": product_id,
                "affected_lots": affected_lots,
                "normal_lots": normal_lots,
                "suspect_lots": suspect_lots,
                "passing_suspect_lots": passing_suspect_lots,
                "untested_lots": untested_lots,
                "affected_count": len(affected_lots),
                "normal_count": len(normal_lots),
                "suspect_count": len(suspect_lots),
                "untested_count": len(untested_lots),
                "fail_modes": dict(fail_modes),
                "yield_trend": yield_trend,
            },
            evidence,
        )


class GetLotContextTool(BaseTool):
    """Resolve the manufacturing and anomaly context for one requested Lot."""

    def __init__(self, repository: FabRepository) -> None:
        super().__init__(
            repository=repository,
            tool_name="get_lot_context",
            owner_agent=AgentKind.MES.value,
        )

    @_measure_tool_latency
    def run(self, tool_input: ToolInput) -> ToolOutput:
        self._validate_tool_name(tool_input)
        lot_id = str(tool_input.parameters["lot_id"]).strip().upper()
        lot_rows = filter_rows(self.repository.rows("lot_master"), lot_id=lot_id)
        if not lot_rows:
            raise LotNotFoundError(f"Lot not found: {lot_id}")

        lot = lot_rows[0]
        process_rows = sorted(
            filter_rows(self.repository.rows("process_history"), lot_id=lot_id),
            key=lambda row: row["started_at"],
        )
        wat_rows = filter_rows(self.repository.rows("wat_result"), lot_id=lot_id)
        defect_rows = filter_rows(self.repository.rows("defect_summary"), lot_id=lot_id)
        hold_rows = filter_rows(self.repository.rows("hold_history"), lot_id=lot_id)
        fdc_rows = filter_rows(self.repository.rows("fdc_feature"), lot_id=lot_id)
        metrology_rows = filter_rows(self.repository.rows("metrology_result"), lot_id=lot_id)
        wat_failed = any(row["pass_fail"] == "false" for row in wat_rows)
        fail_modes = sorted({row["fail_mode"] for row in wat_rows if row["fail_mode"]})
        ooc_features = [row for row in fdc_rows if row["ooc_flag"] == "true"]
        process_operations = {row["operation_no"] for row in process_rows}
        process_wafers = {row["wafer_id"] for row in process_rows}

        wat_observation = (
            f"No WAT records are available for Lot {lot_id}."
            if not wat_rows
            else (
                f"Lot {lot_id} {'fails' if wat_failed else 'does not fail'} WAT; "
                f"fail modes: {', '.join(fail_modes) if fail_modes else 'none'}."
            )
        )
        evidence = [
            EvidenceBuilder.from_tool(
                tool_input=tool_input,
                evidence_id="EV_MES_SOURCE_LOT_CONTEXT",
                evidence_type=EvidenceType.LOT_CONTEXT,
                source_type=EvidenceSourceType.MES,
                observation=(
                    f"Lot {lot_id} is product {lot['product_id']} on route {lot['route_id']} "
                    f"with {len(process_rows)} Wafer-operation records across "
                    f"{len(process_operations)} operations and {len(process_wafers)} Wafers."
                ),
                entities=[
                    EvidenceEntity(
                        entity_type=EntityType.LOT.value,
                        entity_id=lot_id,
                        attributes={
                            "product_id": lot["product_id"],
                            "route_id": lot["route_id"],
                        },
                    ),
                    EvidenceEntity(
                        entity_type=EntityType.PRODUCT.value,
                        entity_id=lot["product_id"],
                    ),
                    EvidenceEntity(
                        entity_type=EntityType.ROUTE.value,
                        entity_id=lot["route_id"],
                    ),
                ],
                confidence=1.0,
                source_id=lot_id,
                source_table="lot_master",
                timestamp=lot.get("finished_at") or lot.get("started_at"),
                metadata={
                    "lot_id": lot_id,
                    "product_id": lot["product_id"],
                    "route_id": lot["route_id"],
                    "process_record_count": len(process_rows),
                    "operation_count": len(process_operations),
                    "wafer_count": len(process_wafers),
                },
            ),
            EvidenceBuilder.from_tool(
                tool_input=tool_input,
                evidence_id="EV_WAT_SOURCE_LOT_ANOMALY",
                evidence_type=(
                    EvidenceType.DATA_MISSING
                    if not wat_rows
                    else (
                        EvidenceType.ELECTRICAL_FAILURE
                        if wat_failed
                        else EvidenceType.NEGATIVE_SIGNAL
                    )
                ),
                source_type=EvidenceSourceType.WAT,
                observation=wat_observation,
                entities=[
                    EvidenceEntity(
                        entity_type=EntityType.LOT.value,
                        entity_id=lot_id,
                    ),
                    *[
                        EvidenceEntity(
                            entity_type=EntityType.WAT_ITEM.value,
                            entity_id=fail_mode,
                        )
                        for fail_mode in fail_modes
                    ],
                ],
                confidence=1.0,
                source_id=f"wat_result:{lot_id}",
                source_table="wat_result",
                source_field="pass_fail",
                timestamp=max((row["tested_at"] for row in wat_rows), default=None),
                metadata={
                    "lot_id": lot_id,
                    "wat_failed": wat_failed,
                    "fail_modes": fail_modes,
                    "wat_record_count": len(wat_rows),
                },
            ),
        ]
        failed_metrology = [row for row in metrology_rows if row["pass_fail"] == "false"]
        normal_cmp_features = [
            row
            for row in fdc_rows
            if row["operation_no"] in {"1500", "5100", "5300", "6100", "6400"}
            and row["severity"] == "NORMAL"
            and row["ooc_flag"] == "false"
        ]
        if failed_metrology and normal_cmp_features:
            evidence.append(
                EvidenceBuilder.from_tool(
                    tool_input=tool_input,
                    evidence_id="EV_FDC_CMP_NORMAL_EXCLUSION",
                    evidence_type=EvidenceType.NEGATIVE_SIGNAL,
                    source_type=EvidenceSourceType.FDC,
                    observation=(
                        f"Lot {lot_id} has out-of-spec metrology while its recorded CMP FDC "
                        "features remain normal, reducing support for a CMP-origin hypothesis."
                    ),
                    entities=[
                        EvidenceEntity(
                            entity_type=EntityType.LOT.value,
                            entity_id=lot_id,
                        ),
                        *[
                            EvidenceEntity(
                                entity_type=EntityType.CHAMBER.value,
                                entity_id=chamber_id,
                            )
                            for chamber_id in sorted(
                                {row["chamber_id"] for row in normal_cmp_features}
                            )
                        ],
                        *[
                            EvidenceEntity(
                                entity_type=EntityType.PARAMETER.value,
                                entity_id=parameter_name,
                            )
                            for parameter_name in sorted(
                                {row["parameter_name"] for row in normal_cmp_features}
                            )
                        ],
                    ],
                    confidence=0.95,
                    source_id=f"fdc_feature:{lot_id}:cmp_normal",
                    source_table="fdc_feature",
                    timestamp=max(row["measured_at"] for row in normal_cmp_features),
                    metadata={
                        "lot_id": lot_id,
                        "normal_cmp_feature_count": len(normal_cmp_features),
                        "failed_metrology_count": len(failed_metrology),
                    },
                )
            )
        all_process_rows = self.repository.rows("process_history")
        recipe_changes: list[dict[str, Any]] = []
        source_rows_by_operation: dict[str, list[Row]] = defaultdict(list)
        for row in process_rows:
            source_rows_by_operation[row["operation_no"]].append(row)
        for operation_no, operation_rows in source_rows_by_operation.items():
            source_recipes = Counter(
                (row["recipe_id"], row["recipe_version"]) for row in operation_rows
            )
            source_recipe, source_count = source_recipes.most_common(1)[0]
            source_started_at = min(row["started_at"] for row in operation_rows)
            previous_rows = [
                row
                for row in all_process_rows
                if row["lot_id"] != lot_id
                and row["route_id"] == lot["route_id"]
                and row["operation_no"] == operation_no
                and row["ended_at"] < source_started_at
            ]
            if not previous_rows:
                continue
            baseline_recipe, baseline_count = Counter(
                (row["recipe_id"], row["recipe_version"]) for row in previous_rows
            ).most_common(1)[0]
            if source_recipe == baseline_recipe:
                continue
            recipe_changes.append(
                {
                    "operation_no": operation_no,
                    "baseline_recipe_id": baseline_recipe[0],
                    "baseline_recipe_version": baseline_recipe[1],
                    "source_recipe_id": source_recipe[0],
                    "source_recipe_version": source_recipe[1],
                    "source_wafer_count": source_count,
                    "baseline_record_count": baseline_count,
                }
            )
        warnings: list[Warning] = []
        recipe_history_missing = False
        if recipe_changes:
            recipe_history_rows = [
                row for row in self.repository.rows("recipe_history") if row["lot_id"] == lot_id
            ]
            recipe_history_missing = not recipe_history_rows
            recipe_change_timestamp = max(
                (row["executed_at"] for row in recipe_history_rows),
                default=max((row["ended_at"] for row in process_rows), default=None),
            )
            evidence.append(
                EvidenceBuilder.from_tool(
                    tool_input=tool_input,
                    evidence_id="EV_MES_RECIPE_CHANGE",
                    evidence_type=EvidenceType.RECIPE_CHANGE,
                    source_type=EvidenceSourceType.MES,
                    observation=(
                        f"Lot {lot_id} has {len(recipe_changes)} operation recipe version "
                        "changes relative to prior Lots on the same route."
                    ),
                    entities=[
                        EvidenceEntity(
                            entity_type=EntityType.LOT.value,
                            entity_id=lot_id,
                        ),
                        *[
                            EvidenceEntity(
                                entity_type=EntityType.OPERATION.value,
                                entity_id=str(item["operation_no"]),
                                attributes={
                                    "baseline_recipe_id": item["baseline_recipe_id"],
                                    "baseline_recipe_version": item["baseline_recipe_version"],
                                },
                            )
                            for item in recipe_changes
                        ],
                        *[
                            EvidenceEntity(
                                entity_type=EntityType.RECIPE.value,
                                entity_id=(
                                    f"{item['source_recipe_id']}:{item['source_recipe_version']}"
                                ),
                            )
                            for item in recipe_changes
                        ],
                    ],
                    confidence=0.99 if recipe_history_rows else 0.9,
                    source_id=(
                        f"recipe_history:{lot_id}"
                        if recipe_history_rows
                        else f"process_history:{lot_id}:recipe_change"
                    ),
                    source_table="recipe_history" if recipe_history_rows else "process_history",
                    source_field="recipe_version",
                    timestamp=recipe_change_timestamp,
                    metadata={"lot_id": lot_id, "recipe_changes": recipe_changes},
                )
            )
            if recipe_history_missing:
                evidence.append(
                    EvidenceBuilder.from_tool(
                        tool_input=tool_input,
                        evidence_id="EV_MES_RECIPE_HISTORY_MISSING",
                        evidence_type=EvidenceType.DATA_MISSING,
                        source_type=EvidenceSourceType.MES,
                        observation=(
                            f"Process history indicates recipe changes for Lot {lot_id}, "
                            "but no corresponding recipe_history records are available."
                        ),
                        entities=[
                            EvidenceEntity(
                                entity_type=EntityType.LOT.value,
                                entity_id=lot_id,
                            ),
                            *[
                                EvidenceEntity(
                                    entity_type=EntityType.RECIPE.value,
                                    entity_id=(
                                        f"{item['source_recipe_id']}:"
                                        f"{item['source_recipe_version']}"
                                    ),
                                )
                                for item in recipe_changes
                            ],
                        ],
                        confidence=1.0,
                        source_id=f"recipe_history:{lot_id}:missing",
                        source_table="recipe_history",
                        timestamp=recipe_change_timestamp,
                        metadata={"lot_id": lot_id, "recipe_changes": recipe_changes},
                    )
                )
                warnings.append(
                    Warning(
                        warning_id="WARN_RECIPE_HISTORY_MISSING",
                        message=(
                            f"Recipe history is unavailable for Lot {lot_id}; "
                            "recipe-change provenance falls back to process history."
                        ),
                        evidence_ids=["EV_MES_RECIPE_HISTORY_MISSING"],
                    )
                )
        if not wat_failed:
            warnings.append(
                Warning(
                    warning_id="WARN_SOURCE_LOT_NOT_WAT_FAILED",
                    message=f"Requested Lot {lot_id} is not marked as failed in WAT data.",
                    evidence_ids=["EV_WAT_SOURCE_LOT_ANOMALY"],
                )
            )

        return _tool_output(
            tool_input,
            {
                "lot_id": lot_id,
                "lot": lot,
                "product_id": lot["product_id"],
                "route_id": lot["route_id"],
                "process_history": process_rows,
                "wat_results": wat_rows,
                "wat_failed": wat_failed,
                "fail_modes": fail_modes,
                "defect_summaries": defect_rows,
                "hold_records": hold_rows,
                "fdc_features": fdc_rows,
                "ooc_features": ooc_features,
                "metrology_results": metrology_rows,
                "recipe_changes": recipe_changes,
                "recipe_history_missing": recipe_history_missing,
            },
            evidence,
            warnings,
        )


class FindImpactLotsTool(BaseTool):
    """Find Lots exposed to the requested Lot's chamber excursion window."""

    def __init__(self, repository: FabRepository) -> None:
        super().__init__(
            repository=repository,
            tool_name="find_impact_lots",
            owner_agent=AgentKind.MES.value,
        )

    @_measure_tool_latency
    def run(self, tool_input: ToolInput) -> ToolOutput:
        self._validate_tool_name(tool_input)
        lot_id = str(tool_input.parameters["lot_id"]).strip().upper()
        source_process_rows = filter_rows(
            self.repository.rows("process_history"),
            lot_id=lot_id,
        )
        if not source_process_rows:
            raise LotNotFoundError(f"Lot process history not found: {lot_id}")

        requested_operation = tool_input.parameters.get("target_operation_no")
        source_fdc_features = filter_rows(self.repository.rows("fdc_feature"), lot_id=lot_id)
        source_ooc_features = [row for row in source_fdc_features if row["ooc_flag"] == "true"]
        source_abnormal_features = [
            row for row in source_fdc_features if row["severity"] != "NORMAL"
        ]
        if requested_operation:
            target_operation_no = str(requested_operation)
        elif source_ooc_features:
            target_operation_no = Counter(
                row["operation_no"] for row in source_ooc_features
            ).most_common(1)[0][0]
        elif source_abnormal_features:
            target_operation_no = Counter(
                row["operation_no"] for row in source_abnormal_features
            ).most_common(1)[0][0]
        elif source_fdc_features:
            target_operation_no = max(source_fdc_features, key=lambda row: row["measured_at"])[
                "operation_no"
            ]
        else:
            target_operation_no = source_process_rows[-1]["operation_no"]

        target_rows = [
            row for row in source_process_rows if row["operation_no"] == target_operation_no
        ]
        if not target_rows:
            raise LotDrivenRCAError(
                f"Lot {lot_id} has no process history at operation {target_operation_no}"
            )

        source_signal = None
        if source_ooc_features:
            source_signal = max(
                source_ooc_features, key=lambda row: abs(_float(row["delta_percent"]))
            )
        elif source_abnormal_features:
            source_signal = max(
                source_abnormal_features,
                key=lambda row: abs(_float(row["delta_percent"])),
            )
        else:
            defect_rows = filter_rows(self.repository.rows("defect_summary"), lot_id=lot_id)
            if defect_rows:
                defect_wafer = max(defect_rows, key=lambda row: int(row["defect_count"]))[
                    "wafer_id"
                ]
                source_signal = next(
                    (
                        row
                        for row in source_fdc_features
                        if row["operation_no"] == target_operation_no
                        and row["wafer_id"] == defect_wafer
                    ),
                    None,
                )
        scoped_target_rows = [
            row
            for row in target_rows
            if (
                not tool_input.parameters.get("equipment_id")
                or row["equipment_id"] == str(tool_input.parameters["equipment_id"])
            )
            and (
                not tool_input.parameters.get("chamber_id")
                or row["chamber_id"] == str(tool_input.parameters["chamber_id"])
            )
            and (
                not tool_input.parameters.get("recipe_id")
                or row["recipe_id"] == str(tool_input.parameters["recipe_id"])
            )
        ]
        if not scoped_target_rows:
            raise ModelValidationError(
                "requested causal Lane has no process exposure at the selected operation"
            )
        source_exposure = next(
            (
                row
                for row in scoped_target_rows
                if source_signal
                and row["wafer_id"] == source_signal["wafer_id"]
                and row["chamber_id"] == source_signal["chamber_id"]
            ),
            scoped_target_rows[0],
        )
        matching_ooc_events = filter_rows(
            self.repository.rows("ooc_event"),
            equipment_id=source_exposure["equipment_id"],
            chamber_id=source_exposure["chamber_id"],
            operation_no=target_operation_no,
        )
        source_lot: Row = next(
            (
                row
                for row in self.repository.rows("lot_master")
                if row["lot_id"] == lot_id
            ),
            {},
        )
        source_analysis_cutoff = str(
            source_lot.get("finished_at") or max(
                row["ended_at"] for row in source_process_rows
            )
        )
        analysis_cutoff = max(
            source_analysis_cutoff,
            *(row["triggered_at"] for row in matching_ooc_events),
        )
        warnings: list[Warning] = []
        chamber_abnormal_features = [
            row
            for row in self.repository.rows("fdc_feature")
            if row["operation_no"] == target_operation_no
            and row["equipment_id"] == source_exposure["equipment_id"]
            and row["chamber_id"] == source_exposure["chamber_id"]
            and row["severity"] != "NORMAL"
            and row["measured_at"] <= analysis_cutoff
        ]
        if matching_ooc_events and chamber_abnormal_features:
            abnormal_wafer_ids = {row["wafer_id"] for row in chamber_abnormal_features}
            abnormal_process_rows = [
                row
                for row in self.repository.rows("process_history")
                if row["operation_no"] == target_operation_no
                and row["wafer_id"] in abnormal_wafer_ids
                and row["chamber_id"] == source_exposure["chamber_id"]
                and row["started_at"] <= analysis_cutoff
            ]
            excursion_start = min(row["started_at"] for row in abnormal_process_rows)
            excursion_end = max(row["ended_at"] for row in abnormal_process_rows)
        elif chamber_abnormal_features and source_abnormal_features:
            source_abnormal_wafer_ids = {
                row["wafer_id"] for row in chamber_abnormal_features if row["lot_id"] == lot_id
            }
            source_abnormal_rows = [
                row
                for row in target_rows
                if row["wafer_id"] in source_abnormal_wafer_ids
                and row["chamber_id"] == source_exposure["chamber_id"]
            ]
            excursion_start = min(row["started_at"] for row in source_abnormal_rows)
            excursion_end = max(row["ended_at"] for row in source_abnormal_rows)
        else:
            excursion_start = source_exposure["started_at"]
            excursion_end = source_exposure["ended_at"]
            warnings.append(
                Warning(
                    warning_id="WARN_IMPACT_SCOPE_NO_OOC_WINDOW",
                    message=(
                        "No matching OOC events were available; impact scope uses only the "
                        "source Lot process interval."
                    ),
                )
            )

        if matching_ooc_events or (chamber_abnormal_features and source_abnormal_features):
            exposed_rows = [
                row
                for row in self.repository.rows("process_history")
                if row["operation_no"] == target_operation_no
                and row["equipment_id"] == source_exposure["equipment_id"]
                and row["chamber_id"] == source_exposure["chamber_id"]
                and _overlaps(
                    row["started_at"],
                    row["ended_at"],
                    excursion_start,
                    excursion_end,
                )
            ]
        else:
            exposed_rows = [source_exposure]
        exposed_lots = sorted({row["lot_id"] for row in exposed_rows} | {lot_id})
        impact_lots = [item for item in exposed_lots if item != lot_id]
        affected_wafers = sorted({row["wafer_id"] for row in exposed_rows})
        source_wafer_id = source_exposure["wafer_id"]
        if chamber_abnormal_features and len(exposed_lots) == 1:
            impact_wafers = list(affected_wafers)
        elif chamber_abnormal_features:
            impact_wafers = [item for item in affected_wafers if not item.startswith(f"{lot_id}_")]
        else:
            impact_wafers = []
        normal_control_features = [
            row
            for row in self.repository.rows("fdc_feature")
            if row["lot_id"] not in exposed_lots
            and row["operation_no"] == target_operation_no
            and row["equipment_id"] == source_exposure["equipment_id"]
            and row["chamber_id"] == source_exposure["chamber_id"]
            and row["severity"] == "NORMAL"
            and row["ooc_flag"] == "false"
            and row["measured_at"] > excursion_end
        ]
        recovery_control_lots = sorted(
            {row["lot_id"] for row in normal_control_features}
        )
        criteria = {
            "operation_no": target_operation_no,
            "equipment_id": source_exposure["equipment_id"],
            "chamber_id": source_exposure["chamber_id"],
            "recipe_id": source_exposure["recipe_id"],
            "excursion_start": excursion_start,
            "excursion_end": excursion_end,
            "selection_rule": "same operation/equipment/chamber with process-time overlap",
            "source_wafer_id": source_wafer_id,
            "analysis_cutoff": analysis_cutoff,
        }

        lane_process_rows = [
            row
            for row in self.repository.rows("process_history")
            if row.get("lot_id") in set(exposed_lots)
        ]
        lane_fdc_rows = [
            row
            for row in self.repository.rows("fdc_feature")
            if row.get("lot_id") in set(exposed_lots)
        ]
        lane_candidates = _shared_exposure_lanes(
            lane_process_rows,
            exposed_lots,
            target_operation_no=target_operation_no,
            fdc_rows=lane_fdc_rows,
            evidence_scope="impact_lots",
        )
        source_lane = next(
            (
                lane
                for lane in lane_candidates
                if lane["operation"] == target_operation_no
                and lane["equipment"] == source_exposure["equipment_id"]
                and lane["chamber"] == source_exposure["chamber_id"]
                and lane["recipe"] == source_exposure["recipe_id"]
            ),
            None,
        )
        if source_lane is None:
            source_lane = {
                "lane_id": ":".join(
                    (
                        "lane",
                        target_operation_no,
                        source_exposure["equipment_id"],
                        source_exposure["chamber_id"],
                        source_exposure["recipe_id"],
                    )
                ),
                "operation": target_operation_no,
                "operation_name": source_exposure.get("operation_name", ""),
                "module": source_exposure.get("module", ""),
                "equipment": source_exposure["equipment_id"],
                "chamber": source_exposure["chamber_id"],
                "recipe": source_exposure["recipe_id"],
                "parameter_scope": [],
                "exposed_lot_ids": list(exposed_lots),
                "exposed_wafer_ids": list(affected_wafers),
                "time_window": [excursion_start, excursion_end],
                "priority_score": 1.0,
                "abnormal_signal_count": len(chamber_abnormal_features),
                "coverage": 1.0,
                "evidence_scope": "impact_lots",
            }
            lane_candidates.insert(0, source_lane)
        lane_scope_token = f"{lot_id}_{target_operation_no}_{len(exposed_lots)}"
        lane_evidence: list[Evidence] = []
        for lane in lane_candidates:
            lane_evidence_id = _evidence_id(
                "EV_MES_SHARED_LANE",
                f"{lane_scope_token}_{lane['lane_id']}",
            )
            lane["evidence_ids"] = [lane_evidence_id]
            lane_evidence.append(
                EvidenceBuilder.from_tool(
                    tool_input=tool_input,
                    evidence_id=lane_evidence_id,
                    evidence_type=EvidenceType.EQUIPMENT_EXPOSURE,
                    source_type=EvidenceSourceType.MES,
                    observation=(
                        f"Lane {lane['lane_id']} covers {len(lane['exposed_lot_ids'])}/"
                        f"{len(exposed_lots)} exposed Lots at operation {lane['operation']} "
                        f"on {lane['equipment']}/{lane['chamber']}."
                    ),
                    entities=[
                        *[
                            EvidenceEntity(
                                entity_type=EntityType.LOT.value,
                                entity_id=lane_lot_id,
                                attributes={"role": "lane_exposure"},
                            )
                            for lane_lot_id in lane["exposed_lot_ids"]
                        ],
                        EvidenceEntity(
                            entity_type=EntityType.OPERATION.value,
                            entity_id=lane["operation"],
                        ),
                        EvidenceEntity(
                            entity_type=EntityType.EQUIPMENT.value,
                            entity_id=lane["equipment"],
                        ),
                        EvidenceEntity(
                            entity_type=EntityType.CHAMBER.value,
                            entity_id=lane["chamber"],
                        ),
                        EvidenceEntity(
                            entity_type=EntityType.RECIPE.value,
                            entity_id=lane["recipe"],
                        ),
                    ],
                    confidence=round(min(0.99, 0.5 + 0.5 * lane["coverage"]), 3),
                    source_id=f"process_history:lane:{lane['lane_id']}",
                    source_table="process_history",
                    timestamp=(lane["time_window"][-1] if lane["time_window"] else None),
                    metadata={
                        "lane_id": lane["lane_id"],
                        "lane": dict(lane),
                        "source_lot_id": lot_id,
                    },
                )
            )

        evidence: list[Evidence] = []
        if matching_ooc_events:
            evidence.append(
                EvidenceBuilder.from_tool(
                    tool_input=tool_input,
                    evidence_id="EV_FDC_EXCURSION_WINDOW",
                    evidence_type=EvidenceType.EXCURSION_WINDOW,
                    source_type=EvidenceSourceType.FDC,
                    observation=(
                        f"Derived excursion window {excursion_start} to {excursion_end} from "
                        f"{len(matching_ooc_events)} matching OOC events."
                    ),
                    entities=[
                        EvidenceEntity(
                            entity_type=EntityType.LOT.value,
                            entity_id=lot_id,
                            attributes={"role": "source"},
                        ),
                        EvidenceEntity(
                            entity_type=EntityType.OPERATION.value,
                            entity_id=target_operation_no,
                        ),
                        EvidenceEntity(
                            entity_type=EntityType.EQUIPMENT.value,
                            entity_id=source_exposure["equipment_id"],
                        ),
                        EvidenceEntity(
                            entity_type=EntityType.CHAMBER.value,
                            entity_id=source_exposure["chamber_id"],
                        ),
                        EvidenceEntity(
                            entity_type=EntityType.EXCURSION.value,
                            entity_id=(
                                f"{source_exposure['equipment_id']}:"
                                f"{source_exposure['chamber_id']}:{target_operation_no}"
                            ),
                            attributes={
                                "start": excursion_start,
                                "end": excursion_end,
                            },
                        ),
                    ],
                    confidence=0.99,
                    source_id=(
                        f"ooc_event:{source_exposure['equipment_id']}:"
                        f"{source_exposure['chamber_id']}:{target_operation_no}"
                    ),
                    source_table="ooc_event",
                    timestamp=excursion_end,
                    metadata={
                        "event_count": len(matching_ooc_events),
                        "excursion_start": excursion_start,
                        "excursion_end": excursion_end,
                        "analysis_cutoff": analysis_cutoff,
                        "lane_id": source_lane["lane_id"],
                    },
                )
            )
        if recovery_control_lots:
            evidence.append(
                EvidenceBuilder.from_tool(
                    tool_input=tool_input,
                    evidence_id="EV_MES_RECOVERY_CONTROLS",
                    evidence_type=EvidenceType.NEGATIVE_SIGNAL,
                    source_type=EvidenceSourceType.ANALYTICS,
                    observation=(
                        f"Found {len(recovery_control_lots)} later control Lots with normal "
                        f"FDC on {source_exposure['equipment_id']}/"
                        f"{source_exposure['chamber_id']} after the excursion."
                    ),
                    entities=[
                        *[
                            EvidenceEntity(
                                entity_type=EntityType.LOT.value,
                                entity_id=control_lot_id,
                                attributes={"role": "recovery_control"},
                            )
                            for control_lot_id in recovery_control_lots
                        ],
                        EvidenceEntity(
                            entity_type=EntityType.EQUIPMENT.value,
                            entity_id=source_exposure["equipment_id"],
                        ),
                        EvidenceEntity(
                            entity_type=EntityType.CHAMBER.value,
                            entity_id=source_exposure["chamber_id"],
                        ),
                        *[
                            EvidenceEntity(
                                entity_type=EntityType.PARAMETER.value,
                                entity_id=parameter_name,
                            )
                            for parameter_name in sorted(
                                {row["parameter_name"] for row in normal_control_features}
                            )
                        ],
                    ],
                    confidence=0.95,
                    source_id=(
                        f"recovery_controls:{source_exposure['equipment_id']}:"
                        f"{source_exposure['chamber_id']}:{target_operation_no}"
                    ),
                    source_table="fdc_feature",
                    timestamp=max(
                        row["measured_at"] for row in normal_control_features
                    ),
                    metadata={
                        "control_lots": recovery_control_lots,
                        "normal_feature_count": len(normal_control_features),
                        "excursion_end": excursion_end,
                    },
                )
            )
        evidence.append(
            EvidenceBuilder.from_tool(
                tool_input=tool_input,
                evidence_id="EV_MES_IMPACT_LOTS",
                evidence_type=(
                    EvidenceType.IMPACT_SCOPE
                    if impact_lots or impact_wafers
                    else EvidenceType.NEGATIVE_SIGNAL
                ),
                source_type=EvidenceSourceType.ANALYTICS,
                observation=(
                    f"Found {len(impact_lots)} additional impact Lots sharing "
                    f"{target_operation_no}/{source_exposure['equipment_id']}/"
                    f"{source_exposure['chamber_id']} during the excursion window."
                ),
                entities=[
                    EvidenceEntity(
                        entity_type=EntityType.LOT.value,
                        entity_id=lot_id,
                        attributes={"role": "source"},
                    ),
                    *[
                        EvidenceEntity(
                            entity_type=EntityType.LOT.value,
                            entity_id=impact_lot_id,
                            attributes={"role": "impact"},
                        )
                        for impact_lot_id in impact_lots
                    ],
                    EvidenceEntity(
                        entity_type=EntityType.OPERATION.value,
                        entity_id=target_operation_no,
                    ),
                    EvidenceEntity(
                        entity_type=EntityType.EQUIPMENT.value,
                        entity_id=source_exposure["equipment_id"],
                    ),
                    EvidenceEntity(
                        entity_type=EntityType.CHAMBER.value,
                        entity_id=source_exposure["chamber_id"],
                    ),
                ],
                confidence=(
                    0.99
                    if matching_ooc_events
                    else (0.95 if chamber_abnormal_features and source_abnormal_features else 0.7)
                ),
                source_id=f"impact_scope:{lot_id}:{target_operation_no}",
                source_table="process_history",
                timestamp=excursion_end,
                metadata={
                    "source_lot_id": lot_id,
                    "impact_lots": impact_lots,
                    "affected_lots": exposed_lots,
                    "criteria": criteria,
                    "affected_wafers": affected_wafers,
                    "impact_wafers": impact_wafers,
                    "lane_id": source_lane["lane_id"],
                },
            )
        )
        evidence.extend(lane_evidence)
        if warnings:
            warnings = [
                Warning(
                    warning_id=warning.warning_id,
                    message=warning.message,
                    severity=warning.severity,
                    evidence_ids=["EV_MES_IMPACT_LOTS"],
                )
                for warning in warnings
            ]

        return _tool_output(
            tool_input,
            {
                "source_lot_id": lot_id,
                "affected_lots": exposed_lots,
                "impact_lots": impact_lots,
                "impact_count": len(impact_lots),
                "affected_wafers": affected_wafers,
                "impact_wafers": impact_wafers,
                "scope_level": "wafer" if len(exposed_lots) == 1 else "mixed",
                "target_operation_no": target_operation_no,
                "source_exposure": source_exposure,
                "impact_criteria": criteria,
                "recovery_control_lots": recovery_control_lots,
                "lane_candidates": lane_candidates,
            },
            evidence,
            warnings,
        )


class AnalyzeLotGenealogyTool(BaseTool):
    """Analyze MES genealogy commonality for affected lots."""

    def __init__(self, repository: FabRepository) -> None:
        super().__init__(
            repository=repository,
            tool_name="analyze_lot_genealogy",
            owner_agent=AgentKind.MES.value,
        )

    @_measure_tool_latency
    def run(self, tool_input: ToolInput) -> ToolOutput:
        self._validate_tool_name(tool_input)
        lot_ids = sorted(str(item) for item in tool_input.parameters["lot_ids"])
        selected_fdc = [
            row for row in self.repository.rows("fdc_feature") if row["lot_id"] in lot_ids
        ]
        requested_operation = tool_input.parameters.get("target_operation_no")
        if requested_operation:
            target_operation_no = str(requested_operation)
        else:
            signal_rows = [row for row in selected_fdc if row["ooc_flag"] == "true"]
            if not signal_rows:
                signal_rows = [row for row in selected_fdc if row["severity"] != "NORMAL"]
            if not signal_rows:
                signal_rows = selected_fdc
            if not signal_rows:
                raise ValueError(
                    "target_operation_no is required when selected Lots have no FDC features"
                )
            target_operation_no = Counter(row["operation_no"] for row in signal_rows).most_common(
                1
            )[0][0]

        process_rows = [
            row for row in self.repository.rows("process_history") if row["lot_id"] in lot_ids
        ]
        lane_candidates = _shared_exposure_lanes(
            process_rows,
            lot_ids,
            target_operation_no=target_operation_no,
            fdc_rows=selected_fdc,
            evidence_scope="affected_lots",
        )
        operation_commonality: list[dict[str, Any]] = []
        grouped: dict[str, list[Row]] = defaultdict(list)
        for row in process_rows:
            grouped[row["operation_no"]].append(row)

        for operation_no, rows in sorted(grouped.items()):
            grouped_assignments: dict[tuple[str, str, str], list[Row]] = defaultdict(list)
            for row in rows:
                grouped_assignments[
                    (row["equipment_id"], row["chamber_id"], row["recipe_id"])
                ].append(row)
            top_key, top_rows = max(
                grouped_assignments.items(),
                key=lambda item: (
                    len({row["lot_id"] for row in item[1]}),
                    len({row["wafer_id"] for row in item[1]}),
                ),
            )
            sample = rows[0]
            operation_lots = {row["lot_id"] for row in rows}
            operation_wafers = {row["wafer_id"] for row in rows}
            assignment_lots = {row["lot_id"] for row in top_rows}
            assignment_wafers = {row["wafer_id"] for row in top_rows}
            operation_commonality.append(
                {
                    "operation_no": operation_no,
                    "operation_name": sample["operation_name"],
                    "module": sample["module"],
                    "equipment_id": top_key[0],
                    "chamber_id": top_key[1],
                    "recipe_id": top_key[2],
                    "lot_count": len(assignment_lots),
                    "wafer_count": len(assignment_wafers),
                    "coverage": len(assignment_lots) / max(1, len(operation_lots)),
                    "wafer_coverage": len(assignment_wafers) / max(1, len(operation_wafers)),
                }
            )

        target_rows = [row for row in process_rows if row["operation_no"] == target_operation_no]
        if not target_rows:
            raise ValueError(
                f"selected Lots have no process history at operation {target_operation_no}"
            )
        target_groups: dict[tuple[str, str, str], list[Row]] = defaultdict(list)
        for row in target_rows:
            target_groups[(row["equipment_id"], row["chamber_id"], row["recipe_id"])].append(row)

        requested_equipment = str(tool_input.parameters.get("equipment_id", ""))
        requested_chamber = str(tool_input.parameters.get("chamber_id", ""))
        requested_recipe = str(tool_input.parameters.get("recipe_id", ""))
        preferred_signal_rows = [
            row
            for row in selected_fdc
            if row["operation_no"] == target_operation_no and row["ooc_flag"] == "true"
        ]
        if not preferred_signal_rows:
            preferred_signal_rows = [
                row
                for row in selected_fdc
                if row["operation_no"] == target_operation_no and row["severity"] != "NORMAL"
            ]
        preferred_pair = (
            (requested_equipment, requested_chamber)
            if requested_equipment and requested_chamber
            else (
                Counter(
                    (row["equipment_id"], row["chamber_id"]) for row in preferred_signal_rows
                ).most_common(1)[0][0]
                if preferred_signal_rows
                else None
            )
        )
        preferred_groups = [
            (key, rows)
            for key, rows in target_groups.items()
            if preferred_pair
            and key[:2] == preferred_pair
            and (not requested_recipe or key[2] == requested_recipe)
        ]
        candidate_groups = preferred_groups or list(target_groups.items())
        target_key, selected_target_rows = max(
            candidate_groups,
            key=lambda item: (
                len({row["lot_id"] for row in item[1]}),
                len({row["wafer_id"] for row in item[1]}),
            ),
        )
        target_lots = {row["lot_id"] for row in selected_target_rows}
        target_wafers = {row["wafer_id"] for row in selected_target_rows}
        all_target_wafers = {row["wafer_id"] for row in target_rows}
        lot_coverage = len(target_lots) / max(1, len(set(lot_ids)))
        wafer_coverage = len(target_wafers) / max(1, len(all_target_wafers))
        hold_rows = [
            row for row in self.repository.rows("hold_history") if row["lot_id"] in lot_ids
        ]

        target_lane = next(
            (
                lane
                for lane in lane_candidates
                if lane["operation"] == target_operation_no
                and lane["equipment"] == target_key[0]
                and lane["chamber"] == target_key[1]
                and lane["recipe"] == target_key[2]
            ),
            None,
        )
        if target_lane is None:
            target_lane = {
                "lane_id": ":".join(
                    ("lane", target_operation_no, target_key[0], target_key[1], target_key[2])
                ),
                "operation": target_operation_no,
                "operation_name": target_rows[0]["operation_name"],
                "module": target_rows[0]["module"],
                "equipment": target_key[0],
                "chamber": target_key[1],
                "recipe": target_key[2],
                "parameter_scope": [],
                "exposed_lot_ids": sorted(target_lots),
                "exposed_wafer_ids": sorted(target_wafers),
                "time_window": [],
                "priority_score": round(lot_coverage, 4),
                "abnormal_signal_count": 0,
                "coverage": round(lot_coverage, 4),
                "evidence_scope": "affected_lots",
            }
            lane_candidates.insert(0, target_lane)
        lane_scope_token = (
            f"{lot_ids[0]}_{lot_ids[-1]}_{len(lot_ids)}"
            if lot_ids
            else "empty"
        )
        lane_evidence: list[Evidence] = []
        for lane in lane_candidates:
            lane_evidence_id = _evidence_id(
                "EV_MES_SHARED_LANE",
                f"{lane_scope_token}_{lane['lane_id']}",
            )
            lane["evidence_ids"] = [lane_evidence_id]
            lane_evidence.append(
                EvidenceBuilder.from_tool(
                    tool_input=tool_input,
                    evidence_id=lane_evidence_id,
                    evidence_type=EvidenceType.EQUIPMENT_EXPOSURE,
                    source_type=EvidenceSourceType.MES,
                    observation=(
                        f"Lane {lane['lane_id']} covers {len(lane['exposed_lot_ids'])}/"
                        f"{len(lot_ids)} selected Lots at operation {lane['operation']} "
                        f"on {lane['equipment']}/{lane['chamber']}."
                    ),
                    entities=[
                        *[
                            EvidenceEntity(
                                entity_type=EntityType.LOT.value,
                                entity_id=lane_lot_id,
                                attributes={"role": "lane_exposure"},
                            )
                            for lane_lot_id in lane["exposed_lot_ids"]
                        ],
                        EvidenceEntity(
                            entity_type=EntityType.OPERATION.value,
                            entity_id=lane["operation"],
                        ),
                        EvidenceEntity(
                            entity_type=EntityType.EQUIPMENT.value,
                            entity_id=lane["equipment"],
                        ),
                        EvidenceEntity(
                            entity_type=EntityType.CHAMBER.value,
                            entity_id=lane["chamber"],
                        ),
                        EvidenceEntity(
                            entity_type=EntityType.RECIPE.value,
                            entity_id=lane["recipe"],
                        ),
                    ],
                    confidence=round(min(0.99, 0.5 + 0.5 * lane["coverage"]), 3),
                    source_id=f"process_history:lane:{lane['lane_id']}",
                    source_table="process_history",
                    timestamp=(lane["time_window"][-1] if lane["time_window"] else None),
                    metadata={
                        "lane_id": lane["lane_id"],
                        "lane": dict(lane),
                        "selected_lot_ids": list(lot_ids),
                    },
                )
            )

        evidence = [
            EvidenceBuilder.from_tool(
                tool_input=tool_input,
                evidence_id="EV_MES_COMMON_CHAMBER",
                evidence_type=EvidenceType.EQUIPMENT_EXPOSURE,
                source_type=EvidenceSourceType.MES,
                observation=(
                    f"{len(target_lots)}/{len(lot_ids)} selected Lots and "
                    f"{len(target_wafers)}/{len(all_target_wafers)} Wafers use operation "
                    f"{target_operation_no}, equipment {target_key[0]}, chamber {target_key[1]}."
                ),
                entities=[
                    *[
                        EvidenceEntity(
                            entity_type=EntityType.LOT.value,
                            entity_id=selected_lot_id,
                            attributes={"role": "selected"},
                        )
                        for selected_lot_id in lot_ids
                    ],
                    EvidenceEntity(
                        entity_type=EntityType.OPERATION.value,
                        entity_id=target_operation_no,
                    ),
                    EvidenceEntity(
                        entity_type=EntityType.EQUIPMENT.value,
                        entity_id=target_key[0],
                    ),
                    EvidenceEntity(
                        entity_type=EntityType.CHAMBER.value,
                        entity_id=target_key[1],
                    ),
                    EvidenceEntity(
                        entity_type=EntityType.RECIPE.value,
                        entity_id=target_key[2],
                    ),
                ],
                confidence=1.0,
                source_id=f"process_history:{target_operation_no}",
                source_table="process_history",
                metadata={
                    "operation_no": target_operation_no,
                    "equipment_id": target_key[0],
                    "chamber_id": target_key[1],
                    "recipe_id": target_key[2],
                    "lot_ids": lot_ids,
                    "wafer_ids": sorted(target_wafers),
                    "lot_coverage": lot_coverage,
                    "wafer_coverage": wafer_coverage,
                    "lane_id": target_lane["lane_id"],
                },
            )
        ]
        evidence.extend(lane_evidence)
        if hold_rows:
            hold_metadata = {
                "hold_count": len(hold_rows),
                "sample_comments": [row["hold_comment"] for row in hold_rows[:3]],
                "hold_ids": [row["hold_id"] for row in hold_rows],
            }
            evidence.extend(
                [
                    EvidenceBuilder.from_tool(
                        tool_input=tool_input,
                        evidence_id="EV_HOLD_COMMENT",
                        evidence_type=EvidenceType.HOLD_EVENT,
                        source_type=EvidenceSourceType.MES,
                        observation=(
                            f"Found {len(hold_rows)} engineering hold comments for affected Lots."
                        ),
                        entities=[
                            EvidenceEntity(
                                entity_type=EntityType.LOT.value,
                                entity_id=hold_lot_id,
                                attributes={
                                    "hold_ids": sorted(
                                        {
                                            row["hold_id"]
                                            for row in hold_rows
                                            if row["lot_id"] == hold_lot_id
                                        }
                                    )
                                },
                            )
                            for hold_lot_id in sorted({row["lot_id"] for row in hold_rows})
                        ],
                        confidence=1.0,
                        source_id="hold_history:affected_lots",
                        source_table="hold_history",
                        source_field="hold_comment",
                        timestamp=max(row["created_at"] for row in hold_rows),
                        metadata=hold_metadata,
                    ),
                    EvidenceBuilder.from_tool(
                        tool_input=tool_input,
                        evidence_id="EV_MES_LOT_HOLD",
                        evidence_type=EvidenceType.HOLD_EVENT,
                        source_type=EvidenceSourceType.MES,
                        observation=(
                            f"Found {len(hold_rows)} containment holds for the selected Lots."
                        ),
                        entities=[
                            EvidenceEntity(
                                entity_type=EntityType.LOT.value,
                                entity_id=hold_lot_id,
                                attributes={
                                    "hold_ids": sorted(
                                        {
                                            row["hold_id"]
                                            for row in hold_rows
                                            if row["lot_id"] == hold_lot_id
                                        }
                                    )
                                },
                            )
                            for hold_lot_id in sorted({row["lot_id"] for row in hold_rows})
                        ],
                        confidence=1.0,
                        source_id="hold_history:selected_lots",
                        source_table="hold_history",
                        timestamp=max(row["created_at"] for row in hold_rows),
                        metadata=hold_metadata,
                    ),
                ]
            )

        return _tool_output(
            tool_input,
            {
                "lot_ids": lot_ids,
                "target_operation_no": target_operation_no,
                "target_commonality": {
                    "equipment_id": target_key[0],
                    "chamber_id": target_key[1],
                    "recipe_id": target_key[2],
                    "lot_count": len(target_lots),
                    "wafer_count": len(target_wafers),
                    "coverage": lot_coverage,
                    "wafer_coverage": wafer_coverage,
                },
                "operation_commonality": operation_commonality,
                "lane_candidates": lane_candidates,
                "hold_count": len(hold_rows),
            },
            evidence,
        )


class AnalyzeParameterShiftTool(BaseTool):
    """Summarize FDC feature shifts for selected lots and equipment."""

    def __init__(self, repository: FabRepository) -> None:
        super().__init__(
            repository=repository,
            tool_name="analyze_parameter_shift",
            owner_agent=AgentKind.FDC.value,
        )

    @_measure_tool_latency
    def run(self, tool_input: ToolInput) -> ToolOutput:
        self._validate_tool_name(tool_input)
        lot_ids = sorted(str(item) for item in tool_input.parameters["lot_ids"])
        if not lot_ids:
            raise ModelValidationError("lot_ids must contain at least one Lot")
        operation_no = str(tool_input.parameters.get("operation_no", "6400"))
        equipment_id = tool_input.parameters.get("equipment_id")
        chamber_id = tool_input.parameters.get("chamber_id")
        recipe_id = str(tool_input.parameters.get("recipe_id", "")).strip()

        rows = [
            row
            for row in self.repository.rows("fdc_feature")
            if row["lot_id"] in lot_ids and row["operation_no"] == operation_no
        ]
        if equipment_id:
            rows = [row for row in rows if row["equipment_id"] == equipment_id]
        if chamber_id:
            rows = [row for row in rows if row["chamber_id"] == chamber_id]
        if recipe_id:
            rows = [row for row in rows if row["recipe_id"] == recipe_id]

        by_parameter: dict[str, list[Row]] = defaultdict(list)
        for row in rows:
            by_parameter[row["parameter_name"]].append(row)

        parameter_summary: list[dict[str, Any]] = []
        evidence: list[Evidence] = []
        warnings: list[Warning] = []
        evidence_ids_by_parameter = {
            "slurry_flow": "EV_FDC_SLURRY_FLOW",
            "endpoint_time": "EV_FDC_ENDPOINT_TIME",
        }
        for parameter_name, parameter_rows in sorted(by_parameter.items()):
            avg_observed = sum(_float(row["observed_value"]) for row in parameter_rows) / len(
                parameter_rows
            )
            avg_baseline = sum(_float(row["baseline_value"]) for row in parameter_rows) / len(
                parameter_rows
            )
            avg_delta = sum(_float(row["delta_percent"]) for row in parameter_rows) / len(
                parameter_rows
            )
            ooc_count = sum(1 for row in parameter_rows if row["ooc_flag"] == "true")
            parameter_summary.append(
                {
                    "parameter_name": parameter_name,
                    "avg_observed": round(avg_observed, 3),
                    "avg_baseline": round(avg_baseline, 3),
                    "avg_delta_percent": round(avg_delta, 3),
                    "ooc_count": ooc_count,
                    "row_count": len(parameter_rows),
                }
            )
            abnormal_rows = [
                row
                for row in parameter_rows
                if row["ooc_flag"] == "true" or row["severity"] != "NORMAL"
            ]
            # A positive parameter-deviation Evidence item may only carry the
            # rows that actually establish the deviation.  Attaching every
            # queried Lot here would turn normal exposure controls into false
            # positive process Evidence.  Negative-signal Evidence retains the
            # complete queried row set because those rows are the observation.
            evidence_rows = abnormal_rows or parameter_rows
            evidence_avg_observed = sum(
                _float(row["observed_value"]) for row in evidence_rows
            ) / len(evidence_rows)
            evidence_avg_baseline = sum(
                _float(row["baseline_value"]) for row in evidence_rows
            ) / len(evidence_rows)
            evidence_avg_delta = sum(
                _float(row["delta_percent"]) for row in evidence_rows
            ) / len(evidence_rows)
            parameter_entities = [
                *[
                    EvidenceEntity(
                        entity_type=EntityType.LOT.value,
                        entity_id=lot_id,
                    )
                    for lot_id in sorted({row["lot_id"] for row in evidence_rows})
                ],
                *[
                    EvidenceEntity(
                        entity_type=EntityType.WAFER.value,
                        entity_id=wafer_id,
                    )
                    for wafer_id in sorted(
                        {row["wafer_id"] for row in evidence_rows if row["wafer_id"]}
                    )
                ],
                EvidenceEntity(
                    entity_type=EntityType.OPERATION.value,
                    entity_id=operation_no,
                ),
                *[
                    EvidenceEntity(
                        entity_type=EntityType.EQUIPMENT.value,
                        entity_id=value,
                    )
                    for value in sorted({row["equipment_id"] for row in evidence_rows})
                ],
                *[
                    EvidenceEntity(
                        entity_type=EntityType.CHAMBER.value,
                        entity_id=value,
                    )
                    for value in sorted({row["chamber_id"] for row in evidence_rows})
                ],
                *[
                    EvidenceEntity(
                        entity_type=EntityType.RECIPE.value,
                        entity_id=recipe_id,
                        attributes={"versions": versions},
                    )
                    for recipe_id, versions in sorted(
                        {
                            row["recipe_id"]: sorted(
                                {
                                    item["recipe_version"]
                                    for item in evidence_rows
                                    if item["recipe_id"] == row["recipe_id"]
                                }
                            )
                            for row in evidence_rows
                            if row["recipe_id"]
                        }.items()
                    )
                ],
                EvidenceEntity(
                    entity_type=EntityType.PARAMETER.value,
                    entity_id=parameter_name,
                    attributes={"unit": evidence_rows[0]["unit"]},
                ),
            ]
            observation = (
                f"{parameter_name} average observed {evidence_avg_observed:.1f} vs "
                f"baseline {evidence_avg_baseline:.1f}; delta "
                f"{evidence_avg_delta:.1f}% with "
                f"{len(abnormal_rows)} abnormal feature records."
            )
            evidence.append(
                EvidenceBuilder.from_tool(
                    tool_input=tool_input,
                    evidence_id=evidence_ids_by_parameter.get(
                        parameter_name,
                        f"EV_FDC_{parameter_name.upper()}",
                    ),
                    evidence_type=(
                        EvidenceType.PARAMETER_DEVIATION
                        if abnormal_rows
                        else EvidenceType.NEGATIVE_SIGNAL
                    ),
                    source_type=EvidenceSourceType.FDC,
                    observation=observation,
                    entities=parameter_entities,
                    confidence=1.0,
                    source_id=f"fdc_feature:{operation_no}:{parameter_name}",
                    source_table="fdc_feature",
                    source_field=parameter_name,
                    timestamp=max(row["measured_at"] for row in evidence_rows),
                    metadata={
                        "operation_no": operation_no,
                        "equipment_id": equipment_id,
                        "chamber_id": chamber_id,
                        "recipe_id": recipe_id or None,
                        "lot_ids": sorted(
                            {row["lot_id"] for row in evidence_rows}
                        ),
                        "selected_lot_ids": lot_ids,
                        "abnormal_lot_ids": sorted(
                            {row["lot_id"] for row in abnormal_rows}
                        ),
                        "ooc_count": ooc_count,
                        "abnormal_row_count": len(abnormal_rows),
                    },
                )
            )

        if not rows:
            missing_evidence = EvidenceBuilder.from_tool(
                tool_input=tool_input,
                evidence_id="EV_FDC_FEATURE_DATA_MISSING",
                evidence_type=EvidenceType.DATA_MISSING,
                source_type=EvidenceSourceType.FDC,
                observation=(
                    "No FDC feature summaries are available for the selected Lots, "
                    "operation, equipment, and chamber."
                ),
                entities=[
                    *[
                        EvidenceEntity(
                            entity_type=EntityType.LOT.value,
                            entity_id=lot_id,
                        )
                        for lot_id in lot_ids
                    ],
                    EvidenceEntity(
                        entity_type=EntityType.OPERATION.value,
                        entity_id=operation_no,
                    ),
                    *(
                        [
                            EvidenceEntity(
                                entity_type=EntityType.EQUIPMENT.value,
                                entity_id=str(equipment_id),
                            )
                        ]
                        if equipment_id
                        else []
                    ),
                    *(
                        [
                            EvidenceEntity(
                                entity_type=EntityType.CHAMBER.value,
                                entity_id=str(chamber_id),
                            )
                        ]
                        if chamber_id
                        else []
                    ),
                ],
                confidence=1.0,
                source_id=f"fdc_feature:{operation_no}:selected_scope:missing",
                source_table="fdc_feature",
                metadata={
                    "operation_no": operation_no,
                    "equipment_id": equipment_id,
                    "chamber_id": chamber_id,
                    "recipe_id": recipe_id or None,
                    "lot_ids": lot_ids,
                },
            )
            evidence.append(missing_evidence)
            warnings.append(
                Warning(
                    warning_id="WARN_FDC_FEATURE_DATA_MISSING",
                    message=(
                        "FDC feature summaries are unavailable for the selected "
                        "Lot and equipment scope."
                    ),
                    evidence_ids=[missing_evidence.evidence_id],
                )
            )

        return _tool_output(
            tool_input,
            {
                "operation_no": operation_no,
                "equipment_id": equipment_id,
                "chamber_id": chamber_id,
                "recipe_id": recipe_id or None,
                "parameter_summary": parameter_summary,
            },
            evidence,
            warnings,
        )


def _maximum_same_side_run(values: list[float], center_line: float) -> tuple[int, str | None]:
    maximum = 0
    current = 0
    current_side: str | None = None
    maximum_side: str | None = None
    for value in values:
        side = "above" if value > center_line else "below" if value < center_line else None
        if side is None:
            current = 0
            current_side = None
        elif side == current_side:
            current += 1
        else:
            current = 1
            current_side = side
        if current > maximum:
            maximum = current
            maximum_side = current_side
    return maximum, maximum_side


def _monotonic_trend(values: list[float], minimum_length: int) -> tuple[bool, str | None]:
    if len(values) < minimum_length:
        return False, None
    for start in range(len(values) - minimum_length + 1):
        window = values[start : start + minimum_length]
        if all(left < right for left, right in pairwise(window)):
            return True, "increasing"
        if all(left > right for left, right in pairwise(window)):
            return True, "decreasing"
    return False, None


class PerformBasicSpcAnalysisTool(BaseTool):
    """Calculate bounded MVP SPC signals from FDC feature summaries."""

    def __init__(self, repository: FabRepository) -> None:
        super().__init__(
            repository=repository,
            tool_name="perform_basic_spc_analysis",
            owner_agent=AgentKind.FDC.value,
        )

    @staticmethod
    def _reference_rows(
        all_rows: list[Row],
        *,
        target_lot_ids: set[str],
        target_rows: list[Row],
        operation_no: str,
        equipment_id: str,
        chamber_id: str,
        parameter_name: str,
        minimum_samples: int,
        target_started_at: datetime,
    ) -> tuple[list[Row], str | None]:
        recipe_keys = {
            (row["recipe_id"], row["recipe_version"])
            for row in target_rows
            if row["recipe_id"] and row["recipe_version"]
        }
        candidates = [
            row
            for row in all_rows
            if row["lot_id"] not in target_lot_ids
            and row["operation_no"] == operation_no
            and row["parameter_name"] == parameter_name
            and row["severity"] == "NORMAL"
            and row["ooc_flag"] == "false"
            and _timestamp(row["measured_at"]) < target_started_at
            and (not recipe_keys or (row["recipe_id"], row["recipe_version"]) in recipe_keys)
        ]
        tiers = (
            (
                "same_chamber",
                [row for row in candidates if row["chamber_id"] == chamber_id],
            ),
            (
                "same_equipment",
                [row for row in candidates if row["equipment_id"] == equipment_id],
            ),
            ("operation_recipe_peer_group", candidates),
        )
        for scope, rows in tiers:
            if len(rows) >= minimum_samples:
                return rows, scope
        return [], None

    @_measure_tool_latency
    def run(self, tool_input: ToolInput) -> ToolOutput:
        self._validate_tool_name(tool_input)
        lot_ids = sorted({str(item) for item in tool_input.parameters["lot_ids"]})
        target_lot_ids = set(lot_ids)
        operation_no = str(tool_input.parameters.get("operation_no", "6400"))
        equipment_id = str(tool_input.parameters["equipment_id"])
        chamber_id = str(tool_input.parameters["chamber_id"])
        recipe_id = str(tool_input.parameters.get("recipe_id", "")).strip()
        minimum_samples = int(tool_input.parameters.get("minimum_baseline_samples", 20))
        sigma_multiplier = float(tool_input.parameters.get("sigma_multiplier", 3.0))
        same_side_run_length = int(tool_input.parameters.get("same_side_run_length", 8))
        trend_run_length = int(tool_input.parameters.get("trend_run_length", 6))
        if minimum_samples < 2:
            raise ValueError("minimum_baseline_samples must be at least 2")
        if sigma_multiplier <= 0:
            raise ValueError("sigma_multiplier must be positive")
        if same_side_run_length < 2 or trend_run_length < 2:
            raise ValueError("SPC run lengths must be at least 2")

        all_rows = self.repository.rows("fdc_feature")
        target_rows = [
            row
            for row in all_rows
            if row["lot_id"] in target_lot_ids
            and row["operation_no"] == operation_no
            and row["equipment_id"] == equipment_id
            and row["chamber_id"] == chamber_id
            and (not recipe_id or row["recipe_id"] == recipe_id)
        ]
        by_parameter: dict[str, list[Row]] = defaultdict(list)
        for row in target_rows:
            by_parameter[row["parameter_name"]].append(row)

        results: list[dict[str, Any]] = []
        evidence: list[Evidence] = []
        insufficient_parameters: list[str] = []
        for parameter_name, parameter_rows in sorted(by_parameter.items()):
            ordered_rows = sorted(parameter_rows, key=lambda row: row["measured_at"])
            reference_rows, baseline_scope = self._reference_rows(
                all_rows,
                target_lot_ids=target_lot_ids,
                target_rows=ordered_rows,
                operation_no=operation_no,
                equipment_id=equipment_id,
                chamber_id=chamber_id,
                parameter_name=parameter_name,
                minimum_samples=minimum_samples,
                target_started_at=_timestamp(ordered_rows[0]["measured_at"]),
            )
            if not reference_rows:
                insufficient_parameters.append(parameter_name)
                continue

            baseline_values = [_float(row["observed_value"]) for row in reference_rows]
            center_line = mean(baseline_values)
            sigma = stdev(baseline_values)
            if sigma <= 1e-12:
                insufficient_parameters.append(parameter_name)
                continue

            lower_control_limit = center_line - sigma_multiplier * sigma
            upper_control_limit = center_line + sigma_multiplier * sigma
            observed_values = [_float(row["observed_value"]) for row in ordered_rows]
            point_violations = []
            for row, value in zip(ordered_rows, observed_values, strict=True):
                if lower_control_limit <= value <= upper_control_limit:
                    continue
                point_violations.append(
                    {
                        "lot_id": row["lot_id"],
                        "wafer_id": row["wafer_id"],
                        "observed_value": round(value, 6),
                        "measured_at": row["measured_at"],
                        "direction": "high" if value > upper_control_limit else "low",
                        "z_score": round((value - center_line) / sigma, 3),
                    }
                )

            maximum_run, run_side = _maximum_same_side_run(observed_values, center_line)
            trend_detected, trend_direction = _monotonic_trend(
                observed_values,
                trend_run_length,
            )
            violated_rules: list[str] = []
            if point_violations:
                violated_rules.append("POINT_BEYOND_3_SIGMA")
            if maximum_run >= same_side_run_length:
                violated_rules.append("RUN_SAME_SIDE")
            if trend_detected:
                violated_rules.append("MONOTONIC_TREND")

            target_mean = mean(observed_values)
            evidence_id = EvidenceBuilder.scoped_evidence_id(
                tool_input,
                _evidence_id("EV_SPC", parameter_name),
            )
            result = {
                "parameter_name": parameter_name,
                "unit": ordered_rows[0]["unit"],
                "status": "OOC" if violated_rules else "IN_CONTROL",
                "baseline_scope": baseline_scope,
                "baseline_sample_count": len(reference_rows),
                "target_sample_count": len(ordered_rows),
                "baseline_window_start": min(row["measured_at"] for row in reference_rows),
                "baseline_window_end": max(row["measured_at"] for row in reference_rows),
                "target_window_start": ordered_rows[0]["measured_at"],
                "target_window_end": ordered_rows[-1]["measured_at"],
                "center_line": round(center_line, 6),
                "sigma": round(sigma, 6),
                "lower_control_limit": round(lower_control_limit, 6),
                "upper_control_limit": round(upper_control_limit, 6),
                "target_mean": round(target_mean, 6),
                "mean_z_score": round((target_mean - center_line) / sigma, 3),
                "point_violation_count": len(point_violations),
                "point_violations": point_violations[:50],
                "maximum_same_side_run": maximum_run,
                "same_side_direction": run_side,
                "trend_detected": trend_detected,
                "trend_direction": trend_direction,
                "violated_rules": violated_rules,
                "evidence_id": evidence_id,
            }
            results.append(result)
            evidence.append(
                EvidenceBuilder.from_tool(
                    tool_input=tool_input,
                    evidence_id=evidence_id,
                    evidence_type=(
                        EvidenceType.SPC_VIOLATION
                        if result["status"] == "OOC"
                        else EvidenceType.NEGATIVE_SIGNAL
                    ),
                    source_type=EvidenceSourceType.ANALYTICS,
                    observation=(
                        f"Minimal SPC classified {parameter_name} as {result['status']}; "
                        f"target mean {target_mean:.3f}, center line {center_line:.3f}, "
                        f"3-sigma limits [{lower_control_limit:.3f}, "
                        f"{upper_control_limit:.3f}], violations: "
                        f"{', '.join(violated_rules) if violated_rules else 'none'}."
                    ),
                    entities=[
                        *[
                            EvidenceEntity(
                                entity_type=EntityType.LOT.value,
                                entity_id=lot_id,
                            )
                            for lot_id in sorted({row["lot_id"] for row in ordered_rows})
                        ],
                        EvidenceEntity(
                            entity_type=EntityType.OPERATION.value,
                            entity_id=operation_no,
                        ),
                        EvidenceEntity(
                            entity_type=EntityType.EQUIPMENT.value,
                            entity_id=equipment_id,
                        ),
                        EvidenceEntity(
                            entity_type=EntityType.CHAMBER.value,
                            entity_id=chamber_id,
                        ),
                        *[
                            EvidenceEntity(
                                entity_type=EntityType.RECIPE.value,
                                entity_id=recipe_id,
                                attributes={"versions": versions},
                            )
                            for recipe_id, versions in sorted(
                                {
                                    row["recipe_id"]: sorted(
                                        {
                                            item["recipe_version"]
                                            for item in ordered_rows
                                            if item["recipe_id"] == row["recipe_id"]
                                        }
                                    )
                                    for row in ordered_rows
                                    if row["recipe_id"]
                                }.items()
                            )
                        ],
                        EvidenceEntity(
                            entity_type=EntityType.PARAMETER.value,
                            entity_id=parameter_name,
                            attributes={"unit": ordered_rows[0]["unit"]},
                        ),
                    ],
                    confidence=1.0,
                    source_id=(f"spc:{operation_no}:{equipment_id}:{chamber_id}:{parameter_name}"),
                    source_table="fdc_feature",
                    source_field=parameter_name,
                    timestamp=ordered_rows[-1]["measured_at"],
                    metadata={key: value for key, value in result.items() if key != "evidence_id"},
                )
            )

        warnings: list[Warning] = []
        if insufficient_parameters or not by_parameter:
            status_evidence = EvidenceBuilder.from_tool(
                tool_input=tool_input,
                evidence_id="EV_SPC_BASELINE_STATUS",
                evidence_type=EvidenceType.DATA_MISSING,
                source_type=EvidenceSourceType.ANALYTICS,
                observation=(
                    "Minimal SPC could not calculate control limits for "
                    + (
                        ", ".join(insufficient_parameters)
                        if insufficient_parameters
                        else "the selected scope because no target features were available"
                    )
                    + "."
                ),
                entities=[
                    *[
                        EvidenceEntity(
                            entity_type=EntityType.LOT.value,
                            entity_id=lot_id,
                        )
                        for lot_id in lot_ids
                    ],
                    EvidenceEntity(
                        entity_type=EntityType.OPERATION.value,
                        entity_id=operation_no,
                    ),
                    EvidenceEntity(
                        entity_type=EntityType.EQUIPMENT.value,
                        entity_id=equipment_id,
                    ),
                    EvidenceEntity(
                        entity_type=EntityType.CHAMBER.value,
                        entity_id=chamber_id,
                    ),
                    *[
                        EvidenceEntity(
                            entity_type=EntityType.PARAMETER.value,
                            entity_id=parameter_name,
                        )
                        for parameter_name in insufficient_parameters
                    ],
                ],
                confidence=1.0,
                source_id=f"spc:{operation_no}:{equipment_id}:{chamber_id}:baseline",
                source_table="fdc_feature",
                metadata={
                    "minimum_baseline_samples": minimum_samples,
                    "insufficient_parameters": insufficient_parameters,
                    "target_row_count": len(target_rows),
                    "recipe_id": recipe_id or None,
                },
            )
            evidence.append(status_evidence)
            warnings.append(
                Warning(
                    warning_id="WARN_SPC_BASELINE_INSUFFICIENT",
                    message=(
                        "Minimal SPC control limits were unavailable for: "
                        + (
                            ", ".join(insufficient_parameters)
                            if insufficient_parameters
                            else "selected scope"
                        )
                        + "."
                    ),
                    evidence_ids=[status_evidence.evidence_id],
                )
            )

        ooc_results = [item for item in results if item["status"] == "OOC"]
        return _tool_output(
            tool_input,
            {
                "lot_ids": lot_ids,
                "operation_no": operation_no,
                "equipment_id": equipment_id,
                "chamber_id": chamber_id,
                "recipe_id": recipe_id or None,
                "method": {
                    "control_limits": f"mean +/- {sigma_multiplier:g} sigma",
                    "minimum_baseline_samples": minimum_samples,
                    "same_side_run_length": same_side_run_length,
                    "trend_run_length": trend_run_length,
                },
                "spc_results": results,
                "analyzed_parameter_count": len(results),
                "ooc_parameter_count": len(ooc_results),
                "calculated_point_violation_count": sum(
                    int(item["point_violation_count"]) for item in results
                ),
                "baseline_insufficient_parameters": insufficient_parameters,
            },
            evidence,
            warnings,
        )


class AnalyzeSpcEvidenceTool(BaseTool):
    """Evaluate versioned SPC baselines using strict Fab context matching."""

    def __init__(self, repository: FabRepository) -> None:
        super().__init__(
            repository=repository,
            tool_name="analyze_spc_evidence",
            owner_agent=AgentKind.FDC.value,
        )

    @staticmethod
    def _matches_profile(row: Row, profile: Row) -> bool:
        return all(
            row.get(field, "") == profile[field]
            for field in (
                "operation_no",
                "equipment_id",
                "chamber_id",
                "recipe_id",
                "recipe_version",
            )
        )

    @staticmethod
    def _fdc_samples(
        rows: list[Row],
        parameter_name: str,
        *,
        aggregate_by_lot: bool,
    ) -> list[SpcSample]:
        if aggregate_by_lot:
            grouped: dict[str, list[Row]] = defaultdict(list)
            for row in rows:
                grouped[row["lot_id"]].append(row)
            return [
                SpcSample(
                    sample_id=f"FDC:{lot_id}:LOT_MEAN:{parameter_name}",
                    lot_id=lot_id,
                    subgroup_id=lot_id,
                    timestamp=max(row["measured_at"] for row in lot_rows),
                    value=mean(_float(row["observed_value"]) for row in lot_rows),
                )
                for lot_id, lot_rows in sorted(grouped.items())
            ]
        return [
            SpcSample(
                sample_id=(f"FDC:{row['lot_id']}:{row.get('wafer_id') or 'LOT'}:{parameter_name}"),
                lot_id=row["lot_id"],
                wafer_id=row.get("wafer_id") or None,
                subgroup_id=row["lot_id"],
                timestamp=row["measured_at"],
                value=_float(row["observed_value"]),
            )
            for row in sorted(rows, key=lambda item: (item["measured_at"], item["lot_id"]))
        ]

    def _wat_samples(self, lot_ids: set[str]) -> list[SpcSample]:
        grouped: dict[str, list[Row]] = defaultdict(list)
        for row in self.repository.rows("wat_result"):
            if row["lot_id"] in lot_ids:
                grouped[row["lot_id"]].append(row)
        return [
            SpcSample(
                sample_id=f"WAT:{lot_id}:FAIL_FRACTION",
                lot_id=lot_id,
                subgroup_id=lot_id,
                timestamp=max(row["tested_at"] for row in rows),
                value=sum(row["pass_fail"] == "false" for row in rows) / len(rows),
                sample_size=len(rows),
                defect_count=sum(row["pass_fail"] == "false" for row in rows),
            )
            for lot_id, rows in sorted(grouped.items())
            if rows
        ]

    def _profile_lots(self, profile: Row, start: str, end: str) -> set[str]:
        return {
            row["lot_id"]
            for row in self.repository.rows("process_history")
            if self._matches_profile(row, profile) and start <= row["ended_at"] <= end
        }

    @_measure_tool_latency
    def run(self, tool_input: ToolInput) -> ToolOutput:
        self._validate_tool_name(tool_input)
        lot_ids = sorted({str(item) for item in tool_input.parameters["lot_ids"]})
        operation_no = str(tool_input.parameters.get("operation_no", "6400"))
        equipment_id = str(tool_input.parameters["equipment_id"])
        chamber_id = str(tool_input.parameters["chamber_id"])
        recipe_id = str(tool_input.parameters.get("recipe_id", "")).strip()
        requested_lot_ids = set(lot_ids)
        scope_rows = self.repository.rows("spc_excursion_lot")
        matching_excursions = {
            row.get("excursion_id", "")
            for row in self.repository.rows("ooc_event")
            if row.get("event_source") == "SPC"
            and row["operation_no"] == operation_no
            and row["equipment_id"] == equipment_id
            and row["chamber_id"] == chamber_id
            and (
                row.get("trigger_lot_id", "") in requested_lot_ids
                or any(
                    scope["lot_id"] in requested_lot_ids
                    for scope in scope_rows
                    if scope["excursion_id"] == row.get("excursion_id", "")
                )
            )
        }
        if matching_excursions:
            lot_ids = sorted(
                requested_lot_ids
                | {
                    row["lot_id"]
                    for row in scope_rows
                    if row["excursion_id"] in matching_excursions
                }
            )
        target_process = [
            row
            for row in self.repository.rows("process_history")
            if row["lot_id"] in lot_ids
            and row["operation_no"] == operation_no
            and row["equipment_id"] == equipment_id
            and row["chamber_id"] == chamber_id
            and (not recipe_id or row["recipe_id"] == recipe_id)
        ]
        recipe_keys = {(row["recipe_id"], row["recipe_version"]) for row in target_process}
        profiles = [
            row
            for row in self.repository.rows("spc_baseline_profile")
            if row["status"] == "REFERENCE"
            and row["operation_no"] == operation_no
            and row["equipment_id"] == equipment_id
            and row["chamber_id"] == chamber_id
            and (row["recipe_id"], row["recipe_version"]) in recipe_keys
        ]
        evidence: list[Evidence] = []
        warnings: list[Warning] = []
        results: list[dict[str, Any]] = []
        insufficient: list[str] = []
        fdc_rows = self.repository.rows("fdc_feature")

        for profile in profiles:
            parameter_name = profile["parameter_name"]
            chart_type = profile["chart_type"]
            baseline_start = profile["baseline_start"]
            baseline_end = profile["baseline_end"]
            if profile["source_table"] == "fdc_feature":
                baseline_rows = [
                    row
                    for row in fdc_rows
                    if self._matches_profile(row, profile)
                    and row["parameter_name"] == parameter_name
                    and baseline_start <= row["measured_at"] <= baseline_end
                ]
                analysis_rows = [
                    row
                    for row in fdc_rows
                    if row["lot_id"] in lot_ids
                    and self._matches_profile(row, profile)
                    and row["parameter_name"] == parameter_name
                ]
                aggregate_by_lot = chart_type == SpcChartType.I_MR.value
                baseline = self._fdc_samples(
                    baseline_rows,
                    parameter_name,
                    aggregate_by_lot=aggregate_by_lot,
                )
                analysis = self._fdc_samples(
                    analysis_rows,
                    parameter_name,
                    aggregate_by_lot=aggregate_by_lot,
                )
            else:
                baseline_lots = self._profile_lots(
                    profile,
                    baseline_start,
                    baseline_end,
                )
                analysis_lots = {
                    row["lot_id"] for row in target_process if self._matches_profile(row, profile)
                }
                baseline = self._wat_samples(baseline_lots)
                analysis = self._wat_samples(analysis_lots)

            minimum = int(profile["minimum_sample_count"])
            if len(baseline) < minimum or not analysis:
                insufficient.append(parameter_name)
                continue
            spec_lower = _float(profile["spec_lower"]) if profile["spec_lower"] else None
            spec_upper = _float(profile["spec_upper"]) if profile["spec_upper"] else None
            if chart_type == SpcChartType.I_MR.value:
                chart = calculate_imr(
                    baseline,
                    analysis,
                    parameter_name=parameter_name,
                    unit=profile["unit"],
                    spec_lower=spec_lower,
                    spec_upper=spec_upper,
                )
            elif chart_type in {SpcChartType.XBAR_S.value, SpcChartType.XBAR_R.value}:
                chart = calculate_xbar(
                    baseline,
                    analysis,
                    parameter_name=parameter_name,
                    unit=profile["unit"],
                    chart_type=chart_type,
                    spec_lower=spec_lower,
                    spec_upper=spec_upper,
                )
            else:
                chart = calculate_p_chart(baseline, analysis, parameter_name=parameter_name)

            evidence_id = EvidenceBuilder.scoped_evidence_id(
                tool_input,
                _evidence_id("EV_SPC", parameter_name),
            )
            result = {
                **chart.to_dict(),
                "baseline_id": profile["baseline_id"],
                "baseline_scope": "strict_versioned_profile",
                "baseline_window": {"start": baseline_start, "end": baseline_end},
                "analysis_lot_ids": lot_ids,
                "evidence_id": evidence_id,
                "target_mean": round(mean(item.value for item in analysis), 6),
            }
            results.append(result)
            evidence.append(
                EvidenceBuilder.from_tool(
                    tool_input=tool_input,
                    evidence_id=evidence_id,
                    evidence_type=(
                        EvidenceType.SPC_VIOLATION
                        if chart.status == "OOC"
                        else EvidenceType.NEGATIVE_SIGNAL
                    ),
                    source_type=EvidenceSourceType.ANALYTICS,
                    observation=(
                        f"{chart_type} classified {parameter_name} as {chart.status} using "
                        f"versioned baseline {profile['baseline_id']}; "
                        f"{len(chart.violations)} Nelson-rule windows were detected."
                    ),
                    entities=[
                        *[
                            EvidenceEntity(
                                entity_type=EntityType.LOT.value,
                                entity_id=lot_id,
                            )
                            for lot_id in lot_ids
                        ],
                        EvidenceEntity(
                            entity_type=EntityType.PRODUCT.value,
                            entity_id=profile["product_id"],
                        ),
                        EvidenceEntity(
                            entity_type=EntityType.OPERATION.value,
                            entity_id=operation_no,
                        ),
                        EvidenceEntity(
                            entity_type=EntityType.EQUIPMENT.value,
                            entity_id=equipment_id,
                        ),
                        EvidenceEntity(
                            entity_type=EntityType.CHAMBER.value,
                            entity_id=chamber_id,
                        ),
                        EvidenceEntity(
                            entity_type=EntityType.RECIPE.value,
                            entity_id=profile["recipe_id"],
                            attributes={"version": profile["recipe_version"]},
                        ),
                        EvidenceEntity(
                            entity_type=EntityType.PARAMETER.value,
                            entity_id=parameter_name,
                            attributes={
                                "unit": profile["unit"],
                                "chart_type": chart_type,
                            },
                        ),
                    ],
                    confidence=1.0,
                    source_id=f"spc_baseline_profile:{profile['baseline_id']}",
                    source_table="spc_baseline_profile",
                    source_field="parameter_name",
                    timestamp=max(item.timestamp for item in analysis),
                    metadata={
                        "baseline_id": profile["baseline_id"],
                        "baseline_window": {"start": baseline_start, "end": baseline_end},
                        "strict_group": {
                            "product_id": profile["product_id"],
                            "operation_no": operation_no,
                            "equipment_id": equipment_id,
                            "chamber_id": chamber_id,
                            "recipe_id": profile["recipe_id"],
                            "recipe_version": profile["recipe_version"],
                            "parameter_name": parameter_name,
                        },
                        "chart": result,
                    },
                )
            )

        if not profiles:
            missing_profile_evidence = EvidenceBuilder.from_tool(
                tool_input=tool_input,
                evidence_id="EV_SPC_PROFILE_DATA_MISSING",
                evidence_type=EvidenceType.DATA_MISSING,
                source_type=EvidenceSourceType.ANALYTICS,
                observation=(
                    "No versioned SPC baseline matched the exact product, operation, "
                    "equipment, chamber, recipe version, and parameter context."
                ),
                entities=[
                    *[
                        EvidenceEntity(
                            entity_type=EntityType.LOT.value,
                            entity_id=lot_id,
                        )
                        for lot_id in lot_ids
                    ],
                    EvidenceEntity(
                        entity_type=EntityType.OPERATION.value,
                        entity_id=operation_no,
                    ),
                    EvidenceEntity(
                        entity_type=EntityType.EQUIPMENT.value,
                        entity_id=equipment_id,
                    ),
                    EvidenceEntity(
                        entity_type=EntityType.CHAMBER.value,
                        entity_id=chamber_id,
                    ),
                ],
                confidence=1.0,
                source_id=(
                    f"spc_baseline_profile:{operation_no}:{equipment_id}:{chamber_id}:missing"
                ),
                source_table="spc_baseline_profile",
                metadata={
                    "lot_ids": lot_ids,
                    "recipe_id": recipe_id or None,
                },
            )
            evidence.append(missing_profile_evidence)
            warnings.append(
                Warning(
                    warning_id="WARN_SPC_PROFILE_NOT_FOUND",
                    message=(
                        "No versioned SPC baseline matched the exact product, operation, "
                        "equipment, chamber, recipe version, and parameter context."
                    ),
                    evidence_ids=[missing_profile_evidence.evidence_id],
                )
            )
        if insufficient:
            insufficient_evidence = EvidenceBuilder.from_tool(
                tool_input=tool_input,
                evidence_id="EV_SPC_BASELINE_DATA_MISSING",
                evidence_type=EvidenceType.DATA_MISSING,
                source_type=EvidenceSourceType.ANALYTICS,
                observation=(
                    "SPC baseline or analysis data were insufficient for: "
                    + ", ".join(insufficient)
                    + "."
                ),
                entities=[
                    *[
                        EvidenceEntity(
                            entity_type=EntityType.LOT.value,
                            entity_id=lot_id,
                        )
                        for lot_id in lot_ids
                    ],
                    EvidenceEntity(
                        entity_type=EntityType.OPERATION.value,
                        entity_id=operation_no,
                    ),
                    EvidenceEntity(
                        entity_type=EntityType.EQUIPMENT.value,
                        entity_id=equipment_id,
                    ),
                    EvidenceEntity(
                        entity_type=EntityType.CHAMBER.value,
                        entity_id=chamber_id,
                    ),
                    *[
                        EvidenceEntity(
                            entity_type=EntityType.PARAMETER.value,
                            entity_id=parameter_name,
                        )
                        for parameter_name in insufficient
                    ],
                ],
                confidence=1.0,
                source_id=(
                    f"spc_baseline_profile:{operation_no}:{equipment_id}:{chamber_id}:insufficient"
                ),
                source_table="spc_baseline_profile",
                metadata={
                    "insufficient_parameters": insufficient,
                    "recipe_id": recipe_id or None,
                },
            )
            evidence.append(insufficient_evidence)
            warnings.append(
                Warning(
                    warning_id="WARN_SPC_BASELINE_INSUFFICIENT",
                    message="SPC baseline or analysis data were insufficient for: "
                    + ", ".join(insufficient)
                    + ".",
                    evidence_ids=[insufficient_evidence.evidence_id],
                )
            )
        ooc_results = [item for item in results if item["status"] == "OOC"]
        return _tool_output(
            tool_input,
            {
                "lot_ids": lot_ids,
                "operation_no": operation_no,
                "equipment_id": equipment_id,
                "chamber_id": chamber_id,
                "recipe_id": recipe_id or None,
                "method": {
                    "engine": "deterministic_advanced_spc",
                    "rules": "Nelson Rules 1-8",
                    "baseline_matching": (
                        "product + operation + equipment + chamber + recipe version + parameter"
                    ),
                },
                "spc_results": results,
                "analyzed_parameter_count": len(results),
                "ooc_parameter_count": len(ooc_results),
                "calculated_point_violation_count": sum(
                    int(item["point_violation_count"]) for item in results
                ),
                "baseline_insufficient_parameters": insufficient,
            },
            evidence,
            warnings,
        )


class FindOocEventsTool(BaseTool):
    """Find FDC OOC events for equipment and chamber."""

    def __init__(self, repository: FabRepository) -> None:
        super().__init__(
            repository=repository,
            tool_name="find_ooc_events",
            owner_agent=AgentKind.FDC.value,
        )

    @_measure_tool_latency
    def run(self, tool_input: ToolInput) -> ToolOutput:
        self._validate_tool_name(tool_input)
        equipment_id = str(tool_input.parameters["equipment_id"])
        chamber_id = str(tool_input.parameters["chamber_id"])
        operation_no = str(tool_input.parameters.get("operation_no", "6400"))

        rows = filter_rows(
            self.repository.rows("ooc_event"),
            equipment_id=equipment_id,
            chamber_id=chamber_id,
            operation_no=operation_no,
        )
        severity_counts = Counter(row["severity"] for row in rows)
        excursions = {row["excursion_id"]: row for row in self.repository.rows("spc_excursion")}
        scope_rows = self.repository.rows("spc_excursion_lot")
        holds = {row["hold_id"]: row for row in self.repository.rows("hold_history")}
        spc_contexts: list[dict[str, Any]] = []
        complete_hold_contexts: list[dict[str, Any]] = []
        missing_hold_contexts: list[dict[str, Any]] = []
        missing_excursion_contexts: list[dict[str, Any]] = []
        for row in rows:
            if row.get("event_source", "FDC") != "SPC":
                continue
            excursion_id = row.get("excursion_id", "")
            scopes = [item for item in scope_rows if item["excursion_id"] == excursion_id]
            trigger_scopes = [item for item in scopes if item["scope_role"] == "TRIGGER"]
            trigger_lot_id = row.get("trigger_lot_id", "")
            trigger_hold_id = row.get("trigger_hold_id", "")
            trigger_hold = holds.get(trigger_hold_id)
            missing_hold_references: list[str] = []
            if not trigger_hold_id:
                missing_hold_references.append("trigger_hold_id")
            elif trigger_hold is None:
                missing_hold_references.append(trigger_hold_id)
            elif trigger_hold.get("lot_id") != trigger_lot_id:
                missing_hold_references.append(f"{trigger_hold_id}:lot_mismatch")
            for scope in scopes:
                scope_hold_id = scope.get("hold_id", "")
                scope_hold = holds.get(scope_hold_id)
                if not scope_hold_id:
                    missing_hold_references.append(f"{scope['lot_id']}:hold_id")
                elif scope_hold is None:
                    missing_hold_references.append(scope_hold_id)
                elif scope_hold.get("lot_id") != scope["lot_id"]:
                    missing_hold_references.append(f"{scope_hold_id}:lot_mismatch")

            context_reasons: list[str] = []
            excursion = excursions.get(excursion_id)
            if not excursion_id or excursion is None:
                context_reasons.append("excursion_record")
            if not trigger_lot_id:
                context_reasons.append("trigger_lot_id")
            if len(trigger_scopes) != 1:
                context_reasons.append("trigger_scope")
            elif trigger_scopes[0]["lot_id"] != trigger_lot_id:
                context_reasons.append("trigger_scope_lot_mismatch")

            hold_link_complete = not missing_hold_references
            excursion_link_complete = not context_reasons
            context = {
                "event_key": row.get("event_key"),
                "trigger_lot_id": trigger_lot_id,
                "trigger_wafer_id": row.get("trigger_wafer_id") or None,
                "trigger_hold": trigger_hold,
                "spc_rule_codes": [
                    item for item in row.get("spc_rule_codes", "").split(";") if item
                ],
                "excursion": excursion,
                "trigger_scope": trigger_scopes[0] if len(trigger_scopes) == 1 else None,
                "impact_scopes": [
                    {**item, "hold": holds.get(item["hold_id"])}
                    for item in scopes
                    if item["scope_role"] == "IMPACT"
                ],
                "hold_link_complete": hold_link_complete,
                "excursion_link_complete": excursion_link_complete,
                "missing_hold_references": sorted(set(missing_hold_references)),
                "missing_context_reasons": sorted(set(context_reasons)),
            }
            spc_contexts.append(context)
            if hold_link_complete and excursion_link_complete:
                complete_hold_contexts.append(context)
            if not hold_link_complete:
                missing_hold_contexts.append(context)
            if not excursion_link_complete:
                missing_excursion_contexts.append(context)

        scope_entities = [
            EvidenceEntity(
                entity_type=EntityType.OPERATION.value,
                entity_id=operation_no,
            ),
            EvidenceEntity(
                entity_type=EntityType.EQUIPMENT.value,
                entity_id=equipment_id,
            ),
            EvidenceEntity(
                entity_type=EntityType.CHAMBER.value,
                entity_id=chamber_id,
            ),
            *[
                EvidenceEntity(
                    entity_type=EntityType.PARAMETER.value,
                    entity_id=parameter_name,
                )
                for parameter_name in sorted(
                    {row["parameter_name"] for row in rows if row["parameter_name"]}
                )
            ],
            *[
                EvidenceEntity(
                    entity_type=EntityType.LOT.value,
                    entity_id=lot_id,
                    attributes={"role": "spc_trigger"},
                )
                for lot_id in sorted(
                    {row.get("trigger_lot_id", "") for row in rows}
                    - {""}
                )
            ],
            *[
                EvidenceEntity(
                    entity_type=EntityType.WAFER.value,
                    entity_id=wafer_id,
                    attributes={"role": "spc_trigger"},
                )
                for wafer_id in sorted(
                    {row.get("trigger_wafer_id", "") for row in rows}
                    - {""}
                )
            ],
            *[
                EvidenceEntity(
                    entity_type=EntityType.EXCURSION.value,
                    entity_id=excursion_id,
                )
                for excursion_id in sorted(
                    {row.get("excursion_id", "") for row in rows}
                    - {""}
                )
            ],
        ]
        evidence = [
            EvidenceBuilder.from_tool(
                tool_input=tool_input,
                evidence_id="EV_OOC_EVENTS",
                evidence_type=(
                    EvidenceType.OOC_EVENT if rows else EvidenceType.NEGATIVE_SIGNAL
                ),
                source_type=(
                    EvidenceSourceType.ANALYTICS
                    if spc_contexts
                    else EvidenceSourceType.FDC
                ),
                observation=(
                    f"Found {len(rows)} OOC events for {equipment_id}/{chamber_id} "
                    f"at operation {operation_no}."
                    if rows
                    else (
                        f"No OOC events were recorded for {equipment_id}/{chamber_id} "
                        f"at operation {operation_no}."
                    )
                ),
                entities=scope_entities,
                confidence=1.0,
                source_id=f"ooc_event:{equipment_id}:{chamber_id}:{operation_no}",
                source_table="ooc_event",
                timestamp=max((row["triggered_at"] for row in rows), default=None),
                metadata={
                    "severity_counts": dict(severity_counts),
                    "spc_contexts": spc_contexts,
                },
            )
        ]
        valid_excursion_contexts = [
            context for context in spc_contexts if context["excursion_link_complete"]
        ]
        if valid_excursion_contexts:
            evidence.append(
                EvidenceBuilder.from_tool(
                    tool_input=tool_input,
                    evidence_id="EV_SPC_OOC_CONTEXT",
                    evidence_type=EvidenceType.EXCURSION_WINDOW,
                    source_type=EvidenceSourceType.ANALYTICS,
                    observation=(
                        f"Resolved {len(valid_excursion_contexts)} SPC OOC trigger(s) to "
                        "valid Excursion windows; "
                        f"{len(complete_hold_contexts)} also have complete Hold links."
                    ),
                    entities=[
                        *scope_entities,
                        *[
                            EvidenceEntity(
                                entity_type=EntityType.LOT.value,
                                entity_id=lot_id,
                                attributes={"role": "spc_impact"},
                            )
                            for lot_id in sorted(
                                {
                                    scope["lot_id"]
                                    for context in valid_excursion_contexts
                                    for scope in context["impact_scopes"]
                                }
                            )
                        ],
                    ],
                    confidence=1.0,
                    source_id=(
                        "spc_excursion:"
                        + ",".join(
                            sorted(
                                {
                                    context["excursion"]["excursion_id"]
                                    for context in valid_excursion_contexts
                                    if context["excursion"]
                                }
                            )
                        )
                    ),
                    source_table="spc_excursion",
                    timestamp=max(row["triggered_at"] for row in rows),
                    metadata={"spc_contexts": valid_excursion_contexts},
                )
            )
        if complete_hold_contexts:
            hold_lot_ids = sorted(
                {
                    context["trigger_lot_id"]
                    for context in complete_hold_contexts
                    if context["trigger_lot_id"]
                }
                | {
                    scope["lot_id"]
                    for context in complete_hold_contexts
                    for scope in context["impact_scopes"]
                }
            )
            evidence.append(
                EvidenceBuilder.from_tool(
                    tool_input=tool_input,
                    evidence_id="EV_SPC_HOLD_CONTEXT",
                    evidence_type=EvidenceType.HOLD_EVENT,
                    source_type=EvidenceSourceType.MES,
                    observation=(
                        f"Verified complete Trigger and Impact Hold links for "
                        f"{len(complete_hold_contexts)} SPC OOC context(s)."
                    ),
                    entities=[
                        EvidenceEntity(
                            entity_type=EntityType.LOT.value,
                            entity_id=lot_id,
                            attributes={
                                "hold_ids": sorted(
                                    {
                                        hold["hold_id"]
                                        for context in complete_hold_contexts
                                        for hold in [
                                            context["trigger_hold"],
                                            *[
                                                scope["hold"]
                                                for scope in context["impact_scopes"]
                                            ],
                                        ]
                                        if hold and hold["lot_id"] == lot_id
                                    }
                                )
                            },
                        )
                        for lot_id in hold_lot_ids
                    ],
                    confidence=1.0,
                    source_id=f"hold_history:{equipment_id}:{chamber_id}:{operation_no}:spc",
                    source_table="hold_history",
                    timestamp=max(
                        hold["created_at"]
                        for context in complete_hold_contexts
                        for hold in [
                            context["trigger_hold"],
                            *[scope["hold"] for scope in context["impact_scopes"]],
                        ]
                        if hold
                    ),
                    metadata={"spc_contexts": complete_hold_contexts},
                )
            )

        context_warnings: list[Warning] = []
        if missing_hold_contexts:
            missing_hold_evidence = EvidenceBuilder.from_tool(
                tool_input=tool_input,
                evidence_id="EV_SPC_HOLD_DATA_MISSING",
                evidence_type=EvidenceType.DATA_MISSING,
                source_type=EvidenceSourceType.MES,
                observation=(
                    f"{len(missing_hold_contexts)} SPC OOC context(s) lack complete, "
                    "Lot-matched Trigger or Impact Hold records."
                ),
                entities=scope_entities,
                confidence=1.0,
                source_id=f"hold_history:{equipment_id}:{chamber_id}:{operation_no}:missing",
                source_table="hold_history",
                metadata={"spc_contexts": missing_hold_contexts},
            )
            evidence.append(missing_hold_evidence)
            context_warnings.append(
                Warning(
                    warning_id="WARN_SPC_HOLD_MISSING",
                    message=(
                        "A real SPC OOC was found, but its Trigger or Impact Hold "
                        "containment records are incomplete."
                    ),
                    evidence_ids=[missing_hold_evidence.evidence_id],
                )
            )
        if missing_excursion_contexts:
            missing_context_evidence = EvidenceBuilder.from_tool(
                tool_input=tool_input,
                evidence_id="EV_SPC_CONTEXT_DATA_MISSING",
                evidence_type=EvidenceType.DATA_MISSING,
                source_type=EvidenceSourceType.ANALYTICS,
                observation=(
                    f"{len(missing_excursion_contexts)} SPC OOC event(s) lack a valid "
                    "Excursion record or Trigger scope link."
                ),
                entities=scope_entities,
                confidence=1.0,
                source_id=f"spc_excursion:{equipment_id}:{chamber_id}:{operation_no}:missing",
                source_table="spc_excursion",
                metadata={"spc_contexts": missing_excursion_contexts},
            )
            evidence.append(missing_context_evidence)
            context_warnings.append(
                Warning(
                    warning_id="WARN_SPC_CONTEXT_MISSING",
                    message=(
                        "An SPC OOC event is missing a valid Excursion or Trigger "
                        "scope relationship."
                    ),
                    evidence_ids=[missing_context_evidence.evidence_id],
                )
            )
        return _tool_output(
            tool_input,
            {
                "equipment_id": equipment_id,
                "chamber_id": chamber_id,
                "operation_no": operation_no,
                "event_count": len(rows),
                "severity_counts": dict(severity_counts),
                "events": rows,
                "spc_contexts": spc_contexts,
            },
            evidence,
            context_warnings,
        )


class SummarizeDefectWatTool(BaseTool):
    """Summarize physical defect and WAT evidence for selected lots."""

    def __init__(self, repository: FabRepository) -> None:
        super().__init__(
            repository=repository,
            tool_name="summarize_defect_wat",
            owner_agent=AgentKind.DEFECT_WAT.value,
        )

    @_measure_tool_latency
    def run(self, tool_input: ToolInput) -> ToolOutput:
        self._validate_tool_name(tool_input)
        lot_ids = sorted(str(item) for item in tool_input.parameters["lot_ids"])
        if not lot_ids:
            raise ModelValidationError("lot_ids must contain at least one Lot")
        evidence_scope = str(tool_input.parameters.get("evidence_scope", "selected_lots"))
        if evidence_scope not in {"selected_lots", "shared_exposure_comparison"}:
            raise ModelValidationError(
                "evidence_scope must be selected_lots or shared_exposure_comparison"
            )
        scope_suffix = "_SHARED_EXPOSURE" if evidence_scope == "shared_exposure_comparison" else ""
        defect_rows = [
            row for row in self.repository.rows("defect_summary") if row["lot_id"] in lot_ids
        ]
        wat_rows = [row for row in self.repository.rows("wat_result") if row["lot_id"] in lot_ids]
        metrology_rows = [
            row for row in self.repository.rows("metrology_result") if row["lot_id"] in lot_ids
        ]

        defect_counts = Counter(row["defect_type"] for row in defect_rows)
        defect_patterns = Counter(row["pattern_type"] for row in defect_rows)
        wat_fail_modes = Counter(row["fail_mode"] for row in wat_rows if row["fail_mode"])
        failed_wat_rows = [row for row in wat_rows if row["pass_fail"] == "false"]
        passing_wat_rows = [row for row in wat_rows if row["pass_fail"] == "true"]
        failed_wat_lot_ids = sorted({row["lot_id"] for row in failed_wat_rows})
        wat_fail_count = len(failed_wat_lot_ids)
        wat_fail_record_count = len(failed_wat_rows)
        missing_wat_lot_ids = sorted(set(lot_ids) - {row["lot_id"] for row in wat_rows})
        metrology_groups: dict[tuple[str, str], list[Row]] = defaultdict(list)
        for row in metrology_rows:
            metrology_groups[(row["measurement_stage"], row["metric_name"])].append(row)

        evidence: list[Evidence] = []
        if defect_rows:
            dominant_defect, dominant_defect_count = defect_counts.most_common(1)[0]
            dominant_pattern = defect_patterns.most_common(1)[0][0]
            dominant_defect_rows = [
                row
                for row in defect_rows
                if row["defect_type"] == dominant_defect
            ]
            evidence.append(
                EvidenceBuilder.from_tool(
                    tool_input=tool_input,
                    evidence_id=_evidence_id("EV_DEFECT", f"{dominant_defect}{scope_suffix}"),
                    evidence_type=EvidenceType.DEFECT_SIGNAL,
                    source_type=EvidenceSourceType.DEFECT,
                    observation=(
                        f"Selected Lots show {dominant_defect_count} {dominant_defect} records; "
                        f"the dominant spatial pattern is {dominant_pattern}."
                    ),
                    entities=[
                        *[
                            EvidenceEntity(
                                entity_type=EntityType.LOT.value,
                                entity_id=lot_id,
                            )
                            for lot_id in sorted(
                                {row["lot_id"] for row in dominant_defect_rows}
                            )
                        ],
                        *[
                            EvidenceEntity(
                                entity_type=EntityType.WAFER.value,
                                entity_id=wafer_id,
                            )
                            for wafer_id in sorted(
                                {row["wafer_id"] for row in dominant_defect_rows}
                            )
                        ],
                        EvidenceEntity(
                            entity_type=EntityType.DEFECT.value,
                            entity_id=dominant_defect,
                            attributes={"dominant_pattern": dominant_pattern},
                        ),
                    ],
                    confidence=1.0,
                    source_id=f"defect_summary:{evidence_scope}:{','.join(lot_ids)}",
                    source_table="defect_summary",
                    timestamp=max(
                        (row["inspected_at"] for row in dominant_defect_rows),
                        default=None,
                    ),
                    metadata={
                        "defect_counts": dict(defect_counts),
                        "pattern_counts": dict(defect_patterns),
                        "evidence_scope": evidence_scope,
                        **_explicit_mechanism_intermediate_metadata(
                            dominant_defect_rows,
                            source_table="defect_summary",
                        ),
                    },
                )
            )
        if wat_fail_modes:
            dominant_fail_mode, dominant_fail_count = wat_fail_modes.most_common(1)[0]
            evidence.append(
                EvidenceBuilder.from_tool(
                    tool_input=tool_input,
                    evidence_id=_evidence_id("EV_WAT", f"{dominant_fail_mode}{scope_suffix}"),
                    evidence_type=EvidenceType.ELECTRICAL_FAILURE,
                    source_type=EvidenceSourceType.WAT,
                    observation=(
                        f"{wat_fail_count} selected Lots fail WAT; {dominant_fail_count} "
                        f"records carry the dominant {dominant_fail_mode} signature."
                    ),
                    entities=[
                        *[
                            EvidenceEntity(
                                entity_type=EntityType.LOT.value,
                                entity_id=lot_id,
                            )
                            for lot_id in failed_wat_lot_ids
                        ],
                        EvidenceEntity(
                            entity_type=EntityType.WAT_ITEM.value,
                            entity_id=dominant_fail_mode,
                        ),
                    ],
                    confidence=1.0,
                    source_id="wat_result:affected_lots",
                    source_table="wat_result",
                    source_field="pass_fail",
                    timestamp=max((row["tested_at"] for row in wat_rows), default=None),
                    metadata={
                        "fail_modes": dict(wat_fail_modes),
                        "wat_fail_count": wat_fail_count,
                        "wat_fail_lot_count": wat_fail_count,
                        "wat_fail_record_count": wat_fail_record_count,
                        "test_equipment_ids": sorted(
                            {
                                row.get("test_equipment_id", "")
                                for row in failed_wat_rows
                                if row.get("test_equipment_id", "")
                            }
                        ),
                        "test_items": sorted(
                            {
                                row.get("test_item", "")
                                for row in failed_wat_rows
                                if row.get("test_item", "")
                            }
                        ),
                        "parameter_names": sorted(
                            {
                                row.get("parameter_name", "")
                                for row in failed_wat_rows
                                if row.get("parameter_name", "")
                            }
                        ),
                    },
                )
            )
        if passing_wat_rows:
            passing_lot_ids = sorted({row["lot_id"] for row in passing_wat_rows})
            evidence.append(
                EvidenceBuilder.from_tool(
                    tool_input=tool_input,
                    evidence_id=f"EV_WAT_PASSING_CONTROLS{scope_suffix}",
                    evidence_type=EvidenceType.NEGATIVE_SIGNAL,
                    source_type=EvidenceSourceType.WAT,
                    observation=(
                        f"{len(passing_wat_rows)} passing WAT control records are available "
                        f"across {len(passing_lot_ids)} selected Lots."
                    ),
                    entities=[
                        *[
                            EvidenceEntity(
                                entity_type=EntityType.LOT.value,
                                entity_id=passing_lot_id,
                            )
                            for passing_lot_id in passing_lot_ids
                        ],
                        *[
                            EvidenceEntity(
                                entity_type=EntityType.WAT_ITEM.value,
                                entity_id=parameter_name,
                            )
                            for parameter_name in sorted(
                                {row["parameter_name"] for row in passing_wat_rows}
                            )
                        ],
                    ],
                    confidence=1.0,
                    source_id="wat_result:selected_lots:passing_controls",
                    source_table="wat_result",
                    source_field="pass_fail",
                    timestamp=max(row["tested_at"] for row in passing_wat_rows),
                    metadata={
                        "pass_record_count": len(passing_wat_rows),
                        "test_equipment_ids": sorted(
                            {
                                row.get("test_equipment_id", "")
                                for row in passing_wat_rows
                                if row.get("test_equipment_id", "")
                            }
                        ),
                        "test_items": sorted(
                            {
                                row.get("test_item", "")
                                for row in passing_wat_rows
                                if row.get("test_item", "")
                            }
                        ),
                        "parameter_names": sorted(
                            {
                                row.get("parameter_name", "")
                                for row in passing_wat_rows
                                if row.get("parameter_name", "")
                            }
                        ),
                    },
                )
            )
        metrology_summaries: list[dict[str, Any]] = []
        for (stage, metric_name), metric_rows in sorted(metrology_groups.items()):
            failed_rows = [row for row in metric_rows if row["pass_fail"] == "false"]
            odd_rows = [
                row for row in metric_rows if int(row["wafer_id"].rsplit("W", 1)[-1]) % 2 == 1
            ]
            even_rows = [
                row for row in metric_rows if int(row["wafer_id"].rsplit("W", 1)[-1]) % 2 == 0
            ]

            def average(rows: list[Row]) -> float | None:
                if not rows:
                    return None
                return round(sum(_float(row["measured_value"]) for row in rows) / len(rows), 3)

            summary = {
                "measurement_stage": stage,
                "metric_name": metric_name,
                "row_count": len(metric_rows),
                "fail_count": len(failed_rows),
                "outlier_wafers": sorted({row["wafer_id"] for row in failed_rows}),
                "overall_average": average(metric_rows),
                "odd_wafer_average": average(odd_rows),
                "even_wafer_average": average(even_rows),
                "unit": metric_rows[0]["unit"],
            }
            metrology_summaries.append(summary)
            # Positive metrology Evidence is scoped only to the rows that are
            # actually out of specification. Passing rows must not inherit the
            # positive deviation merely because they were queried together.
            evidence_rows = failed_rows or metric_rows
            evidence.append(
                EvidenceBuilder.from_tool(
                    tool_input=tool_input,
                    evidence_id=_evidence_id(
                        "EV_METROLOGY",
                        f"{stage}_{metric_name}{scope_suffix}",
                    ),
                    evidence_type=(
                        EvidenceType.METROLOGY_DEVIATION
                        if failed_rows
                        else EvidenceType.NEGATIVE_SIGNAL
                    ),
                    source_type=EvidenceSourceType.ANALYTICS,
                    observation=(
                        f"{stage} {metric_name} has {len(failed_rows)}/{len(metric_rows)} "
                        f"out-of-spec Wafer records; odd/even averages are "
                        f"{summary['odd_wafer_average']} and {summary['even_wafer_average']} "
                        f"{metric_rows[0]['unit']}."
                    ),
                    entities=[
                        *[
                            EvidenceEntity(
                                entity_type=EntityType.LOT.value,
                                entity_id=lot_id,
                            )
                            for lot_id in sorted({row["lot_id"] for row in evidence_rows})
                        ],
                        *[
                            EvidenceEntity(
                                entity_type=EntityType.WAFER.value,
                                entity_id=wafer_id,
                                attributes={"status": "out_of_spec"},
                            )
                            for wafer_id in sorted({row["wafer_id"] for row in failed_rows})
                        ],
                        EvidenceEntity(
                            entity_type=EntityType.PARAMETER.value,
                            entity_id=f"{stage}:{metric_name}",
                            attributes={"unit": metric_rows[0]["unit"]},
                        ),
                    ],
                    confidence=1.0,
                    source_id=f"metrology_result:{stage}:{metric_name}",
                    source_table="metrology_result",
                    source_field="measured_value",
                    timestamp=max(row["measured_at"] for row in evidence_rows),
                    metadata={
                        **summary,
                        **_explicit_mechanism_intermediate_metadata(
                            evidence_rows,
                            source_table="metrology_result",
                        ),
                    },
                )
            )
        has_quality_impact = bool(
            defect_rows
            or failed_wat_rows
            or any(
                int(summary["fail_count"]) > 0
                for summary in metrology_summaries
            )
        )
        if not has_quality_impact:
            quality_data_available = bool(defect_rows or wat_rows or metrology_rows)
            observed_quality_lot_ids = sorted(
                {row["lot_id"] for row in [*defect_rows, *wat_rows, *metrology_rows]}
            )
            evidence.append(
                EvidenceBuilder.from_tool(
                    tool_input=tool_input,
                    evidence_id=f"EV_QUALITY_NO_IMPACT{scope_suffix}",
                    evidence_type=(
                        EvidenceType.NEGATIVE_SIGNAL
                        if quality_data_available
                        else EvidenceType.DATA_MISSING
                    ),
                    source_type=EvidenceSourceType.ANALYTICS,
                    observation=(
                        "Available quality records show no defects, WAT failures, or "
                        "out-of-spec metrology results for the selected Lots."
                        if quality_data_available
                        else (
                            "No Defect, WAT, or Metrology records are available for the "
                            "selected Lots."
                        )
                    ),
                    entities=[
                        EvidenceEntity(
                            entity_type=EntityType.LOT.value,
                            entity_id=lot_id,
                        )
                        for lot_id in (
                            observed_quality_lot_ids if quality_data_available else lot_ids
                        )
                    ],
                    confidence=1.0,
                    source_id="quality:selected_lots:no_impact",
                    source_table="wat_result" if wat_rows else None,
                    timestamp=max((row["tested_at"] for row in wat_rows), default=None),
                    metadata={
                        "lot_ids": lot_ids,
                        "observed_quality_lot_ids": observed_quality_lot_ids,
                    },
                )
            )
        warnings: list[Warning] = []
        if missing_wat_lot_ids and (defect_rows or wat_rows or metrology_rows):
            evidence.append(
                EvidenceBuilder.from_tool(
                    tool_input=tool_input,
                    evidence_id=f"EV_QUALITY_WAT_DATA_MISSING{scope_suffix}",
                    evidence_type=EvidenceType.DATA_MISSING,
                    source_type=EvidenceSourceType.WAT,
                    observation=(
                        f"{len(missing_wat_lot_ids)} selected Lots have no WAT records; "
                        "their electrical failure status is unknown."
                    ),
                    entities=[
                        EvidenceEntity(
                            entity_type=EntityType.LOT.value,
                            entity_id=lot_id,
                        )
                        for lot_id in missing_wat_lot_ids
                    ],
                    confidence=1.0,
                    source_id="wat_result:selected_lots:missing",
                    source_table="wat_result",
                    source_field="pass_fail",
                    metadata={"missing_wat_lot_ids": missing_wat_lot_ids},
                )
            )
            warnings.append(
                Warning(
                    warning_id="WARN_WAT_DATA_MISSING",
                    message=(
                        f"WAT data is unavailable for {len(missing_wat_lot_ids)} selected Lots."
                    ),
                    evidence_ids=["EV_QUALITY_WAT_DATA_MISSING"],
                )
            )
        return _tool_output(
            tool_input,
            {
                "lot_ids": lot_ids,
                "defect_counts": dict(defect_counts),
                "defect_patterns": dict(defect_patterns),
                "wat_fail_modes": dict(wat_fail_modes),
                "wat_fail_count": wat_fail_count,
                "wat_fail_lot_count": wat_fail_count,
                "wat_fail_record_count": wat_fail_record_count,
                "missing_wat_lot_ids": missing_wat_lot_ids,
                "metrology_summaries": metrology_summaries,
                "metrology_fail_count": sum(
                    int(item["fail_count"]) for item in metrology_summaries
                ),
            },
            evidence,
            warnings,
        )


class RetrieveSimilarCaseTool(BaseTool):
    """Retrieve historical RCA case evidence from the knowledge MVP tables."""

    def __init__(self, repository: FabRepository, retriever: Retriever | None = None) -> None:
        super().__init__(
            repository=repository,
            tool_name="retrieve_similar_case",
            owner_agent=AgentKind.KNOWLEDGE.value,
        )
        self.retriever = retriever or KeywordRetriever(KnowledgeAssetRepository(repository))

    @_measure_tool_latency
    def run(self, tool_input: ToolInput) -> ToolOutput:
        self._validate_tool_name(tool_input)
        query = str(tool_input.parameters["query"]).strip().lower()
        if not query:
            raise ModelValidationError("query must be a non-empty string")
        module = str(tool_input.parameters.get("module", "")).lower()
        equipment_type = str(tool_input.parameters.get("equipment_type", "")).lower()
        source_lot_id = str(tool_input.parameters.get("source_lot_id", "")).strip().upper()
        product_id = str(tool_input.parameters.get("product_id", "")).strip()
        detected_operation = str(
            tool_input.parameters.get("detected_operation", "")
        ).strip()
        detected_equipment_id = str(
            tool_input.parameters.get("detected_equipment_id", "")
        ).strip()
        detected_at = str(tool_input.parameters.get("detected_at", "")).strip()
        raw_symptom_types = tool_input.parameters.get("symptom_types", [])
        if not isinstance(raw_symptom_types, list):
            raise ModelValidationError("symptom_types must be a list")
        symptom_types = tuple(
            dict.fromkeys(
                str(item).strip() for item in raw_symptom_types if str(item).strip()
            )
        )
        explicit_module_limit = tool_input.parameters.get(
            "explicit_module_limit", False
        )
        if not isinstance(explicit_module_limit, bool):
            raise ModelValidationError("explicit_module_limit must be a boolean")
        match_evidence_id = str(
            tool_input.parameters.get("match_evidence_id", "EV_KNOWLEDGE_MATCH")
        )
        missing_evidence_id = str(
            tool_input.parameters.get(
                "missing_evidence_id",
                "EV_KNOWLEDGE_NO_CONFIRMED_MATCH",
            )
        )

        retrieval_result = self.retriever.retrieve(
            RetrievalQuery(
                query=query,
                module=module,
                equipment_type=equipment_type,
                source_lot_id=source_lot_id,
                product_id=product_id,
                detected_operation=detected_operation,
                detected_equipment_id=detected_equipment_id,
                detected_at=detected_at,
                symptom_types=symptom_types,
                explicit_module_limit=explicit_module_limit,
            )
        )
        observation_scope = (
            retrieval_result.observation_scope.to_dict()
            if retrieval_result.observation_scope
            else None
        )
        causal_search_scope = (
            retrieval_result.causal_search_scope.to_dict()
            if retrieval_result.causal_search_scope
            else None
        )
        cases = [hit.to_legacy_case() for hit in retrieval_result.hits]
        if not cases:
            missing_evidence = EvidenceBuilder.from_tool(
                tool_input=tool_input,
                evidence_id=missing_evidence_id,
                # Retrieval completed against an available governed source.
                # "No confirmed match" is a negative search observation, not
                # evidence that the source itself was unavailable.
                evidence_type=EvidenceType.NEGATIVE_SIGNAL,
                source_type=EvidenceSourceType.KNOWLEDGE,
                observation="No engineer-confirmed historical RCA case is available.",
                entities=[
                    EvidenceEntity(
                        entity_type=EntityType.KNOWLEDGE_ASSET.value,
                        entity_id="confirmed_rca_cases",
                        attributes={
                            "asset_type": "rca_case_collection",
                            "required_validation_status": "CONFIRMED",
                        },
                    )
                ],
                confidence=1.0,
                source_id="confirmed_rca_cases",
                source_table="rca_case",
                metadata={
                    "validation_status": "CONFIRMED",
                    "retrieval_status": "no_match",
                    "source_available": True,
                },
            )
            return _tool_output(
                tool_input,
                {
                    "query": query,
                    "cases": [],
                    "top_case": None,
                    "documents": [],
                    "retrieval_strategy": "no_match",
                    "score_components": {},
                    "calibrated_relevance": None,
                    "source_confidence": None,
                    "matched_chunk_ids": [],
                    "candidate_lanes": [],
                    "scope_reasons": [],
                    "route_distance": None,
                    "shared_resource_types": [],
                    "scope_fusion_score": None,
                    "observation_scope": observation_scope,
                    "causal_search_scope": causal_search_scope,
                },
                [missing_evidence],
                [
                    Warning(
                        warning_id="WARN_KNOWLEDGE_NO_CONFIRMED_CASE",
                        message="No engineer-confirmed historical RCA case was available.",
                        evidence_ids=[missing_evidence.evidence_id],
                    )
                ],
            )
        best_hit = retrieval_result.top_hit
        if best_hit is None:
            raise ModelValidationError("retriever returned cases without a top hit")
        best_case = best_hit.to_legacy_case()
        documents = [document.to_legacy_row() for document in best_hit.asset.documents]

        evidence = [
            EvidenceBuilder.from_tool(
                tool_input=tool_input,
                evidence_id=match_evidence_id,
                evidence_type=EvidenceType.HISTORICAL_CASE_MATCH,
                source_type=EvidenceSourceType.KNOWLEDGE,
                observation=(
                    f"Historical RCA case {best_case['case_id']} matches query with "
                    f"similarity {best_case['similarity']}."
                ),
                entities=[
                    EvidenceEntity(
                        entity_type=EntityType.KNOWLEDGE_ASSET.value,
                        entity_id=best_case["case_id"],
                        attributes={
                            "asset_type": "rca_case",
                            "validation_status": best_case.get(
                                "validation_status",
                                "CONFIRMED",
                            ),
                            "module": best_case["module"],
                            "equipment_type": best_case["equipment_type"],
                        },
                    ),
                    *[
                        EvidenceEntity(
                            entity_type=EntityType.KNOWLEDGE_ASSET.value,
                            entity_id=document["document_id"],
                            attributes={
                                "asset_type": document["document_type"],
                                "case_id": document["case_id"],
                                "validation_status": document.get(
                                    "validation_status",
                                    "CONFIRMED",
                                ),
                            },
                        )
                        for document in documents
                    ],
                ],
                confidence=float(best_case["similarity"]),
                source_id=best_case["case_id"],
                source_table="rca_case",
                source_field="root_cause",
                timestamp=best_case["created_at"],
                metadata={
                    "case_id": best_case["case_id"],
                    "root_cause": best_case["root_cause"],
                    "validation_status": best_case.get("validation_status", "CONFIRMED"),
                    "documents": documents,
                    "retrieval_strategy": best_hit.retrieval_strategy,
                    "score_components": dict(best_hit.score_components),
                    "calibrated_relevance": best_hit.calibrated_relevance,
                    "source_confidence": best_hit.source_confidence,
                    "matched_chunk_ids": list(best_hit.matched_chunk_ids),
                    "candidate_lanes": list(best_hit.candidate_lanes),
                    "scope_reasons": list(best_hit.scope_reasons),
                    "route_distance": best_hit.route_distance,
                    "shared_resource_types": list(best_hit.shared_resource_types),
                    "scope_fusion_score": best_hit.scope_fusion_score,
                    "observation_scope": observation_scope,
                    "causal_search_scope": causal_search_scope,
                },
            )
        ]
        return _tool_output(
            tool_input,
            {
                "query": query,
                "cases": cases,
                "top_case": best_case,
                "documents": documents,
                "retrieval_strategy": best_hit.retrieval_strategy,
                "score_components": dict(best_hit.score_components),
                "calibrated_relevance": best_hit.calibrated_relevance,
                "source_confidence": best_hit.source_confidence,
                "matched_chunk_ids": list(best_hit.matched_chunk_ids),
                "candidate_lanes": list(best_hit.candidate_lanes),
                "scope_reasons": list(best_hit.scope_reasons),
                "route_distance": best_hit.route_distance,
                "shared_resource_types": list(best_hit.shared_resource_types),
                "scope_fusion_score": best_hit.scope_fusion_score,
                "observation_scope": observation_scope,
                "causal_search_scope": causal_search_scope,
            },
            evidence,
        )
