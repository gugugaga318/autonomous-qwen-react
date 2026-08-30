"""Specialist Agents for the Yield RCA MVP.

Each Agent is a thin orchestration boundary over its assigned Tools. Database
and repository access remain encapsulated by the Tool Layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from yield_rca_core.models import (
    AgentFinding,
    AgentKind,
    Evidence,
    FindingKind,
    ToolInput,
    ToolOutput,
    Warning,
)
from yield_rca_core.tool_layer import (
    AnalyzeLotGenealogyTool,
    AnalyzeParameterShiftTool,
    AnalyzeSpcEvidenceTool,
    FindAffectedLotsTool,
    FindImpactLotsTool,
    FindOocEventsTool,
    GetLotContextTool,
    PerformBasicSpcAnalysisTool,
    RetrieveSimilarCaseTool,
    SummarizeDefectWatTool,
)


def _tool_input(
    *,
    tool_name: str,
    request_id: str,
    parameters: dict[str, Any],
    requested_by: str,
) -> ToolInput:
    return ToolInput(
        tool_name=tool_name,
        request_id=request_id,
        parameters=parameters,
        requested_by=requested_by,
    )


def _evidence_ids(*outputs: ToolOutput) -> list[str]:
    return list(dict.fromkeys(item for output in outputs for item in output.evidence_ids))


def _evidence_payload(*outputs: ToolOutput) -> list[dict[str, Any]]:
    evidence_by_id: dict[str, dict[str, Any]] = {}
    for output in outputs:
        for item in output.data.get("evidence", []):
            evidence_by_id[str(item["evidence_id"])] = dict(item)
    return list(evidence_by_id.values())


def _typed_evidence(*outputs: ToolOutput) -> list[Evidence]:
    """Merge first-class Tool Evidence without rebuilding it from legacy data."""

    evidence_by_id: dict[str, Evidence] = {}
    for output in outputs:
        for item in output.evidence:
            evidence_by_id.setdefault(item.evidence_id, item)
    return list(evidence_by_id.values())


def _warnings(*outputs: ToolOutput, additional: list[Warning] | None = None) -> list[Warning]:
    warnings_by_id = {
        warning.warning_id: warning for output in outputs for warning in output.warnings
    }
    for warning in additional or []:
        warnings_by_id[warning.warning_id] = warning
    return list(warnings_by_id.values())


@dataclass(frozen=True)
class MESAgent:
    """Identify affected lots and MES process commonality."""

    find_affected_lots_tool: FindAffectedLotsTool
    analyze_lot_genealogy_tool: AnalyzeLotGenealogyTool
    get_lot_context_tool: GetLotContextTool | None = None
    find_impact_lots_tool: FindImpactLotsTool | None = None

    def analyze(
        self,
        *,
        request_id: str,
        product_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
        target_operation_no: str | None = None,
    ) -> AgentFinding:
        affected_output = self.find_affected_lots_tool.run(
            _tool_input(
                tool_name="find_affected_lots",
                request_id=f"{request_id}:affected-lots",
                parameters={
                    "product_id": product_id,
                    "start_date": start_date,
                    "end_date": end_date,
                },
                requested_by=AgentKind.MES.value,
            )
        )
        affected_lots = list(affected_output.data["affected_lots"])

        if not affected_lots:
            warning = Warning(
                warning_id="WARN_MES_NO_AFFECTED_LOTS",
                message=f"No affected lots were found for product {product_id}.",
                evidence_ids=list(affected_output.evidence_ids),
            )
            evidence = _typed_evidence(affected_output)
            return AgentFinding(
                finding_id=f"{request_id}:mes",
                agent=AgentKind.MES.value,
                summary=f"No affected lots were identified for {product_id}.",
                confidence=0.2,
                evidence_ids=[item.evidence_id for item in evidence],
                evidence=evidence,
                details={
                    "product_id": product_id,
                    "affected_lots": [],
                    "normal_lots": list(affected_output.data["normal_lots"]),
                    "yield_trend": list(affected_output.data["yield_trend"]),
                },
                warnings=_warnings(affected_output, additional=[warning]),
            )

        genealogy_parameters: dict[str, Any] = {"lot_ids": affected_lots}
        if target_operation_no:
            genealogy_parameters["target_operation_no"] = target_operation_no
        genealogy_output = self.analyze_lot_genealogy_tool.run(
            _tool_input(
                tool_name="analyze_lot_genealogy",
                request_id=f"{request_id}:genealogy",
                parameters=genealogy_parameters,
                requested_by=AgentKind.MES.value,
            )
        )
        commonality = dict(genealogy_output.data["target_commonality"])
        resolved_operation = str(genealogy_output.data["target_operation_no"])
        coverage = float(commonality["coverage"])
        additional_warnings: list[Warning] = []
        if coverage < 0.8:
            additional_warnings.append(
                Warning(
                    warning_id="WARN_MES_WEAK_COMMONALITY",
                    message="MES commonality coverage is below 80% for the target operation.",
                    evidence_ids=list(genealogy_output.evidence_ids),
                )
            )

        evidence = _typed_evidence(affected_output, genealogy_output)
        return AgentFinding(
            finding_id=f"{request_id}:mes",
            agent=AgentKind.MES.value,
            summary=(
                f"{len(affected_lots)} affected lots identified; {coverage:.0%} share "
                f"operation {resolved_operation} on {commonality['equipment_id']}/"
                f"{commonality['chamber_id']}."
            ),
            confidence=round(min(0.99, 0.55 + 0.4 * coverage), 3),
            evidence_ids=[item.evidence_id for item in evidence],
            evidence=evidence,
            details={
                "product_id": product_id,
                "affected_lots": affected_lots,
                "normal_lots": list(affected_output.data["normal_lots"]),
                "suspect_lots": list(affected_output.data["suspect_lots"]),
                "passing_suspect_lots": list(affected_output.data["passing_suspect_lots"]),
                "fail_modes": dict(affected_output.data["fail_modes"]),
                "yield_trend": list(affected_output.data["yield_trend"]),
                "target_operation_no": resolved_operation,
                "target_commonality": commonality,
                "operation_commonality": list(genealogy_output.data["operation_commonality"]),
                "lane_candidates": list(
                    genealogy_output.data.get("lane_candidates", [])
                ),
                "hold_count": int(genealogy_output.data["hold_count"]),
            },
            warnings=_warnings(
                affected_output,
                genealogy_output,
                additional=additional_warnings,
            ),
        )

    def analyze_lot(
        self,
        *,
        request_id: str,
        lot_id: str,
        target_operation_no: str | None = None,
    ) -> AgentFinding:
        """Resolve one abnormal Lot and derive the related impact population."""

        if self.get_lot_context_tool is None or self.find_impact_lots_tool is None:
            raise RuntimeError("MESAgent Lot-driven Tools are not configured")

        context_output = self.get_lot_context_tool.run(
            _tool_input(
                tool_name="get_lot_context",
                request_id=f"{request_id}:lot-context",
                parameters={"lot_id": lot_id},
                requested_by=AgentKind.MES.value,
            )
        )
        impact_parameters: dict[str, Any] = {"lot_id": lot_id}
        if target_operation_no:
            impact_parameters["target_operation_no"] = target_operation_no
        impact_output = self.find_impact_lots_tool.run(
            _tool_input(
                tool_name="find_impact_lots",
                request_id=f"{request_id}:impact-lots",
                parameters=impact_parameters,
                requested_by=AgentKind.MES.value,
            )
        )
        affected_lots = list(impact_output.data["affected_lots"])
        impact_lots = list(impact_output.data["impact_lots"])
        affected_wafers = list(impact_output.data["affected_wafers"])
        impact_wafers = list(impact_output.data["impact_wafers"])
        scope_level = str(impact_output.data["scope_level"])
        resolved_operation = str(impact_output.data["target_operation_no"])
        source_exposure = dict(impact_output.data["source_exposure"])
        genealogy_output = self.analyze_lot_genealogy_tool.run(
            _tool_input(
                tool_name="analyze_lot_genealogy",
                request_id=f"{request_id}:genealogy",
                parameters={
                    "lot_ids": affected_lots,
                    "target_operation_no": resolved_operation,
                    "equipment_id": source_exposure["equipment_id"],
                    "chamber_id": source_exposure["chamber_id"],
                },
                requested_by=AgentKind.MES.value,
            )
        )

        commonality = dict(genealogy_output.data["target_commonality"])
        coverage = float(commonality["coverage"])
        wat_failed = bool(context_output.data["wat_failed"])
        additional_warnings: list[Warning] = []
        if coverage < 0.8:
            additional_warnings.append(
                Warning(
                    warning_id="WARN_MES_WEAK_IMPACT_COMMONALITY",
                    message="Impact Lot commonality coverage is below 80%.",
                    evidence_ids=list(genealogy_output.evidence_ids),
                )
            )

        evidence = _typed_evidence(
            context_output,
            impact_output,
            genealogy_output,
        )
        return AgentFinding(
            finding_id=f"{request_id}:mes",
            agent=AgentKind.MES.value,
            summary=(
                f"Lot {lot_id} {'fails' if wat_failed else 'does not fail'} WAT; "
                f"{len(impact_lots)} additional impact Lots and {len(impact_wafers)} impact "
                f"Wafers share operation "
                f"{resolved_operation} on {commonality['equipment_id']}/"
                f"{commonality['chamber_id']} during the excursion window."
            ),
            confidence=round(min(0.99, 0.55 + 0.4 * coverage), 3),
            evidence_ids=[item.evidence_id for item in evidence],
            evidence=evidence,
            details={
                "investigation_mode": "lot",
                "source_lot_id": lot_id,
                "product_id": context_output.data["product_id"],
                "route_id": context_output.data["route_id"],
                "wat_failed": wat_failed,
                "source_fail_modes": list(context_output.data["fail_modes"]),
                "recipe_changes": list(context_output.data["recipe_changes"]),
                "source_lot": dict(context_output.data["lot"]),
                "source_exposure": source_exposure,
                "affected_lots": affected_lots,
                "impact_lots": impact_lots,
                "affected_wafers": affected_wafers,
                "impact_wafers": impact_wafers,
                "scope_level": scope_level,
                "impact_criteria": dict(impact_output.data["impact_criteria"]),
                "normal_lots": [],
                "yield_trend": [],
                "fail_modes": {mode: 1 for mode in context_output.data["fail_modes"]},
                "target_operation_no": resolved_operation,
                "target_commonality": commonality,
                "operation_commonality": list(genealogy_output.data["operation_commonality"]),
                "lane_candidates": list(
                    impact_output.data.get("lane_candidates", [])
                ),
                "hold_count": int(genealogy_output.data["hold_count"]),
            },
            warnings=_warnings(
                context_output,
                impact_output,
                genealogy_output,
                additional=additional_warnings,
            ),
        )


@dataclass(frozen=True)
class FDCAgent:
    """Analyze FDC feature shifts and chamber-level OOC events."""

    analyze_parameter_shift_tool: AnalyzeParameterShiftTool
    find_ooc_events_tool: FindOocEventsTool
    perform_basic_spc_analysis_tool: PerformBasicSpcAnalysisTool
    analyze_spc_evidence_tool: AnalyzeSpcEvidenceTool | None = None

    def analyze(
        self,
        *,
        request_id: str,
        lot_ids: list[str],
        equipment_id: str,
        chamber_id: str,
        operation_no: str = "6400",
        recipe_id: str | None = None,
    ) -> AgentFinding:
        scoped_parameters: dict[str, Any] = {
            "lot_ids": lot_ids,
            "operation_no": operation_no,
            "equipment_id": equipment_id,
            "chamber_id": chamber_id,
        }
        if recipe_id:
            scoped_parameters["recipe_id"] = recipe_id
        parameter_output = self.analyze_parameter_shift_tool.run(
            _tool_input(
                tool_name="analyze_parameter_shift",
                request_id=f"{request_id}:parameter-shift",
                parameters=dict(scoped_parameters),
                requested_by=AgentKind.FDC.value,
            )
        )
        ooc_output = self.find_ooc_events_tool.run(
            _tool_input(
                tool_name="find_ooc_events",
                request_id=f"{request_id}:ooc-events",
                parameters={
                    "operation_no": operation_no,
                    "equipment_id": equipment_id,
                    "chamber_id": chamber_id,
                },
                requested_by=AgentKind.FDC.value,
            )
        )
        spc_parameters = dict(scoped_parameters)
        if self.analyze_spc_evidence_tool is not None:
            advanced_output = self.analyze_spc_evidence_tool.run(
                _tool_input(
                    tool_name="analyze_spc_evidence",
                    request_id=f"{request_id}:advanced-spc",
                    parameters=spc_parameters,
                    requested_by=AgentKind.FDC.value,
                )
            )
        else:
            advanced_output = None
        if (
            advanced_output is not None
            and int(advanced_output.data["analyzed_parameter_count"]) > 0
        ):
            spc_output = advanced_output
        else:
            spc_output = self.perform_basic_spc_analysis_tool.run(
                _tool_input(
                    tool_name="perform_basic_spc_analysis",
                    request_id=f"{request_id}:basic-spc",
                    parameters=spc_parameters,
                    requested_by=AgentKind.FDC.value,
                )
            )

        parameter_summary = list(parameter_output.data["parameter_summary"])
        event_count = int(ooc_output.data["event_count"])
        spc_results = [dict(item) for item in spc_output.data["spc_results"]]
        spc_ooc_parameter_count = int(spc_output.data["ooc_parameter_count"])
        additional_warnings: list[Warning] = []
        if not parameter_summary:
            additional_warnings.append(
                Warning(
                    warning_id="WARN_FDC_NO_FEATURES",
                    message=(
                        "No FDC feature summaries were found for the selected lots and chamber."
                    ),
                )
            )
        if event_count == 0:
            additional_warnings.append(
                Warning(
                    warning_id="WARN_FDC_NO_OOC_EVENTS",
                    message="No OOC events were found for the selected chamber and operation.",
                    evidence_ids=list(ooc_output.evidence_ids),
                )
            )

        shifts = {
            str(item["parameter_name"]): float(item["avg_delta_percent"])
            for item in parameter_summary
        }
        shift_text = ", ".join(f"{name} {delta:+.1f}%" for name, delta in shifts.items())
        max_abs_shift = max((abs(delta) for delta in shifts.values()), default=0.0)
        confidence = 0.5 + min(0.25, max_abs_shift / 100.0)
        if event_count > 0 or spc_ooc_parameter_count > 0:
            confidence += 0.2

        evidence = _typed_evidence(parameter_output, ooc_output, spc_output)
        return AgentFinding(
            finding_id=f"{request_id}:fdc",
            agent=AgentKind.FDC.value,
            summary=(
                f"{equipment_id}/{chamber_id} shows {shift_text or 'no parameter shift data'} "
                f"with {event_count} recorded OOC events and {spc_ooc_parameter_count} "
                f"SPC OOC parameters at operation {operation_no}."
            ),
            confidence=round(min(0.99, confidence), 3),
            evidence_ids=[item.evidence_id for item in evidence],
            evidence=evidence,
            details={
                "lot_ids": list(lot_ids),
                "operation_no": operation_no,
                "equipment_id": equipment_id,
                "chamber_id": chamber_id,
                "recipe_id": recipe_id,
                "parameter_summary": parameter_summary,
                "event_count": event_count,
                "severity_counts": dict(ooc_output.data["severity_counts"]),
                "events": list(ooc_output.data["events"]),
                "spc_ooc_contexts": list(ooc_output.data.get("spc_contexts", [])),
                "spc_method": dict(spc_output.data["method"]),
                "spc_results": spc_results,
                "spc_analyzed_parameter_count": int(spc_output.data["analyzed_parameter_count"]),
                "spc_ooc_parameter_count": spc_ooc_parameter_count,
                "spc_point_violation_count": int(
                    spc_output.data["calculated_point_violation_count"]
                ),
                "spc_baseline_insufficient_parameters": list(
                    spc_output.data["baseline_insufficient_parameters"]
                ),
            },
            warnings=_warnings(
                parameter_output,
                ooc_output,
                spc_output,
                additional=additional_warnings,
            ),
        )


@dataclass(frozen=True)
class DefectWATAgent:
    """Check consistency between physical defects and electrical WAT failures."""

    summarize_defect_wat_tool: SummarizeDefectWatTool

    def analyze(
        self,
        *,
        request_id: str,
        lot_ids: list[str],
        evidence_scope: str = "selected_lots",
    ) -> AgentFinding:
        output = self.summarize_defect_wat_tool.run(
            _tool_input(
                tool_name="summarize_defect_wat",
                request_id=f"{request_id}:defect-wat",
                parameters={"lot_ids": lot_ids, "evidence_scope": evidence_scope},
                requested_by=AgentKind.DEFECT_WAT.value,
            )
        )
        defect_counts = {
            str(key): int(str(value)) for key, value in dict(output.data["defect_counts"]).items()
        }
        defect_patterns = {
            str(key): int(str(value)) for key, value in dict(output.data["defect_patterns"]).items()
        }
        wat_fail_modes = {
            str(key): int(str(value)) for key, value in dict(output.data["wat_fail_modes"]).items()
        }
        wat_fail_count = int(output.data["wat_fail_count"])
        wat_fail_record_count = int(output.data.get("wat_fail_record_count", wat_fail_count))
        missing_wat_lot_ids = [str(item) for item in output.data.get("missing_wat_lot_ids", [])]
        metrology_summaries = [dict(item) for item in output.data["metrology_summaries"]]
        metrology_fail_count = int(output.data["metrology_fail_count"])

        has_defect_signal = bool(defect_counts)
        has_wat_signal = wat_fail_count > 0
        has_metrology_signal = metrology_fail_count > 0
        additional_warnings: list[Warning] = []
        if not has_defect_signal and not has_metrology_signal:
            additional_warnings.append(
                Warning(
                    warning_id="WARN_DEFECT_NO_SIGNAL",
                    message="No defect summary signal was found for the selected lots.",
                    evidence_ids=[],
                )
            )
        if not has_wat_signal and not has_metrology_signal and not missing_wat_lot_ids:
            additional_warnings.append(
                Warning(
                    warning_id="WARN_WAT_NO_FAILURE",
                    message="No WAT failures were found for the selected lots.",
                    evidence_ids=[],
                )
            )

        if has_metrology_signal and (has_defect_signal or has_wat_signal):
            confidence = 0.95
        elif has_metrology_signal:
            confidence = 0.85
        elif has_defect_signal and has_wat_signal:
            confidence = 0.9
        elif has_defect_signal or has_wat_signal:
            confidence = 0.6
        else:
            confidence = 0.2

        dominant_defect = (
            max(defect_counts, key=lambda item: defect_counts[item]) if defect_counts else "none"
        )
        dominant_pattern = (
            max(defect_patterns, key=lambda item: defect_patterns[item])
            if defect_patterns
            else "none"
        )
        dominant_fail_mode = (
            max(wat_fail_modes, key=lambda item: wat_fail_modes[item]) if wat_fail_modes else "none"
        )
        evidence = _typed_evidence(output)
        return AgentFinding(
            finding_id=f"{request_id}:defect-wat",
            agent=AgentKind.DEFECT_WAT.value,
            summary=(
                f"Selected lots show {dominant_defect}/{dominant_pattern} defect evidence and "
                f"{wat_fail_count} WAT-failing Lots led by {dominant_fail_mode}; "
                f"metrology has {metrology_fail_count} out-of-spec Wafer records."
            ),
            confidence=confidence,
            evidence_ids=[item.evidence_id for item in evidence],
            evidence=evidence,
            details={
                "lot_ids": list(lot_ids),
                "evidence_scope": evidence_scope,
                "defect_counts": defect_counts,
                "defect_patterns": defect_patterns,
                "wat_fail_modes": wat_fail_modes,
                "wat_fail_count": wat_fail_count,
                "wat_fail_record_count": wat_fail_record_count,
                "missing_wat_lot_ids": missing_wat_lot_ids,
                "metrology_summaries": metrology_summaries,
                "metrology_fail_count": metrology_fail_count,
                "physical_electrical_consistent": has_defect_signal and has_wat_signal,
                "physical_signal_supported": (
                    has_metrology_signal or (has_defect_signal and has_wat_signal)
                ),
            },
            warnings=_warnings(output, additional=additional_warnings),
        )


@dataclass(frozen=True)
class KnowledgeAgent:
    """Retrieve historical RCA knowledge relevant to current evidence."""

    retrieve_similar_case_tool: RetrieveSimilarCaseTool

    def analyze(
        self,
        *,
        request_id: str,
        query: str,
        module: str = "",
        equipment_type: str = "",
        source_lot_id: str = "",
        product_id: str = "",
        detected_operation: str = "",
        detected_equipment_id: str = "",
        detected_at: str = "",
        symptom_types: tuple[str, ...] = (),
        explicit_module_limit: bool = False,
    ) -> AgentFinding:
        output = self.retrieve_similar_case_tool.run(
            _tool_input(
                tool_name="retrieve_similar_case",
                request_id=f"{request_id}:similar-case",
                parameters={
                    "query": query,
                    "module": module,
                    "equipment_type": equipment_type,
                    "source_lot_id": source_lot_id,
                    "product_id": product_id,
                    "detected_operation": detected_operation,
                    "detected_equipment_id": detected_equipment_id,
                    "detected_at": detected_at,
                    "symptom_types": list(symptom_types),
                    "explicit_module_limit": explicit_module_limit,
                },
                requested_by=AgentKind.KNOWLEDGE.value,
            )
        )
        evidence = _typed_evidence(output)
        raw_top_case = output.data.get("top_case")
        if not isinstance(raw_top_case, dict):
            return AgentFinding(
                finding_id=f"{request_id}:knowledge",
                agent=AgentKind.KNOWLEDGE.value,
                finding_kind=FindingKind.KNOWLEDGE_DISCOVERY.value,
                summary="No engineer-confirmed historical RCA case is available.",
                confidence=0.0,
                evidence_ids=[item.evidence_id for item in evidence],
                evidence=evidence,
                details={
                    "query": output.data["query"],
                    "top_case": {},
                    "cases": [],
                    "documents": [],
                    "retrieval_strategy": output.data.get("retrieval_strategy", "no_match"),
                    "score_components": dict(output.data.get("score_components", {})),
                    "calibrated_relevance": output.data.get("calibrated_relevance"),
                    "source_confidence": output.data.get("source_confidence"),
                    "matched_chunk_ids": list(output.data.get("matched_chunk_ids", [])),
                    "candidate_lanes": list(output.data.get("candidate_lanes", [])),
                    "scope_reasons": list(output.data.get("scope_reasons", [])),
                    "route_distance": output.data.get("route_distance"),
                    "shared_resource_types": list(
                        output.data.get("shared_resource_types", [])
                    ),
                    "scope_fusion_score": output.data.get("scope_fusion_score"),
                    "observation_scope": output.data.get("observation_scope"),
                    "causal_search_scope": output.data.get("causal_search_scope"),
                },
                warnings=_warnings(output),
            )
        top_case = dict(raw_top_case)
        similarity = float(top_case["similarity"])
        additional_warnings: list[Warning] = []
        if similarity < 0.75:
            additional_warnings.append(
                Warning(
                    warning_id="WARN_KNOWLEDGE_WEAK_MATCH",
                    message="The best historical RCA case has similarity below 0.75.",
                    evidence_ids=list(output.evidence_ids),
                )
            )

        return AgentFinding(
            finding_id=f"{request_id}:knowledge",
            agent=AgentKind.KNOWLEDGE.value,
            finding_kind=FindingKind.KNOWLEDGE_DISCOVERY.value,
            summary=(
                f"Historical case {top_case['case_id']} matched at {similarity:.0%}; "
                f"documented root cause: {top_case['root_cause']}."
            ),
            confidence=round(similarity, 3),
            evidence_ids=[item.evidence_id for item in evidence],
            evidence=evidence,
            details={
                "query": output.data["query"],
                "top_case": top_case,
                "cases": list(output.data["cases"]),
                "documents": list(output.data["documents"]),
                "retrieval_strategy": output.data.get("retrieval_strategy", "keyword"),
                "score_components": dict(output.data.get("score_components", {})),
                "calibrated_relevance": output.data.get("calibrated_relevance"),
                "source_confidence": output.data.get("source_confidence"),
                "matched_chunk_ids": list(output.data.get("matched_chunk_ids", [])),
                "candidate_lanes": list(output.data.get("candidate_lanes", [])),
                "scope_reasons": list(output.data.get("scope_reasons", [])),
                "route_distance": output.data.get("route_distance"),
                "shared_resource_types": list(
                    output.data.get("shared_resource_types", [])
                ),
                "scope_fusion_score": output.data.get("scope_fusion_score"),
                "observation_scope": output.data.get("observation_scope"),
                "causal_search_scope": output.data.get("causal_search_scope"),
            },
            warnings=_warnings(output, additional=additional_warnings),
        )

    def validate_preliminary_candidates(
        self,
        *,
        request_id: str,
        preliminary_candidates: list[dict[str, Any]],
        module: str = "",
        equipment_type: str = "",
        source_lot_id: str = "",
        product_id: str = "",
        detected_operation: str = "",
        detected_equipment_id: str = "",
        detected_at: str = "",
        explicit_module_limit: bool = False,
    ) -> AgentFinding:
        candidate_terms = [
            str(item.get("root_cause", "")).strip()
            for item in preliminary_candidates
            if str(item.get("root_cause", "")).strip()
        ]
        basis_terms = [
            str(item.get("basis", "")).strip()
            for item in preliminary_candidates
            if str(item.get("basis", "")).strip()
        ]
        query = (
            " ".join(dict.fromkeys(candidate_terms + basis_terms)).strip()
            or "legacy preliminary RCA candidate validation"
        )
        output = self.retrieve_similar_case_tool.run(
            _tool_input(
                tool_name="retrieve_similar_case",
                request_id=f"{request_id}:candidate-validation",
                parameters={
                    "query": query,
                    "module": module,
                    "equipment_type": equipment_type,
                    "source_lot_id": source_lot_id,
                    "product_id": product_id,
                    "detected_operation": detected_operation,
                    "detected_equipment_id": detected_equipment_id,
                    "detected_at": detected_at,
                    "explicit_module_limit": explicit_module_limit,
                    "match_evidence_id": "EV_KNOWLEDGE_VALIDATION_MATCH",
                    "missing_evidence_id": "EV_KNOWLEDGE_VALIDATION_DATA_MISSING",
                },
                requested_by=AgentKind.KNOWLEDGE.value,
            )
        )
        evidence = _typed_evidence(output)
        raw_top_case = output.data.get("top_case")
        top_case = dict(raw_top_case) if isinstance(raw_top_case, dict) else {}
        similarity = float(top_case.get("similarity", 0.0))
        validation_kind = "supporting" if top_case else "data_missing"
        validation_results = [
            {
                "root_cause": str(item.get("root_cause", "")).strip(),
                "basis": str(item.get("basis", "")).strip(),
                "validation": validation_kind,
                "knowledge_case_id": str(top_case.get("case_id", "")),
                "evidence_ids": [evidence_item.evidence_id for evidence_item in evidence],
            }
            for item in preliminary_candidates
        ]
        summary = (
            f"Confirmed knowledge case {top_case['case_id']} supports legacy preliminary "
            f"candidate validation at {similarity:.0%}."
            if top_case
            else "No confirmed historical knowledge validates the legacy preliminary candidates."
        )
        return AgentFinding(
            finding_id=f"{request_id}:knowledge-validation",
            agent=AgentKind.KNOWLEDGE.value,
            finding_kind=FindingKind.KNOWLEDGE_VALIDATION.value,
            summary=summary,
            confidence=round(similarity, 3),
            evidence_ids=[item.evidence_id for item in evidence],
            evidence=evidence,
            details={
                "preliminary_candidates": preliminary_candidates,
                "validation_results": validation_results,
                "query": output.data["query"],
                "top_case": top_case,
                "cases": list(output.data["cases"]),
                "documents": list(output.data["documents"]),
                "retrieval_strategy": output.data.get("retrieval_strategy", "keyword"),
                "score_components": dict(output.data.get("score_components", {})),
                "calibrated_relevance": output.data.get("calibrated_relevance"),
                "source_confidence": output.data.get("source_confidence"),
                "matched_chunk_ids": list(output.data.get("matched_chunk_ids", [])),
                "candidate_lanes": list(output.data.get("candidate_lanes", [])),
                "scope_reasons": list(output.data.get("scope_reasons", [])),
                "route_distance": output.data.get("route_distance"),
                "shared_resource_types": list(
                    output.data.get("shared_resource_types", [])
                ),
                "scope_fusion_score": output.data.get("scope_fusion_score"),
                "observation_scope": output.data.get("observation_scope"),
                "causal_search_scope": output.data.get("causal_search_scope"),
            },
            warnings=_warnings(output),
        )
