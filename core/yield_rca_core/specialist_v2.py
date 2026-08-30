"""Bounded Qwen-driven Specialist execution for ``llm_react`` orchestration.

The cross-domain Planner authorizes an :class:`InvestigationAction`.  This
module then lets the owning Specialist choose at most two Python-issued Tool
candidates inside that domain.  Qwen never supplies Tool names or parameters:
it can only select a candidate identifier or finish the local investigation.

Tool outputs remain the sole source of Evidence and Python builds the final
``AgentFinding``.  The second LLM pass may summarize and interpret that
Finding, but its Evidence IDs must be an exact ordered closure of the observed
Tool Evidence and its confidence cannot exceed the deterministic value.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from yield_rca_core.investigation_models import (
    ActionKind,
    InvestigationAction,
    InvestigationIntent,
)
from yield_rca_core.llm_gateway import LLMClient, LLMRequest
from yield_rca_core.models import (
    AgentFinding,
    AgentKind,
    AgentMode,
    Evidence,
    FindingKind,
    ToolInput,
    ToolOutput,
    Warning,
)
from yield_rca_core.specialist_models import (
    MAX_SPECIALIST_TOOL_STEPS,
    SpecialistAnalysis,
    SpecialistDecisionType,
    SpecialistStepRecord,
    SpecialistToolCandidate,
    SpecialistToolDecision,
)

SPECIALIST_V2_VERSION = "v2"
_OUTPUT_ATTEMPTS = 2

_ACTION_AGENTS: dict[str, str] = {
    ActionKind.INSPECT_DEFECT_PATTERN.value: AgentKind.DEFECT_WAT.value,
    ActionKind.VALIDATE_SHARED_DEFECT_PATTERN.value: AgentKind.DEFECT_WAT.value,
    ActionKind.FIND_SHARED_EXPOSURE.value: AgentKind.MES.value,
    ActionKind.ASSESS_IMPACT_SCOPE.value: AgentKind.MES.value,
    ActionKind.INSPECT_RECIPE_CHANGE.value: AgentKind.MES.value,
    ActionKind.INSPECT_FDC_SPC.value: AgentKind.FDC.value,
    ActionKind.VALIDATE_HISTORICAL_CASE.value: AgentKind.KNOWLEDGE.value,
}

_DOMAIN_TOOLS: dict[str, frozenset[str]] = {
    AgentKind.MES.value: frozenset(
        {
            "find_affected_lots",
            "get_lot_context",
            "find_impact_lots",
            "analyze_lot_genealogy",
        }
    ),
    AgentKind.FDC.value: frozenset(
        {
            "analyze_parameter_shift",
            "find_ooc_events",
            "perform_basic_spc_analysis",
            "analyze_spc_evidence",
        }
    ),
    AgentKind.DEFECT_WAT.value: frozenset({"summarize_defect_wat"}),
    AgentKind.KNOWLEDGE.value: frozenset({"retrieve_similar_case"}),
}


class _Tool(Protocol):
    def run(self, tool_input: ToolInput) -> ToolOutput: ...


class SpecialistV2Error(RuntimeError):
    """Typed failure at the bounded Specialist execution boundary."""

    def __init__(self, message: str, *, stage: str, reason: str) -> None:
        super().__init__(message)
        self.stage = stage
        self.reason = reason


@dataclass(frozen=True)
class _ExecutedTool:
    candidate: SpecialistToolCandidate
    decision: SpecialistToolDecision
    output: ToolOutput
    record: SpecialistStepRecord


def _string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _normalized_lot_ids(value: object) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    return list(
        dict.fromkeys(
            item.strip().upper()
            for item in value
            if isinstance(item, str) and item.strip()
        )
    )


def _require_source_lot_in_scope(
    context: Mapping[str, Any],
    lot_ids: list[str],
    *,
    agent: str,
) -> None:
    source_lot_id = _string(context.get("source_lot_id")).upper()
    if source_lot_id and source_lot_id not in lot_ids:
        raise SpecialistV2Error(
            (
                f"{agent} Specialist Lot scope excludes the fixed source Lot; "
                "an impact Lot cannot become the new investigation target"
            ),
            stage="candidate_generation",
            reason="source_lot_scope_violation",
        )


def _output_data(output: ToolOutput) -> dict[str, Any]:
    """Return model-facing Tool data without duplicating the Evidence envelope."""

    return {key: value for key, value in output.data.items() if key != "evidence"}


def _compact_evidence(evidence: Evidence) -> dict[str, Any]:
    """Keep model analysis grounded without repeating full entity envelopes."""

    return {
        "evidence_id": evidence.evidence_id,
        "evidence_type": evidence.evidence_type,
        "source_type": evidence.source_type,
        "summary": evidence.summary,
        "observation": evidence.observation,
        "confidence": evidence.confidence,
    }


def _model_tool_observation(item: _ExecutedTool) -> dict[str, Any]:
    """Bound the model view while RCAState retains the complete ToolOutput."""

    return {
        "tool_name": item.output.tool_name,
        "output_summary": item.record.output_summary,
        "evidence_ids": list(item.output.evidence_ids),
        "evidence": [_compact_evidence(evidence) for evidence in item.output.evidence],
    }


def _merge_evidence(outputs: list[ToolOutput]) -> list[Evidence]:
    evidence_by_id: dict[str, Evidence] = {}
    for output in outputs:
        for item in output.evidence:
            evidence_by_id.setdefault(item.evidence_id, item)
    return list(evidence_by_id.values())


def _merge_warnings(outputs: list[ToolOutput], evidence_ids: set[str]) -> list[Warning]:
    warnings_by_id: dict[str, Warning] = {}
    for output in outputs:
        for warning in output.warnings:
            bounded_ids = [
                evidence_id
                for evidence_id in warning.evidence_ids
                if evidence_id in evidence_ids
            ]
            warnings_by_id[warning.warning_id] = Warning(
                warning_id=warning.warning_id,
                message=warning.message,
                severity=warning.severity,
                evidence_ids=bounded_ids,
                schema_version=warning.schema_version,
            )
    return list(warnings_by_id.values())


def _candidate_id(action: InvestigationAction, tool_name: str) -> str:
    return f"{action.action_id}:candidate:{tool_name}"


def _candidate(
    action: InvestigationAction,
    tool_name: str,
    parameters: dict[str, Any],
    purpose: str,
) -> SpecialistToolCandidate:
    return SpecialistToolCandidate(
        candidate_id=_candidate_id(action, tool_name),
        tool_name=tool_name,
        parameters=parameters,
        purpose=purpose,
    )


def _target_commonality_from_impact(data: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the MES commonality contract from impact-scope exposure data."""

    raw_exposure = data.get("source_exposure")
    exposure = raw_exposure if isinstance(raw_exposure, Mapping) else {}
    affected_lots = _normalized_lot_ids(data.get("affected_lots"))
    affected_wafers = _normalized_lot_ids(data.get("affected_wafers"))
    return {
        "equipment_id": _string(exposure.get("equipment_id")),
        "chamber_id": _string(exposure.get("chamber_id")),
        "recipe_id": _string(exposure.get("recipe_id")),
        "lot_count": len(affected_lots),
        "wafer_count": len(affected_wafers),
        "coverage": 1.0 if affected_lots else 0.0,
        "wafer_coverage": 1.0 if affected_wafers else 0.0,
        "derivation": "find_impact_lots.source_exposure",
    }


def _operation_commonality_from_impact(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_exposure = data.get("source_exposure")
    exposure = raw_exposure if isinstance(raw_exposure, Mapping) else {}
    commonality = _target_commonality_from_impact(data)
    operation_no = _string(data.get("target_operation_no"))
    if not operation_no:
        return []
    return [
        {
            "operation_no": operation_no,
            "operation_name": _string(exposure.get("operation_name")),
            "module": _string(exposure.get("module")),
            "equipment_id": commonality["equipment_id"],
            "chamber_id": commonality["chamber_id"],
            "recipe_id": commonality["recipe_id"],
            "lot_count": commonality["lot_count"],
            "wafer_count": commonality["wafer_count"],
            "coverage": commonality["coverage"],
            "wafer_coverage": commonality["wafer_coverage"],
        }
    ]


def _tool_output_summary(output: ToolOutput) -> str:
    data = output.data
    if output.tool_name == "get_lot_context":
        return (
            f"Resolved Lot {data.get('lot_id', '')} as product "
            f"{data.get('product_id', '')}; WAT failed={bool(data.get('wat_failed', False))}."
        )
    if output.tool_name == "find_impact_lots":
        return (
            f"Found {len(data.get('impact_lots', []))} impact Lots at operation "
            f"{data.get('target_operation_no', '')}."
        )
    if output.tool_name == "find_affected_lots":
        return f"Found {len(data.get('affected_lots', []))} affected Lots."
    if output.tool_name == "analyze_lot_genealogy":
        commonality = data.get("target_commonality", {})
        coverage = commonality.get("coverage", 0.0) if isinstance(commonality, dict) else 0.0
        return (
            f"MES genealogy commonality coverage is {float(coverage):.0%} at operation "
            f"{data.get('target_operation_no', '')}."
        )
    if output.tool_name == "analyze_parameter_shift":
        return f"Summarized {len(data.get('parameter_summary', []))} FDC parameters."
    if output.tool_name == "find_ooc_events":
        return f"Found {int(data.get('event_count', 0))} chamber OOC events."
    if output.tool_name in {"perform_basic_spc_analysis", "analyze_spc_evidence"}:
        return (
            f"SPC analyzed {int(data.get('analyzed_parameter_count', 0))} parameters "
            f"and found {int(data.get('ooc_parameter_count', 0))} OOC parameters."
        )
    if output.tool_name == "summarize_defect_wat":
        return (
            f"Quality evidence contains {len(data.get('defect_counts', {}))} defect "
            f"types, {int(data.get('wat_fail_count', 0))} WAT-failing Lots, and "
            f"{int(data.get('metrology_fail_count', 0))} metrology failures."
        )
    if output.tool_name == "retrieve_similar_case":
        raw_top_case = data.get("top_case")
        top_case = raw_top_case if isinstance(raw_top_case, dict) else {}
        if top_case:
            return (
                f"Retrieved historical case {top_case.get('case_id', '')} with "
                f"similarity {float(top_case.get('similarity', 0.0)):.0%}."
            )
        return "No engineer-confirmed historical case matched the query."
    return f"{output.tool_name} completed with {len(output.evidence_ids)} Evidence items."


@dataclass(frozen=True)
class SpecialistV2Executor:
    """Execute one authorized Specialist action with a two-Tool hard limit."""

    llm_client: LLMClient
    find_affected_lots_tool: _Tool | None = None
    get_lot_context_tool: _Tool | None = None
    find_impact_lots_tool: _Tool | None = None
    analyze_lot_genealogy_tool: _Tool | None = None
    analyze_parameter_shift_tool: _Tool | None = None
    find_ooc_events_tool: _Tool | None = None
    perform_basic_spc_analysis_tool: _Tool | None = None
    analyze_spc_evidence_tool: _Tool | None = None
    summarize_defect_wat_tool: _Tool | None = None
    retrieve_similar_case_tool: _Tool | None = None
    agent_mode: str = AgentMode.LLM.value
    tool_prompt_version: str = "v1"
    analysis_prompt_version: str = "v2"
    direct_single_candidate: bool = False
    llm_analysis_enabled: bool = True

    def execute(
        self,
        action: InvestigationAction,
        *,
        request_id: str,
        context: dict[str, Any],
        max_tool_calls: int = MAX_SPECIALIST_TOOL_STEPS,
    ) -> AgentFinding:
        """Run a bounded local observation/action loop and return one Finding."""

        if type(max_tool_calls) is not int or max_tool_calls <= 0:
            raise SpecialistV2Error(
                "Specialist V2 requires a positive local Tool budget",
                stage="budget",
                reason="tool_budget_exhausted",
            )
        budget = min(MAX_SPECIALIST_TOOL_STEPS, max_tool_calls)
        self._validate_action(action)
        if not isinstance(context, dict):
            raise SpecialistV2Error(
                "Specialist V2 context must be a JSON object",
                stage="candidate_generation",
                reason="invalid_context",
            )

        tools = self._tools()
        executed: list[_ExecutedTool] = []
        effective: list[_ExecutedTool] = []
        executed_candidate_ids: set[str] = set()
        superseded_step_ids: list[str] = []
        validation_error_count = 0
        fallback_reasons: list[str] = []
        direct_selection_count = 0
        stop_reason = "tool_budget_exhausted"

        while len(executed) < budget:
            candidates = self._candidates(
                action,
                context=context,
                effective=effective,
                executed_candidate_ids=executed_candidate_ids,
                remaining_budget=budget - len(executed),
                tools=tools,
            )
            if not candidates:
                stop_reason = (
                    "required_local_evidence_collected"
                    if effective
                    else "no_configured_tool_candidate"
                )
                break

            if self.direct_single_candidate and len(candidates) == 1:
                decision = self._deterministic_decision(
                    action,
                    context=context,
                    candidates=candidates,
                    effective=effective,
                )
                errors = 0
                used_fallback = False
                direct_selection_count += 1
            else:
                decision, errors, used_fallback = self._choose_tool(
                    action,
                    context=context,
                    candidates=candidates,
                    effective=effective,
                    remaining_budget=budget - len(executed),
                )
            validation_error_count += errors
            if used_fallback:
                fallback_reasons.append("tool_selection_output_invalid")
            if decision.decision_type == SpecialistDecisionType.FINISH.value:
                stop_reason = decision.stop_reason or "specialist_finished"
                break

            candidate = next(
                item for item in candidates if item.candidate_id == decision.candidate_id
            )
            result = self._execute_candidate(
                action,
                request_id=request_id,
                decision=decision,
                candidate=candidate,
                step_index=len(executed) + 1,
                tools=tools,
            )
            executed.append(result)
            effective.append(result)
            executed_candidate_ids.add(candidate.candidate_id)

            if (
                candidate.tool_name == "analyze_spc_evidence"
                and int(result.output.data.get("analyzed_parameter_count", 0)) == 0
            ):
                basic_candidate = self._fdc_basic_candidate(action, context)
                basic_tool = tools.get("perform_basic_spc_analysis")
                if basic_tool is None or len(executed) >= budget:
                    raise SpecialistV2Error(
                        "Advanced SPC returned no analyzable parameters but Basic SPC "
                        "fallback is unavailable inside the two-Tool budget",
                        stage="tool_execution",
                        reason="advanced_spc_fallback_unavailable",
                    )
                fallback_decision = SpecialistToolDecision(
                    decision_id=f"{action.action_id}:advanced-spc-fallback",
                    action_id=action.action_id,
                    agent=action.agent,
                    decision_type=SpecialistDecisionType.CALL_TOOL.value,
                    reason=(
                        "Advanced SPC had no analyzable parameters; use the pre-bound "
                        "Basic SPC candidate as the bounded local fallback."
                    ),
                    candidate_id=basic_candidate.candidate_id,
                    stop_reason=None,
                )
                fallback_result = self._execute_candidate(
                    action,
                    request_id=request_id,
                    decision=fallback_decision,
                    candidate=basic_candidate,
                    step_index=len(executed) + 1,
                    tools=tools,
                )
                executed.append(fallback_result)
                effective = [
                    item
                    for item in effective
                    if item.candidate.tool_name != "analyze_spc_evidence"
                ]
                effective.append(fallback_result)
                executed_candidate_ids.add(basic_candidate.candidate_id)
                superseded_step_ids.append(result.record.step_id)
                fallback_reasons.append("advanced_spc_no_analyzable_parameters")
                stop_reason = "advanced_spc_fell_back_to_basic"
                break
        else:
            stop_reason = "tool_budget_exhausted"

        if not effective:
            raise SpecialistV2Error(
                "Specialist V2 produced no Tool Evidence",
                stage="finding_assembly",
                reason="no_tool_evidence",
            )

        deterministic_finding = self._assemble_finding(
            action,
            request_id=request_id,
            context=context,
            outputs=[item.output for item in effective],
        )
        deterministic_analysis = SpecialistAnalysis(
            summary=deterministic_finding.summary,
            confidence=deterministic_finding.confidence,
            evidence_ids=list(deterministic_finding.evidence_ids),
            engineering_interpretation=deterministic_finding.summary,
        )
        if self.llm_analysis_enabled:
            analysis, analysis_errors, analysis_fallback = self._analyze(
                action,
                context=context,
                executed=executed,
                effective=effective,
                deterministic_analysis=deterministic_analysis,
            )
        else:
            # Tool outputs already contain immutable typed Evidence and Python
            # has assembled the only admissible Finding.  A second LLM pass can
            # only paraphrase that result, so production llm_react skips it to
            # avoid duplicate cost, timeout risk, and an extra failure surface.
            analysis = deterministic_analysis
            analysis_errors = 0
            analysis_fallback = False
        validation_error_count += analysis_errors
        if analysis_fallback:
            fallback_reasons.append("analysis_output_invalid")

        bounded_confidence = min(
            deterministic_finding.confidence,
            analysis.confidence,
        )
        fallback_reason = (
            ",".join(dict.fromkeys(fallback_reasons)) if fallback_reasons else None
        )
        return AgentFinding(
            finding_id=deterministic_finding.finding_id,
            task_id=deterministic_finding.task_id,
            agent=deterministic_finding.agent,
            finding_kind=deterministic_finding.finding_kind,
            summary=analysis.summary,
            confidence=bounded_confidence,
            evidence_ids=list(deterministic_finding.evidence_ids),
            evidence=list(deterministic_finding.evidence),
            details={
                **deterministic_finding.details,
                "agent_mode": self.agent_mode,
                "llm_prompt_version": self.analysis_prompt_version,
                "engineering_interpretation": analysis.engineering_interpretation,
                "specialist_v2": {
                    "version": SPECIALIST_V2_VERSION,
                    "action_id": action.action_id,
                    "agent": action.agent,
                    "tool_steps": [item.record.to_dict() for item in executed],
                    "tool_call_count": len(executed),
                    "direct_single_candidate_selection_count": (
                        direct_selection_count
                    ),
                    "stop_reason": stop_reason,
                    "analysis_source": (
                        "deterministic_typed_evidence"
                        if not self.llm_analysis_enabled
                        else (
                            "deterministic_fallback"
                            if analysis_fallback
                            else "qwen"
                        )
                    ),
                    "fallback_reason": fallback_reason,
                    "validation_retry_count": validation_error_count,
                    "superseded_step_ids": superseded_step_ids,
                },
            },
            warnings=list(deterministic_finding.warnings),
        )

    def _validate_action(self, action: InvestigationAction) -> None:
        if not isinstance(action, InvestigationAction):
            raise SpecialistV2Error(
                "Specialist V2 requires an InvestigationAction",
                stage="dispatch",
                reason="invalid_action",
            )
        expected_agent = _ACTION_AGENTS.get(action.kind)
        if expected_agent is None:
            raise SpecialistV2Error(
                f"Action {action.kind!r} is not a Specialist V2 action",
                stage="dispatch",
                reason="unsupported_action",
            )
        if action.agent != expected_agent:
            raise SpecialistV2Error(
                f"Action {action.kind!r} cannot execute in domain {action.agent!r}",
                stage="dispatch",
                reason="agent_domain_mismatch",
            )

    def _tools(self) -> dict[str, _Tool]:
        configured = {
            "find_affected_lots": self.find_affected_lots_tool,
            "get_lot_context": self.get_lot_context_tool,
            "find_impact_lots": self.find_impact_lots_tool,
            "analyze_lot_genealogy": self.analyze_lot_genealogy_tool,
            "analyze_parameter_shift": self.analyze_parameter_shift_tool,
            "find_ooc_events": self.find_ooc_events_tool,
            "perform_basic_spc_analysis": self.perform_basic_spc_analysis_tool,
            "analyze_spc_evidence": self.analyze_spc_evidence_tool,
            "summarize_defect_wat": self.summarize_defect_wat_tool,
            "retrieve_similar_case": self.retrieve_similar_case_tool,
        }
        return {name: tool for name, tool in configured.items() if tool is not None}

    def _candidates(
        self,
        action: InvestigationAction,
        *,
        context: dict[str, Any],
        effective: list[_ExecutedTool],
        executed_candidate_ids: set[str],
        remaining_budget: int,
        tools: dict[str, _Tool],
    ) -> list[SpecialistToolCandidate]:
        if action.agent == AgentKind.MES.value:
            candidates = self._mes_candidates(action, context, effective)
        elif action.agent == AgentKind.FDC.value:
            candidates = self._fdc_candidates(
                action,
                context,
                effective,
                remaining_budget=remaining_budget,
            )
        elif action.agent == AgentKind.DEFECT_WAT.value:
            candidates = [self._defect_candidate(action, context)]
        elif action.agent == AgentKind.KNOWLEDGE.value:
            candidates = [self._knowledge_candidate(action, context)]
        else:
            candidates = []

        allowed = _DOMAIN_TOOLS[action.agent]
        return [
            candidate
            for candidate in candidates
            if candidate.candidate_id not in executed_candidate_ids
            and candidate.tool_name in allowed
            and candidate.tool_name in tools
        ]

    def _mes_candidates(
        self,
        action: InvestigationAction,
        context: dict[str, Any],
        effective: list[_ExecutedTool],
    ) -> list[SpecialistToolCandidate]:
        lot_id = _string(context.get("lot_id")) or _string(context.get("source_lot_id"))
        product_id = _string(context.get("product_id"))
        executed_names = {item.candidate.tool_name for item in effective}
        if lot_id:
            normalized_lot_id = lot_id.upper()
            if "get_lot_context" not in executed_names:
                return [
                    _candidate(
                        action,
                        "get_lot_context",
                        {"lot_id": normalized_lot_id},
                        "Resolve the fixed source Lot manufacturing and anomaly context.",
                    )
                ]
            if "find_impact_lots" not in executed_names:
                parameters: dict[str, Any] = {"lot_id": normalized_lot_id}
                operation_no = _string(context.get("operation_no")) or _string(
                    context.get("target_operation_no")
                )
                if operation_no:
                    parameters["target_operation_no"] = operation_no
                for context_key, parameter_key in (
                    ("equipment_id", "equipment_id"),
                    ("chamber_id", "chamber_id"),
                    ("recipe_id", "recipe_id"),
                ):
                    value = _string(context.get(context_key))
                    if value:
                        parameters[parameter_key] = value
                return [
                    _candidate(
                        action,
                        "find_impact_lots",
                        parameters,
                        (
                            "Find Lots exposed to the source Lot's fixed operation, "
                            "equipment, chamber, and excursion window."
                        ),
                    )
                ]
            return []

        if not product_id:
            raise SpecialistV2Error(
                "MES Specialist requires a trusted source lot_id or product_id",
                stage="candidate_generation",
                reason="missing_mes_scope",
            )
        affected = next(
            (
                item.output
                for item in effective
                if item.candidate.tool_name == "find_affected_lots"
            ),
            None,
        )
        if affected is None:
            start_date, end_date = self._time_window(context)
            return [
                _candidate(
                    action,
                    "find_affected_lots",
                    {
                        "product_id": product_id,
                        "start_date": start_date,
                        "end_date": end_date,
                    },
                    "Resolve the affected Lot population inside the fixed product window.",
                )
            ]
        lot_ids = _normalized_lot_ids(affected.data.get("affected_lots"))
        if not lot_ids or "analyze_lot_genealogy" in executed_names:
            return []
        parameters = {"lot_ids": lot_ids}
        operation_no = _string(context.get("operation_no")) or _string(
            context.get("target_operation_no")
        )
        if operation_no:
            parameters["target_operation_no"] = operation_no
        for context_key, parameter_key in (
            ("equipment_id", "equipment_id"),
            ("chamber_id", "chamber_id"),
            ("recipe_id", "recipe_id"),
        ):
            value = _string(context.get(context_key))
            if value:
                parameters[parameter_key] = value
        return [
            _candidate(
                action,
                "analyze_lot_genealogy",
                parameters,
                "Measure process commonality only across the Tool-selected affected Lots.",
            )
        ]

    def _fdc_parameters(self, context: dict[str, Any]) -> dict[str, Any]:
        lot_ids = _normalized_lot_ids(
            context.get("lot_ids", context.get("affected_lots", []))
        )
        operation_no = _string(context.get("operation_no")) or _string(
            context.get("target_operation_no")
        )
        equipment_id = _string(context.get("equipment_id"))
        chamber_id = _string(context.get("chamber_id"))
        recipe_id = _string(context.get("recipe_id")) or _string(
            context.get("recipe")
        )
        if not lot_ids or not operation_no or not equipment_id or not chamber_id:
            raise SpecialistV2Error(
                "FDC Specialist requires trusted Lot, operation, equipment, and chamber scope",
                stage="candidate_generation",
                reason="missing_fdc_scope",
            )
        _require_source_lot_in_scope(context, lot_ids, agent=AgentKind.FDC.value)
        parameters = {
            "lot_ids": lot_ids,
            "operation_no": operation_no,
            "equipment_id": equipment_id,
            "chamber_id": chamber_id,
        }
        if recipe_id:
            parameters["recipe_id"] = recipe_id
        return parameters

    def _fdc_basic_candidate(
        self,
        action: InvestigationAction,
        context: dict[str, Any],
    ) -> SpecialistToolCandidate:
        return _candidate(
            action,
            "perform_basic_spc_analysis",
            self._fdc_parameters(context),
            "Calculate deterministic minimal SPC limits for the fixed process scope.",
        )

    def _fdc_candidates(
        self,
        action: InvestigationAction,
        context: dict[str, Any],
        effective: list[_ExecutedTool],
        *,
        remaining_budget: int,
    ) -> list[SpecialistToolCandidate]:
        parameters = self._fdc_parameters(context)
        executed_names = {item.candidate.tool_name for item in effective}
        has_spc = bool(
            executed_names & {"perform_basic_spc_analysis", "analyze_spc_evidence"}
        )
        candidates: list[SpecialistToolCandidate] = []
        if "analyze_parameter_shift" not in executed_names:
            candidates.append(
                _candidate(
                    action,
                    "analyze_parameter_shift",
                    parameters,
                    "Quantify parameter shifts for the fixed affected-Lot chamber scope.",
                )
            )
        if "find_ooc_events" not in executed_names:
            candidates.append(
                _candidate(
                    action,
                    "find_ooc_events",
                    {
                        "operation_no": parameters["operation_no"],
                        "equipment_id": parameters["equipment_id"],
                        "chamber_id": parameters["chamber_id"],
                        **(
                            {"recipe_id": parameters["recipe_id"]}
                            if "recipe_id" in parameters
                            else {}
                        ),
                    },
                    "Inspect recorded OOC and containment events for the fixed chamber.",
                )
            )
        if not has_spc:
            candidates.append(self._fdc_basic_candidate(action, context))
            # Advanced SPC is offered only when its required Basic fallback can
            # still fit inside the same hard two-call budget.
            if remaining_budget >= 2 and self.perform_basic_spc_analysis_tool is not None:
                candidates.append(
                    _candidate(
                        action,
                        "analyze_spc_evidence",
                        parameters,
                        (
                            "Evaluate versioned advanced SPC baselines for the exact "
                            "product and process context."
                        ),
                    )
                )
        return candidates

    def _defect_candidate(
        self,
        action: InvestigationAction,
        context: dict[str, Any],
    ) -> SpecialistToolCandidate:
        lot_ids = _normalized_lot_ids(context.get("lot_ids", []))
        if not lot_ids:
            lot_id = _string(context.get("lot_id")) or _string(
                context.get("source_lot_id")
            )
            lot_ids = [lot_id.upper()] if lot_id else []
        if not lot_ids:
            raise SpecialistV2Error(
                "Defect/WAT Specialist requires a trusted non-empty Lot scope",
                stage="candidate_generation",
                reason="missing_defect_scope",
            )
        _require_source_lot_in_scope(
            context,
            lot_ids,
            agent=AgentKind.DEFECT_WAT.value,
        )
        evidence_scope = _string(context.get("evidence_scope")) or (
            "shared_exposure_comparison"
            if action.kind == ActionKind.VALIDATE_SHARED_DEFECT_PATTERN.value
            else "selected_lots"
        )
        return _candidate(
            action,
            "summarize_defect_wat",
            {"lot_ids": lot_ids, "evidence_scope": evidence_scope},
            "Compare physical defect, metrology, and electrical evidence in the fixed Lot scope.",
        )

    def _knowledge_candidate(
        self,
        action: InvestigationAction,
        context: dict[str, Any],
    ) -> SpecialistToolCandidate:
        query = _string(context.get("query"))
        if not query:
            raise SpecialistV2Error(
                "Knowledge Specialist requires a trusted retrieval query",
                stage="candidate_generation",
                reason="missing_knowledge_query",
            )
        return _candidate(
            action,
            "retrieve_similar_case",
            {
                "query": query,
                "module": _string(context.get("module")),
                "equipment_type": _string(context.get("equipment_type")),
                "source_lot_id": _string(context.get("source_lot_id")),
                "product_id": _string(context.get("product_id")),
                "detected_operation": _string(context.get("detected_operation")),
                "detected_equipment_id": _string(
                    context.get("detected_equipment_id")
                ),
                "detected_at": _string(context.get("detected_at")),
                "symptom_types": list(context.get("symptom_types", [])),
                "explicit_module_limit": bool(
                    context.get("explicit_module_limit", False)
                ),
            },
            "Retrieve engineer-confirmed historical cases for the fixed engineering query.",
        )

    @staticmethod
    def _time_window(context: dict[str, Any]) -> tuple[str | None, str | None]:
        raw_window = context.get("time_window")
        window = raw_window if isinstance(raw_window, dict) else {}
        start = _string(context.get("start_date")) or _string(window.get("start")) or _string(
            window.get("start_date")
        )
        end = _string(context.get("end_date")) or _string(window.get("end")) or _string(
            window.get("end_date")
        )
        return start or None, end or None

    def _choose_tool(
        self,
        action: InvestigationAction,
        *,
        context: dict[str, Any],
        candidates: list[SpecialistToolCandidate],
        effective: list[_ExecutedTool],
        remaining_budget: int,
    ) -> tuple[SpecialistToolDecision, int, bool]:
        deterministic = self._deterministic_decision(
            action,
            context=context,
            candidates=candidates,
            effective=effective,
        )
        validation_errors: list[str] = []
        payload = {
            "action_id": action.action_id,
            "agent": action.agent,
            "action": action.to_dict(),
            "engineering_goal": action.reason,
            "trusted_context": dict(context),
            "tool_candidates": [item.to_dict() for item in candidates],
            "completed_steps": [item.record.to_dict() for item in effective],
            "tool_observations": [
                _model_tool_observation(item) for item in effective
            ],
            "remaining_tool_calls": remaining_budget,
            "max_tool_calls": MAX_SPECIALIST_TOOL_STEPS,
            "deterministic_specialist_decision": deterministic.to_dict(),
            "validation_errors": validation_errors,
        }
        error_count = 0
        for attempt in range(_OUTPUT_ATTEMPTS):
            try:
                response = self.llm_client.complete_json(
                    LLMRequest(
                        agent=action.agent,
                        prompt_name="specialist_tool_planner",
                        prompt_version=self.tool_prompt_version,
                        payload=payload,
                    )
                )
                decision = SpecialistToolDecision.from_dict(response.data)
                self._validate_decision(
                    decision,
                    action=action,
                    candidates=candidates,
                    effective=effective,
                )
                return decision, error_count, False
            except Exception as exc:
                error_count += 1
                validation_errors.append(
                    self._validation_error(exc, attempt=attempt + 1)
                )
        return deterministic, error_count, True

    def _deterministic_decision(
        self,
        action: InvestigationAction,
        *,
        context: dict[str, Any],
        candidates: list[SpecialistToolCandidate],
        effective: list[_ExecutedTool],
    ) -> SpecialistToolDecision:
        by_name = {item.tool_name: item for item in candidates}
        preferred_names: list[str]
        if action.agent == AgentKind.FDC.value:
            intent = _string(context.get("investigation_intent"))
            if intent == InvestigationIntent.SPC_CHECK.value:
                preferred_names = [
                    "analyze_spc_evidence",
                    "perform_basic_spc_analysis",
                    "analyze_parameter_shift",
                    "find_ooc_events",
                ]
            else:
                preferred_names = [
                    "analyze_parameter_shift",
                    "perform_basic_spc_analysis",
                    "find_ooc_events",
                    "analyze_spc_evidence",
                ]
        else:
            preferred_names = [
                "get_lot_context",
                "find_impact_lots",
                "find_affected_lots",
                "analyze_lot_genealogy",
                "summarize_defect_wat",
                "retrieve_similar_case",
            ]
        selected = next(
            (by_name[name] for name in preferred_names if name in by_name),
            candidates[0],
        )
        return SpecialistToolDecision(
            decision_id=(
                f"{action.action_id}:specialist-decision:{len(effective) + 1}:deterministic"
            ),
            action_id=action.action_id,
            agent=action.agent,
            decision_type=SpecialistDecisionType.CALL_TOOL.value,
            reason=f"Use the bounded deterministic candidate: {selected.purpose}",
            candidate_id=selected.candidate_id,
            stop_reason=None,
        )

    @staticmethod
    def _validate_decision(
        decision: SpecialistToolDecision,
        *,
        action: InvestigationAction,
        candidates: list[SpecialistToolCandidate],
        effective: list[_ExecutedTool],
    ) -> None:
        if decision.action_id != action.action_id or decision.agent != action.agent:
            raise ValueError("Specialist decision changed the authorized action or agent")
        if decision.decision_type == SpecialistDecisionType.FINISH.value:
            if not SpecialistV2Executor._can_finish(action, effective):
                raise ValueError("Specialist cannot finish before observing Tool Evidence")
            return
        candidate_ids = {item.candidate_id for item in candidates}
        if decision.candidate_id not in candidate_ids:
            raise ValueError("Specialist selected an unknown, executed, or cross-domain candidate")

    @staticmethod
    def _can_finish(
        action: InvestigationAction,
        effective: list[_ExecutedTool],
    ) -> bool:
        if not effective:
            return False
        if action.agent != AgentKind.MES.value:
            return True
        by_name = {item.candidate.tool_name: item.output for item in effective}
        if "get_lot_context" in by_name:
            return "find_impact_lots" in by_name
        affected = by_name.get("find_affected_lots")
        if affected is None:
            return False
        affected_lots = _normalized_lot_ids(affected.data.get("affected_lots"))
        return not affected_lots or "analyze_lot_genealogy" in by_name

    @staticmethod
    def _validation_error(exc: Exception, *, attempt: int) -> str:
        message = str(exc).strip()
        if not message:
            message = type(exc).__name__
        return f"attempt {attempt}: {message[:500]}"

    def _execute_candidate(
        self,
        action: InvestigationAction,
        *,
        request_id: str,
        decision: SpecialistToolDecision,
        candidate: SpecialistToolCandidate,
        step_index: int,
        tools: dict[str, _Tool],
    ) -> _ExecutedTool:
        if candidate.tool_name not in _DOMAIN_TOOLS[action.agent]:
            raise SpecialistV2Error(
                "Specialist candidate crosses its Tool domain",
                stage="tool_execution",
                reason="cross_domain_tool",
            )
        tool = tools.get(candidate.tool_name)
        if tool is None:
            raise SpecialistV2Error(
                f"Tool {candidate.tool_name!r} is not configured",
                stage="tool_execution",
                reason="tool_not_configured",
            )
        parameters = dict(candidate.parameters)
        lane_id = action.scope.get("lane_id")
        if isinstance(lane_id, str) and lane_id.strip():
            parameters.setdefault("lane_id", lane_id.strip())
        tool_input = ToolInput(
            tool_name=candidate.tool_name,
            request_id=f"{request_id}:specialist-step-{step_index}",
            parameters=parameters,
            requested_by=action.agent,
        )
        try:
            output = tool.run(tool_input)
        except Exception as exc:
            raise SpecialistV2Error(
                f"Specialist Tool {candidate.tool_name!r} failed",
                stage="tool_execution",
                reason="tool_call_failed",
            ) from exc
        if not output.success or output.tool_name != candidate.tool_name:
            raise SpecialistV2Error(
                f"Specialist Tool {candidate.tool_name!r} returned an invalid output",
                stage="tool_execution",
                reason="invalid_tool_output",
            )
        record = SpecialistStepRecord(
            step_id=f"{action.action_id}:specialist-step:{step_index}",
            step_index=step_index,
            action_id=action.action_id,
            agent=action.agent,
            decision_id=decision.decision_id,
            candidate_id=candidate.candidate_id,
            tool_name=candidate.tool_name,
            parameters=parameters,
            reason=decision.reason,
            evidence_ids=list(output.evidence_ids),
            output_summary=_tool_output_summary(output),
        )
        return _ExecutedTool(
            candidate=candidate,
            decision=decision,
            output=output,
            record=record,
        )

    def _analyze(
        self,
        action: InvestigationAction,
        *,
        context: dict[str, Any],
        executed: list[_ExecutedTool],
        effective: list[_ExecutedTool],
        deterministic_analysis: SpecialistAnalysis,
    ) -> tuple[SpecialistAnalysis, int, bool]:
        validation_errors: list[str] = []
        payload = {
            "action": action.to_dict(),
            "trusted_context": dict(context),
            "completed_steps": [item.record.to_dict() for item in executed],
            "effective_tool_observations": [
                _model_tool_observation(item) for item in effective
            ],
            "observed_evidence_ids": list(deterministic_analysis.evidence_ids),
            "deterministic_specialist_analysis": deterministic_analysis.to_dict(),
            "validation_errors": validation_errors,
        }
        error_count = 0
        for attempt in range(_OUTPUT_ATTEMPTS):
            try:
                response = self.llm_client.complete_json(
                    LLMRequest(
                        agent=action.agent,
                        prompt_name="specialist_analysis",
                        prompt_version=self.analysis_prompt_version,
                        payload=payload,
                    )
                )
                analysis = SpecialistAnalysis.from_dict(response.data)
                if analysis.evidence_ids != deterministic_analysis.evidence_ids:
                    raise ValueError(
                        "Specialist analysis Evidence IDs are not the exact observed closure"
                    )
                return analysis, error_count, False
            except Exception as exc:
                error_count += 1
                validation_errors.append(
                    self._validation_error(exc, attempt=attempt + 1)
                )
        return deterministic_analysis, error_count, True

    def _assemble_finding(
        self,
        action: InvestigationAction,
        *,
        request_id: str,
        context: dict[str, Any],
        outputs: list[ToolOutput],
    ) -> AgentFinding:
        evidence = _merge_evidence(outputs)
        if not evidence:
            raise SpecialistV2Error(
                "Successful Specialist Tools returned no first-class Evidence",
                stage="finding_assembly",
                reason="missing_evidence_payload",
            )
        evidence_ids = [item.evidence_id for item in evidence]
        warnings = _merge_warnings(outputs, set(evidence_ids))
        if action.agent == AgentKind.MES.value:
            summary, confidence, details = self._assemble_mes(context, outputs)
            suffix = "mes"
            finding_kind = FindingKind.SPECIALIST_OBSERVATION.value
        elif action.agent == AgentKind.FDC.value:
            summary, confidence, details = self._assemble_fdc(context, outputs)
            suffix = "fdc"
            finding_kind = FindingKind.SPECIALIST_OBSERVATION.value
        elif action.agent == AgentKind.DEFECT_WAT.value:
            summary, confidence, details = self._assemble_defect(context, outputs)
            suffix = "defect-wat"
            finding_kind = FindingKind.SPECIALIST_OBSERVATION.value
        else:
            summary, confidence, details = self._assemble_knowledge(context, outputs)
            suffix = "knowledge"
            finding_kind = FindingKind.KNOWLEDGE_DISCOVERY.value
        return AgentFinding(
            finding_id=f"{request_id}:{suffix}",
            agent=action.agent,
            finding_kind=finding_kind,
            summary=summary,
            confidence=confidence,
            evidence_ids=evidence_ids,
            evidence=evidence,
            details=details,
            warnings=warnings,
        )

    @staticmethod
    def _assemble_mes(
        context: dict[str, Any],
        outputs: list[ToolOutput],
    ) -> tuple[str, float, dict[str, Any]]:
        by_name = {output.tool_name: output.data for output in outputs}
        context_data = by_name.get("get_lot_context", {})
        impact_data = by_name.get("find_impact_lots", {})
        affected_data = by_name.get("find_affected_lots", {})
        genealogy_data = by_name.get("analyze_lot_genealogy", {})
        source_lot_id = (
            _string(impact_data.get("source_lot_id"))
            or _string(context_data.get("lot_id"))
            or _string(context.get("lot_id"))
            or _string(context.get("source_lot_id"))
        )

        if impact_data:
            commonality = _target_commonality_from_impact(impact_data)
            affected_lots = _normalized_lot_ids(impact_data.get("affected_lots"))
            impact_lots = _normalized_lot_ids(impact_data.get("impact_lots"))
            operation_no = _string(impact_data.get("target_operation_no"))
            summary = (
                f"Lot {source_lot_id} has {len(impact_lots)} additional impact Lots "
                f"sharing operation {operation_no} on "
                f"{commonality['equipment_id']}/{commonality['chamber_id']}."
            )
            confidence = round(0.55 + 0.4 * float(commonality["coverage"]), 3)
            operation_commonality = _operation_commonality_from_impact(impact_data)
        elif genealogy_data:
            raw_commonality = genealogy_data.get("target_commonality")
            commonality = dict(raw_commonality) if isinstance(raw_commonality, dict) else {}
            affected_lots = _normalized_lot_ids(affected_data.get("affected_lots"))
            impact_lots = []
            operation_no = _string(genealogy_data.get("target_operation_no"))
            coverage = float(commonality.get("coverage", 0.0))
            summary = (
                f"{len(affected_lots)} affected Lots were identified; {coverage:.0%} "
                f"share operation {operation_no} on "
                f"{commonality.get('equipment_id', '')}/{commonality.get('chamber_id', '')}."
            )
            confidence = round(min(0.99, 0.55 + 0.4 * coverage), 3)
            operation_commonality = list(genealogy_data.get("operation_commonality", []))
        else:
            affected_lots = _normalized_lot_ids(
                affected_data.get("affected_lots", [source_lot_id] if source_lot_id else [])
            )
            impact_lots = []
            operation_no = ""
            commonality = {}
            operation_commonality = []
            if affected_data:
                summary = f"{len(affected_lots)} affected Lots were identified."
                confidence = 0.55 if affected_lots else 0.2
            else:
                summary = (
                    f"Resolved source Lot {source_lot_id}; shared exposure still requires "
                    "impact-scope evidence."
                )
                confidence = 0.4

        fail_modes = context_data.get("fail_modes", affected_data.get("fail_modes", {}))
        if isinstance(fail_modes, list):
            normalized_fail_modes = {str(item): 1 for item in fail_modes}
        elif isinstance(fail_modes, dict):
            normalized_fail_modes = dict(fail_modes)
        else:
            normalized_fail_modes = {}
        source_exposure = impact_data.get("source_exposure", {})
        raw_lane_candidates = impact_data.get(
            "lane_candidates",
            genealogy_data.get("lane_candidates", []),
        )
        lane_candidates = [
            dict(item)
            for item in raw_lane_candidates
            if isinstance(item, Mapping) and _string(item.get("lane_id"))
        ] if isinstance(raw_lane_candidates, list) else []
        return summary, confidence, {
            "investigation_mode": "lot" if source_lot_id else "product_window",
            "source_lot_id": source_lot_id,
            "product_id": _string(context_data.get("product_id"))
            or _string(context.get("product_id")),
            "route_id": _string(context_data.get("route_id")),
            "wat_failed": bool(context_data.get("wat_failed", False)),
            "source_fail_modes": list(context_data.get("fail_modes", [])),
            "recipe_changes": list(context_data.get("recipe_changes", [])),
            "source_lot": dict(context_data.get("lot", {})),
            "source_exposure": (
                dict(source_exposure) if isinstance(source_exposure, dict) else {}
            ),
            "affected_lots": affected_lots,
            "impact_lots": impact_lots,
            "affected_wafers": _normalized_lot_ids(
                impact_data.get("affected_wafers", [])
            ),
            "impact_wafers": _normalized_lot_ids(impact_data.get("impact_wafers", [])),
            "scope_level": _string(impact_data.get("scope_level")) or "lot",
            "impact_criteria": dict(impact_data.get("impact_criteria", {})),
            "normal_lots": _normalized_lot_ids(affected_data.get("normal_lots", [])),
            "suspect_lots": _normalized_lot_ids(affected_data.get("suspect_lots", [])),
            "passing_suspect_lots": _normalized_lot_ids(
                affected_data.get("passing_suspect_lots", [])
            ),
            "yield_trend": list(affected_data.get("yield_trend", [])),
            "fail_modes": normalized_fail_modes,
            "target_operation_no": operation_no,
            "target_commonality": commonality,
            "operation_commonality": operation_commonality,
            "lane_candidates": lane_candidates,
            "hold_count": int(
                genealogy_data.get(
                    "hold_count",
                    len(context_data.get("hold_records", [])),
                )
            ),
        }

    @staticmethod
    def _assemble_fdc(
        context: dict[str, Any],
        outputs: list[ToolOutput],
    ) -> tuple[str, float, dict[str, Any]]:
        by_name = {output.tool_name: output.data for output in outputs}
        parameter_data = by_name.get("analyze_parameter_shift", {})
        event_data = by_name.get("find_ooc_events", {})
        spc_data = by_name.get(
            "analyze_spc_evidence",
            by_name.get("perform_basic_spc_analysis", {}),
        )
        parameter_summary = list(parameter_data.get("parameter_summary", []))
        event_count = int(event_data.get("event_count", 0))
        spc_ooc_count = int(spc_data.get("ooc_parameter_count", 0))
        shifts = {
            str(item.get("parameter_name", "")): float(
                item.get("avg_delta_percent", 0.0)
            )
            for item in parameter_summary
            if isinstance(item, dict) and _string(item.get("parameter_name"))
        }
        shift_text = ", ".join(
            f"{name} {delta:+.1f}%" for name, delta in shifts.items()
        )
        max_abs_shift = max((abs(value) for value in shifts.values()), default=0.0)
        confidence = 0.5 + min(0.25, max_abs_shift / 100.0)
        if event_count > 0 or spc_ooc_count > 0:
            confidence += 0.2
        operation_no = _string(context.get("operation_no")) or _string(
            context.get("target_operation_no")
        )
        equipment_id = _string(context.get("equipment_id"))
        chamber_id = _string(context.get("chamber_id"))
        summary = (
            f"{equipment_id}/{chamber_id} shows "
            f"{shift_text or 'no selected parameter-shift result'}, {event_count} recorded "
            f"OOC events, and {spc_ooc_count} SPC OOC parameters at operation {operation_no}."
        )
        return summary, round(min(0.99, confidence), 3), {
            "lot_ids": _normalized_lot_ids(
                context.get("lot_ids", context.get("affected_lots", []))
            ),
            "operation_no": operation_no,
            "equipment_id": equipment_id,
            "chamber_id": chamber_id,
            "parameter_summary": parameter_summary,
            "event_count": event_count,
            "severity_counts": dict(event_data.get("severity_counts", {})),
            "events": list(event_data.get("events", [])),
            "spc_ooc_contexts": list(event_data.get("spc_contexts", [])),
            "spc_method": dict(spc_data.get("method", {})),
            "spc_results": list(spc_data.get("spc_results", [])),
            "spc_analyzed_parameter_count": int(
                spc_data.get("analyzed_parameter_count", 0)
            ),
            "spc_ooc_parameter_count": spc_ooc_count,
            "spc_point_violation_count": int(
                spc_data.get("calculated_point_violation_count", 0)
            ),
            "spc_baseline_insufficient_parameters": list(
                spc_data.get("baseline_insufficient_parameters", [])
            ),
        }

    @staticmethod
    def _assemble_defect(
        context: dict[str, Any],
        outputs: list[ToolOutput],
    ) -> tuple[str, float, dict[str, Any]]:
        data = outputs[-1].data
        defect_counts = {
            str(key): int(value) for key, value in dict(data.get("defect_counts", {})).items()
        }
        defect_patterns = {
            str(key): int(value)
            for key, value in dict(data.get("defect_patterns", {})).items()
        }
        wat_fail_modes = {
            str(key): int(value)
            for key, value in dict(data.get("wat_fail_modes", {})).items()
        }
        wat_fail_count = int(data.get("wat_fail_count", 0))
        metrology_fail_count = int(data.get("metrology_fail_count", 0))
        has_defect = bool(defect_counts)
        has_wat = wat_fail_count > 0
        has_metrology = metrology_fail_count > 0
        if has_metrology and (has_defect or has_wat):
            confidence = 0.95
        elif has_metrology:
            confidence = 0.85
        elif has_defect and has_wat:
            confidence = 0.9
        elif has_defect or has_wat:
            confidence = 0.6
        else:
            confidence = 0.2
        dominant_defect = (
            max(defect_counts, key=lambda item: defect_counts[item])
            if defect_counts
            else "none"
        )
        dominant_pattern = (
            max(defect_patterns, key=lambda item: defect_patterns[item])
            if defect_patterns
            else "none"
        )
        dominant_fail_mode = (
            max(wat_fail_modes, key=lambda item: wat_fail_modes[item])
            if wat_fail_modes
            else "none"
        )
        summary = (
            f"Selected Lots show {dominant_defect}/{dominant_pattern} defect evidence and "
            f"{wat_fail_count} WAT-failing Lots led by {dominant_fail_mode}; metrology "
            f"has {metrology_fail_count} out-of-spec Wafer records."
        )
        return summary, confidence, {
            "lot_ids": _normalized_lot_ids(data.get("lot_ids", context.get("lot_ids", []))),
            "evidence_scope": _string(context.get("evidence_scope"))
            or "selected_lots",
            "defect_counts": defect_counts,
            "defect_patterns": defect_patterns,
            "wat_fail_modes": wat_fail_modes,
            "wat_fail_count": wat_fail_count,
            "wat_fail_record_count": int(
                data.get("wat_fail_record_count", wat_fail_count)
            ),
            "missing_wat_lot_ids": _normalized_lot_ids(
                data.get("missing_wat_lot_ids", [])
            ),
            "metrology_summaries": list(data.get("metrology_summaries", [])),
            "metrology_fail_count": metrology_fail_count,
            "physical_electrical_consistent": has_defect and has_wat,
            "physical_signal_supported": has_metrology or (has_defect and has_wat),
        }

    @staticmethod
    def _assemble_knowledge(
        context: dict[str, Any],
        outputs: list[ToolOutput],
    ) -> tuple[str, float, dict[str, Any]]:
        data = outputs[-1].data
        raw_top_case = data.get("top_case")
        top_case = dict(raw_top_case) if isinstance(raw_top_case, dict) else {}
        if top_case:
            similarity = float(top_case.get("similarity", 0.0))
            summary = (
                f"Historical case {top_case.get('case_id', '')} matched at "
                f"{similarity:.0%}; documented root cause: "
                f"{top_case.get('root_cause', '')}."
            )
        else:
            similarity = 0.0
            summary = "No engineer-confirmed historical RCA case is available."
        return summary, round(similarity, 3), {
            "query": _string(data.get("query")) or _string(context.get("query")),
            "top_case": top_case,
            "cases": list(data.get("cases", [])),
            "documents": list(data.get("documents", [])),
            "retrieval_strategy": _string(data.get("retrieval_strategy")) or "no_match",
            "score_components": dict(data.get("score_components", {})),
            "calibrated_relevance": data.get("calibrated_relevance"),
            "source_confidence": data.get("source_confidence"),
            "matched_chunk_ids": list(data.get("matched_chunk_ids", [])),
            "candidate_lanes": list(data.get("candidate_lanes", [])),
            "scope_reasons": list(data.get("scope_reasons", [])),
            "route_distance": data.get("route_distance"),
            "shared_resource_types": list(data.get("shared_resource_types", [])),
            "scope_fusion_score": data.get("scope_fusion_score"),
            "observation_scope": data.get("observation_scope"),
            "causal_search_scope": data.get("causal_search_scope"),
        }
