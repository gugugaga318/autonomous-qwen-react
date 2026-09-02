"""Markdown report generation from an existing RCAState.

The generator is a pure rendering boundary. It only presents information
already present in RCAState and explicitly marks missing data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from yield_rca_core.models import (
    AgentFinding,
    AgentKind,
    Evidence,
    EvidenceType,
    FindingKind,
    InvestigationMode,
    RCAState,
    Report,
    Warning,
)
from yield_rca_core.question_capability import QUESTION_CAPABILITY_REGISTRY
from yield_rca_core.warning_policy import reconcile_current_warnings


class ReportGenerationError(ValueError):
    """Raised when RCAState cannot support a traceable Report."""


def _deduplicate_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _escape_table_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _format_evidence_ids(evidence_ids: list[str]) -> str:
    if not evidence_ids:
        return "None"
    return ", ".join(f"`{evidence_id}`" for evidence_id in evidence_ids)


def _finding_by_kind(
    state: RCAState,
    *,
    agent: str,
    finding_kind: str,
    legacy_context: str,
) -> AgentFinding | None:
    findings = state.findings_for_kind(finding_kind, agent=agent)
    if len(findings) > 1:
        raise ReportGenerationError(f"RCAState contains multiple {legacy_context} findings")
    if findings:
        return findings[0]

    legacy_findings = [item for item in state.findings if item.agent == agent]
    if len(legacy_findings) > 1:
        raise ReportGenerationError(f"RCAState contains multiple legacy {legacy_context} findings")
    return legacy_findings[0] if legacy_findings else None


def _rca_finding(state: RCAState) -> AgentFinding | None:
    # RCA reasoning is iterative.  RCAState owns the explicit authority
    # pointer; historical Findings remain available for the evidence chain
    # and audit but are never treated as the final result.
    return state.authoritative_rca_finding


def _improvement_finding(state: RCAState) -> AgentFinding | None:
    return _finding_by_kind(
        state,
        agent=AgentKind.IMPROVEMENT.value,
        finding_kind=FindingKind.IMPROVEMENT.value,
        legacy_context="Improvement",
    )


def _evidence_for_finding(state: RCAState, finding: AgentFinding) -> list[Evidence]:
    evidence_by_id = state.evidence_by_id
    return [
        evidence_by_id[evidence_id]
        for evidence_id in finding.evidence_ids
        if evidence_id in evidence_by_id
    ]


def _typed_evidence_ids(
    state: RCAState,
    finding: AgentFinding,
    *,
    evidence_types: set[str],
    source_tools: set[str] | None = None,
    entity_ids: set[str] | None = None,
) -> list[str]:
    evidence_ids: list[str] = []
    for evidence in _evidence_for_finding(state, finding):
        if evidence.evidence_type not in evidence_types:
            continue
        if source_tools is not None and evidence.source_tool not in source_tools:
            continue
        if entity_ids is not None and not any(
            entity.entity_id in entity_ids for entity in evidence.entities
        ):
            continue
        evidence_ids.append(evidence.evidence_id)
    return evidence_ids


def _evidence_ids_by_type(
    state: RCAState,
    *,
    evidence_types: set[str],
    source_tools: set[str] | None = None,
    entity_ids: set[str] | None = None,
) -> list[str]:
    evidence_ids: list[str] = []
    for evidence in state.evidence:
        if evidence.evidence_type not in evidence_types:
            continue
        if source_tools is not None and evidence.source_tool not in source_tools:
            continue
        if entity_ids is not None and not any(
            entity.entity_id in entity_ids for entity in evidence.entities
        ):
            continue
        evidence_ids.append(evidence.evidence_id)
    return evidence_ids


def _format_entities(evidence: Evidence) -> str:
    if not evidence.entities:
        return "Not available"
    return ", ".join(f"{entity.entity_type}: {entity.entity_id}" for entity in evidence.entities)


def _evidence_text(evidence: Evidence) -> str:
    return evidence.observation or evidence.summary


def _state_warnings(state: RCAState) -> list[Warning]:
    historical_finding_warnings = [
        warning
        for finding in state.findings
        for warning in finding.warnings
    ]
    return reconcile_current_warnings(
        state.warnings,
        historical_finding_warnings,
        current_findings=state.findings,
    )


def _validate_evidence_ids(
    evidence_ids: list[str],
    known_evidence_ids: set[str],
    context: str,
) -> None:
    missing = set(evidence_ids) - known_evidence_ids
    if missing:
        raise ReportGenerationError(
            f"{context} references evidence not present in RCAState: {sorted(missing)}"
        )


def _evidence_chain(state: RCAState, rca_finding: AgentFinding | None) -> list[dict[str, Any]]:
    if rca_finding is not None:
        chain = rca_finding.details.get("evidence_chain", [])
        if chain:
            return [dict(item) for item in chain]
    return [
        {
            "stage": finding.agent,
            "claim": finding.summary,
            "confidence": finding.confidence,
            "evidence_ids": list(finding.evidence_ids),
        }
        for finding in state.findings
        if finding.agent not in {AgentKind.RCA_REASONING.value, AgentKind.IMPROVEMENT.value}
    ]


def _recommended_actions(rca_finding: AgentFinding | None) -> list[dict[str, Any]]:
    if rca_finding is None:
        return []
    actions = rca_finding.details.get("recommended_actions", [])
    if not isinstance(actions, list):
        raise ReportGenerationError("recommended_actions must be a list")
    return [dict(item) for item in actions]


def _problem_section(state: RCAState) -> list[str]:
    time_window = state.job.time_window
    start = time_window.get("start") or time_window.get("start_date")
    end = time_window.get("end") or time_window.get("end_date")
    if start and end:
        window_text = f"`{start}` to `{end}`"
    elif start or end:
        window_text = f"`{start or end}`"
    else:
        window_text = "Not available in RCAState."
    product = f"`{state.job.product_id}`" if state.job.product_id else "Not available in RCAState."
    return [
        "## Problem Summary",
        "",
        f"- Job ID: `{state.job.job_id}`",
        f"- User Request: {state.job.user_query}",
        f"- Investigation Mode: `{state.job.investigation_mode}`",
        (
            f"- Investigated Lot: `{state.job.source_lot_id}`"
            if state.job.source_lot_id
            else "- Investigated Lot: Not applicable."
        ),
        f"- Product: {product}",
        f"- Time Window: {window_text}",
    ]


def _affected_lot_section(state: RCAState) -> tuple[list[str], list[str]]:
    mes_findings = [finding for finding in state.findings if finding.agent == AgentKind.MES.value]
    affected_evidence_ids: list[str] = []
    lot_mode = state.job.investigation_mode == InvestigationMode.LOT.value
    relevant_entities = {
        entity_id
        for entity_id in [
            state.job.source_lot_id,
            *state.affected_lots,
            *state.impact_lots,
            *state.affected_wafers,
            *state.impact_wafers,
        ]
        if entity_id
    }
    for finding in mes_findings:
        preferred_types = (
            {
                EvidenceType.LOT_CONTEXT.value,
                EvidenceType.METROLOGY_DEVIATION.value,
                EvidenceType.EXCURSION_WINDOW.value,
                EvidenceType.IMPACT_SCOPE.value,
            }
            if lot_mode
            else {EvidenceType.IMPACT_SCOPE.value}
        )
        preferred = _typed_evidence_ids(
            state,
            finding,
            evidence_types=preferred_types,
            entity_ids=relevant_entities or None,
        )
        if not preferred:
            preferred = _typed_evidence_ids(
                state,
                finding,
                evidence_types=preferred_types,
            )
        affected_evidence_ids.extend(preferred or finding.evidence_ids)
    affected_evidence_ids = _deduplicate_strings(affected_evidence_ids)

    if lot_mode:
        criteria = state.impact_criteria
        lines = ["## Lot Investigation Scope", ""]
        lines.append(f"- Investigated Lot: `{state.job.source_lot_id}`")
        lines.append(f"- Scope Level: `{state.scope_level}`")
        lines.append(f"- Impact Lot Count: {len(state.impact_lots)}")
        lines.append(
            "- Impact Lots: "
            + (
                ", ".join(f"`{lot_id}`" for lot_id in state.impact_lots)
                if state.impact_lots
                else "None identified."
            )
        )
        lines.append(f"- Total Exposed Population: {len(state.affected_lots)}")
        lines.append(f"- Affected Wafer Count: {len(state.affected_wafers)}")
        lines.append(
            "- Affected Wafers: "
            + (
                ", ".join(f"`{wafer_id}`" for wafer_id in state.affected_wafers)
                if state.affected_wafers
                else "None identified."
            )
        )
        lines.append(f"- Impact Wafer Count: {len(state.impact_wafers)}")
        lines.append(
            "- Impact Wafers: "
            + (
                ", ".join(f"`{wafer_id}`" for wafer_id in state.impact_wafers)
                if state.impact_wafers
                else "None identified."
            )
        )
        if criteria:
            lines.extend(
                [
                    (
                        "- Shared Exposure: "
                        f"operation `{criteria.get('operation_no', 'unknown')}`, "
                        f"equipment `{criteria.get('equipment_id', 'unknown')}`, "
                        f"chamber `{criteria.get('chamber_id', 'unknown')}`"
                    ),
                    (
                        "- Excursion Window: "
                        f"`{criteria.get('excursion_start', 'unknown')}` to "
                        f"`{criteria.get('excursion_end', 'unknown')}`"
                    ),
                    f"- Selection Rule: {criteria.get('selection_rule', 'Not available.')}",
                ]
            )
        lines.append(f"- Evidence: {_format_evidence_ids(affected_evidence_ids)}")
        return lines, affected_evidence_ids

    lines = ["## Affected Lots", ""]
    if state.affected_lots:
        lines.extend(
            [
                f"- Count: {len(state.affected_lots)}",
                f"- Lots: {', '.join(f'`{lot_id}`' for lot_id in state.affected_lots)}",
                f"- Evidence: {_format_evidence_ids(affected_evidence_ids)}",
            ]
        )
    else:
        lines.append("Not available in RCAState.")
    return lines, affected_evidence_ids


def _chain_section(chain: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    lines = ["## Evidence Chain", ""]
    cited: list[str] = []
    if not chain:
        lines.append("Not available in RCAState.")
        return lines, cited
    for index, item in enumerate(chain, start=1):
        stage = str(item.get("stage", "unknown")).upper()
        claim = str(item.get("claim", "")).strip()
        if not claim:
            raise ReportGenerationError("evidence_chain item must include a claim")
        evidence_ids = [str(value) for value in item.get("evidence_ids", [])]
        if not evidence_ids:
            raise ReportGenerationError("evidence_chain claim must reference evidence_ids")
        cited.extend(evidence_ids)
        lines.extend(
            [
                f"{index}. **{stage}**: {claim}",
                f"   - Evidence: {_format_evidence_ids(evidence_ids)}",
            ]
        )
    return lines, cited


def _spc_section(state: RCAState) -> tuple[list[str], list[str]]:
    fdc_findings = [item for item in state.findings if item.agent == AgentKind.FDC.value]
    if not fdc_findings:
        lines = ["## SPC Evidence", ""]
        lines.append("Not available in RCAState.")
        return lines, []

    finding = fdc_findings[0]
    raw_results = finding.details.get("spc_results", [])
    results = [dict(item) for item in raw_results if isinstance(item, dict)]
    raw_method = finding.details.get("spc_method", {})
    method = raw_method if isinstance(raw_method, dict) else {}
    advanced = method.get("engine") == "deterministic_advanced_spc"
    lines = ["## SPC Evidence" if advanced else "## Minimal SPC Analysis", ""]
    spc_source_tools = {
        "perform_basic_spc_analysis",
        "perform_advanced_spc_analysis",
        "find_ooc_events",
    }
    cited = _typed_evidence_ids(
        state,
        finding,
        evidence_types={
            EvidenceType.SPC_VIOLATION.value,
            EvidenceType.OOC_EVENT.value,
            EvidenceType.HOLD_EVENT.value,
            EvidenceType.DATA_MISSING.value,
            EvidenceType.NEGATIVE_SIGNAL.value,
        },
        source_tools=spc_source_tools,
    )
    if not cited:
        cited = [
            evidence_id for evidence_id in finding.evidence_ids if evidence_id.startswith("EV_SPC_")
        ]
    if not results:
        insufficient = finding.details.get("spc_baseline_insufficient_parameters", [])
        lines.extend(
            [
                "Control limits could not be calculated from the available baseline data.",
                (
                    "- Parameters without a sufficient baseline: "
                    + ", ".join(f"`{item}`" for item in insufficient)
                    if isinstance(insufficient, list) and insufficient
                    else "- Parameters without a sufficient baseline: Not available."
                ),
                f"- Evidence: {_format_evidence_ids(cited)}",
            ]
        )
        return lines, cited

    if advanced:
        lines.extend(
            [
                f"- Engine: `{method.get('engine')}`",
                f"- Rules: `{method.get('rules')}`",
                f"- Baseline Matching: `{method.get('baseline_matching')}`",
                "",
                (
                    "| Parameter | Chart | Status | Baseline | CL | LCL | UCL | "
                    "Capability | Triggered Rules | Evidence |"
                ),
                "|---|---|---|---|---:|---:|---:|---|---|---|",
            ]
        )
    else:
        lines.extend(
            [
                f"- Control Limits: `{method.get('control_limits', 'Not available')}`",
                (
                    "- Minimum Baseline Samples: "
                    f"{method.get('minimum_baseline_samples', 'Not available')}"
                ),
                "",
                (
                    "| Parameter | Status | Baseline Scope | Center Line | LCL | UCL | "
                    "Target Mean | Point Violations | Triggered Rules | Evidence |"
                ),
                "|---|---|---|---:|---:|---:|---:|---:|---|---|",
            ]
        )
    for item in results:
        evidence_id = str(item.get("evidence_id", ""))
        rules = item.get("violated_rules", [])
        rule_text = ", ".join(str(rule) for rule in rules) if isinstance(rules, list) else ""
        if advanced:
            capability = item.get("capability")
            capability_text = "Not calculated"
            if isinstance(capability, dict):
                capability_text = f"Cpk={capability.get('cpk')}, Ppk={capability.get('ppk')}; " + (
                    "valid" if capability.get("valid_for_decision") else "informational"
                )
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{_escape_table_cell(item.get('parameter_name', 'unknown'))}`",
                        _escape_table_cell(item.get("chart_type", "unknown")),
                        _escape_table_cell(item.get("status", "unknown")),
                        f"`{_escape_table_cell(item.get('baseline_id', 'unknown'))}`",
                        _escape_table_cell(item.get("center_line", "unknown")),
                        _escape_table_cell(item.get("lower_control_limit", "unknown")),
                        _escape_table_cell(item.get("upper_control_limit", "unknown")),
                        _escape_table_cell(capability_text),
                        _escape_table_cell(rule_text or "None"),
                        f"`{_escape_table_cell(evidence_id)}`" if evidence_id else "None",
                    ]
                )
                + " |"
            )
        else:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{_escape_table_cell(item.get('parameter_name', 'unknown'))}`",
                        _escape_table_cell(item.get("status", "unknown")),
                        _escape_table_cell(item.get("baseline_scope", "unknown")),
                        _escape_table_cell(item.get("center_line", "unknown")),
                        _escape_table_cell(item.get("lower_control_limit", "unknown")),
                        _escape_table_cell(item.get("upper_control_limit", "unknown")),
                        _escape_table_cell(item.get("target_mean", "unknown")),
                        _escape_table_cell(item.get("point_violation_count", "unknown")),
                        _escape_table_cell(rule_text or "None"),
                        f"`{_escape_table_cell(evidence_id)}`" if evidence_id else "None",
                    ]
                )
                + " |"
            )

    if advanced:
        lines.extend(["", "### Rule Violations", ""])
        violations_found = False
        for item in results:
            violations = item.get("violations", [])
            if not isinstance(violations, list):
                continue
            grouped_violations: dict[str, list[dict[str, Any]]] = {}
            for raw_violation in violations:
                if isinstance(raw_violation, dict):
                    rule_code = str(raw_violation.get("rule_code", "unknown"))
                    grouped_violations.setdefault(rule_code, []).append(raw_violation)
            for rule_code, rule_violations in grouped_violations.items():
                violations_found = True
                trigger_samples = [
                    str(violation.get("sample_ids", ["unknown"])[-1])
                    for violation in rule_violations
                    if isinstance(violation.get("sample_ids"), list)
                    and violation.get("sample_ids")
                ]
                displayed = trigger_samples[:12]
                sample_text = ", ".join(f"`{sample}`" for sample in displayed)
                if len(trigger_samples) > len(displayed):
                    sample_text += (
                        f", plus {len(trigger_samples) - len(displayed)} endpoints retained "
                        "in evidence metadata"
                    )
                lines.append(
                    f"- `{item.get('parameter_name')}` / `{rule_code}`: {sample_text}"
                )
        if not violations_found:
            lines.append("No Nelson-rule violations were detected.")

        contexts = finding.details.get("spc_ooc_contexts", [])
        if isinstance(contexts, list) and contexts:
            lines.extend(["", "### OOC, Hold, and Excursion Scope", ""])
            context_citations = _typed_evidence_ids(
                state,
                finding,
                evidence_types={EvidenceType.OOC_EVENT.value, EvidenceType.HOLD_EVENT.value},
                source_tools={"find_ooc_events"},
            )
            for context in contexts:
                if not isinstance(context, dict):
                    continue
                trigger_hold = context.get("trigger_hold")
                trigger_hold_id = (
                    trigger_hold.get("hold_id")
                    if isinstance(trigger_hold, dict)
                    else "Not available"
                )
                lines.extend(
                    [
                        f"- OOC Event: `{context.get('event_key')}`",
                        f"- Trigger Lot: `{context.get('trigger_lot_id')}`",
                        f"- Trigger Wafer: `{context.get('trigger_wafer_id') or 'Lot-level'}`",
                        f"- Trigger Hold: `{trigger_hold_id}`",
                    ]
                )
                impact_scopes = context.get("impact_scopes", [])
                if isinstance(impact_scopes, list):
                    impact_text = ", ".join(
                        f"`{scope.get('lot_id')}` -> `{scope.get('hold_id')}`"
                        for scope in impact_scopes
                        if isinstance(scope, dict)
                    )
                    lines.append(f"- Impact Lots and Holds: {impact_text or 'None'}")
                context_entity_ids = {
                    str(value)
                    for value in (
                        context.get("event_key"),
                        context.get("trigger_lot_id"),
                        context.get("trigger_wafer_id"),
                    )
                    if value
                }
                context_evidence = _evidence_ids_by_type(
                    state,
                    evidence_types={EvidenceType.OOC_EVENT.value, EvidenceType.HOLD_EVENT.value},
                    source_tools={"find_ooc_events"},
                    entity_ids=context_entity_ids or None,
                )
                context_evidence = context_evidence or context_citations
                cited.extend(context_evidence)
                lines.append(f"- Evidence: {_format_evidence_ids(context_evidence)}")
    return lines, cited


def _root_cause_sections(
    rca_finding: AgentFinding | None,
) -> tuple[list[str], list[str]]:
    lines = ["## Root Cause", ""]
    confidence_lines = ["## Confidence", ""]
    if rca_finding is None:
        lines.extend(
            [
                "- Status: Not available in RCAState.",
                "- Root Cause: Not available in RCAState.",
                "- Evidence: None",
            ]
        )
        confidence_lines.extend(["- Confidence: Not available in RCAState.", "- Evidence: None"])
        return lines + [""] + confidence_lines, []

    root_cause = str(rca_finding.details.get("root_cause", "")).strip()
    status = str(rca_finding.details.get("status", "")).strip()
    conclusion_status = str(
        rca_finding.details.get("conclusion_status", status)
    ).strip()
    chain_status = rca_finding.details.get("causal_chain_completeness")
    if not root_cause or not status:
        raise ReportGenerationError("RCA finding must include root_cause and status")
    evidence_ids = [
        str(value)
        for value in rca_finding.details.get(
            "root_cause_evidence_ids",
            rca_finding.evidence_ids,
        )
    ]
    if not evidence_ids:
        raise ReportGenerationError("root cause must reference evidence_ids")
    lines.extend(
        [
            f"- Status: `{status}`",
            f"- Conclusion Status: `{conclusion_status}`",
            *(
                [f"- Causal Chain: `{chain_status}`"]
                if chain_status
                else []
            ),
            f"- Root Cause: **{root_cause}**",
            f"- Evidence: {_format_evidence_ids(evidence_ids)}",
        ]
    )
    confidence_lines.extend(
        [
            f"- Confidence: **{rca_finding.confidence:.1%}**",
            f"- Evidence: {_format_evidence_ids(evidence_ids)}",
        ]
    )
    return lines + [""] + confidence_lines, evidence_ids


def _competition_section(state: RCAState) -> list[str]:
    trace = state.competition_trace
    if trace is None:
        return []
    lines = [
        "## Candidate Competition",
        "",
        f"- Requirement: `{trace.competition_requirement}`",
        f"- Candidate Competition Status: `{trace.competition_status}`",
        f"- Competition Type: `{trace.competition_type}`",
        "- Competition Axes: "
        + (
            ", ".join(f"`{item}`" for item in trace.competition_axes)
            if trace.competition_axes
            else "Not evaluated"
        ),
        f"- Scope Assessment Status: `{trace.scope_assessment_status}`",
        f"- Lane Search Status: `{trace.alternative_search_status}`",
    ]
    if trace.competition_failure_reason is not None:
        lines.append(
            f"- Competition Failure: `{trace.competition_failure_reason}`"
        )
    if trace.candidate_lineage:
        lines.extend(["", "### Candidate Lineage", ""])
        for item in trace.candidate_lineage:
            status = str(item.get("lineage_status", "unavailable"))
            root_cause = str(
                item.get("root_cause")
                or item.get("prior_root_cause")
                or "Not available"
            )
            lines.append(f"- `{status}`: {root_cause}")
    return lines


def _action_section(actions: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    lines = ["## Recommended Actions", ""]
    cited: list[str] = []
    if not actions:
        lines.append("No evidence-backed recommended actions are available in RCAState.")
        return lines, cited
    for index, item in enumerate(actions, start=1):
        action = str(item.get("action", "")).strip()
        evidence_ids = [str(value) for value in item.get("evidence_ids", [])]
        if not action:
            raise ReportGenerationError("recommended action must include action text")
        if not evidence_ids:
            raise ReportGenerationError("recommended action must reference evidence_ids")
        cited.extend(evidence_ids)
        lines.extend(
            [
                f"{index}. {action}",
                f"   - Evidence: {_format_evidence_ids(evidence_ids)}",
            ]
        )
    return lines, cited


def _improvement_sections(finding: AgentFinding) -> tuple[list[list[str]], list[str]]:
    details = finding.details
    incident_summary = str(details.get("incident_summary", "")).strip()
    engineering_summary = str(details.get("engineering_summary", "")).strip()
    raw_scope = details.get("scope_assessment", {})
    scope = raw_scope if isinstance(raw_scope, dict) else {}
    raw_recommendations = details.get("recommendations", {})
    recommendations = raw_recommendations if isinstance(raw_recommendations, dict) else {}
    if not incident_summary or not engineering_summary:
        raise ReportGenerationError("Improvement finding must include engineering summaries")

    cited: list[str] = []
    summary_ids = list(finding.evidence_ids)
    cited.extend(summary_ids)
    sections: list[list[str]] = [
        [
            "## Engineering Improvement Summary",
            "",
            f"- Incident: {incident_summary}",
            f"- Engineering Synthesis: {engineering_summary}",
            f"- Conclusion Level: `{scope.get('level', 'event')}`",
            f"- Fab-Level Criteria: {', '.join(scope.get('criteria', [])) or 'None'}",
            f"- Evidence: {_format_evidence_ids(summary_ids)}",
        ]
    ]
    category_titles = (
        ("containment_actions", "Containment Actions"),
        ("corrective_actions", "Corrective Actions"),
        ("recipe_optimization", "Recipe Optimization Recommendations"),
        ("preventive_actions", "Preventive Actions"),
        ("fab_system_optimization", "Fab/System Optimization"),
    )
    for category, title in category_titles:
        lines = [f"## {title}", ""]
        raw_items = recommendations.get(category, [])
        items = [dict(item) for item in raw_items if isinstance(item, dict)]
        if not items:
            lines.append("No evidence-backed recommendation is available for this category.")
            sections.append(lines)
            continue
        for index, item in enumerate(items, start=1):
            action = str(item.get("action", "")).strip()
            rationale = str(item.get("rationale", "")).strip()
            evidence_ids = [str(value) for value in item.get("evidence_ids", [])]
            if not action or not rationale or not evidence_ids:
                raise ReportGenerationError(
                    "Improvement recommendation requires action, rationale, and evidence_ids"
                )
            cited.extend(evidence_ids)
            lines.extend(
                [
                    f"{index}. {action}",
                    f"   - Rationale: {rationale}",
                    f"   - Evidence: {_format_evidence_ids(evidence_ids)}",
                ]
            )
        sections.append(lines)

    sections.append(
        [
            "## Memory Status",
            "",
            f"- Status: `{details.get('memory_status', 'not_persisted')}`",
            (
                "- Approval Requirement: Two different engineers must approve before "
                "long-term memory publication."
                if details.get("requires_two_engineer_approval", False)
                else "- Approval Requirement: Not available."
            ),
            (
                "- Publication State: The workflow marks this content as candidate-ready; "
                "use the Memory Approval API for its current pending, published, or "
                "rejected state."
            ),
        ]
    )
    return sections, _deduplicate_strings(cited)


def _warning_section(
    warnings: list[Warning],
    generated_messages: list[tuple[str, str]],
) -> tuple[list[str], list[str]]:
    lines = ["## Warnings", ""]
    cited: list[str] = []
    if not warnings and not generated_messages:
        lines.append("No warnings.")
        return lines, cited
    for warning in warnings:
        cited.extend(warning.evidence_ids)
        lines.append(
            f"- `{warning.warning_id}` ({warning.severity}): {warning.message} "
            f"Evidence: {_format_evidence_ids(warning.evidence_ids)}"
        )
    for warning_id, message in generated_messages:
        lines.append(f"- `{warning_id}` (warning): {message} Evidence: None")
    return lines, cited


def _question_semantics_section(state: RCAState) -> tuple[list[str], list[str]]:
    """Render the Question/Evidence contract as an auditable report section."""

    if not (
        state.capability_notices
        or state.investigation_questions
        or state.question_evidence_links
        or state.question_update_reviews
    ):
        return [], []

    links_by_question: dict[str, list[Any]] = {}
    for link in state.question_evidence_links:
        links_by_question.setdefault(link.question_id, []).append(link)
    reviews_by_question: dict[str, list[Any]] = {}
    for review in state.question_update_reviews:
        if review.question_id:
            reviews_by_question.setdefault(review.question_id, []).append(review)

    lines = ["## Question–Evidence Semantics", ""]
    cited: list[str] = []
    if state.capability_notices:
        lines.extend(["### Capability Notices", ""])
        for notice in state.capability_notices:
            status = "supported" if notice.supported else "unsupported"
            lines.append(
                f"- `{notice.capability}` (**{status}**, requested by `{notice.request_source}`): "
                f"{notice.reason}"
            )
            if notice.available_alternatives:
                lines.append(
                    "  - Available alternatives: "
                    + ", ".join(f"`{item}`" for item in notice.available_alternatives)
                )

    if state.investigation_questions:
        lines.extend(["### Question Coverage", ""])
        for question in state.investigation_questions:
            capability = QUESTION_CAPABILITY_REGISTRY.get(str(question.question_kind))
            links = links_by_question.get(question.question_id, [])
            satisfied = sorted(
                {
                    link.matched_evidence_group
                    for link in links
                    if link.relation != "unavailable"
                    and capability is not None
                    and link.matched_evidence_group in capability.closure_evidence_groups
                }
            )
            missing = sorted(
                set(capability.closure_evidence_groups) - set(satisfied)
                if capability is not None
                else set()
            )
            lines.extend(
                [
                    f"- **{question.question_id}** "
                    f"(`{question.question_kind}`, `{question.status}`): "
                    f"{question.question}",
                    "  - Compatible Actions: "
                    + (
                        ", ".join(
                            f"`{item}`"
                            for item in sorted(capability.allowed_actions)
                        )
                        if capability is not None and capability.allowed_actions
                        else "None"
                    ),
                    "  - Satisfied groups: "
                    f"{', '.join(f'`{item}`' for item in satisfied) or 'None'}",
                    "  - Missing groups: "
                    f"{', '.join(f'`{item}`' for item in missing) or 'None'}",
                ]
            )
            if question.answer:
                lines.append(f"  - Answer: {question.answer}")
            if question.unavailable_reason:
                lines.append(f"  - Unavailable reason: {question.unavailable_reason}")
            if links:
                lines.append("  - Evidence links:")
                for link in links:
                    cited.append(link.evidence_id)
                    lines.append(
                        f"    - `{link.evidence_id}` via `{link.action_id}` "
                        f"({link.relation}, `{link.matched_evidence_group}`): {link.reason}"
                    )
            else:
                lines.append("  - Evidence links: None")
            for review in reviews_by_question.get(question.question_id, []):
                lines.append(
                    f"  - QuestionUpdate review `{review.disposition}` "
                    f"(`{review.reason_code}`): {review.reason}"
                )

    known_question_ids = {question.question_id for question in state.investigation_questions}
    unscoped_reviews = [
        review
        for review in state.question_update_reviews
        if not review.question_id or review.question_id not in known_question_ids
    ]
    if unscoped_reviews:
        lines.extend(["### Unattached Review Diagnostics", ""])
        for review in unscoped_reviews:
            lines.append(
                f"- Decision `{review.decision_id}`: `{review.disposition}` "
                f"(`{review.reason_code}`): {review.reason}"
            )
    return lines, _deduplicate_strings(cited)


def _reference_section(evidence: list[Evidence], cited_ids: list[str]) -> list[str]:
    evidence_by_id = {item.evidence_id: item for item in evidence}
    lines = [
        "## Typed Evidence Register",
        "",
        "| Evidence ID | Type | Entities | Observation | Source Agent/Tool | Confidence |",
        "|---|---|---|---|---|---:|",
    ]
    for evidence_id in cited_ids:
        item = evidence_by_id[evidence_id]
        source_agent_tool = (
            f"{item.source_agent or item.source_type}/{item.source_tool}"
            if item.source_tool
            else f"{item.source_agent or item.source_type}/Not available"
        )
        confidence = f"{item.confidence:.1%}" if item.confidence is not None else "Not available"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{_escape_table_cell(item.evidence_id)}`",
                    _escape_table_cell(item.evidence_type or "legacy"),
                    _escape_table_cell(_format_entities(item)),
                    _escape_table_cell(_evidence_text(item)),
                    _escape_table_cell(source_agent_tool),
                    _escape_table_cell(confidence),
                ]
            )
            + " |"
        )
    return lines


@dataclass(frozen=True)
class ReportGenerator:
    """Render a traceable Markdown Report without adding domain facts."""

    title: str = "Yield Excursion RCA Report"

    def generate(self, state: RCAState, *, report_id: str | None = None) -> Report:
        if not isinstance(state, RCAState):
            raise ReportGenerationError("state must be an RCAState")
        if not state.evidence:
            raise ReportGenerationError(
                "cannot generate a traceable Report from RCAState without evidence"
            )

        known_evidence_ids = {item.evidence_id for item in state.evidence}
        rca_finding = _rca_finding(state)
        improvement_finding = _improvement_finding(state)
        chain = _evidence_chain(state, rca_finding)
        actions = _recommended_actions(rca_finding)
        warnings = _state_warnings(state)

        generated_warnings: list[tuple[str, str]] = []
        if not state.affected_lots:
            generated_warnings.append(
                ("WARN_REPORT_NO_AFFECTED_LOTS", "Affected lots are not available in RCAState.")
            )
        if rca_finding is None:
            generated_warnings.append(
                ("WARN_REPORT_NO_RCA_RESULT", "RCA reasoning result is not available in RCAState.")
            )
        if not actions and improvement_finding is None:
            generated_warnings.append(
                (
                    "WARN_REPORT_NO_RECOMMENDED_ACTIONS",
                    "Evidence-backed recommended actions are not available in RCAState.",
                )
            )
        if not chain:
            generated_warnings.append(
                ("WARN_REPORT_NO_EVIDENCE_CHAIN", "Evidence chain is not available in RCAState.")
            )

        sections: list[list[str]] = []
        sections.append(_problem_section(state))
        if state.investigation_goal is not None:
            path_lines = ["## Controlled Investigation Path", ""]
            path_lines.append(f"- Intent: `{state.investigation_goal.intent}`")
            for index, record in enumerate(state.action_history, start=1):
                path_lines.append(
                    f"{index}. `{record.action.kind}` ({record.action.agent}): "
                    f"{record.action.reason}"
                )
            path_lines.append(f"- Stop reason: `{state.stop_reason or 'not_available'}`")
            if state.evidence_gaps:
                path_lines.append(f"- Remaining evidence gaps: {', '.join(state.evidence_gaps)}")
            sections.append(path_lines)
        semantic_section, semantic_citations = _question_semantics_section(state)
        if semantic_section:
            sections.append(semantic_section)
        affected_section, affected_citations = _affected_lot_section(state)
        sections.append(affected_section)
        chain_section, chain_citations = _chain_section(chain)
        sections.append(chain_section)
        spc_section, spc_citations = _spc_section(state)
        sections.append(spc_section)
        root_sections, root_citations = _root_cause_sections(rca_finding)
        sections.append(root_sections)
        competition_section = _competition_section(state)
        if competition_section:
            sections.append(competition_section)
        if improvement_finding is not None:
            improvement_sections, action_citations = _improvement_sections(improvement_finding)
            sections.extend(improvement_sections)
        else:
            action_section, action_citations = _action_section(actions)
            sections.append(action_section)
        warning_section, warning_citations = _warning_section(warnings, generated_warnings)
        sections.append(warning_section)

        cited_ids = _deduplicate_strings(
            semantic_citations
            + affected_citations
            + chain_citations
            + spc_citations
            + root_citations
            + action_citations
            + warning_citations
        )
        if not cited_ids:
            cited_ids = [item.evidence_id for item in state.evidence]
        _validate_evidence_ids(cited_ids, known_evidence_ids, "Report")
        sections.append(_reference_section(state.evidence, cited_ids))

        report_title = self.title
        if state.job.source_lot_id:
            report_title = f"Lot-driven RCA Report - {state.job.source_lot_id}"
        elif state.job.product_id:
            report_title = f"{report_title} - {state.job.product_id}"
        markdown_lines = [f"# {report_title}", ""]
        for index, section in enumerate(sections):
            markdown_lines.extend(section)
            if index < len(sections) - 1:
                markdown_lines.append("")

        return Report(
            report_id=report_id or f"{state.job.job_id}:report",
            title=report_title,
            markdown="\n".join(markdown_lines).strip() + "\n",
            cited_evidence_ids=cited_ids,
        )
