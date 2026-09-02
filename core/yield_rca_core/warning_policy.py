"""Current-state warning reconciliation without mutating historical Findings."""

from __future__ import annotations

from collections.abc import Sequence

from yield_rca_core.models import AgentFinding, AgentKind, Warning

SPECIALIST_AGENT_KINDS = frozenset(
    {
        AgentKind.MES.value,
        AgentKind.FDC.value,
        AgentKind.DEFECT_WAT.value,
        AgentKind.KNOWLEDGE.value,
    }
)

_MISSING_FINDINGS_WARNING_ID = "WARN_RCA_MISSING_FINDINGS"


def reconcile_current_warnings(
    existing: Sequence[Warning],
    incoming: Sequence[Warning],
    *,
    current_findings: Sequence[AgentFinding],
) -> list[Warning]:
    """Merge warnings and retire derived warnings disproven by current State."""

    warnings_by_id = {item.warning_id: item for item in existing}
    for item in incoming:
        warnings_by_id[item.warning_id] = item
    present_agents = {finding.agent for finding in current_findings}
    if SPECIALIST_AGENT_KINDS <= present_agents:
        warnings_by_id.pop(_MISSING_FINDINGS_WARNING_ID, None)
    return list(warnings_by_id.values())


__all__ = ["SPECIALIST_AGENT_KINDS", "reconcile_current_warnings"]
