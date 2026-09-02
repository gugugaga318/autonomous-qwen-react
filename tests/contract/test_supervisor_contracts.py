from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core"))

from yield_rca_core.models import (  # noqa: E402
    AgentFinding,
    AgentKind,
    AgentTask,
    RCAJob,
    RCAState,
    TaskPlan,
    TaskStatus,
    Warning,
)
from yield_rca_core.planner_agent import PlannerAgent  # noqa: E402
from yield_rca_core.supervisor import (  # noqa: E402
    SupervisorExecutionError,
    _merge_warnings,
)
from yield_rca_core.workflow import build_csv_workflow  # noqa: E402

SEED_DIR = ROOT / "data" / "seeds" / "golden_case"
QUERY = "Analyze the 40N_SOC yield drop from 2026-07-01 to 2026-07-31."


class SupervisorContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = build_csv_workflow(SEED_DIR)

    def test_supervisor_executes_dependencies_even_when_plan_order_is_reversed(self) -> None:
        original = PlannerAgent().plan(QUERY, plan_id="PLAN_REVERSED")
        reversed_plan = TaskPlan(
            plan_id=original.plan_id,
            objective=original.objective,
            tasks=list(reversed(original.tasks)),
        )
        job = RCAJob(
            job_id="JOB_REVERSED",
            user_query=QUERY,
            product_id="40N_SOC",
            time_window={"start_date": "2026-07-01", "end_date": "2026-07-31"},
        )

        state = self.workflow.supervisor.execute(job, reversed_plan)

        self.assertEqual(state.job.status, TaskStatus.COMPLETED.value)
        self.assertEqual(set(state.completed_task_ids), {task.task_id for task in original.tasks})
        self.assertTrue(
            all(task.status == TaskStatus.COMPLETED.value for task in state.task_plan.tasks)
        )
        self.assertIsNone(state.current_task_id)

    def test_supervisor_maintains_complete_traceable_rca_state(self) -> None:
        state = self.workflow.run(QUERY, job_id="JOB_SUPERVISOR_STATE")

        self.assertIsInstance(state, RCAState)
        self.assertEqual(
            {finding.agent for finding in state.findings},
            {
                AgentKind.MES.value,
                AgentKind.FDC.value,
                AgentKind.DEFECT_WAT.value,
                AgentKind.KNOWLEDGE.value,
                AgentKind.RCA_REASONING.value,
                AgentKind.IMPROVEMENT.value,
            },
        )
        self.assertEqual(len(state.affected_lots), 20)
        self.assertEqual(len(state.hypotheses), 1)
        self.assertEqual(
            state.hypotheses[0].root_cause,
            "CMP_CU03_CH02 slurry delivery degradation",
        )
        self.assertIsNotNone(state.report)
        improvement = next(
            finding for finding in state.findings if finding.agent == AgentKind.IMPROVEMENT.value
        )
        self.assertTrue(improvement.details["recommendations"]["corrective_actions"])
        self.assertTrue(improvement.details["requires_two_engineer_approval"])
        known_evidence = {item.evidence_id for item in state.evidence}
        for finding in state.findings:
            self.assertTrue(set(finding.evidence_ids) <= known_evidence)
            self.assertEqual(
                finding.evidence_ids,
                [evidence.evidence_id for evidence in finding.evidence],
            )
            self.assertEqual(
                [evidence.to_dict() for evidence in finding.evidence],
                finding.details["evidence"],
            )
        self.assertTrue(set(state.report.cited_evidence_ids) <= known_evidence)

    def test_current_state_drops_stale_missing_specialist_warning(self) -> None:
        stale = Warning(
            warning_id="WARN_RCA_MISSING_FINDINGS",
            message="Missing Specialist findings: knowledge.",
        )
        persistent = Warning(
            warning_id="WARN_PERSISTENT_CONTRACT",
            message="A non-derived warning remains active.",
        )
        current_findings = [
            AgentFinding(
                finding_id=f"FINDING_{agent}",
                agent=agent,
                summary=f"{agent} finding is now present.",
                confidence=0.8,
                evidence_ids=[f"EV_{agent.upper()}"],
            )
            for agent in (
                AgentKind.MES.value,
                AgentKind.FDC.value,
                AgentKind.DEFECT_WAT.value,
                AgentKind.KNOWLEDGE.value,
            )
        ]

        warnings = _merge_warnings(
            [stale, persistent],
            [],
            current_findings=current_findings,
        )

        assert {item.warning_id for item in warnings} == {
            "WARN_PERSISTENT_CONTRACT"
        }

    def test_current_state_keeps_missing_specialist_warning_until_complete(self) -> None:
        missing = Warning(
            warning_id="WARN_RCA_MISSING_FINDINGS",
            message="Missing Specialist findings: knowledge.",
        )
        incomplete_findings = [
            AgentFinding(
                finding_id=f"FINDING_{agent}",
                agent=agent,
                summary=f"{agent} finding is present.",
                confidence=0.8,
                evidence_ids=[f"EV_{agent.upper()}"],
            )
            for agent in (
                AgentKind.MES.value,
                AgentKind.FDC.value,
                AgentKind.DEFECT_WAT.value,
            )
        ]

        warnings = _merge_warnings(
            [missing],
            [],
            current_findings=incomplete_findings,
        )

        assert [item.warning_id for item in warnings] == [
            "WARN_RCA_MISSING_FINDINGS"
        ]

    def test_supervisor_rejects_non_executable_registered_kind(self) -> None:
        invalid_plan = TaskPlan(
            plan_id="PLAN_INVALID_SUPERVISOR",
            objective="Attempt to execute Planner as a task.",
            tasks=[
                AgentTask(
                    task_id="task_invalid",
                    agent=AgentKind.PLANNER.value,
                    objective="Invalid runtime task.",
                )
            ],
        )
        job = RCAJob(job_id="JOB_INVALID_SUPERVISOR", user_query="Invalid task plan.")

        with self.assertRaises(SupervisorExecutionError):
            self.workflow.supervisor.execute(job, invalid_plan)

    def test_supervisor_has_no_repository_or_tool_dependency(self) -> None:
        import yield_rca_core.supervisor as supervisor

        source = inspect.getsource(supervisor).lower()
        forbidden_dependencies = (
            "yield_rca_core.repositories",
            "yield_rca_core.tool_layer",
            "csvfabrepository",
            "postgresfabrepository",
            "psycopg",
            "sqlalchemy",
            ".rows(",
        )
        for dependency in forbidden_dependencies:
            self.assertNotIn(dependency, source)


if __name__ == "__main__":
    unittest.main()
