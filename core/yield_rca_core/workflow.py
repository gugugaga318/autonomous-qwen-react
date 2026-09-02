"""Pure Python composition root for the Yield RCA workflow."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

from yield_rca_core.decision_evaluation import evaluate as evaluate_agent_decisions
from yield_rca_core.hybrid_retrieval import KnowledgeLookupRetriever
from yield_rca_core.improvement_agent import ImprovementAgent
from yield_rca_core.intent_planner import QwenIntentPlanner, QwenIntentPlannerError
from yield_rca_core.investigation_models import OrchestrationMode
from yield_rca_core.knowledge_retrieval import (
    KnowledgeAssetRepository,
    TypedKnowledgeRetrieverAdapter,
)
from yield_rca_core.llm_gateway import (
    LLMCallError,
    LLMClient,
    LLMSettings,
    build_llm_client,
    capture_llm_usage,
)
from yield_rca_core.models import InvestigationMode, RCAJob, RCAState
from yield_rca_core.next_action_planner import QwenNextActionPlanner
from yield_rca_core.planner_agent import PlannerAgent
from yield_rca_core.rca_reasoning_agent import RCAReasoningAgent
from yield_rca_core.report_generator import ReportGenerator
from yield_rca_core.repositories import CsvFabRepository, FabRepository, PostgresFabRepository
from yield_rca_core.specialist_agents import DefectWATAgent, FDCAgent, KnowledgeAgent, MESAgent
from yield_rca_core.specialist_v2 import SpecialistV2Executor
from yield_rca_core.supervisor import Supervisor
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
    capture_tool_latencies,
)
from yield_rca_core.workflow_events import emit_workflow_event


def _job_from_goal(
    known_facts: dict[str, object],
    *,
    user_query: str,
    job_id: str,
    explicit_lot_id: str | None,
    declared_unavailable_sources: tuple[str, ...] = (),
) -> RCAJob:
    source_lot_id = str(
        explicit_lot_id or known_facts.get("lot_id") or ""
    ).strip().upper()
    raw_time_window = known_facts.get("time_window")
    time_window = (
        {str(key): str(value) for key, value in raw_time_window.items()}
        if isinstance(raw_time_window, dict)
        else {
            key: str(known_facts[key])
            for key in (
                "start",
                "end",
                "start_date",
                "end_date",
                "month",
                "label",
            )
            if known_facts.get(key) is not None
        }
    )
    product_id = str(known_facts.get("product_id") or "").strip()
    return RCAJob(
        job_id=job_id,
        user_query=user_query,
        investigation_mode=(
            InvestigationMode.LOT.value
            if source_lot_id
            else InvestigationMode.PRODUCT_WINDOW.value
        ),
        source_lot_id=source_lot_id or None,
        product_id=product_id or None,
        time_window=time_window,
        declared_unavailable_sources=list(declared_unavailable_sources),
    )


@dataclass(frozen=True)
class PurePythonRCAWorkflow:
    """Plan and execute one RCA job without HTTP or frontend dependencies."""

    planner: PlannerAgent
    supervisor: Supervisor
    llm_settings: LLMSettings
    orchestration_mode: str = OrchestrationMode.FIXED.value
    intent_planner: QwenIntentPlanner | None = None
    next_action_planner: QwenNextActionPlanner | None = None

    def run(
        self,
        user_query: str,
        *,
        job_id: str,
        plan_id: str | None = None,
        lot_id: str | None = None,
        orchestration_mode_override: str | None = None,
        declared_unavailable_sources: tuple[str, ...] = (),
    ) -> RCAState:
        active_orchestration_mode = orchestration_mode_override or self.orchestration_mode
        OrchestrationMode(active_orchestration_mode)
        started = perf_counter()
        with capture_llm_usage() as llm_usage, capture_tool_latencies() as tool_latencies:
            if active_orchestration_mode == OrchestrationMode.LLM_REACT.value:
                if self.intent_planner is None or self.next_action_planner is None:
                    raise ValueError("llm_react requires configured Qwen planners")
                try:
                    intent_outcome = self.intent_planner.plan_with_diagnostics(
                        user_query,
                        lot_id=lot_id,
                    )
                    intent_plan = intent_outcome.plan
                    emit_workflow_event(
                        "investigation_planned",
                        {
                            "mode": active_orchestration_mode,
                            "goal_id": intent_plan.goal.goal_id,
                            "intent": intent_plan.goal.intent,
                            "summary": intent_plan.goal.summary,
                            "question_count": len(intent_plan.questions),
                            "max_steps": intent_plan.goal.max_steps,
                            "max_tool_calls": intent_plan.goal.max_tool_calls,
                        },
                    )
                except (QwenIntentPlannerError, LLMCallError) as exc:
                    goal = (
                        exc.fallback_plan.goal
                        if isinstance(exc, QwenIntentPlannerError)
                        else self.planner.plan_investigation_goal(
                            user_query,
                            lot_id=lot_id,
                        )
                    )
                    emit_workflow_event(
                        "investigation_planned",
                        {
                            "mode": "controlled_react",
                            "goal_id": goal.goal_id,
                            "intent": goal.intent,
                            "summary": goal.summary,
                            "question_count": 0,
                            "max_steps": goal.max_steps,
                            "max_tool_calls": goal.max_tool_calls,
                            "fallback_from": "llm_react",
                        },
                    )
                    job = _job_from_goal(
                        goal.known_facts,
                        user_query=user_query,
                        job_id=job_id,
                        explicit_lot_id=lot_id,
                        declared_unavailable_sources=declared_unavailable_sources,
                    )
                    state = self.supervisor.execute_controlled(
                        job,
                        goal,
                        tool_latencies=tool_latencies,
                        orchestration_requested_mode=active_orchestration_mode,
                    )
                    state = replace(
                        state,
                        execution_metadata={
                            **state.execution_metadata,
                            "orchestration_requested_mode": "llm_react",
                            "orchestration_mode": "controlled_react",
                            "orchestration_fallback_reason": (
                                "qwen_intent_output_invalid"
                                if isinstance(exc, QwenIntentPlannerError)
                                else "qwen_intent_call_failed"
                            ),
                            "orchestration_fallback_stage": "intent_planning",
                            "orchestration_fallback_after_action_count": 0,
                            **(
                                {
                                    "orchestration_fallback_failure_category": (
                                        "planner_output_invalid"
                                    ),
                                    "orchestration_fallback_attempt_count": exc.attempts,
                                    "orchestration_fallback_validation_errors": list(
                                        exc.validation_errors
                                    ),
                                    "intent_planner_attempt_diagnostics": [
                                        diagnostic.to_dict()
                                        for diagnostic in exc.attempt_diagnostics
                                    ],
                                }
                                if isinstance(exc, QwenIntentPlannerError)
                                else {}
                            ),
                        },
                    )
                else:
                    job = _job_from_goal(
                        intent_plan.goal.known_facts,
                        user_query=user_query,
                        job_id=job_id,
                        explicit_lot_id=lot_id,
                        declared_unavailable_sources=declared_unavailable_sources,
                    )
                    state = self.supervisor.execute_llm_react(
                        job,
                        intent_plan,
                        self.next_action_planner,
                        tool_latencies=tool_latencies,
                    )
                    state = replace(
                        state,
                        execution_metadata={
                            **state.execution_metadata,
                            "intent_planner_attempt_diagnostics": [
                                diagnostic.to_dict()
                                for diagnostic in intent_outcome.attempt_diagnostics
                            ],
                        },
                    )
            else:
                task_plan = self.planner.plan(
                    user_query,
                    plan_id=plan_id,
                    lot_id=lot_id,
                )
                emit_workflow_event(
                    "investigation_planned",
                    {
                        "mode": active_orchestration_mode,
                        "plan_id": task_plan.plan_id,
                        "summary": task_plan.objective,
                        "task_count": len(task_plan.tasks),
                        "agents": [task.agent for task in task_plan.tasks],
                    },
                )
                mes_task = next(task for task in task_plan.tasks if task.agent == "mes")
                raw_window = mes_task.inputs.get("time_window", {})
                time_window = (
                    {str(key): str(value) for key, value in raw_window.items()}
                    if isinstance(raw_window, dict)
                    else {}
                )
                product_id = mes_task.inputs.get("product_id")
                source_lot_id = mes_task.inputs.get("lot_id")
                investigation_mode = str(
                    mes_task.inputs.get(
                        "investigation_mode",
                        InvestigationMode.PRODUCT_WINDOW.value,
                    )
                )
                job = RCAJob(
                    job_id=job_id,
                    user_query=user_query,
                    investigation_mode=investigation_mode,
                    source_lot_id=str(source_lot_id) if source_lot_id else None,
                    product_id=str(product_id) if product_id else None,
                    time_window=time_window,
                    declared_unavailable_sources=list(
                        declared_unavailable_sources
                    ),
                )
                if active_orchestration_mode == OrchestrationMode.CONTROLLED_REACT.value:
                    goal = self.planner.plan_investigation_goal(user_query, lot_id=lot_id)
                    emit_workflow_event(
                        "investigation_planned",
                        {
                            "mode": active_orchestration_mode,
                            "goal_id": goal.goal_id,
                            "intent": goal.intent,
                            "summary": goal.summary,
                            "question_count": 0,
                            "max_steps": goal.max_steps,
                            "max_tool_calls": goal.max_tool_calls,
                        },
                    )
                    state = self.supervisor.execute_controlled(
                        job,
                        goal,
                        tool_latencies=tool_latencies,
                    )
                else:
                    state = self.supervisor.execute(job, task_plan)

        total_tokens = sum(event.total_tokens for event in llm_usage)
        llm_latency_ms = round(sum(event.latency_ms for event in llm_usage), 3)
        actual_orchestration_mode = str(
            state.execution_metadata.get(
                "orchestration_mode",
                active_orchestration_mode,
            )
        )
        metadata = {
            **state.execution_metadata,
            "agent_mode": self.llm_settings.agent_mode,
            "provider": (
                self.supervisor.llm_client.provider if self.supervisor.llm_client else None
            ),
            "model": self.supervisor.llm_client.model if self.supervisor.llm_client else None,
            "prompt_version": (self.planner.prompt_version if self.supervisor.llm_client else None),
            "total_tokens": total_tokens,
            "llm_call_count": len(llm_usage),
            "llm_latency_ms": llm_latency_ms,
            "tool_call_count": len(tool_latencies),
            "tool_latency_ms": round(
                sum(float(record["duration_ms"]) for record in tool_latencies),
                3,
            ),
            "tool_latencies": list(tool_latencies),
            "workflow_duration_ms": round((perf_counter() - started) * 1000.0, 3),
            "reasoning_engine": "hypothesis_v1",
            "hypothesis_engine_mode": "active",
            "orchestration_mode": actual_orchestration_mode,
        }
        final_state = RCAState.from_dict(
            {
                **state.to_dict(),
                "llm_usage": [event.to_dict() for event in llm_usage],
                "execution_metadata": metadata,
            }
        )
        return replace(
            final_state,
            run_evaluation=evaluate_agent_decisions(final_state),
        )


def build_workflow(
    repository: FabRepository,
    *,
    llm_settings: LLMSettings | None = None,
    llm_client: LLMClient | None = None,
    orchestration_mode: str | None = None,
    knowledge_retriever: KnowledgeLookupRetriever | None = None,
    knowledge_asset_repository: KnowledgeAssetRepository | None = None,
) -> PurePythonRCAWorkflow:
    """Assemble Tools, Agents, Supervisor, and Planner around a Repository."""

    settings = llm_settings or LLMSettings.from_env()
    selected_orchestration_mode = (
        orchestration_mode
        or os.getenv("YIELD_RCA_ORCHESTRATION_MODE")
        or OrchestrationMode.FIXED.value
    ).strip()
    try:
        OrchestrationMode(selected_orchestration_mode)
    except ValueError as exc:
        raise ValueError(
            "YIELD_RCA_ORCHESTRATION_MODE must be fixed, controlled_react, or llm_react"
        ) from exc
    shared_llm_client = llm_client if llm_client is not None else build_llm_client(settings)
    if (
        selected_orchestration_mode == OrchestrationMode.LLM_REACT.value
        and shared_llm_client is None
    ):
        raise ValueError(
            "llm_react requires YIELD_RCA_AGENT_MODE=fake or llm and an LLM client"
        )
    find_affected_lots_tool = FindAffectedLotsTool(repository)
    analyze_lot_genealogy_tool = AnalyzeLotGenealogyTool(repository)
    get_lot_context_tool = GetLotContextTool(repository)
    find_impact_lots_tool = FindImpactLotsTool(repository)
    analyze_parameter_shift_tool = AnalyzeParameterShiftTool(repository)
    find_ooc_events_tool = FindOocEventsTool(repository)
    perform_basic_spc_analysis_tool = PerformBasicSpcAnalysisTool(repository)
    summarize_defect_wat_tool = SummarizeDefectWatTool(repository)
    retrieve_similar_case_tool = RetrieveSimilarCaseTool(
        repository,
        retriever=(
            TypedKnowledgeRetrieverAdapter(
                repository,
                knowledge_retriever,
                additional_asset_repositories=(
                    (knowledge_asset_repository,)
                    if knowledge_asset_repository is not None
                    else ()
                ),
            )
            if knowledge_retriever is not None
            else None
        ),
    )
    advanced_spc_tool = (
        AnalyzeSpcEvidenceTool(repository) if repository.rows("spc_baseline_profile") else None
    )
    mes_agent = MESAgent(
        find_affected_lots_tool=find_affected_lots_tool,
        analyze_lot_genealogy_tool=analyze_lot_genealogy_tool,
        get_lot_context_tool=get_lot_context_tool,
        find_impact_lots_tool=find_impact_lots_tool,
    )
    fdc_agent = FDCAgent(
        analyze_parameter_shift_tool=analyze_parameter_shift_tool,
        find_ooc_events_tool=find_ooc_events_tool,
        perform_basic_spc_analysis_tool=perform_basic_spc_analysis_tool,
        analyze_spc_evidence_tool=advanced_spc_tool,
    )
    defect_wat_agent = DefectWATAgent(
        summarize_defect_wat_tool=summarize_defect_wat_tool
    )
    knowledge_agent = KnowledgeAgent(
        retrieve_similar_case_tool=retrieve_similar_case_tool
    )
    specialist_v2_executor = (
        SpecialistV2Executor(
            llm_client=shared_llm_client,
            agent_mode=settings.agent_mode,
            direct_single_candidate=settings.agent_mode == "llm",
            llm_analysis_enabled=settings.agent_mode != "llm",
            find_affected_lots_tool=find_affected_lots_tool,
            get_lot_context_tool=get_lot_context_tool,
            find_impact_lots_tool=find_impact_lots_tool,
            analyze_lot_genealogy_tool=analyze_lot_genealogy_tool,
            analyze_parameter_shift_tool=analyze_parameter_shift_tool,
            find_ooc_events_tool=find_ooc_events_tool,
            perform_basic_spc_analysis_tool=perform_basic_spc_analysis_tool,
            analyze_spc_evidence_tool=advanced_spc_tool,
            summarize_defect_wat_tool=summarize_defect_wat_tool,
            retrieve_similar_case_tool=retrieve_similar_case_tool,
        )
        if shared_llm_client is not None
        else None
    )
    supervisor = Supervisor(
        mes_agent=mes_agent,
        fdc_agent=fdc_agent,
        defect_wat_agent=defect_wat_agent,
        knowledge_agent=knowledge_agent,
        rca_reasoning_agent=RCAReasoningAgent(
            llm_client=shared_llm_client,
            agent_mode=settings.agent_mode,
        ),
        improvement_agent=ImprovementAgent(
            llm_client=shared_llm_client,
            agent_mode=settings.agent_mode,
        ),
        report_generator=ReportGenerator(),
        llm_client=shared_llm_client,
        agent_mode=settings.agent_mode,
        specialist_v2_executor=specialist_v2_executor,
    )
    return PurePythonRCAWorkflow(
        planner=PlannerAgent(
            llm_client=shared_llm_client,
            agent_mode=settings.agent_mode,
        ),
        supervisor=supervisor,
        llm_settings=settings,
        orchestration_mode=selected_orchestration_mode,
        intent_planner=(
            QwenIntentPlanner(shared_llm_client)
            if shared_llm_client is not None
            else None
        ),
        next_action_planner=(
            QwenNextActionPlanner(shared_llm_client)
            if shared_llm_client is not None
            else None
        ),
    )


def build_csv_workflow(
    seed_dir: Path,
    *,
    llm_settings: LLMSettings | None = None,
    llm_client: LLMClient | None = None,
    orchestration_mode: str | None = None,
    knowledge_retriever: KnowledgeLookupRetriever | None = None,
    knowledge_asset_repository: KnowledgeAssetRepository | None = None,
) -> PurePythonRCAWorkflow:
    """Build a workflow over an existing offline seed dataset."""

    return build_workflow(
        CsvFabRepository(seed_dir),
        llm_settings=llm_settings,
        llm_client=llm_client,
        orchestration_mode=orchestration_mode,
        knowledge_retriever=knowledge_retriever,
        knowledge_asset_repository=knowledge_asset_repository,
    )


def build_postgres_workflow(
    database_url: str,
    *,
    llm_settings: LLMSettings | None = None,
    llm_client: LLMClient | None = None,
    orchestration_mode: str | None = None,
    knowledge_retriever: KnowledgeLookupRetriever | None = None,
    knowledge_asset_repository: KnowledgeAssetRepository | None = None,
) -> PurePythonRCAWorkflow:
    """Build a workflow over a previously seeded PostgreSQL database."""

    return build_workflow(
        PostgresFabRepository(database_url),
        llm_settings=llm_settings,
        llm_client=llm_client,
        orchestration_mode=orchestration_mode,
        knowledge_retriever=knowledge_retriever,
        knowledge_asset_repository=knowledge_asset_repository,
    )
