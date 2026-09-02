import { describe, expect, it } from "vitest";

import {
  authoritativeHypothesisFor,
  authoritativeRcaDiagnosisFor,
  authoritativeRcaFindingFor,
  buildRCAJobRequest,
  formatAgentName,
  formatTraceLabel,
  getDefaultLotId,
  getEvidenceChain,
  getTargetChamber,
  getYieldTrend,
  selectAgentTrace,
} from "./selectors";
import type {
  ActionRecord,
  AgentFinding,
  DecisionEvaluation,
  Evidence,
  InvestigationAction,
  InvestigationQuestion,
  PlannerDecision,
  RCAState,
} from "./types";

function stateFixture(): RCAState {
  return {
    job: {
      job_id: "RCA_TEST",
      user_query: "Analyze yield drop.",
      investigation_mode: "product_window",
      source_lot_id: null,
      product_id: "40N_SOC",
      time_window: {},
      status: "completed",
      created_at: "2026-07-20T00:00:00+00:00",
    },
    task_plan: null,
    current_task_id: null,
    completed_task_ids: [],
    affected_lots: ["LOT_A_001"],
    impact_lots: [],
    affected_wafers: [],
    impact_wafers: [],
    scope_level: "lot",
    impact_criteria: {},
    evidence: [],
    findings: [
      {
        finding_id: "MES_FINDING",
        task_id: "task_mes",
        agent: "mes",
        finding_kind: "specialist_observation",
        summary: "MES finding",
        confidence: 0.95,
        evidence_ids: ["EV_MES"],
        warnings: [],
        details: {
          target_commonality: { chamber_id: "CMP_CU03_CH02" },
          yield_trend: [
            {
              date: "2026-07-05",
              lot_count: 7,
              pass_count: 7,
              fail_count: 0,
              pass_rate: 100,
            },
          ],
        },
      },
      {
        finding_id: "RCA_FINDING",
        task_id: "task_rca",
        agent: "rca_reasoning",
        finding_kind: "hypothesis_ranking",
        summary: "RCA finding",
        confidence: 0.95,
        evidence_ids: ["EV_MES"],
        warnings: [],
        details: {
          evidence_chain: [
            {
              stage: "mes",
              claim: "Affected lots share one chamber.",
              confidence: 0.95,
              evidence_ids: ["EV_MES"],
            },
          ],
        },
      },
    ],
    hypotheses: [],
    warnings: [],
    report: null,
    llm_usage: [],
    execution_metadata: {
      agent_mode: "deterministic",
      total_tokens: 0,
      tool_call_count: 6,
    },
  };
}

function traceStateFixture(): RCAState {
  const state = stateFixture();
  state.findings = [];
  state.evidence = [];
  state.execution_metadata = {
    agent_mode: "llm",
    orchestration_requested_mode: "llm_react",
    orchestration_mode: "llm_react",
    tool_latencies: [],
  };
  state.investigation_goal = {
    goal_id: "GOAL_TEST",
    intent: "root_cause",
    summary: "Find the root cause.",
    known_facts: {},
    required_evidence: ["process", "quality"],
    max_steps: 6,
    max_tool_calls: 12,
  };
  state.investigation_questions = [];
  state.action_history = [];
  state.planner_decisions = [];
  state.run_evaluation = null;
  return state;
}

function evidenceFixture(evidenceId: string): Evidence {
  return {
    evidence_id: evidenceId,
    source_type: "table",
    source_id: evidenceId,
    summary: `Evidence ${evidenceId}`,
    source_table: "fixture",
    source_field: "value",
    timestamp: null,
    metadata: {},
  };
}

function questionFixture(questionId: string): InvestigationQuestion {
  return {
    question_id: questionId,
    goal_id: "GOAL_TEST",
    question: `Question ${questionId}?`,
    rationale: "Needed for the investigation.",
    scope: {},
    status: "open",
    answer: null,
    evidence_ids: [],
    unavailable_reason: null,
  };
}

function actionFixture(
  actionId: string,
  agent = "defect_wat",
  kind = "inspect_defect_pattern",
): InvestigationAction {
  return {
    action_id: actionId,
    kind,
    agent,
    reason: `Run ${kind}.`,
    inputs: {},
    scope: { lot_ids: ["LOT_A_001"] },
    required_evidence_ids: [],
    max_attempts: 1,
  };
}

function actionRecordFixture(
  action: InvestigationAction,
  findingIds: string[] = [],
  evidenceIds: string[] = [],
): ActionRecord {
  return {
    action,
    status: "completed",
    produced_finding_ids: findingIds,
    produced_evidence_ids: evidenceIds,
    decision_summary: `Completed ${action.action_id}.`,
  };
}

function findingFixture(
  findingId: string,
  agent: string,
  evidenceIds: string[] = [],
  details: Record<string, unknown> = {},
): AgentFinding {
  return {
    finding_id: findingId,
    task_id: null,
    agent,
    finding_kind:
      agent === "rca_reasoning"
        ? "hypothesis_ranking"
        : "specialist_observation",
    summary: `Finding ${findingId}`,
    confidence: 0.8,
    evidence_ids: evidenceIds,
    details,
    warnings: [],
  };
}

function actDecisionFixture(
  decisionId: string,
  action: InvestigationAction,
  targetQuestionId: string,
): PlannerDecision {
  return {
    decision_id: decisionId,
    goal_id: "GOAL_TEST",
    decision_type: "act",
    reason: `Choose ${action.action_id}.`,
    goal_status: "in_progress",
    proposed_conclusion_level: "candidate",
    next_action: action,
    target_question_ids: [targetQuestionId],
    new_questions: [],
    question_updates: [],
    stop_reason: null,
  };
}

function stopDecisionFixture(decisionId = "DECISION_STOP"): PlannerDecision {
  return {
    decision_id: decisionId,
    goal_id: "GOAL_TEST",
    decision_type: "stop",
    reason: "The investigation boundary has been reached.",
    goal_status: "satisfied",
    proposed_conclusion_level: "supported",
    next_action: null,
    target_question_ids: [],
    new_questions: [],
    question_updates: [],
    stop_reason: "goal_satisfied",
  };
}

function decisionEvaluationFixture(
  decisionId: string,
  newEvidenceIds: string[] = [],
  redundant = false,
): DecisionEvaluation {
  return {
    decision_id: decisionId,
    decision_valid: true,
    evidence_gain: newEvidenceIds.length > 0,
    redundant,
    reason:
      newEvidenceIds.length > 0
        ? "The action added Evidence."
        : "The decision was valid without adding Evidence.",
    new_evidence_ids: newEvidenceIds,
  };
}

function specialistStepFixture({
  stepId,
  stepIndex,
  action,
  toolName,
  evidenceIds,
}: {
  stepId: string;
  stepIndex: number;
  action: InvestigationAction;
  toolName: string;
  evidenceIds: string[];
}): Record<string, unknown> {
  return {
    step_id: stepId,
    step_index: stepIndex,
    action_id: action.action_id,
    agent: action.agent,
    decision_id: `${action.action_id}:specialist-decision-${stepIndex}`,
    candidate_id: `${action.action_id}:candidate:${toolName}`,
    tool_name: toolName,
    parameters: { lot_ids: ["LOT_A_001"] },
    reason: `Use ${toolName}.`,
    evidence_ids: evidenceIds,
    output_summary: `${toolName} completed.`,
    status: "completed",
  };
}

function specialistDetailsFixture(
  action: InvestigationAction,
  toolSteps: Record<string, unknown>[],
  options: {
    fallbackReason?: string | null;
    stopReason?: string;
    supersededStepIds?: string[];
  } = {},
): Record<string, unknown> {
  return {
    engineering_interpretation: `Engineering interpretation for ${action.action_id}.`,
    specialist_v2: {
      version: "v2",
      action_id: action.action_id,
      agent: action.agent,
      tool_steps: toolSteps,
      tool_call_count: toolSteps.length,
      stop_reason: options.stopReason ?? "required_local_evidence_collected",
      analysis_source: "qwen",
      fallback_reason: options.fallbackReason ?? null,
      validation_retry_count: 0,
      superseded_step_ids: options.supersededStepIds ?? [],
    },
  };
}

describe("RCAState display selectors", () => {
  it("submits both the Lot ID and investigation request for controlled investigations", () => {
    expect(
      buildRCAJobRequest(
        "lot",
        "  Investigate scratch in Cu CMP.  ",
        " lot_a_001 ",
      ),
    ).toEqual({
      investigation_mode: "lot",
      lot_id: "LOT_A_001",
      user_query: "Investigate scratch in Cu CMP.",
    });
  });

  it("keeps product-window requests free of a stale Lot ID", () => {
    expect(buildRCAJobRequest("product_window", " Analyze yield drop. ", "LOT_A_001")).toEqual({
      investigation_mode: "product_window",
      user_query: "Analyze yield drop.",
    });
  });

  it("selects the valid demo Lot for the active dataset", () => {
    expect(getDefaultLotId("golden_case")).toBe("LOT_A_001");
    expect(getDefaultLotId("multi_case")).toBe("LOT_A_015");
    expect(getDefaultLotId("spc_case")).toBe("LOT_A_015");
  });

  it("reads backend-provided yield points without recalculating them", () => {
    expect(getYieldTrend(stateFixture())).toEqual([
      {
        date: "2026-07-05",
        lot_count: 7,
        pass_count: 7,
        fail_count: 0,
        pass_rate: 100,
      },
    ]);
  });

  it("reads target chamber and evidence chain from agent findings", () => {
    const state = stateFixture();
    expect(getTargetChamber(state)).toBe("CMP_CU03_CH02");
    expect(getEvidenceChain(state)[0].evidence_ids).toEqual(["EV_MES"]);
  });

  it("prefers finding_kind over the first matching agent finding", () => {
    const state = stateFixture();
    state.findings.unshift({
      finding_id: "RCA_DRAFT",
      task_id: "task_rca_draft",
      agent: "rca_reasoning",
      finding_kind: "hypothesis_generation",
      summary: "Draft RCA finding",
      confidence: 0.4,
      evidence_ids: ["EV_DRAFT"],
      warnings: [],
      details: {
        evidence_chain: [
          {
            stage: "mes",
            claim: "Draft claim.",
            confidence: 0.4,
            evidence_ids: ["EV_DRAFT"],
          },
        ],
      },
    });

    expect(getEvidenceChain(state)[0].evidence_ids).toEqual(["EV_MES"]);
  });

  it("uses the explicit authoritative RCA finding when reasoning is iterative", () => {
    const state = stateFixture();
    const current = state.findings.find((finding) => finding.finding_id === "RCA_FINDING");
    expect(current).toBeDefined();
    state.findings.unshift({
      ...current!,
      finding_id: "RCA_HISTORY",
      details: {
        evidence_chain: [],
      },
    });
    state.authoritative_rca_finding_id = "RCA_FINDING";

    expect(authoritativeRcaFindingFor(state)?.finding_id).toBe("RCA_FINDING");
    expect(authoritativeHypothesisFor(state)).toBeUndefined();
  });

  it("projects RCA diagnosis only from the authoritative Finding", () => {
    const state = stateFixture();
    const current = state.findings.find((finding) => finding.finding_id === "RCA_FINDING");
    expect(current).toBeDefined();
    current!.details = {
      conclusion_status: "inconclusive",
      ranked_candidates: [],
      confirmation_gate: { status: "inconclusive" },
    };
    state.authoritative_rca_finding_id = "RCA_FINDING";
    expect(authoritativeRcaDiagnosisFor(state)?.finding_id).toBe("RCA_FINDING");

    state.authoritative_rca_finding_id = null;
    state.findings.push({ ...current!, finding_id: "RCA_HISTORY" });
    expect(authoritativeRcaDiagnosisFor(state)).toBeUndefined();
  });

  it("projects typed RCA authority without requiring a legacy Finding", () => {
    const state = stateFixture();
    state.findings = [];
    state.authoritative_rca_finding_id = null;
    state.authoritative_rca_result = {
      result_id: "RCA_RESULT_TYPED_UI",
      source_finding_id: null,
      source_hypothesis_id: null,
      conclusion_status: "supported",
      root_cause_candidate_id: "CANDIDATE_TYPED_UI",
      root_cause: "Typed UI root cause",
      confirmation_status: "supported",
      competition_status: "not_required",
      terminal_reason: "candidate_competition_not_required",
      evidence_refs: ["EV_TYPED_UI"],
      schema_version: "1.0",
    };
    state.impact_publication_result = {
      rca_result_id: "RCA_RESULT_TYPED_UI",
      publication_status: "confirmed",
      confirmed_impact_lots: ["LOT_TYPED_UI"],
      evidence_refs: ["EV_TYPED_UI"],
      schema_version: "1.0",
    };

    const diagnosis = authoritativeRcaDiagnosisFor(state);

    expect(diagnosis?.finding_id).toBeNull();
    expect(diagnosis?.result_id).toBe("RCA_RESULT_TYPED_UI");
    expect(diagnosis?.conclusion_status).toBe("supported");
    expect(diagnosis?.root_cause).toBe("Typed UI root cause");
    expect(diagnosis?.impact_lot_gate.publication_status).toBe("confirmed");
    expect(diagnosis?.impact_lot_gate.confirmed_impact_lots).toEqual(["LOT_TYPED_UI"]);
  });

  it("does not fall back to a stale Finding when typed authority has no source", () => {
    const state = stateFixture();
    const staleFinding = state.findings.find(
      (finding) => finding.finding_id === "RCA_FINDING",
    );
    expect(staleFinding).toBeDefined();
    staleFinding!.details = {
      evidence_chain: [
        {
          stage: "rca_reasoning",
          claim: "Stale UI evidence chain.",
          confidence: 0.8,
          evidence_ids: ["EV_STALE_UI"],
        },
      ],
      ranked_candidates: [{ root_cause: "Stale UI candidate" }],
    };
    state.authoritative_rca_finding_id = staleFinding!.finding_id;
    state.authoritative_rca_result = {
      result_id: "RCA_RESULT_UI_WITHOUT_SOURCE",
      source_finding_id: null,
      source_hypothesis_id: null,
      conclusion_status: "inconclusive",
      root_cause_candidate_id: null,
      root_cause: null,
      confirmation_status: "inconclusive",
      competition_status: "exhausted",
      terminal_reason: "no_high_value_action_remains",
      evidence_refs: [],
      schema_version: "1.0",
    };

    expect(authoritativeRcaFindingFor(state)).toBeUndefined();
    expect(getEvidenceChain(state)).toEqual([]);
    expect(authoritativeRcaDiagnosisFor(state)?.ranked_candidates).toEqual([]);
  });

  it("returns empty display data when optional backend fields are absent", () => {
    const state = stateFixture();
    state.findings = [];
    expect(getYieldTrend(state)).toEqual([]);
    expect(getEvidenceChain(state)).toEqual([]);
    expect(getTargetChamber(state)).toBe("Not available");
  });

  it("formats the Improvement Agent timeline label", () => {
    expect(formatAgentName("improvement")).toBe("Improvement");
  });

  it("preserves engineering acronyms in trace labels", () => {
    expect(formatTraceLabel("inspect_fdc_spc")).toBe("Inspect FDC SPC");
    expect(formatTraceLabel("run_rca_reasoning")).toBe("Run RCA Reasoning");
    expect(formatTraceLabel("lot_ids")).toBe("Lot IDs");
  });
});

describe("Agent trace selector", () => {
  it("joins decisions, actions, findings, questions, Evidence, and evaluations by ID", () => {
    const state = traceStateFixture();
    const defectAction = actionFixture("ACTION_DEFECT");
    const reasoningAction = actionFixture(
      "ACTION_RCA",
      "rca_reasoning",
      "run_rca_reasoning",
    );
    const defectDecision = actDecisionFixture(
      "DECISION_DEFECT",
      defectAction,
      "QUESTION_DEFECT",
    );
    const reasoningDecision = actDecisionFixture(
      "DECISION_RCA",
      reasoningAction,
      "QUESTION_RCA",
    );
    const stopDecision = stopDecisionFixture();

    state.investigation_questions = [
      questionFixture("QUESTION_RCA"),
      questionFixture("QUESTION_DEFECT"),
    ];
    state.evidence = [
      evidenceFixture("EV_UNUSED"),
      evidenceFixture("EV_DEFECT"),
    ];
    state.findings = [
      findingFixture("FINDING_RCA", "rca_reasoning", ["EV_DEFECT"]),
      findingFixture("FINDING_DEFECT", "defect_wat", ["EV_DEFECT"]),
    ];
    state.action_history = [
      actionRecordFixture(reasoningAction, ["FINDING_RCA"]),
      actionRecordFixture(
        defectAction,
        ["FINDING_DEFECT"],
        ["EV_DEFECT"],
      ),
    ];
    state.planner_decisions = [
      defectDecision,
      reasoningDecision,
      stopDecision,
    ];
    state.run_evaluation = {
      goal_id: "GOAL_TEST",
      goal_success: true,
      stop_correct: true,
      summary: "The investigation stopped correctly.",
      decision_evaluations: [
        decisionEvaluationFixture("DECISION_STOP"),
        decisionEvaluationFixture("DECISION_RCA"),
        decisionEvaluationFixture("DECISION_DEFECT", ["EV_DEFECT"]),
      ],
    };

    const trace = selectAgentTrace(state);

    expect(trace.nodes.map((node) => node.decision?.decision_id)).toEqual([
      "DECISION_DEFECT",
      "DECISION_RCA",
      "DECISION_STOP",
    ]);
    expect(trace.nodes[0].actionRecord?.action.action_id).toBe("ACTION_DEFECT");
    expect(trace.nodes[0].targetQuestions[0].question_id).toBe(
      "QUESTION_DEFECT",
    );
    expect(trace.nodes[0].findings[0].finding_id).toBe("FINDING_DEFECT");
    expect(trace.nodes[0].evidence[0].evidence_id).toBe("EV_DEFECT");
    expect(trace.nodes[0].newEvidence[0].evidence_id).toBe("EV_DEFECT");
    expect(trace.nodes[1].actionRecord?.action.action_id).toBe("ACTION_RCA");
    expect(trace.nodes[1].findings[0].finding_id).toBe("FINDING_RCA");
    expect(trace.nodes[1].evaluation).toMatchObject({
      decision_id: "DECISION_RCA",
      evidence_gain: false,
      redundant: false,
    });
    expect(trace.nodes[1].newEvidence).toEqual([]);
    expect(trace.nodes[2]).toMatchObject({
      origin: "llm_react",
      action: null,
      actionRecord: null,
    });
    expect(trace.evaluationStatus).toBe("available");
  });

  it("represents an immediate Planner STOP without inventing an ActionRecord", () => {
    const state = traceStateFixture();
    state.planner_decisions = [
      stopDecisionFixture("DECISION_IMMEDIATE_STOP"),
    ];
    state.run_evaluation = {
      goal_id: "GOAL_TEST",
      goal_success: true,
      stop_correct: true,
      summary: "The known facts already answered the request.",
      decision_evaluations: [
        decisionEvaluationFixture("DECISION_IMMEDIATE_STOP"),
      ],
    };

    const trace = selectAgentTrace(state);

    expect(trace.nodes).toHaveLength(1);
    expect(trace.nodes[0].decision?.decision_type).toBe("stop");
    expect(trace.nodes[0].action).toBeNull();
    expect(trace.nodes[0].actionRecord).toBeNull();
    expect(trace.nodes[0].findings).toEqual([]);
    expect(trace.nodes[0].integrityIssues).toEqual([]);
  });

  it("attaches rejected QuestionUpdate reviews to their committed decision", () => {
    const state = traceStateFixture();
    const action = actionFixture("ACTION_REVIEWED");
    state.investigation_questions = [questionFixture("QUESTION_REVIEWED")];
    state.planner_decisions = [
      actDecisionFixture("DECISION_REVIEWED", action, "QUESTION_REVIEWED"),
    ];
    state.action_history = [actionRecordFixture(action)];
    state.question_update_reviews = [
      {
        decision_id: "DECISION_REVIEWED",
        disposition: "rejected",
        reason_code: "non_terminal_status",
        reason: "QuestionUpdate status=open is not a terminal state change.",
        update_index: 0,
        question_id: "QUESTION_REVIEWED",
        claimed_status: "open",
      },
    ];

    const trace = selectAgentTrace(state);

    expect(trace.nodes[0].questionUpdateReviews).toEqual(
      state.question_update_reviews,
    );
    expect(trace.nodes[0].questionUpdates).toEqual([]);
    expect(trace.nodes[0].integrityIssues).toEqual([]);
  });

  it("reports a QuestionUpdate review whose decision is missing", () => {
    const state = traceStateFixture();
    state.question_update_reviews = [
      {
        decision_id: "DECISION_MISSING",
        disposition: "rejected",
        reason_code: "unknown_question",
        reason: "The model invented a Question outside the current trace.",
        update_index: 0,
        question_id: "QUESTION_INVENTED",
        claimed_status: "closed",
      },
    ];

    const trace = selectAgentTrace(state);

    expect(trace.integrityIssues).toContain(
      "QuestionUpdate review references missing PlannerDecision DECISION_MISSING.",
    );
  });

  it("keeps the Qwen prefix and marks only an unmatched fallback tail", () => {
    const state = traceStateFixture();
    const qwenAction = actionFixture("ACTION_QWEN");
    const fallbackActionOne = actionFixture(
      "ACTION_FALLBACK_MES",
      "mes",
      "find_shared_exposure",
    );
    const fallbackActionTwo = actionFixture(
      "ACTION_FALLBACK_FDC",
      "fdc",
      "inspect_fdc_spc",
    );
    state.execution_metadata.orchestration_mode = "controlled_react";
    state.execution_metadata.orchestration_fallback_reason =
      "next_action_planner_output_invalid";
    state.execution_metadata.orchestration_fallback_stage =
      "next_action_planning";
    state.execution_metadata.orchestration_fallback_after_action_count = 1;
    state.execution_metadata.orchestration_fallback_attempt_count = 2;
    state.execution_metadata.orchestration_fallback_validation_errors = [
      "question_updates[0].status must be closed or unavailable",
      "question_updates[0].status must be closed or unavailable",
    ];
    state.execution_metadata.intent_planner_attempt_diagnostics = [
      {
        stage: "intent_planning",
        attempt: 1,
        prompt_name: "intent_planner",
        prompt_version: "v1",
        outcome: "success",
        failure_category: null,
        reason_code: null,
        field_path: null,
        message: null,
        repair_feedback_sent: false,
        candidate_summary: { intent: "root_cause" },
        baseline_diff: { intent_changed: false },
        provider_request_id: null,
      },
    ];
    state.investigation_questions = [questionFixture("QUESTION_QWEN")];
    state.planner_decisions = [
      actDecisionFixture("DECISION_QWEN", qwenAction, "QUESTION_QWEN"),
    ];
    state.action_history = [
      actionRecordFixture(qwenAction),
      actionRecordFixture(fallbackActionOne),
      actionRecordFixture(fallbackActionTwo),
    ];

    const trace = selectAgentTrace(state);

    expect(trace.nodes.map((node) => node.origin)).toEqual([
      "llm_react",
      "controlled_fallback",
      "controlled_fallback",
    ]);
    expect(
      trace.nodes.map(
        (node) => node.actionRecord?.action.action_id ?? node.action?.action_id,
      ),
    ).toEqual([
      "ACTION_QWEN",
      "ACTION_FALLBACK_MES",
      "ACTION_FALLBACK_FDC",
    ]);
    expect(trace.evaluationStatus).toBe("fallback");
    expect(trace.fallbackStage).toBe("next_action_planning");
    expect(trace.fallbackAfterActionCount).toBe(1);
    expect(trace.fallbackAttemptCount).toBe(2);
    expect(trace.fallbackValidationErrors).toEqual([
      "question_updates[0].status must be closed or unavailable",
      "question_updates[0].status must be closed or unavailable",
    ]);
    expect(trace.intentPlannerAttempts).toHaveLength(1);
    expect(trace.intentPlannerAttempts[0].outcome).toBe("success");
  });

  it("rejects malformed Intent Planner attempt metadata from the trace", () => {
    const state = traceStateFixture();
    state.execution_metadata.intent_planner_attempt_diagnostics = [
      {
        stage: "intent_planning",
        attempt: 0,
        outcome: "failure",
      },
    ] as never;

    const trace = selectAgentTrace(state);

    expect(trace.intentPlannerAttempts).toEqual([]);
    expect(trace.integrityIssues).toContain(
      "Intent Planner attempt diagnostics are malformed.",
    );
  });

  it("degrades a legacy state with absent optional trace fields to TaskPlan nodes", () => {
    const state = stateFixture();
    state.execution_metadata = { agent_mode: "deterministic" };
    state.task_plan = {
      plan_id: "PLAN_LEGACY",
      objective: "Run the legacy workflow.",
      tasks: [
        {
          task_id: "TASK_LEGACY_MES",
          agent: "mes",
          objective: "Inspect MES.",
          depends_on: [],
          status: "completed",
          inputs: {},
          finding_kind: "specialist_observation",
        },
      ],
    };
    state.findings[0].task_id = "TASK_LEGACY_MES";
    state.evidence = [evidenceFixture("EV_MES")];

    const trace = selectAgentTrace(state);

    expect(trace).toMatchObject({
      requestedMode: null,
      actualMode: null,
      evaluationStatus: "not_applicable",
    });
    expect(trace.nodes).toHaveLength(1);
    expect(trace.nodes[0].origin).toBe("legacy");
    expect(trace.nodes[0].task?.task_id).toBe("TASK_LEGACY_MES");
    expect(trace.nodes[0].findings[0].finding_id).toBe("MES_FINDING");
    expect(trace.nodes[0].evidence[0].evidence_id).toBe("EV_MES");
  });

  it("keeps a superseded Advanced SPC step without requiring its Evidence in final State", () => {
    const state = traceStateFixture();
    const action = actionFixture("ACTION_FDC", "fdc", "inspect_fdc_spc");
    const advancedStep = specialistStepFixture({
      stepId: "STEP_ADVANCED",
      stepIndex: 1,
      action,
      toolName: "analyze_spc_evidence",
      evidenceIds: ["EV_ADVANCED_SUPERSEDED"],
    });
    const basicStep = specialistStepFixture({
      stepId: "STEP_BASIC",
      stepIndex: 2,
      action,
      toolName: "perform_basic_spc_analysis",
      evidenceIds: ["EV_BASIC"],
    });
    state.investigation_questions = [questionFixture("QUESTION_FDC")];
    state.planner_decisions = [
      actDecisionFixture("DECISION_FDC", action, "QUESTION_FDC"),
    ];
    state.action_history = [
      actionRecordFixture(action, ["FINDING_FDC"], ["EV_BASIC"]),
    ];
    state.evidence = [evidenceFixture("EV_BASIC")];
    state.findings = [
      findingFixture(
        "FINDING_FDC",
        "fdc",
        ["EV_BASIC"],
        specialistDetailsFixture(action, [advancedStep, basicStep], {
          fallbackReason: "advanced_spc_no_analyzable_parameters",
          stopReason: "advanced_spc_fell_back_to_basic",
          supersededStepIds: ["STEP_ADVANCED"],
        }),
      ),
    ];

    const trace = selectAgentTrace(state);
    const specialist = trace.nodes[0].specialistTraces[0];

    expect(specialist.localFallback).toBe(true);
    expect(specialist.toolSteps[0]).toMatchObject({
      stepId: "STEP_ADVANCED",
      superseded: true,
      evidence: [],
    });
    expect(specialist.toolSteps[1].evidence[0].evidence_id).toBe("EV_BASIC");
    expect(
      trace.nodes[0].integrityIssues.some((issue) =>
        issue.includes("EV_ADVANCED_SUPERSEDED"),
      ),
    ).toBe(false);
  });

  it("joins same-named Tool latency records by exact action-scoped request ID", () => {
    const state = traceStateFixture();
    const firstAction = actionFixture("ACTION_DEFECT_FIRST");
    const secondAction = actionFixture("ACTION_DEFECT_SECOND");
    const firstStep = specialistStepFixture({
      stepId: "STEP_FIRST",
      stepIndex: 1,
      action: firstAction,
      toolName: "summarize_defect_wat",
      evidenceIds: ["EV_FIRST"],
    });
    const secondStep = specialistStepFixture({
      stepId: "STEP_SECOND",
      stepIndex: 1,
      action: secondAction,
      toolName: "summarize_defect_wat",
      evidenceIds: ["EV_SECOND"],
    });
    state.investigation_questions = [
      questionFixture("QUESTION_FIRST"),
      questionFixture("QUESTION_SECOND"),
    ];
    state.planner_decisions = [
      actDecisionFixture("DECISION_FIRST", firstAction, "QUESTION_FIRST"),
      actDecisionFixture("DECISION_SECOND", secondAction, "QUESTION_SECOND"),
    ];
    state.action_history = [
      actionRecordFixture(firstAction, ["FINDING_FIRST"], ["EV_FIRST"]),
      actionRecordFixture(secondAction, ["FINDING_SECOND"], ["EV_SECOND"]),
    ];
    state.evidence = [
      evidenceFixture("EV_SECOND"),
      evidenceFixture("EV_FIRST"),
    ];
    state.findings = [
      findingFixture(
        "FINDING_SECOND",
        "defect_wat",
        ["EV_SECOND"],
        specialistDetailsFixture(secondAction, [secondStep]),
      ),
      findingFixture(
        "FINDING_FIRST",
        "defect_wat",
        ["EV_FIRST"],
        specialistDetailsFixture(firstAction, [firstStep]),
      ),
    ];
    state.execution_metadata.tool_latencies = [
      {
        tool_name: "summarize_defect_wat",
        tool_request_id:
          "RCA_TEST:ACTION_DEFECT_SECOND:specialist-step-1",
        agent: "defect_wat",
        outcome: "success",
        duration_ms: 222,
      },
      {
        tool_name: "summarize_defect_wat",
        tool_request_id:
          "RCA_TEST:ACTION_DEFECT_FIRST:specialist-step-1",
        agent: "defect_wat",
        outcome: "success",
        duration_ms: 111,
      },
    ];

    const trace = selectAgentTrace(state);

    expect(
      trace.nodes[0].specialistTraces[0].toolSteps[0].latency?.duration_ms,
    ).toBe(111);
    expect(
      trace.nodes[1].specialistTraces[0].toolSteps[0].latency?.duration_ms,
    ).toBe(222);
  });

  it("surfaces malformed Specialist V2 data as integrity issues without throwing", () => {
    const state = traceStateFixture();
    const action = actionFixture("ACTION_MALFORMED");
    state.investigation_questions = [questionFixture("QUESTION_MALFORMED")];
    state.planner_decisions = [
      actDecisionFixture(
        "DECISION_MALFORMED",
        action,
        "QUESTION_MALFORMED",
      ),
    ];
    state.action_history = [
      actionRecordFixture(action, ["FINDING_MALFORMED"]),
    ];
    state.findings = [
      findingFixture("FINDING_MALFORMED", "defect_wat", [], {
        specialist_v2: "not-an-object",
      }),
    ];

    const trace = selectAgentTrace(state);

    expect(trace.nodes[0].specialistTraces).toHaveLength(1);
    expect(trace.nodes[0].specialistTraces[0].integrityIssues).toContain(
      "Finding specialist_v2 details are not an object.",
    );
    expect(trace.nodes[0].integrityIssues).toContain(
      "Specialist trace FINDING_MALFORMED: Finding specialist_v2 details are not an object.",
    );
  });

  it("reports duplicate and missing ID joins instead of choosing an ambiguous item", () => {
    const state = traceStateFixture();
    const action = actionFixture("ACTION_BROKEN_JOIN");
    const duplicateEvidence = evidenceFixture("EV_DUPLICATE");
    state.investigation_questions = [questionFixture("QUESTION_BROKEN_JOIN")];
    state.planner_decisions = [
      actDecisionFixture(
        "DECISION_BROKEN_JOIN",
        action,
        "QUESTION_BROKEN_JOIN",
      ),
    ];
    state.action_history = [
      actionRecordFixture(
        action,
        [],
        ["EV_DUPLICATE", "EV_MISSING"],
      ),
    ];
    state.evidence = [
      duplicateEvidence,
      { ...duplicateEvidence, summary: "Conflicting duplicate." },
    ];

    const trace = selectAgentTrace(state);

    expect(trace.integrityIssues).toContain(
      "Duplicate Evidence ID: EV_DUPLICATE.",
    );
    expect(trace.nodes[0].integrityIssues).toEqual(
      expect.arrayContaining([
        "Action Evidence ID EV_DUPLICATE is ambiguous.",
        "Action Evidence ID EV_MISSING is missing.",
      ]),
    );
    expect(trace.nodes[0].evidence).toEqual([]);
  });

  it("does not attach one evaluation or ActionRecord to duplicate Planner IDs", () => {
    const state = traceStateFixture();
    const action = actionFixture("ACTION_DUPLICATE");
    const decision = actDecisionFixture(
      "DECISION_DUPLICATE",
      action,
      "QUESTION_DUPLICATE",
    );
    state.investigation_questions = [
      questionFixture("QUESTION_DUPLICATE"),
    ];
    state.planner_decisions = [
      decision,
      {
        ...decision,
        reason: "A malformed snapshot repeated the committed decision.",
      },
    ];
    state.action_history = [actionRecordFixture(action)];
    state.run_evaluation = {
      goal_id: "GOAL_TEST",
      goal_success: false,
      stop_correct: false,
      summary: "Malformed duplicate snapshot.",
      decision_evaluations: [
        decisionEvaluationFixture("DECISION_DUPLICATE"),
      ],
    };

    const trace = selectAgentTrace(state);

    expect(trace.integrityIssues).toEqual(
      expect.arrayContaining([
        "Duplicate PlannerDecision ID: DECISION_DUPLICATE.",
        "Duplicate Planner Action ID: ACTION_DUPLICATE.",
      ]),
    );
    expect(trace.nodes).toHaveLength(3);
    expect(new Set(trace.nodes.map((node) => node.key)).size).toBe(3);
    expect(trace.nodes[0]).toMatchObject({
      evaluation: null,
      actionRecord: null,
    });
    expect(trace.nodes[1]).toMatchObject({
      evaluation: null,
      actionRecord: null,
    });
    expect(trace.nodes[0].integrityIssues).toEqual(
      expect.arrayContaining([
        "PlannerDecision ID DECISION_DUPLICATE is ambiguous.",
        "Planner Action ID ACTION_DUPLICATE is ambiguous.",
      ]),
    );
    expect(trace.nodes[2].actionRecord?.action.action_id).toBe(
      "ACTION_DUPLICATE",
    );
  });
});
