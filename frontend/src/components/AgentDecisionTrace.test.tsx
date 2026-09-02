import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import type {
  ActionRecord,
  DecisionEvaluation,
  InvestigationAction,
  PlannerDecision,
  RCAState,
} from "../types";
import { AgentDecisionTrace } from "./AgentDecisionTrace";

function stateFixture(): RCAState {
  return {
    job: {
      job_id: "JOB_AGENT_TRACE",
      user_query: "Investigate LOT_A_001 scratch in Cu CMP.",
      investigation_mode: "lot",
      source_lot_id: "LOT_A_001",
      product_id: "40N_SOC",
      time_window: {},
      status: "completed",
      created_at: "2026-07-31T00:00:00+00:00",
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
    findings: [],
    hypotheses: [],
    warnings: [],
    report: null,
    llm_usage: [],
    execution_metadata: {
      orchestration_requested_mode: "llm_react",
      orchestration_mode: "llm_react",
    },
    investigation_goal: {
      goal_id: "GOAL_LOT_01",
      intent: "root_cause",
      summary: "Explain the Cu CMP scratch with bounded fab Evidence.",
      known_facts: {
        lot_id: "LOT_A_001",
        defect: "scratch",
        module: "CU_CMP",
      },
      required_evidence: ["defect_signature", "process_mechanism"],
      max_steps: 8,
      max_tool_calls: 20,
    },
    investigation_questions: [
      {
        question_id: "Q_ROOT_CAUSE",
        goal_id: "GOAL_LOT_01",
        question: "Which mechanism caused the Cu CMP scratch?",
        rationale: "The root-cause claim requires fab Evidence.",
        scope: { lot_id: "LOT_A_001", module: "CU_CMP" },
        status: "closed",
        answer: "The bounded observations support an endpoint mechanism.",
        evidence_ids: ["EV_DEFECT_01"],
        unavailable_reason: null,
      },
    ],
    action_history: [],
    planner_decisions: [],
    run_evaluation: null,
    goal_status: "satisfied",
    conclusion_level: "supported",
    evidence_gaps: [],
    stop_reason: "goal_satisfied",
  };
}

function action(
  actionId: string,
  kind: string,
  agent: string,
): InvestigationAction {
  return {
    action_id: actionId,
    kind,
    agent,
    reason: `Execute ${kind} for the original Lot scope.`,
    inputs: { lot_id: "LOT_A_001" },
    scope: { lot_id: "LOT_A_001", module: "CU_CMP" },
    required_evidence_ids: [],
    max_attempts: 1,
  };
}

function actDecision(
  decisionId: string,
  nextAction: InvestigationAction,
): PlannerDecision {
  return {
    decision_id: decisionId,
    goal_id: "GOAL_LOT_01",
    decision_type: "act",
    reason: `Planner selected ${nextAction.kind} after the latest observation.`,
    goal_status: "in_progress",
    proposed_conclusion_level: "signal",
    next_action: nextAction,
    target_question_ids: ["Q_ROOT_CAUSE"],
    new_questions: [],
    question_updates: [],
    stop_reason: null,
  };
}

function stopDecision(
  decisionId = "DECISION_STOP",
  reason = "The evidence gate reached the requested investigation boundary.",
): PlannerDecision {
  return {
    decision_id: decisionId,
    goal_id: "GOAL_LOT_01",
    decision_type: "stop",
    reason,
    goal_status: "satisfied",
    proposed_conclusion_level: "supported",
    next_action: null,
    target_question_ids: [],
    new_questions: [],
    question_updates: [],
    stop_reason: "goal_satisfied",
  };
}

function actionRecord(
  nextAction: InvestigationAction,
  findingId: string,
  summary: string,
): ActionRecord {
  return {
    action: nextAction,
    status: "completed",
    produced_finding_ids: [findingId],
    produced_evidence_ids: ["EV_DEFECT_01"],
    decision_summary: summary,
  };
}

function evaluation(
  decisionId: string,
  {
    evidenceGain,
    redundant = false,
    newEvidenceIds = [],
    reason,
  }: {
    evidenceGain: boolean;
    redundant?: boolean;
    newEvidenceIds?: string[];
    reason: string;
  },
): DecisionEvaluation {
  return {
    decision_id: decisionId,
    decision_valid: true,
    evidence_gain: evidenceGain,
    redundant,
    reason,
    new_evidence_ids: newEvidenceIds,
  };
}

function populatedTraceState(): RCAState {
  const state = stateFixture();
  const defectAction = action(
    "ACTION_DEFECT",
    "inspect_defect_pattern",
    "defect_wat",
  );
  const reasoningAction = action(
    "ACTION_REASONING",
    "run_rca_reasoning",
    "rca_reasoning",
  );
  state.evidence = [
    {
      evidence_id: "EV_DEFECT_01",
      source_type: "defect",
      source_id: "defect:LOT_A_001",
      summary: "The source Lot shows a directional scratch signature.",
      source_table: "defect_summary",
      source_field: "defect_code",
      timestamp: null,
      metadata: {},
      observation: "Directional scratch signature observed.",
    },
  ];
  state.findings = [
    {
      finding_id: "FINDING_DEFECT",
      agent: "defect_wat",
      finding_kind: "specialist_observation",
      summary: "The scratch signature is confirmed.",
      confidence: 0.92,
      evidence_ids: ["EV_DEFECT_01"],
      details: {},
      warnings: [],
    },
    {
      finding_id: "FINDING_REASONING",
      agent: "rca_reasoning",
      finding_kind: "hypothesis_ranking",
      summary: "The existing Evidence supports the ranked mechanism.",
      confidence: 0.84,
      evidence_ids: ["EV_DEFECT_01"],
      details: {},
      warnings: [],
    },
  ];
  state.action_history = [
    actionRecord(
      defectAction,
      "FINDING_DEFECT",
      "The source Lot scratch signature was confirmed.",
    ),
    actionRecord(
      reasoningAction,
      "FINDING_REASONING",
      "The existing Evidence was ranked without inventing new Evidence.",
    ),
  ];
  state.planner_decisions = [
    actDecision("DECISION_DEFECT", defectAction),
    actDecision("DECISION_REASONING", reasoningAction),
    stopDecision(),
  ];
  state.run_evaluation = {
    goal_id: "GOAL_LOT_01",
    goal_success: true,
    stop_correct: true,
    summary:
      "Goal success is true and the Planner stopped at the evidence boundary.",
    decision_evaluations: [
      evaluation("DECISION_DEFECT", {
        evidenceGain: true,
        newEvidenceIds: ["EV_DEFECT_01"],
        reason: "The defect action introduced one new Evidence ID.",
      }),
      evaluation("DECISION_REASONING", {
        evidenceGain: false,
        reason:
          "RCA reasoning added analytical value while reusing validated Evidence.",
      }),
      evaluation("DECISION_STOP", {
        evidenceGain: false,
        reason: "The stop contract matches the completed investigation.",
      }),
    ],
  };
  return state;
}

describe("AgentDecisionTrace server rendering", () => {
  it("renders a complete autonomous trace and keeps no-gain reasoning non-redundant", () => {
    const html = renderToStaticMarkup(
      <AgentDecisionTrace state={populatedTraceState()} />,
    );

    expect(html).toContain("Autonomous Agent Trace");
    expect(html).toContain(
      "<dt>Goal Success</dt><dd>True</dd>",
    );
    expect(html).toContain(
      "<dt>Stop Correct</dt><dd>True</dd>",
    );
    expect(html).toContain("Run RCA Reasoning");
    expect(html).toContain("Evidence Gain: No");
    expect(html).toContain("Redundant: No");
    expect(html).toContain(
      "RCA reasoning added analytical value while reusing validated Evidence.",
    );
    expect(html).toContain("Stop investigation");
    expect(html).toContain("Goal Satisfied");
    expect(html).toContain(
      "The stop contract matches the completed investigation.",
    );
  });

  it("renders goal failure and a correct stop as independent outcomes", () => {
    const state = populatedTraceState();
    if (!state.run_evaluation) throw new Error("fixture requires evaluation");
    state.run_evaluation = {
      ...state.run_evaluation,
      goal_success: false,
      stop_correct: true,
      summary:
        "The evidence remains inconclusive, but the Planner stopped at the correct boundary.",
    };
    state.conclusion_level = "inconclusive";

    const html = renderToStaticMarkup(<AgentDecisionTrace state={state} />);

    expect(html).toContain(
      "<dt>Goal Success</dt><dd>False</dd>",
    );
    expect(html).toContain(
      "<dt>Stop Correct</dt><dd>True</dd>",
    );
    expect(html).toContain(
      "The evidence remains inconclusive, but the Planner stopped at the correct boundary.",
    );
  });

  it("renders a Finalizer outcome separately from the preserved Planner proposal", () => {
    const state = populatedTraceState();
    state.execution_metadata = {
      ...state.execution_metadata,
      terminal_stop_projection_applied: true,
      terminal_stop_projection_trace: "execution_metadata_only",
      terminal_stop_projected_by: "python_investigation_finalizer",
    };
    state.goal_status = "blocked";
    state.conclusion_level = "inconclusive";
    state.stop_reason = "no_high_value_action";

    const html = renderToStaticMarkup(<AgentDecisionTrace state={state} />);

    expect(html).toContain("Python Finalizer outcome");
    expect(html).toContain(
      "The Planner proposal is preserved separately from the authoritative terminal state.",
    );
    expect(html).toContain("No High Value Action");
  });

  it("renders an immediate stop when the autonomous run has no ActionRecord", () => {
    const state = stateFixture();
    const stop = stopDecision(
      "DECISION_IMMEDIATE_STOP",
      "The model proposed an immediate data-unavailable stop.",
    );
    state.planner_decisions = [stop];
    state.run_evaluation = {
      goal_id: "GOAL_LOT_01",
      goal_success: false,
      stop_correct: false,
      summary:
        "The objective was unanswered and a legal evidence action remained.",
      decision_evaluations: [
        evaluation("DECISION_IMMEDIATE_STOP", {
          evidenceGain: false,
          reason: "The stop was structurally valid but premature.",
        }),
      ],
    };
    state.goal_status = "blocked";
    state.conclusion_level = "inconclusive";
    state.stop_reason = "data_unavailable";

    const html = renderToStaticMarkup(<AgentDecisionTrace state={state} />);

    expect(html).toContain("1 trace nodes");
    expect(html).toContain("Stop investigation");
    expect(html).toContain(
      "The model proposed an immediate data-unavailable stop.",
    );
    expect(html).toContain(
      "<dt>Goal Success</dt><dd>False</dd>",
    );
    expect(html).toContain(
      "<dt>Stop Correct</dt><dd>False</dd>",
    );
    expect(html).not.toContain(
      "No autonomous Planner decisions have been recorded yet.",
    );
  });

  it("renders a mid-loop controlled handoff as neutral and preserves its tail", () => {
    const state = populatedTraceState();
    state.run_evaluation = null;
    state.execution_metadata = {
      orchestration_requested_mode: "llm_react",
      orchestration_mode: "controlled_react",
      orchestration_fallback_reason: "qwen_next_action_output_invalid",
      orchestration_fallback_stage: "next_action_planning",
      orchestration_fallback_after_action_count: 2,
      orchestration_fallback_attempt_count: 2,
      orchestration_fallback_validation_errors: [
        "question_updates[0].status must be closed or unavailable",
        "question_updates[0].answer must be null when status is unavailable",
      ],
    };
    const tailAction = action(
      "ACTION_CONTROLLED_TAIL",
      "inspect_fdc_spc",
      "fdc",
    );
    state.action_history = [
      ...(state.action_history ?? []),
      {
        ...actionRecord(
          tailAction,
          "FINDING_DEFECT",
          "Controlled ReAct continued from the retained observations.",
        ),
        produced_finding_ids: [],
        produced_evidence_ids: [],
      },
    ];

    const html = renderToStaticMarkup(<AgentDecisionTrace state={state} />);

    expect(html).toContain("Planner Trace + Controlled Handoff");
    expect(html).toContain("Controlled compatibility handoff");
    expect(html).toContain("Not evaluated");
    expect(html).toContain(
      "Evaluation is not attributed after the compatibility cutover",
    );
    expect(html).toContain("Controlled fallback");
    expect(html).toContain("Qwen output validation failed on 2 attempts");
    expect(html).toContain("Planner validation diagnostics (2)");
    expect(html).toContain(
      "question_updates[0].status must be closed or unavailable",
    );
    expect(html).toContain("Inspect FDC SPC");
    expect(html).toContain(
      "Controlled ReAct continued from the retained observations.",
    );
    expect(html).not.toContain(
      "<dt>Goal Success</dt><dd>False</dd>",
    );
    expect(html).not.toContain(
      "<dt>Stop Correct</dt><dd>False</dd>",
    );
  });

  it("renders typed Intent Planner attempts before an initial controlled handoff", () => {
    const state = stateFixture();
    state.run_evaluation = null;
    state.execution_metadata = {
      orchestration_requested_mode: "llm_react",
      orchestration_mode: "controlled_react",
      orchestration_fallback_reason: "qwen_intent_output_invalid",
      orchestration_fallback_stage: "intent_planning",
      orchestration_fallback_after_action_count: 0,
      orchestration_fallback_attempt_count: 2,
      orchestration_fallback_validation_errors: [
        "IntentPlan is missing fields: goal",
        "Qwen changed an explicit known fact.",
      ],
      intent_planner_attempt_diagnostics: [
        {
          stage: "intent_planning",
          attempt: 1,
          prompt_name: "intent_planner",
          prompt_version: "v1",
          outcome: "failure",
          failure_category: "contract_validation_error",
          reason_code: "malformed_output",
          field_path: "$.goal",
          message: "IntentPlan is missing fields: goal",
          repair_feedback_sent: true,
          candidate_summary: { top_level_fields: [] },
          baseline_diff: { goal_object_missing: true },
          provider_request_id: null,
        },
        {
          stage: "intent_planning",
          attempt: 2,
          prompt_name: "intent_planner",
          prompt_version: "v1",
          outcome: "failure",
          failure_category: "semantic_validation_error",
          reason_code: "known_fact_changed",
          field_path: "$.goal.known_facts.defect",
          message: "Qwen changed an explicit known fact.",
          repair_feedback_sent: false,
          candidate_summary: { intent: "root_cause" },
          baseline_diff: { known_fact_keys_changed: ["defect"] },
          provider_request_id: null,
        },
      ],
    };

    const html = renderToStaticMarkup(<AgentDecisionTrace state={state} />);

    expect(html).toContain("Planner Trace + Controlled Handoff");
    expect(html).toContain("Intent Planner handoff trace");
    expect(html).toContain("Attempt 1");
    expect(html).toContain("Attempt 2");
    expect(html).toContain("Contract Validation Error");
    expect(html).toContain("Semantic Validation Error");
    expect(html).toContain("malformed_output");
    expect(html).toContain("known_fact_changed");
    expect(html).toContain("$.goal.known_facts.defect");
    expect(html).toContain("Sent for retry");
    expect(html).toContain("Safe candidate summary and baseline diff");
    expect(html).not.toContain("Planner validation diagnostics (2)");
  });

  it("shows a successful Intent Planner handoff on an autonomous trace", () => {
    const state = populatedTraceState();
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
        candidate_summary: { intent: "root_cause", question_count: 2 },
        baseline_diff: { intent_changed: false },
        provider_request_id: null,
      },
    ];

    const html = renderToStaticMarkup(<AgentDecisionTrace state={state} />);

    expect(html).toContain("Intent Planner handoff trace");
    expect(html).toContain("Accepted");
    expect(html).toContain(
      "Python accepted the Qwen Goal and Question contract",
    );
  });

  it("renders compact question updates on the decision that committed them", () => {
    const state = populatedTraceState();
    const terminalDecision = state.planner_decisions?.at(-1);
    if (!terminalDecision) throw new Error("terminal decision fixture is required");
    terminalDecision.question_updates = [
      {
        question_id: "Q_ROOT_CAUSE",
        status: "closed",
        answer: "FDC and defect Evidence support the Cu CMP endpoint mechanism.",
        evidence_ids: ["EV_DEFECT_01"],
        unavailable_reason: null,
      },
    ];

    const html = renderToStaticMarkup(<AgentDecisionTrace state={state} />);

    expect(html).toContain("Question state updates");
    expect(html).toContain("Q_ROOT_CAUSE");
    expect(html).toContain(
      "FDC and defect Evidence support the Cu CMP endpoint mechanism.",
    );
    expect(html).toContain("EV_DEFECT_01");
  });

  it("renders a rejected QuestionUpdate without implying that its status changed", () => {
    const state = populatedTraceState();
    state.question_update_reviews = [
      {
        decision_id: "DECISION_DEFECT",
        disposition: "rejected",
        reason_code: "non_terminal_status",
        reason:
          "QuestionUpdate was rejected because status must be closed or unavailable; the Question remains open.",
        update_index: 0,
        question_id: "Q_ROOT_CAUSE",
        claimed_status: "open",
      },
    ];

    const html = renderToStaticMarkup(<AgentDecisionTrace state={state} />);

    expect(html).toContain("QuestionUpdate rejected");
    expect(html).toContain("Q_ROOT_CAUSE");
    expect(html).toContain("Claimed status:");
    expect(html).toContain("open");
    expect(html).toContain("Reason code:");
    expect(html).toContain("non_terminal_status");
    expect(html).toContain("the Question remains open");
    expect(html).toContain(
      "The Agent action was preserved; the invalid Question status claim was not committed.",
    );
  });

  it("shows an Advanced SPC step as superseded while preserving Basic SPC Evidence", () => {
    const state = populatedTraceState();
    const fdcAction = action("ACTION_FDC", "inspect_fdc_spc", "fdc");
    const fdcRecord = actionRecord(
      fdcAction,
      "FINDING_FDC",
      "Basic SPC supplied the effective fallback Evidence.",
    );
    state.planner_decisions = [
      actDecision("DECISION_FDC", fdcAction),
      state.planner_decisions?.[1] as PlannerDecision,
      state.planner_decisions?.[2] as PlannerDecision,
    ];
    state.action_history = [
      fdcRecord,
      state.action_history?.[1] as ActionRecord,
    ];
    state.findings = [
      {
        finding_id: "FINDING_FDC",
        agent: "fdc",
        finding_kind: "specialist_observation",
        summary: "Basic SPC replaced an unusable Advanced SPC result.",
        confidence: 0.8,
        evidence_ids: ["EV_DEFECT_01"],
        details: {
          engineering_interpretation:
            "Use the bounded Basic SPC result for the engineering interpretation.",
          specialist_v2: {
            version: "v2",
            action_id: "ACTION_FDC",
            agent: "fdc",
            tool_call_count: 2,
            stop_reason: "advanced_spc_fell_back_to_basic",
            analysis_source: "qwen",
            fallback_reason: null,
            validation_retry_count: 0,
            superseded_step_ids: ["STEP_ADVANCED"],
            tool_steps: [
              {
                step_id: "STEP_ADVANCED",
                step_index: 1,
                action_id: "ACTION_FDC",
                agent: "fdc",
                decision_id: "SPECIALIST_DECISION_ADVANCED",
                candidate_id: "CANDIDATE_ADVANCED",
                tool_name: "analyze_spc_evidence",
                parameters: { lot_id: "LOT_A_001" },
                reason: "Try the configured Advanced SPC profile first.",
                evidence_ids: ["EV_ADVANCED_SUPERSEDED"],
                output_summary:
                  "The configured profile had no analyzable parameters.",
                status: "completed",
              },
              {
                step_id: "STEP_BASIC",
                step_index: 2,
                action_id: "ACTION_FDC",
                agent: "fdc",
                decision_id: "SPECIALIST_DECISION_BASIC",
                candidate_id: "CANDIDATE_BASIC",
                tool_name: "perform_basic_spc_analysis",
                parameters: { lot_id: "LOT_A_001" },
                reason: "Use Basic SPC inside the remaining Tool budget.",
                evidence_ids: ["EV_DEFECT_01"],
                output_summary: "Basic SPC produced the effective observation.",
                status: "completed",
              },
            ],
          },
        },
        warnings: [],
      },
      state.findings[1],
    ];
    if (!state.run_evaluation) throw new Error("fixture requires evaluation");
    state.run_evaluation = {
      ...state.run_evaluation,
      decision_evaluations: [
        evaluation("DECISION_FDC", {
          evidenceGain: true,
          newEvidenceIds: ["EV_DEFECT_01"],
          reason: "Only the effective Basic SPC Evidence entered the Finding.",
        }),
        ...state.run_evaluation.decision_evaluations.slice(1),
      ],
    };

    const html = renderToStaticMarkup(<AgentDecisionTrace state={state} />);

    expect(html).toContain("2 bounded Tool calls");
    expect(html).toContain("Analyze SPC Evidence");
    expect(html).toContain("Perform Basic SPC Analysis");
    expect(html).toContain("Superseded");
    expect(html).toContain("EV_ADVANCED_SUPERSEDED");
    expect(html).toContain("EV_DEFECT_01");
    expect(html).toContain("Advanced SPC Fell Back To Basic");
    expect(html).not.toContain(
      "Specialist Tool Evidence ID EV_ADVANCED_SUPERSEDED is missing.",
    );
  });

  it("shows Python-owned causal Scope provenance on a Knowledge action", () => {
    const state = populatedTraceState();
    const knowledgeAction = action(
      "ACTION_KNOWLEDGE",
      "validate_historical_case",
      "knowledge",
    );
    state.planner_decisions = [
      actDecision("DECISION_KNOWLEDGE", knowledgeAction),
      state.planner_decisions?.[1] as PlannerDecision,
      state.planner_decisions?.[2] as PlannerDecision,
    ];
    state.action_history = [
      actionRecord(
        knowledgeAction,
        "FINDING_KNOWLEDGE",
        "The cross-Module candidate remained historical context only.",
      ),
      state.action_history?.[1] as ActionRecord,
    ];
    state.findings = [
      {
        finding_id: "FINDING_KNOWLEDGE",
        agent: "knowledge",
        finding_kind: "knowledge_discovery",
        summary: "An upstream historical candidate was retrieved.",
        confidence: 0.72,
        evidence_ids: ["EV_DEFECT_01"],
        details: {
          observation_scope: {
            source_lot_id: "LOT_A_001",
            detected_module: "Cu CMP",
            symptom_types: ["scratch"],
          },
          causal_search_scope: {
            mode: "causal_wide",
            hard_constraints: {},
            soft_hints: { module: "Cu CMP", tags: ["scratch"] },
            expansion_lanes: [
              {
                lane: "upstream_route",
                available: true,
                reason: "Earlier source-Lot route operations are available.",
              },
              {
                lane: "shared_resource",
                available: false,
                reason: "Configured shared-resource context is unavailable.",
              },
            ],
            scope_reason: "Observed Module remains a soft hint.",
          },
          candidate_lanes: ["upstream_route", "global_semantic"],
          scope_reasons: ["Recovered through an upstream route candidate."],
        },
        warnings: [],
      },
      state.findings[1],
    ];

    const html = renderToStaticMarkup(<AgentDecisionTrace state={state} />);

    expect(html).toContain("Python-owned Scope provenance");
    expect(html).toContain("Observed at");
    expect(html).toContain("Detected Module=Cu CMP");
    expect(html).toContain("Upstream Route (available)");
    expect(html).toContain("Shared Resource (unavailable)");
    expect(html).toContain("Selected candidate lane");
    expect(html).toContain("Upstream Route · Global Semantic");
    expect(html).toContain("do not establish current-Lot root cause");
  });

  it("renders a completed legacy autonomous snapshot without evaluation neutrally", () => {
    const state = populatedTraceState();
    state.run_evaluation = null;

    const html = renderToStaticMarkup(<AgentDecisionTrace state={state} />);

    expect(html).toContain(
      "<dt>Goal Success</dt><dd>Unavailable</dd>",
    );
    expect(html).toContain(
      "<dt>Stop Correct</dt><dd>Unavailable</dd>",
    );
    expect(html).not.toContain(
      "<dt>Goal Success</dt><dd>False</dd>",
    );
  });

  it("renders an unfinished autonomous evaluation as pending", () => {
    const state = populatedTraceState();
    state.job.status = "running";
    state.run_evaluation = null;

    const html = renderToStaticMarkup(<AgentDecisionTrace state={state} />);

    expect(html).toContain("<dt>Goal Success</dt><dd>Pending</dd>");
    expect(html).toContain("<dt>Stop Correct</dt><dd>Pending</dd>");
    expect(html).toContain(
      "Decision evaluation is pending until the investigation reaches a terminal state.",
    );
    expect(html).not.toContain("<dt>Goal Success</dt><dd>False</dd>");
  });
});
