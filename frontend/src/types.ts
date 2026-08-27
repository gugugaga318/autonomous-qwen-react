export type TaskStatus =
  | "pending"
  | "queued"
  | "running"
  | "retry_wait"
  | "cancel_requested"
  | "completed"
  | "failed"
  | "cancelled"
  | "skipped";
export type InvestigationMode = "product_window" | "lot";
export type WorkspaceMode = InvestigationMode | "knowledge";
export type KnowledgeQuestionKind =
  | "historical_match"
  | "procedure_guidance"
  | "engineering_note_lookup";
export type KnowledgeDocumentType = "RCA_CASE" | "SOP" | "ENGINEERING_NOTE";
export type OrchestrationMode = "fixed" | "controlled_react" | "llm_react";
export type EvidenceGapStatus = "open" | "closed" | "unavailable";
export type GoalStatus = "in_progress" | "satisfied" | "blocked" | "budget_exhausted";
export type ConclusionLevel =
  | "signal"
  | "candidate"
  | "supported"
  | "conflicted"
  | "inconclusive";
export type FindingKind =
  | "specialist_observation"
  | "knowledge_discovery"
  | "knowledge_validation"
  | "hypothesis_generation"
  | "hypothesis_ranking"
  | "improvement";

export interface CreateRCAJobRequest {
  investigation_mode: InvestigationMode;
  user_query?: string;
  lot_id?: string;
}

export interface RCAJobCreated {
  job_id: string;
  status: TaskStatus;
  created_at: string;
  investigation_mode: InvestigationMode;
  source_lot_id: string | null;
  state_url: string;
  events_url: string;
  report_url: string;
  cancel_url: string;
  idempotency_key: string | null;
  memory_candidate_id: string | null;
  memory_candidate_url: string | null;
}

export interface RuntimeInfo {
  status: "ready";
  agent_mode: "deterministic" | "fake" | "llm";
  model: string;
  dataset: string;
  orchestration_mode: OrchestrationMode;
}

export interface AgentTask {
  task_id: string;
  agent: string;
  objective: string;
  depends_on: string[];
  status: TaskStatus;
  inputs: Record<string, unknown>;
  finding_kind: FindingKind;
}

export interface InvestigationAction {
  action_id: string;
  kind: string;
  agent: string;
  reason: string;
  inputs: Record<string, unknown>;
  scope: Record<string, unknown>;
  required_evidence_ids: string[];
  max_attempts: number;
}

export interface ActionRecord {
  action: InvestigationAction;
  status: "completed" | "skipped" | "failed";
  produced_finding_ids: string[];
  produced_evidence_ids: string[];
  decision_summary: string;
}

export interface InvestigationQuestion {
  question_id: string;
  goal_id: string;
  question: string;
  rationale: string;
  question_kind?: string;
  scope: Record<string, unknown>;
  status: EvidenceGapStatus;
  answer: string | null;
  evidence_ids: string[];
  unavailable_reason: string | null;
  satisfied_evidence_groups?: string[];
  missing_evidence_groups?: string[];
  compatible_action_kinds?: string[];
  evidence_links?: QuestionEvidenceLink[];
}

export interface CapabilityNotice {
  capability: string;
  supported: boolean;
  reason: string;
  available_alternatives: string[];
  request_source: "user" | "qwen" | "system";
}

export interface QuestionEvidenceLink {
  question_id: string;
  evidence_id: string;
  action_id: string;
  relation: "supports" | "contradicts" | "context" | "unavailable";
  matched_evidence_group: string;
  reason: string;
}

export interface QuestionUpdate {
  question_id: string;
  status: "closed" | "unavailable";
  answer: string | null;
  evidence_ids: string[];
  unavailable_reason: string | null;
}

export type QuestionUpdateReasonCode =
  | "accepted"
  | "malformed_collection"
  | "too_many_updates"
  | "malformed_update"
  | "non_terminal_status"
  | "duplicate_question"
  | "unknown_question"
  | "new_question_conflict"
  | "terminal_question"
  | "target_overlap"
  | "unknown_evidence"
  | "evidence_not_applicable"
  | "insufficient_evidence_coverage"
  | "unsupported_capability"
  | "missing_unavailability_evidence";

export interface QuestionUpdateReview {
  decision_id: string;
  disposition: "accepted" | "rejected";
  reason_code: QuestionUpdateReasonCode;
  reason: string;
  update_index: number | null;
  question_id: string | null;
  claimed_status: string | null;
}

export interface PlannerDecision {
  decision_id: string;
  goal_id: string;
  decision_type: "act" | "stop";
  reason: string;
  goal_status: GoalStatus;
  proposed_conclusion_level: ConclusionLevel;
  next_action: InvestigationAction | null;
  target_question_ids: string[];
  new_questions: InvestigationQuestion[];
  question_updates: QuestionUpdate[];
  stop_reason:
    | "goal_satisfied"
    | "critical_contradiction"
    | "no_allowed_action"
    | "budget_exhausted"
    | "data_unavailable"
    | null;
}

export interface DecisionEvaluation {
  decision_id: string;
  decision_valid: boolean;
  evidence_gain: boolean;
  redundant: boolean;
  reason: string;
  new_evidence_ids: string[];
}

export interface RunEvaluation {
  goal_id: string;
  goal_success: boolean;
  stop_correct: boolean;
  summary: string;
  decision_evaluations: DecisionEvaluation[];
}

export interface EvidenceEntity {
  entity_type: string;
  entity_id: string;
  attributes: Record<string, unknown>;
}

export interface Evidence {
  evidence_id: string;
  source_type: string;
  source_id: string;
  summary: string;
  source_table: string | null;
  source_field: string | null;
  timestamp: string | null;
  metadata: Record<string, unknown>;
  evidence_type?: string | null;
  source_agent?: string | null;
  source_tool?: string | null;
  observation?: string | null;
  entities?: EvidenceEntity[];
  confidence?: number | null;
  evidence_schema_version?: string | null;
}

export interface Warning {
  warning_id: string;
  message: string;
  severity: string;
  evidence_ids: string[];
}

export interface AgentFinding {
  finding_id: string;
  task_id?: string | null;
  agent: string;
  finding_kind?: FindingKind;
  summary: string;
  confidence: number;
  evidence_ids: string[];
  evidence?: Evidence[];
  details: Record<string, unknown>;
  warnings: Warning[];
}

export type CausalClaimStatus =
  | "supported"
  | "plausible"
  | "incomplete"
  | "conflicted"
  | "unavailable";

export interface CausalMatrixClaim {
  claim: string;
  status: CausalClaimStatus;
  evidence_ids: string[];
  reason: string;
  facts: Record<string, unknown>;
  support_source: string | null;
}

export interface CausalEvidenceMatrix {
  root_cause: string;
  claims: Record<string, CausalMatrixClaim>;
  status: CausalClaimStatus;
  invalid_evidence_ids: string[];
  data_missing_evidence_ids?: string[];
  data_missing_sources?: DataMissingSourceTrace[];
  causal_chain?: CausalChainTrace | null;
  causal_chain_completeness?: string | null;
  mechanism_support_source: string | null;
}

export interface DataMissingSourceTrace {
  evidence_id: string;
  source_type: string;
  source_id: string;
  source_table: string | null;
  source_field: string | null;
  observation: string;
  entity_ids: string[];
  required_for_confirmation?: boolean;
}

export interface CausalChainTrace {
  status: string;
  stages: Record<string, CausalClaimStatus>;
  evidence_ids: string[];
  missing_stages: string[];
  conflicting_stages: string[];
  data_missing_evidence_ids: string[];
  reason: string;
}

export interface RcaCandidateTrace {
  root_cause: string;
  score: number | null;
  basis: string | null;
  status: string | null;
  evidence_ids: string[];
  supporting_evidence_ids: string[];
  contradicting_evidence_ids: string[];
  rejection_reasons: string[];
  causal_matrix_status: CausalClaimStatus | null;
  causal_chain_completeness?: string | null;
  data_missing_evidence_ids?: string[];
  mechanism_support_source: string | null;
  causal_evidence_matrix: CausalEvidenceMatrix | null;
}

export interface CausalEvidenceGap {
  gap_id: string;
  candidate_index: number;
  claim: string;
  status: string;
  reason: string;
  question_kind: string;
  allowed_actions: string[];
  evidence_ids: string[];
  gap_type?: string;
  data_missing_evidence_ids?: string[];
  unavailable_sources?: DataMissingSourceTrace[];
}

export interface ImpactLotGateRow {
  lot_id: string;
  included: boolean;
  candidate_included?: boolean;
  confirmed?: boolean;
  included_reason: string | null;
  excluded_reason: string | null;
  supporting_evidence_ids: string[];
  data_missing_evidence_ids?: string[];
  non_blocking_data_missing_evidence_ids?: string[];
  data_available?: boolean;
  checks?: Record<string, boolean>;
}

export interface CandidateImpactScopeTrace {
  candidate_index: number;
  candidate_rank?: number | null;
  candidate_root_cause: string | null;
  candidate_scope_status?: string;
  publication_status?: string;
  candidate_impact_lots: string[];
  confirmed_impact_lots: string[];
  data_missing_evidence_ids: string[];
  non_blocking_data_missing_evidence_ids: string[];
}

export interface CausalLaneTrace {
  lane_id: string;
  operation: string;
  operation_name?: string;
  module?: string;
  equipment: string;
  chamber: string;
  recipe: string;
  parameter_scope: string[];
  exposed_lot_ids: string[];
  time_window: string[];
  initial_evidence_ids: string[];
  priority_score: number;
  investigation_status: string;
  pruned_reason: string | null;
}

export interface CandidateChallengeTrace {
  candidate_id: string;
  alternative_candidate_id: string | null;
  evidence_probe_lane_id: string | null;
  challenge_kind: string;
  strongest_alternative_lane_id: string | null;
  supporting_evidence_ids: string[];
  contradicting_evidence_ids: string[];
  unexplained_precursor_evidence_ids: string[];
  distinguishing_gap_ids: string[];
  distinguishing_questions: string[];
  challenge_explanation: string;
  status: string;
}

export interface CompetitionTrace {
  active_lane_ids: string[];
  overflow_lane_ids: string[];
  represented_lane_ids: string[];
  unresolved_lane_ids: string[];
  eliminated_lane_ids: string[];
  blocked_lane_ids: string[];
  lane_resolutions: AlternativeLaneResolutionTrace[];
  alternative_search_status: string;
  competition_requirement: string;
  competition_status: string;
  competition_type: string;
  competition_axes?: string[];
  scope_assessment_status?: string;
  competition_failure_reason: string | null;
  candidate_lineage: Record<string, unknown>[];
  challenge_round_count: number;
  resolution_evidence_ids: string[];
}

export interface AlternativeLaneResolutionTrace {
  lane_id: string;
  status: "retained" | "eliminated" | "unresolved" | "blocked" | "non_discriminative";
  candidate_id: string | null;
  evidence_ids: string[];
  distinguishing_gap_ids: string[];
  reason: string;
}

export interface RcaDiagnosisTrace {
  finding_id: string;
  conclusion_status: string;
  causal_chain_completeness?: string | null;
  data_missing_evidence_ids?: string[];
  root_cause: string | null;
  ranked_candidates: RcaCandidateTrace[];
  evidence_synthesis: Record<string, unknown>;
  causal_evidence_gaps: CausalEvidenceGap[];
  candidate_comparison: Record<string, unknown>;
  causal_lanes: CausalLaneTrace[];
  candidate_challenges: CandidateChallengeTrace[];
  competition_trace: CompetitionTrace | null;
  confirmation_gate: {
    status?: string;
    checks?: Record<string, boolean>;
    reasons?: string[];
    unresolved_gaps?: string[];
    causal_chain_completeness?: string | null;
    data_missing_evidence_ids?: string[];
    blocking_data_missing_evidence_ids?: string[];
    non_blocking_data_missing_evidence_ids?: string[];
  };
  impact_lot_gate: {
    scope_status?: string;
    candidate_scope_status?: string;
    publication_status?: string;
    scope_basis?: string;
    data_missing_evidence_ids?: string[];
    non_blocking_data_missing_evidence_ids?: string[];
    observed_impact_lots?: string[];
    candidate_impact_lots?: string[];
    confirmed_impact_lots?: string[];
    confirmation_blocked_reason?: string | null;
    candidate_scopes?: CandidateImpactScopeTrace[];
    rows?: ImpactLotGateRow[];
  };
}

export interface Hypothesis {
  hypothesis_id: string;
  root_cause: string;
  confidence: number;
  evidence_ids: string[];
  status: string;
  rationale: string;
}

export interface RCAReport {
  report_id: string;
  title: string;
  markdown: string;
  cited_evidence_ids: string[];
  created_at: string;
}

export interface LLMUsageEvent {
  call_id: string;
  agent: string;
  provider: string;
  model: string;
  prompt_version: string;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  cached_tokens: number;
  reasoning_tokens: number;
  latency_ms: number;
  status: "success" | "failed";
}

export interface ToolLatencyRecord {
  tool_name: string;
  tool_request_id: string;
  agent: string;
  outcome: "success" | "failed";
  duration_ms: number;
}

export type PlannerFailureCategory =
  | "transport_provider_failure"
  | "output_parse_error"
  | "contract_validation_error"
  | "semantic_validation_error";

export type IntentPlannerReasonCode =
  | "goal_id_changed"
  | "intent_invalid"
  | "budget_changed"
  | "known_fact_removed"
  | "known_fact_changed"
  | "forbidden_known_fact_added"
  | "unsupported_question_kind"
  | "unrequested_material_trace"
  | "source_lot_scope_mismatch"
  | "malformed_output";

export interface PlannerAttemptDiagnostic {
  stage: "intent_planning";
  attempt: number;
  prompt_name: string;
  prompt_version: string;
  outcome: "success" | "failure";
  failure_category: PlannerFailureCategory | null;
  reason_code: IntentPlannerReasonCode | null;
  field_path: string | null;
  message: string | null;
  repair_feedback_sent: boolean;
  candidate_summary: Record<string, unknown>;
  baseline_diff: Record<string, unknown>;
  provider_request_id: string | null;
}

export interface ExecutionMetadata {
  agent_mode?: "deterministic" | "fake" | "llm";
  provider?: string | null;
  model?: string | null;
  prompt_version?: string | null;
  total_tokens?: number;
  llm_call_count?: number;
  llm_latency_ms?: number;
  tool_call_count?: number;
  tool_latency_ms?: number;
  workflow_duration_ms?: number;
  orchestration_mode?: OrchestrationMode;
  orchestration_requested_mode?: OrchestrationMode;
  orchestration_fallback_reason?: string;
  orchestration_fallback_stage?: "intent_planning" | "next_action_planning";
  orchestration_fallback_after_action_count?: number;
  orchestration_fallback_attempt_count?: number;
  orchestration_fallback_validation_errors?: string[];
  intent_planner_attempt_diagnostics?: PlannerAttemptDiagnostic[];
  tool_latencies?: ToolLatencyRecord[];
}

export interface RCAState {
  job: {
    job_id: string;
    user_query: string;
    investigation_mode: InvestigationMode;
    source_lot_id: string | null;
    product_id: string | null;
    time_window: Record<string, string>;
    declared_unavailable_sources?: string[];
    status: TaskStatus;
    created_at: string;
  };
  task_plan: {
    plan_id: string;
    objective: string;
    tasks: AgentTask[];
  } | null;
  current_task_id: string | null;
  completed_task_ids: string[];
  affected_lots: string[];
  impact_lots: string[];
  affected_wafers: string[];
  impact_wafers: string[];
  scope_level: "lot" | "wafer" | "mixed";
  impact_criteria: Record<string, unknown>;
  evidence: Evidence[];
  findings: AgentFinding[];
  hypotheses: Hypothesis[];
  causal_lanes?: CausalLaneTrace[];
  candidate_challenges?: CandidateChallengeTrace[];
  competition_trace?: CompetitionTrace | null;
  warnings: Warning[];
  report: RCAReport | null;
  llm_usage: LLMUsageEvent[];
  execution_metadata: ExecutionMetadata;
  capability_notices?: CapabilityNotice[];
  investigation_goal?: {
    goal_id: string;
    intent: string;
    summary: string;
    known_facts: Record<string, unknown>;
    required_evidence: string[];
    max_steps: number;
    max_tool_calls: number;
  } | null;
  investigation_questions?: InvestigationQuestion[];
  question_evidence_links?: QuestionEvidenceLink[];
  action_history?: ActionRecord[];
  planner_decisions?: PlannerDecision[];
  question_update_reviews?: QuestionUpdateReview[];
  authoritative_rca_finding_id?: string | null;
  authoritative_hypothesis_id?: string | null;
  rca_diagnosis?: RcaDiagnosisTrace | null;
  run_evaluation?: RunEvaluation | null;
  goal_status?: GoalStatus | null;
  conclusion_level?: ConclusionLevel | null;
  evidence_gaps?: string[];
  stop_reason?: string | null;
}

export type AgentTraceOrigin =
  | "llm_react"
  | "controlled_react"
  | "controlled_fallback"
  | "fixed"
  | "legacy";

export type AgentTraceEvaluationStatus =
  | "available"
  | "pending"
  | "not_applicable"
  | "fallback"
  | "unavailable";

export interface SpecialistToolStepViewModel {
  key: string;
  stepId: string;
  stepIndex: number;
  actionId: string;
  specialistDecisionId: string;
  candidateId: string;
  toolName: string;
  parameters: Record<string, unknown>;
  reason: string;
  evidenceIds: string[];
  evidence: Evidence[];
  outputSummary: string;
  status: "completed" | "failed";
  superseded: boolean;
  toolRequestId: string;
  latency: ToolLatencyRecord | null;
  integrityIssues: string[];
}

export interface SpecialistTraceViewModel {
  findingId: string;
  version: string | null;
  actionId: string;
  agent: string;
  toolCallCount: number;
  stopReason: string | null;
  analysisSource: "qwen" | "deterministic_fallback" | null;
  fallbackReason: string | null;
  validationRetryCount: number;
  localFallback: boolean;
  engineeringInterpretation: string | null;
  supersededStepIds: string[];
  toolSteps: SpecialistToolStepViewModel[];
  integrityIssues: string[];
}

export interface AgentTraceNodeViewModel {
  key: string;
  origin: AgentTraceOrigin;
  task: AgentTask | null;
  decision: PlannerDecision | null;
  evaluation: DecisionEvaluation | null;
  targetQuestions: InvestigationQuestion[];
  newQuestions: InvestigationQuestion[];
  questionUpdates: QuestionUpdate[];
  questionUpdateReviews: QuestionUpdateReview[];
  action: InvestigationAction | null;
  actionRecord: ActionRecord | null;
  findings: AgentFinding[];
  evidence: Evidence[];
  newEvidence: Evidence[];
  specialistTraces: SpecialistTraceViewModel[];
  integrityIssues: string[];
}

export interface AgentTraceViewModel {
  requestedMode: OrchestrationMode | null;
  actualMode: OrchestrationMode | null;
  evaluationStatus: AgentTraceEvaluationStatus;
  runEvaluation: RunEvaluation | null;
  goal: RCAState["investigation_goal"];
  questions: InvestigationQuestion[];
  capabilityNotices: CapabilityNotice[];
  questionEvidenceLinks: QuestionEvidenceLink[];
  nodes: AgentTraceNodeViewModel[];
  fallbackReason: string | null;
  fallbackStage: "intent_planning" | "next_action_planning" | null;
  fallbackAfterActionCount: number | null;
  fallbackAttemptCount: number | null;
  fallbackValidationErrors: string[];
  intentPlannerAttempts: PlannerAttemptDiagnostic[];
  integrityIssues: string[];
}

export interface RCAJobResponse {
  job_id: string;
  status: TaskStatus;
  state: RCAState;
  queue: RCAJobQueueMetadata | null;
}

export interface RCAJobQueueMetadata {
  priority: number;
  attempt_count: number;
  max_attempts: number;
  next_attempt_at: string | null;
  lease_expires_at: string | null;
  cancel_requested_at: string | null;
  started_at: string | null;
  completed_at: string | null;
  error: { error_code?: string; message?: string; retryable?: boolean } | null;
  version: number;
}

export interface RCAJobEvent {
  job_id: string;
  sequence: number;
  event_type: string;
  payload: Record<string, unknown>;
  created_at: string;
}

export type JobStreamConnection = "connecting" | "live" | "reconnecting" | "closed";

export interface CancelRCAJobResponse {
  job_id: string;
  status: TaskStatus;
  cancel_requested_at: string;
  state_url: string;
}

export interface RCAReportResponse {
  job_id: string;
  status: TaskStatus;
  report: RCAReport;
}

export interface YieldTrendPoint {
  date: string;
  lot_count: number;
  pass_count: number;
  fail_count: number;
  pass_rate: number;
}

export interface EvidenceChainItem {
  stage: string;
  claim: string;
  confidence: number;
  evidence_ids: string[];
}

export interface RecommendedAction {
  action: string;
  evidence_ids: string[];
}

export interface SpcViolation {
  rule_code: string;
  description: string;
  direction: string;
  sample_ids: string[];
  lot_ids: string[];
  wafer_ids: string[];
  start_timestamp: string;
  end_timestamp: string;
  evidence_id: string;
}

export interface SpcChartResult {
  chart_type: "I_MR" | "XBAR_S" | "XBAR_R" | "P";
  parameter_name: string;
  unit: string;
  status: "OOC" | "IN_CONTROL";
  baseline_id: string;
  baseline_window: { start: string; end: string };
  center_line: number;
  lower_control_limit: number;
  upper_control_limit: number;
  point_violation_count: number;
  violated_rules: string[];
  violations: SpcViolation[];
  series: Array<{
    sample_id: string;
    lot_id: string;
    wafer_id: string | null;
    timestamp: string;
    value: number;
  }>;
  capability: {
    cp: number | null;
    cpk: number | null;
    pp: number | null;
    ppk: number | null;
    spec_lower: number | null;
    spec_upper: number | null;
    valid_for_decision: boolean;
    warning: string | null;
  } | null;
}

export interface SpcOocContext {
  event_key: string;
  trigger_lot_id: string;
  trigger_wafer_id: string | null;
  trigger_hold: { hold_id: string; hold_code: string } | null;
  spc_rule_codes: string[];
  impact_scopes: Array<{ lot_id: string; hold_id: string }>;
}

export type EngineerRole =
  | "yield_engineer"
  | "process_engineer"
  | "equipment_engineer"
  | "quality_engineer";

export interface MemoryApproval {
  approval_id: string;
  candidate_id: string;
  engineer_id: string;
  engineer_role: EngineerRole;
  decision: "approve" | "reject";
  comment: string;
  decided_at: string;
}

export interface MemoryCandidate {
  candidate_id: string;
  job_id: string;
  status: "pending_approval" | "published" | "rejected";
  scope_level: "event" | "fab";
  title: string;
  incident_summary: string;
  engineering_summary: string;
  root_cause: string;
  confidence: number;
  recommendations: Record<string, Array<Record<string, unknown>>>;
  evidence_ids: string[];
  source_lot_id: string | null;
  product_id: string | null;
  requires_process_engineer_approval: boolean;
  approvals: MemoryApproval[];
  approval_count: number;
  required_approval_count: number;
  has_process_engineer_approval: boolean;
  published_case_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface MemoryCandidateResponse {
  candidate: MemoryCandidate;
}

export interface MemoryApprovalRequest {
  engineer_id: string;
  engineer_role: EngineerRole;
  decision: "approve" | "reject";
  comment: string;
}

export interface KnowledgeLookupRequest {
  query: string;
  question_kind: KnowledgeQuestionKind;
  document_type?: KnowledgeDocumentType;
  module?: string;
  equipment_type?: string;
  operation?: string;
  defect_type?: string;
  tags?: string[];
  source_lot_id?: string;
  product_id?: string;
  detected_at?: string;
  symptom_types?: string[];
  explicit_module_limit?: boolean;
  top_k?: number;
}

export type CausalLane =
  | "same_step"
  | "upstream_route"
  | "shared_resource"
  | "global_semantic";

export interface ScopeFilters {
  module: string;
  equipment_type: string;
  operation: string;
  defect_type: string;
  tags: string[];
}

export interface ObservationScope {
  source_lot_id: string;
  product_id: string;
  detected_module: string;
  detected_operation: string;
  detected_equipment_id: string;
  detected_equipment_type: string;
  detected_at: string;
  symptom_types: string[];
  known_measurements: string[];
  known_defect_attributes: string[];
}

export interface CausalLaneContext {
  lane: CausalLane;
  available: boolean;
  reason: string;
  modules: string[];
  equipment_types: string[];
  route_distance: number | null;
  shared_resource_types: string[];
}

export interface CausalSearchScope {
  mode: "legacy_hard" | "causal_wide" | "explicit_hard";
  hard_constraints: ScopeFilters;
  soft_hints: ScopeFilters;
  expansion_lanes: CausalLaneContext[];
  available_lanes: CausalLane[];
  explicit_user_limits: string[];
  time_boundary: string;
  candidate_budget: number;
  lane_minimum: number;
  scope_reason: string;
}

export interface KnowledgeDocument {
  document_id: string;
  case_id: string | null;
  evaluation_asset_id: string;
  document_type: KnowledgeDocumentType;
  title: string;
  content: string;
  module: string;
  equipment_type: string;
  operation: string;
  defect_type: string;
  tags: string[];
  source_format: string;
  content_sha256: string;
  validation_status: "CONFIRMED";
  publication_policy: string;
  source_candidate_id: string | null;
  created_at: string;
  schema_version: string;
}

export interface KnowledgeLookupHit {
  rank: number;
  document: KnowledgeDocument;
  score: number;
  matched_chunk_ids: string[];
  excerpt: string;
  evidence_id: string;
  relevance_reason: string;
  retrieval_strategy: string;
  score_components: Partial<
    Record<"keyword" | "lexical" | "vector" | "fusion" | "reranker", number>
  >;
  calibrated_relevance: number | null;
  source_confidence: number | null;
  candidate_lanes: CausalLane[];
  scope_reasons: string[];
  route_distance: number | null;
  shared_resource_types: string[];
  scope_fusion_score: number | null;
}

export interface KnowledgeAgentTrace {
  agent: "knowledge";
  action: string;
  execution_reason: string;
  inputs: Record<string, unknown>;
  output_evidence_ids: string[];
  stop_reason: string;
}

export interface KnowledgeLookupResult {
  lookup_id: string;
  intent: "knowledge_lookup";
  question_kind: KnowledgeQuestionKind;
  status: "completed" | "no_match";
  plan: {
    intent: "knowledge_lookup";
    question_kind: KnowledgeQuestionKind;
    action: string;
    query: string;
    allowed_document_types: KnowledgeDocumentType[];
    reason: string;
    module: string;
    equipment_type: string;
    operation: string;
    defect_type: string;
    tags: string[];
    observation_scope: ObservationScope | null;
    causal_search_scope: CausalSearchScope | null;
    explicit_module_limit: boolean;
    top_k: number;
  };
  hits: KnowledgeLookupHit[];
  agent_trace: KnowledgeAgentTrace[];
  answer_boundary: string;
  warnings: string[];
  root_cause_conclusion: null;
  created_at: string;
}

export interface KnowledgeChunk {
  chunk_id: string;
  document_id: string | null;
  candidate_id: string | null;
  chunk_index: number;
  section_type: string;
  heading: string;
  content?: string;
  content_preview?: string;
  token_count: number;
  metadata: Record<string, unknown>;
  validation_status: "STAGED" | "CONFIRMED";
  embedding_status: string;
  schema_version: string;
}

export interface KnowledgeIngestionApproval {
  approval_id: string;
  candidate_id: string;
  engineer_id: string;
  engineer_role: EngineerRole;
  decision: "approve" | "reject";
  comment: string;
  decided_at: string;
  schema_version: string;
}

export interface KnowledgeIngestionCandidate {
  candidate_id: string;
  filename: string;
  source_format: string;
  document_type: KnowledgeDocumentType;
  case_id: string | null;
  title: string;
  parsed_content?: string;
  content_preview?: string;
  content_sha256: string;
  module: string;
  equipment_type: string;
  operation: string;
  defect_type: string;
  tags: string[];
  status: "pending_approval" | "published" | "rejected";
  chunks: KnowledgeChunk[];
  chunk_count: number;
  approvals: KnowledgeIngestionApproval[];
  approval_count: number;
  required_approval_count: number;
  published_document_id: string | null;
  publication_policy: string;
  created_at: string;
  updated_at: string;
  schema_version: string;
}

export interface KnowledgeIngestionResponse {
  candidate: KnowledgeIngestionCandidate;
}

export interface KnowledgeIngestionListResponse {
  candidates: KnowledgeIngestionCandidate[];
}

export interface KnowledgeApprovalRequest {
  engineer_id: string;
  engineer_role: EngineerRole;
  decision: "approve" | "reject";
  comment: string;
}
