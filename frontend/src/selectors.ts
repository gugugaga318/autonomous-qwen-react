import type {
  ActionRecord,
  AgentFinding,
  AgentTask,
  AgentTraceEvaluationStatus,
  AgentTraceNodeViewModel,
  AgentTraceOrigin,
  AgentTraceViewModel,
  CapabilityNotice,
  CreateRCAJobRequest,
  Evidence,
  EvidenceChainItem,
  FindingKind,
  InvestigationAction,
  InvestigationMode,
  PlannerAttemptDiagnostic,
  QuestionEvidenceLink,
  OrchestrationMode,
  PlannerDecision,
  RCAState,
  CausalClaimStatus,
  CausalChainTrace,
  CausalLaneTrace,
  CandidateChallengeTrace,
  CompetitionTrace,
  CausalEvidenceGap,
  CausalEvidenceMatrix,
  CausalMatrixClaim,
  DataMissingSourceTrace,
  ImpactLotGateRow,
  RcaCandidateTrace,
  RcaDiagnosisTrace,
  RecommendedAction,
  SpecialistToolStepViewModel,
  SpecialistTraceViewModel,
  ToolLatencyRecord,
  YieldTrendPoint,
} from "./types";

export function buildRCAJobRequest(
  investigationMode: InvestigationMode,
  query: string,
  lotId: string,
): CreateRCAJobRequest {
  const userQuery = query.trim();
  if (investigationMode === "lot") {
    return {
      investigation_mode: "lot",
      lot_id: lotId.trim().toUpperCase(),
      user_query: userQuery,
    };
  }
  return {
    investigation_mode: "product_window",
    user_query: userQuery,
  };
}

export function getDefaultLotId(dataset: string): string {
  return dataset === "multi_case" || dataset === "spc_case" ? "LOT_A_015" : "LOT_A_001";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function findingFor(
  state: RCAState,
  agent: string,
  findingKind: FindingKind = "specialist_observation",
): AgentFinding | undefined {
  const task = state.task_plan?.tasks.find(
    (item) => item.agent === agent && item.finding_kind === findingKind,
  );
  if (task) {
    const byTask = state.findings.find((finding) => finding.task_id === task.task_id);
    if (byTask) return byTask;
  }

  return (
    state.findings.find(
      (finding) => finding.agent === agent && finding.finding_kind === findingKind,
    ) ?? state.findings.find((finding) => finding.agent === agent)
  );
}

export function authoritativeRcaFindingFor(state: RCAState): AgentFinding | undefined {
  const authoritativeId = state.authoritative_rca_finding_id;
  if (typeof authoritativeId === "string" && authoritativeId.length > 0) {
    return state.findings.find((finding) => finding.finding_id === authoritativeId);
  }
  const rankingFindings = state.findings.filter(
    (finding) =>
      finding.agent === "rca_reasoning" &&
      finding.finding_kind === "hypothesis_ranking",
  );
  if (rankingFindings.length === 1) return rankingFindings[0];
  const rcaFindings = state.findings.filter(
    (finding) => finding.agent === "rca_reasoning",
  );
  return rcaFindings.length === 1 ? rcaFindings[0] : undefined;
}

export function authoritativeHypothesisFor(state: RCAState): RCAState["hypotheses"][number] | undefined {
  const authoritativeId = state.authoritative_hypothesis_id;
  if (typeof authoritativeId === "string" && authoritativeId.length > 0) {
    return state.hypotheses.find((hypothesis) => hypothesis.hypothesis_id === authoritativeId);
  }
  return state.hypotheses.length === 1 ? state.hypotheses[0] : undefined;
}

function stringList(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function causalStatus(value: unknown): CausalClaimStatus {
  return value === "supported" || value === "plausible" || value === "incomplete" || value === "conflicted" || value === "unavailable"
    ? value
    : "unavailable";
}

function matrixClaim(value: unknown, claim: string): CausalMatrixClaim {
  const raw = isRecord(value) ? value : {};
  return {
    claim: typeof raw.claim === "string" ? raw.claim : claim,
    status: causalStatus(raw.status),
    evidence_ids: stringList(raw.evidence_ids),
    reason: typeof raw.reason === "string" ? raw.reason : "",
    facts: isRecord(raw.facts) ? raw.facts : {},
    support_source: typeof raw.support_source === "string" ? raw.support_source : null,
  };
}

function missingSource(value: unknown): DataMissingSourceTrace | null {
  if (!isRecord(value) || typeof value.evidence_id !== "string") return null;
  return {
    evidence_id: value.evidence_id,
    source_type: typeof value.source_type === "string" ? value.source_type : "unknown",
    source_id: typeof value.source_id === "string" ? value.source_id : "",
    source_table: typeof value.source_table === "string" ? value.source_table : null,
    source_field: typeof value.source_field === "string" ? value.source_field : null,
    observation: typeof value.observation === "string" ? value.observation : "",
    entity_ids: stringList(value.entity_ids),
    required_for_confirmation: value.required_for_confirmation === true,
  };
}

function causalLane(value: unknown): CausalLaneTrace | null {
  if (!isRecord(value) || typeof value.lane_id !== "string") return null;
  return {
    lane_id: value.lane_id,
    operation: typeof value.operation === "string" ? value.operation : "",
    operation_name: typeof value.operation_name === "string" ? value.operation_name : "",
    module: typeof value.module === "string" ? value.module : "",
    equipment: typeof value.equipment === "string" ? value.equipment : "",
    chamber: typeof value.chamber === "string" ? value.chamber : "",
    recipe: typeof value.recipe === "string" ? value.recipe : "",
    parameter_scope: stringList(value.parameter_scope),
    exposed_lot_ids: stringList(value.exposed_lot_ids),
    time_window: stringList(value.time_window),
    initial_evidence_ids: stringList(value.initial_evidence_ids),
    priority_score: typeof value.priority_score === "number" ? value.priority_score : 0,
    investigation_status:
      typeof value.investigation_status === "string" ? value.investigation_status : "uninvestigated",
    pruned_reason: typeof value.pruned_reason === "string" ? value.pruned_reason : null,
  };
}

function candidateChallenge(value: unknown): CandidateChallengeTrace | null {
  if (!isRecord(value) || typeof value.candidate_id !== "string") return null;
  return {
    candidate_id: value.candidate_id,
    alternative_candidate_id:
      typeof value.alternative_candidate_id === "string"
        ? value.alternative_candidate_id
        : null,
    evidence_probe_lane_id:
      typeof value.evidence_probe_lane_id === "string"
        ? value.evidence_probe_lane_id
        : typeof value.strongest_alternative_lane_id === "string"
          ? value.strongest_alternative_lane_id
          : null,
    challenge_kind:
      typeof value.challenge_kind === "string" ? value.challenge_kind : "lane_probe",
    strongest_alternative_lane_id:
      typeof value.strongest_alternative_lane_id === "string"
        ? value.strongest_alternative_lane_id
        : null,
    supporting_evidence_ids: stringList(value.supporting_evidence_ids),
    contradicting_evidence_ids: stringList(value.contradicting_evidence_ids),
    unexplained_precursor_evidence_ids: stringList(value.unexplained_precursor_evidence_ids),
    distinguishing_gap_ids: stringList(value.distinguishing_gap_ids),
    distinguishing_questions: stringList(value.distinguishing_questions),
    challenge_explanation:
      typeof value.challenge_explanation === "string" ? value.challenge_explanation : "",
    status: typeof value.status === "string" ? value.status : "open",
  };
}

function competitionTrace(value: unknown): CompetitionTrace | null {
  if (!isRecord(value)) return null;
  return {
    active_lane_ids: stringList(value.active_lane_ids),
    overflow_lane_ids: stringList(value.overflow_lane_ids),
    represented_lane_ids: stringList(value.represented_lane_ids),
    unresolved_lane_ids: stringList(value.unresolved_lane_ids),
    eliminated_lane_ids: stringList(value.eliminated_lane_ids),
    blocked_lane_ids: stringList(value.blocked_lane_ids),
    lane_resolutions: Array.isArray(value.lane_resolutions)
      ? value.lane_resolutions.flatMap((item) => {
          if (!isRecord(item) || typeof item.lane_id !== "string") return [];
          const allowedStatuses = new Set([
            "retained",
            "eliminated",
            "unresolved",
            "blocked",
            "non_discriminative",
          ]);
          const status = typeof item.status === "string" && allowedStatuses.has(item.status)
            ? item.status as "retained" | "eliminated" | "unresolved" | "blocked" | "non_discriminative"
            : "unresolved";
          return [{
            lane_id: item.lane_id,
            status,
            candidate_id: typeof item.candidate_id === "string" ? item.candidate_id : null,
            evidence_ids: stringList(item.evidence_ids),
            distinguishing_gap_ids: stringList(item.distinguishing_gap_ids),
            reason: typeof item.reason === "string" ? item.reason : "",
          }];
        })
      : [],
    alternative_search_status:
      typeof value.alternative_search_status === "string"
        ? value.alternative_search_status
        : "not_searched",
    competition_requirement:
      typeof value.competition_requirement === "string"
        ? value.competition_requirement
        : "not_evaluated",
    competition_status:
      typeof value.competition_status === "string"
        ? value.competition_status
        : "not_evaluated",
    competition_type:
      typeof value.competition_type === "string"
        ? value.competition_type
        : "not_evaluated",
    competition_axes: stringList(value.competition_axes),
    scope_assessment_status:
      typeof value.scope_assessment_status === "string"
        ? value.scope_assessment_status
        : "not_evaluated",
    competition_failure_reason:
      typeof value.competition_failure_reason === "string"
        ? value.competition_failure_reason
        : null,
    candidate_lineage: Array.isArray(value.candidate_lineage)
      ? value.candidate_lineage.filter(isRecord)
      : [],
    challenge_round_count:
      typeof value.challenge_round_count === "number" ? value.challenge_round_count : 0,
    resolution_evidence_ids: stringList(value.resolution_evidence_ids),
  };
}

function causalChain(value: unknown): CausalChainTrace | null {
  if (!isRecord(value)) return null;
  const rawStages = isRecord(value.stages) ? value.stages : {};
  const stages: Record<string, CausalClaimStatus> = {};
  for (const [stage, status] of Object.entries(rawStages)) {
    stages[stage] = causalStatus(status);
  }
  return {
    status: typeof value.status === "string" ? value.status : "incomplete",
    stages,
    evidence_ids: stringList(value.evidence_ids),
    missing_stages: stringList(value.missing_stages),
    conflicting_stages: stringList(value.conflicting_stages),
    data_missing_evidence_ids: stringList(value.data_missing_evidence_ids),
    reason: typeof value.reason === "string" ? value.reason : "",
  };
}

function causalMatrix(value: unknown): CausalEvidenceMatrix | null {
  if (!isRecord(value)) return null;
  const rawClaims = isRecord(value.claims) ? value.claims : {};
  const claims: Record<string, CausalMatrixClaim> = {};
  for (const [claim, item] of Object.entries(rawClaims)) claims[claim] = matrixClaim(item, claim);
  return {
    root_cause: typeof value.root_cause === "string" ? value.root_cause : "",
    claims,
    status: causalStatus(value.status),
    invalid_evidence_ids: stringList(value.invalid_evidence_ids),
    data_missing_evidence_ids: stringList(value.data_missing_evidence_ids),
    data_missing_sources: Array.isArray(value.data_missing_sources)
      ? value.data_missing_sources.flatMap((item) => {
          const source = missingSource(item);
          return source ? [source] : [];
        })
      : [],
    causal_chain: causalChain(value.causal_chain),
    causal_chain_completeness:
      typeof value.causal_chain_completeness === "string"
        ? value.causal_chain_completeness
        : null,
    mechanism_support_source:
      typeof value.mechanism_support_source === "string" ? value.mechanism_support_source : null,
  };
}

function candidateTrace(value: unknown): RcaCandidateTrace | null {
  if (!isRecord(value) || typeof value.root_cause !== "string") return null;
  return {
    root_cause: value.root_cause,
    score: typeof value.score === "number" ? value.score : null,
    basis: typeof value.basis === "string" ? value.basis : null,
    status: typeof value.status === "string" ? value.status : null,
    evidence_ids: stringList(value.evidence_ids),
    supporting_evidence_ids: stringList(value.supporting_evidence_ids),
    contradicting_evidence_ids: stringList(value.contradicting_evidence_ids),
    rejection_reasons: stringList(value.rejection_reasons),
    causal_matrix_status:
      value.causal_matrix_status === null || value.causal_matrix_status === undefined
        ? null
        : causalStatus(value.causal_matrix_status),
    causal_chain_completeness:
      typeof value.causal_chain_completeness === "string"
        ? value.causal_chain_completeness
        : null,
    data_missing_evidence_ids: stringList(value.data_missing_evidence_ids),
    mechanism_support_source:
      typeof value.mechanism_support_source === "string" ? value.mechanism_support_source : null,
    causal_evidence_matrix: causalMatrix(value.causal_evidence_matrix),
  };
}

function diagnosisTrace(value: unknown, findingId: string): RcaDiagnosisTrace | undefined {
  if (!isRecord(value)) return undefined;
  const candidates = Array.isArray(value.ranked_candidates)
    ? value.ranked_candidates.flatMap((item) => {
        const candidate = candidateTrace(item);
        return candidate ? [candidate] : [];
      })
    : [];
  const gaps = Array.isArray(value.causal_evidence_gaps)
    ? value.causal_evidence_gaps.flatMap((item) => {
        if (!isRecord(item) || typeof item.gap_id !== "string") return [];
        return [{
          gap_id: item.gap_id,
          candidate_index: typeof item.candidate_index === "number" ? item.candidate_index : -1,
          claim: typeof item.claim === "string" ? item.claim : "unknown",
          status: typeof item.status === "string" ? item.status : "unresolved",
          reason: typeof item.reason === "string" ? item.reason : "",
          question_kind: typeof item.question_kind === "string" ? item.question_kind : "",
          allowed_actions: stringList(item.allowed_actions),
          evidence_ids: stringList(item.evidence_ids),
          gap_type: typeof item.gap_type === "string" ? item.gap_type : undefined,
          data_missing_evidence_ids: stringList(item.data_missing_evidence_ids),
          unavailable_sources: Array.isArray(item.unavailable_sources)
            ? item.unavailable_sources.flatMap((source) => {
                const parsed = missingSource(source);
                return parsed ? [parsed] : [];
              })
            : [],
        } satisfies CausalEvidenceGap];
      })
    : [];
  const rawGate = isRecord(value.confirmation_gate) ? value.confirmation_gate : {};
  const rawImpact = isRecord(value.impact_lot_gate) ? value.impact_lot_gate : {};
  const impactRows = Array.isArray(rawImpact.rows)
    ? rawImpact.rows.flatMap((item) => {
        if (!isRecord(item) || typeof item.lot_id !== "string") return [];
        return [{
          lot_id: item.lot_id,
          included: item.included === true,
          candidate_included:
            typeof item.candidate_included === "boolean"
              ? item.candidate_included
              : item.included === true,
          confirmed: item.confirmed === true,
          included_reason: typeof item.included_reason === "string" ? item.included_reason : null,
          excluded_reason: typeof item.excluded_reason === "string" ? item.excluded_reason : null,
          supporting_evidence_ids: stringList(item.supporting_evidence_ids),
          data_missing_evidence_ids: stringList(item.data_missing_evidence_ids),
          non_blocking_data_missing_evidence_ids: stringList(
            item.non_blocking_data_missing_evidence_ids,
          ),
          data_available:
            typeof item.data_available === "boolean" ? item.data_available : undefined,
          checks: isRecord(item.checks)
            ? Object.fromEntries(
                Object.entries(item.checks).flatMap(([key, value]) =>
                  typeof value === "boolean" ? [[key, value]] : [],
                ),
              )
            : undefined,
        } satisfies ImpactLotGateRow];
      })
    : [];
  return {
    finding_id: findingId,
    conclusion_status: typeof value.conclusion_status === "string" ? value.conclusion_status : "inconclusive",
    causal_chain_completeness:
      typeof value.causal_chain_completeness === "string"
        ? value.causal_chain_completeness
        : null,
    data_missing_evidence_ids: stringList(value.data_missing_evidence_ids),
    root_cause: typeof value.root_cause === "string" ? value.root_cause : null,
    ranked_candidates: candidates,
    evidence_synthesis: isRecord(value.evidence_synthesis) ? value.evidence_synthesis : {},
    causal_evidence_gaps: gaps,
    candidate_comparison: isRecord(value.candidate_comparison) ? value.candidate_comparison : {},
    causal_lanes: Array.isArray(value.causal_lanes)
      ? value.causal_lanes.flatMap((item) => {
          const lane = causalLane(item);
          return lane ? [lane] : [];
        })
      : [],
    candidate_challenges: Array.isArray(value.candidate_challenges)
      ? value.candidate_challenges.flatMap((item) => {
          const challenge = candidateChallenge(item);
          return challenge ? [challenge] : [];
        })
      : [],
    competition_trace: competitionTrace(value.competition_trace),
    confirmation_gate: {
      status: typeof rawGate.status === "string" ? rawGate.status : undefined,
      checks: isRecord(rawGate.checks)
        ? Object.fromEntries(Object.entries(rawGate.checks).flatMap(([key, item]) => typeof item === "boolean" ? [[key, item]] : []))
        : undefined,
      reasons: stringList(rawGate.reasons),
      unresolved_gaps: stringList(rawGate.unresolved_gaps),
      causal_chain_completeness:
        typeof rawGate.causal_chain_completeness === "string"
          ? rawGate.causal_chain_completeness
          : null,
      data_missing_evidence_ids: stringList(rawGate.data_missing_evidence_ids),
      blocking_data_missing_evidence_ids: stringList(
        rawGate.blocking_data_missing_evidence_ids,
      ),
      non_blocking_data_missing_evidence_ids: stringList(
        rawGate.non_blocking_data_missing_evidence_ids,
      ),
    },
    impact_lot_gate: {
      scope_status: typeof rawImpact.scope_status === "string" ? rawImpact.scope_status : undefined,
      candidate_scope_status:
        typeof rawImpact.candidate_scope_status === "string"
          ? rawImpact.candidate_scope_status
          : undefined,
      publication_status:
        typeof rawImpact.publication_status === "string"
          ? rawImpact.publication_status
          : undefined,
      scope_basis: typeof rawImpact.scope_basis === "string" ? rawImpact.scope_basis : undefined,
      data_missing_evidence_ids: stringList(rawImpact.data_missing_evidence_ids),
      non_blocking_data_missing_evidence_ids: stringList(
        rawImpact.non_blocking_data_missing_evidence_ids,
      ),
      observed_impact_lots: stringList(rawImpact.observed_impact_lots),
      candidate_impact_lots: stringList(rawImpact.candidate_impact_lots),
      confirmed_impact_lots: stringList(rawImpact.confirmed_impact_lots),
      confirmation_blocked_reason:
        typeof rawImpact.confirmation_blocked_reason === "string"
          ? rawImpact.confirmation_blocked_reason
          : null,
      candidate_scopes: Array.isArray(rawImpact.candidate_scopes)
        ? rawImpact.candidate_scopes.flatMap((item) => {
            if (!isRecord(item) || typeof item.candidate_index !== "number") return [];
            return [{
              candidate_index: item.candidate_index,
              candidate_rank: typeof item.candidate_rank === "number" ? item.candidate_rank : null,
              candidate_root_cause:
                typeof item.candidate_root_cause === "string"
                  ? item.candidate_root_cause
                  : null,
              candidate_scope_status:
                typeof item.candidate_scope_status === "string"
                  ? item.candidate_scope_status
                  : undefined,
              publication_status:
                typeof item.publication_status === "string"
                  ? item.publication_status
                  : undefined,
              candidate_impact_lots: stringList(item.candidate_impact_lots),
              confirmed_impact_lots: stringList(item.confirmed_impact_lots),
              data_missing_evidence_ids: stringList(item.data_missing_evidence_ids),
              non_blocking_data_missing_evidence_ids: stringList(
                item.non_blocking_data_missing_evidence_ids,
              ),
            }];
          })
        : [],
      rows: impactRows,
    },
  };
}

export function authoritativeRcaDiagnosisFor(state: RCAState): RcaDiagnosisTrace | undefined {
  const finding = authoritativeRcaFindingFor(state);
  if (!finding) return undefined;
  if (state.rca_diagnosis?.finding_id === finding.finding_id) {
    return state.rca_diagnosis;
  }
  const details = finding.details;
  return diagnosisTrace(
    {
      finding_id: finding.finding_id,
      conclusion_status: details.conclusion_status ?? details.status,
      causal_chain_completeness: details.causal_chain_completeness,
      data_missing_evidence_ids: details.data_missing_evidence_ids,
      root_cause: details.root_cause,
      ranked_candidates: details.ranked_candidates,
      evidence_synthesis: details.evidence_synthesis,
      causal_evidence_gaps: details.causal_evidence_gaps,
      candidate_comparison: details.candidate_comparison,
      causal_lanes: state.causal_lanes,
      candidate_challenges: state.candidate_challenges ?? details.candidate_challenges,
      competition_trace: state.competition_trace,
      confirmation_gate: details.confirmation_gate,
      impact_lot_gate: details.impact_lot_gate,
    },
    finding.finding_id,
  );
}

export function getRcaCandidates(state: RCAState): RcaCandidateTrace[] {
  return authoritativeRcaDiagnosisFor(state)?.ranked_candidates ?? [];
}

export function getCausalEvidenceGaps(state: RCAState): CausalEvidenceGap[] {
  return authoritativeRcaDiagnosisFor(state)?.causal_evidence_gaps ?? [];
}

export function getYieldTrend(state: RCAState): YieldTrendPoint[] {
  const raw = findingFor(state, "mes")?.details.yield_trend;
  if (!Array.isArray(raw)) return [];
  return raw.filter((item): item is YieldTrendPoint => {
    if (!isRecord(item)) return false;
    return (
      typeof item.date === "string" &&
      typeof item.lot_count === "number" &&
      typeof item.pass_count === "number" &&
      typeof item.fail_count === "number" &&
      typeof item.pass_rate === "number"
    );
  });
}

export function getNormalLotCount(state: RCAState): number {
  const value = findingFor(state, "mes")?.details.normal_lots;
  return Array.isArray(value) ? value.length : 0;
}

export function getTargetChamber(state: RCAState): string {
  const value = findingFor(state, "mes")?.details.target_commonality;
  if (!isRecord(value)) return "Not available";
  return typeof value.chamber_id === "string" ? value.chamber_id : "Not available";
}

export function getHoldCount(state: RCAState): number {
  const value = findingFor(state, "mes")?.details.hold_count;
  return typeof value === "number" ? value : 0;
}

export function getEvidenceChain(state: RCAState): EvidenceChainItem[] {
  const raw = authoritativeRcaFindingFor(state)?.details.evidence_chain;
  if (!Array.isArray(raw)) return [];
  return raw.filter((item): item is EvidenceChainItem => {
    if (!isRecord(item)) return false;
    return (
      typeof item.stage === "string" &&
      typeof item.claim === "string" &&
      typeof item.confidence === "number" &&
      Array.isArray(item.evidence_ids) &&
      item.evidence_ids.every((id) => typeof id === "string")
    );
  });
}

export function getRecommendedActions(state: RCAState): RecommendedAction[] {
  const raw = authoritativeRcaFindingFor(state)?.details.recommended_actions;
  if (!Array.isArray(raw)) return [];
  return raw.filter((item): item is RecommendedAction => {
    if (!isRecord(item)) return false;
    return (
      typeof item.action === "string" &&
      Array.isArray(item.evidence_ids) &&
      item.evidence_ids.every((id) => typeof id === "string")
    );
  });
}

export function getFdcShifts(state: RCAState): Array<{
  parameter_name: string;
  avg_delta_percent: number;
}> {
  const raw = findingFor(state, "fdc")?.details.parameter_summary;
  if (!Array.isArray(raw)) return [];
  return raw.flatMap((item) => {
    if (
      !isRecord(item) ||
      typeof item.parameter_name !== "string" ||
      typeof item.avg_delta_percent !== "number"
    ) {
      return [];
    }
    return [
      {
        parameter_name: item.parameter_name,
        avg_delta_percent: item.avg_delta_percent,
      },
    ];
  });
}

export function formatAgentName(agent: string): string {
  const names: Record<string, string> = {
    mes: "MES",
    fdc: "FDC",
    defect_wat: "Defect / WAT",
    knowledge: "Knowledge",
    rca_reasoning: "RCA Reasoning",
    improvement: "Improvement",
  };
  return names[agent] ?? agent.replaceAll("_", " ");
}

export function formatTraceLabel(value: string): string {
  const acronyms: Record<string, string> = {
    cmp: "CMP",
    fdc: "FDC",
    id: "ID",
    ids: "IDs",
    llm: "LLM",
    mes: "MES",
    qwen: "Qwen",
    rca: "RCA",
    react: "ReAct",
    spc: "SPC",
    wat: "WAT",
  };
  return value
    .split("_")
    .map(
      (part) =>
        acronyms[part.toLowerCase()] ??
        `${part.charAt(0).toUpperCase()}${part.slice(1)}`,
    )
    .join(" ");
}

interface UniqueIndex<T> {
  values: Map<string, T>;
  duplicates: Set<string>;
}

function indexUnique<T>(
  items: T[],
  idFor: (item: T) => string,
  label: string,
  integrityIssues: string[],
): UniqueIndex<T> {
  const values = new Map<string, T>();
  const duplicates = new Set<string>();
  for (const item of items) {
    const id = idFor(item);
    if (values.has(id)) {
      duplicates.add(id);
      values.delete(id);
    } else if (!duplicates.has(id)) {
      values.set(id, item);
    }
  }
  for (const id of duplicates) {
    integrityIssues.push(`Duplicate ${label} ID: ${id}.`);
  }
  return { values, duplicates };
}

function resolveIds<T>(
  ids: string[],
  index: UniqueIndex<T>,
  label: string,
  integrityIssues: string[],
): T[] {
  return ids.flatMap((id) => {
    const item = index.values.get(id);
    if (item) return [item];
    integrityIssues.push(
      index.duplicates.has(id)
        ? `${label} ID ${id} is ambiguous.`
        : `${label} ID ${id} is missing.`,
    );
    return [];
  });
}

function orchestrationMode(value: unknown): OrchestrationMode | null {
  return value === "fixed" || value === "controlled_react" || value === "llm_react"
    ? value
    : null;
}

function stringValue(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function stringArray(value: unknown): string[] | null {
  if (!Array.isArray(value) || !value.every((item) => typeof item === "string")) {
    return null;
  }
  return value;
}

function integerValue(value: unknown, minimum = 0): number | null {
  return typeof value === "number" && Number.isInteger(value) && value >= minimum
    ? value
    : null;
}

function isToolLatencyRecord(value: unknown): value is ToolLatencyRecord {
  if (!isRecord(value)) return false;
  return (
    typeof value.tool_name === "string" &&
    typeof value.tool_request_id === "string" &&
    typeof value.agent === "string" &&
    (value.outcome === "success" || value.outcome === "failed") &&
    typeof value.duration_ms === "number"
  );
}

const PLANNER_FAILURE_CATEGORIES = new Set([
  "transport_provider_failure",
  "output_parse_error",
  "contract_validation_error",
  "semantic_validation_error",
]);

const INTENT_PLANNER_REASON_CODES = new Set([
  "goal_id_changed",
  "intent_invalid",
  "budget_changed",
  "known_fact_removed",
  "known_fact_changed",
  "forbidden_known_fact_added",
  "unsupported_question_kind",
  "unrequested_material_trace",
  "source_lot_scope_mismatch",
  "malformed_output",
]);

function isPlannerAttemptDiagnostic(
  value: unknown,
): value is PlannerAttemptDiagnostic {
  if (!isRecord(value)) return false;
  const sharedFieldsAreValid =
    value.stage === "intent_planning" &&
    integerValue(value.attempt, 1) !== null &&
    stringValue(value.prompt_name) !== null &&
    stringValue(value.prompt_version) !== null &&
    typeof value.repair_feedback_sent === "boolean" &&
    isRecord(value.candidate_summary) &&
    isRecord(value.baseline_diff) &&
    (value.provider_request_id === null ||
      stringValue(value.provider_request_id) !== null);
  if (!sharedFieldsAreValid) return false;
  if (value.outcome === "success") {
    return (
      value.failure_category === null &&
      value.reason_code === null &&
      value.field_path === null &&
      value.message === null &&
      value.repair_feedback_sent === false
    );
  }
  if (value.outcome !== "failure") return false;
  return (
    typeof value.failure_category === "string" &&
    PLANNER_FAILURE_CATEGORIES.has(value.failure_category) &&
    typeof value.reason_code === "string" &&
    INTENT_PLANNER_REASON_CODES.has(value.reason_code) &&
    (value.field_path === null || stringValue(value.field_path) !== null) &&
    stringValue(value.message) !== null
  );
}

function evaluationStatus(
  state: RCAState,
  requestedMode: OrchestrationMode | null,
  actualMode: OrchestrationMode | null,
  fallbackReason: string | null,
): AgentTraceEvaluationStatus {
  if (state.run_evaluation) return "available";
  const isFallback =
    requestedMode === "llm_react" &&
    (fallbackReason !== null ||
      (actualMode !== null && actualMode !== "llm_react"));
  if (isFallback) return "fallback";
  if (requestedMode === "llm_react" || actualMode === "llm_react") {
    return state.job.status === "running" || state.job.status === "pending"
      ? "pending"
      : "unavailable";
  }
  return "not_applicable";
}

interface SpecialistParseContext {
  state: RCAState;
  finding: AgentFinding;
  action: InvestigationAction;
  actionRecord: ActionRecord;
  evidenceIndex: UniqueIndex<Evidence>;
  toolLatencies: ToolLatencyRecord[];
}

function parseSpecialistToolStep(
  value: unknown,
  rawIndex: number,
  traceSupersededIds: Set<string>,
  context: SpecialistParseContext,
): SpecialistToolStepViewModel | null {
  const issues: string[] = [];
  if (!isRecord(value)) {
    issues.push(`Specialist Tool step ${rawIndex + 1} is not an object.`);
    return null;
  }
  const stepId = stringValue(value.step_id);
  const stepIndex = integerValue(value.step_index, 1);
  const actionId = stringValue(value.action_id);
  const agent = stringValue(value.agent);
  const specialistDecisionId = stringValue(value.decision_id);
  const candidateId = stringValue(value.candidate_id);
  const toolName = stringValue(value.tool_name);
  const parameters = isRecord(value.parameters) ? value.parameters : null;
  const reason = stringValue(value.reason);
  const evidenceIds = stringArray(value.evidence_ids);
  const outputSummary = stringValue(value.output_summary);
  const status =
    value.status === "completed" || value.status === "failed"
      ? value.status
      : null;
  if (
    stepId === null ||
    stepIndex === null ||
    actionId === null ||
    agent === null ||
    specialistDecisionId === null ||
    candidateId === null ||
    toolName === null ||
    parameters === null ||
    reason === null ||
    evidenceIds === null ||
    outputSummary === null ||
    status === null
  ) {
    return null;
  }
  if (stepIndex > 2) {
    issues.push(`Specialist Tool step ${stepId} exceeds the two-step boundary.`);
  }
  if (actionId !== context.action.action_id) {
    issues.push(`Specialist Tool step ${stepId} references a different action.`);
  }
  if (agent !== context.action.agent || agent !== context.finding.agent) {
    issues.push(`Specialist Tool step ${stepId} references a different Agent.`);
  }

  const superseded = traceSupersededIds.has(stepId);
  const evidence = resolveIds(
    evidenceIds,
    context.evidenceIndex,
    "Specialist Tool Evidence",
    issues,
  );
  if (superseded) {
    for (let index = issues.length - 1; index >= 0; index -= 1) {
      if (issues[index]?.startsWith("Specialist Tool Evidence ID")) {
        issues.splice(index, 1);
      }
    }
  } else {
    const actionEvidenceIds = new Set(context.actionRecord.produced_evidence_ids);
    for (const evidenceId of evidenceIds) {
      if (!actionEvidenceIds.has(evidenceId)) {
        issues.push(
          `Effective Specialist Tool Evidence ID ${evidenceId} is not in its ActionRecord.`,
        );
      }
    }
  }

  const toolRequestId =
    `${context.state.job.job_id}:${context.action.action_id}` +
    `:specialist-step-${stepIndex}`;
  const requestMatches = context.toolLatencies.filter(
    (latency) => latency.tool_request_id === toolRequestId,
  );
  const exactLatencyMatches = requestMatches.filter(
    (latency) => latency.tool_name === toolName && latency.agent === agent,
  );
  let latency: ToolLatencyRecord | null = null;
  if (exactLatencyMatches.length === 1) {
    [latency] = exactLatencyMatches;
  } else if (exactLatencyMatches.length > 1) {
    issues.push(`Tool latency ID ${toolRequestId} is duplicated.`);
  } else if (requestMatches.length > 0) {
    issues.push(`Tool latency ID ${toolRequestId} has a Tool or Agent mismatch.`);
  }

  return {
    key: `${context.finding.finding_id}:${stepId}:${rawIndex}`,
    stepId,
    stepIndex,
    actionId,
    specialistDecisionId,
    candidateId,
    toolName,
    parameters,
    reason,
    evidenceIds,
    evidence,
    outputSummary,
    status,
    superseded,
    toolRequestId,
    latency,
    integrityIssues: issues,
  };
}

function parseSpecialistTrace(
  context: SpecialistParseContext,
): SpecialistTraceViewModel | null {
  const raw = context.finding.details.specialist_v2;
  if (raw === undefined || raw === null) return null;
  const issues: string[] = [];
  if (!isRecord(raw)) {
    return {
      findingId: context.finding.finding_id,
      version: null,
      actionId: context.action.action_id,
      agent: context.action.agent,
      toolCallCount: 0,
      stopReason: null,
      analysisSource: null,
      fallbackReason: null,
      validationRetryCount: 0,
      localFallback: false,
      engineeringInterpretation: null,
      supersededStepIds: [],
      toolSteps: [],
      integrityIssues: ["Finding specialist_v2 details are not an object."],
    };
  }

  const actionId = stringValue(raw.action_id) ?? context.action.action_id;
  const agent = stringValue(raw.agent) ?? context.action.agent;
  if (typeof raw.action_id !== "string") {
    issues.push("Specialist trace action_id is missing or invalid.");
  } else if (actionId !== context.action.action_id) {
    issues.push("Specialist trace references a different action.");
  }
  if (typeof raw.agent !== "string") {
    issues.push("Specialist trace Agent is missing or invalid.");
  } else if (agent !== context.action.agent || agent !== context.finding.agent) {
    issues.push("Specialist trace references a different Agent.");
  }

  const version = stringValue(raw.version);
  if (version === null) issues.push("Specialist trace version is missing or invalid.");
  const stopReason = raw.stop_reason === null ? null : stringValue(raw.stop_reason);
  if (raw.stop_reason !== null && stopReason === null) {
    issues.push("Specialist trace stop reason is invalid.");
  }
  const fallbackReason =
    raw.fallback_reason === null ? null : stringValue(raw.fallback_reason);
  if (raw.fallback_reason !== null && fallbackReason === null) {
    issues.push("Specialist trace fallback reason is invalid.");
  }
  const analysisSource =
    raw.analysis_source === "qwen" ||
    raw.analysis_source === "deterministic_fallback"
      ? raw.analysis_source
      : null;
  if (analysisSource === null) {
    issues.push("Specialist trace analysis source is invalid.");
  }
  const validationRetryCount = integerValue(raw.validation_retry_count) ?? 0;
  if (integerValue(raw.validation_retry_count) === null) {
    issues.push("Specialist trace validation retry count is invalid.");
  }
  const supersededStepIds = stringArray(raw.superseded_step_ids) ?? [];
  if (stringArray(raw.superseded_step_ids) === null) {
    issues.push("Specialist trace superseded step IDs are invalid.");
  }
  const rawSteps = Array.isArray(raw.tool_steps) ? raw.tool_steps : [];
  if (!Array.isArray(raw.tool_steps)) {
    issues.push("Specialist trace Tool steps are not an array.");
  }
  const supersededSet = new Set(supersededStepIds);
  const toolSteps = rawSteps.flatMap((step, index) => {
    const parsed = parseSpecialistToolStep(
      step,
      index,
      supersededSet,
      context,
    );
    if (parsed) return [parsed];
    issues.push(`Specialist Tool step ${index + 1} is malformed and was omitted.`);
    return [];
  });
  toolSteps.sort((left, right) => left.stepIndex - right.stepIndex);

  const stepIds = toolSteps.map((step) => step.stepId);
  const stepIndices = toolSteps.map((step) => step.stepIndex);
  for (const duplicate of stepIds.filter(
    (stepId, index) => stepIds.indexOf(stepId) !== index,
  )) {
    issues.push(`Specialist Tool step ID ${duplicate} is duplicated.`);
  }
  for (const duplicate of stepIndices.filter(
    (stepIndex, index) => stepIndices.indexOf(stepIndex) !== index,
  )) {
    issues.push(`Specialist Tool step index ${duplicate} is duplicated.`);
  }
  for (const supersededStepId of supersededStepIds) {
    if (!stepIds.includes(supersededStepId)) {
      issues.push(
        `Superseded Specialist Tool step ${supersededStepId} is missing.`,
      );
    }
  }

  const toolCallCount = integerValue(raw.tool_call_count) ?? toolSteps.length;
  if (integerValue(raw.tool_call_count) === null) {
    issues.push("Specialist trace Tool-call count is invalid.");
  } else if (toolCallCount !== rawSteps.length) {
    issues.push("Specialist trace Tool-call count does not match its Tool steps.");
  }
  const engineeringInterpretation = stringValue(
    context.finding.details.engineering_interpretation,
  );

  return {
    findingId: context.finding.finding_id,
    version,
    actionId,
    agent,
    toolCallCount,
    stopReason,
    analysisSource,
    fallbackReason,
    validationRetryCount,
    localFallback:
      fallbackReason !== null ||
      analysisSource === "deterministic_fallback" ||
      supersededStepIds.length > 0,
    engineeringInterpretation,
    supersededStepIds,
    toolSteps,
    integrityIssues: issues,
  };
}

export function selectAgentTrace(state: RCAState): AgentTraceViewModel {
  const integrityIssues: string[] = [];
  const questions = state.investigation_questions ?? [];
  const capabilityNotices: CapabilityNotice[] = state.capability_notices ?? [];
  const questionEvidenceLinks: QuestionEvidenceLink[] =
    state.question_evidence_links ?? [];
  const decisions = state.planner_decisions ?? [];
  const questionUpdateReviews = state.question_update_reviews ?? [];
  const records = state.action_history ?? [];
  const runEvaluation = state.run_evaluation ?? null;
  const questionIndex = indexUnique(
    questions,
    (question) => question.question_id,
    "Question",
    integrityIssues,
  );
  const decisionIdIndex = indexUnique(
    decisions,
    (decision) => decision.decision_id,
    "PlannerDecision",
    integrityIssues,
  );
  const plannerActionIndex = indexUnique(
    decisions.flatMap((decision) =>
      decision.next_action === null ? [] : [decision.next_action],
    ),
    (action) => action.action_id,
    "Planner Action",
    integrityIssues,
  );
  const recordIndex = indexUnique(
    records,
    (record) => record.action.action_id,
    "ActionRecord",
    integrityIssues,
  );
  const findingIndex = indexUnique(
    state.findings,
    (finding) => finding.finding_id,
    "Finding",
    integrityIssues,
  );
  const evidenceIndex = indexUnique(
    state.evidence,
    (evidence) => evidence.evidence_id,
    "Evidence",
    integrityIssues,
  );
  const evaluations = runEvaluation?.decision_evaluations ?? [];
  const evaluationIndex = indexUnique(
    evaluations,
    (evaluation) => evaluation.decision_id,
    "DecisionEvaluation",
    integrityIssues,
  );
  const questionUpdateReviewsByDecision = new Map<
    string,
    typeof questionUpdateReviews
  >();
  for (const review of questionUpdateReviews) {
    if (!decisionIdIndex.values.has(review.decision_id)) {
      integrityIssues.push(
        `QuestionUpdate review references missing PlannerDecision ${review.decision_id}.`,
      );
      continue;
    }
    const current = questionUpdateReviewsByDecision.get(review.decision_id) ?? [];
    current.push(review);
    questionUpdateReviewsByDecision.set(review.decision_id, current);
  }

  const actualMode = orchestrationMode(
    state.execution_metadata.orchestration_mode,
  );
  const requestedMode =
    orchestrationMode(state.execution_metadata.orchestration_requested_mode) ??
    actualMode;
  const fallbackReason =
    stringValue(state.execution_metadata.orchestration_fallback_reason);
  const fallbackStage =
    state.execution_metadata.orchestration_fallback_stage === "intent_planning" ||
    state.execution_metadata.orchestration_fallback_stage ===
      "next_action_planning"
      ? state.execution_metadata.orchestration_fallback_stage
      : null;
  const fallbackAfterActionCount = integerValue(
    state.execution_metadata.orchestration_fallback_after_action_count,
  );
  const fallbackAttemptCount = integerValue(
    state.execution_metadata.orchestration_fallback_attempt_count,
    1,
  );
  const rawFallbackValidationErrors =
    state.execution_metadata.orchestration_fallback_validation_errors;
  const fallbackValidationErrors = Array.isArray(rawFallbackValidationErrors)
    ? rawFallbackValidationErrors.filter(
        (error): error is string =>
          typeof error === "string" && error.trim().length > 0,
      )
    : [];
  if (
    rawFallbackValidationErrors !== undefined &&
    (!Array.isArray(rawFallbackValidationErrors) ||
      fallbackValidationErrors.length !== rawFallbackValidationErrors.length)
  ) {
    integrityIssues.push("Planner fallback validation diagnostics are malformed.");
  }
  if (
    fallbackAttemptCount !== null &&
    fallbackValidationErrors.length > 0 &&
    fallbackAttemptCount !== fallbackValidationErrors.length
  ) {
    integrityIssues.push(
      "Planner fallback attempt count does not match its validation diagnostics.",
    );
  }
  const rawIntentPlannerAttempts =
    state.execution_metadata.intent_planner_attempt_diagnostics;
  const intentPlannerAttempts = Array.isArray(rawIntentPlannerAttempts)
    ? rawIntentPlannerAttempts.filter(isPlannerAttemptDiagnostic)
    : [];
  if (
    rawIntentPlannerAttempts !== undefined &&
    (!Array.isArray(rawIntentPlannerAttempts) ||
      intentPlannerAttempts.length !== rawIntentPlannerAttempts.length)
  ) {
    integrityIssues.push("Intent Planner attempt diagnostics are malformed.");
  }
  for (const [index, diagnostic] of intentPlannerAttempts.entries()) {
    if (diagnostic.attempt !== index + 1) {
      integrityIssues.push(
        "Intent Planner attempts are not ordered with contiguous attempt numbers.",
      );
      break;
    }
  }
  if (
    fallbackStage === "intent_planning" &&
    fallbackAttemptCount !== null &&
    intentPlannerAttempts.length > 0 &&
    fallbackAttemptCount !== intentPlannerAttempts.length
  ) {
    integrityIssues.push(
      "Intent Planner handoff count does not match its typed attempt trace.",
    );
  }
  const isFallback =
    requestedMode === "llm_react" &&
    (fallbackReason !== null ||
      (actualMode !== null && actualMode !== "llm_react"));
  const status = evaluationStatus(
    state,
    requestedMode,
    actualMode,
    fallbackReason,
  );
  if (runEvaluation && state.investigation_goal) {
    if (runEvaluation.goal_id !== state.investigation_goal.goal_id) {
      integrityIssues.push("RunEvaluation references a different Goal.");
    }
  }
  if (runEvaluation && isFallback) {
    integrityIssues.push(
      "A fallback investigation unexpectedly contains a RunEvaluation.",
    );
  }

  const rawToolLatencies = state.execution_metadata.tool_latencies;
  const toolLatencies = Array.isArray(rawToolLatencies)
    ? rawToolLatencies.filter((item) => {
        const valid = isToolLatencyRecord(item);
        if (!valid) integrityIssues.push("A Tool latency record is malformed.");
        return valid;
      })
    : [];
  const matchedRecords = new Set<ActionRecord>();

  const buildNode = ({
    key,
    origin,
    task = null,
    decision = null,
    action = null,
    actionRecord = null,
  }: {
    key: string;
    origin: AgentTraceOrigin;
    task?: AgentTask | null;
    decision?: PlannerDecision | null;
    action?: InvestigationAction | null;
    actionRecord?: ActionRecord | null;
  }): AgentTraceNodeViewModel => {
    const nodeIssues: string[] = [];
    const duplicateDecisionId =
      decision !== null &&
      decisionIdIndex.duplicates.has(decision.decision_id);
    if (duplicateDecisionId && decision !== null) {
      nodeIssues.push(
        `PlannerDecision ID ${decision.decision_id} is ambiguous.`,
      );
    }
    const evaluation =
      decision === null || duplicateDecisionId
        ? null
        : evaluationIndex.values.get(decision.decision_id) ?? null;
    if (
      status === "available" &&
      decision !== null &&
      !duplicateDecisionId &&
      evaluation === null
    ) {
      nodeIssues.push(
        `DecisionEvaluation for ${decision.decision_id} is missing or ambiguous.`,
      );
    }
    const targetQuestions =
      decision === null
        ? []
        : resolveIds(
            decision.target_question_ids,
            questionIndex,
            "Target Question",
            nodeIssues,
          );
    const nodeQuestionUpdateReviews =
      decision === null
        ? []
        : (questionUpdateReviewsByDecision.get(decision.decision_id) ?? []);
    for (const review of nodeQuestionUpdateReviews) {
      if (
        review.disposition === "accepted" &&
        !decision?.question_updates.some(
          (update) =>
            update.question_id === review.question_id &&
            update.status === review.claimed_status,
        )
      ) {
        nodeIssues.push(
          "An accepted QuestionUpdate review has no matching committed update.",
        );
      }
    }

    let findings: AgentFinding[] = [];
    let evidence: Evidence[] = [];
    if (actionRecord !== null) {
      findings = resolveIds(
        actionRecord.produced_finding_ids,
        findingIndex,
        "Finding",
        nodeIssues,
      );
      evidence = resolveIds(
        actionRecord.produced_evidence_ids,
        evidenceIndex,
        "Action Evidence",
        nodeIssues,
      );
      if (
        action !== null &&
        (actionRecord.action.kind !== action.kind ||
          actionRecord.action.agent !== action.agent)
      ) {
        nodeIssues.push("Planner Action and ActionRecord do not agree.");
      }
    } else if (task !== null) {
      findings = state.findings.filter(
        (finding) => finding.task_id === task.task_id,
      );
      evidence = resolveIds(
        [...new Set(findings.flatMap((finding) => finding.evidence_ids))],
        evidenceIndex,
        "Task Evidence",
        nodeIssues,
      );
    }

    const newEvidence =
      evaluation === null
        ? []
        : resolveIds(
            evaluation.new_evidence_ids,
            evidenceIndex,
            "New Evidence",
            nodeIssues,
          );
    const specialistTraces =
      action === null || actionRecord === null
        ? []
        : findings.flatMap((finding) => {
            const trace = parseSpecialistTrace({
              state,
              finding,
              action,
              actionRecord,
              evidenceIndex,
              toolLatencies,
            });
            return trace ? [trace] : [];
          });
    for (const trace of specialistTraces) {
      nodeIssues.push(
        ...trace.integrityIssues.map(
          (issue) => `Specialist trace ${trace.findingId}: ${issue}`,
        ),
      );
      for (const step of trace.toolSteps) {
        nodeIssues.push(
          ...step.integrityIssues.map(
            (issue) => `Specialist step ${step.stepId}: ${issue}`,
          ),
        );
      }
    }

    return {
      key,
      origin,
      task,
      decision,
      evaluation,
      targetQuestions,
      newQuestions: decision?.new_questions ?? [],
      questionUpdates: decision?.question_updates ?? [],
      questionUpdateReviews: nodeQuestionUpdateReviews,
      action,
      actionRecord,
      findings,
      evidence,
      newEvidence,
      specialistTraces,
      integrityIssues: nodeIssues,
    };
  };

  const nodes: AgentTraceNodeViewModel[] = [];
  const plannerOrigin: AgentTraceOrigin =
    requestedMode === "llm_react" || actualMode === "llm_react"
      ? "llm_react"
      : "legacy";
  for (const [decisionPosition, decision] of decisions.entries()) {
    const action = decision.next_action;
    const duplicatePlannerAction =
      action !== null && plannerActionIndex.duplicates.has(action.action_id);
    let record: ActionRecord | null = null;
    if (action !== null && !duplicatePlannerAction) {
      record = recordIndex.values.get(action.action_id) ?? null;
      if (record !== null) matchedRecords.add(record);
    }
    const node = buildNode({
      key: decisionIdIndex.duplicates.has(decision.decision_id)
        ? `decision:${decision.decision_id}:${decisionPosition}`
        : `decision:${decision.decision_id}`,
      origin: plannerOrigin,
      decision,
      action,
      actionRecord: record,
    });
    if (duplicatePlannerAction && action !== null) {
      node.integrityIssues.push(
        `Planner Action ID ${action.action_id} is ambiguous.`,
      );
    } else if (action !== null && record === null) {
      node.integrityIssues.push(
        `ActionRecord for ${action.action_id} is missing or ambiguous.`,
      );
    }
    nodes.push(node);
  }

  const unmatchedRecords = records.filter((record) => !matchedRecords.has(record));
  for (const [recordPosition, record] of unmatchedRecords.entries()) {
    let origin: AgentTraceOrigin;
    if (isFallback) {
      origin = "controlled_fallback";
    } else if (decisions.length === 0 && actualMode === "controlled_react") {
      origin = "controlled_react";
    } else {
      origin = "legacy";
    }
    const node = buildNode({
      key: recordIndex.duplicates.has(record.action.action_id)
        ? `action:${record.action.action_id}:${recordPosition}`
        : `action:${record.action.action_id}`,
      origin,
      action: record.action,
      actionRecord: record,
    });
    if (recordIndex.duplicates.has(record.action.action_id)) {
      node.integrityIssues.push(
        `ActionRecord ID ${record.action.action_id} is ambiguous.`,
      );
    }
    if (decisions.length > 0 && !isFallback) {
      node.integrityIssues.push(
        "ActionRecord is not linked to a committed PlannerDecision.",
      );
    }
    nodes.push(node);
  }

  if (decisions.length === 0 && records.length === 0 && state.task_plan) {
    for (const task of state.task_plan.tasks) {
      nodes.push(
        buildNode({
          key: `task:${task.task_id}`,
          origin: actualMode === null ? "legacy" : "fixed",
          task,
        }),
      );
    }
  }

  if (status === "available") {
    const decisionIds = new Set(decisions.map((decision) => decision.decision_id));
    for (const evaluation of evaluations) {
      if (!decisionIds.has(evaluation.decision_id)) {
        integrityIssues.push(
          `DecisionEvaluation ${evaluation.decision_id} has no PlannerDecision.`,
        );
      }
    }
  }

  return {
    requestedMode,
    actualMode,
    evaluationStatus: status,
    runEvaluation,
    goal: state.investigation_goal ?? null,
    questions: [...questionIndex.values.values()],
    capabilityNotices,
    questionEvidenceLinks,
    nodes,
    fallbackReason,
    fallbackStage,
    fallbackAfterActionCount,
    fallbackAttemptCount,
    fallbackValidationErrors,
    intentPlannerAttempts,
    integrityIssues,
  };
}
