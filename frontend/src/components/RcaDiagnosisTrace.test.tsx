import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import type { RCAState } from "../types";
import { RcaDiagnosisTrace } from "./RcaDiagnosisTrace";

function stateFixture(): RCAState {
  return {
    job: {
      job_id: "JOB_RCA_TRACE",
      user_query: "Find the root cause.",
      investigation_mode: "lot",
      source_lot_id: "LOT_SOURCE",
      product_id: null,
      time_window: {},
      status: "completed",
      created_at: "2026-08-15T00:00:00+00:00",
    },
    task_plan: null,
    current_task_id: null,
    completed_task_ids: [],
    affected_lots: [],
    impact_lots: [],
    affected_wafers: [],
    impact_wafers: [],
    scope_level: "lot",
    impact_criteria: {},
    evidence: [],
    findings: [
      {
        finding_id: "RCA_AUTH",
        agent: "rca_reasoning",
        finding_kind: "hypothesis_ranking",
        summary: "RCA",
        confidence: 0.8,
        evidence_ids: [],
        warnings: [],
        details: {
          conclusion_status: "inconclusive",
          root_cause: "Backside pressure excursion",
          ranked_candidates: [
            {
              root_cause: "Backside pressure excursion",
              score: 0.8,
              basis: "qwen",
              status: "inconclusive",
              evidence_ids: ["EV_PROCESS"],
              supporting_evidence_ids: ["EV_PROCESS"],
              contradicting_evidence_ids: [],
              rejection_reasons: [],
              causal_matrix_status: "incomplete",
              causal_evidence_matrix: {
                root_cause: "Backside pressure excursion",
                status: "incomplete",
                invalid_evidence_ids: [],
                data_missing_evidence_ids: ["EV_FDC_MISSING"],
                causal_chain: {
                  status: "incomplete",
                  stages: { parameter: "unavailable" },
                  evidence_ids: [],
                  missing_stages: ["parameter"],
                  conflicting_stages: [],
                  data_missing_evidence_ids: ["EV_FDC_MISSING"],
                  reason: "Parameter source is unavailable.",
                },
                mechanism_support_source: "empirical_convergence",
                claims: {
                  parameter: {
                    claim: "parameter",
                    status: "supported",
                    evidence_ids: ["EV_PROCESS"],
                    reason: "Parameter is aligned.",
                    facts: {},
                    support_source: null,
                  },
                },
              },
            },
          ],
          causal_evidence_gaps: [
            {
              gap_id: "candidate_0.mechanism.incomplete",
              candidate_index: 0,
              claim: "mechanism",
              status: "incomplete",
              reason: "Mechanism evidence is incomplete.",
              question_kind: "process_mechanism",
              allowed_actions: ["retrieve_knowledge"],
              evidence_ids: [],
            },
          ],
          causal_chain_completeness: "incomplete",
          data_missing_evidence_ids: ["EV_FDC_MISSING"],
          candidate_comparison: { preferred_candidate_index: 0 },
          confirmation_gate: {
            status: "inconclusive",
            checks: { parameter: true, mechanism: false },
            reasons: ["mechanism claim is incomplete."],
            unresolved_gaps: ["mechanism.incomplete"],
            causal_chain_completeness: "incomplete",
            data_missing_evidence_ids: ["EV_FDC_MISSING"],
            blocking_data_missing_evidence_ids: ["EV_FDC_MISSING"],
            non_blocking_data_missing_evidence_ids: ["EV_SPC_BASELINE_MISSING"],
          },
          impact_lot_gate: {
            scope_status: "partial",
            candidate_scope_status: "partial",
            publication_status: "withheld",
            scope_basis: "candidate exposure ∩ process excursion window",
            data_missing_evidence_ids: ["EV_FDC_MISSING"],
            non_blocking_data_missing_evidence_ids: ["EV_SPC_BASELINE_MISSING"],
            observed_impact_lots: ["LOT_IMPACT"],
            candidate_impact_lots: ["LOT_IMPACT"],
            confirmed_impact_lots: [],
            confirmation_blocked_reason: "authoritative RCA conclusion is not supported",
            candidate_scopes: [
              {
                candidate_index: 0,
                candidate_rank: 1,
                candidate_root_cause: "Backside pressure excursion",
                candidate_scope_status: "partial",
                publication_status: "withheld",
                candidate_impact_lots: ["LOT_IMPACT"],
                confirmed_impact_lots: [],
                data_missing_evidence_ids: [],
                non_blocking_data_missing_evidence_ids: ["EV_SPC_BASELINE_MISSING"],
              },
              {
                candidate_index: 1,
                candidate_rank: 2,
                candidate_root_cause: "Alternative recipe excursion",
                candidate_scope_status: "unconfirmed",
                publication_status: "unconfirmed",
                candidate_impact_lots: [],
                confirmed_impact_lots: [],
                data_missing_evidence_ids: [],
                non_blocking_data_missing_evidence_ids: [],
              },
            ],
            rows: [
              {
                lot_id: "LOT_IMPACT",
                included: true,
                candidate_included: true,
                confirmed: false,
                included_reason: "Matching exposure and outcome Evidence.",
                excluded_reason: null,
                supporting_evidence_ids: ["EV_PROCESS"],
                data_missing_evidence_ids: [],
                non_blocking_data_missing_evidence_ids: ["EV_SPC_BASELINE_MISSING"],
              },
            ],
          },
        },
      },
    ],
    hypotheses: [],
    causal_lanes: [
      {
        lane_id: "LANE_PRESSURE",
        operation: "OP_4000",
        equipment: "EQ_01",
        chamber: "CH_01",
        recipe: "RCP_01",
        parameter_scope: ["pressure"],
        exposed_lot_ids: ["LOT_SOURCE"],
        time_window: [],
        initial_evidence_ids: ["EV_PROCESS"],
        priority_score: 0.9,
        investigation_status: "evidence_collected",
        pruned_reason: null,
      },
    ],
    candidate_challenges: [
      {
        candidate_id: "C1",
        alternative_candidate_id: "C2",
        evidence_probe_lane_id: "LANE_PRESSURE",
        challenge_kind: "scope",
        strongest_alternative_lane_id: "LANE_PRESSURE",
        supporting_evidence_ids: ["EV_PROCESS"],
        contradicting_evidence_ids: [],
        unexplained_precursor_evidence_ids: ["EV_PRECURSOR"],
        distinguishing_gap_ids: ["candidate_0.mechanism.incomplete"],
        distinguishing_questions: ["Does the precursor explain the shift?"],
        challenge_explanation: "An earlier precursor remains unexplained.",
        status: "unresolved",
      },
    ],
    competition_trace: {
      active_lane_ids: ["LANE_PRESSURE"],
      overflow_lane_ids: [],
      represented_lane_ids: ["LANE_PRESSURE"],
      unresolved_lane_ids: ["LANE_PRESSURE"],
      eliminated_lane_ids: [],
      blocked_lane_ids: [],
      lane_resolutions: [],
      alternative_search_status: "unresolved",
      competition_requirement: "scope_required",
      competition_status: "failed",
      competition_type: "scope",
      competition_failure_reason: "scope_hypothesis_collapsed",
      candidate_lineage: [],
      challenge_round_count: 1,
      resolution_evidence_ids: [],
    },
    warnings: [],
    report: null,
    llm_usage: [],
    execution_metadata: {},
    authoritative_rca_finding_id: "RCA_AUTH",
  };
}

describe("RcaDiagnosisTrace", () => {
  it("renders candidates, matrix states, gaps, gate, and impact-lot reasons", () => {
    const html = renderToStaticMarkup(<RcaDiagnosisTrace state={stateFixture()} />);
    expect(html).toContain("RCA Diagnosis");
    expect(html).toContain("Backside pressure excursion");
    expect(html).toContain("Mechanism evidence is incomplete");
    expect(html).toContain("Confirmation Gate");
    expect(html).toContain("LOT_IMPACT");
    expect(html).toContain("Matching exposure and outcome Evidence.");
    expect(html).toContain("Causal chain");
    expect(html).toContain("Confirmation-blocking source Evidence");
    expect(html).toContain("confirmation Evidence has a substitute");
    expect(html).toContain("EV_FDC_MISSING");
    expect(html).toContain("EV_SPC_BASELINE_MISSING");
    expect(html).toContain("1 candidate / 0 confirmed");
    expect(html).toContain("Candidate scope is retained for audit");
    expect(html).toContain("Candidate 2");
    expect(html).toContain("Alternative recipe excursion");
    expect(html).toContain("candidate exposure ∩ process excursion window");
    expect(html).toContain("Candidate &amp; Lane Competition");
    expect(html).toContain("LANE_PRESSURE");
    expect(html).toContain("scope_hypothesis_collapsed");
    expect(html).toContain("Unexplained precursor Evidence");
    expect(html).toContain("unresolved");
  });

  it("renders an unavailable impact gate even when no lot rows exist", () => {
    const state = stateFixture();
    const finding = state.findings[0];
    const impact = finding.details.impact_lot_gate as Record<string, unknown>;
    impact.rows = [];
    impact.scope_status = "unavailable";
    impact.confirmed_impact_lots = [];
    const html = renderToStaticMarkup(<RcaDiagnosisTrace state={state} />);
    expect(html).toContain("Impact Lot Gate");
    expect(html).toContain("No candidate Impact Lots were evaluated.");
  });

  it("does not render a diagnosis without an authoritative RCA finding", () => {
    const state = stateFixture();
    state.authoritative_rca_finding_id = "MISSING";
    const html = renderToStaticMarkup(<RcaDiagnosisTrace state={state} />);
    expect(html).toBe("");
  });
});
