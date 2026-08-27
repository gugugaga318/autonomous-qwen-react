import { Check, CircleAlert, GitCompareArrows, ShieldCheck } from "lucide-react";

import { authoritativeRcaDiagnosisFor } from "../selectors";
import type { CausalMatrixClaim, RCAState, RcaCandidateTrace } from "../types";

interface RcaDiagnosisTraceProps {
  state: RCAState;
}

const CLAIM_ORDER = [
  "equipment",
  "chamber",
  "operation",
  "parameter",
  "outcome",
  "mechanism",
  "control",
  "contradiction",
  "temporal",
  "scope",
];

function label(value: string): string {
  return value.replaceAll("_", " ");
}

function StatusPill({ status }: { status: string }) {
  const Icon = status === "supported" ? Check : CircleAlert;
  return (
    <span className={`causal-status causal-status-${status}`}>
      <Icon size={12} aria-hidden="true" />
      {label(status)}
    </span>
  );
}

function EvidenceIds({ ids }: { ids: string[] }) {
  if (ids.length === 0) return <span className="causal-muted">No Evidence IDs</span>;
  return (
    <div className="causal-evidence-ids">
      {ids.map((id) => <code key={id}>{id}</code>)}
    </div>
  );
}

function Matrix({ candidate }: { candidate: RcaCandidateTrace }) {
  const matrix = candidate.causal_evidence_matrix;
  if (!matrix) return <p className="causal-muted">Matrix unavailable for this candidate.</p>;
  const claims = CLAIM_ORDER.flatMap((claim) => {
    const item = matrix.claims[claim];
    return item ? [item] : [];
  });
  for (const [claim, item] of Object.entries(matrix.claims)) {
    if (!CLAIM_ORDER.includes(claim)) claims.push(item);
  }
  return (
    <div className="causal-matrix" aria-label="Causal evidence matrix">
      {claims.map((claim) => <ClaimRow claim={claim} key={claim.claim} />)}
      {matrix.causal_chain && (
        <div className="causal-chain-summary">
          <div className="causal-claim-heading">
            <strong>Causal chain</strong>
            <StatusPill status={matrix.causal_chain.status} />
          </div>
          <div className="causal-chain-stages">
            {Object.entries(matrix.causal_chain.stages).map(([stage, status]) => (
              <span key={stage}>
                {label(stage)}: <strong>{label(status)}</strong>
              </span>
            ))}
          </div>
          {matrix.causal_chain.reason && <p>{matrix.causal_chain.reason}</p>}
          {matrix.data_missing_evidence_ids && matrix.data_missing_evidence_ids.length > 0 && (
            <>
              <strong>Unavailable source Evidence</strong>
              <EvidenceIds ids={matrix.data_missing_evidence_ids} />
            </>
          )}
        </div>
      )}
      {matrix.invalid_evidence_ids.length > 0 && (
        <div className="causal-invalid-evidence">
          <strong>Invalid Evidence references</strong>
          <EvidenceIds ids={matrix.invalid_evidence_ids} />
        </div>
      )}
    </div>
  );
}

function ClaimRow({ claim }: { claim: CausalMatrixClaim }) {
  return (
    <div className="causal-claim-row">
      <div className="causal-claim-heading">
        <strong>{label(claim.claim)}</strong>
        <StatusPill status={claim.status} />
      </div>
      <p>{claim.reason || "No diagnostic explanation available."}</p>
      <EvidenceIds ids={claim.evidence_ids} />
    </div>
  );
}

function CandidateCard({ candidate, index, authoritative }: {
  candidate: RcaCandidateTrace;
  index: number;
  authoritative: boolean;
}) {
  return (
    <article className={`causal-candidate ${authoritative ? "causal-candidate-authoritative" : ""}`}>
      <div className="causal-candidate-heading">
        <div>
          <span className="field-label">Candidate {String.fromCharCode(65 + index)}</span>
          <h3>{candidate.root_cause}</h3>
        </div>
        {authoritative && <span className="causal-authoritative">Authoritative</span>}
      </div>
      <div className="causal-candidate-meta">
        <StatusPill status={candidate.causal_matrix_status ?? candidate.status ?? "unavailable"} />
        {candidate.basis && <span>basis: {label(candidate.basis)}</span>}
        {candidate.mechanism_support_source && <span>mechanism: {label(candidate.mechanism_support_source)}</span>}
      </div>
      <Matrix candidate={candidate} />
      {candidate.rejection_reasons.length > 0 && (
        <div className="causal-rejected">
          <strong>Rejected / downgraded</strong>
          <ul>{candidate.rejection_reasons.map((reason) => <li key={reason}>{reason}</li>)}</ul>
        </div>
      )}
    </article>
  );
}

export function RcaDiagnosisTrace({ state }: RcaDiagnosisTraceProps) {
  const diagnosis = authoritativeRcaDiagnosisFor(state);
  if (!diagnosis) return null;
  const preferred = diagnosis.candidate_comparison.preferred_candidate_index;
  const preferredIndex = typeof preferred === "number" ? preferred : null;
  const comparisonExplanation = typeof diagnosis.candidate_comparison.comparison_explanation === "string"
    ? diagnosis.candidate_comparison.comparison_explanation
    : null;
  const comparisonUnresolved = diagnosis.candidate_comparison.unresolved === true;
  const gate = diagnosis.confirmation_gate;
  const gateChecks = gate.checks ? Object.entries(gate.checks) : [];
  const blockingConfirmationMissing = gate.blocking_data_missing_evidence_ids ?? [];
  const nonBlockingConfirmationMissing = gate.non_blocking_data_missing_evidence_ids ?? [];
  const hasClassifiedConfirmationMissing = blockingConfirmationMissing.length > 0
    || nonBlockingConfirmationMissing.length > 0;
  const gaps = diagnosis.causal_evidence_gaps;
  const impactRows = diagnosis.impact_lot_gate.rows ?? [];
  const candidateImpactCount = diagnosis.impact_lot_gate.candidate_impact_lots?.length
    ?? impactRows.filter((row) => row.candidate_included ?? row.included).length;
  const confirmedImpactCount = diagnosis.impact_lot_gate.confirmed_impact_lots?.length ?? 0;
  const candidateImpactScopes = diagnosis.impact_lot_gate.candidate_scopes ?? [];
  const competition = diagnosis.competition_trace;
  const showCompetition = diagnosis.causal_lanes.length > 0
    || diagnosis.candidate_challenges.length > 0
    || competition !== null;
  const showImpact = impactRows.length > 0
    || typeof diagnosis.impact_lot_gate.scope_status === "string"
    || typeof diagnosis.impact_lot_gate.scope_basis === "string"
    || (diagnosis.impact_lot_gate.data_missing_evidence_ids?.length ?? 0) > 0;

  return (
    <section className="rca-diagnosis-section" aria-labelledby="rca-diagnosis-heading">
      <div className="section-heading-row">
        <div>
          <span className="section-kicker">Python-owned validation trace</span>
          <h2 id="rca-diagnosis-heading">RCA Diagnosis</h2>
        </div>
        <div className="causal-diagnosis-status">
          <StatusPill status={diagnosis.conclusion_status} />
          <span>{diagnosis.ranked_candidates.length} candidate{diagnosis.ranked_candidates.length === 1 ? "" : "s"}</span>
        </div>
      </div>

      {diagnosis.ranked_candidates.length > 0 ? (
        <div className="causal-candidate-grid">
          {diagnosis.ranked_candidates.map((candidate, index) => (
            <CandidateCard
              candidate={candidate}
              index={index}
              authoritative={preferredIndex === index || (preferredIndex === null && candidate.root_cause === diagnosis.root_cause)}
              key={`${candidate.root_cause}:${index}`}
            />
          ))}
        </div>
      ) : (
        <p className="empty-copy">No valid causal candidates were produced.</p>
      )}

      {comparisonExplanation && (
        <div className={`causal-comparison-note ${comparisonUnresolved ? "causal-comparison-unresolved" : ""}`}>
          <GitCompareArrows size={15} aria-hidden="true" />
          <div>
            <strong>Candidate comparison</strong>
            <span>{comparisonExplanation}</span>
          </div>
        </div>
      )}

      {showCompetition && (
        <section className="causal-competition-panel" aria-labelledby="causal-competition-heading">
          <div className="causal-subheading">
            <GitCompareArrows size={16} aria-hidden="true" />
            <h3 id="causal-competition-heading">Candidate &amp; Lane Competition</h3>
            <StatusPill status={competition?.competition_status ?? "not_evaluated"} />
            <span>{competition?.challenge_round_count ?? 0} challenge round(s)</span>
          </div>
          {competition && (
            <div className="causal-comparison-note">
              <GitCompareArrows size={15} aria-hidden="true" />
              <div>
                <strong>
                  {competition.competition_type} · {competition.competition_requirement}
                </strong>
                <span>Lane search: {competition.alternative_search_status}</span>
                {(competition.competition_axes?.length ?? 0) > 0 && (
                  <span>Root-cause axes: {competition.competition_axes?.join(", ")}</span>
                )}
                <span>
                  Scope assessment: {competition.scope_assessment_status ?? "not_evaluated"}
                </span>
                {competition.competition_failure_reason && (
                  <span>Competition failure: {competition.competition_failure_reason}</span>
                )}
              </div>
            </div>
          )}
          {diagnosis.causal_lanes.length > 0 && (
            <div className="causal-lane-list">
              {diagnosis.causal_lanes.map((lane) => (
                <article key={lane.lane_id}>
                  <div>
                    <strong>{lane.lane_id}</strong>
                    <StatusPill status={lane.investigation_status} />
                  </div>
                  <p>
                    {[lane.module, lane.operation_name, lane.operation, lane.equipment, lane.chamber, lane.recipe]
                      .filter(Boolean)
                      .join(" · ") || "Lane context unavailable"}
                  </p>
                  {lane.parameter_scope.length > 0 && (
                    <span>Parameters: {lane.parameter_scope.join(", ")}</span>
                  )}
                  {lane.pruned_reason && <span>{lane.pruned_reason}</span>}
                </article>
              ))}
            </div>
          )}
          {diagnosis.candidate_challenges.length > 0 && (
            <div className="causal-challenge-list">
              {diagnosis.candidate_challenges.map((challenge, index) => (
                <article key={`${challenge.candidate_id}:${index}`}>
                  <div>
                    <strong>Challenge {index + 1}</strong>
                    <StatusPill status={challenge.status} />
                  </div>
                  <p>{challenge.challenge_explanation}</p>
                  {challenge.alternative_candidate_id && (
                    <span>
                      Alternative candidate: {challenge.alternative_candidate_id}
                    </span>
                  )}
                  <span>Challenge kind: {challenge.challenge_kind}</span>
                  {challenge.evidence_probe_lane_id && (
                    <span>Evidence probe Lane: {challenge.evidence_probe_lane_id}</span>
                  )}
                  {challenge.strongest_alternative_lane_id && (
                    <span>Strongest alternative: {challenge.strongest_alternative_lane_id}</span>
                  )}
                  {challenge.distinguishing_questions.map((question) => (
                    <span key={question}>Question: {question}</span>
                  ))}
                  {challenge.unexplained_precursor_evidence_ids.length > 0 && (
                    <div className="causal-missing-source-note">
                      <strong>Unexplained precursor Evidence</strong>
                      <EvidenceIds ids={challenge.unexplained_precursor_evidence_ids} />
                    </div>
                  )}
                </article>
              ))}
            </div>
          )}
        </section>
      )}

      <div className="causal-diagnosis-grid">
        <section className="causal-subpanel" aria-labelledby="confirmation-gate-heading">
          <div className="causal-subheading">
            <ShieldCheck size={16} aria-hidden="true" />
            <h3 id="confirmation-gate-heading">Confirmation Gate</h3>
            <StatusPill status={gate.status ?? "unavailable"} />
          </div>
          {gateChecks.length > 0 && (
            <div className="causal-check-list">
              {gateChecks.map(([name, passed]) => (
                <div key={name} className={passed ? "causal-check-pass" : "causal-check-fail"}>
                  {passed ? <Check size={13} aria-hidden="true" /> : <CircleAlert size={13} aria-hidden="true" />}
                  <span>{label(name)}</span>
                  <strong>{passed ? "pass" : "needs evidence"}</strong>
                </div>
              ))}
            </div>
          )}
          {diagnosis.causal_chain_completeness && (
            <div className="causal-gate-summary">
              <span>Causal chain</span>
              <StatusPill status={diagnosis.causal_chain_completeness} />
            </div>
          )}
          {blockingConfirmationMissing.length > 0 && (
            <div className="causal-missing-source-note">
              <strong>Confirmation-blocking source Evidence</strong>
              <EvidenceIds ids={blockingConfirmationMissing} />
            </div>
          )}
          {nonBlockingConfirmationMissing.length > 0 && (
            <div className="causal-missing-source-note">
              <strong>Additional source unavailable; confirmation Evidence has a substitute</strong>
              <EvidenceIds ids={nonBlockingConfirmationMissing} />
            </div>
          )}
          {!hasClassifiedConfirmationMissing
            && (diagnosis.data_missing_evidence_ids?.length ?? 0) > 0 && (
            <div className="causal-missing-source-note">
              <strong>Unavailable source Evidence</strong>
              <EvidenceIds ids={diagnosis.data_missing_evidence_ids ?? []} />
            </div>
          )}
          {(gate.reasons?.length ?? 0) > 0 && (
            <ul className="causal-reason-list">{gate.reasons?.map((reason) => <li key={reason}>{reason}</li>)}</ul>
          )}
        </section>

        <section className="causal-subpanel" aria-labelledby="evidence-gaps-heading">
          <div className="causal-subheading">
            <GitCompareArrows size={16} aria-hidden="true" />
            <h3 id="evidence-gaps-heading">Evidence Gaps</h3>
            <span>{gaps.length}</span>
          </div>
          {gaps.length > 0 ? (
            <ul className="causal-gap-list">
              {gaps.map((gap) => (
                <li key={gap.gap_id}>
                  <div><strong>{label(gap.claim)}</strong><StatusPill status={gap.status} /></div>
                  <p>{gap.reason}</p>
                  <span>Allowed: {gap.allowed_actions.length > 0 ? gap.allowed_actions.map(label).join(", ") : "none"}</span>
                  <EvidenceIds ids={gap.evidence_ids} />
                </li>
              ))}
            </ul>
          ) : <p className="causal-muted">No unresolved causal Evidence gaps.</p>}
        </section>
      </div>

      {showImpact && (
        <section className="causal-impact-panel" aria-labelledby="impact-gate-heading">
          <div className="causal-subheading">
            <GitCompareArrows size={16} aria-hidden="true" />
            <h3 id="impact-gate-heading">Impact Lot Gate</h3>
            <StatusPill
              status={diagnosis.impact_lot_gate.publication_status
                ?? diagnosis.impact_lot_gate.scope_status
                ?? "unavailable"}
            />
            <span>{candidateImpactCount} candidate / {confirmedImpactCount} confirmed</span>
          </div>
          {diagnosis.impact_lot_gate.scope_basis && (
            <p className="causal-impact-basis">{diagnosis.impact_lot_gate.scope_basis}</p>
          )}
          {(diagnosis.impact_lot_gate.data_missing_evidence_ids?.length ?? 0) > 0 && (
            <div className="causal-missing-source-note">
              <strong>Scope data unavailable</strong>
              <EvidenceIds ids={diagnosis.impact_lot_gate.data_missing_evidence_ids ?? []} />
            </div>
          )}
          {diagnosis.impact_lot_gate.confirmation_blocked_reason && (
            <p className="causal-impact-basis">
              Candidate scope is retained for audit; confirmation is withheld because the RCA conclusion is not supported.
            </p>
          )}
          {(diagnosis.impact_lot_gate.non_blocking_data_missing_evidence_ids?.length ?? 0) > 0 && (
            <div className="causal-missing-source-note">
              <strong>Additional source unavailable; substitute scope Evidence is present</strong>
              <EvidenceIds ids={diagnosis.impact_lot_gate.non_blocking_data_missing_evidence_ids ?? []} />
            </div>
          )}
          {candidateImpactScopes.length > 1 && (
            <div className="causal-impact-list">
              {candidateImpactScopes.map((scope) => (
                <div className="causal-impact-excluded" key={scope.candidate_index}>
                  <strong>Candidate {scope.candidate_index + 1}</strong>
                  <StatusPill status={scope.publication_status ?? scope.candidate_scope_status ?? "unconfirmed"} />
                  <span>{scope.candidate_impact_lots.length} candidate / {scope.confirmed_impact_lots.length} confirmed Lots</span>
                  <span>{scope.candidate_root_cause}</span>
                </div>
              ))}
            </div>
          )}
          <div className="causal-impact-list">
            {impactRows.map((row) => {
              const candidateIncluded = row.candidate_included ?? row.included;
              const confirmed = row.confirmed === true;
              return (
                <div className={candidateIncluded ? "causal-impact-included" : "causal-impact-excluded"} key={row.lot_id}>
                  <strong>{row.lot_id}</strong>
                  <StatusPill status={confirmed ? "supported" : candidateIncluded ? "candidate" : "incomplete"} />
                  <span>{candidateIncluded ? row.included_reason : row.excluded_reason}</span>
                  <EvidenceIds ids={row.supporting_evidence_ids} />
                  {(row.data_missing_evidence_ids?.length ?? 0) > 0 && (
                    <EvidenceIds ids={row.data_missing_evidence_ids ?? []} />
                  )}
                  {(row.non_blocking_data_missing_evidence_ids?.length ?? 0) > 0 && (
                    <EvidenceIds ids={row.non_blocking_data_missing_evidence_ids ?? []} />
                  )}
                </div>
              );
            })}
          </div>
          {impactRows.length === 0 && (
            <p className="causal-muted">No candidate Impact Lots were evaluated.</p>
          )}
        </section>
      )}
    </section>
  );
}
