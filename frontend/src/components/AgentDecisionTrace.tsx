import {
  AlertTriangle,
  BrainCircuit,
  CheckCircle2,
  CircleAlert,
  CircleDashed,
  DatabaseZap,
  GitBranch,
  Wrench,
} from "lucide-react";

import {
  formatAgentName,
  formatTraceLabel,
  selectAgentTrace,
} from "../selectors";
import type {
  AgentFinding,
  AgentTraceEvaluationStatus,
  AgentTraceNodeViewModel,
  InvestigationQuestion,
  CapabilityNotice,
  QuestionEvidenceLink,
  PlannerAttemptDiagnostic,
  QuestionUpdate,
  QuestionUpdateReview,
  RCAState,
  SpecialistToolStepViewModel,
  SpecialistTraceViewModel,
} from "../types";

interface AgentDecisionTraceProps {
  state: RCAState;
}

interface VerdictView {
  value: string;
  detail: string;
  tone: "positive" | "negative" | "neutral";
}

function IntentPlannerAttemptTrace({
  attempts,
}: {
  attempts: PlannerAttemptDiagnostic[];
}) {
  if (attempts.length === 0) return null;
  return (
    <section
      className="intent-planner-trace"
      aria-labelledby="intent-planner-trace-heading"
    >
      <div className="intent-planner-trace-heading">
        <div>
          <span className="section-kicker">User request → Goal and Questions</span>
          <h3 id="intent-planner-trace-heading">Intent Planner handoff trace</h3>
        </div>
        <span className="section-count">
          {attempts.length} {attempts.length === 1 ? "attempt" : "attempts"}
        </span>
      </div>
      <ol className="intent-planner-attempt-list">
        {attempts.map((attempt) => (
          <li
            className={`intent-planner-attempt intent-planner-attempt-${attempt.outcome}`}
            key={attempt.attempt}
          >
            <div className="intent-planner-attempt-title">
              <strong>Attempt {attempt.attempt}</strong>
              <span>{attempt.outcome === "success" ? "Accepted" : "Rejected"}</span>
            </div>
            {attempt.outcome === "success" ? (
              <p>
                Python accepted the Qwen Goal and Question contract and handed it
                to Next Action Planning.
              </p>
            ) : (
              <>
                <dl className="intent-planner-attempt-facts">
                  <div>
                    <dt>Category</dt>
                    <dd>{formatTraceLabel(attempt.failure_category ?? "unknown")}</dd>
                  </div>
                  <div>
                    <dt>Reason code</dt>
                    <dd><code>{attempt.reason_code ?? "unknown"}</code></dd>
                  </div>
                  <div>
                    <dt>Field path</dt>
                    <dd><code>{attempt.field_path ?? "not available"}</code></dd>
                  </div>
                  <div>
                    <dt>Repair feedback</dt>
                    <dd>{attempt.repair_feedback_sent ? "Sent for retry" : "Not sent"}</dd>
                  </div>
                </dl>
                <p>{attempt.message ?? "No bounded validation message was recorded."}</p>
              </>
            )}
            <details className="intent-planner-safe-details">
              <summary>Safe candidate summary and baseline diff</summary>
              <div>
                <span>Candidate summary</span>
                <code>{formatValue(attempt.candidate_summary)}</code>
              </div>
              <div>
                <span>Baseline diff</span>
                <code>{formatValue(attempt.baseline_diff)}</code>
              </div>
            </details>
          </li>
        ))}
      </ol>
    </section>
  );
}

function formatValue(value: unknown): string {
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function recordValue(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function stringValues(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter(
        (item): item is string =>
          typeof item === "string" && item.trim().length > 0,
      )
    : [];
}

function compactRecord(value: unknown, emptyText: string): string {
  const record = recordValue(value);
  if (!record) return emptyText;
  const facts = Object.entries(record).flatMap(([key, item]) => {
    if (typeof item === "string" && item.trim()) {
      return [`${formatTraceLabel(key)}=${item}`];
    }
    const values = stringValues(item);
    return values.map((entry) => `${formatTraceLabel(key)}=${entry}`);
  });
  return facts.length > 0 ? facts.join(" · ") : emptyText;
}

function KnowledgeCausalScopeTrace({ findings }: { findings: AgentFinding[] }) {
  const finding = findings.find(
    (item) =>
      item.agent === "knowledge" &&
      recordValue(item.details.causal_search_scope) !== null,
  );
  if (!finding) return null;
  const scope = recordValue(finding.details.causal_search_scope);
  if (!scope) return null;
  const observation = finding.details.observation_scope;
  const lanes: Array<Record<string, unknown> & { lane: string }> = Array.isArray(
    scope.expansion_lanes,
  )
    ? scope.expansion_lanes.flatMap((item) => {
        const lane = recordValue(item);
        const name = lane && typeof lane.lane === "string" ? lane.lane : null;
        return name && lane ? [{ ...lane, lane: name }] : [];
      })
    : [];
  const selectedLanes = stringValues(finding.details.candidate_lanes);
  const scopeReasons = stringValues(finding.details.scope_reasons);
  const baseHardConstraints = compactRecord(
    scope.hard_constraints,
    "Governance constraints only",
  );
  const timeBoundary =
    typeof scope.time_boundary === "string" && scope.time_boundary
      ? `Created At<=${scope.time_boundary}`
      : null;
  const explicitLimits = stringValues(scope.explicit_user_limits);
  const hardConstraints = [
    ...(baseHardConstraints === "Governance constraints only"
      ? []
      : [baseHardConstraints]),
    ...explicitLimits,
    ...(timeBoundary ? [timeBoundary] : []),
  ];
  return (
    <section className="agent-trace-causal-scope" aria-label="Knowledge causal scope">
      <div className="agent-trace-causal-heading">
        <DatabaseZap size={15} aria-hidden="true" />
        <div>
          <span>Python-owned Scope provenance</span>
          <strong>{formatTraceLabel(String(scope.mode ?? "causal search"))}</strong>
        </div>
      </div>
      <dl>
        <div>
          <dt>Observed at</dt>
          <dd>{compactRecord(observation, "No observation context supplied")}</dd>
        </div>
        <div>
          <dt>Search lanes considered</dt>
          <dd>
            {lanes.length > 0
              ? lanes
                  .map(
                    (lane) =>
                      `${formatTraceLabel(lane.lane)} (${
                        lane.available === true ? "available" : "unavailable"
                      })`,
                  )
                  .join(" · ")
              : "No causal lanes recorded"}
          </dd>
        </div>
        <div>
          <dt>Hard constraints</dt>
          <dd>
            {hardConstraints.length > 0
              ? hardConstraints.join(" · ")
              : "Governance constraints only"}
          </dd>
        </div>
        <div>
          <dt>Soft hints</dt>
          <dd>{compactRecord(scope.soft_hints, "None")}</dd>
        </div>
        <div>
          <dt>Selected candidate lane</dt>
          <dd>
            {selectedLanes.length > 0
              ? selectedLanes.map(formatTraceLabel).join(" · ")
              : "No Knowledge candidate selected"}
          </dd>
        </div>
        <div>
          <dt>Scope reason</dt>
          <dd>
            {scopeReasons.length > 0
              ? scopeReasons.join(" · ")
              : String(scope.scope_reason ?? "No reason recorded")}
          </dd>
        </div>
      </dl>
      {lanes
        .filter((lane) => lane.available !== true && typeof lane.reason === "string")
        .map((lane) => (
          <p key={lane.lane}>
            <strong>{formatTraceLabel(lane.lane)} unavailable:</strong>{" "}
            {String(lane.reason)}
          </p>
        ))}
      <small>
        Candidate provenance and retrieval relevance do not establish current-Lot root cause.
      </small>
    </section>
  );
}

function originLabel(origin: AgentTraceNodeViewModel["origin"]): string {
  const labels: Record<AgentTraceNodeViewModel["origin"], string> = {
    llm_react: "Qwen Planner",
    controlled_react: "Controlled ReAct",
    controlled_fallback: "Controlled fallback",
    fixed: "Fixed workflow",
    legacy: "Legacy trace",
  };
  return labels[origin];
}

function unavailableVerdict(
  status: AgentTraceEvaluationStatus,
  metric: "goal" | "stop",
): VerdictView {
  const metricName = metric === "goal" ? "goal outcome" : "stop boundary";
  if (status === "pending") {
    return {
      value: "Pending",
      detail: `The ${metricName} will be evaluated after the run finishes.`,
      tone: "neutral",
    };
  }
  if (status === "fallback") {
    return {
      value: "Not evaluated",
      detail: `The ${metricName} is not graded after a controlled handoff.`,
      tone: "neutral",
    };
  }
  if (status === "not_applicable") {
    return {
      value: "Not applicable",
      detail: `This execution path has no autonomous ${metricName} evaluation.`,
      tone: "neutral",
    };
  }
  return {
    value: "Unavailable",
    detail: `No trustworthy ${metricName} evaluation is available.`,
    tone: "neutral",
  };
}

function verdict(
  status: AgentTraceEvaluationStatus,
  value: boolean | undefined,
  metric: "goal" | "stop",
): VerdictView {
  if (status !== "available" || value === undefined) {
    return unavailableVerdict(
      status === "available" ? "unavailable" : status,
      metric,
    );
  }
  if (metric === "goal") {
    return value
      ? {
          value: "True",
          detail: "The requested objective was answered at an evidence-appropriate level.",
          tone: "positive",
        }
      : {
          value: "False",
          detail: "The requested objective was not fully answered.",
          tone: "negative",
        };
  }
  return value
    ? {
        value: "True",
        detail: "The Planner stopped at an auditable investigation boundary.",
        tone: "positive",
      }
    : {
        value: "False",
        detail: "The stop boundary needs engineering review.",
        tone: "negative",
      };
}

function VerdictCard({
  label,
  verdictView,
}: {
  label: string;
  verdictView: VerdictView;
}) {
  const Icon =
    verdictView.tone === "positive"
      ? CheckCircle2
      : verdictView.tone === "negative"
        ? CircleAlert
        : CircleDashed;
  return (
    <div className={`agent-trace-verdict verdict-${verdictView.tone}`}>
      <Icon size={18} aria-hidden="true" />
      <div>
        <dt>{label}</dt>
        <dd>{verdictView.value}</dd>
        <p>{verdictView.detail}</p>
      </div>
    </div>
  );
}

function ChipList({
  values,
  emptyText = "None",
}: {
  values: string[];
  emptyText?: string;
}) {
  return (
    <div className="trace-chip-list">
      {values.length > 0 ? (
        values.map((value) => <code key={value}>{value}</code>)
      ) : (
        <em>{emptyText}</em>
      )}
    </div>
  );
}

function ObjectFields({
  label,
  value,
}: {
  label: string;
  value: Record<string, unknown>;
}) {
  return (
    <div className="agent-trace-detail-block">
      <h4>{label}</h4>
      <dl className="agent-trace-object-fields">
        {Object.keys(value).length > 0 ? (
          Object.entries(value).map(([key, item]) => (
            <div key={key}>
              <dt>{formatTraceLabel(key)}</dt>
              <dd>
                <code>{formatValue(item)}</code>
              </dd>
            </div>
          ))
        ) : (
          <div>
            <dt>Status</dt>
            <dd>None recorded</dd>
          </div>
        )}
      </dl>
    </div>
  );
}

function CapabilityNoticeList({ notices }: { notices: CapabilityNotice[] }) {
  if (notices.length === 0) return null;
  return (
    <section className="agent-trace-capability-notices" aria-label="Capability notices">
      <div className="agent-trace-capability-heading">
        <CircleAlert size={16} aria-hidden="true" />
        <div>
          <span>Capability boundary</span>
          <strong>Requested data sources and availability</strong>
        </div>
      </div>
      <ul>
        {notices.map((notice) => (
          <li key={`${notice.request_source}:${notice.capability}`}>
            <div className="agent-trace-question-heading">
              <code>{formatTraceLabel(notice.capability)}</code>
              <span
                className={`question-status question-${notice.supported ? "closed" : "unavailable"}`}
              >
                {notice.supported ? "available" : "unsupported"}
              </span>
            </div>
            <p>{notice.reason}</p>
            {notice.available_alternatives.length > 0 && (
              <p>
                <span>Available alternatives</span>{" "}
                {notice.available_alternatives.map(formatTraceLabel).join(", ")}
              </p>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}

function QuestionEvidenceLinks({ links }: { links: QuestionEvidenceLink[] }) {
  if (links.length === 0) {
    return <p className="empty-copy">No applicable Evidence links.</p>;
  }
  return (
    <ul className="agent-trace-question-links">
      {links.map((link) => (
        <li key={`${link.question_id}:${link.evidence_id}:${link.action_id}:${link.relation}`}>
          <div>
            <code>{link.evidence_id}</code>
            <span className={`question-link-relation relation-${link.relation}`}>
              {link.relation}
            </span>
            <span>{formatTraceLabel(link.matched_evidence_group)}</span>
          </div>
          <small>
            Action <code>{link.action_id}</code> · {link.reason}
          </small>
        </li>
      ))}
    </ul>
  );
}

function QuestionList({
  questions,
  links,
}: {
  questions: InvestigationQuestion[];
  links: QuestionEvidenceLink[];
}) {
  const counts = {
    open: questions.filter((question) => question.status === "open").length,
    closed: questions.filter((question) => question.status === "closed").length,
    unavailable: questions.filter((question) => question.status === "unavailable")
      .length,
  };

  return (
    <div className="agent-trace-questions">
      <dl className="agent-trace-question-summary" aria-label="Investigation question status">
        <div>
          <dt>Open</dt>
          <dd>{counts.open}</dd>
        </div>
        <div>
          <dt>Closed</dt>
          <dd>{counts.closed}</dd>
        </div>
        <div>
          <dt>Unavailable</dt>
          <dd>{counts.unavailable}</dd>
        </div>
      </dl>

      {questions.length > 0 ? (
        <details>
          <summary>Investigation questions ({questions.length})</summary>
          <ol>
            {questions.map((question) => (
              <li key={question.question_id}>
                <div className="agent-trace-question-heading">
                  <code>{question.question_id}</code>
                  <span className="trace-badge trace-badge-neutral">
                    {formatTraceLabel(question.question_kind ?? "unsupported")}
                  </span>
                  <span className={`question-status question-${question.status}`}>
                    {question.status}
                  </span>
                </div>
                <strong>{question.question}</strong>
                <p>{question.rationale}</p>
                {question.answer && (
                  <p>
                    <span>Answer</span>
                    {question.answer}
                  </p>
                )}
                {question.unavailable_reason && (
                  <p>
                    <span>Unavailable reason</span>
                    {question.unavailable_reason}
                  </p>
                )}
                <ChipList
                  values={question.evidence_ids}
                  emptyText="No supporting Evidence"
                />
                <div className="agent-trace-question-groups">
                  <div>
                    <span>Satisfied Evidence groups</span>
                    <ChipList
                      values={question.satisfied_evidence_groups ?? []}
                      emptyText="None"
                    />
                  </div>
                  <div>
                    <span>Missing Evidence groups</span>
                    <ChipList
                      values={question.missing_evidence_groups ?? []}
                      emptyText="None"
                    />
                  </div>
                </div>
                <details className="agent-trace-question-links-details">
                  <summary>
                    Question–Evidence links (
                    {links.filter((link) => link.question_id === question.question_id).length}
                    )
                  </summary>
                  <QuestionEvidenceLinks
                    links={links.filter((link) => link.question_id === question.question_id)}
                  />
                </details>
                {(question.compatible_action_kinds?.length ?? 0) > 0 && (
                  <div className="agent-trace-question-compatible-actions">
                    <span>Compatible Actions</span>
                    <ChipList values={question.compatible_action_kinds ?? []} />
                  </div>
                )}
              </li>
            ))}
          </ol>
        </details>
      ) : (
        <p className="empty-copy">No typed investigation questions were recorded.</p>
      )}
    </div>
  );
}

function QuestionUpdateList({
  updates,
  reviews,
}: {
  updates: QuestionUpdate[];
  reviews: QuestionUpdateReview[];
}) {
  const rejectedReviews = reviews.filter(
    (review) => review.disposition === "rejected",
  );
  const acceptedReviews = reviews.filter(
    (review) => review.disposition === "accepted",
  );
  if (updates.length === 0 && rejectedReviews.length === 0) return null;
  return (
    <div className="agent-trace-question-updates">
      <h4>Question state updates</h4>
      <ul>
        {updates.map((update) => (
          <li
            className="agent-trace-question-update-accepted"
            key={`accepted:${update.question_id}`}
          >
            <div className="agent-trace-question-heading">
              <code>{update.question_id}</code>
              <span className={`question-status question-${update.status}`}>
                Question {update.status}
              </span>
            </div>
            {update.answer && <p>{update.answer}</p>}
            {update.unavailable_reason && <p>{update.unavailable_reason}</p>}
            <ChipList
              values={update.evidence_ids}
              emptyText="No supporting Evidence"
            />
            {acceptedReviews
              .filter((review) => review.question_id === update.question_id)
              .map((review) => (
                <p className="agent-trace-question-review-accepted" key={`accepted-review:${review.update_index}`}>
                  Review accepted: <code>{review.reason_code}</code> — {review.reason}
                </p>
              ))}
          </li>
        ))}
        {rejectedReviews.map((review, index) => (
          <li
            className="agent-trace-question-update-rejected"
            key={`rejected:${review.update_index ?? index}:${review.question_id ?? "unknown"}`}
          >
            <div className="agent-trace-question-heading">
              <code>{review.question_id ?? "Question ID not supplied"}</code>
              <span className="question-status question-rejected">
                QuestionUpdate rejected
              </span>
            </div>
            <div className="agent-trace-question-review-facts">
              <span>
                Claimed status: <code>{review.claimed_status ?? "not supplied"}</code>
              </span>
              <span>
                Reason code: <code>{review.reason_code}</code>
              </span>
            </div>
            <p>{review.reason}</p>
            <p className="agent-trace-question-review-preserved">
              The Agent action was preserved; the invalid Question status claim
              was not committed.
            </p>
          </li>
        ))}
      </ul>
    </div>
  );
}

function EvaluationBadges({ node }: { node: AgentTraceNodeViewModel }) {
  const evaluation = node.evaluation;
  if (!evaluation) {
    return (
      <div className="agent-trace-badges" aria-label="Decision evaluation unavailable">
        <span className="trace-badge trace-badge-neutral">Valid: Not evaluated</span>
        <span className="trace-badge trace-badge-neutral">
          Evidence Gain: Not evaluated
        </span>
        <span className="trace-badge trace-badge-neutral">
          Redundant: Not evaluated
        </span>
      </div>
    );
  }

  return (
    <div className="agent-trace-badges" aria-label="Decision evaluation">
      <span
        className={`trace-badge ${
          evaluation.decision_valid ? "trace-badge-positive" : "trace-badge-negative"
        }`}
      >
        {evaluation.decision_valid ? "Valid: Yes" : "Valid: No"}
      </span>
      <span
        className={`trace-badge ${
          evaluation.evidence_gain ? "trace-badge-positive" : "trace-badge-neutral"
        }`}
      >
        {evaluation.evidence_gain
          ? `Evidence Gain: Yes (+${evaluation.new_evidence_ids.length})`
          : "Evidence Gain: No"}
      </span>
      <span
        className={`trace-badge ${
          evaluation.redundant ? "trace-badge-negative" : "trace-badge-neutral"
        }`}
      >
        {evaluation.redundant ? "Redundant: Yes" : "Redundant: No"}
      </span>
    </div>
  );
}

function IntegrityIssues({
  issues,
  label = "Trace integrity issue",
}: {
  issues: string[];
  label?: string;
}) {
  if (issues.length === 0) return null;
  return (
    <div className="agent-trace-integrity" role="status">
      <AlertTriangle size={15} aria-hidden="true" />
      <div>
        <strong>{label}</strong>
        <ul>
          {issues.map((issue) => (
            <li key={issue}>{issue}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}

function ToolStep({ step }: { step: SpecialistToolStepViewModel }) {
  return (
    <li
      className={`specialist-tool-step ${
        step.superseded ? "tool-step-superseded" : ""
      }`}
    >
      <div className="specialist-tool-heading">
        <div>
          <span>Tool step {step.stepIndex}</span>
          <strong>{formatTraceLabel(step.toolName)}</strong>
        </div>
        <div className="agent-trace-badges">
          <span
            className={`trace-badge ${
              step.status === "completed"
                ? "trace-badge-positive"
                : "trace-badge-negative"
            }`}
          >
            {step.status}
          </span>
          {step.superseded && (
            <span className="trace-badge trace-badge-neutral">
              Superseded / audit only
            </span>
          )}
        </div>
      </div>

      <div className="trace-block">
        <span>Why this Tool</span>
        <p>{step.reason}</p>
      </div>
      <ObjectFields label="Bound parameters" value={step.parameters} />
      <div className="trace-block">
        <span>Tool output</span>
        <p>{step.outputSummary}</p>
      </div>
      <div className="agent-trace-tool-evidence">
        <h5>Evidence</h5>
        {step.evidence.length > 0 ? (
          <ul>
            {step.evidence.map((evidence) => (
              <li key={evidence.evidence_id}>
                <code>{evidence.evidence_id}</code>
                <span>{evidence.observation ?? evidence.summary}</span>
              </li>
            ))}
          </ul>
        ) : step.superseded ? (
          <div className="agent-trace-superseded-evidence">
            <p>
              Audit Evidence from this superseded step is not retained in the
              effective Finding.
            </p>
            <ChipList
              values={step.evidenceIds}
              emptyText="No audit Evidence IDs recorded"
            />
          </div>
        ) : (
          <ChipList values={step.evidenceIds} emptyText="No effective Evidence" />
        )}
      </div>
      <dl className="agent-trace-tool-metadata">
        <div>
          <dt>Latency</dt>
          <dd>
            {step.latency
              ? `${step.latency.duration_ms.toFixed(1)} ms`
              : "Not recorded"}
          </dd>
        </div>
        <div>
          <dt>Outcome</dt>
          <dd>{step.latency?.outcome ?? step.status}</dd>
        </div>
        <div>
          <dt>Request</dt>
          <dd>
            <code>{step.toolRequestId || "Not recorded"}</code>
          </dd>
        </div>
      </dl>
      <IntegrityIssues issues={step.integrityIssues} label="Tool trace integrity issue" />
    </li>
  );
}

function SpecialistTrace({ trace }: { trace: SpecialistTraceViewModel }) {
  return (
    <section className="specialist-trace" aria-label={`${formatAgentName(trace.agent)} Tool trace`}>
      <div className="specialist-trace-heading">
        <div>
          <Wrench size={15} aria-hidden="true" />
          <div>
            <span>{formatAgentName(trace.agent)} Specialist</span>
            <strong>{trace.toolCallCount} bounded Tool calls</strong>
          </div>
        </div>
        <div className="agent-trace-badges">
          <span className="trace-badge trace-badge-neutral">
            {trace.analysisSource
              ? formatTraceLabel(trace.analysisSource)
              : "Analysis source unavailable"}
          </span>
          {trace.localFallback && (
            <span className="trace-badge trace-badge-warning">Local fallback</span>
          )}
        </div>
      </div>

      {trace.engineeringInterpretation && (
        <div className="trace-block">
          <span>Engineering interpretation</span>
          <p>{trace.engineeringInterpretation}</p>
        </div>
      )}
      {(trace.stopReason || trace.fallbackReason) && (
        <dl className="agent-trace-tool-metadata">
          {trace.stopReason && (
            <div>
              <dt>Specialist stop</dt>
              <dd>{formatTraceLabel(trace.stopReason)}</dd>
            </div>
          )}
          {trace.fallbackReason && (
            <div>
              <dt>Fallback reason</dt>
              <dd>{formatTraceLabel(trace.fallbackReason)}</dd>
            </div>
          )}
        </dl>
      )}
      <ol className="specialist-tool-list">
        {trace.toolSteps.map((step) => (
          <ToolStep step={step} key={step.key} />
        ))}
      </ol>
      <IntegrityIssues
        issues={trace.integrityIssues}
        label="Specialist trace integrity issue"
      />
    </section>
  );
}

function FindingAndEvidenceDetails({ node }: { node: AgentTraceNodeViewModel }) {
  return (
    <div className="agent-trace-reference-grid">
      <div>
        <h4>Findings</h4>
        {node.findings.length > 0 ? (
          <ul>
            {node.findings.map((finding) => (
              <li key={finding.finding_id}>
                <code>{finding.finding_id}</code>
                <span>{finding.summary}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="empty-copy">No Finding linked to this node.</p>
        )}
      </div>
      <div>
        <h4>Evidence</h4>
        {node.evidence.length > 0 ? (
          <ul>
            {node.evidence.map((evidence) => (
              <li key={evidence.evidence_id}>
                <code>{evidence.evidence_id}</code>
                <span>{evidence.observation ?? evidence.summary}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="empty-copy">No Evidence linked to this node.</p>
        )}
      </div>
    </div>
  );
}

function ActionNode({
  node,
  index,
}: {
  node: AgentTraceNodeViewModel;
  index: number;
}) {
  const action = node.action;
  const agent = action?.agent ?? node.task?.agent ?? "unknown";
  const title = action
    ? formatTraceLabel(action.kind)
    : node.task?.objective ?? "Recorded action";
  const plannerReason =
    node.decision?.reason ?? action?.reason ?? node.task?.objective ?? "No reason recorded.";
  const findingObservation = node.findings
    .map((finding) => finding.summary)
    .join(" ");
  const observation =
    node.actionRecord?.decision_summary ||
    findingObservation ||
    "No observation was linked to this action.";
  const isHandoff = node.origin === "controlled_fallback";

  return (
    <li className={`agent-trace-node trace-origin-${node.origin}`}>
      <div className="agent-trace-node-index">{index + 1}</div>
      <article>
        <header className="agent-trace-node-heading">
          <div>
            <span className="action-agent">
              {isHandoff ? (
                <GitBranch size={15} aria-hidden="true" />
              ) : (
                <BrainCircuit size={15} aria-hidden="true" />
              )}
              {formatAgentName(agent)}
            </span>
            <h3>{title}</h3>
          </div>
          <div className="agent-trace-badges">
            <span
              className={`trace-badge ${
                isHandoff ? "trace-badge-warning" : "trace-badge-neutral"
              }`}
            >
              {originLabel(node.origin)}
            </span>
            {node.actionRecord && (
              <span className={`trace-badge action-status-${node.actionRecord.status}`}>
                {node.actionRecord.status}
              </span>
            )}
          </div>
        </header>

        <EvaluationBadges node={node} />
        <QuestionUpdateList
          updates={node.questionUpdates}
          reviews={node.questionUpdateReviews}
        />

        <div className="agent-trace-observation-grid">
          <div className="trace-block">
            <span>Planner reason</span>
            <p>{plannerReason}</p>
          </div>
          <div className="trace-block">
            <span>Observed result</span>
            <p>{observation}</p>
          </div>
        </div>

        <KnowledgeCausalScopeTrace findings={node.findings} />

        {node.specialistTraces
          .filter((trace) => trace.engineeringInterpretation)
          .map((trace) => (
            <div className="agent-trace-interpretation" key={trace.findingId}>
              <span>Specialist interpretation</span>
              <p>{trace.engineeringInterpretation}</p>
            </div>
          ))}

        <details className="agent-trace-technical">
          <summary>Technical details</summary>
          {action && (
            <div className="agent-trace-action-contract">
              <ObjectFields label="Action input" value={action.inputs} />
              <ObjectFields label="Investigation scope" value={action.scope} />
            </div>
          )}
          <div className="agent-trace-detail-block">
            <h4>Target questions</h4>
            <ChipList
              values={
                node.decision?.target_question_ids ??
                node.targetQuestions.map((question) => question.question_id)
              }
              emptyText="No targeted question"
            />
          </div>
          <FindingAndEvidenceDetails node={node} />
          {node.evaluation && (
            <div className="trace-block">
              <span>Evaluation reason</span>
              <p>{node.evaluation.reason}</p>
            </div>
          )}
          {node.specialistTraces.map((trace) => (
            <SpecialistTrace trace={trace} key={trace.findingId} />
          ))}
          <IntegrityIssues issues={node.integrityIssues} />
        </details>
      </article>
    </li>
  );
}

function StopNode({
  node,
  index,
  state,
}: {
  node: AgentTraceNodeViewModel;
  index: number;
  state: RCAState;
}) {
  const decision = node.decision;
  return (
    <li className={`agent-trace-node agent-trace-stop trace-origin-${node.origin}`}>
      <div className="agent-trace-node-index">
        <CheckCircle2 size={15} aria-hidden="true" />
      </div>
      <article>
        <header className="agent-trace-node-heading">
          <div>
            <span className="action-agent">
              <BrainCircuit size={15} aria-hidden="true" />
              Planner terminal decision
            </span>
            <h3>Stop investigation</h3>
          </div>
          <span className="trace-badge trace-badge-neutral">
            {originLabel(node.origin)}
          </span>
        </header>
        {decision && (
          <>
            <div className="trace-block">
              <span>Planner reason</span>
              <p>{decision.reason}</p>
            </div>
            <dl className="agent-trace-stop-facts">
              <div>
                <dt>Stop reason</dt>
                <dd>
                  {decision.stop_reason
                    ? formatTraceLabel(decision.stop_reason)
                    : "Not recorded"}
                </dd>
              </div>
              <div>
                <dt>Goal status</dt>
                <dd>{formatTraceLabel(decision.goal_status)}</dd>
              </div>
              <div>
                <dt>Proposed conclusion</dt>
                <dd>{formatTraceLabel(decision.proposed_conclusion_level)}</dd>
              </div>
              <div>
                <dt>Evidence-gated conclusion</dt>
                <dd>
                  {state.conclusion_level
                    ? formatTraceLabel(state.conclusion_level)
                    : "Not available"}
                </dd>
              </div>
            </dl>
          </>
        )}
        <div className="agent-trace-detail-block">
          <h4>Remaining evidence gaps</h4>
          <ChipList values={state.evidence_gaps ?? []} emptyText="None" />
        </div>
        <QuestionUpdateList
          updates={node.questionUpdates}
          reviews={node.questionUpdateReviews}
        />
        <EvaluationBadges node={node} />
        {node.evaluation && (
          <div className="trace-block">
            <span>Evaluation reason</span>
            <p>{node.evaluation.reason}</p>
          </div>
        )}
        <IntegrityIssues issues={node.integrityIssues} />
      </article>
      <span className="agent-trace-node-position" aria-hidden="true">
        {index + 1}
      </span>
    </li>
  );
}

function GoalSummary({ trace }: { trace: ReturnType<typeof selectAgentTrace> }) {
  const goal = trace.goal;
  if (!goal) {
    return <p className="empty-copy">No typed autonomous investigation goal was recorded.</p>;
  }
  return (
    <div className="investigation-goal agent-trace-goal">
      <div>
        <span>Goal intent</span>
        <strong>{formatTraceLabel(goal.intent)}</strong>
      </div>
      <div className="goal-summary">
        <span>Investigation objective</span>
        <strong>{goal.summary}</strong>
      </div>
      <div>
        <span>Safety budget</span>
        <strong>
          {goal.max_steps} steps / {goal.max_tool_calls} Tool calls
        </strong>
      </div>
      {Object.keys(goal.known_facts).length > 0 && (
        <div className="goal-facts">
          <span>Known facts</span>
          <div className="trace-chip-list">
            {Object.entries(goal.known_facts).map(([key, value]) => (
              <code key={key}>
                {key}: {formatValue(value)}
              </code>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function evaluationMessage(
  status: AgentTraceEvaluationStatus,
  fallbackReason: string | null,
): string {
  if (status === "pending") {
    return "Decision evaluation is pending until the investigation reaches a terminal state.";
  }
  if (status === "fallback") {
    return fallbackReason
      ? `Evaluation is not attributed after the compatibility cutover: ${formatTraceLabel(
          fallbackReason,
        )}.`
      : "Evaluation is not attributed after the compatibility cutover.";
  }
  if (status === "not_applicable") {
    return "Decision evaluation applies only to a complete autonomous Planner trace.";
  }
  return "A complete and trustworthy autonomous decision evaluation is unavailable.";
}

export function AgentDecisionTrace({ state }: AgentDecisionTraceProps) {
  const trace = selectAgentTrace(state);
  const hasControlledHandoff =
    trace.evaluationStatus === "fallback" ||
    trace.nodes.some((node) => node.origin === "controlled_fallback");
  const hasFinalizerProjection =
    state.execution_metadata.terminal_stop_projection_applied === true &&
    state.execution_metadata.terminal_stop_projection_trace ===
      "execution_metadata_only" &&
    state.execution_metadata.terminal_stop_projected_by ===
      "python_investigation_finalizer";
  const title = hasControlledHandoff
    ? "Planner Trace + Controlled Handoff"
    : "Autonomous Agent Trace";
  const goalVerdict = verdict(
    trace.evaluationStatus,
    trace.runEvaluation?.goal_success,
    "goal",
  );
  const stopVerdict = verdict(
    trace.evaluationStatus,
    trace.runEvaluation?.stop_correct,
    "stop",
  );

  return (
    <section
      className="workflow-section agent-trace-section"
      aria-labelledby="agent-trace-heading"
    >
      <div className="section-heading-row agent-trace-heading">
        <div>
          <span className="section-kicker">
            {hasControlledHandoff
              ? "Planner trace and compatibility continuation"
              : "Plan → Act → Observe → Re-plan"}
          </span>
          <h2 id="agent-trace-heading">{title}</h2>
        </div>
        <span className="section-count">{trace.nodes.length} trace nodes</span>
      </div>

      <dl className="agent-trace-verdict-grid">
        <VerdictCard label="Goal Success" verdictView={goalVerdict} />
        <VerdictCard label="Stop Correct" verdictView={stopVerdict} />
      </dl>
      <p className="agent-trace-evaluation-summary">
        {trace.runEvaluation?.summary ??
          evaluationMessage(trace.evaluationStatus, trace.fallbackReason)}
      </p>

      <IntentPlannerAttemptTrace attempts={trace.intentPlannerAttempts} />

      {hasControlledHandoff && (
        <div className="agent-trace-handoff" role="status">
          <GitBranch size={17} aria-hidden="true" />
          <div>
            <strong>Controlled compatibility handoff</strong>
            <p>
              The Planner handed off
              {trace.fallbackAfterActionCount !== null
                ? ` after ${trace.fallbackAfterActionCount} committed actions`
                : ""}
              {trace.fallbackStage
                ? ` during ${formatTraceLabel(trace.fallbackStage)}`
                : ""}
              {trace.fallbackReason
                ? ` because ${formatTraceLabel(trace.fallbackReason)}`
                : ""}.
            </p>
            {trace.fallbackAttemptCount !== null && (
              <p>
                Qwen output validation failed on {trace.fallbackAttemptCount}
                {trace.fallbackAttemptCount === 1 ? " attempt" : " attempts"}.
              </p>
            )}
            {trace.fallbackValidationErrors.length > 0 &&
              !(
                trace.fallbackStage === "intent_planning" &&
                trace.intentPlannerAttempts.length > 0
              ) && (
              <details className="agent-trace-fallback-diagnostics">
                <summary>
                  Planner validation diagnostics (
                  {trace.fallbackValidationErrors.length})
                </summary>
                <ol>
                  {trace.fallbackValidationErrors.map((error, index) => (
                    <li key={`${index}:${error}`}>
                      <code>{error}</code>
                    </li>
                  ))}
                </ol>
              </details>
            )}
          </div>
        </div>
      )}

      <GoalSummary trace={trace} />
      <CapabilityNoticeList notices={trace.capabilityNotices} />
      <QuestionList
        questions={trace.questions}
        links={trace.questionEvidenceLinks}
      />
      <IntegrityIssues issues={trace.integrityIssues} label="Agent trace integrity issue" />

      {trace.nodes.length > 0 ? (
        <ol className="agent-trace-list">
          {trace.nodes.map((node, index) =>
            node.decision?.decision_type === "stop" ? (
              <StopNode node={node} index={index} state={state} key={node.key} />
            ) : (
              <ActionNode node={node} index={index} key={node.key} />
            ),
          )}
        </ol>
      ) : (
        <div className="agent-trace-empty">
          <DatabaseZap size={20} aria-hidden="true" />
          <p>No autonomous Planner decisions have been recorded yet.</p>
        </div>
      )}

      {hasControlledHandoff && (
        <aside
          className="agent-trace-handoff-outcome"
          aria-labelledby="handoff-outcome-heading"
        >
          <div>
            <GitBranch size={16} aria-hidden="true" />
            <h3 id="handoff-outcome-heading">Controlled handoff outcome</h3>
          </div>
          <dl>
            <div>
              <dt>Goal status</dt>
              <dd>
                {state.goal_status
                  ? formatTraceLabel(state.goal_status)
                  : "Not available"}
              </dd>
            </div>
            <div>
              <dt>Conclusion</dt>
              <dd>
                {state.conclusion_level
                  ? formatTraceLabel(state.conclusion_level)
                  : "Not available"}
              </dd>
            </div>
            <div>
              <dt>Stop reason</dt>
              <dd>
                {state.stop_reason
                  ? formatTraceLabel(state.stop_reason)
                  : "Not available"}
              </dd>
            </div>
            <div>
              <dt>Remaining gaps</dt>
              <dd>
                {state.evidence_gaps && state.evidence_gaps.length > 0
                  ? state.evidence_gaps.join(", ")
                  : "None"}
              </dd>
            </div>
          </dl>
        </aside>
      )}

      {hasFinalizerProjection && !hasControlledHandoff && (
        <aside
          className="agent-trace-handoff-outcome"
          aria-labelledby="finalizer-outcome-heading"
        >
          <div>
            <CheckCircle2 size={16} aria-hidden="true" />
            <h3 id="finalizer-outcome-heading">Python Finalizer outcome</h3>
          </div>
          <p>
            The Planner proposal is preserved separately from the authoritative
            terminal state.
          </p>
          <dl>
            <div>
              <dt>Goal status</dt>
              <dd>
                {state.goal_status
                  ? formatTraceLabel(state.goal_status)
                  : "Not available"}
              </dd>
            </div>
            <div>
              <dt>Conclusion</dt>
              <dd>
                {state.conclusion_level
                  ? formatTraceLabel(state.conclusion_level)
                  : "Not available"}
              </dd>
            </div>
            <div>
              <dt>Stop reason</dt>
              <dd>
                {state.stop_reason
                  ? formatTraceLabel(state.stop_reason)
                  : "Not available"}
              </dd>
            </div>
          </dl>
        </aside>
      )}
    </section>
  );
}
