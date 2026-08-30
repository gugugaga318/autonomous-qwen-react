# Codex Project Handoff

> Generated: 2026-08-30 (Asia/Hong_Kong)
>
> Purpose: durable handoff for a new Codex Agent with no access to the prior chat.
>
> Current instruction: **pause all business-code changes and do not continue fixing
> `FORMAL_009` until the next session completes the read-only audit described below.**

## 1. Repository and Git state

- Repository: `semiconductor-yield-rca-v1-upgrade`
- Current branch: `codex/batch-26-0-lane-first-evidence-synthesis`
- Remote: `origin`
  (`https://github.com/gugugaga318/YieldMind-RCA-Evidence-Grounded-Multi-Agent-Semiconductor-Yield-Root-Cause-Analysis-System.git`)
- Remote branch was at `2ceb42b` (`Batch 26.5 Harden adversarial RCA competition lifecycle`)
  when this handoff was written.
- The accumulated post-26.5 implementation was committed locally as:

  ```text
  c685bc0 Harden candidate evidence closure and scope-grounded confirmation
  ```

- `c685bc0` contains 25 source/test files, including four new files, with
  approximately 4,746 insertions and 103 deletions.
- The automated `git push` attempt was rejected by the execution environment's
  security reviewer because the remote code payload could not be trusted
  automatically. It was **not** retried or bypassed. Verify `git status -sb` and
  push explicitly from a trusted shell if the branch is still ahead.
- This handoff document is intended to be a separate documentation commit after
  `c685bc0` so implementation and handoff history remain distinguishable.

### Local artifacts that must not be committed blindly

The worktree contains many untracked `outputs/` directories from real Qwen
Smoke Tests, replays, formal regressions, and external-score packets. Some
external-score packets may contain or derive from private Ground Truth. Do not
stage `outputs/` wholesale and do not use private answers during a blind RCA
audit.

The unrelated untracked file below predates the current RCA patch and is not
referenced from the repository:

```text
docs/assets/dashboard-overview.png
```

At the point where `c685bc0` was created, all tracked business-code and test
changes were committed. Remaining untracked paths were the local `outputs/`
artifacts and `docs/assets/dashboard-overview.png`. This document is the only
newly authorized repository file after that commit.

## 2. RCA main flow and data flow

The current architecture is an evidence-driven iterative RCA workflow. The
important path is:

```text
RCAJob
  -> Supervisor / Planner
  -> bounded legal Action selection
  -> existing Specialist Agent + Tool execution
  -> immutable typed Evidence
  -> Specialist Finding
  -> causal Lane inventory and lifecycle
  -> Evidence Synthesis
  -> Qwen Candidate Generator (maximum two candidates)
  -> Candidate semantic validation and Evidence closure
  -> Python Causal Evidence Matrix
  -> Qwen Adversarial Challenge / Candidate comparison
  -> Python deterministic Evidence Gaps
  -> Action Value / Information Gain assessment
  -> targeted investigation with existing capabilities
  -> optional later RCA reasoning round
  -> Python Competition progression and Confirmation Gate
  -> independent Python Impact Lot Gate
  -> authoritative Hypothesis/Finding IDs
  -> Report / Memory / API / Frontend selectors
```

### Primary orchestration modules

- `core/yield_rca_core/workflow.py`: top-level workflow construction.
- `core/yield_rca_core/supervisor.py`: executes the investigation, persists
  Evidence/Findings, updates Lane and Competition state, and applies terminal
  governance.
- `core/yield_rca_core/next_action_planner.py`: exposes only Python-legal
  question/action/gap choices to Qwen and validates Qwen's next action.
- `core/yield_rca_core/question_capability.py`: Question -> Action capability
  registry. Targeted investigation must reuse this registry and existing agents.
- `core/yield_rca_core/tool_layer.py`: typed tool execution and Evidence creation.
- `core/yield_rca_core/specialist_agents.py` and `specialist_v2.py`: specialist
  execution and bounded specialist LLM behavior.

### Typed Evidence and Lane projection

- `core/yield_rca_core/evidence_models.py`: immutable Evidence and Entity models.
- `core/yield_rca_core/evidence_builder.py`: Evidence construction helpers.
- `core/yield_rca_core/evidence_synthesis.py`: objective, ID-traceable Lane-first
  synthesis. It must not invent facts or causal conclusions.
- `core/yield_rca_core/causal_scope.py`: derives causal Lane inventory.
- `core/yield_rca_core/causal_lane_lifecycle.py`: Lane creation, deduplication,
  activation, challenge, deferral, elimination, merge, and bounded active snapshot.
- `core/yield_rca_core/incident_evidence.py`: extracts explicit user-reported
  physical inspection observations into typed Evidence. It must not turn generic
  query prose into an observed mechanism.

### RCA reasoning and competition

- `core/yield_rca_core/rca_reasoning_agent.py`: assembles the evidence-bounded
  reasoning context, invokes candidate generation/challenge/comparison, runs the
  Hypothesis Engine, and records the authoritative RCA Finding.
- `core/yield_rca_core/hypothesis_candidate_generator.py`: bounded Qwen candidate
  generation, semantic profiles, candidate fault isolation, Lane-aware dedup,
  one bounded Evidence-closure repair, and competition assessment.
- `core/yield_rca_core/causal_adversarial.py`: Qwen adversarial challenge under a
  Python-owned output contract.
- `core/yield_rca_core/causal_candidate_comparison.py`: candidate comparison.
- `core/yield_rca_core/hypothesis_engine.py`: constructs/ranks hypothesis records
  from model candidates and Python matrices; it must not synthesize an unrequested
  deterministic root-cause candidate on the Qwen path.

### Python-owned verification and terminal decisions

- `core/yield_rca_core/causal_evidence_matrix.py`: Claim <-> Evidence consistency,
  component status, temporal/scope checks, and mechanism support classification.
- `core/yield_rca_core/causal_evidence_gap.py`: deterministic Evidence Gap creation.
- `core/yield_rca_core/investigation_decision.py`: Evidence Gain / Investigation
  Gain, Action Value, information gain, source availability, ranking impact, and
  budget-aware eligibility.
- `core/yield_rca_core/causal_competition.py`: Competition requirements and trace.
- `core/yield_rca_core/causal_competition_progression.py`: terminal Competition
  state machine.
- `core/yield_rca_core/causal_confirmation.py`: Confirmation Gate and independent
  Impact Lot Gate.
- `core/yield_rca_core/investigation_finalizer.py`: Python-governed termination.

### State and publication surfaces

- `core/yield_rca_core/models.py`: `RCAState`, `AgentFinding`, `Hypothesis`,
  authoritative IDs, serialization, and validation.
- `core/yield_rca_core/report_generator.py`: report reads the authoritative RCA
  Finding, while historical RCA Findings remain available for audit.
- `backend/yield_rca_api/memory.py`, `backend/yield_rca_api/schemas.py`, and
  `frontend/src/selectors.ts`: Memory/API/UI projections must follow authoritative
  IDs, never list position or "latest item" assumptions.

## 3. Python / Qwen responsibility boundary

### Qwen owns

- Proposing a maximum of two evidence-bounded root-cause candidates.
- Explaining the physical causal bridge in engineering language.
- Comparing reasonable alternatives.
- Challenging the leading candidate.
- Selecting which Python-provided Evidence Gap is most worth investigating.
- Revising or re-ranking candidates after genuinely new decision-relevant Evidence.

The public candidate payload remains the original four fields:

```json
{
  "root_cause": "...",
  "causal_explanation": "...",
  "supporting_evidence_ids": ["EV_..."],
  "contradicting_evidence_ids": ["EV_..."]
}
```

`candidate_semantic_profiles` is a companion reasoning contract for explicit
claimed scope, comparison scope, mechanism identity, and distinguishing
predictions. It does not let Qwen override objective Evidence fields.

### Python owns

- Reading equipment, chamber, operation, recipe, Lot, parameter, direction,
  magnitude, time/window, outcome, WAT, control, negative signal, contradiction,
  and Knowledge provenance from typed Evidence.
- Evidence synthesis and prompt bounding.
- Claim/Evidence consistency and atomic Lane matching.
- Candidate parsing, fault isolation, Lane-aware duplicate detection, and
  semantic validation.
- Evidence Matrix and deterministic Evidence Gap generation.
- Tool/Agent action legality, exact scope, budgets, loop control, and lifecycle.
- Investigation Gain and Action Value assessment.
- Competition terminal state, Confirmation Gate, and Impact Lot Gate.
- `conclusion_status` and authoritative publication state.

Python must not replace Qwen hypothesis generation with deterministic RCA rules.
Qwen must not declare Evidence sufficient or set the final supported status.

## 4. Architecture invariants that must be preserved

1. **Typed Evidence is the only source of objective facts.** Qwen prose cannot
   create equipment, parameter, direction, Lot, time, or outcome facts.
2. **Historical RCA Findings are immutable audit history.** The current result is
   selected only by `authoritative_rca_finding_id`; hypotheses similarly use
   `authoritative_hypothesis_id`.
3. **Never infer authority by list position.** "Take the last Finding" and "one
   Finding per Agent" are forbidden.
4. **Candidate-level isolation is mandatory.** One malformed candidate or unknown
   Evidence reference must not invalidate another valid candidate.
5. **Empty candidates are legal.** Invalid Qwen output may be repaired once; a
   second failure yields an inconclusive model result, not a fabricated Python
   candidate and not an automatic `controlled_react` fallback.
6. **Maximum two candidates.** Competition means materially distinct primary
   explanations, not paraphrases or a primary mechanism plus a modifier presented
   as an independent root cause.
7. **Evidence coverage is not claimed scope.** A candidate may cite multiple
   recipes for comparison without claiming a multi-recipe effect. Claimed scope,
   comparison scope, mechanism identity, and predictions are separate semantics.
8. **Per-Lane Evidence must match the exact Lane where required.** Same chamber or
   operation is not sufficient when the claim requires recipe/Lane coverage.
9. **Knowledge is contextual mechanism support.** Approved Knowledge can show that
   a mechanism is engineering-plausible, but cannot alone prove it occurred in the
   current Lot. Mechanism support is not limited only to a hard-coded
   `MechanismRule`; typed current-Lot physical intermediate or valid empirical
   discrimination may also support it.
10. **Controls and exclusion evidence are informative, not universal hard gates.**
    They can strengthen a conclusion, but absence does not automatically make a
    candidate impossible unless a case-specific deterministic check requires it.
11. **`data_missing` is not contradiction.** Source unavailable/no-match/insufficient
    signal must preserve unresolved semantics and count as Investigation Gain only
    when it changes investigation state.
12. **Competition cannot terminate active.** Terminal status must be one of the
    explicit confirmed/rejected/blocked/exhausted/budget/failed states with a
    Python-governed reason.
13. **Continue only for a high-value action.** An action must be legal, within
    budget, plausibly change ranking/confirmation, and not repeat the same
    candidate + gap + scope action.
14. **No fallback at budget exhaustion.** Return `inconclusive` or
    `insufficient_evidence`.
15. **Confirmation is Python-owned.** Supported requires exposure, parameter,
    outcome, mechanism, temporal, scope, and alternatives checks with no critical
    unresolved contradiction.
16. **Impact scope is independent from root-cause reasoning.** Shared equipment
    alone never makes a Lot impacted. Confirmation publication requires matching
    exposure, window, operation/recipe, parameter, and compatible outcome.
17. **Do not publish impact lots for an inconclusive root cause.** Candidate impact
    rows may remain in audit detail; confirmed impact lots remain empty.
18. **Controlled behavior remains compatible.** Qwen-path hardening must not silently
    change controlled-path semantics without explicit tests and approval.
19. **Prompt bounds are architecture, not optimization.** Full Lane inventories and
    unbounded Evidence cannot be reintroduced into Planner/Qwen prompts.
20. **`strict_qwen_accepted` is governance telemetry only.** It does not mean the
    root cause is correct or supported.

## 5. Completed repair history and why each design exists

### Batch 24.x: iterative evidence-gated RCA foundation

- **24.0 authoritative RCA Finding cutover:** multiple reasoning rounds retain all
  historical Findings while Report/Memory/API/Frontend read an explicit authority
  pointer. This removed the `multiple RCA Reasoning findings` failure without
  destroying audit history.
- **24.1 Causal Evidence Matrix:** Python verifies whether cited Evidence actually
  supports the candidate's entities, time, scope, parameter, outcome, and
  mechanism. This was required because Evidence type alone did not prevent Qwen
  from citing a related but mismatched observation.
- **24.2 iterative comparison/investigation:** Evidence synthesis, deterministic
  gaps, targeted existing actions, Confirmation Gate, and Impact Gate established
  the "hypothesis -> feedback -> investigate -> re-evaluate" loop. Python still
  does not generate the root cause.
- **24.3 trace/evaluation surfaces:** exposed engineering-readable candidate,
  matrix, gaps, rejected reasons, and impact inclusion/exclusion without exposing
  every internal algorithm detail.

### Batch 25.x: adversarial multi-Lane investigation

- **25.0 causal Lane and Competition state:** represented alternative process paths
  explicitly instead of treating every shared-exposure row as an implicit candidate.
- **25.1 Lane-aware discovery/investigation:** actions are bound to a typed Lane so
  unrelated defect/MES/history actions cannot re-enter a targeted FDC gap.
- **25.2 adversarial challenge:** Qwen challenges the leading explanation and picks
  a Python-provided discriminator; it cannot invent arbitrary actions.
- **25.3 unavailable source and causal chain:** missing sources became explicit
  unresolved/blocked state, and Confirmation requires chain completeness.
- **25.4 independent impact scope and UI trace:** impact inclusion/exclusion became
  auditable and separate from hypothesis reasoning.
- **Lane-aware candidate dedup:** same equipment/operation text is not enough to
  merge candidates. Different Lane, recipe, parameter Evidence, mechanism, or
  discriminator preserves competition; true same-Lane/same-Evidence paraphrases
  remain duplicates with structured rejection audit.

### Batch 26.x: bounded lane-first and competition governance

- **26.0 Lane-first Evidence Synthesis:** replaced large flat Evidence prompts with
  bounded active Lane snapshots, preserving Evidence IDs and objective facts.
- **26.1 evidence-grounded mechanism bridge:** mechanism support must be tied to
  typed Knowledge, rule, current-Lot intermediate, or empirical discrimination;
  Qwen explanation alone is insufficient.
- **26.2 evidence-grounded ranking:** Python matrix strength constrains candidate
  ranking so Qwen preference cannot override objective conflicts.
- **26.3 Lane alternative resolution:** alternatives carry explicit retained,
  unresolved, eliminated, blocked, or non-discriminative state.
- **26.4 conditional third RCA round:** a third reasoning call is permitted only
  after new ranking-relevant typed Evidence, preventing both premature stop and
  unbounded retries.
- **26.5 Competition lifecycle:** explicit requirement/status/type/gap/terminal
  reason and Python-governed stop semantics prevent an active competition from
  being published as complete.

### Post-26.5 implementation in `c685bc0`

- **Explicit Candidate Scope Model:** separates claimed causal reach from comparison
  Evidence coverage. This prevents a candidate that compares two recipes from
  being mislabeled as multi-recipe solely because it cites both.
- **Candidate Evidence Closure:** detects when Qwen asserts a Lane/scope but omits
  already available exposure, parameter, outcome, temporal, or explicit physical
  intermediate citations. One bounded cumulative repair is allowed. Python never
  silently adds Evidence or rewrites the candidate.
- **Cumulative repair/citation regression protection:** repaired candidates must
  preserve earlier valid IDs. This prevents a second Qwen response from fixing one
  gap by dropping previous causal-chain endpoints.
- **Typed mechanism intermediate provenance:** only explicit source-declared
  physical/mechanism intermediate metadata can elevate an observation to mechanism
  support. Pattern names, defect labels, metric names, and Qwen prose cannot do it.
- **Explicit incident observation extraction:** a clearly quoted user inspection
  observation such as `polymer-residue` becomes occurrence-only typed Evidence;
  generic query language does not.
- **Evidence atomicity:** parameter/recipe rows, outcome entities, Lane binding, and
  temporal windows are projected without mixing controls or recipes.
- **Scope-grounded Confirmation and Impact:** broad claims require Lane coverage;
  impact projection can use a broad declared scope without relabeling Lot-specific
  Evidence.
- **Unexplained precursor filtering:** excursion windows, negative signals,
  controls, and missing data cannot be misclassified as unexplained physical
  precursors.
- **Knowledge no-match semantics:** an available source with no confirmed match is
  negative/context evidence rather than fabricated positive mechanism support.
- **Transport/serialization hardening:** candidate-generation transport failure and
  immutable mapping serialization are handled without crashing the workflow.
- **Lane-scoped specialist propagation:** recipe and Lane identity are forwarded
  into FDC tool parameters instead of relying on broad equipment defaults.

## 6. Regression-protected behavior that must not be reimplemented differently

Before changing any of the following, read the listed tests and demonstrate that
the existing invariant is actually insufficient. Do not add a parallel second
implementation.

- Authoritative multi-round Finding selection:
  - `tests/contract/test_supervisor_multi_finding_cutover_contracts.py`
  - `tests/contract/test_report_generator_contracts.py`
- Iterative candidate generation and authoritative prior Finding linkage:
  - `tests/contract/test_qwen_hypothesis_candidate_contracts.py`
- Lane-aware dedup and structured rejected-candidate audit:
  - `tests/contract/test_qwen_hypothesis_candidate_contracts.py`
- Competition lifecycle, bounded rounds, and stop governance:
  - `tests/contract/test_causal_competition_batch265_contracts.py`
  - `tests/contract/test_investigation_decision_patch3_contracts.py`
  - `tests/contract/test_patch4_mechanism_identity_contracts.py`
  - `tests/contract/test_mechanism_centered_competition_patch_contracts.py`
- Candidate Evidence Closure and one cumulative repair:
  - `tests/contract/test_candidate_evidence_closure_patch_contracts.py`
- Evidence atomicity, explicit incident observations, scope checks, and Impact Gate:
  - `tests/contract/test_evidence_atomicity_scope_confirmation_contracts.py`
  - `tests/integration/test_evidence_atomicity_scope_confirmation_replay.py`
- Available Knowledge source with no confirmed match:
  - `tests/contract/test_knowledge_typed_evidence_contracts.py`
- Planner/specialist workflow fallback and transport boundaries:
  - `tests/integration/test_llm_react_workflow.py`
- Frontend authoritative selectors and RCA trace:
  - `frontend/src/selectors.test.ts`
  - `frontend/src/components/RcaDiagnosisTrace.test.tsx`

Known limitation: the current tests do **not** cover every contradiction found in
the latest real `FORMAL_009` run. In particular, existing closure tests do not yet
prove that broad-scope per-Lane closure and Matrix matching use the exact same
Lane semantics after prompt trimming. That is a future design/audit item, not
permission to discard the existing Closure architecture.

## 7. Current validation status

Do not overstate validation. Durable command transcripts for the latest complete
working tree were not available after context compaction.

### Verified in this handoff turn

- `git diff --check`: passed before `c685bc0` was committed.
- No business code was modified during the handoff audit.
- No real Qwen call was made.
- No pytest, mypy, Ruff, compileall, TypeScript, or Vitest command was rerun in
  this handoff turn, as requested.

### Evidence available but not sufficient to claim a clean gate

- `.mypy_cache` and `.ruff_cache` contain entries updated on 2026-08-30, but cache
  timestamps do not prove a successful full run.
- `.pytest_cache/v/cache/lastfailed` is dated 2026-08-28 and contains stale failure
  entries, including an older Candidate Closure payload-size failure and unrelated
  legacy tests. These may predate later fixes, but until rerun they must be treated
  as **unknown current status**, not as passed and not automatically as current
  regressions.
- Full pytest after `c685bc0`: **not durably verified**.
- `core/backend` mypy after `c685bc0`: **not durably verified**.
- Ruff on all changed files after `c685bc0`: **not durably verified**.
- `compileall` after `c685bc0`: **not durably verified**.
- Frontend was not changed by `c685bc0`; frontend TypeScript/Vitest was not rerun.

The next session must begin with read-only inspection. Only after the user
explicitly authorizes validation should it run offline tests. Do not run a paid
Qwen Smoke merely to discover deterministic contract failures.

## 8. Latest real Qwen Smoke: `FORMAL_009`

Primary artifact:

```text
outputs/evidence_atomicity_scope_confirmation_qwen_smoke_FORMAL009_r2/
  run_results.json
  report.md
  states/FORMAL_009.json
```

Observed result:

```text
case_id                         : FORMAL_009
error                           :
job_status                      : completed
actual_orchestration_mode       : llm_react
fallback_reason                 :
hypothesis_status               : inconclusive
conclusion_status               : inconclusive
root_cause                      : inconclusive
action_count                    : 7
ranked_candidate_count          : 1
hypothesis_candidate_count      : 1
competition_requirement         : mechanism_required
competition_status              : exhausted
competition_type                : mechanism
competition_failure_reason      :
competition_gap_reason          : mechanism_alternative_not_generated
competition_terminal_reason     : no_high_value_action_remains
qwen_competition_accepted       : True
investigation_decision_accepted : True
impact_lot_count                : 0
evidence_count                  : 132
llm_call_count                  : 14
llm_call_cap_exceeded           : False
provider_failures               : {}
strict_qwen_accepted            : True
strict_qwen_rejection_reasons   : {}
```

Action sequence:

1. `inspect_defect_pattern`
2. `find_shared_exposure`
3. `validate_shared_defect_pattern` for the `RCP_800D7305` Lane
4. `inspect_fdc_spc` for the `RCP_800D7305` Lane
5. `inspect_fdc_spc` for the `RCP_C5BEFB77` Lane
6. `run_rca_reasoning`
7. `validate_historical_case` for
   `candidate_0.hypothesis_discrimination.mechanism_context`

The sole retained candidate was approximately:

```text
EQ_44E1AC_CH03 / operation 2500 plasma process instability
endpoint_overrun_delta +15.6%
oxygen_flow_response_delta +14.8%
source_power_repeatability +10.5%
-> incomplete polymer residue removal
-> elevated contact resistance / R39/P01 / HIGH_CONTACT_R
```

The numeric candidate values are present in typed FDC Evidence for the
`RCP_800D7305` Lane. Different action-summary averages used a broader aggregation
that included normal rows; this is a presentation/aggregation ambiguity, not
evidence that Qwen invented the candidate numbers.

## 9. Confirmed facts about the latest `FORMAL_009` state

### Execution and safety

- The job completed in `llm_react` without fallback, provider failure, exception,
  or LLM call-cap breach.
- Python correctly refused to publish `supported` while Scope and alternative
  competition remained incomplete.
- `strict_qwen_accepted=True` means the workflow governance checks passed. It does
  not validate the root cause against Ground Truth.
- The final `inconclusive` result is conservative and safe from over-confirmation.

### Typed evidence

- `RCP_800D7305` has typed parameter deviations for endpoint overrun, oxygen flow
  response, and source power repeatability.
- `RCP_C5BEFB77` also has three typed parameter-deviation Evidence records:

  ```text
  EV_FDC_ENDPOINT_OVERRUN_DELTA_LANE_E52308583741B438       +16.3%
  EV_FDC_OXYGEN_FLOW_RESPONSE_DELTA_LANE_E52308583741B438  +14.3%
  EV_FDC_SOURCE_POWER_REPEATABILITY_LANE_E52308583741B438  +10.1%
  ```

- `EV_INCIDENT_OBSERVATION_7BF0FBEC22D880BB` records user-reported
  `polymer-residue` occurrence only; its metadata explicitly says it is not a
  confirmed cause.
- Historical Knowledge retrieval returned:

  ```text
  source_available=True
  retrieval_status=no_match
  ```

  This means the source was available but contained no engineer-confirmed match.

### Candidate and Matrix

- Qwen attempted a second candidate, but its semantic relation was
  `shared_primary_with_modifier`: the same primary mechanism plus recipe-specific
  sensitivity. Python correctly did not count that as an independent primary
  mechanism competitor.
- The retained candidate claimed chamber-level `shared_effect` over both
  `RCP_800D7305` and `RCP_C5BEFB77`.
- Candidate Closure recorded `complete`, while the later Matrix recorded Scope
  `incomplete` because C5 parameter citations were missing from the candidate.
- Confirmation checks passed equipment, chamber, operation, parameter, outcome,
  temporal, and current mechanism classification, but failed Scope, causal-chain
  completeness, and no-equal-alternative.
- Impact Gate found two candidate impact Lots internally (`LOT_706579D262` and
  `LOT_E694A57AFB`) but published zero confirmed impact Lots because the
  authoritative conclusion was inconclusive. This publication behavior is
  expected.

## 10. Unresolved issues, classified

### Confirmed bugs

1. **Prompt-visible vs referenceable Evidence IDs diverge.**
   Candidate synthesis can retain Lane/group/mechanism-bridge IDs after prompt
   trimming removes their typed cards. In the latest run Qwen cited two C5 IDs it
   could see in synthesis, but Python rejected them as unknown because they were
   absent from `typed_evidence_register`.

2. **Candidate Closure and Matrix use inconsistent per-Lane matching.**
   `_candidate_evidence_closure_assessment()` checks a shared-effect broad scope
   per Lane, but `_evidence_matches_claimed_scope()` falls back to broad chamber
   identity after exact Lane mismatch. It can therefore count RCP_800D7305
   Evidence as C5 coverage. Matrix correctly reports C5 process coverage missing.

3. **Lower-priority high-information gaps can be starved.**
   Planner exposes only the minimum-priority gap set before Action Value is
   evaluated. When the priority-1 mechanism refresh becomes ineligible, the
   priority-2 Scope/citation gap is not considered. Therefore
   `no_high_value_action_remains` currently means "no value in the selected
   priority tier," not necessarily "no valuable unresolved action exists."

4. **Knowledge `no_match` can be rendered as source unavailable.**
   Lane resolution applies an older unavailable-source gain from the same Lane to
   the current mechanism challenge without exact gap/discriminator ownership.

5. **Stale candidate rejection reason.**
   Hypothesis Engine can build an initial Matrix, record a critical-conflict
   rejection reason, rebuild the Matrix with semantic scope, and replace only the
   Matrix fields. Final state can say Matrix `incomplete` with no critical conflict
   while retaining `critical claim/Evidence conflict` in rejection reasons.

6. **Stale warning after post-RCA Knowledge action.**
   The final state still includes `WARN_RCA_MISSING_FINDINGS: knowledge` even though
   a Knowledge Finding was added after the authoritative RCA Finding.

7. **Deferred Lane audit semantics leak into unresolved set.**
   Lane lifecycle counts are 2 active, 1 challenged, and 95 deferred, but the
   Competition trace lists many deferred/overflow Lanes as unresolved while the
   competition is `exhausted`.

### Architectural weaknesses

1. **Mechanism competition often stays in prose.** Qwen's challenge named distinct
   possibilities (chamber-wall passivation, residual fluorocarbon oxygen
   scavenging, and recipe-specific ash chemistry), but they were not promoted to
   formal independent candidates or empirical typed discriminators.
2. **Mechanism context action is mainly corroborative.** `validate_historical_case`
   can support plausibility of the leading mechanism but often cannot distinguish
   it from a competing current-Lot physical mechanism.
3. **Observed-intermediate support may join across scope too loosely.** The Matrix
   currently requires a relevant intermediate plus parameter/outcome shared lots,
   but does not clearly require the intermediate itself to share the same exact
   Lot/Lane or require Scope closure first.
4. **Semantic predictions are not fully checked against typed facts.** The retained
   profile stated that C5 had no product outcome, even though the source Lot is in
   C5 and has a typed polymer-residue incident observation. The expected validation
   policy for this contradiction is not yet explicit.
5. **Action summary aggregation can differ from candidate Evidence aggregation.**
   Both may be technically valid, but without an explicit aggregation label the
   audit is easy to misread.

### Expected conservative behavior

- One reasonable candidate plus incomplete adversarial elimination must remain
  inconclusive.
- A modifier of the same primary mechanism need not count as a second independent
  mechanism candidate.
- A Knowledge `no_match` need not trigger another paid RCA reasoning call if it
  cannot change ranking.
- Confirmed impact lots remain empty when the root-cause conclusion is inconclusive.
- Missing SPC control limits may be non-blocking when parameter-deviation Evidence
  itself is available and the Gate classifies the missing baseline as informational.

### Unverified suspicions

- Whether the retained candidate matches the private/external Ground Truth. The
  latest Smoke artifact alone cannot prove this, and blind audit must not read a
  private answer to decide implementation behavior.
- Whether changing mechanism `observed_intermediate` from supported to plausible
  is always correct. The cross-scope weakness is real, but the exact status rule
  needs a general design and replay set, not a FORMAL_009 reaction.
- Whether fixing the prompt Evidence contract would make Qwen produce a valid
  second primary mechanism. The rejected second output was also semantically a
  modifier, so Evidence ID repair alone does not guarantee mechanism competition.
- The regression impact on the full formal suite has not been measured after
  `c685bc0`.

## 11. Prohibited actions for the next session

- Do not add rules keyed to `FORMAL_009`, `EQ_44E1AC`, operation `2500`,
  `RCP_800D7305`, `RCP_C5BEFB77`, or any specific Lot/wafer.
- Do not loosen the Confirmation Gate merely to produce `supported`.
- Do not let Python generate or substitute a formal root-cause candidate on the
  Qwen path.
- Do not treat Qwen's causal explanation as typed Mechanism Evidence.
- Do not require a hard-coded `MechanismRule` as the only path to approved
  mechanism support.
- Do not make normal controls/exclusion evidence a universal hard gate.
- Do not infer claimed scope from the number of entities covered by Evidence.
- Do not treat missing data or Knowledge no-match as contradiction or candidate
  failure.
- Do not rotate through Lanes automatically whenever a source is missing.
- Do not reintroduce unbounded Lane/Evidence payloads to Planner or Qwen.
- Do not create new Agents for these fixes; use existing capabilities.
- Do not decide the authoritative Finding by list order.
- Do not silently attach Evidence IDs to a Qwen candidate.
- Do not conflate candidate citation closure with re-running a data source that
  has already produced the required typed Evidence.
- Do not automatically push, delete outputs, or stage external-score packets.
- Do not call real Qwen until offline contract/replay checks are clean and the
  user explicitly authorizes a paid Smoke.
- Do not interpret `strict_qwen_accepted=True` as root-cause correctness.

## 12. Recommended next-session read-only audit order

1. Read this file completely.
2. Run only read-only Git inspection:

   ```powershell
   git status -sb
   git log --oneline --decorate -12
   git show --stat c685bc0
   git diff origin/codex/batch-26-0-lane-first-evidence-synthesis..HEAD --stat
   ```

3. Confirm whether `c685bc0` and the handoff commit reached the remote. Do not
   push without user approval.
4. Read the latest State without modifying it:

   ```text
   outputs/evidence_atomicity_scope_confirmation_qwen_smoke_FORMAL009_r2/states/FORMAL_009.json
   ```

5. Reconstruct these exact propagation paths:

   ```text
   FDC C5 Evidence
     -> lane-first synthesis IDs
     -> prompt trimming
     -> typed_evidence_register
     -> Candidate 2 validation

   retained Candidate claimed scope
     -> Candidate Evidence Closure
     -> semantic profile
     -> Causal Evidence Matrix Scope
     -> deterministic Scope Gap

   all unresolved Gaps
     -> Planner priority filter
     -> Action option construction
     -> Action Value assessment
     -> terminal no_high_value_action
   ```

6. Read the code in this order:
   - `core/yield_rca_core/evidence_synthesis.py`
   - `core/yield_rca_core/hypothesis_candidate_generator.py`
   - `core/yield_rca_core/causal_evidence_matrix.py`
   - `core/yield_rca_core/causal_evidence_gap.py`
   - `core/yield_rca_core/next_action_planner.py`
   - `core/yield_rca_core/investigation_decision.py`
   - `core/yield_rca_core/causal_competition_progression.py`
   - `core/yield_rca_core/hypothesis_engine.py`
   - `core/yield_rca_core/causal_confirmation.py`
7. Read the new regression tests before proposing changes:
   - `tests/contract/test_candidate_evidence_closure_patch_contracts.py`
   - `tests/contract/test_evidence_atomicity_scope_confirmation_contracts.py`
   - `tests/integration/test_evidence_atomicity_scope_confirmation_replay.py`
8. Produce a read-only architecture proposal that addresses contracts, not the
   single case. Separate Prompt atomicity, Closure/Matrix semantic unification,
   Action Value fallback across priority tiers, and audit consistency.
9. Ask the user to approve the proposal before editing business code.
10. After approval, implement the smallest coherent offline patch and validate
    it with synthetic/contract replay first. Real Qwen Smoke is the final step.

## 13. Suggested offline acceptance design for any future patch

No implementation is authorized by this handoff, but a correct proposal should
eventually demonstrate all of the following without case-specific constants:

- Every Evidence ID visible anywhere in a Candidate Generator request is either
  present in `typed_evidence_register` or explicitly marked non-citable and absent
  from all citation-bearing fields.
- Prompt trimming atomically filters the register, Lane facts, global facts,
  bridge inputs, prior candidates, challenge context, and repair feedback.
- Candidate Closure and Matrix share one exact Lane-coverage predicate for
  per-Lane parameter/outcome roles.
- Broad identity matching remains available only where broad exposure/temporal
  grounding is semantically valid.
- Existing typed Evidence can trigger a reasoning/citation refresh without
  re-running the source tool.
- If the highest-priority action becomes ineligible, the next unresolved gap tier
  is assessed before terminal exhaustion.
- A same-primary-mechanism modifier remains distinguishable from an independent
  primary mechanism candidate.
- `source_available=True, retrieval_status=no_match` remains no-match, not
  unavailable.
- Rebuilt Matrix state synchronizes rejection reasons and warnings.
- Deferred/overflow Lanes do not appear as active unresolved alternatives.
- Controlled-path behavior remains unchanged.

## 14. New Session Bootstrap Prompt

Copy the following into a clean Codex session:

```text
We are continuing the semiconductor-yield-rca-v1-upgrade project from a prior
long-running session. Do not rely on chat history and do not modify code yet.

First read docs/CODEX_HANDOFF.md completely. Then perform only a read-only audit:

1. Inspect git status -sb, current branch, git log --oneline --decorate -12,
   commit c685bc0, and whether the local branch is ahead of origin.
2. Do not stage or inspect private Ground Truth from outputs/external-score packets.
3. Read the latest real Qwen State:
   outputs/evidence_atomicity_scope_confirmation_qwen_smoke_FORMAL009_r2/states/FORMAL_009.json
4. Trace these three contracts in code and State:
   a) prompt-visible Evidence IDs vs typed_evidence_register referenceable IDs;
   b) Candidate Evidence Closure per-Lane matching vs Causal Evidence Matrix Scope;
   c) all unresolved Evidence Gaps -> Planner priority -> Action Value ->
      no_high_value_action_remains.
5. Also verify the documented audit inconsistencies: Knowledge no_match rendered
   unavailable, stale Matrix rejection reason, stale missing-Knowledge warning,
   and deferred Lanes leaking into unresolved competition state.
6. Read the relevant existing regression tests before proposing anything.

Do not fix FORMAL_009, do not add device/recipe/case-specific rules, do not loosen
Confirmation Gate, do not let Python generate a root-cause candidate, and do not
call real Qwen. Distinguish confirmed bug, architectural weakness, expected
conservative behavior, and unverified suspicion.

Return an evidence-backed read-only audit and a minimal architecture-level repair
proposal. Wait for my explicit confirmation before editing any business code or
running paid Smoke Tests.
```
