# Codex Project Handoff

> Originally generated: 2026-08-30 (Asia/Hong_Kong)
>
> Current checkpoint updated: 2026-09-03 (Asia/Hong_Kong)
>
> Purpose: durable handoff for a new Codex Agent with no access to the prior chat.
>
> Current instruction: **Batches 1-5 and Patch 6.1 of the State Ownership refactor
> are committed locally, and an authorized real-Qwen smoke has verified all four
> ownership focus points. Do not start new patches, the Batch 6.3 deep audit, or
> another paid smoke until the next Agent completes the read-only bootstrap audit
> in the current checkpoint and the user approves the next work item.**---

## Current checkpoint — 2026-09-03

This section is authoritative for the next session. The 2026-09-02 historical
section below preserves the Batch 1-3 checkpoint verbatim; the 2026-08-30
sections at the end preserve the pre-refactor history.

### Git state

- Repository: `semiconductor-yield-rca-v1-upgrade`
- Branch: `codex/batch-26-0-lane-first-evidence-synthesis`
- Local HEAD: `8d9192f` (`Rename terminal stop telemetry with a compatibility
  alias`, Patch 6.1)
- Remote tracking HEAD: `b293d6b` (`Add project RCA governance skills`)
- Local branch is **ahead of origin by six commits**:

  ```text
  8d9192f Patch 6.1  telemetry naming migration                (Batch 6.1)
  b22a574 Converge React orchestration into one shared loop    (Batch 5)
  e42f57a Make Finalizer the sole RCA authority writer         (Batch 4)
  779c09e Separate planner proposals from finalizer outcomes   (Batch 3)
  4707019 Migrate RCA authority read paths                     (Batch 2)
  c31b23a Add authoritative RCA result foundation              (Batch 1)
  ```

- No push was requested or performed. A historical push attempt was once
  rejected by the execution environment's security reviewer and was not
  retried; push only from a trusted shell with explicit user approval.
- Only tracked working-tree modification: `docs/CODEX_HANDOFF.md`. It carries
  the previous session's 2026-09-02 checkpoint update (never committed) plus
  this 2026-09-03 checkpoint update. The next docs commit should include both.
- Untracked local artifacts: `outputs/` (including the new
  `batch6_1_qwen_smoke_strict_r1/` and `..._r2/` smoke directories),
  `docs/assets/`, `resume.md`, and `tmp/` (including the unused Run-B driver
  `tmp/batch6_smoke_next_action_fallback.py`, superseded by real run r2; it
  may be deleted locally). Never stage `outputs/` wholesale; some
  `outputs/long_task_3_external_score_packet*` artifacts may contain or
  derive from private Ground Truth.

### Architecture ownership status after Batches 1-5

The target ownership chain is now implemented end to end:

```text
Incident Understanding
  -> Evidence Collection and immutable typed Evidence
  -> Candidate Generation by Qwen (maximum two)
  -> Python Candidate validation and Evidence Closure
  -> Python Causal Evidence Matrix
  -> Investigation Loop (unified ReAct loop: Planner -> legal Action -> Tool)
  -> Competition
  -> Confirmation
  -> Investigation Finalizer (sole AuthoritativeRCAResult writer)
  -> Impact Gate decision -> ImpactPublicationResult (Finalizer writes)
  -> Report / API / Frontend read-only projection (authority preferred)
```

- Production now populates `authoritative_rca_result` and
  `impact_publication_result` on every governed terminal (Batch 4).
- One shared ReAct orchestration loop serves llm_react, controlled_react, and
  fallback; fallback swaps planner + execution profile inside the same State
  and loop (Batch 5).
- Runtime budget preflight/termination is runtime governance, never a planner
  proposal (`PlannerProposal.proposed_by` admits only `llm_qwen` and
  `python_controlled`).

### Completed Batch 4 — `e42f57a` Finalizer consolidation

Files: `core/yield_rca_core/investigation_finalizer.py`,
`core/yield_rca_core/causal_confirmation.py`,
`tests/contract/test_state_ownership_batch3_contracts.py` (fixture update),
new `tests/contract/test_state_ownership_batch4_contracts.py` and
`tests/integration/test_finalizer_authority_projection_integration.py`.

Design:

- `build_authoritative_rca_result` / `_attach_authority_results` live in the
  Finalizer; both finalize paths (`finalize_investigation`,
  `finalize_non_competition_llm_react_terminal`) attach the authority
  objects. `result_id` is deterministic (`RCA_RESULT_{finding_id}`);
  supported results require a confirmed candidate, permitted competition
  status, and evidence (enforced by the value object plus `models`
  cross-validation).
- `derive_impact_publication_result` lives in the Impact Gate module
  (`causal_confirmation.py`) and re-evaluates the stored gate assessment
  against the **terminal** conclusion using the identical publication rule.
  This closes the stale-snapshot channel: a terminal downgrade can no longer
  keep publishing confirmed impact Lots from a reasoning-time snapshot.
- Rejected alternative: a supervisor-side second Impact Gate step would have
  reintroduced a second terminal writer. Decision content stays in the
  Impact Gate module; the single write happens in the Finalizer funnel.
- Validation: full pytest 905 passed / 3 skipped / 49 subtests, frontend 54
  passed, repeated `change-impact-review`: approve with warning.

### Completed Batch 5 — `b22a574` Fallback convergence

Files: new `core/yield_rca_core/planner_interface.py`, rewritten
`core/yield_rca_core/supervisor.py`, new
`tests/contract/test_planner_interface_contracts.py` and
`tests/integration/test_react_loop_characterization.py`.

Design (user-approved 15-point revision):

- One shared `_react_loop` for llm_react, controlled_react, and fallback.
  Mode differences live in an explicit `ReactExecutionProfile` (planner
  preflight, dispatch strategy, budget preflight, stop-event naming,
  decision recording, conclusion capping, finalize routing). The loop never
  branches on the concrete native decision type; `native_decision` is
  audit-only.
- Planner selection is separated from action execution:
  `QwenPlannerAdapter` wraps `decide_with_review` (errors pass through
  untouched); `ControlledPlannerAdapter` wraps `InvestigationPolicy`. The
  `PlannerContext` ownership matrix is documented in the module docstring;
  the controlled adapter deliberately does not forward
  `critical_contradictions` (pre-refactor pin, contract-tested).
- Next-action fallback no longer jumps to a second loop
  (`_continue_controlled` deleted): the loop writes the existing fallback
  metadata and explicitly swaps both planner and profile in the same State.
  Intent-planning fallback reaches the same loop from the first iteration.
- Runtime termination (LLM-call budget preflight, RCA-round budget gate) is
  expressed as `ReactStopBundle(origin="runtime")`, never as a planner
  proposal; the existing `python_runtime` `PlannerDecision` audit records
  are rebuilt from public proposal fields so `planner_decisions` content is
  unchanged. Controlled proposals are NOT injected into `planner_decisions`
  (deliberate; deferred to a separate future batch).
- Budget continuity: state/action/tool/RCA counters and the shared LLM
  client survive the swap; the controlled continuation does not re-arm Qwen
  LLM preflight (matches old `_continue_controlled`).
- Procedure honored: characterization tests pinned the old behavior first
  (green on the pre-refactor implementation), then the shared loop was
  extracted and the same tests stayed green before `_continue_controlled`
  was removed.
- Validation: full pytest 920 passed / 3 skipped / 49 subtests, frontend 54
  passed, `change-impact-review`: approve with warning.

### Completed Batch 6 audit and Patch 6.1 — `8d9192f`

Batch 6 audit verdict (user-mandated wording): **no production dead code
meeting safe-deletion criteria was found**; the only implemented item (C1) is
a telemetry naming compatibility migration, not dead-code removal.

- C1: the runner's `planner_stop_reason` actually carried the Finalizer
  terminal `state.stop_reason`. Patch 6.1 adds canonical
  `terminal_stop_reason` and keeps `planner_stop_reason` as a deprecated
  alias with the same value (`_stop_reason_telemetry_fields` writer helper;
  both case-result sites use it; a source-level guard test forbids bare
  writes). Acceptance reads strictly canonical
  (`_terminal_stop_reason(..., allow_legacy_alias=False)`), so a missing
  canonical write can never be masked by the alias; only the explicit
  historical read path tolerates the alias. Stop provenance telemetry,
  Strict-Qwen semantics, governed-stop statistics, and provider-failure
  classification are untouched.
- C2 (`state.impact_lots`): retained. Active production writers/readers with
  independent operational semantics (fixed-path MES observed scope is not
  confirmed publication). No evidence that its divergence from publication
  lots constitutes a production regression; Batch 6 has no mandate to force
  convergence. Do not "clean it up".
- C3 (all rejected with evidence): `_legacy_preliminary_candidates`,
  `_finding_for_task`/`_finding_for_agent`,
  `InvestigationPolicy.critical_contradictions` (three production callers,
  including the `decision_evaluation` stop oracle), Batch 3 projection
  metadata (`decision_evaluation` reads it), `terminal_conclusion_status_source`
  (intentional audit provenance; tests verify the contract but are not the
  reason it exists), Report/API/UI legacy branches and `memory.py`
  (old-State compatibility), the competition progression legacy
  `failure_reason` shim (old-State readability), and the finding-details
  dual storage (Finalizer derivation input + compatibility).
- Validation: full pytest 928 passed / 3 skipped / 49 subtests, Ruff,
  compileall, `git diff --check`; `change-impact-review`: approve.

### Authorized real-Qwen smoke (2026-09-03) — verified evidence

Scope: minimal paid smoke on FORMAL_009, authorized by the user. Two strict
attempts (r1, r2); no synthetic fallback run was needed. Artifacts:
`outputs/batch6_1_qwen_smoke_strict_r1/` and `..._r2/` (run_results.json,
states/FORMAL_009.json; read-only audit fixtures, not staged).

Four ownership focus verdicts (all verified on real provider traffic):

```text
Batch 3 terminal provenance   r1: python_runtime stop, snapshot present,
                              terminal_stop_projected_by=None correctly
                              (no override occurred).
                              r2: Finalizer overrode the runtime stop ->
                              terminal_stop_projected_by=
                              python_investigation_finalizer present.
                              Both directions of the Batch 3 contract
                              exercised.
Batch 4 authority writer      Both runs: authoritative_rca_result and
                              impact_publication_result present with correct
                              invariants (inconclusive -> root_cause=None,
                              withheld/not_evaluated -> confirmed lots empty,
                              result_id/source ids/id links consistent).
Batch 5 same-loop swap        r2: real model invalidity at decision 7 ->
                              qwen_next_action_output_invalid -> explicit
                              planner+profile swap after 6 actions inside
                              the same State/loop; planner_decisions=6
                              preserved; run continued to completion with a
                              real Qwen candidate (count=1). Strictly better
                              evidence than the synthetic driver.
Batch 6.1 dual keys           Both runs: terminal_stop_reason ==
                              planner_stop_reason in run_results.json.
```

Failure layers (no repository-code defect found):

```text
r1: external provider transport — candidate generation is the largest
    prompt; two attempts timed out at the 60s default (latencies ~122s each)
    -> transport_error -> strict rejected
    ['hypothesis_candidate_fallback','provider_failure',
     'investigation_decision_invalid'].
    Resolution: environment only (YIELD_RCA_LLM_TIMEOUT_SECONDS=240,
    YIELD_RCA_LLM_MAX_RETRIES=2). r2 confirmed candidate generation then
    succeeded.
r2: recurring model-output flakiness class, documented pre-refactor
    (lane_effect_contract_qwen_smoke_FORMAL009_r1 shows the same
    qwen_next_action_output_invalid fallback). Evidence it is not a Batch 5
    regression: next_action_planner.py untouched since e826edd (pre-Batch-1);
    planner prompt inputs field-identical (adapter contract tests); r1/r2 ran
    the same commit and diverged at iteration 5-6, proving provider-side
    nondeterminism at temperature 0. The system behaved as designed: strict
    telemetry honestly reports rejection
    (['orchestration_fallback','orchestration_fallback_reason_present']).
```

Not run / not achieved:

- A clean `strict_qwen_accepted=True` pass was not obtained (twice blocked by
  provider-side factors; the pre-refactor baseline
  `provider_failure_patch_qwen_smoke_FORMAL009_r2` proves it can pass).
  Obtain one opportunistically at the next authorized paid validation.
- The synthetic Run-B driver was superseded by r2 and is not needed.

Reproduction commands (require explicit user authorization, paid):

```text
env: YIELD_RCA_AGENT_MODE=llm, DASHSCOPE_API_KEY=<user key>,
     YIELD_RCA_LLM_TIMEOUT_SECONDS=240, YIELD_RCA_LLM_MAX_RETRIES=2
run: python scripts/run_formal_blind_rca.py --orchestration-mode llm_react
     --case-id FORMAL_009 --confirm-paid-qwen --overwrite
     --output-dir outputs/<new_dir>
```

### Current weaknesses and remaining work

1. No clean strict-Qwen pass on record since Batch 1 (unverified, not a code
   defect; obtain at next authorized smoke).
2. Controlled/fallback terminal decisions are not injected into
   `planner_decisions`, so RunEvaluation stays empty for those terminals —
   deliberate Batch 5 boundary; a future independent cleanup/migration batch
   must design it (it affects decision_evaluation, API, UI, formal
   telemetry, Strict-Qwen).
3. The two dispatchers (`_dispatch_controlled` registry agents vs
   `_dispatch_llm_react` Specialist V2 Lane-scoped context) remain separate
   by design (fallback compatibility); unification is out of scope until a
   separately approved batch.
4. `planner_stop_reason` alias retirement requires an explicit migration plan
   for historical run_results.json consumers before any removal (Batch 6
   discipline; not scheduled).
5. Batch 6.3 deep audit (separate future batch, requires user approval):
   per-function caller/writer/reader/serialization search across
   `next_action_planner.py` (~2800 lines) and `investigation_models.py`.
   Do not presume anything is deletable.
6. Compatibility readers (Report/API/UI legacy branches) remain by design
   until stored-State coverage and stability justify a future removal batch.

### Things the next session must NOT do

- Do not start Batch 6.3, a new patch, or another paid smoke without a
  separate read-only audit and explicit user approval.
- Do not remove the `planner_stop_reason` alias or any C3-listed code.
- Do not "align" `state.impact_lots` with `ImpactPublicationResult` (C2
  retention ruling).
- Do not unify the dispatchers or inject controlled decisions into
  `planner_decisions` without an approved design.
- Do not weaken Planner validation, Confirmation, Competition, or Impact
  gates to obtain a strict pass; a strict rejection is honest telemetry
  (INV-008), not a bug.
- Do not stage `outputs/`, `tmp/`, `docs/assets/`, or `resume.md`; do not
  push without explicit user approval; do not read private Ground Truth or
  external-score packets.
- Do not run real Qwen without `--confirm-paid-qwen` and explicit user
  authorization.

### New Session Bootstrap Prompt

```text
We are continuing the YieldMind RCA State Ownership refactor. Do not rely on
chat history and do not modify business code yet.

First read docs/CODEX_HANDOFF.md completely (the 2026-09-03 checkpoint is
authoritative; later sections are preserved history), then read
.agents/skills/_shared/architecture-invariants.md.

Perform a read-only bootstrap audit:
1. git status -sb, branch, HEAD (8d9192f), and the six commits ahead of
   origin (Batches 1-5 plus Patch 6.1). Do not push.
2. Verify the Batch 4/5/6.1 diffs and their contract/integration tests,
   including the characterization tests in
   tests/integration/test_react_loop_characterization.py.
3. Read the smoke artifacts outputs/batch6_1_qwen_smoke_strict_r1 and _r2
   (run_results.json + states/FORMAL_009.json) and re-verify the four focus
   verdicts recorded in the checkpoint.
4. Confirm the Batch 6 audit rulings (C1 implemented, C2 retained, C3 all
   rejected) still match the current tree.

Then stop and ask the user which item to start: the Batch 6.3 deep audit
(separate per-function reachability audit of next_action_planner.py and
investigation_models.py), a clean strict-Qwen smoke at the next authorized
paid validation, the controlled-decision planner_decisions migration design,
or anything else. Do not begin any of them without explicit approval, and do
not run real Qwen without --confirm-paid-qwen plus explicit authorization.
```

'''


## Historical checkpoint — 2026-09-02 (Batches 1-3)

Superseded by the 2026-09-03 checkpoint above. Preserved verbatim for Batch 1-3
design history: the State Ownership Audit baseline, the recommended ownership
map, the Batch 1-3 designs and validation numbers, and the smoke artifacts known
at that time. Its Git status, unresolved-work list, and bootstrap prompt are
historical rather than current.

### 0.1 Current Git state

- Repository: `semiconductor-yield-rca-v1-upgrade`
- Branch: `codex/batch-26-0-lane-first-evidence-synthesis`
- Local HEAD: `779c09e84bac33d442c082ee39ab1e42d332c0c4`
  (`Separate planner proposals from finalizer outcomes`)
- Remote tracking HEAD:
  `b293d6bdfb4fc2e5cf53af194fd9c7e7949259b8`
  (`Add project RCA governance skills`)
- Local branch is **ahead of origin by three commits**:

  ```text
  779c09e Separate planner proposals from finalizer outcomes     (Batch 3)
  4707019 Migrate RCA authority read paths                       (Batch 2)
  c31b23a Add authoritative RCA result foundation                (Batch 1)
  ```

- Batch 3 was committed only after its offline regression and a repeated
  `change-impact-review` completed with `approve with warning`.
- No push was requested or performed in this checkpoint.
- Immediately after the Batch 3 commit there were no tracked business-code or
  test changes. Updating this handoff makes `docs/CODEX_HANDOFF.md` the only
  intended tracked working-tree modification.
- Existing untracked `outputs/`, `docs/assets/`, `resume.md`, and `tmp/` paths are
  local artifacts. They were not staged, modified, deleted, or committed.
- Some `outputs/long_task_3_external_score_packet*` artifacts may contain or
  derive from private Ground Truth. Never inspect them during blind RCA work and
  never stage `outputs/` wholesale.

### 0.2 Project architecture overview and main data flow

The system is an evidence-grounded semiconductor-yield RCA workflow:

```text
Incident Understanding
  -> Evidence Collection and immutable typed Evidence
  -> Candidate Generation by Qwen (maximum two)
  -> Python Candidate validation and Evidence Closure
  -> Python Causal Evidence Matrix
  -> Investigation Loop (Planner -> legal Action -> Tool -> Evidence)
  -> Competition
  -> Confirmation
  -> Investigation Finalizer
  -> AuthoritativeRCAResult
  -> Impact Gate / ImpactPublicationResult
  -> Report / API / Frontend read-only projection
```

The last four arrows describe the **target ownership chain**. The codebase is in
the middle of migration: Batch 1 introduced the authority value objects, Batch 2
migrated readers, and Batch 3 separated Planner proposals from Finalizer terminal
projections. Production code does not yet populate `AuthoritativeRCAResult` or
`ImpactPublicationResult`; that is deliberately deferred to Batch 4.

### 0.3 Python / Qwen responsibility boundary

Qwen may:

- propose evidence-bounded hypotheses and a maximum of two materially distinct
  formal candidates;
- explain causal mechanisms and challenge/compare candidates;
- select among Python-provided legal investigation options.

Python exclusively owns:

- typed Evidence truth, provenance, entity/Lot/Wafer/Recipe/Lane binding;
- Evidence citation validation and deterministic Evidence Closure;
- Matrix, Competition progression, Confirmation and Impact Gate decisions;
- legal actions, budgets, Action Value and termination governance;
- authoritative RCA and publication state.

Python must not invent, replace, or silently complete a formal Qwen candidate.
Qwen output must not directly declare a supported conclusion or publish impact
scope.

### 0.4 Architecture invariants

The source of truth is:

```text
.agents/skills/_shared/architecture-invariants.md
```

The next Agent must read that file completely. Especially relevant to the
ownership refactor are:

- INV-001/002: candidate authorship and Qwen/Python boundary;
- INV-006/007: separate scope dimensions and independently grounded impact;
- INV-008: Strict-Qwen is execution telemetry, not conclusion correctness;
- INV-009: Confirmation cannot bypass incomplete Competition;
- INV-011/015: Evidence traceability and JSON-safe State boundaries;
- INV-016: authority must use explicit pointers, never list position;
- INV-018: Planner STOP is governed by high-value-action availability and does
  not confirm a root cause;
- INV-019: controlled-path compatibility.

### 0.5 State Ownership Audit baseline

The read-only State Ownership Audit found that terminal engineering semantics
were spread across legacy Finding details, top-level `RCAState` fields,
Competition, Confirmation, Planner decisions, Supervisor projections, Report and
API fallbacks. The highest-risk duplicated concepts were:

```text
root_cause
hypothesis_status
conclusion_status
conclusion_level
goal_status
stop_reason
competition_status / terminal_reason
publication_status
impact_lots / confirmed_impact_lots
```

The target semantic separation is:

```text
competition_status   = whether candidates were sufficiently distinguished
confirmation_status  = whether one candidate passed causal confirmation
conclusion_status    = authoritative engineering RCA conclusion
goal_status          = workflow lifecycle only
stop_reason           = why investigation stopped
terminal_reason       = authoritative terminal rationale
publication_status    = whether root cause / impact scope may be published
```

`goal_status`, `stop_reason`, Planner STOP and orchestration fallback must never
be treated as proof that a root cause is supported.

### 0.6 Recommended ownership map

| Component | Owned state | Must not own |
| --- | --- | --- |
| Candidate Generator | candidates and semantic profiles | Evidence truth or final conclusion |
| Evidence Matrix | claim/Evidence support status | candidate authorship or publication |
| Planner | next action and STOP proposal | terminal RCA conclusion |
| Competition | comparison/progression result | root-cause publication |
| Confirmation | candidate confirmation result | workflow lifecycle |
| Finalizer | sole authoritative RCA conclusion | candidate invention |
| Impact Gate | confirmed publishable impact scope | root-cause generation |
| Workflow | requested orchestration mode | conclusion reinterpretation |
| Supervisor | execution/orchestration state | independent engineering conclusion |
| Report/API/UI | read-only projection | conclusion derivation |

### 0.7 Completed Batch 1 — Authority foundation

Commit: `c31b23a Add authoritative RCA result foundation`

Files:

- `core/yield_rca_core/authoritative_result.py`
- `core/yield_rca_core/models.py`
- `core/yield_rca_core/__init__.py`
- `tests/contract/test_authoritative_rca_result_contracts.py`

Design:

- Added immutable, JSON-safe `AuthoritativeRCAResult` containing only terminal
  engineering authority: result/source IDs, conclusion, publishable root cause,
  confirmation/competition status, terminal reason and Evidence references.
- Added separate immutable `ImpactPublicationResult` containing publication
  status, confirmed impact Lots and Evidence references.
- Added optional fields to `RCAState` with typed `to_dict`/`from_dict` round-trip.
- Enforced invariants such as: non-supported results cannot publish a root cause;
  supported results require a confirmed candidate, permitted Competition status
  and Evidence; non-confirmed impact publication cannot contain confirmed Lots.
- Kept both fields optional so old State remains loadable.
- Intentionally did **not** add a production writer or change RCA behavior.

Why: establish the authority boundary before migrating readers or removing
legacy writers. This prevents a flag-day migration and protects backward
compatibility.

### 0.8 Completed Batch 2 — Read path migration

Commit: `4707019 Migrate RCA authority read paths`

Files:

- `core/yield_rca_core/report_generator.py`
- `backend/yield_rca_api/app.py`
- `backend/yield_rca_api/schemas.py`
- `frontend/src/selectors.ts`
- `frontend/src/types.ts`
- associated Report/API/selector contract tests

Design:

- Report, API projection and frontend selectors prefer
  `authoritative_rca_result` and the matching `impact_publication_result` when
  present.
- Authority linkage uses `result_id`, `source_finding_id` and
  `rca_result_id`; readers do not infer authority from list position.
- Legacy States without the new objects continue through the prior explicit
  Finding/diagnosis compatibility path.
- When an authority object exists, readers do not silently fall back to a
  conflicting legacy result.
- Batch 2 changed readers only; it did not create a production writer.

Why: make downstream surfaces ready before Finalizer becomes the sole writer,
while preserving old serialized State and controlled-path behavior.

### 0.9 Completed Batch 3 — Planner proposal / Finalizer outcome separation

Commit: `779c09e Separate planner proposals from finalizer outcomes`

Core files:

- `core/yield_rca_core/investigation_finalizer.py`
- `core/yield_rca_core/supervisor.py`
- `core/yield_rca_core/decision_evaluation.py`
- `scripts/run_formal_blind_rca.py`
- `frontend/src/components/AgentDecisionTrace.tsx`
- `tests/contract/test_state_ownership_batch3_contracts.py`
- related unit, contract and frontend tests

Confirmed pre-Patch regression:

- after separating Planner STOP proposals from terminal ownership, old
  consumers still assumed the proposal and terminal projection were identical;
- `RunEvaluation` rejected a legitimate Finalizer override;
- Formal telemetry conflated Qwen proposal with Python-governed outcome;
- UI omitted Finalizer outcome;
- `planner_stop_proposed_by` could be overwritten.

Design:

- Preserve the original `PlannerDecision`; Python does not replace it with a
  fabricated STOP decision.
- Preserve `planner_stop_proposed_by` as the real proposal source.
- Record Finalizer projection separately through JSON-safe metadata:

  ```text
  terminal_stop_projection_applied
  terminal_stop_projection_trace
  terminal_stop_projected_by=python_investigation_finalizer
  terminal_state_owner
  superseded_terminal_planner_decision
  ```

- `decision_evaluation` accepts a terminal projection only when the metadata is
  complete and the preserved Planner decision snapshot matches exactly. Missing,
  forged or inconsistent audit metadata remains invalid.
- Formal telemetry now separates Qwen STOP proposal, governed Python STOP and
  Finalizer projection.
- UI now displays Planner proposal and Python Finalizer outcome separately.

Why: Planner owns a proposal, not the terminal engineering conclusion. Keeping
both records avoids fabricating Qwen behavior while allowing deterministic
Finalizer governance to remain auditable.

### 0.10 Batch 3 validation and impact review

Pre-Patch reproduction:

```text
Python targeted: 3 failed, 35 passed
Frontend target: 1 failed, 12 passed
```

Post-Patch verification completed before commit:

```text
Targeted Python                 39 passed
Neighboring Python             73 passed, 6 subtests passed
Full pytest                    896 passed, 3 skipped, 49 subtests passed
Target frontend component      13 passed
Neighboring frontend           38 passed
Full pnpm check                54 passed; TypeScript passed
Ruff                           passed
compileall                     passed
git diff --check               passed
```

Only existing Starlette deprecation and pytest-cache permission warnings were
observed. The repeated `change-impact-review` recommendation was:

```text
approve with warning
```

The review found no confirmed regression in Evidence semantics, entity/Lane
binding, Candidate contract, Closure, Scope, Competition, Confirmation, Impact
Gate, Prompt contract, controlled-path compatibility or JSON serialization.

### 0.11 Current architectural weaknesses and unresolved refactor work

These are architecture weaknesses or planned migration work, not permission to
change behavior without a new audit:

1. **No production authority-object writer yet.** A repository search at this
   checkpoint finds production assignment only during State deserialization;
   construction otherwise exists in tests. Real workflow results therefore still
   use legacy authoritative Finding/state fields. This is the central Batch 4
   task, not an accidental Batch 1-3 omission.
2. **Transitional terminal writes remain.** Supervisor can still project Planner
   proposal fields before Finalizer overwrites governed terminal fields. Batch 3
   made this auditable but did not remove every legacy writer.
3. **Finalizer still updates legacy Finding details and top-level terminal
   fields.** It is not yet the sole producer of the new authority object.
4. **Compatibility readers remain.** Report/API/UI still support old State. These
   branches are required compatibility logic until migration tests and stored
   State coverage permit removal.
5. **Formal naming remains partially legacy.** `planner_stop_reason` can carry the
   final `state.stop_reason`; `terminal_stop_projected_by` disambiguates it, but a
   later legacy-cleanup batch should split or rename the field.
6. **Fallback convergence is not implemented.** `llm_react` fallback still has
   broader controlled-path behavior. The target is to replace only Planner/action
   selection while reusing Matrix, Competition, Confirmation, Finalizer and
   Impact governance.
7. **Legacy cleanup is not authorized.** Duplicate fields and compatibility
   projections must remain until Batch 4/5 are stable and old State tests pass.

### 0.12 Classification at handoff

Confirmed bugs:

- The Batch 3 proposal/projection consumer mismatch is fixed and regression
  protected.
- No new confirmed blocking regression was found in the repeated impact review.

Architectural weaknesses:

- production does not yet populate the new authority objects;
- terminal legacy fields remain multi-writer during migration;
- fallback has not converged to a shared governance chain;
- Report/API compatibility paths still exist by design.

Expected conservative behavior:

- Planner STOP does not mean root cause confirmed;
- one plausible candidate with incomplete Competition remains inconclusive;
- an inconclusive RCA publishes no confirmed impact Lots;
- provider failure or fallback cannot count as a successful Strict-Qwen run.

Unverified suspicions / not-run validation:

- no real Qwen or paid Smoke was run after Batch 1, Batch 2 or Batch 3;
- the provider path has not validated the new Finalizer projection telemetry;
- fallback convergence behavior after future Batch 5 remains unknown;
- no claim is made about private Ground Truth correctness.

### 0.13 Latest real Smoke artifacts

No real Qwen was called during the ownership refactor.

Latest real execution attempt by artifact timestamp:

```text
outputs/lane_effect_contract_qwen_smoke_FORMAL009_r1/
  run_results.json
  states/FORMAL_009.json
```

Observed result (pre-Batch 1-3):

```text
job_status                   completed
actual_orchestration_mode    controlled_react
fallback_reason              qwen_next_action_output_invalid
provider_failure             true
competition_status           failed
competition_failure_reason   candidate_provider_failed
conclusion_status            insufficient_evidence
hypothesis_status            inconclusive
root_cause                   inconclusive
strict_qwen_accepted         false
llm_call_count               11
impact_lot_count             7 (legacy state scope; not confirmed publication)
```

The corresponding pre-refactor State has `goal_status=satisfied` and
`stop_reason=goal_satisfied` while its authoritative Finding reports
`conclusion_status=insufficient_evidence`. It contains neither
`authoritative_rca_result` nor `impact_publication_result`. This artifact is
useful as a reproduction/audit fixture, but it is **not** proof of current HEAD
behavior.

Latest earlier Strict-Qwen accepted baseline:

```text
outputs/provider_failure_patch_qwen_smoke_FORMAL009_r2/
  run_results.json
  states/FORMAL_009.json
```

```text
job_status                         completed
actual_orchestration_mode          llm_react
strict_qwen_accepted               true
provider_failure                   false
hypothesis_status                  inconclusive
conclusion_status                  inconclusive
ranked_candidate_count             1
competition_requirement            mechanism_required
competition_status                 exhausted
competition_gap_reason             mechanism_alternative_not_generated
competition_terminal_reason        no_high_value_action_remains
investigation_decision_accepted    true
qwen_competition_accepted          true
impact_lot_count                    0
evidence_count                      132
llm_call_count                      15
```

Strict-Qwen acceptance proves execution/output governance only; it does not prove
the RCA candidate is correct. Do not read external-score packets to answer that
question.

### 0.14 Compatibility logic that must be preserved

- old State without authority objects must deserialize and remain reportable;
- authority pointers must never be inferred from list order;
- if a new authority object exists, Report/API/UI must project it consistently;
- controlled/fixed behavior must remain compatible unless an approved Batch
  explicitly changes it;
- Planner decision history must remain unchanged by Finalizer projection;
- incomplete/inconclusive RCA must not publish confirmed impact Lots;
- State and projection metadata must remain JSON serializable;
- provider failure must not be disguised as a successful Qwen run;
- Strict-Qwen semantics must remain telemetry-only.

### 0.15 Important files for the next Agent

Ownership foundation and State:

- `core/yield_rca_core/authoritative_result.py`
- `core/yield_rca_core/models.py`
- `tests/contract/test_authoritative_rca_result_contracts.py`

Current terminal writers and governance:

- `core/yield_rca_core/investigation_finalizer.py`
- `core/yield_rca_core/supervisor.py`
- `core/yield_rca_core/decision_evaluation.py`
- `core/yield_rca_core/causal_competition_progression.py`
- `core/yield_rca_core/causal_confirmation.py`
- `core/yield_rca_core/rca_reasoning_agent.py`

Readers/projections:

- `core/yield_rca_core/report_generator.py`
- `backend/yield_rca_api/app.py`
- `backend/yield_rca_api/schemas.py`
- `frontend/src/selectors.ts`
- `frontend/src/components/AgentDecisionTrace.tsx`
- `scripts/run_formal_blind_rca.py`

Governance Skills:

- `.agents/skills/_shared/architecture-invariants.md`
- `.agents/skills/rca-semantic-auditor/SKILL.md`
- `.agents/skills/regression-safe-patch/SKILL.md`
- `.agents/skills/change-impact-review/SKILL.md`
- `.agents/skills/handoff-checkpoint/SKILL.md`

### 0.16 Remaining Batch plan

#### Batch 4 — Finalizer consolidation

- make Finalizer the sole production creator of `AuthoritativeRCAResult`;
- create/link `ImpactPublicationResult` only from the independent Impact Gate;
- populate authority objects from existing confirmed governance results without
  changing Confirmation or Impact criteria;
- keep legacy fields as compatibility projections;
- add contract and integration tests proving all terminal surfaces agree.

#### Batch 5 — Fallback convergence

- define a shared Planner interface;
- make LLM and controlled planners interchangeable action-selection
  implementations;
- reuse the same Evidence, Matrix, Competition, Confirmation, Finalizer and
  Impact chain;
- preserve native `controlled_react` compatibility and provider-failure
  telemetry.

#### Batch 6 — Legacy cleanup

- only after stored-State, controlled-path, Report/API/frontend and Smoke
  validation is stable;
- remove duplicate writers, obsolete projection branches and dead fields in
  small reviewed steps;
- do not delete compatibility fields merely because the new objects exist.

### 0.17 Regression safety plan

Every remaining Batch must protect:

- Python never generates a formal Qwen candidate;
- Evidence IDs cannot appear without typed provenance;
- Confirmation and Impact gates are not weakened;
- observed/candidate impact Lots remain separate from confirmed publication;
- Planner STOP does not imply confirmation;
- provider failure does not count as successful Qwen execution;
- old State remains deserializable;
- Report, API and UI agree with the authority object;
- no new API serialization 500;
- Formal data and private Ground Truth remain untouched.

Required test layers:

- contract tests for single-module ownership and prohibited writes;
- integration tests spanning Finalizer -> authority -> impact -> Report/API/UI;
- old-State round-trip and controlled/fallback compatibility tests;
- full offline regression before any paid Formal Smoke;
- FORMAL_009 must never be the only acceptance case.

### 0.18 Things the next session must NOT do

- Do not start editing before reading this handoff, the shared invariants and the
  Batch 1-3 diffs/tests.
- Do not add case/device/Recipe/Lot-specific rules.
- Do not loosen Confirmation or Impact gates to produce `supported`.
- Do not let Python invent or silently repair Qwen candidate content.
- Do not infer authority from the last Finding or Hypothesis.
- Do not make Report/API/Supervisor re-derive the conclusion.
- Do not treat Planner STOP, workflow completion or `goal_status` as RCA proof.
- Do not remove legacy compatibility paths in Batch 4.
- Do not modify Formal data or read private Ground Truth/external-score packets.
- Do not stage `outputs/` or unrelated local artifacts.
- Do not run real Qwen until deterministic tests and impact review are clean and
  the user explicitly authorizes paid Smoke.
- Do not automatically continue into Batch 5 after completing Batch 4.

### 0.19 New Session Bootstrap Prompt

Copy this prompt into the replacement Agent session:

```text
We are continuing the YieldMind RCA State Ownership refactor. Do not rely on chat
history and do not modify business code yet.

First read docs/CODEX_HANDOFF.md completely, then read
.agents/skills/_shared/architecture-invariants.md. Treat section 0 of the handoff
as the current checkpoint and sections 1-14 as preserved historical context.

Perform a read-only bootstrap audit:
1. Inspect git status -sb, branch, HEAD, origin tracking status and
   git log --oneline --decorate -12.
2. Verify commits c31b23a (Batch 1), 4707019 (Batch 2) and 779c09e (Batch 3).
3. Read their relevant implementation and regression tests before proposing any
   change.
4. Trace current writers/readers of authoritative_rca_result,
   impact_publication_result, conclusion_status, root_cause, conclusion_level,
   goal_status, stop_reason and publication_status.
5. Confirm the documented migration boundary: readers prefer the new authority
   objects, but production does not yet create them.
6. Trace Supervisor -> Competition/Confirmation -> Investigation Finalizer ->
   Report/API/UI, including controlled/fallback paths.
7. Do not inspect outputs/external-score packets or private Ground Truth.
8. Do not run real Qwen or paid Smoke.

Return an evidence-backed Batch 4 design for Finalizer consolidation. It must make
Finalizer the sole AuthoritativeRCAResult writer, keep ImpactPublicationResult
independent, preserve old State and controlled-path compatibility, and avoid any
change to Candidate authorship, Confirmation criteria or Impact criteria.

Wait for explicit approval before invoking regression-safe-patch or editing
business code. Do not automatically continue into Batch 5.
```

---

## Historical checkpoint — 2026-08-30

The following sections preserve the previous checkpoint verbatim for design and
audit history. Their operational status and bootstrap prompt are superseded by
section 0 above.

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
