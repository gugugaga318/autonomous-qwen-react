# Semiconductor Yield RCA Architecture Invariants

本文件是项目级 Codex Skill 共享的架构不变量唯一来源。四个治理 Skill 必须在执行前完整阅读本文件，不得在各自 `SKILL.md` 中复制并维护另一份不变量清单。

## 使用规则

- 把不变量当作设计和审查约束，不把它们当作针对单个 Formal Case 的答案模板。
- 如果代码、测试和文档对某条规则存在冲突，先报告冲突并进行只读审计；不得自行选择最容易让测试通过的解释。
- 新增或修改不变量前，必须同时给出代码、测试或已批准架构文档的依据。
- 未验证的推测只能记录为 `unverified suspicion`，不得升级为 Patch 前提或 Confirmation Gate 规则。
- 权威架构背景见 [CODEX_HANDOFF.md](../../../docs/CODEX_HANDOFF.md)，尤其是 Architecture Invariants、Prohibited Actions 和最新审计状态。

## 核心不变量

### INV-001 — Formal Candidate authorship

Python 负责 Evidence 校验和确定性规则；在 Qwen 路径上，Python 不得代替 Qwen 发明、补充或替换正式 RCA Candidate。

### INV-002 — Qwen/Python responsibility boundary

Qwen 可以提出 hypothesis、semantic scope 和 causal explanation；Evidence 真实性、适用性、确定性校验、最终 Gate 和发布状态仍由 Python 控制。

### INV-003 — Positive Evidence atomicity

Positive Evidence 只能绑定真实 positive/abnormal rows 所代表的 Lot、Wafer 和其他实体。仅因为 normal control 位于查询范围内，不得让它继承 positive Evidence。

### INV-004 — Lane identity preservation

Lane-specific Evidence 必须保留 Recipe、Equipment、Chamber、Operation 和显式 Lane identity；不得把另一 Recipe Lane 的 Evidence 静默当作当前 Lane Evidence。

### INV-005 — No-match is not unavailable

成功访问数据源但没有匹配结果是有效 negative result。`source_available=true, retrieval_status=no_match` 不等于 source unavailable、`DATA_MISSING`、contradiction 或执行失败。

### INV-006 — Scope dimensions remain separate

Scope identity、process coverage、outcome coverage、effect coverage 和 shared-effect support 是不同概念，不得压缩成一个布尔值，也不得用 Evidence 覆盖范围反推 claimed physical scope。

### INV-007 — Impact Lot grounding

Impact Lot inclusion 必须由有效 scope、真实 exposure、temporal alignment、所需 parameter/process Evidence、Recipe/Operation compatibility 和 compatible outcome Evidence共同支撑。共享设备本身不足以确认 Impact Lot。

### INV-008 — Strict Qwen telemetry boundary

`strict_qwen_accepted=True` 只证明真实 Qwen 执行和输出合同满足治理要求，不证明根因正确、Competition 完整、Confirmation Gate 通过或 Impact Lot 正确。

### INV-009 — Confirmation cannot bypass competition

Competition 未完成时，不得仅为了得到 `supported` 而削弱 Confirmation Gate、把同一主机制的改写当作独立候选，或忽略未解决的关键替代解释。

### INV-010 — Candidate repair citation preservation

Candidate repair 不得静默删除此前有效的 Evidence citations。有效 citation 被移除时，必须记录为 citation regression，除非有明确、可审计的语义理由。Python 不得偷偷回填 Qwen 未引用的 Evidence。

### INV-011 — Evidence traceability

Evidence ID、typed payload 和 provenance 必须贯穿 synthesis、prompt projection、candidate generation、closure、Matrix、Confirmation、Impact projection 和最终 State serialization，并保持可追溯。

### INV-012 — Missing citation differs from missing source

已有 typed Evidence 但 Candidate 漏引，与数据源真正不可用是不同状态。两者不得自动产生相同的 Action Value、Lane resolution 或 terminal semantics。

### INV-013 — Broad shared-effect semantics

Broad equipment/chamber shared-effect claim 必须有显式 semantic scope。当前可见 Recipe Lane 的枚举既不自动扩大 claimed physical scope，也不自动证明所有 Lane 的 effect coverage。

### INV-014 — Incident/intermediate provenance boundary

用户提供的 incident 或 physical-intermediate observation 只有在 provenance 完整保留时才能支持“观察到中间现象”。它不自动证明完整 causal mechanism，也不得被改写成用户没有陈述的因果事实。

### INV-015 — Serialization boundary

确定性层产生的 State 和嵌套对象必须能安全完成 `to_dict`、JSON serialization 和 typed round-trip。`mappingproxy`、不可序列化对象或内部 immutable container 不得泄漏到 State 边界。

## 仓库已验证的补充不变量

### INV-016 — Explicit authority pointers

历史 RCA Findings 和 Hypotheses 是不可变审计历史。当前权威结果只能通过 `authoritative_rca_finding_id` 和 `authoritative_hypothesis_id` 选择；不得按列表位置或“最后一个 Finding”推断权威。

### INV-017 — Candidate isolation and bounded generation

单个 malformed Candidate 或 unknown Evidence ID 不得使另一个有效 Candidate 失效。空 Candidate 集合是合法的 inconclusive 结果。Qwen repair 有界，正式候选最多两个且必须是 materially distinct primary explanations。

### INV-018 — Investigation stopping governance

继续调查必须存在合法、预算内、非重复且可能改变 ranking、Confirmation 或关键问题状态的高价值 Action。Budget exhaustion 不得触发确定性根因 fallback。

### INV-019 — Controlled-path compatibility

Qwen 路径的 hardening 不得无意改变 `fixed` 或 `controlled_react` 的既有语义。任何跨路径变化都需要明确的 Blast Radius 说明和兼容性回归测试。

### INV-020 — Prompt bounds are architecture

Planner 和 Qwen prompt 的 Lane/Evidence 数量与 payload 大小边界属于架构合同，不是可随意删除的性能优化。裁剪必须保持所有可引用 ID 与 typed register 原子一致。

### INV-021 — Controls are informative, not universal gates

Normal controls 和 exclusion Evidence 可以提高可信度，但除非存在已批准的确定性规则，否则不得把它们变成所有 Case 的统一硬 Gate。

## 默认治理工作流

```text
Real Smoke / Replay / failing case
  -> rca-semantic-auditor
  -> only if confirmed bug: regression-safe-patch
  -> after tests: change-impact-review
  -> when context/session handoff is needed: handoff-checkpoint
```

不得默认从 Smoke 或失败测试直接进入 Patch。推荐顺序是：

```text
Smoke -> Audit -> Design -> Patch -> Regression -> Impact Review
```

## 不变量维护依据

- [CODEX_HANDOFF.md](../../../docs/CODEX_HANDOFF.md)
- [evaluation-v2-causal-scope-spec.md](../../../docs/evaluation-v2-causal-scope-spec.md)
- [autonomous-qwen-react-spec.md](../../../docs/autonomous-qwen-react-spec.md)
- `core/yield_rca_core/evidence_models.py`
- `core/yield_rca_core/evidence_builder.py`
- `core/yield_rca_core/tool_layer.py`
- `core/yield_rca_core/hypothesis_candidate_generator.py`
- `core/yield_rca_core/causal_evidence_matrix.py`
- `core/yield_rca_core/causal_confirmation.py`
- `core/yield_rca_core/rca_reasoning_agent.py`
- `core/yield_rca_core/models.py`
- 对应 `tests/contract/`、`tests/integration/` 和 `tests/unit/` 回归测试
