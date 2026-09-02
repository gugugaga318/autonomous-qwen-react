---
name: change-impact-review
description: Patch 完成后独立审查 git diff 实际改变的系统行为、兼容性和隐藏回归风险；默认只读，不替代 Full Regression，也不负责继续修复。
---

# Change Impact Review

本 Skill 对已经存在的 Patch 做独立副作用审查。核心问题不是“原 Bug 是否看起来修好”，而是：**这份 diff 实际改变了哪些系统行为？**

## 启动边界

1. 完整阅读共享的 [Architecture Invariants](../_shared/architecture-invariants.md)。
2. 阅读 Patch 的 confirmed root cause、Patch boundary、测试报告和相关 [CODEX_HANDOFF.md](../../../docs/CODEX_HANDOFF.md) 设计背景。
3. 默认只读检查 Git、代码、测试和已有测试结果；不修改业务代码、不重写 Patch、不 stage、commit 或 push。
4. 不把本 Review 当作 Full Regression、真实 Qwen Smoke 或 Formal evaluation 的替代品。
5. 若发现问题，输出 `return to patch` 或 `requires further audit`，不要在同一职责中直接修复。

## Diff 审查顺序

1. 明确 review baseline：工作区 diff、staged diff、某个 commit 或 branch range。不要混合不同 baseline。
2. 阅读完整 diff 和新增/修改测试，而不仅是 `--stat`。
3. 从每个改动点向上游和下游反向追踪调用关系、状态转换和序列化边界。
4. 对照 Patch 声明的 boundary，标出所有实际超出边界的行为变化。
5. 检查测试是否验证语义结果，而非只匹配字符串、固定 Case ID 或实现细节。

## 必查维度

- Evidence semantics
- Entity / Lot / Wafer / Recipe binding
- Lane identity
- Candidate contract and candidate isolation
- Evidence closure and citation regression
- Scope identity / process coverage / outcome coverage / shared-effect semantics
- Competition requirement, axes, lineage and terminal semantics
- Planner legality, priority, Action Value, budget and stopping behavior
- Confirmation Gate and causal-chain behavior
- Impact Gate inclusion, exclusion and publication behavior
- State schema, authority pointers and audit consistency
- JSON serialization, typed round-trip and immutable-container safety
- Backward compatibility and old State loading
- Prompt contract, payload bounds and referenceable Evidence IDs
- Qwen/Python responsibility boundary
- `fixed` / `controlled_react` compatibility
- Existing regression tests and missing neighboring tests

如果一个模块的改动可能影响另一个 Gate、Planner、State、Prompt 或 serialization 层，必须明确列出传播路径；不能因为目标测试通过就忽略。

## 必须输出

对每个实质性行为变化输出：

```text
Changed behavior:
Intended / unintended:
Invariant affected:
Potential hidden regression:
Backward compatibility risk:
Serialization/state risk:
Prompt/schema risk:
Tests covering this change:
Missing tests:
Recommendation:
```

`Recommendation` 只能使用：

- approve
- approve with warning
- return to patch
- requires further audit

最后给出整体 recommendation，并区分：confirmed regression risk、architecture weakness、unverified suspicion 和未运行验证。

## 判定准则

- `approve`：diff 与声明边界一致，相关 invariant 有测试保护，没有发现未解释的跨层变化，且所需离线回归已完成。
- `approve with warning`：没有确认的阻塞缺陷，但存在明确、有限且已记录的未验证风险或未运行的非关键检查。
- `return to patch`：发现 confirmed regression、违反 invariant、遗漏必要测试或实际改动超出 Patch boundary。
- `requires further audit`：无法从当前 diff/测试确认语义，需要 State、Replay 或更深只读追踪；不得猜测后直接修改。

## 停止条件

完成审查和 recommendation 后停止。不要自动进入新的 Patch、Full Regression、真实 Qwen 或 handoff；这些必须由相应 Skill 和用户授权处理。
