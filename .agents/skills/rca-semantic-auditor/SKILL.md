---
name: rca-semantic-auditor
description: 对真实 RCA Smoke、Replay、Formal Case 或复杂测试失败先执行只读语义审计，区分实际 Bug、架构弱点、保守行为和测试缺陷；默认不修改业务代码。
---

# RCA Semantic Auditor

遵守 **READ-ONLY AUDIT FIRST**。本 Skill 的职责是建立事实和故障分类，不是修复代码。

## 启动边界

1. 完整阅读共享的 [Architecture Invariants](../_shared/architecture-invariants.md)。
2. 如果任务延续既有会话，先完整阅读 [CODEX_HANDOFF.md](../../../docs/CODEX_HANDOFF.md)。
3. 默认只允许读取代码、测试、Git metadata、明确指定的 State/Replay/Smoke 输出和公开评估数据。
4. 不修改业务代码、测试、Prompt、State、Formal 数据或 handoff；不 stage、commit、push 或删除输出。
5. 不读取 private Ground Truth 或 external-score packet 内容，除非用户明确授权并且任务确实需要。
6. 不运行真实 Qwen、付费 Smoke 或产生外部副作用。离线测试也只在用户要求执行时运行；单纯审计优先读取已有结果和回归测试。

如果用户同时要求“审计并修复”，先完成审计并输出 Patch 前置结论。只有 root cause 已确认且修改权限明确时，后续才可切换到 `regression-safe-patch`。

## 故障分类顺序

不要因为出现 `inconclusive`、`insufficient_evidence`、`blocked` 或测试失败就假设 Gate 有 Bug。依次检查并区分：

1. execution failure
2. data/evidence failure
3. candidate generation failure
4. evidence closure failure
5. scope modeling failure
6. competition failure
7. planner stopping behavior
8. confirmation behavior
9. impact scope behavior
10. serialization/state failure
11. expected conservative behavior

对每一层追踪：输入合同、权威对象、状态转换、输出合同、上游依赖和下游影响。优先使用 typed fields、Evidence IDs、State 和代码路径，不用自然语言摘要替代证据。

## RCA Smoke 必查项

- Candidate 是否识别了正确的根因主体；只把这项称为“确认”时必须有独立证据或已知答案权限。
- Candidate 是否包含无 typed Evidence 支撑的实体、数字、方向、时间、Lot、Wafer、Recipe、Lane 或物理机制推断。
- Positive Evidence 是否出现 Lot、Wafer、Recipe、Equipment、Chamber、Operation 或 Lane 污染。
- 提示中可见的可引用 Evidence IDs 是否都是 typed register 中的 referenceable IDs。
- Candidate citations 是否闭合，repair 是否删除了此前有效 citation，Python 是否修改过 Candidate citations。
- Exposure、Parameter、Temporal、Outcome、Mechanism 是否分别闭合；不要用总体 Matrix status 掩盖单项状态。
- Scope identity、process coverage、outcome coverage 和 shared-effect support 是否被混淆。
- Competition 是否形成独立 primary mechanism/scope/direction 候选，而非同义改写、modifier 或仅存在于挑战 prose 的替代机制。
- Planner 是否在评估所有 unresolved executable gaps 前过早停止；终止原因是否与 Action Value、预算和 unavailable source 状态一致。
- Confirmation Gate 是否使用正确的 Matrix、Competition 和 causal-chain 状态；不得以期望答案反推 Gate 应通过。
- Impact Lot 是否逐 Lot 满足 scope、exposure、temporal、parameter/process、Recipe/Operation 和 outcome 条件。
- 最终 conclusion、authoritative Finding/Hypothesis、Matrix、Gate、Competition、warnings 和 serialized State 是否一致。

## 审计方法

1. 记录当前 Git branch、HEAD、status 和审计输入路径；不自动 fetch 或更新依赖。
2. 确认哪一个 Finding、Hypothesis、Matrix 和 State 字段具有 authority；禁止按列表位置猜测。
3. 从观察到的最终行为反向追踪到最早发生语义偏差的层。
4. 阅读相关生产代码之前和之后，都检查现有 regression tests，判断是代码违约、测试缺失还是测试本身编码了旧语义。
5. 用最小可证伪陈述记录 suspicion。没有代码或 State 证据时不得写成 confirmed root cause。
6. 明确哪些行为虽然不理想，但符合保守 Gate 和当前架构合同。

## 必须输出

按以下字段输出，每一项都给出代码、测试、State 或 Git 证据；没有证据时写 `unknown`：

```text
Observed behavior:
Expected behavior:
Confirmed facts:
Unverified suspicions:
Confirmed root cause:
Upstream dependency:
Downstream consequence:
Relevant Architecture Invariants:
Classification:
Recommended next step:
```

`Classification` 只能使用一个主分类，并可列出次要分类：

- confirmed bug
- architecture weakness
- expected conservative behavior
- test deficiency
- unknown / requires more evidence

如果存在多个独立问题，为每个问题分别填写上述结构，不要把它们合并成一个模糊 root cause。

## 停止条件

- 如果 root cause 尚未确认，明确写出还缺什么证据并停止；不得进入 Patch。
- 如果结论是 expected conservative behavior，不提出放宽 Gate 的 Patch。
- 如果发现 confirmed bug，只提供最小架构级修复边界和 regression test 建议；除非用户随后明确授权，不修改代码。
