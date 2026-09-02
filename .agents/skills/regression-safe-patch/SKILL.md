---
name: regression-safe-patch
description: 在 RCA Bug 已完成审计且 root cause 已确认后，按回归优先、最小边界和完整副作用说明修改代码；root cause 仍是推测时必须停止并返回审计。
---

# Regression-Safe Patch

本 Skill 只用于已经确认的 Bug。调用本 Skill 不会放宽外部写入、真实 Qwen、付费测试、push、发布或 Formal 数据修改的授权边界。

## Patch 前置条件

1. 完整阅读共享的 [Architecture Invariants](../_shared/architecture-invariants.md)。
2. 阅读当前 [CODEX_HANDOFF.md](../../../docs/CODEX_HANDOFF.md) 中与该 Bug 相关的设计历史、禁止事项和现有回归测试。
3. 取得以下证据：可复现的 observed failure、确定的最早错误层、受影响的合同和至少一个可执行的回归测试设计。
4. 如果 root cause 仍是推测、依赖单一 Formal Case 猜测、或需要先放宽 Gate 才能解释为 Bug，停止修改并切回 `rca-semantic-auditor`。

## 开始修改前必须先输出

```text
Observed failure:
Confirmed root cause:
Affected subsystem:
Upstream dependencies:
Downstream dependencies:
Architecture invariants touched:
Potential regression surface:
Proposed patch boundary:
Files expected to change:
Files explicitly not expected to change:
```

在用户尚未授权修改业务代码时，这份输出只是 Patch 设计，不构成修改授权。

## 强制流程

### 1. Reproduce

用最小 deterministic fixture、contract test 或离线 replay 复现旧行为。优先使用合成数据和公开 fixture；不要用 private Ground Truth 编写测试。

### 2. Root Cause

确定最早破坏合同的位置，区分根因与下游症状。记录代码路径、State 字段和失败条件。

### 3. Blast Radius

列出所有可能受影响的调用者、Gate、Planner、State、Prompt、serialization、controlled path 和兼容性路径。

### 4. Architecture Invariant Review

逐条标出触及的 INV 编号。若 Patch 与某条 invariant 冲突，先停止并重新设计。

### 5. Patch Design

选择能修复最早错误层的最小一致边界。共享语义应修在共享 predicate、projection 或 state transition 中，不在单个 Case 周围加旁路。

### 6. Regression Test Design

每个 confirmed bug 优先添加 reproduction/regression test，并至少添加一个邻接语义测试。测试设计必须证明：

```text
Before patch:
旧行为能够复现失败，且失败原因与 confirmed root cause 一致。

After patch:
Bug 被修复；相邻合法行为、保守行为和兼容路径仍保持。
```

### 7. Minimal Implementation

只修改 Patch boundary 内的代码。出现新的跨层需求时暂停，更新 Blast Radius 和文件清单后再继续。

### 8. Targeted Tests

先运行新增测试和直接相关的现有 tests。确认失败从预期原因转为通过，而不是被 skip、mock 或绕过。

### 9. Neighboring Regression Tests

运行同一语义边界附近的测试，包括至少一个应继续失败或继续保持 inconclusive 的保守场景。

### 10. Full Regression

运行仓库约定的完整离线回归。若环境、时间或授权不允许，明确记录未运行的范围，不得宣称“完整修复完成”。真实 Qwen 和付费 Smoke 必须单独获得用户授权。

### 11. Static Checks

运行项目已有的 formatter、lint、type 或 serialization checks；不引入新的工具链作为修复副作用。

### 12. Git Diff Audit

检查 `git diff --check`、`git diff --stat` 和完整相关 diff。确认没有修改 Formal 数据、输出包、私有数据、无关文档或用户已有改动。

### 13. Completion Report

报告 root cause、修改边界、测试结果、未运行项、剩余风险和需要 `change-impact-review` 重点检查的跨层影响。

## 明确禁止

- 根据单个 Formal Case、设备 ID、Recipe ID、Lot ID、Wafer ID 或 Case ID 写硬编码。
- 为了让结果变成 `supported` 而直接放宽 Confirmation Gate。
- 让 Python 替 Qwen 发明、替换或确定正式 RCA Candidate。
- 让 Python 偷偷补写 Qwen 没引用的 Evidence。
- 因为某个测试失败而跳过、删除或降低已有校验。
- 修改多个架构层但不解释 Blast Radius。
- 没有 regression test 就宣布 confirmed bug 已修复。
- 把 missing citation 当作 source unavailable，或为已有 Evidence 重跑数据源来掩盖 citation 缺口。
- 未经授权运行真实 Qwen、付费测试、修改 Formal evaluation 数据、stage、commit 或 push。

## 完成标准

只有同时满足以下条件才能建议进入 `change-impact-review`：

- confirmed root cause 已在正确架构层修复；
- reproduction test 在修复后通过；
- 至少一个邻接语义测试证明没有误伤；
- 相关回归和静态检查结果已记录；
- diff 中没有未解释的跨层变化；
- 未运行的 Full Regression 或付费验证被明确标为未完成，而不是隐含通过。
