---
name: handoff-checkpoint
description: 在多次 context compaction、跨模块 Patch 或准备新会话时生成或更新 docs/CODEX_HANDOFF.md，保存设计原因、不变量、证据和下一会话只读启动顺序；不继续开发。
---

# Handoff Checkpoint

本 Skill 只负责工程交接。它可以生成或更新 `docs/CODEX_HANDOFF.md`，但不负责继续开发、修 Bug、运行新的 Smoke 或替代 Audit/Impact Review。

## 启动边界

1. 完整阅读共享的 [Architecture Invariants](../_shared/architecture-invariants.md)。
2. 如果 [CODEX_HANDOFF.md](../../../docs/CODEX_HANDOFF.md) 已存在，必须先完整阅读，再增量更新；不得丢失仍然有效的设计历史和禁止事项。
3. 读取当前 Git status、branch、HEAD、相关 log/diff、现有测试结果和用户明确指定的 State/Smoke 输出。
4. 不读取 private Ground Truth 或 external-score packet 内容，不修改 Formal evaluation 数据。
5. 不修改业务代码、测试或 Prompt；除 handoff 文档外不写文件，不 stage、commit、push 或删除输出。
6. 不运行真实 Qwen、付费测试或新的开发验证。只记录已经存在且可证明的结果。

## 事实纪律

- 让一个完全不知道聊天历史的新 Codex Agent 仅靠仓库和 handoff 就能接手。
- 区分 completed、verified、not run、failed、blocked 和 unknown；不得把“计划运行”写成“已通过”。
- 每项 Patch 记录为什么这样设计、保护了哪些 invariant、为什么没有采用明显替代方案。
- 使用准确 commit、文件、测试和 State 路径；不要复制 private Ground Truth。
- 把 confirmed bug、architecture weakness、expected conservative behavior 和 unverified suspicion 分开。
- 如果当前工作区有未提交变化，逐文件描述归属和目的，不假设它们都是当前 Agent 创建的。

## `docs/CODEX_HANDOFF.md` 最低内容

文档至少包含以下内容；可以根据现有结构合并相邻章节，但不得遗漏：

1. Project architecture overview
2. RCA main data flow
3. Python / Qwen responsibility boundary
4. Architecture Invariants，并引用共享 source of truth
5. Completed patches
6. 每项 Patch 的设计原因
7. 已解决且有 regression test 的问题
8. Current unresolved issues
9. Confirmed bugs
10. Architectural weaknesses
11. Expected conservative behaviors
12. Unverified suspicions
13. Latest real Smoke results
14. Relevant replay/test results，含未运行范围
15. Current git status、branch、HEAD 和 remote-tracking caveat
16. Uncommitted changes
17. Important files
18. Important State / output paths
19. Things the next session must NOT do
20. Recommended next read-only audit sequence
21. New Session Bootstrap Prompt

不得只写“修了 A、修了 B、测试通过”。必须保留传播路径、设计原因、回归保护和仍未验证的风险。

## 更新步骤

1. 建立当前 checkpoint 的事实清单和证据路径。
2. 对照旧 handoff，标出新增、失效和仍然有效的内容。
3. 更新架构、Patch 历史、测试结果、风险分类和禁止事项；不要重写成丢失历史的简版。
4. 生成一个可直接复制的新会话 Bootstrap Prompt，要求新 Agent 先读 handoff，再做只读 Git/State/test audit。
5. 检查所有路径、commit、测试数量和状态用词是否与仓库一致。
6. 查看最终文档 diff，确认只修改 `docs/CODEX_HANDOFF.md`，且没有把聊天私密内容或 Ground Truth 写入仓库。

## 完成报告

报告：

- handoff 文件路径；
- 新增或更新的主要章节；
- 当前 Git/测试/Smoke 状态；
- 哪些信息仍是 unknown；
- 新会话应从哪里开始。

完成后停止。不要自动调用 `regression-safe-patch`、继续实现 unresolved issue、运行 Full Regression 或发起真实 Qwen Smoke。
