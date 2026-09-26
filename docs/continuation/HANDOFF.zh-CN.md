# ACE-3 跨对话接续说明

本次快照日期：**2026-09-26**。准确观察时间与当时任务见
[progress-2026-09-26.json](progress-2026-09-26.json)，不要把快照当成实时状态。

## 已完成的事实，不是要重做的任务

- 发布修复 `287d994f2fe4`、接续 `0426fd273809` 已独立审核并 native DONE。
- `b4978b898600` 真正执行一次 CPU 数值检查，结果 **REJECTED**；有 63 次
  local oracle、27 次 projection、9 次 final norm 和 9 次 selected-pair head。
- `6738b75d0cc8` 的 v2 S18 边界实验也真实执行并独立审核，结果
  **boundary_only_explanation_rejected**，18 项终态方向中恢复数为 0。
- 后来的 `bb4f1b293dda` 是 **UNKNOWN_NOT_RUN**，不是第三个数值结果。

这说明执行接口已恢复，不说明被检验的方法成功、项目完成或硬件可用。
数值边界仍是 width896、pair `[34319,13]`、L23/P0，native S16 RTZ、G128 INT4、
FP16 operator/KV、宽于 FP16 的 Q24 residual；不能宣称 strict-FP16-state W4A16。

随后发生过过期修复指令驱动的重复 v6 发布准备。该临时指令已撤下，13:00 UTC
真实启动了基于两份 REJECTED 证据的后续分析。**不要再次创建“fresh v6 binding”
仅为满足已完成的旧 checklist。** 新计算应由具体的新假设或诊断问题决定；
新源码/权限准备仅在该任务确实需要时进行。失败和负结果必须保留。

## 新对话 starter prompt

```text
先读 AGENTS.md、HANDOFF.zh-CN.md、progress-2026-09-26.json 和本次结果摘要。
在原主机运行 python tools/argus_continuation.py inspect，核对实际 runtime、
backlog、最近独立 Reviewer 和 native terminal，不从旧对话猜当前任务。
已经完成的恢复指令、科学尝试和旧 publisher 都不能重放。运行中的任务不要
打断或复制。通过正常 Manager 路径选择真正未完成的研究任务；从实际 REJECTED
结果出发，避免无具体研究依赖的发布/授权循环。先说明 live state 相对快照的变化。
```

原主机 workdir 为 `/home/argustest/ace3-argus`，state root 为
`/home/argustest/.argus-skill-ace3/projects/s-62150b05`。这些只是定位信息，
不是固定 launch command。原始 review/receipt 必须从主机核验。

`tools/argus_continuation.py relay` 仅在没有运行中的 lifecycle 且需要新指令时
显式使用；先读 `--help`，通过实际 WebAPI 的 credential-free localhost 地址，
使用新的 message ID。不要在命令、URL 或 Git 中写凭据。

## 本次发布的范围

六份发布/candidate/capture 源码来自已审核 sealed release，三份 v2/helper/test
来自该实验记录的精确 source hashes；补齐 v1 helper 的配套测试。另归档当前
host publisher 源码，记录 mutable backlog hash 修复，但不将它冒充 sealed source，
也不导出其 host-only helper/权限依赖。不是把后来仍在变化的 live release 模块
混进旧实验的源码证明。另附 Argus Planner 一次格式修复补丁，供对应基线审阅；
不自动升级运行时。[增量清单](source-manifest-2026-09-26.json)记录全部 hash。
**不要用这个源码快照覆盖正在运行的工作区。**

[结果 JSON](../results/STAGE11_REJECTED_20260926.json) 是公开摘要，含原始结果
与 review 文件 hash，不是原样 authoritative receipt。权重、raw arrays、原始
provider/operator transcript、完整状态、权限文件和凭据不上传；详见
[HOST_ONLY_DEPENDENCIES.md](HOST_ONLY_DEPENDENCIES.md)。GitHub 不构成完整主机备份，
也不能恢复或证明外部监督 schedule 正在执行。

## 对等双镜像

- <https://github.com/aHappend/ace-3>
- <https://github.com/Argus-AiTeam/ace-3>
- 两边 `main` 与 `refs/tags/checkpoint-2026-09-26-source` 必须一致。

以 `aHappend` 身份执行每次发布，凭据从用户的外部
`GH_CONFIG_DIR=/home/argustest/argustest2/.gh-config` 获取，不复制其内容，不切换
Copilot 账号。遵循 [DUAL_MIRROR_PUBLICATION.md](DUAL_MIRROR_PUBLICATION.md)，
一边失败就报告 partial sync，不能强推或伪称双同步完成。
