# ACE-3 跨对话接续说明

本次快照日期：**2026-09-28**。准确观察时间与当时任务见
[progress-2026-09-28.json](progress-2026-09-28.json)，不要把快照当成实时状态。

## 当前事实与下一步

- 最新对照 `92e4223ef310` 已真实执行一次九控制 S13–S17 组合并独立审核：
  科学结果 **REJECTED**，9 项 S18 检查与 18 项原始参考方向检查均未通过。
  不能更名后重跑同一组合，详见[结果摘要](../results/STAGE16_REJECTED_20260927.json)。
- 新假设是在 L23/P0 `self_attn.o_proj` 边界使用精确整数 G128 累加，
  保持原 Stage10 FP16 输入与精度边界不变。实现任务 `d12e372181f0`
  已通过独立审核和 native DONE。
- 后继 `7a2022cef21c` 发现真实入口不兼容：node key、dependency 要求与
  原生 mission 不一致，wrapper-first import 的 source census 也不稳定。
  Reviewer 已要求重排，原生状态为 failed；**claim 和数值执行均为零**。
  这不是新机制的数值 REJECTED，更不是成功。
- 原生软件修复任务 `aba2b6c0f3d0` 在快照时运行中。应完成耦合缺陷修复与
  独立审核，再签发一致的全新一次性实验任务；不能修改或复活原失败记录。
- 正常 daemon 已恢复至 `0.1.2+e69764dbdb9bd222`。合法空计划修复和发布已完成，
  不重做封装、审核或发布，不为维持空进程而反复重启。

数值边界仍是 width896、pair `[34319,13]`、L23/P0，native S16 RTZ、G128 INT4、
FP16 operator/KV、宽于 FP16 的 Q24 residual；不能宣称 strict-FP16-state W4A16。
不使用参考输出选位、clamp 或投影来伪装修复；不放宽门槛，不重放原始 prefix、
admission、reference 或已关闭 producer，不动 ACE2，不扩展精度、规模或硬件范围。

## 新对话 starter prompt

```text
先读 AGENTS.md、HANDOFF.zh-CN.md、progress-2026-09-28.json 和本次结果摘要。
在原主机运行 python tools/argus_continuation.py inspect，核对实际 runtime、
backlog、最近独立 Reviewer 和 native terminal，不从旧对话猜当前任务。
已经完成的恢复指令、科学尝试和旧 publisher 都不能重放。运行中的任务不要
打断或复制。先核对 exact-o-proj 软件修复是否已完成独立审核，以及新 science
owner 的真实 claim/执行/结果；不能把 7a2022cef21c 的零次执行叫数值失败。
通过正常 Manager 路径处理具体缺失依赖，不创建通用准备/报告循环。
先说明 live state 相对快照的变化。
```

原主机 workdir 为 `/home/argustest/ace3-argus`，state root 为
`/home/argustest/.argus-skill-ace3/projects/s-62150b05`。这些只是定位信息，
不是固定 launch command。原始 review/receipt 必须从主机核验。

`tools/argus_continuation.py relay` 仅在没有运行中的 lifecycle 且需要新指令时
显式使用；先读 `--help`，通过实际 WebAPI 的 credential-free localhost 地址，
使用新的 message ID。不要在命令、URL 或 Git 中写凭据。

## 本次发布的范围

本次[源码清单](source-manifest-2026-09-28.json)固定 69 份源码/测试（含未变化依赖），
四份新实现/测试与 `d12e372181f0` 的历史 checkpoint 字节一致，但后续发现的
入口缺陷明确保留，不冒称可用发布。正在修改中的 repair 字节不在快照内。
9 月 26 日原科学源码证明保留在 `checkpoint-2026-09-26-source`；后续依赖版本
不能冒充旧 seal。GitHub 同步不会自动升级或恢复运行环境。
**不要用这个源码快照覆盖正在运行的工作区。**

[最新边界说明](../results/STAGE11_EXACT_O_PROJ_20260928.md)和结果 JSON 是公开摘要，含原始结果
与 review 文件 hash，不是原样 authoritative receipt。权重、raw arrays、原始
provider/operator transcript、完整状态、权限文件和凭据不上传；详见
[HOST_ONLY_DEPENDENCIES.md](HOST_ONLY_DEPENDENCIES.md)。GitHub 不构成完整主机备份，
也不能恢复或证明外部监督 schedule 正在执行。

## 对等双镜像

- <https://github.com/aHappend/ace-3>
- <https://github.com/Argus-AiTeam/ace-3>
- 两边 `main` 与 `refs/tags/checkpoint-2026-09-28-source` 必须一致；旧标签不改。

以 `aHappend` 身份执行每次发布，凭据从用户的外部
`GH_CONFIG_DIR=/home/argustest/argustest2/.gh-config` 获取，不复制其内容，不切换
Copilot 账号。遵循 [DUAL_MIRROR_PUBLICATION.md](DUAL_MIRROR_PUBLICATION.md)，
一边失败就报告 partial sync，不能强推或伪称双同步完成。
