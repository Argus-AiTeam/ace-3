# ACE-3 跨对话接续说明

**2026-10-12 checkpoint 输入增量：** 原样发布已独审的 position-3 LM-head
validator 及其定向测试，Python `validate()` 现在必须显式传入 keyword-only
`checkpoint`，CLI `validate` 必须传 `--checkpoint`。准备和核验应使用同一绑定
模型文件，不再静默回退归档内不存在或不同的默认文件；oracle 算法与数值门槛未改。
精确源码及审核范围见
[源码记录](progress-2026-10-12-bound-checkpoint.json)。
本次只复用已有独审和原日志，不重跑科学。测试包含内部证据/模型桩，只证明参数
传递，不冒充完整生产链回归、外部来源核准或 V3 通过。私有 producer、原始日志、
模型数据和活动状态未公开；源码同步也不覆盖运行现场。旧 checkpoint 标签不动。

**2026-10-12 decoder 失败探针增量：** 原样发布一份已审 C++ harness，
注入失败分支不读取 expected trace/final，在记录首个 DUT 输出后明确失败；
正常数值比较保持。完整 SHA 和独审来源见
[单文件源码记录](progress-2026-10-12-decoder-failure-probe.json)。
原 canonical exit0 不等于完整 V2 接受：冻结层间 hash 替换的独立归因、
LD_PRELOAD 跟踪与外部 syscall 观察的区别仍须核准。相关 Makefile/generator/
tracer 未发布，旧失败和审核决定不改写。三层两 token 与 24 层控制器协议
不冒充全模型数值链；Icarus full trace 限制保留。范围 development/not_certified，
不重跑科学、不覆盖运行现场。新 checkpoint 为
`checkpoint-2026-10-12-decoder-failure-probe`，旧标签保持原位。

**2026-10-12 文件门控后续增量：** 两份 FP16 adaptation/QKV testbench 原样继承
已审的输入文件可读性和 VCD 路径可写性检查，打开失败时明确 `$fatal`，不让工具的
非致命文件告警冒充检查成功。精确源码和独立审核摘要见
[文件门控源码记录](progress-2026-10-12-file-gates.json)。保留已发布的 Makefile、
长路径、PASS 标记和 SiLU 修复。未重跑科学实验、未改变预期值或旧终态；
先前执行的 exit0 和独立 review 不覆盖其原生结算拒绝，V1 仍未认证。
新检查点为 `checkpoint-2026-10-12-icarus-file-gates`，历史标签保持原位。

**2026-10-12 增量：** 原样同步已独立审核的 Makefile、FP16 adaptation 和 QKV
testbench 三文件：保留长向量路径、显式 VCD 输出路径，并要求子仿真的真实 PASS
标记。精确 SHA、审核范围及证据限制见
[10 月 12 日源码摘要](progress-2026-10-12.json)。SiLU 修复字节保持不变。
原执行即使 exit0 也未获 V1 验收；本次不重跑实验、不改权威回执或活动现场。
新提案仍须沿正常 Manager、独立 runtime-pass Reviewer 和执行消费者链路审核。
准备任务的禁止执行限制不能复制进未来 execution target，源码打包通过不等于准入。
对应新检查点为 `checkpoint-2026-10-12-vcd-harness`，旧标签不移动。

**2026-10-11 增量：** 已同步原生 Engineer 实现、独立 Reviewer 接受的 SiLU WIDTH
修复。精确源码及原始审核/日志指纹见
[10 月 11 日源码摘要](progress-2026-10-11.json)。这是开发级交付，V1 全量回归仍未
验收；不得把源码同步当运行时采用，也不得在新归档中退回该文件的修复前版本。
对应双仓检查点为 `checkpoint-2026-10-11-silu-width`，既有标签不移动。

下文保留 9 月 28 日历史现场，不代表当前运行中的 owner 或发布后的实时状态。

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
先读 AGENTS.md、HANDOFF.zh-CN.md、progress-2026-10-12-bound-checkpoint.json，
保留此前 progress-2026-10-12-decoder-failure-probe.json 的记录，
保留此前 progress-2026-10-12-file-gates.json 的文件门控记录，
再读此前 progress-2026-10-12.json 和历史结果摘要。
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

10 月 11 日追加一份已审核 RTL 修复；10 月 12 日追加上述摘要标识的三份已审核
构建/测试平台源码及接续说明，随后原样补入两份 testbench 的文件打开门控；
不重发或重新认证下列历史科学实现。原始审核、
开发日志和完整运行状态仍保存在主机，不公开上传；无关旧 admission 日志不作为
本次 FP16/QKV 检查证据，缺少原退出侧车时明确保持未知。

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
