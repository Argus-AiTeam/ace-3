# ACE-3 跨对话接续说明

源码快照时间：**2026-09-24T16:50:59Z**；live state 读取时间：
**2026-09-24T16:58:55Z**。live state 可能在此后继续前进。

## GitHub 双镜像

两个仓库地位完全相同，每次发布都必须由 `aHappend` 同步更新：

- 个人：<https://github.com/aHappend/ace-3>
- 团队：<https://github.com/Argus-AiTeam/ace-3>
- 分支：`main`
- 不可变日期 checkpoint：`refs/tags/checkpoint-2026-09-24-source`
- checkpoint commit：在首次 source publication commit 后生成，并在 push 前由
  final main documentation commit 写入准确 SHA

发布后必须核对两个远端的 commit 和 tree 相同。Git push 不是事务；如果只成功
一个，必须如实记录 partial sync，并继续补齐另一个，不能声称已经双同步。

## 新对话可直接复制的 starter prompt

```text
请先阅读仓库 AGENTS.md、docs/continuation/HANDOFF.zh-CN.md 和
progress-2026-09-24.json。先发现 live Argus daemon/status/runtime/backlog/
current claim/最近 native events/normal Reviewer handoff/registry 与 terminal
outputs；不要从 ambient checkout 猜 runtime。若 worker 或 mission 正在运行，
不要重启、重复执行或抢 ownership，等待合法 terminal。通过 normal Manager
沟通；Argus 负责 science、实现、运行与独立 review，我只提出目标并核对真实
receipts。只有 issuer binding、normal Reviewer 和 native terminal acceptance
才算完成。先报告 live state 相对快照发生了什么变化，再继续任务。
```

## 同机丢对话：首选恢复路径

1. 在原主机和本仓库运行 `python tools/argus_continuation.py inspect`。默认
   命令只读：它核对 daemon PID、从 `daemon.status.json` 发现实际 runtime，
   并列出 backlog、active mission 和最近 native event。
2. 再直接读取 state root 下的 `backlog.jsonl`、`events.jsonl`、
   `handoffs/<mission>/round-*.json`、`continuous.json` 与 terminal output。
   daemon alive 本身不等于 mission 有进度或已经验收。
3. 若存在 running mission，等待真实 terminal，不重启 worker、不复制
   Engineer one-shot、不清 claim/guard，也不要占用 Manager provider lock。
4. 需要给 Manager 一次新信息时，先写一个全新 message file，并使用从未用过的
   message ID；只有在 inspect 显示没有 running lifecycle 后才可显式执行：

   ```bash
   python tools/argus_continuation.py relay \
     --message-file /path/to/new-message.txt \
     --message-id ace3-unique-purpose-YYYYMMDD-NN \
     --manager-url http://127.0.0.1:<active-webapi-port> \
     --confirm-provider-call
   ```

   `--manager-url` 必须是当前 Argus WebAPI 的 credential-free localhost
   origin。若 WebAPI 需要认证，把完整 `Authorization` 值仅放在外部环境变量
   `ARGUS_WEBAPI_AUTHORIZATION`，不要写进命令、Git 或 URL。工具会从 live
   daemon 核对 runtime/profile/role model，通过正式 WebAPI normal Manager
   ingress（由 manager-context 和 session lock 串行化），并要求恰好一个
   provider start 和匹配 completion。它不会内置旧 PID、thread、任务文本或
   consumed message ID；失败/不完整回复不会伪装成 relay success。

当前原主机位置：

- workdir：`/home/argustest/ace3-argus`
- state root：`/home/argustest/.argus-skill-ace3/projects/s-62150b05`
- global root：`/home/argustest/.argus-skill-ace3`

这些是 2026-09-24 的定位信息，不是永久 launch command。runtime 必须每次从
live status 重新发现并验证 import path，不能重演历史上的 wrong-path pin。

## 磁盘/主机丢失：能力边界

GitHub 可恢复本次 curated source、RTL、host/oracle code、tests、contracts、
scripts、文档与 sanitized progress。它**不是**完整运行现场备份，不能恢复
active locks/PIDs/claims、Copilot/GitHub credential、完整 Argus state、模型
权重、全部 raw evidence 或外部 scheduled supervision。详细清单见
`HOST_ONLY_DEPENDENCIES.md`；缺失这些内容时不能声称 full recovery。

仓库文档不会重建旧 schedule126，也不会证明检查仍在持续。恢复后的任何调度
都要在新环境中重新、明确地建立。schedule126 里的 retired runtime/ID 不能
覆盖更新的 intent 和 live receipts。

## 当前科学/工程边界

`04c9ee209234` 只验收了 current-runtime real-lineage emitter 的 operational
contract，science count 为 0；它没有资格化 consumer/science。`e2ba4dbc8651`
保留为 unsealed/replan history；`609b6cad06bc` 已 native FAILED，fresh root
不存在。不能把 Reviewer 的“可在 build root 外使用 disposable temp”解释成
任务已经恢复运行，也不能 force-resume terminal FAILED lifecycle。

后续 lawful action 是 normal Manager 建立合法 successor，先在 disposable
temp 做 exact prevalidation，清理 temp，再只构建新授权 root。只有新的
post-claim Manager issuance、一次允许的 CPU-only `--check`、normal Reviewer
与 native terminal acceptance 全部真实存在后，才能声称 science restored。
不要在 review 中再次执行 Engineer one-shot。

ACE-3 当前 scientific profile 是 native S16 RTZ INT4/FP16 operators/KV，
Q24 signed-64 residual scale `2^-24`，其 residual state 宽于 FP16，不能改称
unchanged strict-FP16-state W4A16。保留原 thresholds、lineage、失败与 UNKNOWN。
GPU/RTL/FPGA/U280/XRT/HBM/Vivado/Vitis/synthesis/PPA/bitstream 仍不在当前
scientific scope；但临时 task restriction 不能变成永久 global ban。

原始 acceptance 要到 host-local handoff/terminal 中核对。公开 JSON 是带原始
hash 的 sanitized summary，不是未经变化的 authoritative receipt。

## GitHub 身份

credential 只由用户在外部提供：
`GH_CONFIG_DIR=/home/argustest/argustest2/.gh-config`，GitHub 身份必须是
`aHappend`。不得复制 token、credential 文件内容，不能改用默认 gh profile。
Copilot backend 仍在 `/home/argustest/.copilot`，与 GitHub 身份无关。

未来发布流程见 `DUAL_MIRROR_PUBLICATION.md`。
