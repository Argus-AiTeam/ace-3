# Argus Compute Engine 3 Mixed-Precision（ACE-3 MP）

ACE-3 MP 是一个证据驱动、处于综合前阶段的混合精度 Transformer 推理加速器项目。
它面向官方
[`Qwen/Qwen2.5-0.5B-Instruct-AWQ`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-AWQ)
checkpoint 建立完整的原生 AWQ 系统边界：非对称打包 INT4 权重、128 group size、
FP16 激活与残差、因果 K/V 状态、decoder 执行、Host 集成和可复现验证。

ACE-3 MP 是 ACE 硬件路线中的独立后继项目，不依赖 ACE-2 的源码树、build 目录、
fixture 路径、runtime 或 evidence store。需要复用的架构思想必须重新实现，或复制
为 ACE-3 自有且带 provenance 的资产。

ACE-3 由开源长期运行 agent harness
**[Argus](https://github.com/lbx154/Argus)** 持续规划、执行、独立审核和保存证据。

## 项目 Contract

| 项目 | 目标 |
| --- | --- |
| 模型 | 官方 Qwen2.5-0.5B-Instruct-AWQ，batch 1 |
| 首个精度 | 原生非对称 AWQ W4A16，G128 |
| Decoder 形状 | 24 层，hidden size 896，intermediate size 4,864 |
| 执行规则 | 每个 represented token 必须经过 indexed RTL layers 0–23 |
| Host 边界 | Tokenizer、embedding、final RMSNorm、tied `lm_head`、greedy selection、decode |
| 验证 | 独立 oracle、认证输入、Icarus 和 Verilator |
| 交付等级 | 先完成可复现 RTL 仿真，再声明综合/PPA/FPGA |

设计覆盖 model-bound tensor loading、原生 AWQ 解包与反量化、完整 projection
reduction、FP16 normalization 和非线性算子、RoPE、因果 K/V cache、attention 与
value composition、decoder-layer 集成、indexed 24 层执行、经过认证的持久
simulator state，以及可读自回归对话所需的 tokenizer/Host/generation 边界。

## 当前状态

ACE-3 仍是活跃研发项目，不是已经综合、部署到 FPGA、流片或完成真实性能测量的
实现。仓库已包含从原生 G128 W4A16 arithmetic lane，到完整官方 projection
reduction、FP16 residual/RMSNorm/SiLU/RoPE、因果 K/V 状态、attention、单个集成
decoder layer，再到全部 24 个 indexed decoder layers 执行的独立审核 RTL 与证据。

已验收的 full-24 fixture 使用了全部 624 个官方 decoder tensor。layer 23 后 Token 1
hidden state 的最大绝对误差为 `0.08988498970425507`，满足公开的 `0.125` bound。
Host final RMSNorm 与 tied software `lm_head` 重现了独立 reference 的 Top-10 排序，
并为固定 `Hello world` fixture 选择 token ID `0`（`!`）。Token 0/global 最大误差仍为
`2.3170627008770595`；这一 FP16 边界行为被明确披露，而不是隐藏。

当前 First Voice milestone 正在把已经审核的 decoder 扩展为自回归系统。24 个紧凑
indexed Verilator binary 已经完成 operational build，支持可保存状态、经过认证的
predecessor lineage 和 caller-held trusted commitment。真实 Hybrid RTL 对话
traversal 正在运行：每个 prompt token 和每个反馈生成 token 都必须通过 RTL layers
0–23，同时保持逐层因果 K/V 状态。Host 只负责 chat serialization、tokenization、
embedding lookup、final RMSNorm、tied-head selection、decode 和 feedback。

目前还没有验收通过的可读 RTL 对话。RTL final RMSNorm、streaming tied
`lm_head`/Top-K、W8A16、BF16/FP16、更大模型尺寸、综合、时序收敛、PPA、FPGA
部署和真实硬件性能仍属于后续 milestone。

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| `ace3/contracts/` | 算术、接口、lineage 和证据的机器可读 contract |
| `ace3/model/` | 独立 bit-level oracle、向量工具和 Host/runtime driver |
| `ace3/rtl/` | 可综合 SystemVerilog 实现 |
| `ace3/tb/` | Icarus 和 Verilator testbench |
| `ace3/fixtures/` | 带 provenance 的小型源码内模型 fixture |
| `design/` | RTL manifest 与 requirement-to-evidence traceability |
| `docs/results/` | 经审核、范围明确的结果说明 |

生成向量、仿真对象、trace、模型权重和本地 agent 状态不属于源码。

建议从[文档导航](docs/INDEX.md)、[当前状态](docs/STATUS.md)、
[架构](docs/ARCHITECTURE.md)和[快速上手](docs/GETTING_STARTED.md)开始。
其中状态页是已验收、进行中和明确未声明结果的权威说明。

## 可复现入口

```sh
# 列出支持的验证与 Model24 入口。
make help

# 运行独立算术 oracle。
make oracle

# 运行源码内 AWQ fixture regression。
make test

# 验证公开的 Model24 controller 与 source/unit evidence；
# 不会重跑已经 sealed 的 full-24 numerical cascade。
make model24-publication-tests

# 运行 First Voice state-lineage 和 compact-builder 定向检查。
make model24-first-voice-hybrid-tests
make model24-first-voice-compact-builder-tests

# 按 docs/GETTING_STARTED.md 准备官方 checkpoint 和 tokenizer 后，
# 运行 checkpoint-bound full-24 RTL cascade。
make model24-controller-rtl-cascade
```

基础 regression 需要 Python 3.10 或更新版本、GNU Make、Icarus Verilog、Verilator
和 C++ 编译器。完整模型执行还需要固定官方 revision
`db09cd27ead7fee40cdee309693cf83601b9c899` 的 checkpoint 与 tokenizer；仓库不会
重新分发这些资产。

验证流程使用 SHA-256 绑定 contract、官方 tensor payload、serialized vector、
simulator binary 和持久状态转换。Icarus 提供有界四态检查，Verilator 执行文档明确
范围内的完整数值路径。仿真 cycle 不是硬件 latency，软件执行也不是 RTL、FPGA 或
silicon 证据。

部分流程需要仓库未附带的模型资产、综合工具或 FPGA 硬件。缺失 prerequisite 会被
明确报告，不会被描述成成功的综合、PPA、FPGA 或硬件运行。

W4A16、W8A16、BF16/FP16、更大模型和实现证据的有序计划见
[路线图](docs/ROADMAP.md)。

## Argus

Argus 为 ACE-3 提供长期工程循环：backlog 与预算监督、skill 匹配、engineer 执行、
独立 reviewer、checkpoint 和基于证据的重新规划。

- 源码：<https://github.com/lbx154/Argus>
- ACE-3 是独立硬件项目；Argus 是用于研发和监督它的通用 agent harness。

## 许可证

ACE-3 源码采用 [Apache License 2.0](LICENSE)。Qwen 模型资产遵循上游许可证，
不包含在本仓库中。
