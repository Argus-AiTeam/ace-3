# ACE-3 MP：ARGUS Mixed-Precision Engine

ACE-3 MP 是一个独立、证据优先的混合精度 Transformer 推理 RTL 项目。首个实现
配置面向官方
[`Qwen/Qwen2.5-0.5B-Instruct-AWQ`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-AWQ)
checkpoint 的原生 AWQ W4A16 执行：非对称打包 INT4 权重、128 group size，以及
FP16 激活、残差和 K/V 状态。

本仓库遵循一条基本原则：**结果的可信度不能超过它实际执行的边界**。软件 oracle、
RTL 仿真、综合、FPGA 和真实硬件测量必须分别报告。

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

设计覆盖原生 AWQ 解包与反量化、完整 projection reduction、FP16 normalization
和非线性算子、RoPE、因果 K/V 状态、attention 与 value composition、decoder-layer
集成、indexed 24 层执行、经过认证的持久 simulator state，以及可读自回归生成所需
的 Host 边界。

## 当前状态

ACE-3 仍是活跃研发项目，不是已经综合、部署到 FPGA、流片或完成真实性能测量的
实现。仓库已包含经过独立审核的 RTL 和 controller 驱动 24 层 decoder cascade
证据。真实 Hybrid RTL 对话 traversal 正在运行，但在完整 transaction chain 和生成
结果通过独立审核前，不能声明为已经验收的可读对话。

| 层级 | 已公开边界 | 状态 |
| --- | --- | --- |
| 原生 AWQ 算术 | G128 W4A16 dot lane、准确 packing 与 FP16 舍入 | RTL 仿真已验收 |
| Projection | 896/128/4864 几何和完整 896 输入官方 `q_proj` reduction | RTL 仿真已验收 |
| FP16 adaptation | Residual、RMSNorm、SiLU/gate、RoPE 和 FP16 K/V 状态 | 有界 RTL 仿真已验收 |
| Attention | Scaled QK、causal softmax 近似和 cached-value composition | 有界 RTL 仿真已验收 |
| Decoder | Indexed decoder 执行及独立参考对照 | 有界 RTL 仿真已验收 |
| Model24 | 无算术的 24 层 controller 和 layer-indexed Verilator cascade | 有界 RTL 仿真已验收 |
| Host decision | accepted fixture 上的 final RMSNorm 与 tied-head top-10/argmax | Host/oracle 边界已验收 |
| First Voice | 可保存 RTL 状态、认证 lineage、trusted tips 和紧凑 indexed-layer builder | 基础设施已验收；全层 runtime evidence 与完整 traversal 进行中 |
| Implementation | 综合、时序、PPA、FPGA 和真实性能 | 尚未声明 |

当前 Hybrid RTL 边界把 chat serialization、tokenization、embedding lookup、final
RMSNorm、tied `lm_head`、greedy selection、decode 和 token feedback 保留在 Host。
每个 prompt 或生成 token 都必须经过 indexed RTL decoder layers 0–23，并维持经过
认证的逐层持久 K/V 状态。纯软件 hidden-state 路径不算完成。

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
