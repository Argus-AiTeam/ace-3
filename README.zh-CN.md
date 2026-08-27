# ACE-3 MP：ARGUS Mixed-Precision Engine

[English](README.md) · [文档导航](docs/INDEX.md) ·
[当前状态](docs/STATUS.md) · [快速上手](docs/GETTING_STARTED.md) ·
[架构](docs/ARCHITECTURE.md) · [路线图](docs/ROADMAP.md)

ACE-3 MP 是一个独立、证据优先的混合精度 Transformer 推理 RTL 项目。首个实现
配置面向官方
[`Qwen/Qwen2.5-0.5B-Instruct-AWQ`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-AWQ)
checkpoint 的原生 AWQ W4A16 执行：非对称打包 INT4 权重、128 group size，以及
FP16 激活、残差和 K/V 状态。

本仓库遵循一条基本原则：**结果的可信度不能超过它实际执行的边界**。软件 oracle、
RTL 仿真、综合、FPGA 和真实硬件测量必须分别报告。

ACE-3 由开源长期运行 agent harness
**[Argus](https://github.com/lbx154/Argus)** 持续规划、执行、独立审核和保存证据。

> **研究预览版：** 当前仓库包含经过验证的 RTL 模块、controller 驱动的 24 层
> RTL cascade，以及支持持久状态 Hybrid RTL 生成的基础设施。目前尚未发布验收通过
> 的可读 RTL 对话、综合/PPA、FPGA bitstream、时序收敛或真实硬件性能结果。

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

## 当前状态

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

完整 claim 边界和当前 milestone 见[当前状态](docs/STATUS.md)。

## 执行边界

```text
Host
  chat template → tokenizer → embedding lookup
                         │
                         ▼
RTL
  layer 0 → layer 1 → ... → layer 23
     │          │                  │
     └── 持久、经过认证的 FP16 K/V 状态 ──┘
                         │
                         ▼
Host
  final RMSNorm → tied lm_head → greedy token → feedback
```

在 First Voice 配置中，每个 prompt token 和每个反馈生成 token 都必须通过全部 24 个
indexed RTL decoder layers。Host 只负责序列化、tokenization、embedding、最终
normalization、tied-head 选择、解码和反馈。纯软件 hidden-state 路径不能被描述为
RTL 对话。

## 仓库结构

```text
ace3/
  contracts/   算术、接口、lineage 和证据的机器可读 contract
  model/       独立 bit-level oracle、向量工具和 Host/runtime driver
  rtl/         可综合 SystemVerilog
  tb/          Icarus 和 Verilator testbench
design/        RTL manifest 与 requirement-to-evidence traceability
docs/
  results/     经审核、范围明确的结果说明
  INDEX.md     文档导航
  STATUS.md    当前验收边界与进行中工作
  ROADMAP.md   有序开发计划
```

生成向量、仿真对象、trace、模型权重和本地 agent 状态不属于源码。

## 可复现入口

依赖：

- Python 3.10 或更新版本；
- GNU Make；
- Icarus Verilog；
- Verilator 和 C++ 编译器。

```sh
make help
make oracle
make test
make model24-publication-tests
make model24-first-voice-hybrid-tests
make model24-first-voice-compact-builder-tests
```

`model24-publication-tests` 验证公开 controller 和 source/unit evidence，不会重跑
sealed full-24 numerical cascade。准备好官方模型资产后，完整 checkpoint-bound RTL
cascade 使用：

```sh
make model24-controller-rtl-cascade
```

完整模型执行还需要官方 checkpoint 和 tokenizer。本仓库不重新分发这些资产。路径、
环境变量和分目标依赖见[快速上手](docs/GETTING_STARTED.md)。

## 验证体系

1. **Contract：** packing、位宽、舍入、reset、stream 行为和 claim scope 均有机器可读定义。
2. **独立 oracle：** Python 参考结果不从 DUT 实现逻辑自动生成。
3. **认证输入：** checkpoint revision、tensor、向量、二进制和状态转换均由 SHA-256 绑定。
4. **独立 simulator：** Icarus 负责有界四态检查，Verilator 负责文档明确范围内的完整二态数值执行。
5. **Fail-closed lineage：** 恢复持久 RTL 状态时必须匹配 caller-held trusted commitment。
6. **明确 non-claim：** 仿真周期不是硬件延迟，软件执行也不是 RTL、FPGA 或 silicon 证据。

固定模型 revision：
`db09cd27ead7fee40cdee309693cf83601b9c899`。

## 精度路线

1. 原生 AWQ W4A16；
2. W8A16；
3. BF16/FP16；
4. 1.5B 和 3B 模型尺寸；
5. 可复现的综合、PPA 和 FPGA 证据。

只有在真实 datapath 和验证 contract 存在后才会加入新的精度模式。

## 文档

- [文档导航](docs/INDEX.md)
- [当前状态与 claim matrix](docs/STATUS.md)
- [架构](docs/ARCHITECTURE.md)
- [快速上手](docs/GETTING_STARTED.md)
- [First Voice Hybrid RTL](docs/FIRST_VOICE_HYBRID_RTL.md)
- [RTL traceability](design/RTL_TRACEABILITY.md)
- [路线图](docs/ROADMAP.md)
- [贡献指南](CONTRIBUTING.md)

## Argus

Argus 为 ACE-3 提供长期工程循环：backlog 与预算监督、skill 匹配、engineer 执行、
独立 reviewer、checkpoint 和基于证据的重新规划。

- 源码：<https://github.com/lbx154/Argus>
- ACE-3 是独立硬件项目；Argus 是用于研发和监督它的通用 agent harness。

## 许可证

ACE-3 源码采用 [Apache License 2.0](LICENSE)。Qwen 模型资产遵循上游许可证，
不包含在本仓库中。
