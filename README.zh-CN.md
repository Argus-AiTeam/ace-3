# ACE-3 MP

[English](README.md) · [快速上手](docs/GETTING_STARTED.md) ·
[架构](docs/ARCHITECTURE.md) · [路线图](docs/ROADMAP.md) ·
[贡献指南](CONTRIBUTING.md)

ACE-3 MP 是一个面向混合精度 Transformer 推理的开源研究型 RTL 项目。首个配置
针对
[`Qwen/Qwen2.5-0.5B-Instruct-AWQ`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-AWQ)
的原生 AWQ W4A16 执行：打包 INT4 权重、128 group size 和 FP16 激活。

本仓库遵循一条基本原则：**每项结果都必须明确执行边界**。软件 oracle、RTL 仿真、
综合、FPGA 和真实硬件测量结果分别报告，不能相互替代。

> **研究预览版：** 当前公开源码包含经过验证的算术和 decoder 构建模块，
> 但还不是完整加速器。目前没有综合、时序、PPA、FPGA bitstream 或真实硬件性能声明。

## 当前公开内容

| 领域 | 已公开的验证边界 |
| --- | --- |
| 原生 AWQ G128 dot lane | 可综合 RTL、独立 bit oracle、Icarus 和 Verilator 仿真 |
| 完整输入 AWQ projection | 参数化 896/128/4864 几何、完整 896 输入官方 `q_proj` reduction、有界仿真 |
| FP16 adaptation | Residual、RMSNorm、SiLU/gate 算子及有界仿真 |
| QKV 路径 | Q/K/V projection 几何、Qwen2 RoPE、indexed FP16 K/V cache |
| Attention | Scaled QK、causal softmax 近似、cached-FP16 value composition |
| Model24 软件 schedule | 确定性的 reduced-geometry 24 层软件/oracle 执行 |

尚未作为验收硬件证据公开的内容：

- 集成 decoder layer RTL 结果；
- RTL 中的完整 24 层和 tied language-model head；
- RTL 支撑的可读多 token 对话；
- 综合、时序、PPA、FPGA 部署或硬件性能。

## 仓库结构

```text
ace3/
  contracts/   机器可读的算术、接口和证据 contract
  model/       独立 bit-level oracle 与确定性向量工具
  rtl/         可综合 SystemVerilog 模块
  tb/          Icarus 与 Verilator testbench
design/        RTL manifest 与 requirement-to-evidence traceability
docs/
  results/     经过审核且范围明确的结果说明
  ARCHITECTURE.md
  GETTING_STARTED.md
  ROADMAP.md
```

生成向量、日志、仿真对象、模型文件和本地状态应放在被忽略的目录中，不属于源码。

## 快速上手

依赖：

- Python 3.10 或更新版本；
- GNU Make；
- Icarus Verilog；
- Verilator 和 C++ 编译器。

查看所有常用入口：

```sh
make help
```

运行不需要模型权重、可独立执行的 Model24 reduced-geometry 软件/oracle smoke：

```sh
make model24-smoke
```

运行独立算术 oracle：

```sh
make oracle
```

完整 RTL regression 使用从官方 checkpoint 提取并经过哈希认证的小型样本。
本仓库不会重新分发模型文件。请将所需文件放入 `official_tensors/`，或将
`OFFICIAL_TENSOR_DIR` 指向只读 fixture 目录：

```sh
make OFFICIAL_TENSOR_DIR="$PWD/official_tensors" test
```

该命令会重新生成向量、验证全部序列化输入、执行篡改拒绝测试、重新编译并运行
Icarus 和 Verilator，并确认源码树没有变化。所需 fixture 文件名及分目标命令见
[快速上手](docs/GETTING_STARTED.md)。

## 已实现的 projection 几何

`ace3_awq_w4a16_projection_engine` 会消费每个输出通道的全部 AWQ group，
并且只在完整 reduction 完成后舍入。

| Qwen2.5 projection | 输入特征 | 输出特征 | AWQ group |
| --- | ---: | ---: | ---: |
| Q / O | 896 | 896 | 7 |
| K / V | 896 | 128 | 7 |
| Gate / Up | 896 | 4864 | 7 |
| Down | 4864 | 896 | 38 |

公开的官方 tensor 数值证据覆盖 layer-0 `q_proj` 的 8 个输出，每个输出都使用全部
896 个输入。其他几何具备 elaboration 和有界接口覆盖，但不宣称已经完成全部官方
tensor 数值匹配。

当前有意采用串行实现的 reference engine，每个 896 输入输出需要 910 个仿真周期；
每个 synthetic-zero 4,864 输入输出需要 4,940 个仿真周期，另加一个输出接受周期。
这些是 RTL 仿真周期数，不是综合频率、真实延迟或吞吐率。

## 验证方法

- Arithmetic contract 明确定义 packing、位宽、舍入、reset 和 stream 行为；
- Python oracle 是独立可执行规范，不从 RTL 自动生成；
- 序列化向量由 SHA-256 绑定，并在仿真前验证；
- Icarus 提供四态 X/Z 检查，Verilator 提供独立二态执行；
- 软件 fallback 永远不能作为 RTL 或硬件完成结果。

模型相关 artifact 绑定官方 checkpoint revision
`db09cd27ead7fee40cdee309693cf83601b9c899`。使用者应按上游模型许可证和条款自行
获得模型资产。

## 许可证

ACE-3 源码采用 [Apache License 2.0](LICENSE)。Qwen 模型及从 checkpoint 提取的
样本属于独立的上游资产，不包含在本仓库中。
