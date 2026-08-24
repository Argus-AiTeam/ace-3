# ACE-3 MP

[English](README.md) | [简体中文](README.zh-CN.md)

ACE-3 MP 是一个面向 Transformer 推理、以证据为先的混合精度加速器项目。
首个实现配置针对官方 Qwen2.5-0.5B-Instruct AWQ 检查点，采用 W4A16 执行。

> **私有预发布阶段：** 本仓库目前尚未宣称已完成完整加速器、综合设计、
> FPGA bitstream、实测硬件性能或芯片流片。

## 为什么开发 ACE-3

ACE-2 探索由有符号 INT4 权重、INT8 激活和 Scale32 元数据组成的严格整数路径。
ACE-3 是一条独立的架构演进路线：在保持 ACE-2 作为独立且不变的基线同时，
增加混合精度执行能力。

初始 AWQ 软件资格验证已经确认：

- 官方 AWQ tensor contract：G128、打包 INT4 `qweight` 和 `qzeros`、
  FP16 scale，以及原生 GEMM 排列；
- 168 个已重建的量化 Transformer Linear 模块；
- CPU reference 能够完成基本对话、指令遵循、多轮记忆、翻译、摘要、
  安全拒答和简单代码生成；
- 模型在严格 JSON 格式、一个事实解释问题和较长代数问题上仍存在不足。

以上均为软件 reference 结果，不是 RTL 或硬件证据。

## 初始配置

| 配置 | 目标 | 状态 |
| --- | --- | --- |
| `AWQ_W4A16` | 原生 AWQ G128 权重与 FP16 激活 | 完整输入的串行 projection RTL 已验证 |
| `AWQ_W4A16_ADAPT` | FP16 residual、RMSNorm 和 SiLU/gate 数据流 | 有界 RTL 仿真已验证 |
| `AWQ_W4A16_QKV` | Q/K/V projection 几何、Qwen RoPE 和 FP16 K/V cache | 有界 RTL 仿真已验证并发布 |
| `AWQ_W4A16_ATTN` | scaled QK、causal softmax 和 cached-FP16 V composition | 有界 RTL 仿真已验证并发布 |
| `ACE_W4A8` | 与现有严格整数路线兼容 | 计划中 |

当前已实现的 RTL 边界包括已验收的 G128 primitive，以及一个参数化串行引擎。
该引擎会组合每个输出通道的全部输入 group，并支持分块输出。

## 仓库结构

```text
ace3/
  rtl/         可综合 ACE-3 RTL
  tb/          RTL testbench
  model/       bit-level 软件 oracle 与向量生成
  contracts/   已实现的精度与接口 contract
docs/          架构与 roadmap
```

生成日志、模型权重、构建输出和本地 evidence bundle 默认不应提交到源码仓库。

## 独立验证

已验收的 G128 dot lane 和完整输入 projection engine 共用仓库根目录的验证入口，
仅依赖 Python、GNU Make、Icarus Verilog 和 Verilator：

```sh
make clean
make OFFICIAL_TENSOR_DIR=/home/argustest/ace-2/build/ace2_chat_demo/qwen25-05b-instruct-awq-software-baseline-cf01/official test
```

`OFFICIAL_TENSOR_DIR` 是显式且可配置的；默认值为上面展示的路径。生成器会原地读取
三个官方 layer-0 `q_proj` 样本文件，验证其冻结的 SHA256，并且绝不会写入该目录。
向量、仿真对象、二进制和日志只生成在被忽略的 `build/` 下；
`build/logs/` 记录每条命令及其结果。

每次运行 `make test` 都会删除并重新生成 `build/vectors/`，重新执行 oracle 和 JSON
校验，重新编译并运行两组 Icarus 测试，并重新构建和运行 Verilator。完整输入
projection 向量也会被独立重新生成、认证和仿真，然后才打印聚合 PASS。
语义检查不会通过 stamp cache 跳过。

Attention target 在此流程上增加固定的 14-query/2-KV-head GQA mapping、
64 元素 FP16 QK accumulation 与 1/8 scaling、causal masking、Q0.24
max-subtracted softmax，以及 cached-FP16 value composition。其官方输入来自
确定性且经过哈希检查的 scale selection，并非捕获的运行时 activation；
相关 claim 仍严格限定在动态 RTL 仿真边界。

历史冻结 manifest 保持逐字节不变。独立、受源码控制的 binding contract 会认证
validator 或 simulator 消费的五个序列化 artifact 的 SHA256、字节数和行数：
`manifest.json`、`meta.hex`、`pairs.hex`、`cases.txt` 和
`vector_params.svh`。仿真前必须先通过验证，`make test` 还会确认被篡改的
`meta.hex` 副本会遭到拒绝。

验证覆盖整数 oracle、确定性 seed `0xACE3CF01`、30 个 case、3,840 个 G128 pair、
精确 accumulator、零 ULP binary16 结果、protocol invariant、Icarus 四态 X/Z probe，
以及独立的 Verilator 执行。当前配置下 Verilator 是二态 simulator，因此 X/Z 声明
仅来自有界 Icarus 测试。这些是动态仿真检查，不是形式验证。

## 完整输入 projection 边界

`ace3_awq_w4a16_projection_engine` 由 `IN_FEATURES` 和 `OUT_FEATURES`
参数化。它会顺序处理连续的输出 tile；每个 AWQ group 消费一条 metadata 和
128 组 activation/qweight pair；将每个精确的 96-bit Q47.48 group accumulator
符号扩展到 102-bit Q53.48 跨 group accumulator；并且只在全部 group 完成后舍入一次。
它不会再次累加 primitive 已经舍入过的 FP16 group 输出。

| 模块 | 输入特征数 | 输出特征数 | Group 数 |
| --- | ---: | ---: | ---: |
| q/o projection | 896 | 896 | 7 |
| k/v projection | 896 | 128 | 7 |
| gate/up projection | 896 | 4864 | 7 |
| down projection | 4864 | 896 | 38 |

官方 tensor 数值证据使用固定 revision
`Qwen/Qwen2.5-0.5B-Instruct-AWQ@db09cd27ead7fee40cdee309693cf83601b9c899`
中经过认证的 layer-0 `q_proj` qweight、qzeros 和 scale，并配合确定性生成的
FP16 activation。测试覆盖通道 4 至 11 的全部 896 个输入。定向输出覆盖跨 group
仅舍入一次时的 cancellation、saturation、subnormal、zero 和 invalid operand。
其他几何已在两个 simulator 中完成 elaboration 和 lint，但不宣称具备官方 tensor
数值一致性证据。

这个有意采用串行 single-lane 的引擎，在 RTL 仿真中的测量延迟为：
896 输入时，从 start 被接受或上一个输出被接受到 `out_valid` 需要 910 cycles；
4,864 输入的 synthetic-zero 输出需要 4,940 cycles。输出 acceptance 额外消耗一个
cycle。这些是仿真周期数，不是综合、时序或性能测量结果。

## 证据政策

每一项已发布 claim 都必须明确其执行边界：

- software reference；
- RTL simulation；
- synthesis 与 timing；
- FPGA deployment；
- 或 measured hardware。

软件 fallback 永远不能作为 RTL 或硬件完成结果报告。尚不支持的精度模式应保持缺失，
而不是使用 placeholder 伪装支持。

## 与 ACE-2 的关系

ACE-2 仍然是独立项目，并继续推进其严格 W4A8 产品化路线。ACE-3 不移动、不重命名、
也不覆盖 ACE-2 RTL。所有复用都必须通过显式、经过 review 的接口和可独立复现的证据完成。

更多信息请参阅：
[架构](docs/ARCHITECTURE.md)、
[Roadmap](docs/ROADMAP.md)、
[Projection 结果](docs/results/AWQ_W4A16_PROJECTION_CF02.md)和
[贡献指南](CONTRIBUTING.md)。
