---
title: Model24 checkpointed layer controller
description: Fixed 24-layer RTL scheduling, checkpoint transitions, and the numerical execution boundary.
---

# Model24 checkpointed layer controller

`ace3_model24_layer_controller` schedules one reusable decoder-layer boundary
across the fixed layer indices 0 through 23. A layer completion must identify the
active index and report no fault. Every accepted completion produces a retained
checkpoint containing the completed layer, next layer, and terminal flag. Layer
N+1 cannot launch until checkpoint N is accepted; token completion cannot occur
until checkpoint 23 is accepted.

Start metadata is limited to two cache slots and positions 0 through 127. An
out-of-range start, early or unsolicited completion, mismatched layer index,
unknown completion metadata, or layer fault latches the controller fault and
suppresses all transactional outputs until clear or reset.

The controller module remains arithmetic-free: it does not contain decoder
arithmetic, tensor loading, residual storage, or K/V payload storage. The
`model24-controller-rtl-cascade` harness gates execution on the controller's
authenticated natural terminal and exact launch order, then runs one compiled,
layer-indexed Verilator decoder instance per accepted launch. Each layer record
binds its 26 official tensors, input and output hidden-state hashes, simulator
terminal, exact integer-oracle trace, and independent PyTorch CPU float64
dequantized-AWQ comparison. This is a controller-driven numerical harness, not
a monolithic controller-plus-decoder RTL image.

The completed fixed-revision run consumed all 624 decoder tensors from
`Qwen/Qwen2.5-0.5B-Instruct-AWQ` revision
`db09cd27ead7fee40cdee309693cf83601b9c899`. The checkpoint SHA-256 is
`c50d807b7bed7ff314308972e0f4bcf4e5a70bc60ad88fc7df53940831ed0c1b`,
the per-layer binding document SHA-256 is
`95a46cfb25d8479a9d9921da9b78e581b1c3746c2645574880dac7ea6825ede0`,
and the ordered layer-0-through-layer-23 controller event stream SHA-256 is
`fcb4c9a6458fa141b143d9a4c7dfd10b2d15e703257067d935f635dc1bf9dbf1`.
The authenticated post-layer-23 hidden state has SHA-256
`97e729f6f905ecb62f498a6a144beecf6b695465d84fdbaf1de777ce9f5a39b6`;
its decision-token maximum absolute error is `0.08988498970425507`, within
the fixed `0.125` tolerance. The complete execution document SHA-256 is
`9d4e048d1316252d67d7e288fd4de0a2a6360f53ff49854c1127b7462578a5c1`.

The run does not execute the tokenizer/dialogue path or tied language-model
head. It also makes no synthesis, timing, area, power, FPGA, latency,
throughput, or silicon claim.
