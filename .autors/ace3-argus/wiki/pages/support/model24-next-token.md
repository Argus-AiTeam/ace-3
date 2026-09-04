---
title: Model24 next-token software boundary
description: Authenticated 24-layer software execution, PyTorch comparison bounds, and unsupported deployment claims.
---

# Model24 next-token software boundary

The `official-model24-next-token` target regenerates an authenticated
software/oracle execution of token IDs 9707 and 1879 through all 24 decoder
layers of the fixed Qwen/Qwen2.5-0.5B-Instruct-AWQ checkpoint. It binds all 624
per-layer tensors to their checkpoint bytes, records 480 deterministic stage
hashes, carries causal FP16 K/V lineage at both positions through every layer,
and feeds the layer-23 terminal row through final RMSNorm and the tied FP16
language-model head.

The model executor uses native asymmetric packed INT4 AWQ W4A16 G128
projections, accepted FP16 residual and RMSNorm primitives, Qwen half-split
RoPE, and a mathematical SiLU independently evaluated before FP16 rounding.
The generated terminal hidden state and full 151,936-logit vector are compared
with a PyTorch CPU float64 dequantized-AWQ reference at an absolute tolerance
of 0.125. The accepted fixed execution has maximum errors below 0.098 and both
paths select token ID 0 (`!`).

This is a deterministic two-input-token, one-next-token software result. It
does not demonstrate multi-token dialogue, full-model RTL, synthesis, timing,
area, power, FPGA execution, latency, or throughput.

## Fixed multi-token dialogue

The `official-model24-dialogue` target authenticates the accepted Model24 tensor
binding and official tokenizer, serializes the fixed chat prompt, and executes a
four-step greedy generation loop through all 24 layers and the tied head. Each
step records the selected token ID, primary and independent-reference logit
hashes, terminal-hidden comparison, and aggregate plus per-layer FP16 K/V cache
parentage.

The fixed execution selects token IDs `9707, 11, 311, 498`, matching the
independent PyTorch CPU float64 dequantized-AWQ argmax at every step, and the
official tokenizer decodes them as `Hello, to you`. It stops at the configured
four-token limit. The primary path uses a 0.25 full-vocabulary absolute-logit
tolerance; the measured maximum is below 0.239. The terminal-hidden tolerance
remains 0.125.

This demonstrates one deterministic fixed-prompt software/oracle dialogue. It
does not demonstrate full-model RTL, synthesis, timing, area, power, FPGA
execution, latency, throughput, or broader dialogue quality.

## Prompt-driven host/runtime

`make model24-host-runtime` reuses the authenticated 24-layer dialogue executor
for two fresh runs: the fixed default prompt and the caller path exercised with
an explicit prompt. The direct entry point,
`ace3/model/model24_host_runtime.py`, accepts an optional `--prompt`; caller
text is serialized with the authenticated official chat template. Evidence
under `build/model24_host_runtime/default/` and
`build/model24_host_runtime/explicit/` preserves prompt and generated token
IDs, decoded text, source and artifact hashes, fixed-revision
checkpoint/tokenizer bindings, deterministic greedy decisions, and per-layer
incremental FP16 K/V lineage.

Validation reauthenticates the live checkpoint and tokenizer and rejects stale
binding metadata, a different expected prompt, or prompt token IDs that do not
match the authenticated tokenizer. This remains a software/oracle runtime. It
does not establish full-model RTL, synthesis, timing closure, PPA, FPGA
execution, product latency, throughput, or broad dialogue quality.

Host/dialogue evidence also carries a validation-only binding closure over the
exact model repository and revision, checkpoint name/hash/size, tokenizer and
configuration artifact names/hashes, canonical prompt record, generated token
IDs, and complete generation record. The focused binding validator rejects
asset, prompt, or generated-token drift without loading the model, invoking
the Model24 executor, or interacting with execution authority. This closure
authenticates evidence lineage only; it is not new dialogue execution evidence.

## Selected-token receipt bridge

`model24_host_runtime.form_next_dialogue_step_from_receipt` is a bounded,
read-only bridge from already-produced selected-token terminal evidence to the
next host token-history step. The receipt binds the fixed checkpoint and
tokenizer, the caller's authenticated prompt record, caller-supplied terminal
evidence and receipt-use authority lineages, the selected token and FP16 logit,
and the complete deterministic 10-entry Top-K payload. Validation checks each
Q24 logit against its FP16 bits, rank ordering, uniqueness, and rank-zero
selection.

For a valid receipt, the bridge decodes the selected token and appends its ID
to the prompt token history. It does not load a model, execute an oracle,
controller, or RTL, submit durable work, mutate lifecycle state, or create or
consume authority. In particular, this validation-only contract does not
authorize or consume r20 execution authority and is not new dialogue execution
or quality evidence.

The chain form requires at least two receipts starting at generation ordinal
zero. Every receipt repeats the fixed checkpoint, tokenizer, and authenticated
prompt record, binds the exact input token history assembled from all preceding
selections, and names the preceding terminal-evidence record. The caller
supplies the accepted terminal-evidence chain and a separately authorized,
unconsumed receipt-use authority lineage for every receipt. Ordinal gaps,
duplicates, reordered receipts, history discontinuities, stale parentage, and
any per-receipt Top-K or authority failure are rejected.

After all receipts validate, the bridge returns the ordered selected tokens,
the resulting complete token history, and a transcript decoded by the supplied
authenticated tokenizer. This remains read-only assembly from existing
receipts: it performs no runtime workload, lifecycle or controller action,
Model24 model/oracle/RTL execution, durable submission, or authority creation
or consumption.

## Systematic continuation first batch

The `official-model24-systematic-continuations` target regenerates a fixed
32-case English/Chinese prompt batch from the reviewed admissible
`showcasecontinuations15c` software/oracle baseline. The checked-in suite balances
continuation, chat, factual, commonsense, code, and short reasoning cases. Its
authenticated JSONL preserves all ordered rows, serialized prompts, input and
generated token IDs, decoded outputs, stop reasons, per-step ACE/PyTorch
agreement or mismatch, and 24-layer FP16 K/V parentage. A JSON summary, concise
Markdown report, and run log are bound by byte counts and SHA-256 hashes;
validation also rejects source, prompt-suite, checkpoint, tokenizer, and
accepted-binding staleness.

The recorded host wall times are diagnostic only. This bounded batch does not
establish broad dialogue quality, product latency, throughput, full-model RTL,
synthesis, PPA, or FPGA behavior. Unreviewed ancestry `486e5d848245` is
explicitly excluded from acceptance and claim-bearing evidence.

## Diagnostic software and RTL-simulation latency

`make model24-latency` downloads `tokenizer.json` and
`tokenizer_config.json` from the fixed official revision and rejects either
file unless its accepted SHA-256 matches. It then performs one fresh
four-token software dialogue continuation and, separately, one fresh
controller trace followed by 24 indexed two-token Verilator decoder
simulations. `build/model24_latency/measurement.json` records the software and
RTL-process wall times separately, while the RTL record also preserves the
abstract controller scheduler count and each decoder harness's total,
per-token, phase, and stall cycle counts. The adjacent manifest binds consumed
sources, commands, fixed assets, logs, controller evidence, simulator
binaries, and per-layer records by byte count and SHA-256.

Wall time uses `time.perf_counter` around each child process and is accompanied
by host affinity, load, CPU, Python, NumPy, PyTorch, and Verilator context.
The numbers are single-run diagnostic software/simulation evidence. Verilator
harness ticks are not hardware clock cycles, and this target makes no
synthesis, timing-closure, PPA, FPGA, hardware-latency, throughput, or
bottleneck claim.
