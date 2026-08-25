# Official Model24 systematic continuations

This directory publishes the independently reviewed, authenticated 32-case
Model24 continuation result set. The suite completed all 32 ordered English and
Chinese cases and 122 generated steps with zero execution failures. It preserves
25 numerical tolerance mismatches across 13 cases; no mismatch is hidden or
reclassified as a passing comparison.

`batch.jsonl` is authoritative. It contains every serialized prompt, raw output,
generated token ID, stop reason, Primary/PyTorch decision, FP16 KV parent/child
lineage record, numerical failure, and content hash. `summary.json` provides the
complete aggregate and source/checkpoint/tokenizer bindings, `manifest.json`
binds the tracked artifacts, `REPORT.md` is the readable all-case projection,
and `run.log` records every case status.

| File | SHA-256 |
| --- | --- |
| `REPORT.md` | `52adab8e83a3a88a7b11d4af3bcc974a045a3454365c0fd6856c1693a7ff5202` |
| `batch.jsonl` | `fa4b4c43670759067ec18691ce450f101ce66123f70ad63d5670c98c3e9c9595` |
| `manifest.json` | `d8f5446644f215f71b6d94ce944e1d8ae789628c344620a662ca4b4d461ff742` |
| `run.log` | `211f0af5caa29f2fcdce87b71e350593d967e8867c879e9f2bffdf125f62fee8` |
| `summary.json` | `6f3472726ef5818f2a42102ec9149b38528eae93d52dd1826873b4bb64835ab6` |

Model weights and tokenizer files are not distributed. Regeneration requires
the official `Qwen/Qwen2.5-0.5B-Instruct-AWQ` checkpoint and tokenizer
authenticated by SHA-256 in `summary.json`:

```sh
make OFFICIAL_MODEL24_CHECKPOINT=/path/to/model.safetensors \
  OFFICIAL_MODEL24_TOKENIZER_DIR=/path/to/tokenizer \
  official-model24-systematic-continuations
```

The result is software/oracle evidence only. It makes no RTL, synthesis, PPA,
FPGA, product-latency, throughput, or broad model-quality claim.
