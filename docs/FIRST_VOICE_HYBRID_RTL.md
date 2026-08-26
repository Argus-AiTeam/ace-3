# First Voice Hybrid RTL

`ace3/model/model24_first_voice_hybrid.py` is a bounded Hybrid RTL driver for
the fixed `Qwen/Qwen2.5-0.5B-Instruct-AWQ` revision
`db09cd27ead7fee40cdee309693cf83601b9c899`. Its machine-readable contract is
`ace3/contracts/model24_first_voice_hybrid.json`.

The accepted indexed decoder has two cache slots and exactly 128 causal
positions. The authenticated chat template is 25 tokens, so the default four
new-token request needs at most 28 represented positions and fits. A request
that could require more than 128 positions fails with
`rtl_context_capacity_exceeded`; it does not run a software fallback or emit an
RTL claim.

Each represented prompt or fed-back generated token traverses all 24 compiled
Verilator decoders. Every layer retains an immutable `--savable` state image
and canonical exact-schema envelope for each represented position. Before
restore, the driver recursively authenticates every retained predecessor state
and envelope plus each transition's retained input activation and output
hidden hash, compiled binary, checkpoint, layer, cache slot, and next position.
Extra or missing fields and noncanonical envelope bytes are rejected. State is
committed only after a natural token terminal. Tokenization, chat serialization, embedding lookup,
final RMSNorm, tied `lm_head`, full-vocabulary greedy selection, and decoding
are host operations. PyTorch CPU float64 dequantized-AWQ execution provides the
independent causal reference.

The traversal keeps only one layer's expanded tensor vectors live at a time.
Compact authenticated tensor manifests persist, while the reusable vector
workspace is removed after every layer transaction. Each raw RTL trace is
hashed, deterministically gzip-compressed, restored and hash-checked, then
replaced by its archive. This preserves per-token/per-layer lineage without
requiring the 422,896,896-byte all-layer expanded-vector footprint.

Compiled RTL uses the same bounded layout. `model24-first-voice-layer-compile`
builds one selected compile-time layer in the reusable
`build/model24_first_voice_hybrid/compact_mdir`, copies and strips only the
self-contained executable into `compiled/layerN/bin/`, writes a canonical
manifest binding the layer, contract, source hashes, exact Verilator
configuration hash, and executable hash, removes the Mdir, and then runs the
retained executable's RTL load-handshake self-test. Success and failure both
remove the reusable Mdir.

Token-0 means the first generated selection. It is selected from the final
prompt token's layer-23 output, so selection itself is not an RTL input. If
generation continues, Token-0 is fed back through every RTL layer at the next
position. The final selection is reported but is not described as represented
unless another decision causes that feedback.

Focused validation:

```sh
make model24-first-voice-hybrid-tests
make model24-first-voice-compact-builder-tests
make model24-first-voice-layer-compile FIRST_VOICE_RTL_LAYER_INDEX=3
```

The target compiles the existing indexed decoder with `--savable`, serializes a
live partially executing RTL model, restores it into a fresh model, and compares
1,024 subsequent cycles. Python tests cover capacity, negative checkpoint,
stale-position and state hashes, every parent/activation/output lineage field,
missing or wrong predecessors, and exact canonical envelope schema. Full execution additionally requires
the authenticated checkpoint/tokenizer and all 24 compiled indexed binaries:

The compact-builder test target proves success/failure Mdir cleanup and rejects
layer, source, configuration, manifest, and stripped-binary tampering. The
selected-layer command is the bounded real-build check. Building all 24 layers
is reserved for a later disk-authorized mission.

```sh
make model24-first-voice-hybrid
```

Runtime evidence stays under ignored
`build/model24_first_voice_hybrid/`. No synthesis, FPGA, PPA, latency, or
throughput claim is made.
