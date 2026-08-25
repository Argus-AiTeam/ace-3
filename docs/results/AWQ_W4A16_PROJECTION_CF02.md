# ACE-3 native-AWQ W4A16 full-input projection milestone

## Implemented boundary

This milestone adds one parameterized, synthesizable, sequential/tiled
projection engine:

- `ace3/rtl/ace3_awq_w4a16_projection_engine.sv`
- `ace3/rtl/ace3_q47_48_to_f16_rne.sv`

The engine instantiates the accepted G128 primitive without modifying it. Each
group contributes its exact signed 96-bit Q47.48 accumulator. The engine
sign-extends and sums those values in signed 102-bit Q53.48, then performs one
binary16 round-to-nearest-ties-to-even conversion after the final group. It
does not add already-rounded FP16 group outputs.

The 102-bit width is `96 + ceil(log2(38))`, where 38 is the maximum group count
for the authenticated 4,864-input shape. This exactly contains any sum of 38
values representable by the primitive contract. There is no internal wrap or
saturation; final binary16 overflow saturates to finite maximum.

## Authenticated geometry and indexing

The generator authenticates fixed
`Qwen/Qwen2.5-0.5B-Instruct-AWQ` revision
`db09cd27ead7fee40cdee309693cf83601b9c899`, including `model-api.json`
SHA256 `9a4a3beea2283031c91d0de501fcb1a8613f9b5f5d6039111eac421833d5a768`.
It authenticates `config.json` with SHA256
`bd20ae34a91eb38230b870d39f56677d1cda1e8b6688ad627e6efb6ca9f44090`.
It records hidden size 896, intermediate size 4,864, 14 attention heads, two KV
heads, and AWQ G128. The sealed tensor contract with SHA256
`b3754c03658534b79ddf8f667049e9122d631f84005fb219faf0e5e9de56e2aa`
independently records the module shapes. Its native-GEMM qzero correction is
bound by SHA256
`601e726d6a524d01bc48ef435831d9fe23cf9a99ad86e22c2382d5af74cded66`;
the initial contract alone has an obsolete ExLlama qzero statement. The
generator also re-probes the correcting AutoAWQ packing source. The one RTL
module elaborates these configurations:

| Modules | IN_FEATURES | OUT_FEATURES | Groups |
| --- | ---: | ---: | ---: |
| q/o | 896 | 896 | 7 |
| k/v | 896 | 128 | 7 |
| gate/up | 896 | 4864 | 7 |
| down | 4864 | 896 | 38 |

For output `o`, group `g`, and input `i`, the stream supplies
`qweight[i][o/8]`, `qzeros[g][o/8]`, and `scales[g][o]`. The logical lane is
`o mod 8`, with accepted native physical nibble order
`[0, 4, 1, 5, 2, 6, 3, 7]`.

## Numerical evidence

Seed `0xACE3CF02` generates seven transactions, 14 complete outputs, 98 group
metadata records, and 12,544 accepted input pairs:

- one eight-output layer-0 `q_proj` tile using authenticated official qweight,
  qzeros, and scales with deterministic generated FP16 activations, channels 4
  through 11, spanning packed output words 0 and 1 and logical lanes 0 through
  7;
- six directed outputs covering cross-group round-once cancellation, positive
  and negative saturation, minimum subnormal, zero, and invalid activation.

The cancellation case has group accumulators `2^24` and `-2^23`, a final exact
sum of `2^23`, and final binary16 zero. Rounding each group first would instead
produce one minimum subnormal, so this vector distinguishes the implemented
round-once contract.

All simulator-consumed files are bound by
`ace3/contracts/awq_w4a16_projection_vector_bindings.json`:

| Artifact | SHA256 |
| --- | --- |
| `manifest.json` | `0619f42d47cb6617da93a857dbbd64828842ae0eb9fc6ef6a726d6b812255331` |
| `transactions.hex` | `7306f9d79f20ff960b3df21121676904368b0a62d94db9f6ced14b948b343bfb` |
| `expected.hex` | `a6ec52b131112fe416b748fbe06c6d506a4fc9a99d695792258d56b648620a4f` |
| `meta.hex` | `5ef481ccbcfd4277717c3c5911552027d56c9f94a90cc96e2970fc3a242fe02d` |
| `pairs.hex` | `e82464ea4f5faa1cd3954aeeed746eeb1c68837b43c74e4afcf1cd8ca1ca6527` |
| `projection_params.svh` | `2294164a2688a570cc082b1fbfc79f374d6386cdf60ff79550c7d535f015d52e` |

## Reproduction and results

Executed from the repository root with Python 3.13.5, GNU Make 4.3, Icarus
Verilog 11.0, and Verilator 4.038:

```sh
make clean
make OFFICIAL_TENSOR_DIR=/path/to/official_tensors test
printf 'corrupt\n' >> build/projection_vectors/pairs.hex
make OFFICIAL_TENSOR_DIR=/path/to/official_tensors test
```

The second non-clean run regenerates both primitive and projection vectors,
revalidates every bound artifact, recompiles all RTL tests, rebuilds both
Verilator harnesses, and reruns every simulation before aggregate success.
The injected projection-pair corruption had SHA256
`8f9f38ff35f7410b7ee5e87b0ba08523688126e1c5c791cf86eeb39c4e358e2f`;
the rerun restored the bound
`e82464ea4f5faa1cd3954aeeed746eeb1c68837b43c74e4afcf1cd8ca1ca6527`.

```text
PROJECTION_ORACLE_PASS checks=9 group_acc_bits=96 cross_acc_bits=102 max_groups=38 round_once=pass
PROJECTION_VECTOR_PASS seed=0xace3cf02 transactions=7 outputs=14 official_outputs=8 groups=98 pairs=12544 revision=db09cd27ead7fee40cdee309693cf83601b9c899 config_sha256=pass official_tensor_sha256=pass
PROJECTION_JSON_VALIDATION_PASS json_files=3 serialized_artifacts=6 sha256=pass transactions=7 outputs=14 official_outputs=8 groups=98 pairs=12544 cross_acc_bits=102 round_once=pass stream_oracle=pass nonvacuity=pass
PROJECTION_TAMPER_REJECTION_PASS artifact=pairs.hex validator_exit=nonzero reason=sha256_mismatch originals=untouched
PROJECTION_GEOMETRY_PASS simulators=iverilog,verilator geometries=4 q_o=896x896 k_v=896x128 gate_up=896x4864 down=4864x896
AWQ_W4A16_PROJECTION_PASS transactions=7 outputs=14 official_outputs=8 groups=98 pairs=12544 ulp_bound=0 first_output_compute_cycles=910 next_output_compute_cycles=910 input_stall_cycles=54 output_backpressure_cycles=98 reset=pass clear=pass protocol=pass four_state_controls=64 four_state_data=3
AWQ_W4A16_PROJECTION_4864_CYCLE_PASS in_features=4864 groups=38 pairs=4864 compute_cycles=4940 result=zero backpressure=pass
AWQ_W4A16_PROJECTION_VERILATOR_PASS transactions=7 outputs=14 official_outputs=8 groups=98 pairs=12544 ulp_bound=0 first_output_compute_cycles=910 next_output_compute_cycles=910 input_stall_cycles=54 output_backpressure_cycles=56 reset=pass clear=pass protocol=pass four_state=unsupported
STANDALONE_VALIDATION_PASS semantic_checks=fresh primitive=pass projection=pass serialized_sha256=pass tamper_rejection=pass iverilog=pass protocol_4state=pass verilator=pass geometry_parameters=pass hygiene=pass
```

The original primitive oracle, frozen manifest, provenance bindings, RTL, and
tests remain regression-covered and unchanged.

## Cycle boundary

The reported counts are clock cycles observed in RTL simulation:

- 896 inputs: 7 groups x (1 metadata + 128 pairs + 1 group collection) =
  910 cycles from accepted start or previous output acceptance to `out_valid`;
- 4,864 inputs: 38 groups x 130 = 4,940 cycles to `out_valid`;
- accepting an asserted output consumes one additional cycle;
- backpressure cycles are excluded from compute latency and reported
  separately.

These counts apply only to the sequential single-lane engine. They are not
synthesized latency, frequency, throughput, area, power, PPA, or FPGA results.

## Limitations

Official-tensor numerical evidence is limited to eight layer-0 `q_proj`
channels because those are the authenticated tensor samples available locally.
Their activations are seeded test stimulus, not captured model activations, so
the outputs are projection checks rather than model-inference results. The
k/v/o/gate/up/down parameter sets have Icarus elaboration and Verilator lint
coverage, while only a synthetic-zero 4,864-input path is dynamically
simulated. Verilator is two-state; bounded X/Z protocol probes run only under
Icarus. No formal verification, tensor SRAM/DMA, parallel lane array, decoder,
full model, readable dialogue, synthesis, PPA, or FPGA claim is made.
