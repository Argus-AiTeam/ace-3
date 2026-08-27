# Contracts

`generated_token_feedback.json` binds the accepted RTL rank-zero token to its
tokenizer metadata, exact tied embedding row, and next-position state commitment.

`final_rmsnorm.json` freezes the parameter-free `ace3_final_rmsnorm` public top,
the fixed 896-element transaction, checkpoint provenance, arithmetic semantics,
and bounded verification counts.

Implemented precision, packing, rounding, streaming, and reset contracts live
here. A contract is descriptive evidence and does not by itself certify RTL.

`awq_w4a16_g128_vectors_manifest.json` is the immutable historical accepted
case manifest. `awq_w4a16_g128_standalone_vector_bindings.json` is a separate
standalone provenance contract that binds every serialized simulator input by
SHA256 without rewriting the historical manifest.

`awq_w4a16_projection_engine.json` defines the additive full-input engine,
fixed-revision authenticated Qwen geometries, corrected native-GEMM indexing,
the 102-bit cross-group accumulator, and cycle contract.
`awq_w4a16_projection_vector_bindings.json` separately binds all complete-input
projection simulator streams.

`attention_block.json` freezes the three attention interfaces, GQA/causal
policy, fixed-point widths, rounding, approximation bound, and error behavior.
`attention_vector_bindings.json` authenticates every generated simulator input.

`decoder_layer0_token_engine.json` limits the composed engine contract to one
fixed-checkpoint decoder layer and two sequential token positions.
`decoder_layer0_vector_bindings.json` binds the complete stage trace, final
outputs, coefficients, and serialized layer-0 tensors used by both simulators.
The accepted integrated claim requires a complete Verilator comparison and
focused Icarus four-state boundaries. Full-trace Icarus execution remains an
explicit runtime limitation rather than a passing numerical claim.

`model24_layer_controller.json` defines the arithmetic-free 24-layer launch,
completion, checkpoint, terminal, and fail-closed sequencing boundary. It does
not extend the numerical decoder claim beyond the separately verified,
layer-indexed engines.
`position1_model24_causal_traversal.json` pins the operator-designated ordered
set of 24 post-position-0 Verilator states, the historical parent binaries, the
corrected child binaries, and the sealed token-2114 feedback evidence used for
a position-1 restore traversal.
Its independent checkpoint oracle is seeded from each matching authenticated
position-0 FP16 K/V trace rather than from an empty cache.
