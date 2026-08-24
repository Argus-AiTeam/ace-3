# Contracts

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
