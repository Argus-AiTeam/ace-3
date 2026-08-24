# Contracts

Implemented precision, packing, rounding, streaming, and reset contracts live
here. A contract is descriptive evidence and does not by itself certify RTL.

`awq_w4a16_g128_vectors_manifest.json` is the immutable historical accepted
case manifest. `awq_w4a16_g128_standalone_vector_bindings.json` is a separate
standalone provenance contract that binds every serialized simulator input by
SHA256 without rewriting the historical manifest.
