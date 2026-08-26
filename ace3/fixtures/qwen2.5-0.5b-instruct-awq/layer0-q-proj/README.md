# Authenticated layer-0 Q projection fixture

This directory contains the six authenticated inputs consumed by the ACE-3
standalone validation default. The model metadata and native-AWQ samples are
byte-identical copies bound to `Qwen/Qwen2.5-0.5B-Instruct-AWQ` revision
`db09cd27ead7fee40cdee309693cf83601b9c899`; the packing reference is bound to
AutoAWQ v0.2.9.

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `config.json` | 837 | `bd20ae34a91eb38230b870d39f56677d1cda1e8b6688ad627e6efb6ca9f44090` |
| `model-api.json` | 5387 | `9a4a3beea2283031c91d0de501fcb1a8613f9b5f5d6039111eac421833d5a768` |
| `autoawq-v0.2.9-packing_utils.py` | 3283 | `65eab3eabe3f55e300ffbab5feac59c49322d985f42dcda4e2288859fb9a4abe` |
| `sample-model_layers_0_self_attn_q_proj-qweight.bin` | 401408 | `db4770023698611ff0115d220590fdb8232fbe5dcbd22fbe80e0bcdc838caf87` |
| `sample-model_layers_0_self_attn_q_proj-qzeros.bin` | 3136 | `3cf7cd5712dd7523db3c7dd47c2b1d582e19545036f75b95ff0331c1fc0c596c` |
| `sample-model_layers_0_self_attn_q_proj-scales.bin` | 12544 | `687adc7d7bcd6e45a065f914dd27a1284b7e48260491bb0d26ae1e13b78ac321` |

The primitive generator consumes the three tensor samples. The projection
generator consumes all six files, while the FP16, QKV/RoPE/cache, and attention
generators consume the config, model metadata, and scale sample. Each generator
authenticates its inputs before use and writes generated vectors only under
ignored `build/` paths. This fixture is a bounded validation input, not a
complete model checkpoint.
