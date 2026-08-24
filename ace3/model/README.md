# Reference models

Bit-level arithmetic oracles and vector generators live here. These models are
verification references, not accelerator implementations or performance paths.

`generate_vectors.py` requires explicit `--official-tensor-dir` and
`--output-dir` arguments. The root `Makefile` binds the official read-only
sample location and places all generated files under `build/vectors/`.
`validate_vectors.py` authenticates every serialized output against the
source-controlled standalone binding contract before any simulator runs.
