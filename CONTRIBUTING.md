# Contributing to ACE-3

Contributions are welcome through GitHub issues and pull requests.

## Development setup

Start with [Getting started](docs/GETTING_STARTED.md). Keep generated vectors,
simulator output, model assets, and local state outside the tracked source tree.

## Change requirements

- Keep synthesizable RTL, testbenches, software oracles, and generated evidence
  in their existing directories.
- Add or update a precision/interface contract with every arithmetic or
  protocol change.
- Compare RTL against an independent oracle, not a transcription of DUT logic.
- Preserve exact model revision and tensor identities for model-bound vectors.
- Add focused tests for reset, backpressure, boundaries, and invalid inputs.
- Report unsupported behavior explicitly; do not add silent floating-point or
  software fallback.
- Do not claim synthesis, FPGA execution, performance, or chip completion
  without the corresponding reproducible artifacts.

## Pull requests

Each pull request should state:

1. the implemented execution boundary;
2. the contract changed or preserved;
3. the smallest reproducible validation command;
4. the evidence produced;
5. known unsupported cases and claim boundaries.

Run the narrowest relevant target first, then the aggregate regression when
official fixtures are available:

```sh
make model24-smoke
make OFFICIAL_TENSOR_DIR=/path/to/official_tensors test
```

Generated model files, credentials, local paths, agent/session state, and
proprietary tool outputs must not be committed.

## Reporting problems

Use a GitHub issue for reproducible bugs and documentation gaps. For security
problems, follow [SECURITY.md](SECURITY.md) instead of opening a public issue.
