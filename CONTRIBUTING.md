# Contributing

ACE-3 is currently a private research and development project.

## Change requirements

- Keep synthesizable RTL, testbenches, software oracles, and generated evidence
  in separate directories.
- Add or update a precision/interface contract with every arithmetic change.
- Compare RTL against an independent oracle, not a copy of DUT logic.
- Preserve exact model revision and tensor identities for model-bound vectors.
- Report unsupported behavior explicitly; do not add silent floating-point or
  software fallback.
- Do not claim synthesis, FPGA execution, performance, or chip completion
  without corresponding artifacts.

## Pull requests

Each pull request should state:

1. the implemented execution boundary;
2. the contract changed or preserved;
3. the smallest reproducible validation command;
4. the evidence produced;
5. known unsupported cases and claim boundaries.

Generated model files, credentials, local state, and proprietary tool outputs
must not be committed.
