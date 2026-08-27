# Contributing to ACE-3

ACE-3 is an evidence-first research RTL project. Changes are reviewed against
their declared execution boundary, not only whether code compiles.

## Before changing code

1. Identify the affected arithmetic, interface, runtime, or evidence contract.
2. Check [current status](docs/STATUS.md) and
   [RTL traceability](design/RTL_TRACEABILITY.md).
3. Keep generated vectors, model assets, simulator objects, traces, and local
   agent state outside source control.
4. Use an independent oracle; do not reproduce DUT logic as the expected
   result.

## Change requirements

- Keep synthesizable RTL, testbenches, software oracles, and generated evidence
  in separate directories.
- Add or update a machine-readable contract for arithmetic or interface changes.
- Preserve exact model revision and tensor identities for model-bound vectors.
- Regenerate and authenticate serialized simulator inputs before execution.
- Include tamper or negative-path coverage for new trust boundaries.
- Report unsupported behavior explicitly; do not add silent software fallback.
- Do not claim synthesis, PPA, FPGA, latency, throughput, or hardware completion
  without the corresponding artifacts.

## Pull request checklist

Every pull request should state:

1. the implemented execution boundary;
2. the contract changed or preserved;
3. the smallest reproducible validation command;
4. the independent oracle or comparison source;
5. the evidence produced;
6. known limitations and explicit non-claims;
7. whether model assets or external tools are required.

Use the repository pull request template. Keep changes focused: documentation,
RTL, oracle, and evidence changes may be reviewed together when they form one
contract-complete milestone, but unrelated experimental work should remain
separate.

## Public release hygiene

Never commit:

- model checkpoints or extracted weights unless redistribution is explicitly
  permitted and intended;
- credentials, tokens, local paths, machine state, or conversation artifacts;
- generated build directories, raw simulator traces, or temporary worktrees;
- evidence that has not completed its declared validation and review.

ACE-3 source is Apache-2.0. Upstream model and tool assets retain their own
licenses.
