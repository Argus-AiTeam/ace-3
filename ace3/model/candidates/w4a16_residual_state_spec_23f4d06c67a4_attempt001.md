# Residual-state recurrence: bounded rejection

**Engineer disposition: `retain_contract_and_stop`. Independent normal Host
Reviewer assessment is pending.** This document explains the accompanying
`ace3/contracts/candidates/w4a16_residual_state_recurrence_23f4d06c67a4_attempt001.json`.
It defines no executable candidate, new precision, adopted policy or launch.

## Evidence and the actual boundary

The genuine reviews `dd6ba8da8fff/round-0001.json` and
`1a3ea71dc640/round-0001.json` retain the actual P0/L0-L8 accepted prefix and
stop the L9-forward cone. Their absolute paths and the original artifact
paths are in the JSON. Historical pending-review prose does not negate
those later reviews; those reviews do not approve this new specification.

The retained parent execution is a **mandatory global numerical FAIL**:
18 local gates pass over 22428 coordinates, while S18 fails one of 896.
At the diagnostic coordinate, actual H/O/D are `662c/b52b/3b1d`.
Their exact sum is `6473999/4096`, correctly rounding to `662d` (1581).
The original binary64 reference is `0x1.8b0476aae57f7p+10`;
the retained exact excess is `1892290766857/2199023255552 > 1/8`.
These numbers describe the failure; they are not recurrence constants.

The reviewed retained decomposition separately records inherited hidden
error `2187923984043/4398046511104`, increment difference
`-2324591441/2199023255552`, and final rounding delta `1777/4096`.
This is a coordinate-specific decomposition, not unique upstream causality
or a performance attribution. Correcting all relevant same-input primitive
deviations through each complete canonical suffix still produces `662d`.
Those controls are not rerun and do not prove all W4A16 implementations fail.

## The current general operator is already compensated within a layer

Let `h`, `o`, `d` denote the real values of authenticated FP16 hidden,
attention-output and down-projection operands for any coordinate. Let
`Q` be the frozen finite binary16 round-to-nearest, ties-to-even operation.
The accepted source specifies:

```
r = Q(h + o)                 # S12, actual FP16 norm2 input
d = existing_MLP(r)          # S13-S17, existing FP16 boundaries
h_next = Q(h + o + d)        # S18, exact sum then one rounding
```

The comments above identify stages, not a candidate implementation.
The frozen final residual core uses exact signed 43-bit Q24 accumulation.
Increasing its transient accumulation precision cannot recover hidden
information lost before its FP16 inputs.

One can write `e = (h + o) - r` and compute `Q(r + e + d)`.
With exact intermediate arithmetic this is identically `Q(h + o + d)`.
It is an alternative expression for the existing single-round operator,
not an unimplemented repair. Rounding `e` or either partial sum changes
that equivalence and needs a separate arithmetic justification.
Feeding unrounded `h+o` to norm2 instead changes the required S13 operand.

## Why inter-layer compensation is not a compatible implementation fix

Consider a general mathematical residual carry `c`, initially zero at
the authenticated FP16 activation root of each token. A full-residual
proposal would use

```
t = h + c + o
r = Q(t)
d = existing_MLP(r)
u = t + d
h_next = Q(u)
c_next = u - h_next
```

A final-only proposal keeps S12/MLP unchanged and uses
`u=h+c+o+d` only at S18. Neither proposal consults the independent
reference. Even so, neither is the current admitted recurrence.

If the carry is discarded at each layer boundary, the final-only proposal
is the existing operator. In particular, setting it to zero at L9 leaves
the retained failing rounding unchanged. If the carry survives, the
effective residual activation becomes `(h,c)`, not the declared single
FP16 hidden value. Calling `c` internal, storing it in another FP16 tensor,
reconstructing it in the host, or leaving K/V in FP16 does not preserve that
cross-layer data contract. This is different from the already permitted
wide *transient* accumulator inside a single FP16-output operator.

A rounded hidden value alone does not determine its lost information:
the exact numbers `1 + 2^-13` and `1 - 2^-13` both round to FP16 1,
but have opposite rounding residues. Retained actual operands may allow
a diagnostic reconstruction of a particular local residue. That does not
make the old output interface a carry producer, prove a compensated-prefix
history, or authorize feeding an extra operand into a new execution.
An old simulator snapshot can contain scratch arrays without those arrays
constituting an admitted portable residual-state ABI.

The carry would track only rounding in the candidate's residual additions.
It is not `original_global_hidden - actual_hidden`; nonlinear, projection
and cache-trajectory differences remain separate. It does not guarantee
global binary64 accuracy or readable dialogue.

## Affected cone, not permission to replay it

No affected cone is executed or invalidated by this rejection. The current
accepted outputs keep their original scoped acceptance.

For the hypothetical root-initialized carry, the first new state would be
produced at L0/P0/S18. L0 numerical outputs may remain identical. The first
possible numerical use is L1/P0/S12 for full-residual compensation, or
L1/P0/S18 for final-only compensation. Supplying `h+c` to nonlinear operators
would instead first potentially change L1/P0/S0. Bypassing the FP16 S12
boundary can already change L0/P0/S13. These are different proposals, not
interchangeable cone descriptions.

Follow changed hidden values into subsequent layers and their K/V producers,
then every reachable later-position computation. Same-layer P0 K/V produced
before a residual-only change is not automatically changed, but its reuse
needs actual arithmetic and state-ABI compatibility evidence. A byte-identical
hidden vector proves neither hidden carry identity nor snapshot compatibility.
The accepted L8 result exposes no admitted nonzero initial carry for L9.

Reuse unchanged, authenticated FP16 outputs and compatible state wherever
the independently justified cone permits. Do not replay accepted ancestors
just to label them again. Do not splice the earlier token-358 tail, the
Q-only fixed-history cone, this P0 prefix or the failed L9 snapshot. Arithmetic
lineage identifies operators and operands; cache/state lineage identifies
own-layer historical producers, positions, validity and saved-machine ABI.
Neither lineage can stand in for the other.

## Interface, rounding and lifetime

The exact public signature is frozen by reference to the retained
continuation `freeze.json#/public_contract`, also present in its manifest.
The module is `ace3_decoder_layer0_token_engine`, with integer parameters
`LAYER_INDEX=0` and `ACCURATE_SILU=(LAYER_INDEX>=3)`; retained L9 uses 9 and 1.
There is no new module alias, carry port, overloaded load kind or hidden
parameter. Load/trace/final FP16 words stay 16-bit. The exact full signature,
not this summary, controls every width and handshake.

Keep ties-to-even, gradual underflow, signed-zero behavior and explicit
nonfinite/overflow/X/Z rejection. At S18 exact zero is negative only for
three negative-zero inputs. The invalid core's zero placeholder is not a
valid result. Rising-edge valid/ready transfers and backpressure must not
duplicate consumption or leak a stale residual into another transaction.

Asynchronous active-low reset and synchronous clear invalidate loads,
contexts, pending outputs and K/V validity; invalid RAM need not be erased.
Reset is not restoration. P0 must start with empty own-layer prior K/V.
Later positions require a compatible, authenticated own-layer saved context.
The retained host uses generated Verilator save/restore state, not a public
RTL restore port. Port compatibility alone cannot establish generated
layout, live-state or arithmetic-history compatibility. The admitted check
of L9's idle state/128 valid entries did not restore/evaluate it and did not
waive its numerical FAIL.

A hypothetical carry must have explicit ownership, valid root initialization
and exact save/restore lifetime. No carry width or migration is assigned here:
there is no compatible extension to adopt under this frozen contract.
Invented-zero restores and reuse of opaque scratch fields are rejected.

## Review boundary

The JSON enumerates the forbidden actions and review prerequisites before
any software candidate, RTL edit or execution. They include independent
mathematical/oracle review, canonical operand/control authentication,
unchanged local and original-global gates, affected-cone justification and
state-ABI analysis. A state/arithmetic extension additionally needs delegated
parent technical disposition, not another per-layer execution permission.

Future regressions must cover general rounding/finite cases, reset/clear,
backpressure, restoration, foreign-state rejection, all applicable numerical
coordinates and legacy behavior. They are requirements, **not results**.
This turn runs only one structural/retained-record consistency check; no
candidate arithmetic, simulator, formal, timing or hardware measurement.
The current option remains `retain_contract_and_stop` pending normal review.
