---
title: Model24 r14 payload-directory lifecycle boundary
description: Exclusive creation, durability, and end-to-end binding of the Model24 controller simulation and RTL cascade output directories.
---

# Model24 r14 payload-directory lifecycle boundary

The r14 lifecycle owns creation of the canonical output root and its exact
`controller-simulation` and `rtl-cascade` children. It validates lexical
ancestry before mutation, rejects every pre-existing filesystem entry
including symlinks, creates the root and both children exclusively, fsyncs the
parent and created directories, and strictly resolves both child paths before
the payload can be invoked. The output root may contain no foreign entry at
that boundary.

The package manifest, review request, Manager authority schema, payload argv,
launch terminal, and post-terminal validation all bind the two child paths
separately from the output root. Terminal validation requires both children to
remain real directories under the canonical root and permits only the
exclusive `launch-terminal.json` alongside them.

The inert controller-entry probe uses the accepted production controller
entry point and the production lifecycle order while replacing only the
Model24 execution body. It requires strict resolution of both child paths and
rejects absent, substituted, pre-existing, symlinked, wrong-parent, partial,
or foreign-entry layouts. Package preparation leaves the canonical review,
authority, consumed marker, terminal evidence, durable-runner receipt and
logs, output root, and both child paths absent.
