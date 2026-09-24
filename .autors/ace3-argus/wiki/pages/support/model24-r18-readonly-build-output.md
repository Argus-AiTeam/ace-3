---
title: Model24 r18 read-only source build-output boundary
description: External Verilator build-root, source immutability, and failed-r17 ancestry constraints for Model24 r18.
---

# Model24 r18 read-only source build-output boundary

The r18 package preserves the r17 binding closure, controller materialization,
lifecycle, terminal contracts, positives, and negatives. It adds exactly one
authenticated source overlay:
`ace3/model/controller_model24_rtl_cascade.py`. The production per-layer Make
command now supplies
`MODEL24_RTL_CASCADE_DIR=<bound payload output_dir>` and rejects an omitted,
different, outside, or symlinked build root. Verilator output is therefore
rooted at `output_dir/compiled/layerN`, never beneath sealed source.

The post-seal production-entry probe keeps the materialized source recursively
read-only, compiles layers 0 and 23 through the controller's production command
constructor into a fresh external root, and authenticates fresh regular
executables without executing them. Complete before and after source records
must match, `source/build` must remain absent, and source-write, stale-binary,
and source-mutation probes must fail closed.

The r17 invocation is preserved as lawful consumed failed ancestry. Its
terminal manifest and failing layer-0 compile log are byte-bound, and its
review, authority, execution inputs, and all predecessor identities are
non-reusable. Preparation leaves the fresh r18 review, authority, canonical
output, terminal, receipt, and log namespaces absent.
