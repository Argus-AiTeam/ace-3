---
title: Decoder layer-0 qzeros address boundary
description: Native-AWQ qzeros geometry, serialized layout, request qualification, and address-width limits.
---

# Decoder layer-0 qzeros address boundary

Layer-0 projections use row-major packed INT4 qzeros words with linear address
`group * words_per_group + word`. Q and O each require `[7,112]` (`0..783`);
K and V each require `[7,16]` (`0..111`); gate and up each require `[7,608]`
(`0..4255`); down requires `[38,112]` (`0..4255`). Each addressed element maps
to one little-endian 32-bit word in the authenticated serialized tensor.

The projection request exposes a 6-bit group, 10-bit word, and 3-bit projection
kind. The checked mapper uses a 16-bit linear address, so the complete required
domain through 4255 is representable without truncation or wrapping. Loaders
must dereference qzeros, scales, qweights, bias, and RoPE data only on their
corresponding valid-and-ready handshake. A live request outside its selected
projection geometry is an error; an idle retained tuple is not a tensor access.

This boundary establishes address and serialization behavior in focused
Icarus and Verilator simulation only. It does not establish integrated decoder
layer-0 execution, full-model behavior, synthesis, PPA, or FPGA results.
