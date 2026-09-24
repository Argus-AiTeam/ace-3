# Decoder layer-0 preload and clear contract

The public decoder layer-0 token-engine interface has three ordered 896-element
preloads: activation (`load_kind=0`), input-normalization n1 (`load_kind=1`),
and post-attention-normalization n2 (`load_kind=2`). Each vector becomes valid
only after indices 0 through 895 are accepted in order.

Asynchronous reset invalidates all three preload-valid flags, K/V cache
validity, cache positions, and transient token state. A start after reset is
rejected until both immutable normalization vectors and an activation vector
have been completely preloaded.

Synchronous clear has a narrower lifecycle. It aborts in-flight token work and
invalidates activation, K/V cache, cache-position, and other transient state.
It retains validity only for completely preloaded immutable n1 and n2 vectors;
an incomplete normalization preload remains invalid. The next transaction
therefore reloads activation only.

Clear does not bypass readiness. Start remains guarded by complete activation,
n1, and n2 validity, a valid cache slot, and an exactly sequential position for
that slot. Reset is the operation that requires n1 and n2 to be preloaded again.

After both rotated K heads complete, the cache-write traversal starts at KV head
0, dimension 0 and covers both heads through dimension 63 before attention may
read. Retaining the final rotated-K head value would start at head 1, omit head
0, and convert the first query-head read into a cache-miss error.
