# ACE-3 continuation handoff

Source snapshot: **2026-09-24T16:50:59Z**; live state observed at
**2026-09-24T16:58:55Z**. Live state may have advanced. The canonical
operator guide is [HANDOFF.zh-CN.md](HANDOFF.zh-CN.md).

The equal mirrors are <https://github.com/aHappend/ace-3> and
<https://github.com/Argus-AiTeam/ace-3>, branch `main`. The dated source ref is
`refs/tags/checkpoint-2026-09-24-source` at
`2b4eaeeec8280a2398ef9311fb606e28a2b5e850`.

On the original host, use `python tools/argus_continuation.py inspect` before
acting. Do not restart a live daemon or duplicate an active mission. Continue
through the normal Manager; verify issuer, normal Reviewer, and native terminal
receipts. `04c9ee209234` is an accepted zero-science operational contract, not
consumer/science qualification; `609b6cad06bc` is terminal FAILED. For disk
loss, this repository restores curated source and documentation only; see
[HOST_ONLY_DEPENDENCIES.md](HOST_ONLY_DEPENDENCIES.md).
