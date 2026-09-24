---
title: Model24 r13 submitter and terminal-evidence boundary
description: Durable-runner root separation, pre-launch validation, and permanent terminal evidence for the Model24 r13 lifecycle.
---

# Model24 r13 submitter and terminal-evidence boundary

The direct durable runner has two distinct filesystem roots. Its submitter
process writes `.argus_subagents` relative to the submitter current working
directory, while `submit --cwd` selects the child payload current working
directory. The r13 contract therefore binds a permanent submitter root under
`/home/argustest`, a registry root derived from that submitter root, and the
sealed candidate source as a separate child cwd.

Pre-submit validation requires the canonical terminal root and all execution
artifacts to be absent. The submission wrapper then exclusively creates the
terminal root and invokes the installed direct runner from that directory.
Child launch validation authenticates the package, independent review,
Manager authority, command, roots, and zero execution state without requiring
a final runner receipt. Receipt, stdout, stderr, and exit-sidecar checks are
post-terminal accountability checks because the runner finalizes those
artifacts only after the child exits.

Terminal evidence remains in the permanent submitter root. The post-terminal
sealer derives the receipt and logs from the bound registry root, validates
one payload invocation when a launch terminal exists, and exclusively creates
`manifest.json`. The preparation boundary leaves the review, authority,
consumed marker, terminal root, runner receipt and logs, Model24 output, and
payload execution absent.
