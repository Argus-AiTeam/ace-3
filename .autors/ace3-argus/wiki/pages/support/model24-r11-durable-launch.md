---
title: Model24 r11 durable-launch review boundary
description: Fail-closed package, review, authority, and durable-runner constraints for the Model24 continuation launch.
---

# Model24 r11 durable-launch review boundary

The r11 launch boundary accepts a package only when its complete package file
set and copied source tree match immutable manifests, its checkpoint matches
the bound size and SHA-256, and its embedded ba953 ancestry matches the recorded
manifest, seal, and review hashes. The embedded ba953 review is nonconforming
rejected history only: it is not a trust root, package acceptance, authorization,
or execution authority, and validation has no external parent-review dependency.
Package validation also requires absent
review, authority, authority-consumption, output, and durable-receipt
namespaces.

Exactly one future independent L2 reviewer may exclusively create the
canonical `review.json`. The receipt must identify reviewer roles and a
reviewer identity distinct from the preparer, deny preparation participation,
report zero execution, and bind the package manifest, package seal, review
request, launcher, validator, and launch contract hashes. An ACCEPT receipt is
not execution authority. The receipt also carries ordered PASS/FAIL results for
the required package, review, launch, and negative fixtures; the exact clean
zero-state observation; and the canonical statement that ba953 ancestry is
non-authoritative. ACCEPT requires every inert result to pass, while REJECT
must identify at least one failed result.

Launch validation requires a separate read-only Manager authority that exactly
binds the accepted review, all six reviewed hashes, canonical launch command
and working directory, task receipt, task-local stdout and stderr logs, output
namespace, cardinality one, and disabled retry, replay, resume, and watcher
semantics. The launcher validates before exclusively consuming authority or
creating output. Only a direct durable-runner receipt in a recent `starting`
state or live `running` state is accepted, and the payload must use the fresh
Model24 path. The durable timeout is five hours.
