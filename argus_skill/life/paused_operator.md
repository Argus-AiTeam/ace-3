# Isolated paused-operator resolution

`paused_operator.py` is a Host-invoked candidate, not an installed daemon route.
Load this file explicitly with `importlib.util.spec_from_file_location`, register
the resulting module in `sys.modules`, then execute its loader. Native Argus
dependencies must already be importable. Do not replace the resident package or
call the generic answer handler.

`resolve_paused_operator` takes native `Backlog` and `CampaignControlStore`
instances, a `PausedOperatorResolution`, native authorization ID/nonce, and
standard pinned `review` and JSON `contract` references. The request binds exact
project ID, absolute state/work roots, task, question, revision, unchanged
objective, directive ID/text, scope, actor, standing-authority ID/text, answer,
and writable paths.

The contract has `kind=paused_operator_resolve_resume`, `binding=request.binding()`
and `sources=source_bindings()`. A current independently reviewed native handoff
must bind its exact bytes. The separate native authorization must permit
`resume_blocked_work`, bind the request as `metadata.paused_operator`, and match
its actor (`source_channel`), directive ID (`source_message_id`), scope and
writable paths. The API cannot issue or widen that authority.

The native Manager-control lock is acquired before the native backlog/claim
lock. Both lock files must already exist; rejected requests cannot create them.
A prepared record retains the original row and leaves it unclaimable. Native
authorization consumption and the control `stage_projection` index precede the
same-ID pending commit. Exact retries recover interruptions at these persistence
boundaries without another authority consumption or metering increment. A
committed replay is read-only, including after a normal subsequent claim.

Invalid bindings, stale review/index/source/authority, nonquiescent tasks, and
conflicting transactions fail explicitly. Conflicting control revisions after
an interrupted consumption require Host reconciliation, not an authority reset.
Torn/corrupt authorization records also fail closed; this is not a torn-write
repair utility or a power-loss durability claim.

The native fresh-attempt increment is retained, while historical errors,
outcomes, iteration expenditure, dependencies, objective and other campaign
gates are preserved. No daemon, worker, objective, scientific branch, failed
continuation, completed evidence or live approval is modified by the API's
validation fixtures. Failed-continuation restoration is outside the normal
entry: the row must already be the exact pending decision in `paused_operator`.

The unittest fixture exercises the actual native typed review writer, current
index and authenticated consumer. Its seals are explicitly disposable interface
fixtures, never independent review approval for live use. The Host must obtain
the required independent Reviewer decision before any live consumption.
