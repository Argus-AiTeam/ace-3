"""Isolated same-task operator resolution; no daemon deployment or generic answer.

The Host supplies an indexed native review of a JSON contract containing
``binding`` and ``sources`` (absolute source paths mapped to SHA-256 digests).
The native authorization's metadata must contain the same ``paused_operator``
binding. Source pins cover this module and its native persistence/claim helpers.

Lock order is Manager control, then backlog/claim. A prepared backlog record
keeps the task paused until the native authorization audit and control index
are committed. Exact replay finishes an interrupted transaction; contradictory
state fails closed. This API never restores a failed continuation, issues
authority, clears campaign gates, starts workers, or changes the objective.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import secrets
import time
from typing import Any

from argus_skill.daemon import reviewed_snapshot
from argus_skill.life import memory
from argus_skill.manager import control_state


_KEY = "paused_operator_resolution"
_ACTION = "resume_blocked_work"


@dataclass(frozen=True)
class PausedOperatorResolution:
    project_id: str
    project_state_root: str
    project_workdir: str
    task_id: str
    decision_id: str
    decision_revision: int
    question: str
    objective: str
    directive_id: str
    directive: str
    scope: str
    actor: str
    standing_authority_id: str
    standing_authority: str
    answer: str
    writable_paths: tuple[str, ...] = ()

    def binding(self) -> dict[str, Any]:
        values = asdict(self)
        for name, value in values.items():
            if name not in {"decision_revision", "writable_paths"}:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{name} must be a nonempty exact string")
        if type(self.decision_revision) is not int or self.decision_revision < 1:
            raise ValueError("decision_revision must be a positive integer")
        if not isinstance(self.writable_paths, tuple) or any(
            not isinstance(path, str) or not path.strip()
            for path in self.writable_paths
        ):
            raise ValueError("writable_paths must be a tuple of nonempty paths")
        values["writable_paths"] = list(self.writable_paths)
        return values


def source_bindings() -> dict[str, str]:
    """Exact executable source membership required in the reviewed contract."""
    paths = (
        Path(__file__),
        Path(memory.__file__),
        Path(control_state.__file__),
        Path(reviewed_snapshot.__file__),
    )
    return {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def _review_contract(
    review: dict[str, Any], contract: dict[str, Any], binding: dict[str, Any]
) -> None:
    reviewed_snapshot._reviewed_artifact(review, contract)
    payload = reviewed_snapshot._bound_object(contract)
    if (
        payload.get("kind") != "paused_operator_resolve_resume"
        or payload.get("binding") != binding
        or payload.get("sources") != source_bindings()
    ):
        raise ValueError("reviewed contract does not bind this request and source")


def _audit_rows(store: control_state.CampaignControlStore) -> list[dict[str, Any]]:
    # The legacy reader skips malformed records. Authority cannot use that rule.
    rows = [
        json.loads(line)
        for line in store.authorization_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("invalid native authorization audit record")
    return rows


def resolve_paused_operator(
    *,
    backlog: memory.Backlog,
    store: control_state.CampaignControlStore,
    request: PausedOperatorResolution,
    authorization_id: str,
    nonce: str,
    review: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Resolve and resume a quiescent paused task, or finish its exact replay.

    Validation/conflict failures perform no persistent writes. An I/O failure
    after preparation is different: it leaves an auditable paused transaction,
    recoverable with the same arguments while its control revision is current.
    A conflicting intervening control revision requires Host reconciliation;
    it never authorizes resetting or reissuing consumed authority.
    """
    binding = request.binding()
    if not authorization_id or not isinstance(nonce, str) or not nonce:
        raise ValueError("native authorization id and nonce are required")
    if (
        backlog.path.resolve() != store.state_root.resolve() / "backlog.jsonl"
        or store.state_root.name != request.project_id
        or str(store.state_root.resolve()) != request.project_state_root
        or str(store.project_root.resolve()) != request.project_workdir
    ):
        raise ValueError("project/store identity mismatch")
    # Native lock creation is not allowed as a side effect of a rejected call.
    if not backlog._lock_path.is_file() or not store.lock_path.is_file():
        raise ValueError("native backlog and control locks must already exist")
    _review_contract(review, contract, binding)

    with store.locked(), backlog._locked():
        _review_contract(review, contract, binding)
        items = backlog._load()
        matching = [item for item in items if item.id == request.task_id]
        if len(matching) != 1:
            raise ValueError("task must exist exactly once")
        item = matching[0]
        card = item.operator_decision
        transaction = card.get(_KEY)
        rows = [
            row for row in _audit_rows(store)
            if row.get("authorization_id") == authorization_id
        ]
        if not rows or rows[0].get("event") != "issued":
            raise ValueError("native authorization was not issued")
        issued = rows[0]
        if (
            not secrets.compare_digest(str(issued.get("nonce", "")), nonce)
            or issued.get("metadata", {}).get("paused_operator") != binding
            or issued.get("scope") != request.scope
            or issued.get("source_message_id") != request.directive_id
            or issued.get("source_channel") != request.actor
            or _ACTION not in issued.get("allowed_actions", [])
            or issued.get("allowed_write_paths") != binding["writable_paths"]
        ):
            raise ValueError("authorization binding mismatch")
        if any(
            row.get(key) != value
            for row in rows[1:]
            for key, value in issued.items()
            if key != "event"
        ):
            raise ValueError("authorization audit identity changed")
        if len(rows) > 2 or rows[-1].get("event") not in {"issued", "consumed"}:
            raise ValueError("conflicting authorization audit")
        consumed = rows[-1] if rows[-1].get("event") == "consumed" else None
        if consumed is not None and (
            consumed.get("consumed_action") != _ACTION
            or consumed.get(_KEY, {}).get("binding") != binding
        ):
            raise ValueError("authorization was consumed by a different operation")

        head = store.read_head()
        snapshot = store.read_snapshot(head)
        if head is None or snapshot is None or any(
            getattr(head, key) != issued.get(key)
            for key in ("campaign_id", "objective_sha256", "campaign_epoch")
        ):
            raise ValueError("campaign identity changed")
        projection = deepcopy(snapshot.get("stage_projection", {}))
        indexed = projection.get(_KEY, {}).get(authorization_id)
        expected_transaction = {
            "authorization_id": authorization_id,
            "binding": binding,
            "review": review,
            "contract": contract,
        }
        if transaction is not None:
            if not isinstance(transaction, dict) or any(
                transaction.get(key) != value
                for key, value in expected_transaction.items()
            ):
                raise ValueError("conflicting same-task resolution replay")
            if consumed is not None and consumed.get(_KEY) != {
                **transaction, "phase": "prepared"
            }:
                raise ValueError("authorization and backlog transaction disagree")
            if transaction.get("phase") == "committed":
                if indexed != {**transaction, "phase": "prepared"} or consumed is None:
                    raise ValueError("committed resolution lacks native audit/index")
                return {"application_status": "already_applied", **deepcopy(transaction)}
            if transaction.get("phase") != "prepared":
                raise ValueError("unknown resolution transaction phase")
        elif consumed is not None or indexed is not None:
            raise ValueError("orphaned authorization/index without paused transaction")

        if (
            item.status != "paused_operator"
            or item.running_owner
            or any(other.status == "running" for other in items)
            or card.get("status") != "pending"
            or card.get("project_id") != request.project_id
            or card.get("item_id") != request.task_id
            or card.get("id") != request.decision_id
            or type(card.get("revision")) is not int
            or card["revision"] != request.decision_revision
            or card.get("question") != request.question
            or item.pending_question != request.question
            or item.objective != request.objective
            or item.owns_paths != list(request.writable_paths)
            or card.get("continuation_item_id")
        ):
            raise ValueError("task is not the exact quiescent paused decision")
        if any(
            other.id != item.id
            and other.status not in {"done", "failed", "aborted", "skipped", "superseded"}
            and other.notes == f"Continues blocked item {item.id}."
            for other in items
        ):
            raise ValueError("a continuation is not terminal")
        current = item.to_jsonable()
        current["operator_decision"] = dict(current["operator_decision"])
        current["operator_decision"].pop(_KEY, None)
        if transaction is not None and transaction.get("before") != current:
            raise ValueError("prepared paused task changed")
        if indexed is not None and indexed != transaction:
            raise ValueError("control index conflicts with paused transaction")
        if indexed is None and (
            head.state_revision != issued.get("state_revision")
            or authorization_id not in snapshot.get("authorization_ids", [])
        ):
            raise ValueError("authorization is stale relative to Manager HEAD")
        if consumed is None:
            expires = issued.get("expires_at", 0)
            if not isinstance(expires, (int, float)) or (
                expires > 0 and time.time() >= expires
            ):
                raise ValueError("authorization expired or has invalid expiry")
        frozen = issued.get("frozen_evidence")
        if not isinstance(frozen, list) or any(
            not isinstance(row, dict) for row in frozen
        ):
            raise ValueError("invalid frozen evidence set")
        if [
            control_state._hash_path(store.project_root, row.get("path"))
            for row in frozen
        ] != frozen:
            raise ValueError("frozen evidence changed")
        backlog._validate_no_dependency_cycles(items)

        if transaction is None:
            transaction = {
                **deepcopy(expected_transaction),
                "phase": "prepared",
                "before": deepcopy(current),
                "prepared_at": time.time(),
            }
            card[_KEY] = transaction
            backlog._save(items)
        if consumed is None:
            control_state._append_jsonl(store.authorization_path, {
                **issued,
                "event": "consumed",
                "consumed_at": time.time(),
                "consumed_action": _ACTION,
                "consumed_state_revision": head.state_revision + 1,
                _KEY: transaction,
            })
        if indexed is None:
            resolutions = dict(projection.get(_KEY, {}))
            resolutions[authorization_id] = deepcopy(transaction)
            projection[_KEY] = resolutions
            store._next_revision_unlocked(
                identity=control_state.CampaignIdentity(
                    head.campaign_id, head.objective_sha256, head.campaign_epoch
                ),
                updates={
                    "stage_projection": projection,
                    "authorization_ids": [
                        value for value in snapshot.get("authorization_ids", [])
                        if value != authorization_id
                    ],
                },
                reason=f"paused_operator same-task resolution: {authorization_id}",
            )
        committed = {**transaction, "phase": "committed"}
        card.update({
            _KEY: committed,
            "status": "resolved",
            "revision": request.decision_revision + 1,
            "decision_revision": request.decision_revision,
            "selected_option": "custom",
            "note": request.answer,
            "resolution_id": authorization_id,
            "resolved_at": transaction["prepared_at"],
            "resume_requested": True,
        })
        item.pending_question = ""
        item.status = "pending"
        item.attempt = max(1, int(item.attempt or 1)) + 1
        item.started_ts = None
        item.finished_ts = None
        backlog._save(items)
        return {"application_status": "accepted", **deepcopy(committed)}
