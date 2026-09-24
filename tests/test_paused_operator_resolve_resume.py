"""Disposable native-store checks, not an independent Reviewer approval."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from argus_skill.core.artifact_binding import parse_artifact_bindings
from argus_skill.core.operator_decision import build_operator_decision
from argus_skill.daemon.reviewed_snapshot import _reviewed_artifact
from argus_skill.life.context_packet import record_reviewed_handoff
from argus_skill.life.memory import Backlog, BacklogItem
from argus_skill.manager.control_state import CampaignControlStore
from argus_skill.reviewer._parsing import ReviewDecision


SOURCE = Path(__file__).resolve().parents[1] / "argus_skill/life/paused_operator.py"
SPEC = importlib.util.spec_from_file_location("isolated_paused_operator", SOURCE)
assert SPEC is not None and SPEC.loader is not None
api = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = api
SPEC.loader.exec_module(api)


def pin(path):
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


class PausedOperatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ["ARGUS_PAUSED_TEST_ROOT"])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.work = self.root / "work"
        self.work.mkdir()
        self.state = self.root / "project"
        self.state.mkdir()
        self.backlog = Backlog(self.state / "backlog.jsonl")
        self.store = CampaignControlStore(self.state, project_root=self.work)
        self.request = api.PausedOperatorResolution(
            project_id="project", project_state_root=str(self.state),
            project_workdir=str(self.work),
            task_id="original", decision_id="decision-original",
            decision_revision=1, question="Resume this exact task?",
            objective="Frozen bounded software task.", directive_id="directive-1",
            directive="Resume the original task only.", scope="bounded-original",
            actor="operator", standing_authority_id="standing-1",
            standing_authority="One same-ID decision resolution.",
            answer="Proceed within the frozen scope.",
        )
        item = BacklogItem(
            id="original", ts=1, title="Original",
            objective=self.request.objective, status="paused_operator",
            pending_question=self.request.question, attempt=4,
            iteration_cost_usd=7.5, last_error="Historical failure",
            started_ts=2, finished_ts=3,
            outcome={"preserved": "failed historical check"},
        )
        item.operator_decision = build_operator_decision(
            item_id=item.id, title=item.title, reason="Operator decision",
            question=item.pending_question, project_id="project",
        )
        self.backlog.add(item)
        self.backlog.add(BacklogItem(
            id="dependent", ts=2, title="Dependent", objective="Do not rewrite",
            status="paused_operator", deps=["original"],
        ))
        self.before = [row.to_jsonable() for row in self.backlog._load()]
        identity = self.store.campaign_identity(objective="Frozen campaign", campaign_epoch=1)
        self.store.commit_revision(
            identity=identity,
            updates={"active_wait": {"wait_id": "preserved"}},
            reason="Disposable test initialization",
        )
        self.authorization = self.store.issue_authorization(
            identity=identity, blocker_fingerprint="paused-original",
            allowed_actions=["resume_blocked_work"], scope=self.request.scope,
            allowed_write_paths=[], evidence_paths=[],
            source_channel=self.request.actor,
            source_message_id=self.request.directive_id,
            metadata={"paused_operator": self.request.binding()},
        )
        self.control_before = self.store.read_snapshot()
        self.contract_path = self.root / "contract.json"
        self.contract_path.write_text(json.dumps({
            "kind": "paused_operator_resolve_resume",
            "binding": self.request.binding(), "sources": api.source_bindings(),
        }))
        handoff_dir = self.root / "disposable-review"
        handoff_dir.mkdir()
        mission = handoff_dir / "mission.json"
        mission.write_text(json.dumps({"mission_id": handoff_dir.name}))
        self.contract = pin(self.contract_path)
        self.typed_review = ReviewDecision(
            status="done", reason="Disposable interface fixture, not live approval",
            next_action="",
            artifact_bindings=parse_artifact_bindings([{
                "ref": str(self.contract_path),
                "sha256": self.contract["sha256"],
            }]),
        )
        self.review_path = record_reviewed_handoff(
            mission_context_path=mission, round_index=1,
            engineer_summary="", review=self.typed_review, checkpoint_path=None,
        )
        self.review = {
            **pin(self.review_path), "mission_id": handoff_dir.name,
            "index_path": str(handoff_dir / "latest.json"),
        }

    def invoke(self, **overrides):
        args = {
            "backlog": self.backlog, "store": self.store,
            "request": self.request,
            "authorization_id": self.authorization.authorization_id,
            "nonce": self.authorization.nonce,
            "review": self.review, "contract": self.contract,
        }
        args.update(overrides)
        return api.resolve_paused_operator(**args)

    def files(self):
        return {
            str(path.relative_to(self.root)): path.read_bytes()
            for path in self.root.rglob("*") if path.is_file()
        }

    def test_same_id_history_metering_native_claim_and_exact_replay(self):
        result = self.invoke()
        self.assertEqual(result["application_status"], "accepted")
        rows = self.backlog._load()
        self.assertEqual([row.id for row in rows], ["original", "dependent"])
        original = rows[0]
        self.assertEqual(original.status, "pending")
        self.assertEqual(original.attempt, 5)
        self.assertEqual(original.iteration_cost_usd, 7.5)
        self.assertEqual(original.last_error, "Historical failure")
        self.assertEqual(original.outcome, self.before[0]["outcome"])
        self.assertEqual(original.objective, self.before[0]["objective"])
        self.assertEqual(original.operator_decision[api._KEY]["before"], self.before[0])
        self.assertEqual(rows[1].to_jsonable(), self.before[1])
        control_after = self.store.read_snapshot()
        for field in ("active_wait", "active_capability", "terminal_evidence"):
            self.assertEqual(control_after[field], self.control_before[field])
        self.assertEqual(
            [row["event"] for row in self.store.authorization_events()],
            ["issued", "consumed"],
        )
        before = self.files()
        self.assertEqual(self.invoke()["application_status"], "already_applied")
        self.assertEqual(before, self.files())
        claimed = self.backlog.claim_next(expected_id="original", owner="normal-host")
        self.assertEqual(claimed.id, "original")
        self.assertEqual(claimed.attempt, 5)
        before = self.files()
        self.assertEqual(self.invoke()["application_status"], "already_applied")
        self.assertEqual(before, self.files())

    def test_invalid_bindings_have_zero_writes(self):
        changes = {
            "project_id": "other", "project_workdir": str(self.root),
            "project_state_root": str(self.root),
            "task_id": "missing", "decision_id": "other", "decision_revision": 2,
            "question": "Different question", "objective": "Different objective",
            "directive_id": "other", "directive": "other", "scope": "other",
            "actor": "other", "standing_authority_id": "other",
            "standing_authority": "other", "answer": "other",
            "writable_paths": ("other",),
        }
        for name, value in changes.items():
            with self.subTest(name=name):
                before = self.files()
                with self.assertRaises(ValueError):
                    self.invoke(request=replace(self.request, **{name: value}))
                self.assertEqual(before, self.files())
        for revision in (True, 0, "1"):
            with self.subTest(revision=revision):
                before = self.files()
                with self.assertRaises(ValueError):
                    self.invoke(request=replace(self.request, decision_revision=revision))
                self.assertEqual(before, self.files())
        before = self.files()
        with self.assertRaises(ValueError):
            self.invoke(nonce="incorrect")
        self.assertEqual(before, self.files())

    def test_stale_task_and_active_claim_reject_without_writes(self):
        for fields in (
            {"pending_question": "changed"},
            {"running_owner": "active-host"},
            {"status": "failed"},
            {"objective": "changed"},
        ):
            with self.subTest(fields=fields):
                with self.backlog._locked():
                    rows = [BacklogItem.from_jsonable(row) for row in self.before]
                    for name, value in fields.items():
                        setattr(rows[0], name, value)
                    self.backlog._save(rows)
                before = self.files()
                with self.assertRaises(ValueError):
                    self.invoke()
                self.assertEqual(before, self.files())

    def test_concurrent_identical_resolve_consumes_only_once(self):
        barrier = threading.Barrier(2)

        def resolve():
            barrier.wait(timeout=5)
            return self.invoke()["application_status"]

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(resolve) for _ in range(2)]
            results = [future.result(timeout=15) for future in futures]
        self.assertCountEqual(results, ["accepted", "already_applied"])
        self.assertEqual(len(self.store.authorization_events()), 2)
        self.assertEqual(self.backlog._load()[0].attempt, 5)

    def test_native_claim_waits_for_resolution_transaction(self):
        entered = threading.Event()
        release = threading.Event()
        claimed_started = threading.Event()
        original_save = self.backlog._save

        def save(items):
            if items[0].status == "pending":
                entered.set()
                if not release.wait(timeout=5):
                    raise TimeoutError("test did not release commit")
            return original_save(items)

        def claim():
            claimed_started.set()
            return self.backlog.claim_next(expected_id="original", owner="host")

        with ThreadPoolExecutor(max_workers=2) as pool:
            with patch.object(self.backlog, "_save", side_effect=save):
                resolution = pool.submit(self.invoke)
                self.assertTrue(entered.wait(timeout=5))
                claimed = pool.submit(claim)
                self.assertTrue(claimed_started.wait(timeout=5))
                self.assertFalse(claimed.done())
                release.set()
                resolution.result(timeout=10)
                self.assertEqual(claimed.result(timeout=10).id, "original")

    def test_failure_boundaries_recover_without_duplicate_authority(self):
        targets = (
            (self.backlog, "_save", 1),
            (api.control_state, "_append_jsonl", 1),
            (self.store, "_next_revision_unlocked", 1),
            (self.backlog, "_save", 2),
        )
        initial = self.files()
        for target, name, occurrence in targets:
            for after_write in (False, True):
                with self.subTest(operation=name, occurrence=occurrence, after=after_write):
                    # Each subcase uses a new disposable native store, never a
                    # rollback or reset of a previously consumed authorization.
                    fixture = PausedOperatorTests()
                    fixture.setUp()
                    try:
                        owner = (
                            fixture.backlog if target is self.backlog else
                            fixture.store if target is self.store else api.control_state
                        )
                        original = getattr(owner, name)
                        calls = 0

                        def fail(*args, **kwargs):
                            nonlocal calls
                            calls += 1
                            if calls == occurrence and not after_write:
                                raise OSError("injected before persistence")
                            value = original(*args, **kwargs)
                            if calls == occurrence and after_write:
                                raise OSError("injected after persistence")
                            return value

                        with patch.object(owner, name, side_effect=fail):
                            with self.assertRaises(OSError):
                                fixture.invoke()
                        result = fixture.invoke()
                        self.assertIn(result["application_status"], {"accepted", "already_applied"})
                        self.assertEqual(fixture.backlog._load()[0].attempt, 5)
                        self.assertEqual(len(fixture.store.authorization_events()), 2)
                        before = fixture.files()
                        fixture.invoke()
                        self.assertEqual(before, fixture.files())
                    finally:
                        fixture.doCleanups()
        self.assertEqual(initial, self.files())

    def test_native_typed_writer_index_consumer_and_supersession(self):
        consumed = _reviewed_artifact(self.review, self.contract)
        self.assertEqual(consumed["producer_role"], "reviewer")
        self.assertEqual(consumed["review"]["artifact_bindings"], [{
            "ref": str(self.contract_path), "sha256": self.contract["sha256"],
        }])
        index = json.loads(Path(self.review["index_path"]).read_text())
        self.assertEqual(index["kind"], "handoff_ref")
        self.assertEqual(index["handoff"]["path"], str(self.review_path))
        print(json.dumps({
            "kind": "disposable_native_review_interface",
            "production_approval": False,
            "writer": "argus_skill.life.context_packet.record_reviewed_handoff",
            "typed_review": type(self.typed_review).__name__,
            "index": index,
            "consumer": "argus_skill.daemon.reviewed_snapshot._reviewed_artifact",
            "consumed_review": consumed,
        }, sort_keys=True))
        record_reviewed_handoff(
            mission_context_path=self.review_path.parent / "mission.json",
            round_index=2, engineer_summary="",
            review=self.typed_review, checkpoint_path=None,
        )
        before = self.files()
        with self.assertRaises(ValueError):
            self.invoke()
        self.assertEqual(before, self.files())


if __name__ == "__main__":
    unittest.main()
