from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "position2_runtime_wrapper_lifecycle.py"
)
SPEC = importlib.util.spec_from_file_location("position2_wrapper_lifecycle", SOURCE)
assert SPEC is not None and SPEC.loader is not None
LIFECYCLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LIFECYCLE)


class Position2RuntimeWrapperLifecycleTests(unittest.TestCase):
    def test_crash_classification_never_combines_retryable_with_consumed_or_terminal(
        self,
    ) -> None:
        for child_started, consumed, terminal in (
            (False, False, False),
            (True, False, True),
            (True, True, True),
            (False, False, True),
        ):
            state = LIFECYCLE.classify_crash_state(
                child_started=child_started,
                consumed=consumed,
                terminal=terminal,
            )
            self.assertTrue(state["invariant_pass"])
            self.assertFalse(
                state["retryable"] and (state["consumed"] or state["terminal"])
            )

    def test_active_wrapper_accepts_inflight_and_rejects_terminal_receipt(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = Path(directory)
            task_id = "ace3-position2-wrapper-rehearsal-test"
            command = "/bound/python -B /bound/driver execute"
            bindings = {
                "task_id": task_id,
                "submission": {"child_command": command},
            }
            logs = registry / f"{task_id}_logs"
            logs.mkdir()
            for name in ("stdout.log", "stderr.log"):
                (logs / name).write_bytes(b"")
            receipt = {
                "state": "running",
                "task_id": task_id,
                "run_id": f"{task_id}-123",
                "mode": "direct",
                "cwd": str(LIFECYCLE.ROOT),
                "command": command,
                "description": LIFECYCLE.compact_json(LIFECYCLE.ROUTE_METADATA),
                "worker_pid": os.getpid(),
            }
            receipt_path = registry / f"{task_id}.json"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            accepted = LIFECYCLE.authenticate_active_wrapper(
                bindings, registry_root=registry
            )
            self.assertEqual(accepted["state"], "running")
            receipt["state"] = "done"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(
                LIFECYCLE.LifecycleError, "prior terminal receipt"
            ):
                LIFECYCLE.authenticate_active_wrapper(
                    bindings, registry_root=registry
                )

    def test_outer_absence_and_child_active_checks_are_separate(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        outer = source[
            source.index("def outer_pre_submit(") :
            source.index("def run_logged_step(")
        ]
        child = source[
            source.index("def authenticate_active_wrapper(") :
            source.index("def runtime_paths(")
        ]
        self.assertIn('require_absent(receipt, "durable runner receipt")', outer)
        self.assertIn("receipt = load_json(receipt_path)", child)
        self.assertNotIn("require_absent(receipt_path", child)
        self.assertIn('{"starting", "running"}', child)

    def test_shared_authority_contract_requires_l2_for_formal_mode(
        self,
    ) -> None:
        driver = {"path": "/bound/driver", "bytes": 1, "sha256": "1" * 64}
        source = {"path": "/bound/source.tar", "bytes": 1, "sha256": "2" * 64}
        package = {
            "package_id": LIFECYCLE.ACCEPTED_PACKAGE_ID,
            "source_archive": source,
        }
        bindings = {
            "task_id": "ace3-position2-fresh-successor-test",
            "identity": "ace3-position2-fresh-successor-test",
            "mode": LIFECYCLE.FORMAL_AUTHORITY_MODE,
            "status": "AUTHORIZED_ONCE_UNCONSUMED",
            "runtime_driver": driver,
            "accepted_package": package,
            "accepted_source_archive": source,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            closure_path = root / "closure.json"
            closure_path.write_text("{}\n", encoding="ascii")
            closure_record = LIFECYCLE.file_record(closure_path)
            review_source_path = root / "review-source.json"
            review_source_path.write_text(
                json.dumps(
                    {
                        "kind": "round_reviewed_handoff",
                        "producer_role": "reviewer",
                        "review": {
                            "status": "continue",
                            "reason": "Independent L2 PASS: test verdict",
                        },
                    }
                ),
                encoding="ascii",
            )
            review_path = root / "l2-review.json"
            review_path.write_text(
                json.dumps(
                    {
                        "kind": (
                            "ace3_position2_runtime_wrapper_"
                            "independent_l2_review_binding"
                        ),
                        "binding_producer_role": "engineer",
                        "verdict_producer_role": "reviewer",
                        "independent": True,
                        "reviewer_level": "L2",
                        "verdict": "PASS",
                        "status": "L2_PASS",
                        "closure_binding": closure_record,
                        "review_source": LIFECYCLE.file_record(
                            review_source_path
                        ),
                    }
                ),
                encoding="ascii",
            )
            authority = {
                "schema_version": 1,
                "kind": LIFECYCLE.AUTHORITY_KIND,
                "mode": LIFECYCLE.FORMAL_AUTHORITY_MODE,
                "status": "AUTHORIZED_ONCE_UNCONSUMED",
                "task_id": bindings["task_id"],
                "identity": bindings["identity"],
                "route_metadata": LIFECYCLE.ROUTE_METADATA,
                "runtime_driver": driver,
                "accepted_package": package,
                "accepted_source_archive": source,
                "formal_execution_authority": True,
                "runtime_consumable": True,
                "reusable": False,
                "one_shot": {
                    "cardinality": 1,
                    "current_authority_consumption_count": 0,
                    "current_durable_submission_count": 0,
                    "current_payload_invocation_count": 0,
                    "current_runtime_invocation_count": 0,
                    "automatic_retry": False,
                    "retry": False,
                    "replay": False,
                    "resume": False,
                },
                "lifecycle_gate": {
                    "required_for_formal_mode": True,
                    "closure_packet": closure_record,
                    "independent_l2_review": LIFECYCLE.file_record(review_path),
                },
                "claim_boundary": "FORMAL_ONE_SHOT_RUNTIME_AUTHORITY",
            }
            LIFECYCLE.validate_authority_document(bindings, authority)
            authority["lifecycle_gate"]["independent_l2_review"] = None
            with self.assertRaisesRegex(
                LIFECYCLE.LifecycleError,
                "formal mode requires independent L2 PASS",
            ):
                LIFECYCLE.validate_authority_document(bindings, authority)


if __name__ == "__main__":
    unittest.main()
