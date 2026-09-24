"""Continuation binding tests: no decoder capture, admission, or numerical rerun."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from ace3.model.candidates import prepare_l6_l8_continuation_v3 as continuation


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="l6-l8-plan-", dir=continuation.host.BUILD)
        self.addCleanup(self.temporary.cleanup)
        self.out = Path(self.temporary.name)
        root = continuation.host.BUILD / "host_capture_v3_54dc8dbc7037_attempt004"
        self.recovery = json.loads((root / "freeze.json").read_text())
        self.original = continuation.host.runtime.document(self.recovery["original_plan"])
        self.parent = continuation.host.record(root / "admission/result.json")

    def test_actual_parent_and_only_new_suffix_paths(self):
        before = copy.deepcopy(self.original)
        plan = continuation.continuation_plan(self.original, self.recovery, self.parent, self.out)
        parent, hidden = continuation.host.admitted_parent(plan, 6, self.parent["sha256"])
        self.assertEqual(parent, self.parent)
        self.assertEqual(hidden, plan["continuation"]["actual_l5_hidden"])
        self.assertEqual(self.original, before)
        self.assertEqual(plan["source_closure"], before["source_closure"])
        self.assertEqual(plan["checkpoint"], before["checkpoint"])
        for layer in continuation.LAYERS:
            old = before["transactions"][str(layer)]
            new = plan["transactions"][str(layer)]
            self.assertIsNone(new["input_state"])
            self.assertEqual(new["cache_slot"], 0)
            self.assertEqual(new["canonical_tensors"], old["canonical_tensors"])
            self.assertEqual(new["exact_monitor"], old["exact_monitor"])
            self.assertEqual(new["simulation_argv"], continuation.host.simulation_command(
                self.out / "runtime" / f"layer{layer:02d}", layer))
            expected = list(old["compile_argv"])
            expected[expected.index("--Mdir") + 1] = str(self.out / "runtime" / f"layer{layer:02d}" / "obj")
            self.assertEqual(new["compile_argv"], expected)
        with self.assertRaisesRegex(ValueError, "immutable output"):
            continuation.host.fresh(Path(plan["transactions"]["5"]["directory"]),
                                    continuation.host.BUILD)
        self.assertFalse((self.out / "runtime").exists())

    def test_state_restore_and_wrong_recovery_are_rejected(self):
        wrong = copy.deepcopy(self.original)
        wrong["transactions"]["6"]["input_state"] = {"path": "L5.state"}
        with self.assertRaisesRegex(ValueError, "own empty P0"):
            continuation.continuation_plan(wrong, self.recovery, self.parent, self.out)
        wrong_recovery = dict(self.recovery, root=str(self.out))
        with self.assertRaisesRegex(ValueError, "explicit recovery"):
            continuation.continuation_plan(self.original, wrong_recovery, self.parent, self.out)

    def test_changed_source_and_missing_host_digest_are_rejected(self):
        wrong = copy.deepcopy(self.original)
        wrong["source_closure"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "compatibly admitted"):
            continuation.continuation_plan(wrong, self.recovery, self.parent, self.out)
        with self.assertRaisesRegex(ValueError, "Host"):
            continuation.host.admitted_parent({}, 6, None)

    def test_command_files_require_host_digests_and_exclusive_paths(self):
        plan = continuation.continuation_plan(self.original, self.recovery, self.parent, self.out)
        commands = continuation.command_files(plan, self.out)
        self.assertEqual(len(commands), 6)
        for rec in commands:
            text = Path(rec["path"]).read_text()
            self.assertIn("sha256sum --check", text)
            self.assertIn("separately Host-observed digest required", text)
            self.assertNotIn("--layer 5", text)
            self.assertNotIn("--state-in", text)
        with self.assertRaises(FileExistsError):
            continuation.command_files(plan, self.out)


if __name__ == "__main__":
    unittest.main()
