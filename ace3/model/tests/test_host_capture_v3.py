"""Capture-component regressions; child processes here are not RTL simulations."""

import json
from pathlib import Path
import sys
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import host_capture_v3 as capture
from ace3.model.candidates import runtime_admission_v3 as runtime
from ace3.model.candidates import capture_harness_v3 as harness


class HostCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="host-capture-unit-", dir=capture.BUILD)
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def test_real_process_capture_and_exclusive_output(self):
        argv = [sys.executable, "-c", "print('capture process regression, not RTL')"]
        result = capture.observe(argv, self.root, "child")
        self.assertTrue(result["executed"])
        self.assertTrue(result["natural_exit"])
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(json.loads((self.root / "child.command.json").read_text())["argv"], argv)
        self.assertGreater(result["pid"], 0)
        self.assertGreaterEqual(result["elapsed_seconds"], 0)
        with self.assertRaises(FileExistsError):
            capture.observe(argv, self.root, "child")
        self.assertFalse((self.root / "manifest.json").exists())

    def test_failed_process_stays_failed(self):
        with self.assertRaises(capture.subprocess.CalledProcessError):
            capture.observe([sys.executable, "-c", "raise SystemExit(7)"], self.root, "failure")
        result = json.loads((self.root / "failure.process.json").read_text())
        self.assertEqual(result["returncode"], 7)
        self.assertEqual(result["failure_taxonomy"], "execution_failure")
        self.assertFalse((self.root / "manifest.json").exists())

    def test_no_execution_is_distinct(self):
        with self.assertRaises(FileNotFoundError):
            capture.observe([str(self.root / "absent")], self.root, "missing")
        result = json.loads((self.root / "missing.process.json").read_text())
        self.assertFalse(result["executed"])
        self.assertFalse(result["natural_exit"])
        self.assertEqual(result["failure_taxonomy"], "evaluator_no_execution")

    def test_safe_paths_and_binding_drift(self):
        path = capture.fresh(self.root / "fresh", self.root)
        path.write_bytes(b"original")
        rec = capture.record(path)
        capture.authenticate(rec)
        with self.assertRaises(ValueError):
            capture.fresh(path, self.root)
        with self.assertRaises(ValueError):
            capture.fresh(self.root.parent / "escape", self.root)
        link = self.root / "link"
        link.symlink_to(self.root / "absent", target_is_directory=True)
        with self.assertRaises(ValueError):
            capture.fresh(link / "new", self.root)
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "binding drift"):
            capture.authenticate(rec)

    def test_native_trace_abi_not_synthetic_stage_prefix(self):
        self.assertEqual(runtime.trace_rows(b"000000010002be62\n"), [(1, 0, 2, 0xbe62)])
        self.assertEqual(runtime.trace_rows(b"00000012137f3c00\n"), [(18, 0, 4991, 0x3c00)])
        with self.assertRaisesRegex(ValueError, "token ordinal"):
            runtime.trace_rows(b"0100000000003c00\n")

    def test_extracts_all_actual_stages_and_cache_not_monitor(self):
        rows = runtime.trace_rows(runtime.artifact(runtime.TRACE_LAYOUT))
        coordinates = [row[:3] for row in rows]
        self.assertEqual({row[0] for row in rows}, set(range(19)))
        final = [row[3] for row in rows if row[0] == 18]
        final_path = self.root / "final.hex"
        final_path.write_text("".join(f"{i:06x}{word:04x}\n" for i, word in enumerate(final)))
        hidden = np.zeros(896, dtype="<u2")
        transaction = {"raw": {"trace": runtime.TRACE_LAYOUT}, "output": capture.record(final_path)}
        arrays = capture.actual_arrays(transaction, hidden, coordinates)
        for stage, count in capture.local.SIZES.items():
            self.assertEqual(arrays[f"stage{stage:02d}"].shape, (count,))
        np.testing.assert_array_equal(arrays["output_cache_k"][0], arrays["stage06"])
        self.assertEqual(arrays["input_cache_v"].shape, (0, 128))
        with self.assertRaisesRegex(ValueError, "order/coverage"):
            capture.actual_arrays(transaction, hidden, coordinates[::-1])

    def test_future_parent_trust_and_monitor_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "Host"):
            capture.admitted_parent({}, 6, None)
        with self.assertRaisesRegex(ValueError, "accepted L4"):
            capture.admitted_parent({}, 5, "0" * 64)
        path = self.root / "monitor.npz"
        with path.open("xb") as stream:
            np.savez(stream, input_hidden=np.ones(896, dtype="<u2"))
        with self.assertRaisesRegex(ValueError, "stale monitor"):
            capture.load_monitor(capture.record(path), np.zeros(896, dtype="<u2"))

    def admission_fixture(self):
        directory = self.root / "attempt003"
        directory.mkdir()
        checkpoint = directory / "checkpoint"
        checkpoint.write_bytes(b"unit fixture, not model weights")
        with (directory / "actual_operands.npz").open("xb") as stream:
            np.savez(stream, input_hidden=np.zeros(896, dtype="<u2"))
        capture.write(directory / "manifest.json", {
            "actual_operands": capture.record(directory / "actual_operands.npz")})
        plan = {
            "schema": capture.SCHEMA, "policy_id": capture.policy.POLICY_ID,
            "scope": {"layers": list(capture.LAYERS), "position": 0, "history": [9707]},
            "root": str(directory), "transactions": {"5": {"directory": str(directory)}},
            "checkpoint": capture.record(checkpoint), "bindings": [capture.record(checkpoint)],
        }
        capture.write(directory / "freeze.json", plan)
        return plan, capture.record(directory / "manifest.json")["sha256"]

    def test_admission_requires_external_digest_before_evaluation(self):
        out = self.root / "admission"
        with patch.object(runtime, "_trusted_context") as context, \
                patch.object(capture.policy, "evaluate_actual_rtl_result") as evaluate:
            for digest in (None, "", "0" * 63, "0" * 65, "g" * 64, "A" * 64, 123):
                with self.subTest(digest=digest), self.assertRaisesRegex(ValueError, "parent-Host"):
                    capture.admit({}, 5, digest, out=out)
            context.assert_not_called()
            evaluate.assert_not_called()
        self.assertFalse(out.exists())

    def test_admission_rejects_digest_mismatch_before_output_or_evaluation(self):
        plan, _ = self.admission_fixture()
        out = self.root / "admission"
        with patch.object(runtime, "_trusted_context") as context, \
                patch.object(capture.policy, "evaluate_actual_rtl_result") as evaluate:
            with self.assertRaisesRegex(ValueError, "external Host digest"):
                capture.admit(plan, 5, "0" * 64, out=out)
            context.assert_not_called()
            evaluate.assert_not_called()
        self.assertFalse(out.exists())

    def test_additive_admission_rejects_existing_nested_and_escaping_roots(self):
        plan, digest = self.admission_fixture()
        original = Path(plan["root"])
        recovery = self.root / "attempt004"
        recovery.mkdir()
        sentinel = recovery / "sentinel"
        sentinel.write_bytes(b"preserved recovery")
        link = self.root / "link"
        link.symlink_to(recovery, target_is_directory=True)
        dangling = self.root / "dangling"
        dangling.symlink_to(self.root / "absent", target_is_directory=True)
        file = self.root / "existing-file"
        file.write_bytes(b"preserved file")
        paths = (self.root, original, recovery, file, sentinel, original / "fresh",
                 recovery / "fresh", self.root.parent / "escape", Path("relative"),
                 link, link / "fresh", dangling, dangling / "fresh",
                 recovery / ".." / "fresh")
        with patch.object(capture, "BUILD", self.root), \
                patch.object(runtime, "_trusted_context") as context, \
                patch.object(capture.policy, "evaluate_actual_rtl_result") as evaluate:
            for out in paths:
                with self.subTest(out=out), self.assertRaises(ValueError):
                    capture.admit(plan, 5, digest, out=out)
            context.assert_not_called()
            evaluate.assert_not_called()
        self.assertEqual(sentinel.read_bytes(), b"preserved recovery")
        self.assertEqual(file.read_bytes(), b"preserved file")
        self.assertEqual(set(recovery.iterdir()), {sentinel})
        self.assertFalse((original / "fresh").exists())

    def test_external_and_legacy_admission_outputs_bind_invocation_and_keep_gates(self):
        plan, digest = self.admission_fixture()
        directory = Path(plan["root"])
        context = {"canonical": {}, "trajectory": {}, "binary64": []}

        def numerical_fixture(**kwargs):
            kwargs["emit"]({"stage": 0, "status": "PASS", "fixture_only": True})
            return {"status": "PASS", "local_references": {
                "stage00": np.zeros(896, dtype="<u2")}}

        with patch.object(capture, "BUILD", self.root), \
                patch.object(runtime, "_trusted_context", return_value=context), \
                patch.object(capture, "tensors_from", return_value={}), \
                patch.object(runtime, "validate_runtime", return_value={"node": [5, 0]}) as validate, \
                patch.object(capture.policy, "evaluate_p0_transaction",
                             side_effect=numerical_fixture) as numerical, \
                patch.object(capture, "observe") as observe:
            self.assertEqual(capture.admit(plan, 5, digest), 0)
            retained = {p.name: capture.record(p) for p in (directory / "admission").iterdir()}
            for action in ("admit", "admit-recovery"):
                out = self.root / action
                argv = ["host_capture_v3", action, "--plan", str(directory / "freeze.json"),
                        "--layer", "5", "--out", str(out), "--trusted-manifest-sha256", digest]
                with patch.object(sys, "argv", argv), \
                        patch.object(capture, "load_recovery", return_value=plan) as recovery:
                    self.assertEqual(capture.main(), 0)
                    if action == "admit-recovery":
                        recovery.assert_called_once_with(directory / "freeze.json")
                    else:
                        recovery.assert_not_called()
                result = json.loads((out / "result.json").read_text())
                command = runtime.document(result["admission_invocation"])
                self.assertEqual(command["output_root"], str(out))
                self.assertEqual(command["manifest"], capture.record(directory / "manifest.json"))
                self.assertEqual(command["plan"], capture.record(directory / "freeze.json"))
                self.assertEqual(command["trusted_manifest_sha256"], digest)
                self.assertEqual(command["argv"], argv)
                self.assertEqual(validate.call_args.kwargs["receipt"], command["manifest"])
                self.assertEqual(validate.call_args.kwargs["trusted_manifest_sha256"], digest)
                self.assertEqual(result["local_references"], capture.record(out / "local_references.npz"))
                self.assertTrue((out / "stage00.json").is_file())
                with self.assertRaisesRegex(ValueError, "immutable output"):
                    capture.admit(plan, 5, digest, out=out)
                command["output_root"] = str(self.root / "tampered")
                (out / "command.json").write_text(json.dumps(command))
                with self.assertRaises(ValueError):
                    runtime.document(result["admission_invocation"])
            self.assertEqual(validate.call_count, 3)
            self.assertEqual(numerical.call_count, 3)
            observe.assert_not_called()
        self.assertEqual(retained, {
            p.name: capture.record(p) for p in (directory / "admission").iterdir()})

    def test_external_admission_preserves_blocked_and_failed_results(self):
        plan, digest = self.admission_fixture()
        context = {"canonical": {}, "trajectory": {}, "binary64": []}
        with patch.object(capture, "BUILD", self.root), \
                patch.object(runtime, "_trusted_context", return_value=context), \
                patch.object(capture, "tensors_from", return_value={}), \
                patch.object(runtime, "validate_runtime") as validate, \
                patch.object(capture.policy, "evaluate_p0_transaction") as numerical:
            for error in ("source identity mismatch", "cache producer identity mismatch"):
                out = self.root / error.replace(" ", "-")
                validate.side_effect = ValueError(error)
                self.assertEqual(capture.admit(plan, 5, digest, out=out), 1)
                result = json.loads((out / "result.json").read_text())
                self.assertEqual(result["status"], "BLOCKED")
                self.assertEqual(result["numerical_status"], "NOT_EVALUATED")
                self.assertIn(error, result["reason"])
                runtime.document(result["admission_invocation"])
            numerical.assert_not_called()
            validate.side_effect = None
            validate.return_value = {"node": [5, 0]}
            numerical.return_value = {"status": "FAIL"}
            out = self.root / "numerical-failure"
            self.assertEqual(capture.admit(plan, 5, digest, out=out), 1)
            result = json.loads((out / "result.json").read_text())
            self.assertEqual(result["status"], result["numerical_status"])
            self.assertEqual(result["status"], "FAIL")

    def test_cli_recovery_binding_drift_and_capture_output_override_fail_closed(self):
        plan, digest = self.admission_fixture()
        directory = Path(plan["root"])
        plan.update(schema="ace3-v3-retained-recovery-plan-v1",
                    scope={"layers": [5], "position": 0, "history": [9707]},
                    original_plan=plan["checkpoint"], original_manifest=plan["checkpoint"],
                    original_admission=plan["checkpoint"])
        (directory / "freeze.json").write_text(json.dumps(plan))
        (directory / "checkpoint").write_bytes(b"drift")
        out = self.root / "external"
        with patch.object(capture.policy, "evaluate_actual_rtl_result") as evaluate, \
                patch.object(capture, "capture") as execute:
            for action, reason in (("admit-recovery", "binding drift"),
                                   ("capture", "frozen output path")):
                argv = ["host_capture_v3", action, "--plan", str(directory / "freeze.json"),
                        "--layer", "5", "--out", str(out), "--trusted-manifest-sha256", digest]
                with patch.object(sys, "argv", argv), self.assertRaisesRegex(ValueError, reason):
                    capture.main()
            evaluate.assert_not_called()
            execute.assert_not_called()
        self.assertFalse(out.exists())

    def test_retained_l5_coordinate_oracle_preserves_original_failure(self):
        root = capture.BUILD / "host_capture_v3_66725e8b68d5_attempt003/runtime/layer05"
        manifest = runtime.document(capture.record(root / "manifest.json"))
        transaction = runtime.document(manifest["transaction"])
        data = runtime.artifact(transaction["raw"]["trace"])
        records = [struct.unpack(">BHBHH", bytes.fromhex(line.decode("ascii")))
                   for line in data.splitlines()]
        self.assertEqual(len(records), 23324)
        self.assertEqual({(r[0], r[1]) for r in records}, {(0, 0)})
        coordinates = [r[:3] for r in runtime.trace_rows(runtime.artifact(runtime.TRACE_LAYOUT))]
        hidden = runtime.hex_words(runtime.artifact(manifest["parent"]), indexed=True)
        arrays = capture.actual_arrays(transaction, hidden, coordinates)
        for stage, count in capture.local.SIZES.items():
            rows = [(r[3], r[4]) for r in records if r[2] == stage]
            self.assertEqual(len(rows), count)
            if stage in (8, 9):
                self.assertEqual([r[0] for r in rows], [0] * 14)
                expected = [r[1] for r in rows]
            else:
                self.assertEqual(sorted(r[0] for r in rows), list(range(count)))
                expected = [r[1] for r in sorted(rows)]
            np.testing.assert_array_equal(arrays[f"stage{stage:02d}"], expected)
        capture.local.validate_lineage(arrays, hidden, position=0, history=[9707])
        with np.load(root / "actual_operands.npz", allow_pickle=False) as old:
            self.assertEqual(np.count_nonzero(old["stage05"] != old["stage06"]), 124)
            with self.assertRaisesRegex(ValueError, "cache producer identity"):
                capture.local.validate_lineage(old, hidden, position=0, history=[9707])
        self.assertEqual(json.loads((root / "admission/result.json").read_text())["reason"],
                         "ValueError: cache producer identity mismatch")

    def test_coordinate_rejection_independent_of_layout_authority(self):
        rows = runtime.trace_rows(runtime.artifact(runtime.TRACE_LAYOUT))
        cases = []
        cases.append(rows[:-1])
        for stage, index in ((5, 128), (5, -1), (8, 1), (9, 1)):
            changed = rows.copy()
            n = next(i for i, r in enumerate(changed) if r[0] == stage)
            s, p, _, value = changed[n]
            changed[n] = (s, p, index, value)
            cases.append(changed)
        duplicate = rows.copy()
        n = next(i for i, r in enumerate(duplicate) if r[:3] == (5, 0, 32))
        duplicate[n] = (5, 0, 0, duplicate[n][3])
        cases.append(duplicate)
        for replacement in ((19, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 65536)):
            cases.append([replacement, *rows[1:]])
        reordered = rows.copy()
        a = next(i for i, r in enumerate(rows) if r[0] == 8)
        b = next(i for i, r in enumerate(rows) if r[0] == 9)
        reordered[a], reordered[b] = reordered[b], reordered[a]
        cases.append(reordered)
        extra = rows.copy()
        extra.insert(a, rows[a])
        cases.append(extra)
        for number, changed in enumerate(cases):
            with self.subTest(case=number), self.assertRaises(ValueError):
                runtime.canonical_stages(changed, [r[:3] for r in changed])
        with self.assertRaisesRegex(ValueError, "order/coverage"):
            runtime.canonical_stages(rows[::-1], [r[:3] for r in rows])

    def test_command_keeps_reset_state_and_bounded_outputs(self):
        argv = capture.simulation_command(self.root, 5)
        self.assertNotIn("--state-in", argv)
        self.assertEqual(argv[argv.index("--transaction-position") + 1], "0")
        self.assertEqual(argv[argv.index("--state-out") + 1], str(self.root / "candidate.state"))
        self.assertEqual(argv.count(harness.CAPTURE_FLAG), 1)

    def test_capture_monitor_values_do_not_constrain_actual_parent(self):
        arrays = {f"stage{s:02d}": np.zeros(n, dtype="<u2")
                  for s, n in capture.local.SIZES.items()}
        arrays["input_hidden"] = np.zeros(896, dtype="<u2")
        path = self.root / "monitor.npz"
        with path.open("xb") as stream:
            np.savez(stream, **arrays)
        hidden = np.ones(896, dtype="<u2")
        with self.assertRaisesRegex(ValueError, "stale monitor"):
            capture.load_monitor(capture.record(path), hidden)
        loaded = capture.load_monitor(capture.record(path), hidden, capture_only=True)
        np.testing.assert_array_equal(loaded["input_hidden"], arrays["input_hidden"])

    def test_only_authentic_reviewed_runtime_policy_successor(self):
        selected = capture.reviewed_policy_source()
        self.assertEqual(selected, capture.record(capture.policy.__file__))
        original = json.loads((capture.SOFTWARE / "freeze.json").read_text())
        old = next(r for r in original["sources"]
                   if Path(r["path"]).name == "decoder_gate_policy_v3.py")
        self.assertNotEqual(old["sha256"], selected["sha256"])
        frozen = json.loads((capture.RUNTIME_SOFTWARE / "freeze.json").read_text())
        target = next(r for r in frozen["sources"]
                      if Path(r["path"]).name == "decoder_gate_policy_v3.py")
        target["sha256"] = "0" * 64
        (self.root / "freeze.json").write_text(json.dumps(frozen))
        with patch.object(capture, "RUNTIME_SOFTWARE", self.root):
            with self.assertRaisesRegex(ValueError, "binding drift"):
                capture.reviewed_policy_source()


if __name__ == "__main__":
    unittest.main()
