"""No retained science is dispatched by these synthetic focused regressions."""

from fractions import Fraction as F
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import q24_s16_l23_p0_value_content_s18_family_counterfactual_v1 as d


class CounterfactualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = d.selection.finite_table()
        cls.oracle = d.module_from(d.ORACLE, "family_independent_test_oracle")

    def fixture(self, control):
        words = np.full(896, 0x3c00, dtype="<u2")
        coords = list(d.COMMON) + ([] if control == "mapped_all" else [783, 827])
        words[coords] = 0x4000
        ref = np.ones(896, dtype="<f8")
        rows = [{"index": i, **d.bridge.s18_quantities(f"{int(w):04x}", float(r).hex())}
                for i, (w, r) in enumerate(zip(words, ref))]
        return words, ref, rows

    def test_exact_nearest_passable_independent_exhaustive(self):
        for word, reference in ((0x4000, F(1)), (0xc000, F(-1)),
                                (0x3c01, F(1)), (0x8000, F(0))):
            with self.subTest(word=word):
                got, floor = d.selection.nearest(word, reference, self.table)
                values, words = self.table
                q = min(abs(v - reference) for v in values)
                allowed = [(v, w) for v, w in zip(values, words)
                           if abs(v - reference) - q <= F(1, 8)]
                expected = word if any(w == word for _, w in allowed) else min(
                    allowed, key=lambda p: (abs(p[0] - self.oracle.scalar(word)), p[0], p[1]))[1]
                self.assertEqual((got, floor), (expected, q))

    def test_nine_real_guarded_suffixes_and_independent_oracle(self):
        audit = d.new_audit()
        operands = (np.ones(896, dtype="<f2"), np.zeros((2, 896), dtype="<f2"))
        operands[1][0] = 1 / 1024
        refs = {"fp16": np.zeros(151936, dtype="<u2"),
                "binary64": np.zeros(151936, dtype="<f8")}
        refs["fp16"][34319], refs["binary64"][34319] = 0x4000, 2
        all_rows = []
        with d.guarded(d.OUTPUT, audit):
            for control in d.CONTROLS:
                words, ref, rows = self.fixture(control)
                raw = {"stage18": words, "stage17": words.copy(),
                       "output_i": np.arange(896, dtype="<i8"),
                       "output_z": np.zeros(896, dtype="u1"),
                       "final_logits": np.zeros(2, dtype="<u2")}
                classified = {"control": control,
                              "failure_coordinates": list(d.COMMON) + (
                                  [] if control == "mapped_all" else [783, 827]),
                              "paired_margins": [
                                  {"reference_branch": branch, "pair": d.PAIR,
                                   "retained_actual_margin": "0",
                                   "substituted_actual_margin": "0",
                                   "retained_reference_margin": "2", "required_direction": "2"}
                                  for branch in ("fp16", "binary64")]}
                payload, result = d.run_control(
                    control, raw, {"control": control, "S18": rows}, classified,
                    ref, refs, operands, self.table, self.oracle, audit)
                self.assertEqual(result["S18"]["status"], "PASS")
                for key, value in raw.items():
                    self.assertEqual(value.tobytes(), payload[key].tobytes())
                all_rows.append(result)
        self.assertEqual(d.classify(all_rows), "SUPPORTED")
        self.assertEqual(audit["forbidden_events"], [])
        self.assertEqual(audit["final_rmsnorm_invocations"], 9)
        self.assertEqual(audit["pair_head_invocations"], 9)
        self.assertEqual(audit["writes"], [])

    def test_reference_and_classification_splices(self):
        for defect in ("word", "reference", "extra_failure", "mapped_family"):
            with self.subTest(defect=defect):
                words, ref, rows = self.fixture("mapped_all")
                if defect == "word":
                    words[53] = 0x4001
                elif defect == "reference":
                    ref[53] = 2
                elif defect == "extra_failure":
                    words[0] = 0x4000
                    rows[0] = {"index": 0, **d.bridge.s18_quantities("4000", float(1).hex())}
                else:
                    rows[783]["accepted"] = False
                with self.assertRaises(d.bridge.BindingError):
                    d.intervene(words, ref, rows, "mapped_all", self.table)

    def test_strict_direction_uses_original_baseline_not_value_content(self):
        refs = {"fp16": np.zeros(151936, dtype="<u2"),
                "binary64": np.zeros(151936, dtype="<f8")}
        refs["fp16"][34319], refs["binary64"][34319] = 0x4000, 2
        margins = [{"pair": d.PAIR, "reference_branch": b, "retained_actual_margin": "1",
                    "substituted_actual_margin": "0", "retained_reference_margin": "2",
                    "required_direction": "1"} for b in refs]
        for word, expected in ((0x3800, False), (0x3c00, False), (0x3e00, True)):
            rows = d.paired_margins(np.array([word, 0], dtype="<u2"),
                                   np.zeros(2, dtype="<u2"), margins, refs)
            self.assertTrue(all(r["direction_pass"] is expected for r in rows))
        margins[0]["required_direction"] = "-1"
        with self.assertRaises(d.bridge.BindingError):
            d.paired_margins(np.zeros(2, dtype="<u2"), np.zeros(2, dtype="<u2"), margins, refs)

    def test_numerical_rejection_and_complete_census(self):
        rows = [{"control": c, "S18": {"status": "PASS"},
                 "paired_margins": [{"reference_branch": b, "direction_pass": True}
                                    for b in ("fp16", "binary64")]} for c in d.CONTROLS]
        self.assertEqual(d.classify(rows), "SUPPORTED")
        rows[0]["S18"]["status"] = "FAIL"
        self.assertEqual(d.classify(rows), "REJECTED")
        rows[0]["S18"]["status"] = "PASS"
        rows[8]["paired_margins"][1]["direction_pass"] = False
        self.assertEqual(d.classify(rows), "REJECTED")
        with self.assertRaises(d.bridge.BindingError):
            d.classify(rows[::-1])

    def test_write_service_and_producer_guards(self):
        for event, args in (
            ("subprocess.Popen", ()), ("socket.connect", ()),
            ("os.rename", ()), ("os.mkdir", ()),
            ("import", ("torch",)), ("open", (str(d.SOURCE), "w", 577)),
            ("open", (str(d.OUTPUT / "occupied"), "w", 577)),
        ):
            with self.subTest(event=event):
                audit = d.new_audit()
                with d.guarded(d.OUTPUT, audit):
                    with self.assertRaises(d.bridge.BindingError):
                        d.audit_event(event, args)
                self.assertTrue(audit["forbidden_events"])
        audit = d.new_audit()
        with d.guarded(d.OUTPUT, audit):
            with self.assertRaises(d.bridge.BindingError):
                d.suffix.check(audit)
        self.assertTrue(audit["forbidden_events"])

    def test_occupied_root_refuses_without_dispatch(self):
        with patch.object(d.OUTPUT.__class__, "exists", return_value=True), \
                patch.object(d.subprocess, "check_output", return_value="argus/full-projection\n"), \
                patch.object(d.subprocess, "run") as git:
            git.return_value.returncode = 0
            with self.assertRaisesRegex(d.bridge.BindingError, "occupied"):
                d.execute()

    def test_corrupted_suffix_is_not_accepted_by_oracle(self):
        words = np.full(896, 0x3c00, dtype="<u2")
        operands = (np.ones(896, dtype="<f2"), np.zeros((2, 896), dtype="<f2"))
        norm, account = d.suffix.rmsnorm(words, operands[0])
        logits = d.suffix.logits(norm, operands[1])
        norm[1] ^= 1
        with self.assertRaises(AssertionError):
            self.oracle.verify_suffix(words, operands, norm, account, logits)
