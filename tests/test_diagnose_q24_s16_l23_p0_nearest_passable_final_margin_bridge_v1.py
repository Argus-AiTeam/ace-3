from contextlib import redirect_stdout
from fractions import Fraction
import io
import json
import os
from pathlib import Path
import subprocess
import types
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l23_p0_nearest_passable_final_margin_bridge_v1 as d


class FinalMarginBridgeTests(unittest.TestCase):
    def test_coordinate_identity_independent_expansion(self):
        for h0, h1, w, v in ((2, 3, 5, -7), (2, 2, -3, 4), (0, 0, 2, 9), (-5, 2, -3, -4)):
            y0, y1, s0, s1 = map(Fraction, ("3/2", "-7/8", "2/3", "4/5"))
            terms, observed = d.coordinate_terms(
                Fraction(h0), Fraction(h1), y0, y1, Fraction(w), Fraction(v), s0, s1)
            self.assertEqual(terms[0], v * w * s0 * (h1 - h0))
            self.assertEqual(terms[1] + terms[2], v * w * h1 * (s1 - s0))
            self.assertEqual(observed, v * (y1 - y0))
            self.assertEqual(sum(terms), observed)

    def fixture(self):
        zeros = (Fraction(),) * d.WIDTH
        h0 = (Fraction(2), Fraction(3)) + zeros[2:]
        h1 = (Fraction(4), Fraction(3)) + zeros[2:]
        y0 = (Fraction(1), Fraction(2)) + zeros[2:]
        y1 = (Fraction(3), Fraction(4)) + zeros[2:]
        weights = (Fraction(1),) * d.WIDTH
        head = {13: weights, 319: (Fraction(-2),) * d.WIDTH,
                34319: (Fraction(3),) * d.WIDTH}
        rows = {row: d.row_account(head[row], y0, y1, Fraction(row, 32), Fraction(row, 16))
                for row in d.ROWS}
        return h0, h1, y0, y1, weights, head, rows

    def test_all_pairs_close_with_unchanged_coordinates_and_head_boundaries(self):
        h0, h1, y0, y1, w, head, rows = self.fixture()
        for pair in d.PAIRS:
            r = d.pair_account(pair, h0, h1, y0, y1, w, head,
                               Fraction(1, 2), Fraction(3, 4), {0}, rows)
            expected = Fraction(pair[0] - pair[1], 32)
            self.assertEqual(Fraction(r["decomposition_sum"]), expected)
            self.assertNotEqual(Fraction(r["unchanged_coordinate_contribution"]), 0)
            self.assertEqual(r["unchanged_coordinate_direct_hidden"], "0")
            self.assertEqual(r["closure_residual"], "0")
            self.assertEqual(len(r["coordinates"]), 896)

    def test_reversed_pair_negates_every_coordinate_term(self):
        h0, h1, y0, y1, w, head, rows = self.fixture()
        reports = [d.pair_account(pair, h0, h1, y0, y1, w, head,
                                  Fraction(1, 2), Fraction(3, 4), {0}, rows)
                   for pair in d.PAIRS[:2]]
        for a, b in zip(reports[0]["coordinates"], reports[1]["coordinates"], strict=True):
            for key in d.TERMS:
                self.assertEqual(Fraction(a[key]), -Fraction(b[key]))

    def test_selected_row_splice_rejected(self):
        h0, h1, y0, y1, w, head, rows = self.fixture()
        rows[319]["exact_dot_delta"] = "0"
        with self.assertRaisesRegex(ValueError, "selected tied-head"):
            d.pair_account(d.PAIRS[0], h0, h1, y0, y1, w, head,
                           Fraction(1, 2), Fraction(3, 4), {0}, rows)

    def test_omitted_changed_coordinate_rejected(self):
        h0, h1, y0, y1, w, head, rows = self.fixture()
        with self.assertRaisesRegex(ValueError, "unchanged coordinate"):
            d.pair_account(d.PAIRS[0], h0, h1, y0, y1, w, head,
                           Fraction(1, 2), Fraction(3, 4), set(), rows)

    def test_integer_anchor_is_retained_ratio_not_sqrt(self):
        self.assertEqual(d.retained_anchor({"root_q24": 41335592, "mean_q48": 1708631221823978}),
                         Fraction(1 << 24, 41335592))
        for root in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                d.retained_anchor({"root_q24": root, "mean_q48": 1})

    def test_fp16_words_are_exact_and_nonfinite_rejected(self):
        self.assertEqual(d.words(np.array([0x0001, 0x8000, 0xbc00], dtype="<u2"), 3),
                         (Fraction(1, 1 << 24), Fraction(), Fraction(-1)))
        for array in (np.array([0x7c00], dtype="<u2"), np.zeros(1, dtype="<f2")):
            with self.assertRaises(ValueError):
                d.words(array, 1)

    def test_all_protected_fields_include_zero_sign_and_q24_width(self):
        old = {"stage18": np.array([0x8000], dtype="<u2"),
               "output_i": np.array([1 << 40], dtype="<i8")}
        for name in old:
            changed = {k: v.copy() for k, v in old.items()}
            changed[name][0] = 0
            with self.assertRaisesRegex(ValueError, "protected field"):
                d.preserve(old, changed)
        d.preserve(old, old)

    def test_bound_hash_and_byte_count_are_both_required(self):
        content = b"retained"
        pin = {"path": str(d.ROOT / "build/not-created"), "bytes": len(content),
               "sha256": d.hashlib.sha256(content).hexdigest()}
        with patch.object(Path, "read_bytes", return_value=content):
            self.assertEqual(d.bound_bytes(pin), content)
            for mutation in ({"sha256": "0" * 64}, {"bytes": len(content) + 1}):
                with self.assertRaises(ValueError):
                    d.bound_bytes({**pin, **mutation})

    def test_current_pin_splice_rejected_even_when_cached(self):
        path = str(d.SOURCE)
        current = {"path": path, "bytes": 12, "sha256": "a" * 64}
        with self.assertRaisesRegex(ValueError, "pin changed"):
            d.authenticate_pins({**current, "sha256": "b" * 64}, {path: current})

    def test_read_only_blocks_path_low_level_writes_and_external_dispatch(self):
        audit = d.new_audit()
        target = d.ROOT / "build/nearest-final-margin-must-not-exist"
        with d.read_only(audit):
            for call in (lambda: target.open("wb"), lambda: os.open(target, os.O_CREAT | os.O_WRONLY),
                         lambda: target.unlink(), lambda: subprocess.run(["true"])):
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit, {"evidence_writes": 3, "forbidden_dispatches": 1,
                                 "model_operator_calls": 0})

    def test_read_only_blocks_preloaded_model_function(self):
        module = types.ModuleType("ace3.forbidden_test_operator")
        exec("def run():\n    return 1\n", module.__dict__)
        audit = d.new_audit()
        with d.read_only(audit), self.assertRaisesRegex(RuntimeError, "operator calls"):
            module.run()
        self.assertEqual(audit["model_operator_calls"], 1)

    def test_unknown_is_one_json_with_nonadmission_and_no_fallback(self):
        output = io.StringIO()
        with patch.object(d, "authenticate", side_effect=ValueError("pin defect")), redirect_stdout(output):
            self.assertEqual(d.main(["--check"]), 1)
        result = json.loads(output.getvalue())
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertIn("pin defect", result["error"])
        self.assertFalse(result["candidate_admitted"])
        self.assertEqual(result["audit"], d.new_audit())

    def test_success_is_one_json_and_no_evidence_write(self):
        output = io.StringIO()
        with patch.object(d, "authenticate", return_value=()), patch.object(
                d, "report", return_value={"status": "COMPLETE"}), redirect_stdout(output):
            self.assertEqual(d.main(["--check"]), 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["audit"], d.new_audit())


if __name__ == "__main__":
    unittest.main()
