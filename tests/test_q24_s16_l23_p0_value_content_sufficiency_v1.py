import io
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import q24_s16_l23_p0_value_content_sufficiency_v1 as d


class ValueSufficiencyTests(unittest.TestCase):
    def fixture(self):
        actual = {key: np.zeros(896, dtype="<u2") for key in d.PROTECTED}
        for stage in (3, 6, 7):
            actual[f"stage{stage:02d}"] = np.zeros(128, dtype="<u2")
        actual["stage09"] = np.full(14, 0x3c00, dtype="<u2")
        for kind in ("k", "v"):
            actual["input_cache_" + kind] = np.empty((0, 128), dtype="<u2")
            actual["output_cache_" + kind] = np.zeros((1, 128), dtype="<u2")
        reference = {key: value.copy() for key, value in actual.items()}
        reference["stage03"][:] = reference["stage07"][:] = 0x3c00
        return actual, reference

    def rows(self):
        return [{"control": label, "reports": [{"stage": s, "status": "PASS"}
                                              for s in range(10, 19)]} for label in d.CONTROLS]

    def test_all_nine_pass(self):
        self.assertEqual(d.scientific_status(self.rows()), "PASS")

    def test_one_failure_rejects_rescue(self):
        rows = self.rows()
        rows[-1]["reports"][-1]["status"] = "FAIL"
        self.assertEqual(d.scientific_status(rows), "FAIL")

    def test_missing_control_is_defect(self):
        with self.assertRaises((RuntimeError, ValueError)):
            d.scientific_status(self.rows()[:-1])

    def test_reordered_controls_is_defect(self):
        with self.assertRaises((RuntimeError, ValueError)):
            d.scientific_status(self.rows()[::-1])

    def test_missing_stage_is_defect(self):
        rows = self.rows()
        rows[0]["reports"].pop()
        with self.assertRaises((RuntimeError, ValueError)):
            d.scientific_status(rows)

    def test_unknown_is_not_scientific_failure(self):
        rows = self.rows()
        rows[0]["reports"][0]["status"] = "UNKNOWN"
        with self.assertRaises((RuntimeError, ValueError)):
            d.scientific_status(rows)

    def test_only_stage07_substituted(self):
        actual, reference = self.fixture()
        with patch.object(d.layer.preflight.parent, "check_state"):
            _, arrays = d.prepare(actual, reference)
        d.protected(actual, arrays, reference)
        self.assertTrue(np.all(arrays["stage07"] == 0x3c00))
        self.assertTrue(np.all(arrays["output_cache_v"] == 0))
        self.assertFalse(arrays["stage07"].flags.writeable)
        self.assertNotIn("stage10", arrays)

    def test_nonunit_probability_rejected(self):
        actual, reference = self.fixture()
        actual["stage09"][0] = 0
        with self.assertRaises((RuntimeError, ValueError)):
            d.prepare(actual, reference)

    def test_cache_splice_rejected(self):
        actual, reference = self.fixture()
        actual["output_cache_v"][0, 0] = 1
        with patch.object(d.layer.preflight.parent, "check_state"):
            with self.assertRaises((RuntimeError, ValueError)):
                d.prepare(actual, reference)

    def test_gqa_value_mapping(self):
        arrays = {"stage07": np.arange(128, dtype="<f2").view("<u2"),
                  "stage09": np.full(14, 0x3c00, dtype="<u2")}
        stages = d.suffix_stages({}, {}, arrays)
        self.assertEqual(next(stages), 10)
        expected = np.repeat(arrays["stage07"].reshape(2, 64), 7, axis=0).reshape(-1)
        np.testing.assert_array_equal(arrays["stage10"], expected)

    def test_signed_paired_margins(self):
        old = np.zeros(34320, dtype="<f2")
        new = old.copy()
        ref = old.copy()
        old[319], new[319], ref[319] = 2, 3, 1
        rows = d.margin_rows(old.view("<u2"), new.view("<u2"),
                             {"logits_fp16": ref.view("<u2"), "logits_binary64": ref.astype("<f8")})
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]["retained_margin_change"], "1")
        self.assertEqual(rows[0]["substituted_margin_change"], "2")
        self.assertEqual(rows[0]["paired_intervention_delta"], "1")

    def test_forbidden_write_and_native_prefix(self):
        audit = {"file_write_opens": 0, "forbidden_calls": 0}
        with d.write_scope(d.OUTPUT, audit):
            with self.assertRaises((RuntimeError, ValueError)):
                io.open(d.ROOT / "forbidden.tmp", "w")
        with d.final.no_dispatch(audit):
            with self.assertRaises(RuntimeError):
                d.native._stages({}, 23, {}, {})
        self.assertEqual(audit["forbidden_calls"], 1)
