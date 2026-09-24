import io
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import q24_s16_l23_p0_inherited_residual_contrast_v1 as d


class InheritedResidualTests(unittest.TestCase):
    def fixture(self):
        actual = {f"stage{s:02d}": np.zeros(n, dtype="<u2")
                  for s, n in d.layer.prior.local.SIZES.items()}
        actual["stage09"][:] = 0x3c00
        for prefix in ("input", "scratch", "output"):
            actual[prefix + "_i"] = np.zeros(896, dtype="<i8")
            actual[prefix + "_z"] = np.zeros(896, dtype="u1")
        actual["input_hidden"] = np.zeros(896, dtype="<u2")
        for kind in ("k", "v"):
            actual["input_cache_" + kind] = np.empty((0, 128), dtype="<u2")
            actual["output_cache_" + kind] = np.zeros((1, 128), dtype="<u2")
        actual["s16_unrounded_binary64"] = np.zeros(4864, dtype="<f8")
        reference = {key: value.copy() for key, value in actual.items()}
        reference["input_hidden"][:] = 0x3c00
        return actual, reference

    def rows(self):
        return [{"control": label, "preservation_checks": {"input_hidden": True},
                 "reports": [{"stage": s, "status": "PASS"} for s in d.STAGES]}
                for label in d.CONTROLS]

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

    def test_preservation_defect_is_not_pass(self):
        rows = self.rows()
        rows[0]["preservation_checks"]["input_hidden"] = False
        with self.assertRaises((RuntimeError, ValueError)):
            d.scientific_status(rows)

    def test_only_residual_operand_substituted(self):
        actual, reference = self.fixture()
        parent, arrays = d.prepare(actual, reference)
        self.assertTrue(all(d.protected(actual, arrays, reference, parent).values()))
        np.testing.assert_array_equal(parent["h"], reference["input_hidden"])
        self.assertTrue(np.all(arrays["input_hidden"] == 0))
        self.assertTrue(np.all(parent["i"] == 1 << 24))
        self.assertFalse(parent["i"].flags.writeable)
        self.assertNotIn("stage12", arrays)

    def test_signed_zero_lift(self):
        actual, reference = self.fixture()
        reference["input_hidden"][0] = 0x8000
        parent, _ = d.prepare(actual, reference)
        self.assertEqual((parent["i"][0], parent["z"][0], parent["h"][0]), (0, 1, 0x8000))

    def test_nonunit_probability_rejected(self):
        actual, reference = self.fixture()
        actual["stage09"][0] = 0
        with self.assertRaises((RuntimeError, ValueError)):
            d.prepare(actual, reference)

    def test_cache_splice_rejected(self):
        actual, reference = self.fixture()
        actual["output_cache_v"][0, 0] = 1
        with self.assertRaises((RuntimeError, ValueError)):
            d.prepare(actual, reference)

    def test_corrupt_retained_state_rejected(self):
        actual, reference = self.fixture()
        actual["input_i"][0] = 1 << 24
        with self.assertRaises((RuntimeError, ValueError)):
            d.prepare(actual, reference)

    def test_nonfinite_reference_rejected(self):
        actual, reference = self.fixture()
        reference["input_hidden"][0] = 0x7c00
        with self.assertRaises((RuntimeError, ValueError)):
            d.prepare(actual, reference)

    def test_attention_mutation_rejected(self):
        actual, reference = self.fixture()
        parent, arrays = d.prepare(actual, reference)
        arrays["stage11"] = np.ones(896, dtype="<u2")
        with self.assertRaises((RuntimeError, ValueError)):
            d.protected(actual, arrays, reference, parent)

    def test_complete_cone_keeps_attention_and_uses_intervened_scratch(self):
        actual, reference = self.fixture()
        parent, arrays = d.prepare(actual, reference)
        tensors = {"model.layers.23.post_attention_layernorm.weight": np.ones(896, dtype="<f2")}

        def projection(tensors, name, words):
            self.assertIn(name.rsplit(".", 1)[-1], ("gate_proj", "up_proj", "down_proj"))
            return np.zeros(896 if name.endswith("down_proj") else 4864, dtype="<u2")

        with d.final.no_dispatch({"forbidden_calls": 0}), patch.object(d.native, "projection", projection):
            self.assertEqual(list(d.suffix_stages(tensors, parent, arrays)), list(d.STAGES))
        for stage in (12, 18):
            d.expected_stage(stage, arrays, parent, tensors)
        np.testing.assert_array_equal(arrays["stage18"], reference["input_hidden"])
        self.assertTrue(all(d.protected(actual, arrays, reference, parent).values()))

    def test_signed_paired_margins(self):
        old = np.zeros(34320, dtype="<f2")
        new, ref = old.copy(), old.copy()
        old[319], new[319], ref[319] = 2, 3, 1
        rows = d.base.margin_rows(old.view("<u2"), new.view("<u2"),
                                 {"logits_fp16": ref.view("<u2"), "logits_binary64": ref.astype("<f8")})
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]["retained_margin_change"], "1")
        self.assertEqual(rows[0]["substituted_margin_change"], "2")
        self.assertEqual(rows[0]["paired_intervention_delta"], "1")

    def test_forbidden_write_and_native_prefix(self):
        audit = {"file_write_opens": 0, "forbidden_calls": 0}
        with d.base.write_scope(d.OUTPUT, audit):
            with self.assertRaises((RuntimeError, ValueError)):
                io.open(d.ROOT / "forbidden.tmp", "w")
        with d.final.no_dispatch(audit):
            with self.assertRaises(RuntimeError):
                d.native._stages({}, 23, {}, {})
        self.assertEqual(audit["forbidden_calls"], 1)
