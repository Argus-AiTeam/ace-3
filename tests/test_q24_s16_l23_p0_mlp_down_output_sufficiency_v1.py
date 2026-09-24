import io
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import q24_s16_l23_p0_mlp_down_output_sufficiency_v1 as d


class MlpDownOutputTests(unittest.TestCase):
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
        reference["stage17"][:] = 0x3c00
        return actual, reference

    def rows(self):
        return [{"control": label, "status": "PASS", "reports": [{"stage": 18, "status": "PASS"}],
                 "final_rescued": "NOT_DEFINED", "preservation_checks": {"all": True},
                 "source_operand_state_KV_lineage_checks": "PASS"} for label in d.CONTROLS]

    def test_all_nine_pass(self):
        self.assertEqual(d.scientific_status(self.rows()), "PASS")

    def test_one_failure_rejects_all_nine_rescue(self):
        rows = self.rows()
        rows[-1]["status"] = rows[-1]["reports"][0]["status"] = "FAIL"
        self.assertEqual(d.scientific_status(rows), "FAIL")

    def test_missing_or_reordered_controls_rejected(self):
        for rows in (self.rows()[:-1], self.rows()[::-1]):
            with self.assertRaises(ValueError):
                d.scientific_status(rows)

    def test_missing_or_extra_stage_rejected(self):
        for reports in ([], [{"stage": 17, "status": "PASS"}, {"stage": 18, "status": "PASS"}]):
            rows = self.rows()
            rows[0]["reports"] = reports
            with self.assertRaises(ValueError):
                d.scientific_status(rows)

    def test_unknown_is_not_numerical_failure(self):
        rows = self.rows()
        rows[0]["status"] = rows[0]["reports"][0]["status"] = "UNKNOWN"
        with self.assertRaises(ValueError):
            d.scientific_status(rows)

    def test_source_or_preservation_defect_rejected(self):
        for field, value in (("preservation_checks", {"all": False}),
                             ("source_operand_state_KV_lineage_checks", "UNKNOWN")):
            rows = self.rows()
            rows[0][field] = value
            with self.assertRaises(ValueError):
                d.scientific_status(rows)

    def test_final_diagnostics_do_not_classify_rescue(self):
        rows = self.rows()
        rows[0]["final_comparisons"] = {"error": 1000}
        self.assertEqual(d.scientific_status(rows), "PASS")
        rows[0]["final_rescued"] = "PASS"
        with self.assertRaises(ValueError):
            d.scientific_status(rows)

    def test_only_stage17_substituted(self):
        actual, reference = self.fixture()
        arrays = d.prepare(actual, reference)
        self.assertTrue(all(d.protected(actual, arrays, reference).values()))
        self.assertEqual(set(arrays), set(actual) | {"retained_" + key for key in d.AFFECTED})
        np.testing.assert_array_equal(arrays["stage17"], reference["stage17"])
        self.assertFalse(arrays["stage17"].flags.writeable)
        self.assertTrue(np.all(arrays["retained_stage17"] == 0))

    def test_signed_zero_substitution_preserved(self):
        actual, reference = self.fixture()
        reference["stage17"][0] = 0x8000
        self.assertEqual(d.prepare(actual, reference)["stage17"][0], 0x8000)

    def test_invalid_reference_shape_dtype_or_nonfinite_rejected(self):
        for words in (np.zeros(895, dtype="<u2"), np.zeros(896, dtype="<f2"),
                      np.full(896, 0x7c00, dtype="<u2")):
            actual, reference = self.fixture()
            reference["stage17"] = words
            with self.assertRaises(ValueError):
                d.prepare(actual, reference)

    def test_kv_source_splice_rejected(self):
        for key in ("output_cache_v", "output_cache_k"):
            actual, reference = self.fixture()
            actual[key][0, 0] = 1
            with self.assertRaises(ValueError):
                d.prepare(actual, reference)

    def test_nonunit_probability_rejected(self):
        actual, reference = self.fixture()
        actual["stage09"][0] = 0
        with self.assertRaises(ValueError):
            d.prepare(actual, reference)

    def test_corrupt_input_scratch_or_output_state_rejected(self):
        for prefix in ("input", "scratch", "output"):
            actual, reference = self.fixture()
            actual[prefix + "_i"][0] = 1 << 24
            with self.assertRaises(ValueError):
                d.prepare(actual, reference)

    def test_protected_attention_inherited_and_mlp_mutations_rejected(self):
        for key in ("stage07", "stage11", "input_i", "scratch_i", "stage16", "retained_stage18"):
            actual, reference = self.fixture()
            arrays = d.prepare(actual, reference)
            arrays[key] = np.ones_like(arrays[key])
            with self.assertRaises(ValueError):
                d.protected(actual, arrays, reference)

    def test_substitution_mutation_rejected(self):
        actual, reference = self.fixture()
        arrays = d.prepare(actual, reference)
        arrays["stage17"] = actual["stage17"]
        with self.assertRaises(ValueError):
            d.protected(actual, arrays, reference)

    def test_reserved_payload_collision_rejected(self):
        actual, reference = self.fixture()
        actual["retained_stage17"] = actual["stage17"]
        with self.assertRaises(ValueError):
            d.prepare(actual, reference)

    def test_native_s18_matches_independent_rational_oracle_with_wide_state(self):
        words = np.full(896, 0x3c00, dtype="<u2")
        scratch = d.native.state.lift(words)
        scratch["i"] += 1
        addend = np.full(896, 0xbc00, dtype="<u2")
        audit = d.new_audit()
        audit["controls"].append(d.CONTROLS[0])
        with d.suffix_only(audit):
            actual = d.native.state.add(scratch, addend)
        expected = d.layer.prior.retained.transition_reference(scratch, addend)
        d.layer.prior.retained.verify_parent(actual, expected)
        self.assertTrue(np.all(actual["i"] == 1))
        self.assertTrue(np.all(actual["h"] == 1))
        self.assertEqual(audit["stage_dispatches"], [[d.CONTROLS[0], 23, 0, 18]])

    def test_unchanged_s18_threshold_at_and_above_one_eighth(self):
        actual = np.zeros(896, dtype="<u2")
        reference = np.zeros(896, dtype="<u2")
        binary64 = np.zeros(896, dtype="<f8")
        for bits, status in ((0x3000, "PASS"), (0x3001, "FAIL")):
            actual[0] = bits
            report = d.layer.stage_report(18, {"stage18": actual},
                                          {"stage18": reference}, binary64, None)
            self.assertEqual(report["status"], status)
            self.assertEqual(d.report_summary(report)["excess_budget"], "1/8")
            self.assertEqual(report["fp16_role"], "independent-whole-FP16-trajectory-diagnostic")

    def test_forbidden_decoder_reference_and_external_dispatch(self):
        audit = d.new_audit()
        with d.suffix_only(audit):
            for function in (d.native._stages, d.base.suffix_stages, d.native.projection,
                             d.native.rne, d.native.toward_zero, d.layer.expected_stage,
                             d.layer.prior.local.local_reference, d.final.subprocess.Popen):
                with self.assertRaises(RuntimeError):
                    function()
        self.assertEqual(audit["forbidden_calls"], 8)

    def test_extra_s18_dispatch_rejected(self):
        actual, _ = self.fixture()
        state = d.layer.prior.retained.state_from(actual, "scratch", "stage12")
        audit = d.new_audit()
        audit["controls"].append(d.CONTROLS[0])
        with d.suffix_only(audit):
            d.native.state.add(state, actual["stage17"])
            with self.assertRaises(ValueError):
                d.native.state.add(state, actual["stage17"])

    def test_exact_dispatch_and_write_census(self):
        audit = d.new_audit()
        audit.update(controls=list(d.CONTROLS),
                     stage_dispatches=[[label, 23, 0, 18] for label in d.CONTROLS],
                     stage18_oracle_invocations=9, final_rmsnorm_invocations=9,
                     lm_head_invocations=9, top_k_invocations=27, file_write_opens=11)
        d.check_counts(audit)
        for field in ("top_k_invocations", "native_S0_S17_invocations", "file_write_opens"):
            with self.assertRaises(ValueError):
                d.check_counts({**audit, field: audit[field] + 1})

    def test_occupied_output_refuses_before_dispatch(self):
        with patch.object(d, "OUTPUT", d.ROOT), patch.object(d.native.state, "add") as add:
            with self.assertRaises(ValueError):
                d.fresh_output()
            add.assert_not_called()

    def test_write_and_mutation_guards(self):
        audit = d.new_audit()
        with d.write_scope(audit):
            with self.assertRaises(ValueError):
                io.open(d.SOURCE, "w")
            with self.assertRaises(RuntimeError):
                d.os.unlink(d.SOURCE)
        self.assertEqual(audit["file_write_opens"], 0)
        self.assertEqual(audit["forbidden_mutations"], 1)

    def test_control_uses_actual_scratch_and_substituted_stage17(self):
        actual, reference = self.fixture()
        arrays = d.prepare(actual, reference)
        scores = np.zeros(34320, dtype="<u2")
        refs = {"logits_fp16": scores, "logits_binary64": scores.astype("<f8")}
        audit = d.new_audit()
        with patch.object(d.final, "rmsnorm", return_value=(reference["stage17"], {})) as norm, \
                patch.object(d.final, "logits", return_value=scores), \
                patch.object(d.final, "comparisons", return_value={"diagnostic": True}), \
                d.suffix_only(audit):
            row = d.run_control(d.CONTROLS[0], actual, arrays, reference,
                                np.ones(896, dtype="<f8"), [None, None], refs,
                                {"logits": scores}, audit)
        np.testing.assert_array_equal(arrays["stage18"], reference["stage17"])
        self.assertTrue(np.all(arrays["output_i"] == 1 << 24))
        self.assertIs(norm.call_args.args[0], arrays["stage18"])
        self.assertEqual(row["status"], "PASS")
        self.assertEqual(row["final_rescued"], "NOT_DEFINED")
        self.assertEqual(len(row["paired_final_margins"]), 6)
        self.assertEqual(audit["final_rmsnorm_invocations"], 1)
