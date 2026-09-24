import io
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import q24_s16_l23_p0_s18_immediate_operand_combined_sufficiency_v1 as d


class CombinedImmediateOperandTests(unittest.TestCase):
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
        reference["stage12"][:] = 0x4000
        reference["stage17"][:] = 0x3c00
        reference["stage18"][:] = 0x4200
        return actual, reference

    def rows(self):
        return [{"control": label, "status": "PASS", "reports": [{"stage": 18, "status": "PASS"}],
                 "final_rescued": "NOT_DEFINED", "preservation_checks": {"all": True},
                 "source_operand_state_KV_lineage_checks": "PASS"} for label in d.CONTROLS]

    def test_all_nine_pass(self):
        self.assertEqual(d.scientific_status(self.rows()), "PASS")

    def test_one_failure_rejects_all_nine(self):
        rows = self.rows()
        rows[-1]["status"] = rows[-1]["reports"][0]["status"] = "FAIL"
        self.assertEqual(d.scientific_status(rows), "FAIL")

    def test_incomplete_reordered_or_extra_controls_rejected(self):
        for rows in (self.rows()[:-1], self.rows()[::-1], self.rows() + self.rows()[:1]):
            with self.assertRaises(ValueError):
                d.scientific_status(rows)

    def test_invalid_stage_or_unknown_not_numerical_failure(self):
        for stage, status in ((17, "PASS"), (18, "UNKNOWN")):
            rows = self.rows()
            rows[0]["status"] = status
            rows[0]["reports"] = [{"stage": stage, "status": status}]
            with self.assertRaises(ValueError):
                d.scientific_status(rows)

    def test_protection_defects_rejected(self):
        for field, value in (("preservation_checks", {"all": False}),
                             ("source_operand_state_KV_lineage_checks", "UNKNOWN")):
            rows = self.rows()
            rows[0][field] = value
            with self.assertRaises(ValueError):
                d.scientific_status(rows)

    def test_no_final_predicate(self):
        rows = self.rows()
        rows[0]["final_comparisons"] = {"error": 1000}
        self.assertEqual(d.scientific_status(rows), "PASS")
        rows[0]["final_rescued"] = "PASS"
        with self.assertRaises(ValueError):
            d.scientific_status(rows)

    def test_only_immediate_operands_changed(self):
        actual, reference = self.fixture()
        arrays = d.prepare(actual, reference)
        self.assertTrue(all(d.protected(actual, arrays, reference).values()))
        self.assertEqual(set(arrays), set(actual) | set(d.OPERAND_FIELDS)
                         | {"retained_" + key for key in d.prior.AFFECTED})
        np.testing.assert_array_equal(arrays["stage18_operand_h"], reference["stage12"])
        np.testing.assert_array_equal(arrays["stage17"], reference["stage17"])
        self.assertTrue(np.all(arrays["scratch_i"] == 0))
        self.assertTrue(all(not value.flags.writeable for value in arrays.values()))

    def test_signed_zero_lift_and_add_preserved(self):
        actual, reference = self.fixture()
        reference["stage12"][:] = reference["stage17"][:] = 0x8000
        arrays = d.prepare(actual, reference)
        state = {key: arrays["stage18_operand_" + key] for key in ("i", "z", "h")}
        audit = d.new_audit()
        audit["controls"].append(d.CONTROLS[0])
        with d.suffix_only(audit):
            successor = d.native.state.add(state, arrays["stage17"])
        expected = d.layer.prior.retained.transition_reference(state, reference["stage17"])
        d.layer.prior.retained.verify_parent(successor, expected)
        self.assertTrue(np.all(successor["h"] == 0x8000))
        self.assertTrue(np.all(successor["z"] == 1))

    def test_both_reference_shapes_dtypes_and_nonfinite_rejected(self):
        for key in ("stage12", "stage17"):
            for words in (np.zeros(895, dtype="<u2"), np.zeros(896, dtype="<f2"),
                          np.full(896, 0x7c00, dtype="<u2")):
                actual, reference = self.fixture()
                reference[key] = words
                with self.assertRaises(ValueError):
                    d.prepare(actual, reference)

    def test_missing_reference_rejected(self):
        for key in ("stage12", "stage17"):
            actual, reference = self.fixture()
            del reference[key]
            with self.assertRaises(KeyError):
                d.prepare(actual, reference)

    def test_kv_and_probability_splices_rejected(self):
        for key in ("output_cache_v", "output_cache_k", "stage09"):
            actual, reference = self.fixture()
            actual[key].flat[0] = 1
            with self.assertRaises(ValueError):
                d.prepare(actual, reference)

    def test_actual_input_scratch_output_state_corruption_rejected(self):
        for prefix in ("input", "scratch", "output"):
            actual, reference = self.fixture()
            actual[prefix + "_i"][0] = 1 << 24
            with self.assertRaises(ValueError):
                d.prepare(actual, reference)

    def test_retained_and_substitution_mutations_rejected(self):
        for key in ("input_i", "scratch_i", "stage07", "stage12", "stage16",
                    "retained_stage17", "retained_stage18", "stage17",
                    *d.OPERAND_FIELDS):
            actual, reference = self.fixture()
            arrays = d.prepare(actual, reference)
            arrays[key] = np.ones_like(arrays[key])
            with self.assertRaises(ValueError):
                d.protected(actual, arrays, reference)

    def test_reserved_payload_fields_rejected(self):
        for key in (*d.OPERAND_FIELDS, "retained_stage17"):
            actual, reference = self.fixture()
            actual[key] = actual["stage17"]
            with self.assertRaises(ValueError):
                d.prepare(actual, reference)

    def test_threshold_and_complete_failure_summary(self):
        actual = np.zeros(896, dtype="<u2")
        reference = {"stage18": actual.copy()}
        binary64 = np.zeros(896, dtype="<f8")
        for bits, status, count in ((0x3000, "PASS", 0), (0x3001, "FAIL", 1)):
            actual[62] = bits
            report = d.layer.stage_report(18, {"stage18": actual}, reference, binary64, None)
            summary = d.report_summary(report)
            self.assertEqual(summary["status"], status)
            self.assertEqual(summary["excess_budget"], "1/8")
            self.assertEqual(summary["failure_count"], count)
            self.assertEqual(len(summary["failed_coordinates"]), count)
            if count:
                self.assertEqual(summary["failed_coordinates"][0]["index"], 62)
            self.assertEqual(len(report["binary64_v1"]["rows"]), 896)

    def test_forbidden_decoder_reference_external_and_extra_dispatch(self):
        actual, reference = self.fixture()
        arrays = d.prepare(actual, reference)
        operand = {key: arrays["stage18_operand_" + key] for key in ("i", "z", "h")}
        audit = d.new_audit()
        audit["controls"].append(d.CONTROLS[0])
        with d.suffix_only(audit):
            for function in (d.native._stages, d.base.suffix_stages, d.native.projection,
                             d.native.rne, d.native.toward_zero, d.layer.expected_stage,
                             d.layer.prior.local.local_reference, d.final.subprocess.Popen):
                with self.assertRaises(RuntimeError):
                    function()
            d.native.state.add(operand, arrays["stage17"])
            with self.assertRaises(ValueError):
                d.native.state.add(operand, arrays["stage17"])
        self.assertEqual(audit["forbidden_calls"], 8)

    def test_exact_dispatch_and_write_counts(self):
        audit = d.new_audit()
        audit.update(controls=list(d.CONTROLS),
                     stage_dispatches=[[label, 23, 0, 18] for label in d.CONTROLS],
                     stage12_operand_lifts=9, stage18_oracle_invocations=9,
                     final_rmsnorm_invocations=9, lm_head_invocations=9,
                     top_k_invocations=27, file_write_opens=11)
        d.check_counts(audit)
        for key in ("stage12_operand_lifts", "native_S0_S17_invocations", "file_write_opens"):
            with self.assertRaises(ValueError):
                d.check_counts({**audit, key: audit[key] + 1})

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

    def test_control_uses_both_reference_operands_not_actual_scratch(self):
        actual, reference = self.fixture()
        arrays = d.prepare(actual, reference)
        scores = np.zeros(34320, dtype="<u2")
        refs = {"logits_fp16": scores, "logits_binary64": scores.astype("<f8")}
        audit = d.new_audit()
        with patch.object(d.final, "rmsnorm", return_value=(reference["stage18"], {})) as norm, \
                patch.object(d.final, "logits", return_value=scores), \
                patch.object(d.final, "comparisons", return_value={"diagnostic": True}), \
                d.suffix_only(audit):
            row = d.run_control(d.CONTROLS[0], actual, arrays, reference,
                                np.full(896, 3., dtype="<f8"), [None, None], refs,
                                {"logits": scores}, audit)
        np.testing.assert_array_equal(arrays["stage18"], reference["stage18"])
        self.assertTrue(np.all(arrays["output_i"] == 3 << 24))
        self.assertTrue(np.all(arrays["scratch_i"] == 0))
        self.assertIs(norm.call_args.args[0], arrays["stage18"])
        self.assertEqual(row["status"], "PASS")
        self.assertEqual(row["final_rescued"], "NOT_DEFINED")
        self.assertEqual(len(row["paired_final_margins"]), 6)
        self.assertEqual(audit["stage_dispatches"], [[d.CONTROLS[0], 23, 0, 18]])
