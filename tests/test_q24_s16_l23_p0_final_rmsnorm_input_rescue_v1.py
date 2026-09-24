import io
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import q24_s16_l23_p0_final_rmsnorm_input_rescue_v1 as d


class FinalInputTests(unittest.TestCase):
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
        reference["stage18"][:] = 0x3c00
        return actual, reference

    def rows_audit(self):
        rows = [{"control": label, "rescued": "NOT_DEFINED", "preservation_checks": {"all": True}}
                for label in d.CONTROLS]
        audit = {
            "controls": list(d.CONTROLS), "final_rmsnorm_invocations": 9,
            "lm_head_invocations": 9, "top_k_invocations": 27,
            "native_L0_L23_invocations": 0, "reference_producer_invocations": 0,
            "external_invocations": 0, "retained_evidence_writes": 0,
            "forbidden_calls": 0, "forbidden_mutations": 0, "file_write_opens": 11,
        }
        return rows, audit

    def test_only_final_input_substituted(self):
        actual, reference = self.fixture()
        arrays = d.prepare(actual, reference)
        self.assertEqual(set(arrays), set(actual) | {"final_input_hidden"})
        self.assertTrue(all(d.protected(actual, arrays, reference).values()))
        self.assertTrue(np.all(arrays["stage18"] == 0))
        np.testing.assert_array_equal(arrays["final_input_hidden"], reference["stage18"])
        self.assertFalse(arrays["final_input_hidden"].flags.writeable)

    def test_signed_zero_preserved(self):
        actual, reference = self.fixture()
        reference["stage18"][0] = 0x8000
        self.assertEqual(d.prepare(actual, reference)["final_input_hidden"][0], 0x8000)

    def test_nonfinite_input_rejected(self):
        actual, reference = self.fixture()
        reference["stage18"][0] = 0x7c00
        with self.assertRaises((RuntimeError, ValueError)):
            d.prepare(actual, reference)

    def test_wrong_input_shape_rejected(self):
        actual, reference = self.fixture()
        reference["stage18"] = reference["stage18"][:-1]
        with self.assertRaises((RuntimeError, ValueError)):
            d.prepare(actual, reference)

    def test_cache_splice_rejected(self):
        actual, reference = self.fixture()
        actual["output_cache_v"][0, 0] = 1
        with self.assertRaises((RuntimeError, ValueError)):
            d.prepare(actual, reference)

    def test_probability_splice_rejected(self):
        actual, reference = self.fixture()
        actual["stage09"][0] = 0
        with self.assertRaises((RuntimeError, ValueError)):
            d.prepare(actual, reference)

    def test_q24_output_splice_rejected(self):
        actual, reference = self.fixture()
        actual["output_i"][0] = 1 << 24
        with self.assertRaises((RuntimeError, ValueError)):
            d.prepare(actual, reference)

    def test_historical_output_mutation_rejected(self):
        actual, reference = self.fixture()
        arrays = d.prepare(actual, reference)
        arrays["stage18"] = reference["stage18"].copy()
        with self.assertRaises((RuntimeError, ValueError)):
            d.protected(actual, arrays, reference)

    def test_substitution_mutation_rejected(self):
        actual, reference = self.fixture()
        arrays = d.prepare(actual, reference)
        arrays["final_input_hidden"] = actual["stage18"].copy()
        with self.assertRaises((RuntimeError, ValueError)):
            d.protected(actual, arrays, reference)

    def test_diagnostics_not_a_rescue_predicate(self):
        rows, audit = self.rows_audit()
        for row in rows:
            row["final_comparisons"] = {"arbitrary_diagnostic_error": 999}
        d.check_delivery(rows, audit)
        for forbidden in ("PASS", "FAIL", "UNKNOWN"):
            rows[0]["rescued"] = forbidden
            with self.subTest(forbidden=forbidden), self.assertRaises((RuntimeError, ValueError)):
                d.check_delivery(rows, audit)

    def test_missing_or_reordered_control_rejected(self):
        rows, audit = self.rows_audit()
        for changed in (rows[:-1], rows[::-1]):
            with self.assertRaises((RuntimeError, ValueError)):
                d.check_delivery(changed, audit)

    def test_preservation_defect_rejected(self):
        rows, audit = self.rows_audit()
        rows[0]["preservation_checks"]["all"] = False
        with self.assertRaises((RuntimeError, ValueError)):
            d.check_delivery(rows, audit)

    def test_dispatch_census_enforced(self):
        rows, audit = self.rows_audit()
        audit["top_k_invocations"] += 1
        with self.assertRaises((RuntimeError, ValueError)):
            d.check_delivery(rows, audit)

    def test_final_artifact_census_serializable(self):
        names = ["command.json", "compile_tests.json", "result.json"]
        names += [label + ".npz" for label in d.CONTROLS]
        with patch.object(d, "OUTPUT") as output:
            output.iterdir.return_value = [d.ROOT / name for name in reversed(names)]
            d.check_artifact_census()
        output.iterdir.assert_called_once_with()

    def test_final_artifact_census_rejects_missing_or_extra(self):
        names = ["command.json", "compile_tests.json", "result.json"]
        names += [label + ".npz" for label in d.CONTROLS]
        for changed in (names[:-1], [name for name in names if name != "result.json"],
                        names + ["unexpected.npz"], names + ["failure.json"]):
            with self.subTest(names=changed), patch.object(d, "OUTPUT") as output:
                output.iterdir.return_value = [d.ROOT / name for name in changed]
                with self.assertRaisesRegex(ValueError, "final artifact census mismatch"):
                    d.check_artifact_census()

    def test_occupied_output_zero_dispatch(self):
        with patch.object(d, "OUTPUT", d.ROOT), patch.object(d.final, "rmsnorm") as norm:
            with self.assertRaises((RuntimeError, ValueError)):
                d.fresh_output()
            norm.assert_not_called()

    def test_forbidden_writes_and_mutations(self):
        audit = {"file_write_opens": 0, "forbidden_mutations": 0}
        with d.write_scope(audit):
            with self.assertRaises((RuntimeError, ValueError)):
                io.open(d.SOURCE, "w")
            with self.assertRaises(RuntimeError):
                d.os.unlink(d.SOURCE)
        self.assertEqual(audit, {"file_write_opens": 0, "forbidden_mutations": 1})

    def test_forbidden_decoder_and_external_dispatch(self):
        audit = {"forbidden_calls": 0}
        with d.final_only(audit):
            for function in (d.native._stages, d.base.suffix_stages,
                             d.native.projection, d.native.state.add, d.final.subprocess.Popen):
                with self.assertRaises(RuntimeError):
                    function()
        self.assertEqual(audit["forbidden_calls"], 5)

    def test_run_control_uses_only_selected_boundary(self):
        actual, reference = self.fixture()
        arrays = d.prepare(actual, reference)
        scores = np.zeros(34320, dtype="<u2")
        refs = {"logits_fp16": scores, "logits_binary64": scores.astype("<f8")}
        _, audit = self.rows_audit()
        audit.update(controls=[], final_rmsnorm_invocations=0,
                     lm_head_invocations=0, top_k_invocations=0)
        with d.final_only(audit), \
                patch.object(d.final, "rmsnorm", return_value=(reference["stage18"], {})) as norm, \
                patch.object(d.final, "logits", return_value=scores), \
                patch.object(d.final, "comparisons", return_value={"diagnostic": True}):
            row = d.run_control(d.CONTROLS[0], actual, arrays, reference, [None, None],
                                refs, {"logits": scores}, audit)
        self.assertIs(norm.call_args.args[0], arrays["final_input_hidden"])
        self.assertEqual(row["rescued"], "NOT_DEFINED")
        self.assertEqual(audit["top_k_invocations"], 3)
        self.assertEqual(len(row["paired_final_margins"]), 6)

    def test_existing_oracle_diagnostics_and_margin_signs(self):
        old = np.zeros(34320, dtype="<f2")
        new, ref = old.copy(), old.copy()
        old[319], new[319], ref[319] = 2, 3, 1
        metric = d.final.comparison(new.view("<u2"), ref.astype("<f8"), ref.view("<u2"))
        self.assertEqual(metric["binary64_max_absolute_error"], "2")
        self.assertEqual(metric["reviewed_fp16_mismatch_count"], 1)
        self.assertIn("diagnostic", metric["comparison_scope"])
        rows = d.base.margin_rows(old.view("<u2"), new.view("<u2"),
                                 {"logits_fp16": ref.view("<u2"), "logits_binary64": ref.astype("<f8")})
        self.assertEqual(rows[0]["retained_margin_change"], "1")
        self.assertEqual(rows[0]["substituted_margin_change"], "2")
        self.assertEqual(rows[0]["paired_intervention_delta"], "1")
