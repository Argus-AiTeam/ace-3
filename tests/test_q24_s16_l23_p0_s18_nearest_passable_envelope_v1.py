import copy
from fractions import Fraction
import io
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import q24_s16_l23_p0_s18_nearest_passable_envelope_v1 as d


class EnvelopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = d.diagnostic.finite_table()

    def scalar(self, word=0x3c00, reference=2.0):
        row = d.diagnostic.scalar(word, 0, word, word, reference, self.table)
        row["index"] = 0
        return row

    def fixture(self):
        row = self.scalar()
        actual = {"stage18": np.full(896, 0x3c00, dtype="<u2"),
                  "output_i": np.full(896, 1 << 24, dtype="<i8"),
                  "output_z": np.zeros(896, dtype="u1"),
                  "stage17": np.zeros(896, dtype="<u2"),
                  "output_cache_v": np.zeros((1, 128), dtype="<u2"),
                  "final_logits": np.zeros(34320, dtype="<u2")}
        control = {"rows": [{**row, "index": i} for i in range(896)]}
        reference = np.full(896, 2.0, dtype="<f8")
        return actual, control, reference

    def rows_audit(self):
        rows = [{"control": label, "S18_status": "PASS", "S18_binary64_failures": 0,
                 "coordinates": 896, "final_rescued": "NOT_DEFINED",
                 "preservation_checks": {"all": True},
                 "changed_coordinates": 110, "retained_S18_failures": 110}
                for label in d.CONTROLS]
        audit = d.new_audit()
        audit.update(controls=list(d.CONTROLS), s18_word_interventions=9,
                     scalar_gate_evaluations=16128, final_rmsnorm_invocations=9,
                     lm_head_invocations=9, top_k_invocations=27, file_write_opens=11)
        return rows, audit

    def test_nearest_passable_matches_exhaustive_oracle(self):
        for word, reference in ((0x3c00, 2.0), (0xc000, -1.0), (0x7bff, 65488.0),
                                (0x0001, -0.25), (0xbc00, 1.00048828125)):
            with self.subTest(word=word, reference=reference):
                row = self.scalar(word, reference)
                selected, report = d.select_word(word, row, reference, self.table)
                value, target = Fraction(float(np.array(word, dtype="<u2").view("<f2"))), Fraction(reference)
                decoded = [(Fraction(float(np.array(w, dtype="<u2").view("<f2"))), w)
                           for w in range(65536) if w & 0x7c00 != 0x7c00]
                floor = min(abs(v - target) for v, _ in decoded)
                passable = [(v, w) for v, w in decoded if abs(v - target) - floor <= Fraction(1, 8)]
                expected = min(passable, key=lambda p: (abs(p[0] - value), p[0], p[1]))[1]
                self.assertEqual(selected, word if row["accepted"] else expected)
                self.assertTrue(report["intervened_gate"]["accepted"])

    def test_exact_budget_is_inclusive(self):
        row = self.scalar(0x3f80, 2.0)
        selected, report = d.select_word(0x3f80, row, 2.0, self.table)
        self.assertEqual(selected, 0x3f80)
        self.assertEqual(report["threshold_margin"], "0")
        self.assertFalse(report["changed"])

    def test_passing_signed_zero_is_preserved_despite_word_tie(self):
        for word in (0, 0x8000):
            selected, report = d.select_word(word, self.scalar(word, 0.0), 0.0, self.table)
            self.assertEqual(selected, word)
            self.assertEqual(report["nearest_passable_ties"], ["0000", "8000"])
            self.assertTrue(report["tie_present"])

    def test_failing_zero_endpoint_tie_uses_unsigned_word(self):
        word = 0xbc00
        selected, report = d.select_word(word, self.scalar(word, 0.125), 0.125, self.table)
        self.assertEqual(selected, 0)
        self.assertEqual(report["nearest_passable_ties"], ["0000", "8000"])
        self.assertEqual(report["signed_delta"], "1")

    def test_equal_distance_uses_exact_value_before_word(self):
        table = ((Fraction(-1), Fraction(1)), (0xbc00, 0x3c00))
        words, delta = d.diagnostic.nearest_words(Fraction(0), table)
        self.assertEqual(words, [0xbc00, 0x3c00])
        self.assertEqual(delta, 1)
        self.assertEqual(min(words, key=lambda w: (
            abs(d.rational.fp16_value(w)), d.rational.fp16_value(w), w)), 0xbc00)

    def test_changed_interval_rejected(self):
        row = self.scalar()
        row["passable_interval"]["first_index"] += 1
        with self.assertRaisesRegex(ValueError, "interval defect"):
            d.select_word(0x3c00, row, 2.0, self.table)

    def test_changed_tie_set_rejected(self):
        row = self.scalar()
        row["nearest_passable_words_to_actual"] = ["3c00"]
        with self.assertRaisesRegex(ValueError, "tie set defect"):
            d.select_word(0x3c00, row, 2.0, self.table)

    def test_reference_or_budget_splice_rejected(self):
        for field, value in (("reference_binary64_hex", "0x0.0p+0"), ("excess_budget", "1/4"),
                             ("accepted", True), ("passable_signed_changes", ["0"])):
            row = self.scalar()
            row[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                d.select_word(0x3c00, row, 2.0, self.table)

    def test_nonfinite_inputs_rejected(self):
        for word, reference in ((0x7c00, 2.0), (0x3c00, float("inf"))):
            with self.assertRaises(ValueError):
                d.select_word(word, self.scalar(), reference, self.table)

    def test_intervention_keeps_all_retained_fields(self):
        actual, row, ref = self.fixture()
        arrays, reports = d.intervene(actual, row, ref, self.table)
        self.assertTrue(all(d.preserve(actual, arrays).values()))
        self.assertEqual(set(arrays), set(actual) | {"intervened_stage18"})
        self.assertEqual(len(reports), 896)
        self.assertFalse(arrays["intervened_stage18"].flags.writeable)
        self.assertTrue(np.all(arrays["intervened_stage18"] == 0x3f80))
        self.assertTrue(np.all(actual["stage18"] == 0x3c00))

    def test_mixed_pass_fail_changes_only_failures(self):
        actual, row, ref = self.fixture()
        actual["stage18"][0] = 0x8000
        ref[0] = 0
        row["rows"][0] = self.scalar(0x8000, 0.0)
        arrays, reports = d.intervene(actual, row, ref, self.table)
        self.assertEqual(arrays["intervened_stage18"][0], 0x8000)
        self.assertEqual(sum(r["changed"] for r in reports), 895)

    def test_coordinate_census_and_shape_rejected(self):
        actual, row, ref = self.fixture()
        for changed, refs in (({"rows": row["rows"][::-1]}, ref), (row, ref[:-1])):
            with self.assertRaises(ValueError):
                d.intervene(actual, changed, refs, self.table)

    def test_retained_mutation_rejected(self):
        actual, _, _ = self.fixture()
        for key in actual:
            changed = copy.deepcopy(actual)
            changed[key].flat[0] += 1
            with self.subTest(field=key), self.assertRaises(ValueError):
                d.preserve(actual, changed)

    def test_reserved_intervention_fields_rejected(self):
        actual, row, ref = self.fixture()
        actual["intervened_stage18"] = actual["stage18"]
        with self.assertRaises(ValueError):
            d.intervene(actual, row, ref, self.table)

    def test_delivery_is_scalar_only(self):
        rows, audit = self.rows_audit()
        d.check_delivery(rows, audit)
        rows[0]["final_comparisons"] = {"error": 999}
        d.check_delivery(rows, audit)
        rows[0]["final_rescued"] = "PASS"
        with self.assertRaises(ValueError):
            d.check_delivery(rows, audit)

    def test_missing_reordered_or_failed_control_rejected(self):
        rows, audit = self.rows_audit()
        for changed in (rows[:-1], rows[::-1]):
            with self.assertRaises(ValueError):
                d.check_delivery(changed, audit)
        rows[0]["S18_binary64_failures"] = 1
        with self.assertRaises(ValueError):
            d.check_delivery(rows, audit)

    def test_exact_dispatch_counts_enforced(self):
        rows, audit = self.rows_audit()
        for key in audit:
            if key == "controls":
                continue
            changed = {**audit, key: audit[key] + 1}
            with self.subTest(counter=key), self.assertRaises(ValueError):
                d.check_delivery(rows, changed)

    def test_forbidden_producer_and_external_calls(self):
        audit = d.new_audit()
        with d.final_only(audit):
            for function in (d.base.suffix_stages, d.diagnostic.parent.native.projection,
                             d.diagnostic.parent.native.state.add, d.layer.stage_report,
                             d.diagnostic.control_report, d.final.subprocess.Popen):
                with self.assertRaises(RuntimeError):
                    function()
        self.assertEqual(audit["forbidden_calls"], 6)

    def test_forbidden_writes(self):
        audit = d.new_audit()
        with d.write_scope(audit):
            with self.assertRaises(ValueError):
                io.open(d.SOURCE, "w")
            with self.assertRaises(RuntimeError):
                d.os.unlink(d.SOURCE)
        self.assertEqual(audit["file_write_opens"], 0)
        self.assertEqual(audit["forbidden_mutations"], 1)

    def test_occupied_output_zero_dispatch(self):
        with patch.object(d, "OUTPUT", d.ROOT), patch.object(d.final, "rmsnorm") as norm:
            with self.assertRaises(ValueError):
                d.fresh_output()
            norm.assert_not_called()

    def test_suffix_consumes_intervened_words_only(self):
        actual, row, ref = self.fixture()
        arrays, reports = d.intervene(actual, row, ref, self.table)
        refs = {"logits_fp16": actual["final_logits"],
                "logits_binary64": actual["final_logits"].astype("<f8")}
        audit = d.new_audit()
        with d.final_only(audit), \
                patch.object(d.final, "rmsnorm", return_value=(arrays["intervened_stage18"], {})) as norm, \
                patch.object(d.final, "logits", return_value=actual["final_logits"]), \
                patch.object(d.final, "comparisons", return_value={"diagnostic": True}):
            report = d.run_control(d.CONTROLS[0], actual, arrays, reports, [None, None], refs, audit)
        self.assertIs(norm.call_args.args[0], arrays["intervened_stage18"])
        self.assertEqual(report["final_rescued"], "NOT_DEFINED")
        self.assertEqual(len(report["paired_final_margins"]), 6)
        self.assertTrue(all(d.preserve(actual, arrays).values()))

    def test_margin_changes_remain_paired_diagnostics(self):
        words = np.array([1, -2], dtype="<f2").view("<u2")
        weights = np.array([[1, 2], [-1, 0.5]], dtype="<f2")
        np.testing.assert_array_equal(d.final.logits(words, weights).view("<f2"), [-3, -2])
        normalized, _ = d.final.rmsnorm(
            np.zeros(896, dtype="<u2"), np.ones(896, dtype="<f2"))
        np.testing.assert_array_equal(normalized, np.zeros(896, dtype="<u2"))
        old = np.zeros(34320, dtype="<f2")
        new, ref = old.copy(), old.copy()
        old[319], new[319], ref[319] = 2, 3, 1
        rows = d.base.margin_rows(old.view("<u2"), new.view("<u2"),
                                 {"logits_fp16": ref.view("<u2"), "logits_binary64": ref.astype("<f8")})
        self.assertEqual(rows[0]["retained_actual_margin"], "2")
        self.assertEqual(rows[0]["retained_reference_margin"], "1")
        self.assertEqual(rows[0]["retained_margin_change"], "1")
        self.assertEqual(rows[0]["substituted_margin_change"], "2")
        self.assertEqual(rows[0]["paired_intervention_delta"], "1")
