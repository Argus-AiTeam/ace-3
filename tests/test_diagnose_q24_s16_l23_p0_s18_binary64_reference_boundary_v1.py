import io
import struct
import unittest
from fractions import Fraction
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_l23_p0_s18_binary64_reference_boundary_v1 as d


class BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = d.finite_table()

    def scalar(self, left=0x3c00, right=0, actual=0x3c00, fp16=0x3c00, reference=1.0):
        return d.scalar(left, right, actual, fp16, reference, self.table)

    def test_all_finite_words_independent_struct_oracle(self):
        self.assertEqual(len(self.table[0]), 63488)
        for value, word in zip(*self.table):
            self.assertEqual(value, Fraction.from_float(struct.unpack("<e", struct.pack("<H", word))[0]))

    def test_exact_sum(self):
        row = self.scalar(left=0x3c00, right=0x3800, actual=0x3e00, fp16=0x3e00, reference=1.5)
        self.assertEqual(row["independent_rational_s18"], "3/2")
        self.assertTrue(row["local_word_matches"])

    def test_rne_even_tie(self):
        row = self.scalar(right=0x1000)
        self.assertEqual(row["independent_rational_s18"], "2049/2048")
        self.assertEqual(row["independent_rne_word"], "3c00")

    def test_rne_odd_tie(self):
        row = self.scalar(left=0x3c01, right=0x1000, actual=0x3c02, fp16=0x3c02)
        self.assertEqual(row["independent_rne_word"], "3c02")

    def test_negative_zero(self):
        row = self.scalar(left=0x8000, right=0x8000, actual=0x8000, fp16=0x8000, reference=-0.0)
        self.assertEqual(row["independent_q24_z"], 1)
        self.assertEqual(row["independent_rne_word"], "8000")
        self.assertEqual(set(row["nearest_binary64_fp16_words"]), {"0000", "8000"})

    def test_cancellation_zero(self):
        row = self.scalar(right=0xbc00, actual=0, fp16=0, reference=0.0)
        self.assertEqual(row["independent_rne_word"], "0000")
        self.assertEqual(row["independent_q24_z"], 0)

    def test_subnormal(self):
        row = self.scalar(left=1, right=1, actual=2, fp16=2, reference=2**-23)
        self.assertEqual(row["independent_rational_s18"], "1/8388608")
        self.assertTrue(row["accepted"])

    def test_exact_budget_pass(self):
        row = self.scalar(actual=0x3c80)
        self.assertEqual(row["excess_error"], "1/8")
        self.assertTrue(row["accepted"])

    def test_next_word_over_budget_fails(self):
        row = self.scalar(actual=0x3c81)
        self.assertFalse(row["accepted"])
        self.assertEqual(row["nearest_passable_words_to_actual"], ["3c80"])

    def test_reference_midpoint_two_nearest(self):
        row = self.scalar(reference=1 + 2**-11)
        self.assertEqual(row["nearest_binary64_fp16_words"], ["3c00", "3c01"])
        self.assertEqual(row["representation_floor"], "1/2048")

    def test_passable_interval_exact_boundaries(self):
        for ref in (-30.684, -0.0, 0.001, 1.0, 65504.0):
            row = self.scalar(reference=ref)
            span = row["passable_interval"]
            values, words = self.table
            low, high = span["first_index"], span["last_index"]
            radius = Fraction(row["representation_floor"]) + Fraction(1, 8)
            exact = Fraction.from_float(ref)
            self.assertLessEqual(abs(values[low] - exact), radius)
            self.assertLessEqual(abs(values[high] - exact), radius)
            if low:
                self.assertGreater(abs(values[low - 1] - exact), radius)
            if high + 1 < len(values):
                self.assertGreater(abs(values[high + 1] - exact), radius)
            self.assertEqual(span["first_word"], f"{words[low]:04x}")

    def test_signed_decomposition(self):
        row = self.scalar(left=0x3c01, right=0x1000, actual=0x3c02, reference=0.7)
        self.assertEqual(Fraction(row["fixed_operand_minus_binary64"])
                         + Fraction(row["actual_minus_fixed_operand"]),
                         Fraction(row["actual_minus_binary64"]))

    def test_incompatibility_not_representation_impossibility(self):
        row = {**self.scalar(reference=0.5), "index": 0, "local_state_matches": True}
        decision = d.decide([row])
        self.assertEqual(decision["decision"], "FIXED_FP16_OPERAND_REFERENCE_BOUNDARY_INCOMPATIBILITY")
        self.assertFalse(decision["fp16_representation_cannot_pass"])
        self.assertFalse(decision["fixed_operand_correct_rne_can_rescue"])

    def test_local_discrepancy_unknown(self):
        row = {**self.scalar(actual=0x3c01), "index": 0, "local_state_matches": True}
        decision = d.decide([row])
        self.assertEqual(decision["status"], "UNKNOWN")
        self.assertEqual(decision["local_discrepancy_indices"], [0])

    def test_state_defect_unknown(self):
        row = {**self.scalar(), "index": 0, "local_state_matches": False}
        self.assertEqual(d.decide([row])["status"], "UNKNOWN")

    def test_reference_discrepancy_unknown(self):
        row = {**self.scalar(fp16=0x3c01), "index": 0, "local_state_matches": True}
        self.assertEqual(d.decide([row])["decision"], "FP16_REFERENCE_S18_BOUNDARY_DISCREPANCY")
        self.assertEqual(d.decide([row])["status"], "UNKNOWN")

    def test_invalid_operand_refused(self):
        for word in (0x7c00, 0xfc00, 0x7e00):
            with self.assertRaises(ValueError):
                self.scalar(left=word)

    def test_nonfinite_reference_refused(self):
        for value in (float("inf"), float("nan")):
            with self.assertRaises((ValueError, OverflowError)):
                self.scalar(reference=value)

    def test_dispatch_guards(self):
        audit = {"forbidden_calls": 0}
        with d.no_dispatch(audit):
            for function in (d.parent.native.state.add, d.final.rmsnorm,
                             d.base.suffix_stages, d.layer.stage_report, d.parent.run_control):
                with self.assertRaises(RuntimeError):
                    function()
        self.assertEqual(audit["forbidden_calls"], 5)

    def test_write_guard_no_actual_writes(self):
        audit = {"file_write_opens": 0}
        with patch.object(io, "open", return_value=io.StringIO()) as opening:
            with d.base.write_scope(d.OUTPUT, audit):
                with self.assertRaisesRegex(ValueError, "write outside fresh output root"):
                    (d.ROOT / "accepted-evidence.json").open("w")
                self.assertEqual(opening.call_count, 0)
        self.assertEqual(audit["file_write_opens"], 0)

    def header(self):
        audit = d.parent.new_audit()
        audit.update(
            controls=list(d.CONTROLS), stage_dispatches=[[c, 23, 0, 18] for c in d.CONTROLS],
            stage18_oracle_invocations=9, stage12_operand_lifts=9,
            final_rmsnorm_invocations=9, lm_head_invocations=9, top_k_invocations=27,
            file_write_opens=12)
        return {"task_id": d.parent.TASK, "revision": 1, "diagnostic_id": d.parent.NAME,
                "execution_status": "COMPLETE", "execution_count": 1, "status": "FAIL",
                "flags": d.parent.FLAGS.copy(), "audit": audit, "final_rescued": "NOT_DEFINED"}

    def test_final_write_census_authenticated(self):
        d.check_combined_header(self.header())

    def test_changed_threshold_boundary_refused(self):
        header = self.header()
        header["final_rescued"] = "PASS"
        with self.assertRaisesRegex(ValueError, "final-head boundary changed"):
            d.check_combined_header(header)

    def test_missing_control_refused(self):
        header = self.header()
        header["audit"]["controls"].pop()
        with self.assertRaisesRegex(ValueError, "control dispatch census mismatch"):
            d.check_combined_header(header)

    def test_prior_forbidden_dispatch_refused(self):
        header = self.header()
        header["audit"]["native_L0_L22_invocations"] = 1
        with self.assertRaisesRegex(ValueError, "dispatch/write census mismatch: native_L0_L22_invocations"):
            d.check_combined_header(header)

    def test_admission_marker_refused(self):
        header = self.header()
        header["flags"]["candidate_admitted"] = True
        with self.assertRaisesRegex(ValueError, "combined non-admission flags changed"):
            d.check_combined_header(header)

    def test_filesystem_mutation_guard(self):
        audit = {"file_write_opens": 0, "forbidden_mutations": 0}
        with d.write_scope(audit):
            with self.assertRaises(RuntimeError):
                d.os.unlink(d.ROOT / "accepted-evidence.json")
        self.assertEqual(audit["forbidden_mutations"], 1)

    def test_reserved_attempt_cannot_reenter(self):
        with patch.object(d.Path, "iterdir", return_value=iter([d.OUTPUT / "command.json"])):
            with self.assertRaisesRegex(ValueError, "occupied output; zero diagnostic dispatch"):
                d.fresh_output()
