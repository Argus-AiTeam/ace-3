"""Independent rational/FP16 selected-head oracle and fail-closed scope tests."""

from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from fractions import Fraction
import importlib.util
import io
import json
import os
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_terminal_remainder_final_rmsnorm_headonly_three_sixtyfourths_vector_dose_classifier_v1 as d


EVIDENCE = REPORT = OUTPUTS = IDENTITY = DELTA = SUMMARIES = PINS = None
spec = importlib.util.spec_from_file_location("reviewed_independent_fp16_oracle", d.closed.TEST)
if spec is None or spec.loader is None:
    raise RuntimeError("independent FP16 oracle unavailable")
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


class HeadOnlyTests(unittest.TestCase):
    def test_exact_input_order_and_independent_hidden_rne(self):
        expected_delta = tuple(oracle.q(a) - oracle.q(b) for a, b in zip(
            EVIDENCE["references"]["rmsnorm_fp16"].view("<f2"),
            EVIDENCE["references"]["rmsnorm_binary64"], strict=True))
        self.assertEqual(DELTA, expected_delta)
        self.assertEqual(REPORT["frozen_final_hidden_residual"], list(map(str, DELTA)))
        for (c, p), output in OUTPUTS.items():
            original = EVIDENCE["arrays"][c]["rmsnorm"]
            sign = -1 if p == "forward" else 1
            targets = [oracle.q(a) + Fraction(sign * 3, 64) * b
                       for a, b in zip(original.view("<f2"), DELTA, strict=True)]
            operand = output["operand"]
            self.assertEqual(operand["coordinates"], list(range(896)))
            self.assertEqual(operand["coordinate_order"], "all_input_order")
            self.assertEqual(operand["exact_targets"], list(map(str, targets)))
            self.assertEqual(operand["working_fp16_words"], [oracle.nearest(v) for v in targets])
            self.assertEqual(output["working_hidden"].tolist(), operand["working_fp16_words"])
            self.assertEqual(operand["rounding_remainders"], [
                str(oracle.q(a) - b) for a, b in zip(output["working_hidden"].view("<f2"), targets, strict=True)])
            self.assertEqual(operand["changed_coordinates"],
                             np.flatnonzero(original != output["working_hidden"]).tolist())
            self.assertFalse(output["working_hidden"].flags.writeable)
            self.assertFalse(np.array_equal(output["working_hidden"], EVIDENCE["archives"][c]["stage18"]))

    def test_independent_exact_selected_dots_and_logit_rne(self):
        rows = [EVIDENCE["rows"][i] for i in d.IDS]
        for (c, _), out in OUTPUTS.items():
            for key, values, words_key in (
                    ("baseline_dots", map(oracle.q, EVIDENCE["arrays"][c]["rmsnorm"].view("<f2")), "baseline_logits"),
                    ("working_dots", map(oracle.q, out["working_hidden"].view("<f2")), "logits"),
                    ("target_dots", map(Fraction, out["operand"]["exact_targets"]), None)):
                values = list(values)
                expected = tuple(sum((a * oracle.q(b) for a, b in zip(values, row, strict=True)),
                                     Fraction()) for row in rows)
                self.assertEqual(out[key], expected)
                if words_key:
                    self.assertEqual(out[words_key].tolist(), [oracle.nearest(v) for v in expected])
        for row in REPORT["contrasts"]:
            i, j = d.IDS.index(row["left_id"]), d.IDS.index(row["right_id"])
            for key, margin in (("baseline_row_dots", "baseline_dot_margin"),
                                ("target_row_dots", "target_dot_margin"),
                                ("working_row_dots", "working_dot_margin")):
                self.assertEqual(Fraction(row[key][i]) - Fraction(row[key][j]), Fraction(row[margin]))
            self.assertEqual(Fraction(row["working_dot_margin"]) - Fraction(row["target_dot_margin"]),
                             Fraction(row["hidden_rne_margin_remainder"]))
            self.assertEqual(Fraction(row["intervened_actual_margin"]) - Fraction(row["working_dot_margin"]),
                             Fraction(row["working_head_rne_remainder"]))

    def test_synthetic_ties_subnormal_zero_and_saturation(self):
        words = np.full(896, 0x3c00, dtype="<u2")
        words[:4] = [0, 0x8000, 0x3c00, 0x3c01]
        delta = (Fraction(1, 1 << 21), Fraction(-1, 1 << 21),
                 Fraction(1, 128), Fraction(1, 128)) + (Fraction(),) * 892
        for p, sign in (("forward", -1), ("reverse", 1)):
            output, targets, _ = d.prepare(words, delta, p)
            self.assertEqual(output.tolist(), [oracle.nearest(
                oracle.q(a) + Fraction(sign * 3, 64) * b) for a, b in zip(words.view("<f2"), delta, strict=True)])
            self.assertEqual(len(targets), 896)
        for target in (Fraction(1, 1 << 25), Fraction(3, 1 << 25),
                       Fraction(1) + Fraction(1, 2048), Fraction(1) + Fraction(3, 2048)):
            bits, saturated = d.parent.head.fixed_to_f16(target.numerator, target.denominator.bit_length() - 1)
            self.assertFalse(saturated)
            self.assertEqual(bits, oracle.nearest(target))
        with self.assertRaises(ValueError):
            d.prepare(words, (Fraction(1 << 30),) * 896, "reverse")

    def test_selection_and_dose_mutations_refused(self):
        words = EVIDENCE["arrays"]["scratch"]["rmsnorm"]
        for name, value in (
                ("IDS", (319, 1566)), ("PAIRS", d.PAIRS[::-1]), ("BRANCHES", d.BRANCHES[::-1]),
                ("COORDINATES", (363,)), ("COORDINATES", d.COORDINATES[::-1]),
                ("POLARITIES", d.POLARITIES[::-1]), ("DOSE", Fraction(1, 32)),
                ("DOSE", Fraction(1, 16)), ("DOSE", 0.046875)):
            with patch.object(d, name, value), self.assertRaises(ValueError):
                d.prepare(words, DELTA, "forward")
        for delta in (DELTA[:-1], list(DELTA), (Fraction(1, 3),) * 896):
            with self.assertRaises(ValueError):
                d.prepare(words, delta, "forward")
        with self.assertRaises(ValueError):
            d.prepare(words, DELTA, "other")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            d.main(["--check", "--dose", "1/32"])

    def test_protected_source_operand_state_kv_lineage_and_references(self):
        d.closed.protect(EVIDENCE, IDENTITY)
        for key in EVIDENCE:
            changed = {**EVIDENCE, key: "mutation"}
            with self.assertRaises(ValueError):
                d.closed.protect(changed, IDENTITY)
        for kind in ("rmsnorm", "logits"):
            changed = deepcopy(EVIDENCE)
            changed["arrays"]["scratch"][kind][0] ^= 1
            with self.assertRaises(ValueError):
                d.closed.protect(changed, IDENTITY)
        for key in ("rmsnorm_fp16", "rmsnorm_binary64", "logits_fp16", "logits_binary64"):
            changed = deepcopy(EVIDENCE)
            changed["references"][key][0] += 1
            with self.assertRaises(ValueError):
                d.closed.protect(changed, IDENTITY)

    def test_closed_receipt_mutations_refused(self):
        for index in range(8):
            summaries = deepcopy(SUMMARIES)
            summaries[index]["status"] = "UNKNOWN"
            with self.assertRaises(ValueError):
                d.bind_doses(EVIDENCE, summaries)
        for key, value in (
                ("protected_input_identity", "foreign"), ("flags", {}), ("native_exit", 1),
                ("stdout_json_documents", 2), ("strict_dose_agreements", 72),
                ("reverse_zero_obstructions", 0), ("recovered_reverse_zero_contrasts", 0),
                ("dispatch_and_write_audit", {}), ("changed_coordinate_counts", {})):
            summaries = deepcopy(SUMMARIES)
            summaries[-1][key] = value
            with self.assertRaises(ValueError):
                d.bind_doses(EVIDENCE, summaries)
        values = SUMMARIES[-1]["contrast_values"]
        mutations = [values[:-1], values[::-1], [values[0]] * 36]
        for i in range(len(d.CONTRAST_FIELDS)):
            changed = deepcopy(values)
            changed[0][i] = "foreign"
            mutations.append(changed)
        for values in mutations:
            summaries = deepcopy(SUMMARIES)
            summaries[-1]["contrast_values"] = values
            with self.assertRaises(ValueError):
                d.bind_doses(EVIDENCE, summaries)

    def test_current_parent_review_receipt_pin_mutations(self):
        for pin in (*PINS, *d.RECEIPTS, d.parent.record(d.SOURCE), d.parent.record(d.TEST)):
            with self.assertRaises(ValueError):
                d.base.read_bound({**pin, "sha256": "0" * 64})
        with patch.object(d, "BRACKET_CALL", "missing"), self.assertRaises(ValueError):
            d.retained_doses()

    def test_forbidden_replay_write_and_unselected_dispatch(self):
        audit = {"forbidden_calls": 0, "final_rmsnorm_invocations": 0, "selected_row_head_invocations": 0}
        operations = [
            lambda: d.parent.execute("forbidden"), lambda: d.parent.rmsnorm(None, None),
            lambda: d.parent.norm.rmsnorm(None, None), lambda: d.bridge.measure(),
            lambda: d.bridge.margin.selected_coordinates(None),
            lambda: d.hotspot.audit_coordinates(None, None, None),
            lambda: d.parent.logits(None, None), lambda: d.parent.head.decode_array_q24(None),
            lambda: subprocess.Popen(["false"]), lambda: io.open(d.SOURCE, "w"),
            lambda: os.open(d.TEST, os.O_WRONLY | os.O_TRUNC), lambda: d.SOURCE.unlink()]
        for module in (d.closed, d.bracket.vector, d.bracket.complement, d.bracket.reverse,
                       d.bracket.half, d.bracket.quarter, d.bracket.eighth,
                       d.bracket.sixteenth, d.bracket.thirtysecond, d.bracket):
            for name in ("prepare", "check", "run_tests"):
                operations.append(lambda module=module, name=name: getattr(module, name)())
        with d.head_only(audit, [], np.zeros((2, 896), dtype="<f2")):
            for operation in operations:
                with self.assertRaises(RuntimeError):
                    operation()
        self.assertEqual(audit["forbidden_calls"], len(operations))
        self.assertEqual(audit["selected_row_head_invocations"], 0)
        self.assertEqual(audit["final_rmsnorm_invocations"], 0)

    def test_head_operand_and_row_splices_refused(self):
        words = EVIDENCE["arrays"]["scratch"]["rmsnorm"]
        rows = np.stack([EVIDENCE["rows"][i] for i in d.IDS])
        for bad_words, bad_rows in ((words[:-1], rows), (words, rows[::-1]), (words, rows[:1])):
            audit = {"forbidden_calls": 0, "final_rmsnorm_invocations": 0, "selected_row_head_invocations": 0}
            with self.assertRaises(ValueError), d.head_only(audit, [words], rows):
                d.parent.logits(bad_words, bad_rows)
            self.assertEqual(audit["selected_row_head_invocations"], 0)

    def test_classification_both_scientific_successors(self):
        rows = deepcopy(REPORT["contrasts"])
        for row in rows:
            old, ref = Fraction(row["retained_actual_margin"]), Fraction(row["fixed_reference_margin"])
            row["observed_delta"] = row["closed_bracket_delta"]
            new = old + Fraction(row["observed_delta"])
            row["intervened_actual_margin"], row["intervened_margin_change"] = str(new), str(new - ref)
        self.assertEqual(d.classify(rows), ("supported", d.HEAD_SUCCESSOR))
        row = rows[0]
        row["observed_delta"] = "0"
        row["intervened_actual_margin"] = row["retained_actual_margin"]
        row["intervened_margin_change"] = row["retained_margin_change"]
        self.assertEqual(d.classify(rows), ("rejected", d.NORM_SUCCESSOR))
        self.assertEqual(d.classify(REPORT["contrasts"]), (REPORT["classification"], REPORT["successor"]))

    def test_classification_integrity_and_exact_separation(self):
        rows = REPORT["contrasts"]
        for changed in (rows[:-1], rows[::-1], [rows[0]] * 72):
            with self.assertRaises(ValueError):
                d.classify(changed)
        for key in ("target_dot_delta", "frozen_hidden_dot_residual", "baseline_dot_margin",
                    "working_dot_margin", "target_dot_margin", "pre_head_rne_delta",
                    "retained_margin_change", "intervened_margin_change", "closed_bracket_delta"):
            changed = deepcopy(rows)
            changed[0][key] = str(Fraction(changed[0][key]) + 1)
            with self.assertRaises(ValueError):
                d.classify(changed)
        for row in rows:
            residual, target = Fraction(row["frozen_hidden_dot_residual"]), Fraction(row["target_dot_delta"])
            self.assertLess(abs(residual / 32), abs(target))
            self.assertLess(abs(target), abs(residual / 16))

    def test_single_json_unknown_and_false_admission_flags(self):
        for error in (ValueError("authentication defect"), RuntimeError("scope defect")):
            output = io.StringIO()
            with patch.object(d, "check", side_effect=error), redirect_stdout(output):
                code = d.main(["--check"])
            value = json.loads(output.getvalue())
            self.assertEqual((code, value["status"]), (1, "UNKNOWN"))
            self.assertIn(str(error), value["integrity_error"])
            self.assertNotIn("successor", value)
        output = io.StringIO()
        with patch.object(d, "check", return_value={"status": "rejected"}), redirect_stdout(output):
            self.assertEqual(d.main(["--check"]), 0)
        self.assertEqual(json.loads(output.getvalue()), {"status": "rejected"})
        for key, value in d.FLAGS.items():
            if key not in ("historical_failures_preserved", "original_global_reference_unchanged"):
                self.assertIn(value, (False, 0), key)
