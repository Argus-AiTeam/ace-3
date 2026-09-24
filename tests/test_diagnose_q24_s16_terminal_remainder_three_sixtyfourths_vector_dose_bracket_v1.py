"""Independent exact bracket oracle, inherited integrity tests and new dose guards."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from fractions import Fraction
import importlib.util
import io
import json
import os
import subprocess
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_terminal_remainder_three_sixtyfourths_vector_dose_bracket_v1 as d


EVIDENCE = REPORT = OUTPUTS = IDENTITY = DELTA = SUMMARIES = None
spec = importlib.util.spec_from_file_location("reviewed_bracket_integrity_tests", d.thirtysecond.TEST)
if spec is None or spec.loader is None:
    raise RuntimeError("reviewed independent tests unavailable")
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
oracle = prior.oracle


class ThreeSixtyFourthsTests(prior.ThirtySecondDoseTests):
    @classmethod
    def setUpClass(cls):
        prior.d = d
        prior.EVIDENCE, prior.REPORT, prior.OUTPUTS = EVIDENCE, REPORT, OUTPUTS
        prior.IDENTITY, prior.DELTA, prior.SUMMARIES = IDENTITY, DELTA, SUMMARIES
        super().setUpClass()

    def test_exact896_thirtysecond_targets_and_independent_rne(self):
        delta = tuple(oracle.q(a) - oracle.q(b) for a, b in zip(
            EVIDENCE["reference_archive"]["stage18"].view("<f2"), EVIDENCE["binary64"], strict=True))
        self.assertEqual(DELTA, delta)
        self.assertEqual(REPORT["frozen_original_terminal_vector"], list(map(str, delta)))
        for (control, polarity), output in OUTPUTS.items():
            original, operand = EVIDENCE["archives"][control]["stage18"], output["operand"]
            sign = -1 if polarity == "forward" else 1
            targets = [oracle.q(a) + Fraction(sign * 3, 64) * b
                       for a, b in zip(original.view("<f2"), delta, strict=True)]
            expected = [oracle.nearest(v) for v in targets]
            self.assertEqual(operand["coordinates"], list(range(896)))
            self.assertEqual(operand["coordinate_count"], 896)
            self.assertEqual(operand["coordinate_order"], "all_input_order")
            self.assertEqual(operand["polarity"], polarity)
            self.assertEqual(operand["dose"], "3/64")
            self.assertEqual(operand["signed_dose"], str(Fraction(sign * 3, 64)))
            self.assertEqual(operand["exact_targets"], list(map(str, targets)))
            self.assertEqual(output["working_stage18"].tolist(), expected)
            self.assertEqual(operand["working_fp16_words"], expected)
            self.assertEqual(operand["rounding_remainders"], [
                str(oracle.q(a) - b) for a, b in zip(
                    output["working_stage18"].view("<f2"), targets, strict=True)])
            self.assertEqual(operand["frozen_vector_identity"], d.closed.digest(delta))
            changed = np.flatnonzero(original != output["working_stage18"]).tolist()
            self.assertEqual(operand["changed_coordinates"], changed)
            for receipt in SUMMARIES[-2:]:
                self.assertNotEqual(len(changed), receipt["changed_coordinate_counts"][polarity][control])
            self.assertFalse(output["working_stage18"].flags.writeable)

    def test_zero_subnormal_ties_even_and_no_rounded_sixteenth_averaging(self):
        words = np.full(896, 0x3c00, dtype="<u2")
        words[:4] = [0, 0x8000, 0x3c00, 0x3c01]
        delta = (Fraction(1, 1 << 21), Fraction(-1, 1 << 21),
                 Fraction(1, 128), Fraction(1, 128)) + (Fraction(),) * 892
        for polarity, sign in (("forward", -1), ("reverse", 1)):
            result, operand = d.prepare(words, delta, polarity)
            self.assertEqual(result.tolist(), [
                oracle.nearest(oracle.q(a) + Fraction(sign * 3, 64) * b)
                for a, b in zip(words.view("<f2"), delta, strict=True)])
            self.assertEqual(result[4:].tolist(), words[4:].tolist())
            self.assertEqual(len(operand["exact_targets"]), 896)
        # Synthetic ties test the converter, not a closed-dose operand replay.
        for value in (Fraction(1, 1 << 25), Fraction(3, 1 << 25),
                      Fraction(1) + Fraction(1, 2048), Fraction(1) + Fraction(3, 2048)):
            bits, saturated = d.parent.head.fixed_to_f16(value.numerator, value.denominator.bit_length() - 1)
            self.assertFalse(saturated)
            self.assertEqual(bits, oracle.nearest(value))
        result, _ = d.prepare(words, delta, "reverse")
        self.assertEqual(result[3], 0x3c01)
        self.assertNotEqual(result[3], 0x3c02)

    def test_selection_order_pair_polarity_and_dose_search_refused(self):
        words = EVIDENCE["archives"]["scratch"]["stage18"]
        for name, value in (("COORDINATES", (363,)), ("COORDINATES", d.COORDINATES[::-1]),
                            ("COORDINATES", d.COORDINATES[:-1] + (0,)),
                            ("IDS", (319, 1566)), ("PAIRS", ((319, 34319),)),
                            ("BRANCHES", ("binary64", "fp16")), ("POLARITIES", ("reverse", "forward")),
                            *(("DOSE", v) for v in (Fraction(1, 32), Fraction(1, 16),
                                Fraction(1, 64), Fraction(1, 8), Fraction(1, 4),
                                Fraction(1, 2), Fraction(1), 0.046875))):
            with patch.object(d, name, value), self.assertRaises(ValueError):
                d.prepare(words, DELTA, "forward")
        with self.assertRaises(ValueError):
            d.prepare(words, DELTA, "unknown")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            d.main(["--check", "--dose", "1/32"])
        self.assertEqual(error.exception.code, 2)

    def test_closed_receipt_outcome_census_and_comparator_mutations(self):
        mutations = [(i, "status", "UNKNOWN") for i in range(7)]
        mutations += [(i, "protected_input_identity", "foreign") for i in range(1, 7)]
        mutations += [(6, k, v) for k, v in (
            ("status", "supported"), ("retained_common_component", "PASS"), ("native_exit", 1),
            ("stdout_json_documents", 2), ("strict_dose_agreements", 72),
            ("directional_agreements", 72), ("contrast_count", 71),
            ("tests", {"executed": 0, "errors": 0, "failures": 0, "skipped": 0}),
            ("dispatch_and_write_audit", {}), ("changed_coordinate_counts", {}),
            ("contrast_fields", list(d.CONTRAST_FIELDS)[::-1]))]
        for index, key, value in mutations:
            summaries = deepcopy(SUMMARIES)
            summaries[index][key] = value
            with self.assertRaises(ValueError):
                d.bind_doses(EVIDENCE, *summaries)
        for receipt_index, fields in ((5, d.thirtysecond.CONTRAST_FIELDS), (6, d.CONTRAST_FIELDS)):
            values = SUMMARIES[receipt_index]["contrast_values"]
            bad_rows = [values[:-1], values[::-1], [values[0]] * 36, [values[0][:-1], *values[1:]]]
            for index in range(len(fields)):
                changed = deepcopy(values)
                changed[0][index] = "0"
                bad_rows.append(changed)
            if receipt_index == 6:
                changed = deepcopy(values)
                zero = next(v for v in changed if v[6] == "0")
                zero[6] = "1/128"
                bad_rows.append(changed)
            for values in bad_rows:
                summaries = deepcopy(SUMMARIES)
                summaries[receipt_index]["contrast_values"] = values
                with self.assertRaises(ValueError):
                    d.bind_doses(EVIDENCE, *summaries)

    def test_source_test_review_receipt_and_current_pin_mutations(self):
        for module, mission, source, test, review in (
                *d.closed.CHAIN, d.vector.CLOSED_CHAIN, d.complement.VECTOR_CHAIN,
                d.reverse.COMPLEMENT_CHAIN, d.half.REVERSE_CHAIN, d.quarter.HALF_CHAIN,
                d.eighth.QUARTER_CHAIN, d.sixteenth.EIGHTH_CHAIN,
                d.thirtysecond.SIXTEENTH_CHAIN, d.THIRTYSECOND_CHAIN):
            round_name = "round-0002.json" if module is d.reverse else "round-0001.json"
            for path in (module.SOURCE, module.TEST, d.closed.HANDOFFS / mission / round_name):
                with self.assertRaises(ValueError):
                    d.base.read_bound({"path": str(path), "sha256": "0" * 64})
        for path in (d.SOURCE, d.TEST, *(r["path"] for r in d.RECEIPTS)):
            with self.assertRaises(ValueError):
                d.base.read_bound({"path": str(path), "sha256": "0" * 64})
        with patch.object(d, "THIRTYSECOND_CALL", "missing"), self.assertRaises(ValueError):
            d.retained_doses()

    def test_forbidden_ranking_union_closed_replay_prefix_and_writes(self):
        operations = [
            lambda: d.parent.execute("forbidden"), lambda: d.parent.rmsnorm(None, None),
            lambda: d.parent.logits(None, None), lambda: d.bridge.measure(),
            lambda: d.bridge.margin.selected_coordinates(None),
            lambda: d.hotspot.audit_coordinates(None, None, None), lambda: subprocess.Popen(["false"]),
            lambda: os.open(d.SOURCE, os.O_WRONLY | os.O_TRUNC), lambda: d.SOURCE.unlink()]
        for module in (d.closed, d.vector, d.complement, d.reverse, d.half, d.quarter,
                       d.eighth, d.sixteenth, d.thirtysecond):
            for name in ("check", "run_tests", "prepare"):
                operations.append(lambda module=module, name=name: getattr(module, name)())
        for module in (d.closed, d.vector, d.complement, d.reverse):
            operations.append(lambda module=module: module.preregister())
        destinations = (d.SOURCE, d.TEST, d.THIRTYSECOND_RECEIPT["path"],
                        next(iter(d.base.PINS.values()))["path"])
        audit = {"forbidden_calls": 0}
        with d.hotspot.read_only(audit), d.no_closed_replay(audit):
            for operation in operations:
                with self.assertRaises(RuntimeError):
                    operation()
            for path in destinations:
                with self.assertRaises(RuntimeError):
                    io.open(path, "w")
        self.assertEqual(audit["forbidden_calls"], len(operations) + len(destinations))

    def test_strict_thirtysecond_classification_zero_opposite_equal_and_overshoot(self):
        rows = [{**r, "observed_delta": r[d.DOSE_KEYS[1]]} for r in REPORT["contrasts"]]
        self.assertEqual(d.classify(rows), "supported")
        for index in range(len(rows)):
            for value in ("0", str(-Fraction(rows[index][d.DOSE_KEYS[1]])),
                          str(2 * Fraction(rows[index][d.DOSE_KEYS[1]])), rows[index][d.DOSE_KEYS[0]]):
                changed = deepcopy(rows)
                changed[index]["observed_delta"] = value
                self.assertEqual(d.classify(changed), "rejected")
        for key in d.DOSE_KEYS:
            changed = deepcopy(rows)
            changed[0][key] = str(-Fraction(changed[0][key]))
            with self.assertRaises(ValueError):
                d.classify(changed)
        for bad in (rows[:-1], rows[::-1], [rows[0]] * 72,
                    [{**r, "predicted_delta": "0"} for r in rows],
                    [{**r, "observed_delta": "NaN"} for r in rows],
                    [{**r, "dose_comparable": False} for r in rows],
                    [{**r, "thirtysecond_zero_obstruction": not r["thirtysecond_zero_obstruction"]} for r in rows],
                    [{**r, d.DOSE_KEYS[0]: r[d.DOSE_KEYS[1]]} for r in rows]):
            with self.assertRaises(ValueError):
                d.classify(bad)
        self.assertEqual(d.classify(REPORT["contrasts"]), REPORT["classification"])

    def test_fixed_reference_obstructions_and_complete_sixteenth_comparators(self):
        retained_rows = {tuple(v[:4]): dict(zip(d.CONTRAST_FIELDS, v, strict=True))
                         for v in SUMMARIES[-1]["contrast_values"]}
        self.assertEqual(len(retained_rows), 36)
        self.assertEqual(len(REPORT["contrasts"]), 72)
        self.assertEqual(sum(r["thirtysecond_zero_obstruction"] for r in REPORT["contrasts"]), 32)
        used = set()
        for row, obstruction in zip(REPORT["contrasts"], REPORT["retained_common_selection_obstructions"], strict=True):
            for key, value in obstruction.items():
                self.assertEqual(row[key], value)
            key = row["control"], row["polarity"], row["left_id"], row["right_id"]
            retained = retained_rows[key]
            used.add(key)
            self.assertEqual(Fraction(row["predicted_delta"]), Fraction(retained["predicted_delta"]) * Fraction(3, 2))
            self.assertEqual(row[d.DOSE_KEYS[0]], retained["observed_delta"])
            for field in d.DOSE_KEYS[1:]:
                self.assertEqual(row[field], retained[field])
            self.assertEqual(row["thirtysecond_receipt_branch"], "binary64")
            self.assertEqual(row["thirtysecond_comparator_derivation"], "closed_ordered_pair_margin_movement")
            self.assertIs(row["dose_comparable"], True)
            self.assertEqual(row["reference_movement_identity"], "fixed_reference_subtraction_cancels")
            left, right = row["left_id"], row["right_id"]
            actual = EVIDENCE["arrays"][row["control"]]["logits"].view("<f2")
            ref = EVIDENCE["references"]["logits_" + row["branch"]]
            ref = ref.view("<f2") if row["branch"] == "fp16" else ref
            self.assertEqual(Fraction(row["retained_actual_margin"]), oracle.q(actual[left]) - oracle.q(actual[right]))
            self.assertEqual(Fraction(row["fixed_reference_margin"]), oracle.q(ref[left]) - oracle.q(ref[right]))
            self.assertEqual(Fraction(row["observed_delta"]),
                             Fraction(row["intervened_margin_change"]) - Fraction(row["retained_margin_change"]))
            self.assertEqual(Fraction(row["observed_delta"]),
                             Fraction(row["intervened_actual_margin"]) - Fraction(row["retained_actual_margin"]))
            if row["branch"] == "fp16":
                self.assertEqual(row["retained_branch_terminal_component_signed"], "0")
        self.assertEqual(used, set(retained_rows))
        self.assertEqual(REPORT["retained_common_component"], "UNKNOWN")
        self.assertEqual(REPORT["counterfactual_common_component"], "UNKNOWN")

    def test_one_stdout_json_unknown_and_closed_non_admission_flags(self):
        for status in ("supported", "rejected"):
            output = io.StringIO()
            with patch.object(d, "check", return_value={"status": status}), redirect_stdout(output):
                self.assertEqual(d.main(["--check"]), 0)
            self.assertEqual(json.loads(output.getvalue()), {"status": status})
            self.assertEqual(len(output.getvalue().splitlines()), 1)
        output = io.StringIO()
        with patch.object(d, "check", side_effect=ValueError("injected integrity drift")), redirect_stdout(output):
            self.assertEqual(d.main(["--check"]), 1)
        result = json.loads(output.getvalue())
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertIn("injected integrity drift", result["integrity_error"])
        for key, value in d.thirtysecond.FLAGS.items():
            self.assertEqual(d.FLAGS[key], value)
        for key in ("candidate_admitted", "strict_FP16_state_claim", "new_token_claim",
                    "full_model_claim", "reference_reanchoring", "accepted_prefix_replay",
                    "admission_replay", "coordinate_ranking", "selected_union_dependency",
                    "closed_thirtysecond_vector_intervention_replay", "thirtysecond_dose_recomputation", "dose_search"):
            self.assertIs(d.FLAGS[key], False)
        for key in ("retained_evidence_writes", "reference_producer_dispatch",
                    "GPU_dispatch", "RTL_dispatch", "admission_dispatch"):
            self.assertEqual(d.FLAGS[key], 0)
        self.assertEqual(len(REPORT["closed_results"]), 9)
        for index, row in enumerate(REPORT["closed_results"]):
            self.assertEqual(row["classification"], "rejected" if index == 8 else "supported")
            self.assertEqual(row["directional_contrasts"],
                             "40/72" if index == 8 else "72/72" if index >= 4 else "36/36")
            self.assertIs(row["replayed"], False)
        d.closed.protect(EVIDENCE, IDENTITY)
