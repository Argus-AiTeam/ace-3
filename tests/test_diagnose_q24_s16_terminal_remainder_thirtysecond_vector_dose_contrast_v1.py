"""Independent thirty-second-dose conversion/suffix oracle and integrity mutations."""

from contextlib import redirect_stderr, redirect_stdout
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

from ace3.model.candidates import diagnose_q24_s16_terminal_remainder_thirtysecond_vector_dose_contrast_v1 as d


EVIDENCE = REPORT = OUTPUTS = IDENTITY = DELTA = SUMMARIES = None
spec = importlib.util.spec_from_file_location("reviewed_thirtysecond_suffix_oracle", d.closed.TEST)
if spec is None or spec.loader is None:
    raise RuntimeError("reviewed independent suffix oracle unavailable")
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


class ThirtySecondDoseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is None:
            raise RuntimeError("run native --check for authenticated nonzero focused tests")

    def test_independent_suffix_oracle_both_polarities_all_controls(self):
        rows = np.stack([EVIDENCE["rows"][i] for i in d.IDS])
        self.assertEqual(len(OUTPUTS), 18)
        for output in OUTPUTS.values():
            norm, logits, mean, root = oracle.oracle(
                output["working_stage18"], EVIDENCE["weight_array"], rows)
            self.assertEqual(output["rmsnorm"].tolist(), norm)
            self.assertEqual(output["logits"].tolist(), logits)
            self.assertEqual(output["scalars"], {"mean_q48": mean, "root_q24": root})

    def test_exact896_thirtysecond_targets_and_independent_rne(self):
        delta = tuple(oracle.q(a) - oracle.q(b) for a, b in zip(
            EVIDENCE["reference_archive"]["stage18"].view("<f2"),
            EVIDENCE["binary64"], strict=True))
        self.assertEqual(DELTA, delta)
        self.assertEqual(REPORT["frozen_original_terminal_vector"], list(map(str, delta)))
        for (control, polarity), output in OUTPUTS.items():
            original, operand = EVIDENCE["archives"][control]["stage18"], output["operand"]
            sign = -1 if polarity == "forward" else 1
            targets = [oracle.q(a) + sign * b / 32 for a, b in zip(
                original.view("<f2"), delta, strict=True)]
            expected = [oracle.nearest(v) for v in targets]
            self.assertEqual(operand["coordinates"], list(range(896)))
            self.assertEqual(operand["coordinate_count"], 896)
            self.assertEqual(operand["coordinate_order"], "all_input_order")
            self.assertEqual(operand["polarity"], polarity)
            self.assertEqual(operand["dose"], "1/32")
            self.assertEqual(operand["signed_dose"], str(Fraction(sign, 32)))
            self.assertEqual(operand["exact_targets"], list(map(str, targets)))
            self.assertEqual(output["working_stage18"].tolist(), expected)
            self.assertEqual(operand["working_fp16_words"], expected)
            self.assertEqual(operand["rounding_remainders"], [
                str(oracle.q(a) - b) for a, b in zip(
                    output["working_stage18"].view("<f2"), targets, strict=True)])
            self.assertEqual(operand["frozen_vector_identity"], d.closed.digest(delta))
            self.assertEqual(operand["changed_coordinates"],
                             np.flatnonzero(original != output["working_stage18"]).tolist())
            self.assertFalse(output["working_stage18"].flags.writeable)

    def test_zero_subnormal_ties_even_and_no_rounded_sixteenth_averaging(self):
        words = np.full(896, 0x3c00, dtype="<u2")
        words[:5] = [0, 0x8000, 0x3c00, 0x3c01, 0x3c01]
        delta = (Fraction(1, 1 << 20), Fraction(-1, 1 << 20),
                 Fraction(1, 64), Fraction(1, 128), Fraction(1, 64)) + (Fraction(),) * 891
        for polarity, sign in (("forward", -1), ("reverse", 1)):
            result, operand = d.prepare(words, delta, polarity)
            self.assertEqual(result.tolist(), [
                oracle.nearest(oracle.q(a) + sign * b / 32)
                for a, b in zip(words.view("<f2"), delta, strict=True)])
            self.assertEqual(result[5:].tolist(), words[5:].tolist())
            self.assertEqual(len(operand["exact_targets"]), 896)
        result, _ = d.prepare(words, delta, "reverse")
        self.assertEqual(result[[2, 4]].tolist(), [0x3c00, 0x3c02])
        actual = oracle.q(words.view("<f2")[3])
        rounded_sixteenth = np.array([oracle.nearest(actual + delta[3] / 16)], dtype="<u2")
        averaged = oracle.nearest((actual + oracle.q(rounded_sixteenth.view("<f2")[0])) / 2)
        self.assertNotEqual(result[3], averaged)

    def test_selection_order_pair_polarity_and_dose_search_refused(self):
        words = EVIDENCE["archives"]["scratch"]["stage18"]
        for name, value in (("COORDINATES", (363,)), ("COORDINATES", d.COORDINATES[::-1]),
                            ("COORDINATES", d.COORDINATES[:-1] + (0,)),
                            ("IDS", (319, 1566)), ("PAIRS", ((319, 34319),)),
                            ("BRANCHES", ("binary64", "fp16")),
                            ("POLARITIES", ("reverse", "forward")),
                            *(("DOSE", v) for v in (
                                Fraction(1, 16), Fraction(1, 64), Fraction(1, 8),
                                Fraction(1, 4), Fraction(1, 2), Fraction(1), 0.03125))):
            with patch.object(d, name, value), self.assertRaises(ValueError):
                d.prepare(words, DELTA, "forward")
        with self.assertRaises(ValueError):
            d.prepare(words, DELTA, "unknown")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            d.main(["--check", "--dose", "1/64"])
        self.assertEqual(error.exception.code, 2)

    def test_shape_precision_nonfinite_nondyadic_saturation_refused(self):
        words = EVIDENCE["archives"]["scratch"]["stage18"]
        for bad in (words[:-1], words.view("<f2"), np.full(896, 0x7c00, dtype="<u2"),
                    np.full(896, 0x7e00, dtype="<u2")):
            with self.assertRaises(ValueError):
                d.prepare(bad, DELTA, "forward")
        for bad in (DELTA[:-1], list(DELTA), (Fraction(1, 3),) * 896,
                    (Fraction(10**9),) * 896):
            for polarity in d.POLARITIES:
                with self.assertRaises(ValueError):
                    d.prepare(words, bad, polarity)

    def test_closed_receipt_outcome_census_and_comparator_mutations(self):
        mutations = [(i, "status", "UNKNOWN") for i in range(6)]
        mutations += [(i, "protected_input_identity", "foreign") for i in range(1, 6)]
        mutations += [(5, k, v) for k, v in (
            ("retained_common_component", "PASS"), ("native_exit", 1),
            ("stdout_json_documents", 2), ("strict_dose_agreements", 71),
            ("directional_agreements", 71), ("contrast_count", 71),
            ("tests", {"executed": 0, "errors": 0, "failures": 0, "skipped": 0}),
            ("dispatch_and_write_audit", {}), ("changed_coordinate_counts", {}),
            ("contrast_fields", list(d.CONTRAST_FIELDS)[::-1]))]
        for index, key, value in mutations:
            summaries = deepcopy(SUMMARIES)
            summaries[index][key] = value
            with self.assertRaises(ValueError):
                d.bind_doses(EVIDENCE, *summaries)
        for index in range(4):
            for key in ("predicted_delta", "observed_delta"):
                summaries = deepcopy(SUMMARIES)
                summaries[index]["contrasts"][0][key] = "0"
                with self.assertRaises(ValueError):
                    d.bind_doses(EVIDENCE, *summaries)
        summaries = deepcopy(SUMMARIES)
        summaries[4]["distinct_forward_pair_movements"] *= 2
        with self.assertRaises(ValueError):
            d.bind_doses(EVIDENCE, *summaries)
        values = SUMMARIES[5]["contrast_values"]
        bad_rows = [values[:-1], values[::-1], [values[0]] * 36,
                    [values[0][:-1], *values[1:]]]
        for index in range(len(d.CONTRAST_FIELDS)):
            changed = deepcopy(values)
            changed[0][index] = "0"
            bad_rows.append(changed)
        for changed in bad_rows:
            summaries = deepcopy(SUMMARIES)
            summaries[5]["contrast_values"] = changed
            with self.assertRaises(ValueError):
                d.bind_doses(EVIDENCE, *summaries)

    def test_source_test_review_receipt_and_current_pin_mutations(self):
        for module, mission, source, test, review in (
                *d.closed.CHAIN, d.vector.CLOSED_CHAIN, d.complement.VECTOR_CHAIN,
                d.reverse.COMPLEMENT_CHAIN, d.half.REVERSE_CHAIN,
                d.quarter.HALF_CHAIN, d.eighth.QUARTER_CHAIN,
                d.sixteenth.EIGHTH_CHAIN, d.SIXTEENTH_CHAIN):
            round_name = "round-0002.json" if module is d.reverse else "round-0001.json"
            for path in (module.SOURCE, module.TEST, d.closed.HANDOFFS / mission / round_name):
                with self.assertRaises(ValueError):
                    d.base.read_bound({"path": str(path), "sha256": "0" * 64})
        for path in (d.SOURCE, d.TEST, *(r["path"] for r in d.RECEIPTS)):
            with self.assertRaises(ValueError):
                d.base.read_bound({"path": str(path), "sha256": "0" * 64})
        with patch.object(d, "SIXTEENTH_CALL", "missing"), self.assertRaises(ValueError):
            d.retained_doses()

    def test_operand_q24_kv_reference_weight_and_asset_drift(self):
        for key in ("stage18", "input_i", "input_z", "input_hidden", "input_cache_k",
                    "input_cache_v", "output_cache_k", "output_cache_v"):
            changed = {**EVIDENCE, "archives": dict(EVIDENCE["archives"])}
            archive = dict(EVIDENCE["archives"]["scratch"])
            archive[key] = archive[key].copy()
            if archive[key].size:
                archive[key].view("u1").flat[0] ^= 1
            else:
                archive[key] = np.zeros((1,), dtype=archive[key].dtype)
            changed["archives"]["scratch"] = archive
            with self.assertRaises(ValueError):
                d.closed.protect(changed, IDENTITY)
        for key in ("binary64", "weight_array"):
            changed = {**EVIDENCE, key: EVIDENCE[key].copy()}
            changed[key].view("u1").flat[0] ^= 1
            with self.assertRaises(ValueError):
                d.closed.protect(changed, IDENTITY)
        for key in ("references", "assets", "hidden", "rows"):
            with self.assertRaises(ValueError):
                d.closed.protect({**EVIDENCE, key: {}}, IDENTITY)

    def test_lineage_threshold_and_historical_failure_mutations(self):
        for key, value in (("source_operand_state_KV_RTZ_checks", "FAIL"),
                           ("candidate_admitted", True), ("S18_failure_indices", [])):
            result = deepcopy(EVIDENCE["result"])
            result["controls"][0]["parent"]["retained_L23"][key] = value
            with self.assertRaises(ValueError):
                d.base.check_history(result)
        result = deepcopy(EVIDENCE["result"])
        result["controls"][0]["parent"]["retained_L23"]["retained_L21"]["control"] = "foreign"
        with self.assertRaises(ValueError):
            d.base.check_history(result)
        result = deepcopy(EVIDENCE["result"])
        result["preflight"]["thresholds"] = {}
        with self.assertRaises(ValueError):
            d.closed.protect({**EVIDENCE, "result": result}, IDENTITY)

    def test_forbidden_ranking_union_closed_replay_prefix_and_writes(self):
        operations = [
            lambda: d.parent.execute("forbidden"), lambda: d.parent.rmsnorm(None, None),
            lambda: d.parent.logits(None, None), lambda: d.bridge.measure(),
            lambda: d.bridge.margin.selected_coordinates(None),
            lambda: d.hotspot.audit_coordinates(None, None, None),
            lambda: subprocess.Popen(["false"]),
            lambda: os.open(d.SOURCE, os.O_WRONLY | os.O_TRUNC), lambda: d.SOURCE.unlink(),
        ]
        for module in (d.closed, d.vector, d.complement, d.reverse, d.half, d.quarter,
                       d.eighth, d.sixteenth):
            for name in ("check", "run_tests", "prepare"):
                operations.append(lambda module=module, name=name: getattr(module, name)())
        for module in (d.closed, d.vector, d.complement, d.reverse):
            operations.append(lambda module=module: module.preregister())
        destinations = (d.SOURCE, d.TEST, d.SIXTEENTH_RECEIPT["path"],
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

    def test_suffix_operand_order_scale_and_budget_refusal(self):
        words = OUTPUTS["scratch", "forward"]["working_stage18"]
        weights = EVIDENCE["weight_array"]
        rows = np.stack([EVIDENCE["rows"][i] for i in d.IDS])
        changed = words.copy()
        changed[363] ^= 1
        with self.assertRaises(ValueError):
            with d.closed.suffix_only({"forbidden_calls": 0}, [words], weights, rows):
                d.parent.rmsnorm(changed, weights)
        changed_weights = weights.copy()
        changed_weights.view("<u2")[0] ^= 1
        with self.assertRaises(ValueError):
            with d.closed.suffix_only({"forbidden_calls": 0}, [words], weights, rows):
                d.parent.rmsnorm(words, changed_weights)
        for operation in (lambda: d.parent.logits(words, rows[::-1]),
                          lambda: d.parent.logits(words, np.stack([*rows, rows[0]])),
                          lambda: d.parent.rmsnorm(words, weights)):
            with self.assertRaises(RuntimeError):
                with d.closed.suffix_only({"forbidden_calls": 0}, [], weights, rows):
                    operation()

    def test_strict_thirtysecond_classification_zero_opposite_equal_and_overshoot(self):
        rows = [{**r, "observed_delta": str(Fraction(r[d.DOSE_KEYS[0]]) / 2)}
                for r in REPORT["contrasts"]]
        self.assertEqual(d.classify(rows), "supported")
        for key in d.DOSE_KEYS:
            for factor in (0, -1, 1, 2):
                changed = deepcopy(rows)
                changed[0]["observed_delta"] = str(factor * Fraction(changed[0][key]))
                self.assertEqual(d.classify(changed), "rejected")
            for value in ("0", str(-Fraction(rows[0][key]))):
                changed = deepcopy(rows)
                changed[0][key] = value
                with self.assertRaises(ValueError):
                    d.classify(changed)
        for bad in (rows[:-1], rows[::-1], [rows[0]] * 72,
                    [{**r, "predicted_delta": "0"} for r in rows],
                    [{**r, "observed_delta": "NaN"} for r in rows],
                    [{**r, "dose_comparable": False} for r in rows],
                    [{**r, d.DOSE_KEYS[0]: r[d.DOSE_KEYS[1]]} for r in rows]):
            with self.assertRaises(ValueError):
                d.classify(bad)
        self.assertEqual(d.classify(REPORT["contrasts"]), REPORT["classification"])

    def test_fixed_reference_obstructions_and_complete_sixteenth_comparators(self):
        retained_rows = {
            tuple(v[:4]): dict(zip(d.CONTRAST_FIELDS, v, strict=True))
            for v in SUMMARIES[5]["contrast_values"]}
        self.assertEqual(len(retained_rows), 36)
        self.assertEqual(len(REPORT["contrasts"]), 72)
        used = set()
        for row, obstruction in zip(REPORT["contrasts"],
                                    REPORT["retained_common_selection_obstructions"], strict=True):
            for key, value in obstruction.items():
                self.assertEqual(row[key], value)
            key = row["control"], row["polarity"], row["left_id"], row["right_id"]
            retained = retained_rows[key]
            used.add(key)
            self.assertEqual(Fraction(row["predicted_delta"]), Fraction(retained["predicted_delta"]) / 2)
            self.assertEqual(row[d.DOSE_KEYS[0]], retained["observed_delta"])
            for field in d.DOSE_KEYS[1:]:
                self.assertEqual(row[field], retained[field])
            self.assertEqual(row["sixteenth_receipt_branch"], "binary64")
            self.assertEqual(row["sixteenth_comparator_derivation"], "closed_ordered_pair_margin_movement")
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
        with patch.object(d, "check", side_effect=ValueError("injected integrity drift")):
            with redirect_stdout(output):
                self.assertEqual(d.main(["--check"]), 1)
        result = json.loads(output.getvalue())
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertIn("injected integrity drift", result["integrity_error"])
        for key, value in d.sixteenth.FLAGS.items():
            self.assertEqual(d.FLAGS[key], value)
        for key in ("candidate_admitted", "strict_FP16_state_claim", "new_token_claim",
                    "full_model_claim", "reference_reanchoring", "accepted_prefix_replay",
                    "admission_replay", "coordinate_ranking", "selected_union_dependency",
                    "closed_reverse_vector_intervention_replay", "closed_half_vector_intervention_replay",
                    "closed_quarter_vector_intervention_replay", "closed_eighth_vector_intervention_replay",
                    "closed_sixteenth_vector_intervention_replay", "full_dose_recomputation",
                    "half_dose_recomputation", "quarter_dose_recomputation",
                    "eighth_dose_recomputation", "sixteenth_dose_recomputation", "dose_search"):
            self.assertIs(d.FLAGS[key], False)
        for key in ("retained_evidence_writes", "reference_producer_dispatch",
                    "GPU_dispatch", "RTL_dispatch", "admission_dispatch"):
            self.assertEqual(d.FLAGS[key], 0)
        self.assertEqual(len(REPORT["closed_results"]), 8)
        for index, row in enumerate(REPORT["closed_results"]):
            self.assertEqual(row["classification"], "supported")
            self.assertEqual(row["directional_contrasts"], "72/72" if index >= 4 else "36/36")
            self.assertIs(row["replayed"], False)
        d.closed.protect(EVIDENCE, IDENTITY)
