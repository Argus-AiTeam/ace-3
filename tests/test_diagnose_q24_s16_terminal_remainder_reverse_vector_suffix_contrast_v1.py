"""Independent reverse operand/suffix oracle and fail-closed mutation coverage."""

from contextlib import redirect_stdout
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

from ace3.model.candidates import diagnose_q24_s16_terminal_remainder_reverse_vector_suffix_contrast_v1 as d


EVIDENCE = REPORT = OUTPUTS = IDENTITY = DELTA = None
spec = importlib.util.spec_from_file_location("reviewed_exact_suffix_oracle", d.closed.TEST)
if spec is None or spec.loader is None:
    raise RuntimeError("reviewed independent suffix oracle unavailable")
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


class ReverseVectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is None:
            raise RuntimeError("run native --check for authenticated nonzero focused tests")

    def test_independent_integer_suffix_oracle_all_controls(self):
        rows = np.stack([EVIDENCE["rows"][i] for i in d.IDS])
        for output in OUTPUTS.values():
            norm, logits, mean, root = oracle.oracle(
                output["working_stage18"], EVIDENCE["weight_array"], rows)
            self.assertEqual(output["rmsnorm"].tolist(), norm)
            self.assertEqual(output["logits"].tolist(), logits)
            self.assertEqual(output["scalars"], {"mean_q48": mean, "root_q24": root})

    def test_all896_reverse_targets_independent_rne_and_input_order(self):
        expected_delta = tuple(oracle.q(a) - oracle.q(b) for a, b in zip(
            EVIDENCE["reference_archive"]["stage18"].view("<f2"),
            EVIDENCE["binary64"], strict=True))
        self.assertEqual(DELTA, expected_delta)
        self.assertEqual(REPORT["frozen_original_terminal_vector"], list(map(str, DELTA)))
        for control, output in OUTPUTS.items():
            original = EVIDENCE["archives"][control]["stage18"]
            operand = output["operand"]
            self.assertEqual(operand["coordinates"], list(range(896)))
            self.assertEqual(operand["coordinate_count"], 896)
            self.assertEqual(operand["frozen_vector_identity"], d.closed.digest(DELTA))
            targets = [oracle.q(a) + b for a, b in zip(
                original.view("<f2"), expected_delta, strict=True)]
            self.assertEqual(operand["exact_targets"], list(map(str, targets)))
            expected = [oracle.nearest(v) for v in targets]
            self.assertEqual(output["working_stage18"].tolist(), expected)
            self.assertEqual(operand["working_fp16_words"], expected)
            self.assertEqual(operand["rounding_remainders"], [
                str(oracle.q(a) - b) for a, b in zip(
                    output["working_stage18"].view("<f2"), targets, strict=True)])
            self.assertEqual(operand["changed_coordinates"],
                             np.flatnonzero(original != output["working_stage18"]).tolist())
            self.assertFalse(output["working_stage18"].flags.writeable)

    def test_zero_coordinates_and_reverse_not_forward_polarity(self):
        words = np.full(896, 0x3c00, dtype="<u2")
        delta = tuple(Fraction(1, 8) if i % 2 else Fraction() for i in range(896))
        output, operand = d.prepare(words, delta)
        self.assertEqual(output.tolist(), [oracle.nearest(Fraction(1) + v) for v in delta])
        self.assertEqual(operand["changed_coordinates"], list(range(1, 896, 2)))

    def test_selection_order_union_and_pair_mutations_refused(self):
        for name, value in (("COORDINATES", (363,)), ("COORDINATES", d.COORDINATES[:-1]),
                            ("COORDINATES", d.COORDINATES[::-1]),
                            ("COORDINATES", d.COORDINATES[:-1] + (0,)),
                            ("IDS", (319, 1566)), ("PAIRS", ((319, 34319),)),
                            ("BRANCHES", ("binary64", "fp16"))):
            with patch.object(d, name, value), self.assertRaises(ValueError):
                d.prepare(EVIDENCE["archives"]["scratch"]["stage18"], DELTA)

    def test_shape_precision_nonfinite_nondyadic_and_saturation_refused(self):
        words = EVIDENCE["archives"]["scratch"]["stage18"]
        for bad in (words[:-1], words.view("<f2"), np.full(896, 0x7c00, dtype="<u2"),
                    np.full(896, 0x7e00, dtype="<u2")):
            with self.assertRaises(ValueError):
                d.prepare(bad, DELTA)
        for bad in (DELTA[:-1], list(DELTA), (Fraction(1, 3),) * 896,
                    (Fraction(10**9),) * 896):
            with self.assertRaises(ValueError):
                d.prepare(words, bad)

    def test_retained_prediction_not_selected_union_or_full_vector_bridge(self):
        summary = REPORT["retained_forward_summary"]
        by_key = {(r["control"], *r["pair"]): r for r in summary["contrasts"]}
        self.assertEqual(d.retained_predictions(), summary)
        self.assertNotIn("report", EVIDENCE)
        for row in REPORT["contrasts"]:
            retained = by_key[row["control"], row["left_id"], row["right_id"]]
            self.assertEqual(Fraction(row["predicted_delta"]), -Fraction(retained["predicted_delta"]))
            self.assertEqual(Fraction(row["full_vector_linear_bridge_delta"]),
                             -Fraction(retained["full_vector_linear_bridge_delta"]))
            self.assertNotEqual(row["predicted_delta"], row["full_vector_linear_bridge_delta"])
        changed = deepcopy(summary)
        changed["contrasts"][0]["full_vector_linear_bridge_delta"] = "0"
        with self.assertRaises(ValueError):
            d.preregister(EVIDENCE, changed)
        changed = deepcopy(summary)
        changed["contrasts"].reverse()
        with self.assertRaises(ValueError):
            d.preregister(EVIDENCE, changed)

    def test_fixed_margin_obstructions_and_original_references(self):
        self.assertEqual(len(REPORT["contrasts"]), 36)
        for row, old in zip(REPORT["contrasts"], REPORT["retained_common_selection_obstructions"],
                            strict=True):
            for key, value in old.items():
                self.assertEqual(row[key], value)
            reference = EVIDENCE["references"]["logits_" + row["branch"]]
            reference = reference.view("<f2") if row["branch"] == "fp16" else reference
            actual = EVIDENCE["arrays"][row["control"]]["logits"].view("<f2")
            self.assertEqual(Fraction(row["retained_actual_margin"]),
                             oracle.q(actual[row["left_id"]]) - oracle.q(actual[row["right_id"]]))
            self.assertEqual(Fraction(row["fixed_reference_margin"]),
                             oracle.q(reference[row["left_id"]]) - oracle.q(reference[row["right_id"]]))
            self.assertEqual(Fraction(row["observed_delta"]),
                             Fraction(row["intervened_margin_change"]) -
                             Fraction(row["retained_margin_change"]))
            if row["branch"] == "fp16":
                self.assertEqual(row["retained_branch_terminal_component_signed"], "0")
        self.assertEqual(REPORT["retained_common_component"], "UNKNOWN")
        self.assertEqual(REPORT["counterfactual_common_component"], "UNKNOWN")

    def test_directional_classification_zero_opposite_and_invalid_census(self):
        rows = [{**r, "observed_delta": r["predicted_delta"]} for r in REPORT["contrasts"]]
        self.assertEqual(d.classify(rows), "supported")
        for value in ("0", str(-Fraction(rows[0]["predicted_delta"]))):
            changed = deepcopy(rows)
            changed[0]["observed_delta"] = value
            self.assertEqual(d.classify(changed), "rejected")
        for bad in (rows[:-1], rows[::-1], [rows[0]] * 36,
                    [{**r, "predicted_delta": "0"} for r in rows],
                    [{**r, "observed_delta": "NaN"} for r in rows]):
            with self.assertRaises(ValueError):
                d.classify(bad)
        self.assertEqual(REPORT["classification"], d.classify(REPORT["contrasts"]))

    def test_source_test_review_receipt_and_current_pin_mutations(self):
        for module, mission, source, test, review in (
                *d.closed.CHAIN, d.vector.CLOSED_CHAIN,
                d.complement.VECTOR_CHAIN, d.COMPLEMENT_CHAIN):
            for path in (module.SOURCE, module.TEST,
                         d.closed.HANDOFFS / mission / "round-0001.json"):
                with self.assertRaises(ValueError):
                    d.base.read_bound({"path": str(path), "sha256": "0" * 64})
        for path in (d.SOURCE, d.TEST, d.PREDICTION_RECEIPT["path"]):
            with self.assertRaises(ValueError):
                d.base.read_bound({"path": str(path), "sha256": "0" * 64})
        with patch.object(d, "PREDICTION_CALL", "missing"), self.assertRaises(ValueError):
            d.retained_predictions()

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
        for key in ("references", "assets", "hidden"):
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

    def test_forbidden_ranking_union_closed_replay_and_writes(self):
        operations = [
            lambda: d.parent.execute("forbidden"), lambda: d.parent.rmsnorm(None, None),
            lambda: d.parent.logits(None, None), lambda: d.bridge.measure(),
            lambda: d.bridge.margin.report(None), lambda: d.bridge.margin.selected_coordinates(None),
            lambda: d.hotspot.audit_coordinates(None, None, None),
            lambda: subprocess.Popen(["false"]),
            lambda: os.open(d.SOURCE, os.O_WRONLY | os.O_TRUNC), lambda: d.SOURCE.unlink(),
        ]
        for module in (d.closed, d.vector, d.complement):
            for name in ("check", "run_tests", "prepare", "preregister"):
                operations.append(lambda module=module, name=name: getattr(module, name)())
        audit = {"forbidden_calls": 0}
        destinations = (d.SOURCE, d.TEST, d.PREDICTION_RECEIPT["path"],
                        next(iter(d.base.PINS.values()))["path"])
        with d.hotspot.read_only(audit), d.no_reselection_or_closed_replay(audit):
            for operation in operations:
                with self.assertRaises(RuntimeError):
                    operation()
            for path in destinations:
                with self.assertRaises(RuntimeError):
                    io.open(path, "w")
        self.assertEqual(audit["forbidden_calls"], len(operations) + len(destinations))

    def test_suffix_operand_order_scale_and_budget_refusal(self):
        words = OUTPUTS["scratch"]["working_stage18"]
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

    def test_one_unknown_json_and_non_admission_closed_result_flags(self):
        output = io.StringIO()
        with patch.object(d, "check", side_effect=ValueError("injected integrity drift")):
            with redirect_stdout(output):
                self.assertEqual(d.main(["--check"]), 1)
        result = json.loads(output.getvalue())
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertIn("injected integrity drift", result["integrity_error"])
        for key in ("candidate_admitted", "policy_adopted", "successor_published",
                    "strict_FP16_state_claim", "new_token_claim", "full_model_claim",
                    "reference_reanchoring", "accepted_prefix_replay", "admission_replay",
                    "coordinate_ranking", "selected_union_dependency", "selected_union_reconstruction",
                    "closed_coordinate363_intervention_replay", "closed_full_vector_intervention_replay",
                    "closed_non363_intervention_replay", "middle_pair_intervention_replay"):
            self.assertIs(d.FLAGS[key], False)
        for key in ("retained_evidence_writes", "reference_producer_dispatch",
                    "GPU_dispatch", "RTL_dispatch", "admission_dispatch"):
            self.assertEqual(d.FLAGS[key], 0)
        self.assertEqual(len(REPORT["closed_results"]), 3)
        for row in REPORT["closed_results"]:
            self.assertEqual(row["classification"], "supported")
            self.assertEqual(row["directional_contrasts"], "36/36")
            self.assertIs(row["replayed"], False)
        d.closed.protect(EVIDENCE, IDENTITY)
