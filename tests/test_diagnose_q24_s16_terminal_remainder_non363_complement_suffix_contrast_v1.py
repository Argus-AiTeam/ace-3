"""Independent non-363 operand/suffix oracle and authenticated refusal gates."""

from contextlib import redirect_stdout
from copy import deepcopy
from fractions import Fraction
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_terminal_remainder_non363_complement_suffix_contrast_v1 as d


EVIDENCE = REPORT = OUTPUTS = IDENTITY = DELTA = RETAINED = None
spec = importlib.util.spec_from_file_location("reviewed_exact_suffix_oracle", d.closed.TEST)
if spec is None or spec.loader is None:
    raise RuntimeError("reviewed independent suffix oracle unavailable")
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


class TerminalNon363Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is None:
            raise RuntimeError("run the native --check for authenticated nonzero focused tests")

    def test_independent_integer_suffix_oracle_all_controls(self):
        rows = np.stack([EVIDENCE["rows"][i] for i in d.IDS])
        for result in OUTPUTS.values():
            norm, logits, mean, root = oracle.oracle(
                result["working_stage18"], EVIDENCE["weight_array"], rows)
            self.assertEqual(result["rmsnorm"].tolist(), norm)
            self.assertEqual(result["logits"].tolist(), logits)
            self.assertEqual(result["scalars"], {"mean_q48": mean, "root_q24": root})

    def test_all895_original_targets_and_protected_coordinate363(self):
        expected_delta = tuple(oracle.q(a)-oracle.q(b) for a, b in zip(
            EVIDENCE["reference_archive"]["stage18"].view("<f2"),
            EVIDENCE["binary64"], strict=True))
        self.assertEqual(DELTA, expected_delta)
        self.assertEqual(REPORT["frozen_original_terminal_vector"], list(map(str, DELTA)))
        for control, result in OUTPUTS.items():
            original = EVIDENCE["archives"][control]["stage18"]
            operand = result["operand"]
            coordinates = [i for i in range(896) if i != 363]
            self.assertEqual(operand["coordinates"], coordinates)
            self.assertEqual(operand["coordinate_count"], 895)
            self.assertEqual(operand["frozen_vector_identity"], d.closed.digest(expected_delta))
            self.assertEqual(len(operand["exact_targets"]), 895)
            self.assertEqual(len(operand["rounding_remainders"]), 895)
            expected = original.tolist()
            for index, coordinate in enumerate(coordinates):
                target = oracle.q(original.view("<f2")[coordinate])-expected_delta[coordinate]
                expected[coordinate] = oracle.nearest(target)
                self.assertEqual(Fraction(operand["exact_targets"][index]), target)
                self.assertEqual(Fraction(operand["rounding_remainders"][index]),
                                 oracle.q(result["working_stage18"].view("<f2")[coordinate])-target)
            self.assertEqual(result["working_stage18"].tolist(), expected)
            self.assertEqual(operand["working_fp16_words"], expected)
            self.assertEqual(operand["changed_coordinates"],
                             np.flatnonzero(original != result["working_stage18"]).tolist())
            self.assertNotIn(363, operand["changed_coordinates"])
            self.assertEqual(operand["coordinate363_original_word"], int(original[363]))
            self.assertEqual(operand["coordinate363_working_word"], int(original[363]))
            self.assertFalse(result["working_stage18"].flags.writeable)

    def test_mask_includes_zeros_selected_coordinates_and_preserves_negative_zero363(self):
        words = np.full(896, 0x3c00, dtype="<u2")
        words[363] = 0x8000
        delta = tuple(Fraction(1, 8) if i % 2 else Fraction() for i in range(896))
        working, operand = d.prepare(words, delta)
        self.assertEqual(working.tolist(), [
            0x8000 if i == 363 else oracle.nearest(Fraction(1)-delta[i]) for i in range(896)])
        self.assertEqual(operand["changed_coordinates"], [i for i in range(1, 896, 2) if i != 363])
        selected = set(RETAINED["rows"][0]["retained_gate_values"]["excluded_selected_coordinates"])
        self.assertTrue(selected.issubset(d.COORDINATES))
        self.assertGreater(len(OUTPUTS["scratch"]["operand"]["changed_coordinates"]), 1)

    def test_selection_mutations_refused_without_coordinate_ranking(self):
        for name, value in (("COORDINATES", tuple(range(896))),
                            ("COORDINATES", (363,)),
                            ("COORDINATES", tuple(i for i in range(896) if i != 362)),
                            ("COORDINATES", d.COORDINATES[::-1]),
                            ("COORDINATES", d.COORDINATES[:-1]+(0,)),
                            ("EXCLUDED_COORDINATE", 362),
                            ("IDS", (319, 1566)), ("PAIRS", ((319, 34319),)),
                            ("BRANCHES", ("binary64", "fp16"))):
            with patch.object(d, name, value):
                with self.assertRaises(ValueError):
                    d.preregister(EVIDENCE)
                with self.assertRaises(ValueError):
                    d.prepare(EVIDENCE["archives"]["scratch"]["stage18"], DELTA)

    def test_operand_shape_precision_nonfinite_and_saturation_refusal(self):
        words = EVIDENCE["archives"]["scratch"]["stage18"]
        for bad in (words[:-1], words.view("<f2"), np.full(896, 0x7c00, dtype="<u2"),
                    np.full(896, 0x7e00, dtype="<u2")):
            with self.assertRaises(ValueError):
                d.prepare(bad, DELTA)
        for bad in (DELTA[:-1], list(DELTA), (Fraction(1, 3),)*896, (Fraction(10**9),)*896):
            with self.assertRaises(ValueError):
                d.prepare(words, bad)

    def test_independent_retained_complement_prediction_not_all895_bridge(self):
        for row, obstruction in zip(REPORT["contrasts"], RETAINED["rows"], strict=True):
            control = next(c for c in EVIDENCE["report"]["controls"] if c["control"] == row["control"])
            pair = next(p for p in control["pairs"] if (p["left_id"], p["right_id"])
                        == (row["left_id"], row["right_id"]))
            binary = pair["branches"]["binary64"]
            anchor = Fraction(binary["unchanged_unselected_bridge"]
                              ["reference_norm"]["inverse_norm_anchor"])
            left, right = EVIDENCE["rows"][row["left_id"]], EVIDENCE["rows"][row["right_id"]]
            terms = [DELTA[i]*oracle.q(EVIDENCE["weight_array"][i])*anchor
                     *(oracle.q(left[i])-oracle.q(right[i])) for i in range(896)]
            excluded = set(binary["gate_values"]["excluded_selected_coordinates"])
            complement = sum((v for i, v in enumerate(terms) if i != 363 and i not in excluded),
                             Fraction())
            self.assertEqual(Fraction(row["predicted_delta"]), -complement)
            self.assertEqual(Fraction(row["retained_non363_complement_contribution"]), complement)
            self.assertEqual(Fraction(row["retained_coordinate363_contribution"]), terms[363])
            self.assertEqual(Fraction(row["retained_terminal_component"]["signed"]),
                             complement+terms[363])
            self.assertEqual(Fraction(row["all895_linear_bridge_delta"]),
                             -sum((v for i, v in enumerate(terms) if i != 363), Fraction()))
            self.assertNotEqual(Fraction(row["predicted_delta"]),
                                Fraction(row["all895_linear_bridge_delta"]))
            self.assertEqual(obstruction["retained_gate_values"],
                             pair["branches"][row["branch"]]["gate_values"])

    def test_fixed_reference_margins_and_common_obstruction_bindings(self):
        self.assertEqual(len(REPORT["contrasts"]), 36)
        self.assertEqual(len(RETAINED["rows"]), 36)
        for row, obstruction in zip(REPORT["contrasts"], RETAINED["rows"], strict=True):
            for key in ("control", "left_id", "right_id", "branch"):
                self.assertEqual(row[key], obstruction[key])
            for field in ("retained_actual_margin", "retained_margin_change"):
                self.assertEqual(row[field], obstruction["retained_gate_values"][field])
            reference = EVIDENCE["references"]["logits_"+row["branch"]]
            reference = reference.view("<f2") if row["branch"] == "fp16" else reference
            self.assertEqual(Fraction(row["fixed_reference_margin"]),
                             oracle.q(reference[row["left_id"]])-oracle.q(reference[row["right_id"]]))
            self.assertEqual(Fraction(row["observed_delta"]),
                             Fraction(row["intervened_margin_change"])-Fraction(row["retained_margin_change"]))
        self.assertEqual(REPORT["retained_common_component"], "UNKNOWN")
        self.assertEqual(REPORT["counterfactual_common_component"], "UNKNOWN")
        self.assertEqual(RETAINED["selection"], EVIDENCE["report"]["selection"])

    def test_directional_classification_zero_opposite_and_invalid_census(self):
        supported = [{**row, "observed_delta": row["predicted_delta"]}
                     for row in REPORT["contrasts"]]
        self.assertEqual(d.classify(supported), "supported")
        for value in ("0", str(-Fraction(supported[0]["predicted_delta"]))):
            rows = deepcopy(supported)
            rows[0]["observed_delta"] = value
            self.assertEqual(d.classify(rows), "rejected")
        expected = "supported" if all(Fraction(r["predicted_delta"])*Fraction(r["observed_delta"]) > 0
                                      for r in REPORT["contrasts"]) else "rejected"
        self.assertEqual(REPORT["classification"], expected)
        for rows in (supported[:-1], supported[::-1], [supported[0]]*36,
                     [{**r, "predicted_delta": "0"} for r in supported],
                     [{**r, "observed_delta": "NaN"} for r in supported]):
            with self.assertRaises(ValueError):
                d.classify(rows)

    def test_source_test_review_and_current_pin_mutations_refused(self):
        for module, mission, source, test, review in (
                *d.closed.CHAIN, d.vector.CLOSED_CHAIN, d.VECTOR_CHAIN):
            for path in (module.SOURCE, module.TEST,
                         d.closed.HANDOFFS / mission / "round-0001.json"):
                with self.assertRaises(ValueError):
                    d.base.read_bound({"path": str(path), "sha256": "0"*64})
        for path in (d.SOURCE, d.TEST):
            with self.assertRaises(ValueError):
                d.base.read_bound({"path": str(path), "sha256": "0"*64})
        for index in (2, 3, 4):
            chain = list(d.VECTOR_CHAIN)
            chain[index] = "0"*64
            with patch.object(d, "VECTOR_CHAIN", tuple(chain)):
                with self.assertRaises(ValueError):
                    d.authenticate_chain()

    def test_actual_operand_q24_state_and_kv_drift_refused(self):
        for key in ("stage18", "input_i", "input_z", "input_hidden",
                    "input_cache_k", "input_cache_v", "output_cache_k", "output_cache_v"):
            changed = dict(EVIDENCE)
            changed["archives"] = dict(EVIDENCE["archives"])
            archive = dict(EVIDENCE["archives"]["scratch"])
            archive[key] = archive[key].copy()
            if archive[key].size:
                archive[key].view("u1").flat[0] ^= 1
            else:
                archive[key] = np.zeros((1,), dtype=archive[key].dtype)
            changed["archives"]["scratch"] = archive
            with self.assertRaises(ValueError):
                d.closed.protect(changed, IDENTITY)

    def test_original_reference_norm_weight_and_token_asset_drift_refused(self):
        for container, key in (("reference_archive", "stage18"),
                               ("references", "logits_fp16"), ("references", "logits_binary64")):
            changed = dict(EVIDENCE)
            changed[container] = dict(EVIDENCE[container])
            changed[container][key] = EVIDENCE[container][key].copy()
            changed[container][key].view("u1").flat[0] ^= 1
            with self.assertRaises(ValueError):
                d.closed.protect(changed, IDENTITY)
        for key in ("binary64", "weight_array"):
            changed = {**EVIDENCE, key: EVIDENCE[key].copy()}
            changed[key].view("u1").flat[0] ^= 1
            with self.assertRaises(ValueError):
                d.closed.protect(changed, IDENTITY)
        with self.assertRaises(ValueError):
            d.closed.protect({**EVIDENCE, "assets": {}}, IDENTITY)

    def test_lineage_thresholds_and_historical_failures_refused(self):
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

    def test_forbidden_replay_write_external_and_both_closed_interventions(self):
        audit = {"forbidden_calls": 0}
        destinations = (
            d.SOURCE, d.TEST, d.ROOT / "chip-execution-authority.json",
            d.closed.HANDOFFS / d.VECTOR_CHAIN[1] / "round-0001.json",
            Path(next(iter(d.base.PINS.values()))["path"]),
        )
        operations = [
            lambda: d.parent.execute("forbidden"), lambda: d.parent.rmsnorm(None, None),
            lambda: d.parent.logits(None, None), lambda: d.bridge.check(),
            lambda: d.hotspot.check(), lambda: subprocess.Popen(["false"]),
            lambda: os.open(d.SOURCE, os.O_WRONLY | os.O_TRUNC), lambda: d.SOURCE.unlink(),
        ]
        for module in (d.closed, d.vector):
            for name in ("check", "run_tests", "prepare", "preregister"):
                operations.append(lambda module=module, name=name: getattr(module, name)())
        with d.hotspot.read_only(audit), d.no_closed_interventions(audit):
            for operation in operations:
                with self.assertRaises(RuntimeError):
                    operation()
            for path in destinations:
                with self.assertRaises(RuntimeError):
                    io.open(path, "w")
        self.assertEqual(audit["forbidden_calls"], len(operations)+len(destinations))

    def test_suffix_coordinate363_splice_row_scale_order_and_budget_guards(self):
        words = OUTPUTS["scratch"]["working_stage18"]
        weights = EVIDENCE["weight_array"]
        rows = np.stack([EVIDENCE["rows"][i] for i in d.IDS])
        changed = words.copy()
        changed[363] ^= 1
        for bad in (changed, words[:-1]):
            audit = {"forbidden_calls": 0, "final_rmsnorm_invocations": 0,
                     "selected_row_head_invocations": 0}
            with self.assertRaises(ValueError):
                with d.closed.suffix_only(audit, [words], weights, rows):
                    d.parent.rmsnorm(bad, weights)
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

    def test_integrity_error_emits_exactly_one_unknown_json(self):
        output = io.StringIO()
        with patch.object(d, "check", side_effect=ValueError("injected integrity drift")):
            with redirect_stdout(output):
                self.assertEqual(d.main(["--check"]), 1)
        result = json.loads(output.getvalue())
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertIn("injected integrity drift", result["integrity_error"])
        self.assertEqual(result["normal_host_review"], "REQUIRED")
        self.assertEqual(result["flags"], d.FLAGS)

    def test_non_admission_flags_and_both_closed_results_preserved(self):
        for flag in ("candidate_admitted", "policy_adopted", "successor_published",
                     "strict_FP16_state_claim", "new_token_claim", "full_model_claim",
                     "reference_reanchoring", "accepted_prefix_replay", "admission_replay",
                     "coordinate_reselection", "closed_coordinate363_intervention_replay",
                     "closed_full_vector_intervention_replay", "coordinate363_operand_mutated",
                     "middle_pair_intervention_replay", "reference_independent_native_repair_claim"):
            self.assertIs(d.FLAGS[flag], False)
        for flag in ("retained_evidence_writes", "reference_producer_dispatch",
                     "GPU_dispatch", "RTL_dispatch", "admission_dispatch"):
            self.assertEqual(d.FLAGS[flag], 0)
        for key in ("closed_coordinate363_result", "closed_full_vector_result"):
            self.assertEqual(REPORT[key]["classification"], "supported")
            self.assertEqual(REPORT[key]["directional_contrasts"], "36/36")
            self.assertIs(REPORT[key]["replayed"], False)
        d.closed.protect(EVIDENCE, IDENTITY)
