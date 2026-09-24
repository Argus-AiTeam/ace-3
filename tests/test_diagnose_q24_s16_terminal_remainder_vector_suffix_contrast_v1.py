"""Independent exact suffix oracle, full-vector selection and negative integrity gates."""

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

from ace3.model.candidates import diagnose_q24_s16_terminal_remainder_vector_suffix_contrast_v1 as d


EVIDENCE = REPORT = OUTPUTS = IDENTITY = DELTA = RETAINED = None
spec = importlib.util.spec_from_file_location("reviewed_exact_suffix_oracle", d.closed.TEST)
if spec is None or spec.loader is None:
    raise RuntimeError("reviewed independent suffix oracle unavailable")
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


class TerminalVectorTests(unittest.TestCase):
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

    def test_every_original_vector_coordinate_and_rne_operand(self):
        fp16 = EVIDENCE["reference_archive"]["stage18"].view("<f2")
        binary64 = EVIDENCE["binary64"]
        self.assertEqual(DELTA, tuple(oracle.q(a)-oracle.q(b)
                                     for a, b in zip(fp16, binary64, strict=True)))
        self.assertEqual(REPORT["frozen_original_terminal_vector"], list(map(str, DELTA)))
        for control, result in OUTPUTS.items():
            original = EVIDENCE["archives"][control]["stage18"]
            operand = result["operand"]
            self.assertEqual(operand["coordinate_count"], 896)
            self.assertEqual(operand["frozen_vector_identity"], d.closed.digest(DELTA))
            expected = []
            for i in range(896):
                target = oracle.q(original.view("<f2")[i])-DELTA[i]
                expected.append(oracle.nearest(target))
                self.assertEqual(Fraction(operand["exact_targets"][i]), target)
                self.assertEqual(Fraction(operand["rounding_remainders"][i]),
                                 oracle.q(result["working_stage18"].view("<f2")[i])-target)
            self.assertEqual(result["working_stage18"].tolist(), expected)
            self.assertEqual(operand["working_fp16_words"], expected)
            self.assertEqual(operand["changed_coordinates"],
                             np.flatnonzero(original != result["working_stage18"]).tolist())
            self.assertGreater(len(operand["changed_coordinates"]), 1)
            self.assertFalse(result["working_stage18"].flags.writeable)

    def test_full_vector_includes_previously_selected_and_zero_coordinates(self):
        words = np.full(896, 0x3c00, dtype="<u2")
        delta = tuple(Fraction(1, 8) if i % 2 else Fraction() for i in range(896))
        working, operand = d.prepare(words, delta)
        self.assertEqual(working.tolist(), [oracle.nearest(Fraction(1)-v) for v in delta])
        self.assertEqual(len(operand["exact_targets"]), 896)
        self.assertEqual(len(operand["rounding_remainders"]), 896)
        self.assertEqual(operand["changed_coordinates"], list(range(1, 896, 2)))
        selected = set(RETAINED["rows"][0]["retained_gate_values"]["excluded_selected_coordinates"])
        self.assertTrue(selected.issubset(d.COORDINATES))
        self.assertTrue(any(i != 363 for i in OUTPUTS["scratch"]["operand"]["changed_coordinates"]))

    def test_selection_mutations_refused_without_hotspot_ranking(self):
        for name, value in (("COORDINATES", tuple(range(895))),
                            ("COORDINATES", (363,)),
                            ("COORDINATES", tuple(reversed(range(896)))),
                            ("IDS", (319, 1566)), ("PAIRS", ((319, 34319),))):
            with patch.object(d, name, value):
                with self.assertRaises(ValueError):
                    d.preregister(EVIDENCE)
        self.assertIs(d.FLAGS["coordinate_reselection"], False)

    def test_operand_shape_precision_and_nonfinite_refusal(self):
        words = EVIDENCE["archives"]["scratch"]["stage18"]
        for bad in (words[:-1], words.view("<f2"), np.full(896, 0x7c00, dtype="<u2")):
            with self.assertRaises(ValueError):
                d.prepare(bad, DELTA)
        for bad in (DELTA[:-1], list(DELTA), (Fraction(1, 3),)*896, (Fraction(10**9),)*896):
            with self.assertRaises(ValueError):
                d.prepare(words, bad)

    def test_fixed_reference_margins_and_common_obstruction_bindings(self):
        self.assertEqual(len(REPORT["contrasts"]), 36)
        self.assertEqual(len(RETAINED["rows"]), 36)
        for row, obstruction in zip(REPORT["contrasts"], RETAINED["rows"], strict=True):
            for key in ("control", "left_id", "right_id", "branch"):
                self.assertEqual(row[key], obstruction[key])
            gate = obstruction["retained_gate_values"]
            for field in ("retained_actual_margin", "retained_margin_change"):
                self.assertEqual(row[field], gate[field])
            reference = EVIDENCE["references"]["logits_"+row["branch"]]
            reference = reference.view("<f2") if row["branch"] == "fp16" else reference
            self.assertEqual(Fraction(row["fixed_reference_margin"]),
                             oracle.q(reference[row["left_id"]])-oracle.q(reference[row["right_id"]]))
            self.assertEqual(Fraction(row["observed_delta"]),
                             Fraction(row["intervened_margin_change"])-Fraction(row["retained_margin_change"]))
            self.assertEqual(Fraction(row["predicted_delta"]),
                             -Fraction(row["retained_terminal_component"]["signed"]))
        self.assertEqual(REPORT["retained_common_component"], "UNKNOWN")
        self.assertEqual(REPORT["counterfactual_common_component"], "UNKNOWN")
        self.assertEqual(RETAINED["selection"], EVIDENCE["report"]["selection"])

    def test_directional_classification_and_cancelled_opposite_outcomes(self):
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
                     [{**r, "predicted_delta": "0"} for r in supported]):
            with self.assertRaises(ValueError):
                d.classify(rows)

    def test_source_test_and_review_pin_mutations_refused(self):
        for module, mission, source, test, review in (*d.closed.CHAIN, d.CLOSED_CHAIN):
            for path in (module.SOURCE, module.TEST,
                         d.closed.HANDOFFS / mission / "round-0001.json"):
                with self.assertRaises(ValueError):
                    d.base.read_bound({"path": str(path), "sha256": "0"*64})
        for path in (d.SOURCE, d.TEST):
            with self.assertRaises(ValueError):
                d.base.read_bound({"path": str(path), "sha256": "0"*64})

    def test_source_operand_state_kv_and_original_reference_drift_refused(self):
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

    def test_forbidden_replay_write_external_and_closed_intervention_guards(self):
        audit = {"forbidden_calls": 0}
        destinations = (
            d.SOURCE, d.TEST, d.ROOT / "chip-execution-authority.json",
            d.closed.HANDOFFS / d.CLOSED_CHAIN[1] / "round-0001.json",
            Path(next(iter(d.base.PINS.values()))["path"]),
        )
        operations = (
            lambda: d.parent.execute("forbidden"),
            lambda: d.parent.rmsnorm(None, None),
            lambda: d.parent.logits(None, None),
            lambda: d.bridge.check(),
            lambda: d.hotspot.check(),
            lambda: d.closed.check(),
            lambda: d.closed.run_tests(None, None, None, None, None),
            lambda: d.closed.prepare(None, None, None),
            lambda: d.closed.preregister(None),
            lambda: subprocess.Popen(["false"]),
            lambda: os.open(d.SOURCE, os.O_WRONLY | os.O_TRUNC),
            lambda: d.SOURCE.unlink(),
        )
        with d.hotspot.read_only(audit), d.no_closed_intervention(audit):
            for operation in operations:
                with self.assertRaises(RuntimeError):
                    operation()
            for path in destinations:
                with self.assertRaises(RuntimeError):
                    io.open(path, "w")
        self.assertEqual(audit["forbidden_calls"], len(operations)+len(destinations))

    def test_suffix_operand_row_order_budget_and_scale_guards(self):
        words = OUTPUTS["scratch"]["working_stage18"]
        weights = EVIDENCE["weight_array"]
        rows = np.stack([EVIDENCE["rows"][i] for i in d.IDS])
        for bad in (np.zeros_like(words), words[:-1]):
            audit = {"forbidden_calls": 0, "final_rmsnorm_invocations": 0,
                     "selected_row_head_invocations": 0}
            with self.assertRaises(ValueError):
                with d.closed.suffix_only(audit, [words], weights, rows):
                    d.parent.rmsnorm(bad, weights)
        for operation in (lambda: d.parent.logits(words, rows[::-1]),
                          lambda: d.parent.logits(words, np.stack([*rows, rows[0]])),
                          lambda: d.parent.rmsnorm(words, weights)):
            with self.assertRaises(RuntimeError):
                with d.closed.suffix_only({"forbidden_calls": 0}, [], weights, rows):
                    operation()

    def test_integrity_error_emits_single_unknown_json(self):
        output = io.StringIO()
        with patch.object(d, "check", side_effect=ValueError("injected integrity drift")):
            with redirect_stdout(output):
                self.assertEqual(d.main(["--check"]), 1)
        result = json.loads(output.getvalue())
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertIn("injected integrity drift", result["integrity_error"])
        self.assertEqual(result["normal_host_review"], "REQUIRED")
        self.assertEqual(result["flags"], d.FLAGS)

    def test_non_admission_flags_and_closed_scalar_result_preserved(self):
        for flag in ("candidate_admitted", "policy_adopted", "successor_published",
                     "strict_FP16_state_claim", "new_token_claim", "full_model_claim",
                     "reference_reanchoring", "accepted_prefix_replay", "admission_replay",
                     "coordinate_reselection", "closed_coordinate363_intervention_replay",
                     "middle_pair_intervention_replay", "reference_independent_native_repair_claim"):
            self.assertIs(d.FLAGS[flag], False)
        for flag in ("retained_evidence_writes", "reference_producer_dispatch",
                     "GPU_dispatch", "RTL_dispatch", "admission_dispatch"):
            self.assertEqual(d.FLAGS[flag], 0)
        self.assertEqual(REPORT["closed_coordinate363_result"]["classification"], "supported")
        self.assertEqual(REPORT["closed_coordinate363_result"]["directional_contrasts"], "36/36")
        self.assertIs(REPORT["closed_coordinate363_result"]["replayed"], False)
        d.closed.protect(EVIDENCE, IDENTITY)
