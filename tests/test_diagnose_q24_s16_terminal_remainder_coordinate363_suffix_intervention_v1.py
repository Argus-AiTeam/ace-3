"""Independent exact FP16/integer suffix oracle and retained-input refusal cases."""

from copy import deepcopy
from fractions import Fraction
import io
import math
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_terminal_remainder_coordinate363_suffix_intervention_v1 as d


EVIDENCE = REPORT = OUTPUTS = IDENTITY = RETAINED = None


def q(value):
    return Fraction(float(value))


def nearest(value, zero_sign=0):
    sign = 0x8000 if value < 0 or value == 0 and zero_sign else 0
    value = abs(value)
    lo, hi = 0, 0x7bff
    while lo < hi:
        mid = (lo+hi+1)//2
        decoded = q(np.asarray(mid, dtype="<u2").view("<f2"))
        if decoded <= value:
            lo = mid
        else:
            hi = mid-1
    candidates = (lo, min(lo+1, 0x7bff))
    best = min(candidates, key=lambda bits: (
        abs(q(np.asarray(bits, dtype="<u2").view("<f2"))-value), bits & 1))
    return sign | best


def rounded_division(value, denominator):
    quotient, remainder = divmod(value, denominator)
    return quotient + int(2*remainder > denominator
                          or 2*remainder == denominator and quotient % 2 == 1)


def oracle(words, weights, rows):
    x = [int(q(v)*(1 << 24)) for v in words.view("<f2")]
    w = [int(q(v)*(1 << 24)) for v in weights]
    mean = rounded_division(sum(v*v for v in x)+281474977*896, 896)
    root = math.isqrt(mean)
    normalized = []
    for i, (a, b) in enumerate(zip(x, w, strict=True)):
        product = a*b
        value = Fraction(rounded_division(abs(product), root), 1 << 24)
        if product < 0:
            value = -value
        normalized.append(nearest(value, bool(int(words[i]) >> 15)
                                  ^ bool(int(weights.view("<u2")[i]) >> 15)))
    exact = [q(v) for v in np.asarray(normalized, dtype="<u2").view("<f2")]
    logits = [nearest(sum((v*q(weight) for v, weight in zip(exact, row, strict=True)),
                          Fraction())) for row in rows]
    return normalized, logits, mean, root


class TerminalSuffixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is None:
            raise RuntimeError("run the native --check for authenticated nonzero focused tests")

    def test_independent_suffix_oracle_all_controls(self):
        rows = np.stack([EVIDENCE["rows"][i] for i in d.IDS])
        for result in OUTPUTS.values():
            norm, logits, mean, root = oracle(result["working_stage18"],
                                             EVIDENCE["weight_array"], rows)
            self.assertEqual(result["rmsnorm"].tolist(), norm)
            self.assertEqual(result["logits"].tolist(), logits)
            self.assertEqual(result["scalars"], {"mean_q48": mean, "root_q24": root})

    def test_exact_coordinate_operand_and_rounding(self):
        for control, result in OUTPUTS.items():
            original = EVIDENCE["archives"][control]["stage18"]
            delta = q(EVIDENCE["reference_archive"]["stage18"].view("<f2")[363])-q(EVIDENCE["binary64"][363])
            target = q(original.view("<f2")[363])-delta
            self.assertEqual(int(result["working_stage18"][363]), nearest(target))
            self.assertEqual(Fraction(result["operand"]["exact_target"]), target)
            self.assertTrue(np.array_equal(np.delete(original, 363),
                                           np.delete(result["working_stage18"], 363)))
            self.assertFalse(result["working_stage18"].flags.writeable)

    def test_frozen_selection_and_hotspot_mutations_refused(self):
        predictions = d.preregister(RETAINED)
        self.assertEqual(len(predictions), 18)
        with patch.object(d, "COORDINATE", 362):
            with self.assertRaises(ValueError):
                d.preregister(RETAINED)
        for ties in ([362], [362, 363], []):
            retained = deepcopy(RETAINED)
            retained["pairs"][0]["rows"][1]["top_terminal_magnitude_ties"] = ties
            with self.assertRaises(ValueError):
                d.preregister(retained)
        retained = deepcopy(RETAINED)
        retained["pairs"][0]["rows"].pop()
        with self.assertRaises(ValueError):
            d.preregister(retained)

    def test_direction_and_fixed_original_references(self):
        rows = REPORT["contrasts"]
        self.assertEqual(len(rows), 36)
        for row in rows:
            ref = EVIDENCE["references"]["logits_"+row["branch"]]
            if row["branch"] == "fp16":
                ref = ref.view("<f2")
            self.assertEqual(Fraction(row["fixed_reference_margin"]),
                             q(ref[row["left_id"]])-q(ref[row["right_id"]]))
            self.assertEqual(Fraction(row["observed_delta"]),
                             Fraction(row["intervened_margin_change"])-Fraction(row["retained_margin_change"]))
        expected = "supported" if all(Fraction(r["predicted_delta"])*Fraction(r["observed_delta"]) > 0
                                      for r in rows) else "rejected"
        self.assertEqual(REPORT["classification"], expected)
        self.assertEqual(REPORT["retained_common_component"], "UNKNOWN")
        self.assertEqual(REPORT["counterfactual_common_component"], "UNKNOWN")

    def test_rejected_absent_opposite_and_unknown_incomplete(self):
        rows = [{"predicted_delta": "1", "observed_delta": "2"} for _ in range(36)]
        self.assertEqual(d.classify(rows), "supported")
        for value in ("0", "-1"):
            self.assertEqual(d.classify([{**r, "observed_delta": value} for r in rows]), "rejected")
        with self.assertRaises(ValueError):
            d.classify(rows[:-1])

    def test_state_kv_operand_and_reference_mutations_refused(self):
        for key in ("stage18", "input_i", "input_z", "input_hidden",
                    "input_cache_k", "input_cache_v", "output_cache_k", "output_cache_v"):
            changed = dict(EVIDENCE)
            changed["archives"] = dict(EVIDENCE["archives"])
            archive = dict(changed["archives"]["scratch"])
            archive[key] = archive[key].copy()
            if archive[key].size:
                archive[key].flat[0] ^= 1
            else:
                archive[key] = np.zeros((1,), dtype=archive[key].dtype)
            changed["archives"]["scratch"] = archive
            with self.assertRaises(ValueError):
                d.protect(changed, IDENTITY)
        changed = {**EVIDENCE, "binary64": EVIDENCE["binary64"].copy()}
        changed["binary64"][363] += 1
        with self.assertRaises(ValueError):
            d.protect(changed, IDENTITY)
        for container, key in (("reference_archive", "stage18"),
                               ("references", "logits_fp16"),
                               ("references", "logits_binary64"),
                               ("arrays", "scratch")):
            changed = dict(EVIDENCE)
            changed[container] = dict(EVIDENCE[container])
            if container == "arrays":
                changed[container][key] = dict(EVIDENCE[container][key])
                value = EVIDENCE[container][key]["rmsnorm"].copy()
                value.flat[0] ^= 1
                changed[container][key]["rmsnorm"] = value
            else:
                value = EVIDENCE[container][key].copy()
                value.view("u1").flat[0] ^= 1
                changed[container][key] = value
            with self.assertRaises(ValueError):
                d.protect(changed, IDENTITY)

    def test_source_pins_and_review_mutations_refused(self):
        for module, mission, source, test, review in d.CHAIN:
            for path in (module.SOURCE, module.TEST, d.HANDOFFS / mission / "round-0001.json"):
                with self.assertRaises(ValueError):
                    d.base.read_bound({"path": str(path), "sha256": "0"*64})
        for path in (d.SOURCE, d.TEST):
            with self.assertRaises(ValueError):
                d.base.read_bound({"path": str(path), "sha256": "0"*64})
        d.protect(EVIDENCE, IDENTITY)

    def test_lineage_and_historical_failure_mutations_refused(self):
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

    def test_forbidden_replay_write_and_external_guards(self):
        audit = {"forbidden_calls": 0}
        with d.hotspot.read_only(audit):
            for operation in (
                lambda: d.parent.execute("forbidden"),
                lambda: d.parent.rmsnorm(None, None),
                lambda: d.parent.logits(None, None),
                lambda: d.hotspot.dominance.check(),
                lambda: d.hotspot.census.check(),
                lambda: d.bridge.check(),
                lambda: subprocess.Popen(["false"]),
                lambda: io.open(d.SOURCE, "w"),
                lambda: os.open(d.SOURCE, os.O_WRONLY | os.O_TRUNC),
                lambda: Path(d.SOURCE).unlink(),
            ):
                with self.assertRaises(RuntimeError):
                    operation()
            for destination in (
                d.TEST, d.ROOT / "chip-execution-authority.json",
                d.HANDOFFS / d.CHAIN[0][1] / "round-0001.json",
                Path(next(iter(d.base.PINS.values()))["path"]),
            ):
                with self.assertRaises(RuntimeError):
                    io.open(destination, "w")
        self.assertEqual(audit["forbidden_calls"], 14)

    def test_suffix_operand_and_scale_guards(self):
        words = OUTPUTS["scratch"]["working_stage18"]
        weights = EVIDENCE["weight_array"]
        rows = np.stack([EVIDENCE["rows"][i] for i in d.IDS])
        for bad in (np.zeros_like(words), words[:-1]):
            audit = {"forbidden_calls": 0, "final_rmsnorm_invocations": 0,
                     "selected_row_head_invocations": 0}
            with self.assertRaises(ValueError):
                with d.suffix_only(audit, [words], weights, rows):
                    d.parent.rmsnorm(bad, weights)
        audit = {"forbidden_calls": 0}
        with self.assertRaises(RuntimeError):
            with d.suffix_only(audit, [], weights, rows):
                d.parent.logits(words, np.stack([*rows, rows[0]]))

    def test_non_admission_flags_and_thresholds(self):
        for flag in ("candidate_admitted", "policy_adopted", "successor_published",
                     "strict_FP16_state_claim", "new_token_claim", "full_model_claim",
                     "reference_reanchoring", "accepted_prefix_replay", "admission_replay"):
            self.assertIs(d.FLAGS[flag], False)
        self.assertEqual(d.FLAGS["retained_evidence_writes"], 0)
        d.base.check_history(EVIDENCE["result"])
