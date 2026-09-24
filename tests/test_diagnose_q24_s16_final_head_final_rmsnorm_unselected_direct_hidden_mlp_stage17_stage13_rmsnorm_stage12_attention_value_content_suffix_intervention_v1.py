"""Independent exact directional, bit-operand, closure and dispatch checks."""

from fractions import Fraction
import copy
import os
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_stage13_rmsnorm_stage12_attention_value_content_suffix_intervention_v1 as d


EVIDENCE = None


def half(word):
    exponent, mantissa = (int(word) >> 10) & 31, int(word) & 1023
    if exponent == 31:
        raise ValueError("nonfinite half")
    value = (Fraction(mantissa, 1 << 24) if exponent == 0 else
             Fraction(1024 + mantissa) * Fraction(2) ** (exponent - 25))
    return -value if int(word) & 32768 else value


class InterventionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is None:
            raise RuntimeError("Run the diagnostic's native --check; no unguarded fixture replay")
        (cls.inputs, cls.retained, cls.plan, cls.e, cls.outputs, cls.rows,
         cls.reports, cls.tensors, cls.operands) = EVIDENCE

    def test_reviewed_parent_source_and_test_pins(self):
        for pin in d.PARENT_PINS:
            self.assertEqual(d.parent.record(d.ROOT / pin["path"]), pin)
        with patch.object(d, "PARENT_PINS", ({**d.PARENT_PINS[0], "sha256": "0"*64},)):
            with self.assertRaises(ValueError):
                d.authenticate_parent()

    def test_independent_native_nibble_value_prediction_and_selection(self):
        tensors = self.inputs[0][0][0][0][2]
        totals = [[Fraction() for _ in range(128)] for _ in self.rows]
        columns = {}
        for account in self.retained["attention_score_value_accounts"]:
            core = self.retained["attention_score_value_local_accounts"][account["local_account_key"]]
            coordinate = core["output_coordinate"]
            if coordinate not in columns:
                shift = 4*(0, 4, 1, 5, 2, 6, 3, 7)[coordinate % 8]
                columns[coordinate] = [
                    (((int(tensors["qweight"][i, coordinate//8]) >> shift) & 15)
                     - ((int(tensors["qzeros"][i//128, coordinate//8]) >> shift) & 15))
                    * half(tensors["scales"].view("<u2")[i//128, coordinate]) for i in range(896)]
            a = self.e["archives"][core["control"]]["stage07"]
            r = self.e["reference_archive"]["stage07"]
            factor = Fraction(account["multipliers"][d.previous.FIELDS[-1]])
            for k in range(128):
                kv, dim = divmod(k, 64)
                weight = sum((columns[coordinate][h*64+dim] for h in range(kv*7, (kv+1)*7)),
                             Fraction())
                totals[account["row_index"]][k] += (half(a[k])-half(r[k]))*weight*factor
        masses = [sum((abs(row[k]) for row in totals), Fraction()) for k in range(128)]
        k = min(range(128), key=lambda i: (-masses[i], i))
        self.assertEqual(self.plan["value_coordinate"], k)
        self.assertEqual(self.plan["coordinate_selection_masses"], list(map(str, masses)))
        self.assertEqual(self.plan["predicted_row_deltas"], [str(-row[k]) for row in totals])

    def test_exact_single_component_and_unchanged_scores_state_kv(self):
        k = self.plan["value_coordinate"]
        for c in d.CONTROLS:
            actual, changed = self.e["archives"][c], self.outputs[c]
            expected = actual["stage07"].copy()
            expected[k] = self.e["reference_archive"]["stage07"][k]
            np.testing.assert_array_equal(changed["stage07"], expected)
            for key in d.suffix.PROTECTED:
                self.assertEqual(changed[key].tobytes(), actual[key].tobytes(), key)
            np.testing.assert_array_equal(changed["stage09"], np.full(14, 0x3c00, dtype="<u2"))
            expected_av = expected.reshape(2, 64).repeat(7, axis=0).reshape(896)
            np.testing.assert_array_equal(changed["stage10"], expected_av)

    def test_independent_mlp_and_margin_closure(self):
        for row, retained in zip(self.rows, self.retained["rows"], strict=True):
            c = row["control"]
            old, new = self.e["archives"][c], self.outputs[c]
            terms = [(half(new["stage17"][h["output_coordinate"]])
                      - half(old["stage17"][h["output_coordinate"]]))
                     * Fraction(h["retained_weighted_row_factor"]) for h in retained["hotspots"]]
            self.assertEqual(Fraction(row["mlp_maxima_delta"]), sum(terms, Fraction()))
            baseline = self.e["arrays"][c]["logits"]
            delta = (half(new["pair_logits"][0])-half(new["pair_logits"][1])
                     - half(baseline[34319])+half(baseline[13]))
            self.assertEqual(Fraction(row["margin_delta"]), delta)
            prediction = Fraction(row["predicted_delta"])
            for field in ("mlp", "margin"):
                measured = row["mlp_maxima_delta" if field == "mlp" else "margin_delta"]
                self.assertEqual(prediction+Fraction(row[field+"_prediction_remainder"]),
                                 Fraction(measured))

    def test_independent_pair_head_dot_oracle(self):
        for arrays in self.outputs.values():
            hidden = list(map(half, arrays["final_rmsnorm"]))
            for i, weights in enumerate(self.operands[1].view("<u2")):
                value = sum((h*half(w) for h, w in zip(hidden, weights, strict=True)), Fraction())
                expected = np.asarray(float(value), dtype="<f2").view("<u2").item()
                self.assertEqual(int(arrays["pair_logits"][i]), expected)

    def test_original_references_and_thresholds_not_reanchored(self):
        summary = self.e["result"]["preflight"]
        for branch in ("fp16", "binary64"):
            self.assertEqual(summary["final_reference"]["reference"]["input_"+branch],
                             summary["L23_original_reference"]["reference"][branch])
        self.assertEqual(self.retained["common_component"], "UNKNOWN")
        d.base.check_history(self.e["result"])
        for reports in self.reports.values():
            self.assertEqual([r["stage"] for r in reports], list(range(10, 19)))

    def test_directional_classifier_supported_rejected_and_invalid(self):
        rows = [{**r, "predicted_delta": "2", "mlp_maxima_delta": "3",
                 "margin_delta": "1", "direction_observed": True} for r in self.rows]
        reports = {c: [{**r, "status": "PASS"} for r in rs] for c, rs in self.reports.items()}
        self.assertEqual(d.classify(rows, reports), "supported")
        for value in ("0", "-1"):
            changed = [{**rows[0], "margin_delta": value, "direction_observed": False}, *rows[1:]]
            self.assertEqual(d.classify(changed, reports), "rejected")
        failed = copy.deepcopy(reports)
        failed[d.CONTROLS[0]][-1]["status"] = "FAIL"
        self.assertEqual(d.classify(rows, failed), "rejected")
        with self.assertRaises(ValueError):
            d.classify([{**rows[0], "predicted_delta": "0"}, *rows[1:]], reports)
        with self.assertRaises(ValueError):
            d.classify(rows[:-1], reports)

    def test_operand_state_kv_and_score_mutations_rejected(self):
        c, k = d.CONTROLS[0], self.plan["value_coordinate"]
        actual, arrays, ref = self.e["archives"][c], self.outputs[c], self.e["reference_archive"]
        for key in (*d.suffix.PROTECTED, "stage07"):
            changed = arrays[key].copy()
            if changed.size:
                changed.view("u1").flat[0] ^= 1
            else:
                changed = np.zeros((1, 128), dtype=changed.dtype)
            with self.assertRaises(ValueError, msg=key):
                d.protected(actual, {**arrays, key: changed}, ref, k)
        with self.assertRaises(ValueError):
            d.prepare(actual, ref, 128)

    def test_source_token_reference_threshold_and_lineage_mutations_rejected(self):
        for field, value in (("source_token_id", 9708), ("source_position", 1)):
            with self.assertRaises(ValueError):
                d.source.bind_authority({**self.e, "result": {**self.e["result"], field: value}})
        for field in ("thresholds", "L23_original_reference", "final_reference"):
            mutated = {**self.e, "result": {**self.e["result"], "preflight": {
                **self.e["result"]["preflight"], field: {}}}}
            with self.assertRaises(ValueError):
                d.source.bind_authority(mutated)

    def test_nonunit_probability_and_changed_downstream_closure_rejected(self):
        for key, value in (("common_component", "mlp_stage17"), ("left_id", 319)):
            with self.assertRaises(ValueError):
                d.preregister({**self.retained, key: value})
        accounts = self.retained["attention_score_value_accounts"]
        changed = {**accounts[0], "closure_residuals": {d.previous.FIELDS[0]: "1"}}
        with self.assertRaises(ValueError):
            d.preregister({**self.retained, "attention_score_value_accounts": [changed, *accounts[1:]]})
        controls = copy.deepcopy(self.retained["attention_score_value_controls"])
        controls[0]["scores_by_absolute_delta"][0]["actual_probability"] = "0"
        with self.assertRaises(ValueError):
            d.preregister({**self.retained, "attention_score_value_controls": controls})

    def test_forbidden_predecessors_prefix_references_full_head_and_writes(self):
        audit = {"forbidden_calls": 0}
        with d.suffix_only(audit, {}, self.tensors, self.operands):
            for call in (
                lambda: d.previous.check(), lambda: d.previous.collect(None),
                lambda: d.suffix.execute(), lambda: d.native._stages(None, None, None, None),
                lambda: d.native.projection(None, "model.layers.0.self_attn.q_proj", None),
                lambda: d.parent.top_k(None, words=True),
                lambda: subprocess.run(["forbidden"]), lambda: os.system("forbidden"),
                lambda: open(d.SOURCE, "w"), lambda: d.TEST.write_text("forbidden"),
                lambda: os.open(d.TEST, os.O_WRONLY),
                lambda: d.TEST.unlink(), lambda: d.TEST.rename(d.SOURCE),
            ):
                with self.assertRaises(RuntimeError):
                    call()
            with self.assertRaises(ValueError):
                d.parent.logits(None, np.zeros((3, 896), dtype="<f2"))
        self.assertEqual(audit["forbidden_calls"], 13)

    def test_preserved_false_zero_nonadmission_flags_and_finite_outputs(self):
        for name in ("candidate_admitted", "policy_adopted", "successor_published",
                     "strict_FP16_state_claim", "new_token_claim", "full_model_claim",
                     "accepted_prefix_replay", "reference_recomputation"):
            self.assertIs(d.FLAGS[name], False)
        for name in ("score_replay", "softmax_replay", "v_projection_replay",
                     "retained_evidence_writes", "prefix_invocations", "admission_invocations"):
            self.assertEqual(d.FLAGS[name], 0)
        for arrays in self.outputs.values():
            for stage in range(10, 19):
                self.assertTrue(np.isfinite(arrays[f"stage{stage:02d}"].view("<f2")).all())
