"""Independent half-bit/native-nibble oracles for the stage10 o_proj bridge."""

from fractions import Fraction
import io
import json
import math
import os
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_stage13_rmsnorm_stage12_attention_o_projection_input_bridge_v1 as d
from ace3.model.candidates import q24_s16_toward_zero_native_v1 as native


EVIDENCE = None


def half(word):
    exponent, mantissa = (word >> 10) & 31, word & 1023
    if exponent == 31:
        raise ValueError("nonfinite FP16")
    value = (Fraction(mantissa, 1 << 24) if exponent == 0
             else Fraction(1024+mantissa)*Fraction(2)**(exponent-25))
    return -value if word & 32768 else value


def weight(tensors, output, coordinate):
    shift = 4*(0, 4, 1, 5, 2, 6, 3, 7)[output % 8]
    q = (int(tensors["qweight"][coordinate, output//8]) % (1 << 32))//(1 << shift) % 16
    z = (int(tensors["qzeros"][coordinate//128, output//8]) % (1 << 32))//(1 << shift) % 16
    return (q-z)*half(int(tensors["scales"].view("<u2")[coordinate//128, output]))


def mass(values):
    values = list(values)
    signed, absolute = sum(values, Fraction()), sum(map(abs, values), Fraction())
    return {"signed": str(signed), "absolute": str(absolute),
            "cancellation_absolute_mass": str(absolute-abs(signed))}


def replace_path(value, path, replacement):
    if not path:
        return replacement
    result = value.copy()
    result[path[0]] = replace_path(value[path[0]], path[1:], replacement)
    return result


class OProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is None:
            audit = {"forbidden_calls": 0}
            cls.inputs = d.collect(audit)
            with d.read_only(audit):
                cls.result = d.report(*cls.inputs)
            if audit["forbidden_calls"]:
                raise AssertionError("forbidden fixture work")
        else:
            cls.inputs, cls.result = EVIDENCE
        cls.parent_inputs, cls.retained = cls.inputs
        cls.attention_inputs = cls.parent_inputs[0][0]
        cls.e = d.source.evidence(cls.attention_inputs[0])
        cls.oracles, columns = {}, {}
        for key, core in cls.result["attention_o_projection_local_accounts"].items():
            output = core["output_coordinate"]
            if output not in columns:
                columns[output] = [weight(cls.attention_inputs[2], output, i) for i in range(896)]
            a, r = cls.e["archives"][core["control"]], cls.e["reference_archive"]
            actual = [half(int(x)) for x in a["stage10"]]
            reference = [half(int(x)) for x in r["stage10"]]
            weights = columns[output]
            terms = [(x-y)*w for x, y, w in zip(actual, reference, weights, strict=True)]
            selected = sorted(range(896), key=lambda i: (-abs(terms[i]), i))[:8]
            actual_sum = sum((x*w for x, w in zip(actual, weights, strict=True)), Fraction())
            reference_sum = sum((x*w for x, w in zip(reference, weights, strict=True)), Fraction())
            cls.oracles[key] = {
                "actual": actual, "reference": reference, "weights": weights,
                "terms": terms, "selected": selected, "actual_sum": actual_sum,
                "reference_sum": reference_sum, "actual_o": half(int(a["stage11"][output])),
                "reference_o": half(int(r["stage11"][output])),
                "masses": {
                    "selected": mass(terms[i] for i in selected),
                    "unselected": mass(terms[i] for i in range(896) if i not in selected),
                    "full": mass(terms)},
            }

    def test_independent_half_bits_and_native_nibble_order(self):
        self.assertEqual(half(1), Fraction(1, 1 << 24))
        self.assertEqual(half(0x8000), 0)
        self.assertEqual(half(0xbc00), -1)
        self.assertEqual(half(0x7bff), 65504)
        tensors = {
            "qweight": np.full((896, 112), 0x76543210, dtype="<i4"),
            "qzeros": np.full((7, 112), 0x11111111, dtype="<i4"),
            "scales": np.ones((7, 896), dtype="<f2")}
        for output, nibble in enumerate((0, 4, 1, 5, 2, 6, 3, 7)):
            self.assertEqual(weight(tensors, output, 128), nibble-1)
            self.assertEqual(d.attention.weight_column(tensors, output)[128], nibble-1)
        with self.assertRaises(ValueError):
            half(0x7c00)

    def test_every_exact_stage10_projection_sum_and_separate_boundaries(self):
        for key, o in self.oracles.items():
            core = self.result["attention_o_projection_local_accounts"][key]
            self.assertEqual(core["actual_exact_input_weight_sum"], str(o["actual_sum"]))
            self.assertEqual(core["original_exact_input_weight_sum"], str(o["reference_sum"]))
            expected = (sum(o["terms"], Fraction()), o["actual_o"]-o["actual_sum"],
                        o["reference_sum"]-o["reference_o"])
            self.assertEqual([Fraction(core[t]) for t in d.TERMS], list(expected))
            self.assertEqual(sum(expected), o["actual_o"]-o["reference_o"])
            self.assertEqual(core["retained_o_projection_delta"], str(sum(expected)))
            self.assertEqual(core["closure_residual"], "0")
            self.assertEqual((core["input_stage"], core["output_stage"], core["source_token_id"]),
                             ("stage10_fp16", "stage11_fp16", 9707))

    def test_every_ranked_coordinate_and_signed_absolute_cancellation_mass(self):
        for key, o in self.oracles.items():
            core = self.result["attention_o_projection_local_accounts"][key]
            selected = core["largest_absolute_input_coordinates"]
            self.assertEqual([e["coordinate"] for e in selected], o["selected"])
            for entry in selected:
                i = entry["coordinate"]
                self.assertEqual(entry, {
                    "coordinate": i, "input_group": i//128, "actual_input": str(o["actual"][i]),
                    "original_input_fp16_input": str(o["reference"][i]),
                    "input_delta": str(o["actual"][i]-o["reference"][i]),
                    "weight": str(o["weights"][i]), "signed_contribution": str(o["terms"][i])})
            self.assertEqual(core["input_coordinate_totals"], o["masses"])

    def test_every_seven_g128_complement_groups_and_closure(self):
        for key, o in self.oracles.items():
            split = self.result["attention_o_projection_local_accounts"][key]["unselected_stage10_inputs"]
            self.assertEqual(split["coordinate_count"], 888)
            self.assertEqual(split["excluded_ranked_coordinates"], o["selected"])
            self.assertEqual(split["totals"], o["masses"]["unselected"])
            groups = split["groups_in_input_order"]
            self.assertEqual([g["input_group"] for g in groups], list(range(7)))
            self.assertEqual(sum(g["coordinate_count"] for g in groups), 888)
            self.assertEqual(sum(g["excluded_ranked_coordinate_count"] for g in groups), 8)
            for index, group in enumerate(groups):
                indices = [i for i in range(index*128, (index+1)*128) if i not in o["selected"]]
                self.assertEqual((group["start_coordinate"], group["end_coordinate_exclusive"]),
                                 (index*128, (index+1)*128))
                self.assertEqual(group["coordinate_count"], len(indices))
                self.assertEqual({k: group[k] for k in mass([])}, mass(o["terms"][i] for i in indices))

    def test_entire_inherited_selected_value_source_report_unchanged(self):
        additions = {
            "attention_o_projection_local_accounts", "attention_o_projection_input_accounts",
            "attention_o_projection_input_summary", "attention_o_projection_input_weighting"}
        self.assertEqual({k: v for k, v in self.result.items() if k not in additions}, self.retained)
        self.assertEqual(set(self.oracles), set(self.retained["attention_stage11_local_accounts"]))

    def test_every_logical_account_downstream_identity_and_summary(self):
        accounts = self.result["attention_o_projection_input_accounts"]
        sources = list(d.sources(self.retained))
        self.assertEqual(len(accounts), len(sources))
        for account, (location, entry) in zip(accounts, sources, strict=True):
            self.assertEqual({k: account[k] for k in location}, location)
            old = entry["attention_stage11_source_value"]
            self.assertEqual(account["local_account_key"], old["local_account_key"])
            self.assertEqual(account["multipliers"], old["multipliers"])
            o = self.oracles[account["local_account_key"]]
            for field in d.FIELDS:
                multiplier = Fraction(old["multipliers"][field])
                expected = [v*multiplier for v in (
                    sum(o["terms"], Fraction()), o["actual_o"]-o["actual_sum"],
                    o["reference_sum"]-o["reference_o"])]
                self.assertEqual([Fraction(t[field]) for t in account["terms"]], expected)
                self.assertEqual(str(sum(expected)), old["retained_parent_attention_stage11_delta"][field])
                self.assertEqual(account["closure_residuals"][field], "0")
                for group in d.GROUPS:
                    m = o["masses"][group]
                    signed = Fraction(m["signed"])*multiplier
                    absolute = Fraction(m["absolute"])*abs(multiplier)
                    self.assertEqual(account["input_coordinate_totals"][group][field],
                                     {"signed": str(signed), "absolute": str(absolute),
                                      "cancellation_absolute_mass": str(absolute-abs(signed))})
        summary = self.result["attention_o_projection_input_summary"]
        self.assertEqual(len(accounts), 2304)
        self.assertEqual(summary["account_count"], 2304)
        self.assertEqual(summary["input_coordinate_count"], 2304*896)
        self.assertEqual(summary["unselected_input_coordinate_count"], 2304*888)
        self.assertEqual(summary["unselected_input_group_count"], 2304*7)
        for index, term in enumerate(d.TERMS):
            for field in d.FIELDS:
                self.assertEqual(summary["term_totals"][term][field],
                                 mass(Fraction(a["terms"][index][field]) for a in accounts))

    def test_independently_propagated_norm_gate_up_down_row_opposite_factors(self):
        norm_inputs = self.attention_inputs[0][0]
        projection_inputs = norm_inputs[0]
        for row in self.retained["rows"]:
            for hotspot in row["hotspots"]:
                for selected in hotspot["selected_stage16_coordinates"]:
                    j = selected["selected_down_input"]["coordinate"]
                    down = weight(projection_inputs[0][4], hotspot["output_coordinate"], j)
                    for kind in ("gate_proj", "up_proj"):
                        opposite = half(int(self.e["reference_archive"][
                            "stage15" if kind == "gate_proj" else "stage14"][j]))
                        outer = down*opposite*Fraction(hotspot["retained_weighted_row_factor"])
                        for entry in selected[kind]["ranked_stage13_rmsnorm_sources"]:
                            i = entry["retained_ranked_input"]["coordinate"]
                            norm = half(int(norm_inputs[1].view("<u2")[i]))
                            anchor = Fraction(norm_inputs[2]["original_input_fp16"]["inverse_norm_anchor"])
                            gu = weight(projection_inputs[1][kind], j, i)
                            expected = (Fraction(1), norm*anchor, norm*anchor*gu, norm*anchor*gu*outer)
                            self.assertEqual(tuple(Fraction(entry["attention_stage11_source_value"]
                                                            ["multipliers"][f]) for f in d.FIELDS), expected)

    def test_local_and_downstream_mutations_rejected(self):
        location, entry = next(d.sources(self.retained))
        inherited = entry["attention_stage11_source_value"]
        key = inherited["local_account_key"]
        core = self.result["attention_o_projection_local_accounts"][key]
        local = self.retained["attention_stage11_local_accounts"][key]
        for name in (*d.TERMS, "retained_o_projection_delta", "control", "output_coordinate"):
            with self.assertRaises(ValueError):
                d.bind_local({**core, name: "123"}, local)
        field = d.FIELDS[-1]
        for path in (["multipliers", field], ["terms", 4, field],
                     ["retained_parent_attention_stage11_delta", field], ["closure_residuals", field]):
            with self.assertRaises(ValueError):
                d.weighted_account(core, replace_path(inherited, path, "123"), location)

    def test_parent_pins_latest_complement_and_binding_mutations(self):
        self.assertIn("value_projection_unselected_stage00_complement", str(d.previous.SOURCE))
        for pin in d.PARENT_PINS:
            with self.assertRaises(ValueError):
                d.base.read_bound({**pin, "sha256": "0"*64})
        with patch.object(d, "PARENT_PINS", ({**d.PARENT_PINS[0], "sha256": "0"*64},)):
            with self.assertRaises(ValueError):
                d.authenticate_parent()
        with patch.object(d.previous, "report", return_value=self.retained):
            for field, value in (("common_component", "ADMITTED"), ("internal_reference", "binary64"),
                                 ("binary64_internal_stages", "RECONSTRUCTED"),
                                 ("value_projection_local_accounts", {}),
                                 ("attention_stage11_local_accounts", {})):
                with self.assertRaises(ValueError):
                    d.bind_parent(self.parent_inputs, {**self.retained, field: value})

    def test_actual_original_stage10_stage11_state_and_lineage_rejected(self):
        control = d.parent.CONTROLS[0]
        norm = self.attention_inputs[0][0][1]
        for reference in (False, True):
            prefix = ["reference_archive"] if reference else ["archives", control]
            archive = self.e["reference_archive"] if reference else self.e["archives"][control]
            keys = ("stage03", "stage07", "stage10", "stage11")
            if not reference:
                keys += ("input_i", "input_z", "scratch_i", "scratch_z",
                         "output_cache_k", "output_cache_v")
            for key in keys:
                changed = archive[key].copy()
                changed.flat[0] ^= 1
                with self.assertRaises(ValueError):
                    d.source.rmsnorm.bind_inputs(replace_path(self.e, [*prefix, key], changed), norm)

    def test_canonical_o_proj_operands_and_empty_p0_kv_rejected(self):
        inputs, tensors = self.attention_inputs[0], self.attention_inputs[2]
        for suffix, tensor in tensors.items():
            changed = tensor.copy()
            changed.view("u1").flat[0] ^= 1
            with self.assertRaises(ValueError):
                d.source.bind_attention(inputs, {**tensors, suffix: changed})
        archive = self.e["archives"][d.parent.CONTROLS[0]]
        for kind in ("k", "v"):
            with self.assertRaises(ValueError):
                d.attention.attention_operands(
                    {**archive, "input_cache_"+kind: np.zeros((1, 128), dtype="<u2")}, actual=True)

    def test_source_token_reference_threshold_authority_mutations_rejected(self):
        binding = ["result", "preflight", "L23_original_reference", "reference"]
        for path, value in (
            ([*binding, "prior_kv"], "nonempty"),
            ([*binding, "canonical", d.attention.PROJECTION+"qweight", "sha256"], "0"*64),
            ([*binding, "fp16", "sha256"], "0"*64),
            (["result", "preflight", "thresholds"], {}),
            (["result", "controls", 0, "parent", "terminal_archive", "sha256"], "0"*64)):
            with self.assertRaises(ValueError):
                d.source.bind_authority(replace_path(self.e, path, value))
        changed = {**self.e["result"], "source_token_id": 9708}
        with self.assertRaises(ValueError):
            d.source.bind_authority({**self.e, "result": changed})

    def test_zero_ties_negative_weights_group_edges_and_invalid_inputs(self):
        zero = [Fraction()] * 896
        actual, weights = zero.copy(), zero.copy()
        selected = [0, 127, 128, 255, 256, 383, 384, 895]
        for i in selected:
            actual[i], weights[i] = Fraction(1), Fraction(3 if i % 2 else -3)
        actual[500], weights[500] = Fraction(1), Fraction(1)
        actual[501], weights[501] = Fraction(1), Fraction(-1)
        core = d.projection_account(actual, zero, weights, Fraction(2), Fraction(-1), 895)
        split = core["unselected_stage10_inputs"]
        self.assertEqual(split["excluded_ranked_coordinates"], selected)
        self.assertEqual(split["totals"], mass([Fraction(1), Fraction(-1)]))
        self.assertEqual([g["coordinate_count"] for g in split["groups_in_input_order"]],
                         [126, 126, 126, 127, 128, 128, 127])
        core = d.projection_account(zero, zero, zero, Fraction(), Fraction(), 0)
        self.assertEqual(core["unselected_stage10_inputs"]["excluded_ranked_coordinates"], list(range(8)))
        self.assertEqual(core["input_coordinate_totals"]["full"], mass([]))
        for coordinate in (-1, 896, True):
            with self.assertRaises(ValueError):
                d.projection_account(zero, zero, zero, Fraction(), Fraction(), coordinate)
        with self.assertRaises(ValueError):
            d.projection_account(zero[:-1], zero, zero, Fraction(), Fraction(), 0)

    def test_forbidden_checks_operator_replay_reconstruction_and_writes(self):
        target = d.parent.OUTPUT / "forbidden-o-projection-input-bridge"
        calls = (
            lambda: d.previous.check(), lambda: d.previous.run_tests(None, None),
            lambda: d.previous.collect(None), lambda: d.source.check(),
            lambda: d.attention.measure(), lambda: d.parent.execute("forbidden"),
            lambda: d.parent.rmsnorm(None, None), lambda: d.parent.logits(None, None),
            lambda: native._stages(None, None, None, None),
            lambda: d.bridge.residual.local.local_reference(None),
            lambda: d.bridge.margin.contributions.dyadic_dot(None, None),
            lambda: math.exp(0), lambda: np.exp(0),
            lambda: subprocess.run(["forbidden"]), lambda: os.system("forbidden"),
            lambda: target.write_text("forbidden"), lambda: open(target, "wb"),
            lambda: os.open(target, os.O_WRONLY | os.O_CREAT), lambda: target.mkdir(),
            lambda: os.replace(target, target), lambda: target.unlink())
        audit = {"forbidden_calls": 0}
        with d.read_only(audit):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit["forbidden_calls"], len(calls))

    def test_original_global_references_unknown_history_and_typed_nonadmission(self):
        d.source.bind_authority(self.e)
        for branch in ("fp16", "binary64"):
            self.assertEqual(self.e["result"]["preflight"]["final_reference"]["reference"]["input_"+branch],
                             self.e["result"]["preflight"]["L23_original_reference"]["reference"][branch])
        self.assertEqual(self.result["common_component"], "UNKNOWN")
        self.assertTrue(self.result["stop_nested_bridge_expansion"])
        self.assertEqual(json.dumps(d.FLAGS, sort_keys=True),
                         json.dumps({**d.previous.FLAGS, "stage10_o_projection_input_causal_allocation": False},
                                    sort_keys=True))
        self.assertTrue(all(v is False or (type(v) is int and v == 0) for v in d.FLAGS.values()))
        for text in ("Q24", "strict-FP16-state", "not solely", "888", "binary64"):
            self.assertIn(text, d.BOUNDARY)

    def test_cli_exactly_one_json_and_no_write_replay_options(self):
        stream = io.StringIO()
        with patch.object(d, "check", return_value={"report": "fixture"}), patch("sys.stdout", stream):
            d.main(["--check"])
        result, end = json.JSONDecoder().raw_decode(stream.getvalue())
        self.assertEqual(result, {"report": "fixture"})
        self.assertEqual(stream.getvalue()[end:], "\n")
        for args in ([], ["--check", "--output", "forbidden"], ["--check", "--replay"],
                     ["--check", "--binary64-attention"], ["--check", "--reference"]):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                d.main(args)


if __name__ == "__main__":
    unittest.main()
