"""Independent half-bit/native-nibble and product oracles for retained P0 contrasts."""

from fractions import Fraction
import io
import json
import math
import os
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_stage13_rmsnorm_stage12_attention_score_value_contrast_bridge_v1 as d
from ace3.model.candidates import q24_s16_toward_zero_native_v1 as native


EVIDENCE = None


def half(word):
    exponent, mantissa = (word >> 10) & 31, word & 1023
    if exponent == 31:
        raise ValueError("nonfinite half")
    value = (Fraction(mantissa, 1 << 24) if exponent == 0
             else Fraction(1024+mantissa)*Fraction(2)**(exponent-25))
    return -value if word & 32768 else value


def weight(tensors, output, coordinate):
    shift = 4*(0, 4, 1, 5, 2, 6, 3, 7)[output % 8]
    q = (int(tensors["qweight"][coordinate, output//8]) >> shift) & 15
    z = (int(tensors["qzeros"][coordinate//128, output//8]) >> shift) & 15
    return (q-z)*half(int(tensors["scales"].view("<u2")[coordinate//128, output]))


def effects(products):
    r, p, v, a = products
    return dict(zip((*d.PRODUCTS, *d.EFFECTS),
                    (*products, p-r, a-p, v-r, a-v, a-p-v+r, a-r), strict=True))


def replace_path(value, path, replacement):
    if not path:
        return replacement
    result = value.copy()
    result[path[0]] = (replacement if len(path) == 1
                       else replace_path(value[path[0]], path[1:], replacement))
    return result


class ScoreValueTests(unittest.TestCase):
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
        cls.attention_inputs = cls.parent_inputs[0][0][0]
        cls.e = d.source.evidence(cls.attention_inputs[0])
        cls.oracles, cls.group_oracles, columns = {}, {}, {}
        for key, core in cls.result["attention_score_value_local_accounts"].items():
            coordinate = core["output_coordinate"]
            if coordinate not in columns:
                columns[coordinate] = [weight(cls.attention_inputs[2], coordinate, i)
                                       for i in range(896)]
            actual = cls.e["archives"][core["control"]]
            reference = cls.e["reference_archive"]
            components = []
            for index in range(128):
                kv, dim = divmod(index, 64)
                products = [Fraction() for _ in range(4)]
                for head in range(kv*7, (kv+1)*7):
                    w = columns[coordinate][head*64+dim]
                    pa, pr = half(int(actual["stage09"][head])), half(int(reference["stage09"][head]))
                    va, vr = half(int(actual["stage07"][index])), half(int(reference["stage07"][index]))
                    for i, value in enumerate((pr*vr*w, pa*vr*w, pr*va*w, pa*va*w)):
                        products[i] += value
                components.append(effects(products))
            cls.oracles[key] = components
            order = sorted(range(128), key=lambda i: (-abs(components[i]["signed_contribution"]), i))
            cls.group_oracles[key] = {
                group: {name: sum((components[i][name] for i in indices), Fraction())
                        for name in (*d.PRODUCTS, *d.EFFECTS)}
                for group, indices in zip(d.GROUPS, (order[:8], order[8:], order), strict=True)}

    def test_independent_half_bit_and_native_nibble_oracles(self):
        self.assertEqual(half(1), Fraction(1, 1 << 24))
        self.assertEqual(half(0x8000), 0)
        self.assertEqual(half(0xbc00), -1)
        self.assertEqual(half(0x7bff), 65504)
        with self.assertRaises(ValueError):
            half(0x7c00)
        tensors = {"qweight": np.full((896, 112), 0x76543210, dtype="<i4"),
                   "qzeros": np.full((7, 112), 0x11111111, dtype="<i4"),
                   "scales": np.ones((7, 896), dtype="<f2")}
        for output, nibble in enumerate((0, 4, 1, 5, 2, 6, 3, 7)):
            self.assertEqual(weight(tensors, output, 128), nibble-1)
            self.assertEqual(d.attention.weight_column(tensors, output)[128], nibble-1)

    def test_all_authenticated_scores_probabilities_and_deterministic_ranks(self):
        controls = self.result["attention_score_value_controls"]
        self.assertEqual([row["control"] for row in controls], list(d.parent.CONTROLS))
        reference = self.e["reference_archive"]
        for row in controls:
            actual = self.e["archives"][row["control"]]
            scores = [half(int(x)) for x in actual["stage08"]]
            originals = [half(int(x)) for x in reference["stage08"]]
            order = sorted(range(14), key=lambda h: (-abs(scores[h]-originals[h]), h))
            self.assertEqual([e["query_head"] for e in row["scores_by_absolute_delta"]], order)
            for entry in row["scores_by_absolute_delta"]:
                h = entry["query_head"]
                self.assertEqual(entry, {
                    "query_head": h, "kv_head": h//7, "actual_score": str(scores[h]),
                    "reference_score": str(originals[h]), "score_delta": str(scores[h]-originals[h]),
                    "actual_probability": "1", "reference_probability": "1", "probability_delta": "0"})
                self.assertEqual(half(int(actual["stage09"][h])), 1)
                self.assertEqual(half(int(reference["stage09"][h])), 1)
            self.assertEqual(row["changed_score_head_count"], sum(a != r for a, r in zip(scores, originals)))
            self.assertEqual((row["source_position"], row["source_token_id"],
                              row["changed_probability_head_count"]), (0, 9707, 0))

    def test_score_ties_zero_and_rejected_nonunit_or_inexact_operands(self):
        zero, unit = [Fraction()]*14, [Fraction(1)]*14
        actual = zero.copy()
        actual[13], actual[0], actual[7] = Fraction(-3), Fraction(3), Fraction(1)
        self.assertEqual([x["query_head"] for x in d.score_rows(actual, zero, unit, unit)][:3],
                         [0, 13, 7])
        self.assertEqual([x["query_head"] for x in d.score_rows(zero, zero, unit, unit)],
                         list(range(14)))
        for args in ((zero[:-1], zero, unit, unit), ([0]*14, zero, unit, unit),
                     (zero, zero, zero, unit), (zero, zero, unit, zero)):
            with self.assertRaises(ValueError):
                d.score_rows(*args)

    def test_every_gqa_component_four_products_and_both_swap_orders(self):
        for key, oracle in self.oracles.items():
            core = self.result["attention_score_value_local_accounts"][key]
            order = sorted(range(128), key=lambda i: (-abs(oracle[i]["signed_contribution"]), i))
            components = core["value_component_contrasts"]
            self.assertEqual([entry["value_coordinate"] for entry in components], order)
            for entry in components:
                index = entry["value_coordinate"]
                kv, dim = divmod(index, 64)
                self.assertEqual((entry["kv_head"], entry["head_dimension"], entry["query_heads"]),
                                 (kv, dim, list(range(kv*7, (kv+1)*7))))
                self.assertEqual(entry["contrasts"], {k: str(v) for k, v in oracle[index].items()})
                for name in d.ZERO_EFFECTS:
                    self.assertEqual(entry["contrasts"][name], "0")

    def test_selected_complement_source_products_and_separate_boundary_closure(self):
        for key, oracle in self.oracles.items():
            core = self.result["attention_score_value_local_accounts"][key]
            order = [entry["value_coordinate"] for entry in core["value_component_contrasts"]]
            for group, indices in zip(d.GROUPS, (order[:8], order[8:], order), strict=True):
                expected = {name: str(sum((oracle[i][name] for i in indices), Fraction()))
                            for name in (*d.PRODUCTS, *d.EFFECTS)}
                self.assertEqual(core["value_component_group_contrasts"][group], expected)
            token = core["source_tokens"][0]
            self.assertEqual((token["source_position"], token["source_token_id"]), (0, 9707))
            self.assertEqual(token["contrasts"], core["value_component_group_contrasts"]["full"])
            self.assertEqual(token["largest_absolute_value_components"], core["value_component_contrasts"][:8])
            o = self.retained["attention_o_projection_local_accounts"][key]
            a = self.e["archives"][core["control"]]
            r = self.e["reference_archive"]
            delta = half(int(a["stage11"][core["output_coordinate"]]))-half(int(r["stage11"][core["output_coordinate"]]))
            self.assertEqual(Fraction(token["contrasts"]["signed_contribution"])
                             + Fraction(core["av_boundary_remainder"])
                             + sum((Fraction(core[name]) for name in d.previous.TERMS[1:]), Fraction()),
                             delta)
            for name in d.previous.TERMS[1:]:
                self.assertEqual(core[name], o[name])
            self.assertEqual(core["closure_residual"], "0")

    def test_every_downstream_product_effect_group_and_closure(self):
        for account, (location, entry) in zip(self.result["attention_score_value_accounts"],
                                            d.previous.sources(self.retained), strict=True):
            old = entry["attention_stage11_source_value"]
            self.assertEqual({name: account[name] for name in location}, location)
            self.assertEqual(account["multipliers"], old["multipliers"])
            oracle = self.group_oracles[account["local_account_key"]]
            for field in d.FIELDS:
                multiplier = Fraction(old["multipliers"][field])
                for name in (*d.PRODUCTS, *d.EFFECTS):
                    self.assertEqual(account["contrasts"][name][field],
                                     str(oracle["full"][name]*multiplier))
                    for group in d.GROUPS:
                        self.assertEqual(account["value_component_group_contrasts"][group][name][field],
                                         str(oracle[group][name]*multiplier))
                self.assertEqual(Fraction(account["contrasts"]["signed_contribution"][field])
                                 + sum((Fraction(e[field]) for e in account["boundaries"].values()), Fraction()),
                                 Fraction(old["retained_parent_attention_stage11_delta"][field]))
                self.assertEqual(account["closure_residuals"][field], "0")

    def test_independent_norm_gate_up_down_row_opposite_reference_multipliers(self):
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
                            self.assertEqual(tuple(Fraction(entry["attention_stage11_source_value"]
                                                            ["multipliers"][f]) for f in d.FIELDS),
                                             (1, norm*anchor, norm*anchor*gu, norm*anchor*gu*outer))

    def test_summary_census_and_exact_signed_absolute_totals(self):
        summary = self.result["attention_score_value_summary"]
        accounts = self.result["attention_score_value_accounts"]
        self.assertEqual((summary["control_count"], summary["score_pair_count"],
                          summary["reference_score_count"], summary["account_count"]),
                         (9, 126, 14, 2304))
        self.assertEqual(len(accounts), 2304)
        self.assertEqual(summary["shared_local_account_count"], len(self.oracles))
        self.assertEqual(summary["value_component_count"], 2304*128)
        for name in (*d.PRODUCTS, *d.EFFECTS):
            for field in d.FIELDS:
                values = [Fraction(a["contrasts"][name][field]) for a in accounts]
                signed, absolute = sum(values, Fraction()), sum(map(abs, values), Fraction())
                self.assertEqual(summary["contrast_totals"][name][field], {
                    "signed": str(signed), "absolute": str(absolute),
                    "cancellation_absolute_mass": str(absolute-abs(signed))})

    def test_entire_inherited_report_and_original_global_references_unchanged(self):
        additions = {"attention_score_value_controls", "attention_score_value_local_accounts",
                     "attention_score_value_accounts", "attention_score_value_summary",
                     "attention_score_value_weighting"}
        self.assertEqual({k: v for k, v in self.result.items() if k not in additions}, self.retained)
        for branch in ("fp16", "binary64"):
            self.assertEqual(self.e["result"]["preflight"]["final_reference"]["reference"]["input_"+branch],
                             self.e["result"]["preflight"]["L23_original_reference"]["reference"][branch])
        self.assertEqual(self.result["common_component"], "UNKNOWN")
        self.assertTrue(self.result["stop_nested_bridge_expansion"])

    def test_latest_parent_pins_and_inherited_value_o_proj_source_mutations_rejected(self):
        self.assertTrue(str(d.previous.SOURCE).endswith("attention_o_projection_input_bridge_v1.py"))
        for pin in d.PARENT_PINS:
            self.assertEqual(d.parent.record(d.ROOT / pin["path"]), pin)
            with self.assertRaises(ValueError):
                d.base.read_bound({**pin, "sha256": "0"*64})
        with patch.object(d, "PARENT_PINS", ({**d.PARENT_PINS[0], "sha256": "0"*64},)):
            with self.assertRaises(ValueError):
                d.authenticate_parent()
        with patch.object(d.previous, "report", return_value=self.retained):
            for name, value in (("internal_reference", "binary64"),
                                ("binary64_internal_stages", "RECONSTRUCTED"),
                                ("value_projection_local_accounts", {}),
                                ("attention_stage11_local_accounts", {}),
                                ("attention_o_projection_local_accounts", {}),
                                ("attention_o_projection_input_accounts", [])):
                with self.assertRaises(ValueError):
                    d.bind_parent(self.parent_inputs, {**self.retained, name: value})

    def test_actual_original_score_probability_archive_and_lineage_mutations_rejected(self):
        control = d.parent.CONTROLS[0]
        for original in (False, True):
            prefix = ["reference_archive"] if original else ["archives", control]
            archive = self.e["reference_archive"] if original else self.e["archives"][control]
            for name in ("stage08", "stage09"):
                changed = archive[name].copy()
                changed.flat[0] ^= 1
                mutated = replace_path(self.e, [*prefix, name], changed)
                with patch.object(d.source, "evidence", return_value=mutated):
                    with self.assertRaises(ValueError):
                        d.bind_scores(self.attention_inputs[0], self.attention_inputs[2])
        for array in (np.zeros(13, dtype="<u2"), np.zeros(14, dtype="<f2"),
                      np.full(14, 0x7c00, dtype="<u2")):
            with self.assertRaises(ValueError):
                d.attention.words(array, (14,))

    def test_state_kv_stage00_value_and_o_projection_lineage_mutations_rejected(self):
        control = d.parent.CONTROLS[0]
        norm = self.attention_inputs[0][0][1]
        for original in (False, True):
            prefix = ["reference_archive"] if original else ["archives", control]
            archive = self.e["reference_archive"] if original else self.e["archives"][control]
            names = ("stage00", "stage03", "stage07", "stage10", "stage11")
            if not original:
                names += ("input_i", "input_z", "scratch_i", "scratch_z", "output_cache_k", "output_cache_v")
            for name in names:
                changed = archive[name].copy()
                changed.flat[0] ^= 1
                with self.assertRaises(ValueError):
                    d.source.rmsnorm.bind_inputs(replace_path(self.e, [*prefix, name], changed), norm)
        for kind in ("k", "v"):
            with self.assertRaises(ValueError):
                d.attention.attention_operands({
                    **self.e["archives"][control], "input_cache_"+kind: np.zeros((1, 128), dtype="<u2")},
                    actual=True)

    def test_source_token_position_authority_threshold_and_canonical_mutations_rejected(self):
        binding = ["result", "preflight", "L23_original_reference", "reference"]
        for path, value in (
            (["result", "source_token_id"], 9708), (["result", "source_position"], 1),
            ([*binding, "prior_kv"], "nonempty"), ([*binding, "fp16", "sha256"], "0"*64),
            ([*binding, "canonical", d.attention.PROJECTION+"qweight", "sha256"], "0"*64),
            (["result", "preflight", "thresholds"], {}),
            (["result", "controls", 0, "parent", "terminal_archive", "sha256"], "0"*64)):
            with self.assertRaises(ValueError):
                d.source.bind_authority(replace_path(self.e, path, value))
        for name, tensor in self.attention_inputs[2].items():
            changed = tensor.copy()
            changed.view("u1").flat[0] ^= 1
            with self.assertRaises(ValueError):
                d.bind_scores(self.attention_inputs[0], {**self.attention_inputs[2], name: changed})

    def test_local_product_and_downstream_mutations_rejected(self):
        location, entry = next(d.previous.sources(self.retained))
        inherited = entry["attention_stage11_source_value"]
        o = self.retained["attention_o_projection_input_accounts"][0]
        core = self.result["attention_score_value_local_accounts"][inherited["local_account_key"]]
        field = d.FIELDS[-1]
        for name in (*d.PRODUCTS, *d.ZERO_EFFECTS, "value_at_reference_probability"):
            mutated = replace_path(core, ["source_tokens", 0, "contrasts", name], "123")
            with self.assertRaises(ValueError):
                d.weighted_account(mutated, inherited, o, location)
        for path in (["multipliers", field], ["terms", 1, field],
                     ["retained_parent_attention_stage11_delta", field], ["closure_residuals", field]):
            with self.assertRaises(ValueError):
                d.weighted_account(core, replace_path(inherited, path, "123"), o, location)
        for path in (["multipliers", field], ["terms", 0, field], ["terms", 1, field],
                     ["closure_residuals", field], ["local_account_key"]):
            with self.assertRaises(ValueError):
                d.weighted_account(core, inherited, replace_path(o, path, "123"), location)

    def test_forbidden_predecessor_checks_operator_replay_reconstruction_and_writes(self):
        target = d.parent.OUTPUT / "forbidden-score-value-contrast-bridge"
        calls = (
            lambda: d.previous.check(), lambda: d.previous.run_tests(None, None),
            lambda: d.previous.collect(None), lambda: d.source.check(),
            lambda: d.contrast.check(), lambda: d.contrast.measure(),
            lambda: d.contrast.focused_tests(None), lambda: d.contrast.report(None, None, None),
            lambda: d.attention.measure(), lambda: d.parent.execute("forbidden"),
            lambda: d.parent.rmsnorm(None, None), lambda: d.parent.logits(None, None),
            lambda: native._stages(None, None, None, None),
            lambda: d.bridge.residual.local.local_reference(None),
            lambda: d.bridge.margin.contributions.dyadic_dot(None, None),
            lambda: math.exp(0), lambda: np.exp(0), lambda: subprocess.run(["forbidden"]),
            lambda: os.system("forbidden"), lambda: target.write_text("forbidden"),
            lambda: open(target, "wb"), lambda: os.open(target, os.O_WRONLY | os.O_CREAT),
            lambda: target.mkdir(), lambda: os.replace(target, target), lambda: target.unlink())
        audit = {"forbidden_calls": 0}
        with d.read_only(audit):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit["forbidden_calls"], len(calls))

    def test_typed_nonadmission_cli_single_json_and_no_output_or_replay_options(self):
        self.assertEqual(d.FLAGS, {**d.previous.FLAGS, "score_replay": 0, "softmax_replay": 0,
                                  "score_value_causal_allocation": False})
        self.assertTrue(all(v is False or (type(v) is int and v == 0) for v in d.FLAGS.values()))
        for text in ("Q24", "strict-FP16-state", "binary64", "not Q/K", "FP16 operator/KV"):
            self.assertIn(text, d.BOUNDARY)
        stream = io.StringIO()
        with patch.object(d, "check", return_value={"report": "fixture"}), patch("sys.stdout", stream):
            d.main(["--check"])
        value, end = json.JSONDecoder().raw_decode(stream.getvalue())
        self.assertEqual(value, {"report": "fixture"})
        self.assertEqual(stream.getvalue()[end:], "\n")
        for args in ([], ["--check", "--output", "forbidden"], ["--check", "--replay"],
                     ["--check", "--binary64-attention"], ["--check", "--reference"]):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                d.main(args)


if __name__ == "__main__":
    unittest.main()
