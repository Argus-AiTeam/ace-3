"""Independent half-bit/GEMM-nibble oracles for the retained P0 attention bridge."""

from copy import deepcopy
from fractions import Fraction
import io
import json
import math
import os
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_stage13_rmsnorm_stage12_attention_source_value_bridge_v1 as d
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


class AttentionSourceValueTests(unittest.TestCase):
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
        cls.parent_inputs, cls.retained, cls.tensors = cls.inputs
        cls.e = d.evidence(cls.parent_inputs)
        cls.norm_inputs = cls.parent_inputs[0]
        cls.projection_inputs = cls.norm_inputs[0]
        cls.oracles = {}
        columns = {}
        for key, local in cls.result["attention_stage11_local_accounts"].items():
            control, output = local["control"], local["output_coordinate"]
            if output not in columns:
                columns[output] = [weight(cls.tensors, output, j) for j in range(896)]
            a, r = cls.e["archives"][control], cls.e["reference_archive"]
            pa, pr = ([half(int(w)) for w in x["stage09"]] for x in (a, r))
            va, vr = ([half(int(w)) for w in x["stage07"]] for x in (a, r))
            components = []
            for v in range(128):
                kv, dim = divmod(v, 64)
                parts = [Fraction(), Fraction(), Fraction()]
                for head in range(7*kv, 7*kv+7):
                    w = columns[output][head*64+dim]
                    dp, dv = pa[head]-pr[head], va[v]-vr[v]
                    parts[0] += dp*vr[v]*w
                    parts[1] += pr[head]*dv*w
                    parts[2] += dp*dv*w
                components.append(parts)
            av_delta = sum(((half(int(a["stage10"][j]))-half(int(r["stage10"][j])))
                            * columns[output][j] for j in range(896)), Fraction())
            exact = sum((sum(parts, Fraction()) for parts in components), Fraction())
            delta = half(int(a["stage11"][output]))-half(int(r["stage11"][output]))
            order = sorted(range(128), key=lambda v: (-abs(sum(components[v])), v))
            cls.oracles[key] = (components, order, av_delta-exact, delta-av_delta, delta)

    def sources(self, result=None):
        for row in (self.result if result is None else result)["rows"]:
            for hotspot in row["hotspots"]:
                for selected in hotspot["selected_stage16_coordinates"]:
                    for kind in ("gate_proj", "up_proj"):
                        for source in selected[kind]["ranked_stage13_rmsnorm_sources"]:
                            yield row, hotspot, selected, kind, source

    def first(self):
        for _, _, _, _, source in self.sources():
            account = source["attention_stage11_source_value"]
            if all(Fraction(account["retained_parent_attention_stage11_delta"][field])
                   for field in d.FIELDS):
                return source, self.result["attention_stage11_local_accounts"][account["local_account_key"]]
        raise AssertionError("no retained attention delta with nonzero downstream weighting")

    def test_all_local_accounts_independent_bits_gqa_and_native_nibbles(self):
        for key, local in self.result["attention_stage11_local_accounts"].items():
            components, order, av, projection, delta = self.oracles[key]
            self.assertEqual([e["value_coordinate"] for e in local["full_value_component_ranking"]], order)
            for entry in local["full_value_component_ranking"]:
                v = entry["value_coordinate"]
                self.assertEqual([Fraction(entry[p]) for p in d.attention.PARTS], components[v])
                self.assertEqual(Fraction(entry["signed_contribution"]), sum(components[v]))
                self.assertEqual((entry["source_position"], entry["source_token_id"]), (0, 9707))
                self.assertEqual(entry["query_heads"], list(range((v//64)*7, (v//64+1)*7)))
            self.assertEqual(Fraction(local["av_boundary_remainder"]), av)
            self.assertEqual(Fraction(local["o_projection_boundary_remainder"]), projection)
            self.assertEqual(Fraction(local["retained_o_projection_delta"]), delta)
            self.assertEqual(av, 0)
            self.assertTrue(all(parts[0] == parts[2] == 0 for parts in components))

    def test_every_parent_attention_term_and_all_downstream_multipliers(self):
        count = 0
        for row, hotspot, selected, kind, source in self.sources():
            account = source["attention_stage11_source_value"]
            components, _, av, projection, delta = self.oracles[account["local_account_key"]]
            i = source["retained_ranked_input"]["coordinate"]
            output = selected["selected_down_input"]["coordinate"]
            norm = half(int(self.norm_inputs[1].view("<u2")[i]))
            anchor = Fraction(self.norm_inputs[2]["original_input_fp16"]["inverse_norm_anchor"])
            gate_up = weight(self.projection_inputs[1][kind], output, i)
            down = weight(self.projection_inputs[0][4], hotspot["output_coordinate"], output)
            opposite = half(int(self.e["reference_archive"]["stage15" if kind == "gate_proj" else "stage14"][output]))
            row_factor = Fraction(hotspot["retained_weighted_row_factor"])
            multipliers = (Fraction(1), norm*anchor, norm*anchor*gate_up,
                           norm*anchor*gate_up*down*opposite*row_factor)
            values = [sum((p[k] for p in components), Fraction()) for k in range(3)] + [av, projection]
            target = source["direct_stage12_residual_source"]["terms"][1]
            self.assertEqual(account["retained_parent_attention_stage11_delta"], target)
            for field, multiplier in zip(d.FIELDS, multipliers, strict=True):
                self.assertEqual(Fraction(account["multipliers"][field]), multiplier)
                self.assertEqual([Fraction(t[field]) for t in account["terms"]],
                                 [v*multiplier for v in values])
                self.assertEqual(Fraction(target[field]), delta*multiplier)
                self.assertEqual(account["totals"][field], mass(v*multiplier for v in values))
                self.assertEqual(account["closure_residuals"][field], "0")
            count += 1
        self.assertEqual(count, 2304)

    def test_selected_unselected_full_component_masses(self):
        for key, local in self.result["attention_stage11_local_accounts"].items():
            components, order, _, _, _ = self.oracles[key]
            self.assertEqual(local["largest_absolute_value_components"], local["full_value_component_ranking"][:8])
            for group, coordinates in zip(d.GROUPS, (order[:8], order[8:], order), strict=True):
                for part, index in zip(d.PARTS, (0, 1, 2, None), strict=True):
                    values = [sum(components[v]) if index is None else components[v][index] for v in coordinates]
                    self.assertEqual(local["value_component_totals"][group][part], mass(values))
        for _, _, _, _, source in self.sources():
            a = source["attention_stage11_source_value"]
            local = self.result["attention_stage11_local_accounts"][a["local_account_key"]]
            for field in d.FIELDS:
                multiplier = Fraction(a["multipliers"][field])
                for group in d.GROUPS:
                    for part in d.PARTS:
                        unit = local["value_component_totals"][group][part]
                        observed = a["value_component_totals"][group][field][part]
                        signed = Fraction(unit["signed"])*multiplier
                        absolute = Fraction(unit["absolute"])*abs(multiplier)
                        self.assertEqual(observed, {"signed": str(signed), "absolute": str(absolute),
                                                   "cancellation_absolute_mass": str(absolute-abs(signed))})

    def test_projection_row_and_global_summaries(self):
        def check(observed, accounts):
            self.assertEqual(observed["account_count"], len(accounts))
            self.assertEqual(observed["value_component_count"], len(accounts)*128)
            for field in d.FIELDS:
                self.assertEqual(observed["totals"][field],
                                 mass(Fraction(t[field]) for a in accounts for t in a["terms"]))
                for index, name in enumerate(d.TERMS):
                    self.assertEqual(observed["term_totals"][name][field],
                                     mass(Fraction(a["terms"][index][field]) for a in accounts))
                for group in d.GROUPS:
                    for part in d.PARTS:
                        entries = [a["value_component_totals"][group][field][part] for a in accounts]
                        signed = sum((Fraction(e["signed"]) for e in entries), Fraction())
                        absolute = sum((Fraction(e["absolute"]) for e in entries), Fraction())
                        self.assertEqual(observed["value_component_totals"][group][field][part],
                                         {"signed": str(signed), "absolute": str(absolute),
                                          "cancellation_absolute_mass": str(absolute-abs(signed))})
                values = [a["value_component_totals"]["full"][field]["signed_contribution"] for a in accounts]
                signed = sum((Fraction(v["signed"]) for v in values), Fraction())
                absolute = sum((Fraction(v["absolute"]) for v in values), Fraction())
                boundaries = [Fraction(t[field]) for a in accounts for t in a["terms"][3:]]
                signed += sum(boundaries, Fraction())
                absolute += sum(map(abs, boundaries), Fraction())
                self.assertEqual(observed["component_and_boundary_totals"][field],
                                 {"signed": str(signed), "absolute": str(absolute),
                                  "cancellation_absolute_mass": str(absolute-abs(signed))})
        all_accounts = []
        for row in self.result["rows"]:
            row_accounts = []
            for hotspot in row["hotspots"]:
                for selected in hotspot["selected_stage16_coordinates"]:
                    for kind in ("gate_proj", "up_proj"):
                        p = selected[kind]
                        accounts = [s["attention_stage11_source_value"] for s in p["ranked_stage13_rmsnorm_sources"]]
                        check(p["attention_stage11_source_value_summary"], accounts)
                        row_accounts.extend(accounts)
            check(row["attention_stage11_source_value_summary"], row_accounts)
            all_accounts.extend(row_accounts)
        check(self.result["attention_stage11_source_value_summary"], all_accounts)

    def test_coordinate741_controls_branches_and_no_internal_binary64(self):
        self.assertEqual([(r["control"], r["branch"]) for r in self.result["rows"]],
                         [(c, b) for c in d.parent.CONTROLS for b in ("fp16", "binary64")])
        for row in self.result["rows"]:
            self.assertEqual(row["attention_stage11_source_value_summary"]["account_count"], 128)
            if row["control"] == "frozen_inherited":
                self.assertEqual(row["hotspots"][0]["output_coordinate"], 741)
                self.assertEqual(row["retained_maximum_coordinate_ties"], [741])
        self.assertEqual(self.result["internal_reference"], "original_input_L23_fp16")
        self.assertEqual(self.result["binary64_internal_stages"], "NOT_RETAINED_NO_RECONSTRUCTION")

    def test_lossless_parent_and_unchanged_unknown_gates(self):
        stripped = deepcopy(self.result)
        for key in ("attention_stage11_local_accounts", "attention_stage11_source_value_summary",
                    "attention_stage11_source_value_identity", "attention_stage11_source_value_weighting"):
            stripped.pop(key)
        for row in stripped["rows"]:
            row.pop("attention_stage11_source_value_summary")
            for hotspot in row["hotspots"]:
                for selected in hotspot["selected_stage16_coordinates"]:
                    for kind in ("gate_proj", "up_proj"):
                        selected[kind].pop("attention_stage11_source_value_summary")
                        for source in selected[kind]["ranked_stage13_rmsnorm_sources"]:
                            source.pop("attention_stage11_source_value")
        self.assertEqual(stripped, self.retained)
        self.assertEqual(self.result["common_component"], "UNKNOWN")
        self.assertTrue(self.result["stop_nested_bridge_expansion"])

    def test_tampered_parent_selection_rejected(self):
        path = ["rows", 0, "hotspots", 0, "selected_stage16_coordinates", 0,
                "gate_proj", "ranked_stage13_rmsnorm_sources"]
        old = self.retained
        for key in path:
            old = old[key]
        changed = replace_path(self.retained, path, list(reversed(old)))
        with self.assertRaises(ValueError):
            d.report(self.parent_inputs, changed, self.tensors)

    def test_tampered_parent_targets_multipliers_and_coordinate(self):
        source, local = self.first()
        root = ["direct_stage12_residual_source"]
        for path in (
            ["retained_ranked_input", "coordinate"],
            *[[*root, key] for key in ("norm_weight", "original_input_fp16_scalar_anchor",
                                      "gate_up_weight", "coordinate_linear_operand_factor")],
            *[[*root, "terms", 1, field] for field in d.FIELDS],
        ):
            with self.assertRaises((ValueError, ZeroDivisionError)):
                d.account(local, replace_path(source, path, -1 if path[-1] == "coordinate" else "0"))

    def test_tampered_attention_arrays_actual_original_bits_shape_dtype(self):
        control = d.parent.CONTROLS[0]
        for reference in (False, True):
            prefix = ["reference_archive"] if reference else ["archives", control]
            archive = self.e["reference_archive"] if reference else self.e["archives"][control]
            for stage in ("stage07", "stage09", "stage10", "stage11"):
                changed = archive[stage].copy()
                changed.flat[0] ^= 1
                e = replace_path(self.e, [*prefix, stage], changed)
                with self.assertRaises(ValueError):
                    d.rmsnorm.bind_inputs(e, self.norm_inputs[1])
        for array in (archive["stage09"][:-1], archive["stage09"].astype("<i4")):
            with self.assertRaises(ValueError):
                d.attention.attention_operands({**archive, "stage09": array}, actual=False)
        load_attention = d.attention.load_attention
        for reference in (False, True):
            def changed_binding(*args):
                actual, original, tensors, pins = load_attention(*args)
                operands = original if reference else actual[control]
                operands[3][0] += Fraction(1, 1 << 24)
                return actual, original, tensors, pins

            message = ("original-input FP16" if reference else "actual") + " attention operand binding changed"
            with patch.object(d.attention, "load_attention", side_effect=changed_binding):
                with self.assertRaisesRegex(ValueError, message):
                    d.bind_attention(self.parent_inputs, self.tensors)

    def test_tampered_empty_kv_lineage_q24_and_signed_zero(self):
        control = d.parent.CONTROLS[0]
        archive = self.e["archives"][control]
        for stage in ("stage03", "stage06", "input_i", "input_z", "scratch_i", "scratch_z",
                      "output_cache_k", "output_cache_v"):
            changed = archive[stage].copy()
            changed.flat[0] ^= 1
            with self.assertRaises(ValueError):
                d.rmsnorm.bind_inputs(replace_path(self.e, ["archives", control, stage], changed),
                                     self.norm_inputs[1])
        for kind in ("k", "v"):
            with self.assertRaises(ValueError):
                d.attention.attention_operands(
                    {**archive, "input_cache_"+kind: np.zeros((1, 128), dtype="<u2")}, actual=True)

    def test_tampered_canonical_o_projection_tensors(self):
        for name in self.tensors:
            changed = self.tensors[name].copy()
            changed.view("u1").flat[0] ^= 1
            with self.assertRaises(ValueError):
                d.bind_attention(self.parent_inputs, {**self.tensors, name: changed})
        with self.assertRaises(ValueError):
            d.bind_attention(self.parent_inputs, {**self.tensors, "binary64": self.tensors["scales"]})

    def test_tampered_original_source_tensor_kv_and_threshold_authority(self):
        binding = ["result", "preflight", "L23_original_reference", "reference"]
        for path, value in (
            ([*binding, "prior_kv"], "nonempty"),
            ([*binding, "fp16", "sha256"], "0"*64),
            ([*binding, "canonical", d.attention.PROJECTION+"qweight", "sha256"], "0"*64),
            (["result", "preflight", "thresholds"], {}),
            (["result", "controls", 0, "parent", "terminal_archive", "sha256"], "0"*64),
            (["result", "preflight", "final_reference", "reference", "input_binary64", "sha256"], "0"*64),
        ):
            with self.assertRaises(ValueError):
                d.bind_authority(replace_path(self.e, path, value))
        source, local = self.first()
        changed = deepcopy(local)
        changed["full_value_component_ranking"][0]["source_token_id"] = 9708
        with self.assertRaises((ValueError, d.complement.jsonschema.ValidationError)):
            d.complement.split_unselected({k: changed[k] for k in d.attention.ACCOUNT_SCHEMA["properties"]})
        with self.assertRaises(ValueError):
            d.base.read_bound({**d.parent.record(d.SOURCE), "sha256": "0"*64})

    def test_zero_negative_weights_and_cancellation(self):
        unit = mass([Fraction(3), Fraction(-3), Fraction(1)])
        for factor in (Fraction(0), Fraction(-2), Fraction(1, 3)):
            self.assertEqual(d.scaled_mass(unit, factor),
                             mass(v*factor for v in (Fraction(3), Fraction(-3), Fraction(1))))
        self.assertEqual(d.merged_mass([mass([Fraction(3)]), mass([Fraction(-3)])]),
                         mass([Fraction(3), Fraction(-3)]))

    def test_forbidden_replay_write_and_internal_attention_reconstruction(self):
        target = d.parent.OUTPUT / "forbidden-stage11-attention-source-value"
        torch = d.parent.preflight.parent.producer.legacy.torch
        calls = (
            lambda: d.previous.check(), lambda: d.previous.collect(None),
            lambda: d.previous.run_tests(None, None), lambda: d.attention.measure(),
            lambda: d.attention.report(None, None, None, None, None),
            lambda: d.complement.measure(), lambda: d.parent.rmsnorm(None, None),
            lambda: d.parent.logits(None, None), lambda: d.parent.execute("forbidden"),
            lambda: native._stages(None, None, None, None),
            lambda: d.bridge.residual.local.local_reference(None),
            lambda: d.bridge.margin.contributions.dyadic_dot(None, None),
            lambda: d.rmsnorm.gate_up.measure(), lambda: d.rmsnorm.hidden.measure(),
            lambda: math.exp(0), lambda: np.exp(0), lambda: torch.sigmoid(None),
            lambda: torch.nn.functional.silu(None), lambda: torch.cuda._lazy_init(),
            lambda: subprocess.run(["forbidden"]), lambda: os.system("forbidden"),
            lambda: target.write_text("forbidden"), lambda: open(target, "wb"),
            lambda: os.open(target, os.O_WRONLY | os.O_CREAT), lambda: target.mkdir(),
            lambda: os.replace(target, target), lambda: target.unlink(),
        )
        audit = {"forbidden_calls": 0}
        with d.read_only(audit):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit["forbidden_calls"], len(calls))
        with self.assertRaises(ValueError):
            d.attention.words(np.ones((14,), dtype="<f8"), (14,))

    def test_original_references_history_and_claim_boundary(self):
        d.bind_authority(self.e)
        for branch in ("fp16", "binary64"):
            self.assertEqual(self.e["result"]["preflight"]["final_reference"]["reference"]["input_"+branch],
                             self.e["result"]["preflight"]["L23_original_reference"]["reference"][branch])
        self.assertTrue(all(not v for v in d.FLAGS.values()))
        for text in ("UNKNOWN", "Q24", "original-input FP16", "strict-FP16-state",
                     "not solely", "no reselection"):
            self.assertIn(text, d.BOUNDARY)

    def test_cli_one_json_document_and_no_extra_options(self):
        stream = io.StringIO()
        with patch.object(d, "check", return_value={"report": "fixture"}), patch("sys.stdout", stream):
            d.main(["--check"])
        value, end = json.JSONDecoder().raw_decode(stream.getvalue())
        self.assertEqual(value, {"report": "fixture"})
        self.assertEqual(stream.getvalue()[end:], "\n")
        for args in ([], ["--check", "--output", "forbidden"], ["--check", "--binary64-attention"],
                     ["--check", "--replay"], ["--check", "--reference"]):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                d.main(args)


if __name__ == "__main__":
    unittest.main()
