"""Independent FP16-bit and packed-GEMM oracles for the selected V input bridge."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_stage13_rmsnorm_stage12_attention_value_projection_input_bridge_v1 as d
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


class ValueProjectionInputTests(unittest.TestCase):
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
        cls.parent_inputs, cls.retained, cls.operands = cls.inputs
        cls.e = d.previous.evidence(cls.parent_inputs[0])
        cls.oracles = {}
        columns = {}
        for key, core in cls.result["value_projection_local_accounts"].items():
            v, control = core["output_coordinate"], core["control"]
            if v not in columns:
                columns[v] = [weight(cls.operands[2], v, i) for i in range(896)]
            a, r = cls.e["archives"][control], cls.e["reference_archive"]
            actual, reference = ([half(int(w)) for w in x["stage00"]] for x in (a, r))
            terms = [(x-y)*w for x, y, w in zip(actual, reference, columns[v], strict=True)]
            bias = half(int(cls.operands[2]["bias"].view("<u2")[v]))
            asum = sum((x*w for x, w in zip(actual, columns[v], strict=True)), bias)
            rsum = sum((x*w for x, w in zip(reference, columns[v], strict=True)), bias)
            av, rv = half(int(a["stage03"][v])), half(int(r["stage03"][v]))
            cls.oracles[key] = (terms, sorted(range(896), key=lambda i: (-abs(terms[i]), i)),
                                asum, rsum, av-asum, rsum-rv, av-rv)

    def sources(self, result=None):
        for row in (self.result if result is None else result)["rows"]:
            for hotspot in row["hotspots"]:
                for selected in hotspot["selected_stage16_coordinates"]:
                    for kind in ("gate_proj", "up_proj"):
                        for source in selected[kind]["ranked_stage13_rmsnorm_sources"]:
                            yield row, hotspot, selected, kind, source

    def accounts(self):
        for *_, source in self.sources():
            yield from source["attention_value_projection_input"]["selected_value_components"]

    def test_all_exact_input_sums_and_separate_projection_boundaries(self):
        for key, core in self.result["value_projection_local_accounts"].items():
            terms, _, asum, rsum, ab, rb, delta = self.oracles[key]
            for name, expected in (
                ("actual_exact_input_weight_sum_with_bias", asum),
                ("original_exact_input_weight_sum_with_bias", rsum),
                ("exact_stage00_input_delta", sum(terms)),
                ("actual_projection_boundary", ab),
                ("negative_original_input_fp16_projection_boundary", rb),
                ("projection_boundary_remainder", ab+rb), ("retained_projection_delta", delta),
            ):
                self.assertEqual(Fraction(core[name]), expected)
            self.assertEqual(sum(terms)+ab+rb, delta)
            self.assertEqual(core["closure_residual"], "0")

    def test_all_ranked_coordinates_and_contiguous_g128_groups(self):
        for key, core in self.result["value_projection_local_accounts"].items():
            terms, order, *_ = self.oracles[key]
            self.assertEqual([r["coordinate"] for r in core["largest_absolute_input_coordinates"]], order[:8])
            for row in core["largest_absolute_input_coordinates"]:
                i = row["coordinate"]
                self.assertEqual(Fraction(row["signed_contribution"]), terms[i])
                self.assertEqual(Fraction(row["weight"]),
                                 weight(self.operands[2], core["output_coordinate"], i))
                self.assertEqual(Fraction(row["input_delta"]),
                                 Fraction(row["actual_input"])-Fraction(row["original_input_fp16_input"]))
            groups = core["groups_by_absolute_signed_sum"]
            expected = [mass(terms[g*128:(g+1)*128]) for g in range(7)]
            self.assertEqual([g["input_group"] for g in groups],
                             sorted(range(7), key=lambda g: (-abs(Fraction(expected[g]["signed"])), g)))
            for group in groups:
                g = group["input_group"]
                self.assertEqual((group["start_coordinate"], group["end_coordinate_exclusive"],
                                  group["coordinate_count"]), (g*128, (g+1)*128, 128))
                self.assertEqual({k: group[k] for k in expected[g]}, expected[g])

    def test_all_selected_complement_and_full_coordinate_masses(self):
        for key, core in self.result["value_projection_local_accounts"].items():
            terms, order, *_ = self.oracles[key]
            for group, indices in zip(d.GROUPS, (order[:8], order[8:], order), strict=True):
                self.assertEqual(core["input_coordinate_totals"][group], mass(terms[i] for i in indices))
        for a in self.accounts():
            core = self.result["value_projection_local_accounts"][a["projection_account_key"]]
            for group in d.GROUPS:
                unit = core["input_coordinate_totals"][group]
                for field in d.FIELDS:
                    factor = Fraction(a["input_coordinate_multipliers"][field])
                    signed, absolute = Fraction(unit["signed"])*factor, Fraction(unit["absolute"])*abs(factor)
                    self.assertEqual(a["input_coordinate_totals"][group][field],
                                     {"signed": str(signed), "absolute": str(absolute),
                                      "cancellation_absolute_mass": str(absolute-abs(signed))})

    def test_every_selected_parent_term_and_independent_downstream_weights(self):
        count = 0
        residual_inputs = self.parent_inputs[0]
        norm_inputs = residual_inputs[0]
        projection_inputs = norm_inputs[0]
        columns = {}
        for row, hotspot, selected, kind, source in self.sources():
            i = source["retained_ranked_input"]["coordinate"]
            j = selected["selected_down_input"]["coordinate"]
            norm = half(int(norm_inputs[1].view("<u2")[i]))
            anchor = Fraction(norm_inputs[2]["original_input_fp16"]["inverse_norm_anchor"])
            gu = weight(projection_inputs[1][kind], j, i)
            down = weight(projection_inputs[0][4], hotspot["output_coordinate"], j)
            opposite = half(int(self.e["reference_archive"]["stage15" if kind == "gate_proj" else "stage14"][j]))
            factor = down*opposite*Fraction(hotspot["retained_weighted_row_factor"])
            downstream = (Fraction(1), norm*anchor, norm*anchor*gu, norm*anchor*gu*factor)
            if i not in columns:
                columns[i] = [weight(self.parent_inputs[2], i, c) for c in range(896)]
            local = self.retained["attention_stage11_local_accounts"][
                source["attention_stage11_source_value"]["local_account_key"]]
            accounts = source["attention_value_projection_input"]["selected_value_components"]
            self.assertEqual([a["retained_parent_value_component"] for a in accounts],
                             local["largest_absolute_value_components"])
            for rank, a in enumerate(accounts, 1):
                v = a["retained_parent_value_component"]["value_coordinate"]
                kv, dim = divmod(v, 64)
                gqa = sum((columns[i][h*64+dim] for h in range(kv*7, (kv+1)*7)), Fraction())
                _, _, asum, rsum, ab, rb, delta = self.oracles[a["projection_account_key"]]
                self.assertEqual(a["value_hotspot_rank"], rank)
                self.assertEqual(Fraction(a["value_component_factor"]), gqa)
                self.assertEqual([t["term"] for t in a["terms"]], list(d.TERMS))
                for field, m in zip(d.FIELDS, downstream, strict=True):
                    values = [x*gqa*m for x in (asum-rsum, ab, rb)]
                    self.assertEqual(Fraction(a["input_coordinate_multipliers"][field]), gqa*m)
                    self.assertEqual([Fraction(t[field]) for t in a["terms"]], values)
                    self.assertEqual(Fraction(a["retained_parent_contributions"][field]), delta*gqa*m)
                    self.assertEqual(sum(values), delta*gqa*m)
                    self.assertEqual(a["totals"][field], mass(values))
                    self.assertEqual(a["closure_residuals"][field], "0")
                count += 1
        self.assertEqual(count, 18432)

    def test_projection_row_and_global_summary_totals(self):
        def check(observed, accounts):
            self.assertEqual(observed["selected_value_component_count"], len(accounts))
            self.assertEqual(observed["input_coordinate_term_count"], len(accounts)*896)
            self.assertEqual(observed["input_group_count"], len(accounts)*7)
            for field in d.FIELDS:
                self.assertEqual(observed["totals"][field],
                                 mass(Fraction(t[field]) for a in accounts for t in a["terms"]))
                for i, term in enumerate(d.TERMS):
                    self.assertEqual(observed["term_totals"][term][field],
                                     mass(Fraction(a["terms"][i][field]) for a in accounts))
                for group in d.GROUPS:
                    entries = [a["input_coordinate_totals"][group][field] for a in accounts]
                    signed = sum((Fraction(e["signed"]) for e in entries), Fraction())
                    absolute = sum((Fraction(e["absolute"]) for e in entries), Fraction())
                    self.assertEqual(observed["input_coordinate_totals"][group][field],
                                     {"signed": str(signed), "absolute": str(absolute),
                                      "cancellation_absolute_mass": str(absolute-abs(signed))})
        all_accounts = []
        for row in self.result["rows"]:
            row_accounts = []
            for h in row["hotspots"]:
                for s in h["selected_stage16_coordinates"]:
                    for kind in ("gate_proj", "up_proj"):
                        p = s[kind]
                        accounts = [a for source in p["ranked_stage13_rmsnorm_sources"]
                                    for a in source["attention_value_projection_input"]["selected_value_components"]]
                        check(p["attention_value_projection_input_summary"], accounts)
                        row_accounts.extend(accounts)
            check(row["attention_value_projection_input_summary"], row_accounts)
            all_accounts.extend(row_accounts)
        check(self.result["attention_value_projection_input_summary"], all_accounts)

    def test_lossless_parent_selections_other_terms_and_unknown_gates(self):
        stripped = deepcopy(self.result)
        for key in ("value_projection_local_accounts", "attention_value_projection_input_summary",
                    "attention_value_projection_input_identity", "attention_value_projection_input_weighting"):
            stripped.pop(key)
        for row in stripped["rows"]:
            row.pop("attention_value_projection_input_summary")
            for h in row["hotspots"]:
                for s in h["selected_stage16_coordinates"]:
                    for kind in ("gate_proj", "up_proj"):
                        s[kind].pop("attention_value_projection_input_summary")
                        for source in s[kind]["ranked_stage13_rmsnorm_sources"]:
                            source.pop("attention_value_projection_input")
        self.assertEqual(stripped, self.retained)
        self.assertEqual(self.result["common_component"], "UNKNOWN")
        self.assertTrue(self.result["stop_nested_bridge_expansion"])

    def test_control_branch_representatives_and_original_reference(self):
        self.assertEqual([(r["control"], r["branch"]) for r in self.result["rows"]],
                         [(c, b) for c in d.parent.CONTROLS for b in ("fp16", "binary64")])
        for row in self.result["rows"]:
            self.assertEqual(row["attention_value_projection_input_summary"]["selected_value_component_count"], 1024)
            if row["control"] == "frozen_inherited":
                self.assertEqual(row["hotspots"][0]["output_coordinate"], 741)
        self.assertEqual(self.result["internal_reference"], "original_input_L23_fp16")
        self.assertEqual(self.result["binary64_internal_stages"], "NOT_RETAINED_NO_RECONSTRUCTION")

    def test_tampered_parent_selection_rejected(self):
        key = next(iter(self.retained["attention_stage11_local_accounts"]))
        selected = self.retained["attention_stage11_local_accounts"][key]["largest_absolute_value_components"]
        path = ["attention_stage11_local_accounts", key, "largest_absolute_value_components"]
        with self.assertRaises(ValueError):
            d.previous.validate_report(self.parent_inputs, replace_path(self.retained, path, selected[::-1]))

    def test_tampered_value_component_source_and_target(self):
        a = next(self.accounts())
        core = self.result["value_projection_local_accounts"][a["projection_account_key"]]
        source = next(self.sources())[-1]
        downstream = source["attention_value_projection_input"]["downstream_multipliers"]
        component = a["retained_parent_value_component"]
        for key, replacement in (("value_coordinate", -1), ("source_token_id", 9708),
                                 ("source_position", 1), ("query_heads", []),
                                 ("value", "123"), ("probability", "1"), ("interaction", "1")):
            with self.assertRaises(ValueError):
                d.component_account(core, {**component, key: replacement},
                                    Fraction(a["value_component_factor"]), downstream,
                                    a["projection_account_key"], 1)

    def test_tampered_stage00_operands_and_v_tensors(self):
        actual, reference, tensors = self.operands
        control = d.parent.CONTROLS[0]
        changed = actual[control].copy()
        changed[0] += 1
        with self.assertRaises(ValueError):
            d.bind_values(self.parent_inputs, ({**actual, control: changed}, reference, tensors))
        changed = reference.copy()
        changed[0] += 1
        with self.assertRaises(ValueError):
            d.bind_values(self.parent_inputs, (actual, changed, tensors))
        canonical = self.e["result"]["preflight"]["L23_original_reference"]["reference"]["canonical"]
        for suffix, tensor in tensors.items():
            changed = tensor.copy()
            changed.view("u1").flat[0] ^= 1
            with self.assertRaises(ValueError):
                d.value.bind_tensor(changed, canonical[d.value.PROJECTION+suffix], suffix)
            with self.assertRaises(ValueError):
                d.value.bind_tensor(tensor.astype("<f8"), canonical[d.value.PROJECTION+suffix], suffix)
        with self.assertRaises(ValueError):
            d.previous.attention.words(np.ones((896,), dtype="<f8"), (896,))

    def test_tampered_stage00_v_state_and_output_cache_lineage(self):
        control = d.parent.CONTROLS[0]
        norm = self.parent_inputs[0][0][1]
        for reference in (False, True):
            prefix = ["reference_archive"] if reference else ["archives", control]
            archive = self.e["reference_archive"] if reference else self.e["archives"][control]
            keys = ("stage00", "stage03", "stage07")
            if not reference:
                keys += ("input_i", "input_z", "scratch_i", "scratch_z", "output_cache_k", "output_cache_v")
            for key in keys:
                changed = archive[key].copy()
                changed.flat[0] ^= 1
                with self.assertRaises(ValueError):
                    d.previous.rmsnorm.bind_inputs(replace_path(self.e, [*prefix, key], changed), norm)
        for kind in ("k", "v"):
            with self.assertRaises(ValueError):
                d.previous.attention.attention_operands(
                    {**self.e["archives"][control], "input_cache_"+kind: np.zeros((1, 128), dtype="<u2")},
                    actual=True)

    def test_tampered_source_tensor_kv_threshold_and_parent_pins(self):
        binding = ["result", "preflight", "L23_original_reference", "reference"]
        for path, replacement in (
            ([*binding, "prior_kv"], "nonempty"),
            ([*binding, "canonical", d.value.PROJECTION+"qweight", "sha256"], "0"*64),
            ([*binding, "fp16", "sha256"], "0"*64),
            (["result", "preflight", "thresholds"], {}),
            (["result", "controls", 0, "parent", "terminal_archive", "sha256"], "0"*64),
        ):
            with self.assertRaises(ValueError):
                d.previous.bind_authority(replace_path(self.e, path, replacement))
        for pin in d.PARENT_PINS:
            with self.assertRaises(ValueError):
                d.base.read_bound({**pin, "sha256": "0"*64})

    def test_zero_ties_signed_cancellation_and_shared_bias(self):
        zero = [Fraction()] * 896
        actual, weights = zero.copy(), zero.copy()
        actual[0] = actual[1] = Fraction(1)
        weights[0], weights[1] = Fraction(3), Fraction(-3)
        core = d.projection_account(actual, zero, weights, Fraction(2), Fraction(2), Fraction(2), 0)
        self.assertEqual([r["coordinate"] for r in core["largest_absolute_input_coordinates"]], list(range(8)))
        self.assertEqual(core["input_coordinate_totals"]["full"], mass([Fraction(3), Fraction(-3)]))
        self.assertEqual(core["actual_projection_boundary"], "0")
        self.assertEqual(core["negative_original_input_fp16_projection_boundary"], "0")
        self.assertEqual([g["input_group"] for g in core["groups_by_absolute_signed_sum"]], list(range(7)))
        for multiplier in (Fraction(), Fraction(-2), Fraction(1, 3)):
            self.assertEqual(d.previous.scaled_mass(core["input_coordinate_totals"]["full"], multiplier),
                             mass([3*multiplier, -3*multiplier]))

    def test_forbidden_predecessor_replay_writes_and_binary64_attention(self):
        target = d.parent.OUTPUT / "forbidden-stage11-value-input-bridge"
        calls = (
            lambda: d.previous.check(), lambda: d.previous.collect(None),
            lambda: d.previous.run_tests(None, None), lambda: d.value.measure(),
            lambda: d.value.report(None, None, None, None), lambda: d.value.check(),
            lambda: d.parent.execute("forbidden"), lambda: d.parent.rmsnorm(None, None),
            lambda: d.parent.logits(None, None), lambda: native._stages(None, None, None, None),
            lambda: d.bridge.residual.local.local_reference(None),
            lambda: d.bridge.margin.contributions.dyadic_dot(None, None),
            lambda: math.exp(0), lambda: np.exp(0),
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
        for key, replacement in (("internal_reference", "binary64"),
                                 ("binary64_internal_stages", "RECONSTRUCTED")):
            with self.assertRaises(ValueError):
                d.report(self.parent_inputs, {**self.retained, key: replacement}, self.operands)

    def test_original_global_references_history_and_claim_boundary(self):
        d.previous.bind_authority(self.e)
        for branch in ("fp16", "binary64"):
            self.assertEqual(self.e["result"]["preflight"]["final_reference"]["reference"]["input_"+branch],
                             self.e["result"]["preflight"]["L23_original_reference"]["reference"][branch])
        self.assertTrue(all(not value for value in d.FLAGS.values()))
        for text in ("UNKNOWN", "Q24", "strict-FP16-state", "not solely", "no reselection"):
            self.assertIn(text, d.BOUNDARY)

    def test_cli_exactly_one_json_document_and_no_write_replay_options(self):
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
