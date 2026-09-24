"""Independent bit-level oracle and mutation checks for stage00 complements."""

from fractions import Fraction
import io
import json
import math
import os
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_stage13_rmsnorm_stage12_attention_value_projection_unselected_stage00_complement_v1 as d
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


class ComplementTests(unittest.TestCase):
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
        cls.operands = cls.parent_inputs[2]
        cls.attention_inputs = cls.parent_inputs[0]
        cls.e = d.previous.previous.evidence(cls.attention_inputs[0])
        cls.oracles, cls.coordinate_masses, cls.input_sums, columns = {}, {}, {}, {}
        for key, core in cls.retained["value_projection_local_accounts"].items():
            v = core["output_coordinate"]
            if v not in columns:
                columns[v] = [weight(cls.operands[2], v, i) for i in range(896)]
            a, r = cls.e["archives"][core["control"]], cls.e["reference_archive"]
            terms = [(half(int(x))-half(int(y)))*w for x, y, w in
                     zip(a["stage00"], r["stage00"], columns[v], strict=True)]
            selected = sorted(range(896), key=lambda i: (-abs(terms[i]), i))[:8]
            cls.oracles[key] = terms, selected
            cls.coordinate_masses[key] = {
                "selected": mass(terms[i] for i in selected),
                "unselected": mass(terms[i] for i in range(896) if i not in selected),
            }
            cls.input_sums[key] = sum(terms, Fraction())

    def test_exact_complement_for_every_shared_account(self):
        for key, (terms, selected) in self.oracles.items():
            split = self.result["value_projection_local_accounts"][key]["unselected_stage00_inputs"]
            self.assertEqual(split["excluded_ranked_coordinates"], selected)
            self.assertEqual(split["totals"], mass(terms[i] for i in range(896) if i not in selected))
            self.assertEqual(split["coordinate_count"], 888)
            self.assertEqual(split["closure_residual"], "0")

    def test_seven_g128_groups_counts_order_and_cancellation(self):
        for key, (terms, selected) in self.oracles.items():
            groups = self.result["value_projection_local_accounts"][key]["unselected_stage00_inputs"]["groups_in_input_order"]
            self.assertEqual([g["input_group"] for g in groups], list(range(7)))
            self.assertEqual(sum(g["coordinate_count"] for g in groups), 888)
            self.assertEqual(sum(g["excluded_ranked_coordinate_count"] for g in groups), 8)
            for index, group in enumerate(groups):
                indices = [i for i in range(index*128, (index+1)*128) if i not in selected]
                self.assertEqual((group["start_coordinate"], group["end_coordinate_exclusive"]),
                                 (index*128, (index+1)*128))
                self.assertEqual(group["coordinate_count"], len(indices))
                self.assertEqual({k: group[k] for k in mass([])}, mass(terms[i] for i in indices))

    def test_selected_entries_and_entire_parent_report_unchanged(self):
        stripped = {**self.result}
        stripped.pop("unselected_stage00_complement_summary")
        stripped.pop("unselected_stage00_complement_weighting")
        stripped["value_projection_local_accounts"] = {
            key: {k: v for k, v in core.items() if k != "unselected_stage00_inputs"}
            for key, core in stripped["value_projection_local_accounts"].items()}
        self.assertEqual(stripped, self.retained)

    def test_every_downstream_selected_unselected_boundary_identity(self):
        count = 0
        for account in d.accounts(self.result):
            key = account["projection_account_key"]
            terms, selected = self.oracles[key]
            core = self.retained["value_projection_local_accounts"][key]
            split = self.result["value_projection_local_accounts"][key]["unselected_stage00_inputs"]
            d.weighted_closure(core, split, account)
            for field in d.FIELDS:
                factor = Fraction(account["input_coordinate_multipliers"][field])
                for group in ("selected", "unselected"):
                    expected = self.coordinate_masses[key][group]
                    signed = Fraction(expected["signed"])*factor
                    absolute = Fraction(expected["absolute"])*abs(factor)
                    self.assertEqual(account["input_coordinate_totals"][group][field],
                                     {"signed": str(signed), "absolute": str(absolute),
                                      "cancellation_absolute_mass": str(absolute-abs(signed))})
                boundary = sum(Fraction(core[t]) for t in d.previous.TERMS[1:])
                self.assertEqual((self.input_sums[key]+boundary)*factor,
                                 Fraction(account["retained_parent_contributions"][field]))
            count += 1
        summary = self.result["unselected_stage00_complement_summary"]
        self.assertEqual(count, 18432)
        self.assertEqual(summary["selected_value_component_count"], count)
        self.assertEqual(summary["unselected_input_coordinate_count"], 16367616)
        self.assertEqual(summary["unselected_input_group_count"], 129024)

    def test_downstream_multipliers_independently_propagated(self):
        norm_inputs = self.attention_inputs[0][0]
        projection_inputs = norm_inputs[0]
        columns = {}
        for row in self.retained["rows"]:
            for hotspot in row["hotspots"]:
                for selected in hotspot["selected_stage16_coordinates"]:
                    j = selected["selected_down_input"]["coordinate"]
                    down = weight(projection_inputs[0][4], hotspot["output_coordinate"], j)
                    for kind in ("gate_proj", "up_proj"):
                        opposite = half(int(self.e["reference_archive"][
                            "stage15" if kind == "gate_proj" else "stage14"][j]))
                        outer = down*opposite*Fraction(hotspot["retained_weighted_row_factor"])
                        for source in selected[kind]["ranked_stage13_rmsnorm_sources"]:
                            i = source["retained_ranked_input"]["coordinate"]
                            norm = half(int(norm_inputs[1].view("<u2")[i]))
                            anchor = Fraction(norm_inputs[2]["original_input_fp16"]["inverse_norm_anchor"])
                            gu = weight(projection_inputs[1][kind], j, i)
                            factors = (Fraction(1), norm*anchor, norm*anchor*gu, norm*anchor*gu*outer)
                            if i not in columns:
                                columns[i] = [weight(self.attention_inputs[2], i, c) for c in range(896)]
                            for a in source["attention_value_projection_input"]["selected_value_components"]:
                                v = a["retained_parent_value_component"]["value_coordinate"]
                                kv, dim = divmod(v, 64)
                                gqa = sum((columns[i][h*64+dim] for h in range(kv*7, (kv+1)*7)), Fraction())
                                self.assertEqual([Fraction(a["input_coordinate_multipliers"][f])
                                                  for f in d.FIELDS], [gqa*f for f in factors])

    def test_selected_order_and_projection_mutation_rejected(self):
        core = next(iter(self.retained["value_projection_local_accounts"].values()))
        args = (self.operands[0][core["control"]], self.operands[1],
                d.previous.value.weight_column(self.operands[2], core["output_coordinate"]))
        for key, changed in (
            ("largest_absolute_input_coordinates", core["largest_absolute_input_coordinates"][::-1]),
            ("input_coordinate_totals", {}), ("groups_by_absolute_signed_sum", []),
            ("retained_projection_delta", "123"), ("source_token_id", 9708)):
            with self.assertRaises(ValueError):
                d.split_unselected(*args, {**core, key: changed})

    def test_downstream_mutations_rejected(self):
        account = next(d.accounts(self.retained))
        core = self.result["value_projection_local_accounts"][account["projection_account_key"]]
        field = d.FIELDS[0]
        for path in (["input_coordinate_multipliers", field],
                     ["input_coordinate_totals", "unselected", field, "absolute"],
                     ["terms", 1, field], ["retained_parent_contributions", field],
                     ["closure_residuals", field]):
            with self.assertRaises(ValueError):
                d.weighted_closure(core, core["unselected_stage00_inputs"],
                                   replace_path(account, path, "123"))

    def test_parent_binding_and_source_test_pins_rejected(self):
        for pin in d.PARENT_PINS:
            with self.assertRaises(ValueError):
                d.base.read_bound({**pin, "sha256": "0"*64})
        with patch.object(d, "PARENT_PINS", ({**d.PARENT_PINS[0], "sha256": "0"*64},)):
            with self.assertRaises(ValueError):
                d.authenticate_parent()
        with patch.object(d.previous, "report", return_value=self.retained):
            for field, value in (("common_component", "ADMITTED"), ("internal_reference", "binary64"),
                                 ("binary64_internal_stages", "RECONSTRUCTED")):
                with self.assertRaises(ValueError):
                    d.bind_parent(self.parent_inputs, {**self.retained, field: value})

    def test_actual_original_stage00_and_v_tensors_rejected(self):
        actual, reference, tensors = self.operands
        control = d.parent.CONTROLS[0]
        changed = actual[control].copy()
        changed[0] += 1
        with self.assertRaises(ValueError):
            d.previous.bind_values(self.attention_inputs, ({**actual, control: changed}, reference, tensors))
        changed = reference.copy()
        changed[0] += 1
        with self.assertRaises(ValueError):
            d.previous.bind_values(self.attention_inputs, (actual, changed, tensors))
        canonical = self.e["result"]["preflight"]["L23_original_reference"]["reference"]["canonical"]
        for suffix, tensor in tensors.items():
            changed = tensor.copy()
            changed.view("u1").flat[0] ^= 1
            with self.assertRaises(ValueError):
                d.previous.value.bind_tensor(changed, canonical[d.previous.value.PROJECTION+suffix], suffix)

    def test_stage00_stage03_stage07_state_and_output_cache_rejected(self):
        control = d.parent.CONTROLS[0]
        norm = self.attention_inputs[0][0][1]
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
                    d.previous.previous.rmsnorm.bind_inputs(replace_path(self.e, [*prefix, key], changed), norm)

    def test_source_authority_thresholds_and_empty_p0_kv_rejected(self):
        binding = ["result", "preflight", "L23_original_reference", "reference"]
        for path, replacement in (
            ([*binding, "prior_kv"], "nonempty"),
            ([*binding, "canonical", d.previous.value.PROJECTION+"qweight", "sha256"], "0"*64),
            ([*binding, "fp16", "sha256"], "0"*64),
            (["result", "preflight", "thresholds"], {}),
            (["result", "controls", 0, "parent", "terminal_archive", "sha256"], "0"*64)):
            with self.assertRaises(ValueError):
                d.previous.previous.bind_authority(replace_path(self.e, path, replacement))
        archive = self.e["archives"][d.parent.CONTROLS[0]]
        for kind in ("k", "v"):
            with self.assertRaises(ValueError):
                d.previous.previous.attention.attention_operands(
                    {**archive, "input_cache_"+kind: np.zeros((1, 128), dtype="<u2")}, actual=True)

    def test_zero_ties_negative_weights_and_group_edge_exclusions(self):
        zero = [Fraction()] * 896
        actual, weights = zero.copy(), zero.copy()
        for i in (0, 127, 128, 255, 256, 383, 384, 895):
            actual[i], weights[i] = Fraction(1), Fraction(3 if i % 2 else -3)
        actual[500], weights[500] = Fraction(1), Fraction(1)
        actual[501], weights[501] = Fraction(1), Fraction(-1)
        core = d.previous.projection_account(actual, zero, weights, Fraction(2), Fraction(2), Fraction(2), 0)
        split = d.split_unselected(actual, zero, weights, core)
        self.assertEqual(split["excluded_ranked_coordinates"], [0, 127, 128, 255, 256, 383, 384, 895])
        self.assertEqual(split["totals"], mass([Fraction(1), Fraction(-1)]))
        self.assertEqual([g["coordinate_count"] for g in split["groups_in_input_order"]],
                         [126, 126, 126, 127, 128, 128, 127])
        core = d.previous.projection_account(zero, zero, zero, Fraction(), Fraction(), Fraction(), 0)
        split = d.split_unselected(zero, zero, zero, core)
        self.assertEqual(split["excluded_ranked_coordinates"], list(range(8)))
        self.assertEqual(split["totals"], mass([]))

    def test_forbidden_checks_operator_replay_and_internal_reconstruction(self):
        calls = (
            lambda: d.previous.check(), lambda: d.previous.run_tests(None, None),
            lambda: d.previous.collect(None), lambda: d.previous.value.measure(),
            lambda: d.previous.previous.check(), lambda: d.parent.execute("forbidden"),
            lambda: d.parent.rmsnorm(None, None), lambda: d.parent.logits(None, None),
            lambda: native._stages(None, None, None, None),
            lambda: d.bridge.residual.local.local_reference(None),
            lambda: d.bridge.margin.contributions.dyadic_dot(None, None),
            lambda: math.exp(0), lambda: np.exp(0),
            lambda: subprocess.run(["forbidden"]), lambda: os.system("forbidden"))
        audit = {"forbidden_calls": 0}
        with d.read_only(audit):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit["forbidden_calls"], len(calls))

    def test_forbidden_evidence_writes(self):
        target = d.parent.OUTPUT / "forbidden-stage00-complement"
        calls = (lambda: target.write_text("forbidden"), lambda: open(target, "wb"),
                 lambda: os.open(target, os.O_WRONLY | os.O_CREAT), lambda: target.mkdir(),
                 lambda: os.replace(target, target), lambda: target.unlink())
        audit = {"forbidden_calls": 0}
        with d.read_only(audit):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit["forbidden_calls"], len(calls))

    def test_original_references_history_unknown_and_nonadmission_flags(self):
        d.previous.previous.bind_authority(self.e)
        for branch in ("fp16", "binary64"):
            self.assertEqual(self.e["result"]["preflight"]["final_reference"]["reference"]["input_"+branch],
                             self.e["result"]["preflight"]["L23_original_reference"]["reference"][branch])
        self.assertEqual(self.result["common_component"], "UNKNOWN")
        self.assertTrue(self.result["stop_nested_bridge_expansion"])
        self.assertEqual(
            json.dumps(d.FLAGS, sort_keys=True),
            json.dumps({**d.previous.FLAGS, "stage00_complement_causal_allocation": False},
                       sort_keys=True))
        self.assertTrue(all(value is False or (type(value) is int and value == 0)
                            for value in d.FLAGS.values()))
        for text in ("Q24", "strict-FP16-state", "not solely", "no reselection", "888"):
            self.assertIn(text, d.BOUNDARY)

    def test_cli_single_json_and_no_write_replay_options(self):
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
