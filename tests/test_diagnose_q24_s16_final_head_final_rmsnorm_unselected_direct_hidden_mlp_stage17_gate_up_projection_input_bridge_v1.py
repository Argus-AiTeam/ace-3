"""Independent half-bit/native-nibble oracle for the middle-pair stage13 bridge."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_gate_up_projection_input_bridge_v1 as d
from ace3.model.candidates import q24_s16_toward_zero_native_v1 as native


EVIDENCE = None


def half(word):
    exponent, mantissa = (word >> 10) & 31, word & 1023
    if exponent == 31:
        raise ValueError("nonfinite retained FP16")
    value = (Fraction(mantissa, 1 << 24) if exponent == 0
             else Fraction(1024+mantissa)*Fraction(2)**(exponent-25))
    return -value if word & 32768 else value


def weights(tensors, output):
    lane = (0, 4, 1, 5, 2, 6, 3, 7)[output % 8]
    result = []
    for coordinate in range(896):
        group = coordinate//128
        q = (int(tensors["qweight"][coordinate, output//8]) % (1 << 32))//16**lane % 16
        z = (int(tensors["qzeros"][group, output//8]) % (1 << 32))//16**lane % 16
        result.append((q-z)*half(int(tensors["scales"].view("<u2")[group, output])))
    return result


def mass(values):
    signed, absolute = sum(values, Fraction()), sum(map(abs, values), Fraction())
    return {"signed": str(signed), "absolute": str(absolute),
            "cancellation_absolute_mass": str(absolute-abs(signed))}


class Stage13BridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is None:
            audit = {"forbidden_calls": 0}
            cls.inputs = d.collect(audit)
            with d.read_only(audit):
                cls.result = d.report(*cls.inputs)
            if audit["forbidden_calls"]:
                raise AssertionError("forbidden work during fixture authentication")
        else:
            cls.inputs, cls.result = EVIDENCE
        cls.previous_inputs, cls.tensors, cls.retained = cls.inputs
        cls.e = cls.previous_inputs[0]
        cls.columns, cls.oracles = {}, {}
        for row in cls.result["rows"]:
            control = row["control"]
            delta = [half(int(a))-half(int(r)) for a, r in zip(
                cls.e["archives"][control]["stage13"],
                cls.e["reference_archive"]["stage13"], strict=True)]
            for hotspot in row["hotspots"]:
                for selected in hotspot["selected_stage16_coordinates"]:
                    for kind in ("gate_proj", "up_proj"):
                        output = selected[kind]["output_coordinate"]
                        if (kind, output) not in cls.columns:
                            cls.columns[kind, output] = weights(cls.tensors[kind], output)
                        column = cls.columns[kind, output]
                        cls.oracles[control, kind, output] = (
                            delta, column, [a*w for a, w in zip(delta, column, strict=True)])

    def accounts(self):
        for row in self.result["rows"]:
            for hotspot in row["hotspots"]:
                for selected in hotspot["selected_stage16_coordinates"]:
                    for kind in ("gate_proj", "up_proj"):
                        account = selected[kind]
                        yield row, hotspot, selected, kind, account, self.oracles[
                            row["control"], kind, account["output_coordinate"]]

    def test_all_896_input_terms_and_ranked_coordinates_independently(self):
        count = 0
        for _, _, _, _, account, (delta, column, terms) in self.accounts():
            expected = [
                {"coordinate": i, "input_group": i//128, "input_delta": str(delta[i]),
                 "weight": str(column[i]), "signed_contribution": str(value),
                 "absolute_contribution": str(abs(value))}
                for i, value in enumerate(terms)]
            self.assertEqual(account["input_coordinates"], expected)
            top = sorted(range(896), key=lambda i: (-abs(terms[i]), i))[:8]
            self.assertEqual(account["largest_absolute_coordinates"], [expected[i] for i in top])
            self.assertEqual(account["input_totals"], mass(terms))
            self.assertEqual(account["exact_input_delta"], str(sum(terms, Fraction())))
            self.assertEqual(account["sum_absolute_contributions"], mass(terms)["absolute"])
            self.assertEqual(account["coordinate_count"], 896)
            count += len(terms)
        self.assertEqual(count, 258048)
        self.assertEqual(count, self.result["input_coordinate_term_count"])

    def test_all_g128_groups_ranked_unranked_and_cancellation_totals(self):
        groups = 0
        for _, _, _, _, account, (_, _, terms) in self.accounts():
            for group, entry in enumerate(account["groups_in_input_order"]):
                total = mass(terms[group*128:(group+1)*128])
                self.assertEqual(entry, {
                    "input_group": group, "start_coordinate": group*128,
                    "end_coordinate_exclusive": (group+1)*128, "coordinate_count": 128,
                    "signed_contribution": total["signed"],
                    "sum_absolute_contributions": total["absolute"],
                    "cancellation_absolute_mass": total["cancellation_absolute_mass"]})
                groups += 1
            top = sorted(range(896), key=lambda i: (-abs(terms[i]), i))[:8]
            ranked = mass([terms[i] for i in top])
            rest = mass([value for i, value in enumerate(terms) if i not in top])
            self.assertEqual(account["ranked_input_totals"], ranked)
            self.assertEqual(account["unranked_input_totals"], rest)
            self.assertEqual(account["ranked_signed_sum"], ranked["signed"])
            self.assertEqual(account["ranked_absolute_sum"], ranked["absolute"])
            self.assertEqual(account["unranked_signed_remainder"], rest["signed"])
            self.assertEqual(account["unranked_absolute_remainder"], rest["absolute"])
        self.assertEqual(groups, 2016)
        self.assertEqual(groups, self.result["input_group_count"])

    def test_every_projection_boundary_closes_to_retained_stage16_operand(self):
        for row, _, selected, kind, account, (_, _, terms) in self.accounts():
            stage, operand = {"gate_proj": ("stage14", "gate"), "up_proj": ("stage15", "up")}[kind]
            output = selected["selected_down_input"]["coordinate"]
            a = half(int(self.e["archives"][row["control"]][stage][output]))
            r = half(int(self.e["reference_archive"][stage][output]))
            exact = sum(terms, Fraction())
            remainder = a-r-exact
            self.assertEqual(account["projection"], kind)
            self.assertEqual(account["output_coordinate"], output)
            self.assertEqual(account["input_stage"], "stage13_fp16")
            self.assertEqual(account["output_stage"], stage+"_fp16")
            self.assertEqual(account["reference"], "original_input_L23_fp16")
            self.assertEqual(account["actual_projection_output"], str(a))
            self.assertEqual(account["reference_projection_output"], str(r))
            self.assertEqual(account["retained_projection_delta"], str(a-r))
            self.assertEqual(selected["bridge"]["actual"][operand], str(a))
            self.assertEqual(selected["bridge"]["reference_operands"][operand], str(r))
            self.assertEqual(selected["bridge"][operand+"_delta"], str(a-r))
            self.assertEqual(account["projection_boundary_remainder"], str(remainder))
            self.assertEqual(exact+remainder, a-r)
            self.assertEqual(account["projection_closure_residual"], "0")
            self.assertEqual(account["input_and_boundary_cancellation_mass"],
                             str(sum(map(abs, terms), Fraction())+abs(remainder)-abs(a-r)))
            self.assertTrue(account["exact_local_identity"])
            self.assertTrue(account["retained_stage16_operand_delta_unchanged"])

    def test_retained_selections_counts_pair_and_common_gates(self):
        self.assertEqual(self.result["retained_stage16_bridge"], self.retained)
        self.assertEqual((self.result["left_id"], self.result["right_id"]), (34319, 13))
        self.assertEqual((self.result["layer"], self.result["position"]), (23, 0))
        self.assertEqual([self.result[key] for key in (
            "row_count", "hotspot_count", "selected_stage16_coordinate_count",
            "projection_account_count", "selected_input_coordinate_count")],
            [18, 18, 144, 288, 2304])
        self.assertEqual(self.result["common_component"], "UNKNOWN")
        self.assertTrue(self.result["stop_nested_bridge_expansion"])
        self.assertTrue(self.result["prior_pair_and_common_gates_unchanged"])
        self.assertEqual([p["component"] for p in self.previous_inputs[2]["pairs"]],
                         ["UNKNOWN", "mlp_stage17", "UNKNOWN"])
        for row, old in zip(self.result["rows"], self.retained["rows"], strict=True):
            self.assertEqual({k: v for k, v in row.items() if k != "hotspots"},
                             {k: v for k, v in old.items() if k != "hotspots"})
            for hotspot, prior in zip(row["hotspots"], old["hotspots"], strict=True):
                self.assertEqual({k: v for k, v in hotspot.items() if k != "selected_stage16_coordinates"},
                                 {k: v for k, v in prior.items() if k != "selected_stage16_coordinates"})
                self.assertEqual([{k: v for k, v in entry.items() if k not in ("gate_proj", "up_proj")}
                                  for entry in hotspot["selected_stage16_coordinates"]],
                                 prior["selected_stage16_coordinates"])

    def test_coordinate741_and_all_control_branch_representatives(self):
        expected = {
            "fp16": "972712872287995455840195/19342813113834066795298816",
            "binary64": "1955800403415784668441195/38685626227668133590597632"}
        self.assertEqual([(r["control"], r["branch"]) for r in self.result["rows"]],
                         [(c, b) for c in d.hotspot.census.CONTROLS for b in ("fp16", "binary64")])
        for branch, contribution in expected.items():
            row = next(r for r in self.result["rows"]
                       if r["control"] == "frozen_inherited" and r["branch"] == branch)
            self.assertEqual(row["retained_maximum_coordinate_ties"], [741])
            hotspot = row["hotspots"][0]
            self.assertEqual(hotspot["output_coordinate"], 741)
            self.assertEqual(hotspot["summary"]["retained_coordinate_signed_contribution"], contribution)
            self.assertEqual(len(hotspot["selected_stage16_coordinates"]), 8)
        for control in d.hotspot.census.CONTROLS:
            rows = [r for r in self.result["rows"] if r["control"] == control]
            for kind in ("gate_proj", "up_proj"):
                self.assertEqual(
                    [s[kind] for h in rows[0]["hotspots"] for s in h["selected_stage16_coordinates"]],
                    [s[kind] for h in rows[1]["hotspots"] for s in h["selected_stage16_coordinates"]])

    def test_tampered_retained_pair_control_branch_hotspot_stage16_and_gates(self):
        for mutate in (
            lambda r: r.update(left_id=319),
            lambda r: r["rows"][0].update(control="scratch"),
            lambda r: r["rows"][0].update(branch="binary64"),
            lambda r: r["rows"][0].update(retained_maximum_coordinate_ties=[62]),
            lambda r: r["rows"][0]["hotspots"][0].update(output_coordinate=62),
            lambda r: r["rows"][0]["hotspots"][0]["selected_stage16_coordinates"].reverse(),
            lambda r: r["rows"][0]["hotspots"][0]["selected_stage16_coordinates"][0]
                ["bridge"].update(gate_delta="1"),
            lambda r: r.update(common_component="mlp_stage17"),
            lambda r: r.update(stop_nested_bridge_expansion=False),
        ):
            changed = deepcopy(self.retained)
            mutate(changed)
            with self.assertRaises(ValueError):
                d.report(self.previous_inputs, self.tensors, changed)

    def test_tampered_actual_reference_stage13_gate_up_bits_geometry_and_census(self):
        control = d.hotspot.census.CONTROLS[0]
        for reference in (False, True):
            original = self.e["reference_archive"] if reference else self.e["archives"][control]
            for stage in ("stage13", "stage14", "stage15"):
                changed_bits = original[stage].copy()
                changed_bits[0] ^= 1
                for vector in (changed_bits, original[stage][:-1], original[stage].astype("<i4")):
                    archive = {**original, stage: vector}
                    changed = ({**self.e, "reference_archive": archive} if reference else
                               {**self.e, "archives": {**self.e["archives"], control: archive}})
                    with self.assertRaises(ValueError):
                        d.bind_inputs(changed, self.tensors)
        with self.assertRaises(ValueError):
            d.bind_inputs({**self.e, "archives": dict(reversed(list(self.e["archives"].items())))},
                          self.tensors)

    def test_tampered_stage16_retained_operands_rejected_before_accounting(self):
        control = d.hotspot.census.CONTROLS[0]
        for reference in (False, True):
            original = self.e["reference_archive"] if reference else self.e["archives"][control]
            vector = original["stage16"].copy()
            vector[0] ^= 1
            archive = {**original, "stage16": vector}
            changed = ({**self.e, "reference_archive": archive} if reference else
                       {**self.e, "archives": {**self.e["archives"], control: archive}})
            with self.assertRaises(ValueError):
                d.report((changed, *self.previous_inputs[1:]), self.tensors, self.retained)

    def test_all_six_canonical_tensor_bindings_and_tampering(self):
        canonical = self.e["result"]["preflight"]["L23_original_reference"]["reference"]["canonical"]
        bindings = self.result["gate_up_projection_tensor_bindings"]
        self.assertEqual(len(bindings), 6)
        for kind in ("gate_proj", "up_proj"):
            prefix = "model.layers.23.mlp."+kind+"."
            for suffix in ("qweight", "qzeros", "scales"):
                self.assertEqual(bindings[prefix+suffix], canonical[prefix+suffix])
                tensor = self.tensors[kind][suffix]
                changed = tensor.copy()
                changed.view("u1").flat[0] ^= 1
                for value in (changed, tensor[:-1], tensor.astype("<f8")):
                    tensors = {**self.tensors, kind: {**self.tensors[kind], suffix: value}}
                    with self.assertRaises(ValueError):
                        d.bind_inputs(self.e, tensors)
                with self.assertRaises(ValueError):
                    d.gate_up.bind_tensor(tensor, {**canonical[prefix+suffix], "name": "wrong"},
                                          kind, suffix)
            with self.assertRaises(ValueError):
                d.gate_up.tensor_names({**canonical, prefix+"bias": {}}, kind)
        for tensors in ({k: v for k, v in self.tensors.items() if k != "up_proj"},
                        {**self.tensors, "down_proj": self.tensors["up_proj"]},
                        {**self.tensors, "gate_proj": {**self.tensors["gate_proj"], "bias": np.zeros(1)}}):
            with self.assertRaises(ValueError):
                d.bind_inputs(self.e, tensors)

    def test_tampered_new_input_rank_group_remainder_and_gate_reports(self):
        for mutate in (
            lambda r: r["rows"][0]["hotspots"][0]["selected_stage16_coordinates"][0]["gate_proj"]
                ["input_coordinates"][0].update(signed_contribution="1"),
            lambda r: r["rows"][0]["hotspots"][0]["selected_stage16_coordinates"][0]["up_proj"]
                ["largest_absolute_coordinates"].reverse(),
            lambda r: r["rows"][0]["hotspots"][0]["selected_stage16_coordinates"][0]["gate_proj"]
                ["groups_in_input_order"][0].update(cancellation_absolute_mass="1"),
            lambda r: r["rows"][0]["hotspots"][0]["selected_stage16_coordinates"][0]["up_proj"]
                .update(projection_boundary_remainder="1"),
            lambda r: r.update(input_coordinate_term_count=0),
            lambda r: r.update(stop_nested_bridge_expansion=False),
        ):
            changed = deepcopy(self.result)
            mutate(changed)
            with self.assertRaises(ValueError):
                d.validate_report(self.inputs, changed)

    def test_original_global_references_thresholds_and_historical_failures(self):
        self.assertEqual(self.result["internal_reference"], "original_input_L23_fp16")
        self.assertEqual(self.result["binary64_internal_stages"], "NOT_RETAINED_NO_RECONSTRUCTION")
        for branch in ("fp16", "binary64"):
            self.assertEqual(self.e["result"]["preflight"]["final_reference"]["reference"]["input_"+branch],
                             self.e["result"]["preflight"]["L23_original_reference"]["reference"][branch])
        for field, value in (("candidate_admitted", True), ("S18_failure_indices", []),
                             ("source_operand_state_KV_RTZ_checks", "FAIL")):
            changed = deepcopy(self.e["result"])
            changed["controls"][0]["parent"]["retained_L23"][field] = value
            with self.assertRaises(ValueError):
                d.base.check_history(changed)
        changed = deepcopy(self.e["result"])
        changed["controls"][0]["parent"]["L23_failures"][0]["excess_budget"] = "1"
        with self.assertRaises(ValueError):
            d.base.check_history(changed)

    def test_zero_cancellation_ties_and_separate_projection_remainder(self):
        reference = {"stage13": [Fraction(0)]*896, "stage14": [Fraction(0)]*4864,
                     "stage15": [Fraction(0)]*4864}
        actual = {**reference, "stage13": [Fraction(1), Fraction(-1)]+[Fraction(0)]*894,
                  "stage14": [Fraction(3)]+[Fraction(0)]*4863,
                  "stage15": [Fraction(-2)]+[Fraction(0)]*4863}
        selected = {
            "selected_down_input": {"coordinate": 0},
            "bridge": {"actual": {"gate": "3", "up": "-2"},
                       "reference_operands": {"gate": "0", "up": "0"},
                       "gate_delta": "3", "up_delta": "-2"}}
        for kind, remainder in (("gate_proj", "3"), ("up_proj", "-2")):
            account = d.account(actual, reference, [Fraction(1)]*896, kind, selected)
            self.assertEqual(account["input_totals"], mass([Fraction(1), Fraction(-1)]))
            self.assertEqual(account["projection_boundary_remainder"], remainder)
            self.assertEqual([r["coordinate"] for r in account["largest_absolute_coordinates"]],
                             list(range(8)))
            zero = d.account(actual, reference, [Fraction(0)]*896, kind, selected)
            self.assertEqual(zero["input_totals"], mass([]))
        for coordinate in (-1, 4864, True):
            with self.assertRaises(ValueError):
                d.account(actual, reference, [Fraction(1)]*896, "gate_proj",
                          {**selected, "selected_down_input": {"coordinate": coordinate}})
        with self.assertRaises(ValueError):
            d.account(actual, reference, [1.0]*896, "gate_proj", selected)
        with self.assertRaises(ValueError):
            d.account(actual, reference, [Fraction(1)]*896, "gate_proj",
                      {**selected, "bridge": {**selected["bridge"], "gate_delta": "0"}})

    def test_native_nibble_order_no_plus_one_and_fp16_scale_oracle(self):
        tensors = {suffix: np.zeros(shape, dtype=dtype)
                   for suffix, (shape, dtype) in d.gate_up.SPECS.items()}
        tensors["qweight"][:] = np.array(0xFEDCBA98, dtype="<u4").view("<i4")
        tensors["qzeros"][:] = 0x12345678
        for group in range(7):
            tensors["scales"][group] = np.float16((group+1)/16)
        for output in (*range(8), 4863):
            actual = d.gate_up.weight_column(tensors, output)
            self.assertEqual(actual, weights(tensors, output))
            for group in range(7):
                self.assertEqual(actual[group*128], actual[(group+1)*128-1])
        self.assertEqual(d.gate_up.weight_column(tensors, 0), [Fraction(0)]*896)

    def test_forbidden_replay_nonlinear_dispatch_and_writes(self):
        target = d.parent.OUTPUT / "forbidden-middle-pair-stage13-bridge"
        torch = d.parent.preflight.parent.producer.legacy.torch
        calls = (
            lambda: d.previous.check(), lambda: d.previous.run_tests(None, None),
            lambda: d.previous.collect(None), lambda: d.gate_up.measure(),
            lambda: d.gate_up.report(None, None, None, None, None),
            lambda: d.bridge.measure(), lambda: d.bridge.report(None),
            lambda: d.hotspot.check(), lambda: d.down.measure(),
            lambda: d.parent.rmsnorm(None, None), lambda: d.parent.logits(None, None),
            lambda: d.parent.execute("forbidden"), lambda: d.parent.head.decode_array_q24(None),
            lambda: native._stages(None, None, None, None),
            lambda: d.bridge.residual.local.local_reference(None),
            lambda: d.bridge.margin.contributions.dyadic_dot(None, None),
            lambda: math.exp(0), lambda: np.exp(0), lambda: torch.sigmoid(None),
            lambda: torch.nn.functional.silu(None), lambda: subprocess.run(["forbidden"]),
            lambda: os.system("forbidden"), lambda: torch.cuda._lazy_init(),
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

    def test_source_authentication_and_non_admission_flags(self):
        with self.assertRaises(ValueError):
            d.base.read_bound({**d.parent.record(d.previous.SOURCE), "sha256": "0"*64})
        self.assertTrue(all(not flag for flag in d.FLAGS.values()))
        self.assertIn("common UNKNOWN", d.BOUNDARY)
        self.assertIn("not solely rounding", d.BOUNDARY)
        self.assertIn("Q24", d.BOUNDARY)
        self.assertIn("original-input FP16", d.BOUNDARY)

    def test_cli_refuses_extra_paths_and_abbreviations(self):
        for argv in ([], ["--execute"], ["--check", "--out", "/tmp/forbidden"],
                     ["--check", "--reference"], ["--check", "--che"]):
            with patch.object(d, "check", side_effect=AssertionError("dispatch")), \
                    patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit) as error:
                d.main(argv)
            self.assertEqual(error.exception.code, 2)

    def test_stdout_exactly_one_json_document(self):
        output = io.StringIO()
        with patch.object(d, "check", return_value={"report": self.result}), patch("sys.stdout", output):
            d.main(["--check"])
        value, end = json.JSONDecoder().raw_decode(output.getvalue())
        self.assertEqual(value, {"report": self.result})
        self.assertEqual(output.getvalue()[end:], "\n")
