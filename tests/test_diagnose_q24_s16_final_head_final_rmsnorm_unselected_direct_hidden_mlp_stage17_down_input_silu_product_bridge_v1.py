"""Independent retained-FP16-bit oracle for selected stage16 four-term accounts."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_down_input_silu_product_bridge_v1 as d
from ace3.model.candidates import q24_s16_toward_zero_native_v1 as native


EVIDENCE = None


def half(word):
    exponent, mantissa = (word >> 10) & 31, word & 1023
    if exponent == 31:
        raise ValueError("nonfinite retained FP16 operand")
    value = (Fraction(mantissa, 1 << 24) if exponent == 0
             else Fraction(1024+mantissa)*Fraction(2)**(exponent-25))
    return -value if word & 32768 else value


def weights(tensors, output):
    lane = (0, 4, 1, 5, 2, 6, 3, 7)[output % 8]
    result = []
    for coordinate in range(4864):
        group = coordinate//128
        q = (int(tensors["qweight"][coordinate, output//8]) % (1 << 32))//16**lane % 16
        z = (int(tensors["qzeros"][group, output//8]) % (1 << 32))//16**lane % 16
        result.append((q-z)*half(int(tensors["scales"].view("<u2")[group, output])))
    return result


def mass(values):
    signed, absolute = sum(values, Fraction()), sum(map(abs, values), Fraction())
    return {"signed": str(signed), "absolute": str(absolute),
            "cancellation_absolute_mass": str(absolute-abs(signed))}


class Stage16BridgeTests(unittest.TestCase):
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
        cls.e, cls.c, cls.dom, cls.observed, cls.tensors, cls.retained = cls.inputs
        cls.columns, cls.expected = {}, {}
        for row in cls.result["rows"]:
            control, branch = row["control"], row["branch"]
            pair = next(p for c in cls.e["report"]["controls"] if c["control"] == control
                        for p in c["pairs"] if (p["left_id"], p["right_id"]) == (34319, 13))
            anchor = Fraction(pair["branches"][branch]["unchanged_unselected_bridge"]
                              ["reference_norm"]["inverse_norm_anchor"])
            for account in row["hotspots"]:
                output = account["output_coordinate"]
                if output not in cls.columns:
                    cls.columns[output] = weights(cls.tensors, output)
                factor = (half(int(cls.e["weight_array"].view("<u2")[output]))*anchor
                          * (half(int(cls.e["rows"][34319].view("<u2")[output]))
                             - half(int(cls.e["rows"][13].view("<u2")[output]))))
                actual, reference = cls.e["archives"][control], cls.e["reference_archive"]
                deltas = [half(int(a))-half(int(r))
                          for a, r in zip(actual["stage16"], reference["stage16"], strict=True)]
                terms = [delta*w for delta, w in zip(deltas, cls.columns[output], strict=True)]
                stage17 = half(int(actual["stage17"][output]))-half(int(reference["stage17"][output]))
                cls.expected[control, branch, output] = factor, terms, stage17

    def accounts(self):
        for row in self.result["rows"]:
            for account in row["hotspots"]:
                yield row, account, self.expected[row["control"], row["branch"],
                                                 account["output_coordinate"]]

    def test_every_local_identity_from_independent_retained_bits(self):
        count = 0
        for row, account, _ in self.accounts():
            for selected in account["selected_stage16_coordinates"]:
                local = selected["bridge"]
                coordinate = selected["selected_down_input"]["coordinate"]
                actual, reference = (
                    [half(int(archive[stage][coordinate]))
                     for stage in ("stage14", "stage15", "stage16")]
                    for archive in (self.e["archives"][row["control"]], self.e["reference_archive"]))
                ag, au, ay = actual
                rg, ru, ry = reference
                delta_gate, delta_up = ag-rg, au-ru
                boundary_actual, boundary_reference = ay-ag*au, ry-rg*ru
                values = [delta_gate*ru, rg*delta_up, delta_gate*delta_up,
                          boundary_actual-boundary_reference]
                self.assertEqual(sum(values, Fraction()), ay-ry)
                self.assertEqual(local["actual"], dict(zip(("gate", "up", "stage16"), map(str, actual))))
                self.assertEqual(local["reference_operands"],
                                 dict(zip(("gate", "up", "stage16"), map(str, reference))))
                for key, expected in (
                    ("gate_delta", delta_gate), ("up_delta", delta_up),
                    ("retained_stage16_delta", ay-ry),
                    ("actual_raw_gate_up_product", ag*au), ("reference_raw_gate_up_product", rg*ru),
                    ("raw_gate_up_product_delta", ag*au-rg*ru),
                    ("actual_silu_product_remainder", boundary_actual),
                    ("reference_silu_product_remainder", boundary_reference),
                    ("silu_product_boundary_delta", boundary_actual-boundary_reference),
                ):
                    self.assertEqual(local[key], str(expected))
                self.assertEqual([Fraction(t["signed_contribution"])
                                  for t in local["terms_in_operand_order"]], values)
                self.assertEqual([t["term"] for t in local["terms_in_operand_order"]], [
                    "gate_delta_times_reference_up", "reference_gate_times_up_delta",
                    "gate_up_interaction", "silu_product_boundary_delta"])
                self.assertEqual([t["term_index"] for t in local["ranked_absolute_terms"]],
                                 sorted(range(4), key=lambda i: (-abs(values[i]), i)))
                self.assertEqual(local["reference"], "original_input_L23_fp16")
                self.assertEqual(local["baseline"], "raw_gate_times_up_NOT_SILU")
                self.assertEqual(local["separate_silu_product_rounding"], "NOT_IDENTIFIABLE_WITHOUT_REPLAY")
                self.assertTrue(local["exact_local_identity"])
                count += 1
        self.assertGreater(count, 0)
        self.assertEqual(count, self.result["bridge_account_count"])
        self.assertEqual(self.result["closure_term_count"], 4*count)

    def test_every_weighted_term_and_stage17_closure_independently(self):
        for row, account, (factor, terms, stage17) in self.accounts():
            output = account["output_coordinate"]
            selected_ids = sorted(range(4864), key=lambda i: (-abs(terms[i]), i))[:8]
            self.assertEqual([c["selected_down_input"]["coordinate"]
                              for c in account["selected_stage16_coordinates"]], selected_ids)
            self.assertEqual(account["retained_weighted_row_factor"], str(factor))
            expanded = []
            for rank, selected in enumerate(account["selected_stage16_coordinates"], 1):
                self.assertEqual(selected["stage16_coordinate_rank"], rank)
                local, pin = selected["bridge"], selected["selected_down_input"]
                coordinate = pin["coordinate"]
                weight = self.columns[output][coordinate]
                self.assertEqual(pin["weight"], str(weight))
                self.assertEqual(pin["signed_contribution"], str(terms[coordinate]))
                values = [Fraction(t["signed_contribution"]) for t in local["terms_in_operand_order"]]
                weighted = [factor*weight*v for v in values]
                self.assertEqual([Fraction(t["signed_contribution"])
                                  for t in local["weighted_terms_in_operand_order"]],
                                 [weight*v for v in values])
                for key, value in (("weighted_actual_silu_product_remainder",
                                    weight*Fraction(local["actual_silu_product_remainder"])),
                                   ("weighted_reference_silu_product_remainder",
                                    weight*Fraction(local["reference_silu_product_remainder"]))):
                    self.assertEqual(local[key], str(value))
                self.assertEqual([Fraction(t["signed_contribution"])
                                  for t in selected["coordinate_weighted_terms_in_operand_order"]], weighted)
                self.assertEqual([Fraction(t["absolute_contribution"])
                                  for t in selected["coordinate_weighted_terms_in_operand_order"]],
                                 list(map(abs, weighted)))
                self.assertEqual(sum(weighted, Fraction()), factor*terms[coordinate])
                self.assertEqual(selected["coordinate_weighted_term_totals"], mass(weighted))
                self.assertEqual(selected["weighted_selected_input_closure_residual"], "0")
                expanded.extend(weighted)
            selected = [factor*terms[i] for i in selected_ids]
            unselected = [factor*t for i, t in enumerate(terms) if i not in selected_ids]
            boundary = factor*(stage17-sum(terms, Fraction()))
            summary = account["summary"]
            self.assertEqual(summary["selected_coordinate_count"], 8)
            self.assertEqual(summary["weighted_selected_total"], mass(selected))
            self.assertEqual(summary["weighted_unselected_total"], mass(unselected))
            self.assertEqual(summary["coordinate_weighted_term_totals"], mass(expanded))
            for i, term in enumerate(summary["coordinate_weighted_operand_terms_in_order"]):
                self.assertEqual(term["totals"], mass(expanded[i::4]))
            self.assertEqual(summary["weighted_projection_boundary_remainder"], str(boundary))
            self.assertEqual(sum(expanded, Fraction())+sum(unselected, Fraction())+boundary, factor*stage17)
            self.assertEqual(summary["retained_coordinate_signed_contribution"], str(factor*stage17))
            self.assertEqual(summary["retained_coordinate_absolute_contribution"], str(abs(factor*stage17)))
            self.assertEqual(summary["term_expansion_cancellation_mass"],
                             str(sum(map(abs, expanded), Fraction())-sum(map(abs, selected), Fraction())))
            self.assertEqual(summary["expanded_input_and_boundary_cancellation_mass"],
                             str(sum(map(abs, expanded+unselected), Fraction())+abs(boundary)-abs(factor*stage17)))
            self.assertEqual(summary["weighted_stage17_coordinate_closure_residual"], "0")
            self.assertTrue(summary["exact_selected_to_stage17_identity"])

    def test_coordinate741_control_branch_representatives(self):
        expected = {
            "fp16": "972712872287995455840195/19342813113834066795298816",
            "binary64": "1955800403415784668441195/38685626227668133590597632",
        }
        for branch, contribution in expected.items():
            row = next(r for r in self.result["rows"]
                       if r["control"] == "frozen_inherited" and r["branch"] == branch)
            self.assertEqual(row["retained_maximum_coordinate_ties"], [741])
            account = row["hotspots"][0]
            self.assertEqual(account["output_coordinate"], 741)
            self.assertEqual(account["summary"]["retained_coordinate_signed_contribution"], contribution)
            self.assertEqual(len(account["selected_stage16_coordinates"]), 8)
        self.assertEqual([(r["control"], r["branch"]) for r in self.result["rows"]],
                         [(c, b) for c in d.hotspot.census.CONTROLS for b in ("fp16", "binary64")])
        self.assertEqual(self.result["row_count"], 18)

    def test_retained_selections_pair_common_and_reference_gates(self):
        self.assertEqual((self.result["left_id"], self.result["right_id"]), (34319, 13))
        self.assertEqual(self.result["retained_down_input_bridge"], self.retained)
        self.assertEqual(self.retained["retained_hotspot_audit"], self.observed)
        self.assertEqual(self.observed["retained_dominance_audit"], self.dom)
        self.assertEqual(self.dom["retained_obstruction_census"], self.c)
        self.assertEqual([p["component"] for p in self.dom["pairs"]], ["UNKNOWN", "mlp_stage17", "UNKNOWN"])
        self.assertEqual(self.result["common_component"], "UNKNOWN")
        self.assertTrue(self.result["stop_nested_bridge_expansion"])
        self.assertTrue(self.result["prior_pair_and_common_gates_unchanged"])
        for row, old in zip(self.result["rows"], self.retained["rows"], strict=True):
            self.assertEqual({k: v for k, v in row.items() if k != "hotspots"},
                             {k: v for k, v in old.items() if k != "hotspots"})
            self.assertEqual(row["binary64_internal_stages"], "NOT_RETAINED_NO_RECONSTRUCTION")
            for account, previous in zip(row["hotspots"], old["hotspots"], strict=True):
                self.assertEqual([c["selected_down_input"] for c in account["selected_stage16_coordinates"]],
                                 [previous["input_coordinates"][i]
                                  for i in previous["input_order"]["selected_coordinates"]])

    def test_tampered_retained_pair_control_branch_hotspot_and_gates(self):
        for mutate in (
            lambda r: r.update(left_id=319),
            lambda r: r["rows"][0].update(control="scratch"),
            lambda r: r["rows"][0].update(branch="binary64"),
            lambda r: r["rows"][0].update(retained_maximum_coordinate_ties=[62]),
            lambda r: r["rows"][0]["hotspots"][0]["input_order"]["selected_coordinates"].reverse(),
            lambda r: r["rows"][0]["hotspots"][0]["input_coordinates"][0].update(weight="1"),
            lambda r: r.update(common_component="mlp_stage17"),
            lambda r: r.update(stop_nested_bridge_expansion=False),
        ):
            changed = deepcopy(self.retained)
            mutate(changed)
            with self.assertRaises(ValueError):
                d.report(*self.inputs[:-1], changed)

    def test_tampered_actual_and_reference_gate_up_stage16_vectors(self):
        control = d.hotspot.census.CONTROLS[0]
        for reference in (False, True):
            original = self.e["reference_archive"] if reference else self.e["archives"][control]
            for stage in ("stage14", "stage15", "stage16", "stage17"):
                archive = {**original, stage: original[stage].copy()}
                archive[stage][0] ^= 1
                changed = ({**self.e, "reference_archive": archive} if reference else
                           {**self.e, "archives": {**self.e["archives"], control: archive}})
                with self.assertRaises(ValueError):
                    d.bind_inputs(changed, self.tensors)

    def test_tampered_tensor_geometry_and_archive_census(self):
        for suffix in self.tensors:
            tensors = {**self.tensors, suffix: self.tensors[suffix].copy()}
            tensors[suffix].flat[0] += 1
            with self.assertRaises(ValueError):
                d.bind_inputs(self.e, tensors)
        control = d.hotspot.census.CONTROLS[0]
        original = self.e["archives"][control]
        for vector in (original["stage14"][:-1], original["stage14"].astype("<i4")):
            changed = {**self.e, "archives": {**self.e["archives"], control: {**original, "stage14": vector}}}
            with self.assertRaises(ValueError):
                d.bind_inputs(changed, self.tensors)
        with self.assertRaises(ValueError):
            d.bind_inputs({**self.e, "archives": {**self.e["archives"], "scratch": original}}, self.tensors)

    def test_tampered_new_local_weighted_closure_and_gate_report(self):
        for mutate in (
            lambda r: r["rows"][0]["hotspots"][0]["selected_stage16_coordinates"][0]["bridge"].update(gate_delta="1"),
            lambda r: r["rows"][0]["hotspots"][0]["selected_stage16_coordinates"][0]
                ["coordinate_weighted_terms_in_operand_order"][0].update(signed_contribution="1"),
            lambda r: r["rows"][0]["hotspots"][0]["summary"].update(weighted_projection_boundary_remainder="1"),
            lambda r: r.update(stop_nested_bridge_expansion=False),
        ):
            changed = deepcopy(self.result)
            mutate(changed)
            with self.assertRaises(ValueError):
                d.validate_report(self.inputs, changed)

    def test_zero_interaction_and_cancellation_boundaries(self):
        reference = dict(zip(("gate", "up", "stage16"), map(Fraction, (2, 3, 5))))
        actual = dict(zip(("gate", "up", "stage16"), map(Fraction, (4, 7, 11))))
        for weight in (Fraction(0), Fraction(-2), Fraction(3, 4)):
            result = d.silu.account(actual, reference, 0, weight)
            self.assertEqual([Fraction(t["signed_contribution"])
                              for t in result["terms_in_operand_order"]],
                             [Fraction(6), Fraction(8), Fraction(8), Fraction(-16)])
            self.assertEqual(result["weighted_stage16_delta"], str(6*weight))
            same = d.silu.account(reference, reference, 4863, weight)
            self.assertTrue(all(Fraction(t["signed_contribution"]) == 0
                                for t in same["terms_in_operand_order"]))
        for coordinate in (-1, 4864, True):
            with self.assertRaises(ValueError):
                d.silu.account(actual, reference, coordinate, Fraction(1))
        with self.assertRaises(ValueError):
            d.silu.account({**actual, "gate": 4.0}, reference, 0, Fraction(1))

    def test_historical_failures_thresholds_and_original_references(self):
        for field, value in (("candidate_admitted", True), ("S18_failure_indices", []),
                             ("source_operand_state_KV_RTZ_checks", "FAIL")):
            result = deepcopy(self.e["result"])
            result["controls"][0]["parent"]["retained_L23"][field] = value
            with self.assertRaises(ValueError):
                d.base.check_history(result)
        result = deepcopy(self.e["result"])
        result["controls"][0]["parent"]["L23_failures"][0]["excess_budget"] = "1"
        with self.assertRaises(ValueError):
            d.base.check_history(result)
        for branch in ("fp16", "binary64"):
            self.assertEqual(self.e["result"]["preflight"]["final_reference"]["reference"]["input_"+branch],
                             self.e["result"]["preflight"]["L23_original_reference"]["reference"][branch])

    def test_source_authentication_and_non_admission_flags(self):
        with self.assertRaises(ValueError):
            d.base.read_bound({**d.parent.record(d.previous.SOURCE), "sha256": "0"*64})
        self.assertTrue(all(not flag for flag in d.FLAGS.values()))
        self.assertIn("common UNKNOWN", d.BOUNDARY)
        self.assertIn("NOT_IDENTIFIABLE_WITHOUT_REPLAY", d.BOUNDARY)

    def test_forbidden_replay_nonlinear_dispatch_and_writes(self):
        target = d.parent.OUTPUT / "forbidden-stage16-silu-product-bridge"
        torch = d.parent.preflight.parent.producer.legacy.torch
        calls = (
            lambda: d.previous.check(), lambda: d.previous.run_tests(None, None, None, None, None, None),
            lambda: d.silu.check(), lambda: d.silu.measure(), lambda: d.silu.report(None, None, None),
            lambda: d.bridge.measure(), lambda: d.bridge.report(None),
            lambda: d.hotspot.check(), lambda: d.down.measure(),
            lambda: d.parent.rmsnorm(None, None), lambda: d.parent.logits(None, None),
            lambda: d.parent.execute("forbidden"), lambda: d.parent.head.decode_array_q24(None),
            lambda: native._stages(None, None, None, None),
            lambda: d.bridge.residual.local.local_reference(None),
            lambda: d.bridge.margin.contributions.dyadic_dot(None, None),
            lambda: math.exp(0), lambda: np.exp(0), lambda: torch.sigmoid(None),
            lambda: torch.nn.functional.silu(None),
            lambda: subprocess.run(["forbidden"]), lambda: os.system("forbidden"),
            lambda: torch.cuda._lazy_init(),
            lambda: target.write_text("forbidden"), lambda: open(target, "wb"),
            lambda: os.open(target, os.O_WRONLY | os.O_CREAT),
            lambda: target.mkdir(), lambda: os.replace(target, target), lambda: target.unlink(),
        )
        audit = {"forbidden_calls": 0}
        with d.read_only(audit):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit["forbidden_calls"], len(calls))

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
