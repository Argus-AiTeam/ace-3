"""Independent integer-grid oracle for every retained unselected down input."""

from copy import deepcopy
from fractions import Fraction
import io
import json
import math
import os
import subprocess
import unittest
from unittest.mock import patch

import jsonschema
import numpy as np

from ace3.model.candidates import diagnose_q24_s16_final_head_l23_mlp_unselected_down_input_silu_product_bridge_v1 as d


EVIDENCE = None


def half_grid(word):
    exponent, mantissa = (word >> 10) & 31, word & 1023
    if exponent == 31:
        raise ValueError("nonfinite half")
    value = mantissa if exponent == 0 else (1024 + mantissa) << (exponent - 1)
    return -value if word & 32768 else value


def mass(signed, absolute, denominator):
    return {
        "signed_sum": str(Fraction(signed, denominator)),
        "absolute_sum": str(Fraction(absolute, denominator)),
        "cancellation_absolute_mass": str(Fraction(absolute - abs(signed), denominator)),
    }


class UnselectedDownInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is None:
            with d.read_only({"forbidden_calls": 0}):
                cls.evidence, _, _ = d.measure()
        else:
            cls.evidence = EVIDENCE
        cls.previous, cls.report = cls.evidence
        cls.prior_projection = cls.previous[0]
        cls.actual, cls.reference = cls.previous[1:3]
        cls.archives, cls.reference_archive, cls.tensors = cls.prior_projection[0][3:6]
        cls.grid = {
            label: [[half_grid(int(word)) for word in archive[stage]]
                    for stage in ("stage14", "stage15", "stage16")]
            for label, archive in (*cls.archives.items(), ("reference", cls.reference_archive))
        }

    def hotspots(self):
        for row in self.report["controls"]:
            for branch in row["branches"]:
                for hotspot in branch["hotspots"]:
                    yield row["control"], hotspot

    def test_01_all_complement_terms_independent_integer_and_native_nibble_oracle(self):
        count = 0
        scale_words = self.tensors["scales"].view("<u2")
        denominator = 1 << 72
        for label, hotspot in self.hotspots():
            split = hotspot["unselected_bridge"]
            output = hotspot["final_head_channel"]["coordinate"]
            lane, packed = (0, 4, 1, 5, 2, 6, 3, 7)[output % 8], output // 8
            excluded = {x["selected_down_input"]["coordinate"] for x in hotspot["stage16_coordinates"]}
            signed, absolute, delta_sum, delta_absolute, ab_sum, rb_sum = [0]*4, [0]*4, 0, 0, 0, 0
            for i in range(4864):
                if i in excluded:
                    continue
                ag, au, ay = (v[i] for v in self.grid[label])
                rg, ru, ry = (v[i] for v in self.grid["reference"])
                q = int(self.tensors["qweight"][i, packed]) % (1 << 32) // 16**lane % 16
                z = int(self.tensors["qzeros"][i // 128, packed]) % (1 << 32) // 16**lane % 16
                w = (q - z) * half_grid(int(scale_words[i // 128, output]))
                # Endpoint products on a common integer grid, not production Fraction arithmetic.
                values = (w*(ag*ru-rg*ru), w*(rg*au-rg*ru),
                          w*(ag*au-ag*ru-rg*au+rg*ru),
                          w*(((ay-ry) << 24)-ag*au+rg*ru))
                delta = w * ((ay-ry) << 24)
                self.assertEqual(sum(values), delta)
                for j, value in enumerate(values):
                    signed[j] += value
                    absolute[j] += abs(value)
                delta_sum += delta
                delta_absolute += abs(delta)
                ab_sum += w*((ay << 24)-ag*au)
                rb_sum += w*((ry << 24)-rg*ru)
                count += 1
            self.assertEqual(split["weighted_terms_in_operand_order"], [
                {"term": name, "totals": mass(s, a, denominator)}
                for name, s, a in zip(d.TERMS, signed, absolute, strict=True)])
            self.assertEqual(split["unselected_down_input_totals"],
                             mass(delta_sum, delta_absolute, denominator))
            self.assertEqual(split["weighted_actual_silu_product_remainder"],
                             str(Fraction(ab_sum, denominator)))
            self.assertEqual(split["weighted_reference_silu_product_remainder"],
                             str(Fraction(rb_sum, denominator)))
        self.assertEqual(count, 699264)

    def test_02_exact_signed_and_absolute_closure_and_cancellation(self):
        for _, hotspot in self.hotspots():
            s = hotspot["unselected_bridge"]
            categories = [t["totals"] for t in s["weighted_terms_in_operand_order"]]
            signed = sum(Fraction(t["signed_sum"]) for t in categories)
            absolute = sum(Fraction(t["absolute_sum"]) for t in categories)
            net_absolute = sum(abs(Fraction(t["signed_sum"])) for t in categories)
            target = s["unselected_down_input_totals"]
            self.assertEqual(str(signed), target["signed_sum"])
            expected = {
                "component_absolute_sum": absolute,
                "within_coordinate_cancellation_mass": absolute-Fraction(target["absolute_sum"]),
                "across_coordinate_cancellation_mass": Fraction(target["absolute_sum"])-abs(signed),
                "within_category_across_coordinate_cancellation_mass": absolute-net_absolute,
                "between_category_net_cancellation_mass": net_absolute-abs(signed),
                "total_component_cancellation_mass": absolute-abs(signed),
            }
            for key, value in expected.items():
                self.assertEqual(s[key], str(value))
                self.assertGreaterEqual(value, 0)
            self.assertEqual(Fraction(s["weighted_actual_silu_product_remainder"])
                             - Fraction(s["weighted_reference_silu_product_remainder"]),
                             Fraction(categories[3]["signed_sum"]))
            self.assertEqual(Fraction(s["selected_down_input_signed_sum"])+signed
                             + Fraction(s["down_projection_boundary_remainder"]),
                             Fraction(s["retained_stage17_delta"]))

    def test_03_selected_accounts_complement_and_boundary_preserved_verbatim(self):
        for row, old in zip(self.report["controls"], self.previous[3]["controls"], strict=True):
            for branch, prior in zip(row["branches"], old["branches"], strict=True):
                self.assertEqual(branch["numeric_id"], prior["numeric_id"])
                for hotspot, original in zip(branch["hotspots"], prior["hotspots"], strict=True):
                    self.assertEqual({k: v for k, v in hotspot.items() if k != "unselected_bridge"}, original)
                    s, summary = hotspot["unselected_bridge"], original["summary"]
                    self.assertEqual(s["excluded_selected_coordinates"], sorted(
                        item["selected_down_input"]["coordinate"] for item in original["stage16_coordinates"]))
                    self.assertEqual(s["coordinate_count"], 4856)
                    for key in ("down_projection_boundary_remainder", "retained_stage17_delta"):
                        self.assertEqual(s[key], summary[key])
                    for field, target in (("signed_sum", "signed"), ("absolute_sum", "absolute")):
                        self.assertEqual(s["unselected_down_input_totals"][field],
                                         summary[f"unselected_down_input_{target}_remainder"])

    def test_04_original_fp16_reference_and_failure_history_unchanged(self):
        self.assertEqual(self.report["silu_product_bridge_report"], self.previous[3])
        self.assertEqual(self.report["L23_reference"], self.previous[3]["L23_reference"])
        d.base.check_history(self.prior_projection[0][0][0])
        for row in self.report["controls"]:
            self.assertEqual([b["final_head_reference_branch"] for b in row["branches"]],
                             ["fp16", "binary64"])
            for branch in row["branches"]:
                for hotspot in branch["hotspots"]:
                    self.assertEqual(hotspot["unselected_bridge"]["reference"], "original_input_L23_fp16")

    def test_05_all_nine_authenticated_projection_tensor_bindings(self):
        canonical = self.prior_projection[0][0][0]["preflight"]["L23_original_reference"]["reference"]["canonical"]
        for suffix, tensor in self.tensors.items():
            pin = canonical[d.down.PREFIX + suffix]
            self.assertIs(d.down.bind_tensor(tensor, pin, suffix), tensor)
            with self.assertRaises(ValueError):
                d.down.bind_tensor(tensor, {**pin, "sha256": "0"*64}, suffix)
        for kind in ("gate_proj", "up_proj"):
            for suffix, tensor in self.prior_projection[3][kind].items():
                pin = canonical[d.bridge.gate_up.PREFIXES[kind] + suffix]
                self.assertIs(d.bridge.gate_up.bind_tensor(tensor, pin, kind, suffix), tensor)
                with self.assertRaises(ValueError):
                    d.bridge.gate_up.bind_tensor(tensor, {**pin, "sha256": "0"*64}, kind, suffix)

    def test_06_strict_split_schema_scope_and_order(self):
        split = next(self.hotspots())[1]["unselected_bridge"]
        validator = jsonschema.Draft202012Validator(d.SPLIT_SCHEMA)
        validator.validate(split)
        for key, value in (("coordinate_count", 4864), ("reference", "binary64"),
                           ("baseline", "silu_output"), ("component_absolute_sum", "NaN"),
                           ("exact_stage17_identity", False), ("extra", 0),
                           ("weighted_terms_in_operand_order", list(reversed(split["weighted_terms_in_operand_order"]))),
                           ("excluded_selected_coordinates", [0]*8)):
            with self.assertRaises(jsonschema.ValidationError):
                validator.validate({**split, key: value})
        for key in ("unselected_coordinate_account_count", "weighted_term_count", "layer", "position"):
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate(-1, d.REPORT_SCHEMA["properties"][key])

    def synthetic(self):
        values = dict.fromkeys(d.bridge.OPERAND_NAMES, Fraction(0))
        actual = {key: [value]*4864 for key, value in values.items()}
        reference = {key: list(vector) for key, vector in actual.items()}
        coordinates = [{"selected_down_input": {
            "coordinate": i, "weight": "1", "input_delta": "0", "signed_contribution": "0"}}
            for i in range(8)]
        hotspot = {"stage16_coordinates": coordinates, "summary": {
            "selected_weighted_down_totals": mass(0, 0, 1),
            "unselected_down_input_signed_remainder": "0",
            "unselected_down_input_absolute_remainder": "0",
            "down_projection_boundary_remainder": "7", "retained_stage17_delta": "7"}}
        return actual, reference, [Fraction(1)]*4864, hotspot

    def test_07_zero_subnormal_signed_weight_and_boundary_cancellation(self):
        a, r, w, h = self.synthetic()
        a["gate"][8], a["up"][8] = Fraction(1, 1 << 24), Fraction(-1, 1 << 24)
        r["gate"][8], r["up"][8] = -a["gate"][8], -a["up"][8]
        w[8] = Fraction(-3, 2)
        result = d.split_unselected(a, r, w, h)
        self.assertEqual(result["unselected_down_input_totals"], mass(0, 0, 1))
        self.assertGreater(Fraction(result["component_absolute_sum"]), 0)
        self.assertEqual(result["component_absolute_sum"], result["total_component_cancellation_mass"])
        self.assertEqual(result, d.split_unselected(a, r, w, h))
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)

    def test_08_invalid_operand_geometry_and_selection_rejected(self):
        a, r, w, h = self.synthetic()
        for bad in ({}, {**a, "gate": [Fraction(0)]*4863}, {**a, "up": [0.0]*4864}):
            with self.assertRaises(ValueError):
                d.split_unselected(bad, r, w, h)
        for bad_weights in (w[:-1], [1.0]*4864):
            with self.assertRaises(ValueError):
                d.split_unselected(a, r, bad_weights, h)
        for coordinate in (-1, 4864, True, 0.5, 1):
            bad = deepcopy(h)
            bad["stage16_coordinates"][0]["selected_down_input"]["coordinate"] = coordinate
            with self.assertRaises(ValueError):
                d.split_unselected(a, r, w, bad)

    def test_09_remainder_selected_weight_and_stage17_splices_rejected(self):
        a, r, w, h = self.synthetic()
        for field in ("unselected_down_input_signed_remainder", "unselected_down_input_absolute_remainder",
                      "down_projection_boundary_remainder", "retained_stage17_delta"):
            bad = deepcopy(h)
            bad["summary"][field] = "123"
            with self.assertRaises(ValueError):
                d.split_unselected(a, r, w, bad)
        for field in ("weight", "input_delta", "signed_contribution"):
            bad = deepcopy(h)
            bad["stage16_coordinates"][0]["selected_down_input"][field] = "123"
            with self.assertRaises(ValueError):
                d.split_unselected(a, r, w, bad)

    def test_10_archive_nonfinite_state_kv_and_lineage_gates(self):
        archive = self.archives[d.parent.CONTROLS[0]]
        for stage in ("stage14", "stage15", "stage16"):
            for value in (np.zeros(4863, dtype="<u2"), np.zeros(4864, dtype="<f2"),
                          np.full(4864, 0x7c00, dtype="<u2")):
                with self.assertRaises(ValueError):
                    d.bridge.operands({**archive, stage: value})
        bad = archive["output_cache_v"].copy()
        bad.flat[0] ^= 1
        for key, value in (("output_cache_v", bad), ("input_i", np.zeros(896, dtype="<i4"))):
            with self.assertRaises(ValueError):
                d.residual.operands({**archive, key: value}, actual=True)
        retained = deepcopy(self.prior_projection[0][0][0])
        retained["preflight"]["L23_original_reference"]["reference"]["prior_kv"] = "substituted"
        with self.assertRaises(ValueError):
            d.residual.load_residuals(retained)

    def test_11_forbidden_native_reference_operator_and_previous_check_dispatch(self):
        audit = {"forbidden_calls": 0}
        calls = [
            lambda: d.residual.native._stages(None, None, None, None),
            lambda: d.residual.native.projection(None, None, None),
            lambda: d.residual.native.toward_zero(None),
            lambda: d.residual.local.local_reference(None, None, None, None),
            lambda: d.parent.execute(None), lambda: d.parent.rmsnorm(None, None),
            lambda: d.parent.logits(None, None), lambda: d.bridge.check(),
            lambda: d.bridge.focused_tests(None), lambda: d.bridge.gate_up.check(),
            lambda: d.down.check(), lambda: d.residual.measure(), lambda: d.channels.check(),
            lambda: math.exp(1), lambda: np.exp(1), lambda: subprocess.Popen(["false"]),
            lambda: os.system("false"),
        ]
        with d.read_only(audit):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit["forbidden_calls"], len(calls))

    def test_12_forbidden_writes_do_not_touch_evidence(self):
        audit = {"forbidden_calls": 0}
        calls = [
            lambda: d.parent.write(None, None), lambda: open(d.SOURCE, "w"),
            lambda: io.open(d.SOURCE, "a"), lambda: d.SOURCE.write_bytes(b"forbidden"),
            lambda: os.open(d.SOURCE, os.O_WRONLY), lambda: os.unlink(d.SOURCE),
            lambda: os.rename(d.SOURCE, d.TEST), lambda: os.mkdir(d.SOURCE),
        ]
        with d.read_only(audit):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit["forbidden_calls"], len(calls))

    def test_13_stdout_json_only_and_no_execution_or_output_options(self):
        stream = io.StringIO()
        with patch.object(d, "check", return_value={"ok": True}), patch("sys.stdout", stream):
            d.main(["--check"])
        self.assertEqual(stream.getvalue(), '{"ok": true}\n')
        for arguments in ([], ["--execute"], ["--check", "--output", "forbidden"],
                          ["--check", "--reference", "substituted"], ["--che"]):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                d.main(arguments)
        for key in ("native_dispatch", "prefix_dispatch", "admission_dispatch", "evidence_writes",
                    "earlier_checks", "silu_replay", "reference_replay", "product_operator_replay"):
            self.assertEqual(d.FLAGS[key], 0)

    def test_14_exact_command_and_account_source_gates(self):
        self.assertEqual(d.COMMAND,
                         "PYTHONPATH=/home/argustest/ace3-argus PYTHONDONTWRITEBYTECODE=1 "
                         "/home/argustest/miniconda3/bin/python -B -m ace3.model.candidates."
                         "diagnose_q24_s16_final_head_l23_mlp_unselected_down_input_silu_product_bridge_v1 --check")
        with patch.object(os, "getuid", return_value=0), self.assertRaises(ValueError):
            d.check()
        with patch.object(d, "SOURCE", d.TEST), self.assertRaises(ValueError):
            d.check()


if __name__ == "__main__":
    unittest.main()
