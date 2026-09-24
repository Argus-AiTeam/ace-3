"""Independent half-bit/native-nibble oracles for retained gate/up accounts."""

from copy import deepcopy
from fractions import Fraction
import io
import json
import os
import subprocess
import unittest
from unittest.mock import patch

import jsonschema
import numpy as np

from ace3.model.candidates import diagnose_q24_s16_final_head_l23_mlp_gate_up_projection_input_coordinates_v1 as d


EVIDENCE = None


def half(word):
    exponent, mantissa = (word >> 10) & 31, word & 1023
    if exponent == 31:
        raise ValueError("nonfinite FP16")
    magnitude = (Fraction(mantissa, 16777216) if exponent == 0
                 else Fraction(1024 + mantissa) * Fraction(2) ** (exponent - 25))
    return -magnitude if word & 32768 else magnitude


def oracle_weights(tensors, output):
    lane = (0, 4, 1, 5, 2, 6, 3, 7)[output % 8]
    column = output // 8
    bits = tensors["scales"].view("<u2")
    result = []
    for i in range(896):
        q = (int(tensors["qweight"][i, column]) % (1 << 32)) // (16 ** lane) % 16
        z = (int(tensors["qzeros"][i // 128, column]) % (1 << 32)) // (16 ** lane) % 16
        result.append((q - z) * half(int(bits[i // 128, output])))
    return result


class GateUpInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is None:
            with d.read_only({"forbidden_calls": 0}):
                cls.evidence, _, _ = d.measure()
        else:
            cls.evidence = EVIDENCE
        cls.previous, cls.actual, cls.reference, cls.tensors, cls.report = cls.evidence
        cls.archives, cls.reference_archive = cls.previous[3:5]
        cls.binding = cls.previous[0][0]["preflight"]["L23_original_reference"]
        cls.oracles = {}
        columns = {}
        for row in cls.report["controls"]:
            label = row["control"]
            delta = [half(int(a)) - half(int(r)) for a, r in zip(
                cls.archives[label]["stage13"], cls.reference_archive["stage13"], strict=True)]
            for branch in row["branches"]:
                for hotspot in branch["hotspots"]:
                    for selected in hotspot["stage16_coordinates"]:
                        for kind in ("gate_proj", "up_proj"):
                            output = selected[kind]["output_coordinate"]
                            if (kind, output) not in columns:
                                columns[kind, output] = oracle_weights(cls.tensors[kind], output)
                            key = (label, kind, output)
                            if key not in cls.oracles:
                                weights = columns[kind, output]
                                cls.oracles[key] = (delta, weights, [
                                    a * w for a, w in zip(delta, weights, strict=True)])

    def accounts(self):
        for row in self.report["controls"]:
            for branch in row["branches"]:
                for hotspot in branch["hotspots"]:
                    for selected in hotspot["stage16_coordinates"]:
                        for kind in ("gate_proj", "up_proj"):
                            yield row["control"], kind, selected[kind]

    def test_01_all_terms_and_ranked_coordinates(self):
        count = 0
        for label, kind, account in self.accounts():
            delta, weights, terms = self.oracles[label, kind, account["output_coordinate"]]
            top = sorted(range(896), key=lambda i: (-abs(terms[i]), i))[:8]
            self.assertEqual(account["largest_absolute_coordinates"], [
                {"coordinate": i, "input_group": i // 128, "input_delta": str(delta[i]),
                 "weight": str(weights[i]), "signed_contribution": str(terms[i])} for i in top])
            self.assertEqual(Fraction(account["exact_input_delta"]), sum(terms))
            self.assertEqual(Fraction(account["sum_absolute_contributions"]), sum(map(abs, terms)))
            count += len(terms)
        self.assertEqual(count, 2064384)

    def test_02_all_groups_and_signed_absolute_remainders(self):
        groups = 0
        for label, kind, account in self.accounts():
            terms = self.oracles[label, kind, account["output_coordinate"]][2]
            for group, entry in enumerate(account["groups_in_input_order"]):
                chunk = terms[group * 128:(group + 1) * 128]
                self.assertEqual(entry, {
                    "input_group": group, "start_coordinate": group * 128,
                    "end_coordinate_exclusive": (group + 1) * 128, "coordinate_count": 128,
                    "signed_contribution": str(sum(chunk)),
                    "sum_absolute_contributions": str(sum(map(abs, chunk))),
                })
                groups += 1
            top = sorted(range(896), key=lambda i: (-abs(terms[i]), i))[:8]
            rest = [term for i, term in enumerate(terms) if i not in top]
            self.assertEqual(Fraction(account["ranked_signed_sum"]), sum(terms[i] for i in top))
            self.assertEqual(Fraction(account["ranked_absolute_sum"]), sum(abs(terms[i]) for i in top))
            self.assertEqual(Fraction(account["unranked_signed_remainder"]), sum(rest))
            self.assertEqual(Fraction(account["unranked_absolute_remainder"]), sum(map(abs, rest)))
        self.assertEqual(groups, 16128)

    def test_03_retained_projection_boundaries_from_bits(self):
        for label, kind, account in self.accounts():
            stage = {"gate_proj": "stage14", "up_proj": "stage15"}[kind]
            output = account["output_coordinate"]
            a = half(int(self.archives[label][stage][output]))
            r = half(int(self.reference_archive[stage][output]))
            exact = sum(self.oracles[label, kind, output][2])
            self.assertEqual(account["input_stage"], "stage13_fp16")
            self.assertEqual(account["output_stage"], stage + "_fp16")
            self.assertEqual(account["actual_projection_output"], str(a))
            self.assertEqual(account["reference_projection_output"], str(r))
            self.assertEqual(account["retained_projection_delta"], str(a - r))
            self.assertEqual(account["projection_boundary_remainder"], str(a - r - exact))
            self.assertEqual(exact + Fraction(account["projection_boundary_remainder"]), a - r)

    def test_04_exact_selection_and_logical_census(self):
        selections = accounts = ranked = 0
        for row, original in zip(self.report["controls"], self.previous[6]["controls"], strict=True):
            self.assertEqual(row["control"], original["control"])
            for branch, old in zip(row["branches"], original["branches"], strict=True):
                self.assertEqual(branch["numeric_id"], old["numeric_id"])
                self.assertEqual(branch["final_head_reference_branch"], old["final_head_reference_branch"])
                for hotspot, prior in zip(branch["hotspots"], old["hotspots"], strict=True):
                    self.assertEqual(hotspot["hotspot_rank"], prior["hotspot_rank"])
                    self.assertEqual(hotspot["final_head_channel"], prior["final_head_channel"])
                    selected = hotspot["stage16_coordinates"]
                    self.assertEqual([s["stage16_coordinate_rank"] for s in selected], list(range(1, 9)))
                    self.assertEqual([s["selected_down_input"] for s in selected],
                                     prior["down_projection"]["largest_absolute_coordinates"])
                    for item in selected:
                        selections += 1
                        for kind in ("gate_proj", "up_proj"):
                            self.assertEqual(item[kind]["projection"], kind)
                            self.assertEqual(item[kind]["output_coordinate"],
                                             item["selected_down_input"]["coordinate"])
                            accounts += 1
                            ranked += len(item[kind]["largest_absolute_coordinates"])
        self.assertEqual((selections, accounts, ranked), (1152, 2304, 18432))

    def test_05_original_references_and_history_unchanged(self):
        self.assertEqual(self.report["down_projection_input_coordinate_report"], self.previous[6])
        self.assertEqual(self.report["L23_reference"], self.previous[6]["L23_reference"])
        self.assertEqual(self.report["L23_reference"]["fp16"], self.binding["reference"]["fp16"])
        self.assertFalse(self.report["L23_reference"]["original_global_references_reanchored"])
        self.assertEqual(self.report["L23_reference"]["binary64_internal_stages"],
                         "NOT_RETAINED_NO_RECONSTRUCTION")
        d.base.check_history(self.previous[0][0])

    def test_06_schema_json_and_deterministic_account(self):
        jsonschema.Draft202012Validator.check_schema(d.OUTPUT_SCHEMA)
        jsonschema.Draft202012Validator(d.REPORT_SCHEMA).validate(self.report)
        text = json.dumps(self.report, sort_keys=True, allow_nan=False)
        self.assertEqual(json.dumps(json.loads(text), sort_keys=True, allow_nan=False), text)
        label, kind, account = next(self.accounts())
        output = account["output_coordinate"]
        self.assertEqual(d.account(self.actual[label], self.reference,
                                   d.weight_column(self.tensors[kind], output), kind, output), account)

    def test_07_schema_rejects_counts_scope_order_and_extras(self):
        validator = jsonschema.Draft202012Validator(d.REPORT_SCHEMA)
        for key, value in (("projection_account_count", 2303), ("input_coordinate_term_count", 2064383),
                           ("selected_stage16_coordinate_count", 1151), ("input_group_count", 16127),
                           ("selected_input_coordinate_count", 18431), ("layer", 22), ("position", 1),
                           ("extra", 0), ("controls", list(reversed(self.report["controls"])))):
            with self.assertRaises(jsonschema.ValidationError):
                validator.validate({**self.report, key: value})
        sample = next(self.accounts())[2]
        validator = jsonschema.Draft202012Validator(d.ACCOUNT_SCHEMA)
        for key, value in (("output_stage", "stage15_fp16"), ("output_coordinate", 4864),
                           ("reference", "binary64"), ("input_stage", "stage12_fp16"),
                           ("exact_input_delta", "NaN"), ("exact_local_identity", False),
                           ("groups_in_input_order", list(reversed(sample["groups_in_input_order"]))),
                           ("largest_absolute_coordinates", sample["largest_absolute_coordinates"][:-1])):
            with self.assertRaises(jsonschema.ValidationError):
                validator.validate({**sample, key: value})

    def test_08_canonical_tensor_names_geometry_and_binding(self):
        canonical = self.binding["reference"]["canonical"]
        count = 0
        expected = {"qweight": ((896, 608), "<i4"), "qzeros": ((7, 608), "<i4"),
                    "scales": ((7, 4864), "<f2")}
        self.assertEqual(d.SPECS, expected)
        for kind in ("gate_proj", "up_proj"):
            prefix = "model.layers.23.mlp." + kind + "."
            self.assertEqual(d.tensor_names(canonical, kind), [prefix + s for s in expected])
            for suffix, (shape, dtype) in expected.items():
                tensor, pin = self.tensors[kind][suffix], canonical[prefix + suffix]
                self.assertEqual((tensor.shape, tensor.dtype.str), (shape, dtype))
                self.assertEqual(self.report["gate_up_projection_tensor_bindings"][prefix + suffix], pin)
                self.assertIs(d.bind_tensor(tensor, pin, kind, suffix), tensor)
                for field, value in (("sha256", "0" * 64), ("shape", []), ("dtype", "float32"),
                                     ("name", "model.layers.22.mlp." + kind + "." + suffix)):
                    with self.assertRaises(ValueError):
                        d.bind_tensor(tensor, {**pin, field: value}, kind, suffix)
                count += 1
        self.assertEqual(count, 6)

    def test_09_no_bias_missing_tensor_and_cross_projection_rejected(self):
        canonical = self.binding["reference"]["canonical"]
        for kind in ("gate_proj", "up_proj"):
            with self.assertRaises(ValueError):
                d.tensor_names({**canonical, d.PREFIXES[kind] + "bias": {}}, kind)
            with self.assertRaises(ValueError):
                d.tensor_names({k: v for k, v in canonical.items()
                                if k != d.PREFIXES[kind] + "qzeros"}, kind)
        with self.assertRaises(ValueError):
            d.bind_tensor(self.tensors["gate_proj"]["scales"],
                          canonical[d.PREFIXES["up_proj"] + "scales"], "gate_proj", "scales")
        for suffix in ("bias", "unknown"):
            with self.assertRaises(ValueError):
                d.bind_tensor(None, {}, "gate_proj", suffix)

    def test_10_native_nibbles_signed_words_and_group_edges(self):
        tensors = {"qweight": np.full((896, 608), -19088744, dtype="<i4"),
                   "qzeros": np.full((7, 608), 0x76543210, dtype="<i4"),
                   "scales": np.ones((7, 4864), dtype="<f2")}
        tensors["scales"][1::2] = np.float16(-0.5)
        for coordinate in (*range(8), 62, 4863):
            self.assertEqual(d.weight_column(tensors, coordinate), oracle_weights(tensors, coordinate))
        self.assertNotEqual(d.weight_column(tensors, 0)[127], d.weight_column(tensors, 0)[128])

    def test_11_half_bits_shapes_nonfinite_and_tensor_tampering(self):
        bits = [0, 0x8000, 1, 0x8001, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x7bff, 0xfbff]
        self.assertEqual(d.residual.words(np.array(bits, dtype="<u2"), (len(bits),)),
                         [half(word) for word in bits])
        archive = self.archives[d.parent.CONTROLS[0]]
        for stage, width in (("stage13", 896), ("stage14", 4864), ("stage15", 4864)):
            for bad in (np.zeros(width - 1, dtype="<u2"), np.zeros(width, dtype="<f2"),
                        np.full(width, 0x7c00, dtype="<u2"), np.full(width, 0xfe00, dtype="<u2")):
                with self.assertRaises(ValueError):
                    d.operands({**archive, stage: bad})
        pin = self.binding["reference"]["canonical"][d.PREFIXES["gate_proj"] + "scales"]
        for value in (np.float16("inf"), np.float16("nan"), np.float16(1)):
            bad = self.tensors["gate_proj"]["scales"].copy()
            bad.flat[0] = value
            with self.assertRaises(ValueError):
                d.bind_tensor(bad, pin, "gate_proj", "scales")

    def test_12_invalid_account_operands(self):
        label, kind, sample = next(self.accounts())
        output = sample["output_coordinate"]
        weights = d.weight_column(self.tensors[kind], output)
        for coordinate in (-1, 4864, True, 1.5):
            with self.assertRaises(ValueError):
                d.weight_column(self.tensors[kind], coordinate)
            with self.assertRaises(ValueError):
                d.account(self.actual[label], self.reference, weights, kind, coordinate)
        with self.assertRaises(ValueError):
            d.account(self.actual[label], self.reference, weights[:-1], kind, output)
        with self.assertRaises(ValueError):
            d.account(self.actual[label], self.reference, weights, "down_proj", output)
        with self.assertRaises(ValueError):
            d.account({**self.actual[label], "stage13": [0.0] * 896},
                      self.reference, weights, kind, output)

    def test_13_ties_cancellation_and_distinct_boundary_remainders(self):
        reference = {"stage13": [Fraction()] * 896, "stage14": [Fraction()] * 4864,
                     "stage15": [Fraction()] * 4864}
        actual = {"stage13": [Fraction((-1) ** i) for i in range(896)],
                  "stage14": [Fraction(3, 2)] * 4864, "stage15": [Fraction(-5, 2)] * 4864}
        for kind, remainder in (("gate_proj", "3/2"), ("up_proj", "-5/2")):
            account = d.account(actual, reference, [Fraction(1)] * 896, kind, 4863)
            self.assertEqual([e["coordinate"] for e in account["largest_absolute_coordinates"]],
                             list(range(8)))
            self.assertEqual(account["exact_input_delta"], "0")
            self.assertEqual(account["sum_absolute_contributions"], "896")
            self.assertEqual(account["unranked_signed_remainder"], "0")
            self.assertEqual(account["unranked_absolute_remainder"], "888")
            self.assertEqual(account["projection_boundary_remainder"], remainder)

    def test_14_state_kv_reference_and_selection_gates(self):
        bad = deepcopy(self.previous[0][0])
        bad["preflight"]["L23_original_reference"]["reference"]["prior_kv"] = "substituted"
        with self.assertRaises(ValueError):
            d.residual.load_residuals(bad)
        bad = deepcopy(self.previous[0][0])
        bad["preflight"]["L23_original_reference"]["reference"]["fp16"] = {}
        with self.assertRaises(ValueError):
            d.residual.load_residuals(bad)
        archive = self.archives[d.parent.CONTROLS[0]]
        for key, value in (("input_i", np.zeros(896, dtype="<i4")),
                           ("scratch_z", np.full(896, 2, dtype="u1")),
                           ("input_cache_k", np.zeros((1, 128), dtype="<u2"))):
            with self.assertRaises(ValueError):
                d.residual.operands({**archive, key: value}, actual=True)
        bad_kv = archive["output_cache_v"].copy()
        bad_kv.flat[0] ^= 1
        with self.assertRaises(ValueError):
            d.residual.operands({**archive, "output_cache_v": bad_kv}, actual=True)
        for duplicate in (False, True):
            bad = deepcopy(self.previous[6])
            selected = bad["controls"][0]["branches"][0]["hotspots"][0]["down_projection"][
                "largest_absolute_coordinates"]
            if duplicate:
                selected[1] = selected[0]
            else:
                selected.reverse()
            with self.assertRaises(ValueError):
                d.report(bad, self.actual, self.reference, self.tensors, self.binding)

    def test_15_forbidden_dispatch_and_previous_checks(self):
        audit = {"forbidden_calls": 0}
        calls = [
            lambda: d.residual.native.projection(None, None, None),
            lambda: d.residual.native._stages(None, None, None, None),
            lambda: d.residual.native.toward_zero(None),
            lambda: d.residual.local.local_reference(None, None, None, None),
            lambda: d.parent.execute(None), lambda: d.parent.rmsnorm(None, None),
            lambda: d.parent.logits(None, None), lambda: d.parent.preflight.parent.execute(None),
            lambda: d.down.check(), lambda: d.down.focused_tests(None),
            lambda: d.residual.check(), lambda: d.residual.measure(),
            lambda: d.channels.check(), lambda: d.base.check(),
            lambda: subprocess.Popen(["false"]), lambda: os.system("false"),
        ]
        with d.read_only(audit):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit["forbidden_calls"], len(calls))

    def test_16_forbidden_writes(self):
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
        for key in ("native_dispatch", "prefix_dispatch", "admission_dispatch",
                    "evidence_writes", "earlier_checks", "local_operator_replay",
                    "mlp_down_projection_replay", "mlp_gate_up_projection_replay",
                    "silu_replay", "rmsnorm_replay"):
            self.assertEqual(d.FLAGS[key], 0)

    def test_17_stdout_only_cli_and_no_execution_output_flags(self):
        stream = io.StringIO()
        with patch.object(d, "check", return_value={"ok": True}), patch("sys.stdout", stream):
            d.main(["--check"])
        self.assertEqual(stream.getvalue(), '{"ok": true}\n')
        for argv in ([], ["--execute"], ["--check", "--execute"],
                     ["--check", "--output", "x"], ["--check", "--che"]):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                d.main(argv)

    def test_18_identity_and_evidence_fail_closed(self):
        with patch.object(d.os, "getuid", return_value=1), patch.object(d, "measure") as measure:
            with self.assertRaises(ValueError):
                d.check()
            measure.assert_not_called()
        for pin in (d.base.PINS["result"], d.channels.PINS["contribution_source"],
                    self.binding["reference"]["fp16"]):
            with self.assertRaises(ValueError):
                d.base.read_bound({**pin, "sha256": "0" * 64})


if __name__ == "__main__":
    unittest.main()
