"""Independent integer/half-bit oracles for all retained down-input accounts."""

from copy import deepcopy
from fractions import Fraction
import hashlib
import io
import json
import os
import subprocess
import unittest
from unittest.mock import patch

import jsonschema
import numpy as np

from ace3.model.candidates import diagnose_q24_s16_final_head_l23_mlp_down_projection_input_coordinates_v1 as d


EVIDENCE = None


def half(word):
    exponent, mantissa = (word >> 10) & 31, word & 1023
    if exponent == 31:
        raise ValueError("nonfinite FP16")
    magnitude = (Fraction(mantissa, 16777216) if exponent == 0
                 else Fraction(1024 + mantissa) * Fraction(2) ** (exponent - 25))
    return -magnitude if word & 32768 else magnitude


def oracle_weights(tensors, output):
    # Unpack physical nibbles to logical GEMM order independently of SHIFTS.
    lane = (0, 4, 1, 5, 2, 6, 3, 7)[output % 8]
    column = output // 8
    scale_bits = tensors["scales"].view("<u2")
    values = []
    for i in range(4864):
        q = (int(tensors["qweight"][i, column]) % (1 << 32)) // (16 ** lane) % 16
        z = (int(tensors["qzeros"][i // 128, column]) % (1 << 32)) // (16 ** lane) % 16
        values.append((q - z) * half(int(scale_bits[i // 128, output])))
    return values


class DownInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is None:
            with d.read_only({"forbidden_calls": 0}):
                cls.evidence, _, _ = d.measure()
        else:
            cls.evidence = EVIDENCE
        (cls.previous, cls.actual, cls.reference, cls.archives,
         cls.reference_archive, cls.tensors, cls.report) = cls.evidence
        cls.binding = cls.previous[0]["preflight"]["L23_original_reference"]
        cls.oracles = {}
        columns = {}
        for row in cls.report["controls"]:
            label = row["control"]
            delta = [half(int(a)) - half(int(r)) for a, r in zip(
                cls.archives[label]["stage16"], cls.reference_archive["stage16"], strict=True)]
            for branch in row["branches"]:
                for hotspot in branch["hotspots"]:
                    output = hotspot["down_projection"]["output_coordinate"]
                    if output not in columns:
                        columns[output] = oracle_weights(cls.tensors, output)
                    weights = columns[output]
                    cls.oracles[label, output] = (delta, weights, [
                        a * w for a, w in zip(delta, weights, strict=True)])

    def accounts(self):
        for row in self.report["controls"]:
            for branch in row["branches"]:
                for hotspot in branch["hotspots"]:
                    yield row["control"], hotspot["down_projection"]

    def test_01_all_terms_and_top_coordinates(self):
        count = 0
        for label, account in self.accounts():
            delta, weights, terms = self.oracles[label, account["output_coordinate"]]
            top = sorted(range(4864), key=lambda i: (-abs(terms[i]), i))[:8]
            self.assertEqual(account["largest_absolute_coordinates"], [
                {"coordinate": i, "input_group": i // 128, "input_delta": str(delta[i]),
                 "weight": str(weights[i]), "signed_contribution": str(terms[i])} for i in top])
            self.assertEqual(Fraction(account["exact_input_delta"]), sum(terms))
            self.assertEqual(Fraction(account["sum_absolute_contributions"]), sum(map(abs, terms)))
            count += len(terms)
        self.assertEqual(count, 700416)

    def test_02_all_groups_and_remainders(self):
        count = 0
        for label, account in self.accounts():
            terms = self.oracles[label, account["output_coordinate"]][2]
            groups = account["groups_in_input_order"]
            for group, entry in enumerate(groups):
                chunk = terms[group * 128:(group + 1) * 128]
                self.assertEqual(entry, {
                    "input_group": group, "start_coordinate": group * 128,
                    "end_coordinate_exclusive": (group + 1) * 128, "coordinate_count": 128,
                    "signed_contribution": str(sum(chunk)),
                    "sum_absolute_contributions": str(sum(map(abs, chunk))),
                })
                count += 1
            top = sorted(range(4864), key=lambda i: (-abs(terms[i]), i))[:8]
            rest = [value for i, value in enumerate(terms) if i not in top]
            self.assertEqual(Fraction(account["unranked_signed_remainder"]), sum(rest))
            self.assertEqual(Fraction(account["unranked_absolute_remainder"]), sum(map(abs, rest)))
            self.assertEqual(Fraction(account["ranked_signed_sum"]), sum(terms[i] for i in top))
            self.assertEqual(Fraction(account["ranked_absolute_sum"]), sum(abs(terms[i]) for i in top))
        self.assertEqual(count, 5472)

    def test_03_boundary_remainders_independent_bits(self):
        for label, account in self.accounts():
            output = account["output_coordinate"]
            a = half(int(self.archives[label]["stage17"][output]))
            r = half(int(self.reference_archive["stage17"][output]))
            exact = sum(self.oracles[label, output][2])
            self.assertEqual(account["actual_stage17"], str(a))
            self.assertEqual(account["reference_stage17"], str(r))
            self.assertEqual(account["retained_projection_delta"], str(a - r))
            self.assertEqual(account["down_projection_boundary_remainder"], str(a - r - exact))
            self.assertEqual(exact + Fraction(account["down_projection_boundary_remainder"]), a - r)

    def test_04_census_and_retained_hotspot_selection(self):
        accounts = 0
        for row, previous in zip(self.report["controls"], self.previous[5]["controls"], strict=True):
            self.assertEqual(row["control"], previous["control"])
            self.assertEqual([b["final_head_reference_branch"] for b in row["branches"]],
                             list(d.channels.BRANCHES))
            for branch in row["branches"]:
                original = previous["branches"][branch["final_head_reference_branch"]]
                self.assertEqual(branch["numeric_id"], original["numeric_id"])
                self.assertEqual([h["hotspot_rank"] for h in branch["hotspots"]], list(range(1, 9)))
                self.assertEqual([h["final_head_channel"] for h in branch["hotspots"]],
                                 original["channels"]["full_absolute_signed_ranking"][:8])
                for hotspot in branch["hotspots"]:
                    self.assertEqual(hotspot["final_head_channel"]["coordinate"],
                                     hotspot["down_projection"]["output_coordinate"])
                    accounts += 1
        self.assertEqual(accounts, 144)
        self.assertEqual(self.report["final_head_channel_report"], self.previous[5])
        self.assertEqual(self.report["L23_reference"]["fp16"], self.binding["reference"]["fp16"])
        self.assertFalse(self.report["L23_reference"]["original_global_references_reanchored"])

    def test_05_schema_json_and_deterministic_reconstruction(self):
        jsonschema.Draft202012Validator.check_schema(d.OUTPUT_SCHEMA)
        jsonschema.Draft202012Validator(d.REPORT_SCHEMA).validate(self.report)
        rebuilt = d.report(self.previous[5], self.actual, self.reference, self.tensors, self.binding)
        text = json.dumps(self.report, sort_keys=True, allow_nan=False)
        self.assertEqual(text, json.dumps(rebuilt, sort_keys=True, allow_nan=False))
        self.assertEqual(json.loads(text), self.report)

    def test_06_schema_rejects_count_order_scope_and_extra(self):
        validator = jsonschema.Draft202012Validator(d.REPORT_SCHEMA)
        for field, value in (("projection_account_count", 143), ("input_coordinate_term_count", 700415),
                             ("input_group_count", 5471), ("layer", 22), ("position", 1),
                             ("extra", 0), ("controls", list(reversed(self.report["controls"])))):
            with self.assertRaises(jsonschema.ValidationError):
                validator.validate({**self.report, field: value})
        sample = next(self.accounts())[1]
        account_validator = jsonschema.Draft202012Validator(d.ACCOUNT_SCHEMA)
        for field, value in (("groups_in_input_order", list(reversed(sample["groups_in_input_order"]))),
                             ("largest_absolute_coordinates", sample["largest_absolute_coordinates"][:-1]),
                             ("reference", "binary64"), ("extra", 0),
                             ("exact_local_identity", False), ("exact_input_delta", "NaN")):
            with self.assertRaises(jsonschema.ValidationError):
                account_validator.validate({**sample, field: value})
        row = self.report["controls"][0]
        with self.assertRaises(jsonschema.ValidationError):
            validator.validate({**self.report, "controls": [
                {**row, "branches": list(reversed(row["branches"]))}, *self.report["controls"][1:]]})

    def test_07_native_nibbles_groups_and_signed_words(self):
        tensors = {
            "qweight": np.full((4864, 112), -19088744, dtype="<i4"),
            "qzeros": np.full((38, 112), 0x76543210, dtype="<i4"),
            "scales": np.ones((38, 896), dtype="<f2"),
        }
        tensors["scales"][1::2] = np.float16(-0.5)
        for coordinate in (*range(8), 62, 895):
            self.assertEqual(d.weight_column(tensors, coordinate), oracle_weights(tensors, coordinate))
        self.assertNotEqual(d.weight_column(tensors, 0)[127], d.weight_column(tensors, 0)[128])

    def test_08_fp16_and_tensor_bindings(self):
        bits = [0, 0x8000, 1, 0x8001, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x7bff, 0xfbff]
        self.assertEqual(d.residual.words(np.array(bits, dtype="<u2"), (len(bits),)),
                         [half(word) for word in bits])
        for suffix, tensor in self.tensors.items():
            pin = self.binding["reference"]["canonical"][d.PREFIX + suffix]
            self.assertEqual(pin["sha256"], hashlib.sha256(tensor.tobytes()).hexdigest())
            self.assertIs(d.bind_tensor(tensor, pin, suffix), tensor)
            for field, value in (("sha256", "0" * 64), ("name", "model.layers.22.mlp.down_proj." + suffix),
                                 ("shape", []), ("dtype", "float32")):
                with self.assertRaises(ValueError):
                    d.bind_tensor(tensor, {**pin, field: value}, suffix)
            bad = tensor.copy()
            bad.flat[0] += 1
            with self.assertRaises(ValueError):
                d.bind_tensor(bad, pin, suffix)

    def test_09_operand_shape_and_nonfinite_rejection(self):
        for bits in (np.zeros(4863, dtype="<u2"), np.zeros(4864, dtype="<f2"),
                     np.full(4864, 0x7c00, dtype="<u2"), np.full(4864, 0xfe00, dtype="<u2")):
            with self.assertRaises(ValueError):
                d.residual.words(bits, (4864,))
        label = d.parent.CONTROLS[0]
        weights = d.weight_column(self.tensors, 62)
        for coordinate in (-1, 896, True, 1.5):
            with self.assertRaises(ValueError):
                d.account(self.actual[label], self.reference, weights, coordinate)
            with self.assertRaises(ValueError):
                d.weight_column(self.tensors, coordinate)
        with self.assertRaises(ValueError):
            d.account(self.actual[label], self.reference, weights[:-1], 62)

    def test_10_input_reference_and_state_kv_lineage_rejections(self):
        bad = deepcopy(self.previous[0])
        bad["preflight"]["L23_original_reference"]["reference"]["prior_kv"] = "substituted"
        with self.assertRaises(ValueError):
            d.residual.load_residuals(bad)
        bad = deepcopy(self.previous[0])
        bad["preflight"]["L23_original_reference"]["reference"]["fp16"] = {}
        with self.assertRaises(ValueError):
            d.residual.load_residuals(bad)
        archive = self.archives[d.parent.CONTROLS[0]]
        for key, value in (("input_i", np.zeros(896, dtype="<i4")),
                           ("scratch_z", np.full(896, 2, dtype="u1")),
                           ("input_cache_k", np.zeros((1, 128), dtype="<u2"))):
            with self.assertRaises(ValueError):
                d.residual.operands({**archive, key: value}, actual=True)
        bad = archive["output_cache_v"].copy()
        bad.flat[0] ^= 1
        with self.assertRaises(ValueError):
            d.residual.operands({**archive, "output_cache_v": bad}, actual=True)

    def test_11_hotspot_selection_tamper_rejections(self):
        for kind in ("control", "branch", "reorder", "duplicate"):
            bad = deepcopy(self.previous[5])
            row = bad["controls"][0]
            if kind == "control":
                bad["controls"].reverse()
            elif kind == "branch":
                row["branches"] = dict(reversed(list(row["branches"].items())))
            else:
                selected = row["branches"]["fp16"]["channels"]["largest_absolute_signed"]
                if kind == "reorder":
                    selected.reverse()
                else:
                    selected[1] = selected[0]
            with self.assertRaises(ValueError):
                d.report(bad, self.actual, self.reference, self.tensors, self.binding)

    def test_12_ties_cancellation_and_boundary_are_not_discarded(self):
        reference = {"stage16": [Fraction()] * 4864, "stage17": [Fraction()] * 896}
        actual = {"stage16": [Fraction((-1) ** i) for i in range(4864)],
                  "stage17": [Fraction(3, 2)] * 896}
        account = d.account(actual, reference, [Fraction(1)] * 4864, 62)
        self.assertEqual([e["coordinate"] for e in account["largest_absolute_coordinates"]], list(range(8)))
        self.assertEqual(account["exact_input_delta"], "0")
        self.assertEqual(account["sum_absolute_contributions"], "4864")
        self.assertEqual(account["unranked_signed_remainder"], "0")
        self.assertEqual(account["unranked_absolute_remainder"], "4856")
        self.assertEqual(account["down_projection_boundary_remainder"], "3/2")

    def test_13_forbidden_dispatch_guards(self):
        audit = {"forbidden_calls": 0}
        calls = [
            lambda: d.residual.native.projection(None, None, None),
            lambda: d.residual.native._stages(None, None, None, None),
            lambda: d.residual.local.local_reference(None, None, None, None),
            lambda: d.parent.execute(None), lambda: d.parent.rmsnorm(None, None),
            lambda: d.parent.logits(None, None), lambda: d.parent.preflight.parent.execute(None),
            lambda: d.residual.check(), lambda: d.residual.measure(),
            lambda: d.channels.check(), lambda: d.channels.focused_tests(None),
            lambda: d.base.check(), lambda: subprocess.Popen(["false"]),
            lambda: os.system("false"),
        ]
        with d.read_only(audit):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit["forbidden_calls"], len(calls))

    def test_14_forbidden_write_guards(self):
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
                    "evidence_writes", "local_operator_replay", "earlier_checks",
                    "mlp_down_projection_replay"):
            self.assertEqual(d.FLAGS[key], 0)

    def test_15_cli_one_document_only_and_no_output_mode(self):
        stream = io.StringIO()
        with patch.object(d, "check", return_value={"ok": True}), patch("sys.stdout", stream):
            d.main(["--check"])
        self.assertEqual(stream.getvalue(), '{"ok": true}\n')
        for argv in ([], ["--execute"], ["--check", "--output", "x"], ["--check", "--che"]):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                d.main(argv)

    def test_16_identity_and_evidence_pins_fail_closed(self):
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
