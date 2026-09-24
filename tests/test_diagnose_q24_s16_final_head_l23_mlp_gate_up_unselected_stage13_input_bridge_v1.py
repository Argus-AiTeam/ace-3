"""Independent half-bit and packed-nibble oracles for the stage13 complement."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_l23_mlp_gate_up_unselected_stage13_input_bridge_v1 as d


EVIDENCE = None


def half(word):
    exponent, mantissa = (word >> 10) & 31, word & 1023
    if exponent == 31:
        raise ValueError("nonfinite FP16")
    value = (Fraction(mantissa, 16777216) if exponent == 0
             else Fraction(1024 + mantissa) * Fraction(2) ** (exponent - 25))
    return -value if word & 32768 else value


def oracle_weights(tensors, output):
    lane = (0, 4, 1, 5, 2, 6, 3, 7)[output % 8]
    bits = tensors["scales"].view("<u2")
    result = []
    for i in range(896):
        q = (int(tensors["qweight"][i, output // 8]) % (1 << 32)) // (16 ** lane) % 16
        z = (int(tensors["qzeros"][i // 128, output // 8]) % (1 << 32)) // (16 ** lane) % 16
        result.append((q - z) * half(int(bits[i // 128, output])))
    return result


def accounts(report):
    for row in report["controls"]:
        for branch in row["branches"]:
            for hotspot in branch["hotspots"]:
                for selected in hotspot["stage16_coordinates"]:
                    for kind in ("gate_proj", "up_proj"):
                        yield row["control"], kind, selected[kind]


class UnrankedStage13Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is None:
            with d.read_only({"forbidden_calls": 0}):
                evidence, _, _ = d.measure()
        else:
            evidence = EVIDENCE
        cls.previous, cls.report = evidence
        cls.down_evidence, cls.actual, cls.reference, cls.tensors, cls.original = cls.previous
        cls.archives, cls.reference_archive = cls.down_evidence[3:5]
        cls.binding = cls.down_evidence[0][0]["preflight"]["L23_original_reference"]
        cls.terms, columns = {}, {}
        for label, kind, account in accounts(cls.report):
            output = account["output_coordinate"]
            key = (label, kind, output)
            if key not in cls.terms:
                if (kind, output) not in columns:
                    columns[kind, output] = oracle_weights(cls.tensors[kind], output)
                cls.terms[key] = [
                    (half(int(a)) - half(int(r))) * w
                    for a, r, w in zip(cls.archives[label]["stage13"],
                                       cls.reference_archive["stage13"],
                                       columns[kind, output], strict=True)]

    def sample(self):
        label, kind, account = next(accounts(self.original))
        return self.actual[label], self.reference, d.gate_up.weight_column(
            self.tensors[kind], account["output_coordinate"]), account

    def test_01_all_unranked_terms_independent_oracle(self):
        count = 0
        for label, kind, account in accounts(self.report):
            terms = self.terms[label, kind, account["output_coordinate"]]
            ranked = sorted(range(896), key=lambda i: (-abs(terms[i]), i))[:8]
            split = account["unranked_stage13_inputs"]
            self.assertEqual(split["excluded_ranked_coordinates"], ranked)
            rest = [term for i, term in enumerate(terms) if i not in ranked]
            self.assertEqual(Fraction(split["unranked_signed_sum"]), sum(rest))
            self.assertEqual(Fraction(split["unranked_absolute_sum"]), sum(map(abs, rest)))
            count += len(rest)
        self.assertEqual(count, 2045952)
        self.assertEqual(self.report["unranked_input_coordinate_count"], count)

    def test_02_every_g128_complement_and_partition(self):
        count = 0
        for label, kind, account in accounts(self.report):
            terms = self.terms[label, kind, account["output_coordinate"]]
            split = account["unranked_stage13_inputs"]
            excluded = set(split["excluded_ranked_coordinates"])
            for group, entry in enumerate(split["groups_in_input_order"]):
                indices = [i for i in range(group * 128, (group + 1) * 128) if i not in excluded]
                self.assertEqual(entry, {
                    "input_group": group, "start_coordinate": group * 128,
                    "end_coordinate_exclusive": (group + 1) * 128,
                    "coordinate_count": len(indices),
                    "excluded_ranked_coordinate_count": 128 - len(indices),
                    "signed_contribution": str(sum(terms[i] for i in indices)),
                    "sum_absolute_contributions": str(sum(abs(terms[i]) for i in indices)),
                })
                count += 1
            self.assertEqual(sum(g["coordinate_count"] for g in split["groups_in_input_order"]), 888)
            self.assertEqual(sum(g["excluded_ranked_coordinate_count"]
                                 for g in split["groups_in_input_order"]), 8)
        self.assertEqual(count, 16128)

    def test_03_exact_remainders_and_retained_projection_identities(self):
        for label, kind, account in accounts(self.report):
            split = account["unranked_stage13_inputs"]
            signed, absolute = (Fraction(split[key]) for key in
                                ("unranked_signed_sum", "unranked_absolute_sum"))
            self.assertEqual(str(signed), account["unranked_signed_remainder"])
            self.assertEqual(str(absolute), account["unranked_absolute_remainder"])
            self.assertEqual(signed, sum(Fraction(g["signed_contribution"])
                                        for g in split["groups_in_input_order"]))
            self.assertEqual(absolute, sum(Fraction(g["sum_absolute_contributions"])
                                          for g in split["groups_in_input_order"]))
            self.assertEqual(signed + Fraction(account["ranked_signed_sum"]),
                             Fraction(account["exact_input_delta"]))
            self.assertEqual(absolute + Fraction(account["ranked_absolute_sum"]),
                             Fraction(account["sum_absolute_contributions"]))
            stage = {"gate_proj": "stage14", "up_proj": "stage15"}[kind]
            i = account["output_coordinate"]
            retained = half(int(self.archives[label][stage][i])) - half(int(self.reference_archive[stage][i]))
            self.assertEqual(retained, signed + Fraction(account["ranked_signed_sum"])
                             + Fraction(account["projection_boundary_remainder"]))
            self.assertEqual(str(retained), account["retained_projection_delta"])
            self.assertGreaterEqual(absolute, abs(signed))

    def test_04_original_accounts_and_selection_preserved(self):
        stripped = deepcopy(self.report)
        del stripped["unranked_input_coordinate_count"]
        count = 0
        for _, _, account in accounts(stripped):
            del account["unranked_stage13_inputs"]
            count += 1
        self.assertEqual(stripped, self.original)
        self.assertEqual(count, 2304)
        self.assertEqual(self.report["selected_input_coordinate_count"], 18432)

    def test_05_original_references_lineage_history_and_bindings(self):
        self.assertEqual(self.report["L23_reference"], self.original["L23_reference"])
        self.assertEqual(self.report["L23_reference"]["fp16"], self.binding["reference"]["fp16"])
        self.assertFalse(self.report["L23_reference"]["original_global_references_reanchored"])
        self.assertEqual(self.report["L23_reference"]["binary64_internal_stages"],
                         "NOT_RETAINED_NO_RECONSTRUCTION")
        self.assertEqual(self.report["gate_up_projection_tensor_bindings"],
                         self.original["gate_up_projection_tensor_bindings"])
        self.assertEqual(self.report["down_projection_input_coordinate_report"], self.down_evidence[6])
        d.base.check_history(self.down_evidence[0][0])

    def test_06_schema_and_deterministic_split(self):
        jsonschema.Draft202012Validator.check_schema(d.OUTPUT_SCHEMA)
        jsonschema.Draft202012Validator(d.REPORT_SCHEMA).validate(self.report)
        split = d.split_unranked(*self.sample())
        self.assertEqual(split, next(accounts(self.report))[2]["unranked_stage13_inputs"])
        text = json.dumps(split, sort_keys=True, allow_nan=False)
        self.assertEqual(json.dumps(json.loads(text), sort_keys=True, allow_nan=False), text)

    def test_07_schema_rejects_scope_counts_and_extras(self):
        validator = jsonschema.Draft202012Validator(d.REPORT_SCHEMA)
        for key, value in (("unranked_input_coordinate_count", 2045951), ("control_count", 8),
                           ("layer", 22), ("position", 1), ("extra", 0),
                           ("controls", list(reversed(self.report["controls"])))):
            with self.assertRaises(jsonschema.ValidationError):
                validator.validate({**self.report, key: value})
        sample = next(accounts(self.report))[2]["unranked_stage13_inputs"]
        validator = jsonschema.Draft202012Validator(d.SPLIT_SCHEMA)
        for key, value in (("coordinate_count", 896), ("reference", "binary64"),
                           ("unranked_signed_sum", "NaN"), ("exact_projection_identity", False),
                           ("excluded_ranked_coordinates", [0] * 8),
                           ("groups_in_input_order", list(reversed(sample["groups_in_input_order"])))):
            with self.assertRaises(jsonschema.ValidationError):
                validator.validate({**sample, key: value})

    def test_08_ties_cancellation_and_distinct_boundaries(self):
        reference = {"stage13": [Fraction()] * 896, "stage14": [Fraction()] * 4864,
                     "stage15": [Fraction()] * 4864}
        actual = {"stage13": [Fraction((-1) ** i) for i in range(896)],
                  "stage14": [Fraction(3, 2)] * 4864, "stage15": [Fraction(-5, 2)] * 4864}
        weights = [Fraction(1)] * 896
        for kind, boundary in (("gate_proj", "3/2"), ("up_proj", "-5/2")):
            original = d.gate_up.account(actual, reference, weights, kind, 4863)
            split = d.split_unranked(actual, reference, weights, original)
            self.assertEqual(split["excluded_ranked_coordinates"], list(range(8)))
            self.assertEqual(split["unranked_signed_sum"], "0")
            self.assertEqual(split["unranked_absolute_sum"], "888")
            self.assertEqual([g["coordinate_count"] for g in split["groups_in_input_order"]],
                             [120, 128, 128, 128, 128, 128, 128])
            self.assertEqual(original["projection_boundary_remainder"], boundary)
        zero = d.gate_up.account(reference, reference, weights, "gate_proj", 0)
        self.assertEqual(d.split_unranked(reference, reference, weights, zero)["unranked_absolute_sum"], "0")

    def test_09_group_edges_and_negative_net(self):
        reference = {"stage13": [Fraction()] * 896, "stage14": [Fraction()] * 4864}
        actual = {"stage13": [Fraction(-1)] * 896, "stage14": [Fraction(7)] * 4864}
        edges = [0, 127, 128, 255, 256, 767, 768, 895]
        for i in edges:
            actual["stage13"][i] = Fraction(-10)
        weights = [Fraction(1)] * 896
        original = d.gate_up.account(actual, reference, weights, "gate_proj", 0)
        split = d.split_unranked(actual, reference, weights, original)
        self.assertEqual(split["excluded_ranked_coordinates"], edges)
        self.assertEqual(split["unranked_signed_sum"], "-888")
        self.assertEqual(split["unranked_absolute_sum"], "888")
        self.assertEqual([g["excluded_ranked_coordinate_count"] for g in split["groups_in_input_order"]],
                         [2, 2, 1, 0, 0, 1, 2])

    def test_10_ranked_complement_tampering_rejected(self):
        actual, reference, weights, original = self.sample()
        for mode in ("duplicate", "reverse", "missing", "range", "bool", "weight", "delta"):
            bad = deepcopy(original)
            selected = bad["largest_absolute_coordinates"]
            if mode == "duplicate":
                selected[1] = selected[0]
            elif mode == "reverse":
                selected.reverse()
            elif mode == "missing":
                selected.pop()
            elif mode in ("range", "bool"):
                selected[0]["coordinate"] = 896 if mode == "range" else True
            else:
                selected[0]["weight" if mode == "weight" else "input_delta"] = "999"
            with self.assertRaises(ValueError):
                d.split_unranked(actual, reference, weights, bad)

    def test_11_exact_closure_tampering_rejected(self):
        actual, reference, weights, original = self.sample()
        for key in ("unranked_signed_remainder", "unranked_absolute_remainder",
                    "ranked_signed_sum", "ranked_absolute_sum", "exact_input_delta",
                    "sum_absolute_contributions", "actual_projection_output",
                    "reference_projection_output", "retained_projection_delta",
                    "projection_boundary_remainder"):
            with self.assertRaises(ValueError):
                d.split_unranked(actual, reference, weights,
                                {**original, key: str(Fraction(original[key]) + 1)})
        bad = deepcopy(original)
        bad["groups_in_input_order"][0]["signed_contribution"] = "999"
        with self.assertRaises(ValueError):
            d.split_unranked(actual, reference, weights, bad)

    def test_12_invalid_operands_and_reference_rejected(self):
        actual, reference, weights, original = self.sample()
        for bad_weights in (weights[:-1], [0.0] * 896):
            with self.assertRaises(ValueError):
                d.split_unranked(actual, reference, bad_weights, original)
        for vector in (actual["stage13"][:-1], [0.0] * 896):
            with self.assertRaises(ValueError):
                d.split_unranked({**actual, "stage13": vector}, reference, weights, original)
        for key, value in (("reference", "binary64"), ("input_stage", "stage12_fp16"),
                           ("output_stage", "stage16_fp16"), ("output_coordinate", True),
                           ("projection", "down_proj")):
            with self.assertRaises(ValueError):
                d.split_unranked(actual, reference, weights, {**original, key: value})

    def test_13_native_awq_nibbles_and_tensor_authentication(self):
        tensors = {"qweight": np.full((896, 608), -19088744, dtype="<i4"),
                   "qzeros": np.full((7, 608), 0x76543210, dtype="<i4"),
                   "scales": np.ones((7, 4864), dtype="<f2")}
        tensors["scales"][1::2] = np.float16(-0.5)
        for coordinate in (*range(8), 4863):
            self.assertEqual(d.gate_up.weight_column(tensors, coordinate),
                             oracle_weights(tensors, coordinate))
        canonical = self.binding["reference"]["canonical"]
        for kind in ("gate_proj", "up_proj"):
            prefix = d.gate_up.PREFIXES[kind]
            with self.assertRaises(ValueError):
                d.gate_up.tensor_names({**canonical, prefix + "bias": {}}, kind)
            for suffix in ("qweight", "qzeros", "scales"):
                with self.assertRaises(ValueError):
                    d.gate_up.bind_tensor(self.tensors[kind][suffix],
                                         {**canonical[prefix + suffix], "sha256": "0" * 64},
                                         kind, suffix)

    def test_14_state_kv_reference_and_source_gates(self):
        for key, value in (("prior_kv", "substituted"), ("fp16", {})):
            bad = deepcopy(self.down_evidence[0][0])
            bad["preflight"]["L23_original_reference"]["reference"][key] = value
            with self.assertRaises(ValueError):
                d.residual.load_residuals(bad)
        archive = self.archives[d.parent.CONTROLS[0]]
        bad_kv = archive["output_cache_v"].copy()
        bad_kv.flat[0] ^= 1
        for key, value in (("input_i", np.zeros(896, dtype="<i4")),
                           ("scratch_z", np.full(896, 2, dtype="u1")),
                           ("output_cache_v", bad_kv)):
            with self.assertRaises(ValueError):
                d.residual.operands({**archive, key: value}, actual=True)
        for pin in (d.base.PINS["result"], d.channels.PINS["contribution_source"],
                    self.binding["reference"]["fp16"]):
            with self.assertRaises(ValueError):
                d.base.read_bound({**pin, "sha256": "0" * 64})

    def test_15_forbidden_dispatch_and_earlier_checks(self):
        audit = {"forbidden_calls": 0}
        calls = [
            lambda: d.residual.native.projection(None, None, None),
            lambda: d.residual.native._stages(None, None, None, None),
            lambda: d.residual.native.toward_zero(None),
            lambda: d.residual.local.local_reference(None, None, None, None),
            lambda: d.parent.execute(None), lambda: d.parent.rmsnorm(None, None),
            lambda: d.parent.logits(None, None), lambda: d.parent.preflight.parent.execute(None),
            lambda: d.gate_up.check(), lambda: d.gate_up.focused_tests(None),
            lambda: d.down.check(), lambda: d.residual.check(), lambda: d.residual.measure(),
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
        self.assertEqual(d.FLAGS, d.gate_up.FLAGS)
        self.assertTrue(all(value == 0 for value in d.FLAGS.values()))

    def test_17_stdout_only_cli_rejects_execution_and_output_flags(self):
        stream = io.StringIO()
        with patch.object(d, "check", return_value={"ok": True}), patch("sys.stdout", stream):
            d.main(["--check"])
        self.assertEqual(stream.getvalue(), '{"ok": true}\n')
        for argv in ([], ["--execute"], ["--check", "--execute"],
                     ["--check", "--output", "x"], ["--check", "--che"]):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                d.main(argv)

    def test_18_identity_fails_before_measurement(self):
        with patch.object(d.os, "getuid", return_value=1), patch.object(d, "measure") as measure:
            with self.assertRaises(ValueError):
                d.check()
            measure.assert_not_called()


if __name__ == "__main__":
    unittest.main()
