"""Independent scalar/bit oracles for bounded P0 attention attribution."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_attention_source_value_attribution_v1 as d
from ace3.model.candidates import local_operator_reference_v3 as local
from ace3.model.candidates import q24_s16_toward_zero_native_v1 as native


EVIDENCE = None


def half(word):
    sign = -1 if word & 32768 else 1
    exponent, mantissa = (word >> 10) & 31, word & 1023
    if exponent == 31:
        raise ValueError("nonfinite half")
    return sign * (Fraction(mantissa, 2 ** 24) if exponent == 0
                   else Fraction(1024 + mantissa) * Fraction(2) ** (exponent - 25))


def native_nibble(word, lane):
    # Official physical GEMM lane order, inverted without production shifts.
    physical = (0, 2, 4, 6, 1, 3, 5, 7).index(lane)
    return ((int(word) & 0xffffffff) // (16 ** physical)) % 16


def column(tensors, output):
    return [
        (native_nibble(tensors["qweight"][j, output // 8], output % 8)
         - native_nibble(tensors["qzeros"][j // 128, output // 8], output % 8))
        * half(int(tensors["scales"].view("<u2")[j // 128, output]))
        for j in range(896)
    ]


class AttentionAttributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is None:
            with d.read_only({"forbidden_calls": 0}):
                cls.evidence, _, _ = d.measure()
        else:
            cls.evidence = EVIDENCE
        cls.channel_evidence, cls.actual, cls.reference, cls.tensors, cls.report = cls.evidence
        cls.result = cls.channel_evidence[0]

    def accounts(self):
        for row in self.report["controls"]:
            for branch, item in row["branches"].items():
                for hotspot in item["hotspots"]:
                    yield row["control"], branch, hotspot

    def test_all_value_components_independent_scalar_oracle(self):
        cache = {}
        for label, _, hotspot in self.accounts():
            output = hotspot["final_head_channel"]["coordinate"]
            if output not in cache:
                cache[output] = column(self.tensors, output)
            weights = cache[output]
            va, vr = self.actual[label][1], self.reference[1]
            expected = [
                sum(weights[head * 64 + dim] * (va[kv * 64 + dim] - vr[kv * 64 + dim])
                    for head in range(kv * 7, kv * 7 + 7))
                for kv in range(2) for dim in range(64)]
            for entry in hotspot["attention"]["full_value_component_ranking"]:
                index = entry["value_coordinate"]
                self.assertEqual(Fraction(entry["signed_contribution"]), expected[index])
                self.assertEqual(Fraction(entry["value"]), expected[index])
                self.assertEqual(entry["probability"], "0")
                self.assertEqual(entry["interaction"], "0")

    def test_native_awq_all_lanes_groups_and_no_plus_one(self):
        packed = np.full((896, 112), 0x76543210, dtype="<i4")
        zeros = np.full((7, 112), 0x01234567, dtype="<i4")
        scales = np.full((7, 896), 2 ** -24, dtype="<f2")
        scales[1::2] = -0.5
        tensors = {"qweight": packed, "qzeros": zeros, "scales": scales}
        for output in range(8):
            self.assertEqual(d.weight_column(tensors, output), column(tensors, output))
        self.assertEqual(d.weight_column(tensors, 0)[0], -7 * Fraction(1, 2 ** 24))
        packed[:] = -1
        self.assertEqual(d.weight_column(tensors, 895), column(tensors, 895))

    def test_local_remainders_and_exact_identity(self):
        for label, _, hotspot in self.accounts():
            i, account = hotspot["final_head_channel"]["coordinate"], hotspot["attention"]
            values = [Fraction(e["signed_contribution"]) for e in account["full_value_component_ranking"]]
            delta = self.actual[label][3][i] - self.reference[3][i]
            self.assertEqual(sum(values), Fraction(account["exact_attention_sum"]))
            self.assertEqual(account["av_boundary_remainder"], "0")
            self.assertEqual(Fraction(account["retained_o_projection_delta"]), delta)
            self.assertEqual(delta - sum(values), Fraction(account["o_projection_boundary_remainder"]))
            ranked = sum(values[:8])
            self.assertEqual(Fraction(account["ranked_signed_sum"]), ranked)
            self.assertEqual(Fraction(account["unranked_signed_remainder"]), sum(values) - ranked)

    def test_source_token_gqa_mapping_and_census(self):
        pins = [(row["parent"]["terminal_archive"], self.actual[row["control"]])
                for row in self.result["controls"]]
        pins.append((self.result["preflight"]["L23_original_reference"]["reference"]["fp16"],
                     self.reference))
        for pin, operands in pins:
            with np.load(io.BytesIO(d.base.read_bound(pin)), allow_pickle=False) as z:
                for name, values in zip(("stage09", "stage07", "stage10", "stage11"),
                                        operands, strict=True):
                    self.assertEqual(values, [half(int(word)) for word in z[name]])
        count = 0
        for _, _, hotspot in self.accounts():
            count += 1
            account = hotspot["attention"]
            self.assertEqual(account["source_token_count"], 1)
            token = account["source_tokens"][0]
            self.assertEqual((token["source_position"], token["source_token_id"]), (0, 9707))
            self.assertEqual(token["signed_contribution"], account["exact_attention_sum"])
            self.assertEqual(token["probability"], token["interaction"])
            for item in account["full_value_component_ranking"]:
                kv, dim = divmod(item["value_coordinate"], 64)
                self.assertEqual((item["kv_head"], item["head_dimension"]), (kv, dim))
                self.assertEqual(item["query_heads"], list(range(kv * 7, kv * 7 + 7)))
        self.assertEqual(count, self.report["hotspot_count"])
        self.assertEqual(count * 896, self.report["query_value_product_count"])
        self.assertEqual(count * 128, self.report["value_component_count"])

    def test_order_and_top_hotspot_selection(self):
        self.assertEqual([r["control"] for r in self.report["controls"]], list(d.parent.CONTROLS))
        for row, original in zip(self.report["controls"], self.channel_evidence[5]["controls"], strict=True):
            self.assertEqual(list(row["branches"]), list(d.channels.BRANCHES))
            for name, branch in row["branches"].items():
                source = original["branches"][name]
                self.assertEqual(branch["numeric_id"], source["numeric_id"])
                self.assertEqual([h["hotspot_rank"] for h in branch["hotspots"]], list(range(1, 9)))
                self.assertEqual([h["final_head_channel"] for h in branch["hotspots"]],
                                 source["channels"]["largest_absolute_signed"])
        for _, _, hotspot in self.accounts():
            account = hotspot["attention"]
            entries = account["full_value_component_ranking"]
            self.assertEqual(sorted(e["value_coordinate"] for e in entries), list(range(128)))
            self.assertEqual(entries, sorted(entries, key=lambda e: (
                -abs(Fraction(e["signed_contribution"])), e["value_coordinate"])))
            self.assertEqual(account["largest_absolute_value_components"], entries[:8])

    def test_report_serialization_and_schema(self):
        jsonschema.Draft202012Validator.check_schema(d.OUTPUT_SCHEMA)
        jsonschema.Draft202012Validator(d.REPORT_SCHEMA).validate(self.report)
        text = json.dumps(self.report, sort_keys=True, allow_nan=False)
        self.assertEqual(json.loads(text), self.report)
        self.assertEqual(text, json.dumps(deepcopy(self.report), sort_keys=True, allow_nan=False))

    def test_schema_rejects_counts_extra_and_scope(self):
        validator = jsonschema.Draft202012Validator(d.REPORT_SCHEMA)
        for field, value in (("hotspot_count", 143), ("controls", []),
                             ("value_component_count", 0), ("extra", True)):
            with self.assertRaises(jsonschema.ValidationError):
                validator.validate({**self.report, field: value})
        account = next(self.accounts())[2]["attention"]
        validator = jsonschema.Draft202012Validator(d.ACCOUNT_SCHEMA)
        for field, value in (("position", 1), ("layer", 22),
                             ("attention_reference", "binary64"), ("full_value_component_ranking", [])):
            with self.assertRaises(jsonschema.ValidationError):
                validator.validate({**account, field: value})

    def test_original_references_and_histories_unchanged(self):
        retained = self.report["final_head_channel_report"]
        self.assertEqual(retained, self.channel_evidence[5])
        self.assertEqual(retained["original_global_reference"], self.result["preflight"]["final_reference"])
        self.assertEqual(retained["thresholds"], self.result["preflight"]["thresholds"])
        self.assertEqual(retained["retained_L23_stage_reports"], {"PASS": 162, "FAIL": 9})
        ref = self.report["attention_reference"]
        self.assertEqual(ref["binary64_stage_status"], "NOT_RETAINED_NO_RECONSTRUCTION")
        self.assertIs(ref["final_head_branches_reanchored"], False)
        self.assertEqual(ref["fp16"], retained["original_global_reference"]["reference"]["input_fp16"])
        self.assertIn("NOT propagation", d.BOUNDARY)
        for row in retained["controls"]:
            self.assertEqual(row["L21_L22_L23_status"], ["FAIL"] * 3)
            self.assertTrue(all(row["retained_failures"].values()))
        for field, value in (("candidate_admitted", True),
                             ("source_operand_state_KV_RTZ_checks", "FAIL")):
            bad = deepcopy(self.result)
            bad["controls"][0]["parent"]["retained_L23"][field] = value
            with self.assertRaises(ValueError):
                d.base.check_history(bad)

    def test_missing_reordered_control_and_branch_rejected(self):
        for actual in ({}, dict(reversed(list(self.actual.items())))):
            with self.assertRaises(ValueError):
                d.report(self.channel_evidence[5], actual, self.reference, self.tensors, {})
        bad = deepcopy(self.channel_evidence[5])
        bad["controls"][0]["branches"] = {}
        with self.assertRaises(ValueError):
            d.report(bad, self.actual, self.reference, self.tensors, {})

    def test_invalid_words_and_coordinate_refused(self):
        for array, shape in ((np.zeros(1, dtype="<f2"), (1,)),
                             (np.zeros(2, dtype="<u2"), (1,)),
                             (np.array([0x7c00], dtype="<u2"), (1,)),
                             (np.array([0x7e00], dtype="<u2"), (1,))):
            with self.assertRaises(ValueError):
                d.words(array, shape)
        self.assertEqual(d.words(np.array([1, 0x8001], dtype="<u2"), (2,)),
                         [Fraction(1, 2 ** 24), -Fraction(1, 2 ** 24)])
        for coordinate in (-1, 896, True):
            with self.assertRaises(ValueError):
                d.weight_column(self.tensors, coordinate)

    def test_probability_v_cache_and_gqa_splices_rejected(self):
        pin = self.result["controls"][0]["parent"]["terminal_archive"]
        with np.load(io.BytesIO(d.base.read_bound(pin)), allow_pickle=False) as z:
            original = {k: z[k] for k in z.files}
        for name in ("stage03", "stage07", "stage09", "stage10", "output_cache_v"):
            bad = deepcopy(original)
            bad[name].flat[0] ^= 1
            with self.assertRaises(ValueError):
                d.attention_operands(bad, actual=True)
        bad = {**original, "input_cache_v": np.zeros((1, 128), dtype="<u2")}
        with self.assertRaises(ValueError):
            d.attention_operands(bad, actual=True)

    def test_attention_reference_and_tensor_binding_tamper_rejected(self):
        assets = self.result["preflight"]["assets"]
        for field, value in (("prior_kv", "other"), ("fp16", {})):
            bad = deepcopy(self.result)
            bad["preflight"]["L23_original_reference"]["reference"][field] = value
            with self.assertRaises(ValueError):
                d.load_attention(bad, assets)
        bad = deepcopy(self.result)
        bad["preflight"]["L23_original_reference"]["reference"]["canonical"][d.PROJECTION + "qzeros"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            d.load_attention(bad, assets)

    def test_pinned_source_and_archive_tamper_rejected(self):
        pins = [self.result["controls"][0]["parent"]["terminal_archive"],
                *d.channels.PINS.values(), *d.base.PINS.values()]
        for pin in pins:
            with self.assertRaises(ValueError):
                d.base.read_bound({**pin, "sha256": "0" * 64})

    def test_probability_value_interaction_and_boundary_synthetic(self):
        actual = ([Fraction(2)] * 14, [Fraction(3)] * 128,
                  [Fraction(7)] * 896, [Fraction(20)] * 896)
        reference = ([Fraction(1)] * 14, [Fraction(1)] * 128,
                     [Fraction(1)] * 896, [Fraction(4)] * 896)
        weights = [Fraction()] * 896
        weights[0], weights[64] = 2, -1
        account = d.account(actual, reference, weights, 0)
        entry = account["full_value_component_ranking"][0]
        self.assertEqual((entry["probability"], entry["value"], entry["interaction"]), ("1", "2", "2"))
        self.assertEqual(account["exact_attention_sum"], "5")
        self.assertEqual(account["av_boundary_remainder"], "1")
        self.assertEqual(account["o_projection_boundary_remainder"], "10")
        self.assertEqual(account["retained_o_projection_delta"], "16")

    def test_ties_zero_and_signed_cancellation(self):
        actual = ([Fraction(1)] * 14, [Fraction(2)] * 128,
                  [Fraction(2)] * 896, [Fraction()] * 896)
        reference = ([Fraction(1)] * 14, [Fraction(1)] * 128,
                     [Fraction(1)] * 896, [Fraction()] * 896)
        weights = [Fraction()] * 896
        weights[0], weights[1], weights[64] = 2, -1, -1
        account = d.account(actual, reference, weights, 0)
        self.assertEqual([x["value_coordinate"] for x in account["largest_absolute_value_components"]],
                         list(range(8)))
        self.assertEqual(account["exact_attention_sum"], "0")
        self.assertEqual(account["full_value_component_ranking"][0]["signed_contribution"], "1")
        self.assertEqual(account["full_value_component_ranking"][1]["signed_contribution"], "-1")
        zero = d.account(reference, reference, weights, 0)
        self.assertEqual(zero["exact_attention_sum"], "0")

    def test_forbidden_native_dispatch_and_evidence_writes(self):
        audit = {"forbidden_calls": 0}
        calls = [
            lambda: native.projection(None, None, None),
            lambda: native._stages(None, None, None, None),
            lambda: local.local_reference(None, None, None, None),
            lambda: d.parent.execute(None), lambda: d.parent.rmsnorm(None, None),
            lambda: d.parent.logits(None, None), lambda: d.parent.preflight.parent.execute(None),
            lambda: d.channels.check(), lambda: d.channels.focused_tests(None),
            lambda: d.channels.hotspots.check(), lambda: d.base.check(),
            lambda: d.channels.contributions.dyadic_dot(None, None),
            lambda: d.parent.write(None, None), lambda: subprocess.Popen(["false"]),
            lambda: os.system("false"), lambda: open(d.SOURCE, "w"),
            lambda: d.SOURCE.write_text("forbidden"), lambda: os.open(d.SOURCE, os.O_WRONLY),
            lambda: os.unlink(d.SOURCE), lambda: os.rename(d.SOURCE, d.TEST),
            lambda: d.channels.contributions.preflight.parent.producer.legacy.torch.cuda._lazy_init(),
        ]
        with d.read_only(audit):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit["forbidden_calls"], len(calls))
        for flag in ("native_dispatch", "prefix_dispatch", "admission_dispatch", "evidence_writes",
                     "RTL_dispatch", "GPU_dispatch", "hardware_dispatch", "attention_replay",
                     "o_projection_replay"):
            self.assertEqual(d.FLAGS[flag], 0)

    def test_cli_one_json_no_execution_or_output_modes(self):
        stream = io.StringIO()
        with patch.object(d, "check", return_value={"result": 1}), patch("sys.stdout", stream):
            d.main(["--check"])
        self.assertEqual(stream.getvalue(), '{"result": 1}\n')
        for args in ([], ["--execute"], ["--check", "--out", "build/new"], ["--che"]):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit) as error:
                d.main(args)
            self.assertEqual(error.exception.code, 2)

    def test_source_gate_rejects_wrong_workdir_before_read(self):
        with patch.object(d.Path, "cwd", return_value=d.ROOT.parent), \
                patch.object(d, "measure", side_effect=AssertionError("must not measure")):
            with self.assertRaises(ValueError):
                d.check()


if __name__ == "__main__":
    unittest.main()
