"""Independent half-bit identities for the retained L23 SiLU/product bridge."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_l23_mlp_silu_product_bridge_v1 as d


EVIDENCE = None


def half(word):
    exponent, mantissa = (word >> 10) & 31, word & 1023
    if exponent == 31:
        raise ValueError("nonfinite half")
    value = (Fraction(mantissa, 1 << 24) if exponent == 0
             else Fraction(1024 + mantissa) * Fraction(2) ** (exponent - 25))
    return -value if word & 32768 else value


class SiluProductBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is None:
            with d.read_only({"forbidden_calls": 0}):
                cls.evidence, _, _ = d.measure()
        else:
            cls.evidence = EVIDENCE
        cls.previous, cls.actual, cls.reference, cls.report = cls.evidence
        cls.archives, cls.reference_archive = cls.previous[0][3:5]

    def hotspots(self):
        for row in self.report["controls"]:
            for branch in row["branches"]:
                for hotspot in branch["hotspots"]:
                    yield row["control"], hotspot

    def accounts(self):
        for label, hotspot in self.hotspots():
            for item in hotspot["stage16_coordinates"]:
                yield label, item

    def oracle(self, label, coordinate):
        a, r = ([half(int(archive[stage][coordinate]))
                 for stage in ("stage14", "stage15", "stage16")]
                for archive in (self.archives[label], self.reference_archive))
        ag, au, ay = a
        rg, ru, ry = r
        # Endpoint expansion, independently decoded from authenticated archive words.
        terms = [ag * ru - rg * ru, rg * au - rg * ru,
                 ag * au - ag * ru - rg * au + rg * ru,
                 ay - ry - ag * au + rg * ru]
        return a, r, terms

    def assert_totals(self, got, terms):
        self.assertEqual(got, {
            "signed_sum": str(sum(terms, Fraction())),
            "absolute_sum": str(sum(map(abs, terms), Fraction())),
            "cancellation_absolute_mass": str(sum(map(abs, terms), Fraction()) - abs(sum(terms))),
        })

    def test_01_every_operand_term_and_endpoint_from_half_bits(self):
        count = 0
        for label, item in self.accounts():
            account = item["bridge"]
            a, r, terms = self.oracle(label, account["coordinate"])
            self.assertEqual(account["actual"], dict(zip(d.OPERAND_NAMES, map(str, a), strict=True)))
            self.assertEqual(account["reference_operands"],
                             dict(zip(d.OPERAND_NAMES, map(str, r), strict=True)))
            self.assertEqual([Fraction(t["signed_contribution"])
                              for t in account["terms_in_operand_order"]], terms)
            self.assertEqual(Fraction(account["gate_delta"]), a[0] - r[0])
            self.assertEqual(Fraction(account["up_delta"]), a[1] - r[1])
            self.assertEqual(Fraction(account["retained_stage16_delta"]), a[2] - r[2])
            self.assertEqual(sum(terms), a[2] - r[2])
            count += 1
        self.assertEqual(count, 1152)

    def test_02_actual_reference_and_delta_boundary_remainders(self):
        for label, item in self.accounts():
            account = item["bridge"]
            a, r, terms = self.oracle(label, account["coordinate"])
            self.assertEqual(Fraction(account["actual_raw_gate_up_product"]), a[0] * a[1])
            self.assertEqual(Fraction(account["reference_raw_gate_up_product"]), r[0] * r[1])
            self.assertEqual(Fraction(account["raw_gate_up_product_delta"]), sum(terms[:3]))
            self.assertEqual(Fraction(account["actual_silu_product_remainder"]), a[2] - a[0] * a[1])
            self.assertEqual(Fraction(account["reference_silu_product_remainder"]), r[2] - r[0] * r[1])
            self.assertEqual(Fraction(account["silu_product_boundary_delta"]), terms[3])
            self.assertEqual(account["baseline"], "raw_gate_times_up_NOT_SILU")
            self.assertEqual(account["separate_silu_product_rounding"], "NOT_IDENTIFIABLE_WITHOUT_REPLAY")

    def test_03_signed_absolute_interaction_cancellation_and_ranks(self):
        for label, item in self.accounts():
            account = item["bridge"]
            terms = self.oracle(label, account["coordinate"])[2]
            self.assert_totals(account["operand_totals"], terms[:3])
            self.assert_totals(account["closure_totals"], terms)
            self.assertEqual([entry["term_index"] for entry in account["ranked_absolute_terms"]],
                             sorted(range(4), key=lambda i: (-abs(terms[i]), i)))
            self.assertEqual(account["ranked_absolute_terms"], sorted(
                account["terms_in_operand_order"],
                key=lambda e: (-abs(Fraction(e["signed_contribution"])), e["term_index"])))

    def test_04_authenticated_down_weight_and_weighted_closure(self):
        tensors = self.previous[0][5]
        scales = tensors["scales"].view("<u2")
        for label, hotspot in self.hotspots():
            output = hotspot["final_head_channel"]["coordinate"]
            lane, packed = (0, 4, 1, 5, 2, 6, 3, 7)[output % 8], output // 8
            for item in hotspot["stage16_coordinates"]:
                account = item["bridge"]
                i = account["coordinate"]
                q = int(tensors["qweight"][i, packed]) % (1 << 32) // 16 ** lane % 16
                z = int(tensors["qzeros"][i // 128, packed]) % (1 << 32) // 16 ** lane % 16
                weight = (q - z) * half(int(scales[i // 128, output]))
                a, r, terms = self.oracle(label, i)
                weighted = [weight * term for term in terms]
                self.assertEqual(account["down_weight"], str(weight))
                self.assertEqual([Fraction(t["signed_contribution"])
                                  for t in account["weighted_terms_in_operand_order"]], weighted)
                self.assert_totals(account["weighted_closure_totals"], weighted)
                self.assertEqual(account["weighted_stage16_delta"], str(sum(weighted)))
                self.assertEqual(account["weighted_stage16_delta"], item["selected_down_input"]["signed_contribution"])
                self.assertEqual(Fraction(account["weighted_actual_silu_product_remainder"]),
                                 weight * (a[2] - a[0] * a[1]))
                self.assertEqual(Fraction(account["weighted_reference_silu_product_remainder"]),
                                 weight * (r[2] - r[0] * r[1]))

    def test_05_hotspot_totals_ranking_and_unselected_stage17_remainders(self):
        for label, hotspot in self.hotspots():
            accounts = [item["bridge"] for item in hotspot["stage16_coordinates"]]
            summary = hotspot["summary"]
            stage16 = [self.oracle(label, a["coordinate"])[0][2]
                       - self.oracle(label, a["coordinate"])[1][2] for a in accounts]
            weighted = [delta * Fraction(a["down_weight"])
                        for delta, a in zip(stage16, accounts, strict=True)]
            self.assert_totals(summary["selected_stage16_totals"], stage16)
            self.assert_totals(summary["selected_weighted_down_totals"], weighted)
            self.assertEqual(summary["stage16_coordinates_ranked_by_absolute_delta"],
                             [accounts[i]["coordinate"] for i in sorted(range(8), key=lambda i: (
                                 -abs(stage16[i]), accounts[i]["coordinate"]))])
            for i, entry in enumerate(summary["weighted_operand_terms_in_order"]):
                self.assert_totals(entry["totals"], [
                    self.oracle(label, a["coordinate"])[2][i] * Fraction(a["down_weight"])
                    for a in accounts])
            self.assertEqual(sum(weighted) + Fraction(summary["unselected_down_input_signed_remainder"])
                             + Fraction(summary["down_projection_boundary_remainder"]),
                             Fraction(summary["retained_stage17_delta"]))

    def test_06_selection_history_references_and_all_nine_bindings_preserved(self):
        self.assertEqual(self.report["gate_up_projection_input_coordinate_report"], self.previous[4])
        self.assertEqual(self.report["L23_reference"], self.previous[4]["L23_reference"])
        d.base.check_history(self.previous[0][0][0])
        canonical = self.previous[0][0][0]["preflight"]["L23_original_reference"]["reference"]["canonical"]
        count = 0
        for kind in ("gate_proj", "up_proj"):
            for suffix in d.gate_up.SPECS:
                name = d.gate_up.PREFIXES[kind] + suffix
                tensor = self.previous[3][kind][suffix]
                self.assertIs(d.gate_up.bind_tensor(tensor, canonical[name], kind, suffix), tensor)
                with self.assertRaises(ValueError):
                    d.gate_up.bind_tensor(tensor, {**canonical[name], "sha256": "0" * 64}, kind, suffix)
                count += 1
        for suffix, tensor in self.previous[0][5].items():
            pin = canonical[d.down.PREFIX + suffix]
            self.assertIs(d.down.bind_tensor(tensor, pin, suffix), tensor)
            with self.assertRaises(ValueError):
                d.down.bind_tensor(tensor, {**pin, "sha256": "0" * 64}, suffix)
            count += 1
        self.assertEqual(count, 9)
        for row, prior in zip(self.report["controls"], self.previous[4]["controls"], strict=True):
            for branch, old_branch in zip(row["branches"], prior["branches"], strict=True):
                self.assertEqual(branch["numeric_id"], old_branch["numeric_id"])
                for hotspot, old in zip(branch["hotspots"], old_branch["hotspots"], strict=True):
                    self.assertEqual(hotspot["final_head_channel"], old["final_head_channel"])
                    self.assertEqual([s["selected_down_input"] for s in hotspot["stage16_coordinates"]],
                                     [s["selected_down_input"] for s in old["stage16_coordinates"]])

    def test_07_json_schema_and_deterministic_scalar_account(self):
        jsonschema.Draft202012Validator.check_schema(d.OUTPUT_SCHEMA)
        jsonschema.Draft202012Validator(d.REPORT_SCHEMA).validate(self.report)
        _, item = next(self.accounts())
        account = item["bridge"]
        self.assertEqual(d.account(
            {k: Fraction(v) for k, v in account["actual"].items()},
            {k: Fraction(v) for k, v in account["reference_operands"].items()},
            account["coordinate"], Fraction(account["down_weight"])), account)
        text = json.dumps(account, sort_keys=True, allow_nan=False)
        self.assertEqual(json.dumps(json.loads(text), sort_keys=True, allow_nan=False), text)

    def test_08_schema_rejects_scope_nonfinite_and_extra_fields(self):
        account = next(self.accounts())[1]["bridge"]
        validator = jsonschema.Draft202012Validator(d.ACCOUNT_SCHEMA)
        for key, value in (("coordinate", 4864), ("reference", "binary64"),
                           ("baseline", "silu_output"), ("gate_delta", "NaN"),
                           ("exact_local_identity", False), ("extra", 1),
                           ("terms_in_operand_order", list(reversed(account["terms_in_operand_order"])))):
            with self.assertRaises(jsonschema.ValidationError):
                validator.validate({**account, key: value})
        for key in ("bridge_account_count", "closure_term_count", "layer", "position"):
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate(-1, d.REPORT_SCHEMA["properties"][key])

    def test_09_zero_subnormal_sign_and_cancellation_synthetic_oracle(self):
        for ag, au, ay, rg, ru, ry in (
            (0, 0, 1, 0x8000, 0, 0), (1, 0x8001, 0, 0x8001, 1, 0x8000),
            (0x3c00, 0xbc00, 0x3c00, 0xbc00, 0x3c00, 0x3c00),
            (0x7bff, 0xfbff, 0x0400, 0xfbff, 0x7bff, 0x03ff),
        ):
            a = dict(zip(d.OPERAND_NAMES, map(half, (ag, au, ay)), strict=True))
            r = dict(zip(d.OPERAND_NAMES, map(half, (rg, ru, ry)), strict=True))
            result = d.account(a, r, 4863, Fraction(-3, 2))
            expected = [a["gate"] * r["up"] - r["gate"] * r["up"],
                        r["gate"] * a["up"] - r["gate"] * r["up"],
                        (a["gate"] - r["gate"]) * (a["up"] - r["up"]),
                        a["stage16"] - r["stage16"] - a["gate"] * a["up"] + r["gate"] * r["up"]]
            self.assertEqual([Fraction(t["signed_contribution"])
                              for t in result["terms_in_operand_order"]], expected)
            self.assert_totals(result["closure_totals"], expected)
        a = {"gate": Fraction(1), "up": Fraction(1), "stage16": Fraction(0)}
        result = d.account(a, a, 0, Fraction(0))
        self.assertEqual([t["term_index"] for t in result["ranked_absolute_terms"]], list(range(4)))

    def test_10_invalid_scalar_operands_rejected(self):
        a = dict.fromkeys(d.OPERAND_NAMES, Fraction(1))
        for coordinate in (-1, 4864, True, 0.5):
            with self.assertRaises(ValueError):
                d.account(a, a, coordinate, Fraction(1))
        for bad in ({}, {**a, "gate": 1.0}, {**a, "extra": Fraction(1)}):
            with self.assertRaises(ValueError):
                d.account(bad, a, 0, Fraction(1))
        with self.assertRaises(ValueError):
            d.account(a, a, 0, 1.0)

    def test_11_archive_shapes_dtypes_nonfinite_and_state_kv_gates(self):
        archive = self.archives[d.parent.CONTROLS[0]]
        for stage in ("stage14", "stage15", "stage16"):
            for value in (np.zeros(4863, dtype="<u2"), np.zeros(4864, dtype="<f2"),
                          np.full(4864, 0x7c00, dtype="<u2"), np.full(4864, 0xfe00, dtype="<u2")):
                with self.assertRaises(ValueError):
                    d.operands({**archive, stage: value})
        for key, value in (("input_i", np.zeros(896, dtype="<i4")),
                           ("scratch_z", np.full(896, 2, dtype="u1")),
                           ("input_cache_k", np.zeros((1, 128), dtype="<u2"))):
            with self.assertRaises(ValueError):
                d.residual.operands({**archive, key: value}, actual=True)
        bad = archive["output_cache_v"].copy()
        bad.flat[0] ^= 1
        with self.assertRaises(ValueError):
            d.residual.operands({**archive, "output_cache_v": bad}, actual=True)

    def test_12_original_reference_and_lineage_splices_rejected(self):
        original = self.previous[0][0][0]
        for key, value in (("prior_kv", "substituted"), ("fp16", {})):
            bad = deepcopy(original)
            bad["preflight"]["L23_original_reference"]["reference"][key] = value
            with self.assertRaises(ValueError):
                d.residual.load_residuals(bad)
        # Exercise cross-report splice guards without rerunning the large schema walk.
        for field in ("coordinate", "actual_projection_output", "input_delta"):
            bad = deepcopy(self.previous[4])
            selected = bad["controls"][0]["branches"][0]["hotspots"][0]["stage16_coordinates"][0]
            if field == "coordinate":
                selected["gate_proj"]["output_coordinate"] = -1
            elif field == "actual_projection_output":
                selected["gate_proj"][field] = "1234567"
            else:
                selected["selected_down_input"][field] = "1234567"
            with patch.object(jsonschema.Draft202012Validator, "validate"):
                with self.assertRaises(ValueError):
                    d.report(bad, self.actual, self.reference)

    def test_13_forbidden_replay_and_earlier_checks(self):
        audit = {"forbidden_calls": 0}
        calls = [
            lambda: d.residual.native._stages(None, None, None, None),
            lambda: d.residual.native.projection(None, None, None),
            lambda: d.residual.native.toward_zero(None),
            lambda: d.residual.local.local_reference(None, None, None, None),
            lambda: d.parent.execute(None), lambda: d.parent.rmsnorm(None, None),
            lambda: d.parent.logits(None, None), lambda: d.gate_up.check(),
            lambda: d.gate_up.focused_tests(None), lambda: d.down.check(),
            lambda: d.residual.measure(), lambda: d.channels.check(), lambda: d.base.check(),
            lambda: math.exp(1), lambda: np.exp(1), lambda: subprocess.Popen(["false"]),
            lambda: os.system("false"),
        ]
        with d.read_only(audit):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit["forbidden_calls"], len(calls))

    def test_14_forbidden_writes(self):
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

    def test_15_stdout_only_cli_and_execution_options_rejected(self):
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

    def test_16_exact_command_account_and_source_gates(self):
        self.assertEqual(d.COMMAND,
                         "PYTHONPATH=/home/argustest/ace3-argus PYTHONDONTWRITEBYTECODE=1 "
                         "/home/argustest/miniconda3/bin/python -B -m "
                         "ace3.model.candidates.diagnose_q24_s16_final_head_l23_mlp_silu_product_bridge_v1 --check")
        with patch.object(os, "getuid", return_value=0):
            with self.assertRaises(ValueError):
                d.check()
        pin = d.parent.record(d.SOURCE)
        with self.assertRaises(ValueError):
            d.base.read_bound({**pin, "sha256": "0" * 64})


if __name__ == "__main__":
    unittest.main()
