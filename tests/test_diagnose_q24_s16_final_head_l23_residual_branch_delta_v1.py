"""Independent FP16 bit-decoding and exact residual-boundary oracles."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_l23_residual_branch_delta_v1 as d


EVIDENCE = None


def half(word):
    sign = -1 if word & 0x8000 else 1
    exponent, mantissa = (word >> 10) & 31, word & 1023
    if exponent == 31:
        raise ValueError("nonfinite FP16")
    return sign * (Fraction(mantissa, 1 << 24) if exponent == 0
                   else Fraction(1024 + mantissa) * Fraction(2) ** (exponent - 25))


class ResidualBranchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is None:
            with d.read_only({"forbidden_calls": 0}):
                cls.evidence, _, _ = d.measure()
        else:
            cls.evidence = EVIDENCE
        (cls.channel_evidence, cls.actual, cls.reference,
         cls.archives, cls.reference_archive, cls.report) = cls.evidence
        cls.result = cls.channel_evidence[0]

    def accounts(self):
        for row in self.report["controls"]:
            for branch in row["branches"]:
                for hotspot in branch["hotspots"]:
                    yield row["control"], branch["final_head_reference_branch"], hotspot

    def test_01_all_boundary_deltas_independent_bits(self):
        for label, _, hotspot in self.accounts():
            account = hotspot["residual"]
            i = account["coordinate"]
            for stage in d.STAGES:
                a = half(int(self.archives[label][stage][i]))
                r = half(int(self.reference_archive[stage][i]))
                self.assertEqual(account["boundaries"][stage], {
                    "actual_exact": str(a), "reference_exact": str(r),
                    "actual_minus_reference": str(a - r)})

    def test_02_exact_residual_remainders(self):
        for label, _, hotspot in self.accounts():
            account = hotspot["residual"]
            i = account["coordinate"]
            a = [half(int(self.archives[label][stage][i])) for stage in d.STAGES]
            r = [half(int(self.reference_archive[stage][i])) for stage in d.STAGES]
            ar, rr = a[3] - sum(a[:3]), r[3] - sum(r[:3])
            delta_sum = sum(a[:3]) - sum(r[:3])
            self.assertEqual(Fraction(account["branch_delta_sum"]), delta_sum)
            self.assertEqual(Fraction(account["actual_boundary_remainder"]), ar)
            self.assertEqual(Fraction(account["reference_boundary_remainder"]), rr)
            self.assertEqual(Fraction(account["boundary_remainder_delta"]), ar - rr)
            self.assertEqual(a[3] - r[3], delta_sum + ar - rr)

    def test_03_actual_q24_parts_independent_integer_oracle(self):
        for label, _, hotspot in self.accounts():
            account, archive = hotspot["residual"], self.archives[label]
            i = account["coordinate"]
            state = [Fraction(int(archive[key + "_i"][i]), 16777216)
                     for key in ("input", "scratch", "output")]
            h, attn, down, out = [half(int(archive[stage][i])) for stage in d.STAGES]
            expected = [state[0] - h, state[1] - state[0] - attn,
                        state[2] - state[1] - down, out - state[2]]
            self.assertEqual(account["actual_q24_boundary_parts"],
                             dict(zip(d.ACTUAL_PARTS, map(str, expected), strict=True)))
            self.assertEqual(sum(expected), out - h - attn - down)

    def test_04_reference_add_boundary_parts(self):
        for _, _, hotspot in self.accounts():
            account = hotspot["residual"]
            i = account["coordinate"]
            h, attn, down, out, scratch = [
                half(int(self.reference_archive[stage][i]))
                for stage in (*d.STAGES, "stage12")]
            expected = [scratch - h - attn, out - scratch - down]
            self.assertEqual(account["reference_fp16_boundary_parts"],
                             dict(zip(d.REFERENCE_PARTS, map(str, expected), strict=True)))
            self.assertEqual(sum(expected), Fraction(account["reference_boundary_remainder"]))

    def test_05_census_and_selected_hotspot_order(self):
        self.assertEqual([row["control"] for row in self.report["controls"]], list(d.parent.CONTROLS))
        count = 0
        for row, retained in zip(self.report["controls"], self.channel_evidence[5]["controls"], strict=True):
            self.assertEqual([b["final_head_reference_branch"] for b in row["branches"]],
                             list(d.channels.BRANCHES))
            for branch in row["branches"]:
                source = retained["branches"][branch["final_head_reference_branch"]]
                self.assertEqual(branch["numeric_id"], source["numeric_id"])
                self.assertEqual([h["hotspot_rank"] for h in branch["hotspots"]], list(range(1, 9)))
                selected = source["channels"]["full_absolute_signed_ranking"][:8]
                self.assertEqual([h["final_head_channel"] for h in branch["hotspots"]], selected)
                self.assertEqual([h["residual"]["coordinate"] for h in branch["hotspots"]],
                                 [s["coordinate"] for s in selected])
                count += len(selected)
        self.assertEqual(count, 144)
        self.assertEqual(count * len(d.STAGES), self.report["boundary_delta_count"])

    def test_06_fp16_decode_edge_cases(self):
        bits = [0, 0x8000, 1, 0x8001, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x7bff, 0xfbff]
        self.assertEqual(d.words(np.array(bits, dtype="<u2"), (len(bits),)),
                         [half(word) for word in bits])

    def test_07_invalid_boundaries_and_coordinates(self):
        for array in (np.zeros(896, dtype="<f2"), np.zeros(895, dtype="<u2"),
                      np.full(896, 0x7c00, dtype="<u2"), np.full(896, 0xfe00, dtype="<u2")):
            with self.assertRaises(ValueError):
                d.words(array)
        for index in (-1, 896, True, 0.5):
            with self.assertRaises(ValueError):
                d.account(self.actual[d.parent.CONTROLS[0]], self.reference, index)

    def test_08_q24_and_kv_tamper_rejected(self):
        original = self.archives[d.parent.CONTROLS[0]]
        mutations = (
            ("input_i", np.zeros(896, dtype="<i4")),
            ("output_z", np.full(896, 2, dtype="u1")),
            ("scratch_z", np.ones(896, dtype="u1")),
            ("input_cache_k", np.zeros((1, 128), dtype="<u2")),
        )
        for key, value in mutations:
            with self.assertRaises(ValueError):
                d.operands({**original, key: value}, actual=True)
        for key in ("output_cache_k", "output_cache_v", "stage03"):
            bad = deepcopy(original)
            bad[key].flat[0] ^= 1
            with self.assertRaises(ValueError):
                d.operands(bad, actual=True)

    def test_09_reference_scope_splices_rejected(self):
        for key, value in (("prior_kv", "other"), ("fp16", {})):
            bad = deepcopy(self.result)
            bad["preflight"]["L23_original_reference"]["reference"][key] = value
            with self.assertRaises(ValueError):
                d.load_residuals(bad)

    def test_10_archive_hash_tamper_and_duplicate_fields_rejected(self):
        for pin in (self.result["controls"][0]["parent"]["terminal_archive"],
                    self.result["preflight"]["L23_original_reference"]["reference"]["fp16"]):
            with self.assertRaises(ValueError):
                d.read_archive({**pin, "sha256": "0" * 64})
        with patch.object(d.base, "read_bound", return_value=b""), patch.object(d.np, "load") as load:
            load.return_value.__enter__.return_value.files = ["stage18", "stage18"]
            with self.assertRaises(ValueError):
                d.read_archive({})

    def test_11_controls_branches_and_hotspot_tamper_rejected(self):
        binding = self.result["preflight"]["L23_original_reference"]
        for actual in ({}, dict(reversed(list(self.actual.items())))):
            with self.assertRaises(ValueError):
                d.report(self.channel_evidence[5], actual, self.reference, binding)
        for kind in ("branch", "missing", "reordered", "duplicate"):
            bad = deepcopy(self.channel_evidence[5])
            branches = bad["controls"][0]["branches"]
            selected = branches["fp16"]["channels"]["largest_absolute_signed"]
            if kind == "branch":
                bad["controls"][0]["branches"] = dict(reversed(list(branches.items())))
            elif kind == "missing":
                selected.pop()
            elif kind == "reordered":
                selected.reverse()
            else:
                selected[1] = selected[0]
            with self.assertRaises(ValueError):
                d.report(bad, self.actual, self.reference, binding)

    def test_12_schema_and_json_round_trip(self):
        jsonschema.Draft202012Validator.check_schema(d.OUTPUT_SCHEMA)
        jsonschema.Draft202012Validator(d.REPORT_SCHEMA).validate(self.report)
        text = json.dumps(self.report, sort_keys=True, allow_nan=False)
        self.assertEqual(json.loads(text), self.report)

    def test_13_schema_rejects_count_order_scope_and_extra(self):
        validator = jsonschema.Draft202012Validator(d.REPORT_SCHEMA)
        for field, value in (("hotspot_count", 143), ("boundary_delta_count", 575),
                             ("layer", 22), ("position", 1), ("extra", 0),
                             ("controls", list(reversed(self.report["controls"])))):
            with self.assertRaises(jsonschema.ValidationError):
                validator.validate({**self.report, field: value})
        row = deepcopy(self.report["controls"][0])
        row["branches"].reverse()
        with self.assertRaises(jsonschema.ValidationError):
            validator.validate({**self.report, "controls": [row, *self.report["controls"][1:]]})
        branch = deepcopy(self.report["controls"][0]["branches"][0])
        branch["hotspots"].reverse()
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.Draft202012Validator(d.branch_schema("fp16")).validate(branch)
        account = next(self.accounts())[2]["residual"]
        for field, value in (("extra", 1), ("coordinate", 896),
                             ("branch_delta_sum", "nan"), ("residual_reference", "binary64"),
                             ("boundaries", {}), ("exact_residual_branch_identity", False)):
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.Draft202012Validator(d.ACCOUNT_SCHEMA).validate({**account, field: value})

    def test_14_report_determinism_fresh_arithmetic(self):
        repeated = d.report(self.channel_evidence[5], self.actual, self.reference,
                            self.result["preflight"]["L23_original_reference"])
        self.assertEqual(json.dumps(repeated, sort_keys=True, allow_nan=False),
                         json.dumps(self.report, sort_keys=True, allow_nan=False))

    def test_15_history_references_and_nonadmission_unchanged(self):
        retained = self.report["final_head_channel_report"]
        self.assertEqual(retained, self.channel_evidence[5])
        self.assertEqual(retained["thresholds"], self.result["preflight"]["thresholds"])
        self.assertEqual(retained["original_global_reference"], self.result["preflight"]["final_reference"])
        self.assertEqual(retained["retained_L23_stage_reports"], {"PASS": 162, "FAIL": 9})
        self.assertEqual(self.report["L23_reference"]["fp16"],
                         retained["original_global_reference"]["reference"]["input_fp16"])
        self.assertFalse(self.report["L23_reference"]["original_global_references_reanchored"])
        for row in retained["controls"]:
            self.assertEqual(row["L21_L22_L23_status"], ["FAIL"] * 3)
            self.assertTrue(all(row["retained_failures"].values()))
        for field, value in (("candidate_admitted", True),
                             ("source_operand_state_KV_RTZ_checks", "FAIL")):
            bad = deepcopy(self.result)
            bad["controls"][0]["parent"]["retained_L23"][field] = value
            with self.assertRaises(ValueError):
                d.base.check_history(bad)
        self.assertIn("wider than FP16", d.BOUNDARY)
        self.assertIn("not attributed solely to rounding", d.BOUNDARY)

    def test_16_synthetic_signed_nonzero_remainders(self):
        a = {key: [Fraction(value)] * 896 for key, value in {
            "input_hidden": 2, "stage11": -3, "stage17": 5, "stage18": 7,
            "stage12": 0, "input_q24": Fraction(5, 2),
            "scratch_q24": 1, "output_q24": 6}.items()}
        r = {key: [Fraction(value)] * 896 for key, value in {
            "input_hidden": 1, "stage11": 2, "stage17": -1,
            "stage18": 4, "stage12": 4}.items()}
        account = d.account(a, r, 62)
        self.assertEqual(account["branch_delta_sum"], "2")
        self.assertEqual(account["boundary_remainder_delta"], "1")
        self.assertEqual(account["actual_q24_boundary_parts"],
                         dict(zip(d.ACTUAL_PARTS, ("1/2", "3/2", "0", "1"), strict=True)))
        self.assertEqual(account["reference_fp16_boundary_parts"],
                         dict(zip(d.REFERENCE_PARTS, ("1", "1"), strict=True)))
        self.assertEqual(account["boundaries"]["stage18"]["actual_minus_reference"], "3")

    def test_17_forbidden_dispatch_audit(self):
        audit = {"forbidden_calls": 0}
        calls = [
            lambda: d.native.projection(None, None, None),
            lambda: d.native._stages(None, None, None, None),
            lambda: d.native.rne(None), lambda: d.native.toward_zero(None),
            lambda: d.local.local_reference(None, None, None, None),
            lambda: d.parent.execute(None), lambda: d.parent.rmsnorm(None, None),
            lambda: d.parent.logits(None, None), lambda: d.parent.preflight.parent.execute(None),
            lambda: d.channels.check(), lambda: d.channels.focused_tests(None),
            lambda: d.channels.hotspots.check(), lambda: d.base.check(),
            lambda: d.channels.contributions.dyadic_dot(None, None),
            lambda: subprocess.Popen(["false"]), lambda: os.system("false"),
            lambda: d.channels.contributions.preflight.parent.producer.legacy.torch.cuda._lazy_init(),
        ]
        with d.read_only(audit):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit["forbidden_calls"], len(calls))

    def test_18_forbidden_filesystem_audit(self):
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
                    "evidence_writes", "local_operator_replay", "earlier_checks"):
            self.assertEqual(d.FLAGS[key], 0)

    def test_19_cli_json_only_and_unsupported_modes(self):
        stream = io.StringIO()
        with patch.object(d, "check", return_value={"ok": True}), patch("sys.stdout", stream):
            d.main(["--check"])
        self.assertEqual(stream.getvalue(), '{"ok": true}\n')
        for argv in ([], ["--execute"], ["--check", "--output", "x"], ["--check", "--che"]):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                d.main(argv)

    def test_20_source_and_identity_preflight_fail_closed(self):
        with patch.object(d.os, "getuid", return_value=1), patch.object(d, "measure") as measure:
            with self.assertRaises(ValueError):
                d.check()
            measure.assert_not_called()
        for pin in (d.base.PINS["result"], d.channels.PINS["contribution_source"]):
            with self.assertRaises(ValueError):
                d.base.read_bound({**pin, "sha256": "0" * 64})


if __name__ == "__main__":
    unittest.main()
