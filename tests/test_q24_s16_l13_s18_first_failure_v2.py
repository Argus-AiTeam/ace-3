"""Synthetic focused regressions; no native accepted-layer execution."""

import ast
import copy
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l13_s18_first_failure_v2 as d
from tests.test_q24_s16_l13_s18_first_failure_v1 import lineage_fixture, reports_fixture


EXPECTED_TESTS = 16


def reference_fixture():
    extension = {
        "reference_only": True, "accepted_ancestor_oracle_replays": 0,
        "policy_id": d.gates.POLICY_ID,
        "global_reference_policy": "legacy-binary64-AWQ-fully-independent-propagation",
        "original_binary64_parent": "original64", "original_fp16_parent": "original16",
        "layers": {},
    }
    previous64, previous16 = "original64", "original16"
    for layer in range(9, 14):
        item = {"input_binary64": previous64, "input_fp16": previous16,
                "prior_kv": "own empty P0", "binary64": f"global{layer}", "fp16": f"fp16{layer}"}
        extension["layers"][str(layer)] = item
        previous64, previous16 = item["binary64"], item["fp16"]
    return extension


class DiagnosticTests(unittest.TestCase):
    def test_mandatory_selection(self):
        result, reports = reports_fixture()
        self.assertEqual(d.select_failure(result, reports)["index"], 62)

    def test_reject_wrong_failure(self):
        for mutation in ("index", "missing", "earlier", "policy"):
            result, reports = reports_fixture()
            if mutation == "index":
                result["first_failure"]["index"] = 0
            elif mutation == "missing":
                reports[18]["binary64_v1"]["failures"] = []
            elif mutation == "earlier":
                reports[0]["status"] = "FAIL"
            else:
                reports[18]["policy_id"] = "other"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                d.select_failure(result, reports)

    def test_exact_failure(self):
        row = d.measure(0x6630, float.fromhex("0x1.8bd7b2092532cp+10"))
        self.assertEqual(Fraction(row["q"]), Fraction(407084766411, 1099511627776))
        self.assertEqual(Fraction(row["excess_error"]), Fraction(142671047477, 549755813888))
        self.assertFalse(row["accepted"])
        self.assertEqual(Fraction(row["excess_over_budget"]),
                         Fraction(row["excess_error"]) - Fraction(1, 8))

    def test_gate_boundaries(self):
        self.assertTrue(d.measure(0x3c80, 1.0)["accepted"])
        self.assertFalse(d.measure(0x3c81, 1.0)["accepted"])
        self.assertEqual(Fraction(d.measure(0x6400, 1024.5)["q"]), Fraction(1, 2))
        self.assertEqual(Fraction(d.measure(0x8000, -0.0)["q"]), 0)
        for value in (float("nan"), float("inf"), 65505.0):
            with self.assertRaises(ValueError):
                d.measure(0, value)

    def test_rtz_reference(self):
        values = np.array([1.0008, -1.0008, 2**-25, -2**-25, 0.0, -0.0], dtype="<f8")
        self.assertEqual(d.rtz_reference(values).tolist(), [0x3c00, 0xbc00, 0, 0x8000, 0, 0x8000])

    def test_lineage_valid(self):
        self.assertEqual(d.check_lineage(*lineage_fixture())["L12_output_to_L13_input_IZH"], "PASS")

    def test_lineage_mutations(self):
        for key in ("input_i", "input_z", "input_hidden", "scratch_i", "scratch_z",
                    "stage12", "output_i", "output_z", "stage18", "stage11", "stage17",
                    "output_cache_k", "output_cache_v", "stage16"):
            arrays, parent, parent_arrays, kv = lineage_fixture()
            arrays[key].flat[0] = 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_lineage(arrays, parent, parent_arrays, kv)
        arrays, parent, parent_arrays, kv = lineage_fixture()
        kv["k"][0, 0] = 1
        with self.assertRaises(ValueError):
            d.check_lineage(arrays, parent, parent_arrays, kv)

    def test_source_binding(self):
        with tempfile.TemporaryDirectory(dir=d.ROOT / "build", prefix="q24_s16_l13_test_") as tmp:
            path = Path(tmp) / "source"
            path.write_text("original")
            record = d.retained.record(path)
            inputs = d.BoundInputs()
            self.assertEqual(inputs.bind(record), path)
            changed = {**record, "sha256": "0" * 64}
            with self.assertRaises(ValueError):
                inputs.bind(changed)
            path.write_text("changed")
            with self.assertRaises(ValueError):
                d.BoundInputs().bind(record)

    def test_external_and_symlink_input_rejected(self):
        with tempfile.TemporaryDirectory(dir=d.ROOT / "build", prefix="q24_s16_l13_test_") as tmp:
            link = Path(tmp) / "external"
            link.symlink_to("/etc/hosts")
            for path in (link, Path("/etc/hosts"), Path("relative")):
                with self.assertRaises(ValueError):
                    d.BoundInputs().bind({"path": str(path), "bytes": 0, "sha256": "unused"})

    def test_counters(self):
        document = {"native_L0_L8_invocations": 0, "rtl_invocations": 0}
        d.check_counters(document)
        for name in document:
            for value in (1, -1, False, "0", None):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    d.check_counters({**document, name: value})
            missing = dict(document)
            del missing[name]
            with self.assertRaises(KeyError):
                d.check_counters(missing)

    def test_original_reference_chain(self):
        d.check_reference_chain(reference_fixture())

    def test_reference_reanchoring_rejected(self):
        baseline = reference_fixture()
        for field in ("input_binary64", "input_fp16", "prior_kv"):
            extension = copy.deepcopy(baseline)
            extension["layers"]["11"][field] = "candidate"
            with self.assertRaises(ValueError):
                d.check_reference_chain(extension)
        for field, value in (("reference_only", False), ("accepted_ancestor_oracle_replays", 1),
                             ("policy_id", "other"), ("global_reference_policy", "local")):
            with self.assertRaises(ValueError):
                d.check_reference_chain({**baseline, field: value})

    def test_signed_decomposition(self):
        arrays, parent, _, _ = lineage_fixture()
        parent_ref = np.zeros(896, dtype="<f8")
        global_ref = parent_ref.copy()
        parent_ref[62], global_ref[62] = 0.5, 0.25
        row = d.decomposition(arrays, parent, parent_ref, global_ref, arrays["stage17"], 62)
        self.assertEqual(Fraction(row["L12_signed_Q24_drift"]), Fraction(-1, 2))
        self.assertEqual(Fraction(row["L13_net_increment_error"]), Fraction(1, 4))
        self.assertEqual(Fraction(row["signed_global_error"]), Fraction(-1, 4))
        arrays["output_i"][62] = 1
        with self.assertRaises(ValueError):
            d.decomposition(arrays, parent, parent_ref, global_ref, arrays["stage17"], 62)

    def test_no_native_dispatch(self):
        tree = ast.parse(Path(d.__file__).read_text())
        calls = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
                 for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, (ast.Name, ast.Attribute))}
        self.assertFalse(calls & {"system", "Popen", "execute_layers", "_stages", "continuation_stages",
                                 "save", "savez", "savez_compressed", "_reference_layer_step"})

    def test_wrong_input(self):
        with self.assertRaises(ValueError):
            d.diagnose(d.prior.INPUT)

    def test_no_admission_or_unique_attribution(self):
        contract = json.loads(d.CONTRACT.read_text())
        for key in ("candidate_admitted", "policy_adopted", "successor_published"):
            self.assertIs(contract[key], False)
        self.assertEqual(contract["normal_host_review"], "REQUIRED")
        d.check_counters(contract)
        for rescued in (True, False):
            report = d.classify({"accepted": False}, {"accepted": rescued}, {"accepted": rescued})
            self.assertEqual(report["classification"], "inconclusive")
            self.assertEqual(report["S17_local_reference_substitution_rescues"], rescued)


if __name__ == "__main__":
    unittest.main()
