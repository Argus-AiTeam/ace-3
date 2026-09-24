"""Exact-cell, full-vector and read-only boundary checks; no native execution."""

import copy
from fractions import Fraction
import io
import json
import os
import subprocess
import sys
from types import ModuleType
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import (
    diagnose_q24_s16_l20_to_l21_coordinate62_admissible_threshold_v1 as d,
)


class AdmissibleThresholdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads(d.CONTRACT.read_text())
        cls.integer = 26514683648
        cls.reference = Fraction.from_float(float.fromhex("0x1.8b12540d033edp+10"))
        cls.parent = {
            "i": np.zeros(896, dtype="<i8"),
            "z": np.zeros(896, dtype="u1"),
            "h": np.zeros(896, dtype="<u2"),
        }
        cls.parent["i"][62] = cls.integer
        cls.parent["h"][62] = 0x662C
        cls.parent["z"][0] = 1
        cls.parent["h"][0] = 0x8000
        cls.reference_vector = np.zeros(896, dtype="<f8")
        cls.reference_vector[62] = float(cls.reference)
        cls.trajectory = {"stage18": cls.parent["h"].copy()}

    def test_contract(self):
        d.check_contract(self.contract)
        self.assertEqual(self.contract["interface"], ["--check"])
        self.assertFalse(self.contract["candidate_admitted"])

    def test_contract_changes_rejected(self):
        for key, value in (("excess_budget", "3/4"), ("policy_id", "other"),
                           ("native_layer_invocations", False), ("focused_tests", 19),
                           ("normal_host_review", "OPTIONAL"), ("interface", ["--execute"])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_contract({**self.contract, key: value})
        for section, key in (("method", "coordinate"), ("local_gate", "ordered_FP16_ULP")):
            changed = copy.deepcopy(self.contract)
            changed[section][key] = True
            with self.assertRaises(ValueError):
                d.check_contract(changed)

    def test_context(self):
        self.assertEqual(d.upstream.producer.context()["executable"], str(d.PYTHON))
        with patch.dict(os.environ, {"PYTHONPATH": "/foreign"}):
            with self.assertRaisesRegex(ValueError, "repo-bound"):
                d.upstream.producer.context()

    def test_origins(self):
        records = d.origins()
        for name in (d.MODULE, d.TEST_MODULE, d.upstream.MODULE, d.upstream.LEGACY_TEST):
            self.assertEqual(records[name]["path"],
                             str(d.ROOT.joinpath(*name.split(".")).with_suffix(".py")))

    def test_foreign_origin(self):
        module = ModuleType("ace3.model.candidates.foreign")
        module.__file__ = "/foreign/source.py"
        with patch.dict(sys.modules, {module.__name__: module}):
            with self.assertRaisesRegex(ValueError, "origin mismatch"):
                d.origins()
        with patch.object(d, "__file__", "/foreign/source.py"):
            with self.assertRaisesRegex(ValueError, "origin mismatch"):
                d.origins()

    def test_evidence_digest_before_read(self):
        inputs = d.margin.margin.prior.BoundInputs()
        with patch.object(d.retained, "record", return_value={"sha256": "0" * 64}):
            with patch.object(inputs, "bind") as bind:
                with self.assertRaisesRegex(ValueError, "pinned sensitivity"):
                    d.read_observation(inputs)
                bind.assert_not_called()

    def test_input_binding(self):
        inputs = d.margin.margin.prior.BoundInputs()
        record = d.retained.record(d.SOURCE)
        inputs.bind(record)
        for key, value in (("sha256", "0" * 64), ("bytes", 0), ("path", "/foreign/source.py")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                inputs.bind({**record, key: value})

    def test_frozen_observation_rejects_relabeling(self):
        # Failure at the first immutable field rejects before reading any samples.
        for value in ("PREFLIGHT_VALIDATED", "PASS", "SEARCH_EXHAUSTED"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "status"):
                d.check_observation({"diagnostic_id": d.upstream.ID, "status": value}, {}, {})
        with self.assertRaisesRegex(ValueError, "reason"):
            d.check_observation({"diagnostic_id": d.upstream.ID, "status": "INCONCLUSIVE",
                                 "reason": "SEARCH_EXHAUSTED"}, {}, {})

    def test_exact_selected_interval(self):
        result = d.interval(self.integer, self.reference)
        self.assertEqual(result["first_passing_FP16_word"], "662c")
        self.assertEqual(result["last_passing_FP16_word"], "662c")
        self.assertEqual(result["minimum_Q24_integer"], 3159 * (1 << 23))
        self.assertEqual(result["maximum_Q24_integer"], 3161 * (1 << 23))
        self.assertEqual(result["admissible_Q24_integer_count"], (1 << 24) + 1)
        self.assertEqual(result["reference_binary64_hex"], "0x1.8b12540d033edp+10")
        self.assertEqual(result["excess_budget"], "1/8")
        self.assertTrue(result["cells"][0]["lower_tie_inclusive"])
        self.assertTrue(result["cells"][0]["upper_tie_inclusive"])

    def test_selected_endpoints_and_exact_deltas(self):
        result = d.interval(self.integer, self.reference)
        rows = result["candidate_endpoints"]
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(rows[0]["Q24_integer"], rows[1]["Q24_integer"])
        self.assertEqual(rows[0]["input_FP16_word"], rows[1]["input_FP16_word"])
        for row in rows:
            self.assertEqual(row["delta_Q24_units"], row["Q24_integer"] - self.integer)
            self.assertEqual(Fraction(row["delta_exact"]),
                             Fraction(row["delta_Q24_units"], 1 << 24))
        self.assertEqual(result["baseline_control_delta_Q24_units"], 0)
        self.assertFalse(result["L21_search_coverage_complete"])
        self.assertFalse(result["minimality_proven"])

    def test_odd_tie_exclusion(self):
        result = d.interval(2050 * d.Q, Fraction(2050))
        self.assertEqual(result["first_passing_FP16_word"], "6801")
        self.assertEqual(result["last_passing_FP16_word"], "6801")
        self.assertEqual(result["minimum_Q24_integer"], 2049 * d.Q + 1)
        self.assertEqual(result["maximum_Q24_integer"], 2051 * d.Q - 1)
        self.assertFalse(result["cells"][0]["lower_tie_inclusive"])
        self.assertFalse(result["cells"][0]["upper_tie_inclusive"])

    def test_multiple_cells_independent_fp16_oracle(self):
        result = d.interval(64 * d.Q, Fraction(64))
        words = np.arange(0x5000, 0x5800, dtype="<u2")
        values = words.view("<f2").astype(np.float64)
        passing = words[np.abs(values - 64.0) <= 0.125]
        self.assertEqual([int(row["input_FP16_word"], 16) for row in result["cells"]],
                         passing.tolist())
        self.assertEqual(result["candidate_endpoint_count"], 2 * len(passing))
        for cell in result["cells"]:
            word = int(cell["input_FP16_word"], 16)
            for integer in (cell["minimum_Q24_integer"], cell["maximum_Q24_integer"]):
                # All selected dyadic inputs are exact in binary64; no double rounding.
                expected = int(np.array([integer / d.Q], dtype="<f2").view("<u2")[0])
                self.assertEqual(expected, word)
        for left, right in zip(result["cells"], result["cells"][1:]):
            self.assertEqual(left["maximum_Q24_integer"] + 1, right["minimum_Q24_integer"])

    def test_invalid_interval_domain(self):
        for integer, reference in ((True, self.reference), (1 << 63, self.reference),
                                   (self.integer, float(self.reference)),
                                   (self.integer, Fraction(1, 3)),
                                   (self.integer, Fraction(-1580)),
                                   (self.integer, Fraction(65504))):
            with self.subTest(integer=integer, reference=reference), self.assertRaises(ValueError):
                d.interval(integer, reference)

    def test_coordinate_copy_preserves_zero_sign_and_parent(self):
        before = {key: value.copy() for key, value in self.parent.items()}
        for integer in (self.integer, self.integer + 1, 3159 * (1 << 23), 3161 * (1 << 23)):
            changed = d.change_coordinate(self.parent, integer)
            self.assertEqual(int(changed["i"][62]), integer)
            self.assertEqual(int(changed["h"][62]), 0x662C)
            self.assertTrue(np.array_equal(changed["z"], self.parent["z"]))
            for key in self.parent:
                self.assertFalse(np.shares_memory(changed[key], self.parent[key]))
                self.assertTrue(np.array_equal(np.delete(changed[key], 62),
                                               np.delete(self.parent[key], 62)))
                self.assertTrue(np.array_equal(self.parent[key], before[key]))

    def test_invalid_state_and_integer(self):
        for integer in (True, np.int64(1), 1.0, "1", 1 << 63, -(1 << 63) - 1):
            with self.subTest(integer=integer), self.assertRaises(ValueError):
                d.change_coordinate(self.parent, integer)
        for key in ("i", "z", "h"):
            parent = {name: value.copy() for name, value in self.parent.items()}
            parent[key][62] = 0
            if key == "z":
                parent[key][62] = 2
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.change_coordinate(parent, self.integer)

    def test_full_vector_boundaries(self):
        result = d.analyze(self.parent, self.trajectory, self.reference_vector)
        for label, gate in result["full_vector_gate_checks"].items():
            self.assertEqual(gate["coordinates"], 896)
            self.assertEqual(gate["failed_indices"], [62] if label.startswith("one_") else [])
            self.assertEqual(gate["status"], "FAIL" if label.startswith("one_") else "PASS")

    def test_other_coordinate_failure_and_reference_substitution(self):
        reference = self.reference_vector.copy()
        reference[7] = 1
        with self.assertRaisesRegex(ValueError, "full-vector L20"):
            d.analyze(self.parent, self.trajectory, reference)
        with self.assertRaisesRegex(ValueError, "reference substitution"):
            d.margin.check_reference({"original_reference": {"input_binary64": "original"}},
                                     {"input_binary64": "actual"})

    def test_native_and_external_execution_guards(self):
        with d.no_execution():
            for layer in range(24):
                with self.subTest(layer=layer), self.assertRaisesRegex(RuntimeError, "forbidden"):
                    d.upstream.producer.native.candidate._stages(layer=layer)
            for function, args in ((subprocess.Popen, (["verilator"],)),
                                   (os.system, ("verilator",)), (d.retained.save, ())):
                with self.assertRaisesRegex(RuntimeError, "forbidden"):
                    function(*args)

    def test_check_only_interface(self):
        for argv in ([], ["--execute"], ["--check", "--execute"], ["--check", "--out", "existing"]):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    d.main(argv)
                self.assertEqual(raised.exception.code, 2)

    def test_blocked_is_not_success(self):
        stdout = io.StringIO()
        with patch.object(d, "check", side_effect=ValueError("binding mismatch")):
            with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
                self.assertEqual(d.main(["--check"]), 1)
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIn("binding mismatch", result["error"])
        for name in ("native_L0_L8_invocations", "native_L9_plus_invocations",
                     "native_layer_invocations", "rtl_invocations"):
            self.assertEqual(result[name], 0)
        for name in ("candidate_admitted", "policy_adopted", "successor_published"):
            self.assertIs(result[name], False)


if __name__ == "__main__":
    unittest.main()
