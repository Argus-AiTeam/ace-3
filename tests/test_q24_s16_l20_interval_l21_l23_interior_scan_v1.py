"""Strict-interior preflight regressions; no native or upstream test execution."""

import copy
from contextlib import redirect_stderr
from fractions import Fraction
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l20_interval_l21_l23_interior_scan_v1 as d


class InteriorScanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads(d.CONTRACT.read_text())
        cls.document = json.loads((d.INPUT / "diagnostic.stdout.json").read_text())
        cls.terminal = json.loads((d.INPUT / "terminal.json").read_text())
        cls.preparation = json.loads((d.endpoint.INPUT / "diagnostic.stdout.json").read_text())
        cls.parent = {
            "i": np.zeros(896, dtype="<i8"), "z": np.zeros(896, dtype="u1"),
            "h": np.zeros(896, dtype="<u2"),
        }
        cls.parent["i"][62] = 26514683648
        cls.parent["h"][62] = 0x662C
        cls.parent["z"][0] = 1
        cls.parent["h"][0] = 0x8000

    def test_contract_and_only_check_interface(self):
        d.check_contract(self.contract)
        self.assertEqual(self.contract["interface"], ["--check"])
        self.assertEqual(self.contract["reproduction"], d.COMMAND)

    def test_contract_threshold_reference_scope_and_flags(self):
        for key, value in (("interface", ["--execute"]), ("focused_tests", 19),
                           ("excess_budget", "1/4"), ("reference_policy", "local"),
                           ("native_layers", [21]), ("reproduction", "python --check")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_contract({**self.contract, key: value})
        for key, value in d.FLAGS.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_contract({**self.contract, key: not value})
        for section, key, value in (("method", "probe_count", 16),
                                    ("local_gate", "ordered_FP16_ULP", True)):
            changed = copy.deepcopy(self.contract)
            changed[section][key] = value
            with self.assertRaises(ValueError):
                d.check_contract(changed)

    def test_repo_bound_context(self):
        self.assertEqual(d.upstream.producer.context()["executable"], str(d.PYTHON))
        with patch.dict(os.environ, {"PYTHONPATH": "/foreign"}):
            with self.assertRaises(ValueError):
                d.upstream.producer.context()
        with patch.object(Path, "cwd", return_value=Path("/foreign")):
            with self.assertRaises(ValueError):
                d.upstream.producer.context()

    def test_candidate_and_test_source_origins(self):
        records = d.origins()
        for name in (d.MODULE, d.TEST_MODULE, d.endpoint.MODULE, d.upstream.LEGACY_TEST):
            self.assertEqual(records[name]["path"],
                             str(d.ROOT.joinpath(*name.split(".")).with_suffix(".py")))

    def test_foreign_origin_rejected(self):
        module = ModuleType("ace3.model.candidates.foreign_interior")
        module.__file__ = "/foreign/source.py"
        with patch.dict(sys.modules, {module.__name__: module}):
            with self.assertRaisesRegex(ValueError, "origin mismatch"):
                d.origins()

    def test_pinned_evidence_rejected_before_parse(self):
        inputs = d.prepared.margin.margin.prior.BoundInputs()
        with patch.object(d.retained, "record", return_value={"sha256": "0" * 64}):
            with patch.object(inputs, "read") as read:
                with self.assertRaisesRegex(ValueError, "pinned evidence"):
                    d.v1.pinned(inputs, d.INPUT / "terminal.json", d.PINS["terminal.json"])
                read.assert_not_called()

    def test_exact_schedule_against_independent_rational_oracle(self):
        lo, hi, baseline = 26499612672, 26516389888, 26514683648
        actual = d.interior_points(lo, hi, baseline)
        expected = [int(Fraction(16 - k, 16) * lo + Fraction(k, 16) * hi)
                    for k in range(1, 16)]
        self.assertEqual(actual, expected)
        self.assertEqual(actual[0], 26500661248)
        self.assertEqual(actual[-1], 26515341312)
        self.assertEqual(len(actual), 15)
        self.assertEqual(set(b - a for a, b in zip(actual, actual[1:])), {1048576})

    def test_strict_interior_and_executed_endpoint_exclusion(self):
        analysis = self.document["analysis"]
        points = d.interior_points(analysis["minimum_Q24_integer"],
                                   analysis["maximum_Q24_integer"],
                                   analysis["baseline_Q24_integer"])
        self.assertEqual(points, sorted(set(points)))
        self.assertTrue(all(analysis["minimum_Q24_integer"] < p
                            < analysis["maximum_Q24_integer"] for p in points))
        self.assertNotIn(analysis["baseline_Q24_integer"], points)
        self.assertTrue(set(points).isdisjoint(
            row["Q24_integer"] for row in self.document["candidate_set"]))

    def test_empty_reversed_small_and_baseline_collisions_rejected(self):
        for bounds in ((10, 10, 10), (20, 10, 15), (0, 15, 7), (0, 16, 8),
                       (0, 17, 0), (0, 17, 17)):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                d.interior_points(*bounds)

    def test_noninteger_and_overflow_bounds_rejected(self):
        for value in (True, 1.0, "1", None, 1 << 63, -(1 << 63) - 1):
            for index in range(3):
                bounds = [0, 1000, 999]
                bounds[index] = value
                with self.subTest(value=value, index=index), self.assertRaises(ValueError):
                    d.interior_points(*bounds)

    def test_distinct_q24_states_with_identical_h(self):
        rows = d.schedule(self.document["analysis"], self.parent)
        self.assertEqual(len({row["state_binding"]["i"]["sha256"] for row in rows}), 15)
        self.assertEqual(len({row["state_binding"]["h"]["sha256"] for row in rows}), 1)
        self.assertEqual({row["input_FP16_word"] for row in rows}, {"662c"})
        self.assertEqual({row["status"] for row in rows}, {"NOT_EXECUTED"})
        self.assertEqual([row["probe_index"] for row in rows], list(range(15)))

    def test_coordinate_only_copy_preserves_parent_and_zero_tags(self):
        before = {key: value.copy() for key, value in self.parent.items()}
        changed = d.prepared.change_coordinate(self.parent, 26500661248)
        for key in self.parent:
            self.assertTrue(np.array_equal(self.parent[key], before[key]))
            self.assertFalse(np.shares_memory(self.parent[key], changed[key]))
        self.assertEqual(np.flatnonzero(changed["i"] != before["i"]).tolist(), [62])
        self.assertTrue(np.array_equal(changed["z"], before["z"]))
        self.assertTrue(np.array_equal(changed["h"], before["h"]))

    def test_schedule_rejects_parent_cell_and_interval_substitution(self):
        for key, value in (("baseline_Q24_integer", 26514683649), ("cells", []),
                           ("minimum_Q24_integer", 26499612671)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.schedule({**self.document["analysis"], key: value}, self.parent)
        changed = copy.deepcopy(self.document["analysis"])
        changed["cells"][0]["input_FP16_word"] = "662d"
        with self.assertRaises(ValueError):
            d.schedule(changed, self.parent)
        bad = {key: value.copy() for key, value in self.parent.items()}
        bad["h"][62] = 0x662D
        with self.assertRaises(ValueError):
            d.schedule(self.document["analysis"], bad)

    def test_frozen_endpoint_history_preserved(self):
        d.check_history(self.document, self.terminal, self.preparation)
        self.assertEqual(self.document["status"], "ENDPOINT_GATE_FAIL")
        self.assertEqual(self.terminal["status"], "EXHAUSTED_NO_PASS")
        self.assertEqual(self.document["historical_sensitivity"]["native_layer_invocations"], 43)

    def test_history_rejects_reference_state_lineage_and_interval_changes(self):
        for key in ("analysis", "parent_binding", "original_L20_reference", "lineage",
                    "reference_policy", "historical_sensitivity", "historical_preparation"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_history({**self.document, key: None}, self.terminal, self.preparation)

    def test_history_rejects_relabelled_results_and_invocations(self):
        for key, value in (("status", "PASS"), ("native_layer_invocations", 0),
                           ("native_L0_L8_invocations", 1), ("rtl_invocations", 1),
                           ("native_layers", [21, 22]), ("candidate_admitted", True)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_history({**self.document, key: value}, self.terminal, self.preparation)
        for key, value in (("status", "PASS"), ("returncode", 0),
                           ("historical_artifacts_unchanged", False)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_history(self.document, {**self.terminal, key: value}, self.preparation)

    def test_history_rejects_executed_endpoint_substitution(self):
        for key, value in (("Q24_integer", 26500661248), ("cell_endpoint", "interior"),
                           ("state_binding", {}), ("first_failure", None)):
            changed = copy.deepcopy(self.document)
            changed["candidate_set"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_history(changed, self.terminal, self.preparation)

    def test_history_rejects_full_vector_gate_and_kv_corruption(self):
        for field in ("parent_gate", "local_gate", "state", "kv", "continued"):
            changed = copy.deepcopy(self.document)
            row = changed["candidate_set"][0]
            if field == "parent_gate":
                row["L20_global_gate"]["binary64_v1"]["coordinates"] = 1
            elif field == "local_gate":
                row["layers"][0]["reports"][0]["local_operator_fp16"]["passed"] = False
            elif field == "state":
                row["layers"][0]["reports"][0]["residual_state_lineage"] = "FAIL"
            elif field == "kv":
                row["layers"][0]["prior_layer_kv_consumed"] = True
            else:
                row["layers"].append(row["layers"][0])
            with self.subTest(field=field), self.assertRaises(ValueError):
                d.check_history(changed, self.terminal, self.preparation)

    def test_native_dispatch_is_blocked_including_l0_l8(self):
        with d.v1.no_execution():
            for layer in (0, 8, 9, 20, 21, 22, 23):
                with self.subTest(layer=layer), self.assertRaisesRegex(RuntimeError, "forbidden"):
                    d.endpoint.native.candidate._stages(None, layer, None, {})

    def test_external_execution_and_state_publication_blocked(self):
        with d.v1.no_execution():
            for call in (lambda: subprocess.Popen(["forbidden"]),
                         lambda: os.system("forbidden"),
                         lambda: np.savez("forbidden.npz", x=np.zeros(1))):
                with self.assertRaisesRegex(RuntimeError, "forbidden"):
                    call()

    def test_exact_focused_count_and_zero_failures_errors_skips(self):
        outcome = unittest.TestResult()
        outcome.testsRun = d.EXPECTED_TESTS
        d.check_test_result(outcome, d.EXPECTED_TESTS)
        for count in (0, 19, d.EXPECTED_TESTS - 1, d.EXPECTED_TESTS + 1):
            with self.assertRaises(ValueError):
                d.check_test_result(outcome, count)
        for field in ("failures", "errors", "skipped"):
            bad = unittest.TestResult()
            bad.testsRun = d.EXPECTED_TESTS
            getattr(bad, field).append((self, "synthetic rejected outcome"))
            with self.subTest(field=field), self.assertRaises(ValueError):
                d.check_test_result(bad, d.EXPECTED_TESTS)
        outcome.testsRun -= 1
        with self.assertRaises(ValueError):
            d.check_test_result(outcome, d.EXPECTED_TESTS)

    def test_execute_interface_rejected_before_any_check(self):
        with patch.object(d, "check") as check, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                d.main(["--check", "--execute"])
            self.assertEqual(raised.exception.code, 2)
            check.assert_not_called()


if __name__ == "__main__":
    unittest.main()
