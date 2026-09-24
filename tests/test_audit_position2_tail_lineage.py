"""Current tail runtime mapping, separate from the retained preparer diagnosis."""

import ast
import copy
from fractions import Fraction
import inspect
from pathlib import Path
import unittest
from unittest.mock import mock_open, patch

from ace3.model import prepare_position2_tail_binding as binding
from ace3.model import run_position2_tail as runtime


class CurrentTailRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.contract = binding.load(binding.CONTRACT)
        self.np = binding.np

    def compare(self, actual, reference):
        bits = self.np.asarray([actual, reference], dtype="<f2").view("<u2")
        return runtime.compare_rmsnorm(binding, bits[:1], bits[1:], self.contract)

    def test_contract_and_runtime_use_exact_current_policy(self):
        runtime.validate_rmsnorm_contract(self.contract)
        self.assertEqual(self.contract["schema"], "ace3_position2_tail_binding_v2")
        self.assertFalse(self.contract["final_rmsnorm"]["legacy_mathematical_diagnostic"]["admission_gate"])
        self.assertFalse(self.contract["admission"]["unchanged_tail_replay_authorized"])
        self.assertFalse(self.contract["admission"]["execution_authority"])

    def test_finite_controls_against_independent_fraction_oracle(self):
        cases = ((256, 256), (1, 1.125), (1024, 1025), (1000, 1001),
                 (1023.5, 1022.5), (-1000, -1001), (1000, 1002),
                 (0, 2**-24), (-0.0, 0.0), (2**-14, 0))
        for reference, actual in cases:
            with self.subTest(reference=reference, actual=actual):
                bits = self.np.asarray([actual, reference], dtype="<f2").view("<u2")
                def ordered(raw):
                    return 0x8000 - (raw & 0x7fff) if raw & 0x8000 else 0x8000 + raw
                distance = abs(ordered(int(bits[0])) - ordered(int(bits[1])))
                error = abs(Fraction(actual) - Fraction(reference))
                relative = error / max(abs(Fraction(reference)), Fraction(1, 2**14))
                accepted = error <= Fraction(1, 8) or (
                    relative < Fraction(1, 1000) and distance <= 1)
                result = self.compare(actual, reference)
                self.assertEqual(result["failure_count"], int(not accepted))
                self.assertEqual(result["max_relative_error"], float(relative))
                self.assertEqual(result["max_ulp_distance"], distance)

    def test_nonfinite_inputs_fail_on_either_side(self):
        for raw in (0x7c00, 0xfc00, 0x7e00, 0xfe00):
            bad = self.np.asarray([raw], dtype="<u2")
            good = self.np.asarray([0], dtype="<u2")
            for actual, reference in ((bad, good), (good, bad)):
                with self.assertRaisesRegex(binding.evidence.AttemptError, "nonfinite"):
                    runtime.compare_rmsnorm(binding, actual, reference, self.contract)

    def test_legacy_or_changed_contract_is_not_silently_adopted(self):
        for field in runtime.RMSNORM_COMPARISON:
            contract = copy.deepcopy(self.contract)
            contract["final_rmsnorm"]["comparison"][field] = "legacy"
            with self.assertRaisesRegex(RuntimeError, "contract/reference"):
                runtime.validate_rmsnorm_contract(contract)
        del contract["final_rmsnorm"]["comparison"]
        with self.assertRaisesRegex(RuntimeError, "contract/reference"):
            runtime.validate_rmsnorm_contract(contract)

    def test_runtime_consumes_interstage_reference_not_diagnostic_as_gate(self):
        source = inspect.getsource(runtime.execute)
        tree = ast.parse(source)
        gate = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                    and ast.unparse(node.test) == "len(differences) or current['failure_count']")
        self.assertNotIn("material", ast.unparse(gate.test))
        assignments = {node.targets[0].id: ast.unparse(node.value)
                       for node in ast.walk(tree) if isinstance(node, ast.Assign)
                       and isinstance(node.targets[0], ast.Name)}
        self.assertIn("fp16_interstage_rmsnorm_expected.hex", assignments["interstage"])
        self.assertEqual(assignments["current"],
                         "compare_rmsnorm(preparation, actual, interstage, contract)")
        self.assertIn("mathematical_rmsnorm_expected.hex", assignments["mathematical"])
        oracle_source = inspect.getsource(runtime.rms_expectation)
        self.assertNotIn("preparation.rms_expectation(", oracle_source)
        self.assertIn("preparation.evidence.fp16_interstage_expected", oracle_source)
        self.assertIn("preparation.independent_rmsnorm", oracle_source)

    def test_runtime_oracle_writes_distinct_active_and_diagnostic_references(self):
        bits = [0x3000 + (index * 37) % 0x1800 for index in range(896)]
        weights = self.np.asarray(
            [0x3800 + (index * 11) % 0x600 for index in range(896)], dtype="<u2")
        files = {}
        def open_output(path, *args, **kwargs):
            files[path.name] = []
            handle = mock_open()()
            handle.writelines.side_effect = files[path.name].extend
            return handle
        with patch.object(binding.np, "memmap", return_value=weights), \
                patch.object(Path, "open", open_output):
            expected, receipt = runtime.rms_expectation(
                binding, Path("unused"), bits, {"model.norm.weight": {"offset": 0}},
                self.contract)
        def written(name):
            return [int(row, 16) for row in files[name]]
        self.assertEqual(written("final_rmsnorm_expected.hex"), expected)
        self.assertEqual(receipt["fp16_interstage_comparison"]["failure_count"], 0)
        active = written("fp16_interstage_rmsnorm_expected.hex")
        diagnostic = written("mathematical_rmsnorm_expected.hex")
        self.assertEqual(len(active), 896)
        self.assertNotEqual(active, diagnostic)
        self.assertEqual(receipt["integer_mismatches"], 0)

    def test_foreign_parent_rejected_before_artifact_access(self):
        with patch.object(binding, "authenticate") as authenticate:
            with self.assertRaisesRegex(binding.evidence.AttemptError, "^foreign_parent$"):
                binding.admit({"parent": {"attempt_id": "model24_layer23_attempt001"}},
                              Path("unused"), self.contract)
            authenticate.assert_not_called()

    def test_source_admission_does_not_authorize_unchanged_replay(self):
        with patch.object(binding, "admit") as admit:
            with self.assertRaisesRegex(RuntimeError, "unchanged_tail_replay_not_authorized"):
                runtime.admit_execution(binding, {}, Path("unused"), self.contract)
            admit.assert_called_once()

    def test_consumed_marker_rejected_without_authentication_or_mutation(self):
        parent = self.contract["parent"]
        package = {"parent": {
            "attempt_id": parent["attempt_id"], "review": {"path": str(binding.REVIEW)},
            "launch": {"path": str(binding.PARENT / "result.json"),
                       "sha256": parent["launch_result_sha256"]},
            "seal": {"path": str(binding.PARENT / "seal.json"), "sha256": parent["seal_sha256"]},
            "result": {"path": str(binding.LAYER / "result.json"),
                       "sha256": parent["layer_result_sha256"]},
            "terminal": {"path": str(binding.LAYER / "position002/raw/final.hex"),
                         "sha256": parent["terminal_file_sha256"]},
        }}
        with patch.object(Path, "exists", lambda path: path.name == "consumed.json"), \
                patch.object(binding, "authenticate") as authenticate:
            with self.assertRaisesRegex(binding.evidence.AttemptError, "^consumed_attempt$"):
                binding.admit(package, Path("unused"), self.contract)
            authenticate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
