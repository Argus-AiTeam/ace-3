"""Remaining-layer binding regressions; no candidate execution or ancestor admission."""

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import remaining_layers_v3 as remaining
from ace3.model.candidates import remaining_decoder_gate_policy_v3 as evaluator


def synthetic_arguments(layer):
    local = remaining.host.local
    arrays = {f"stage{s:02d}": np.zeros(n, dtype="<u2") for s, n in local.SIZES.items()}
    arrays["stage09"][:] = np.float16(1).view("<u2")
    arrays["input_hidden"] = np.zeros(896, dtype="<u2")
    for kind in ("k", "v"):
        arrays["input_cache_" + kind] = np.empty((0, 128), dtype="<u2")
        arrays["output_cache_" + kind] = np.zeros((1, 128), dtype="<u2")
    tensors, canonical = {}, {}
    for name, (shape, dtype) in local.tensor_shapes(layer).items():
        value = np.zeros(shape, dtype=dtype)
        tensors[name] = value
        canonical[name] = {"name": name, "dtype": dtype, "shape": list(shape),
                           "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    return dict(arrays=arrays, tensors=tensors, canonical_records=canonical,
                expected_hidden=arrays["input_hidden"].copy(),
                trajectory={s: arrays[f"stage{s:02d}"].copy() for s in range(19)},
                reference_binary64=np.zeros(896, dtype="<f8"), layer=layer, position=0,
                history=[9707], policy=remaining.host.policy.POLICY_ID, emit=lambda r: None)


class RemainingLayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ["ACE3_REMAINING_PLAN"])
        cls.plan = json.loads((cls.root / "freeze.json").read_text())
        cls.extension = remaining.host.runtime.document(cls.plan["reference_extension"])

    def test_all_fifteen_source_bound_layers_and_public_parameters(self):
        remaining.validate_plan(self.plan)
        self.assertEqual(len(self.plan["continuation"]["command_files"]), 30)
        self.assertFalse((self.root / "runtime").exists())
        public = (self.root / "public_contract.sv").read_text()
        contract = remaining.host.reference_contract(self.plan)
        for layer in remaining.LAYERS:
            with self.subTest(layer=layer):
                self.assertIn(f".LAYER_INDEX({layer})", public)
                context = remaining.host.runtime._trusted_context(contract, layer)
                self.assertEqual(context["binary64"].shape, (896,))
                self.assertEqual(context["binary64"].dtype, np.dtype("<f8"))
                self.assertEqual(set(context["trajectory"]), set(range(19)))
                self.assertEqual(len(context["canonical"]), 26)
                directory = self.root / "runtime" / f"layer{layer:02d}"
                self.assertEqual(self.plan["transactions"][str(layer)]["simulation_argv"],
                                 remaining.host.simulation_command(directory, layer))
                for action in ("capture", "admit"):
                    text = (self.root / f"{action}-layer{layer:02d}.command.sh").read_text()
                    self.assertIn(f"--layer {layer}", text)
                    self.assertIn("separately Host-observed digest required", text)
                    self.assertNotIn("--state-in", text)

    def test_original_reference_chains_never_use_actual_hidden(self):
        actual = self.plan["continuation"]["actual_l8_hidden"]
        self.assertNotEqual(actual, self.extension["original_fp16_parent"])
        self.assertNotEqual(actual, self.extension["original_binary64_parent"])
        previous16 = self.extension["original_fp16_parent"]
        previous64 = self.extension["original_binary64_parent"]
        for layer in remaining.LAYERS:
            row = self.extension["layers"][str(layer)]
            self.assertEqual(row["input_fp16"], previous16)
            self.assertEqual(row["input_binary64"], previous64)
            self.assertEqual(row["prior_kv"], "own empty P0")
            previous16, previous64 = row["fp16"], row["binary64"]
        self.assertEqual(self.extension["accepted_ancestor_oracle_replays"], 0)

    def test_wrong_controls_state_paths_and_parent_fail_closed(self):
        for field, value in (
            ("input_state", {"path": "layer08.state"}),
            ("cache_slot", 1),
            ("parameters", {"LAYER_INDEX": 8, "ACCURATE_SILU": 1}),
            ("directory", str(self.root / "runtime/layer08")),
            ("canonical_tensors", self.plan["transactions"]["10"]["canonical_tensors"])):
            with self.subTest(field=field):
                wrong = copy.deepcopy(self.plan)
                wrong["transactions"]["9"][field] = value
                with self.assertRaisesRegex(ValueError, "noncanonical"):
                    remaining.validate_plan(wrong)
        wrong = copy.deepcopy(self.plan)
        wrong["continuation"]["actual_l8_hidden"] = self.extension["original_fp16_parent"]
        with self.assertRaisesRegex(ValueError, "actual L8 hidden"):
            remaining.validate_plan(wrong)

    def test_missing_unsupported_and_spliced_global_references_block(self):
        base = json.loads(remaining.host.policy.CONTRACT.read_text())
        with self.assertRaisesRegex(ValueError, "missing original reference extension"):
            remaining.host.runtime._trusted_context(base, 9)
        for layer in (4, 24, -1, True):
            with self.subTest(layer=layer), self.assertRaisesRegex(ValueError, "unsupported"):
                remaining.host.runtime._trusted_context(base, layer)
        with tempfile.TemporaryDirectory(dir=remaining.host.BUILD) as temporary:
            directory = Path(temporary)
            cases = []
            wrong = copy.deepcopy(self.extension)
            wrong["original_binary64_parent"] = self.plan["continuation"]["actual_l8_hidden"]
            cases.append(wrong)
            wrong = copy.deepcopy(self.extension)
            wrong["layers"]["10"]["input_binary64"] = wrong["original_binary64_parent"]
            cases.append(wrong)
            wrong = copy.deepcopy(self.extension)
            wrong["scope"]["position"] = 1
            cases.append(wrong)
            for index, value in enumerate(cases):
                path = directory / f"wrong{index}.json"
                remaining.host.write(path, value)
                with self.subTest(index=index), self.assertRaises(ValueError):
                    remaining.host.runtime._trusted_context(
                        dict(base, reference_extension=remaining.host.record(path)), 9)
            path = directory / "invalid.npy"
            with path.open("xb") as stream:
                np.save(stream, np.full(896, np.nan, dtype="<f8"))
            wrong = copy.deepcopy(self.extension)
            wrong["layers"]["23"]["binary64"] = remaining.host.record(path)
            record = directory / "nonfinite.json"
            remaining.host.write(record, wrong)
            with self.assertRaisesRegex(ValueError, "invalid original global"):
                remaining.host.runtime._trusted_context(
                    dict(base, reference_extension=remaining.host.record(record)), 23)

    def test_corrected_coordinates_cover_every_remaining_layer(self):
        runtime = remaining.host.runtime
        coordinates = remaining.original_context()["trace_coordinates"]
        # Nonconstant words expose half-split RoPE order and repeated S8/S9 index zero.
        values = {s: np.arange(n, dtype="<u2") for s, n in remaining.host.local.SIZES.items()}
        for layer in remaining.LAYERS:
            offsets = {8: 0, 9: 0}
            rows = []
            for stage, position, index in coordinates:
                offset = offsets[stage] if stage in offsets else index
                rows.append((stage, position, index, int(values[stage][offset])))
                if stage in offsets:
                    offsets[stage] += 1
            decoded = runtime.canonical_stages(rows, coordinates)
            for stage in values:
                np.testing.assert_array_equal(decoded[f"stage{stage:02d}"], values[stage])
            with self.subTest(layer=layer), self.assertRaisesRegex(ValueError, "order/coverage"):
                runtime.canonical_stages(rows[::-1], coordinates)

    def test_external_manifest_required_before_any_suffix_numerics(self):
        with patch.object(evaluator, "evaluate_p0_transaction") as numerical:
            result = evaluator.evaluate_actual_rtl_result(
                runtime_admission={}, reference_extension=self.plan["reference_extension"],
                evaluator_extension=self.plan["evaluator_extension"],
                arrays={}, tensors={}, canonical_records={}, expected_hidden=None,
                trajectory={}, reference_binary64=None, layer=23, position=0,
                history=[9707], policy=remaining.host.policy.POLICY_ID, emit=lambda r: None)
            self.assertEqual(result["status"], "BLOCKED")
            self.assertEqual(result["numerical_status"], "NOT_EVALUATED")
            numerical.assert_not_called()

    def test_historical_policy_identity_and_separate_extension_scope(self):
        self.assertEqual(remaining.host.reviewed_policy_source(), self.plan["historical_policy_source"])
        evaluator.validate_binding(self.plan["evaluator_extension"])
        args = synthetic_arguments(9)
        with self.assertRaisesRegex(ValueError, "L5-L8"):
            remaining.host.policy.evaluate_p0_transaction(**args)
        for layer in (8, 24, True):
            with self.subTest(layer=layer), self.assertRaisesRegex(ValueError, "L9-L23"):
                evaluator.evaluate_p0_transaction(**dict(args, layer=layer))
        for value in (None, dict(self.plan["evaluator_extension"], id="unversioned")):
            wrong = copy.deepcopy(self.plan)
            wrong["evaluator_extension"] = value
            with self.assertRaisesRegex(ValueError, "evaluator extension"):
                remaining.validate_plan(wrong)
        wrong = copy.deepcopy(self.plan["evaluator_extension"])
        wrong["sources"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "evaluator extension"):
            evaluator.validate_binding(wrong)

    def test_synthetic_suffix_endpoints_use_real_unchanged_numerics(self):
        for layer in (9, 23):
            args = synthetic_arguments(layer)
            with self.subTest(layer=layer):
                result = evaluator.evaluate_p0_transaction(**args)
                self.assertEqual(result["status"], "PASS")
                self.assertEqual(len(result["reports"]), 19)
                self.assertEqual(set(result["local_references"]),
                                 {f"stage{s:02d}" for s in range(18)})
                self.assertEqual(result["reports"][-1]["node"], [layer, 0, 18])
                self.assertEqual(result["reports"][-1]["binary64_v1"]["failure_count"], 0)

    def test_suffix_arithmetic_error_and_global_drift_still_fail(self):
        args = synthetic_arguments(9)
        args["arrays"]["stage00"][0] = np.float16(1).view("<u2")
        result = evaluator.evaluate_p0_transaction(**args)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual([r["stage"] for r in result["reports"]], [0])
        args["arrays"]["stage00"][0] = 0
        args["reference_binary64"][0] = 1
        result = evaluator.evaluate_p0_transaction(**args)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(len(result["reports"]), 19)
        self.assertTrue(all(r["status"] == "PASS" for r in result["reports"][:-1]))
        self.assertEqual(result["reports"][-1]["binary64_v1"]["failure_count"], 1)

    def test_manifest_must_bind_the_same_evaluator_before_context_or_numerics(self):
        runtime = remaining.host.runtime
        with tempfile.TemporaryDirectory(dir=remaining.host.BUILD) as temporary:
            path = Path(temporary) / "synthetic_manifest.json"
            remaining.host.write(path, {
                "schema": runtime.SCHEMA, "policy_id": evaluator.base.POLICY_ID,
                "evidence_kind": "actual_rtl", "node": [9, 0], "history": [9707],
                "reference_extension": self.plan["reference_extension"],
                "evaluator_extension": dict(self.plan["evaluator_extension"], id="wrong"),
            })
            receipt = remaining.host.record(path)
            with patch.object(runtime, "_trusted_context") as context, \
                    patch.object(evaluator, "evaluate_p0_transaction") as numerical:
                result = evaluator.evaluate_actual_rtl_result(
                    runtime_admission=receipt, trusted_runtime_manifest_sha256=receipt["sha256"],
                    reference_extension=self.plan["reference_extension"],
                    evaluator_extension=self.plan["evaluator_extension"], **synthetic_arguments(9))
                self.assertEqual(result["status"], "BLOCKED")
                self.assertIn("runtime remaining evaluator extension mismatch", result["reason"])
                context.assert_not_called()
                numerical.assert_not_called()

    def test_suffix_result_carries_version_binding_after_synthetic_runtime_admission(self):
        with patch.object(remaining.host.runtime, "validate_runtime",
                          return_value={"fixture": "synthetic, not runtime evidence"}) as admission:
            result = evaluator.evaluate_actual_rtl_result(
                runtime_admission={}, reference_extension=self.plan["reference_extension"],
                evaluator_extension=self.plan["evaluator_extension"], **synthetic_arguments(9))
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["evaluator_extension"], self.plan["evaluator_extension"])
        self.assertEqual(admission.call_args.kwargs["contract"]["evaluator_extension"],
                         self.plan["evaluator_extension"])
        self.assertEqual(len(result["reports"]), 19)

    def test_host_suffix_admission_selects_extension_without_changing_legacy_entrypoint(self):
        host = remaining.host
        args = synthetic_arguments(9)
        with tempfile.TemporaryDirectory(dir=host.BUILD) as temporary:
            directory = Path(temporary)
            with (directory / "actual_operands.npz").open("xb") as stream:
                np.savez(stream, **args["arrays"])
            host.write(directory / "manifest.json", {
                "actual_operands": host.record(directory / "actual_operands.npz")})
            plan = copy.deepcopy(self.plan)
            plan["root"] = str(directory)
            plan["transactions"]["9"]["directory"] = str(directory)
            host.write(directory / "freeze.json", plan)
            digest = host.record(directory / "manifest.json")["sha256"]
            context = {"canonical": args["canonical_records"], "trajectory": args["trajectory"],
                       "binary64": args["reference_binary64"]}
            with patch.object(host.runtime, "_trusted_context", return_value=context), \
                    patch.object(host, "tensors_from", return_value=args["tensors"]), \
                    patch.object(host.runtime, "validate_runtime",
                                 return_value={"fixture": "synthetic, not runtime evidence"}), \
                    patch.object(host.policy, "evaluate_actual_rtl_result") as legacy, \
                    patch.object(host, "observe") as execution:
                self.assertEqual(host.admit(plan, 9, digest), 0)
                legacy.assert_not_called()
                execution.assert_not_called()
            result = json.loads((directory / "admission/result.json").read_text())
            self.assertEqual(result["evaluator_extension"], self.plan["evaluator_extension"])
            self.assertEqual(len(result["reports"]), 19)

    def test_suffix_actual_parent_must_retain_evaluator_lineage(self):
        host = remaining.host
        receipt = {"path": str(self.root / "runtime/layer09/admission/result.json"),
                   "bytes": 1, "sha256": "1" * 64}
        parent = {
            "status": "PASS", "numerical_status": "PASS", "policy_id": host.policy.POLICY_ID,
            "evidence_kind": "actual_rtl_numerical_evaluation",
            "runtime_admission": {
                "node": [9, 0], "history": [9707],
                "source_sha256": {Path(r["path"]).name: r["sha256"] for r in self.plan["source_closure"]},
                "output": {"fixture": "synthetic parent"}},
        }
        with patch.object(host, "record", return_value=receipt), \
                patch.object(host.runtime, "document", return_value=parent):
            with self.assertRaisesRegex(ValueError, "evaluator lineage"):
                host.admitted_parent(self.plan, 10, receipt["sha256"])
            parent["evaluator_extension"] = self.plan["evaluator_extension"]
            self.assertEqual(host.admitted_parent(self.plan, 10, receipt["sha256"]),
                             (receipt, parent["runtime_admission"]["output"]))

    def test_exclusive_roots_and_command_outputs(self):
        with self.assertRaisesRegex(ValueError, "immutable output"):
            remaining.host.fresh(self.root, remaining.host.BUILD)
        with self.assertRaises(FileExistsError):
            remaining.command_files(self.plan, self.root)


if __name__ == "__main__":
    unittest.main()
