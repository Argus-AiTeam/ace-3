"""L11 producer preflight: real read-only binding and synthetic CPU wiring."""

import ast
import copy
from fractions import Fraction
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l11_coordinate62_producer_cone_v1 as d


def fixture():
    parent = d.paired.mapped_parent(np.ones(896, dtype="<f8"))
    arrays = {key: np.full(896, 0x3c00, dtype="<u2") for key in ("stage11", "stage17")}
    scratch, output = d.producer.compose(parent, arrays["stage11"], arrays["stage17"])
    arrays.update({"scratch_" + k: v for k, v in scratch.items() if k != "h"})
    arrays.update({"output_" + k: v for k, v in output.items() if k != "h"})
    arrays.update(stage12=scratch["h"], stage18=output["h"])
    original = {key: np.full(896, value, dtype="<f8") for key, value in (
        ("input", 2), ("s11", 3), ("s17", 4), ("residual", 5), ("s18", 9))}
    return {"parent": parent, "arrays": arrays}, original


def rows(rescues=()):
    return [{"label": label, "index62": {
        "accepted": label in rescues,
        "actual_fp16_bits": "662f" if label in rescues else "6630",
    }} for label, _ in d.plan()]


class ProducerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = d.authenticate()

    def test_plan_and_contract(self):
        contract = d.check_contract()
        self.assertEqual(len(d.plan()), 13)
        self.assertEqual(len(set(label for label, _ in d.plan())), 13)
        self.assertEqual(len(d.expected_dispatches()), 28)
        self.assertEqual(set(d.expected_dispatches()), {11, 12, 13})
        self.assertEqual(contract["native_layer_invocations"], 28)

    def test_authenticated_selected_sources_and_l11_parent(self):
        evidence = self.data["selected_L9_evidence"]
        self.assertEqual(evidence["result"]["sha256"], d.selected.SELECTED[1])
        self.assertEqual(evidence["entry_source"], d.record(Path(d.entry.__file__)))
        d.paired.same_arrays(self.data["layers"][10]["output"], self.data["layers"][11]["parent"])
        self.assertEqual(evidence["policy_binding"]["policy_id"], d.prior.gates.POLICY_ID)
        self.assertFalse(self.data["historical"][0]["source_compatible"])
        self.assertFalse(self.data["historical"][0]["execution_authorized_by_this_diagnostic"])
        for layer in d.NATIVE_LAYERS:
            refs = self.data["extension"]["layers"]
            self.assertEqual(refs[str(layer)]["input_binary64"], refs[str(layer - 1)]["binary64"])

    def test_parent_and_original_reference_splices_rejected(self):
        for kind in ("parent", "reference"):
            data = {**self.data, "layers": self.data["layers"].copy(),
                    "extension": copy.deepcopy(self.data["extension"])}
            if kind == "parent":
                data["layers"][11] = {**data["layers"][11],
                                      "parent": {k: v.copy() for k, v in data["layers"][11]["parent"].items()}}
                data["layers"][11]["parent"]["i"][62] += 1
            else:
                data["extension"]["layers"]["11"]["input_binary64"] = {"path": "spliced"}
            with patch.object(d.selected, "authenticate", return_value=data), self.assertRaises(ValueError):
                d.authenticate()

    def test_bound_input_tamper_and_foreign_path_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside isolated repository"):
            d.upstream.bind_input({"path": "/tmp/input", "bytes": 0, "sha256": "bad"}, {})
        item = self.data["selected_L9_evidence"]["result"]
        with self.assertRaises(ValueError):
            d.upstream.bind_input({**item, "sha256": "wrong"}, {})

    def test_repo_origins_and_import_only_ancestor_tests(self):
        origins = d.source_context()
        for name in (d.MODULE, d.TEST_MODULE, d.ANCESTOR_TEST):
            self.assertEqual(origins[name]["path"],
                             str(d.ROOT.joinpath(*name.split(".")).with_suffix(".py")))
        self.assertTrue(all(Path(item["path"]).is_relative_to(d.ROOT)
                            for item in origins.values() if "path" in item))

    def test_foreign_module_origin_rejected(self):
        with patch.dict(sys.modules, {"ace3.model.foreign": SimpleNamespace(__file__="/tmp/foreign.py")}):
            with self.assertRaisesRegex(ValueError, "module origin mismatch"):
                d.source_context()

    def test_actual_cut_reproduces_authenticated_l11(self):
        layer = self.data["layers"][11]
        _, original = fixture()
        _, output = d.cut_parent(layer, original, "actual")
        d.paired.same_arrays(output, self.data["layers"][11]["output"])

    def test_factorial_exact_additive_nonmutating_scalar_cuts(self):
        layer, original = fixture()
        before = {k: v.copy() for k, v in layer["arrays"].items()}
        values = {}
        for label, parts in d.plan()[:8]:
            _, output = d.cut_parent(layer, original, label)
            values[parts] = int(output["i"][62])
            self.assertTrue(np.all(output["i"][np.arange(896) != 62] == 3 << 24))
        self.assertEqual(values[()], 3 << 24)
        self.assertEqual(values[d.producer.BRANCHES], 9 << 24)
        self.assertEqual(values[d.producer.BRANCHES] - values[()],
                         sum(values[(b,)] - values[()] for b in d.producer.BRANCHES))
        d.paired.same_arrays(before, layer["arrays"])

    def test_scratch_and_down_keep_fp16_boundary(self):
        layer, original = fixture()
        operands, output = d.cut_parent(layer, original, "scratch")
        self.assertEqual(output["i"][62], 6 << 24)
        self.assertEqual(operands["down"].dtype, np.dtype("<u2"))
        np.testing.assert_array_equal(operands["down"], layer["arrays"]["stage17"])
        _, output = d.cut_parent(layer, original, "scratch_down")
        self.assertEqual(output["i"][62], 9 << 24)

    def test_mapped_endpoints_and_invalid_cut(self):
        layer, original = fixture()
        for label in ("mapped62", "mapped_all"):
            _, output = d.cut_parent(layer, original, label)
            self.assertEqual(output["i"][62], 9 << 24)
            self.assertEqual(output["i"][0], (3 if label == "mapped62" else 9) << 24)
        with self.assertRaises(ValueError):
            d.cut_parent(layer, original, "unknown")
        layer["parent"]["h"][62] = 0
        with self.assertRaises(ValueError):
            d.cut_parent(layer, original, "frozen_o")

    def test_signed_zero_and_q24_ties_even(self):
        values = np.zeros(896, dtype="<f8")
        values[:4] = [-0.0, -2**-25, 2**-25, 3 * 2**-25]
        mapped = d.paired.mapped_parent(values)
        np.testing.assert_array_equal(mapped["i"][:4], [0, 0, 0, 2])
        np.testing.assert_array_equal(mapped["z"][:4], [1, 1, 0, 0])
        np.testing.assert_array_equal(mapped["h"][:3], [0x8000, 0x8000, 0])

    def test_exact_threshold_and_historical_failure_unchanged(self):
        self.assertTrue(d.prior.measure(0x3c80, 1.0)["accepted"])
        self.assertFalse(d.prior.measure(0x3c81, 1.0)["accepted"])
        reference = float.fromhex("0x1.8bd7b2092532cp+10")
        failed = d.prior.measure(0x6630, reference)
        self.assertFalse(failed["accepted"])
        self.assertTrue(d.prior.measure(0x662f, reference)["accepted"])
        self.assertEqual(Fraction(failed["excess_budget"]), Fraction(1, 8))

    def test_conditional_not_unique_attribution(self):
        result = d.classify(rows(("frozen_down",)))
        self.assertEqual(result["classification"], "conditional_single_branch_sufficiency")
        self.assertEqual(result["sufficient_frozen_single_branches"], ["down"])
        self.assertFalse(result["unique_upstream_producer_attributed"])
        for rescues in ((), ("frozen_o", "frozen_down"), ("frozen_o_down",)):
            self.assertEqual(d.classify(rows(rescues))["classification"],
                             "unresolved_under_tested_branch_mappings")

    def test_changed_baseline_or_incomplete_controls_rejected(self):
        for invalid in (rows()[:-1], list(reversed(rows())), rows(("actual",))):
            with self.assertRaises(ValueError):
                d.classify(invalid)

    def test_dispatch_rejects_every_out_of_scope_layer_before_delegation(self):
        with patch.object(d.upstream, "execute_layer") as execute:
            for layer in (*range(11), *range(14, 24), -1, True, 11.0, "11"):
                with self.subTest(layer=layer), self.assertRaisesRegex(ValueError, "outside L11-L13"):
                    d.execute_layer(layer, {}, {}, [])
            execute.assert_not_called()

    def test_native_stage_guard_rejects_bypass_and_restores_delegate(self):
        with patch.object(d.native, "stages", return_value="synthetic") as stages:
            dispatch = []
            with d.guard_stages(dispatch):
                for layer in (*range(11), 14, True):
                    with self.assertRaises(ValueError):
                        d.native.stages({}, layer, {}, {})
                stages.assert_not_called()
                self.assertEqual(d.native.stages({}, 11, {}, {}), "synthetic")
            self.assertIs(d.native.stages, stages)
            self.assertEqual(dispatch, [11])
            stages.assert_called_once_with({}, 11, {}, {})

    def test_suffix_threads_actual_state_and_original_references(self):
        data = self.data
        initial = data["layers"][11]["output"]
        dispatch, observed = [], []

        def execute(tensors, layer, parent, trajectory, reference):
            self.assertIs(tensors, data["layers"][layer]["tensors"])
            self.assertIs(trajectory, data["layers"][layer]["trajectory"])
            self.assertIs(reference, data["layers"][layer]["reference"])
            d.paired.same_arrays(parent, data["layers"][layer]["parent"])
            return tuple(data["layers"][layer][k] for k in ("arrays", "locals", "reports"))

        with patch.object(d.upstream, "execute_layer", side_effect=execute):
            d.execute_suffix(initial, data, lambda layer, *unused: observed.append(layer), dispatch)
        self.assertEqual(dispatch, [12, 13])
        self.assertEqual(observed, [12, 13])

    def test_output_scope_exclusivity_and_real_filesystem_publication(self):
        for path in (d.ROOT / "build/wrong", d.ROOT / ("build/" + d.OUTPUT_PREFIX)):
            with self.assertRaises(ValueError):
                d.output_path(path)
        with tempfile.TemporaryDirectory(prefix=d.OUTPUT_PREFIX + "test_", dir=d.ROOT / "build") as directory:
            out = Path(directory)
            with self.assertRaisesRegex(ValueError, "already exists"):
                d.output_path(out)
            link = out / "link"
            link.symlink_to(out / "missing")
            with self.assertRaisesRegex(ValueError, "symlink"):
                d.output_path(link)
            d.write(out / "validation.json", {"once": True})
            with self.assertRaises(FileExistsError):
                d.write(out / "validation.json", {"twice": True})
            self.assertEqual(json.loads((out / "validation.json").read_text()), {"once": True})

    def test_check_mode_validates_once_without_diagnostic_or_result(self):
        with tempfile.TemporaryDirectory(prefix=d.OUTPUT_PREFIX + "test_", dir=d.ROOT / "build") as directory:
            out = Path(directory) / "output"
            with patch.object(d, "output_path", return_value=out), \
                    patch.object(d, "validate", return_value={}) as validate, \
                    patch.object(d, "diagnose") as diagnose, \
                    patch.object(sys, "argv", [d.MODULE, "--check", "--out", str(out)]), \
                    patch.object(sys, "stdout", io.StringIO()):
                self.assertEqual(d.main(), 0)
            validate.assert_called_once_with(out)
            diagnose.assert_not_called()
            self.assertFalse((out / "result.json").exists())

    def test_static_no_external_execution_or_policy_publication(self):
        tree = ast.parse(Path(d.__file__).read_text())
        calls = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
                 for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))}
        self.assertFalse(calls & {"system", "Popen", "publish_parent", "execute_layers",
                                 "continuation_stages", "run_factory", "cuda", "set_default_dtype"})
        delegates = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and node.func.attr == "execute_layer"]
        self.assertEqual(len(delegates), 1)
        self.assertIs(d.cut_parent, d.legacy.cut_parent)
        self.assertIs(d.plan, d.producer.plan)
        contract = d.check_contract()
        for key in ("candidate_admitted", "policy_adopted", "successor_published",
                    "accepted_L0_L8_execution", "scientific_result_claim"):
            self.assertIs(contract[key], False)


if __name__ == "__main__":
    unittest.main()
