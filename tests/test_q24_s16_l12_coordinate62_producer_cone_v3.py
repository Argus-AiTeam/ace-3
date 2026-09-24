"""Focused v3 surface tests; native execution is replaced by explicit test doubles."""

import ast
from copy import deepcopy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l12_coordinate62_producer_cone_v3 as d
from tests import test_q24_s16_l12_coordinate62_producer_cone_v2 as fixtures


EXPECTED_TESTS = 30


class ProducerConeV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads(d.CONTRACT.read_text())

    def test_contract_authorizes_only_non_admission_dispatch(self):
        d.check_contract(self.contract)
        self.assertTrue(self.contract["native_control_dispatch_authorized"])
        self.assertEqual(self.contract["dispatch_purpose"], "isolated_non_admission_counterfactual_suffix")
        self.assertEqual(self.contract["maximum_native_layer_invocations"], 15)

    def test_contract_rejects_admission_and_rtl(self):
        for key, value in (("candidate_admitted", True), ("policy_adopted", True),
                           ("successor_published", True), ("rtl_invocations", 1),
                           ("normal_host_review", "OPTIONAL"), ("accepted_prefix_replay", True),
                           ("scientific_result_claim", True)):
            changed = {**self.contract, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_contract(changed)

    def test_contract_rejects_scope_and_budget_changes(self):
        for key, value in (("maximum_native_layer_invocations", 16),
                           ("native_control_dispatch_authorized", False),
                           ("suffix_extent", {"position": 1, "L12": 2, "L13": 13}),
                           ("dispatch_purpose", "admission"), ("native_L0_L8_invocations", 1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_contract({**self.contract, key: value})

    def test_contract_rejects_threshold_and_reference_changes(self):
        for key in ("thresholds", "inherits_sha256", "reviewed_result_sha256",
                    "retained_freeze_sha256", "preserved_fields", "inputs"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_contract({**self.contract, key: "substituted"})

    def test_contract_rejects_boolean_integer_substitution(self):
        changed = deepcopy(self.contract)
        changed["suffix_extent"]["position"] = False
        with self.assertRaises(ValueError):
            d.check_contract(changed)

    def test_preserved_history_and_arithmetic(self):
        self.assertEqual(len(d.history_records()), 6)
        for name in ("plan", "compose", "frozen_parent", "original_branches", "authenticate", "classify"):
            self.assertIs(getattr(d, name), getattr(d.v2, name))

    def test_factorial_plan(self):
        self.assertEqual([label for label, _ in d.plan()], self.contract["controls"])
        self.assertEqual(len(d.plan()), 13)
        self.assertEqual(len({parts for _, parts in d.plan()[:8]}), 8)

    def test_baseline_exact_q24(self):
        scratch, output = d.frozen_parent(*fixtures.fixture(), ())[-2:]
        self.assertEqual(scratch["i"][62], 2 << 24)
        self.assertEqual(output["i"][62], 3 << 24)
        self.assertEqual(output["h"][62], 0x4200)

    def test_complete_inherited_coordinate_and_no_mutation(self):
        parent, actual, original = fixtures.fixture()
        incoming, _, _, _, output = d.frozen_parent(parent, actual, original, ("inherited",))
        self.assertEqual(incoming["i"][62], 2 << 24)
        self.assertEqual(incoming["h"][62], 0x4000)
        self.assertEqual(output["i"][62], 4 << 24)
        unchanged = np.arange(896) != 62
        for key in parent:
            np.testing.assert_array_equal(incoming[key][unchanged], parent[key][unchanged])
        self.assertTrue(np.all(parent["i"] == 1 << 24))

    def test_fp16_o_and_down_cuts(self):
        for parts, expected in ((("o",), 5), (("down",), 6), (("o", "down"), 8)):
            _, o, down, _, output = d.frozen_parent(*fixtures.fixture(), parts)
            self.assertEqual(o.dtype, np.dtype("<u2"))
            self.assertEqual(down.dtype, np.dtype("<u2"))
            self.assertEqual(output["i"][62], expected << 24)

    def test_exact_additive_factorial(self):
        outputs = {parts: d.frozen_parent(*fixtures.fixture(), parts)[-1]["i"][62]
                   for _, parts in d.plan()[:8]}
        self.assertEqual(outputs[d.legacy.BRANCHES], 9 << 24)
        self.assertEqual(outputs[d.legacy.BRANCHES] - outputs[()],
                         sum(outputs[(branch,)] - outputs[()] for branch in d.legacy.BRANCHES))

    def test_inconsistent_state_and_unknown_cut_rejected(self):
        parent, actual, original = fixtures.fixture()
        with self.assertRaises(ValueError):
            d.frozen_parent(parent, actual, original, ("unknown",))
        parent["h"][62] = 0
        with self.assertRaises(ValueError):
            d.compose(parent, actual["stage11"], actual["stage17"])

    def test_signed_zero_and_q24_mapping(self):
        values = np.zeros(896, dtype="<f8")
        values[:4] = [-0.0, -2**-25, 2**-25, 3 * 2**-25]
        mapped = d.paired.mapped_parent(values)
        np.testing.assert_array_equal(mapped["i"][:4], [0, 0, 0, 2])
        np.testing.assert_array_equal(mapped["z"][:4], [1, 1, 0, 0])

    def test_exact_threshold_and_retained_failure(self):
        self.assertTrue(d.prior.measure(0x3c80, 1.0)["accepted"])
        self.assertFalse(d.prior.measure(0x3c81, 1.0)["accepted"])
        reference = float.fromhex("0x1.8bd7b2092532cp+10")
        self.assertFalse(d.prior.measure(0x6630, reference)["accepted"])
        self.assertTrue(d.prior.measure(0x662f, reference)["accepted"])
        self.assertEqual(d.prior.measure(0x6630, reference)["excess_budget"], "1/8")

    def test_classification_never_unique_producer(self):
        result = d.classify(fixtures.rows(("frozen_down",)))
        self.assertEqual(result["classification"], "conditional_single_branch_sufficiency")
        self.assertFalse(result["unique_upstream_producer_attributed"])
        self.assertTrue(result["missing_evidence"])
        self.assertEqual(d.classify(fixtures.rows())["classification"], "unresolved_producer_interaction")

    def test_incomplete_and_changed_baseline_rejected(self):
        changed = fixtures.rows()
        changed[0]["index62"]["accepted"] = True
        for rows in (fixtures.rows()[:-1], list(reversed(fixtures.rows())), changed):
            with self.assertRaises(ValueError):
                d.classify(rows)

    def test_schedule_exact_layers_and_budget(self):
        data = {"layers": {12: {"trajectory": object(), "reference": object()}},
                "trajectory": object(), "reference": object(), "tensors": object(), "tensors12": object()}
        controls = d.NativeControls(data)
        parent = object()
        with patch.object(d.upstream, "execute_layer", return_value=("arrays", "locals", "reports")) as run:
            for label, layer in controls.schedule:
                self.assertEqual(controls.run(label, layer, parent), ("arrays", "locals", "reports"))
                source = data["layers"][12] if layer == 12 else data
                self.assertIs(run.call_args.args[3], source["trajectory"])
                self.assertIs(run.call_args.args[4], source["reference"])
            controls.finish()
            self.assertEqual(run.call_count, 15)
            self.assertEqual([call.args[1] for call in run.call_args_list].count(12), 2)
            with self.assertRaises(ValueError):
                controls.run("actual", 13, parent)
            self.assertEqual(run.call_count, 15)

    def test_schedule_rejects_reorder_wrong_layer_and_boolean(self):
        controls = d.NativeControls({})
        with patch.object(d.upstream, "execute_layer") as run:
            for label, layer in (("actual", 13), ("baseline", 9), ("baseline", True), ("baseline", 12.0)):
                with self.assertRaises(ValueError):
                    controls.run(label, layer, {})
            run.assert_not_called()
        self.assertEqual(controls.count, 0)

    def test_incomplete_schedule_rejected(self):
        with self.assertRaises(ValueError):
            d.NativeControls({}).finish()

    def test_foreign_source_origin_rejected(self):
        with patch.dict(d.sys.modules, {"ace3.model.foreign": SimpleNamespace(__file__="/tmp/foreign.py")}):
            with self.assertRaisesRegex(ValueError, "module origin mismatch"):
                d.source_context()

    def test_hidden_pythonpath_rejected(self):
        with patch.dict(d.os.environ, {"PYTHONPATH": ""}):
            with self.assertRaisesRegex(ValueError, "repo-bound"):
                d.source_context()

    def test_foreign_or_tampered_input_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside isolated repository"):
            d.upstream.bind_input({"path": "/tmp/input", "bytes": 0, "sha256": "bad"}, {})
        item = {"path": str(d.ROOT / "build/producer-input"), "bytes": 1, "sha256": "a"}
        with patch.object(d.upstream, "record", return_value={**item, "sha256": "b"}):
            with self.assertRaisesRegex(ValueError, "binding mismatch"):
                d.upstream.bind_input(item, {})

    def test_reference_substitution_and_splice_rejected(self):
        extension = {"reference_only": True, "policy_id": d.prior.gates.POLICY_ID,
                     "global_reference_policy": "actual-parent"}
        with self.assertRaisesRegex(ValueError, "reference substituted"):
            d.coordinate.check_reference_chain(extension)
        extension.update(global_reference_policy="legacy-binary64-AWQ-fully-independent-propagation",
                         original_binary64_parent="original64", original_fp16_parent="original16",
                         layers={"9": {"input_binary64": "spliced", "input_fp16": "original16",
                                        "prior_kv": "own empty P0"}})
        with self.assertRaisesRegex(ValueError, "spliced original-input reference"):
            d.coordinate.check_reference_chain(extension)

    def test_wrong_receipt_and_reviewed_identity_rejected(self):
        with self.assertRaisesRegex(ValueError, "parent receipt lineage mismatch"):
            d.coordinate.check_parent_receipt({"next_layer": 14}, {}, {}, {}, {})
        with self.assertRaisesRegex(ValueError, "reviewed v3 coordinate scope mismatch"):
            d.v2.check_reviewed({"diagnostic_id": d.legacy.ID})

    def test_cli_rejects_ambiguous_modes_and_outputs(self):
        for args in ([], ["--execute"], ["--check", "--execute"],
                     ["--check", "--out", "build/forbidden"]):
            with patch.object(d.sys, "argv", [d.MODULE, *args]), \
                    patch.object(d.sys, "stderr", io.StringIO()), patch.object(d, "validate") as validate:
                with self.assertRaises(SystemExit) as exc:
                    d.main()
                self.assertEqual(exc.exception.code, 2)
                validate.assert_not_called()

    def test_cli_check_has_no_dispatch(self):
        with patch.object(d.sys, "argv", [d.MODULE, "--check"]), \
                patch.object(d, "validate", return_value={"checked": True}) as validate, \
                patch.object(d, "diagnose") as diagnose, patch.object(d.sys, "stdout", io.StringIO()):
            self.assertEqual(d.main(), 0)
            validate.assert_called_once_with()
            diagnose.assert_not_called()

    def test_cli_execution_cannot_bypass_validation(self):
        with patch.object(d.sys, "argv", [d.MODULE, "--execute", "--out", "build/test"]), \
                patch.object(d, "output_path", return_value=Path("build/test")), \
                patch.object(d, "validate", side_effect=ValueError("invalid check")), \
                patch.object(d, "diagnose") as diagnose:
            with self.assertRaisesRegex(ValueError, "invalid check"):
                d.main()
            diagnose.assert_not_called()
        events = []
        out = MagicMock()
        out.mkdir.side_effect = lambda **kwargs: events.append(("mkdir", kwargs))
        with patch.object(d.sys, "argv", [d.MODULE, "--execute", "--out", "build/test"]), \
                patch.object(d, "output_path", return_value=out), \
                patch.object(d, "validate", side_effect=lambda: events.append("validate") or {}) as validate, \
                patch.object(d, "diagnose", side_effect=lambda *args: events.append("diagnose") or {}) as diagnose, \
                patch.object(d, "write"), patch.object(d, "record", return_value={}), \
                patch.object(d.sys, "stdout", io.StringIO()):
            self.assertEqual(d.main(), 0)
            validate.assert_called_once_with()
            diagnose.assert_called_once()
            self.assertEqual(events, ["validate", ("mkdir", {"exist_ok": False}), "diagnose"])

    def test_output_scope_and_existing_paths_rejected(self):
        for value in (d.ROOT / "build/wrong", d.ROOT / ("other/" + d.OUTPUT_PREFIX + "fresh"),
                      d.ROOT / ("build/" + d.OUTPUT_PREFIX)):
            with self.assertRaises(ValueError):
                d.output_path(value)
        value = d.ROOT / ("build/" + d.OUTPUT_PREFIX + "test")
        with patch.object(Path, "exists", return_value=True):
            with self.assertRaisesRegex(ValueError, "already exists"):
                d.output_path(value)

    def test_no_rtl_subprocess_or_publication_surface(self):
        tree = ast.parse(Path(d.__file__).read_text())
        calls = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
                 for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, (ast.Name, ast.Attribute))}
        self.assertFalse(calls & {"system", "Popen", "publish_parent", "execute_layers", "stages"})

    def test_diagnose_rejects_foreign_validation_before_native(self):
        with patch.object(d, "authenticate") as authenticate, patch.object(d.upstream, "execute_layer") as run:
            with self.assertRaisesRegex(ValueError, "invalid pre-dispatch validation"):
                d.diagnose(Path("build/unused"), io.StringIO(), self.contract,
                           {"diagnostic_id": "foreign"})
            authenticate.assert_not_called()
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
