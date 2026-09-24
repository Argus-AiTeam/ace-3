from __future__ import annotations

import ast
import copy
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "ace3/model"
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

import emit_transaction5_layer4_executor_review as reviewer  # noqa: E402
import prepare_transaction5_layer4_executor as transaction5  # noqa: E402


class Transaction5Layer4ExecutorTest(unittest.TestCase):
    def test_exact_scope_and_future_commands_are_fixed(self) -> None:
        launch = transaction5.launch_contract()
        self.assertEqual(transaction5.TRANSACTION_INDEX, 5)
        self.assertEqual(transaction5.LAYER_INDEX, 4)
        self.assertEqual(transaction5.PERMITTED_INDICES, [5])
        self.assertEqual(
            transaction5.FORBIDDEN_INDICES,
            [0, 1, 2, 3, 4, *range(6, 26)],
        )
        self.assertEqual(launch["cwd"], str(ROOT))
        self.assertEqual(launch["argv"][-1], "5")
        self.assertEqual(launch["start_cursor"], 5)
        self.assertEqual(launch["exit_cursor"], 6)
        self.assertIn(
            "MODEL24_RTL_LAYER_INDEX=4",
            transaction5.compile_argv(),
        )
        self.assertEqual(transaction5.simulation_argv()[2], "4")
        self.assertEqual(
            transaction5.simulation_argv()[-1],
            str(
                transaction5.transaction_descriptor()["inputs"][
                    "position2_kv_parent"
                ]["path"]
            ),
        )

    def test_descriptor_is_transaction005_layer04_only(self) -> None:
        descriptor = transaction5.transaction_descriptor()
        self.assertEqual(descriptor["transaction_index"], 5)
        self.assertEqual(descriptor["layer_index"], 4)
        self.assertEqual(
            descriptor["inputs"]["predecessor"]["source_transaction_index"],
            4,
        )
        self.assertEqual(descriptor["inputs"]["transaction_position"], 3)

    def test_parent_acceptance_and_zero_namespace_validate(self) -> None:
        ledger = transaction5.validate_parent_acceptance()
        self.assertEqual(ledger["state_generation"], 5)
        self.assertEqual(ledger["next_transaction_index"], 5)
        transaction5.validate_absence()

    def test_manifest_rejects_wrong_bindings(self) -> None:
        if not transaction5.PACKAGE_ROOT.exists():
            self.skipTest("sealed transaction005 package is absent")
        manifest = transaction5.load_json(transaction5.PACKAGE_MANIFEST)
        mutations = (
            lambda item: item.update({"transaction_index": 6}),
            lambda item: item.update({"layer_index": 5}),
            lambda item: item.update({"authoritative_generation": 4}),
            lambda item: item["position004_input_state"].update(
                {"sha256": "0" * 64}
            ),
            lambda item: item["official_frozen_evidence"][
                "model_checkpoint"
            ].update({"sha256": "0" * 64}),
            lambda item: item["official_frozen_evidence"][
                "tied_projection"
            ].update({"sha256": "0" * 64}),
            lambda item: item["compile"]["argv"].__setitem__(
                3,
                "MODEL24_RTL_LAYER_INDEX=3",
            ),
            lambda item: item["simulation"]["argv"].__setitem__(2, "3"),
            lambda item: item.update({"permitted_transaction_indices": [5, 6]}),
            lambda item: item["activity_counters"].update(
                {"transaction_execution": 1}
            ),
            lambda item: item.update({"execution_authorized": True}),
        )
        for mutate in mutations:
            candidate = copy.deepcopy(manifest)
            mutate(candidate)
            with self.assertRaises(transaction5.Transaction5Error):
                transaction5.validate_manifest_document(candidate)

    def test_reviewer_is_source_disjoint_and_nonexecuting(self) -> None:
        source = Path(reviewer.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {
            alias.name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertTrue(
            {
                "fcntl",
                "importlib",
                "numpy",
                "safetensors",
                "shutil",
                "subprocess",
                "torch",
            }.isdisjoint(imported)
        )
        self.assertNotIn("execute_exact_layer_transaction(", source)
        self.assertNotIn("publish_generation6(", source)
        self.assertNotIn("manager-authority.json\", \"w", source)

    def test_sealed_package_and_hostile_controls_validate(self) -> None:
        if not transaction5.PACKAGE_ROOT.exists():
            self.skipTest("sealed transaction005 package is absent")
        if transaction5.REVIEW.exists():
            manifest = transaction5.validate_package(require_zero_runtime=False)
            review = transaction5.validate_review()
            status = review["status"]
            controls = review["adversarial_controls"]
            reason = review.get("reason")
        else:
            manifest = transaction5.validate_package()
            status, controls, reason = reviewer.assess_package(
                transaction5.PACKAGE_ROOT
            )
        self.assertEqual(status, "PASS", reason)
        self.assertEqual(
            set(controls),
            {
                "wrong-transaction",
                "wrong-layer",
                "wrong-parent",
                "wrong-position004-state",
                "wrong-model",
                "wrong-projection",
                "wrong-compile-argv",
                "wrong-simulation-argv",
                "wrong-output-namespace",
                "broad-scope",
                "nonzero-counter",
                "premature-authority",
            },
        )
        self.assertEqual(manifest["activity_counters"], transaction5.ZERO_COUNTERS)
        self.assertFalse(transaction5.TRANSACTION5.exists())
        self.assertFalse(transaction5.GENERATION6.exists())


if __name__ == "__main__":
    unittest.main()
