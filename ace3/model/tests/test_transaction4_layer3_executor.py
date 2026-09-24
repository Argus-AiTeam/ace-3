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

import emit_transaction4_layer3_executor_review as reviewer  # noqa: E402
import prepare_transaction4_layer3_executor as transaction4  # noqa: E402


class Transaction4Layer3ExecutorTest(unittest.TestCase):
    def test_exact_scope_and_launch_are_fixed(self) -> None:
        launch = transaction4.launch_contract()
        self.assertEqual(transaction4.TRANSACTION_INDEX, 4)
        self.assertEqual(transaction4.LAYER_INDEX, 3)
        self.assertEqual(transaction4.PERMITTED_INDICES, [4])
        self.assertEqual(
            transaction4.FORBIDDEN_INDICES,
            [0, 1, 2, 3, *range(5, 26)],
        )
        self.assertEqual(launch["cwd"], str(ROOT))
        self.assertEqual(launch["argv"][-1], "4")
        self.assertEqual(launch["start_cursor"], 4)
        self.assertEqual(launch["exit_cursor"], 5)

    def test_descriptor_is_transaction004_layer03_only(self) -> None:
        descriptor = transaction4.transaction_descriptor()
        self.assertEqual(descriptor["transaction_index"], 4)
        self.assertEqual(descriptor["layer_index"], 3)
        self.assertEqual(
            descriptor["inputs"]["predecessor"]["source_transaction_index"],
            3,
        )
        self.assertEqual(descriptor["inputs"]["transaction_position"], 3)

    def test_wrong_scope_and_nonzero_state_are_rejected(self) -> None:
        if not transaction4.PACKAGE_ROOT.exists():
            self.skipTest("sealed transaction004 package is absent")
        manifest = transaction4.load_json(transaction4.PACKAGE_MANIFEST)
        mutations = (
            lambda item: item.update({"transaction_index": 5}),
            lambda item: item.update({"layer_index": 4}),
            lambda item: item.update({"permitted_transaction_indices": [4, 5]}),
            lambda item: item.update({"required_parent_checkpoint_index": 2}),
            lambda item: item["activity_counters"].update(
                {"transaction_execution": 1}
            ),
            lambda item: item.update({"execution_authorized": True}),
        )
        for mutate in mutations:
            candidate = copy.deepcopy(manifest)
            mutate(candidate)
            with self.assertRaises(transaction4.Transaction4Error):
                transaction4.validate_manifest_document(candidate)

    def test_reviewer_is_disjoint_from_execution_imports(self) -> None:
        source = Path(reviewer.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {
            alias.name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertTrue(
            {"importlib", "numpy", "safetensors", "subprocess", "torch"}.isdisjoint(
                imported
            )
        )
        self.assertNotIn("execute_exact_layer_transaction(", source)
        self.assertNotIn("manager-authority.json\", \"w", source)

    def test_live_parent_and_zero_namespace_validate(self) -> None:
        ledger = transaction4.validate_generation4()
        self.assertEqual(ledger["state_generation"], 4)
        self.assertEqual(ledger["next_transaction_index"], 4)
        transaction4.validate_absence()

    def test_sealed_package_and_hostile_controls_validate(self) -> None:
        if not transaction4.PACKAGE_ROOT.exists():
            self.skipTest("sealed transaction004 package is absent")
        manifest = transaction4.validate_package()
        status, controls, reason = reviewer.assess_package(
            transaction4.PACKAGE_ROOT
        )
        self.assertEqual(status, "PASS", reason)
        self.assertEqual(
            set(controls),
            {
                "wrong-transaction",
                "wrong-layer",
                "wrong-parent",
                "broad-scope",
                "nonzero-counter",
                "premature-authority",
            },
        )
        self.assertEqual(manifest["activity_counters"], transaction4.ZERO_COUNTERS)


if __name__ == "__main__":
    unittest.main()
