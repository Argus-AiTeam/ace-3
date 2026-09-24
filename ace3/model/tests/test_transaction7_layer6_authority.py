from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "ace3/model"
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

import emit_transaction7_layer6_authority_review as reviewer  # noqa: E402
import prepare_transaction7_layer6_authority as authority  # noqa: E402


class Transaction7Layer6AuthorityTest(unittest.TestCase):
    def test_authority_binds_exact_parent_target_package_and_launch(self) -> None:
        manifest = authority.validate_package()
        authority.validate_parent()
        document = authority.authority_document(manifest)
        self.assertEqual(document["status"], "AUTHORIZED_NOT_CONSUMED")
        self.assertEqual(document["authority_cardinality"], 1)
        self.assertEqual(document["authoritative_generation"], 7)
        self.assertEqual(document["authoritative_cursor"], 7)
        self.assertEqual(document["required_parent_checkpoint_index"], 6)
        self.assertEqual(document["target_generation"], 8)
        self.assertEqual(document["target_cursor"], 8)
        self.assertEqual(document["target_checkpoint_index"], 7)
        self.assertEqual(document["transaction_index"], 7)
        self.assertEqual(document["layer_index"], 6)
        self.assertEqual(document["authorized_launch"], authority.expected_launch())
        self.assertEqual(
            document["package_seal"]["sha256"],
            "1386cf9320fe54fba47e9f227f963301f95fae3328f6f4e2c442e9d61aaee46d",
        )
        self.assertEqual(
            document["package_acceptance"]["sha256"],
            "e503a873f7df236edd4c2fea548ed212fe527fcd8b924390edc22166deb0a266",
        )
        self.assertEqual(document["activity_counters"], authority.ZERO_COUNTERS)
        self.assertFalse(document["authority_consumed"])

    def test_40_hostile_controls_make_zero_workload_calls(self) -> None:
        manifest = authority.validate_package()
        document = authority.authority_document(manifest)
        controls, workload_counts = reviewer.adversarial_controls(document)
        self.assertEqual(len(controls), 40)
        self.assertEqual(len(set(controls)), 40)
        self.assertEqual(sum(workload_counts.values()), 0)
        self.assertEqual(
            {f"transaction{index:03d}-scope" for index in range(8, 26)},
            {control for control in controls if control.startswith("transaction")},
        )

    def test_reviewer_is_source_disjoint_from_workload(self) -> None:
        source = Path(reviewer.__file__).read_text(encoding="utf-8")
        reviewer.source_boundary(source)
        tree = ast.parse(source)
        imported = {
            alias.name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertTrue(reviewer.FORBIDDEN_IMPORTS.isdisjoint(imported))

    def test_published_authority_assesses_pass_without_review_write(self) -> None:
        if not authority.AUTHORITY_PACKAGE.exists():
            self.skipTest("sealed transaction007 authority is absent")
        document = authority.validate_authority_package()
        specification = importlib.util.spec_from_file_location(
            "sealed_transaction7_executor",
            authority.PACKAGE / "transaction7_executor.py",
        )
        self.assertIsNotNone(specification)
        self.assertIsNotNone(specification.loader)
        executor = importlib.util.module_from_spec(specification)
        sys.modules[specification.name] = executor
        specification.loader.exec_module(executor)
        self.assertEqual(executor.validate_authority(), document)
        status, controls, workload_counts, reason = reviewer.assess_authority(
            authority.AUTHORITY_PACKAGE,
            authority.AUTHORITY_REVIEW,
            require_output_absent=not authority.AUTHORITY_REVIEW.exists(),
        )
        self.assertEqual(status, "PASS", reason)
        self.assertEqual(len(controls), 40)
        self.assertEqual(sum(workload_counts.values()), 0)
        self.assertEqual(document["status"], "AUTHORIZED_NOT_CONSUMED")
        self.assertFalse(document["authority_consumed"])
        self.assertFalse(document["transaction007_executed"])
        self.assertFalse(document["generation8_exists"])

    def test_second_issuance_is_rejected(self) -> None:
        if not authority.AUTHORITY_PACKAGE.exists():
            self.skipTest("sealed transaction007 authority is absent")
        before = authority.file_record(authority.AUTHORITY)
        with self.assertRaises(authority.AuthorityError):
            authority.issue_authority()
        self.assertEqual(authority.file_record(authority.AUTHORITY), before)


if __name__ == "__main__":
    unittest.main()
