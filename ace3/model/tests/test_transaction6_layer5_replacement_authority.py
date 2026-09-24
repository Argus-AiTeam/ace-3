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

import emit_transaction6_layer5_replacement_authority_review as reviewer  # noqa: E402
import prepare_transaction6_layer5_replacement_authority as authority  # noqa: E402


class Transaction6Layer5ReplacementAuthorityTest(unittest.TestCase):
    def test_authority_binds_r4_host_parent_launch_and_supersession(self) -> None:
        manifest = authority.validate_package()
        authority.validate_parent()
        document = authority.authority_document(manifest)
        self.assertEqual(document["status"], "AUTHORIZED_NOT_CONSUMED")
        self.assertEqual(document["authority_cardinality"], 1)
        self.assertEqual(document["authoritative_generation"], 6)
        self.assertEqual(document["authoritative_cursor"], 6)
        self.assertEqual(document["required_parent_checkpoint_index"], 5)
        self.assertEqual(document["transaction_index"], 6)
        self.assertEqual(document["layer_index"], 5)
        self.assertEqual(document["authorized_launch"], authority.expected_launch())
        self.assertEqual(
            document["host_reviewer_decision"]["sha256"],
            "5dc0031bcf4ec67244da03813b8a8434bc236b5b6699b8bd08c1e034c784158b",
        )
        self.assertEqual(
            document["package_acceptance"]["sha256"],
            "7bf470cf31e5e3b1185e051d2191c8e739409e69c0af485180f528e4ff704a1a",
        )
        self.assertEqual(
            document["supersession"],
            authority.expected_supersession(),
        )
        self.assertEqual(document["activity_counters"], authority.ZERO_COUNTERS)
        self.assertFalse(document["authority_consumed"])

    def test_29_hostile_controls_make_zero_workload_calls(self) -> None:
        manifest = authority.validate_package()
        document = authority.authority_document(manifest)
        controls, workload_counts = reviewer.adversarial_controls(document)
        self.assertEqual(len(controls), 29)
        self.assertEqual(len(set(controls)), 29)
        self.assertEqual(sum(workload_counts.values()), 0)

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

    def test_published_authority_is_package_compatible_and_passes(self) -> None:
        if not authority.AUTHORITY_PACKAGE.exists():
            self.skipTest("sealed r4 replacement authority is absent")
        document = authority.validate_authority_package()
        specification = importlib.util.spec_from_file_location(
            "sealed_transaction6_executor",
            authority.PACKAGE / "transaction6_executor.py",
        )
        self.assertIsNotNone(specification)
        self.assertIsNotNone(specification.loader)
        module = importlib.util.module_from_spec(specification)
        sys.modules[specification.name] = module
        specification.loader.exec_module(module)
        self.assertEqual(module.validate_authority(), document)
        status, controls, workload_counts, reason = reviewer.assess_authority(
            authority.AUTHORITY_PACKAGE,
            authority.AUTHORITY_REVIEW,
            require_output_absent=not authority.AUTHORITY_REVIEW.exists(),
        )
        self.assertEqual(status, "PASS", reason)
        self.assertEqual(len(controls), 29)
        self.assertEqual(sum(workload_counts.values()), 0)
        review = reviewer.load_json(authority.AUTHORITY_REVIEW)
        self.assertEqual(review["status"], "PASS")
        self.assertEqual(len(review["adversarial_controls"]), 29)
        self.assertEqual(review["total_workload_calls"], 0)
        self.assertTrue(
            all(
                calls == 0
                for calls in review["hostile_control_workload_calls"].values()
            )
        )
        self.assertEqual(review["authority_cardinality"], 1)
        self.assertFalse(review["authority_consumed"])
        self.assertFalse(review["transaction006_executed"])
        self.assertFalse(review["generation7_exists"])

    def test_second_issuance_is_rejected(self) -> None:
        if not authority.AUTHORITY_PACKAGE.exists():
            self.skipTest("sealed r4 replacement authority is absent")
        before = authority.file_record(authority.AUTHORITY)
        with self.assertRaises(authority.AuthorityError):
            authority.issue_authority()
        self.assertEqual(authority.file_record(authority.AUTHORITY), before)


if __name__ == "__main__":
    unittest.main()
