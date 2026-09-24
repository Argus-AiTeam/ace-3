from __future__ import annotations

import ast
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "ace3/model"
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

import emit_transaction5_layer4_authority_review as reviewer  # noqa: E402
import prepare_transaction5_layer4_authority as authority  # noqa: E402
import prepare_transaction5_layer4_executor as executor  # noqa: E402


class Transaction5Layer4AuthorityTest(unittest.TestCase):
    def test_authority_binds_exact_accepted_boundary_and_launch(self) -> None:
        if not authority.AUTHORITY_PACKAGE.exists():
            self.skipTest("sealed transaction005 authority is absent")
        document = authority.validate_authority_package()
        self.assertEqual(executor.validate_authority(), document)
        manifest = authority.load_json(authority.PACKAGE_MANIFEST)
        self.assertEqual(document["status"], "AUTHORIZED_NOT_CONSUMED")
        self.assertEqual(document["producer_role"], "manager")
        self.assertEqual(document["authority_cardinality"], 1)
        self.assertEqual(document["transaction_index"], 5)
        self.assertEqual(document["layer_index"], 4)
        self.assertEqual(document["permitted_transaction_indices"], [5])
        self.assertEqual(
            document["forbidden_transaction_indices"],
            [0, 1, 2, 3, 4, *range(6, 26)],
        )
        self.assertEqual(
            document["package_seal"]["sha256"],
            "93d54e75d1dcb2316343770bda7bf404d4f61af9a42818cc28d11dd92e34c262",
        )
        self.assertEqual(
            document["package_manifest"]["sha256"],
            "a777568726ae600b946f9fb236dac45292bce003ace734c85eae22a404ce622e",
        )
        self.assertEqual(
            document["package_acceptance"]["sha256"],
            "60a2c22eaa95942629329e7c00e5b8779f33d72a6c9f8219fd37f6382e1964cb",
        )
        self.assertEqual(document["authorized_launch"], manifest["launch"])
        self.assertEqual(
            document["authoritative_parent"]["pointer"]["sha256"],
            "adee838870e758a5cc32a1a79248938b22d8e9d701501dce525c9098958bdda0",
        )
        self.assertEqual(
            document["authoritative_parent"]["checkpoint004"]["sha256"],
            "ff157b01b83d07d8bf167c420115659072a54415054454674625503d7decd610",
        )
        self.assertEqual(
            document["authoritative_parent"]["position004_input_state"]["sha256"],
            "bc17b1015874eba58edd49c6b29c74b37057576823711f5983476264de0c65ac",
        )
        self.assertEqual(document["activity_counters"], authority.ZERO_COUNTERS)
        self.assertFalse(document["authority_consumed"])
        self.assertFalse(document["replay_authorized"])

    def test_hostile_controls_reject_without_workload_calls(self) -> None:
        manifest, _ = authority.validate_accepted_package()
        document = (
            authority.load_json(authority.AUTHORITY)
            if authority.AUTHORITY_PACKAGE.exists()
            else authority.authority_document(manifest)
        )
        controls, workload_counts = reviewer.adversarial_controls(document)
        expected = {
            "wrong-parent",
            "wrong-transaction",
            "wrong-layer",
            "wrong-argv",
            "wrong-evidence",
            "wrong-namespace",
            "replay-authorized",
            "authority-consumed",
            *(f"transaction{index:03d}-scope" for index in range(6, 26)),
        }
        self.assertEqual(set(controls), expected)
        self.assertEqual(set(workload_counts), expected)
        self.assertTrue(all(count == 0 for count in workload_counts.values()))

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
        self.assertNotIn("execute_exact_layer_transaction(", source)
        self.assertNotIn("execute_layer_transaction(", source)

    def test_assessment_passes_unconsumed_with_zero_workload(self) -> None:
        if not authority.AUTHORITY_PACKAGE.exists():
            self.skipTest("sealed transaction005 authority is absent")
        status, controls, workload_counts, reason = reviewer.assess_authority(
            authority.AUTHORITY_PACKAGE,
            authority.AUTHORITY_REVIEW,
            require_output_absent=not authority.AUTHORITY_REVIEW.exists(),
        )
        self.assertEqual(status, "PASS", reason)
        self.assertEqual(len(controls), 28)
        self.assertEqual(sum(workload_counts.values()), 0)
        self.assertFalse(authority.TRANSACTION5.exists())
        self.assertFalse(authority.GENERATION6.exists())
        self.assertFalse(authority.FUTURE_ROOT.exists())

    def test_second_issuance_is_rejected_without_replacement(self) -> None:
        if not authority.AUTHORITY_PACKAGE.exists():
            self.skipTest("sealed transaction005 authority is absent")
        before = authority.file_record(authority.AUTHORITY)
        with self.assertRaises(authority.AuthorityError):
            authority.issue_authority()
        self.assertEqual(authority.file_record(authority.AUTHORITY), before)


if __name__ == "__main__":
    unittest.main()
