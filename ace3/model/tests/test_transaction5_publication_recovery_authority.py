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

import emit_transaction5_publication_recovery_authority_review as reviewer  # noqa: E402
import prepare_transaction5_publication_recovery_authority as authority  # noqa: E402


class Transaction5PublicationRecoveryAuthorityTest(unittest.TestCase):
    def test_zero_state_rejects_consumption_publication_and_later_work(self) -> None:
        zero = {
            "generation6_exists": False,
            "generation6_staging_exists": False,
            "publication_consumption_exists": False,
            "publication_terminal_exists": False,
            "publication_failure_exists": False,
            "later_transaction_indices": [],
        }
        authority.validate_zero_state(**zero)
        for mutation in (
            {"generation6_exists": True},
            {"generation6_staging_exists": True},
            {"publication_consumption_exists": True},
            {"publication_terminal_exists": True},
            {"publication_failure_exists": True},
            {"later_transaction_indices": [6]},
        ):
            candidate = dict(zero)
            candidate.update(mutation)
            with self.subTest(mutation=mutation):
                with self.assertRaises(authority.AuthorityError):
                    authority.validate_zero_state(**candidate)

    def test_authority_binds_exact_publish_argv_and_source(self) -> None:
        if not authority.DEFAULT_OUTPUT.exists():
            self.skipTest("sealed authority package is absent")
        document = authority.validate_authority_package()
        self.assertEqual(
            document["authorized_publish"]["argv"],
            authority.authorized_publish_argv(),
        )
        self.assertEqual(
            document["publication_source"],
            authority.file_record(
                authority.DEFAULT_OUTPUT / authority.PUBLICATION_SOURCE_NAME
            ),
        )
        self.assertEqual(document["producer_role"], "manager")
        self.assertTrue(document["manager_provenance"]["manager_direct"])
        self.assertEqual(
            document["original_transaction005_authority_consumption"][
                "consumption"
            ],
            authority.file_record(authority.ORIGINAL_CONSUMPTION),
        )
        self.assertEqual(
            document["frozen_transaction005_evidence"]["fail_closed_terminal"][
                "sha256"
            ],
            "12d0aa1e19a083b7027b1ec7568f7870bddf64389019eb5ad5703a1af0b029de",
        )
        self.assertEqual(
            document["frozen_transaction005_evidence"]["simulation_log"]["sha256"],
            "1b5af7c5641c1c9dd8d30eb74b4a9c10e3eb76397caa2bfc4ca8c5245f6cfd06",
        )
        self.assertEqual(document["permitted_operations"], authority.PERMITTED_OPERATIONS)
        self.assertIn("vector generation", document["prohibitions"])
        self.assertIn("new transaction identity", document["prohibitions"])
        self.assertEqual(document["activity_counters"], authority.ZERO_ACTIVITY)
        self.assertFalse(document["authority_consumed"])
        self.assertFalse(document["publication_performed"])
        self.assertFalse(document["generation6_exists"])
        self.assertTrue(document["transactions006_025_absent"])

    def test_authority_rejects_binding_and_counter_mutations(self) -> None:
        if not authority.DEFAULT_OUTPUT.exists():
            self.skipTest("sealed authority package is absent")
        document = authority.load_json(
            authority.DEFAULT_OUTPUT / authority.AUTHORITY_FILENAME
        )
        mutations = (
            lambda item: item.update({"producer_role": "engineer"}),
            lambda item: item[
                "original_transaction005_authority_consumption"
            ]["consumption"].update({"sha256": "0" * 64}),
            lambda item: item["prohibitions"].remove("vector generation"),
            lambda item: item["accepted_r1_recovery"]["independent_review"].update(
                {"sha256": "0" * 64}
            ),
            lambda item: item["frozen_transaction005_evidence"]["raw_final"].update(
                {"sha256": "0" * 64}
            ),
            lambda item: item["authoritative_parent"].update({"cursor": 4}),
            lambda item: item["publication_source"].update({"sha256": "0" * 64}),
            lambda item: item["authorized_publish"]["argv"].append("--retry"),
            lambda item: item["activity_counters"].update(
                {"authority_consumption": 1}
            ),
            lambda item: item["activity_counters"].update(
                {"generation6_publication": 1}
            ),
            lambda item: item["activity_counters"].update({"model_execution": 1}),
            lambda item: item["activity_counters"].update({"oracle_execution": 1}),
            lambda item: item["activity_counters"].update({"rtl_simulation": 1}),
            lambda item: item["activity_counters"].update({"transaction_retry": 1}),
            lambda item: item["activity_counters"].update({"transaction_replay": 1}),
            lambda item: item["activity_counters"].update({"transaction_resume": 1}),
            lambda item: item["activity_counters"].update(
                {"transactions006_025_execution": 1}
            ),
        )
        for mutation in mutations:
            candidate = copy.deepcopy(document)
            mutation(candidate)
            with self.subTest(candidate=candidate):
                with self.assertRaises(reviewer.ReviewError):
                    reviewer.validate_authority_document(
                        candidate, authenticate_records=False
                    )

    def test_sources_have_no_model_oracle_rtl_execution_entrypoint(self) -> None:
        for module in (authority, reviewer):
            path = Path(module.__file__)
            source = path.read_text(encoding="utf-8")
            reviewer.source_boundary(source)
            tree = ast.parse(source)
            imports = {
                alias.name.split(".", 1)[0]
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            }
            self.assertTrue(
                {"importlib", "numpy", "safetensors", "subprocess", "torch"}.isdisjoint(
                    imports
                )
            )
            calls = {
                node.func.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            } | {
                node.func.attr
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            }
            self.assertTrue(
                {
                    "execute_exact_layer_transaction",
                    "execute_layer_transaction",
                }.isdisjoint(calls)
            )

    def test_independent_review_accepts_with_all_zero_counters(self) -> None:
        if not authority.DEFAULT_OUTPUT.exists() or not authority.DEFAULT_REVIEW.exists():
            self.skipTest("sealed authority package or review is absent")
        review = authority.validate_authority_review()
        self.assertEqual(review["status"], "PASS", review["reason"])
        self.assertTrue(review["manager_provenance_verified"])
        self.assertTrue(review["original_consumption_binding_verified"])
        self.assertTrue(review["vector_generation_prohibited"])
        self.assertEqual(review["activity_counters"], authority.ZERO_ACTIVITY)
        self.assertFalse(review["transaction005_retry_replay_resume"])
        self.assertFalse(review["model_oracle_rtl_execution"])
        self.assertFalse(review["authority_consumed"])
        self.assertFalse(review["generation6_publication_performed"])
        self.assertTrue(review["transactions006_025_absent"])
        self.assertEqual(
            set(review["adversarial_controls"]),
            {
                "manager-provenance",
                "original-authority-consumption-binding",
                "vector-generation-prohibition",
                "rejected-r1-recovery-review",
                "altered-frozen-evidence",
                "wrong-parent",
                "altered-publication-source",
                "altered-publish-argv",
                "authority-consumption",
                "generation6-publication",
                "transaction005-retry",
                "transaction005-replay",
                "transaction005-resume",
                "model-execution",
                "oracle-execution",
                "vector-generation",
                "rtl-execution",
                "transaction006-artifact",
                "forbidden-computation-import",
            },
        )

    def test_sealed_authority_is_reviewable_without_emitting_review(self) -> None:
        if not authority.DEFAULT_OUTPUT.exists():
            self.skipTest("sealed authority package is absent")
        status, controls, reason = reviewer.assess(
            authority.DEFAULT_OUTPUT, authority.DEFAULT_REVIEW
        )
        self.assertEqual(status, "PASS", reason)
        self.assertIn("altered-publish-argv", controls)
        self.assertFalse(authority.DEFAULT_REVIEW.exists())

    def test_generation6_payloads_build_without_consuming_or_publishing(self) -> None:
        if not authority.DEFAULT_OUTPUT.exists() or not authority.DEFAULT_REVIEW.exists():
            self.skipTest("sealed authority package or review is absent")
        self.assertFalse(authority.PUBLICATION_CONSUMPTION.exists())
        self.assertFalse(authority.GENERATION6.exists())
        files, manifest, pointer = authority.generation6_payloads(
            authority.DEFAULT_OUTPUT, authority.DEFAULT_REVIEW
        )
        self.assertIn("checkpoints/transaction-005.json", files)
        self.assertIn(b'"generation": 6', manifest)
        self.assertIn(b'"generation": 6', pointer)
        self.assertFalse(authority.PUBLICATION_CONSUMPTION.exists())
        self.assertFalse(authority.GENERATION6.exists())


if __name__ == "__main__":
    unittest.main()
