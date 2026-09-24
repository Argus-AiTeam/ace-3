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

import emit_transaction4_publication_recovery_authority_review as reviewer  # noqa: E402
import prepare_transaction4_publication_recovery_authority as authority  # noqa: E402


class Transaction4PublicationRecoveryAuthorityTest(unittest.TestCase):
    def test_zero_state_rejects_consumption_publication_and_later_work(self) -> None:
        zero = {
            "generation5_exists": False,
            "generation5_staging_exists": False,
            "publication_consumption_exists": False,
            "publication_terminal_exists": False,
            "publication_failure_exists": False,
            "later_transaction_indices": [],
        }
        authority.validate_zero_state(**zero)
        for mutation in (
            {"generation5_exists": True},
            {"generation5_staging_exists": True},
            {"publication_consumption_exists": True},
            {"publication_terminal_exists": True},
            {"publication_failure_exists": True},
            {"later_transaction_indices": [5]},
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
            document["original_transaction004_authority_consumption"][
                "consumption"
            ],
            authority.file_record(authority.ORIGINAL_CONSUMPTION),
        )
        self.assertEqual(
            document["frozen_transaction004_evidence"]["fail_closed_terminal"][
                "sha256"
            ],
            "a2815d8f9fbf53ff63f5af1f9646008ef42826ada52fa0e9c9364819477e0e72",
        )
        self.assertEqual(
            document["frozen_transaction004_evidence"]["simulation_log"]["sha256"],
            "b2b0e11ecdaa947af0ed6c659f876c8e1d8e18d562d1a19c048cb64f158fec5e",
        )
        self.assertEqual(document["permitted_operations"], authority.PERMITTED_OPERATIONS)
        self.assertIn("vector generation", document["prohibitions"])
        self.assertIn("new transaction identity", document["prohibitions"])
        self.assertEqual(document["activity_counters"], authority.ZERO_ACTIVITY)
        self.assertFalse(document["authority_consumed"])
        self.assertFalse(document["publication_performed"])
        self.assertFalse(document["generation5_exists"])
        self.assertTrue(document["transactions005_025_absent"])

    def test_authority_rejects_binding_and_counter_mutations(self) -> None:
        if not authority.DEFAULT_OUTPUT.exists():
            self.skipTest("sealed authority package is absent")
        document = authority.load_json(
            authority.DEFAULT_OUTPUT / authority.AUTHORITY_FILENAME
        )
        mutations = (
            lambda item: item.update({"producer_role": "engineer"}),
            lambda item: item[
                "original_transaction004_authority_consumption"
            ]["consumption"].update({"sha256": "0" * 64}),
            lambda item: item["prohibitions"].remove("vector generation"),
            lambda item: item["accepted_r3_recovery"]["independent_review"].update(
                {"sha256": "0" * 64}
            ),
            lambda item: item["frozen_transaction004_evidence"]["raw_final"].update(
                {"sha256": "0" * 64}
            ),
            lambda item: item["authoritative_parent"].update({"cursor": 5}),
            lambda item: item["publication_source"].update({"sha256": "0" * 64}),
            lambda item: item["authorized_publish"]["argv"].append("--retry"),
            lambda item: item["activity_counters"].update(
                {"authority_consumption": 1}
            ),
            lambda item: item["activity_counters"].update(
                {"generation5_publication": 1}
            ),
            lambda item: item["activity_counters"].update({"model_execution": 1}),
            lambda item: item["activity_counters"].update({"oracle_execution": 1}),
            lambda item: item["activity_counters"].update({"rtl_simulation": 1}),
            lambda item: item["activity_counters"].update({"transaction_retry": 1}),
            lambda item: item["activity_counters"].update({"transaction_replay": 1}),
            lambda item: item["activity_counters"].update({"transaction_resume": 1}),
            lambda item: item["activity_counters"].update(
                {"transactions005_025_execution": 1}
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
        self.assertFalse(review["transaction004_retry_replay_resume"])
        self.assertFalse(review["model_oracle_rtl_execution"])
        self.assertFalse(review["authority_consumed"])
        self.assertFalse(review["generation5_publication_performed"])
        self.assertTrue(review["transactions005_025_absent"])
        self.assertEqual(
            set(review["adversarial_controls"]),
            {
                "manager-provenance",
                "original-authority-consumption-binding",
                "vector-generation-prohibition",
                "rejected-r3-recovery-review",
                "altered-frozen-evidence",
                "wrong-parent",
                "altered-publication-source",
                "altered-publish-argv",
                "authority-consumption",
                "generation5-publication",
                "transaction004-retry",
                "transaction004-replay",
                "transaction004-resume",
                "model-execution",
                "oracle-execution",
                "vector-generation",
                "rtl-execution",
                "transaction005-artifact",
                "forbidden-computation-import",
            },
        )

    def test_generation5_payloads_build_without_consuming_or_publishing(self) -> None:
        if not authority.DEFAULT_OUTPUT.exists() or not authority.DEFAULT_REVIEW.exists():
            self.skipTest("sealed authority package or review is absent")
        self.assertFalse(authority.PUBLICATION_CONSUMPTION.exists())
        self.assertFalse(authority.GENERATION5.exists())
        files, manifest, pointer = authority.generation5_payloads(
            authority.DEFAULT_OUTPUT, authority.DEFAULT_REVIEW
        )
        self.assertIn("checkpoints/transaction-004.json", files)
        self.assertIn(b'"generation": 5', manifest)
        self.assertIn(b'"generation": 5', pointer)
        self.assertFalse(authority.PUBLICATION_CONSUMPTION.exists())
        self.assertFalse(authority.GENERATION5.exists())


if __name__ == "__main__":
    unittest.main()
