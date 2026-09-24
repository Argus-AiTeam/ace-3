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

import emit_transaction4_publication_recovery_review as reviewer  # noqa: E402
import prepare_transaction4_publication_recovery as recovery  # noqa: E402


TERMINAL = (
    b"schema=ace3_decoder_token_transaction_v1 layer_index=3 position=3 "
    b"natural_terminal=1 exit_code=0 trace_count=23408 final_count=896 "
    b"done_count=1\n"
)
COMPARISON = {
    "exact_integer_oracle_output_sha256": (
        "152347b94d879e16b8e5d514ac7a0b38c954336f138bd3f5e27b57965d350cfa"
    ),
    "implementation": "independent ACE-3 integer W4A16 oracle",
    "integer_mismatches": 0,
    "inter_layer_boundary": "binary16 round after every RTL layer",
    "rtl_matches_exact_integer_oracle": True,
}


class Transaction4PublicationRecoveryTest(unittest.TestCase):
    def test_terminal_and_comparison_are_strict(self) -> None:
        self.assertEqual(
            recovery.parse_terminal(TERMINAL),
            {"trace_count": 23408, "final_count": 896, "done_count": 1},
        )
        recovery.validate_comparison(COMPARISON)
        for old, new in (
            (b"natural_terminal=1", b"natural_terminal=0"),
            (b"exit_code=0", b"exit_code=1"),
            (b"done_count=1", b"done_count=2"),
            (b"final_count=896", b"final_count=895"),
            (b"done_count=1", b"done_count=1 natural_terminal=1"),
        ):
            with self.subTest(old=old):
                with self.assertRaises(recovery.RecoveryError):
                    recovery.parse_terminal(TERMINAL.replace(old, new))
        altered = dict(COMPARISON)
        altered["integer_mismatches"] = 1
        with self.assertRaises(recovery.RecoveryError):
            recovery.validate_comparison(altered)

    def test_mapping_defect_is_bound_to_frozen_sources(self) -> None:
        finding = recovery.validate_mapping_sources()
        self.assertEqual(
            finding["terminal_parser_authenticates"],
            {"natural_terminal": 1, "exit_code": 0},
        )
        self.assertEqual(
            finding["terminal_parser_returns"],
            ["trace_count", "final_count", "done_count"],
        )
        self.assertIsNone(
            finding["returned_rtl_record_top_level_natural_terminal"]
        )
        self.assertIsNone(finding["returned_rtl_record_raw_natural_terminal"])
        self.assertEqual(
            finding["receipt_adapter_observed_value_type"], "NoneType"
        )
        self.assertIsNone(finding["receipt_adapter_observed_value"])
        self.assertEqual(finding["failing_expression"], "natural_terminal is True")
        self.assertFalse(finding["failing_expression_result"])
        self.assertTrue(finding["corrected_reconstructed_receipt_value"])
        self.assertIsNotNone(finding["terminal_parser"])
        self.assertIsNotNone(finding["returned_rtl_record_builder"])

    def test_namespace_rejects_replay_resume_and_later_work(self) -> None:
        zero = {
            "generation5_exists": False,
            "generation5_staging_exists": False,
            "receipt_candidate_exists": False,
            "execution_evidence_exists": False,
            "success_terminal_exists": False,
            "later_transactions": [],
        }
        recovery.validate_namespace(**zero)
        for mutation in (
            {"generation5_exists": True},
            {"generation5_staging_exists": True},
            {"receipt_candidate_exists": True},
            {"execution_evidence_exists": True},
            {"success_terminal_exists": True},
            {"later_transactions": [5]},
        ):
            candidate = dict(zero)
            candidate.update(mutation)
            with self.subTest(mutation=mutation):
                with self.assertRaises(recovery.RecoveryError):
                    recovery.validate_namespace(**candidate)

    def test_reconstructed_receipt_has_no_timing_or_execution_claim(self) -> None:
        adjudication = recovery.adjudicate()
        receipt = adjudication["reconstructed_receipt"]
        self.assertIsNone(receipt["timing"]["transaction_seconds"])
        self.assertEqual(
            receipt["result"],
            {
                "exact_integer_oracle_match": True,
                "natural_rtl_terminal": True,
                "output_hidden_elements": 896,
                "output_state_position": 4,
            },
        )
        self.assertEqual(
            receipt["publication_recovery"],
            {
                "kind": "frozen-evidence-receipt-reconstruction",
                "adapts_only": "natural_rtl_terminal None-to-True reconstruction",
                "model_rerun": False,
                "oracle_rerun": False,
                "rtl_rerun": False,
                "transaction_retry": False,
                "transaction_replay": False,
                "transaction_resume": False,
            },
        )
        self.assertEqual(
            adjudication["production_completion_validation"],
            recovery.validate_with_production_completion_validator(
                recovery.descriptor(), receipt
            ),
        )
        rejected = copy.deepcopy(receipt)
        rejected["result"]["natural_rtl_terminal"] = False
        with self.assertRaises(recovery.RecoveryError):
            recovery.validate_with_production_completion_validator(
                recovery.descriptor(), rejected
            )

    def test_adjudication_rejects_authority_or_publication_mutations(self) -> None:
        document = recovery.adjudicate()
        mutations = (
            lambda item: item["recovery_boundary"]["activity_counters"].update(
                {"authority_consumption": 1}
            ),
            lambda item: item["recovery_boundary"].update(
                {"authority_reusable": True}
            ),
            lambda item: item["recovery_boundary"].update(
                {"generation5_publication_authorized": True}
            ),
            lambda item: item["recovery_boundary"].update(
                {"separately_reviewed_successor_required": False}
            ),
            lambda item: item["recovery_boundary"].update(
                {"transactions005_025_absent": False}
            ),
            lambda item: item["source_boundary"].update(
                {"receipt_adapter_observed_value": 1}
            ),
            lambda item: item["production_completion_validation"].update(
                {"status": "REJECT"}
            ),
            lambda item: item["reconstructed_receipt"][
                "publication_recovery"
            ].update({"transaction_replay": True}),
        )
        for operation in mutations:
            candidate = copy.deepcopy(document)
            operation(candidate)
            with self.assertRaises(recovery.RecoveryError):
                recovery.validate_adjudication(candidate, authenticate_evidence=False)

    def test_sources_have_no_execution_or_publication_entrypoint(self) -> None:
        for module in (recovery, reviewer):
            path = Path(module.__file__)
            source = path.read_text(encoding="utf-8")
            recovery.validate_source_boundary(path)
            tree = ast.parse(source)
            imported = {
                alias.name.split(".", 1)[0]
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            }
            self.assertTrue(
                {"numpy", "subprocess", "torch"}.isdisjoint(imported)
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
                    "publish_generation5",
                }.isdisjoint(calls)
            )
            self.assertIn("spec_from_file_location", calls)
            self.assertIn("validate_completion_receipt", calls)

    def test_sealed_package_and_independent_review_accept(self) -> None:
        if not recovery.DEFAULT_OUTPUT.exists() or not recovery.DEFAULT_REVIEW.exists():
            self.skipTest("sealed recovery package or review is absent")
        adjudication = recovery.validate_package()
        review = recovery.load_json(recovery.DEFAULT_REVIEW)
        status, controls, reason = reviewer.assess_package(
            recovery.DEFAULT_OUTPUT, recovery.DEFAULT_REVIEW
        )
        self.assertEqual(status, "PASS", reason)
        self.assertEqual(review["status"], "PASS")
        self.assertFalse(review["institutional_role_authority_claimed"])
        self.assertFalse(review["transaction004_retry_replay_resume"])
        self.assertFalse(review["new_authority_consumed"])
        self.assertFalse(review["generation5_publication_authorized"])
        self.assertFalse(review["generation5_publication_performed"])
        self.assertTrue(review["separately_reviewed_successor_required"])
        self.assertIsNone(review["receipt_adapter_observed_value"])
        self.assertFalse(review["receipt_adapter_mapping_result"])
        self.assertEqual(
            review["production_completion_validator_result"], "PASS"
        )
        self.assertEqual(
            review["production_completion_validator"],
            recovery.EXPECTED_PRODUCTION_CONTROL,
        )
        self.assertEqual(
            set(controls),
            {
                "altered-terminal",
                "altered-comparison",
                "altered-mapping",
                "altered-production-validation",
                "retry",
                "replay",
                "resume",
                "authority-consumption",
                "generation5-publication",
                "transaction005-execution",
                "execution-import",
            },
        )
        self.assertEqual(
            adjudication["recovery_boundary"]["activity_counters"],
            recovery.ZERO_ACTIVITY,
        )


if __name__ == "__main__":
    unittest.main()
