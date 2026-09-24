from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "ace3/model"
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

import prepare_transaction3_publication_recovery as recovery  # noqa: E402
import emit_transaction3_publication_recovery_review as reviewer  # noqa: E402
import emit_transaction3_publication_recovery_authority_review as authority_reviewer  # noqa: E402


TERMINAL = (
    b"schema=ace3_decoder_token_transaction_v1 layer_index=2 position=3 "
    b"natural_terminal=1 exit_code=0 trace_count=23408 final_count=896 "
    b"done_count=1\n"
)
COMPARISON = {
    "exact_integer_oracle_output_sha256": (
        "95e7911ce6afbc8dba048e3910b22489288fedfdeee33b47021622d937b67463"
    ),
    "implementation": "independent ACE-3 integer W4A16 oracle",
    "integer_mismatches": 0,
    "inter_layer_boundary": "binary16 round after every RTL layer",
    "rtl_matches_exact_integer_oracle": True,
}


class Transaction3PublicationRecoveryTest(unittest.TestCase):
    def test_natural_terminal_is_strictly_authenticated(self) -> None:
        self.assertEqual(
            recovery.parse_terminal(TERMINAL),
            {"trace_count": 23408, "final_count": 896, "done_count": 1},
        )
        for replacement in (
            (b"natural_terminal=1", b"natural_terminal=0"),
            (b"exit_code=0", b"exit_code=1"),
            (b"done_count=1", b"done_count=2"),
            (b"final_count=896", b"final_count=895"),
            (
                b"done_count=1",
                b"done_count=1 natural_terminal=1",
            ),
        ):
            with self.subTest(replacement=replacement):
                with self.assertRaises(recovery.RecoveryError):
                    recovery.parse_terminal(TERMINAL.replace(*replacement))

    def test_comparison_rejects_any_nonexact_result(self) -> None:
        recovery.validate_comparison(COMPARISON)
        mutations = (
            ("integer_mismatches", 1),
            ("rtl_matches_exact_integer_oracle", False),
            ("exact_integer_oracle_output_sha256", "0" * 64),
        )
        for key, value in mutations:
            changed = dict(COMPARISON)
            changed[key] = value
            with self.subTest(key=key):
                with self.assertRaises(recovery.RecoveryError):
                    recovery.validate_comparison(changed)

    def test_hidden_output_rejects_altered_rows_and_hash(self) -> None:
        rows = [
            f"00{index:04x}{(index * 17) & 0xffff:04x}\n"
            for index in range(recovery.HIDDEN_SIZE)
        ]
        payload = "".join(rows).encode("ascii")
        expected = hashlib.sha256(
            b"".join(
                (((index * 17) & 0xFFFF).to_bytes(2, "little"))
                for index in range(recovery.HIDDEN_SIZE)
            )
        ).hexdigest()
        self.assertEqual(recovery.semantic_hidden_sha256(payload), expected)
        changed = bytearray(payload)
        changed[-3] = ord("0") if changed[-3] != ord("0") else ord("1")
        self.assertNotEqual(
            recovery.semantic_hidden_sha256(bytes(changed)),
            expected,
        )
        with self.assertRaises(recovery.RecoveryError):
            recovery.semantic_hidden_sha256(payload.rsplit(b"\n", 2)[0] + b"\n")

    def test_inventory_rejects_altered_state_hash_and_partial_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "terminal.txt").write_bytes(TERMINAL)
            (root / "position004.state").write_bytes(b"verilatorsave01\nstate")
            inventory = recovery.tree_inventory(root)
            recovery.authenticate_inventory(root, inventory)
            (root / "position004.state").write_bytes(b"verilatorsave01\naltered")
            with self.assertRaises(recovery.RecoveryError):
                recovery.authenticate_inventory(root, inventory)
            (root / "position004.state").unlink()
            with self.assertRaises(recovery.RecoveryError):
                recovery.authenticate_inventory(root, inventory)

    def test_wrong_parent_and_replay_are_rejected(self) -> None:
        pointer = {
            "schema_version": 1,
            "status": "COMMITTED",
            "generation": 3,
            "runtime_identity": recovery.RUNTIME_IDENTITY,
        }
        ledger = {
            "state_generation": 3,
            "next_transaction_index": 3,
            "completed_transaction_count": 3,
            "authoritative_state_root": str(recovery.GENERATION3),
            "completed_receipts": [{}, {}, {}],
        }
        recovery.validate_parent(pointer, ledger)
        wrong = dict(pointer)
        wrong["generation"] = 2
        with self.assertRaises(recovery.RecoveryError):
            recovery.validate_parent(wrong, ledger)
        wrong_ledger = dict(ledger)
        wrong_ledger["next_transaction_index"] = 4
        with self.assertRaises(recovery.RecoveryError):
            recovery.validate_parent(pointer, wrong_ledger)

        original_load = recovery.load_json
        recovery.load_json = lambda path: ledger
        try:
            for mutation in (
                {"generation4_exists": True},
                {"staging_exists": True},
                {"receipt_candidate_exists": True},
                {"execution_evidence_exists": True},
                {"success_terminal_exists": True},
                {"later_transactions": [4]},
            ):
                arguments = {
                    "generation4_exists": False,
                    "staging_exists": False,
                    "receipt_candidate_exists": False,
                    "execution_evidence_exists": False,
                    "success_terminal_exists": False,
                    "later_transactions": [],
                }
                arguments.update(mutation)
                with self.subTest(mutation=mutation):
                    with self.assertRaises(recovery.RecoveryError):
                        recovery.require_publishable_namespace(
                            pointer,
                            **arguments,
                        )
        finally:
            recovery.load_json = original_load

    def test_receipt_timing_is_an_explicit_nonclaim(self) -> None:
        descriptor = {
            "operation": "position3-decoder-layer",
            "input_binding_sha256": "a" * 64,
            "required_result": {
                "exact_integer_oracle_match": True,
                "natural_rtl_terminal": True,
                "output_hidden_elements": 896,
                "output_state_position": 4,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hidden_path = root / "final.hex"
            state_path = root / "position004.state"
            hidden_path.write_bytes(b"x")
            state_path.write_bytes(b"y")
            receipt = {
                "schema_version": 1,
                "kind": "ace3_position3_transaction_completion",
                "status": "COMPLETE",
                "transaction_index": 3,
                "operation": descriptor["operation"],
                "input_binding_sha256": descriptor["input_binding_sha256"],
                "result": copy.deepcopy(descriptor["required_result"]),
                "outputs": {
                    "hidden": recovery.file_record(hidden_path),
                    "state": recovery.file_record(state_path),
                },
                "output_semantics": {
                    "hidden": {
                        "dtype": "FP16",
                        "elements": 896,
                        "semantic_sha256": recovery.EXPECTED_HIDDEN_SEMANTIC_SHA256,
                    },
                    "state": {"layer_index": 2, "position": 4},
                },
                "timing": {
                    "started_at": None,
                    "completed_at": None,
                    "transaction_seconds": None,
                    "basis": "transaction-003 timing was not durably preserved",
                },
            }
            recovery.validate_receipt(descriptor, receipt)
            receipt["timing"]["transaction_seconds"] = 1.0
            with self.assertRaises(recovery.RecoveryError):
                recovery.validate_receipt(descriptor, receipt)

    def test_source_has_no_child_process_model_or_rtl_entrypoint(self) -> None:
        source = Path(recovery.__file__).read_text(encoding="utf-8")
        recovery.validate_package_source_boundary(Path(recovery.__file__))
        tree = ast.parse(source)
        imported = {
            alias.name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertTrue(
            {"subprocess", "numpy", "torch", "importlib"}.isdisjoint(imported)
        )
        self.assertNotIn("run_command", source)
        self.assertNotIn("execute_exact_layer_transaction(", source)

    def test_reviewer_emitter_is_disjoint_and_runs_hostile_controls(self) -> None:
        source = Path(reviewer.__file__).read_text(encoding="utf-8")
        reviewer.source_boundary(source, reviewer_only=True)
        tree = ast.parse(source)
        imported = {
            alias.name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertTrue(
            {"subprocess", "numpy", "torch", "importlib", "fcntl", "shutil"}.isdisjoint(
                imported
            )
        )
        if recovery.DEFAULT_OUTPUT.exists():
            status, controls, reason = reviewer.assess_package(
                recovery.DEFAULT_OUTPUT,
                recovery.DEFAULT_REVIEW,
            )
            self.assertEqual(status, "PASS", reason)
            self.assertEqual(
                set(controls),
                {
                    "altered-terminal",
                    "altered-comparison",
                    "altered-output",
                    "altered-state",
                    "altered-hash",
                    "partial-artifacts",
                    "wrong-parent-generation",
                    "replay",
                    "executable-model-rtl-call",
                    "reviewer-publication-call",
                },
            )

    def test_review_reject_is_not_publication_eligibility(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "package"
            package.mkdir()
            (package / "package-seal.json").write_text("{}\n", encoding="ascii")
            (package / "adjudication.json").write_text("{}\n", encoding="ascii")
            review_path = root / "review.json"
            review = {
                "schema_version": 1,
                "kind": "ace3_transaction3_publication_recovery_independent_review",
                "status": "REJECT",
                "producer_role": "reviewer",
                "preparation_participation": False,
                "recovery_authority_created": False,
                "publication_performed": False,
                "package_seal": recovery.file_record(
                    package / "package-seal.json"
                ),
                "adjudication": recovery.file_record(
                    package / "adjudication.json"
                ),
            }
            review_path.write_bytes(recovery.canonical_json(review))
            review_path.chmod(0o444)
            with self.assertRaises(recovery.RecoveryError):
                recovery.validate_review(package, review_path)

    def test_authority_contract_rejects_binding_and_scope_mutations(self) -> None:
        if not recovery.DEFAULT_OUTPUT.exists():
            self.skipTest("sealed r7 package is absent")
        adjudication, _ = recovery.validate_accepted_r7(
            recovery.DEFAULT_OUTPUT,
            recovery.DEFAULT_REVIEW,
        )
        authority = recovery.authority_document(
            recovery.DEFAULT_OUTPUT,
            recovery.DEFAULT_REVIEW,
            recovery.DEFAULT_AUTHORITY_REVIEW,
            adjudication,
        )
        authority_reviewer.validate_authority_document(
            authority,
            authenticate=False,
        )
        mutations = (
            lambda item: item["frozen_evidence"]["comparison"].update(
                {"sha256": "0" * 64}
            ),
            lambda item: item["authoritative_parent"].update(
                {"latest_checkpoint_index": 3}
            ),
            lambda item: item["authorized_argv"].append("--replay"),
            lambda item: item["activity_counters"].update({"publication": 1}),
            lambda item: item.update({"new_transaction_identity": "transaction-004"}),
            lambda item: item["prohibitions"].remove("RTL simulation"),
        )
        for mutation in mutations:
            candidate = copy.deepcopy(authority)
            mutation(candidate)
            with self.subTest(candidate=candidate):
                with self.assertRaises(authority_reviewer.AuthorityReviewError):
                    authority_reviewer.validate_authority_document(
                        candidate,
                        authenticate=False,
                    )

    def test_authority_reviewer_runs_required_adversarial_controls(self) -> None:
        if not recovery.DEFAULT_OUTPUT.exists():
            self.skipTest("sealed r7 package is absent")
        adjudication, review = recovery.validate_accepted_r7(
            recovery.DEFAULT_OUTPUT,
            recovery.DEFAULT_REVIEW,
        )
        authority = recovery.authority_document(
            recovery.DEFAULT_OUTPUT,
            recovery.DEFAULT_REVIEW,
            recovery.DEFAULT_AUTHORITY_REVIEW,
            adjudication,
        )
        source = (
            recovery.DEFAULT_OUTPUT / "publication_recovery.py"
        ).read_text(encoding="utf-8")
        controls = authority_reviewer.adversarial_controls(
            authority,
            review,
            source,
        )
        self.assertEqual(
            set(controls),
            {
                "altered-frozen-evidence",
                "wrong-parentage",
                "rejected-r7-review",
                "writable-r7-review",
                "competing-authority",
                "replay-capable-argv",
                "forbidden-computation-import",
                "forbidden-computation-call",
                "pre-existing-generation4",
                "later-transaction-namespace",
            },
        )

    def test_authority_source_has_no_computation_entrypoint(self) -> None:
        source = Path(authority_reviewer.__file__).read_text(encoding="utf-8")
        authority_reviewer.source_boundary(source, reviewer_only=True)
        tree = ast.parse(source)
        imported = {
            alias.name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertTrue(
            {"subprocess", "numpy", "torch", "safetensors", "importlib"}.isdisjoint(
                imported
            )
        )

    def test_live_authority_namespace_rejects_competition_and_later_work(
        self,
    ) -> None:
        authority_reviewer.validate_live_namespace(
            generation4_exists=False,
            staging_exists=False,
            later_transactions=[],
            authority_count=1,
        )
        for mutation in (
            {"generation4_exists": True},
            {"staging_exists": True},
            {"later_transactions": [4]},
            {"authority_count": 2},
        ):
            arguments = {
                "generation4_exists": False,
                "staging_exists": False,
                "later_transactions": [],
                "authority_count": 1,
            }
            arguments.update(mutation)
            with self.subTest(mutation=mutation):
                with self.assertRaises(authority_reviewer.AuthorityReviewError):
                    authority_reviewer.validate_live_namespace(**arguments)


if __name__ == "__main__":
    unittest.main()
