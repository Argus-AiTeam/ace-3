#!/usr/bin/env python3
"""Focused contract tests for the position-2 terminal status seal."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / "ace3/contracts/position2_canonical_evidence_status.json"
SHA256 = re.compile(r"[0-9a-f]{64}")


class Position2CanonicalEvidenceStatusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.status = json.loads(CONTRACT.read_text(encoding="utf-8"))

    def test_records_v10_candidate_without_publishing_canonical_pass(self) -> None:
        self.assertEqual(
            self.status["status"],
            "V10_RUNTIME_PASS_AWAITING_INDEPENDENT_REVIEW",
        )
        self.assertEqual(
            self.status["canonical_pass_evidence"],
            {
                "path": "build/model24_selected_token_position2/evidence.json",
                "state_at_seal": (
                    "ABSENT_PENDING_INDEPENDENT_V10_ACCEPTANCE"
                ),
                "position2_runtime_pass_claimed": False,
            },
        )
        identity = self.status["consumed_identity"]
        self.assertEqual(
            identity["package_id"],
            "ace3-position2-fresh-v6-20260831t081500z",
        )
        self.assertEqual(identity["terminal_state"], "RUNTIME_FAIL_NON_RETRIABLE")
        self.assertEqual(identity["durable_runner"]["state"], "error")
        self.assertEqual(identity["durable_runner"]["exit_code"], 2)
        self.assertEqual(identity["generate_exit_code"], 1)
        self.assertFalse(identity["validate_invoked"])
        self.assertTrue(identity["authority_consumed"])
        self.assertEqual(identity["successful_output"]["state_at_seal"], "ABSENT")

    def test_records_required_canonical_counts_once(self) -> None:
        self.assertEqual(
            self.status["canonical_lifecycle_counts"],
            {
                "authority": 1,
                "submission": 1,
                "payload": 1,
                "generate_runtime": 1,
                "validator": 0,
                "verdict": 1,
            },
        )
        observed = self.status["observed_terminal_counts"]
        self.assertEqual(observed["authority_consumptions"], 1)
        self.assertEqual(observed["durable_runner_submissions"], 1)
        self.assertEqual(observed["payload_invocations"], 1)
        self.assertEqual(observed["driver_invocations"], 1)
        self.assertEqual(observed["generate_invocations"], 1)
        self.assertEqual(observed["validate_invocations"], 0)
        self.assertEqual(observed["terminal_manifests"], 1)
        self.assertEqual(observed["terminal_review_verdicts"], 1)
        for key in ("retries", "replays", "resumes", "relaunches"):
            self.assertEqual(observed[key], 0)

    def test_terminal_review_is_read_only_and_non_claim_bearing(self) -> None:
        review = self.status["terminal_failure_review"]
        self.assertEqual(review["mode"], "READ_ONLY_POST_TERMINAL")
        self.assertEqual(
            review["verdict"],
            "ACCEPTED_AUTHENTICATED_NON_CLAIM_BEARING_FAILURE",
        )
        self.assertEqual(review["verdict_count"], 1)
        self.assertEqual(review["runtime_invocations_by_review"], 0)
        self.assertEqual(review["validator_invocations_by_review"], 0)
        self.assertFalse(review["authority_reused_by_review"])

    def test_binds_terminal_logs_and_partial_layer0_artifacts(self) -> None:
        evidence = self.status["authenticated_evidence"]
        required = {
            "accepted_package_manifest",
            "accepted_package_review",
            "publication_attestation",
            "source_archive",
            "runtime_driver",
            "execution_authority",
            "authority_review",
            "authority_consumed",
            "payload_invocation",
            "durable_runner_receipt",
            "runner_stdout",
            "runner_stderr",
            "runner_exit_code",
            "runtime_summary",
            "exactly_once_terminal",
            "generate_invocation",
            "generate_terminal",
            "generate_stdout",
            "generate_stderr",
            "layer00_compile_log",
            "layer00_compiled_simulator",
            "layer00_position000_simulation_log",
            "layer00_vector_manifest",
            "layer00_position000_inputs",
            "layer00_position000_rope_coefficients",
        }
        self.assertEqual(set(evidence), required)
        for record in evidence.values():
            self.assertTrue(
                record["path"].startswith("build/")
                or record["path"].startswith(".argus_subagents/")
            )
            self.assertGreaterEqual(record["bytes"], 0)
            self.assertIsNotNone(SHA256.fullmatch(record["sha256"]))
        for key in ("runner_stdout", "runner_stderr", "generate_stdout"):
            self.assertEqual(evidence[key]["bytes"], 0)
            self.assertEqual(
                evidence[key]["sha256"],
                "e3b0c44298fc1c149afbf4c8996fb924"
                "27ae41e4649b934ca495991b7852b855",
            )

    def test_classifies_missing_replay_vectors_without_a_runtime_claim(self) -> None:
        failure = self.status["failure_boundary"]
        self.assertEqual(
            failure["classification"],
            "POSITION2_LAYER0_REPLAY_VECTOR_INPUT_FAIL",
        )
        self.assertEqual(
            failure["claim_classification"],
            "AUTHENTICATED_NON_CLAIM_BEARING_FAILURE",
        )
        self.assertIn("cannot open", failure["first_observed_error"])
        self.assertTrue(failure["first_observed_error"].endswith("/trace.hex"))
        self.assertEqual(
            {
                Path(path).name
                for path in failure["required_replay_vector_files_absent"]
            },
            {"trace.hex", "final.hex", "boundary_manifest.json"},
        )
        self.assertEqual(failure["compiled_layers"], [0])
        self.assertEqual(failure["validated_layers"], [])
        self.assertEqual(failure["integer_oracle_comparisons"], 0)
        self.assertEqual(failure["position2_layer_review"], "NOT_REACHED")

    def test_preserves_absent_validator_and_pass_evidence(self) -> None:
        absent = self.status["absent_terminal_evidence"]
        self.assertEqual(
            set(absent),
            {
                "canonical_pass",
                "runtime_output",
                "validate_invocation",
                "validate_terminal",
            },
        )
        self.assertTrue(absent["canonical_pass"].endswith("/evidence.json"))
        self.assertIn(
            "canonical publication withheld",
            self.status["claim_boundary"]["position2_runtime"],
        )

    def test_consumed_identity_cannot_advance(self) -> None:
        self.assertEqual(
            set(self.status["no_relaunch_boundary"]["prohibited"]),
            {
                "retry",
                "resume",
                "replay",
                "relaunch",
                "reuse_consumed_identity",
                "validate_consumed_identity",
            },
        )
        self.assertIn(
            "ace3-position2-fresh-v5-20260831T032100Z",
            self.status["previous_failed_identities"],
        )
        for boundary in self.status["advancement_boundary"].values():
            self.assertEqual(boundary["invocations_by_this_seal"], 0)
            self.assertFalse(boundary["claim_advanced"])
        self.assertEqual(
            self.status["review_boundary"],
            {
                "v6_terminal_failure_review": "COMPLETE",
                "fresh_successor_package_review": "PASS",
                "fresh_successor_authority_review": (
                    "L2_PASS_UNCONSUMED_READY_FOR_SEPARATE_RUNTIME_PASS_TASK"
                ),
                "v8_consumption_attempt_review": "PENDING",
                "v9_consumption_attempt_review": "PENDING",
                "v10_runtime_pass_review": (
                    "PENDING_SINGLE_NO_EXECUTION_INDEPENDENT_REVIEW"
                ),
                "next_owner": "reviewer",
            },
        )

    def test_authenticates_v10_canonical_runtime_and_causal_lineage(
        self,
    ) -> None:
        attempt = self.status["v10_runtime_submission"]
        self.assertEqual(
            attempt["status"],
            "RUNTIME_PASS_AWAITING_INDEPENDENT_REVIEW",
        )
        self.assertEqual(attempt["durable_runner"]["submissions"], 1)
        self.assertEqual(attempt["durable_runner"]["exit_code"], 0)
        self.assertEqual(
            attempt["authority_consumption"],
            {
                "manager_designated_canonical_runtime_authority": True,
                "cardinality": 1,
                "basis": (
                    "one exclusive durable-runner receipt and one driver "
                    "invocation"
                ),
                "separate_authority_artifact": "NOT_MATERIALIZED",
                "independent_runtime_review": "PENDING",
            },
        )
        counts = attempt["lifecycle_counts"]
        self.assertEqual(counts["canonical_validator_invocations"], 2)
        self.assertEqual(counts["generate_invocations"], 1)
        self.assertEqual(counts["validate_invocations"], 1)
        self.assertEqual(counts["accepted_24_layer_traversals"], 1)
        self.assertEqual(counts["terminal_manifests"], 1)
        for key in ("retries", "replays", "resumes", "relaunches"):
            self.assertEqual(counts[key], 0)

        for record in attempt["authenticated_evidence"].values():
            path = ROOT / record["path"]
            self.assertTrue(path.is_file())
            self.assertFalse(path.is_symlink())
            self.assertEqual(path.stat().st_size, record["bytes"])
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                record["sha256"],
            )

        runtime_root = (
            ROOT
            / "build/model24_selected_token_position2_runs"
            / attempt["runtime_identity"]
        )
        summary = json.loads(
            (runtime_root / "v10-runtime-submission.json").read_text(
                encoding="utf-8"
            )
        )
        evidence = json.loads(
            (runtime_root / "evidence.json").read_text(encoding="utf-8")
        )
        terminal = json.loads(
            (runtime_root / "terminal.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["status"], attempt["status"])
        self.assertEqual(terminal["status"], attempt["status"])
        self.assertEqual(
            summary["canonical_validator"]["invocation_counts"],
            {"generate": 1, "validate": 1, "total": 2},
        )
        self.assertEqual(summary["canonical_validator"]["verdict"], "PASS")
        self.assertEqual(
            terminal["counters"]["accepted_24_layer_traversals"],
            1,
        )

        binding = attempt["source_and_fixture_bindings"]
        source_path = summary["accepted_source_path"]
        self.assertEqual(
            source_path["source_archive"]["sha256"],
            binding["source_archive"]["sha256"],
        )
        self.assertEqual(
            source_path["source_set"]["canonical_sha256"],
            binding["accepted_source_set"]["canonical_sha256"],
        )
        accepted_sources = {
            item["path"]: item for item in source_path["source_set"]["files"]
        }
        consumed_sources = source_path[
            "canonical_evidence_consumed_sources"
        ]
        self.assertEqual(len(accepted_sources), 30)
        self.assertEqual(len(consumed_sources), 30)
        extracted_root = runtime_root / "source"
        for record in consumed_sources.values():
            path = Path(record["path"])
            relative = path.relative_to(extracted_root).as_posix()
            self.assertIn(relative, accepted_sources)
            self.assertEqual(record["bytes"], accepted_sources[relative]["bytes"])
            self.assertEqual(
                record["sha256"],
                accepted_sources[relative]["sha256"],
            )
            self.assertEqual(path.stat().st_size, record["bytes"])
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                record["sha256"],
            )

        fixtures = summary["fixture_checkpoint_bindings"]
        self.assertEqual(fixtures["checkpoint"], {
            **fixtures["checkpoint"],
            "bytes": binding["checkpoint"]["bytes"],
            "sha256": binding["checkpoint"]["sha256"],
        })
        self.assertEqual(
            fixtures["official_tied_embedding"],
            {
                **fixtures["official_tied_embedding"],
                **binding["official_tied_embedding"],
            },
        )
        self.assertEqual(
            fixtures["fresh_token_inputs"],
            binding["fresh_token_inputs"],
        )

        traversal = evidence["current_continuation_attempt"]
        self.assertEqual(traversal["layer_order"], list(range(24)))
        self.assertEqual(traversal["natural_terminal_layers"], 24)
        self.assertEqual(len(traversal["layers"]), 24)
        ordered_terminals = []
        previous_outputs = None
        for layer_index, layer in enumerate(traversal["layers"]):
            self.assertEqual(layer["layer_index"], layer_index)
            positions = (
                layer["fresh_prefix"]["position0"],
                layer["fresh_prefix"]["position1"],
                layer,
            )
            for position, record in enumerate(positions):
                self.assertEqual(record["position"], position)
                self.assertEqual(record["raw"]["done_count"], 1)
                self.assertEqual(record["raw"]["final_count"], 896)
                terminal_record = record["raw"]["terminal"]
                path = Path(terminal_record["path"])
                self.assertEqual(path.stat().st_size, terminal_record["bytes"])
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    terminal_record["sha256"],
                )
                ordered_terminals.append((layer_index, position, path))

            prefix0, prefix1, position2 = positions
            parentage = layer["fp16_kv_parentage"]
            self.assertEqual(parentage["source_positions"], [0, 1])
            self.assertEqual(parentage["target_position"], 2)
            self.assertTrue(parentage["exact_integer_oracle_match"])
            self.assertEqual(
                parentage["position0_trace"],
                prefix0["raw"]["trace"],
            )
            self.assertEqual(
                parentage["position1_trace"],
                prefix1["raw"]["trace"],
            )
            self.assertEqual(
                parentage["position2_state"],
                prefix1["output_state"],
            )
            self.assertEqual(layer["position01_state"], prefix1["output_state"])
            self.assertTrue(
                layer["independent_oracle_comparison"][
                    "rtl_matches_exact_integer_oracle"
                ]
            )
            if previous_outputs is not None:
                self.assertEqual(
                    [
                        prefix0["input"]["sha256"],
                        prefix1["input"]["sha256"],
                        position2["input"]["sha256"],
                    ],
                    previous_outputs,
                )
            previous_outputs = [
                prefix0["output"]["semantic_sha256"],
                prefix1["output"]["semantic_sha256"],
                position2["output"]["semantic_sha256"],
            ]

        self.assertEqual(len(ordered_terminals), 72)
        self.assertEqual(
            [(layer, position) for layer, position, _ in ordered_terminals],
            [
                (layer, position)
                for layer in range(24)
                for position in range(3)
            ],
        )
        self.assertEqual(
            traversal["post_layer23"]["hidden_sha256"],
            attempt["accepted_traversal"]["post_layer23_hidden_sha256"],
        )
        self.assertTrue(traversal["post_layer23"]["natural_terminal"])
        self.assertTrue(
            traversal["post_layer23"][
                "independent_oracle_within_tolerance"
            ]
        )
        self.assertEqual(
            attempt["v9_boundary"]["canonical_validator_invocations"],
            0,
        )
        self.assertFalse(attempt["v9_boundary"]["canonical_runtime_pass"])

    def test_seals_reviewed_v8_authority_at_zero_execution(self) -> None:
        successor = self.status["fresh_successor_runtime_authority"]
        self.assertEqual(
            successor["status"],
            "L2_PASS_UNCONSUMED_READY_FOR_SEPARATE_RUNTIME_PASS_TASK",
        )
        self.assertEqual(
            successor["accepted_package"]["package_id"],
            "ace3-position2-fresh-v7-20260831t084959z",
        )
        self.assertEqual(
            successor["runtime_identity"],
            "ace3-position2-fresh-v8-20260831t090000z",
        )
        self.assertEqual(
            successor["execution_authority"]["status"],
            "AUTHORIZED_ONCE_UNCONSUMED",
        )
        self.assertEqual(
            successor["authority_review"]["verdict"],
            "PASS",
        )
        self.assertEqual(
            successor["rejected_predecessor_authority"]["classification"],
            "REJECTED_PRE_EXECUTION_PREFLIGHT_NON_CONSUMED",
        )
        self.assertEqual(
            successor["lifecycle_counts_at_authority_review_seal"],
            {
                "authority_consumptions": 0,
                "durable_runner_submissions": 0,
                "payload_invocations": 0,
                "runtime_invocations": 0,
                "validator_invocations": 0,
                "terminal_manifests": 0,
            },
        )
        self.assertFalse(
            successor["claim_boundary"]["position2_runtime_pass_claimed"]
        )
        for key in (
            "source_review",
            "focused_regression",
            "package_validation",
            "source_archive",
            "package_review",
        ):
            self.assertIsNotNone(
                SHA256.fullmatch(
                    successor["accepted_package"][key]["sha256"]
                )
            )
        for key in ("execution_authority", "authority_review", "runtime_driver"):
            self.assertIsNotNone(
                SHA256.fullmatch(successor[key]["sha256"])
            )

    def test_records_v8_pre_consumption_terminal_without_runtime_claim(self) -> None:
        attempt = self.status["v8_consumption_attempt"]
        self.assertEqual(
            attempt["status"],
            "FAIL_NON_RETRIABLE_AWAITING_INDEPENDENT_REVIEW",
        )
        self.assertEqual(
            attempt["runtime_identity"],
            "ace3-position2-fresh-v8-20260831t090000z",
        )
        self.assertEqual(attempt["durable_runner"]["state"], "error")
        self.assertEqual(attempt["durable_runner"]["exit_code"], 2)
        self.assertEqual(attempt["durable_runner"]["submissions"], 1)
        self.assertEqual(
            attempt["lifecycle_counts"],
            {
                "authority_consumptions": 0,
                "durable_runner_submissions": 1,
                "driver_invocations": 1,
                "payload_invocations": 0,
                "generate_invocations": 0,
                "validate_invocations": 0,
                "terminal_manifests": 1,
                "terminal_review_verdicts": 0,
                "retries": 0,
                "replays": 0,
                "resumes": 0,
                "relaunches": 0,
            },
        )
        self.assertEqual(
            attempt["failure_boundary"]["classification"],
            "POSITION2_V8_LAUNCH_TIME_RECEIPT_FRESHNESS_FAIL",
        )
        self.assertEqual(
            attempt["failure_boundary"]["first_observed_error"],
            "durable runner receipt already exists",
        )
        self.assertEqual(
            attempt["runtime_payload_binding"],
            {
                "authority_to_driver": "AUTHENTICATED",
                "payload_invocation": "ABSENT",
                "runtime_output": "ABSENT",
                "vector_binding": "NOT_REACHED",
            },
        )
        self.assertFalse(
            attempt["validator_verdict_consistency"]["runtime_pass_claimed"]
        )
        self.assertEqual(
            attempt["validator_verdict_consistency"][
                "independent_terminal_review"
            ],
            "PENDING",
        )
        for record in attempt["authenticated_evidence"].values():
            self.assertGreaterEqual(record["bytes"], 0)
            self.assertIsNotNone(SHA256.fullmatch(record["sha256"]))
        self.assertFalse(
            attempt["upstream_preservation"]["source_snapshot_regenerated"]
        )
        self.assertFalse(
            attempt["upstream_preservation"]["package_manifest_regenerated"]
        )

    def test_seals_v9_rehearsal_only_scope_mismatch_without_relaunch(self) -> None:
        attempt = self.status["v9_consumption_attempt"]
        self.assertEqual(
            attempt["status"],
            (
                "SEALED_NON_RETRIABLE_AUTHORITY_DRIVER_SCOPE_MISMATCH_"
                "AWAITING_INDEPENDENT_REVIEW"
            ),
        )
        self.assertEqual(
            attempt["runtime_identity"],
            "ace3-position2-fresh-v9-20260831t100202z",
        )
        self.assertEqual(attempt["durable_runner"]["state"], "done")
        self.assertEqual(attempt["durable_runner"]["exit_code"], 0)
        self.assertEqual(attempt["durable_runner"]["submissions"], 1)
        self.assertEqual(
            attempt["lifecycle_counts"],
            {
                "authority_consumptions": 1,
                "durable_runner_submissions": 1,
                "driver_invocations": 1,
                "payload_invocations": 1,
                "source_archive_extractions": 1,
                "verilator_transactions": 1,
                "validate_invocations": 0,
                "terminal_manifests": 1,
                "terminal_review_verdicts": 0,
                "retries": 0,
                "replays": 0,
                "resumes": 0,
                "relaunches": 0,
            },
        )
        self.assertEqual(
            attempt["runtime_terminal"]["kind"],
            "ace3_position2_disposable_rehearsal_terminal",
        )
        self.assertEqual(
            attempt["runtime_terminal"]["status"],
            "REHEARSAL_PASS_AWAITING_INDEPENDENT_L2",
        )
        self.assertTrue(attempt["runtime_terminal"]["consumed"])
        self.assertFalse(attempt["runtime_terminal"]["retryable"])
        self.assertEqual(
            attempt["failure_boundary"]["classification"],
            "POSITION2_V9_FORMAL_AUTHORITY_BOUND_TO_REHEARSAL_ONLY_DRIVER",
        )
        self.assertTrue(attempt["failure_boundary"]["fresh_verilator_build"])
        self.assertEqual(
            attempt["failure_boundary"]["accepted_rtl_transactions"], 1
        )
        self.assertEqual(
            attempt["failure_boundary"]["integer_oracle_comparisons"], 0
        )
        self.assertFalse(attempt["failure_boundary"]["validate_reached"])
        self.assertFalse(
            attempt["failure_boundary"]["canonical_pass_evidence_produced"]
        )
        self.assertEqual(
            attempt["runtime_payload_binding"][
                "canonical_position2_validator"
            ],
            "ABSENT_NOT_IMPLEMENTED_BY_BOUND_DRIVER",
        )
        self.assertEqual(
            attempt["validator_verdict_consistency"][
                "independent_terminal_review"
            ],
            "PENDING",
        )
        self.assertFalse(
            attempt["validator_verdict_consistency"]["runtime_pass_claimed"]
        )
        for record in attempt["authenticated_evidence"].values():
            self.assertGreaterEqual(record["bytes"], 0)
            self.assertIsNotNone(SHA256.fullmatch(record["sha256"]))
            path = ROOT / record["path"]
            self.assertTrue(path.is_file())
            self.assertFalse(path.is_symlink())
            self.assertEqual(path.stat().st_size, record["bytes"])
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                record["sha256"],
            )
        for path in attempt["absent_artifacts"].values():
            self.assertFalse((ROOT / path).exists())
        self.assertTrue(attempt["no_relaunch_boundary"]["authority_consumed"])
        self.assertFalse(attempt["no_relaunch_boundary"]["retryable"])
        self.assertEqual(
            set(attempt["no_relaunch_boundary"]["prohibited"]),
            {
                "retry",
                "resume",
                "replay",
                "relaunch",
                "reuse_consumed_identity",
                "retroactive_validation",
            },
        )


if __name__ == "__main__":
    unittest.main()
