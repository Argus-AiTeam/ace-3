#!/usr/bin/env python3
"""Exhaustive reduced-geometry and mutation tests for model24 execution."""

from __future__ import annotations

import copy
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

MODEL_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MODEL_DIR.parents[1]
sys.path.insert(0, str(MODEL_DIR))

from generate_model24_execution_vectors import generate  # noqa: E402
from model24_execution_oracle import (  # noqa: E402
    ContractError,
    ExecutionMachine,
    Geometry,
    LAYER_DESCRIPTOR_SHA256,
    OFFICIAL_GEOMETRY,
    ORACLE_GEOMETRY,
    PER_LAYER_OPERATIONS,
    FP16KVCache,
    SoftwareOracleEngine,
    argmax_first,
    argmax_lowest,
    build_vector_artifacts,
    canonical_json_bytes,
    expected_schedule,
    grouped_lm_head_interface,
    kv_address,
    kv_owner,
    load_json_bytes,
    residual_handoffs,
    require_provenance_commit,
    reduced_execution_document,
    reduced_tensor_inventory,
    sha256_bytes,
    validate_execution_contract,
    validate_trajectory,
    validate_vector_bindings,
)
from validate_model24_execution_vectors import validate_vector_directory  # noqa: E402


class Model24ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        contracts = REPOSITORY_ROOT / "ace3" / "contracts"
        cls.contract_payload = (contracts / "model24_execution.json").read_bytes()
        cls.bindings_payload = (
            contracts / "model24_execution_vector_bindings.json"
        ).read_bytes()
        cls.tensor_payload = (contracts / "model24_tensor_map.json").read_bytes()
        cls.control_payload = (contracts / "model24_control.json").read_bytes()
        cls.oracle_source_payload = (
            MODEL_DIR / "model24_execution_oracle.py"
        ).read_bytes()
        cls.contract = load_json_bytes(cls.contract_payload, "execution contract")
        cls.bindings = load_json_bytes(cls.bindings_payload, "vector bindings")

    def test_official_contract_binds_all_reviewed_surfaces(self) -> None:
        validate_execution_contract(
            self.contract,
            self.tensor_payload,
            self.control_payload,
        )
        validate_vector_bindings(
            self.bindings,
            sha256_bytes(self.contract_payload),
            self.oracle_source_payload,
        )
        self.assertEqual(self.contract["parent_commit"], "3cf65b762d928e02e2b64fbba4389e294e1aa2c5")
        self.assertEqual(len(self.contract["layers"]["bindings"]), 24)
        self.assertEqual(
            [
                binding["namespace"]
                for binding in self.contract["layers"]["bindings"]
            ],
            [f"model.layers.{layer_id}." for layer_id in range(24)],
        )
        self.assertEqual(
            tuple(
                binding["descriptor_sha256"]
                for binding in self.contract["layers"]["bindings"]
            ),
            LAYER_DESCRIPTOR_SHA256,
        )
        snapshot = self.contract["decoder_snapshot_compatibility"]
        self.assertIn("not decoder acceptance", snapshot["status"])

    def test_provenance_accepts_additive_commits_and_fails_closed(
        self,
    ) -> None:
        def git(repository: Path, *arguments: str) -> str:
            result = subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Model24 Test",
                    "-c",
                    "user.email=model24-test@example.invalid",
                    *arguments,
                ],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            repository = root / "published"
            repository.mkdir()
            git(repository, "init", "--quiet")
            (repository / "contract").write_text("pinned\n", encoding="ascii")
            git(repository, "add", "contract")
            git(repository, "commit", "--quiet", "-m", "pinned provenance")
            provenance_commit = git(repository, "rev-parse", "HEAD")

            for commit_number in (1, 2):
                additive_path = repository / f"additive-{commit_number}"
                additive_path.write_text(f"{commit_number}\n", encoding="ascii")
                git(repository, "add", additive_path.name)
                git(
                    repository,
                    "commit",
                    "--quiet",
                    "-m",
                    f"additive commit {commit_number}",
                )
                require_provenance_commit(repository, provenance_commit)
            git(repository, "checkout", "--quiet", "--orphan", "unrelated")
            (repository / "other").write_text("unrelated\n", encoding="ascii")
            git(repository, "add", "other")
            git(repository, "commit", "--quiet", "-m", "unrelated history")
            with self.assertRaisesRegex(
                ContractError,
                "required provenance commit .* is not an ancestor of HEAD",
            ):
                require_provenance_commit(repository, provenance_commit)

            non_repository = root / "not-a-repository"
            non_repository.mkdir()
            with self.assertRaisesRegex(
                ContractError,
                "unable to verify required provenance commit",
            ):
                require_provenance_commit(non_repository, provenance_commit)

            with mock.patch(
                "model24_execution_oracle.subprocess.run",
                side_effect=FileNotFoundError("git"),
            ) as run_git:
                with self.assertRaisesRegex(ContractError, "unable to execute git"):
                    require_provenance_commit(repository, provenance_commit)
                run_git.assert_called_once_with(
                    [
                        "git",
                        "merge-base",
                        "--is-ancestor",
                        provenance_commit,
                        "HEAD",
                    ],
                    cwd=repository,
                    check=False,
                    capture_output=True,
                    text=True,
                    shell=False,
                )

    def test_official_schedule_and_lineage_are_complete(self) -> None:
        events = expected_schedule()
        self.assertEqual(len(events), 483)
        self.assertEqual(events[0]["operation"], "embedding_lookup")
        self.assertEqual(
            [event["operation"] for event in events[-2:]],
            ["final_rmsnorm", "lm_head"],
        )
        for layer_id in range(24):
            block = [
                event
                for event in events
                if event["layer_id"] == layer_id
            ]
            self.assertEqual(
                [event["operation"] for event in block],
                list(PER_LAYER_OPERATIONS),
            )
            self.assertEqual(
                {event["namespace"] for event in block},
                {f"model.layers.{layer_id}."},
            )
        handoffs = residual_handoffs(OFFICIAL_GEOMETRY)
        self.assertEqual(handoffs[0]["input"], "embedding.output")
        for layer_id in range(1, 24):
            self.assertEqual(
                handoffs[layer_id]["input"],
                handoffs[layer_id - 1]["output"],
            )
        validate_trajectory(events)

    def test_official_kv_ownership_and_final_interfaces(self) -> None:
        addresses = {
            kv_address(1, layer_id, 37, 1, 63, kind)
            for layer_id in range(24)
            for kind in ("K", "V")
        }
        self.assertEqual(len(addresses), 48)
        self.assertNotEqual(kv_owner(0, "K"), kv_owner(0, "V"))
        lm_head = grouped_lm_head_interface()
        self.assertEqual(lm_head["groups_per_logit"], 7)
        self.assertEqual(lm_head["output_logits"], 151936)
        self.assertEqual(argmax_first([5, 5, 4]), 0)
        self.assertEqual(argmax_first([-4, 9, 9]), 1)
        self.assertEqual(argmax_lowest([5.0, 5.0, 4.0]), 0)
        with self.assertRaises(ContractError):
            argmax_first([])
        with self.assertRaises(ContractError):
            argmax_first([1, True])

    def test_small_geometries_exhaust_every_legal_address(self) -> None:
        for layers in (1, 2, 3):
            geometry = Geometry(layers, 2, 2, 2, 2, 4, 2, 5)
            events = expected_schedule(geometry)
            validate_trajectory(events, geometry)
            addresses = [
                kv_address(slot, layer, position, head, dimension, kind, geometry)
                for kind in ("K", "V")
                for slot in range(geometry.cache_slots)
                for layer in range(geometry.layers)
                for position in range(geometry.positions)
                for head in range(geometry.kv_heads)
                for dimension in range(geometry.head_dim)
            ]
            self.assertEqual(
                len(addresses),
                2
                * geometry.cache_slots
                * geometry.layers
                * geometry.positions
                * geometry.kv_heads
                * geometry.head_dim,
            )
            self.assertEqual(len(addresses), len(set(addresses)))
            self.assertEqual(sorted(addresses), list(range(len(addresses))))

    def test_every_kv_bound_and_type_fails_closed(self) -> None:
        geometry = Geometry(2, 2, 2, 2, 2, 4, 2, 5)
        illegal = (
            (-1, 0, 0, 0, 0, "K"),
            (2, 0, 0, 0, 0, "K"),
            (0, -1, 0, 0, 0, "K"),
            (0, 2, 0, 0, 0, "K"),
            (0, 0, -1, 0, 0, "K"),
            (0, 0, 2, 0, 0, "K"),
            (0, 0, 0, -1, 0, "K"),
            (0, 0, 0, 2, 0, "K"),
            (0, 0, 0, 0, -1, "K"),
            (0, 0, 0, 0, 2, "K"),
            (0, 0, 0, 0, 0, "X"),
            (False, 0, 0, 0, 0, "K"),
        )
        for arguments in illegal:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ContractError):
                    kv_address(*arguments, geometry=geometry)

    def test_schedule_fault_latches_until_reset(self) -> None:
        geometry = Geometry(1, 1, 1, 1, 1, 2, 1, 2)
        events = expected_schedule(geometry)
        machine = ExecutionMachine(geometry)
        bad = dict(events[0])
        bad["operation"] = "q_proj"
        with self.assertRaises(ContractError):
            machine.accept(bad)
        self.assertTrue(machine.faulted)
        with self.assertRaisesRegex(ContractError, "faulted"):
            machine.accept(events[0])
        machine.reset()
        for event in events:
            machine.accept(event)
        self.assertTrue(machine.done)
        with self.assertRaisesRegex(ContractError, "completed"):
            machine.accept(events[-1])

    def test_every_schedule_mutation_is_rejected(self) -> None:
        geometry = ORACLE_GEOMETRY.schedule_geometry()
        events = expected_schedule(geometry)
        mutations = {
            "missing": events[:10] + events[11:],
            "duplicate": events[:11] + [events[10]] + events[11:],
            "reordered": events[:10] + [events[11], events[10]] + events[12:],
            "extra": events + [events[-1]],
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                with self.assertRaises(ContractError):
                    validate_trajectory(mutation, geometry)

    def test_reduced_oracle_executes_every_layer_tensor_and_transition(self) -> None:
        document = reduced_execution_document()
        inventory = reduced_tensor_inventory()
        self.assertEqual(document["tensor_count"], 627)
        self.assertEqual(len(inventory), 627)
        self.assertEqual(len(document["executions"]), 2)
        self.assertEqual(
            document["consumed_tensor_names"],
            [record["name"] for record in inventory],
        )
        fixture_hashes = {
            record["name"]: record["fixture_descriptor_sha256"]
            for record in inventory
        }
        self.assertEqual(
            fixture_hashes["model.embed_tokens.weight"],
            fixture_hashes["lm_head.weight"],
        )
        for execution_index, execution in enumerate(document["executions"]):
            self.assertEqual(execution["control_event_count"], 483)
            self.assertEqual(len(execution["events"]), 483)
            self.assertEqual(
                {event["layer_id"] for event in execution["events"] if event["layer_id"] is not None},
                set(range(24)),
            )
            self.assertTrue(
                all(event["output"]["dtype"] == "FP16" for event in execution["events"])
            )
            residuals = [
                event["residual_transition"]
                for event in execution["events"]
                if "residual_transition" in event
            ]
            self.assertEqual(len(residuals), 48)
            for layer_id in range(1, 24):
                self.assertEqual(
                    residuals[layer_id * 2]["input"],
                    residuals[(layer_id - 1) * 2 + 1]["output"],
                )
            kv_transitions = [
                event["kv_transition"]
                for event in execution["events"]
                if "kv_transition" in event
            ]
            self.assertEqual(len(kv_transitions), 48)
            self.assertEqual(
                {transition["layer_id"] for transition in kv_transitions},
                set(range(24)),
            )
            self.assertTrue(
                all(transition["format"] == "FP16" for transition in kv_transitions)
            )
            read_positions = [
                transition["positions"]
                for transition in kv_transitions
                if transition["action"] == "read"
            ]
            self.assertEqual(
                read_positions,
                [list(range(execution_index + 1))] * 24,
            )
            self.assertEqual(execution["final_norm"]["dtype"], "FP16")
            self.assertTrue(execution["lm_head"]["tied"])
            self.assertEqual(execution["lm_head"]["group_size"], 128)
            self.assertEqual(execution["lm_head"]["groups_per_logit"], 2)
            self.assertEqual(
                execution["output_token_id"],
                argmax_lowest(
                    [
                        struct.unpack("<e", bits.to_bytes(2, "little"))[0]
                        for bits in execution["lm_head"]["logit_bits"]
                    ]
                ),
            )

    def test_reduced_execution_is_byte_identical_and_tie_breaks_lowest(self) -> None:
        first = canonical_json_bytes(reduced_execution_document())
        second = canonical_json_bytes(reduced_execution_document())
        self.assertEqual(first, second)
        self.assertEqual(argmax_lowest([9.0, 9.0, -1.0]), 0)

    def test_stale_kv_owner_and_untied_head_fail_closed(self) -> None:
        cache = FP16KVCache()
        cache._owners[(0, 0, "K")] = "kv.layer.23.K"
        values = [0.0] * (ORACLE_GEOMETRY.kv_heads * ORACLE_GEOMETRY.head_dim)
        with self.assertRaisesRegex(ContractError, "stale K KV ownership"):
            cache.write(0, 0, 0, values, values)
        with self.assertRaisesRegex(ContractError, "not tied"):
            SoftwareOracleEngine(tied_head=False)

    def test_contract_and_reviewed_map_mutations_are_rejected(self) -> None:
        mutated = copy.deepcopy(self.contract)
        mutated["schedule"]["event_count"] = 482
        with self.assertRaises(ContractError):
            validate_execution_contract(
                mutated,
                self.tensor_payload,
                self.control_payload,
            )
        mutated_tensor = self.tensor_payload.replace(
            b'"layers": 24',
            b'"layers": 23',
            1,
        )
        with self.assertRaisesRegex(ContractError, "tensor map SHA256"):
            validate_execution_contract(
                self.contract,
                mutated_tensor,
                self.control_payload,
            )
        with self.assertRaisesRegex(ContractError, "execution vector bindings"):
            validate_vector_bindings(
                self.bindings,
                "0" * 64,
                self.oracle_source_payload,
            )
        with self.assertRaisesRegex(ContractError, "oracle_source_sha256"):
            validate_vector_bindings(
                self.bindings,
                sha256_bytes(self.contract_payload),
                self.oracle_source_payload + b"\n# tampered\n",
            )

    def test_vectors_are_byte_reproducible_and_validate(self) -> None:
        first = build_vector_artifacts(
            sha256_bytes(self.contract_payload),
            sha256_bytes(self.bindings_payload),
            self.oracle_source_payload,
        )
        second = build_vector_artifacts(
            sha256_bytes(self.contract_payload),
            sha256_bytes(self.bindings_payload),
            self.oracle_source_payload,
        )
        self.assertEqual(first, second)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "vectors"
            generated = generate(REPOSITORY_ROOT, output_dir)
            self.assertEqual(generated, first)
            summary = validate_vector_directory(REPOSITORY_ROOT, output_dir)
            self.assertEqual(summary["official_events"], 483)
            self.assertEqual(summary["generated_token_count"], 2)

    def test_vector_mutations_missing_extra_and_duplicate_keys_are_rejected(self) -> None:
        artifacts = build_vector_artifacts(
            sha256_bytes(self.contract_payload),
            sha256_bytes(self.bindings_payload),
            self.oracle_source_payload,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            for name, payload in artifacts.items():
                (output_dir / name).write_bytes(payload)

            official_path = output_dir / "official_schedule.json"
            original = official_path.read_bytes()
            official_path.write_bytes(original.replace(b'"event_count":483', b'"event_count":482', 1))
            with self.assertRaisesRegex(ContractError, "oracle regeneration"):
                validate_vector_directory(REPOSITORY_ROOT, output_dir)
            official_path.write_bytes(original)

            (output_dir / "extra.json").write_text("{}\n", encoding="ascii")
            with self.assertRaisesRegex(ContractError, "artifact set"):
                validate_vector_directory(REPOSITORY_ROOT, output_dir)
            (output_dir / "extra.json").unlink()

            manifest_path = output_dir / "manifest.json"
            original_manifest = manifest_path.read_bytes()
            manifest_path.write_bytes(
                original_manifest.replace(
                    b'{"algorithm":',
                    b'{"schema_version":1,"algorithm":',
                    1,
                )
            )
            with self.assertRaisesRegex(ContractError, "duplicate JSON key"):
                validate_vector_directory(REPOSITORY_ROOT, output_dir)
            manifest_path.write_bytes(original_manifest)

            official_path.unlink()
            with self.assertRaisesRegex(ContractError, "artifact set"):
                validate_vector_directory(REPOSITORY_ROOT, output_dir)

if __name__ == "__main__":
    unittest.main()
