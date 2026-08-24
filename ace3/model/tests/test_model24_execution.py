#!/usr/bin/env python3
"""Exhaustive reduced-geometry and mutation tests for model24 execution."""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

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
    PER_LAYER_OPERATIONS,
    argmax_first,
    build_vector_artifacts,
    expected_schedule,
    grouped_lm_head_interface,
    kv_address,
    kv_owner,
    load_json_bytes,
    residual_handoffs,
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

    def test_vectors_are_byte_reproducible_and_validate(self) -> None:
        first = build_vector_artifacts(
            sha256_bytes(self.contract_payload),
            sha256_bytes(self.bindings_payload),
        )
        second = build_vector_artifacts(
            sha256_bytes(self.contract_payload),
            sha256_bytes(self.bindings_payload),
        )
        self.assertEqual(first, second)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "vectors"
            generated = generate(REPOSITORY_ROOT, output_dir)
            self.assertEqual(generated, first)
            summary = validate_vector_directory(REPOSITORY_ROOT, output_dir)
            self.assertEqual(summary["official_events"], 483)

    def test_vector_mutations_missing_extra_and_duplicate_keys_are_rejected(self) -> None:
        artifacts = build_vector_artifacts(
            sha256_bytes(self.contract_payload),
            sha256_bytes(self.bindings_payload),
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

    def test_argmax_rejects_empty_and_non_integer_controls(self) -> None:
        with self.assertRaises(ContractError):
            argmax_first([])
        with self.assertRaises(ContractError):
            argmax_first([1, True])


if __name__ == "__main__":
    unittest.main()
