#!/usr/bin/env python3
"""Mutation and coverage tests for the ACE-3 model24 control contracts."""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_DIR))

from model24_oracle import (  # noqa: E402
    ContractError,
    OFFICIAL_CONFIG,
    PER_LAYER_OPERATIONS,
    expected_operation_events,
    expected_tensor_records,
    kv_address,
    load_json_document,
    validate_contract_documents,
)


class Model24ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        contracts = MODEL_DIR.parent / "contracts"
        cls.tensor_map = load_json_document(
            contracts / "model24_tensor_map.json"
        )
        cls.control = load_json_document(contracts / "model24_control.json")

    def validate(self, tensor_map: object, control: object) -> dict[str, int]:
        return validate_contract_documents(tensor_map, control)

    def test_complete_inventory_touches_all_627_ranges(self) -> None:
        summary = self.validate(self.tensor_map, self.control)
        self.assertEqual(summary["tensor_count"], 627)
        self.assertEqual(summary["touched_range_count"], 627)
        self.assertEqual(summary["layer_namespace_count"], 24)
        self.assertEqual(summary["tensor_family_count"], 26)
        expected = expected_tensor_records()
        self.assertEqual(
            [record["absolute_file_offsets"] for record in self.tensor_map["tensors"]],
            [record["absolute_file_offsets"] for record in expected],
        )

    def test_sequence_exercises_24_distinct_layer_namespaces(self) -> None:
        events = expected_operation_events()
        self.assertEqual(len(events), 483)
        layer_events = [event for event in events if event["layer_id"] is not None]
        self.assertEqual(len(layer_events), 24 * len(PER_LAYER_OPERATIONS))
        for layer_id in range(24):
            block = [
                event for event in layer_events if event["layer_id"] == layer_id
            ]
            self.assertEqual(
                [event["operation"] for event in block],
                list(PER_LAYER_OPERATIONS),
            )
            self.assertEqual(
                {event["layer_namespace"] for event in block},
                {f"model.layers.{layer_id}."},
            )
        self.assertEqual(
            [event["operation"] for event in events[-2:]],
            ["final_rmsnorm", "lm_head"],
        )
        scheduled_tensors = {
            tensor_name
            for event in events
            for tensor_name in event["tensor_names"]
        }
        self.assertEqual(
            scheduled_tensors,
            {record["name"] for record in self.tensor_map["tensors"]},
        )

    def test_kv_addresses_are_layer_and_bank_disjoint(self) -> None:
        addresses = {
            kv_address(1, layer_id, 37, 1, 63, kind)
            for kind in ("K", "V")
            for layer_id in range(24)
        }
        self.assertEqual(len(addresses), 48)
        for arguments in (
            (0, -1, 0, 0, 0, "K"),
            (0, 24, 0, 0, 0, "K"),
            (0, 0, 128, 0, 0, "K"),
            (0, 0, 0, 2, 0, "K"),
            (0, 0, 0, 0, 64, "K"),
            (0, 0, 0, 0, 0, "X"),
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ContractError):
                    kv_address(*arguments)

    def test_missing_tensor_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.tensor_map)
        del mutated["tensors"][313]
        with self.assertRaises(ContractError):
            self.validate(mutated, self.control)

    def test_repeated_layer_namespace_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.control)
        mutated["layer_namespaces"]["bindings"][23]["namespace"] = (
            "model.layers.22."
        )
        with self.assertRaises(ContractError):
            self.validate(self.tensor_map, mutated)

    def test_malformed_tensor_address_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.tensor_map)
        mutated["tensors"][313]["absolute_file_offsets"][0] += 2
        with self.assertRaises(ContractError):
            self.validate(mutated, self.control)

    def test_broken_weight_tying_is_rejected(self) -> None:
        mutated_map = copy.deepcopy(self.tensor_map)
        mutated_map["tied_output"]["value_sha256"] = "0" * 64
        with self.assertRaises(ContractError):
            self.validate(mutated_map, self.control)
        mutated_control = copy.deepcopy(self.control)
        mutated_control["final_projection"]["lm_head"]["tied_to"] = (
            "model.layers.23.mlp.down_proj.qweight"
        )
        with self.assertRaises(ContractError):
            self.validate(self.tensor_map, mutated_control)

    def test_broken_layer_order_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.control)
        operations = mutated["schedule"]["per_layer_operation_order"]
        operations[1], operations[2] = operations[2], operations[1]
        with self.assertRaises(ContractError):
            self.validate(self.tensor_map, mutated)

    def test_geometry_is_not_vacuously_parameterized(self) -> None:
        mutated_config = copy.deepcopy(OFFICIAL_CONFIG)
        mutated_config["num_hidden_layers"] = 23
        with self.assertRaises(ContractError):
            expected_tensor_records(mutated_config)

    def test_duplicate_json_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            duplicate_contract = (
                Path(temporary_directory) / "duplicate_contract.json"
            )
            duplicate_contract.write_text(
                '{"schema_version": 1, "schema_version": 1}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ContractError,
                "duplicate JSON key: schema_version",
            ):
                load_json_document(duplicate_contract)


if __name__ == "__main__":
    unittest.main()
