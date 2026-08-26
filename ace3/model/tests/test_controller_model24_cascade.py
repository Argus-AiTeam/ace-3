#!/usr/bin/env python3
"""Focused fail-closed tests for the controller-driven Model24 cascade."""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MODEL_DIR.parents[1]
sys.path.insert(0, str(MODEL_DIR))

from controller_model24_cascade import (  # noqa: E402
    ControllerCascadeError,
    _binding_layers,
    _hidden_comparison,
    _load_json,
    _run_accepted_decoder_layer,
    _tensor_map,
    build_binding_document,
    parse_controller_events,
    parse_simulation_terminal,
    validate_binding_document,
)


class ControllerModel24CascadeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tensor_payload = (
            REPOSITORY_ROOT / "ace3" / "contracts" / "model24_tensor_map.json"
        ).read_bytes()
        cls.bindings = build_binding_document(
            REPOSITORY_ROOT,
            cls.tensor_payload,
        )

    def test_all_layer_bindings_are_distinct_and_complete(self) -> None:
        layers = _binding_layers(_tensor_map(self.tensor_payload))
        self.assertEqual([layer["layer_id"] for layer in layers], list(range(24)))
        self.assertTrue(all(layer["tensor_count"] == 26 for layer in layers))
        names = {
            record["name"]
            for layer in layers
            for record in layer["tensors"]
        }
        self.assertEqual(len(names), 24 * 26)

    def test_tampered_layer_binding_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.bindings)
        tampered["layers"][7]["descriptor_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            ControllerCascadeError,
            "layer binding manifest mismatch",
        ):
            validate_binding_document(
                tampered,
                REPOSITORY_ROOT,
                self.tensor_payload,
            )

    def test_duplicate_binding_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(ControllerCascadeError, "duplicate key"):
            _load_json(
                b'{"schema_version":1,"schema_version":1}\n',
                "layer bindings",
            )

    def test_controller_trace_requires_checkpoint_gating(self) -> None:
        words = []
        for layer_id in range(24):
            words.extend(
                (
                    0x10000000 | layer_id,
                    0x20000000
                    | ((1 if layer_id == 23 else 0) << 10)
                    | ((layer_id + 1) << 5)
                    | layer_id,
                )
            )
        words.append(0x30000000 | 23)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "controller_events.hex"
            path.write_text(
                "".join(f"{word:08x}\n" for word in words),
                encoding="ascii",
            )
            self.assertEqual(parse_controller_events(path), list(range(24)))
            words[15], words[16] = words[16], words[15]
            path.write_text(
                "".join(f"{word:08x}\n" for word in words),
                encoding="ascii",
            )
            with self.assertRaisesRegex(
                ControllerCascadeError,
                "checkpoint-gated",
            ):
                parse_controller_events(path)

    def test_terminal_gate_rejects_failure_and_duplicate_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "terminal.txt"
            path.write_text(
                "schema=ace3_model24_controller_raw_v1 natural_terminal=0 "
                "exit_code=2 launches=1 checkpoints=0 done=0 "
                "terminal_layer=none\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(ControllerCascadeError, "natural"):
                parse_simulation_terminal(path)
            path.write_text(
                "schema=ace3_model24_controller_raw_v1 natural_terminal=1 "
                "natural_terminal=1 exit_code=0 launches=24 checkpoints=24 "
                "done=1 terminal_layer=23\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(
                ControllerCascadeError,
                "malformed or ambiguous",
            ):
                parse_simulation_terminal(path)

    def test_layer3_switches_from_reviewed_rational_to_accurate_silu(self) -> None:
        hidden = np.zeros((2, 896), dtype="<u2")
        observed: list[bool] = []

        def run_token(
            _values: object,
            activation: list[int],
            _token: int,
            _cache_k: object,
            _cache_v: object,
            *,
            accurate_silu: bool,
        ) -> tuple[list[int], list[object]]:
            observed.append(accurate_silu)
            return activation, [object()] * 23338

        binding = {"namespace": "model.layers.0.", "tensors": []}
        with mock.patch(
            "controller_model24_cascade.run_decoder_layer_token",
            side_effect=run_token,
        ):
            _run_accepted_decoder_layer(2, binding, {}, hidden)
            _run_accepted_decoder_layer(3, binding, {}, hidden)
        self.assertEqual(observed, [False, False, True, True])

    def test_two_token_comparison_rejects_token0_only_tolerance_miss(self) -> None:
        produced = np.zeros((2, 896), dtype="<u2")
        reference = np.zeros((2, 896), dtype=np.float64)
        reference[0, 62] = 0.126
        reference[1, 17] = 0.01
        comparison = _hidden_comparison(produced, reference)
        self.assertFalse(comparison["within_tolerance"])
        self.assertFalse(comparison["tokens"][0]["within_tolerance"])
        self.assertTrue(comparison["tokens"][1]["within_tolerance"])
        self.assertEqual(comparison["tokens"][0]["max_abs_error"], 0.126)


if __name__ == "__main__":
    unittest.main()
