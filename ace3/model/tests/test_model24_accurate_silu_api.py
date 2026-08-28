#!/usr/bin/env python3
"""CPU-only closure tests for the Model24 accurate-SiLU API."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_DIR))

import controller_model24_rtl_cascade as controller  # noqa: E402
import decoder_layer0_oracle as decoder  # noqa: E402
import fp16_adaptation_oracle as adaptation  # noqa: E402
import model24_execution_oracle as oracle  # noqa: E402


HANDOFF_PAYLOAD = (
    b"0000003c00\n"
    b"0000014000\n"
    b"0100004200\n"
    b"0100014400\n"
)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def artifact(payload: bytes) -> dict[str, object]:
    return {"bytes": len(payload), "sha256": sha256(payload)}


def expected_trace_payload() -> bytes:
    rows = ((0, (0x3C00, 0x4000)), (1, (0x4200, 0x4400)))
    return "".join(
        f"{token:02x}{token:04x}10{index:04x}{value:04x}\n"
        for token, values in rows
        for index, value in enumerate(values)
    ).encode("ascii")


def expected_final_payload() -> bytes:
    rows = ((0, (0x3C00, 0x4000)), (1, (0x4200, 0x4400)))
    return "".join(
        f"{token:02x}{index:04x}{value:04x}\n"
        for token, values in rows
        for index, value in enumerate(values)
    ).encode("ascii")


class Model24AccurateSiluApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.subprocess_guard = mock.patch.object(
            subprocess,
            "run",
            side_effect=AssertionError("subprocess execution is forbidden"),
        )
        self.subprocess_guard.start()

    def tearDown(self) -> None:
        self.subprocess_guard.stop()

    def materialize(
        self,
        root: Path,
        layer_index: int,
        flags: list[bool],
        accurate_silu: bool | None | object,
        *,
        through_controller: bool = False,
    ) -> dict[str, object]:
        handoff_path = root / f"layer{layer_index}-input.hex"
        handoff_path.write_bytes(HANDOFF_PAYLOAD)
        output_dir = root / f"layer{layer_index}-vectors"
        binding = oracle.indexed_layer_binding(layer_index)

        def run_token(
            _values: dict[str, list[int]],
            activation: list[int],
            position: int,
            _cache_k: list[list[int]],
            _cache_v: list[list[int]],
            use_accurate_silu: bool = False,
        ) -> tuple[list[int], list[tuple[int, int, int, int]]]:
            flags.append(use_accurate_silu)
            return activation, [
                (16, index, value, position)
                for index, value in enumerate(activation)
            ]

        function = (
            controller.materialize_indexed_decoder_vectors
            if through_controller
            else oracle.materialize_indexed_decoder_vectors
        )
        kwargs: dict[str, object] = {
            "layer_index": layer_index,
            "expected_handoff_sha256": sha256(HANDOFF_PAYLOAD),
        }
        if accurate_silu is not DEFAULT:
            kwargs["accurate_silu"] = accurate_silu
        with (
            mock.patch.object(
                oracle,
                "OFFICIAL_GEOMETRY",
                SimpleNamespace(hidden_size=2),
            ),
            mock.patch.object(
                oracle,
                "_layer_tensor_payloads",
                return_value=({}, [], binding),
            ),
            mock.patch.object(
                oracle,
                "run_decoder_layer_token",
                side_effect=run_token,
            ),
        ):
            return function(
                root / "checkpoint.safetensors",
                root / "tensor-map.json",
                handoff_path,
                output_dir,
                **kwargs,
            )

    def test_controller_layer0_real_materializer_accepts_explicit_true(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            flags: list[bool] = []
            manifest = self.materialize(
                Path(temporary),
                0,
                flags,
                True,
                through_controller=True,
            )

        self.assertEqual(flags, [True, True])
        self.assertEqual(
            manifest["input_handoff"],
            {
                "sha256": sha256(HANDOFF_PAYLOAD),
                "rows": 4,
                "shape": [2, 2],
                "dtype": "F16",
                "record_format": "token[7:0] index[15:0] f16[15:0]",
                "source": "authenticated official token embedding rows",
                "source_layer_index": None,
                "consumer_layer_index": 0,
                "byte_preserved_as": "inputs.hex",
            },
        )
        self.assertEqual(
            manifest["numeric_profile"]["silu"],
            "exp range-reduced degree-7 Q24",
        )

    def test_explicit_false_retains_rational_silu(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            flags: list[bool] = []
            manifest = self.materialize(Path(temporary), 7, flags, False)

        self.assertEqual(flags, [False, False])
        self.assertEqual(
            manifest["numeric_profile"]["silu"],
            "accepted rational Q24",
        )

    def test_default_indexed_layer_silu_profile_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            layer2_flags: list[bool] = []
            layer2 = self.materialize(root, 2, layer2_flags, DEFAULT)
            layer3_flags: list[bool] = []
            layer3 = self.materialize(root, 3, layer3_flags, DEFAULT)

        self.assertEqual(layer2_flags, [False, False])
        self.assertEqual(layer3_flags, [True, True])
        self.assertEqual(
            layer2["numeric_profile"]["silu"],
            "accepted rational Q24",
        )
        self.assertEqual(
            layer3["numeric_profile"]["silu"],
            "exp range-reduced degree-7 Q24",
        )

    def test_decoder_token_oracle_selects_requested_silu(self) -> None:
        selected: list[int] = []

        def module(
            _values: dict[str, list[int]],
            prefix: str,
            activations: list[int],
            out_features: int,
            _bias: list[int] | None = None,
        ) -> list[int]:
            if prefix == "mlp.down_proj":
                selected.extend(activations)
                return activations
            if prefix in ("mlp.gate_proj", "mlp.up_proj"):
                return [0x3C00]
            return [0] * out_features

        values = {
            "model.layers.0.input_layernorm.weight:": [0x3C00],
            "model.layers.0.post_attention_layernorm.weight:": [0x3C00],
            "model.layers.0.self_attn.q_proj.bias:": [0],
            "model.layers.0.self_attn.k_proj.bias:": [],
            "model.layers.0.self_attn.v_proj.bias:": [],
        }
        with (
            mock.patch.multiple(
                decoder,
                HIDDEN=1,
                INTERMEDIATE=1,
                HEADS=0,
                KV_HEADS=0,
                HEAD_DIM=1,
            ),
            mock.patch.object(decoder, "_module", side_effect=module),
            mock.patch.object(
                decoder,
                "silu_gate",
                return_value=(0x3555, False, False),
            ),
            mock.patch.object(
                decoder,
                "silu_gate_exp",
                return_value=(0x3666, False, False),
            ),
        ):
            decoder.run_token(values, [0x3C00], 0, [], [], accurate_silu=False)
            decoder.run_token(values, [0x3C00], 0, [], [], accurate_silu=True)

        self.assertEqual(selected, [0x3555, 0x3666])

    def test_accurate_silu_matches_mathematical_fp16_samples(self) -> None:
        def f16_bits(value: float) -> int:
            return int(np.asarray(value, dtype=np.float16).view(np.uint16))

        rational_differences = 0
        up = 1.5
        for gate in (-4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0):
            expected = f16_bits((gate / (1.0 + math.exp(-gate))) * up)
            accurate = adaptation.silu_gate_exp(
                f16_bits(gate), f16_bits(up)
            )[0]
            rational = adaptation.silu_gate(f16_bits(gate), f16_bits(up))[0]
            self.assertEqual(accurate, expected)
            rational_differences += rational != expected
        self.assertGreater(rational_differences, 0)

    def test_retained_resume_regenerates_with_real_materializer(self) -> None:
        trace = expected_trace_payload()
        final = expected_final_payload()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            layer_id = 7
            layer_dir = root / "layer07"
            raw_dir = layer_dir / "raw"
            vector_dir = layer_dir / "vectors"
            raw_dir.mkdir(parents=True)
            vector_dir.mkdir()
            binding = oracle.indexed_layer_binding(layer_id)
            handoff_binding = {
                "sha256": sha256(HANDOFF_PAYLOAD),
                "rows": 4,
                "shape": [2, 2],
                "dtype": "F16",
                "record_format": "token[7:0] index[15:0] f16[15:0]",
                "source": "authenticated decoder layer 6 raw final rows",
                "source_layer_index": 6,
                "consumer_layer_index": 7,
                "byte_preserved_as": "inputs.hex",
            }
            manifest = {
                "layer_index": layer_id,
                "layer_binding": binding,
                "input_handoff": handoff_binding,
            }
            vector_payloads = {
                "boundary_manifest.json": json.dumps(manifest).encode("ascii"),
                "inputs.hex": HANDOFF_PAYLOAD,
                "trace.hex": trace,
                "final.hex": final,
            }
            comparison = {"within_tolerance": True}
            comparison_payload = (
                json.dumps(comparison, sort_keys=True, separators=(",", ":"))
                .encode("ascii")
                + b"\n"
            )
            raw_payloads = {
                "trace.hex": trace,
                "final.hex": final,
                "terminal.txt": controller.natural_terminal(layer_id).encode("ascii"),
                "comparison.json": comparison_payload,
            }
            for name, payload in vector_payloads.items():
                (vector_dir / name).write_bytes(payload)
            for name, payload in raw_payloads.items():
                (raw_dir / name).write_bytes(payload)
            predecessor_path = root / "predecessor.hex"
            predecessor_path.write_bytes(HANDOFF_PAYLOAD)
            checkpoint_path = root / "checkpoint.safetensors"
            checkpoint_path.write_bytes(b"not opened by the focused test\n")
            tensor_map_path = root / "tensor-map.json"
            tensor_map_path.write_bytes(b"not opened by the focused test\n")
            consumed = [{"name": "tensor", "sha256": "tensor-sha256"}]
            sources = {"controller.py": "source-sha256"}
            record = {
                "layer_index": layer_id,
                "namespace": binding["namespace"],
                "descriptor_sha256": binding["descriptor_sha256"],
                "input_raw_sha256": sha256(HANDOFF_PAYLOAD),
                "output_raw_sha256": sha256(final),
                "consumed_tensors": consumed,
                "comparison": comparison,
                "sources": sources,
                "artifacts": {
                    name: artifact(payload) for name, payload in raw_payloads.items()
                },
                "oracle_artifacts": {
                    name: artifact(payload)
                    for name, payload in vector_payloads.items()
                },
            }
            (layer_dir / "record.json").write_text(
                json.dumps(record), encoding="ascii"
            )
            flags: list[bool] = []

            def run_token(
                _values: dict[str, list[int]],
                activation: list[int],
                position: int,
                _cache_k: list[list[int]],
                _cache_v: list[list[int]],
                use_accurate_silu: bool = False,
            ) -> tuple[list[int], list[tuple[int, int, int, int]]]:
                flags.append(use_accurate_silu)
                return activation, [
                    (16, index, value, position)
                    for index, value in enumerate(activation)
                ]

            with (
                mock.patch.object(
                    oracle,
                    "OFFICIAL_GEOMETRY",
                    SimpleNamespace(hidden_size=2),
                ),
                mock.patch.object(
                    oracle,
                    "_layer_tensor_payloads",
                    return_value=({}, [], binding),
                ),
                mock.patch.object(
                    oracle,
                    "run_decoder_layer_token",
                    side_effect=run_token,
                ),
                mock.patch.object(
                    controller,
                    "run_command",
                    side_effect=AssertionError("RTL execution is forbidden"),
                ),
            ):
                validated = controller.validate_completed_layer(
                    layer_id,
                    layer_dir,
                    sha256(HANDOFF_PAYLOAD),
                    binding,
                    consumed,
                    sources,
                    checkpoint_path=checkpoint_path,
                    tensor_map_path=tensor_map_path,
                    predecessor_path=predecessor_path,
                )

        self.assertEqual(validated, record)
        self.assertEqual(flags, [True, True])


DEFAULT = object()


if __name__ == "__main__":
    unittest.main()
