#!/usr/bin/env python3
"""Focused tests for authenticated indexed-layer package construction."""

from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

MODEL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_DIR))

import model24_execution_oracle as oracle  # noqa: E402


class Model24IndexedPackageConstructionTests(unittest.TestCase):
    LAYER2_CAPTURE_CPP = """
if(layer_index != 2)
    throw std::runtime_error("this sealed invocation requires layer index 2");
"schema=ace3-layer2-simulator-terminal-v1\\nlayer_index=2\\nnatural_terminal=1\\n"
std::cout << "ACE3_LAYER2_CAPTURE_NATURAL_TERMINAL";
"schema=ace3-layer2-simulator-terminal-v1\\nlayer_index=2\\nnatural_terminal=0\\n"
std::cerr << "ACE3_LAYER2_CAPTURE_EXCEPTION";
tensor(dir, "layer2_input_layernorm_weight.fp16le.bin");
tensor(dir, "layer2_post_attention_layernorm_weight.fp16le.bin");
projection(dir, "layer2_self_attn_q_proj", true);
projection(dir, "layer2_self_attn_k_proj", true);
projection(dir, "layer2_self_attn_v_proj", true);
projection(dir, "layer2_self_attn_o_proj", false);
projection(dir, "layer2_mlp_gate_proj", false);
projection(dir, "layer2_mlp_up_proj", false);
projection(dir, "layer2_mlp_down_proj", false);
"""
    LAYER2_CAPTURE_HEADER = (
        'counts_.append("schema=ace3-layer2-raw-counts-v1\\nlayer_index=2\\n");'
    )

    def test_stale_layer2_capture_is_rejected_for_layer3(self) -> None:
        with self.assertRaisesRegex(
            oracle.ContractError,
            "capture source layer-index binding mismatch",
        ):
            oracle.validate_indexed_capture_sources(
                self.LAYER2_CAPTURE_CPP,
                self.LAYER2_CAPTURE_HEADER,
                3,
            )

    def test_capture_retarget_propagates_layer3_to_every_evidence_sink(self) -> None:
        cpp_source, raw_evidence_header = oracle.retarget_indexed_capture_sources(
            self.LAYER2_CAPTURE_CPP,
            self.LAYER2_CAPTURE_HEADER,
            source_layer_index=2,
            target_layer_index=3,
        )

        oracle.validate_indexed_capture_sources(cpp_source, raw_evidence_header, 3)
        self.assertNotIn("ace3-layer2", cpp_source)
        self.assertNotIn("ACE3_LAYER2_CAPTURE", cpp_source)
        self.assertNotIn("layer_index=2", cpp_source)
        self.assertNotIn("layer2_", cpp_source)
        self.assertNotIn("ace3-layer2", raw_evidence_header)
        self.assertNotIn("layer_index=2", raw_evidence_header)

    def test_layer3_capture_requests_only_existing_layer3_tensor_files(self) -> None:
        cpp_source, _ = oracle.retarget_indexed_capture_sources(
            self.LAYER2_CAPTURE_CPP,
            self.LAYER2_CAPTURE_HEADER,
            source_layer_index=2,
            target_layer_index=3,
        )
        filenames = oracle.indexed_capture_tensor_filenames(cpp_source, 3)
        self.assertEqual(len(filenames), 26)
        self.assertTrue(all(name.startswith("layer3_") for name in filenames))

        with tempfile.TemporaryDirectory() as temporary:
            vector_dir = Path(temporary)
            tensor_dir = vector_dir / "tensors"
            tensor_dir.mkdir()
            for name in filenames:
                (tensor_dir / name).write_bytes(b"authenticated tensor fixture")

            with self.assertRaisesRegex(
                oracle.ContractError,
                "capture tensor layer-index binding mismatch",
            ):
                oracle.validate_indexed_capture_tensor_files(
                    self.LAYER2_CAPTURE_CPP,
                    vector_dir,
                    3,
                )
            self.assertEqual(
                oracle.validate_indexed_capture_tensor_files(
                    cpp_source,
                    vector_dir,
                    3,
                ),
                filenames,
            )
            (tensor_dir / filenames[-1]).unlink()
            with self.assertRaisesRegex(
                oracle.ContractError,
                "capture tensor files missing",
            ):
                oracle.validate_indexed_capture_tensor_files(
                    cpp_source,
                    vector_dir,
                    3,
                )

    def test_layer2_manifest_binds_authenticated_layer1_handoff(self) -> None:
        payload = (
            b"0000003c00\n"
            b"0000014000\n"
            b"0100004200\n"
            b"0100014400\n"
        )
        digest = hashlib.sha256(payload).hexdigest()
        layer_binding = {
            "layer_id": 2,
            "namespace": "model.layers.2.",
            "descriptor_sha256": "d" * 64,
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            handoff_path = root / "layer1-final.rows"
            handoff_path.write_bytes(payload)
            output_dir = root / "layer2-vectors"
            with (
                mock.patch.object(
                    oracle,
                    "OFFICIAL_GEOMETRY",
                    SimpleNamespace(hidden_size=2),
                ),
                mock.patch.object(
                    oracle,
                    "_layer_tensor_payloads",
                    return_value=({}, [], layer_binding),
                ),
                mock.patch.object(
                    oracle,
                    "run_decoder_layer_token",
                    side_effect=lambda _values, activation, *_args: (
                        activation,
                        [],
                    ),
                ),
            ):
                manifest = oracle.materialize_indexed_decoder_vectors(
                    root / "checkpoint",
                    root / "tensor-map",
                    handoff_path,
                    output_dir,
                    layer_index=2,
                    expected_handoff_sha256=digest,
                )

            self.assertEqual(manifest["layer_index"], 2)
            self.assertEqual(manifest["layer_binding"], layer_binding)
            self.assertEqual(
                manifest["input_handoff"],
                {
                    "sha256": digest,
                    "rows": 4,
                    "shape": [2, 2],
                    "dtype": "F16",
                    "record_format": "token[7:0] index[15:0] f16[15:0]",
                    "source": "authenticated decoder layer 1 raw final rows",
                    "source_layer_index": 1,
                    "consumer_layer_index": 2,
                    "byte_preserved_as": "inputs.hex",
                },
            )
            self.assertEqual((output_dir / "inputs.hex").read_bytes(), payload)

    def test_layer3_manifest_binds_authenticated_layer2_handoff(self) -> None:
        payload = (
            b"0000003c00\n"
            b"0000014000\n"
            b"0100004200\n"
            b"0100014400\n"
        )
        digest = hashlib.sha256(payload).hexdigest()
        layer_binding = {
            "layer_id": 3,
            "namespace": "model.layers.3.",
            "descriptor_sha256": "e" * 64,
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            handoff_path = root / "layer2-final.rows"
            handoff_path.write_bytes(payload)
            output_dir = root / "layer3-vectors"
            with (
                mock.patch.object(
                    oracle,
                    "OFFICIAL_GEOMETRY",
                    SimpleNamespace(hidden_size=2),
                ),
                mock.patch.object(
                    oracle,
                    "_layer_tensor_payloads",
                    return_value=({}, [], layer_binding),
                ),
                mock.patch.object(
                    oracle,
                    "run_decoder_layer_token",
                    side_effect=lambda _values, activation, *_args: (
                        activation,
                        [],
                    ),
                ),
            ):
                manifest = oracle.materialize_indexed_decoder_vectors(
                    root / "checkpoint",
                    root / "tensor-map",
                    handoff_path,
                    output_dir,
                    layer_index=3,
                    expected_handoff_sha256=digest,
                )

            self.assertEqual(manifest["layer_index"], 3)
            self.assertEqual(manifest["layer_binding"], layer_binding)
            self.assertEqual(
                manifest["input_handoff"],
                {
                    "sha256": digest,
                    "rows": 4,
                    "shape": [2, 2],
                    "dtype": "F16",
                    "record_format": "token[7:0] index[15:0] f16[15:0]",
                    "source": "authenticated decoder layer 2 raw final rows",
                    "source_layer_index": 2,
                    "consumer_layer_index": 3,
                    "byte_preserved_as": "inputs.hex",
                },
            )
            self.assertEqual((output_dir / "inputs.hex").read_bytes(), payload)

    def test_wrong_layer1_handoff_hash_is_rejected_before_output(self) -> None:
        payload = (
            b"0000003c00\n"
            b"0000014000\n"
            b"0100004200\n"
            b"0100014400\n"
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            handoff_path = root / "layer1-final.rows"
            handoff_path.write_bytes(payload)
            output_dir = root / "layer2-vectors"
            with (
                mock.patch.object(
                    oracle,
                    "OFFICIAL_GEOMETRY",
                    SimpleNamespace(hidden_size=2),
                ),
                mock.patch.object(
                    oracle,
                    "_layer_tensor_payloads",
                    return_value=(
                        {},
                        [],
                        {
                            "layer_id": 2,
                            "namespace": "model.layers.2.",
                            "descriptor_sha256": "d" * 64,
                        },
                    ),
                ),
            ):
                with self.assertRaisesRegex(
                    oracle.ContractError,
                    "two-token handoff SHA256 mismatch",
                ):
                    oracle.materialize_indexed_decoder_vectors(
                        root / "checkpoint",
                        root / "tensor-map",
                        handoff_path,
                        output_dir,
                        layer_index=2,
                        expected_handoff_sha256="0" * 64,
                    )

            self.assertFalse(output_dir.exists())

    def test_layer2_requires_explicit_predecessor_handoff_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(
                oracle.ContractError,
                "expected predecessor handoff SHA256 is required",
            ):
                oracle.materialize_indexed_decoder_vectors(
                    root / "checkpoint",
                    root / "tensor-map",
                    root / "layer1-final.rows",
                    root / "layer2-vectors",
                    layer_index=2,
                )


if __name__ == "__main__":
    unittest.main()
