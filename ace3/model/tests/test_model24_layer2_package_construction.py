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

    @classmethod
    def layer3_capture(cls) -> tuple[str, str]:
        return oracle.retarget_indexed_capture_sources(
            cls.LAYER2_CAPTURE_CPP,
            cls.LAYER2_CAPTURE_HEADER,
            source_layer_index=2,
            target_layer_index=3,
        )

    def test_layer2_capture_binding_regression(self) -> None:
        oracle.validate_indexed_capture_sources(
            self.LAYER2_CAPTURE_CPP,
            self.LAYER2_CAPTURE_HEADER,
            2,
        )
        filenames = oracle.indexed_capture_tensor_filenames(
            self.LAYER2_CAPTURE_CPP,
            2,
        )
        self.assertEqual(len(filenames), 26)
        self.assertTrue(all(name.startswith("layer2_") for name in filenames))
        self.assertEqual(
            self.LAYER2_CAPTURE_CPP.count(
                "schema=ace3-layer2-simulator-terminal-v1"
            ),
            2,
        )
        self.assertEqual(self.LAYER2_CAPTURE_CPP.count("layer_index=2"), 2)

    def test_layer3_capture_binding_regression(self) -> None:
        cpp_source, raw_evidence_header = self.layer3_capture()

        oracle.validate_indexed_capture_sources(cpp_source, raw_evidence_header, 3)
        filenames = oracle.indexed_capture_tensor_filenames(cpp_source, 3)
        self.assertEqual(len(filenames), 26)
        self.assertTrue(all(name.startswith("layer3_") for name in filenames))
        self.assertEqual(
            cpp_source.count("schema=ace3-layer3-simulator-terminal-v1"),
            2,
        )
        self.assertEqual(cpp_source.count("layer_index=3"), 2)
        self.assertNotIn("ace3-layer2", cpp_source)
        self.assertNotIn("layer_index=2", raw_evidence_header)

    def test_wrong_requested_index_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            oracle.ContractError,
            "capture source layer-index binding mismatch",
        ):
            oracle.validate_indexed_capture_sources(
                self.LAYER2_CAPTURE_CPP,
                self.LAYER2_CAPTURE_HEADER,
                3,
            )

    def test_wrong_argv_index_is_rejected(self) -> None:
        wrong_argv_source = self.LAYER2_CAPTURE_CPP.replace(
            "if(layer_index != 2)",
            "if(layer_index != 3)",
            1,
        )
        with self.assertRaisesRegex(
            oracle.ContractError,
            "capture source layer-index binding mismatch",
        ):
            oracle.validate_indexed_capture_sources(
                wrong_argv_source,
                self.LAYER2_CAPTURE_HEADER,
                2,
            )

    def test_wrong_terminal_index_is_rejected(self) -> None:
        cpp_source, raw_evidence_header = self.layer3_capture()
        wrong_terminal_source = cpp_source.replace(
            "layer_index=3\\nnatural_terminal=1",
            "layer_index=2\\nnatural_terminal=1",
            1,
        )
        with self.assertRaisesRegex(
            oracle.ContractError,
            "capture source layer-index binding mismatch",
        ):
            oracle.validate_indexed_capture_sources(
                wrong_terminal_source,
                raw_evidence_header,
                3,
            )

    def test_wrong_tensor_namespace_is_rejected(self) -> None:
        cpp_source, raw_evidence_header = self.layer3_capture()
        wrong_tensor_source = cpp_source.replace(
            "layer3_input_layernorm_weight.fp16le.bin",
            "layer2_input_layernorm_weight.fp16le.bin",
            1,
        )
        with self.assertRaisesRegex(
            oracle.ContractError,
            "capture tensor layer-index binding mismatch",
        ):
            oracle.validate_indexed_capture_sources(
                wrong_tensor_source,
                raw_evidence_header,
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
