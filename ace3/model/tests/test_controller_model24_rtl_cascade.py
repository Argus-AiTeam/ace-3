#!/usr/bin/env python3
"""Focused scale-aware comparison tests for the Model24 RTL cascade."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_DIR))

from controller_model24_rtl_cascade import (  # noqa: E402
    RtlCascadeError,
    layer_comparison_record,
    natural_terminal,
    prepare_fresh_output,
    require_integer_oracle_bit_exact,
    require_layer_comparison,
    same_handoff_reference_layer,
    scale_aware_fp16_hidden_comparison,
    validate_completed_layer,
)


def fp16_bits(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype="<f2").view("<u2")


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def artifact(payload: bytes) -> dict[str, object]:
    return {"bytes": len(payload), "sha256": sha256(payload)}


INPUT_PAYLOAD = b"authenticated predecessor\n"
TRACE_PAYLOAD = b"integer trace\n"
FINAL_PAYLOAD = b"integer final\n"
CHECKPOINT_PAYLOAD = b"authenticated checkpoint fixture\n"
TENSOR_MAP_PAYLOAD = b"authenticated tensor map fixture\n"


def fixture_manifest() -> dict[str, object]:
    return {
        "layer_index": 7,
        "layer_binding": {
            "layer_id": 7,
            "namespace": "model.layers.7.",
            "descriptor_sha256": "descriptor",
        },
        "input_handoff": {
            "sha256": sha256(INPUT_PAYLOAD),
            "source": "authenticated decoder layer 6 raw final rows",
            "source_layer_index": 6,
            "consumer_layer_index": 7,
            "byte_preserved_as": "inputs.hex",
        },
    }


def regenerate_fixture_oracle(
    checkpoint_path: Path,
    tensor_map_path: Path,
    handoff_path: Path,
    output_dir: Path,
    *,
    layer_index: int,
    expected_handoff_sha256: str,
    accurate_silu: bool,
) -> dict[str, object]:
    if checkpoint_path.read_bytes() != CHECKPOINT_PAYLOAD:
        raise ValueError("checkpoint fixture authentication failed")
    if tensor_map_path.read_bytes() != TENSOR_MAP_PAYLOAD:
        raise ValueError("tensor map fixture authentication failed")
    if handoff_path.read_bytes() != INPUT_PAYLOAD:
        raise ValueError("handoff fixture authentication failed")
    if layer_index != 7 or expected_handoff_sha256 != sha256(INPUT_PAYLOAD):
        raise ValueError("regenerated oracle lineage mismatch")
    if accurate_silu is not True:
        raise ValueError("regenerated oracle numeric profile mismatch")
    manifest = fixture_manifest()
    output_dir.mkdir(parents=True)
    (output_dir / "trace.hex").write_bytes(TRACE_PAYLOAD)
    (output_dir / "final.hex").write_bytes(FINAL_PAYLOAD)
    return manifest


def write_completed_layer_fixture(root: Path) -> tuple[dict, dict, list, dict]:
    layer_id = 7
    layer_dir = root / "layer07"
    raw_dir = layer_dir / "raw"
    vector_dir = layer_dir / "vectors"
    raw_dir.mkdir(parents=True)
    vector_dir.mkdir()
    input_payload = INPUT_PAYLOAD
    trace = TRACE_PAYLOAD
    final = FINAL_PAYLOAD
    terminal = natural_terminal(layer_id).encode("ascii")
    comparison = {"within_tolerance": True}
    comparison_payload = json.dumps(
        comparison, sort_keys=True, separators=(",", ":")
    ).encode("ascii") + b"\n"
    binding = {
        "layer_id": layer_id,
        "namespace": "model.layers.7.",
        "descriptor_sha256": "descriptor",
    }
    consumed = [{"name": "tensor", "sha256": "tensor-sha256"}]
    sources = {"controller.py": "source-sha256"}
    input_sha256 = sha256(input_payload)
    manifest = fixture_manifest()
    vector_payloads = {
        "boundary_manifest.json": json.dumps(manifest).encode("ascii"),
        "inputs.hex": input_payload,
        "trace.hex": trace,
        "final.hex": final,
    }
    raw_payloads = {
        "trace.hex": trace,
        "final.hex": final,
        "terminal.txt": terminal,
        "comparison.json": comparison_payload,
    }
    for name, payload in vector_payloads.items():
        (vector_dir / name).write_bytes(payload)
    for name, payload in raw_payloads.items():
        (raw_dir / name).write_bytes(payload)
    (root / "checkpoint.safetensors").write_bytes(CHECKPOINT_PAYLOAD)
    (root / "tensor-map.json").write_bytes(TENSOR_MAP_PAYLOAD)
    (root / "predecessor.hex").write_bytes(input_payload)
    record = {
        "layer_index": layer_id,
        "namespace": binding["namespace"],
        "descriptor_sha256": binding["descriptor_sha256"],
        "input_raw_sha256": input_sha256,
        "output_raw_sha256": sha256(final),
        "consumed_tensors": consumed,
        "comparison": comparison,
        "sources": sources,
        "artifacts": {
            name: artifact(payload) for name, payload in raw_payloads.items()
        },
        "oracle_artifacts": {
            name: artifact(payload) for name, payload in vector_payloads.items()
        },
    }
    (layer_dir / "record.json").write_text(json.dumps(record), encoding="ascii")
    return binding, record, consumed, sources


class ControllerModel24RtlCascadeTests(unittest.TestCase):
    def test_completed_layer_positive_control_authenticates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binding, record, consumed, sources = write_completed_layer_fixture(root)
            with patch(
                "controller_model24_rtl_cascade."
                "materialize_indexed_decoder_vectors",
                side_effect=regenerate_fixture_oracle,
            ) as regenerate:
                validated = validate_completed_layer(
                    7,
                    root / "layer07",
                    record["input_raw_sha256"],
                    binding,
                    consumed,
                    sources,
                    checkpoint_path=root / "checkpoint.safetensors",
                    tensor_map_path=root / "tensor-map.json",
                    predecessor_path=root / "predecessor.hex",
                )
            self.assertEqual(validated, record)
            regenerate.assert_called_once()

    def test_completed_layer_corruption_fails_closed(self) -> None:
        corruptions = {
            "wrong_trace": lambda root, record: (
                (root / "layer07/raw/trace.hex").write_bytes(b"wrong trace\n"),
                record["artifacts"].__setitem__(
                    "trace.hex", artifact(b"wrong trace\n")
                ),
            ),
            "wrong_final": lambda root, record: (
                (root / "layer07/raw/final.hex").write_bytes(b"wrong final\n"),
                record["artifacts"].__setitem__(
                    "final.hex", artifact(b"wrong final\n")
                ),
                record.__setitem__("output_raw_sha256", sha256(b"wrong final\n")),
            ),
            "false_record_hash": lambda _root, record: record["artifacts"].__setitem__(
                "trace.hex", artifact(b"false claimed trace\n")
            ),
            "claimed_output_actual_final_mismatch": lambda _root, record: (
                record.__setitem__("output_raw_sha256", "0" * 64)
            ),
            "wrong_predecessor_digest": lambda _root, record: (
                record.__setitem__("input_raw_sha256", "1" * 64)
            ),
            "wrong_predecessor_layer": lambda root, record: (
                (lambda path, manifest: (
                    manifest["input_handoff"].__setitem__("source_layer_index", 5),
                    path.write_text(json.dumps(manifest), encoding="ascii"),
                    record["oracle_artifacts"].__setitem__(
                        "boundary_manifest.json", artifact(path.read_bytes())
                    ),
                ))(
                    root / "layer07/vectors/boundary_manifest.json",
                    json.loads(
                        (root / "layer07/vectors/boundary_manifest.json").read_text()
                    ),
                )
            ),
            "wrong_source_lineage": lambda _root, record: (
                record.__setitem__("sources", {"controller.py": "wrong"})
            ),
        }
        for label, corrupt in corruptions.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                binding, record, consumed, sources = write_completed_layer_fixture(root)
                corrupt(root, record)
                (root / "layer07/record.json").write_text(
                    json.dumps(record), encoding="ascii"
                )
                with patch(
                    "controller_model24_rtl_cascade."
                    "materialize_indexed_decoder_vectors",
                    side_effect=regenerate_fixture_oracle,
                ), self.assertRaises((RtlCascadeError, KeyError, ValueError)):
                    validate_completed_layer(
                        7,
                        root / "layer07",
                        sha256(INPUT_PAYLOAD),
                        binding,
                        consumed,
                        sources,
                        checkpoint_path=root / "checkpoint.safetensors",
                        tensor_map_path=root / "tensor-map.json",
                        predecessor_path=root / "predecessor.hex",
                    )

    def test_coordinated_retained_corruption_fails_regenerated_oracle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binding, record, consumed, sources = write_completed_layer_fixture(root)
            corrupt_trace = b"coordinated corrupt trace\n"
            corrupt_final = b"coordinated corrupt final\n"
            for relative, payload in (
                ("raw/trace.hex", corrupt_trace),
                ("raw/final.hex", corrupt_final),
                ("vectors/trace.hex", corrupt_trace),
                ("vectors/final.hex", corrupt_final),
            ):
                (root / "layer07" / relative).write_bytes(payload)
            record["artifacts"]["trace.hex"] = artifact(corrupt_trace)
            record["artifacts"]["final.hex"] = artifact(corrupt_final)
            record["oracle_artifacts"]["trace.hex"] = artifact(corrupt_trace)
            record["oracle_artifacts"]["final.hex"] = artifact(corrupt_final)
            record["output_raw_sha256"] = sha256(corrupt_final)
            (root / "layer07/record.json").write_text(
                json.dumps(record), encoding="ascii"
            )

            with patch(
                "controller_model24_rtl_cascade."
                "materialize_indexed_decoder_vectors",
                side_effect=regenerate_fixture_oracle,
            ) as regenerate, self.assertRaisesRegex(
                RtlCascadeError,
                "differs from independent integer oracle",
            ):
                validate_completed_layer(
                    7,
                    root / "layer07",
                    sha256(INPUT_PAYLOAD),
                    binding,
                    consumed,
                    sources,
                    checkpoint_path=root / "checkpoint.safetensors",
                    tensor_map_path=root / "tensor-map.json",
                    predecessor_path=root / "predecessor.hex",
                )
            regenerate.assert_called_once()

    def test_fresh_preflight_excludes_stale_layer8_execution_scratch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            stale_layer = output_dir / "layers" / "layer08"
            stale_binary = output_dir / "compiled" / "layer8"
            checkpoint = output_dir / "checkpoint" / "model.safetensors"
            receipt = output_dir / ".argus_subagents" / "receipt.json"
            for path in (
                stale_layer,
                stale_binary,
                checkpoint.parent,
                receipt.parent,
            ):
                path.mkdir(parents=True, exist_ok=True)
            (stale_layer / "final.hex").write_text(
                "stale\n", encoding="ascii"
            )
            (stale_binary / "sim").write_text("stale\n", encoding="ascii")
            checkpoint.write_text("checkpoint\n", encoding="ascii")
            receipt.write_text("receipt\n", encoding="ascii")

            record = prepare_fresh_output(output_dir)

            self.assertTrue(
                record["layer8_execution_scratch_before"]["layers/layer08"]
            )
            self.assertTrue(
                record["layer8_execution_scratch_before"]["compiled/layer8"]
            )
            self.assertEqual(
                record["layer8_execution_scratch_after"],
                {"layers/layer08": False, "compiled/layer8": False},
            )
            self.assertFalse(record["stale_layer8_reused"])
            self.assertFalse((output_dir / "layers").exists())
            self.assertFalse((output_dir / "compiled").exists())
            self.assertTrue(checkpoint.is_file())
            self.assertTrue(receipt.is_file())
            self.assertTrue((output_dir / "fresh_preflight.json").is_file())

    def test_layer_reference_restarts_from_authenticated_fp16_handoff(self) -> None:
        handoff = np.zeros((2, 896), dtype=np.uint16)
        handoff[0, 62] = fp16_bits(np.asarray(np.float16(1580.0))).item()
        handoff[0, 262] = fp16_bits(np.asarray(np.float16(75.4375))).item()

        with patch(
            "controller_model24_rtl_cascade._reference_layer",
            side_effect=lambda _layer, _tensors, activation: activation,
        ) as reference_layer:
            result = same_handoff_reference_layer(7, {}, handoff)

        reference_layer.assert_called_once()
        self.assertEqual(float(result[0, 62]), 1580.0)
        self.assertEqual(float(result[0, 262]), 75.4375)

    def test_layer7_accumulated_drift_is_reported_but_local_passes(self) -> None:
        produced_values = np.zeros((2, 896), dtype=np.float16)
        produced_values[0, 62] = np.float16(1581.0)
        produced_values[0, 262] = np.float16(75.375)
        produced = fp16_bits(produced_values)
        propagated_reference = produced_values.astype(np.float64)
        propagated_reference[0, 62] = 1578.8711739171279
        propagated_reference[0, 262] = 75.52008442876374
        same_handoff_reference = produced_values.astype(np.float64)
        same_handoff_reference[0, 62] = 1580.3088638650381
        same_handoff_reference[0, 262] = 75.42602408017562

        comparison = layer_comparison_record(
            7,
            produced,
            same_handoff_reference,
            propagated_reference,
        )

        accumulated = comparison["accumulated_end_to_end_drift"]
        self.assertTrue(comparison["within_tolerance"])
        self.assertEqual(comparison["accepted_by_relative_ulp"], 1)
        self.assertFalse(accumulated["within_tolerance"])
        self.assertEqual(accumulated["failure_count"], 2)
        self.assertEqual(
            accumulated["acceptance_role"],
            "reported only; not used by the layer gate",
        )

    def test_one_ulp_sub_point001_relative_boundary_is_accepted(self) -> None:
        reference = np.full((2, 896), 256.0, dtype=np.float64)
        produced = fp16_bits(reference).copy()
        adjacent = np.nextafter(np.float16(256.0), np.float16(np.inf))
        produced[0, 31] = fp16_bits(np.asarray(adjacent)).item()

        comparison = scale_aware_fp16_hidden_comparison(produced, reference)

        self.assertTrue(comparison["within_tolerance"])
        self.assertGreater(comparison["max_abs_error"], 0.125)
        self.assertEqual(comparison["max_ulp_distance"], 1)
        self.assertLess(comparison["max_relative_error"], 0.001)
        self.assertEqual(comparison["accepted_by_relative_ulp"], 1)

    def test_material_rtl_and_operator_defects_are_rejected(self) -> None:
        reference = np.full((2, 896), 16.0, dtype=np.float64)
        for label, replacement in (
            ("rtl_sign_bit", np.float16(-16.0)),
            ("operator_bypass", np.float16(17.0)),
        ):
            with self.subTest(label=label):
                produced = fp16_bits(reference).copy()
                produced[1, 23] = fp16_bits(np.asarray(replacement)).item()
                comparison = scale_aware_fp16_hidden_comparison(
                    produced,
                    reference,
                )
                self.assertFalse(comparison["within_tolerance"])
                self.assertEqual(comparison["failure_count"], 1)
                self.assertEqual(
                    comparison["first_failure"]["hidden_index"],
                    23,
                )

    def test_integer_oracle_bit_exactness_is_mandatory(self) -> None:
        trace = b"trace\n"
        final = b"final\n"
        require_integer_oracle_bit_exact(7, trace, trace, final, final)

        for label, rtl_trace, rtl_final, message in (
            ("trace", b"corrupt\n", final, "RTL trace differs"),
            ("final", trace, b"corrupt\n", "RTL final differs"),
        ):
            with self.subTest(label=label):
                with self.assertRaisesRegex(RtlCascadeError, message):
                    require_integer_oracle_bit_exact(
                        7,
                        rtl_trace,
                        trace,
                        rtl_final,
                        final,
                    )

    def test_layer_gate_fails_before_lineage_can_advance(self) -> None:
        reference = np.ones((2, 896), dtype=np.float64)
        produced = fp16_bits(reference).copy()
        produced[1, 23] = fp16_bits(np.asarray(np.float16(2.0))).item()
        comparison = scale_aware_fp16_hidden_comparison(produced, reference)

        with self.assertRaisesRegex(
            RtlCascadeError,
            r"layer 7 .*token 1 hidden 23.*ulp=",
        ):
            require_layer_comparison(7, comparison)


if __name__ == "__main__":
    unittest.main()
