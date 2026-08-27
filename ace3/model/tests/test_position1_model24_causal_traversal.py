from __future__ import annotations

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

MODEL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL))

from model24_oracle import CHECKPOINT_SHA256, MODEL_REPOSITORY, MODEL_REVISION
from model24_execution_oracle import TENSOR_MAP_SHA256
from position1_model24_causal_traversal import (
    LAYER_COUNT,
    PARENT_SCHEMA,
    POSITION,
    SCHEMA,
    SELECTED_TOKEN,
    TraversalError,
    SEMANTIC_KV_SCHEMA,
    _bound_file_matches,
    _layer_reference_comparisons,
    _load_parent_kv,
    canonical_json,
    hash_file,
    parse_semantic_kv_payload,
    sha256,
    validate_semantic_kv_preload,
    validate_parent_document,
    verify_result,
    write_json,
)


class LayerReferencePrecisionTests(unittest.TestCase):
    def test_hard_gate_uses_embedding_seeded_contract_precision(self):
        state = SimpleNamespace(
            reference_k=torch.empty((0, 2, 64), dtype=torch.float64),
            reference_v=torch.empty((0, 2, 64), dtype=torch.float64),
        )
        parent_k = torch.zeros((1, 2, 64), dtype=torch.float64)
        parent_v = torch.zeros((1, 2, 64), dtype=torch.float64)
        input_bits = np.asarray([0x3C00, 0xC000], dtype="<u2")
        output_bits = input_bits.copy()
        contract_input = torch.tensor([0.5, -0.5], dtype=torch.float64)
        continuous_input = torch.tensor([0.25, -0.25], dtype=torch.float64)
        seen_inputs = []
        seen_cache_lengths = []

        def fake_step(layer_state, hidden, position):
            self.assertEqual(position, 1)
            seen_inputs.append(hidden.clone())
            seen_cache_lengths.append(layer_state.reference_k.shape[0])
            layer_state.reference_k = torch.cat(
                (layer_state.reference_k, parent_k), dim=0
            )
            return hidden

        with mock.patch(
            "position1_model24_causal_traversal._reference_layer_step",
            side_effect=fake_step,
        ):
            contract_output, _, _, contract, _, local = _layer_reference_comparisons(
                state,
                contract_input,
                continuous_input,
                input_bits,
                output_bits,
                parent_k,
                parent_v,
                1,
            )

        self.assertEqual(seen_cache_lengths, [1, 1, 1])
        torch.testing.assert_close(seen_inputs[0][0], contract_input)
        torch.testing.assert_close(
            seen_inputs[2][0],
            torch.tensor([1.0, -2.0], dtype=torch.float64),
        )
        torch.testing.assert_close(contract_output, contract_input)
        self.assertGreater(contract["max_abs_error"], 0.125)
        self.assertLessEqual(local["max_abs_error"], 0.125)

    def test_prior_rtl_input_cannot_change_hard_gate(self):
        state = SimpleNamespace(
            reference_k=torch.empty((0, 2, 64), dtype=torch.float64),
            reference_v=torch.empty((0, 2, 64), dtype=torch.float64),
        )
        parent_k = torch.zeros((1, 2, 64), dtype=torch.float64)
        parent_v = torch.zeros((1, 2, 64), dtype=torch.float64)
        output_bits = np.asarray([0x3C00, 0x4000], dtype="<u2")

        with mock.patch(
            "position1_model24_causal_traversal._reference_layer_step",
            side_effect=lambda state, hidden, position: hidden,
        ):
            hard_results = []
            for local_bits in (
                np.asarray([0x0000, 0x0000], dtype="<u2"),
                np.asarray([0x7BFF, 0xFBFF], dtype="<u2"),
            ):
                _, _, _, hard, _, _ = _layer_reference_comparisons(
                    state,
                    torch.tensor([1.0, 2.0], dtype=torch.float64),
                    torch.tensor([1.0, 2.0], dtype=torch.float64),
                    local_bits,
                    output_bits,
                    parent_k,
                    parent_v,
                    1,
                )
                hard_results.append(hard)
        self.assertEqual(hard_results[0], hard_results[1])

    def test_injected_upstream_drift_fails_even_when_local_matches(self):
        state = SimpleNamespace(
            reference_k=torch.empty((0, 2, 64), dtype=torch.float64),
            reference_v=torch.empty((0, 2, 64), dtype=torch.float64),
        )
        zeros = torch.zeros((1, 2, 64), dtype=torch.float64)
        output_bits = np.asarray([0x3C00, 0x4000], dtype="<u2")
        with mock.patch(
            "position1_model24_causal_traversal._reference_layer_step",
            side_effect=lambda state, hidden, position: hidden,
        ):
            _, _, _, hard, _, local = _layer_reference_comparisons(
                state,
                torch.tensor([1.5, 2.5], dtype=torch.float64),
                torch.tensor([1.0, 2.0], dtype=torch.float64),
                output_bits,
                output_bits,
                zeros,
                zeros,
                1,
            )
        self.assertGreater(hard["max_abs_error"], 0.125)
        self.assertEqual(local["max_abs_error"], 0.0)

    def test_result_rejects_prior_rtl_local_gate_substitution(self):
        contract_path = MODEL.parent / "contracts/position1_model24_causal_traversal.json"
        contract = json.loads(contract_path.read_text(encoding="ascii"))
        artifact = {"bytes": 1, "sha256": "a" * 64}
        layers = []
        for layer_index in range(LAYER_COUNT):
            layers.append({
                "layer_index": layer_index,
                "semantic_kv_payload": artifact,
                "semantic_kv_readback": artifact,
                "transaction": {
                    "natural_terminal": True,
                    "semantic_kv_preload": artifact,
                    "semantic_kv_readback": artifact,
                },
                "independent_reference": {
                    "seed": "selected token embedding; prior RTL hidden is never consumed",
                    "inter_layer_boundary": "independent binary16 round after every reference layer",
                    "absolute_tolerance": 0.125,
                    "max_abs_error": 0.125,
                    "within_tolerance": True,
                    "gating": True,
                },
                "continuous_float64_reference": {"gating": False},
                "local_reference": {"gating": False},
            })
        result = {
            "schema": SCHEMA,
            "selected_token": SELECTED_TOKEN,
            "position": POSITION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "tensor_map_sha256": TENSOR_MAP_SHA256,
            "build_manifest_sha256": contract["execution_build"]["build_manifest_sha256"],
            "parent_build_manifest_sha256": contract["parent_import"]["build_manifest_sha256"],
            "parent_set_sha256": contract["parent_import"]["parent_set_sha256"],
            "layers": layers,
            "natural_terminal_layers": LAYER_COUNT,
            "hard_gate": "embedding-seeded contract-precision cumulative reference only",
        }
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "result.json"
            write_json(result_path, result)
            verify_result(result_path, contract_path)
            layers[-1]["independent_reference"]["gating"] = False
            layers[-1]["local_reference"]["gating"] = True
            write_json(result_path, result)
            with self.assertRaisesRegex(TraversalError, "oracle"):
                verify_result(result_path, contract_path)


def fixture() -> dict:
    layers = [
        {
            "layer_index": index,
            "state": {"bytes": 100 + index, "sha256": f"{index + 1:064x}"},
            "parent_kv": {
                "k_sha256": f"{index + 101:064x}",
                "v_sha256": f"{index + 201:064x}",
                "elements_each": 128,
                "format": "FP16",
            },
        }
        for index in range(LAYER_COUNT)
    ]
    return {
        "schema": PARENT_SCHEMA,
        "model_binding": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "tensor_map_sha256": TENSOR_MAP_SHA256,
        },
        "build_manifest_sha256": "a" * 64,
        "layers": layers,
    }


class ParentImportNegativeTests(unittest.TestCase):
    def test_valid(self):
        value = fixture()
        validate_parent_document(value, sha256(canonical_json(value)))

    def reject(self, mutate, message):
        value = fixture()
        expected = sha256(canonical_json(value))
        mutate(value)
        with self.assertRaisesRegex(TraversalError, message):
            validate_parent_document(value, expected)

    def test_absent(self): self.reject(lambda d: d["layers"].pop(), "count")
    def test_reordered(self): self.reject(lambda d: d["layers"].reverse(), "order")
    def test_duplicated(self): self.reject(lambda d: d["layers"][1].update(state=d["layers"][0]["state"]), "duplicated")
    def test_stale(self): self.reject(lambda d: d["layers"][2]["state"].update(sha256="0" * 64), "stale")
    def test_checkpoint(self): self.reject(lambda d: d["model_binding"].update(checkpoint_sha256="0" * 64), "checkpoint/vector")


def write_parent_trace(path: Path, *, value_offset: int = 0, swap_first: bool = False, omit_last: bool = False) -> None:
    rows = []
    for index in range(128):
        pair = [
            f"00000006{index:04x}{(value_offset + index + 6) & 0xffff:04x}\n",
            f"00000007{index:04x}{(value_offset + index + 7) & 0xffff:04x}\n",
        ]
        if index == 0 and swap_first:
            pair.reverse()
        rows.extend(pair)
    if omit_last:
        rows.pop()
    with gzip.open(path, "wt", encoding="ascii", newline="") as output:
        output.writelines(rows)


class ParentKvImportNegativeTests(unittest.TestCase):
    def test_missing_parent_kv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.hex.gz"
            write_parent_trace(path, omit_last=True)
            with self.assertRaisesRegex(TraversalError, "count"):
                _load_parent_kv(path, layer_index=0)

    def test_reordered_parent_kv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.hex.gz"
            write_parent_trace(path, swap_first=True)
            with self.assertRaisesRegex(TraversalError, "index/order"):
                _load_parent_kv(path, layer_index=0)

    def test_substituted_parent_kv(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.hex.gz"
            second = Path(directory) / "second.hex.gz"
            write_parent_trace(first)
            write_parent_trace(second, value_offset=1)
            expected, _, _ = _load_parent_kv(first, layer_index=0)
            with self.assertRaisesRegex(TraversalError, "substituted"):
                _load_parent_kv(second, expected, layer_index=0)


def write_semantic_payload(path: Path, *, layer: int = 0, position: int = 0) -> None:
    rows = []
    for index in range(128):
        rows.append(f"{layer:02x}00{position:04x}06{index:04x}{index + 6:04x}\n")
        rows.append(f"{layer:02x}00{position:04x}07{index:04x}{index + 7:04x}\n")
    path.write_text("".join(rows), encoding="ascii", newline="")


class SemanticKvPreloadTests(unittest.TestCase):
    def mutate_rejects(self, mutate, message):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preload.hex"
            write_semantic_payload(path)
            rows = path.read_text(encoding="ascii").splitlines(keepends=True)
            mutate(rows)
            path.write_text("".join(rows), encoding="ascii", newline="")
            with self.assertRaisesRegex(TraversalError, message):
                parse_semantic_kv_payload(path, 0)

    def test_positive_exact_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preload.hex"
            write_semantic_payload(path)
            self.assertEqual(parse_semantic_kv_payload(path, 0)["elements_each"], 128)

    def test_wrong_count(self):
        self.mutate_rejects(lambda rows: rows.pop(), "count")

    def test_reordered(self):
        self.mutate_rejects(lambda rows: rows.__setitem__(slice(0, 2), rows[0:2][::-1]), "reordered")

    def test_duplicated(self):
        self.mutate_rejects(lambda rows: rows.__setitem__(2, rows[0]), "duplicated")

    def test_cross_layer(self):
        self.mutate_rejects(lambda rows: rows.__setitem__(0, "01" + rows[0][2:]), "cross-layer")

    def test_wrong_position(self):
        self.mutate_rejects(lambda rows: rows.__setitem__(0, rows[0][:4] + "0001" + rows[0][8:]), "position")

    def test_tampered_payload_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "layer00.hex"
            manifest = root / "layer00.json"
            write_semantic_payload(payload)
            parent_kv = parse_semantic_kv_payload(payload, 0)
            parent = {
                "trace": {"bytes": 1, "sha256": "1" * 64},
                "parent_kv": parent_kv,
                "trusted_tip": {"kind": "test-tip", "layer_index": 0},
            }
            parent_document = {"model_binding": fixture()["model_binding"]}
            document = {
                "schema": SEMANTIC_KV_SCHEMA,
                "model_binding": parent_document["model_binding"],
                "parent_set_sha256": "2" * 64,
                "layer_index": 0,
                "cache_slot": 0,
                "source_position": 0,
                "execution_position": 1,
                "execution_token": 2114,
                "tensor_binding": {
                    "key": "trace-stage-6-rotated-key-fp16",
                    "value": "trace-stage-7-value-fp16",
                    "ordering": "kv-head-major-dimension-minor",
                },
                "source_trace": parent["trace"],
                "parent_kv": parent_kv,
                "trusted_tip": parent["trusted_tip"],
                "payload": {"path": payload.name, **hash_file(payload)},
            }
            write_json(manifest, document)
            rows = payload.read_text(encoding="ascii").splitlines(keepends=True)
            rows[-1] = rows[-1][:-5] + "ffff\n"
            payload.write_text("".join(rows), encoding="ascii", newline="")
            with self.assertRaisesRegex(TraversalError, "tampered"):
                validate_semantic_kv_preload(
                    manifest, payload, layer_index=0, parent=parent,
                    parent_document=parent_document,
                    parent_set_sha256="2" * 64,
                )


class HistoricalBuildFileBindingTests(unittest.TestCase):
    def test_path_is_checked_separately_from_file_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.bin"
            path.write_bytes(b"bound artifact")
            record = {
                "path": "layer0/artifact.bin",
                "bytes": len(b"bound artifact"),
                "sha256": sha256(b"bound artifact"),
            }
            self.assertTrue(
                _bound_file_matches(path, record, "layer0/artifact.bin")
            )

    def test_wrong_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.bin"
            path.write_bytes(b"bound artifact")
            record = {
                "path": "layer1/artifact.bin",
                "bytes": len(b"bound artifact"),
                "sha256": sha256(b"bound artifact"),
            }
            self.assertFalse(
                _bound_file_matches(path, record, "layer0/artifact.bin")
            )

    def test_wrong_file_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.bin"
            path.write_bytes(b"bound artifact")
            record = {
                "path": "layer0/artifact.bin",
                "bytes": len(b"bound artifact") + 1,
                "sha256": sha256(b"bound artifact"),
            }
            self.assertFalse(
                _bound_file_matches(path, record, "layer0/artifact.bin")
            )


if __name__ == "__main__":
    unittest.main()
