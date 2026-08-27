from __future__ import annotations

import gzip
import sys
import tempfile
import unittest
from pathlib import Path

MODEL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL))

from model24_oracle import CHECKPOINT_SHA256, MODEL_REPOSITORY, MODEL_REVISION
from model24_execution_oracle import TENSOR_MAP_SHA256
from position1_model24_causal_traversal import (
    LAYER_COUNT,
    PARENT_SCHEMA,
    TraversalError,
    _bound_file_matches,
    _load_parent_kv,
    canonical_json,
    sha256,
    validate_parent_document,
)


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
