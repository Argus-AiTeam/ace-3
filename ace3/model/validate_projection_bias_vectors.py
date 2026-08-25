#!/usr/bin/env python3
"""Authenticate and independently recompute biased AWQ projection vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from awq_bit_oracle import q47_48_to_f16
from fp16_adaptation_oracle import decode_f16_q24
from projection_oracle import complete_projection_output

FILES = {"cases.hex", "meta.hex", "pairs.hex", "expected.hex", "manifest.json"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-tensor-dir", required=True, type=Path)
    parser.add_argument("--generated-dir", required=True, type=Path)
    parser.add_argument("--bindings", required=True, type=Path)
    args = parser.parse_args()
    root = args.generated_dir.resolve(strict=True)
    official = args.official_tensor_dir.resolve(strict=True)
    bindings = json.loads(args.bindings.read_text())
    require(bindings.get("kind") == "ace3_projection_bias_vector_bindings", "binding kind")
    require({p.name for p in root.iterdir() if p.is_file()} == FILES, "artifact inventory")
    artifacts = {x["file"]: x for x in bindings["serialized_artifacts"]}
    require(set(artifacts) == FILES, "binding inventory")
    for name, item in artifacts.items():
        payload = (root / name).read_bytes()
        require(hashlib.sha256(payload).hexdigest() == item["sha256"], f"{name} hash")
        require(len(payload) == item["byte_count"], f"{name} size")
    manifest = json.loads((root / "manifest.json").read_text())
    require(manifest["source_sha256"] == bindings["source_sha256"], "source hashes")
    for name, digest in bindings["source_sha256"].items():
        require(hashlib.sha256((official / name).read_bytes()).hexdigest() == digest,
                f"source artifact hash: {name}")
    headers = (root / "cases.hex").read_text().splitlines()
    meta = (root / "meta.hex").read_text().splitlines()
    pairs = (root / "pairs.hex").read_text().splitlines()
    expected = (root / "expected.hex").read_text().splitlines()
    require(len(headers) == len(meta) == len(expected) == 5 and len(pairs) == 640,
            "fixed record counts")
    for number, summary in enumerate(manifest["cases"]):
        header = int(headers[number], 16)
        lane, bias = (header >> 16) & 0xFF, header & 0xFFFF
        require(header >> 24 == number and lane == summary["lane"] and bias == summary["bias"],
                f"header {number}")
        word = int(meta[number], 16)
        qzero, scale = word & 0xFFFFFFFF, word >> 32
        segment = [int(x, 16) for x in pairs[number * 128:(number + 1) * 128]]
        activations = [x >> 32 for x in segment]
        qweights = [x & 0xFFFFFFFF for x in segment]
        accumulator, result, invalid, saturation, groups = complete_projection_output(
            activations, qweights, [qzero], [scale], lane, bias)
        packed = number | (accumulator & ((1 << 102) - 1)) << 8
        packed |= result << 110 | int(invalid) << 126 | int(saturation) << 127
        require(int(expected[number], 16) == packed, f"oracle stream {number}")
        require((summary["accumulator"], summary["group_accumulator"], summary["result"],
                 summary["invalid"], summary["saturation"]) ==
                (accumulator, groups[0], result, invalid, saturation), f"manifest {number}")
    directed = manifest["cases"][0]
    require(directed["result"] == 1 and directed["accumulator"] == 1 << 23,
            "post-round distinction")
    pre_round, _ = q47_48_to_f16(
        directed["accumulator"] + (decode_f16_q24(directed["bias"])[0] << 24))
    require(pre_round == 2, "directed case fails to distinguish pre-accumulator bias")
    require(manifest["cases"][2]["invalid"] and manifest["cases"][3]["saturation"],
            "nonfinite/saturation coverage")
    require(manifest["cases"][4]["bias"] == 0x8000 and manifest["cases"][4]["result"] == 0,
            "signed-zero coverage")
    print("PROJECTION_BIAS_VALIDATION_PASS artifacts=5 cases=5 pairs=640 "
          "sha256=pass source=pass post_round=pass oracle=pass")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"PROJECTION_BIAS_VALIDATION_FAIL {error}", file=sys.stderr)
        raise SystemExit(1) from error
