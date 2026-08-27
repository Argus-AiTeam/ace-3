#!/usr/bin/env python3
"""Independent exact reference and authenticated vectors for the streaming lm_head."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import struct
from pathlib import Path
from typing import Any, Iterable

import numpy as np

MODEL_REPOSITORY = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
MODEL_REVISION = "db09cd27ead7fee40cdee309693cf83601b9c899"
CHECKPOINT_SHA256 = "c50d807b7bed7ff314308972e0f4bcf4e5a70bc60ad88fc7df53940831ed0c1b"
CHECKPOINT_BYTES = 730_652_248
TIED_WEIGHT_SHA256 = "d74257dc547b48be5ae7b93f1c9af072c0c42dbbb85503078e25c59cd09e68d0"
FINAL_NORM_SHA256 = "1dd25d7720c68bc10838374200238c26626a624119cac0b45bff44bc43c354fe"
HIDDEN_SIZE = 896
VOCAB_SIZE = 151936
TOP_K = 10
EPSILON_Q48 = 281_474_977


class ReferenceError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReferenceError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while payload := stream.read(1024 * 1024):
            digest.update(payload)
    return digest.hexdigest()


def sha256_range(path: Path, offset: int, length: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(offset)
        remaining = length
        while remaining:
            payload = stream.read(min(1024 * 1024, remaining))
            require(bool(payload), "checkpoint tensor payload is truncated")
            digest.update(payload)
            remaining -= len(payload)
    return digest.hexdigest()


def tensor_records(path: Path) -> dict[str, dict[str, Any]]:
    require(path.stat().st_size == CHECKPOINT_BYTES, "checkpoint byte count mismatch")
    require(sha256_file(path) == CHECKPOINT_SHA256, "checkpoint SHA256 mismatch")
    with path.open("rb") as stream:
        header_length_raw = stream.read(8)
        require(len(header_length_raw) == 8, "truncated safetensors length")
        header_length = struct.unpack("<Q", header_length_raw)[0]
        header = json.loads(stream.read(header_length))
    data_base = 8 + header_length
    result: dict[str, dict[str, Any]] = {}
    for name in ("model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"):
        require(name in header, f"missing checkpoint tensor {name}")
        record = header[name]
        begin, end = record["data_offsets"]
        result[name] = {
            "dtype": record["dtype"],
            "shape": record["shape"],
            "offset": data_base + begin,
            "bytes": end - begin,
        }
    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        record = result[name]
        require(record["dtype"] == "F16", f"{name} dtype mismatch")
        require(record["shape"] == [VOCAB_SIZE, HIDDEN_SIZE], f"{name} shape mismatch")
        require(
            sha256_range(path, record["offset"], record["bytes"]) == TIED_WEIGHT_SHA256,
            f"{name} digest mismatch",
        )
    norm = result["model.norm.weight"]
    require(norm["dtype"] == "F16", "model.norm.weight dtype mismatch")
    require(norm["shape"] == [HIDDEN_SIZE], "model.norm.weight shape mismatch")
    require(
        sha256_range(path, norm["offset"], norm["bytes"]) == FINAL_NORM_SHA256,
        "model.norm.weight digest mismatch",
    )
    require(
        result["model.embed_tokens.weight"]["offset"] != result["lm_head.weight"]["offset"],
        "tied tensors unexpectedly share storage",
    )
    return result


def round_div_even(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    doubled = remainder * 2
    return quotient + int(doubled > denominator or (doubled == denominator and quotient & 1))


def decode_f16_q24(bits: int) -> tuple[int, bool, int]:
    sign = (bits >> 15) & 1
    exponent = (bits >> 10) & 0x1F
    fraction = bits & 0x3FF
    if exponent == 0x1F:
        return 0, False, sign
    magnitude = fraction if exponent == 0 else (0x400 | fraction) << (exponent - 1)
    return (-magnitude if sign else magnitude), True, sign


def fixed_to_f16(value: int, fraction_bits: int) -> tuple[int, bool]:
    sign = int(value < 0)
    magnitude = abs(value)
    if magnitude == 0:
        return 0, False
    most = magnitude.bit_length() - 1
    unbiased = most - fraction_bits
    if unbiased > 15:
        return (sign << 15) | 0x7BFF, True
    if unbiased >= -14:
        shift = most - 10
        if shift > 0:
            retained = magnitude >> shift
            discarded = magnitude & ((1 << shift) - 1)
            guard = (discarded >> (shift - 1)) & 1
            sticky = bool(discarded & ((1 << (shift - 1)) - 1)) if shift > 1 else False
            retained += int(guard and (sticky or (retained & 1)))
        else:
            retained = magnitude << (-shift)
        if retained >= 0x800:
            retained >>= 1
            unbiased += 1
        if unbiased > 15:
            return (sign << 15) | 0x7BFF, True
        return (sign << 15) | ((unbiased + 15) << 10) | (retained & 0x3FF), False
    sub_shift = fraction_bits - 24
    if sub_shift >= 0:
        retained = magnitude >> sub_shift
        discarded = magnitude & ((1 << sub_shift) - 1) if sub_shift else 0
        guard = (discarded >> (sub_shift - 1)) & 1 if sub_shift else 0
        sticky = bool(discarded & ((1 << (sub_shift - 1)) - 1)) if sub_shift > 1 else False
        retained += int(guard and (sticky or (retained & 1)))
    else:
        retained = magnitude << (-sub_shift)
    if retained >= 0x400:
        return (sign << 15) | 0x0400, False
    return (sign << 15) | retained, False


def build_final_rmsnorm_hidden(path: Path, records: dict[str, dict[str, Any]]) -> list[int]:
    terminal_q24 = [(((index * 73) % 513) - 256) << 14 for index in range(HIDDEN_SIZE)]
    terminal_bits = [fixed_to_f16(value, 24)[0] for value in terminal_q24]
    norm = records["model.norm.weight"]
    with path.open("rb") as stream:
        stream.seek(norm["offset"])
        norm_bits = np.frombuffer(stream.read(norm["bytes"]), dtype="<u2").astype(np.uint16)
    decoded_terminal = [decode_f16_q24(bits)[0] for bits in terminal_bits]
    decoded_norm = [decode_f16_q24(int(bits)) for bits in norm_bits]
    require(all(item[1] for item in decoded_norm), "nonfinite final norm weight")
    mean_q48 = round_div_even(
        sum(value * value for value in decoded_terminal) + EPSILON_Q48 * HIDDEN_SIZE,
        HIDDEN_SIZE,
    )
    rms_q24 = math.isqrt(mean_q48)
    require(rms_q24 > 0, "final RMSNorm divisor is zero")
    normalized: list[int] = []
    for activation, (weight, _, _) in zip(decoded_terminal, decoded_norm, strict=True):
        product = activation * weight
        value = round_div_even(abs(product), rms_q24)
        if product < 0:
            value = -value
        bits, saturated = fixed_to_f16(value, 24)
        require(not saturated, "final RMSNorm output saturated")
        normalized.append(bits)
    return normalized


def decode_array_q24(bits: np.ndarray) -> np.ndarray:
    unsigned = bits.astype(np.uint16, copy=False)
    exponent = ((unsigned >> 10) & 0x1F).astype(np.int64)
    require(bool(np.all(exponent != 0x1F)), "nonfinite tied weight")
    fraction = (unsigned & 0x3FF).astype(np.int64)
    magnitude = np.where(exponent == 0, fraction, np.left_shift(0x400 | fraction, np.maximum(exponent - 1, 0)))
    return np.where((unsigned & 0x8000) != 0, -magnitude, magnitude).astype(np.int64)


def selected_tokens(hidden_sha256: str, count: int = 12) -> list[int]:
    seed = hashlib.sha256(
        (CHECKPOINT_SHA256 + hidden_sha256 + "ace3-lm-head-logit-checks-v1").encode("ascii")
    ).digest()
    selected = {0, VOCAB_SIZE - 1}
    counter = 0
    while len(selected) < count:
        block = hashlib.sha256(seed + counter.to_bytes(4, "little")).digest()
        selected.add(int.from_bytes(block[:8], "little") % VOCAB_SIZE)
        counter += 1
    return sorted(selected)


def exact_traversal(
    path: Path,
    records: dict[str, dict[str, Any]],
    hidden_bits: list[int],
) -> tuple[list[tuple[int, int, int]], dict[int, tuple[int, int]], str]:
    hidden_q24 = decode_array_q24(np.asarray(hidden_bits, dtype="<u2"))
    hidden_sha256 = hashlib.sha256(np.asarray(hidden_bits, dtype="<u2").tobytes()).hexdigest()
    selected = set(selected_tokens(hidden_sha256))
    checks: dict[int, tuple[int, int]] = {}
    heap: list[tuple[int, int, int, int]] = []
    logits_digest = hashlib.sha256()
    weights = records["model.embed_tokens.weight"]
    mapped = np.memmap(
        path,
        dtype="<u2",
        mode="r",
        offset=weights["offset"],
        shape=(VOCAB_SIZE, HIDDEN_SIZE),
    )
    for begin in range(0, VOCAB_SIZE, 512):
        end = min(begin + 512, VOCAB_SIZE)
        weight_q24 = decode_array_q24(mapped[begin:end])
        accumulators = np.sum(weight_q24 * hidden_q24, axis=1, dtype=np.int64)
        for row, accumulator_value in enumerate(accumulators, start=begin):
            accumulator = int(accumulator_value)
            bits, saturated = fixed_to_f16(accumulator, 48)
            require(not saturated, f"official logit {row} saturated")
            value, finite, _ = decode_f16_q24(bits)
            require(finite, f"official logit {row} is nonfinite")
            logits_digest.update(struct.pack("<H", bits))
            if row in selected:
                checks[row] = (bits, accumulator)
            item = (value, -row, row, bits)
            if len(heap) < TOP_K:
                heapq.heappush(heap, item)
            elif item[:2] > heap[0][:2]:
                heapq.heapreplace(heap, item)
    winners = sorted(heap, key=lambda item: (-item[0], item[2]))
    require(set(checks) == selected, "selected logit checks are incomplete")
    return [(item[2], item[3], item[0]) for item in winners], checks, logits_digest.hexdigest()


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.write_text("".join(f"{line}\n" for line in lines), encoding="ascii")


def generate(checkpoint: Path, output_dir: Path) -> dict[str, Any]:
    records = tensor_records(checkpoint)
    hidden = build_final_rmsnorm_hidden(checkpoint, records)
    hidden_payload = np.asarray(hidden, dtype="<u2").tobytes()
    hidden_sha256 = hashlib.sha256(hidden_payload).hexdigest()
    winners, checks, logits_sha256 = exact_traversal(checkpoint, records, hidden)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_lines(output_dir / "hidden.hex", (f"{bits:04x}" for bits in hidden))
    write_lines(
        output_dir / "topk.txt",
        (f"{rank} {token} {bits:04x} {value}" for rank, (token, bits, value) in enumerate(winners)),
    )
    write_lines(
        output_dir / "checks.txt",
        (f"{token} {bits:04x} {accumulator & ((1 << 96) - 1):024x}" for token, (bits, accumulator) in sorted(checks.items())),
    )
    embedding = records["model.embed_tokens.weight"]
    manifest = {
        "schema_version": 1,
        "kind": "ace3_streaming_tied_lm_head_official_vectors",
        "model": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "checkpoint_filename": checkpoint.name,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "checkpoint_bytes": CHECKPOINT_BYTES,
            "streamed_tensor": "model.embed_tokens.weight",
            "tied_peer": "lm_head.weight",
            "tied_value_sha256": TIED_WEIGHT_SHA256,
            "weight_offset": embedding["offset"],
            "weight_bytes": embedding["bytes"],
        },
        "geometry": {"hidden_size": HIDDEN_SIZE, "vocab_size": VOCAB_SIZE, "top_k": TOP_K},
        "hidden": {
            "source": "exact final RMSNorm of the deterministic 896-wide structural terminal fixture using authenticated model.norm.weight",
            "sha256": hidden_sha256,
        },
        "selection": {
            "policy": "descending rounded-FP16 numeric logit, then ascending token ID",
            "selected_check_policy": "SHA256(checkpoint digest || hidden digest || domain), independent of logit values, plus boundary IDs",
            "selected_check_token_ids": sorted(checks),
        },
        "logits_sha256": logits_sha256,
        "top_k": [
            {"rank": rank, "token_id": token, "logit_f16_bits": bits, "logit_q24": value}
            for rank, (token, bits, value) in enumerate(winners)
        ],
        "claim_boundary": {
            "demonstrated": "authenticated official-shape standalone final-RMSNorm-to-tied-lm_head traversal",
            "not_demonstrated": ["integrated dialogue", "synthesis", "PPA", "FPGA", "latency"],
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_lines(
        output_dir / "run.cfg",
        (
            f"checkpoint_bytes={CHECKPOINT_BYTES}",
            f"weight_offset={embedding['offset']}",
            f"weight_bytes={embedding['bytes']}",
            f"hidden_size={HIDDEN_SIZE}",
            f"vocab_size={VOCAB_SIZE}",
            f"top_k={TOP_K}",
            f"check_count={len(checks)}",
        ),
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = generate(args.checkpoint, args.output_dir)
    print(
        "LM_HEAD_REFERENCE_PASS "
        f"vocab={manifest['geometry']['vocab_size']} hidden={manifest['geometry']['hidden_size']} "
        f"top_token={manifest['top_k'][0]['token_id']} checks={len(manifest['selection']['selected_check_token_ids'])}"
    )


if __name__ == "__main__":
    main()
