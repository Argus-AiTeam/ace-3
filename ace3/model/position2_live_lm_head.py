#!/usr/bin/env python3
"""Authenticate position-2 traversal output and execute the live tied lm_head."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch

try:
    from . import streaming_lm_head_reference as lm_reference
    from .fp16_adaptation_oracle import rmsnorm
except ImportError:
    import streaming_lm_head_reference as lm_reference
    from fp16_adaptation_oracle import rmsnorm

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "build/model24_rtl_cascade/checkpoint/model.safetensors"
PARENT_EVIDENCE = ROOT / "build/model24_selected_token_position2/evidence.json"
SELECTED_INPUT_TOKEN = 271
POSITION = 2
LAYER_COUNT = 24
HIDDEN_SIZE = 896
VOCAB_SIZE = 151936
TOP_K = 10
EXPECTED_WEIGHTS = HIDDEN_SIZE * VOCAB_SIZE
SOURCE_PATHS = {
    "position2_lm_head": "ace3/model/position2_live_lm_head.py",
    "position2_traversal_validator": (
        "ace3/model/validate_selected_token_position2_traversal.py"
    ),
    "streaming_lm_head_reference": "ace3/model/streaming_lm_head_reference.py",
    "fp16_adaptation_oracle": "ace3/model/fp16_adaptation_oracle.py",
    "streaming_lm_head_rtl": "ace3/rtl/ace3_streaming_tied_lm_head_topk.sv",
    "fp16_fixed_rtl": "ace3/rtl/ace3_fp16_fixed.sv",
    "q47_48_rounder_rtl": "ace3/rtl/ace3_q47_48_to_f16_rne.sv",
    "verilator_harness": "ace3/tb/ace3_streaming_tied_lm_head_topk_main.cpp",
    "focused_tests": "ace3/model/tests/test_position2_live_lm_head.py",
    "makefile": "Makefile",
}
PARENT_VALIDATOR = ROOT / SOURCE_PATHS["position2_traversal_validator"]
RTL_RECEIPT = re.compile(
    r"STREAMING_LM_HEAD_OFFICIAL_PASS hidden=(\d+) vocab=(\d+) "
    r"weights=(\d+) top_token=(\d+) checks=(\d+) cycles=(\d+) "
    r"integrated_dialogue=not_run synthesis=not_run ppa=not_measured "
    r"fpga=not_run latency=not_claimed\n"
)


class EvidenceError(RuntimeError):
    """Raised when a consumed artifact or semantic claim is invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while payload := stream.read(1024 * 1024):
            digest.update(payload)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def require_file_record(
    path: Path, record: Mapping[str, Any], label: str
) -> dict[str, Any]:
    actual = file_record(path)
    require(actual == record, f"{label} content binding mismatch")
    return actual


def require_expected_hash(path: Path, expected: str, label: str) -> dict[str, Any]:
    record = file_record(path)
    require(record["sha256"] == expected, f"{label} SHA-256 mismatch")
    return record


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    document = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_keys,
    )
    require(isinstance(document, dict), f"{path} must contain a JSON object")
    return document


def canonical_json(document: Mapping[str, Any]) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")


def load_terminal_bits(
    path: Path, hidden_size: int = HIDDEN_SIZE
) -> np.ndarray:
    rows = path.read_text(encoding="ascii").splitlines()
    require(len(rows) == hidden_size, "parent terminal hidden row count mismatch")
    values = np.empty(hidden_size, dtype="<u2")
    for index, row in enumerate(rows):
        require(
            len(row) == 10
            and row[:2] == "00"
            and int(row[2:6], 16) == index,
            f"parent terminal hidden row {index} ordering mismatch",
        )
        values[index] = int(row[6:10], 16)
    return values


def source_records(
    repository_root: Path = ROOT,
    paths: Mapping[str, str] = SOURCE_PATHS,
) -> dict[str, dict[str, Any]]:
    return {
        label: file_record(repository_root / relative_path)
        for label, relative_path in paths.items()
    }


def require_source_bindings(
    stored: Mapping[str, Any],
    repository_root: Path = ROOT,
    paths: Mapping[str, str] = SOURCE_PATHS,
) -> None:
    actual = source_records(repository_root, paths)
    require(set(stored) == set(actual), "consumed source closure is incomplete")
    for label, record in actual.items():
        require(stored.get(label) == record, f"{label} source binding mismatch")


def validate_parent_evidence(evidence_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(PARENT_VALIDATOR),
            "validate",
            "--output",
            str(evidence_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (completed.stderr or completed.stdout).strip()
    require(
        completed.returncode == 0,
        f"parent traversal evidence is stale: {output}",
    )


def authenticate_parent_traversal(
    evidence_path: Path = PARENT_EVIDENCE,
    parent_validator: Callable[[Path], None] | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    require(evidence_path.is_file(), "parent traversal evidence is missing")
    evidence_record = file_record(evidence_path)
    evidence = load_json(evidence_path)
    require(
        evidence.get("schema_version") == 2
        and evidence.get("kind")
        == "ace3_selected_token_position2_continuation_evidence",
        "parent traversal evidence identity mismatch",
    )
    require(
        evidence.get("status") == "COMPLETE",
        "parent traversal evidence is not COMPLETE",
    )
    validate_parent = parent_validator or validate_parent_evidence
    validate_parent(evidence_path)
    require_file_record(
        evidence_path,
        evidence_record,
        "parent traversal evidence",
    )
    model = evidence.get("model")
    selected_token = evidence.get("selected_token")
    traversal = evidence.get("current_continuation_attempt")
    require(
        isinstance(model, dict)
        and model.get("repository") == lm_reference.MODEL_REPOSITORY
        and model.get("revision") == lm_reference.MODEL_REVISION
        and model.get("checkpoint_sha256") == lm_reference.CHECKPOINT_SHA256
        and isinstance(selected_token, dict)
        and selected_token.get("selected_token_id") == SELECTED_INPUT_TOKEN
        and isinstance(traversal, dict),
        "parent traversal model or selected-token binding mismatch",
    )
    layers = traversal.get("layers")
    require(
        traversal.get("status") == "COMPLETE"
        and traversal.get("execution") == "current-worktree compiled Verilator RTL"
        and traversal.get("selected_token_id") == SELECTED_INPUT_TOKEN
        and traversal.get("position") == POSITION
        and traversal.get("layer_order") == list(range(LAYER_COUNT))
        and traversal.get("natural_terminal_layers") == LAYER_COUNT,
        "parent traversal identity mismatch",
    )
    require(
        isinstance(layers, list)
        and [layer.get("layer_index") for layer in layers]
        == list(range(LAYER_COUNT)),
        "parent traversal layer ordering mismatch",
    )
    layer23 = layers[-1]
    output = layer23.get("output")
    post_layer23 = traversal.get("post_layer23")
    require(
        isinstance(output, dict)
        and isinstance(post_layer23, dict)
        and isinstance(output.get("path"), str)
        and post_layer23.get("natural_terminal") is True
        and post_layer23.get("independent_oracle_within_tolerance") is True
        and post_layer23.get("hidden_sha256") == output.get("semantic_sha256"),
        "parent layer-23 terminal binding mismatch",
    )
    terminal_path = Path(output["path"])
    require(terminal_path.is_file(), "parent layer-23 terminal is missing")
    terminal_record = require_file_record(
        terminal_path,
        {
            "path": output.get("path"),
            "bytes": output.get("bytes"),
            "sha256": output.get("sha256"),
        },
        "parent layer-23 terminal",
    )
    terminal_bits = load_terminal_bits(terminal_path)
    terminal_hidden_sha256 = sha256_bytes(terminal_bits.tobytes())
    require(
        terminal_hidden_sha256 == output.get("semantic_sha256"),
        "parent terminal semantic SHA-256 mismatch",
    )
    return {
        "evidence": evidence_record,
        "status": "COMPLETE",
        "selected_token_id": SELECTED_INPUT_TOKEN,
        "position": POSITION,
        "natural_terminal_layers": LAYER_COUNT,
        "layer23_terminal": terminal_record,
        "terminal_hidden_sha256": terminal_hidden_sha256,
    }, terminal_bits


def topk_from_bits(bits: np.ndarray) -> list[tuple[int, int, int]]:
    heap: list[tuple[int, int, int, int]] = []
    for token_id, raw in enumerate(np.asarray(bits, dtype="<u2")):
        value, finite, _ = lm_reference.decode_f16_q24(int(raw))
        require(finite, f"nonfinite logit at token {token_id}")
        item = (value, -token_id, token_id, int(raw))
        if len(heap) < TOP_K:
            heapq.heappush(heap, item)
        elif item[:2] > heap[0][:2]:
            heapq.heapreplace(heap, item)
    winners = sorted(heap, key=lambda item: (-item[0], item[2]))
    return [(item[2], item[3], item[0]) for item in winners]


def final_rmsnorm(
    checkpoint: Path,
    records: Mapping[str, Mapping[str, Any]],
    terminal_bits: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    norm = records["model.norm.weight"]
    with checkpoint.open("rb") as stream:
        stream.seek(norm["offset"])
        norm_bits = np.frombuffer(stream.read(norm["bytes"]), dtype="<u2").copy()
    outputs, mean_q48, rms_q24 = rmsnorm(
        terminal_bits.tolist(), norm_bits.tolist()
    )
    require(
        all(not invalid and not saturated for _, invalid, saturated in outputs),
        "final RMSNorm produced invalid or saturated output",
    )
    normalized = np.asarray([raw for raw, _, _ in outputs], dtype="<u2")
    terminal_f64 = terminal_bits.view("<f2").astype(np.float64)
    norm_f64 = norm_bits.view("<f2").astype(np.float64)
    torch_input = torch.from_numpy(terminal_f64)
    torch_weight = torch.from_numpy(norm_f64)
    torch_output = (
        torch_input
        * torch.rsqrt(
            torch.mean(torch_input * torch_input)
            + lm_reference.EPSILON_Q48 / float(1 << 48)
        )
        * torch_weight
    )
    difference = np.abs(
        normalized.view("<f2").astype(np.float64) - torch_output.numpy()
    )
    return normalized, {
        "input_sha256": sha256_bytes(terminal_bits.tobytes()),
        "output_sha256": sha256_bytes(normalized.tobytes()),
        "mean_q48": mean_q48,
        "rms_q24": rms_q24,
        "weight": {
            "tensor": "model.norm.weight",
            "dtype": "FP16",
            "shape": [HIDDEN_SIZE],
            "sha256": lm_reference.FINAL_NORM_SHA256,
        },
        "accepted_path": "ACE-3 exact Q24/Q48 RMSNorm with binary16 RNE output",
        "pytorch_float64_reference": {
            "max_abs_error": float(difference.max()),
            "mean_abs_error": float(difference.mean()),
        },
    }


def full_vocabulary_oracles(
    checkpoint: Path,
    records: Mapping[str, Mapping[str, Any]],
    hidden_bits: np.ndarray,
) -> tuple[dict[str, Any], dict[int, tuple[int, int]], list[tuple[int, int, int]]]:
    torch.set_num_threads(1)
    hidden_q24 = lm_reference.decode_array_q24(hidden_bits)
    hidden_f64 = hidden_bits.view("<f2").astype(np.float64)
    selected = set(
        lm_reference.selected_tokens(sha256_bytes(hidden_bits.tobytes()))
    )
    exact = np.empty(VOCAB_SIZE, dtype="<u2")
    pytorch = np.empty(VOCAB_SIZE, dtype="<u2")
    checks: dict[int, tuple[int, int]] = {}
    weights = records["model.embed_tokens.weight"]
    mapped = np.memmap(
        checkpoint,
        dtype="<u2",
        mode="r",
        offset=weights["offset"],
        shape=(VOCAB_SIZE, HIDDEN_SIZE),
    )
    hidden_tensor = torch.from_numpy(hidden_f64)
    for begin in range(0, VOCAB_SIZE, 512):
        end = min(begin + 512, VOCAB_SIZE)
        weight_bits = np.asarray(mapped[begin:end], dtype="<u2")
        accumulators = np.sum(
            lm_reference.decode_array_q24(weight_bits) * hidden_q24,
            axis=1,
            dtype=np.int64,
        )
        for offset, accumulator_value in enumerate(accumulators):
            token_id = begin + offset
            accumulator = int(accumulator_value)
            raw, saturated = lm_reference.fixed_to_f16(accumulator, 48)
            require(not saturated, f"official logit {token_id} saturated")
            exact[token_id] = raw
            if token_id in selected:
                checks[token_id] = (raw, accumulator)
        values = torch.from_numpy(
            weight_bits.view("<f2").astype(np.float64)
        ).matmul(hidden_tensor)
        pytorch[begin:end] = np.asarray(values.numpy(), dtype="<f2").view("<u2")
    mismatches = int(np.count_nonzero(exact != pytorch))
    exact_top = topk_from_bits(exact)
    pytorch_top = topk_from_bits(pytorch)
    require(mismatches == 0, f"full-vocabulary oracle mismatch count is {mismatches}")
    require(exact_top == pytorch_top, "full-vocabulary Top-K oracle mismatch")
    require(set(checks) == selected, "selected exact checks are incomplete")
    receipt = {
        "coverage": {
            "vocab_size": VOCAB_SIZE,
            "exact_logits": VOCAB_SIZE,
            "pytorch_logits": VOCAB_SIZE,
            "rounded_fp16_mismatches": mismatches,
        },
        "exact_integer": {
            "implementation": (
                "FP16 operands decoded to Q16.24, exact Q32.48 products, "
                "signed int64 reduction, one binary16 RNE output rounding"
            ),
            "logits_sha256": sha256_bytes(exact.tobytes()),
        },
        "pytorch_oracle": {
            "implementation": (
                "PyTorch CPU float64 matmul over independently decoded FP16 "
                "operands, one binary16 RNE output rounding"
            ),
            "logits_sha256": sha256_bytes(pytorch.tobytes()),
        },
        "agreement": {
            "all_rounded_fp16_logits_equal": True,
            "top_k_equal": True,
            "selected_token_equal": True,
        },
        "selection_policy": (
            "descending rounded finite FP16 numeric logit; equal logits use "
            "ascending token ID"
        ),
        "selected_token_id": exact_top[0][0],
        "selected_logit_f16_bits": exact_top[0][1],
        "selected_check_token_ids": sorted(selected),
        "top_k": [
            {
                "rank": rank,
                "token_id": token,
                "logit_f16_bits": raw,
                "logit_q24": value,
            }
            for rank, (token, raw, value) in enumerate(exact_top)
        ],
    }
    return receipt, checks, exact_top


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.write_text("".join(f"{line}\n" for line in lines), encoding="ascii")


def compute_oracle(
    checkpoint: Path,
    parent_evidence: Path = PARENT_EVIDENCE,
) -> dict[str, Any]:
    parent, terminal_bits = authenticate_parent_traversal(parent_evidence)
    records = lm_reference.tensor_records(checkpoint)
    normalized, final_norm = final_rmsnorm(
        checkpoint, records, terminal_bits
    )
    oracle, checks, winners = full_vocabulary_oracles(
        checkpoint, records, normalized
    )
    return {
        "parent": parent,
        "checkpoint": file_record(checkpoint),
        "final_rmsnorm": final_norm,
        "oracle": oracle,
        "_normalized": normalized,
        "_checks": checks,
        "_winners": winners,
        "_records": records,
    }


def prepare(
    checkpoint: Path,
    output_dir: Path,
    parent_evidence: Path = PARENT_EVIDENCE,
) -> dict[str, Any]:
    require(not (output_dir / "oracle.json").exists(), "oracle output already exists")
    computed = compute_oracle(checkpoint, parent_evidence)
    vectors = output_dir / "vectors"
    vectors.mkdir(parents=True, exist_ok=False)
    normalized = computed.pop("_normalized")
    checks = computed.pop("_checks")
    winners = computed.pop("_winners")
    records = computed.pop("_records")
    write_lines(vectors / "hidden.hex", (f"{int(raw):04x}" for raw in normalized))
    write_lines(
        vectors / "checks.txt",
        (
            f"{token} {raw:04x} {accumulator & ((1 << 96) - 1):024x}"
            for token, (raw, accumulator) in sorted(checks.items())
        ),
    )
    write_lines(
        vectors / "topk.txt",
        (
            f"{rank} {token} {raw:04x} {value}"
            for rank, (token, raw, value) in enumerate(winners)
        ),
    )
    embedding = records["model.embed_tokens.weight"]
    write_lines(
        vectors / "run.cfg",
        (
            f"checkpoint_bytes={checkpoint.stat().st_size}",
            f"weight_offset={embedding['offset']}",
            f"weight_bytes={embedding['bytes']}",
            f"hidden_size={HIDDEN_SIZE}",
            f"vocab_size={VOCAB_SIZE}",
            f"top_k={TOP_K}",
            f"check_count={len(checks)}",
        ),
    )
    (output_dir / "oracle.json").write_bytes(canonical_json(computed))
    return computed


def parse_rtl_receipt(payload: str, oracle: Mapping[str, Any]) -> dict[str, Any]:
    match = RTL_RECEIPT.fullmatch(payload)
    require(match is not None, "official RTL terminal receipt is malformed")
    hidden, logits, weights, token, checks, cycles = map(int, match.groups())
    require(hidden == HIDDEN_SIZE, "official RTL hidden count mismatch")
    require(logits == VOCAB_SIZE, "official RTL full-vocabulary count mismatch")
    require(weights == EXPECTED_WEIGHTS, "official RTL weight count mismatch")
    require(
        token == oracle["selected_token_id"],
        "official RTL selected token mismatch",
    )
    require(
        checks == len(oracle["selected_check_token_ids"]),
        "official RTL selected-check count mismatch",
    )
    require(cycles > 0, "official RTL cycle count must be positive")
    return {
        "natural_terminal": True,
        "hidden_elements": hidden,
        "accepted_logits": logits,
        "accepted_weights": weights,
        "selected_token_id": token,
        "selected_logit_f16_bits": oracle["selected_logit_f16_bits"],
        "matched_selected_checks": checks,
        "reported_simulation_cycles": cycles,
        "top_k": oracle["top_k"],
    }


def artifact_records(output_dir: Path) -> dict[str, dict[str, Any]]:
    names = (
        "oracle.json",
        "vectors/hidden.hex",
        "vectors/checks.txt",
        "vectors/topk.txt",
        "vectors/run.cfg",
        "tests.log",
        "prepare.log",
        "compile.log",
        "rtl.log",
    )
    return {name: file_record(output_dir / name) for name in names}


def assemble_evidence(
    output_dir: Path,
    binary: Path,
    computed: Mapping[str, Any],
) -> dict[str, Any]:
    rtl = parse_rtl_receipt(
        (output_dir / "rtl.log").read_text(encoding="utf-8"),
        computed["oracle"],
    )
    tests_log = (output_dir / "tests.log").read_text(encoding="utf-8")
    require("Ran " in tests_log and "\nOK\n" in tests_log, "focused tests did not pass")
    return {
        "schema_version": 1,
        "kind": "ace3_position2_live_final_rmsnorm_streaming_lm_head_evidence",
        "status": "COMPLETE",
        "model": {
            "repository": lm_reference.MODEL_REPOSITORY,
            "revision": lm_reference.MODEL_REVISION,
            "checkpoint": computed["checkpoint"],
            "geometry": {
                "hidden_size": HIDDEN_SIZE,
                "vocab_size": VOCAB_SIZE,
                "top_k": TOP_K,
            },
        },
        "input": {
            "token_id": SELECTED_INPUT_TOKEN,
            "position": POSITION,
            "source": "authenticated parent position-2 layer-23 terminal hidden state",
        },
        "parent_traversal": computed["parent"],
        "final_rmsnorm": computed["final_rmsnorm"],
        "full_vocabulary_oracles": computed["oracle"],
        "official_shape_streaming_tied_lm_head": {
            **rtl,
            "implementation": "current-worktree Verilator RTL",
            "binary": file_record(binary),
            "terminal_log": file_record(output_dir / "rtl.log"),
            "raw_full_logit_vector_materialized": False,
        },
        "selected_next_token": {
            "token_id": computed["oracle"]["selected_token_id"],
            "logit_f16_bits": computed["oracle"]["selected_logit_f16_bits"],
            "exact_integer_pytorch_agreement": True,
            "rtl_top_k_agreement": True,
        },
        "consumed_sources": source_records(),
        "artifacts": artifact_records(output_dir),
        "claim_boundary": {
            "demonstrated": (
                "token 271 at position 2 from the bound parent traversal through "
                "accepted exact final RMSNorm and a live official-shape streaming "
                "tied lm_head/Top-K, with exact-integer and PyTorch full-vocabulary "
                "agreement"
            ),
            "dialogue": "not claimed",
            "synthesis": "not run",
            "ppa": "not measured",
            "fpga": "not run",
            "latency": "not measured",
            "throughput": "not measured",
        },
    }


def seal(output_dir: Path, binary: Path, evidence_path: Path) -> dict[str, Any]:
    require(not evidence_path.exists(), "evidence output already exists")
    computed = load_json(output_dir / "oracle.json")
    evidence = assemble_evidence(output_dir, binary, computed)
    evidence_path.write_bytes(canonical_json(evidence))
    return evidence


def validate(
    output_dir: Path,
    binary: Path,
    evidence_path: Path,
    parent_evidence: Path = PARENT_EVIDENCE,
) -> dict[str, Any]:
    stored = load_json(evidence_path)
    require_source_bindings(stored.get("consumed_sources", {}))
    for name, record in stored.get("artifacts", {}).items():
        require_file_record(output_dir / name, record, name)
    require_file_record(
        binary,
        stored.get("official_shape_streaming_tied_lm_head", {}).get("binary", {}),
        "official RTL binary",
    )
    computed = compute_oracle(CHECKPOINT, parent_evidence)
    computed.pop("_normalized")
    computed.pop("_checks")
    computed.pop("_winners")
    computed.pop("_records")
    require(
        load_json(output_dir / "oracle.json") == computed,
        "stored full-vocabulary oracle is stale",
    )
    fresh = assemble_evidence(output_dir, binary, computed)
    require(stored == fresh, "stored position-2 lm_head evidence is stale")
    return stored


def print_summary(document: Mapping[str, Any], evidence_path: Path) -> None:
    print(
        "POSITION2_LIVE_LM_HEAD_PASS "
        f"input_token={document['input']['token_id']} "
        f"position={document['input']['position']} "
        f"final_rmsnorm_sha256={document['final_rmsnorm']['output_sha256']} "
        f"selected_next_token={document['selected_next_token']['token_id']} "
        f"vocab={document['model']['geometry']['vocab_size']} "
        "exact_pytorch_mismatches=0 rtl_topk_match=1 "
        f"evidence={evidence_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="operation", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument(
        "--parent-evidence", type=Path, default=PARENT_EVIDENCE
    )
    seal_parser = subparsers.add_parser("seal")
    seal_parser.add_argument("--output-dir", type=Path, required=True)
    seal_parser.add_argument("--binary", type=Path, required=True)
    seal_parser.add_argument("--evidence", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--output-dir", type=Path, required=True)
    validate_parser.add_argument("--binary", type=Path, required=True)
    validate_parser.add_argument("--evidence", type=Path, required=True)
    validate_parser.add_argument(
        "--parent-evidence", type=Path, default=PARENT_EVIDENCE
    )
    args = parser.parse_args()
    try:
        if args.operation == "prepare":
            document = prepare(
                args.checkpoint.resolve(strict=True),
                args.output_dir.resolve(),
                args.parent_evidence.resolve(),
            )
            print(
                "POSITION2_LIVE_LM_HEAD_PREPARE_PASS "
                f"selected_next_token={document['oracle']['selected_token_id']} "
                "exact_pytorch_mismatches=0"
            )
            return
        output_dir = args.output_dir.resolve(strict=True)
        binary = args.binary.resolve(strict=True)
        evidence = args.evidence.resolve() if args.operation == "seal" else args.evidence.resolve(strict=True)
        document = (
            seal(output_dir, binary, evidence)
            if args.operation == "seal"
            else validate(
                output_dir,
                binary,
                evidence,
                args.parent_evidence.resolve(),
            )
        )
    except (EvidenceError, OSError, ValueError, KeyError) as error:
        raise SystemExit(f"POSITION2_LIVE_LM_HEAD_FAIL {error}") from error
    print_summary(document, evidence)


if __name__ == "__main__":
    main()
