#!/usr/bin/env python3
"""Prepare and validate the repair8-to-lm_head/position-2 execution package."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

ACCEPTED_COMMIT = "1b1f50c0cee8ddcd7e4f83a7b6c2fc03d8435dfe"
REPAIR8_RESULT_SHA256 = "da2c696b9701e86944ddfb22a28bb51f99f64a55bbe6e325aa58da7bc7d420c9"
TERMINAL_HIDDEN_SHA256 = "3e99fe3e1d30e6350d32f3ef45311f4a81114b27c13a5e01ef04f1a1ca0bf429"
CHECKPOINT_SHA256 = "c50d807b7bed7ff314308972e0f4bcf4e5a70bc60ad88fc7df53940831ed0c1b"
TIED_WEIGHT_SHA256 = "d74257dc547b48be5ae7b93f1c9af072c0c42dbbb85503078e25c59cd09e68d0"
TOKENIZER_SHA256 = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
TOKENIZER_CONFIG_SHA256 = "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583"
MODEL_REPOSITORY = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
MODEL_REVISION = "db09cd27ead7fee40cdee309693cf83601b9c899"
HIDDEN_SIZE = 896
VOCAB_SIZE = 151936
TOP_K = 10
LAYER_COUNT = 24
SOURCE_PATHS = (
    "ace3/rtl/ace3_streaming_tied_lm_head_topk.sv",
    "ace3/rtl/ace3_fp16_fixed.sv",
    "ace3/rtl/ace3_q47_48_to_f16_rne.sv",
    "ace3/tb/ace3_streaming_tied_lm_head_topk_main.cpp",
    "ace3/tb/ace3_streaming_tied_lm_head_topk_tb.sv",
    "ace3/model/streaming_lm_head_reference.py",
    "ace3/model/fp16_adaptation_oracle.py",
    "ace3/model/position1_model24_causal_traversal.py",
    "ace3/contracts/position1_model24_causal_traversal.json",
)


class PreparationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreparationError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while payload := stream.read(1024 * 1024):
            digest.update(payload)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def canonical_json(document: Mapping[str, Any]) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")


def load_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(document, dict), f"{path} root must be an object")
    return document


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def export_sources(repository: Path, output_dir: Path) -> dict[str, Any]:
    require(not output_dir.exists(), "sealed source directory already exists")
    output_dir.mkdir(parents=True)
    records = []
    for source_path in SOURCE_PATHS:
        payload = subprocess.check_output(
            ["git", "show", f"{ACCEPTED_COMMIT}:{source_path}"], cwd=repository
        )
        destination = output_dir / source_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        records.append(
            {
                "path": source_path,
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
        )
    manifest = {
        "schema_version": 1,
        "kind": "ace3_repair8_lm_head_sealed_sources",
        "accepted_commit": ACCEPTED_COMMIT,
        "files": records,
    }
    (output_dir / "source_manifest.json").write_bytes(canonical_json(manifest))
    return manifest


def load_terminal_bits(path: Path, hidden_size: int = HIDDEN_SIZE) -> list[int]:
    lines = path.read_text(encoding="ascii").splitlines()
    require(len(lines) == hidden_size, "terminal hidden row count mismatch")
    values = []
    for index, line in enumerate(lines):
        require(len(line) == 10, f"terminal hidden line {index} framing mismatch")
        require(int(line[:2], 16) == 0, f"terminal hidden line {index} token index mismatch")
        require(int(line[2:6], 16) == index, f"terminal hidden line {index} feature index mismatch")
        values.append(int(line[6:10], 16))
    return values


def terminal_payload(bits: Iterable[int]) -> bytes:
    return np.asarray(list(bits), dtype="<u2").tobytes()


def authenticate_repair8(result_path: Path) -> tuple[dict[str, Any], Path, list[dict[str, Any]]]:
    require(sha256_file(result_path) == REPAIR8_RESULT_SHA256, "repair8 result SHA256 mismatch")
    result = load_json(result_path)
    require(result.get("checkpoint_sha256") == CHECKPOINT_SHA256, "repair8 checkpoint mismatch")
    require(result.get("selected_token") == 2114 and result.get("position") == 1, "repair8 token/position mismatch")
    require(result.get("terminal_hidden_sha256") == TERMINAL_HIDDEN_SHA256, "repair8 terminal binding mismatch")
    require(result.get("natural_terminal_layers") == LAYER_COUNT, "repair8 natural terminal count mismatch")
    layers = result.get("layers")
    require(isinstance(layers, list) and len(layers) == LAYER_COUNT, "repair8 layer count mismatch")
    terminal_path = result_path.parent / "execution/transactions/position001/layer23/raw/final.hex"
    bits = load_terminal_bits(terminal_path)
    require(sha256_bytes(terminal_payload(bits)) == TERMINAL_HIDDEN_SHA256, "repair8 raw terminal SHA256 mismatch")
    states = []
    for layer_index, layer in enumerate(layers):
        require(layer.get("layer_index") == layer_index, "repair8 layer ordering mismatch")
        transaction = layer.get("transaction", {})
        require(transaction.get("natural_terminal") is True, f"repair8 layer {layer_index} is not natural terminal")
        state_dir = result_path.parent / f"execution/states/layer{layer_index:02d}/position002"
        envelope_path = state_dir / "envelope.json"
        state_path = state_dir / "state"
        envelope = load_json(envelope_path)
        state_record = file_record(state_path)
        require(envelope.get("layer_index") == layer_index, f"repair8 layer {layer_index} envelope mismatch")
        require(envelope.get("next_position") == 2, f"repair8 layer {layer_index} next-position mismatch")
        require(envelope.get("model_binding", {}).get("checkpoint_sha256") == CHECKPOINT_SHA256, f"repair8 layer {layer_index} model mismatch")
        require(envelope.get("state", {}).get("sha256") == state_record["sha256"], f"repair8 layer {layer_index} state hash mismatch")
        require(envelope.get("state", {}).get("bytes") == state_record["bytes"], f"repair8 layer {layer_index} state byte mismatch")
        require(transaction.get("state_sha256") == state_record["sha256"], f"repair8 layer {layer_index} result/state mismatch")
        states.append(
            {
                "layer_index": layer_index,
                "next_position": 2,
                "output_hidden_sha256": envelope.get("output_hidden_sha256"),
                "envelope": file_record(envelope_path),
                "state": state_record,
            }
        )
    require(states[-1]["output_hidden_sha256"] == TERMINAL_HIDDEN_SHA256, "layer23 state terminal mismatch")
    return result, terminal_path, states


def authenticate_tokenizer(tokenizer_dir: Path) -> dict[str, Any]:
    tokenizer = tokenizer_dir / "tokenizer.json"
    config = tokenizer_dir / "tokenizer_config.json"
    require(sha256_file(tokenizer) == TOKENIZER_SHA256, "tokenizer.json SHA256 mismatch")
    require(sha256_file(config) == TOKENIZER_CONFIG_SHA256, "tokenizer_config.json SHA256 mismatch")
    return {"tokenizer": file_record(tokenizer), "tokenizer_config": file_record(config)}


def topk_from_bits(bits: np.ndarray, reference: Any, top_k: int = TOP_K) -> list[tuple[int, int, int]]:
    heap: list[tuple[int, int, int, int]] = []
    for token_id, raw in enumerate(np.asarray(bits, dtype="<u2")):
        value, finite, _ = reference.decode_f16_q24(int(raw))
        require(finite, f"nonfinite logit at token {token_id}")
        item = (value, -token_id, token_id, int(raw))
        if len(heap) < top_k:
            heapq.heappush(heap, item)
        elif item[:2] > heap[0][:2]:
            heapq.heapreplace(heap, item)
    return [(item[2], item[3], item[0]) for item in sorted(heap, key=lambda item: (-item[0], item[2]))]


def full_vocabulary_oracle(
    checkpoint: Path,
    records: Mapping[str, Mapping[str, Any]],
    hidden_bits: list[int],
    reference: Any,
) -> tuple[dict[str, Any], dict[int, tuple[int, int]], list[tuple[int, int, int]]]:
    torch.set_num_threads(1)
    hidden_u16 = np.asarray(hidden_bits, dtype="<u2")
    hidden_q24 = reference.decode_array_q24(hidden_u16)
    hidden_f64 = hidden_u16.view("<f2").astype(np.float64)
    hidden_sha256 = sha256_bytes(hidden_u16.tobytes())
    selected_checks = set(reference.selected_tokens(hidden_sha256))
    checks: dict[int, tuple[int, int]] = {}
    exact_digest = hashlib.sha256()
    torch_digest = hashlib.sha256()
    exact_chunks = []
    torch_chunks = []
    mismatches = 0
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
            reference.decode_array_q24(weight_bits) * hidden_q24,
            axis=1,
            dtype=np.int64,
        )
        exact_bits = np.empty(end - begin, dtype="<u2")
        for offset, accumulator_value in enumerate(accumulators):
            token_id = begin + offset
            accumulator = int(accumulator_value)
            raw, saturated = reference.fixed_to_f16(accumulator, 48)
            require(not saturated, f"official logit {token_id} saturated")
            exact_bits[offset] = raw
            if token_id in selected_checks:
                checks[token_id] = (raw, accumulator)
        weights_f64 = weight_bits.view("<f2").astype(np.float64)
        torch_values = torch.from_numpy(weights_f64).matmul(hidden_tensor).numpy()
        torch_bits = np.asarray(torch_values, dtype="<f2").view("<u2")
        mismatches += int(np.count_nonzero(exact_bits != torch_bits))
        exact_digest.update(exact_bits.tobytes())
        torch_digest.update(torch_bits.tobytes())
        exact_chunks.append(exact_bits)
        torch_chunks.append(torch_bits)
    exact_all = np.concatenate(exact_chunks)
    torch_all = np.concatenate(torch_chunks)
    exact_top = topk_from_bits(exact_all, reference)
    torch_top = topk_from_bits(torch_all, reference)
    require(mismatches == 0, f"independent full-vocabulary integer mismatch count is {mismatches}")
    require(exact_top == torch_top, "independent full-vocabulary Top-K mismatch")
    require(set(checks) == selected_checks, "selected logit checks are incomplete")
    receipt = {
        "schema_version": 1,
        "kind": "ace3_repair8_lm_head_independent_oracle",
        "coverage": {"vocab_size": VOCAB_SIZE, "logit_count": len(exact_all), "integer_mismatches": mismatches},
        "numeric_boundary": {
            "rtl": "exact FP16 products accumulated in signed Q47.48; round once to binary16 RNE",
            "independent": "PyTorch CPU float64 matmul of authenticated FP16 tied weights and FP16 hidden; round once to binary16 RNE",
        },
        "selection_policy": "descending rounded finite FP16 numeric logit; equal logits use ascending token ID",
        "hidden_sha256": hidden_sha256,
        "exact_logits_sha256": exact_digest.hexdigest(),
        "independent_logits_sha256": torch_digest.hexdigest(),
        "selected_token_id": exact_top[0][0],
        "selected_logit_f16_bits": exact_top[0][1],
        "top_k": [
            {"rank": rank, "token_id": token, "logit_f16_bits": raw, "logit_q24": value}
            for rank, (token, raw, value) in enumerate(exact_top)
        ],
    }
    return receipt, checks, exact_top


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.write_text("".join(f"{line}\n" for line in lines), encoding="ascii")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    require(not output_dir.exists(), "preparation output already exists")
    source_manifest = load_json(args.source_dir / "source_manifest.json")
    require(source_manifest.get("accepted_commit") == ACCEPTED_COMMIT, "sealed source commit mismatch")
    for record in source_manifest.get("files", []):
        path = args.source_dir / record["path"]
        require(path.is_file() and sha256_file(path) == record["sha256"], f"sealed source mismatch: {record['path']}")
    repair8, terminal_path, states = authenticate_repair8(args.repair8_result)
    terminal_bits = load_terminal_bits(terminal_path)
    tokenizer = authenticate_tokenizer(args.tokenizer_dir)
    reference = load_module(args.source_dir / "ace3/model/streaming_lm_head_reference.py", "sealed_lm_head_reference")
    rmsnorm = load_module(args.source_dir / "ace3/model/fp16_adaptation_oracle.py", "sealed_fp16_adaptation")
    records = reference.tensor_records(args.checkpoint)
    norm = records["model.norm.weight"]
    with args.checkpoint.open("rb") as stream:
        stream.seek(norm["offset"])
        norm_bits = np.frombuffer(stream.read(norm["bytes"]), dtype="<u2").tolist()
    outputs, mean_q48, rms_q24 = rmsnorm.rmsnorm(terminal_bits, norm_bits)
    require(all(not invalid and not saturated for _, invalid, saturated in outputs), "final RMSNorm invalid or saturated")
    normalized_bits = [raw for raw, _, _ in outputs]
    receipt, checks, winners = full_vocabulary_oracle(args.checkpoint, records, normalized_bits, reference)

    output_dir.mkdir(parents=True)
    vectors = output_dir / "vectors"
    vectors.mkdir()
    write_lines(vectors / "hidden.hex", (f"{raw:04x}" for raw in normalized_bits))
    write_lines(vectors / "topk.txt", (f"{rank} {token} {raw:04x} {value}" for rank, (token, raw, value) in enumerate(winners)))
    write_lines(vectors / "checks.txt", (f"{token} {raw:04x} {acc & ((1 << 96) - 1):024x}" for token, (raw, acc) in sorted(checks.items())))
    embedding = records["model.embed_tokens.weight"]
    write_lines(
        vectors / "run.cfg",
        (
            f"checkpoint_bytes={args.checkpoint.stat().st_size}",
            f"weight_offset={embedding['offset']}",
            f"weight_bytes={embedding['bytes']}",
            f"hidden_size={HIDDEN_SIZE}",
            f"vocab_size={VOCAB_SIZE}",
            f"top_k={TOP_K}",
            f"check_count={len(checks)}",
        ),
    )
    (output_dir / "oracle_receipt.json").write_bytes(canonical_json(receipt))
    selected_token = receipt["selected_token_id"]
    position2_contract = {
        "schema_version": 1,
        "kind": "ace3_position2_model24_causal_preparation",
        "source": {"repair8_result": file_record(args.repair8_result), "terminal_hidden_sha256": TERMINAL_HIDDEN_SHA256},
        "model": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION, "checkpoint_sha256": CHECKPOINT_SHA256},
        "selected_token": {
            "token_id": selected_token,
            "source": "independent full-vocabulary oracle prediction; later official lm_head receipt must match exactly before traversal authority",
            "official_receipt_required": True,
        },
        "positions": {"source": 1, "target": 2},
        "kv_feedback": {"layer_count": LAYER_COUNT, "states": states},
        "execution_authority": "withheld pending independent review and a separate cardinality-one authority",
    }
    (output_dir / "position2_contract.json").write_bytes(canonical_json(position2_contract))
    generated = {
        name: file_record(output_dir / name)
        for name in ("oracle_receipt.json", "position2_contract.json")
    }
    generated.update(
        {
            f"vectors/{name}": file_record(vectors / name)
            for name in ("hidden.hex", "topk.txt", "checks.txt", "run.cfg")
        }
    )
    environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "LC_ALL": "C",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
        "TZ": "UTC",
    }
    package = {
        "schema_version": 1,
        "kind": "ace3_repair8_lm_head_position2_preparation",
        "accepted_source_commit": ACCEPTED_COMMIT,
        "construction_source_manifest": file_record(args.source_dir / "source_manifest.json"),
        "contract": file_record(args.contract),
        "repair8": {
            "result": file_record(args.repair8_result),
            "terminal_raw": file_record(terminal_path),
            "terminal_hidden_sha256": TERMINAL_HIDDEN_SHA256,
            "result_claim_boundary": repair8.get("claim_boundary"),
        },
        "model": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "checkpoint": file_record(args.checkpoint),
            "tied_weight_sha256": TIED_WEIGHT_SHA256,
            "tokenizer": tokenizer,
            "geometry": {"hidden_size": HIDDEN_SIZE, "vocab_size": VOCAB_SIZE, "top_k": TOP_K},
        },
        "final_rmsnorm": {
            "input_sha256": TERMINAL_HIDDEN_SHA256,
            "output_sha256": receipt["hidden_sha256"],
            "mean_q48": mean_q48,
            "rms_q24": rms_q24,
            "implementation": "accepted commit fp16_adaptation_oracle.py Q26 RMS divisor",
        },
        "independent_oracle": generated["oracle_receipt.json"],
        "preflight": {
            "icarus_four_state": file_record(args.icarus_log),
            "verilator_numerical_protocol": file_record(args.verilator_log),
            "official_shape_binary": file_record(args.compiled_binary),
        },
        "official_lm_head": {
            "task_id": "position1-repair8-tied-lm-head-q26-official",
            "output_identity": "/dev/shm/ace3-position1-model24-6c135134/build/position1_repair8_tied_lm_head_q26_official",
            "cardinality": 1,
            "timeout_seconds": 43200,
            "environment": environment,
            "argv": [
                str(args.compiled_binary.resolve()),
                "--checkpoint", str(args.checkpoint.resolve()),
                "--config", str((vectors / "run.cfg").resolve()),
                "--hidden", str((vectors / "hidden.hex").resolve()),
                "--checks", str((vectors / "checks.txt").resolve()),
                "--topk", str((vectors / "topk.txt").resolve()),
            ],
            "receipt_schema": {
                "required": ["selected_token_id", "selected_logit_f16_bits", "top_k", "accepted_logits", "accepted_weights", "natural_terminal"],
                "selected_token_id": selected_token,
                "selected_logit_f16_bits": receipt["selected_logit_f16_bits"],
                "accepted_logits": VOCAB_SIZE,
                "accepted_weights": VOCAB_SIZE * HIDDEN_SIZE,
                "integer_mismatches": 0,
            },
            "execution_authority": "withheld",
        },
        "position2": {
            "task_id": "position2-model24-causal-q26-official",
            "output_identity": "/dev/shm/ace3-position1-model24-6c135134/build/position2_model24_causal_q26_official",
            "cardinality": 1,
            "timeout_seconds": 43200,
            "selected_token_id": selected_token,
            "official_lm_head_receipt_required": True,
            "contract": generated["position2_contract.json"],
            "execution_authority": "withheld",
        },
        "generated": generated,
        "claim_boundary": {
            "demonstrated": "read-only repair8 authentication, Q26 final RMSNorm, full-vocabulary path-distinct oracle agreement, and simulator/binary preflight preparation",
            "official_lm_head": "not executed",
            "position2_traversal": "not executed",
            "dialogue": "not claimed",
            "synthesis": "not run",
            "ppa": "not measured",
            "fpga": "not run",
            "latency": "not measured",
            "throughput": "not measured",
            "broad_generation_quality": "not claimed",
        },
    }
    (output_dir / "package.json").write_bytes(canonical_json(package))
    return package


def validate(package_path: Path) -> dict[str, Any]:
    package = load_json(package_path)
    require(package.get("kind") == "ace3_repair8_lm_head_position2_preparation", "package kind mismatch")
    require(package.get("accepted_source_commit") == ACCEPTED_COMMIT, "package source commit mismatch")
    require(package["repair8"]["result"]["sha256"] == REPAIR8_RESULT_SHA256, "package repair8 result mismatch")
    require(package["repair8"]["terminal_hidden_sha256"] == TERMINAL_HIDDEN_SHA256, "package terminal mismatch")
    require(package["model"]["checkpoint"]["sha256"] == CHECKPOINT_SHA256, "package checkpoint mismatch")
    require(package["model"]["geometry"]["vocab_size"] == VOCAB_SIZE, "package vocabulary mismatch")
    require(package["official_lm_head"]["execution_authority"] == "withheld", "lm_head authority was not withheld")
    require(package["position2"]["execution_authority"] == "withheld", "position2 authority was not withheld")
    require(package["official_lm_head"]["receipt_schema"]["integer_mismatches"] == 0, "package integer mismatch")
    for section in ("construction_source_manifest", "contract"):
        record = package[section]
        path = Path(record["path"])
        require(path.stat().st_size == record["bytes"] and sha256_file(path) == record["sha256"], f"package {section} hash mismatch")
    for record in package["generated"].values():
        path = Path(record["path"])
        require(path.stat().st_size == record["bytes"] and sha256_file(path) == record["sha256"], f"generated artifact mismatch: {path}")
    for record in package["preflight"].values():
        path = Path(record["path"])
        require(path.stat().st_size == record["bytes"] and sha256_file(path) == record["sha256"], f"preflight artifact mismatch: {path}")
    result_path = Path(package["repair8"]["result"]["path"])
    _, _, states = authenticate_repair8(result_path)
    require(len(states) == LAYER_COUNT, "position2 state cardinality mismatch")
    oracle = load_json(Path(package["independent_oracle"]["path"]))
    require(oracle["coverage"] == {"vocab_size": VOCAB_SIZE, "logit_count": VOCAB_SIZE, "integer_mismatches": 0}, "oracle coverage mismatch")
    require(oracle["selected_token_id"] == package["position2"]["selected_token_id"], "selected token lineage mismatch")
    require([item["rank"] for item in oracle["top_k"]] == list(range(TOP_K)), "Top-K rank mismatch")
    return {"selected_token_id": oracle["selected_token_id"], "top_k": len(oracle["top_k"]), "states": len(states)}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export-sources")
    export.add_argument("--repository", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    make = sub.add_parser("prepare")
    for name in ("source-dir", "repair8-result", "checkpoint", "tokenizer-dir", "contract", "compiled-binary", "icarus-log", "verilator-log", "output-dir"):
        make.add_argument(f"--{name}", type=Path, required=True)
    check = sub.add_parser("validate")
    check.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export-sources":
        manifest = export_sources(args.repository.resolve(strict=True), args.output_dir.resolve())
        print(f"REPAIR8_LM_HEAD_SOURCE_EXPORT_PASS files={len(manifest['files'])} commit={ACCEPTED_COMMIT}")
    elif args.command == "prepare":
        for name in ("source_dir", "repair8_result", "checkpoint", "tokenizer_dir", "contract", "compiled_binary", "icarus_log", "verilator_log"):
            setattr(args, name, getattr(args, name).resolve(strict=True))
        package = prepare(args)
        print(
            "REPAIR8_LM_HEAD_POSITION2_PREPARATION_PASS "
            f"vocab={VOCAB_SIZE} top_token={package['position2']['selected_token_id']} "
            "integer_mismatches=0 official_lm_head=not_executed position2=not_executed"
        )
    else:
        summary = validate(args.package.resolve(strict=True))
        print(f"REPAIR8_LM_HEAD_POSITION2_VALIDATION_PASS selected_token={summary['selected_token_id']} top_k={summary['top_k']} states={summary['states']}")


if __name__ == "__main__":
    main()
