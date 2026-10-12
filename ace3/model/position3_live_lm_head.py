#!/usr/bin/env python3
"""Authenticate position-3 traversal output and execute the live tied lm_head."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

try:
    from . import position2_live_lm_head as base
    from . import streaming_lm_head_reference as lm_reference
except ImportError:
    import position2_live_lm_head as base
    import streaming_lm_head_reference as lm_reference

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "build/model24_rtl_cascade/checkpoint/model.safetensors"
TOKENIZER_DIR = ROOT / "model24_execution_vectors/tokenizer"
PARENT_EVIDENCE = ROOT / "build/model24_selected_token_position3/evidence.json"
POSITION = 3
LAYER_COUNT = 24
HIDDEN_SIZE = 896
VOCAB_SIZE = 151936
TOP_K = 10
PROMPT_TOKEN_HISTORY = [151644, 2114, 271]
NUMERIC_PROFILE = (
    "native asymmetric packed INT4 AWQ W4A16 G128, no qzero plus-one "
    "adjustment, FP16 activations and FP16 K/V"
)
TOKENIZER_SHA256 = (
    "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
)
TOKENIZER_CONFIG_SHA256 = (
    "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583"
)
SOURCE_PATHS = {
    "position3_lm_head": "ace3/model/position3_live_lm_head.py",
    "position3_traversal_validator": (
        "ace3/model/validate_selected_token_position3_traversal.py"
    ),
    "position2_lm_head_primitives": "ace3/model/position2_live_lm_head.py",
    "streaming_lm_head_reference": "ace3/model/streaming_lm_head_reference.py",
    "fp16_adaptation_oracle": "ace3/model/fp16_adaptation_oracle.py",
    "streaming_lm_head_rtl": "ace3/rtl/ace3_streaming_tied_lm_head_topk.sv",
    "fp16_fixed_rtl": "ace3/rtl/ace3_fp16_fixed.sv",
    "q47_48_rounder_rtl": "ace3/rtl/ace3_q47_48_to_f16_rne.sv",
    "verilator_harness": "ace3/tb/ace3_streaming_tied_lm_head_topk_main.cpp",
    "focused_tests": "ace3/model/tests/test_position3_live_lm_head.py",
    "contract": "ace3/contracts/position3_live_lm_head.json",
    "makefile": "Makefile",
}
PARENT_VALIDATOR = ROOT / SOURCE_PATHS["position3_traversal_validator"]
RECHECK_CONDITION = (
    "recheck when build/model24_selected_token_position3/evidence.json exists "
    "with status COMPLETE and passes fresh validation of its exact position-3 "
    "selected token, 24 ordered current-worktree Verilator layers, layer-23 "
    "terminal hidden row, official checkpoint/tokenizer hashes, and "
    "source/artifact closure"
)

EvidenceError = base.EvidenceError
file_record = base.file_record
load_terminal_bits = base.load_terminal_bits
parse_rtl_receipt = base.parse_rtl_receipt
require = base.require
require_expected_hash = base.require_expected_hash
require_file_record = base.require_file_record
topk_from_bits = base.topk_from_bits

ParentValidator = Callable[[Path], None]


def load_json(path: Path) -> dict[str, Any]:
    return base.load_json(path)


def source_records(
    repository_root: Path = ROOT,
    paths: Mapping[str, str] = SOURCE_PATHS,
) -> dict[str, dict[str, Any]]:
    return base.source_records(repository_root, paths)


def require_source_bindings(
    stored: Mapping[str, Any],
    repository_root: Path = ROOT,
    paths: Mapping[str, str] = SOURCE_PATHS,
) -> None:
    base.require_source_bindings(stored, repository_root, paths)


def validate_parent_evidence(evidence_path: Path) -> None:
    document = load_json(evidence_path)
    preflight = document.get("authenticated_launch_preflight")
    require(
        isinstance(preflight, dict) and isinstance(preflight.get("path"), str),
        "parent traversal launch preflight binding is missing",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(PARENT_VALIDATOR),
            "validate",
            "--preflight",
            preflight["path"],
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
    parent_validator: ParentValidator | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    require(
        evidence_path.is_file(),
        f"position-3 traversal evidence is missing: {evidence_path}",
    )
    evidence_record = file_record(evidence_path)
    evidence = load_json(evidence_path)
    require(
        evidence.get("schema_version") == 1
        and evidence.get("kind")
        == "ace3_selected_token_position3_continuation_evidence",
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
    position3_input = evidence.get("position3_input")
    traversal = evidence.get("current_continuation_attempt")
    require(
        model
        == {
            "repository": lm_reference.MODEL_REPOSITORY,
            "revision": lm_reference.MODEL_REVISION,
            "checkpoint_sha256": lm_reference.CHECKPOINT_SHA256,
            "numeric_profile": NUMERIC_PROFILE,
        },
        "parent traversal model binding mismatch",
    )
    require(
        isinstance(position3_input, dict)
        and isinstance(traversal, dict)
        and isinstance(position3_input.get("selected_token_id"), int),
        "parent traversal position-3 input binding is missing",
    )
    selected_token_id = position3_input["selected_token_id"]
    traversal_history = [*PROMPT_TOKEN_HISTORY, selected_token_id]
    require(
        0 <= selected_token_id < VOCAB_SIZE
        and position3_input.get("position") == POSITION
        and position3_input.get("prompt_token_history") == PROMPT_TOKEN_HISTORY
        and position3_input.get("traversal_token_history") == traversal_history,
        "parent traversal selected-token or history binding mismatch",
    )
    layers = traversal.get("layers")
    require(
        traversal.get("status") == "COMPLETE"
        and traversal.get("execution")
        == "current-worktree compiled Verilator RTL"
        and traversal.get("operation")
        == "selected-token-position3-full-traversal"
        and traversal.get("selected_token_id") == selected_token_id
        and traversal.get("position") == POSITION
        and traversal.get("prompt_token_history") == PROMPT_TOKEN_HISTORY
        and traversal.get("traversal_token_history") == traversal_history
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
    output = layers[-1].get("output")
    post_layer23 = traversal.get("post_layer23")
    require(
        isinstance(output, dict)
        and isinstance(post_layer23, dict)
        and isinstance(output.get("path"), str)
        and post_layer23.get("natural_terminal") is True
        and post_layer23.get("independent_integer_oracle_match") is True
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
    terminal_hidden_sha256 = base.sha256_bytes(terminal_bits.tobytes())
    require(
        terminal_hidden_sha256 == output.get("semantic_sha256"),
        "parent terminal semantic SHA-256 mismatch",
    )
    return {
        "evidence": evidence_record,
        "status": "COMPLETE",
        "selected_token_id": selected_token_id,
        "position": POSITION,
        "prompt_token_history": PROMPT_TOKEN_HISTORY,
        "traversal_token_history": traversal_history,
        "natural_terminal_layers": LAYER_COUNT,
        "launch_preflight": evidence["authenticated_launch_preflight"],
        "layer23_terminal": terminal_record,
        "terminal_hidden_sha256": terminal_hidden_sha256,
    }, terminal_bits


def authenticate_tokenizer(
    tokenizer_dir: Path,
    tokenizer_sha256: str = TOKENIZER_SHA256,
    config_sha256: str = TOKENIZER_CONFIG_SHA256,
) -> dict[str, Any]:
    return {
        "repository": lm_reference.MODEL_REPOSITORY,
        "revision": lm_reference.MODEL_REVISION,
        "tokenizer": require_expected_hash(
            tokenizer_dir / "tokenizer.json",
            tokenizer_sha256,
            "official tokenizer.json",
        ),
        "tokenizer_config": require_expected_hash(
            tokenizer_dir / "tokenizer_config.json",
            config_sha256,
            "official tokenizer_config.json",
        ),
    }


def compute_oracle(
    checkpoint: Path,
    tokenizer_dir: Path,
    parent_evidence: Path = PARENT_EVIDENCE,
) -> dict[str, Any]:
    parent, terminal_bits = authenticate_parent_traversal(parent_evidence)
    tokenizer = authenticate_tokenizer(tokenizer_dir)
    records = lm_reference.tensor_records(checkpoint)
    normalized, final_norm = base.final_rmsnorm(
        checkpoint, records, terminal_bits
    )
    oracle, checks, winners = base.full_vocabulary_oracles(
        checkpoint, records, normalized
    )
    return {
        "parent": parent,
        "checkpoint": file_record(checkpoint),
        "tokenizer": tokenizer,
        "final_rmsnorm": final_norm,
        "oracle": oracle,
        "_normalized": normalized,
        "_checks": checks,
        "_winners": winners,
        "_records": records,
    }


def prepare(
    checkpoint: Path,
    tokenizer_dir: Path,
    output_dir: Path,
    parent_evidence: Path = PARENT_EVIDENCE,
) -> dict[str, Any]:
    require(not (output_dir / "oracle.json").exists(), "oracle output already exists")
    computed = compute_oracle(checkpoint, tokenizer_dir, parent_evidence)
    vectors = output_dir / "vectors"
    vectors.mkdir(parents=True, exist_ok=False)
    normalized = computed.pop("_normalized")
    checks = computed.pop("_checks")
    winners = computed.pop("_winners")
    records = computed.pop("_records")
    base.write_lines(
        vectors / "hidden.hex", (f"{int(raw):04x}" for raw in normalized)
    )
    base.write_lines(
        vectors / "checks.txt",
        (
            f"{token} {raw:04x} {accumulator & ((1 << 96) - 1):024x}"
            for token, (raw, accumulator) in sorted(checks.items())
        ),
    )
    base.write_lines(
        vectors / "topk.txt",
        (
            f"{rank} {token} {raw:04x} {value}"
            for rank, (token, raw, value) in enumerate(winners)
        ),
    )
    embedding = records["model.embed_tokens.weight"]
    base.write_lines(
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
    (output_dir / "oracle.json").write_bytes(base.canonical_json(computed))
    return computed


def artifact_records(output_dir: Path) -> dict[str, dict[str, Any]]:
    return base.artifact_records(output_dir)


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
        "kind": "ace3_position3_live_final_rmsnorm_streaming_lm_head_evidence",
        "status": "COMPLETE",
        "model": {
            "repository": lm_reference.MODEL_REPOSITORY,
            "revision": lm_reference.MODEL_REVISION,
            "checkpoint": computed["checkpoint"],
            "tokenizer": computed["tokenizer"],
            "geometry": {
                "hidden_size": HIDDEN_SIZE,
                "vocab_size": VOCAB_SIZE,
                "top_k": TOP_K,
            },
        },
        "input": {
            "token_id": computed["parent"]["selected_token_id"],
            "position": POSITION,
            "source": (
                "authenticated position-3 traversal layer-23 terminal hidden state"
            ),
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
                "the authenticated position-3 terminal hidden state through "
                "exact final RMSNorm and a live official-shape streaming tied "
                "lm_head/Top-K, with exact-integer and PyTorch full-vocabulary "
                "agreement"
            ),
            "position4_traversal": "not run",
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
    evidence_path.write_bytes(base.canonical_json(evidence))
    return evidence


def validate(
    output_dir: Path,
    binary: Path,
    evidence_path: Path,
    parent_evidence: Path = PARENT_EVIDENCE,
    tokenizer_dir: Path = TOKENIZER_DIR,
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
    computed = compute_oracle(CHECKPOINT, tokenizer_dir, parent_evidence)
    computed.pop("_normalized")
    computed.pop("_checks")
    computed.pop("_winners")
    computed.pop("_records")
    require(
        load_json(output_dir / "oracle.json") == computed,
        "stored full-vocabulary oracle is stale",
    )
    fresh = assemble_evidence(output_dir, binary, computed)
    require(stored == fresh, "stored position-3 lm_head evidence is stale")
    return stored


def print_summary(document: Mapping[str, Any], evidence_path: Path) -> None:
    print(
        "POSITION3_LIVE_LM_HEAD_PASS "
        f"input_token={document['input']['token_id']} "
        f"position={document['input']['position']} "
        f"final_rmsnorm_sha256={document['final_rmsnorm']['output_sha256']} "
        f"selected_next_token={document['selected_next_token']['token_id']} "
        f"vocab={document['model']['geometry']['vocab_size']} "
        "exact_pytorch_mismatches=0 rtl_topk_match=1 "
        "position4_traversal=not_run dialogue=not_claimed "
        f"evidence={evidence_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="operation", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    prepare_parser.add_argument("--tokenizer-dir", type=Path, default=TOKENIZER_DIR)
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
    validate_parser.add_argument("--tokenizer-dir", type=Path, default=TOKENIZER_DIR)
    args = parser.parse_args()
    try:
        if args.operation == "prepare":
            document = prepare(
                args.checkpoint.resolve(strict=True),
                args.tokenizer_dir.resolve(),
                args.output_dir.resolve(),
                args.parent_evidence.resolve(),
            )
            print(
                "POSITION3_LIVE_LM_HEAD_PREPARE_PASS "
                f"selected_next_token={document['oracle']['selected_token_id']} "
                "exact_pytorch_mismatches=0"
            )
            return
        output_dir = args.output_dir.resolve(strict=True)
        binary = args.binary.resolve(strict=True)
        evidence = (
            args.evidence.resolve()
            if args.operation == "seal"
            else args.evidence.resolve(strict=True)
        )
        document = (
            seal(output_dir, binary, evidence)
            if args.operation == "seal"
            else validate(
                output_dir,
                binary,
                evidence,
                args.parent_evidence.resolve(),
                args.tokenizer_dir.resolve(),
            )
        )
    except (EvidenceError, OSError, ValueError, KeyError) as error:
        raise SystemExit(
            f"POSITION3_LIVE_LM_HEAD_NOT_READY {error}; {RECHECK_CONDITION}"
        ) from error
    print_summary(document, evidence)


if __name__ == "__main__":
    main()
