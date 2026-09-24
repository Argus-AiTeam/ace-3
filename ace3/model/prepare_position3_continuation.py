#!/usr/bin/env python3
"""Prepare a fail-closed position-3 traversal package from position-2 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
POSITION2_EVIDENCE = ROOT / "build/model24_selected_token_position2/evidence.json"
POSITION2_LM_HEAD_EVIDENCE = (
    ROOT / "build/model24_selected_token_position2_lm_head/evidence.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "build/model24_selected_token_position3_preparation/package"
)
POSITION2_VALIDATOR = (
    ROOT / "ace3/model/validate_selected_token_position2_traversal.py"
)
POSITION2_LM_HEAD_VALIDATOR = ROOT / "ace3/model/position2_live_lm_head.py"
MODEL_REPOSITORY = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
MODEL_REVISION = "db09cd27ead7fee40cdee309693cf83601b9c899"
CHECKPOINT_SHA256 = (
    "c50d807b7bed7ff314308972e0f4bcf4e5a70bc60ad88fc7df53940831ed0c1b"
)
POSITION0_TOKEN_ID = 151644
POSITION1_TOKEN_ID = 2114
POSITION2_TOKEN_ID = 271
POSITION3 = 3
LAYER_COUNT = 24
HIDDEN_SIZE = 896
VOCAB_SIZE = 151936
SOURCE_PATHS = {
    "position3_continuation_preparer": (
        "ace3/model/prepare_position3_continuation.py"
    ),
    "focused_tests": (
        "ace3/model/tests/test_prepare_position3_continuation.py"
    ),
    "contract": "ace3/contracts/position3_continuation_preparation.json",
    "position2_traversal_validator": (
        "ace3/model/validate_selected_token_position2_traversal.py"
    ),
    "position2_lm_head": "ace3/model/position2_live_lm_head.py",
    "makefile": "Makefile",
}
RECHECK_CONDITION = (
    "recheck when build/model24_selected_token_position2/evidence.json exists "
    "with status COMPLETE and "
    "build/model24_selected_token_position2_lm_head/evidence.json is freshly "
    "regenerated with status COMPLETE and bound to that exact traversal evidence"
)

Validator = Callable[[Path], None]
EmbeddingProvider = Callable[[Path, int], Sequence[int]]


class PreparationError(RuntimeError):
    """Raised when continuation inputs or package contents are not authentic."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreparationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while payload := stream.read(1024 * 1024):
            digest.update(payload)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    require(resolved.is_file(), f"regular file required: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def require_file_record(
    record: Mapping[str, Any], label: str
) -> dict[str, Any]:
    require(isinstance(record.get("path"), str), f"{label} path is missing")
    path = Path(record["path"])
    require(path.is_file(), f"{label} is missing: {path}")
    actual = file_record(path)
    require(actual == record, f"{label} content binding mismatch")
    return actual


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


def source_records(
    repository_root: Path = ROOT,
    paths: Mapping[str, str] = SOURCE_PATHS,
) -> dict[str, dict[str, Any]]:
    return {
        label: file_record(repository_root / relative_path)
        for label, relative_path in paths.items()
    }


def authenticate_record_closure(
    records: Any, label: str
) -> dict[str, dict[str, Any]]:
    require(isinstance(records, dict) and records, f"{label} closure is missing")
    authenticated: dict[str, dict[str, Any]] = {}
    for name, record in sorted(records.items()):
        require(isinstance(record, dict), f"{label} {name} record is malformed")
        authenticated[name] = require_file_record(record, f"{label} {name}")
    return authenticated


def run_position2_validator(evidence_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(POSITION2_VALIDATOR),
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
        f"position-2 traversal fresh validation failed: {output}",
    )


def run_lm_head_validator(evidence_path: Path) -> None:
    document = load_json(evidence_path)
    binary = document.get("official_shape_streaming_tied_lm_head", {}).get(
        "binary", {}
    )
    parent = document.get("parent_traversal", {}).get("evidence", {})
    require(
        isinstance(binary, dict) and isinstance(parent, dict),
        "position-2 lm_head validation inputs are missing",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(POSITION2_LM_HEAD_VALIDATOR),
            "validate",
            "--output-dir",
            str(evidence_path.parent),
            "--binary",
            str(binary.get("path", "")),
            "--parent-evidence",
            str(parent.get("path", "")),
            "--evidence",
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
        f"position-2 lm_head fresh validation failed: {output}",
    )


def authenticate_position2_traversal(
    evidence_path: Path,
    validator: Validator = run_position2_validator,
) -> dict[str, Any]:
    require(
        evidence_path.is_file(),
        f"position-2 traversal evidence is missing: {evidence_path}",
    )
    evidence_record = file_record(evidence_path)
    document = load_json(evidence_path)
    require(
        document.get("schema_version") == 2
        and document.get("kind")
        == "ace3_selected_token_position2_continuation_evidence",
        "position-2 traversal evidence identity mismatch",
    )
    require(
        document.get("status") == "COMPLETE",
        "position-2 traversal evidence is not COMPLETE",
    )
    model = document.get("model")
    selected = document.get("selected_token")
    traversal = document.get("current_continuation_attempt")
    require(
        isinstance(model, dict)
        and model.get("repository") == MODEL_REPOSITORY
        and model.get("revision") == MODEL_REVISION
        and model.get("checkpoint_sha256") == CHECKPOINT_SHA256,
        "position-2 traversal model binding mismatch",
    )
    require(
        isinstance(selected, dict)
        and selected.get("selected_token_id") == POSITION2_TOKEN_ID,
        "position-2 selected-token binding mismatch",
    )
    require(
        isinstance(traversal, dict)
        and traversal.get("status") == "COMPLETE"
        and traversal.get("position") == 2
        and traversal.get("selected_token_id") == POSITION2_TOKEN_ID
        and traversal.get("layer_order") == list(range(LAYER_COUNT))
        and traversal.get("natural_terminal_layers") == LAYER_COUNT,
        "position-2 traversal completion binding mismatch",
    )
    seeds = traversal.get("predecessor_seed_bindings")
    require(
        isinstance(seeds, dict)
        and seeds.get("position0", {}).get("token_id") == POSITION0_TOKEN_ID
        and seeds.get("position1", {}).get("token_id") == POSITION1_TOKEN_ID,
        "position-2 prompt-token parentage mismatch",
    )
    layers = traversal.get("layers")
    require(
        isinstance(layers, list)
        and [layer.get("layer_index") for layer in layers]
        == list(range(LAYER_COUNT)),
        "position-2 layer ordering mismatch",
    )
    kv_parentage = []
    for layer_index, layer in enumerate(layers):
        require(
            layer.get("position") == 2,
            f"position-2 layer {layer_index} position mismatch",
        )
        predecessor = layer.get("predecessor_state")
        output_state = layer.get("output_state")
        output_hidden = layer.get("output")
        require(
            isinstance(predecessor, dict)
            and isinstance(output_state, dict)
            and isinstance(output_hidden, dict),
            f"position-2 layer {layer_index} parentage is incomplete",
        )
        replay_parent = layer.get("predecessor_replay", {}).get(
            "position1", {}
        ).get("output_state")
        require(
            predecessor == replay_parent,
            f"position-2 layer {layer_index} predecessor-state lineage mismatch",
        )
        kv_parentage.append(
            {
                "layer_index": layer_index,
                "parent_position": 2,
                "next_position": POSITION3,
                "position1_predecessor_state": require_file_record(
                    predecessor,
                    f"position-2 layer {layer_index} predecessor state",
                ),
                "position2_output_state": require_file_record(
                    output_state,
                    f"position-2 layer {layer_index} output state",
                ),
                "position2_output_hidden": require_file_record(
                    {
                        "path": output_hidden.get("path"),
                        "bytes": output_hidden.get("bytes"),
                        "sha256": output_hidden.get("sha256"),
                    },
                    f"position-2 layer {layer_index} output hidden",
                ),
            }
        )
    sources = authenticate_record_closure(
        document.get("consumed_sources"), "position-2 traversal sources"
    )
    validator(evidence_path)
    require(
        file_record(evidence_path) == evidence_record,
        "position-2 traversal evidence changed during validation",
    )
    return {
        "record": evidence_record,
        "model": dict(model),
        "prompt_token_history": [
            POSITION0_TOKEN_ID,
            POSITION1_TOKEN_ID,
            POSITION2_TOKEN_ID,
        ],
        "kv_parentage": kv_parentage,
        "source_bindings": sources,
    }


def authenticate_position2_lm_head(
    evidence_path: Path,
    parent: Mapping[str, Any],
    validator: Validator = run_lm_head_validator,
    expected_checkpoint_sha256: str = CHECKPOINT_SHA256,
) -> dict[str, Any]:
    require(
        evidence_path.is_file(),
        f"fresh position-2 lm_head evidence is missing: {evidence_path}",
    )
    evidence_record = file_record(evidence_path)
    document = load_json(evidence_path)
    require(
        document.get("schema_version") == 1
        and document.get("kind")
        == "ace3_position2_live_final_rmsnorm_streaming_lm_head_evidence",
        "position-2 lm_head evidence identity mismatch",
    )
    require(
        document.get("status") == "COMPLETE",
        "position-2 lm_head evidence is not COMPLETE",
    )
    model = document.get("model")
    checkpoint = model.get("checkpoint") if isinstance(model, dict) else None
    require(
        isinstance(model, dict)
        and model.get("repository") == MODEL_REPOSITORY
        and model.get("revision") == MODEL_REVISION
        and model.get("geometry", {}).get("hidden_size") == HIDDEN_SIZE
        and model.get("geometry", {}).get("vocab_size") == VOCAB_SIZE
        and isinstance(checkpoint, dict),
        "position-2 lm_head model binding mismatch",
    )
    checkpoint_record = require_file_record(checkpoint, "official checkpoint")
    require(
        checkpoint_record["sha256"] == expected_checkpoint_sha256,
        "official checkpoint SHA-256 mismatch",
    )
    lm_parent = document.get("parent_traversal")
    require(
        isinstance(lm_parent, dict)
        and lm_parent.get("evidence") == parent["record"]
        and lm_parent.get("status") == "COMPLETE"
        and lm_parent.get("selected_token_id") == POSITION2_TOKEN_ID
        and lm_parent.get("position") == 2
        and lm_parent.get("natural_terminal_layers") == LAYER_COUNT,
        "position-2 lm_head is not bound to the exact COMPLETE traversal evidence",
    )
    require(
        document.get("input")
        == {
            "token_id": POSITION2_TOKEN_ID,
            "position": 2,
            "source": (
                "authenticated parent position-2 layer-23 terminal hidden state"
            ),
        },
        "position-2 lm_head input binding mismatch",
    )
    selected = document.get("selected_next_token")
    oracle = document.get("full_vocabulary_oracles")
    rtl = document.get("official_shape_streaming_tied_lm_head")
    require(
        isinstance(selected, dict)
        and isinstance(oracle, dict)
        and isinstance(rtl, dict),
        "position-2 lm_head result is incomplete",
    )
    token_id = selected.get("token_id")
    require(
        isinstance(token_id, int)
        and 0 <= token_id < VOCAB_SIZE
        and token_id == oracle.get("selected_token_id")
        and token_id == rtl.get("selected_token_id")
        and selected.get("exact_integer_pytorch_agreement") is True
        and selected.get("rtl_top_k_agreement") is True
        and oracle.get("agreement", {}).get(
            "all_rounded_fp16_logits_equal"
        )
        is True
        and oracle.get("agreement", {}).get("selected_token_equal") is True
        and oracle.get("agreement", {}).get("top_k_equal") is True
        and oracle.get("coverage", {}).get("vocab_size") == VOCAB_SIZE
        and oracle.get("coverage", {}).get("rounded_fp16_mismatches") == 0
        and rtl.get("natural_terminal") is True
        and rtl.get("accepted_logits") == VOCAB_SIZE,
        "position-2 lm_head selected-token or full-vocabulary binding mismatch",
    )
    sources = authenticate_record_closure(
        document.get("consumed_sources"), "position-2 lm_head sources"
    )
    artifacts = authenticate_record_closure(
        document.get("artifacts"), "position-2 lm_head artifacts"
    )
    binary = require_file_record(rtl.get("binary", {}), "position-2 lm_head binary")
    terminal = require_file_record(
        rtl.get("terminal_log", {}), "position-2 lm_head terminal log"
    )
    validator(evidence_path)
    require(
        file_record(evidence_path) == evidence_record,
        "position-2 lm_head evidence changed during validation",
    )
    return {
        "record": evidence_record,
        "checkpoint": checkpoint_record,
        "selected_token_id": token_id,
        "selected_logit_f16_bits": selected.get("logit_f16_bits"),
        "source_bindings": sources,
        "artifacts": artifacts,
        "binary": binary,
        "terminal_log": terminal,
    }


def load_official_embedding(checkpoint: Path, token_id: int) -> Sequence[int]:
    import numpy as np
    from safetensors import safe_open

    with safe_open(checkpoint, framework="np") as source:
        tensor = source.get_tensor("model.embed_tokens.weight")
        require(
            tensor.shape == (VOCAB_SIZE, HIDDEN_SIZE)
            and tensor.dtype == np.dtype("<f2"),
            "official tied embedding geometry or dtype mismatch",
        )
        return np.asarray(tensor[token_id], dtype="<f2").view("<u2").tolist()


def embedding_payload(bits: Sequence[int]) -> bytes:
    require(len(bits) == HIDDEN_SIZE, "position-3 embedding element count mismatch")
    rows = []
    for index, raw in enumerate(bits):
        require(
            isinstance(raw, int) and 0 <= raw <= 0xFFFF,
            f"position-3 embedding element {index} is not uint16",
        )
        rows.append(f"00{index:04x}{raw:04x}\n")
    return "".join(rows).encode("ascii")


def assemble_manifest(
    output_dir: Path,
    parent: Mapping[str, Any],
    lm_head: Mapping[str, Any],
    input_record: Mapping[str, Any],
    own_sources: Mapping[str, Any],
) -> dict[str, Any]:
    selected_token = lm_head["selected_token_id"]
    return {
        "schema_version": 1,
        "kind": "ace3_selected_token_position3_traversal_launch_package",
        "status": "READY",
        "model": parent["model"],
        "position3_input": {
            "position": POSITION3,
            "selected_token_id": selected_token,
            "selected_logit_f16_bits": lm_head["selected_logit_f16_bits"],
            "prompt_token_history": parent["prompt_token_history"],
            "traversal_token_history": [
                *parent["prompt_token_history"],
                selected_token,
            ],
            "embedding": {
                **input_record,
                "dtype": "FP16",
                "elements": HIDDEN_SIZE,
                "tensor": "model.embed_tokens.weight",
                "token_id": selected_token,
            },
        },
        "layer_kv_parentage": {
            "source_position": 2,
            "target_position": POSITION3,
            "layer_order": list(range(LAYER_COUNT)),
            "layers": parent["kv_parentage"],
        },
        "parents": {
            "position2_traversal": parent["record"],
            "position2_lm_head": lm_head["record"],
        },
        "source_bindings": {
            "preparation": dict(own_sources),
            "position2_traversal": parent["source_bindings"],
            "position2_lm_head": lm_head["source_bindings"],
        },
        "consumed_artifacts": {
            "official_checkpoint": lm_head["checkpoint"],
            "position2_lm_head_artifacts": lm_head["artifacts"],
            "position2_lm_head_binary": lm_head["binary"],
            "position2_lm_head_terminal_log": lm_head["terminal_log"],
            "position2_layer_states": [
                {
                    "layer_index": layer["layer_index"],
                    "position1_predecessor_state": layer[
                        "position1_predecessor_state"
                    ],
                    "position2_output_state": layer["position2_output_state"],
                    "position2_output_hidden": layer["position2_output_hidden"],
                }
                for layer in parent["kv_parentage"]
            ],
        },
        "launch": {
            "operation": "selected-token-position3-full-traversal",
            "input_position": POSITION3,
            "layer_order": list(range(LAYER_COUNT)),
            "required_embedding": input_record["path"],
            "required_fp16_kv_state_count": LAYER_COUNT,
            "execution_performed": False,
            "execution_authority": False,
        },
        "claim_boundary": {
            "demonstrated": (
                "authenticated preparation of token embedding and 24-layer FP16 "
                "K/V parentage for a position-3 traversal"
            ),
            "position3_traversal": "not executed",
            "dialogue": "not claimed",
            "synthesis": "not run",
            "ppa": "not measured",
            "fpga": "not run",
            "latency": "not measured",
            "throughput": "not measured",
        },
    }


def prepare(
    position2_evidence: Path,
    lm_head_evidence: Path,
    output_dir: Path,
    *,
    position2_validator: Validator = run_position2_validator,
    lm_head_validator: Validator = run_lm_head_validator,
    embedding_provider: EmbeddingProvider = load_official_embedding,
    repository_root: Path = ROOT,
    own_source_paths: Mapping[str, str] = SOURCE_PATHS,
    expected_checkpoint_sha256: str = CHECKPOINT_SHA256,
) -> dict[str, Any]:
    require(not output_dir.exists(), f"preparation output already exists: {output_dir}")
    parent = authenticate_position2_traversal(
        position2_evidence, position2_validator
    )
    lm_head = authenticate_position2_lm_head(
        lm_head_evidence,
        parent,
        lm_head_validator,
        expected_checkpoint_sha256,
    )
    own_sources = source_records(repository_root, own_source_paths)
    bits = embedding_provider(
        Path(lm_head["checkpoint"]["path"]), lm_head["selected_token_id"]
    )
    payload = embedding_payload(bits)
    output_dir.mkdir(parents=True)
    input_path = output_dir / "position3_input.hex"
    input_path.write_bytes(payload)
    manifest = assemble_manifest(
        output_dir,
        parent,
        lm_head,
        file_record(input_path),
        own_sources,
    )
    (output_dir / "launch_manifest.json").write_bytes(canonical_json(manifest))
    return manifest


def validate(
    position2_evidence: Path,
    lm_head_evidence: Path,
    output_dir: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    manifest_path = output_dir / "launch_manifest.json"
    input_path = output_dir / "position3_input.hex"
    require(
        output_dir.is_dir()
        and {path.name for path in output_dir.iterdir()}
        == {"launch_manifest.json", "position3_input.hex"},
        "position-3 package artifact closure mismatch",
    )
    stored = load_json(manifest_path)
    position2_validator = kwargs.get(
        "position2_validator", run_position2_validator
    )
    lm_head_validator = kwargs.get("lm_head_validator", run_lm_head_validator)
    expected_checkpoint_sha256 = kwargs.get(
        "expected_checkpoint_sha256", CHECKPOINT_SHA256
    )
    parent = authenticate_position2_traversal(
        position2_evidence, position2_validator
    )
    lm_head = authenticate_position2_lm_head(
        lm_head_evidence,
        parent,
        lm_head_validator,
        expected_checkpoint_sha256,
    )
    embedding_provider = kwargs.get(
        "embedding_provider", load_official_embedding
    )
    expected_payload = embedding_payload(
        embedding_provider(
            Path(lm_head["checkpoint"]["path"]), lm_head["selected_token_id"]
        )
    )
    require(
        input_path.read_bytes() == expected_payload,
        "position-3 embedding content binding mismatch",
    )
    own_sources = source_records(
        kwargs.get("repository_root", ROOT),
        kwargs.get("own_source_paths", SOURCE_PATHS),
    )
    fresh = assemble_manifest(
        output_dir,
        parent,
        lm_head,
        file_record(input_path),
        own_sources,
    )
    require(stored == fresh, "stored position-3 launch manifest is stale")
    return stored


def print_summary(document: Mapping[str, Any], output_dir: Path) -> None:
    print(
        "POSITION3_CONTINUATION_PREPARATION_PASS "
        f"selected_token={document['position3_input']['selected_token_id']} "
        f"prompt_tokens={len(document['position3_input']['prompt_token_history'])} "
        f"history_tokens={len(document['position3_input']['traversal_token_history'])} "
        f"kv_parent_layers={len(document['layer_kv_parentage']['layers'])} "
        f"package={output_dir / 'launch_manifest.json'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("prepare", "validate"))
    parser.add_argument(
        "--position2-evidence", type=Path, default=POSITION2_EVIDENCE
    )
    parser.add_argument(
        "--lm-head-evidence", type=Path, default=POSITION2_LM_HEAD_EVIDENCE
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        if args.operation == "prepare":
            document = prepare(
                args.position2_evidence.resolve(),
                args.lm_head_evidence.resolve(),
                args.output_dir.resolve(),
            )
        else:
            document = validate(
                args.position2_evidence.resolve(),
                args.lm_head_evidence.resolve(),
                args.output_dir.resolve(strict=True),
            )
    except (PreparationError, OSError, ValueError, KeyError) as error:
        raise SystemExit(
            f"POSITION3_CONTINUATION_NOT_READY {error}; {RECHECK_CONDITION}"
        ) from error
    print_summary(document, args.output_dir.resolve())


if __name__ == "__main__":
    main()
