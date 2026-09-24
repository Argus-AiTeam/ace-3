#!/usr/bin/env python3
"""Prepare a fail-closed position-4 traversal package from position-3 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
POSITION3_EVIDENCE = ROOT / "build/model24_selected_token_position3/evidence.json"
POSITION3_LM_HEAD_EVIDENCE = (
    ROOT / "build/model24_selected_token_position3_lm_head/evidence.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "build/model24_selected_token_position4_preparation/package"
)
MODEL_REPOSITORY = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
MODEL_REVISION = "db09cd27ead7fee40cdee309693cf83601b9c899"
CHECKPOINT_SHA256 = (
    "c50d807b7bed7ff314308972e0f4bcf4e5a70bc60ad88fc7df53940831ed0c1b"
)
TOKENIZER_SHA256 = (
    "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
)
TOKENIZER_CONFIG_SHA256 = (
    "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583"
)
NUMERIC_PROFILE = (
    "native asymmetric packed INT4 AWQ W4A16 G128, no qzero plus-one "
    "adjustment, FP16 activations and FP16 K/V"
)
PROMPT_TOKEN_HISTORY = [151644, 2114, 271]
POSITION3 = 3
POSITION4 = 4
LAYER_COUNT = 24
HIDDEN_SIZE = 896
VOCAB_SIZE = 151936
SOURCE_PATHS = {
    "position4_continuation_preparer": (
        "ace3/model/prepare_position4_continuation.py"
    ),
    "focused_tests": (
        "ace3/model/tests/test_prepare_position4_continuation.py"
    ),
    "contract": "ace3/contracts/position4_continuation_preparation.json",
    "position3_traversal_validator": (
        "ace3/model/validate_selected_token_position3_traversal.py"
    ),
    "position3_lm_head": "ace3/model/position3_live_lm_head.py",
    "makefile": "Makefile",
}
RECHECK_CONDITION = (
    "recheck when "
    "build/model24_selected_token_position3_lm_head/evidence.json exists with "
    "status COMPLETE and is bound to "
    "build/model24_selected_token_position3/evidence.json with status COMPLETE, "
    "and both authenticate the exact selected token, official "
    "checkpoint/tokenizer hashes, all 24 position-3 FP16 K/V states, and "
    "source/artifact closure"
)


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


def require_file_record(record: Any, label: str) -> dict[str, Any]:
    require(isinstance(record, dict), f"{label} record is malformed")
    require(isinstance(record.get("path"), str), f"{label} path is missing")
    path = Path(record["path"])
    require(path.is_file(), f"{label} is missing: {path}")
    actual = file_record(path)
    require(
        {name: record.get(name) for name in actual} == actual,
        f"{label} content binding mismatch",
    )
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
        authenticated[name] = require_file_record(record, f"{label} {name}")
    return authenticated


def authenticate_record_tree(value: Any, label: str) -> None:
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"}.issubset(value):
            require_file_record(value, label)
            return
        for name, child in value.items():
            authenticate_record_tree(child, f"{label} {name}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            authenticate_record_tree(child, f"{label} {index}")


def load_complete_position3_lm_head(
    evidence_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    require(
        evidence_path.is_file(),
        f"position-3 lm_head evidence is missing: {evidence_path}",
    )
    evidence_record = file_record(evidence_path)
    document = load_json(evidence_path)
    require(
        document.get("schema_version") == 1
        and document.get("kind")
        == "ace3_position3_live_final_rmsnorm_streaming_lm_head_evidence",
        "position-3 lm_head evidence identity mismatch",
    )
    require(
        document.get("status") == "COMPLETE",
        "position-3 lm_head evidence is not COMPLETE",
    )
    return evidence_record, document


def authenticate_position3_traversal(
    evidence_path: Path,
    expected_checkpoint_sha256: str = CHECKPOINT_SHA256,
) -> dict[str, Any]:
    require(
        evidence_path.is_file(),
        f"position-3 traversal evidence is missing: {evidence_path}",
    )
    evidence_record = file_record(evidence_path)
    document = load_json(evidence_path)
    require(
        document.get("schema_version") == 1
        and document.get("kind")
        == "ace3_selected_token_position3_continuation_evidence",
        "position-3 traversal evidence identity mismatch",
    )
    require(
        document.get("status") == "COMPLETE",
        "position-3 traversal evidence is not COMPLETE",
    )
    model = document.get("model")
    position3_input = document.get("position3_input")
    traversal = document.get("current_continuation_attempt")
    require(
        isinstance(model, dict)
        and model.get("repository") == MODEL_REPOSITORY
        and model.get("revision") == MODEL_REVISION
        and model.get("checkpoint_sha256") == expected_checkpoint_sha256
        and model.get("numeric_profile") == NUMERIC_PROFILE,
        "position-3 traversal model binding mismatch",
    )
    require(
        isinstance(position3_input, dict)
        and isinstance(traversal, dict)
        and isinstance(position3_input.get("selected_token_id"), int),
        "position-3 traversal input binding is missing",
    )
    selected_token_id = position3_input["selected_token_id"]
    traversal_history = [*PROMPT_TOKEN_HISTORY, selected_token_id]
    require(
        0 <= selected_token_id < VOCAB_SIZE
        and position3_input.get("position") == POSITION3
        and position3_input.get("prompt_token_history") == PROMPT_TOKEN_HISTORY
        and position3_input.get("traversal_token_history") == traversal_history,
        "position-3 selected-token or history binding mismatch",
    )
    layers = traversal.get("layers")
    require(
        traversal.get("status") == "COMPLETE"
        and traversal.get("execution")
        == "current-worktree compiled Verilator RTL"
        and traversal.get("operation")
        == "selected-token-position3-full-traversal"
        and traversal.get("selected_token_id") == selected_token_id
        and traversal.get("position") == POSITION3
        and traversal.get("prompt_token_history") == PROMPT_TOKEN_HISTORY
        and traversal.get("traversal_token_history") == traversal_history
        and traversal.get("layer_order") == list(range(LAYER_COUNT))
        and traversal.get("natural_terminal_layers") == LAYER_COUNT,
        "position-3 traversal completion binding mismatch",
    )
    require(
        isinstance(layers, list)
        and [layer.get("layer_index") for layer in layers]
        == list(range(LAYER_COUNT)),
        "position-3 layer ordering mismatch",
    )
    kv_parentage = []
    for layer_index, layer in enumerate(layers):
        predecessor = layer.get("predecessor_state")
        output_state = layer.get("output_state")
        output_hidden = layer.get("output")
        authenticated_parentage = layer.get("authenticated_kv_parentage")
        require(
            layer.get("position") == POSITION3
            and isinstance(predecessor, dict)
            and isinstance(output_state, dict)
            and isinstance(output_hidden, dict)
            and isinstance(authenticated_parentage, dict),
            f"position-3 layer {layer_index} K/V parentage is incomplete",
        )
        require(
            predecessor == authenticated_parentage.get("position2_output_state"),
            f"position-3 layer {layer_index} predecessor-state lineage mismatch",
        )
        authenticate_record_tree(layer, f"position-3 layer {layer_index}")
        kv_parentage.append(
            {
                "layer_index": layer_index,
                "parent_position": POSITION3,
                "next_position": POSITION4,
                "position2_predecessor_state": require_file_record(
                    predecessor,
                    f"position-3 layer {layer_index} predecessor state",
                ),
                "position3_output_state": require_file_record(
                    output_state,
                    f"position-3 layer {layer_index} output state",
                ),
                "position3_output_hidden": require_file_record(
                    output_hidden,
                    f"position-3 layer {layer_index} output hidden",
                ),
            }
        )
    post_layer23 = traversal.get("post_layer23")
    require(
        isinstance(post_layer23, dict)
        and post_layer23.get("natural_terminal") is True
        and post_layer23.get("independent_integer_oracle_match") is True
        and post_layer23.get("hidden_sha256")
        == layers[-1].get("output", {}).get("semantic_sha256"),
        "position-3 layer-23 terminal binding mismatch",
    )
    sources = authenticate_record_closure(
        document.get("consumed_sources"), "position-3 traversal sources"
    )
    require(
        file_record(evidence_path) == evidence_record,
        "position-3 traversal evidence changed during authentication",
    )
    return {
        "record": evidence_record,
        "model": dict(model),
        "selected_token_id": selected_token_id,
        "prompt_token_history": list(PROMPT_TOKEN_HISTORY),
        "traversal_token_history": traversal_history,
        "kv_parentage": kv_parentage,
        "source_bindings": sources,
    }


def authenticate_position3_lm_head(
    evidence_path: Path,
    evidence_record: Mapping[str, Any],
    document: Mapping[str, Any],
    parent: Mapping[str, Any],
    *,
    expected_checkpoint_sha256: str = CHECKPOINT_SHA256,
    expected_tokenizer_sha256: str = TOKENIZER_SHA256,
    expected_tokenizer_config_sha256: str = TOKENIZER_CONFIG_SHA256,
) -> dict[str, Any]:
    model = document.get("model")
    checkpoint = model.get("checkpoint") if isinstance(model, dict) else None
    tokenizer = model.get("tokenizer") if isinstance(model, dict) else None
    require(
        isinstance(model, dict)
        and model.get("repository") == MODEL_REPOSITORY
        and model.get("revision") == MODEL_REVISION
        and model.get("geometry", {}).get("hidden_size") == HIDDEN_SIZE
        and model.get("geometry", {}).get("vocab_size") == VOCAB_SIZE
        and isinstance(checkpoint, dict)
        and isinstance(tokenizer, dict),
        "position-3 lm_head model binding mismatch",
    )
    checkpoint_record = require_file_record(checkpoint, "official checkpoint")
    require(
        checkpoint_record["sha256"] == expected_checkpoint_sha256,
        "official checkpoint SHA-256 mismatch",
    )
    require(
        tokenizer.get("repository") == MODEL_REPOSITORY
        and tokenizer.get("revision") == MODEL_REVISION,
        "official tokenizer repository or revision mismatch",
    )
    tokenizer_record = require_file_record(
        tokenizer.get("tokenizer"), "official tokenizer.json"
    )
    tokenizer_config_record = require_file_record(
        tokenizer.get("tokenizer_config"), "official tokenizer_config.json"
    )
    require(
        tokenizer_record["sha256"] == expected_tokenizer_sha256,
        "official tokenizer.json SHA-256 mismatch",
    )
    require(
        tokenizer_config_record["sha256"] == expected_tokenizer_config_sha256,
        "official tokenizer_config.json SHA-256 mismatch",
    )
    require(
        Path(tokenizer_record["path"]).name == "tokenizer.json"
        and Path(tokenizer_config_record["path"]).name == "tokenizer_config.json"
        and Path(tokenizer_record["path"]).parent
        == Path(tokenizer_config_record["path"]).parent,
        "official tokenizer file closure mismatch",
    )
    lm_parent = document.get("parent_traversal")
    require(
        isinstance(lm_parent, dict)
        and lm_parent.get("evidence") == parent["record"]
        and lm_parent.get("status") == "COMPLETE"
        and lm_parent.get("selected_token_id") == parent["selected_token_id"]
        and lm_parent.get("position") == POSITION3
        and lm_parent.get("prompt_token_history")
        == parent["prompt_token_history"]
        and lm_parent.get("traversal_token_history")
        == parent["traversal_token_history"]
        and lm_parent.get("natural_terminal_layers") == LAYER_COUNT,
        "position-3 lm_head is not bound to the exact COMPLETE traversal evidence",
    )
    require(
        document.get("input")
        == {
            "token_id": parent["selected_token_id"],
            "position": POSITION3,
            "source": (
                "authenticated position-3 traversal layer-23 terminal hidden state"
            ),
        },
        "position-3 lm_head input binding mismatch",
    )
    selected = document.get("selected_next_token")
    oracle = document.get("full_vocabulary_oracles")
    rtl = document.get("official_shape_streaming_tied_lm_head")
    require(
        isinstance(selected, dict)
        and isinstance(oracle, dict)
        and isinstance(rtl, dict),
        "position-3 lm_head result is incomplete",
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
        "position-3 lm_head selected-token or full-vocabulary binding mismatch",
    )
    sources = authenticate_record_closure(
        document.get("consumed_sources"), "position-3 lm_head sources"
    )
    artifacts = authenticate_record_closure(
        document.get("artifacts"), "position-3 lm_head artifacts"
    )
    binary = require_file_record(rtl.get("binary"), "position-3 lm_head binary")
    terminal = require_file_record(
        rtl.get("terminal_log"), "position-3 lm_head terminal log"
    )
    require(
        file_record(evidence_path) == evidence_record,
        "position-3 lm_head evidence changed during authentication",
    )
    return {
        "record": dict(evidence_record),
        "checkpoint": checkpoint_record,
        "tokenizer": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "tokenizer": tokenizer_record,
            "tokenizer_config": tokenizer_config_record,
        },
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
    require(len(bits) == HIDDEN_SIZE, "position-4 embedding element count mismatch")
    rows = []
    for index, raw in enumerate(bits):
        require(
            isinstance(raw, int) and 0 <= raw <= 0xFFFF,
            f"position-4 embedding element {index} is not uint16",
        )
        rows.append(f"00{index:04x}{raw:04x}\n")
    return "".join(rows).encode("ascii")


def assemble_manifest(
    parent: Mapping[str, Any],
    lm_head: Mapping[str, Any],
    input_record: Mapping[str, Any],
    own_sources: Mapping[str, Any],
) -> dict[str, Any]:
    selected_token = lm_head["selected_token_id"]
    prompt_history = parent["traversal_token_history"]
    return {
        "schema_version": 1,
        "kind": "ace3_selected_token_position4_traversal_launch_package",
        "status": "READY",
        "model": parent["model"],
        "position4_input": {
            "position": POSITION4,
            "selected_token_id": selected_token,
            "selected_logit_f16_bits": lm_head["selected_logit_f16_bits"],
            "prompt_token_history": prompt_history,
            "traversal_token_history": [*prompt_history, selected_token],
            "embedding": {
                **input_record,
                "dtype": "FP16",
                "elements": HIDDEN_SIZE,
                "tensor": "model.embed_tokens.weight",
                "token_id": selected_token,
            },
        },
        "layer_kv_parentage": {
            "source_position": POSITION3,
            "target_position": POSITION4,
            "layer_order": list(range(LAYER_COUNT)),
            "layers": parent["kv_parentage"],
        },
        "parents": {
            "position3_traversal": parent["record"],
            "position3_lm_head": lm_head["record"],
        },
        "source_bindings": {
            "preparation": dict(own_sources),
            "position3_traversal": parent["source_bindings"],
            "position3_lm_head": lm_head["source_bindings"],
        },
        "consumed_artifacts": {
            "official_checkpoint": lm_head["checkpoint"],
            "official_tokenizer": lm_head["tokenizer"],
            "position3_lm_head_artifacts": lm_head["artifacts"],
            "position3_lm_head_binary": lm_head["binary"],
            "position3_lm_head_terminal_log": lm_head["terminal_log"],
            "position3_layer_states": [
                {
                    "layer_index": layer["layer_index"],
                    "position2_predecessor_state": layer[
                        "position2_predecessor_state"
                    ],
                    "position3_output_state": layer["position3_output_state"],
                    "position3_output_hidden": layer["position3_output_hidden"],
                }
                for layer in parent["kv_parentage"]
            ],
        },
        "launch": {
            "operation": "selected-token-position4-full-traversal",
            "input_position": POSITION4,
            "layer_order": list(range(LAYER_COUNT)),
            "required_embedding": input_record["path"],
            "required_fp16_kv_state_count": LAYER_COUNT,
            "execution_performed": False,
            "execution_authority": False,
        },
        "claim_boundary": {
            "demonstrated": (
                "authenticated preparation of the selected-token embedding and "
                "24-layer FP16 K/V parentage for a position-4 traversal"
            ),
            "position4_traversal": "not executed",
            "dialogue": "not claimed",
            "synthesis": "not run",
            "ppa": "not measured",
            "fpga": "not run",
            "latency": "not measured",
            "throughput": "not measured",
        },
    }


def not_ready_document(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "ace3_selected_token_position4_continuation_preparation_evidence",
        "status": "NOT_READY",
        "target_position": POSITION4,
        "reason": reason,
        "recheck_condition": RECHECK_CONDITION,
        "execution_performed": False,
        "claim_boundary": {
            "position4_traversal": "not executed",
            "dialogue": "not claimed",
            "synthesis": "not run",
            "ppa": "not measured",
            "fpga": "not run",
            "latency": "not measured",
            "throughput": "not measured",
        },
    }


def emit_not_ready(output_dir: Path, reason: str) -> None:
    if output_dir.exists():
        return
    output_dir.mkdir(parents=True)
    (output_dir / "not_ready.json").write_bytes(
        canonical_json(not_ready_document(reason))
    )


def prepare(
    position3_evidence: Path,
    lm_head_evidence: Path,
    output_dir: Path,
    *,
    embedding_provider=load_official_embedding,
    repository_root: Path = ROOT,
    own_source_paths: Mapping[str, str] = SOURCE_PATHS,
    expected_checkpoint_sha256: str = CHECKPOINT_SHA256,
    expected_tokenizer_sha256: str = TOKENIZER_SHA256,
    expected_tokenizer_config_sha256: str = TOKENIZER_CONFIG_SHA256,
) -> dict[str, Any]:
    try:
        require(
            not output_dir.exists(),
            f"preparation output already exists: {output_dir}",
        )
        evidence_record, lm_head_document = load_complete_position3_lm_head(
            lm_head_evidence
        )
        parent = authenticate_position3_traversal(
            position3_evidence, expected_checkpoint_sha256
        )
        lm_head = authenticate_position3_lm_head(
            lm_head_evidence,
            evidence_record,
            lm_head_document,
            parent,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            expected_tokenizer_sha256=expected_tokenizer_sha256,
            expected_tokenizer_config_sha256=(
                expected_tokenizer_config_sha256
            ),
        )
        own_sources = source_records(repository_root, own_source_paths)
        payload = embedding_payload(
            embedding_provider(
                Path(lm_head["checkpoint"]["path"]),
                lm_head["selected_token_id"],
            )
        )
        output_dir.mkdir(parents=True)
        input_path = output_dir / "position4_input.hex"
        input_path.write_bytes(payload)
        manifest = assemble_manifest(
            parent,
            lm_head,
            file_record(input_path),
            own_sources,
        )
        (output_dir / "launch_manifest.json").write_bytes(
            canonical_json(manifest)
        )
        return manifest
    except (PreparationError, OSError, ValueError, KeyError) as error:
        emit_not_ready(output_dir, str(error))
        raise


def validate(
    position3_evidence: Path,
    lm_head_evidence: Path,
    output_dir: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    manifest_path = output_dir / "launch_manifest.json"
    input_path = output_dir / "position4_input.hex"
    require(
        output_dir.is_dir()
        and {path.name for path in output_dir.iterdir()}
        == {"launch_manifest.json", "position4_input.hex"},
        "position-4 package artifact closure mismatch",
    )
    stored = load_json(manifest_path)
    evidence_record, lm_head_document = load_complete_position3_lm_head(
        lm_head_evidence
    )
    expected_checkpoint_sha256 = kwargs.get(
        "expected_checkpoint_sha256", CHECKPOINT_SHA256
    )
    parent = authenticate_position3_traversal(
        position3_evidence, expected_checkpoint_sha256
    )
    lm_head = authenticate_position3_lm_head(
        lm_head_evidence,
        evidence_record,
        lm_head_document,
        parent,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_tokenizer_sha256=kwargs.get(
            "expected_tokenizer_sha256", TOKENIZER_SHA256
        ),
        expected_tokenizer_config_sha256=kwargs.get(
            "expected_tokenizer_config_sha256", TOKENIZER_CONFIG_SHA256
        ),
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
        "position-4 embedding content binding mismatch",
    )
    own_sources = source_records(
        kwargs.get("repository_root", ROOT),
        kwargs.get("own_source_paths", SOURCE_PATHS),
    )
    fresh = assemble_manifest(
        parent,
        lm_head,
        file_record(input_path),
        own_sources,
    )
    require(stored == fresh, "stored position-4 launch manifest is stale")
    return stored


def print_summary(document: Mapping[str, Any], output_dir: Path) -> None:
    print(
        "POSITION4_CONTINUATION_PREPARATION_PASS "
        f"selected_token={document['position4_input']['selected_token_id']} "
        f"prompt_tokens={len(document['position4_input']['prompt_token_history'])} "
        f"history_tokens={len(document['position4_input']['traversal_token_history'])} "
        f"kv_parent_layers={len(document['layer_kv_parentage']['layers'])} "
        f"package={output_dir / 'launch_manifest.json'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("prepare", "validate"))
    parser.add_argument(
        "--position3-evidence", type=Path, default=POSITION3_EVIDENCE
    )
    parser.add_argument(
        "--lm-head-evidence", type=Path, default=POSITION3_LM_HEAD_EVIDENCE
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        if args.operation == "prepare":
            document = prepare(
                args.position3_evidence.resolve(),
                args.lm_head_evidence.resolve(),
                args.output_dir.resolve(),
            )
        else:
            document = validate(
                args.position3_evidence.resolve(),
                args.lm_head_evidence.resolve(),
                args.output_dir.resolve(strict=True),
            )
    except (PreparationError, OSError, ValueError, KeyError) as error:
        raise SystemExit(
            f"POSITION4_CONTINUATION_NOT_READY {error}; {RECHECK_CONDITION}"
        ) from error
    print_summary(document, args.output_dir.resolve())


if __name__ == "__main__":
    main()
