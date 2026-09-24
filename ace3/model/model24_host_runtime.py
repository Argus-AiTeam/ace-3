#!/usr/bin/env python3
"""Authenticated prompt-driven host/runtime for the Model24 software executor."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from fp16_adaptation_oracle import decode_f16_q24
from model24_execution_oracle import (
    DEFAULT_OFFICIAL_CHECKPOINT,
    DEFAULT_OFFICIAL_TOKENIZER_DIR,
    FIXED_CHAT_MESSAGES,
    FIXED_CHAT_SERIALIZATION,
    FIXED_CHAT_TOKEN_IDS,
    OFFICIAL_TOP_K,
    TOKENIZER_CONFIG_SHA256,
    TOKENIZER_SHA256,
    authenticate_tokenizer,
)
from model24_oracle import (
    CHECKPOINT_SHA256,
    CHECKPOINT_SIZE,
    OFFICIAL_CONFIG,
    authenticate_checkpoint,
)
from official_model24_dialogue import (
    ARTIFACT_NAME as DIALOGUE_ARTIFACT_NAME,
    DialogueExecutionError,
    EVIDENCE_KIND as DIALOGUE_EVIDENCE_KIND,
    _authenticate_model24_binding,
    _binding_path,
    _canonical_json,
    _json_without_duplicates,
    _load_model,
    _sha256_bytes,
    create_binding_lineage,
    execute_loaded_prompt,
    official_tokenizer_binding,
    validate_binding_lineage,
    validate_document as validate_dialogue_document,
)
from official_model24_next_token import (
    LAYER_COUNT,
    LAYER_TENSOR_COUNT,
    MODEL_REPOSITORY,
    MODEL_REVISION,
)
from official_model24_showcase import _official_chat_serialization

ARTIFACT_NAME = "model24_host_runtime.json"
MANIFEST_NAME = "manifest.json"
EVIDENCE_KIND = "ace3_model24_host_runtime"
MANIFEST_KIND = "ace3_model24_host_runtime_manifest"
SELECTED_TOKEN_RECEIPT_KIND = "ace3_model24_selected_token_dialogue_receipt"
SELECTED_TOKEN_AUTHORITY_KIND = (
    "ace3_model24_selected_token_dialogue_receipt_authority"
)
NEXT_DIALOGUE_STEP_KIND = "ace3_model24_selected_token_dialogue_step"
SELECTED_TOKEN_CHAIN_KIND = "ace3_model24_selected_token_dialogue_chain"
SELECTED_TOKEN_EXPORT_KIND = (
    "ace3_model24_accepted_evidence_receipt_chain_export"
)
SELECTED_TOKEN_SOURCE_PROVENANCE_KIND = (
    "ace3_model24_accepted_evidence_provenance"
)
SELECTED_TOKEN_SOURCE_PARENT_KIND = (
    "ace3_model24_accepted_evidence_initial_parent"
)
SELECTED_TOKEN_SOURCE_TERMINAL_KIND = (
    "ace3_model24_accepted_evidence_terminal_step"
)
DIALOGUE_MANIFEST_KIND = (
    "ace3_official_model24_multitoken_dialogue_manifest"
)
SELECTED_TOKEN_POLICY = (
    "descending rounded finite FP16 numeric logit; "
    "equal logits use ascending token ID"
)
DEFAULT_MAX_NEW_TOKENS = 4
FOCUSED_EXPLICIT_PROMPT = "Reply with one short sentence about the moon."
SYSTEM_PROMPT = "You are a concise assistant."
TIED_HEAD_INTEGRATION_ARTIFACT_NAME = "integration.json"
TIED_HEAD_INTEGRATION_KIND = "ace3_model24_tokenizer_host_integration"
TIED_HEAD_INTEGRATION_MANIFEST_KIND = (
    "ace3_model24_tokenizer_host_integration_manifest"
)
TIED_HEAD_INTEGRATION_ATTEMPT_ID = (
    "model24_tokenizer_host_integration_attempt001"
)
TIED_HEAD_ATTEMPT_ID = "model24_tied_lm_head_topk_attempt002"
TIED_HEAD_SEAL_SHA256 = (
    "c701c116343609d84c894b9c14fbeca3d37075d527c9a28bcaccb196a8557190"
)
FINAL_RMSNORM_SEAL_SHA256 = (
    "f5cfac5693b75bb2a570ff2ce90c8fcfa68b3e5c8c86bb71f73b8c8618f78a67"
)
LAYER23_SEAL_SHA256 = (
    "2c7adde9827e16ddd9f495df502788aa2d3ef8a09fc307723281bf43a49c3646"
)
TIED_HEAD_SELECTED_TOKEN_ID = 0
TIED_HEAD_SELECTED_LOGIT_F16_BITS = 0x4C1D
MODEL24_SEED_TEXT = "Hello world"
MODEL24_SEED_TOKEN_IDS = (9707, 1879)
SOURCE_PATHS = (
    "ace3/model/model24_host_runtime.py",
    "ace3/model/official_model24_dialogue.py",
    "ace3/model/official_model24_showcase.py",
)


class HostRuntimeError(RuntimeError):
    """Raised when host/runtime execution or evidence validation fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HostRuntimeError(message)


def _reject_replay_markers(value: Any, context: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require(
                not (
                    isinstance(key, str)
                    and "replay" in key.casefold()
                ),
                f"{context} contains a replay marker",
            )
            _reject_replay_markers(item, context)
    elif isinstance(value, list):
        for item in value:
            _reject_replay_markers(item, context)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _source_bindings() -> list[dict[str, Any]]:
    root = _repository_root()
    return [
        {
            "path": relative_path,
            "bytes": (root / relative_path).stat().st_size,
            "sha256": _sha256_bytes((root / relative_path).read_bytes()),
        }
        for relative_path in SOURCE_PATHS
    ]


def _authenticate_assets(checkpoint_path: Path, tokenizer_dir: Path) -> Any:
    try:
        authenticate_checkpoint(checkpoint_path)
    except Exception as error:
        raise HostRuntimeError(
            f"official checkpoint authentication failed: {error}"
        ) from error
    try:
        return authenticate_tokenizer(tokenizer_dir)
    except Exception as error:
        raise HostRuntimeError(
            f"official tokenizer authentication failed: {error}"
        ) from error


def _prepare_prompt(
    tokenizer: Any,
    tokenizer_dir: Path,
    prompt_text: str | None,
) -> dict[str, Any]:
    if prompt_text is None:
        source = "default"
        input_text = FIXED_CHAT_MESSAGES[-1][1]
        messages = [
            {"role": role, "content": content}
            for role, content in FIXED_CHAT_MESSAGES
        ]
        serialization = FIXED_CHAT_SERIALIZATION
    else:
        _require(
            isinstance(prompt_text, str) and bool(prompt_text.strip()),
            "caller prompt must contain non-whitespace text",
        )
        source = "caller_provided"
        input_text = prompt_text
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]
        serialization = _official_chat_serialization(tokenizer_dir, messages)

    token_ids = tokenizer.encode(
        serialization,
        add_special_tokens=False,
    ).ids
    _require(bool(token_ids), "prompt tokenization produced no token IDs")
    if prompt_text is None:
        _require(
            token_ids == list(FIXED_CHAT_TOKEN_IDS),
            "default prompt token IDs mismatch",
        )
    return {
        "source": source,
        "input_text": input_text,
        "messages": messages,
        "serialization": serialization,
        "serialization_utf8_sha256": _sha256_bytes(
            serialization.encode("utf-8")
        ),
        "token_ids": token_ids,
        "decoded_roundtrip": tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
        ),
    }


def _model_binding(checkpoint_path: Path) -> dict[str, Any]:
    return {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "checkpoint": {
            "filename": checkpoint_path.name,
            "sha256": CHECKPOINT_SHA256,
            "bytes": CHECKPOINT_SIZE,
        },
        "accepted_model24_execution_binding": _authenticate_model24_binding(
            _binding_path()
        ),
        "authenticated_layer_count": LAYER_COUNT,
        "authenticated_layer_tensor_count": LAYER_COUNT * LAYER_TENSOR_COUNT,
        "tied_lm_head": "model.embed_tokens.weight",
    }


def _read_json_document(path: Path, context: str) -> dict[str, Any]:
    _require(path.is_file(), f"{context} is missing: {path}")
    document = _json_without_duplicates(path.read_bytes(), context)
    _require(isinstance(document, dict), f"{context} is not a JSON object")
    return document


def _verify_file_record(
    record: Mapping[str, Any],
    expected_path: Path,
    context: str,
) -> bytes:
    _require(isinstance(record, dict), f"{context} record is malformed")
    path = Path(str(record.get("path", "")))
    _require(
        path.is_absolute() and path.resolve() == expected_path.resolve(),
        f"{context} path mismatch",
    )
    _require(path.is_file(), f"{context} is missing: {path}")
    payload = path.read_bytes()
    _require(
        record.get("bytes") == len(payload)
        and record.get("sha256") == _sha256_bytes(payload),
        f"{context} authentication mismatch",
    )
    return payload


def _sealed_artifact_payload(
    root: Path,
    seal: Mapping[str, Any],
    relative_path: str,
    context: str,
) -> bytes:
    artifacts = seal.get("artifacts")
    _require(isinstance(artifacts, list), f"{context} artifact list is malformed")
    matches = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, dict)
        and artifact.get("relative_path") == relative_path
    ]
    _require(len(matches) == 1, f"{context} artifact is not uniquely sealed")
    return _verify_file_record(
        matches[0],
        root / relative_path,
        context,
    )


def _load_tied_head_lineage(head_attempt_dir: Path) -> dict[str, Any]:
    root = head_attempt_dir.resolve()
    _require(root.name == TIED_HEAD_ATTEMPT_ID, "tied-head attempt identity mismatch")
    seal_path = root / "sealed_output_manifest.json"
    seal_payload = seal_path.read_bytes() if seal_path.is_file() else b""
    _require(
        _sha256_bytes(seal_payload) == TIED_HEAD_SEAL_SHA256,
        "tied-head sealed receipt authentication mismatch",
    )
    seal = _json_without_duplicates(seal_payload, "tied-head sealed receipt")
    _require(
        seal.get("schema_version") == 1
        and seal.get("kind")
        == "ace3_tied_lm_head_topk_attempt002_sealed_manifest"
        and seal.get("attempt_id") == TIED_HEAD_ATTEMPT_ID
        and seal.get("official_attempt") == 2
        and seal.get("status") == "ENGINEERING_PASS_REVIEW_REQUIRED",
        "tied-head sealed receipt identity mismatch",
    )

    documents = {
        name: _json_without_duplicates(
            _sealed_artifact_payload(
                root,
                seal,
                relative_path,
                f"tied-head {name}",
            ),
            f"tied-head {name}",
        )
        for name, relative_path in {
            "status": "status.json",
            "adjudication": "adjudication.json",
            "comparison": "comparison.json",
            "oracle": "oracle/oracle.json",
            "inputs": "input_manifest.json",
            "preservation": "preservation.json",
        }.items()
    }
    status = documents["status"]
    adjudication = documents["adjudication"]
    comparison = documents["comparison"]
    oracle = documents["oracle"]
    inputs = documents["inputs"]
    preservation = documents["preservation"]
    _require(
        status.get("status") == "PASS"
        and status.get("attempt_id") == TIED_HEAD_ATTEMPT_ID
        and status.get("selected_token_id") == TIED_HEAD_SELECTED_TOKEN_ID
        and status.get("selected_logit_f16_bits") == "4c1d"
        and status.get("natural_terminal") is True
        and status.get("complete_streamed_vocabulary_coverage") is True,
        "tied-head selected-token status mismatch",
    )
    _require(
        adjudication.get("status") == "ENGINEERING_PASS_REVIEW_REQUIRED"
        and adjudication.get("attempt_id") == TIED_HEAD_ATTEMPT_ID
        and adjudication.get("review_requirement", {}).get("status") == "PENDING",
        "tied-head adjudication metadata mismatch",
    )
    _require(
        comparison.get("status") == "PASS"
        and comparison.get("failure_count") == 0
        and comparison.get("records_compared") == OFFICIAL_CONFIG["vocab_size"]
        and comparison.get("logit_bit_mismatch_count") == 0
        and comparison.get("accumulator_mismatch_count") == 0
        and comparison.get("top_k_mismatch_count") == 0
        and comparison.get("selected_token_id") == TIED_HEAD_SELECTED_TOKEN_ID
        and comparison.get("selected_logit_f16_bits") == "4c1d"
        and comparison.get("selected_token_logit_match") is True,
        "tied-head independent comparison mismatch",
    )
    _require(
        oracle.get("status") == "PASS"
        and oracle.get("selected_token_id") == TIED_HEAD_SELECTED_TOKEN_ID
        and oracle.get("selected_logit_f16_bits") == "4c1d",
        "tied-head oracle selection mismatch",
    )

    top_k: list[dict[str, int]] = []
    oracle_entries = oracle.get("top_k_entries")
    _require(
        isinstance(oracle_entries, list) and len(oracle_entries) == OFFICIAL_TOP_K,
        "tied-head oracle Top-K length mismatch",
    )
    for rank, entry in enumerate(oracle_entries):
        _require(isinstance(entry, dict), "tied-head oracle Top-K entry malformed")
        logit_bits = int(str(entry.get("logit_f16_bits", "")), 16)
        decoded_q24, finite, _, _ = decode_f16_q24(logit_bits)
        normalized = {
            "rank": rank,
            "token_id": entry.get("token_id"),
            "logit_f16_bits": logit_bits,
            "logit_q24": entry.get("logit_q24"),
        }
        _require(
            entry.get("rank") == rank
            and type(normalized["token_id"]) is int
            and finite
            and normalized["logit_q24"] == decoded_q24,
            "tied-head oracle Top-K payload mismatch",
        )
        top_k.append(normalized)
    _require(
        top_k
        == sorted(
            top_k,
            key=lambda entry: (-entry["logit_q24"], entry["token_id"]),
        ),
        "tied-head oracle Top-K ordering mismatch",
    )
    raw_top_k_payload = _sealed_artifact_payload(
        root,
        seal,
        "raw/topk.txt",
        "tied-head raw Top-K",
    )
    expected_raw_top_k = "".join(
        f"{entry['rank']} {entry['token_id']} "
        f"{entry['logit_f16_bits']:04x}\n"
        for entry in top_k
    ).encode("ascii")
    _require(
        raw_top_k_payload == expected_raw_top_k,
        "tied-head raw and independent Top-K payloads differ",
    )

    _require(
        inputs.get("attempt_id") == TIED_HEAD_ATTEMPT_ID
        and inputs.get("consumed_final_rmsnorm_token_index") == 1,
        "tied-head final-RMSNorm input lineage mismatch",
    )
    final_predecessor = inputs.get("predecessor")
    _require(
        isinstance(final_predecessor, dict),
        "final-RMSNorm predecessor lineage is malformed",
    )
    final_root = root.parent / "model24_final_rmsnorm_attempt001"
    final_seal_payload = _verify_file_record(
        final_predecessor.get("seal", {}),
        final_root / "sealed_output_manifest.json",
        "final-RMSNorm predecessor seal",
    )
    _require(
        _sha256_bytes(final_seal_payload) == FINAL_RMSNORM_SEAL_SHA256,
        "final-RMSNorm predecessor seal mismatch",
    )
    final_seal = _json_without_duplicates(
        final_seal_payload,
        "final-RMSNorm predecessor seal",
    )
    _require(
        final_seal.get("status") == "ENGINEERING_PASS_REVIEW_REQUIRED"
        and final_seal.get("kind")
        == "ace3_final_rmsnorm_attempt001_sealed_manifest"
        and final_seal.get("official_attempt") == 1,
        "final-RMSNorm predecessor identity mismatch",
    )
    final_status = _json_without_duplicates(
        _verify_file_record(
            final_predecessor.get("status", {}),
            final_root / "status.json",
            "final-RMSNorm predecessor status",
        ),
        "final-RMSNorm predecessor status",
    )
    _require(
        final_status.get("status") == "PASS"
        and final_status.get("natural_terminal") is True
        and final_status.get("integer_oracle_bit_exact") is True
        and final_status.get("fp16_policy_within_tolerance") is True,
        "final-RMSNorm predecessor status mismatch",
    )
    final_inputs = _json_without_duplicates(
        _sealed_artifact_payload(
            final_root,
            final_seal,
            "input_manifest.json",
            "final-RMSNorm input manifest",
        ),
        "final-RMSNorm input manifest",
    )
    layer23 = final_inputs.get("predecessor")
    _require(isinstance(layer23, dict), "layer-23 predecessor lineage is malformed")
    layer23_root = root.parent / "model24_layer23_attempt001"
    layer23_seal_payload = _verify_file_record(
        layer23.get("seal", {}),
        layer23_root / "sealed_output_manifest.json",
        "layer-23 predecessor seal",
    )
    _require(
        _sha256_bytes(layer23_seal_payload) == LAYER23_SEAL_SHA256,
        "layer-23 predecessor seal mismatch",
    )
    layer23_status = _json_without_duplicates(
        _verify_file_record(
            layer23.get("status", {}),
            layer23_root / "status.json",
            "layer-23 predecessor status",
        ),
        "layer-23 predecessor status",
    )
    _require(
        layer23_status.get("status") == "PASS"
        and layer23_status.get("layer_index") == 23
        and layer23_status.get("official_attempt_candidate") == 1,
        "layer-23 predecessor status mismatch",
    )
    _require(
        preservation.get("status") == "PASS"
        and preservation.get("predecessor_preserved") is True
        and preservation.get("before_ordered_file_set_sha256")
        == preservation.get("after_ordered_file_set_sha256")
        and preservation.get("prior_attempt001", {}).get("preserved") is True,
        "tied-head predecessor preservation mismatch",
    )
    return {
        "tied_head": {
            "attempt_id": TIED_HEAD_ATTEMPT_ID,
            "seal_sha256": TIED_HEAD_SEAL_SHA256,
            "sealed_status": seal["status"],
            "runtime_status": status["status"],
            "review_status": status["review_status"],
            "adjudication_review_status": adjudication["review_requirement"][
                "status"
            ],
        },
        "final_rmsnorm": {
            "attempt_id": "model24_final_rmsnorm_attempt001",
            "seal_sha256": FINAL_RMSNORM_SEAL_SHA256,
            "runtime_status": final_status["status"],
            "consumed_token_index": inputs[
                "consumed_final_rmsnorm_token_index"
            ],
        },
        "layer23": {
            "attempt_id": "model24_layer23_attempt001",
            "seal_sha256": LAYER23_SEAL_SHA256,
            "runtime_status": layer23_status["status"],
        },
        "predecessors_preserved": True,
        "top_k": top_k,
    }


def _tokenizer_host_binding(tokenizer: Any, tokenizer_dir: Path) -> dict[str, Any]:
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    config_path = tokenizer_dir / "tokenizer_config.json"
    tokenizer_payload = tokenizer_path.read_bytes()
    config_payload = config_path.read_bytes()
    _require(
        _sha256_bytes(tokenizer_payload) == TOKENIZER_SHA256
        and _sha256_bytes(config_payload) == TOKENIZER_CONFIG_SHA256,
        "official tokenizer asset binding mismatch",
    )
    messages = [
        {"role": role, "content": content}
        for role, content in FIXED_CHAT_MESSAGES
    ]
    serialization = _official_chat_serialization(tokenizer_dir, messages)
    chat_token_ids = tokenizer.encode(
        serialization,
        add_special_tokens=False,
    ).ids
    _require(
        serialization == FIXED_CHAT_SERIALIZATION
        and chat_token_ids == list(FIXED_CHAT_TOKEN_IDS),
        "official Qwen chat-template binding mismatch",
    )
    seed_token_ids = tokenizer.encode(
        MODEL24_SEED_TEXT,
        add_special_tokens=False,
    ).ids
    _require(
        seed_token_ids == list(MODEL24_SEED_TOKEN_IDS)
        and tokenizer.decode(seed_token_ids, skip_special_tokens=False)
        == MODEL24_SEED_TEXT,
        "accepted Model24 seed prompt/history mismatch",
    )
    return {
        "official_assets": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "tokenizer": {
                "filename": tokenizer_path.name,
                "bytes": len(tokenizer_payload),
                "sha256": TOKENIZER_SHA256,
            },
            "tokenizer_config": {
                "filename": config_path.name,
                "bytes": len(config_payload),
                "sha256": TOKENIZER_CONFIG_SHA256,
            },
        },
        "chat_template_probe": {
            "messages": messages,
            "serialization": serialization,
            "serialization_utf8_sha256": _sha256_bytes(
                serialization.encode("utf-8")
            ),
            "token_ids": chat_token_ids,
        },
        "seed_prompt": {
            "serialization_kind": "plain_text_seed_not_chat_template",
            "text": MODEL24_SEED_TEXT,
            "token_ids": seed_token_ids,
        },
    }


def create_tied_head_tokenizer_host_integration(
    head_attempt_dir: Path,
    tokenizer_dir: Path,
) -> dict[str, Any]:
    try:
        tokenizer = authenticate_tokenizer(tokenizer_dir)
    except Exception as error:
        raise HostRuntimeError(
            f"official tokenizer authentication failed: {error}"
        ) from error
    lineage = _load_tied_head_lineage(head_attempt_dir)
    tokenizer_binding = _tokenizer_host_binding(tokenizer, tokenizer_dir)
    top_k = lineage.pop("top_k")
    selected = top_k[0]
    _require(
        selected["token_id"] == TIED_HEAD_SELECTED_TOKEN_ID
        and selected["logit_f16_bits"] == TIED_HEAD_SELECTED_LOGIT_F16_BITS,
        "selected token/logit differs from tied-head receipt",
    )
    input_history = list(MODEL24_SEED_TOKEN_IDS)
    resulting_history = [*input_history, selected["token_id"]]
    decoded_token = tokenizer.decode(
        [selected["token_id"]],
        skip_special_tokens=False,
    )
    transcript = tokenizer.decode(
        resulting_history,
        skip_special_tokens=False,
    )
    _require(
        decoded_token == "!" and transcript == "Hello world!",
        "official tokenizer selected-token transcript mismatch",
    )
    return {
        "schema_version": 1,
        "kind": TIED_HEAD_INTEGRATION_KIND,
        "attempt_id": TIED_HEAD_INTEGRATION_ATTEMPT_ID,
        "source_lineage": lineage,
        "tokenizer_binding": tokenizer_binding,
        "selection": {
            "policy": SELECTED_TOKEN_POLICY,
            "selected_token_id": selected["token_id"],
            "selected_logit_f16_bits": selected["logit_f16_bits"],
            "top_k": top_k,
        },
        "host_result": {
            "input_token_history": input_history,
            "appended_token_id": selected["token_id"],
            "resulting_token_history": resulting_history,
            "decoded_token": decoded_token,
            "decoded_transcript_fragment": transcript,
        },
        "claim_boundary": {
            "demonstrated": (
                "authenticated current-lineage token-0 transfer from the "
                "accepted 24-layer/final-RMSNorm/tied-head RTL evidence into "
                "official tokenizer host history"
            ),
            "rtl_reexecution": "not run by this host integration",
            "persistent_kv_feedback": "not run",
            "generated_tokens": 1,
            "readable_fragment": transcript,
            "synthesis": "not run",
            "ppa": "not measured",
            "fpga": "not run",
        },
    }


def validate_tied_head_tokenizer_host_integration(
    document: Mapping[str, Any],
    head_attempt_dir: Path,
    tokenizer_dir: Path,
) -> dict[str, Any]:
    expected = create_tied_head_tokenizer_host_integration(
        head_attempt_dir,
        tokenizer_dir,
    )
    _require(
        document.get("schema_version") == 1
        and document.get("kind") == TIED_HEAD_INTEGRATION_KIND
        and document.get("attempt_id") == TIED_HEAD_INTEGRATION_ATTEMPT_ID,
        "tokenizer/host integration identity mismatch",
    )
    _require(
        document.get("source_lineage") == expected["source_lineage"],
        "tokenizer/host predecessor-lineage drift",
    )
    _require(
        document.get("tokenizer_binding") == expected["tokenizer_binding"],
        "tokenizer/chat-template binding drift",
    )
    actual_selection = document.get("selection")
    expected_selection = expected["selection"]
    _require(
        isinstance(actual_selection, dict)
        and actual_selection.get("selected_token_id")
        == TIED_HEAD_SELECTED_TOKEN_ID
        and actual_selection.get("selected_token_id")
        == expected_selection["selected_token_id"],
        "selected-token drift",
    )
    _require(
        actual_selection.get("selected_logit_f16_bits")
        == TIED_HEAD_SELECTED_LOGIT_F16_BITS
        and actual_selection.get("selected_logit_f16_bits")
        == expected_selection["selected_logit_f16_bits"],
        "selected-logit drift",
    )
    _require(
        actual_selection.get("policy") == SELECTED_TOKEN_POLICY
        and actual_selection.get("top_k") == expected_selection["top_k"],
        "deterministic Top-K payload/order drift",
    )
    _require(
        document.get("host_result") == expected["host_result"]
        and document.get("tokenizer_binding", {}).get("seed_prompt")
        == expected["tokenizer_binding"]["seed_prompt"],
        "prompt/token-history or decoded transcript drift",
    )
    _require(
        document.get("claim_boundary") == expected["claim_boundary"],
        "tokenizer/host claim boundary drift",
    )
    return {
        "attempt_id": TIED_HEAD_INTEGRATION_ATTEMPT_ID,
        "selected_token_id": TIED_HEAD_SELECTED_TOKEN_ID,
        "selected_logit_f16_bits": "0x4c1d",
        "top_k_entries": OFFICIAL_TOP_K,
        "resulting_token_history": expected["host_result"][
            "resulting_token_history"
        ],
        "decoded_transcript_fragment": expected["host_result"][
            "decoded_transcript_fragment"
        ],
    }


def generate_tied_head_tokenizer_host_integration(
    output_dir: Path,
    head_attempt_dir: Path,
    tokenizer_dir: Path,
) -> dict[str, bytes]:
    _require(
        not output_dir.exists() or not any(output_dir.iterdir()),
        "fresh tokenizer/host integration output directory is not empty",
    )
    document = create_tied_head_tokenizer_host_integration(
        head_attempt_dir,
        tokenizer_dir,
    )
    summary = validate_tied_head_tokenizer_host_integration(
        document,
        head_attempt_dir,
        tokenizer_dir,
    )
    evidence_payload = _canonical_json(document)
    manifest = {
        "schema_version": 1,
        "kind": TIED_HEAD_INTEGRATION_MANIFEST_KIND,
        "attempt_id": TIED_HEAD_INTEGRATION_ATTEMPT_ID,
        "artifacts": {
            TIED_HEAD_INTEGRATION_ARTIFACT_NAME: {
                "bytes": len(evidence_payload),
                "sha256": _sha256_bytes(evidence_payload),
            }
        },
        "summary": summary,
    }
    payloads = {
        TIED_HEAD_INTEGRATION_ARTIFACT_NAME: evidence_payload,
        MANIFEST_NAME: _canonical_json(manifest),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        (output_dir / name).write_bytes(payload)
    return payloads


def validate_tied_head_tokenizer_host_integration_directory(
    evidence_dir: Path,
    head_attempt_dir: Path,
    tokenizer_dir: Path,
) -> dict[str, Any]:
    _require(evidence_dir.is_dir(), "tokenizer/host integration directory missing")
    _require(
        {path.name for path in evidence_dir.iterdir() if path.is_file()}
        == {TIED_HEAD_INTEGRATION_ARTIFACT_NAME, MANIFEST_NAME},
        "tokenizer/host integration artifact set mismatch",
    )
    evidence_payload = (
        evidence_dir / TIED_HEAD_INTEGRATION_ARTIFACT_NAME
    ).read_bytes()
    manifest = _read_json_document(
        evidence_dir / MANIFEST_NAME,
        "tokenizer/host integration manifest",
    )
    _require(
        manifest.get("schema_version") == 1
        and manifest.get("kind") == TIED_HEAD_INTEGRATION_MANIFEST_KIND
        and manifest.get("attempt_id") == TIED_HEAD_INTEGRATION_ATTEMPT_ID,
        "tokenizer/host integration manifest identity mismatch",
    )
    artifact = manifest.get("artifacts", {}).get(
        TIED_HEAD_INTEGRATION_ARTIFACT_NAME,
        {},
    )
    _require(
        artifact.get("bytes") == len(evidence_payload)
        and artifact.get("sha256") == _sha256_bytes(evidence_payload),
        "tokenizer/host integration manifest authentication failed",
    )
    document = _json_without_duplicates(
        evidence_payload,
        TIED_HEAD_INTEGRATION_ARTIFACT_NAME,
    )
    summary = validate_tied_head_tokenizer_host_integration(
        document,
        head_attempt_dir,
        tokenizer_dir,
    )
    _require(
        manifest.get("summary") == summary,
        "tokenizer/host integration manifest summary mismatch",
    )
    return summary


def _claim_boundary() -> dict[str, str]:
    return {
        "demonstrated": (
            "one deterministic greedy multi-token software/oracle execution for "
            "the authenticated prompt, checkpoint, tokenizer, and Model24 executor"
        ),
        "broader_quality": "not assessed by this single prompt execution",
        "rtl": "full 24-layer host dialogue execution not demonstrated in RTL",
        "synthesis": "not run",
        "ppa": "not measured",
        "fpga": "not run",
        "latency": "not measured",
        "throughput": "not measured",
    }


def _receipt_model_binding() -> dict[str, Any]:
    return {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "checkpoint": {
            "filename": "model.safetensors",
            "sha256": CHECKPOINT_SHA256,
            "bytes": CHECKPOINT_SIZE,
        },
    }


def _receipt_prompt_lineage(prompt: Mapping[str, Any]) -> dict[str, Any]:
    serialization = prompt.get("serialization")
    token_ids = prompt.get("token_ids")
    _require(
        isinstance(serialization, str)
        and isinstance(token_ids, list)
        and bool(token_ids)
        and all(type(token_id) is int for token_id in token_ids),
        "expected prompt lineage is malformed",
    )
    serialization_sha256 = _sha256_bytes(serialization.encode("utf-8"))
    _require(
        prompt.get("serialization_utf8_sha256") == serialization_sha256,
        "expected prompt lineage is not authenticated",
    )
    return {
        "source": prompt.get("source"),
        "serialization_utf8_sha256": serialization_sha256,
        "token_ids": token_ids,
    }


def _accepted_evidence_spec(
    evidence_kind: object,
) -> tuple[str, str]:
    if evidence_kind == EVIDENCE_KIND:
        return ARTIFACT_NAME, MANIFEST_KIND
    if evidence_kind == DIALOGUE_EVIDENCE_KIND:
        return DIALOGUE_ARTIFACT_NAME, DIALOGUE_MANIFEST_KIND
    raise HostRuntimeError("accepted Model24 evidence kind mismatch")


def _accepted_source_provenance(
    evidence_payload: bytes,
    manifest_payload: bytes,
    evidence_name: str,
    evidence_kind: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": SELECTED_TOKEN_SOURCE_PROVENANCE_KIND,
        "manifest_name": MANIFEST_NAME,
        "manifest_sha256": _sha256_bytes(manifest_payload),
        "evidence_name": evidence_name,
        "evidence_sha256": _sha256_bytes(evidence_payload),
        "evidence_kind": evidence_kind,
    }


def _validate_accepted_asset_lineage(
    document: Mapping[str, Any],
) -> None:
    model = document.get("model_binding", {})
    _require(
        isinstance(model, dict)
        and {
            key: model.get(key)
            for key in ("repository", "revision", "checkpoint")
        }
        == _receipt_model_binding(),
        "accepted Model24 checkpoint lineage mismatch",
    )
    try:
        expected_execution_binding = _authenticate_model24_binding(
            _binding_path()
        )
    except DialogueExecutionError as error:
        raise HostRuntimeError(str(error)) from error
    _require(
        model.get("accepted_model24_execution_binding")
        == expected_execution_binding,
        "accepted Model24 execution lineage mismatch",
    )

    tokenizer_binding = document.get("tokenizer_binding", {})
    expected_tokenizer = official_tokenizer_binding()
    required_tokenizer_fields = {
        "tokenizer_artifact",
        "tokenizer_sha256",
        "config_artifact",
        "tokenizer_config_sha256",
    }
    _require(
        isinstance(tokenizer_binding, dict)
        and required_tokenizer_fields <= set(tokenizer_binding)
        and all(
            tokenizer_binding.get(field) == expected_tokenizer[field]
            for field in required_tokenizer_fields
        )
        and all(
            field not in tokenizer_binding
            or tokenizer_binding[field] == expected_tokenizer[field]
            for field in ("repository", "revision", "eos_token_id")
        ),
        "accepted Model24 tokenizer lineage mismatch",
    )
    if "binding_lineage" in document:
        try:
            validate_binding_lineage(document)
        except DialogueExecutionError as error:
            raise HostRuntimeError(str(error)) from error


def _validate_accepted_source_hashes(
    document: Mapping[str, Any],
    expected_source_bindings: list[Mapping[str, Any]] | None,
) -> None:
    if document.get("kind") != EVIDENCE_KIND:
        _require(
            expected_source_bindings is None,
            "accepted Model24 source bindings are not applicable",
        )
        return
    bindings = document.get("source_bindings")
    _require(
        isinstance(bindings, list)
        and all(isinstance(binding, dict) for binding in bindings)
        and [binding.get("path") for binding in bindings] == list(SOURCE_PATHS),
        "accepted Model24 source binding paths mismatch",
    )
    _require(
        all(
            isinstance(binding, dict)
            and set(binding) == {"path", "bytes", "sha256"}
            and type(binding["bytes"]) is int
            and binding["bytes"] > 0
            and isinstance(binding["sha256"], str)
            and len(binding["sha256"]) == 64
            and all(
                character in "0123456789abcdef"
                for character in binding["sha256"]
            )
            for binding in bindings
        ),
        "accepted Model24 source binding hash mismatch",
    )
    _require(
        isinstance(expected_source_bindings, list)
        and bindings == expected_source_bindings,
        "accepted Model24 source binding provenance mismatch",
    )


def _validate_accepted_evidence_payloads(
    evidence_payload: bytes,
    manifest_payload: bytes,
    tokenizer: Any,
    expected_source_provenance: Mapping[str, Any],
    expected_source_bindings: list[Mapping[str, Any]] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(
        isinstance(evidence_payload, bytes)
        and isinstance(manifest_payload, bytes),
        "accepted Model24 evidence payloads must be bytes",
    )
    document = _json_without_duplicates(
        evidence_payload,
        "accepted Model24 evidence",
    )
    _reject_replay_markers(document, "accepted Model24 evidence")
    _reject_replay_markers(
        expected_source_bindings,
        "accepted Model24 expected source bindings",
    )
    _validate_accepted_source_hashes(document, expected_source_bindings)
    evidence_name, manifest_kind = _accepted_evidence_spec(
        document.get("kind")
    )
    manifest = _json_without_duplicates(manifest_payload, MANIFEST_NAME)
    _reject_replay_markers(manifest, "accepted Model24 manifest")
    _reject_replay_markers(
        expected_source_provenance,
        "accepted Model24 source provenance",
    )
    _require(
        manifest.get("schema_version") == 1
        and manifest.get("kind") == manifest_kind,
        "accepted Model24 manifest identity mismatch",
    )
    artifacts = manifest.get("artifacts")
    _require(
        isinstance(artifacts, dict)
        and set(artifacts) == {evidence_name},
        "accepted Model24 manifest artifact set mismatch",
    )
    artifact = artifacts[evidence_name]
    _require(
        isinstance(artifact, dict)
        and set(artifact) == {"bytes", "sha256"}
        and artifact.get("bytes") == len(evidence_payload)
        and artifact.get("sha256") == _sha256_bytes(evidence_payload),
        "accepted Model24 evidence manifest authentication failed",
    )

    source_provenance = _accepted_source_provenance(
        evidence_payload,
        manifest_payload,
        evidence_name,
        document["kind"],
    )
    _require(
        isinstance(expected_source_provenance, dict)
        and source_provenance == expected_source_provenance,
        "accepted Model24 source provenance mismatch",
    )
    _validate_accepted_asset_lineage(document)
    try:
        summary = validate_dialogue_document(
            document,
            tokenizer,
            expected_kind=document["kind"],
            expected_prompt_serialization=document["prompt"]["serialization"],
            expected_prompt_token_ids=document["prompt"]["token_ids"],
        )
    except DialogueExecutionError as error:
        raise HostRuntimeError(str(error)) from error

    if document["kind"] == EVIDENCE_KIND:
        generation = document["generation"]
        expected_summary = {
            "prompt_source": document["prompt"]["source"],
            "prompt_tokens": len(document["prompt"]["token_ids"]),
            "generated_tokens": summary["steps"],
            "generated_token_ids": summary["generated_token_ids"],
            "decoded_text": summary["decoded_text"],
            "decoded_utf8_sha256": generation["decoded_utf8_sha256"],
            "stop_reason": summary["stop_reason"],
        }
    else:
        expected_summary = summary
    _require(
        manifest.get("summary") == expected_summary,
        "accepted Model24 manifest summary mismatch",
    )
    return document, source_provenance


def export_selected_token_receipt_chain_from_evidence(
    evidence_payload: bytes,
    manifest_payload: bytes,
    tokenizer: Any,
    expected_source_provenance: Mapping[str, Any],
    expected_source_bindings: list[Mapping[str, Any]] | None,
    receipt_authorities: list[Mapping[str, Any]],
    expected_authority_lineages: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Export receipts from authenticated existing evidence without execution."""

    _reject_replay_markers(
        receipt_authorities,
        "selected-token receipt authority chain",
    )
    _reject_replay_markers(
        expected_authority_lineages,
        "selected-token expected authority chain",
    )
    document, source_provenance = _validate_accepted_evidence_payloads(
        evidence_payload,
        manifest_payload,
        tokenizer,
        expected_source_provenance,
        expected_source_bindings,
    )
    prompt = document["prompt"]
    prompt_lineage = _receipt_prompt_lineage(prompt)
    steps = document["generation"]["steps"]
    _require(
        len(steps) >= 2
        and isinstance(receipt_authorities, list)
        and len(receipt_authorities) == len(steps)
        and isinstance(expected_authority_lineages, list)
        and len(expected_authority_lineages) == len(steps),
        "accepted Model24 receipt authority chain mismatch",
    )

    initial_parent = {
        "schema_version": 1,
        "kind": SELECTED_TOKEN_SOURCE_PARENT_KIND,
        "source_provenance": copy.deepcopy(source_provenance),
        "prompt_lineage": copy.deepcopy(prompt_lineage),
    }
    terminal_evidence_chain: list[dict[str, Any]] = [initial_parent]
    receipts: list[dict[str, Any]] = []
    token_history = list(prompt_lineage["token_ids"])
    for ordinal, step in enumerate(steps):
        _require(
            isinstance(step, dict) and step.get("ordinal") == ordinal,
            "accepted Model24 terminal step ordering mismatch",
        )
        token = step["token"]
        top_k = [
            {
                "rank": entry["rank"],
                "token_id": entry["token_id"],
                "logit_f16_bits": entry["logit_f16_bits"],
                "logit_q24": entry["logit_q24"],
            }
            for entry in token["top_k"]
        ]
        terminal_evidence = {
            "schema_version": 1,
            "kind": SELECTED_TOKEN_SOURCE_TERMINAL_KIND,
            "source_provenance": copy.deepcopy(source_provenance),
            "generation_ordinal": ordinal,
            "input_token_history": list(token_history),
            "source_step_sha256": _sha256_bytes(_canonical_json(step)),
            "terminal_hidden_sha256": step["terminal_hidden"]["sha256"],
            "logits_sha256": step["logits"]["sha256"],
            "selected_token_id": token["argmax_token_id"],
        }
        authority = receipt_authorities[ordinal]
        expected_lineage = expected_authority_lineages[ordinal]
        _require(
            isinstance(authority, dict)
            and set(authority)
            == {
                "kind",
                "lineage",
                "receipt_use_authorized",
                "authority_consumed",
            },
            "selected-token receipt authority identity mismatch",
        )
        receipt = {
            "schema_version": 1,
            "kind": SELECTED_TOKEN_RECEIPT_KIND,
            "model_binding": _receipt_model_binding(),
            "tokenizer_binding": official_tokenizer_binding(),
            "prompt_lineage": copy.deepcopy(prompt_lineage),
            "input_token_history": list(token_history),
            "parent_terminal_evidence": copy.deepcopy(
                terminal_evidence_chain[-1]
            ),
            "terminal_evidence": terminal_evidence,
            "authority": copy.deepcopy(authority),
            "selection": {
                "generation_ordinal": ordinal,
                "vocab_size": token["vocab_size"],
                "selection_policy": SELECTED_TOKEN_POLICY,
                "selected_token_id": token["argmax_token_id"],
                "selected_logit_f16_bits": top_k[0]["logit_f16_bits"],
                "top_k": top_k,
            },
        }
        validate_selected_token_receipt(
            receipt,
            prompt,
            terminal_evidence,
            expected_lineage,
        )
        receipts.append(receipt)
        terminal_evidence_chain.append(terminal_evidence)
        token_history.append(token["argmax_token_id"])

    dialogue = form_dialogue_from_receipt_chain(
        receipts,
        tokenizer,
        prompt,
        terminal_evidence_chain,
        expected_authority_lineages,
    )
    generation = document["generation"]
    _require(
        dialogue["generated_token_ids"] == generation["generated_token_ids"]
        and dialogue["resulting_token_history"] == token_history,
        "accepted Model24 generated token history mismatch",
    )
    decoded_transcript = tokenizer.decode(
        generation["decoded_token_ids"],
        skip_special_tokens=False,
    )
    _require(
        decoded_transcript == generation["decoded_text"]
        and _sha256_bytes(decoded_transcript.encode("utf-8"))
        == generation["decoded_utf8_sha256"],
        "accepted Model24 decoded transcript mismatch",
    )
    return {
        "schema_version": 1,
        "kind": SELECTED_TOKEN_EXPORT_KIND,
        "source_provenance": source_provenance,
        "receipts": receipts,
        "receipt_chain": dialogue,
        "accepted_generated_token_history": list(
            generation["generated_token_ids"]
        ),
        "decoded_transcript": decoded_transcript,
        "effect_boundary": copy.deepcopy(dialogue["effect_boundary"]),
    }


def validate_selected_token_receipt(
    receipt: Mapping[str, Any],
    expected_prompt: Mapping[str, Any],
    expected_terminal_evidence: Mapping[str, Any],
    expected_authority_lineage: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a selected-token receipt without invoking or mutating runtime."""

    _reject_replay_markers(receipt, "selected-token receipt")
    _reject_replay_markers(
        expected_authority_lineage,
        "selected-token expected authority lineage",
    )
    _require(
        receipt.get("schema_version") == 1
        and receipt.get("kind") == SELECTED_TOKEN_RECEIPT_KIND,
        "selected-token receipt identity mismatch",
    )
    _require(
        receipt.get("model_binding") == _receipt_model_binding(),
        "selected-token checkpoint lineage mismatch",
    )
    _require(
        receipt.get("tokenizer_binding") == official_tokenizer_binding(),
        "selected-token tokenizer lineage mismatch",
    )
    _require(
        receipt.get("prompt_lineage")
        == _receipt_prompt_lineage(expected_prompt),
        "selected-token prompt lineage mismatch",
    )
    terminal_evidence = receipt.get("terminal_evidence")
    _require(
        isinstance(terminal_evidence, dict)
        and terminal_evidence == expected_terminal_evidence,
        "selected-token terminal evidence lineage mismatch",
    )

    authority = receipt.get("authority")
    _require(
        isinstance(authority, dict)
        and authority.get("kind") == SELECTED_TOKEN_AUTHORITY_KIND,
        "selected-token receipt authority identity mismatch",
    )
    _require(
        authority.get("lineage") == expected_authority_lineage,
        "selected-token receipt authority lineage mismatch",
    )
    _require(
        authority.get("receipt_use_authorized") is True,
        "selected-token receipt use is not authorized",
    )
    _require(
        authority.get("authority_consumed") is False,
        "selected-token receipt authority is already consumed",
    )

    selection = receipt.get("selection")
    _require(
        isinstance(selection, dict)
        and type(selection.get("generation_ordinal")) is int
        and selection["generation_ordinal"] >= 0
        and selection.get("vocab_size") == OFFICIAL_CONFIG["vocab_size"]
        and selection.get("selection_policy") == SELECTED_TOKEN_POLICY,
        "selected-token selection metadata mismatch",
    )
    selected_token_id = selection.get("selected_token_id")
    selected_logit = selection.get("selected_logit_f16_bits")
    _require(
        type(selected_token_id) is int
        and 0 <= selected_token_id < OFFICIAL_CONFIG["vocab_size"],
        "selected token ID is outside the official vocabulary",
    )
    _require(
        type(selected_logit) is int and 0 <= selected_logit <= 0xFFFF,
        "selected logit is not a binary16 payload",
    )

    top_k = selection.get("top_k")
    _require(
        isinstance(top_k, list) and len(top_k) == OFFICIAL_TOP_K,
        "selected-token top-k payload length mismatch",
    )
    valid_top_k = True
    for rank, entry in enumerate(top_k):
        if not isinstance(entry, dict):
            valid_top_k = False
            break
        token_id = entry.get("token_id")
        logit_bits = entry.get("logit_f16_bits")
        logit_q24 = entry.get("logit_q24")
        if not (
            type(entry.get("rank")) is int
            and entry["rank"] == rank
            and type(token_id) is int
            and 0 <= token_id < OFFICIAL_CONFIG["vocab_size"]
            and type(logit_bits) is int
            and 0 <= logit_bits <= 0xFFFF
            and type(logit_q24) is int
        ):
            valid_top_k = False
            break
        decoded_q24, finite, _, _ = decode_f16_q24(logit_bits)
        if not finite or decoded_q24 != logit_q24:
            valid_top_k = False
            break
    _require(valid_top_k, "selected-token top-k payload mismatch")
    _require(
        len({entry["token_id"] for entry in top_k}) == OFFICIAL_TOP_K
        and top_k
        == sorted(
            top_k,
            key=lambda entry: (-entry["logit_q24"], entry["token_id"]),
        ),
        "selected-token top-k ordering mismatch",
    )
    _require(
        selected_token_id == top_k[0]["token_id"],
        "selected token ID does not match top-k rank zero",
    )
    _require(
        selected_logit == top_k[0]["logit_f16_bits"],
        "selected logit does not match top-k rank zero",
    )
    return {
        "generation_ordinal": selection["generation_ordinal"],
        "selected_token_id": selected_token_id,
        "selected_logit_f16_bits": selected_logit,
        "top_k": top_k,
    }


def form_next_dialogue_step_from_receipt(
    receipt: Mapping[str, Any],
    tokenizer: Any,
    expected_prompt: Mapping[str, Any],
    expected_terminal_evidence: Mapping[str, Any],
    expected_authority_lineage: Mapping[str, Any],
) -> dict[str, Any]:
    """Form the next token-history step from already-produced evidence."""

    selection = validate_selected_token_receipt(
        receipt,
        expected_prompt,
        expected_terminal_evidence,
        expected_authority_lineage,
    )
    token_id = selection["selected_token_id"]
    decoded_token = tokenizer.decode([token_id], skip_special_tokens=False)
    _require(
        isinstance(decoded_token, str),
        "official tokenizer selected-token decode did not return text",
    )
    prompt_lineage = _receipt_prompt_lineage(expected_prompt)
    return {
        "schema_version": 1,
        "kind": NEXT_DIALOGUE_STEP_KIND,
        "generation_ordinal": selection["generation_ordinal"],
        "prompt_lineage": prompt_lineage,
        "selected_token": {
            "token_id": token_id,
            "logit_f16_bits": selection["selected_logit_f16_bits"],
            "decoded_token": decoded_token,
        },
        "next_model_input_token_ids": [
            *prompt_lineage["token_ids"],
            token_id,
        ],
        "effect_boundary": {
            "runtime_workload_invoked": False,
            "controller_invoked": False,
            "model_or_rtl_invoked": False,
            "durable_submission_created": False,
            "authority_created": False,
            "authority_consumed": False,
        },
    }


def form_dialogue_from_receipt_chain(
    receipts: list[Mapping[str, Any]],
    tokenizer: Any,
    expected_prompt: Mapping[str, Any],
    expected_terminal_evidence_chain: list[Mapping[str, Any]],
    expected_authority_lineages: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate an ordered receipt chain and assemble its dialogue record."""

    _require(
        isinstance(receipts, list)
        and len(receipts) >= 2
        and all(isinstance(receipt, dict) for receipt in receipts),
        "selected-token receipt chain must contain at least two receipts",
    )
    _require(
        isinstance(expected_terminal_evidence_chain, list)
        and len(expected_terminal_evidence_chain) == len(receipts) + 1
        and all(
            isinstance(evidence, dict)
            for evidence in expected_terminal_evidence_chain
        ),
        "selected-token terminal evidence chain mismatch",
    )
    _require(
        isinstance(expected_authority_lineages, list)
        and len(expected_authority_lineages) == len(receipts)
        and all(
            isinstance(lineage, dict)
            for lineage in expected_authority_lineages
        ),
        "selected-token receipt authority chain mismatch",
    )

    prompt_lineage = _receipt_prompt_lineage(expected_prompt)
    token_history = list(prompt_lineage["token_ids"])
    generated_token_ids: list[int] = []
    selected_tokens: list[dict[str, Any]] = []

    for position, receipt in enumerate(receipts):
        selection_record = receipt.get("selection")
        _require(
            isinstance(selection_record, dict)
            and selection_record.get("generation_ordinal") == position,
            "selected-token receipt positions are not monotone and ungapped",
        )
        _require(
            receipt.get("prompt_lineage") == prompt_lineage,
            "selected-token prompt lineage mismatch",
        )
        _require(
            receipt.get("input_token_history") == token_history,
            "selected-token token history discontinuity",
        )
        _require(
            receipt.get("parent_terminal_evidence")
            == expected_terminal_evidence_chain[position],
            "selected-token terminal evidence parent mismatch",
        )

        selection = validate_selected_token_receipt(
            receipt,
            expected_prompt,
            expected_terminal_evidence_chain[position + 1],
            expected_authority_lineages[position],
        )
        token_id = selection["selected_token_id"]
        decoded_token = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
        )
        _require(
            isinstance(decoded_token, str),
            "official tokenizer selected-token decode did not return text",
        )
        generated_token_ids.append(token_id)
        token_history.append(token_id)
        selected_tokens.append(
            {
                "generation_ordinal": position,
                "token_id": token_id,
                "logit_f16_bits": selection["selected_logit_f16_bits"],
                "decoded_token": decoded_token,
            }
        )

    decoded_dialogue_transcript = tokenizer.decode(
        token_history,
        skip_special_tokens=False,
    )
    _require(
        isinstance(decoded_dialogue_transcript, str),
        "official tokenizer dialogue decode did not return text",
    )
    return {
        "schema_version": 1,
        "kind": SELECTED_TOKEN_CHAIN_KIND,
        "prompt_lineage": prompt_lineage,
        "receipt_count": len(receipts),
        "selected_tokens": selected_tokens,
        "generated_token_ids": generated_token_ids,
        "resulting_token_history": token_history,
        "decoded_dialogue_transcript": decoded_dialogue_transcript,
        "effect_boundary": {
            "runtime_workload_invoked": False,
            "lifecycle_invoked": False,
            "controller_invoked": False,
            "model_oracle_or_rtl_invoked": False,
            "durable_submission_created": False,
            "authority_created": False,
            "authority_consumed": False,
        },
    }


def execute_host_runtime(
    checkpoint_path: Path,
    tokenizer_dir: Path,
    prompt_text: str | None = None,
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> dict[str, Any]:
    _require(
        type(max_new_tokens) is int and max_new_tokens > 1,
        "max_new_tokens must be an integer greater than one",
    )
    tokenizer = _authenticate_assets(checkpoint_path, tokenizer_dir)
    prompt = _prepare_prompt(tokenizer, tokenizer_dir, prompt_text)

    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    embeddings, final_norm, reference_lm_head, states = _load_model(
        checkpoint_path
    )
    generation = execute_loaded_prompt(
        checkpoint_path,
        tokenizer,
        embeddings,
        final_norm,
        reference_lm_head,
        states,
        prompt["token_ids"],
        max_new_tokens=max_new_tokens,
    )
    document = {
        "schema_version": 1,
        "kind": EVIDENCE_KIND,
        "source_bindings": _source_bindings(),
        "model_binding": _model_binding(checkpoint_path),
        "tokenizer_binding": official_tokenizer_binding(),
        "runtime_profile": {
            "entry_point": "ace3/model/model24_host_runtime.py",
            "executor": "ace3/model/official_model24_dialogue.py",
            "selection": "deterministic greedy full-vocabulary argmax",
            "kv_cache": "incremental FP16 per layer",
        },
        "prompt": prompt,
        "generation": generation,
        "claim_boundary": _claim_boundary(),
    }
    document["binding_lineage"] = create_binding_lineage(document)
    return document


def validate_document(
    document: Mapping[str, Any],
    tokenizer: Any,
    tokenizer_dir: Path,
    expected_prompt_text: str | None,
) -> dict[str, Any]:
    _require(document.get("schema_version") == 1, "schema version mismatch")
    _require(document.get("kind") == EVIDENCE_KIND, "evidence kind mismatch")
    _require(
        document.get("source_bindings") == _source_bindings(),
        "host/runtime source binding mismatch",
    )
    model = document.get("model_binding", {})
    checkpoint = model.get("checkpoint", {})
    _require(
        model.get("repository") == MODEL_REPOSITORY
        and model.get("revision") == MODEL_REVISION
        and checkpoint.get("sha256") == CHECKPOINT_SHA256
        and checkpoint.get("bytes") == CHECKPOINT_SIZE,
        "checkpoint binding mismatch",
    )
    tokenizer_binding = document.get("tokenizer_binding", {})
    _require(
        tokenizer_binding.get("tokenizer_sha256") == TOKENIZER_SHA256
        and tokenizer_binding.get("tokenizer_config_sha256")
        == TOKENIZER_CONFIG_SHA256,
        "tokenizer binding mismatch",
    )
    try:
        validate_binding_lineage(document)
    except DialogueExecutionError as error:
        raise HostRuntimeError(str(error)) from error

    expected_prompt = _prepare_prompt(
        tokenizer,
        tokenizer_dir,
        expected_prompt_text,
    )
    actual_prompt = document.get("prompt", {})
    for name in (
        "source",
        "input_text",
        "messages",
        "serialization",
        "serialization_utf8_sha256",
        "token_ids",
        "decoded_roundtrip",
    ):
        _require(
            actual_prompt.get(name) == expected_prompt[name],
            f"prompt {name} mismatch",
        )

    summary = validate_dialogue_document(
        document,
        tokenizer,
        expected_kind=EVIDENCE_KIND,
        expected_prompt_serialization=expected_prompt["serialization"],
        expected_prompt_token_ids=expected_prompt["token_ids"],
    )
    generation = document["generation"]
    _require(
        len(generation["generated_token_ids"]) >= 2,
        "runtime did not produce multi-token output",
    )
    _require(
        isinstance(generation["decoded_text"], str)
        and bool(generation["decoded_text"].strip()),
        "runtime decoded output is empty",
    )
    _require(
        document.get("claim_boundary") == _claim_boundary(),
        "claim boundary mismatch",
    )
    return {
        "prompt_source": expected_prompt["source"],
        "prompt_tokens": len(expected_prompt["token_ids"]),
        "generated_tokens": len(generation["generated_token_ids"]),
        "generated_token_ids": generation["generated_token_ids"],
        "decoded_text": generation["decoded_text"],
        "decoded_utf8_sha256": generation["decoded_utf8_sha256"],
        "stop_reason": summary["stop_reason"],
    }


def generate(
    output_dir: Path,
    checkpoint_path: Path = DEFAULT_OFFICIAL_CHECKPOINT,
    tokenizer_dir: Path = DEFAULT_OFFICIAL_TOKENIZER_DIR,
    prompt_text: str | None = None,
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> dict[str, bytes]:
    document = execute_host_runtime(
        checkpoint_path,
        tokenizer_dir,
        prompt_text,
        max_new_tokens=max_new_tokens,
    )
    tokenizer = authenticate_tokenizer(tokenizer_dir)
    summary = validate_document(
        document,
        tokenizer,
        tokenizer_dir,
        prompt_text,
    )
    evidence_payload = _canonical_json(document)
    manifest = {
        "schema_version": 1,
        "kind": MANIFEST_KIND,
        "artifacts": {
            ARTIFACT_NAME: {
                "bytes": len(evidence_payload),
                "sha256": _sha256_bytes(evidence_payload),
            }
        },
        "summary": summary,
    }
    payloads = {
        ARTIFACT_NAME: evidence_payload,
        MANIFEST_NAME: _canonical_json(manifest),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in output_dir.iterdir()}
    _require(
        existing <= set(payloads),
        f"output directory contains unexpected files: {sorted(existing - set(payloads))}",
    )
    for name, payload in payloads.items():
        (output_dir / name).write_bytes(payload)
    return payloads


def validate_directory(
    evidence_dir: Path,
    checkpoint_path: Path = DEFAULT_OFFICIAL_CHECKPOINT,
    tokenizer_dir: Path = DEFAULT_OFFICIAL_TOKENIZER_DIR,
    expected_prompt_text: str | None = None,
) -> dict[str, Any]:
    _require(
        evidence_dir.is_dir(),
        f"evidence directory is missing: {evidence_dir}",
    )
    actual_names = {path.name for path in evidence_dir.iterdir() if path.is_file()}
    _require(
        actual_names == {ARTIFACT_NAME, MANIFEST_NAME},
        "evidence artifact set mismatch",
    )
    tokenizer = _authenticate_assets(checkpoint_path, tokenizer_dir)
    evidence_payload = (evidence_dir / ARTIFACT_NAME).read_bytes()
    manifest = _json_without_duplicates(
        (evidence_dir / MANIFEST_NAME).read_bytes(),
        MANIFEST_NAME,
    )
    _require(manifest.get("kind") == MANIFEST_KIND, "manifest kind mismatch")
    artifact = manifest.get("artifacts", {}).get(ARTIFACT_NAME, {})
    _require(
        artifact.get("bytes") == len(evidence_payload)
        and artifact.get("sha256") == _sha256_bytes(evidence_payload),
        "evidence manifest authentication failed",
    )
    document = _json_without_duplicates(evidence_payload, ARTIFACT_NAME)
    summary = validate_document(
        document,
        tokenizer,
        tokenizer_dir,
        expected_prompt_text,
    )
    _require(summary == manifest.get("summary"), "manifest summary mismatch")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "operation",
        choices=(
            "generate",
            "validate",
            "integrate-tied-head",
            "validate-tied-head-integration",
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument(
        "--tied-head-attempt-dir",
        type=Path,
        default=(
            _repository_root()
            / "build"
            / TIED_HEAD_ATTEMPT_ID
        ),
    )
    parser.add_argument(
        "--official-checkpoint",
        type=Path,
        default=DEFAULT_OFFICIAL_CHECKPOINT,
    )
    parser.add_argument(
        "--official-tokenizer-dir",
        type=Path,
        default=DEFAULT_OFFICIAL_TOKENIZER_DIR,
    )
    parser.add_argument("--prompt")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.operation == "integrate-tied-head":
            _require(args.output_dir is not None, "--output-dir is required")
            payloads = generate_tied_head_tokenizer_host_integration(
                args.output_dir.resolve(),
                args.tied_head_attempt_dir.resolve(),
                args.official_tokenizer_dir.resolve(),
            )
            document = json.loads(
                payloads[TIED_HEAD_INTEGRATION_ARTIFACT_NAME]
            )
            result = document["host_result"]
            print(
                "MODEL24_TOKENIZER_HOST_INTEGRATION_PASS "
                f"attempt={document['attempt_id']} "
                f"selected_token={result['appended_token_id']} "
                "selected_logit=0x4c1d "
                f"history={result['resulting_token_history']} "
                f"decoded={result['decoded_transcript_fragment']!r}"
            )
        elif args.operation == "validate-tied-head-integration":
            _require(args.evidence_dir is not None, "--evidence-dir is required")
            summary = validate_tied_head_tokenizer_host_integration_directory(
                args.evidence_dir.resolve(),
                args.tied_head_attempt_dir.resolve(),
                args.official_tokenizer_dir.resolve(),
            )
            print(
                "MODEL24_TOKENIZER_HOST_INTEGRATION_VALIDATION_PASS "
                f"attempt={summary['attempt_id']} "
                f"selected_token={summary['selected_token_id']} "
                f"selected_logit={summary['selected_logit_f16_bits']} "
                f"history={summary['resulting_token_history']} "
                f"decoded={summary['decoded_transcript_fragment']!r}"
            )
        elif args.operation == "generate":
            _require(args.output_dir is not None, "--output-dir is required")
            payloads = generate(
                args.output_dir.resolve(),
                args.official_checkpoint.resolve(),
                args.official_tokenizer_dir.resolve(),
                args.prompt,
                max_new_tokens=args.max_new_tokens,
            )
            document = json.loads(payloads[ARTIFACT_NAME])
            summary = document["generation"]
            print(
                "MODEL24_HOST_RUNTIME_GENERATION_PASS "
                f"prompt_source={document['prompt']['source']} "
                f"prompt_tokens={len(document['prompt']['token_ids'])} "
                f"generated_tokens={len(summary['generated_token_ids'])} "
                f"token_ids={summary['generated_token_ids']} "
                f"decoded_text={summary['decoded_text']!r}"
            )
        else:
            _require(args.evidence_dir is not None, "--evidence-dir is required")
            summary = validate_directory(
                args.evidence_dir.resolve(),
                args.official_checkpoint.resolve(),
                args.official_tokenizer_dir.resolve(),
                args.prompt,
            )
            print(
                "MODEL24_HOST_RUNTIME_VALIDATION_PASS "
                f"prompt_source={summary['prompt_source']} "
                f"prompt_tokens={summary['prompt_tokens']} "
                f"generated_tokens={summary['generated_tokens']} "
                f"token_ids={summary['generated_token_ids']} "
                f"decoded_text={summary['decoded_text']!r}"
            )
    except (
        DialogueExecutionError,
        HostRuntimeError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise SystemExit(f"MODEL24_HOST_RUNTIME_FAIL {error}") from error


if __name__ == "__main__":
    main()
