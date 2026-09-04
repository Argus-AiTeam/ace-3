#!/usr/bin/env python3
"""Authenticated prompt-driven host/runtime for the Model24 software executor."""

from __future__ import annotations

import argparse
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
    DialogueExecutionError,
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
SELECTED_TOKEN_POLICY = (
    "descending rounded finite FP16 numeric logit; "
    "equal logits use ascending token ID"
)
DEFAULT_MAX_NEW_TOKENS = 4
FOCUSED_EXPLICIT_PROMPT = "Reply with one short sentence about the moon."
SYSTEM_PROMPT = "You are a concise assistant."
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


def validate_selected_token_receipt(
    receipt: Mapping[str, Any],
    expected_prompt: Mapping[str, Any],
    expected_terminal_evidence: Mapping[str, Any],
    expected_authority_lineage: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a selected-token receipt without invoking or mutating runtime."""

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
    parser.add_argument("operation", choices=("generate", "validate"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
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
        if args.operation == "generate":
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
