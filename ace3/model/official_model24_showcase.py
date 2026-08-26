#!/usr/bin/env python3
"""Transparent official-checkpoint continuation and chat showcase."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping

import torch

from model24_execution_oracle import (
    DEFAULT_OFFICIAL_CHECKPOINT,
    DEFAULT_OFFICIAL_TOKENIZER_DIR,
    EOS_TOKEN_ID,
    FIXED_CHAT_MESSAGES,
    FIXED_CHAT_SERIALIZATION,
    TOKENIZER_CONFIG_SHA256,
    TOKENIZER_SHA256,
    authenticate_tokenizer,
)
from model24_oracle import (
    CHECKPOINT_SHA256,
    CHECKPOINT_SIZE,
    authenticate_checkpoint,
)
from official_model24_dialogue import (
    LOGITS_ABSOLUTE_TOLERANCE,
    MODEL24_BINDING_SHA256,
    DialogueExecutionError,
    _authenticate_model24_binding,
    _binding_path,
    _canonical_json,
    _load_model,
    _reset_kv_caches,
    _sha256_bytes,
    execute_loaded_prompt,
    validate_document,
)
from official_model24_next_token import (
    LAYER_COUNT,
    LAYER_TENSOR_COUNT,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE,
)

ARTIFACT_NAME = "official_model24_showcase.json"
MANIFEST_NAME = "manifest.json"
MARKDOWN_NAME = "SHOWCASE.md"
EVIDENCE_KIND = "ace3_official_model24_continuation_showcase"
DEFAULT_MAX_NEW_TOKENS = 6

PROMPT_SPECS = (
    {
        "id": "english_continuation_i_am",
        "mode": "continuation",
        "input_text": "I am",
        "messages": None,
    },
    {
        "id": "english_continuation_france",
        "mode": "continuation",
        "input_text": "The capital of France is",
        "messages": None,
    },
    {
        "id": "english_chat_request",
        "mode": "chat",
        "input_text": "Reply with one short sentence about the moon.",
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {
                "role": "user",
                "content": "Reply with one short sentence about the moon.",
            },
        ],
    },
    {
        "id": "chinese_continuation",
        "mode": "continuation",
        "input_text": "\u4e2d\u56fd\u7684\u9996\u90fd\u662f",
        "messages": None,
    },
    {
        "id": "chinese_chat_request",
        "mode": "chat",
        "input_text": "\u8bf7\u7528\u4e00\u53e5\u7b80\u77ed\u7684\u4e2d\u6587\u95ee\u5019\u6211\u3002",
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {
                "role": "user",
                "content": "\u8bf7\u7528\u4e00\u53e5\u7b80\u77ed\u7684\u4e2d\u6587\u95ee\u5019\u6211\u3002",
            },
        ],
    },
    {
        "id": "python_code_completion",
        "mode": "continuation",
        "input_text": "def add(a, b):\n    return",
        "messages": None,
    },
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DialogueExecutionError(message)


def _official_chat_serialization(
    tokenizer_dir: Path,
    messages: list[dict[str, str]],
) -> str:
    config_payload = (tokenizer_dir / "tokenizer_config.json").read_bytes()
    _require(
        _sha256_bytes(config_payload) == TOKENIZER_CONFIG_SHA256,
        "official tokenizer config SHA256 mismatch",
    )
    config = json.loads(config_payload)
    chat_template = config.get("chat_template")
    _require(
        isinstance(chat_template, str) and bool(chat_template),
        "official chat template is missing",
    )
    try:
        from transformers import PreTrainedTokenizerFast
    except ImportError as error:
        raise DialogueExecutionError(
            "transformers is required to apply the official chat template"
        ) from error
    template_host = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_dir / "tokenizer.json")
    )
    template_host.chat_template = chat_template
    serialization = template_host.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    fixed_messages = [
        {"role": role, "content": content}
        for role, content in FIXED_CHAT_MESSAGES
    ]
    _require(
        template_host.apply_chat_template(
            fixed_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        == FIXED_CHAT_SERIALIZATION,
        "official chat template fixed-prompt authentication failed",
    )
    return serialization


def _serialize_prompt(
    spec: Mapping[str, Any],
    tokenizer_dir: Path,
) -> str:
    if spec["mode"] == "continuation":
        return spec["input_text"]
    _require(spec["mode"] == "chat", f"unsupported prompt mode {spec['mode']!r}")
    return _official_chat_serialization(tokenizer_dir, spec["messages"])


def _model_binding() -> dict[str, Any]:
    return {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "checkpoint": {
            "filename": DEFAULT_OFFICIAL_CHECKPOINT.name,
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
            "six deterministic software/oracle continuations using the official "
            "checkpoint, tokenizer, chat template, 24-layer executor, tied "
            "full-vocabulary head, and incremental FP16 KV"
        ),
        "broader_quality": "not assessed beyond the six preserved prompts",
        "rtl": "full 24-layer showcase execution not demonstrated in RTL",
        "synthesis": "not run",
        "ppa": "not measured",
        "fpga": "not run",
        "latency": "not measured",
        "throughput": "not measured",
    }


def _comparison_failures(
    prompt_id: str,
    generation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    failures = []
    for step in generation["steps"]:
        for metric, comparison in (
            (
                "terminal_hidden",
                step["terminal_hidden"]["independent_reference"],
            ),
            ("logits", step["logits"]["independent_reference"]),
        ):
            if not comparison["within_tolerance"]:
                failures.append(
                    {
                        "prompt_id": prompt_id,
                        "step": step["ordinal"],
                        "type": "comparison_tolerance_exceeded",
                        "metric": metric,
                        "max_abs_error": comparison["max_abs_error"],
                        "absolute_tolerance": comparison["absolute_tolerance"],
                    }
                )
    return failures


def execute_showcase(
    checkpoint_path: Path,
    tokenizer_dir: Path,
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> dict[str, Any]:
    _require(
        type(max_new_tokens) is int and max_new_tokens > 1,
        "showcase max_new_tokens must be an integer greater than one",
    )
    try:
        authenticate_checkpoint(checkpoint_path)
    except Exception as error:
        raise DialogueExecutionError(
            f"official checkpoint authentication failed: {error}"
        ) from error
    tokenizer = authenticate_tokenizer(tokenizer_dir)
    binding = _model_binding()
    binding["checkpoint"]["filename"] = checkpoint_path.name

    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    embeddings, final_norm, reference_lm_head, states = _load_model(
        checkpoint_path
    )
    rows = []
    failures = []
    for spec in PROMPT_SPECS:
        serialization = _serialize_prompt(spec, tokenizer_dir)
        prompt_ids = tokenizer.encode(
            serialization,
            add_special_tokens=False,
        ).ids
        prompt = {
            "serialization": serialization,
            "serialization_utf8_sha256": _sha256_bytes(
                serialization.encode("utf-8")
            ),
            "token_ids": prompt_ids,
            "decoded_roundtrip": tokenizer.decode(
                prompt_ids,
                skip_special_tokens=False,
            ),
        }
        row: dict[str, Any] = {
            "id": spec["id"],
            "mode": spec["mode"],
            "input_text": spec["input_text"],
            "messages": spec["messages"],
            "prompt": prompt,
        }
        try:
            generation = execute_loaded_prompt(
                checkpoint_path,
                tokenizer,
                embeddings,
                final_norm,
                reference_lm_head,
                states,
                prompt_ids,
                max_new_tokens=max_new_tokens,
                enforce_tolerances=False,
            )
            steps = generation["steps"]
            row_failures = _comparison_failures(spec["id"], generation)
            row.update(
                {
                    "status": (
                        "pass"
                        if not row_failures
                        else "pass_with_comparison_failures"
                    ),
                    "generation": generation,
                    "raw_decoded_output": generation["decoded_text"],
                    "max_hidden_abs_error": max(
                        step["terminal_hidden"]["independent_reference"][
                            "max_abs_error"
                        ]
                        for step in steps
                    ),
                    "max_logit_abs_error": max(
                        step["logits"]["independent_reference"]["max_abs_error"]
                        for step in steps
                    ),
                    "failures": row_failures,
                }
            )
            failures.extend(row_failures)
        except DialogueExecutionError as error:
            failure = {
                "prompt_id": spec["id"],
                "type": "execution_failure",
                "exception_type": type(error).__name__,
                "message": str(error),
            }
            row.update(
                {
                    "status": "execution_failure",
                    "generation": None,
                    "raw_decoded_output": None,
                    "max_hidden_abs_error": None,
                    "max_logit_abs_error": None,
                    "failures": [failure],
                }
            )
            failures.append(failure)
            _reset_kv_caches(states)
        rows.append(row)

    return {
        "schema_version": 1,
        "kind": EVIDENCE_KIND,
        "model_binding": binding,
        "tokenizer_binding": {
            "tokenizer_sha256": TOKENIZER_SHA256,
            "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
            "chat_template_sha256": _sha256_bytes(
                json.loads(
                    (tokenizer_dir / "tokenizer_config.json").read_bytes()
                )["chat_template"].encode("utf-8")
            ),
            "eos_token_id": EOS_TOKEN_ID,
        },
        "numeric_profile": {
            "primary": (
                "native asymmetric packed INT4 AWQ W4A16 G128 with exact "
                "ACE-3 FP16 operators and incremental FP16 KV"
            ),
            "independent_reference": (
                "PyTorch CPU float64 dequantized-AWQ Qwen2 with causal KV"
            ),
            "projection_qzero_adjustment": "none",
            "selection": "greedy full-vocabulary argmax, lowest token ID tie-break",
            "hidden_absolute_tolerance": TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE,
            "logit_absolute_tolerance": LOGITS_ABSOLUTE_TOLERANCE,
        },
        "max_new_tokens": max_new_tokens,
        "rows": rows,
        "failures": failures,
        "claim_boundary": _claim_boundary(),
    }


def validate_showcase_document(
    document: Mapping[str, Any],
    tokenizer: Any,
    tokenizer_dir: Path,
) -> dict[str, Any]:
    _require(document.get("schema_version") == 1, "schema version mismatch")
    _require(document.get("kind") == EVIDENCE_KIND, "evidence kind mismatch")
    _require(
        document["tokenizer_binding"]["tokenizer_sha256"] == TOKENIZER_SHA256
        and document["tokenizer_binding"]["tokenizer_config_sha256"]
        == TOKENIZER_CONFIG_SHA256,
        "tokenizer binding mismatch",
    )
    expected_by_id = {spec["id"]: spec for spec in PROMPT_SPECS}
    rows = document["rows"]
    _require(
        len(rows) == len(PROMPT_SPECS)
        and {row["id"] for row in rows} == set(expected_by_id),
        "showcase prompt coverage mismatch",
    )
    _require(
        all(row["generation"] is not None for row in rows),
        "showcase contains execution failures",
    )

    total_steps = 0
    max_hidden_error = 0.0
    max_logit_error = 0.0
    outputs = []
    derived_failures = []
    for row in rows:
        spec = expected_by_id[row["id"]]
        _require(
            row["mode"] == spec["mode"]
            and row["input_text"] == spec["input_text"]
            and row["messages"] == spec["messages"],
            f"prompt {row['id']} definition mismatch",
        )
        expected_serialization = _serialize_prompt(spec, tokenizer_dir)
        _require(
            row["prompt"]["serialization"] == expected_serialization,
            f"prompt {row['id']} serialization mismatch",
        )
        summary = validate_document(
            {
                "schema_version": 1,
                "kind": EVIDENCE_KIND,
                "model_binding": document["model_binding"],
                "prompt": row["prompt"],
                "generation": row["generation"],
                "claim_boundary": document["claim_boundary"],
            },
            tokenizer,
            expected_kind=EVIDENCE_KIND,
            expected_prompt_serialization=None,
            expected_prompt_token_ids=None,
            require_tolerances=False,
        )
        _require(
            2 <= summary["steps"] <= document["max_new_tokens"],
            f"prompt {row['id']} did not generate several bounded tokens",
        )
        hidden_error = max(
            step["terminal_hidden"]["independent_reference"]["max_abs_error"]
            for step in row["generation"]["steps"]
        )
        logit_error = max(
            step["logits"]["independent_reference"]["max_abs_error"]
            for step in row["generation"]["steps"]
        )
        _require(
            row["max_hidden_abs_error"] == hidden_error
            and row["max_logit_abs_error"] == logit_error
            and row["raw_decoded_output"] == summary["decoded_text"],
            f"prompt {row['id']} summary mismatch",
        )
        row_failures = _comparison_failures(row["id"], row["generation"])
        _require(
            row["failures"] == row_failures
            and row["status"]
            == (
                "pass"
                if not row_failures
                else "pass_with_comparison_failures"
            ),
            f"prompt {row['id']} comparison failures mismatch",
        )
        derived_failures.extend(row_failures)
        total_steps += summary["steps"]
        max_hidden_error = max(max_hidden_error, hidden_error)
        max_logit_error = max(max_logit_error, logit_error)
        outputs.append(
            {
                "id": row["id"],
                "generated_token_ids": summary["generated_token_ids"],
                "raw_decoded_output": summary["decoded_text"],
                "stop_reason": summary["stop_reason"],
            }
        )
    _require(
        document["failures"] == derived_failures,
        "top-level comparison failures mismatch",
    )
    return {
        "prompts": len(rows),
        "steps": total_steps,
        "failures": len(derived_failures),
        "prompts_with_failures": sum(
            bool(row["failures"]) for row in rows
        ),
        "max_hidden_abs_error": max_hidden_error,
        "max_logit_abs_error": max_logit_error,
        "outputs": outputs,
    }


def _markdown(document: Mapping[str, Any]) -> bytes:
    lines = [
        "# Official Model24 continuation showcase",
        "",
        "The adjacent `official_model24_showcase.json` is authoritative and "
        "contains every prompt, token ID, raw output, stop reason, per-token "
        "Primary/PyTorch comparison, FP16 KV lineage record, and failure.",
        "",
        "| Prompt | Mode | Generated token IDs | Raw decoded output | Stop | Status |",
        "|---|---|---|---|---|---|",
    ]
    for row in document["rows"]:
        generation = row["generation"]
        token_ids = "failure" if generation is None else json.dumps(
            generation["generated_token_ids"]
        )
        output = (
            row["failures"][0]["message"]
            if generation is None
            else row["raw_decoded_output"]
        )
        stop = "failure" if generation is None else generation["stop_reason"]
        escaped_output = json.dumps(output, ensure_ascii=False).replace("|", "\\|")
        lines.append(
            f"| `{row['id']}` | {row['mode']} | `{token_ids}` | "
            f"`{escaped_output}` | {stop} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "Scope: software/oracle evidence only. No RTL, FPGA, latency, "
            "throughput, synthesis, or PPA claim is made.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def generate(
    output_dir: Path,
    checkpoint_path: Path = DEFAULT_OFFICIAL_CHECKPOINT,
    tokenizer_dir: Path = DEFAULT_OFFICIAL_TOKENIZER_DIR,
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> dict[str, bytes]:
    document = execute_showcase(
        checkpoint_path,
        tokenizer_dir,
        max_new_tokens=max_new_tokens,
    )
    evidence_payload = _canonical_json(document)
    markdown_payload = _markdown(document)
    tokenizer = authenticate_tokenizer(tokenizer_dir)
    if any(row["generation"] is None for row in document["rows"]):
        summary = {
            "prompts": len(document["rows"]),
            "steps": sum(
                len(row["generation"]["steps"])
                for row in document["rows"]
                if row["generation"] is not None
            ),
            "failures": len(document["failures"]),
        }
    else:
        summary = validate_showcase_document(document, tokenizer, tokenizer_dir)
    manifest = {
        "schema_version": 1,
        "kind": "ace3_official_model24_continuation_showcase_manifest",
        "artifacts": {
            ARTIFACT_NAME: {
                "bytes": len(evidence_payload),
                "sha256": _sha256_bytes(evidence_payload),
            },
            MARKDOWN_NAME: {
                "bytes": len(markdown_payload),
                "sha256": _sha256_bytes(markdown_payload),
            },
        },
        "summary": summary,
    }
    payloads = {
        ARTIFACT_NAME: evidence_payload,
        MANIFEST_NAME: _canonical_json(manifest),
        MARKDOWN_NAME: markdown_payload,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in output_dir.iterdir()}
    _require(
        existing <= set(payloads),
        f"output directory contains unexpected files: {sorted(existing - set(payloads))}",
    )
    for name, payload in payloads.items():
        (output_dir / name).write_bytes(payload)
    _require(
        all(row["generation"] is not None for row in document["rows"]),
        "showcase generation recorded execution failures",
    )
    return payloads


def validate_directory(
    vector_dir: Path,
    checkpoint_path: Path = DEFAULT_OFFICIAL_CHECKPOINT,
    tokenizer_dir: Path = DEFAULT_OFFICIAL_TOKENIZER_DIR,
) -> dict[str, Any]:
    _require(vector_dir.is_dir(), f"evidence directory is missing: {vector_dir}")
    actual_names = {path.name for path in vector_dir.iterdir() if path.is_file()}
    _require(
        actual_names == {ARTIFACT_NAME, MANIFEST_NAME, MARKDOWN_NAME},
        "showcase artifact set mismatch",
    )
    evidence_payload = (vector_dir / ARTIFACT_NAME).read_bytes()
    markdown_payload = (vector_dir / MARKDOWN_NAME).read_bytes()
    manifest = json.loads((vector_dir / MANIFEST_NAME).read_bytes())
    for name, payload in (
        (ARTIFACT_NAME, evidence_payload),
        (MARKDOWN_NAME, markdown_payload),
    ):
        record = manifest["artifacts"][name]
        _require(
            record["bytes"] == len(payload)
            and record["sha256"] == _sha256_bytes(payload),
            f"{name} manifest authentication failed",
        )
    try:
        authenticate_checkpoint(checkpoint_path)
    except Exception as error:
        raise DialogueExecutionError(
            f"official checkpoint authentication failed: {error}"
        ) from error
    _authenticate_model24_binding(_binding_path())
    tokenizer = authenticate_tokenizer(tokenizer_dir)
    document = json.loads(evidence_payload)
    summary = validate_showcase_document(document, tokenizer, tokenizer_dir)
    _require(summary == manifest["summary"], "manifest summary mismatch")
    completed = [
        row["generation"]
        for row in document["rows"]
        if row["generation"] is not None
    ]
    _require(completed, "showcase has no completed generations")
    max_new_tokens = completed[0]["max_new_tokens"]
    _require(
        all(row["max_new_tokens"] == max_new_tokens for row in completed),
        "showcase max_new_tokens mismatch",
    )
    with TemporaryDirectory(prefix="ace3-showcase-validation-") as temporary:
        regenerated = generate(
            Path(temporary),
            checkpoint_path,
            tokenizer_dir,
            max_new_tokens=max_new_tokens,
        )
    _require(
        all(
            regenerated[name] == (vector_dir / name).read_bytes()
            for name in regenerated
        ),
        "independent evidence regeneration mismatch",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("generate", "validate"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--vector-dir", type=Path)
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
                max_new_tokens=args.max_new_tokens,
            )
            document = json.loads(payloads[ARTIFACT_NAME])
            summary = validate_showcase_document(
                document,
                authenticate_tokenizer(args.official_tokenizer_dir.resolve()),
                args.official_tokenizer_dir.resolve(),
            )
            prefix = "OFFICIAL_MODEL24_SHOWCASE_GENERATION_PASS"
        else:
            _require(args.vector_dir is not None, "--vector-dir is required")
            summary = validate_directory(
                args.vector_dir.resolve(),
                args.official_checkpoint.resolve(),
                args.official_tokenizer_dir.resolve(),
            )
            prefix = "OFFICIAL_MODEL24_SHOWCASE_VALIDATION_PASS"
        print(
            f"{prefix} prompts={summary['prompts']} steps={summary['steps']} "
            f"failures={summary['failures']} "
            f"max_hidden_error={summary['max_hidden_abs_error']:.9f} "
            f"max_logit_error={summary['max_logit_abs_error']:.9f}"
        )
    except (DialogueExecutionError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"OFFICIAL_MODEL24_SHOWCASE_FAIL {error}") from error


if __name__ == "__main__":
    main()
