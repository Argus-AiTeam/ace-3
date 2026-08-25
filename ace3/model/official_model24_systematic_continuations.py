#!/usr/bin/env python3
"""Systematic official Model24 continuation batch and evidence validator."""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from model24_execution_oracle import (
    DEFAULT_OFFICIAL_CHECKPOINT,
    DEFAULT_OFFICIAL_TOKENIZER_DIR,
    EOS_TOKEN_ID,
    TOKENIZER_CONFIG_SHA256,
    TOKENIZER_SHA256,
    authenticate_tokenizer,
)
from model24_oracle import authenticate_checkpoint
from official_model24_dialogue import (
    DialogueExecutionError,
    _load_model,
    _reset_kv_caches,
    execute_loaded_prompt,
    validate_document,
)
from official_model24_next_token import _json_without_duplicates
from official_model24_showcase import (
    LOGITS_ABSOLUTE_TOLERANCE,
    TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE,
    _canonical_json,
    _model_binding,
    _serialize_prompt,
    _sha256_bytes,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROMPT_SUITE_PATH = Path(__file__).with_name(
    "official_model24_systematic_prompts.json"
)
ARTIFACT_NAME = "batch.jsonl"
SUMMARY_NAME = "summary.json"
MANIFEST_NAME = "manifest.json"
MARKDOWN_NAME = "REPORT.md"
RUN_LOG_NAME = "run.log"
EVIDENCE_KIND = "ace3_official_model24_systematic_continuation_batch"
SUMMARY_KIND = "ace3_official_model24_systematic_continuation_summary"
MANIFEST_KIND = "ace3_official_model24_systematic_continuation_manifest"
PROMPT_SUITE_KIND = "ace3_official_model24_systematic_prompt_suite"
REVIEWED_BASELINE = "showcasecontinuations15c"
EXCLUDED_UNREVIEWED_EVIDENCE = "486e5d848245"
DEFAULT_MAX_NEW_TOKENS = 4
EXPECTED_CASE_COUNT = 32
EXPECTED_CATEGORY_COUNTS = {
    "chat": 6,
    "code": 5,
    "commonsense": 5,
    "continuation": 6,
    "factual": 6,
    "reasoning": 4,
}
EXPECTED_LANGUAGE_COUNTS = {"en": 16, "zh": 16}
CONSUMED_ASSET_PATHS = (
    "ace3/contracts/model24_execution_vector_bindings.json",
    "ace3/model/attention_oracle.py",
    "ace3/model/awq_bit_oracle.py",
    "ace3/model/fp16_adaptation_oracle.py",
    "ace3/model/model24_execution_oracle.py",
    "ace3/model/model24_oracle.py",
    "ace3/model/official_model24_dialogue.py",
    "ace3/model/official_model24_next_token.py",
    "ace3/model/official_model24_showcase.py",
    "ace3/model/official_model24_systematic_continuations.py",
    "ace3/model/official_model24_systematic_prompts.json",
    "ace3/model/official_single_decoder_layer.py",
    "ace3/model/projection_oracle.py",
    "ace3/model/qwen2_rope_oracle.py",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DialogueExecutionError(message)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_prompt_suite(path: Path = PROMPT_SUITE_PATH) -> dict[str, Any]:
    payload = path.read_bytes()
    suite = _json_without_duplicates(payload, path.name)
    _validate_prompt_suite(suite)
    return suite


def _validate_prompt_suite(suite: Mapping[str, Any]) -> None:
    _require(suite.get("schema_version") == 1, "prompt suite schema mismatch")
    _require(suite.get("kind") == PROMPT_SUITE_KIND, "prompt suite kind mismatch")
    _require(
        suite.get("reviewed_baseline") == REVIEWED_BASELINE,
        "prompt suite reviewed baseline mismatch",
    )
    cases = suite.get("cases")
    _require(
        isinstance(cases, list) and len(cases) == EXPECTED_CASE_COUNT,
        "prompt suite must contain exactly 32 cases",
    )
    ids = [case.get("id") for case in cases]
    _require(
        all(isinstance(case_id, str) and case_id for case_id in ids)
        and len(set(ids)) == EXPECTED_CASE_COUNT,
        "prompt suite case IDs must be unique non-empty strings",
    )
    categories = Counter(case.get("category") for case in cases)
    languages = Counter(case.get("language") for case in cases)
    _require(
        dict(sorted(categories.items())) == EXPECTED_CATEGORY_COUNTS,
        "prompt suite category balance mismatch",
    )
    _require(
        dict(sorted(languages.items())) == EXPECTED_LANGUAGE_COUNTS,
        "prompt suite language balance mismatch",
    )
    for case in cases:
        mode = case.get("mode")
        input_text = case.get("input_text")
        messages = case.get("messages")
        _require(mode in ("continuation", "chat"), f"invalid mode for {case['id']}")
        _require(
            isinstance(input_text, str) and bool(input_text),
            f"empty input text for {case['id']}",
        )
        if mode == "continuation":
            _require(messages is None, f"continuation {case['id']} has messages")
        else:
            _require(
                messages
                == [
                    {
                        "role": "system",
                        "content": "You are a concise assistant.",
                    },
                    {"role": "user", "content": input_text},
                ],
                f"chat messages mismatch for {case['id']}",
            )


def _asset_bindings() -> list[dict[str, Any]]:
    bindings = []
    for relative_path in CONSUMED_ASSET_PATHS:
        path = REPOSITORY_ROOT / relative_path
        payload = path.read_bytes()
        bindings.append(
            {
                "path": relative_path,
                "bytes": len(payload),
                "sha256": _sha256_bytes(payload),
            }
        )
    return bindings


def _tokenizer_binding(tokenizer_dir: Path) -> dict[str, Any]:
    config = json.loads((tokenizer_dir / "tokenizer_config.json").read_bytes())
    return {
        "tokenizer_sha256": TOKENIZER_SHA256,
        "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
        "chat_template_sha256": _sha256_bytes(
            config["chat_template"].encode("utf-8")
        ),
        "eos_token_id": EOS_TOKEN_ID,
    }


def _provenance() -> dict[str, Any]:
    return {
        "reviewed_admissible_baseline": REVIEWED_BASELINE,
        "baseline_use": (
            "accepted Model24 showcase executor, bindings, and validation boundary"
        ),
        "excluded_claim_bearing_evidence": [EXCLUDED_UNREVIEWED_EVIDENCE],
        "exclusion_reason": (
            "unreviewed ancestry is not used as acceptance or claim-bearing evidence"
        ),
    }


def _numeric_profile() -> dict[str, Any]:
    return {
        "primary": (
            "ACE-3 native asymmetric packed INT4 AWQ W4A16 G128 with exact "
            "FP16 operators and incremental FP16 KV"
        ),
        "independent_reference": (
            "PyTorch CPU float64 dequantized-AWQ Qwen2 with causal KV"
        ),
        "projection_qzero_adjustment": "none",
        "selection": "greedy full-vocabulary argmax, lowest token ID tie-break",
        "hidden_absolute_tolerance": TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE,
        "logit_absolute_tolerance": LOGITS_ABSOLUTE_TOLERANCE,
    }


def _claim_boundary() -> dict[str, str]:
    return {
        "demonstrated": (
            "one fixed 32-case deterministic software/oracle batch using the "
            "official checkpoint, tokenizer, 24-layer executor, tied head, and "
            "incremental FP16 KV"
        ),
        "broader_quality": (
            "not assessed beyond the preserved bounded English/Chinese prompt suite"
        ),
        "rtl": "full 24-layer batch execution not demonstrated in RTL",
        "synthesis": "not run",
        "ppa": "not measured",
        "fpga": "not run",
        "latency": "diagnostic wall time only; product latency not measured",
        "throughput": "not measured",
    }


def _metadata(
    checkpoint_path: Path,
    tokenizer_dir: Path,
    suite: Mapping[str, Any],
) -> dict[str, Any]:
    model_binding = _model_binding()
    model_binding["checkpoint"]["filename"] = checkpoint_path.name
    return {
        "provenance": _provenance(),
        "suite_binding": {
            "path": str(PROMPT_SUITE_PATH.relative_to(REPOSITORY_ROOT)),
            "sha256": _sha256_bytes(PROMPT_SUITE_PATH.read_bytes()),
            "case_count": len(suite["cases"]),
            "category_counts": EXPECTED_CATEGORY_COUNTS,
            "language_counts": EXPECTED_LANGUAGE_COUNTS,
        },
        "asset_bindings": _asset_bindings(),
        "model_binding": model_binding,
        "tokenizer_binding": _tokenizer_binding(tokenizer_dir),
        "numeric_profile": _numeric_profile(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_threads": 8,
            "deterministic_algorithms": True,
        },
        "claim_boundary": _claim_boundary(),
    }


def _prompt_record(
    case: Mapping[str, Any],
    tokenizer: Any,
    tokenizer_dir: Path,
) -> dict[str, Any]:
    serialization = _serialize_prompt(case, tokenizer_dir)
    token_ids = tokenizer.encode(serialization, add_special_tokens=False).ids
    return {
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


def _agreement_record(step: Mapping[str, Any]) -> dict[str, str]:
    token = step["token"]
    hidden = step["terminal_hidden"]["independent_reference"]
    logits = step["logits"]["independent_reference"]
    return {
        "argmax": (
            "agreement"
            if token["argmax_matches_independent_reference"]
            else "mismatch"
        ),
        "terminal_hidden_tolerance": (
            "agreement" if hidden["within_tolerance"] else "mismatch"
        ),
        "logit_tolerance": (
            "agreement" if logits["within_tolerance"] else "mismatch"
        ),
    }


def _step_failures(
    prompt_id: str,
    generation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    failures = []
    for step in generation["steps"]:
        token = step["token"]
        if not token["argmax_matches_independent_reference"]:
            failures.append(
                {
                    "prompt_id": prompt_id,
                    "step": step["ordinal"],
                    "type": "argmax_mismatch",
                    "ace_token_id": token["argmax_token_id"],
                    "pytorch_token_id": token[
                        "independent_reference_argmax_token_id"
                    ],
                }
            )
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


def _record_generation_agreements(generation: Mapping[str, Any]) -> None:
    for step in generation["steps"]:
        step["ace_vs_pytorch"] = _agreement_record(step)


def execute_batch(
    checkpoint_path: Path,
    tokenizer_dir: Path,
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, Any]]:
    _require(
        type(max_new_tokens) is int and max_new_tokens > 1,
        "batch max_new_tokens must be an integer greater than one",
    )
    try:
        authenticate_checkpoint(checkpoint_path)
    except Exception as error:
        raise DialogueExecutionError(
            f"official checkpoint authentication failed: {error}"
        ) from error
    suite = _load_prompt_suite()
    tokenizer = authenticate_tokenizer(tokenizer_dir)
    metadata = _metadata(checkpoint_path, tokenizer_dir, suite)
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)

    batch_start = time.perf_counter()
    embeddings, final_norm, reference_lm_head, states = _load_model(checkpoint_path)
    rows = []
    for ordinal, case in enumerate(suite["cases"]):
        prompt = _prompt_record(case, tokenizer, tokenizer_dir)
        row: dict[str, Any] = {
            "ordinal": ordinal,
            "case_count": EXPECTED_CASE_COUNT,
            "id": case["id"],
            "category": case["category"],
            "language": case["language"],
            "mode": case["mode"],
            "input_text": case["input_text"],
            "messages": case["messages"],
            "prompt": prompt,
        }
        case_start = time.perf_counter()
        try:
            generation = execute_loaded_prompt(
                checkpoint_path,
                tokenizer,
                embeddings,
                final_norm,
                reference_lm_head,
                states,
                prompt["token_ids"],
                max_new_tokens=max_new_tokens,
                enforce_tolerances=False,
            )
            _record_generation_agreements(generation)
            failures = _step_failures(case["id"], generation)
            row.update(
                {
                    "status": "pass" if not failures else "completed_with_mismatches",
                    "generation": generation,
                    "raw_decoded_output": generation["decoded_text"],
                    "max_hidden_abs_error": max(
                        step["terminal_hidden"]["independent_reference"][
                            "max_abs_error"
                        ]
                        for step in generation["steps"]
                    ),
                    "max_logit_abs_error": max(
                        step["logits"]["independent_reference"]["max_abs_error"]
                        for step in generation["steps"]
                    ),
                    "failures": failures,
                }
            )
        except DialogueExecutionError as error:
            failure = {
                "prompt_id": case["id"],
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
            _reset_kv_caches(states)
        row["wall_time_seconds_diagnostic"] = time.perf_counter() - case_start
        rows.append(row)
    diagnostics = {
        "batch_wall_time_seconds_diagnostic": time.perf_counter() - batch_start,
        "case_wall_time_seconds_diagnostic": sum(
            row["wall_time_seconds_diagnostic"] for row in rows
        ),
    }
    return rows, diagnostics, metadata


def _validate_diagnostic_time(value: Any, label: str) -> float:
    _require(
        isinstance(value, float) and math.isfinite(value) and value > 0.0,
        f"{label} must be a positive finite diagnostic float",
    )
    return value


def _validate_generation(
    row: Mapping[str, Any],
    metadata: Mapping[str, Any],
    tokenizer: Any,
) -> dict[str, Any]:
    generation = row["generation"]
    _require(generation is not None, f"case {row['id']} has no generation")
    normalized = copy.deepcopy(generation)
    for original, adjusted in zip(
        generation["steps"],
        normalized["steps"],
        strict=True,
    ):
        token = original["token"]
        ace_token = token["argmax_token_id"]
        reference_token = token["independent_reference_argmax_token_id"]
        _require(
            type(ace_token) is int
            and type(reference_token) is int
            and token["argmax_matches_independent_reference"]
            == (ace_token == reference_token),
            f"case {row['id']} step {original['ordinal']} argmax record mismatch",
        )
        _require(
            original.get("ace_vs_pytorch") == _agreement_record(original),
            f"case {row['id']} step {original['ordinal']} agreement record mismatch",
        )
        if ace_token != reference_token:
            adjusted["token"]["independent_reference_argmax_token_id"] = ace_token
            adjusted["token"]["argmax_matches_independent_reference"] = True
    validation_summary = validate_document(
        {
            "schema_version": 1,
            "kind": EVIDENCE_KIND,
            "model_binding": metadata["model_binding"],
            "prompt": row["prompt"],
            "generation": normalized,
            "claim_boundary": metadata["claim_boundary"],
        },
        tokenizer,
        expected_kind=EVIDENCE_KIND,
        expected_prompt_serialization=None,
        expected_prompt_token_ids=None,
        require_tolerances=False,
    )
    _require(
        2 <= validation_summary["steps"] <= generation["max_new_tokens"],
        f"case {row['id']} did not generate multiple bounded tokens",
    )
    return validation_summary


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    tokenizer: Any,
    tokenizer_dir: Path,
    *,
    require_complete: bool,
) -> dict[str, Any]:
    suite = _load_prompt_suite()
    cases = suite["cases"]
    _require(
        len(rows) == EXPECTED_CASE_COUNT,
        "batch row count is not the complete 32-case suite",
    )
    _require(
        [row.get("id") for row in rows] == [case["id"] for case in cases],
        "batch prompt order or coverage mismatch",
    )
    all_failures = []
    outputs = []
    total_steps = 0
    execution_failures = 0
    max_hidden_error = 0.0
    max_logit_error = 0.0
    serializations = []
    for ordinal, (row, case) in enumerate(zip(rows, cases, strict=True)):
        _require(
            row.get("ordinal") == ordinal
            and row.get("case_count") == EXPECTED_CASE_COUNT,
            f"case {case['id']} ordinal or batch count mismatch",
        )
        for field in (
            "id",
            "category",
            "language",
            "mode",
            "input_text",
            "messages",
        ):
            _require(
                row.get(field) == case[field],
                f"case {case['id']} {field} mismatch",
            )
        expected_prompt = _prompt_record(case, tokenizer, tokenizer_dir)
        _require(
            row.get("prompt") == expected_prompt,
            f"case {case['id']} prompt serialization or token stream mismatch",
        )
        serializations.append(expected_prompt["serialization"])
        _validate_diagnostic_time(
            row.get("wall_time_seconds_diagnostic"),
            f"case {case['id']} wall time",
        )
        generation = row.get("generation")
        if generation is None:
            execution_failures += 1
            failures = row.get("failures")
            _require(
                row.get("status") == "execution_failure"
                and isinstance(failures, list)
                and len(failures) == 1
                and failures[0].get("prompt_id") == case["id"]
                and failures[0].get("type") == "execution_failure"
                and isinstance(failures[0].get("message"), str),
                f"case {case['id']} execution failure record mismatch",
            )
            _require(
                row.get("raw_decoded_output") is None
                and row.get("max_hidden_abs_error") is None
                and row.get("max_logit_abs_error") is None,
                f"case {case['id']} execution failure has generated summaries",
            )
            all_failures.extend(failures)
            continue
        validation_summary = _validate_generation(row, metadata, tokenizer)
        failures = _step_failures(case["id"], generation)
        hidden_error = max(
            step["terminal_hidden"]["independent_reference"]["max_abs_error"]
            for step in generation["steps"]
        )
        logit_error = max(
            step["logits"]["independent_reference"]["max_abs_error"]
            for step in generation["steps"]
        )
        _require(
            row.get("failures") == failures
            and row.get("status")
            == ("pass" if not failures else "completed_with_mismatches"),
            f"case {case['id']} mismatch reporting mismatch",
        )
        _require(
            row.get("raw_decoded_output") == validation_summary["decoded_text"]
            and row.get("max_hidden_abs_error") == hidden_error
            and row.get("max_logit_abs_error") == logit_error,
            f"case {case['id']} output summary mismatch",
        )
        total_steps += validation_summary["steps"]
        max_hidden_error = max(max_hidden_error, hidden_error)
        max_logit_error = max(max_logit_error, logit_error)
        all_failures.extend(failures)
        outputs.append(
            {
                "id": case["id"],
                "generated_token_ids": validation_summary["generated_token_ids"],
                "raw_decoded_output": validation_summary["decoded_text"],
                "stop_reason": validation_summary["stop_reason"],
            }
        )
    _require(
        len(set(serializations)) == EXPECTED_CASE_COUNT,
        "batch contains duplicate serialized prompts",
    )
    if require_complete:
        _require(
            execution_failures == 0,
            "batch contains execution failures or partial rows",
        )
    return {
        "cases": EXPECTED_CASE_COUNT,
        "completed_cases": EXPECTED_CASE_COUNT - execution_failures,
        "execution_failures": execution_failures,
        "steps": total_steps,
        "mismatches": len(
            [failure for failure in all_failures if failure["type"] != "execution_failure"]
        ),
        "cases_with_mismatches": sum(
            row.get("status") == "completed_with_mismatches" for row in rows
        ),
        "max_hidden_abs_error": max_hidden_error,
        "max_logit_abs_error": max_logit_error,
        "failures": all_failures,
        "outputs": outputs,
    }


def _build_summary(
    rows: Sequence[Mapping[str, Any]],
    diagnostics: Mapping[str, Any],
    metadata: Mapping[str, Any],
    tokenizer: Any,
    tokenizer_dir: Path,
    *,
    require_complete: bool,
) -> dict[str, Any]:
    batch_wall_time = _validate_diagnostic_time(
        diagnostics.get("batch_wall_time_seconds_diagnostic"),
        "batch wall time",
    )
    case_wall_time = _validate_diagnostic_time(
        diagnostics.get("case_wall_time_seconds_diagnostic"),
        "aggregate case wall time",
    )
    _require(
        batch_wall_time >= case_wall_time,
        "batch diagnostic wall time is shorter than aggregate case wall time",
    )
    results = _validate_rows(
        rows,
        metadata,
        tokenizer,
        tokenizer_dir,
        require_complete=require_complete,
    )
    return {
        "schema_version": 1,
        "kind": SUMMARY_KIND,
        **metadata,
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        "diagnostics": {
            "label": (
                "host wall time is diagnostic only and is not product latency "
                "or throughput evidence"
            ),
            "batch_wall_time_seconds_diagnostic": batch_wall_time,
            "case_wall_time_seconds_diagnostic": case_wall_time,
        },
        "results": results,
        "batch_status": (
            "incomplete"
            if results["execution_failures"]
            else (
                "completed_with_mismatches"
                if results["mismatches"]
                else "pass"
            )
        ),
    }


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json(row) for row in rows)


def _markdown(
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> bytes:
    results = summary["results"]
    lines = [
        "# Official Model24 systematic continuation quality report",
        "",
        f"Reviewed admissible baseline: `{REVIEWED_BASELINE}`. The unreviewed "
        f"`{EXCLUDED_UNREVIEWED_EVIDENCE}` ancestry is explicitly excluded from "
        "acceptance and claim-bearing evidence.",
        "",
        "The adjacent `batch.jsonl` is authoritative and preserves all 32 ordered "
        "cases, prompt serializations and token IDs, raw decoded outputs, stop "
        "reasons, every ACE-vs-PyTorch step comparison, FP16 KV lineage, diagnostic "
        "wall times, and failures.",
        "",
        f"Status: **{summary['batch_status']}**; completed "
        f"{results['completed_cases']}/{results['cases']} cases and "
        f"{results['steps']} generated steps, with {results['mismatches']} "
        "preserved comparison mismatches.",
        "",
        "| # | Case | Category | Lang | Token IDs | Raw decoded output | Stop | Result |",
        "|---:|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        generation = row["generation"]
        token_ids = (
            "execution failure"
            if generation is None
            else json.dumps(generation["generated_token_ids"])
        )
        output = (
            row["failures"][0]["message"]
            if generation is None
            else row["raw_decoded_output"]
        )
        stop = "execution_failure" if generation is None else generation["stop_reason"]
        escaped = json.dumps(output, ensure_ascii=False).replace("|", "\\|")
        lines.append(
            f"| {row['ordinal'] + 1} | `{row['id']}` | {row['category']} | "
            f"{row['language']} | `{token_ids}` | `{escaped}` | {stop} | "
            f"{row['status']} |"
        )
    lines.extend(
        [
            "",
            "Scope: bounded software/oracle evidence only. Diagnostic host wall "
            "time is not product latency or throughput. No full-model RTL, FPGA, "
            "synthesis, PPA, broad dialogue-quality, latency, or throughput claim "
            "is made.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _run_log(
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> bytes:
    lines = [
        f"BASELINE reviewed={REVIEWED_BASELINE} "
        f"excluded_unreviewed={EXCLUDED_UNREVIEWED_EVIDENCE}",
    ]
    for row in rows:
        steps = 0 if row["generation"] is None else len(row["generation"]["steps"])
        lines.append(
            f"CASE ordinal={row['ordinal']} id={row['id']} status={row['status']} "
            f"steps={steps} failures={len(row['failures'])} "
            f"wall_time_seconds_diagnostic={row['wall_time_seconds_diagnostic']:.9f}"
        )
    results = summary["results"]
    lines.append(
        f"BATCH status={summary['batch_status']} cases={results['cases']} "
        f"completed={results['completed_cases']} steps={results['steps']} "
        f"mismatches={results['mismatches']} execution_failures="
        f"{results['execution_failures']} wall_time_seconds_diagnostic="
        f"{summary['diagnostics']['batch_wall_time_seconds_diagnostic']:.9f}"
    )
    return ("\n".join(lines) + "\n").encode("ascii")


def generate(
    output_dir: Path,
    checkpoint_path: Path = DEFAULT_OFFICIAL_CHECKPOINT,
    tokenizer_dir: Path = DEFAULT_OFFICIAL_TOKENIZER_DIR,
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> dict[str, Any]:
    _require(
        max_new_tokens == DEFAULT_MAX_NEW_TOKENS,
        f"systematic batch is fixed at {DEFAULT_MAX_NEW_TOKENS} new tokens",
    )
    rows, diagnostics, metadata = execute_batch(
        checkpoint_path,
        tokenizer_dir,
        max_new_tokens=max_new_tokens,
    )
    tokenizer = authenticate_tokenizer(tokenizer_dir)
    summary = _build_summary(
        rows,
        diagnostics,
        metadata,
        tokenizer,
        tokenizer_dir,
        require_complete=False,
    )
    payloads = {
        ARTIFACT_NAME: _jsonl(rows),
        SUMMARY_NAME: _canonical_json(summary),
        MARKDOWN_NAME: _markdown(summary, rows),
        RUN_LOG_NAME: _run_log(summary, rows),
    }
    manifest = {
        "schema_version": 1,
        "kind": MANIFEST_KIND,
        "artifacts": {
            name: {
                "bytes": len(payload),
                "sha256": _sha256_bytes(payload),
            }
            for name, payload in payloads.items()
        },
        "results": {
            key: summary["results"][key]
            for key in (
                "cases",
                "completed_cases",
                "execution_failures",
                "steps",
                "mismatches",
                "cases_with_mismatches",
            )
        },
        "reviewed_baseline": REVIEWED_BASELINE,
        "excluded_claim_bearing_evidence": [EXCLUDED_UNREVIEWED_EVIDENCE],
    }
    payloads[MANIFEST_NAME] = _canonical_json(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in output_dir.iterdir() if path.is_file()}
    _require(
        existing <= set(payloads),
        f"output directory contains unexpected files: {sorted(existing - set(payloads))}",
    )
    for name, payload in payloads.items():
        (output_dir / name).write_bytes(payload)
    _require(
        summary["results"]["execution_failures"] == 0,
        "systematic batch recorded execution failures; evidence was preserved",
    )
    return summary


def _parse_jsonl(payload: bytes) -> list[dict[str, Any]]:
    _require(payload.endswith(b"\n"), "batch JSONL is truncated or lacks final newline")
    lines = payload.splitlines()
    _require(
        len(lines) == EXPECTED_CASE_COUNT and all(lines),
        "batch JSONL row count is incomplete, duplicated, or blank",
    )
    return [
        _json_without_duplicates(line, f"{ARTIFACT_NAME}:{ordinal + 1}")
        for ordinal, line in enumerate(lines)
    ]


def validate_directory(
    vector_dir: Path,
    checkpoint_path: Path = DEFAULT_OFFICIAL_CHECKPOINT,
    tokenizer_dir: Path = DEFAULT_OFFICIAL_TOKENIZER_DIR,
) -> dict[str, Any]:
    _require(vector_dir.is_dir(), f"evidence directory is missing: {vector_dir}")
    expected_names = {
        ARTIFACT_NAME,
        SUMMARY_NAME,
        MANIFEST_NAME,
        MARKDOWN_NAME,
        RUN_LOG_NAME,
    }
    actual_names = {path.name for path in vector_dir.iterdir() if path.is_file()}
    _require(actual_names == expected_names, "systematic batch artifact set mismatch")
    manifest = _json_without_duplicates(
        (vector_dir / MANIFEST_NAME).read_bytes(),
        MANIFEST_NAME,
    )
    _require(
        manifest.get("schema_version") == 1
        and manifest.get("kind") == MANIFEST_KIND,
        "systematic batch manifest identity mismatch",
    )
    payloads = {
        name: (vector_dir / name).read_bytes()
        for name in expected_names - {MANIFEST_NAME}
    }
    _require(
        set(manifest.get("artifacts", {})) == set(payloads),
        "manifest artifact coverage mismatch",
    )
    for name, payload in payloads.items():
        record = manifest["artifacts"][name]
        _require(
            record.get("bytes") == len(payload)
            and record.get("sha256") == _sha256_bytes(payload),
            f"{name} manifest authentication failed",
        )
    try:
        authenticate_checkpoint(checkpoint_path)
    except Exception as error:
        raise DialogueExecutionError(
            f"official checkpoint authentication failed: {error}"
        ) from error
    suite = _load_prompt_suite()
    tokenizer = authenticate_tokenizer(tokenizer_dir)
    expected_metadata = _metadata(checkpoint_path, tokenizer_dir, suite)
    summary = _json_without_duplicates(payloads[SUMMARY_NAME], SUMMARY_NAME)
    for key, value in expected_metadata.items():
        _require(summary.get(key) == value, f"stale or mutated {key} binding")
    _require(
        summary.get("schema_version") == 1
        and summary.get("kind") == SUMMARY_KIND
        and summary.get("max_new_tokens") == DEFAULT_MAX_NEW_TOKENS,
        "systematic batch summary identity mismatch",
    )
    rows = _parse_jsonl(payloads[ARTIFACT_NAME])
    rebuilt = _build_summary(
        rows,
        summary["diagnostics"],
        expected_metadata,
        tokenizer,
        tokenizer_dir,
        require_complete=True,
    )
    _require(summary == rebuilt, "systematic batch summary mismatch")
    _require(
        payloads[MARKDOWN_NAME] == _markdown(summary, rows),
        "systematic batch report mismatch",
    )
    _require(
        payloads[RUN_LOG_NAME] == _run_log(summary, rows),
        "systematic batch run log mismatch",
    )
    _require(
        manifest["results"]
        == {
            key: summary["results"][key]
            for key in (
                "cases",
                "completed_cases",
                "execution_failures",
                "steps",
                "mismatches",
                "cases_with_mismatches",
            )
        }
        and manifest.get("reviewed_baseline") == REVIEWED_BASELINE
        and manifest.get("excluded_claim_bearing_evidence")
        == [EXCLUDED_UNREVIEWED_EVIDENCE],
        "manifest result or provenance mismatch",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        checkpoint_path = args.official_checkpoint.resolve()
        tokenizer_dir = args.official_tokenizer_dir.resolve()
        if args.operation == "generate":
            _require(args.output_dir is not None, "--output-dir is required")
            summary = generate(
                args.output_dir.resolve(),
                checkpoint_path,
                tokenizer_dir,
            )
            prefix = "OFFICIAL_MODEL24_SYSTEMATIC_CONTINUATIONS_GENERATION_PASS"
        else:
            _require(args.vector_dir is not None, "--vector-dir is required")
            summary = validate_directory(
                args.vector_dir.resolve(),
                checkpoint_path,
                tokenizer_dir,
            )
            prefix = "OFFICIAL_MODEL24_SYSTEMATIC_CONTINUATIONS_VALIDATION_PASS"
        results = summary["results"]
        print(
            f"{prefix} cases={results['cases']} completed={results['completed_cases']} "
            f"steps={results['steps']} mismatches={results['mismatches']} "
            f"execution_failures={results['execution_failures']} "
            f"baseline={REVIEWED_BASELINE}"
        )
    except (DialogueExecutionError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(
            f"OFFICIAL_MODEL24_SYSTEMATIC_CONTINUATIONS_FAIL {error}"
        ) from error


if __name__ == "__main__":
    main()
