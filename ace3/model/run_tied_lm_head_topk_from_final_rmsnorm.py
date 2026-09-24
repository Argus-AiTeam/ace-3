#!/usr/bin/env python3
"""Run official-shape tied lm_head/Top-K RTL from the sealed final RMSNorm."""

from __future__ import annotations

import hashlib
import heapq
import importlib.util
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


ROOT = Path("/home/argustest/ace3-argus")
PREDECESSOR = ROOT / "build/model24_final_rmsnorm_attempt001"
PRIOR_ATTEMPT = ROOT / "build/model24_tied_lm_head_topk_attempt001"
ATTEMPT_ID = "model24_tied_lm_head_topk_attempt002"
OFFICIAL_ATTEMPT = 2
OUTPUT = ROOT / f"build/{ATTEMPT_ID}"
MISSION = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/"
    "0f784a0c042c/mission.json"
)
CHECKPOINT = ROOT / "build/model24_rtl_cascade/checkpoint/model.safetensors"
REFERENCE = ROOT / "ace3/model/streaming_lm_head_reference.py"
FROZEN_CONTRACT = ROOT / "ace3/contracts/streaming_tied_lm_head_topk.json"
DESIGN_MANIFEST = ROOT / "design/RTL_MANIFEST.json"
RTL_TRACEABILITY = ROOT / "design/RTL_TRACEABILITY.md"
RTL_SOURCES = (
    ROOT / "ace3/rtl/ace3_fp16_fixed.sv",
    ROOT / "ace3/rtl/ace3_q47_48_to_f16_rne.sv",
    ROOT / "ace3/rtl/ace3_streaming_tied_lm_head_topk.sv",
)
PROTOCOL_TB = ROOT / "ace3/tb/ace3_streaming_tied_lm_head_topk_tb.sv"
TOP_MODULE = "ace3_streaming_tied_lm_head_topk"
MODEL_REPOSITORY = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
MODEL_REVISION = "db09cd27ead7fee40cdee309693cf83601b9c899"
CHECKPOINT_SHA256 = "c50d807b7bed7ff314308972e0f4bcf4e5a70bc60ad88fc7df53940831ed0c1b"
CHECKPOINT_BYTES = 730_652_248
TIED_WEIGHT_SHA256 = "d74257dc547b48be5ae7b93f1c9af072c0c42dbbb85503078e25c59cd09e68d0"
HIDDEN_SIZE = 896
VOCAB_SIZE = 151_936
TOP_K = 10
EXPECTED_WEIGHTS = HIDDEN_SIZE * VOCAB_SIZE
FINAL_ROWS = HIDDEN_SIZE * 2
NATURAL_TERMINAL_RE = re.compile(
    rb"schema=ace3_tied_lm_head_raw_v1 natural_terminal=1 exit_code=0 "
    rb"hidden_count=896 weight_count=136134656 logit_count=151936 "
    rb"top_count=10 cycles=([1-9][0-9]*)\n"
)
FAILURE_TERMINAL_RE = re.compile(
    rb"schema=ace3_tied_lm_head_raw_v1 natural_terminal=0 exit_code=2 "
    rb"hidden_count=896 weight_count=896 logit_count=1 top_count=0 "
    rb"cycles=([1-9][0-9]*)\n"
)


class AttemptError(RuntimeError):
    """Raised when an evidence or execution boundary fails closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AttemptError(message)


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    document = json.loads(
        path.read_text(encoding="ascii"),
        object_pairs_hook=reject_duplicate_keys,
    )
    require(isinstance(document, dict), f"JSON object required: {path}")
    return document


def canonical_json(document: object) -> bytes:
    return (
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def write_json(path: Path, document: object) -> None:
    with path.open("xb") as stream:
        stream.write(canonical_json(document))


def write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)


def write_lines(path: Path, lines: Iterable[str]) -> None:
    with path.open("x", encoding="ascii", newline="\n") as stream:
        for line in lines:
            stream.write(f"{line}\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_range(path: Path, offset: int, length: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(offset)
        remaining = length
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            require(bool(chunk), "checkpoint tensor payload is truncated")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_record(path: Path, root: Path | None = None) -> dict[str, Any]:
    stat = path.stat()
    record: dict[str, Any] = {
        "path": str(path),
        "bytes": stat.st_size,
        "sha256": sha256(path),
        "mode": f"{stat.st_mode & 0o777:04o}",
    }
    if root is not None:
        record["relative_path"] = str(path.relative_to(root))
    return record


def snapshot(root: Path) -> list[dict[str, Any]]:
    return [
        file_record(path, root)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def snapshot_digest(records: list[dict[str, Any]]) -> str:
    return sha256_bytes(canonical_json(records))


def authenticate_predecessor(
    mission: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    require(PREDECESSOR.is_dir(), "final-RMSNorm predecessor is missing")
    before = snapshot(PREDECESSOR)
    seal = load_json(PREDECESSOR / "sealed_output_manifest.json")
    require(
        seal.get("kind") == "ace3_final_rmsnorm_attempt001_sealed_manifest"
        and seal.get("status") == "ENGINEERING_PASS_REVIEW_REQUIRED"
        and seal.get("official_attempt") == 1
        and seal.get("operation") == "final_rmsnorm",
        "final-RMSNorm seal identity is malformed",
    )
    artifacts = {
        row.get("relative_path"): row
        for row in seal.get("artifacts", [])
        if isinstance(row, dict) and isinstance(row.get("relative_path"), str)
    }
    require(
        len(artifacts) == seal.get("artifact_count"),
        "final-RMSNorm sealed artifact set is malformed or duplicated",
    )
    for relative, expected in artifacts.items():
        path = PREDECESSOR / relative
        require(path.is_file(), f"sealed predecessor artifact missing: {relative}")
        require(
            path.stat().st_size == expected.get("bytes")
            and sha256(path) == expected.get("sha256"),
            f"sealed predecessor artifact changed: {relative}",
        )
    status = load_json(PREDECESSOR / "status.json")
    gate = load_json(PREDECESSOR / "natural_terminal_gate.json")
    integer = load_json(PREDECESSOR / "integer_comparison.json")
    policy = load_json(PREDECESSOR / "fp16_policy_comparison.json")
    preservation = load_json(PREDECESSOR / "preservation.json")
    require(
        status.get("status") == "PASS"
        and status.get("failure_count") == 0
        and status.get("first_material_mismatch") is None
        and gate.get("natural_terminal") is True
        and gate.get("actual_exit_code") == 0
        and integer.get("bit_exact") is True
        and integer.get("failure_count") == 0
        and policy.get("within_tolerance") is True
        and policy.get("failure_count") == 0
        and preservation.get("predecessor_preserved") is True,
        "final-RMSNorm predecessor is not a complete numerical PASS",
    )
    require(
        mission.get("kind") == "mission_context"
        and mission.get("node_key")
        == "w4a16-tied-lm-head-topk-from-final-rmsnorm"
        and "build/model24_final_rmsnorm_attempt001"
        in str(mission.get("objective"))
        and "accepted" in str(mission.get("objective")).lower(),
        "live objective does not authorize this predecessor consumption",
    )
    raw_path = PREDECESSOR / "raw/final.hex"
    expected_raw = artifacts.get("raw/final.hex")
    require(
        expected_raw is not None
        and raw_path.stat().st_size == expected_raw.get("bytes")
        and sha256(raw_path) == expected_raw.get("sha256"),
        "final-RMSNorm raw output is not bound by its seal",
    )
    rows = raw_path.read_bytes().splitlines()
    require(len(rows) == FINAL_ROWS, "final-RMSNorm row count mismatch")
    values = np.empty((2, HIDDEN_SIZE), dtype="<u2")
    for ordinal, row in enumerate(rows):
        require(
            len(row) == 10 and re.fullmatch(rb"[0-9a-f]{10}", row) is not None,
            f"malformed final-RMSNorm row {ordinal}",
        )
        token = int(row[0:2], 16)
        feature = int(row[2:6], 16)
        require(
            token == ordinal // HIDDEN_SIZE
            and feature == ordinal % HIDDEN_SIZE,
            f"misordered final-RMSNorm row {ordinal}",
        )
        values[token, feature] = int(row[6:10], 16)
    hidden = values[-1].copy()
    hidden_payload = hidden.tobytes()
    return (
        {
            "root": str(PREDECESSOR),
            "seal": file_record(PREDECESSOR / "sealed_output_manifest.json"),
            "status": file_record(PREDECESSOR / "status.json"),
            "natural_terminal_gate": file_record(
                PREDECESSOR / "natural_terminal_gate.json"
            ),
            "integer_comparison": file_record(
                PREDECESSOR / "integer_comparison.json"
            ),
            "fp16_policy_comparison": file_record(
                PREDECESSOR / "fp16_policy_comparison.json"
            ),
            "raw_final": {
                **file_record(raw_path),
                "dtype": "F16",
                "shape": [2, HIDDEN_SIZE],
                "consumed_token_index": 1,
                "consumed_hidden_sha256": sha256_bytes(hidden_payload),
            },
            "artifact_review_status": status.get("review_status"),
            "consumption_authority": {
                "source": "live_operator_objective",
                "mission": file_record(MISSION),
                "node_key": mission["node_key"],
            },
        },
        hidden,
        before,
    )


def authenticate_prior_attempt() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    require(PRIOR_ATTEMPT.is_dir(), "official attempt001 is missing")
    before = snapshot(PRIOR_ATTEMPT)
    seal_path = PRIOR_ATTEMPT / "sealed_output_manifest.json"
    seal = load_json(seal_path)
    require(
        seal.get("kind") == "ace3_tied_lm_head_topk_attempt001_sealed_manifest"
        and seal.get("status") == "ENGINEERING_PASS_REVIEW_REQUIRED"
        and seal.get("attempt_id") == "model24_tied_lm_head_topk_attempt001"
        and seal.get("official_attempt") == 1,
        "official attempt001 seal identity is malformed",
    )
    artifacts = {
        row.get("relative_path"): row
        for row in seal.get("artifacts", [])
        if isinstance(row, dict) and isinstance(row.get("relative_path"), str)
    }
    require(
        len(artifacts) == seal.get("artifact_count"),
        "official attempt001 sealed artifact set is malformed or duplicated",
    )
    for relative, expected in artifacts.items():
        path = PRIOR_ATTEMPT / relative
        require(path.is_file(), f"attempt001 artifact missing: {relative}")
        require(
            path.stat().st_size == expected.get("bytes")
            and sha256(path) == expected.get("sha256"),
            f"attempt001 artifact changed: {relative}",
        )
    status = load_json(PRIOR_ATTEMPT / "status.json")
    require(
        status.get("status") == "PASS"
        and status.get("failure_count") == 0
        and status.get("natural_terminal") is True,
        "official attempt001 is not a complete numerical PASS",
    )
    return (
        {
            "root": str(PRIOR_ATTEMPT),
            "attempt_id": ATTEMPT_ID,
            "seal": file_record(seal_path),
            "status": file_record(PRIOR_ATTEMPT / "status.json"),
            "file_count": len(before),
            "ordered_file_set_sha256": snapshot_digest(before),
        },
        before,
    )


def authenticate_checkpoint() -> dict[str, Any]:
    require(CHECKPOINT.is_file(), "official checkpoint is missing")
    require(
        CHECKPOINT.stat().st_size == CHECKPOINT_BYTES,
        "official checkpoint byte count mismatch",
    )
    require(sha256(CHECKPOINT) == CHECKPOINT_SHA256, "checkpoint SHA256 mismatch")
    with CHECKPOINT.open("rb") as stream:
        header_size_raw = stream.read(8)
        require(len(header_size_raw) == 8, "truncated safetensors length")
        header_size = struct.unpack("<Q", header_size_raw)[0]
        header = json.loads(
            stream.read(header_size), object_pairs_hook=reject_duplicate_keys
        )
    data_base = 8 + header_size
    tensors: dict[str, dict[str, Any]] = {}
    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        require(name in header, f"official checkpoint missing {name}")
        record = header[name]
        begin, end = record["data_offsets"]
        tensor = {
            "name": name,
            "dtype": record["dtype"],
            "shape": record["shape"],
            "absolute_offset": data_base + begin,
            "bytes": end - begin,
        }
        require(
            tensor["dtype"] == "F16"
            and tensor["shape"] == [VOCAB_SIZE, HIDDEN_SIZE],
            f"official tensor geometry mismatch: {name}",
        )
        tensor["sha256"] = sha256_range(
            CHECKPOINT, tensor["absolute_offset"], tensor["bytes"]
        )
        require(
            tensor["sha256"] == TIED_WEIGHT_SHA256,
            f"official tied value mismatch: {name}",
        )
        tensors[name] = tensor
    require(
        tensors["model.embed_tokens.weight"]["absolute_offset"]
        != tensors["lm_head.weight"]["absolute_offset"],
        "official tied tensors unexpectedly share storage",
    )
    return {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "checkpoint": file_record(CHECKPOINT),
        "streamed_tensor": tensors["model.embed_tokens.weight"],
        "tied_peer": tensors["lm_head.weight"],
        "binding": (
            "distinct checkpoint tensors with identical authenticated FP16 values; "
            "RTL streams model.embed_tokens.weight as tied lm_head.weight"
        ),
    }


def harness_source() -> str:
    return r'''#include "Vace3_streaming_tied_lm_head_topk.h"
#include "verilated.h"

#include <cstdint>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

static uint64_t cycles = 0;

static void fail(const std::string& message) {
    throw std::runtime_error(message);
}

static std::map<std::string, uint64_t> read_config(const std::string& path) {
    std::ifstream input(path);
    if (!input) fail("cannot open config");
    std::map<std::string, uint64_t> result;
    std::string line;
    while (std::getline(input, line)) {
        const auto separator = line.find('=');
        if (separator == std::string::npos) fail("invalid config line");
        if (!result.emplace(
                line.substr(0, separator),
                std::stoull(line.substr(separator + 1))
            ).second) fail("duplicate config field");
    }
    return result;
}

static std::vector<uint16_t> read_hidden(const std::string& path) {
    std::ifstream input(path);
    if (!input) fail("cannot open hidden input");
    std::vector<uint16_t> result;
    std::string line;
    while (std::getline(input, line)) {
        if (line.size() != 4) fail("malformed hidden input");
        result.push_back(static_cast<uint16_t>(std::stoul(line, nullptr, 16)));
    }
    return result;
}

static void tick(Vace3_streaming_tied_lm_head_topk& top) {
    top.clk_i = 0;
    top.eval();
    top.clk_i = 1;
    top.eval();
    ++cycles;
}

static void idle(Vace3_streaming_tied_lm_head_topk& top) {
    top.clear_i = 0;
    top.start_valid_i = 0;
    top.hidden_valid_i = 0;
    top.hidden_index_i = 0;
    top.hidden_f16_i = 0;
    top.hidden_last_i = 0;
    top.hidden_end_i = 0;
    top.weight_valid_i = 0;
    top.weight_token_index_i = 0;
    top.weight_feature_index_i = 0;
    top.weight_f16_i = 0;
    top.weight_last_feature_i = 0;
    top.weight_last_token_i = 0;
    top.weight_end_i = 0;
    top.logit_ready_i = 0;
    top.top_ready_i = 0;
    top.done_ready_i = 0;
    top.eval();
}

static void durable_file(const std::string& path) {
    const int descriptor = open(path.c_str(), O_WRONLY | O_APPEND);
    if (descriptor < 0) fail("cannot reopen raw output for durability");
    if (fsync(descriptor) != 0) {
        close(descriptor);
        fail("cannot make partial raw output durable");
    }
    close(descriptor);
}

static void write_terminal(
    const std::string& path,
    bool natural,
    int exit_code,
    uint64_t hidden_count,
    uint64_t weight_count,
    uint64_t logit_count,
    uint64_t top_count
) {
    std::ofstream output(path, std::ios::trunc);
    if (!output) return;
    output << "schema=ace3_tied_lm_head_raw_v1"
           << " natural_terminal=" << (natural ? 1 : 0)
           << " exit_code=" << exit_code
           << " hidden_count=" << hidden_count
           << " weight_count=" << weight_count
           << " logit_count=" << logit_count
           << " top_count=" << top_count
           << " cycles=" << cycles << "\n";
    output.flush();
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    std::map<std::string, std::string> paths;
    for (int index = 1; index + 1 < argc; index += 2)
        paths[argv[index]] = argv[index + 1];
    for (const char* name : {
             "--checkpoint", "--config", "--hidden", "--raw-logits",
             "--raw-topk", "--terminal", "--fail-after-logits"
         })
        if (!paths.count(name)) {
            std::cerr << "missing argument " << name << "\n";
            return 2;
        }

    uint64_t hidden_count = 0;
    uint64_t weight_count = 0;
    uint64_t logit_count = 0;
    uint64_t top_count = 0;
    int descriptor = -1;
    void* mapping = MAP_FAILED;
    uint64_t mapped_bytes = 0;
    std::ofstream raw_logits;
    std::ofstream raw_topk;
    try {
        const auto config = read_config(paths["--config"]);
        const auto hidden = read_hidden(paths["--hidden"]);
        const uint64_t fail_after = std::stoull(paths["--fail-after-logits"]);
        if (hidden.size() != config.at("hidden_size"))
            fail("hidden vector count mismatch");
        descriptor = open(paths["--checkpoint"].c_str(), O_RDONLY);
        if (descriptor < 0) fail("cannot open checkpoint");
        struct stat status{};
        if (fstat(descriptor, &status) != 0 ||
            static_cast<uint64_t>(status.st_size) != config.at("checkpoint_bytes"))
            fail("checkpoint byte count mismatch");
        mapped_bytes = status.st_size;
        mapping = mmap(nullptr, status.st_size, PROT_READ, MAP_PRIVATE, descriptor, 0);
        if (mapping == MAP_FAILED) fail("cannot mmap checkpoint");
        const auto* bytes = static_cast<const uint8_t*>(mapping);
        const uint64_t weight_offset = config.at("weight_offset");
        const uint64_t weight_bytes = config.at("weight_bytes");
        if (weight_offset + weight_bytes > mapped_bytes)
            fail("weight range outside checkpoint");
        const auto* weights = bytes + weight_offset;

        raw_logits.open(paths["--raw-logits"], std::ios::trunc);
        raw_topk.open(paths["--raw-topk"], std::ios::trunc);
        if (!raw_logits || !raw_topk) fail("cannot open raw output");

        Vace3_streaming_tied_lm_head_topk top;
        idle(top);
        top.rst_ni = 0;
        tick(top);
        tick(top);
        top.rst_ni = 1;
        top.eval();
        if (!top.start_ready_o) fail("DUT not ready after reset");
        top.start_valid_i = 1;
        tick(top);
        top.start_valid_i = 0;
        for (uint32_t index = 0; index < hidden.size(); ++index) {
            if (!top.hidden_ready_o) fail("hidden channel stalled");
            top.hidden_index_i = index;
            top.hidden_f16_i = hidden[index];
            top.hidden_last_i = index + 1 == hidden.size();
            top.hidden_end_i = top.hidden_last_i;
            top.hidden_valid_i = 1;
            tick(top);
            ++hidden_count;
        }
        top.hidden_valid_i = 0;
        top.hidden_last_i = 0;
        top.hidden_end_i = 0;

        const uint32_t hidden_size = config.at("hidden_size");
        const uint32_t vocab_size = config.at("vocab_size");
        for (uint32_t token = 0; token < vocab_size; ++token) {
            for (uint32_t feature = 0; feature < hidden_size; ++feature) {
                if (!top.weight_ready_o) fail("weight channel stalled");
                const uint64_t offset =
                    (static_cast<uint64_t>(token) * hidden_size + feature) * 2;
                const uint16_t bits = static_cast<uint16_t>(weights[offset]) |
                    (static_cast<uint16_t>(weights[offset + 1]) << 8);
                top.weight_token_index_i = token;
                top.weight_feature_index_i = feature;
                top.weight_f16_i = bits;
                top.weight_last_feature_i = feature + 1 == hidden_size;
                top.weight_last_token_i =
                    (token + 1 == vocab_size) && top.weight_last_feature_i;
                top.weight_end_i = top.weight_last_token_i;
                top.weight_valid_i = 1;
                tick(top);
                ++weight_count;
            }
            top.weight_valid_i = 0;
            top.weight_last_feature_i = 0;
            top.weight_last_token_i = 0;
            top.weight_end_i = 0;
            if (!top.logit_valid_o || top.logit_token_index_o != token)
                fail("missing or misindexed logit");
            if (top.logit_saturation_o || top.invalid_operand_o || top.error_valid_o)
                fail("invalid or saturated official logit");
            raw_logits << std::dec << token << " "
                       << std::hex << std::setfill('0') << std::setw(4)
                       << static_cast<uint32_t>(top.logit_f16_o) << " "
                       << std::setw(8) << top.acc_q47_48_o[2]
                       << std::setw(8) << top.acc_q47_48_o[1]
                       << std::setw(8) << top.acc_q47_48_o[0] << "\n";
            ++logit_count;
            if (!raw_logits) fail("raw logit write failed");
            if (fail_after && logit_count == fail_after) {
                raw_logits.flush();
                durable_file(paths["--raw-logits"]);
                fail("injected failure after durable raw write");
            }
            top.logit_ready_i = 1;
            tick(top);
            top.logit_ready_i = 0;
        }

        for (uint32_t rank = 0; rank < config.at("top_k"); ++rank) {
            if (!top.top_valid_o || top.top_rank_o != rank)
                fail("missing or misindexed top-k output");
            raw_topk << std::dec << rank << " " << top.top_token_index_o << " "
                     << std::hex << std::setfill('0') << std::setw(4)
                     << static_cast<uint32_t>(top.top_logit_f16_o) << "\n";
            ++top_count;
            top.top_ready_i = 1;
            tick(top);
            top.top_ready_i = 0;
        }
        if (!top.done_valid_o) fail("missing completion output");
        top.done_ready_i = 1;
        tick(top);
        top.done_ready_i = 0;
        if (!top.start_ready_o || top.busy_o || top.error_valid_o)
            fail("DUT did not return idle");
        raw_logits.flush();
        raw_topk.flush();
        if (!raw_logits || !raw_topk) fail("raw output flush failed");
        raw_logits.close();
        raw_topk.close();
        write_terminal(
            paths["--terminal"], true, 0, hidden_count, weight_count,
            logit_count, top_count
        );
        std::cout << "ACE3_TIED_LM_HEAD_RTL_NATURAL_PASS hidden="
                  << hidden_count << " weights=" << weight_count
                  << " logits=" << logit_count << " top=" << top_count
                  << " cycles=" << cycles
                  << " oracle_read=0 synthesis=not_run ppa=not_measured"
                  << " fpga=not_run\n";
        munmap(mapping, mapped_bytes);
        close(descriptor);
        return 0;
    } catch (const std::exception& error) {
        raw_logits.flush();
        raw_topk.flush();
        raw_logits.close();
        raw_topk.close();
        write_terminal(
            paths["--terminal"], false, 2, hidden_count, weight_count,
            logit_count, top_count
        );
        if (mapping != MAP_FAILED) munmap(mapping, mapped_bytes);
        if (descriptor >= 0) close(descriptor);
        std::cerr << "ACE3_TIED_LM_HEAD_RTL_FAILURE " << error.what() << "\n";
        return 2;
    }
}
'''


def run_command(
    command: list[str],
    stdout_path: Path,
    stderr_path: Path,
    timeout: int,
) -> tuple[subprocess.CompletedProcess[bytes], float]:
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    elapsed = time.monotonic() - started
    write_bytes(stdout_path, completed.stdout)
    write_bytes(stderr_path, completed.stderr)
    return completed, elapsed


def compile_and_protocol() -> tuple[Path, dict[str, Any], dict[str, Any]]:
    verilator = shutil.which("verilator")
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    require(verilator is not None, "verilator is unavailable")
    require(iverilog is not None and vvp is not None, "Icarus tools are unavailable")
    harness = OUTPUT / "source/tied_lm_head_main.cpp"
    harness.write_text(harness_source(), encoding="ascii", newline="\n")
    object_dir = OUTPUT / "tmp/obj"
    command = [
        verilator,
        "--cc",
        "--exe",
        "--build",
        "-j",
        "8",
        "-Wno-fatal",
        "--top-module",
        TOP_MODULE,
        "--Mdir",
        str(object_dir),
        "-CFLAGS",
        "-O3",
        *(str(path) for path in RTL_SOURCES),
        str(harness),
    ]
    completed, elapsed = run_command(
        command,
        OUTPUT / "compile.stdout",
        OUTPUT / "compile.stderr",
        120,
    )
    require(completed.returncode == 0, "official-shape Verilator build failed")
    built = object_dir / f"V{TOP_MODULE}"
    require(built.is_file(), "Verilator executable is missing")
    binary = OUTPUT / f"compiled/V{TOP_MODULE}"
    shutil.copy2(built, binary)
    binary.chmod(0o500)
    compile_record = {
        "schema_version": 1,
        "kind": "ace3_tied_lm_head_official_shape_compile",
        "status": "PASS",
        "command": command,
        "exit_code": completed.returncode,
        "wall_seconds": elapsed,
        "tool": subprocess.run(
            [verilator, "--version"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "binary": file_record(binary),
        "stdout": file_record(OUTPUT / "compile.stdout"),
        "stderr": file_record(OUTPUT / "compile.stderr"),
    }
    write_json(OUTPUT / "compile.json", compile_record)

    protocol_binary = OUTPUT / "compiled/streaming_lm_head_protocol.vvp"
    protocol_compile = [
        iverilog,
        "-g2012",
        "-s",
        f"{TOP_MODULE}_tb",
        "-o",
        str(protocol_binary),
        *(str(path) for path in RTL_SOURCES),
        str(PROTOCOL_TB),
    ]
    protocol_compiled, compile_elapsed = run_command(
        protocol_compile,
        OUTPUT / "protocol_compile.stdout",
        OUTPUT / "protocol_compile.stderr",
        60,
    )
    require(protocol_compiled.returncode == 0, "protocol Icarus compile failed")
    protocol_run = [vvp, str(protocol_binary)]
    protocol_completed, run_elapsed = run_command(
        protocol_run,
        OUTPUT / "protocol_simulation.stdout",
        OUTPUT / "protocol_simulation.stderr",
        60,
    )
    require(
        protocol_completed.returncode == 0
        and b"STREAMING_LM_HEAD_PROTOCOL_PASS logits=5 top_k=3 four_state=2"
        in protocol_completed.stdout,
        "streaming protocol/tie-policy simulation failed",
    )
    protocol_record = {
        "schema_version": 1,
        "kind": "ace3_tied_lm_head_protocol_simulation",
        "status": "PASS",
        "compile_command": protocol_compile,
        "simulation_command": protocol_run,
        "compile_exit_code": protocol_compiled.returncode,
        "simulation_exit_code": protocol_completed.returncode,
        "compile_wall_seconds": compile_elapsed,
        "simulation_wall_seconds": run_elapsed,
        "tie_policy_checked": "equal logits use ascending token ID",
        "four_state_probes": 2,
        "stdout": file_record(OUTPUT / "protocol_simulation.stdout"),
        "stderr": file_record(OUTPUT / "protocol_simulation.stderr"),
    }
    write_json(OUTPUT / "protocol_simulation.json", protocol_record)
    return binary, compile_record, protocol_record


def simulator_command(
    binary: Path,
    raw_root: Path,
    fail_after_logits: int,
) -> list[str]:
    return [
        str(binary),
        "--checkpoint",
        str(CHECKPOINT),
        "--config",
        str(OUTPUT / "vectors/run.cfg"),
        "--hidden",
        str(OUTPUT / "vectors/hidden.hex"),
        "--raw-logits",
        str(raw_root / "logits.txt"),
        "--raw-topk",
        str(raw_root / "topk.txt"),
        "--terminal",
        str(raw_root / "terminal.txt"),
        "--fail-after-logits",
        str(fail_after_logits),
    ]


def negative_gate_probes(binary: Path) -> dict[str, Any]:
    strace = shutil.which("strace")
    require(strace is not None, "strace is required for simulator open tracing")
    raw_root = OUTPUT / "negative/raw"
    simulator = simulator_command(binary, raw_root, 1)
    trace = OUTPUT / "negative/open.trace"
    command = [
        strace,
        "-qq",
        "-f",
        "-e",
        "trace=openat",
        "-o",
        str(trace),
        *simulator,
    ]
    completed, elapsed = run_command(
        command,
        OUTPUT / "negative/simulation.stdout",
        OUTPUT / "negative/simulation.stderr",
        30,
    )
    terminal = raw_root / "terminal.txt"
    logits = raw_root / "logits.txt"
    require(completed.returncode == 2, "injected simulator failure did not fail")
    require(
        FAILURE_TERMINAL_RE.fullmatch(terminal.read_bytes()) is not None,
        "injected failure terminal was not fail-closed",
    )
    require(
        len(logits.read_bytes().splitlines()) == 1,
        "durable partial raw row was not preserved",
    )
    trace_payload = trace.read_bytes()
    require(trace_payload.isascii(), "open trace is not ASCII")
    lowered_trace = trace_payload.lower()
    forbidden = (b"/oracle/", b"comparison.json", b"expected")
    require(
        all(value not in lowered_trace for value in forbidden),
        "simulator opened an oracle/comparison artifact",
    )
    require(
        not (OUTPUT / "negative/comparison.json").exists(),
        "comparison ran after injected simulator failure",
    )
    duplicate = OUTPUT / "negative/duplicate_terminal.txt"
    payload = terminal.read_bytes().rstrip(b"\n") + b" natural_terminal=1\n"
    write_bytes(duplicate, payload)
    require(
        NATURAL_TERMINAL_RE.fullmatch(payload) is None
        and FAILURE_TERMINAL_RE.fullmatch(payload) is None,
        "duplicate terminal field was accepted",
    )
    record = {
        "schema_version": 1,
        "kind": "ace3_tied_lm_head_negative_terminal_gates",
        "status": "PASS",
        "injected_failure": {
            "command": command,
            "simulator_command": simulator,
            "exit_code": completed.returncode,
            "wall_seconds": elapsed,
            "partial_raw_rows": 1,
            "natural_terminal": False,
            "comparison_created": False,
        },
        "duplicate_terminal_rejected": True,
        "simulator_oracle_opens": 0,
        "terminal": file_record(terminal),
        "partial_raw": file_record(logits),
        "open_trace": file_record(trace),
        "duplicate_terminal": file_record(duplicate),
    }
    write_json(OUTPUT / "negative_gate_checks.json", record)
    return record


def run_official_simulation(binary: Path) -> dict[str, Any]:
    command = simulator_command(binary, OUTPUT / "raw", 0)
    write_json(
        OUTPUT / "execution_started.json",
        {
            "schema_version": 1,
            "kind": "ace3_tied_lm_head_execution_started",
            "attempt_id": ATTEMPT_ID,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "oracle_arguments": [],
        },
    )
    completed, elapsed = run_command(
        command,
        OUTPUT / "simulation.stdout",
        OUTPUT / "simulation.stderr",
        240,
    )
    record = {
        "schema_version": 1,
        "kind": "ace3_tied_lm_head_official_shape_simulation",
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "command": command,
        "exit_code": completed.returncode,
        "wall_seconds": elapsed,
        "stdout": file_record(OUTPUT / "simulation.stdout"),
        "stderr": file_record(OUTPUT / "simulation.stderr"),
        "simulator_oracle_arguments": [],
    }
    write_json(OUTPUT / "simulation.json", record)
    require(completed.returncode == 0, "official-shape RTL simulation failed")
    return record


def parse_raw_logits(path: Path) -> tuple[np.ndarray, list[int]]:
    rows = path.read_bytes().splitlines()
    require(len(rows) == VOCAB_SIZE, "raw logit coverage is incomplete")
    bits = np.empty(VOCAB_SIZE, dtype="<u2")
    accumulators: list[int] = []
    for token, row in enumerate(rows):
        fields = row.split(b" ")
        require(
            len(fields) == 3
            and fields[0] == str(token).encode("ascii")
            and re.fullmatch(rb"[0-9a-f]{4}", fields[1]) is not None
            and re.fullmatch(rb"[0-9a-f]{24}", fields[2]) is not None,
            f"malformed or misordered raw logit {token}",
        )
        bits[token] = int(fields[1], 16)
        accumulators.append(int(fields[2], 16))
    return bits, accumulators


def parse_raw_topk(path: Path) -> list[tuple[int, int, int]]:
    rows = path.read_bytes().splitlines()
    require(len(rows) == TOP_K, "raw Top-K coverage is incomplete")
    result: list[tuple[int, int, int]] = []
    for expected_rank, row in enumerate(rows):
        fields = row.split(b" ")
        require(
            len(fields) == 3
            and fields[0] == str(expected_rank).encode("ascii")
            and re.fullmatch(rb"[0-9]+", fields[1]) is not None
            and re.fullmatch(rb"[0-9a-f]{4}", fields[2]) is not None,
            f"malformed or misordered raw Top-K rank {expected_rank}",
        )
        token = int(fields[1])
        require(0 <= token < VOCAB_SIZE, "raw Top-K token is out of range")
        result.append((expected_rank, token, int(fields[2], 16)))
    require(
        len({token for _, token, _ in result}) == TOP_K,
        "raw Top-K contains duplicate tokens",
    )
    return result


def natural_terminal_gate(
    simulation: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray, list[int], list[tuple[int, int, int]]]:
    terminal = OUTPUT / "raw/terminal.txt"
    logits_path = OUTPUT / "raw/logits.txt"
    topk_path = OUTPUT / "raw/topk.txt"
    require(
        simulation.get("status") == "PASS"
        and simulation.get("exit_code") == 0
        and terminal.is_file()
        and logits_path.is_file()
        and topk_path.is_file(),
        "simulation did not reach the natural terminal gate",
    )
    terminal_payload = terminal.read_bytes()
    require(terminal_payload.isascii(), "terminal contains non-ASCII bytes")
    match = NATURAL_TERMINAL_RE.fullmatch(terminal_payload)
    require(match is not None, "terminal is missing, malformed, or ambiguous")
    bits, accumulators = parse_raw_logits(logits_path)
    topk = parse_raw_topk(topk_path)
    gate = {
        "schema_version": 1,
        "kind": "ace3_tied_lm_head_natural_terminal_gate",
        "status": "PASS",
        "natural_terminal": True,
        "actual_exit_code": simulation["exit_code"],
        "recorded_exit_code": 0,
        "hidden_count": HIDDEN_SIZE,
        "weight_count": EXPECTED_WEIGHTS,
        "logit_count": VOCAB_SIZE,
        "top_count": TOP_K,
        "reported_cycles": int(match.group(1)),
        "complete_streamed_vocabulary_coverage": True,
        "terminal": file_record(terminal),
        "raw_logits": file_record(logits_path),
        "raw_topk": file_record(topk_path),
        "simulation": file_record(OUTPUT / "simulation.json"),
    }
    write_json(OUTPUT / "natural_terminal_gate.json", gate)
    return gate, bits, accumulators, topk


def load_reference() -> Any:
    spec = importlib.util.spec_from_file_location(
        "ace3_tied_lm_head_independent_oracle", REFERENCE
    )
    require(spec is not None and spec.loader is not None, "cannot load oracle")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def oracle_comparison(
    hidden: np.ndarray,
    model: Mapping[str, Any],
    gate: Mapping[str, Any],
    actual_bits: np.ndarray,
    actual_accumulators: list[int],
    actual_topk: list[tuple[int, int, int]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    require(
        gate.get("natural_terminal") is True,
        "oracle comparison attempted before natural terminal",
    )
    reference = load_reference()
    hidden_q24 = reference.decode_array_q24(hidden)
    embedding = model["streamed_tensor"]
    mapped = np.memmap(
        CHECKPOINT,
        dtype="<u2",
        mode="r",
        offset=embedding["absolute_offset"],
        shape=(VOCAB_SIZE, HIDDEN_SIZE),
    )
    expected_bits = np.empty(VOCAB_SIZE, dtype="<u2")
    expected_accumulators: list[int] = []
    heap: list[tuple[int, int, int, int]] = []
    oracle_lines: list[str] = []
    logits_digest = hashlib.sha256()
    for begin in range(0, VOCAB_SIZE, 512):
        end = min(begin + 512, VOCAB_SIZE)
        weight_q24 = reference.decode_array_q24(mapped[begin:end])
        chunk = np.sum(weight_q24 * hidden_q24, axis=1, dtype=np.int64)
        for token, accumulator_value in enumerate(chunk, start=begin):
            accumulator = int(accumulator_value)
            bits, saturated = reference.fixed_to_f16(accumulator, 48)
            require(not saturated, f"oracle logit {token} saturated")
            value, finite, _ = reference.decode_f16_q24(bits)
            require(finite, f"oracle logit {token} is nonfinite")
            expected_bits[token] = bits
            expected_accumulators.append(accumulator & ((1 << 96) - 1))
            logits_digest.update(struct.pack("<H", bits))
            oracle_lines.append(
                f"{token} {bits:04x} {accumulator & ((1 << 96) - 1):024x}"
            )
            item = (value, -token, token, bits)
            if len(heap) < TOP_K:
                heapq.heappush(heap, item)
            elif item[:2] > heap[0][:2]:
                heapq.heapreplace(heap, item)
    write_lines(OUTPUT / "oracle/logits.txt", oracle_lines)
    winners = sorted(heap, key=lambda item: (-item[0], item[2]))
    expected_topk = [
        (rank, item[2], item[3], item[0])
        for rank, item in enumerate(winners)
    ]
    write_lines(
        OUTPUT / "oracle/topk.txt",
        (
            f"{rank} {token} {bits:04x} {value}"
            for rank, token, bits, value in expected_topk
        ),
    )
    del mapped

    bit_mismatches = np.flatnonzero(actual_bits != expected_bits)
    accumulator_mismatches = [
        token
        for token, (actual, expected) in enumerate(
            zip(actual_accumulators, expected_accumulators, strict=True)
        )
        if actual != expected
    ]
    expected_top_shape = [
        (rank, token, bits) for rank, token, bits, _ in expected_topk
    ]
    topk_mismatches = [
        rank
        for rank, (actual, expected) in enumerate(
            zip(actual_topk, expected_top_shape, strict=True)
        )
        if actual != expected
    ]
    selected_token = actual_topk[0][1]
    selected_bits = actual_topk[0][2]
    selected_logit_match = int(actual_bits[selected_token]) == selected_bits
    actual_order_values = [
        reference.decode_f16_q24(bits)[0] for _, _, bits in actual_topk
    ]
    tie_policy_valid = all(
        actual_order_values[index] > actual_order_values[index + 1]
        or (
            actual_order_values[index] == actual_order_values[index + 1]
            and actual_topk[index][1] < actual_topk[index + 1][1]
        )
        for index in range(TOP_K - 1)
    )
    failure_count = (
        int(bit_mismatches.size)
        + len(accumulator_mismatches)
        + len(topk_mismatches)
        + int(not selected_logit_match)
        + int(not tie_policy_valid)
    )
    first_mismatch: dict[str, Any] | None = None
    if bit_mismatches.size:
        token = int(bit_mismatches[0])
        first_mismatch = {
            "field": "logit_f16_bits",
            "token_id": token,
            "actual": f"{int(actual_bits[token]):04x}",
            "expected": f"{int(expected_bits[token]):04x}",
        }
    elif accumulator_mismatches:
        token = accumulator_mismatches[0]
        first_mismatch = {
            "field": "accumulator_q47_48",
            "token_id": token,
            "actual": f"{actual_accumulators[token]:024x}",
            "expected": f"{expected_accumulators[token]:024x}",
        }
    elif topk_mismatches:
        rank = topk_mismatches[0]
        first_mismatch = {
            "field": "top_k",
            "rank": rank,
            "actual": actual_topk[rank],
            "expected": expected_top_shape[rank],
        }
    elif not selected_logit_match:
        first_mismatch = {"field": "selected_token_logit_binding"}
    elif not tie_policy_valid:
        first_mismatch = {"field": "deterministic_top_k_order"}

    oracle = {
        "schema_version": 1,
        "kind": "ace3_tied_lm_head_exact_integer_fp16_policy_oracle",
        "status": "PASS" if failure_count == 0 else "FAIL",
        "source": file_record(REFERENCE),
        "independence": (
            "Python/NumPy exact Q24 decode, integer dot accumulation, and "
            "single FP16 RNE conversion; no RTL output is used to rank candidates"
        ),
        "geometry": {
            "hidden_size": HIDDEN_SIZE,
            "vocab_size": VOCAB_SIZE,
            "top_k": TOP_K,
        },
        "numeric_policy": {
            "operand_decode": "exact finite binary16 to signed Q16.24",
            "product": "exact signed Q32.48",
            "accumulation": "exact row dot product with no intermediate rounding",
            "row_rounding": "binary16 round-to-nearest ties-to-even",
            "top_k": "descending rounded-FP16 value, then ascending token ID",
        },
        "complete_logit_coverage": VOCAB_SIZE,
        "logits_sha256": logits_digest.hexdigest(),
        "selected_token_id": expected_topk[0][1],
        "selected_logit_f16_bits": f"{expected_topk[0][2]:04x}",
        "top_k_entries": [
            {
                "rank": rank,
                "token_id": token,
                "logit_f16_bits": f"{bits:04x}",
                "logit_q24": value,
            }
            for rank, token, bits, value in expected_topk
        ],
        "logits": file_record(OUTPUT / "oracle/logits.txt"),
        "topk": file_record(OUTPUT / "oracle/topk.txt"),
    }
    write_json(OUTPUT / "oracle/oracle.json", oracle)
    comparison = {
        "schema_version": 1,
        "kind": "ace3_tied_lm_head_exact_oracle_comparison",
        "status": "PASS" if failure_count == 0 else "FAIL",
        "natural_terminal_gate": file_record(
            OUTPUT / "natural_terminal_gate.json"
        ),
        "records_compared": VOCAB_SIZE,
        "complete_streamed_vocabulary_coverage": True,
        "logit_bit_mismatch_count": int(bit_mismatches.size),
        "accumulator_mismatch_count": len(accumulator_mismatches),
        "top_k_mismatch_count": len(topk_mismatches),
        "selected_token_id": selected_token,
        "selected_logit_f16_bits": f"{selected_bits:04x}",
        "selected_token_logit_match": selected_logit_match,
        "deterministic_tie_policy": "ascending token ID for equal rounded logits",
        "deterministic_top_k_order_valid": tie_policy_valid,
        "failure_count": failure_count,
        "first_material_mismatch": first_mismatch,
        "actual": {
            "logits": file_record(OUTPUT / "raw/logits.txt"),
            "topk": file_record(OUTPUT / "raw/topk.txt"),
        },
        "expected": {
            "oracle": file_record(OUTPUT / "oracle/oracle.json"),
            "logits": file_record(OUTPUT / "oracle/logits.txt"),
            "topk": file_record(OUTPUT / "oracle/topk.txt"),
        },
    }
    write_json(OUTPUT / "comparison.json", comparison)
    require(failure_count == 0, "official-shape RTL/oracle comparison failed")
    return oracle, comparison


def seal_attempt(
    predecessor_before: list[dict[str, Any]],
    prior_attempt_before: list[dict[str, Any]],
    simulation: Mapping[str, Any],
    gate: Mapping[str, Any],
    comparison: Mapping[str, Any],
    timings: Mapping[str, float],
) -> None:
    predecessor_after = snapshot(PREDECESSOR)
    preserved = predecessor_before == predecessor_after
    prior_attempt_after = snapshot(PRIOR_ATTEMPT)
    prior_attempt_preserved = prior_attempt_before == prior_attempt_after
    preservation = {
        "schema_version": 1,
        "kind": "ace3_tied_lm_head_predecessor_preservation",
        "status": "PASS" if preserved else "FAIL",
        "predecessor": str(PREDECESSOR),
        "before_file_count": len(predecessor_before),
        "after_file_count": len(predecessor_after),
        "before_ordered_file_set_sha256": snapshot_digest(predecessor_before),
        "after_ordered_file_set_sha256": snapshot_digest(predecessor_after),
        "predecessor_preserved": preserved,
        "prior_attempt001": {
            "root": str(PRIOR_ATTEMPT),
            "before_file_count": len(prior_attempt_before),
            "after_file_count": len(prior_attempt_after),
            "before_ordered_file_set_sha256": snapshot_digest(prior_attempt_before),
            "after_ordered_file_set_sha256": snapshot_digest(prior_attempt_after),
            "preserved": prior_attempt_preserved,
        },
    }
    write_json(OUTPUT / "preservation.json", preservation)
    require(preserved, "final-RMSNorm predecessor changed during execution")
    require(prior_attempt_preserved, "official attempt001 changed during execution")
    timing = {
        "schema_version": 1,
        "kind": "ace3_tied_lm_head_phase_timing",
        "clock": "time.monotonic",
        "phases_seconds": dict(timings),
        "official_simulation_wall_seconds": simulation["wall_seconds"],
        "reported_simulation_cycles": gate["reported_cycles"],
        "performance_claim": "measurement only; no bottleneck or hardware claim",
    }
    write_json(OUTPUT / "timing.json", timing)
    passed = (
        simulation.get("status") == "PASS"
        and gate.get("status") == "PASS"
        and comparison.get("status") == "PASS"
        and comparison.get("failure_count") == 0
        and comparison.get("first_material_mismatch") is None
        and comparison.get("selected_token_logit_match") is True
        and comparison.get("deterministic_top_k_order_valid") is True
        and preserved
        and prior_attempt_preserved
    )
    status = {
        "schema_version": 1,
        "kind": "ace3_tied_lm_head_topk_runtime_status",
        "status": "PASS" if passed else "FAIL",
        "attempt_id": ATTEMPT_ID,
        "official_attempt": OFFICIAL_ATTEMPT,
        "source_stage": "accepted_final_rmsnorm_token1",
        "operation": "tied_lm_head_top_k",
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "natural_terminal": gate["natural_terminal"],
        "simulation_exit_code": simulation["exit_code"],
        "complete_streamed_vocabulary_coverage": True,
        "accepted_weights": EXPECTED_WEIGHTS,
        "accepted_logits": VOCAB_SIZE,
        "ordered_top_k_entries": TOP_K,
        "selected_token_id": comparison["selected_token_id"],
        "selected_logit_f16_bits": comparison["selected_logit_f16_bits"],
        "failure_count": comparison["failure_count"],
        "first_material_mismatch": comparison["first_material_mismatch"],
        "predecessor_preserved": preserved,
        "prior_attempt001_preserved": prior_attempt_preserved,
        "review_status": "PENDING_INDEPENDENT_REVIEW",
        "advance_to_tokenizer_host_integration": False,
        "claim_boundary": {
            "demonstrated": (
                "computer-local official-shape streaming tied lm_head/Top-K RTL "
                "simulation from the authenticated final-RMSNorm token-1 output"
            ),
            "not_demonstrated": [
                "independent reviewer PASS",
                "tokenizer or host integration",
                "persistent multi-token K/V",
                "readable dialogue",
                "synthesis, PPA, bitstream, FPGA, or silicon",
            ],
        },
    }
    write_json(OUTPUT / "status.json", status)
    require(passed, "tied lm_head/Top-K engineering adjudication failed")
    write_json(
        OUTPUT / "adjudication.json",
        {
            "schema_version": 1,
            "kind": "ace3_tied_lm_head_topk_engineer_adjudication",
            "status": "ENGINEERING_PASS_REVIEW_REQUIRED",
            "attempt_id": ATTEMPT_ID,
            "failure_count": 0,
            "first_material_mismatch": None,
            "natural_terminal": True,
            "complete_streamed_vocabulary_coverage": True,
            "predecessor_preserved": True,
            "prior_attempt001_preserved": True,
            "review_requirement": {
                "required": True,
                "status": "PENDING",
                "independent_reviewer_must_authenticate": [
                    "sealed final-RMSNorm predecessor and consumed token-1 row",
                    "official tied embed_tokens/lm_head tensor binding",
                    "simulator command contains no oracle inputs",
                    "natural terminal and complete raw vocabulary coverage",
                    "exact integer/FP16-policy logit and accumulator comparison",
                    "selected token and ordered deterministic Top-K",
                    "negative terminal gates and predecessor preservation",
                ],
            },
            "advance_to_tokenizer_host_integration": False,
        },
    )
    artifacts = [
        file_record(path, OUTPUT)
        for path in sorted(OUTPUT.rglob("*"))
        if path.is_file() and path.name != "sealed_output_manifest.json"
    ]
    input_manifest = OUTPUT / "input_manifest.json"
    authenticated_inputs = load_json(input_manifest)
    write_json(
        OUTPUT / "sealed_output_manifest.json",
        {
            "schema_version": 1,
            "kind": "ace3_tied_lm_head_topk_attempt002_sealed_manifest",
            "status": "ENGINEERING_PASS_REVIEW_REQUIRED",
            "attempt_id": ATTEMPT_ID,
            "official_attempt": OFFICIAL_ATTEMPT,
            "operation": "tied_lm_head_top_k",
            "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_manifest": file_record(input_manifest, OUTPUT),
            "authenticated_design_records": authenticated_inputs["design_records"],
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
        },
    )
    for path in sorted(OUTPUT.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(0o400)
        elif path.is_dir():
            path.chmod(0o500)
    OUTPUT.chmod(0o500)


def main() -> int:
    require(Path.cwd().resolve() == ROOT, f"run from {ROOT}")
    require(not OUTPUT.exists(), f"fresh attempt already exists: {OUTPUT}")
    OUTPUT.mkdir(parents=True, mode=0o700)
    timings: dict[str, float] = {}
    try:
        for relative in (
            "vectors",
            "source",
            "compiled",
            "tmp",
            "raw",
            "oracle",
            "negative",
            "negative/raw",
        ):
            (OUTPUT / relative).mkdir()
        attempt = {
            "schema_version": 1,
            "kind": "ace3_tied_lm_head_topk_attempt",
            "attempt_id": ATTEMPT_ID,
            "official_attempt": OFFICIAL_ATTEMPT,
            "fresh": True,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "cwd": str(ROOT),
            "branch": "argus/full-projection",
            "host": platform.node(),
            "python": sys.version.split()[0],
            "scope": (
                "final-RMSNorm token-1 output through official tied lm_head and "
                "deterministic Top-K; no decoder-layer replay"
            ),
        }
        write_json(OUTPUT / "attempt.json", attempt)
        mission = load_json(MISSION)

        started = time.monotonic()
        predecessor, hidden, predecessor_before = authenticate_predecessor(mission)
        prior_attempt, prior_attempt_before = authenticate_prior_attempt()
        model = authenticate_checkpoint()
        timings["authentication"] = time.monotonic() - started
        write_lines(OUTPUT / "vectors/hidden.hex", (f"{int(bits):04x}" for bits in hidden))
        embedding = model["streamed_tensor"]
        write_lines(
            OUTPUT / "vectors/run.cfg",
            (
                f"checkpoint_bytes={CHECKPOINT_BYTES}",
                f"weight_offset={embedding['absolute_offset']}",
                f"weight_bytes={embedding['bytes']}",
                f"hidden_size={HIDDEN_SIZE}",
                f"vocab_size={VOCAB_SIZE}",
                f"top_k={TOP_K}",
            ),
        )
        write_json(
            OUTPUT / "predecessor_before.json",
            {
                "schema_version": 1,
                "root": str(PREDECESSOR),
                "file_count": len(predecessor_before),
                "ordered_file_set_sha256": snapshot_digest(predecessor_before),
                "files": predecessor_before,
            },
        )
        input_manifest = {
            "schema_version": 1,
            "kind": "ace3_tied_lm_head_topk_authenticated_inputs",
            "attempt_id": ATTEMPT_ID,
            "predecessor": predecessor,
            "prior_official_attempt": prior_attempt,
            "no_decoder_layer_replay": True,
            "consumed_final_rmsnorm_token_index": 1,
            "official_model": model,
            "geometry": {
                "hidden_size": HIDDEN_SIZE,
                "vocab_size": VOCAB_SIZE,
                "top_k": TOP_K,
                "streamed_weight_values": EXPECTED_WEIGHTS,
            },
            "sources": {
                "rtl": [file_record(path) for path in RTL_SOURCES],
                "protocol_tb": file_record(PROTOCOL_TB),
                "oracle": {
                    "path": str(REFERENCE),
                    "opened_only_after_natural_terminal_gate": True,
                },
            },
            "design_records": {
                "frozen_contract": file_record(FROZEN_CONTRACT),
                "rtl_manifest": file_record(DESIGN_MANIFEST),
                "human_traceability": file_record(RTL_TRACEABILITY),
            },
            "simulator_oracle_input_policy": "no expected or oracle artifact arguments",
        }
        write_json(OUTPUT / "input_manifest.json", input_manifest)

        started = time.monotonic()
        binary, _, _ = compile_and_protocol()
        timings["compile_and_protocol_simulation"] = time.monotonic() - started
        started = time.monotonic()
        negative_gate_probes(binary)
        timings["negative_gate_probes"] = time.monotonic() - started
        started = time.monotonic()
        simulation = run_official_simulation(binary)
        timings["official_rtl_simulation"] = time.monotonic() - started
        started = time.monotonic()
        gate, actual_bits, actual_accumulators, actual_topk = (
            natural_terminal_gate(simulation)
        )
        timings["natural_terminal_gate"] = time.monotonic() - started
        started = time.monotonic()
        _, comparison = oracle_comparison(
            hidden,
            model,
            gate,
            actual_bits,
            actual_accumulators,
            actual_topk,
        )
        timings["independent_oracle_and_comparison"] = time.monotonic() - started
        started = time.monotonic()
        seal_attempt(
            predecessor_before,
            prior_attempt_before,
            simulation,
            gate,
            comparison,
            timings,
        )
        timings["sealing"] = time.monotonic() - started
        print(
            "ACE3_TIED_LM_HEAD_TOPK_ATTEMPT002_ENGINEERING_PASS "
            f"vocab={VOCAB_SIZE} weights={EXPECTED_WEIGHTS} "
            f"selected_token={comparison['selected_token_id']} "
            f"selected_logit={comparison['selected_logit_f16_bits']} "
            "mismatches=0 natural_terminal=1 predecessor_preserved=1 "
            "review=pending tokenizer_host=not_started"
        )
        return 0
    except Exception as error:
        if OUTPUT.exists() and not (OUTPUT / "failure.json").exists():
            try:
                write_json(
                    OUTPUT / "failure.json",
                    {
                        "schema_version": 1,
                        "kind": "ace3_tied_lm_head_topk_attempt_failure",
                        "attempt_id": ATTEMPT_ID,
                        "status": "FAIL",
                        "error": str(error),
                        "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
            except Exception:
                pass
        print(f"TIED_LM_HEAD_TOPK_ATTEMPT_FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
