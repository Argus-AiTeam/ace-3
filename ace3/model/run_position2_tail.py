#!/usr/bin/env python3
"""Execute a fresh, raw-output-only tail from the position-2 binding package."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
RMSNORM_COMPARISON = {
    "reference": "qwen2_fp16_interstage_final_rmsnorm",
    "recurrence": "FP32 mean-square/rsqrt; FP16 normalized activation; FP16 weight product",
    "acceptance": "finite AND (absolute <= 0.125 OR (relative < 0.001 AND ordered_FP16_ULP <= 1))",
    "relative_denominator": "max(abs(reference), 2^-14)",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write(path: Path, value: object) -> None:
    with path.open("x", encoding="ascii") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def record(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def authenticate(item: dict) -> None:
    require(record(Path(item["path"])) == {
        key: item[key] for key in ("path", "bytes", "sha256")
    }, f"artifact drift: {item['path']}")


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def command(output: Path, name: str, argv: list[str]) -> dict:
    write(output / f"{name}.command.json", {"argv": argv, "cwd": str(ROOT)})
    started = time.monotonic()
    with (output / f"{name}.stdout").open("xb") as stdout, (
        output / f"{name}.stderr"
    ).open("xb") as stderr:
        completed = subprocess.run(argv, cwd=ROOT, stdout=stdout, stderr=stderr,
                                   check=False)
    result = {"exit_code": completed.returncode,
              "wall_seconds": time.monotonic() - started,
              "command": record(output / f"{name}.command.json"),
              "stdout": record(output / f"{name}.stdout"),
              "stderr": record(output / f"{name}.stderr")}
    write(output / f"{name}.exit.json", result)
    require(completed.returncode == 0, f"{name} exited {completed.returncode}")
    return result


RMS_HARNESS = r'''#include "Vace3_fp16_rmsnorm_core.h"
#include "verilated.h"
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

static uint64_t cycles = 0;
static void fail(const std::string& text) {
    std::cerr << text << "\n"; std::exit(2);
}
static std::vector<uint16_t> read_hex(const std::string& path) {
    std::ifstream file(path);
    if (!file) fail("cannot open " + path);
    std::vector<uint16_t> values;
    std::string line;
    while (std::getline(file, line)) {
        if (line.size() != 4 || line.find_first_not_of("0123456789abcdef") != std::string::npos)
            fail("malformed FP16 input");
        values.push_back(static_cast<uint16_t>(std::stoul(line, nullptr, 16)));
    }
    if (!file.eof() || values.size() != 896) fail("input count/read failure");
    return values;
}
static void tick(Vace3_fp16_rmsnorm_core& top) {
    top.clk_i = 0; top.eval(); top.clk_i = 1; top.eval(); ++cycles;
}
int main(int argc, char** argv) {
    if (argc != 5) fail("usage: rms INPUT WEIGHTS OUTPUT TERMINAL");
    Verilated::commandArgs(argc, argv);
    const auto inputs = read_hex(argv[1]), weights = read_hex(argv[2]);
    std::ofstream output(argv[3]);
    if (!output) fail("cannot open raw output");
    Vace3_fp16_rmsnorm_core top;
    top.clear_i = 0; top.start_valid_i = 0; top.element_count_i = 896;
    top.in_valid_i = 0; top.activation_f16_i = 0; top.weight_f16_i = 0;
    top.out_ready_i = 0; top.rst_ni = 0;
    tick(top); tick(top); top.rst_ni = 1; tick(top);
    top.start_valid_i = 1; top.eval();
    if (!top.start_ready_o) fail("start not ready");
    tick(top); top.start_valid_i = 0;
    for (unsigned i = 0; i < 896; ++i) {
        top.activation_f16_i = inputs[i]; top.weight_f16_i = weights[i];
        top.in_valid_i = 1; top.eval();
        if (!top.in_ready_o) fail("input not ready");
        tick(top);
    }
    top.in_valid_i = 0;
    unsigned waits = 0;
    while (!top.out_valid_o && waits <= 46) { tick(top); ++waits; }
    if (waits != 46 || top.rms_q24_o == 0) fail("square root protocol failure");
    const uint64_t root = top.rms_q24_o;
    for (unsigned i = 0; i < 896; ++i) {
        top.eval();
        if (!top.out_valid_o || top.out_index_o != i ||
            top.out_last_o != (i == 895) || top.invalid_operand_o || top.saturation_o)
            fail("output protocol/status failure");
        output << std::hex << std::setfill('0') << std::setw(4)
               << static_cast<unsigned>(top.out_f16_o) << "\n";
        top.out_ready_i = 1; tick(top); top.out_ready_i = 0;
    }
    top.eval();
    if (top.out_valid_o || !top.start_ready_o) fail("RMSNorm did not return idle");
    output.close();
    if (!output) fail("raw output write failure");
    std::ofstream terminal(argv[4]);
    terminal << "natural_terminal=1 exit_code=0 input_count=896 output_count=896"
             << " transaction_count=1 cycles=" << cycles << " rms_q24=" << root << "\n";
    terminal.close();
    if (!terminal) fail("terminal write failure");
    top.final();
    std::cout << "POSITION2_FINAL_RMSNORM_NATURAL_EXIT cycles=" << cycles << "\n";
    return 0;
}
'''


def build(output: Path, contract: dict, key: str, source: Path,
          rtl: list[Path], compiler: str) -> Path:
    top = contract[key]["top"]
    obj = output / f"{key}_obj"
    argv = [compiler, "--cc", "--exe", "--build", "-Wno-fatal",
            "--top-module", top, "--Mdir", str(obj)]
    argv += [f"-G{name}={value}"
             for name, value in contract[key]["parameters"].items()]
    command(output, f"{key}_compile", argv + [str(p) for p in rtl] + [str(source)])
    binary = obj / f"V{top}"
    require(binary.is_file(), f"compiled binary missing: {top}")
    write(output / f"{key}_binary.json", record(binary))
    return binary


def validate_rmsnorm_contract(contract: dict) -> None:
    require(contract["final_rmsnorm"].get("comparison") == RMSNORM_COMPARISON,
            "final RMSNorm contract/reference mapping mismatch")


def compare_rmsnorm(preparation, actual, expected, contract: dict) -> dict:
    validate_rmsnorm_contract(contract)
    return preparation.fp16_interstage_comparison(actual, expected)


def admit_execution(preparation, package: dict, package_dir: Path, contract: dict) -> None:
    validate_rmsnorm_contract(contract)
    preparation.admit(package, package_dir, contract)
    require(contract["admission"].get("unchanged_tail_replay_authorized") is True,
            "unchanged_tail_replay_not_authorized; historical authority remains consumed")


def rms_expectation(preparation, output: Path, bits: list[int],
                    tensors: dict, contract: dict) -> tuple[list[int], dict]:
    np = preparation.np
    norm = tensors["model.norm.weight"]
    weights = np.memmap(preparation.CHECKPOINT, dtype="<u2", mode="r",
                        offset=norm["offset"], shape=(896,))
    inputs = np.asarray([bits], dtype="<u2")
    interstage = preparation.evidence.fp16_interstage_expected(inputs, weights)[0]
    expected = preparation.independent_rmsnorm(bits, weights.tolist())
    production, mean, divisor = preparation.fixed.rmsnorm(bits, weights.tolist())
    require(not any(row[1] or row[2] for row in production), "RMSNorm invalid/saturated")
    require([row[0] for row in production] == expected,
            "independent RMSNorm integer mismatch")
    current = compare_rmsnorm(preparation, np.asarray(expected, dtype="<u2"),
                              interstage, contract)
    require(current["failure_count"] == 0, "FP16-interstage RMSNorm material mismatch")
    x = inputs[0].view("<f2").astype(np.float64)
    w = weights.view("<f2").astype(np.float64)
    mathematical = (x / np.sqrt(np.mean(x * x) + 1e-6) * w).astype("<f2").view("<u2")
    for name, values in (("tail_input.hex", bits), ("norm_weight.hex", weights),
                         ("final_rmsnorm_expected.hex", expected),
                         ("fp16_interstage_rmsnorm_expected.hex", interstage),
                         ("mathematical_rmsnorm_expected.hex", mathematical)):
        with (output / name).open("x", encoding="ascii") as stream:
            stream.writelines(f"{int(raw):04x}\n" for raw in values)
    return expected, {
        "integer_mismatches": 0, "fp16_interstage_comparison": current,
        "mathematical_comparison_role": "legacy diagnostic only; never an admission gate",
        "mean_q48": mean, "rms_q24": divisor,
        "input_semantic_sha256": preparation.head.sha256_bytes(
            preparation.head.terminal_payload(bits)),
        "output_semantic_sha256": preparation.head.sha256_bytes(
            preparation.head.terminal_payload(expected)),
    }


def execute(package_dir: Path, output: Path) -> int:
    require(output.parent == ROOT / "build", "output must be directly under build/")
    output.mkdir(exist_ok=False)
    phase = "package_admission"
    result = {"status": "EVALUATOR_NO_EXECUTION", "runtime_tail_invocations": 0,
              "natural_exit": False, "independent_review": "pending Host Reviewer",
              "claim_boundary": "Computer-local W4A16 position-2 tail only; no "
              "decoder revalidation, dialogue, synthesis, PPA, FPGA or hardware claim."}
    try:
        write(output / "invocation.command.json",
              {"argv": [sys.executable, "-B", *sys.argv], "cwd": str(ROOT)})
        for suffix in (".command.sh", ".submit.command.sh"):
            source = output.with_name(output.name + suffix)
            shutil.copyfile(source, output / source.name)
        package = load(package_dir / "runtime_pass_package.json")
        package_seal = load(package_dir / "seal.json")
        for item in package_seal["files"]:
            authenticate(item)
        frozen_sources = output / "sources"
        for item in package["sources"]:
            source = Path(item["frozen"]["path"])
            relative = source.relative_to(package_dir / "sources")
            destination = frozen_sources / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        for relative in ("ace3/model/run_position2_tail.py",
                         "ace3/model/run_tied_lm_head_topk_from_final_rmsnorm.py"):
            shutil.copyfile(ROOT / relative, frozen_sources / relative)
        for name in ("runtime_pass_package.json", "frozen.json", "seal.json"):
            shutil.copyfile(package_dir / name, output / f"parent_{name}")
        sys.path.insert(0, str(frozen_sources / "ace3/model"))
        import numpy as np
        import prepare_position2_tail_binding as preparation

        preparation.PARENT = ROOT / "build/layer23_runtime_launch_attempt002"
        preparation.LAYER = ROOT / "build/model24_corrected_q_layers22_23_attempt001/layer23"
        preparation.CHECKPOINT = Path(package["checkpoint"]["path"])
        contract = load(frozen_sources / "ace3/contracts/position2_tail_binding.json")
        admit_execution(preparation, package, package_dir, contract)
        source = frozen_sources / "ace3/model/run_tied_lm_head_topk_from_final_rmsnorm.py"
        spec = importlib.util.spec_from_file_location("tail_head_driver", source)
        require(spec is not None and spec.loader is not None, "cannot load head driver")
        head_driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(head_driver)
        compiler = shutil.which("verilator")
        require(compiler is not None, "host Verilator unavailable; inspect declared local containers")
        command(output, "verilator_version", [compiler, "--version"])
        write(output / "frozen.json", {
            "package": record(package_dir / "runtime_pass_package.json"),
            "parent": package["parent"], "contract": contract,
            "public_interfaces": load(package_dir / "frozen.json")["public_interfaces"],
            "sources": [record(p) for p in sorted(frozen_sources.rglob("*")) if p.is_file()],
            "compiler": record(Path(compiler).resolve()), "python": sys.version,
            "numpy": np.__version__, "torch": preparation.head.torch.__version__,
            "environment": {key: os.environ.get(key) for key in
                            ("PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "LD_PRELOAD")},
            "score_policy": "exact accepted-policy FP16 RMSNorm, all rounded logits, "
                            "selected exact accumulators, deterministic Top-K and selected token; "
                            "active FP16-interstage RMSNorm gate from frozen contract; "
                            "legacy mathematical RMSNorm comparison is diagnostic only"})
        phase = "independent_rmsnorm_oracle"
        oracle_dir = output / "oracle"
        oracle_dir.mkdir()
        bits = [int(line, 16) for line in (package_dir / "tail_input.hex").read_text().splitlines()]
        expected, rms_oracle = rms_expectation(
            preparation, oracle_dir, bits, package["tensors"], contract)
        write(output / "rmsnorm_oracle.json", rms_oracle)
        raw = output / "raw"
        raw.mkdir()
        rms_source = output / "rmsnorm_main.cpp"
        with rms_source.open("x", encoding="ascii") as stream:
            stream.write(RMS_HARNESS)
        head_source = output / "lm_head_main.cpp"
        with head_source.open("x", encoding="ascii") as stream:
            stream.write(head_driver.harness_source())
        rtl = frozen_sources / "ace3/rtl"
        phase = "compile_public_contract"
        rms_binary = build(output, contract, "final_rmsnorm", rms_source,
                           [rtl / "ace3_fp16_fixed.sv", rtl / "ace3_fp16_rmsnorm_core.sv"],
                           compiler)
        head_binary = build(output, contract, "tied_head", head_source,
                            [rtl / name for name in ("ace3_fp16_fixed.sv",
                             "ace3_q47_48_to_f16_rne.sv", "ace3_streaming_tied_lm_head_topk.sv")],
                            compiler)
        write(output / "authenticated_inputs.json", {
            "layer23_terminal": package["parent"]["terminal"],
            "rmsnorm_input": record(oracle_dir / "tail_input.hex"),
            "norm_weight": record(oracle_dir / "norm_weight.hex"),
            "checkpoint": package["checkpoint"], "tensors": package["tensors"],
            "generated_harnesses": [record(rms_source), record(head_source)]})
        phase = "final_rmsnorm_runtime"
        result["runtime_tail_invocations"] = 1
        command(output, "final_rmsnorm_runtime", [str(rms_binary),
                str(oracle_dir / "tail_input.hex"), str(oracle_dir / "norm_weight.hex"),
                str(raw / "final_rmsnorm.hex"), str(raw / "final_rmsnorm.terminal")])
        terminal = (raw / "final_rmsnorm.terminal").read_text()
        match = re.fullmatch(r"natural_terminal=1 exit_code=0 input_count=896 output_count=896 "
                             r"transaction_count=1 cycles=(\d+) rms_q24=(\d+)\n", terminal)
        require(match is not None, "RMSNorm natural terminal missing/malformed")
        require(int(match[2]) == rms_oracle["rms_q24"], "RMSNorm divisor mismatch")
        lines = (raw / "final_rmsnorm.hex").read_text().splitlines()
        require(len(lines) == 896 and all(re.fullmatch("[0-9a-f]{4}", row) for row in lines),
                "RMSNorm output shape/encoding mismatch")
        actual = np.asarray([int(row, 16) for row in lines], dtype="<u2")
        differences = np.flatnonzero(actual != expected)
        interstage = np.asarray([int(row, 16) for row in
            (oracle_dir / "fp16_interstage_rmsnorm_expected.hex").read_text().splitlines()], dtype="<u2")
        current = compare_rmsnorm(preparation, actual, interstage, contract)
        # Preserve the historical mathematical comparison as a diagnostic, not a gate.
        mathematical = np.asarray([int(row, 16) for row in
            (oracle_dir / "mathematical_rmsnorm_expected.hex").read_text().splitlines()], dtype="<u2")
        absolute = np.abs(actual.view("<f2").astype(np.float64) -
                          mathematical.view("<f2").astype(np.float64))
        relative = absolute / np.maximum(np.abs(mathematical.view("<f2").astype(np.float64)), 1e-30)
        ulp = np.asarray([abs(preparation.evidence.ordered_f16(int(a)) -
                             preparation.evidence.ordered_f16(int(b)))
                          for a, b in zip(actual, mathematical, strict=True)])
        material = np.flatnonzero((absolute > .125) & (relative > .001) & (ulp > 1))
        rms_receipt = {"natural_exit": True, "cycles": int(match[1]),
                       "integer_mismatches": len(differences),
                       "material_mismatches": current["failure_count"],
                       "fp16_interstage_comparison": current,
                       "active_reference": record(oracle_dir / "fp16_interstage_rmsnorm_expected.hex"),
                       "legacy_mathematical_diagnostic": {
                           "role": "diagnostic only; never an admission gate",
                           "reference": record(oracle_dir / "mathematical_rmsnorm_expected.hex"),
                           "finite": bool(np.isfinite(mathematical.view("<f2")).all()),
                           "material_mismatches": len(material)},
                       "raw_output": record(raw / "final_rmsnorm.hex"),
                       "actual_output_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
                       "oracle": record(output / "rmsnorm_oracle.json")}
        write(output / "final_rmsnorm.json", rms_receipt)
        if len(differences) or current["failure_count"]:
            indices = list(differences)
            if current["failure_count"]:
                indices.append(current["first_material_mismatch_index"])
            index = int(min(indices))
            result.update(status="TAIL_MISMATCH", natural_exit=True,
                          earliest_tail_mismatch={"stage": "final_rmsnorm", "index": index,
                           "actual_bits": int(actual[index]), "expected_bits": int(expected[index]),
                           "fp16_interstage_expected_bits": int(interstage[index])},
                          failure_taxonomy="tail_numerical_mismatch",
                          root_cause_hypothesis="RMSNorm RTL differs from the independently recomputed policy",
                          regression="same authenticated layer23 vector against frozen Decimal RMSNorm")
            return 2
        phase = "lm_head_runtime"
        weights = package["tensors"]["model.embed_tokens.weight"]
        config = {"checkpoint_bytes": package["checkpoint"]["bytes"],
                  "weight_offset": weights["offset"], "weight_bytes": weights["bytes"],
                  "hidden_size": 896, "vocab_size": 151936, "top_k": 10}
        config_path = output / "head.cfg"
        with config_path.open("x", encoding="ascii") as stream:
            stream.writelines(f"{key}={value}\n" for key, value in config.items())
        write(output / "head_input.json", {"actual_rmsnorm": record(raw / "final_rmsnorm.hex"),
                                         "config": record(config_path), "weights": weights})
        command(output, "lm_head_runtime", [str(head_binary), "--checkpoint",
                package["checkpoint"]["path"], "--config", str(config_path),
                "--hidden", str(raw / "final_rmsnorm.hex"), "--raw-logits", str(raw / "logits.txt"),
                "--raw-topk", str(raw / "topk.txt"), "--terminal", str(raw / "head.terminal"),
                "--fail-after-logits", "0"])
        head_terminal = (raw / "head.terminal").read_text()
        match = re.fullmatch(r"schema=ace3_tied_lm_head_raw_v1 natural_terminal=1 exit_code=0 "
                             r"hidden_count=896 weight_count=136134656 logit_count=151936 "
                             r"top_count=10 cycles=(\d+)\n", head_terminal)
        require(match is not None, "lm_head natural terminal missing/malformed")
        raw_bits, raw_accumulators = head_driver.parse_raw_logits(raw / "logits.txt")
        raw_topk = head_driver.parse_raw_topk(raw / "topk.txt")
        authenticate(package["checkpoint"])
        phase = "independent_lm_head_oracle"
        reference, torch = preparation.reference, preparation.head.torch
        torch.set_num_threads(1)
        hidden_q24 = reference.decode_array_q24(actual)
        mapped = np.memmap(preparation.CHECKPOINT, dtype="<u2", mode="r",
                           offset=weights["offset"], shape=(151936, 896))
        hidden_f64 = torch.from_numpy(actual.view("<f2").astype(np.float64))
        expected_bits = np.empty(151936, dtype="<u2")
        expected_acc = np.empty(151936, dtype=np.int64)
        independent_bits = np.empty(151936, dtype="<u2")
        for begin in range(0, 151936, 512):
            end = min(begin + 512, 151936)
            chunk = np.asarray(mapped[begin:end], dtype="<u2")
            decoded = reference.decode_array_q24(chunk)
            bound = int(np.max(np.abs(decoded))) * sum(abs(int(v)) for v in hidden_q24)
            require(bound < (1 << 63), "integer oracle accumulation bound exceeded")
            expected_acc[begin:end] = np.sum(decoded * hidden_q24, axis=1, dtype=np.int64)
            for offset, accumulator in enumerate(expected_acc[begin:end]):
                value, saturated = reference.fixed_to_f16(int(accumulator), 48)
                require(not saturated, "oracle logit saturation")
                expected_bits[begin + offset] = value
            values = torch.from_numpy(chunk.view("<f2").astype(np.float64)).matmul(hidden_f64).numpy()
            independent_bits[begin:end] = values.astype("<f2").view("<u2")
        require(np.array_equal(expected_bits, independent_bits),
                "independent float64 and accepted-policy head oracles disagree")
        with (oracle_dir / "logits.u16").open("xb") as stream:
            stream.write(independent_bits.tobytes())
        top = preparation.head.topk_from_bits(independent_bits, reference)
        expected_topk = [(rank, token, value) for rank, (token, value, _) in enumerate(top)]
        selected = sorted(set(reference.selected_tokens(rms_receipt["actual_output_sha256"])) |
                          {token for _, token, _ in expected_topk})
        checks = [{"token_id": token, "actual_bits": int(raw_bits[token]),
                   "expected_bits": int(independent_bits[token]),
                   "actual_accumulator_hex": f"{raw_accumulators[token]:024x}",
                   "expected_accumulator_hex": f"{int(expected_acc[token]) & ((1 << 96) - 1):024x}"}
                  for token in selected]
        acc_failures = [row["token_id"] for row in checks
                        if row["actual_accumulator_hex"] != row["expected_accumulator_hex"]]
        logit_failures = np.flatnonzero(raw_bits != independent_bits)
        top_failures = [rank for rank in range(10) if raw_topk[rank] != expected_topk[rank]]
        write(output / "lm_head_oracle.json", {
            "independent": "PyTorch CPU float64 on authenticated FP16 weights and actual RTL RMSNorm",
            "accepted_policy": "exact Q47.48 sum, one binary16 RNE; independent bit agreement",
            "integer_vs_float64_mismatches": 0, "logit_count": 151936,
            "oracle_output": record(oracle_dir / "logits.u16"),
            "hidden": record(raw / "final_rmsnorm.hex")})
        authenticate(package["checkpoint"])
        write(output / "lm_head.json", {"natural_exit": True, "cycles": int(match[1]),
              "logit_mismatches": len(logit_failures), "selected_accumulator_mismatches": len(acc_failures),
              "selected_checks": checks, "raw_logits": record(raw / "logits.txt")})
        write(output / "topk.json", {"actual": raw_topk, "expected": expected_topk,
              "mismatch_count": len(top_failures), "selection_policy": package["selection_policy"]})
        write(output / "selected_token.json", {"actual_token_id": raw_topk[0][1],
              "actual_logit_bits": raw_topk[0][2], "expected_token_id": expected_topk[0][1],
              "expected_logit_bits": expected_topk[0][2], "matches": raw_topk[0] == expected_topk[0],
              "parent_layer23": package["parent"]["terminal"], "tail_input": rms_receipt["raw_output"]})
        mismatch = None
        if len(logit_failures) or acc_failures:
            token = int(min([*logit_failures, *acc_failures]))
            mismatch = {"stage": "lm_head", "token_id": token,
                        "actual_bits": int(raw_bits[token]), "expected_bits": int(independent_bits[token]),
                        "actual_accumulator_hex": f"{raw_accumulators[token]:024x}",
                        "expected_accumulator_hex": f"{int(expected_acc[token]) & ((1 << 96) - 1):024x}"}
        elif top_failures:
            rank = top_failures[0]
            mismatch = {"stage": "topk", "rank": rank, "actual": raw_topk[rank],
                        "expected": expected_topk[rank]}
        result.update(status="TAIL_MISMATCH" if mismatch else "TAIL_PASS_REVIEW_REQUIRED",
                      natural_exit=True, earliest_tail_mismatch=mismatch,
                      final_rmsnorm_material_mismatches=current["failure_count"],
                      logit_mismatches=len(logit_failures), topk_mismatches=len(top_failures),
                      selected_accumulator_mismatches=len(acc_failures),
                      selected_token_id=raw_topk[0][1])
        if mismatch:
            result.update(failure_taxonomy="tail_numerical_mismatch",
                          root_cause_hypothesis=f"{mismatch['stage']} RTL differs from frozen numerical policy",
                          regression="authenticated actual-RMSNorm-fed full-vocabulary tail comparison")
        return 2 if mismatch else 0
    except (OSError, ValueError, KeyError, RuntimeError, ImportError, subprocess.SubprocessError) as error:
        result.update(status="TAIL_EXECUTION_FAILURE" if result["runtime_tail_invocations"]
                      else "EVALUATOR_NO_EXECUTION", error=str(error),
                      failure_taxonomy="execution_or_admission_failure",
                      root_cause_hypothesis=f"{phase}: {error}",
                      regression="fresh source-bound package admission, public compilation and tail execution")
        return 2
    finally:
        result["terminal_phase"] = phase
        write(output / "result.json", result)
        write(output / "seal.json", {"files": [record(p) for p in sorted(output.rglob("*"))
                                               if p.is_file()]})
        for path in output.rglob("*"):
            if path.is_file():
                path.chmod(0o555 if os.access(path, os.X_OK) else 0o444)
        for path in sorted(output.rglob("*"), reverse=True):
            if path.is_dir():
                path.chmod(0o555)
        output.chmod(0o555)
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(execute(args.package.resolve(), args.output.resolve()))
