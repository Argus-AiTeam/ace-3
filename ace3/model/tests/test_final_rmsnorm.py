#!/usr/bin/env python3
"""Executable acceptance for the standalone final-RMSNorm boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

HIDDEN_SIZE = 896
OFFICIAL_SHA256 = "c50d807b7bed7ff314308972e0f4bcf4e5a70bc60ad88fc7df53940831ed0c1b"


def run(command: list[str], log_dir: Path, label: str, expected_success: bool = True) -> int:
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{label}.command").write_text(shlex.join(command), encoding="utf-8")
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    (log_dir / f"{label}.stdout").write_text(result.stdout, encoding="utf-8")
    (log_dir / f"{label}.stderr").write_text(result.stderr, encoding="utf-8")
    (log_dir / f"{label}.status").write_text(f"{result.returncode}\n", encoding="ascii")
    if expected_success != (result.returncode == 0):
        raise RuntimeError(
            f"unexpected status for {label}: {result.returncode}\n{result.stdout}\n{result.stderr}"
        )
    return result.returncode


def assert_no_oracle_open(trace_path: Path, manifest: Path, expected: Path) -> None:
    trace = trace_path.read_text(encoding="utf-8", errors="replace")
    for forbidden in (str(manifest), str(expected)):
        if forbidden in trace:
            raise RuntimeError(f"invalid evidence gate opened oracle artifact: {forbidden}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--contract-only", action="store_true")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"immutable run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    logs = run_dir / "logs"
    rtl = [
        root / "ace3/rtl/ace3_fp16_fixed.sv",
        root / "ace3/rtl/ace3_fp16_rmsnorm_core.sv",
        root / "ace3/rtl/ace3_final_rmsnorm.sv",
    ]
    contract = json.loads((root / "ace3/contracts/final_rmsnorm.json").read_text())
    if contract["public_top"]["module"] != "ace3_final_rmsnorm":
        raise RuntimeError("public top mismatch")
    if contract["public_top"]["parameters"] != [] or len(contract["public_top"]["ports"]) != 18:
        raise RuntimeError("public parameter/port contract mismatch")
    contract_vvp = run_dir / "contract.vvp"
    run(
        ["iverilog", "-g2012", "-s", "ace3_final_rmsnorm", "-o", str(contract_vvp), *map(str, rtl)],
        logs,
        "contract_compile",
    )
    if args.contract_only:
        (run_dir / "summary.json").write_text(
            json.dumps({"contract_compile": "pass", "public_top": "ace3_final_rmsnorm"}) + "\n"
        )
        return 0

    checkpoint_before = args.checkpoint.stat()
    vectors = run_dir / "vectors"
    model = root / "ace3/model/final_rmsnorm_model.py"
    compare = root / "ace3/model/final_rmsnorm_compare.py"
    run(
        [sys.executable, str(model), "--checkpoint", str(args.checkpoint), "--output-dir", str(vectors)],
        logs,
        "generate_vectors",
    )
    run(
        [sys.executable, str(model), "--checkpoint", str(run_dir / "missing.safetensors"), "--output-dir", str(run_dir / "missing-output")],
        logs,
        "malformed_missing_checkpoint",
        expected_success=False,
    )
    run(
        [sys.executable, str(model), "--checkpoint", str(args.checkpoint), "--output-dir", str(run_dir / "wrong-digest-output"), "--expected-sha256", "0" * 64],
        logs,
        "malformed_checkpoint_digest",
        expected_success=False,
    )
    checkpoint_after = args.checkpoint.stat()
    if (checkpoint_before.st_ino, checkpoint_before.st_size, checkpoint_before.st_mtime_ns, checkpoint_before.st_mode) != (
        checkpoint_after.st_ino,
        checkpoint_after.st_size,
        checkpoint_after.st_mtime_ns,
        checkpoint_after.st_mode,
    ):
        raise RuntimeError("checkpoint metadata changed during acceptance")

    iverilog_vvp = run_dir / "icarus" / "ace3_final_rmsnorm.vvp"
    iverilog_vvp.parent.mkdir()
    run(
        ["iverilog", "-g2012", "-s", "ace3_final_rmsnorm_tb", "-o", str(iverilog_vvp), *map(str, rtl), str(root / "ace3/tb/ace3_final_rmsnorm_tb.sv")],
        logs,
        "icarus_build",
    )
    manifest = vectors / "manifest.json"
    expected = vectors / "expected.memh"
    activations = vectors / "activations.memh"
    weights = vectors / "weights.memh"
    icarus_raw = run_dir / "icarus" / "case0.raw"
    icarus_terminal = run_dir / "icarus" / "case0.terminal"
    run(
        ["vvp", str(iverilog_vvp), f"+ACTIVATIONS={activations}", f"+WEIGHTS={weights}", f"+RAW={icarus_raw}", f"+TERMINAL={icarus_terminal}", "+CASE=0"],
        logs,
        "icarus_case0",
    )
    icarus_report = run_dir / "icarus" / "case0.report.json"
    run(
        [sys.executable, str(compare), "--terminal", str(icarus_terminal), "--raw", str(icarus_raw), "--manifest", str(manifest), "--expected", str(expected), "--report", str(icarus_report), "--simulator-exit-code", "0"],
        logs,
        "icarus_compare",
    )

    fail_raw = run_dir / "icarus" / "fail-after-one.raw"
    fail_terminal = run_dir / "icarus" / "fail-after-one.terminal"
    fail_rc = run(
        ["vvp", str(iverilog_vvp), f"+ACTIVATIONS={activations}", f"+WEIGHTS={weights}", f"+RAW={fail_raw}", f"+TERMINAL={fail_terminal}", "+CASE=0", "+FAIL_AFTER=1"],
        logs,
        "icarus_injected_failure",
        expected_success=False,
    )
    if len(fail_raw.read_text(encoding="ascii").splitlines()) != 1:
        raise RuntimeError("injected-failure raw row did not survive")
    if "natural_terminal=0" not in fail_terminal.read_text(encoding="ascii"):
        raise RuntimeError("injected-failure terminal is not marked unnatural")
    fail_report = run_dir / "icarus" / "fail-after-one.report.json"
    fail_trace = run_dir / "icarus" / "fail-after-one.strace"
    run(
        ["strace", "-f", "-e", "trace=openat", "-o", str(fail_trace), sys.executable, str(compare), "--terminal", str(fail_terminal), "--raw", str(fail_raw), "--manifest", str(manifest), "--expected", str(expected), "--report", str(fail_report), "--simulator-exit-code", str(fail_rc)],
        logs,
        "gate_injected_failure",
        expected_success=False,
    )
    if fail_report.exists():
        raise RuntimeError("comparison report exists for unnatural simulator completion")
    assert_no_oracle_open(fail_trace, manifest, expected)

    duplicate_terminal = run_dir / "icarus" / "duplicate.terminal"
    duplicate_terminal.write_text(
        icarus_terminal.read_text(encoding="ascii") + "natural_terminal=1\n", encoding="ascii"
    )
    duplicate_report = run_dir / "icarus" / "duplicate.report.json"
    duplicate_trace = run_dir / "icarus" / "duplicate.strace"
    run(
        ["strace", "-f", "-e", "trace=openat", "-o", str(duplicate_trace), sys.executable, str(compare), "--terminal", str(duplicate_terminal), "--raw", str(icarus_raw), "--manifest", str(manifest), "--expected", str(expected), "--report", str(duplicate_report), "--simulator-exit-code", "0"],
        logs,
        "gate_duplicate_terminal",
        expected_success=False,
    )
    if duplicate_report.exists():
        raise RuntimeError("comparison report exists for duplicate terminal")
    assert_no_oracle_open(duplicate_trace, manifest, expected)

    truncated_raw = run_dir / "icarus" / "truncated.raw"
    truncated_raw.write_text("\n".join(icarus_raw.read_text(encoding="ascii").splitlines()[:-1]) + "\n", encoding="ascii")
    run(
        [sys.executable, str(compare), "--terminal", str(icarus_terminal), "--raw", str(truncated_raw), "--manifest", str(manifest), "--expected", str(expected), "--report", str(run_dir / "icarus" / "truncated.report.json"), "--simulator-exit-code", "0"],
        logs,
        "gate_truncated_raw",
        expected_success=False,
    )

    mdir = run_dir / "verilator" / "obj_dir"
    mdir.parent.mkdir()
    run(
        ["verilator", "-Wall", "-Wno-fatal", "--cc", "--exe", "--build", "--top-module", "ace3_final_rmsnorm", "--Mdir", str(mdir), *map(str, rtl), str(root / "ace3/tb/ace3_final_rmsnorm_main.cpp"), "-CFLAGS", "-std=c++17", "-o", "ace3_final_rmsnorm_sim"],
        logs,
        "verilator_build",
    )
    simulator = mdir / "ace3_final_rmsnorm_sim"
    verilator_comparisons = 0
    for case_index in range(4):
        raw = run_dir / "verilator" / f"case{case_index}.raw"
        terminal = run_dir / "verilator" / f"case{case_index}.terminal"
        report = run_dir / "verilator" / f"case{case_index}.report.json"
        run(
            [str(simulator), "--activations", str(activations), "--weights", str(weights), "--raw", str(raw), "--terminal", str(terminal), "--case", str(case_index)],
            logs,
            f"verilator_case{case_index}",
        )
        run(
            [sys.executable, str(compare), "--terminal", str(terminal), "--raw", str(raw), "--manifest", str(manifest), "--expected", str(expected), "--report", str(report), "--simulator-exit-code", "0"],
            logs,
            f"verilator_compare_case{case_index}",
        )
        verilator_comparisons += json.loads(report.read_text())["comparisons"]

    if len(icarus_raw.read_text(encoding="ascii").splitlines()) != HIDDEN_SIZE:
        raise RuntimeError("Icarus did not cover all 896 four-state outputs")
    if verilator_comparisons != 4 * HIDDEN_SIZE:
        raise RuntimeError("Verilator comparison count mismatch")
    tracked_package = [
        root / "ace3/rtl/ace3_final_rmsnorm.sv",
        root / "ace3/contracts/final_rmsnorm.json",
        root / "ace3/model/final_rmsnorm_model.py",
        root / "ace3/model/final_rmsnorm_compare.py",
        root / "ace3/model/tests/test_final_rmsnorm.py",
        root / "ace3/tb/ace3_final_rmsnorm_tb.sv",
        root / "ace3/tb/ace3_final_rmsnorm_main.cpp",
        root / "docs/FINAL_RMSNORM.md",
    ]
    forbidden = (
        "ACE" + "-2",
        "/home/" + "argustest/ace3-argus/" + "model24_execution_vectors/model.safetensors",
    )
    for path in tracked_package:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                raise RuntimeError(f"tracked package hygiene violation in {path}: {token}")
    summary = {
        "schema": "ace3-final-rmsnorm-acceptance-v1",
        "checkpoint_sha256": OFFICIAL_SHA256,
        "contract_compile": "pass",
        "malformed_input_checks": 5,
        "icarus_four_state_outputs": HIDDEN_SIZE,
        "icarus_oracle_comparisons": HIDDEN_SIZE,
        "verilator_oracle_comparisons": verilator_comparisons,
        "mismatches": 0,
        "natural_terminal_gate": "pass",
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
