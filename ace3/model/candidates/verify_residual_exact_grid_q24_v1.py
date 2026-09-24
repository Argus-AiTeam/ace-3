"""Fresh, contract-frozen isolated Q24 semantic attempt; no model execution."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / "ace3/model/candidates/build"
RTL = Path("ace3/rtl/candidates/residual_exact_grid_q24_v1")
MODEL = Path("ace3/model/candidates")
CONTRACTS = Path("ace3/contracts/candidates")
TEST = Path("ace3/model/tests/test_residual_exact_grid_q24_v1.py")
SOURCES = (
    RTL / "ace3_residual_exact_grid_q24_public.svh",
    RTL / "ace3_residual_exact_grid_q24_core.sv",
    RTL / "ace3_residual_exact_grid_q24_elab.sv",
    RTL / "ace3_residual_exact_grid_q24_tb.sv",
    MODEL / "residual_exact_grid_q24_reference_v1.py",
    MODEL / "residual_exact_grid_q24_runtime_v1.py",
    MODEL / "verify_residual_exact_grid_q24_v1.py",
    TEST,
    *(CONTRACTS / name for name in (
        "residual_exact_grid_q24_v1.json", "residual_exact_grid_q24_state_v1.json",
        "residual_exact_grid_q24_policy_v1.json",
        "residual_exact_grid_q24_primitive_regression_v1.json")),
)
LEGACY = (
    *(MODEL / (name + ".py") for name in (
        "decoder_gate_policy", "decoder_gate_policy_v3", "local_operator_reference_v3",
        "runtime_admission_v3", "binary64_fp16_excess_v1", "run_single_round_residual_rtl")),
    *(CONTRACTS / name for name in (
        "decoder_gate_policy_v2.json", "decoder_gate_policy_v3.json",
        "binary64_fp16_excess_v1.json")),
    Path("ace3/model/tests/test_decoder_gate_policy.py"),
    Path("ace3/model/tests/test_decoder_gate_policy_v3.py"),
    MODEL / "run_residual_exact_grid_q24_v1.py",
)
PREVIOUS = BUILD / "q24_contract_runtime_05f93e5fa139_attempt001"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, document):
    with path.open("x") as output:
        json.dump(document, output, indent=2)
        output.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-id", required=True)
    args = parser.parse_args()
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]*_attempt[0-9]+", args.attempt_id) is None:
        parser.error("expected a path-free attempt ID ending in _attemptNNN")
    tools = {name: shutil.which(name) for name in ("iverilog", "vvp")}
    if not all(tools.values()):
        parser.error("host tools unresolved; inspect declared local containers before claiming unavailable")
    attempt = BUILD / args.attempt_id
    attempt.mkdir(parents=True, exist_ok=False)
    frozen = attempt / "source"
    snapshots = {}
    for relative in SOURCES + LEGACY:
        destination = frozen / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = (ROOT / relative).read_bytes()
        with destination.open("xb") as output:
            output.write(data)
        snapshots[str(relative)] = hashlib.sha256(data).hexdigest()
    preserved = {str(PREVIOUS / name): digest(PREVIOUS / name)
                 for name in ("freeze.json", "result.json")}
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    records = []

    def execute(name, argv, cwd=frozen):
        selected_env = dict(environment, PYTHONPATH=str(cwd))
        sidecar = {"argv": argv, "cwd": str(cwd), "timeout_seconds": 90,
                   "environment": {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(cwd)}}
        save_json(attempt / f"{name}.command.json", sidecar)
        with (attempt / f"{name}.command.sh").open("x") as output:
            output.write("cd " + shlex.quote(str(cwd)) + " && " +
                         "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=" + shlex.quote(str(cwd)) +
                         " " + shlex.join(argv) + "\n")
        started = time.monotonic()
        timed_out = False
        try:
            completed = subprocess.run(argv, cwd=cwd, env=selected_env,
                                       capture_output=True, text=True, timeout=90, check=False)
            code, stdout, stderr = completed.returncode, completed.stdout, completed.stderr
        except subprocess.TimeoutExpired as error:
            timed_out = True
            code = -1
            stdout = error.stdout or b""
            stderr = error.stderr or b""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            stderr += "\nTIMEOUT: command exceeded frozen 90-second limit\n"
        with (attempt / f"{name}.log").open("x") as output:
            output.write(stdout + stderr)
        record = dict(sidecar, name=name, exit_code=code, timed_out=timed_out,
                      elapsed_seconds=time.monotonic() - started,
                      log=str(attempt / f"{name}.log"))
        records.append(record)
        return code, stdout + stderr

    versions = {}
    for name, argv in (
        ("python", [sys.executable, "--version"]),
        ("iverilog", [tools["iverilog"], "-V"]),
        ("vvp", [tools["vvp"], "-V"]),
    ):
        code, output = execute("version_" + name, argv)
        versions[name] = {"exit_code": code, "output": output}
    contract = json.loads((frozen / CONTRACTS / "residual_exact_grid_q24_primitive_regression_v1.json").read_text())
    save_json(attempt / "freeze.json", {
        "attempt_id": args.attempt_id, "created_at": time.time(),
        "scope": "isolated_primitive_reference_codec",
        "contract": contract, "sources": snapshots, "tools": versions,
        "preserved_compile_only_attempt": preserved,
        "commands_recorded_before_execution": True,
        "score_policy": contract["score_policy"],
        "review": "PENDING_NORMAL_HOST_REVIEWER",
    })
    include = frozen / RTL
    core = include / "ace3_residual_exact_grid_q24_core.sv"
    common = [tools["iverilog"], "-g2012", "-I", str(include)]
    elaboration, _ = execute("public_elaboration", common + [
        "-s", "ace3_residual_exact_grid_q24_elab", "-o", str(attempt / "public.vvp"),
        str(core), str(include / "ace3_residual_exact_grid_q24_elab.sv")])
    compiled, _ = execute("primitive_compile", common + [
        "-s", "ace3_residual_exact_grid_q24_tb", "-o", str(attempt / "primitive.vvp"),
        str(core), str(include / "ace3_residual_exact_grid_q24_tb.sv")])
    vectors = attempt / "vectors.txt"
    generated, _ = execute("vectors", [
        sys.executable, "-B", "-m", "ace3.model.tests.test_residual_exact_grid_q24_v1",
        "--vectors", str(vectors)])
    input_digest = digest(vectors) if generated == 0 else None
    row_count = int(vectors.read_text().splitlines()[0]) if generated == 0 else 0
    save_json(attempt / "inputs.json", {
        "file": str(vectors), "sha256": input_digest, "rows": row_count,
        "generator": snapshots[str(TEST)], "seed": contract["expectations"]["seed"],
        "expectations": "struct binary16 codec plus exact integer sums, not candidate output",
    })
    unit_code, unit_output = execute("primitive_python", [
        sys.executable, "-B", "-m", "unittest", "-v",
        "ace3.model.tests.test_residual_exact_grid_q24_v1"])
    legacy_code, legacy_output = execute("legacy_v2_v3_python", [
        sys.executable, "-B", "-m", "unittest", "-v",
        "ace3.model.tests.test_decoder_gate_policy",
        "ace3.model.tests.test_decoder_gate_policy_v3"], cwd=ROOT)
    launched = elaboration == compiled == generated == 0
    sim_code, sim_output = (None, "")
    if launched:
        sim_code, sim_output = execute("primitive_simulation", [
            tools["vvp"], str(attempt / "primitive.vvp"),
            "+vectors=" + str(vectors), "+wave=" + str(attempt / "protocol.vcd")])
    expected_marker = (f"Q24_PRIMITIVE_PASS rows={row_count} pipeline_pairs=1792 "
                       "recurrence_steps=8 xz_payloads=6 control_diagnostics=15")
    semantic_pass = sim_code == 0 and sim_output.splitlines().count(expected_marker) == 1
    bindings = all(digest(ROOT / path) == value and digest(frozen / path) == value
                   for path, value in snapshots.items())
    bindings = bindings and all(digest(Path(path)) == value for path, value in preserved.items())
    bindings = bindings and generated == 0 and digest(vectors) == input_digest
    unit_pass = unit_code == 0 and re.search(r"Ran 7 tests\b", unit_output) is not None
    legacy_pass = legacy_code == 0 and re.search(r"Ran 13 tests\b", legacy_output) is not None
    passed = (semantic_pass and unit_pass and legacy_pass and bindings and
              all(record["exit_code"] == 0 for record in records))
    result = {
        "attempt_id": args.attempt_id, "status": "PASS" if passed else "FAIL",
        "primitive_numerical_protocol": "PASS" if semantic_pass else
            ("FAIL" if launched else "NOT_EVALUATED"),
        "reference_codec_tests": "PASS" if unit_pass else "FAIL",
        "legacy_v2_v3_tests": "PASS" if legacy_pass else "FAIL",
        "source_input_and_history_bindings_unchanged": bindings,
        "simulator_executions": int(launched), "vector_rows": row_count,
        "semantic_marker": expected_marker if semantic_pass else None,
        "decoder_controller_model_tokenizer_token_kv": "NOT_EVALUATED",
        "hardware": "NOT_EVALUATED", "independent_review": "PENDING_NORMAL_HOST_REVIEWER",
        "commands": records,
    }
    if not passed:
        result.update(
            failure_taxonomy="primitive_regression_failure" if launched else "evaluator_no_execution",
            root_cause_hypothesis="a frozen arithmetic/protocol expectation, Python check, or input binding disagrees; see the first failing command/marker",
            regression="retain this attempt; repair only the demonstrated source defect and use a fresh attempt ID")
    save_json(attempt / "result.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "commands"}, indent=2))
    print("result=" + str(attempt / "result.json"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
