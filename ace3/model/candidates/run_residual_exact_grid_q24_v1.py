"""Select the isolated compile route or an explicitly admitted decoder adapter.

The default retains the isolated compile route. --decoder-route selects the
ordered actual decoder route with explicit independently admitted integration.
The decoder controller and codec do not supply a model-specific adapter or
software-screen admission. Missing either is an execution blocker, not a
successful no-op or permission to import a primitive/FP16-only model parent.
"""

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


ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / "ace3/model/candidates/build"
RTL = Path("ace3/rtl/candidates/residual_exact_grid_q24_v1")
MODEL = Path("ace3/model/candidates")
CONTRACTS = Path("ace3/contracts/candidates")
SOURCES = (
    RTL / "ace3_residual_exact_grid_q24_public.svh",
    RTL / "ace3_residual_exact_grid_q24_core.sv",
    RTL / "ace3_residual_exact_grid_q24_elab.sv",
    MODEL / "residual_exact_grid_q24_reference_v1.py",
    MODEL / "residual_exact_grid_q24_runtime_v1.py",
    MODEL / "run_residual_exact_grid_q24_v1.py",
    CONTRACTS / "residual_exact_grid_q24_v1.json",
    CONTRACTS / "residual_exact_grid_q24_state_v1.json",
    CONTRACTS / "residual_exact_grid_q24_policy_v1.json",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--decoder-route", type=Path,
                        help="source-bound integration module selected by independent review")
    parser.add_argument("--decoder-route-sha256",
                        help="separately trusted integration source SHA256")
    parser.add_argument("--software-admission", type=Path,
                        help="normal Host Reviewer receipt for the software/state screen")
    parser.add_argument("--software-admission-sha256")
    args = parser.parse_args()
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]*_attempt[0-9]+", args.attempt_id) is None:
        parser.error("attempt ID must be a path-free lowercase name ending in _attemptNNN")
    if args.decoder_route is not None:
        return decoder_route(args)
    if any((args.decoder_route_sha256, args.software_admission, args.software_admission_sha256)):
        parser.error("decoder admission arguments require --decoder-route")
    tools = {name: shutil.which(name) for name in ("iverilog", "verilator")}
    missing = [name for name, path in tools.items() if path is None]
    if missing:
        parser.error(f"host tools unresolved: {missing}; inspect declared local containers before claiming unavailable")
    attempt = BUILD / args.attempt_id
    attempt.mkdir(parents=True, exist_ok=False)
    frozen_root = attempt / "source"
    source_records = []
    for relative in SOURCES:
        source = ROOT / relative
        destination = frozen_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        destination.write_bytes(data)
        source_records.append({"path": str(relative), "sha256": hashlib.sha256(data).hexdigest()})
        if relative.suffix == ".json":
            json.loads(data)
        elif relative.suffix == ".py":
            compile(data, str(relative), "exec")

    include = frozen_root / RTL
    core = include / "ace3_residual_exact_grid_q24_core.sv"
    harness = include / "ace3_residual_exact_grid_q24_elab.sv"
    commands = [
        [sys.executable, "--version"],
        [tools["iverilog"], "-V"],
        [tools["verilator"], "--version"],
        [sys.executable, "-B", "-c",
         "import ace3.model.candidates.residual_exact_grid_q24_reference_v1 as oracle; "
         "import ace3.model.candidates.residual_exact_grid_q24_runtime_v1 as codec; "
         "print('IMPORT_ONLY: independent oracle and isolated codec; no arithmetic or codec calls')"],
        [tools["iverilog"], "-g2012", "-I", str(include),
         "-s", "ace3_residual_exact_grid_q24_elab", "-o", str(attempt / "elaboration.vvp"),
         str(core), str(harness)],
        [tools["verilator"], "--cc", "--top-module", "ace3_residual_exact_grid_q24_core",
         "-I" + str(include), "--Mdir", str(attempt / "obj_dir"), str(core)],
    ]
    freeze = {
        "attempt_id": args.attempt_id,
        "created_at": time.time(),
        "scope": "compile_import_elaboration_only",
        "sources": source_records,
        "commands": commands,
        "cwd": str(frozen_root),
        "environment": {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(frozen_root)},
        "score_policy": "route PASS iff every recorded command exits zero; no numerical inference",
        "official_test_inputs": [],
        "simulator_executions_permitted": 0,
    }
    (attempt / "freeze.json").write_text(json.dumps(freeze, indent=2) + "\n")
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(frozen_root))
    records = []
    for index, command in enumerate(commands):
        started = time.monotonic()
        completed = subprocess.run(command, cwd=frozen_root, env=environment,
                                   capture_output=True, text=True, check=False)
        log = attempt / f"command{index:02d}.log"
        log.write_text(completed.stdout + completed.stderr)
        records.append({
            "argv": command, "exit_code": completed.returncode, "log": str(log),
            "stdout": completed.stdout, "stderr": completed.stderr,
            "elapsed_seconds": time.monotonic() - started,
        })
    passed = all(record["exit_code"] == 0 for record in records)
    result = {
        "attempt_id": args.attempt_id,
        "route_executability": "PASS" if passed else "FAIL",
        "primitive_numerical_behavior": "NOT_EVALUATED",
        "primitive_protocol_behavior": "NOT_EVALUATED",
        "portable_state_behavior": "NOT_EVALUATED",
        "decoder_controller_model_tokenizer_token_kv": "NOT_EVALUATED",
        "hardware": "NOT_EVALUATED",
        "simulator_executions": 0,
        "commands": records,
        "independent_review": "PENDING_NORMAL_HOST_REVIEWER",
    }
    if not passed:
        result["failure_taxonomy"] = "isolated_route_compile_import_elaboration_failure"
        result["root_cause_hypothesis"] = "a frozen entrypoint is incompatible with the recorded host compiler or importer; inspect the failing command log"
        result["regression"] = "same literal public ABI/import route in a fresh repaired attempt; no vector simulation"
    (attempt / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "attempt_id": args.attempt_id,
        "route_executability": result["route_executability"],
        "result": str(attempt / "result.json"),
        "command_exit_codes": [record["exit_code"] for record in records],
        "numerical_status": "NOT_EVALUATED",
        "simulator_executions": 0,
    }, indent=2))
    return 0 if passed else 1


def decoder_route(args) -> int:
    from ace3.model.candidates.residual_exact_grid_q24_decoder_route_v1 import run_root_through_l8

    for path, expected in (
        (args.decoder_route, args.decoder_route_sha256),
        (args.software_admission, args.software_admission_sha256),
    ):
        if path is None or expected is None or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError("decoder route requires separately bound source and software review")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("decoder route source/software review digest mismatch")
    review = json.loads(args.software_admission.read_text())
    if (review.get("producer_role") != "reviewer" or
            review.get("kind") != "round_reviewed_handoff" or
            review.get("review", {}).get("status") != "done"):
        raise ValueError("decoder RTL requires completed independent software/state review")
    # A generic done receipt is insufficient: bind the exact admitted adapter.
    bindings = review.get("review", {}).get("artifact_bindings", {})
    if bindings.get("q24_decoder_route_sha256") != args.decoder_route_sha256:
        raise ValueError("software review does not admit this decoder integration source")
    attempt = ROOT / "build" / args.attempt_id
    attempt.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("ace3_bound_q24_decoder_route", args.decoder_route)
    if spec is None or spec.loader is None:
        raise ValueError("cannot import bound Q24 integration module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    integration = module.open_runtime(attempt)
    parent = run_root_through_l8(
        integration.root, integration.execute, integration.admit, integration.persist)
    with (attempt / "l8_parent.json").open("x") as output:
        output.write(parent.text)
    print(json.dumps({"scope": "P0/L0-L8", "parent_sha256": parent.sha256,
                      "next_layer": 9, "review": "PENDING_NORMAL_HOST_REVIEWER"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
