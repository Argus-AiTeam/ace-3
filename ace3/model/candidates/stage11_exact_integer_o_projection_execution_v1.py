"""Mission-bound one-use Stage11 suffix entry; --preflight never claims or computes."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

import numpy as np

from ace3.model.candidates import stage11_exact_integer_o_projection_contrast_v1 as library


ROOT, require, pin = library.ROOT, library.require, library.pin
SOURCE = Path(__file__).resolve()
NAME = SOURCE.stem
TEST = ROOT / f"tests/test_{NAME}.py"
HANDOFFS = library.retained.HANDOFFS
SOFTWARE_MISSION = "d12e372181f0"
NODE_KEY = "stage11-exact-integer-o-projection-execution"


def project_sources():
    return library.project_sources() | {str(SOURCE), str(TEST)}


def command(handoff, sha256):
    return shlex.join([
        "python3", "-B", "-m", "ace3.model.candidates." + NAME, "--execute",
        "--handoff", str(handoff), "--sha256", sha256,
    ]) + "\n"


def authenticate_sources(records):
    require(isinstance(records, list)
            and all(isinstance(r, dict) and set(r) == {"path", "bytes", "sha256"} for r in records),
            "complete source byte pins")
    require(len(library.project_sources()) == 62 and len(project_sources()) == 64
            and [r["path"] for r in records] == sorted(project_sources()), "exact source/test membership")
    for record in records:
        require(type(record["bytes"]) is int and record["bytes"] > 0
                and isinstance(record["sha256"], str)
                and re.fullmatch("[0-9a-f]{64}", record["sha256"]) is not None, "source byte pin shape")
        compile(library.retained.selection.bound(record), record["path"], "exec")


def preflight(handoff, sha256):
    require(Path.cwd() == ROOT and os.getuid() == 1000 and sys.dont_write_bytecode
            and not sys.flags.optimize, "isolated source/account/bytecode context")
    require(subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
        == "argus/full-projection", "isolated branch")
    require(re.fullmatch("[0-9a-f]{64}", sha256) is not None, "handoff SHA256")
    document = json.loads(library.retained.selection.bound({"path": str(handoff), "sha256": sha256}))
    mission_id = document["mission_id"]
    require(isinstance(mission_id, str) and re.fullmatch("[0-9a-f]{12}", mission_id)
            and mission_id not in {SOFTWARE_MISSION, library.BASELINE_MISSION,
                                   library.silu.STAGE17_MISSION, library.gate_up.MISSION,
                                   library.norm.MISSION}, "fresh science mission identity")
    require(handoff == HANDOFFS / mission_id / "stage11-exact-integer-execution.json"
            and handoff.resolve() == handoff, "Manager handoff path")
    require(type(document["schema_version"]) is int and document["schema_version"] == 1
            and type(document["science_budget"]) is int and document["science_budget"] == 1,
            "one-use handoff schema/budget")
    mission_pin = document["mission"]
    require(mission_pin["path"] == str(HANDOFFS / mission_id / "mission.json"), "native mission path")
    mission = json.loads(library.retained.selection.bound(mission_pin))
    require(mission["mission_id"] == mission_id and mission["scope"] == "bounded"
            and mission["node_key"] == NODE_KEY and mission["execution_workdir"] == str(ROOT)
            and mission["deps"] == [] and "review:required" in mission["tags"]
            and not mission.get("authorization_id") and not mission.get("action"),
            "science owner/dependency preflight")
    review_pin = document["review"]
    review_path = Path(review_pin["path"])
    require(review_path.parent == HANDOFFS / SOFTWARE_MISSION
            and re.fullmatch(r"round-[0-9]{4}\.json", review_path.name), "software review path")
    review = json.loads(library.retained.selection.bound(review_pin))
    require(review["mission_id"] == SOFTWARE_MISSION and review["producer_role"] == "reviewer"
            and review["review"]["status"] == "done", "independent software acceptance")
    authenticate_sources(document["source_pins"])
    out = Path(document["output"])
    require(out.is_absolute() and out.resolve() == out and out.is_relative_to(ROOT / "build")
            and out.name == f"{NAME}-{mission_id}-attempt001", "canonical isolated output binding")
    require(document["output_claim"] == str(out / "command.json"), "canonical output-claim binding")
    require(subprocess.run(["git", "check-ignore", "--quiet", str(out)], cwd=ROOT).returncode == 0,
            "output must be ignored")
    require(out.is_dir() and {p.name for p in out.iterdir()} == {"command.sh"},
            "fresh Manager-provisioned output missing or occupied")
    sidecar = out / "command.sh"
    require(sidecar.resolve() == sidecar and sidecar.is_file()
            and sidecar.read_bytes() == command(handoff, sha256).encode("ascii"), "command/root binding")
    audit = library.base.new_audit()
    with library.guarded(audit):
        values = library.inputs()
    closure, _, reference, refs, operands, tensors, _, records, prepared = values
    require(audit["writes"] == audit["forbidden_events"] == audit["stage_dispatches"] == [],
            "preflight must be read-only")
    document["input_pins"] = [pin(handoff), mission_pin, review_pin, pin(sidecar), *records]
    document["retained_inputs"] = {
        "baseline": library.RESULT_PIN, "review": library.REVIEW_PIN,
        "stage10": {c: library.retained.digest(raw["stage10"])
                    for c, (_, raw) in zip(library.CONTROLS, prepared, strict=True)},
        "tensor_pins": {k: library.retained.digest(v) for k, v in tensors.items()},
        "operand_pins": [library.retained.digest(v) for v in operands],
        "reference_pins": [library.retained.digest(reference),
                           *[library.retained.digest(refs[b]) for b in ("fp16", "binary64")]],
        "thresholds": closure["thresholds"],
    }
    return document, out


def prepare(handoff, sha256):
    document, out = preflight(handoff, sha256)
    oracle = library.retained.module_from(library.TEST, NAME + "_oracle")
    authenticate_sources(document["source_pins"])
    return document, out, oracle


def preflight_receipt(document, out, handoff, sha256):
    return {
        "schema_version": 1, "status": "PREFLIGHT_ONLY", "mission_id": document["mission_id"],
        "output": str(out), "output_claim": document["output_claim"],
        "output_claim_acquired": False, "science_budget": 1, "science_invocations": 0,
        "command_bytes": command(handoff, sha256), "source_pins": document["source_pins"],
        "input_pins": document["input_pins"], "retained_inputs": document["retained_inputs"],
        "non_admission": True, "normal_independent_review": "REQUIRED",
        "automatic_replay_authorized": False, "boundary": library.BOUNDARY,
    }


@contextmanager
def writing(out, audit):
    prior = library.prior
    require(prior._active is None and sys.getprofile() is None, "nested write guard")
    if not prior._hook_installed:
        sys.addaudithook(prior.audit_event)
        prior._hook_installed = True
    prior._active = (out, audit)
    try:
        yield
    finally:
        prior._active = None


def save(out, name, value):
    with (out / name).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def emit_payloads(out, result, payloads):
    require(list(payloads) == list(library.CONTROLS), "payload control census")
    require(result["status"] == library.scientific_status(result["controls"]), "result classification/schema")
    library.dispatch_census(result["audit"])
    for row in result["controls"]:
        control = row["control"]
        path = out / (control + ".npz")
        with path.open("xb") as stream:
            np.savez(stream, **payloads[control])
        row["payload"] = pin(path)
        loaded = library.down.archive(row["payload"])
        require({k: library.retained.digest(v) for k, v in loaded.items()}
                == {k: library.retained.digest(v) for k, v in payloads[control].items()},
                "emitter/consumer payload mismatch")


def execute(handoff, sha256):
    document, out, oracle = prepare(handoff, sha256)
    writes, audit = library.base.new_audit(), library.base.new_audit()
    with writing(out, writes):
        save(out, "command.json", {
            "mission_id": document["mission_id"], "mission": document["mission"],
            "output": str(out), "output_claim": document["output_claim"],
            "command_bytes": command(handoff, sha256), "argv": sys.argv,
            "cwd": str(ROOT), "uid": os.getuid(), "pid": os.getpid(),
            "executable": sys.executable, "timestamp": time.time(), "science_budget": 1,
            "authorization_id": "", "action": "", "source_pins": document["source_pins"],
            "input_pins": document["input_pins"], "claim": "exclusive command.json; no retries",
        })
    result = preflight_receipt(document, out, handoff, sha256)
    result.update(status="UNKNOWN", output_claim_acquired=True, controls=[], audit=audit,
                  flags=library.base.closure_source.FLAGS)
    try:
        library.native.torch.set_num_threads(1)
        evaluated, payloads = library.evaluate(oracle, audit=audit)
        require(evaluated["source_pins"] == [
            p for p in document["source_pins"] if p["path"] in library.project_sources()],
            "evaluator source census")
        result.update(evaluated)
        result["source_pins"] = document["source_pins"]
        result["input_pins"] += document["input_pins"]
        authenticate_sources(document["source_pins"])
        for record in result["input_pins"]:
            library.gate_up.historical_record(record)
        with writing(out, writes):
            emit_payloads(out, result, payloads)
        require(writes["writes"] == [str(out / "command.json")]
                + [str(out / (c + ".npz")) for c in library.CONTROLS]
                and writes["forbidden_events"] == [], "exclusive payload/write census")
    except Exception as error:
        # A claimed failure is consumed UNKNOWN evidence, never a retry.
        result.update(status="UNKNOWN", error_type=type(error).__name__, error=str(error))
    result["science_invocations"] = audit.get("science_invocations", 0)
    result["audit"]["writes"] = writes["writes"][1:].copy()
    result["audit"]["forbidden_events"].extend(writes["forbidden_events"])
    result["evidence_write_census"] = sorted([p.name for p in out.iterdir()] + ["result.json"])
    with writing(out, writes):
        save(out, "result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    parser.add_argument("--handoff", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args(argv)
    if args.preflight:
        document, out, _ = prepare(args.handoff, args.sha256)
        result = preflight_receipt(document, out, args.handoff, args.sha256)
    else:
        result = execute(args.handoff, args.sha256)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 1 if result["status"] == "UNKNOWN" else 0


if __name__ == "__main__":
    raise SystemExit(main())
