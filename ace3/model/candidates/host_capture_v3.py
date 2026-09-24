"""Parent-invoked P0 capture producer; capture never supplies its own trust input."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np
from safetensors import safe_open

from ace3.model.candidates import decoder_gate_policy_v3 as policy
from ace3.model.candidates import local_operator_reference_v3 as local
from ace3.model.candidates import runtime_admission_v3 as runtime
from ace3.model.candidates import capture_harness_v3 as harness
from ace3.model.candidates.run_single_round_residual_rtl import retained_record as record
from ace3.model.candidates.run_single_round_residual_rtl import retained_write as write


ROOT = runtime.ROOT
BUILD = ROOT / "build"
SOFTWARE = BUILD / "local_operator_v3_229c5f00e558_attempt001/evaluation"
RUNTIME_SOFTWARE = BUILD / "local_operator_v3_229c5f00e558_runtime_attempt001"
REVIEWS = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs")
LAYERS = (5, 6, 7, 8)
REMAINING_LAYERS = tuple(range(9, 24))
SCHEMA = "ace3-v3-parent-host-capture-plan-v1"


def adapter():
    # The established vector/transaction module uses project-local imports.
    model = str(ROOT / "ace3/model")
    if model not in sys.path:
        sys.path.insert(0, model)
    from ace3.model import validate_selected_token_position2_traversal
    return validate_selected_token_position2_traversal


def authenticate(rec):
    expected = {key: rec[key] for key in ("path", "bytes", "sha256")}
    local.require(record(rec["path"]) == expected, f"binding drift: {rec['path']}")
    return Path(rec["path"])


def fresh(path, parent):
    path, parent = Path(path), Path(parent).resolve()
    local.require(path.is_absolute() and path == path.resolve()
                  and path.is_relative_to(parent) and path != parent,
                  "output must be a resolved child of the bounded build path")
    local.require(not os.path.lexists(path), f"immutable output already exists: {path}")
    return path


def observe(argv, directory, label):
    """Record command before spawn; never manufacture terminal evidence on interruption."""
    write(directory / f"{label}.command.json", {
        "argv": list(argv), "cwd": str(ROOT),
        "environment": {k: os.environ.get(k) for k in
                        ("PATH", "PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    })
    start, wall = time.monotonic(), time.time_ns()
    log = directory / f"{label}.log"
    with log.open("xb") as stream:
        try:
            process = subprocess.Popen(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        except OSError as error:
            write(directory / f"{label}.process.json", {
                "argv": list(argv), "executed": False, "natural_exit": False,
                "error": str(error), "started_ns": wall,
                "failure_taxonomy": "evaluator_no_execution",
                "root_cause_hypothesis": "The named executable could not be spawned.",
                "regression": "Spawn failure must not create a capture manifest.",
            })
            raise
        write(directory / f"{label}.started.json", {
            "pid": process.pid, "parent_pid": os.getpid(), "started_ns": wall,
        })
        code = process.wait()
    result = {
        "argv": list(argv), "pid": process.pid, "executed": True, "returncode": code,
        "natural_exit": code >= 0, "started_ns": wall, "finished_ns": time.time_ns(),
        "elapsed_seconds": time.monotonic() - start, "log": record(log),
    }
    if code:
        result.update(
            failure_taxonomy="execution_failure",
            root_cause_hypothesis="The observed process failed; inspect its retained log.",
            regression="Nonzero or signaled execution must never produce an admitted manifest.")
    write(directory / f"{label}.process.json", result)
    if code:
        raise subprocess.CalledProcessError(code, argv)
    return result


def simulation_command(directory, layer):
    return [
        str(directory / ("obj/V" + runtime.TOP)), "--layer-index", str(layer),
        "--vector-dir", str(directory / "vectors"), "--tensor-dir", str(directory / "vectors"),
        "--raw-dir", str(directory / "runtime/raw"), "--transaction-position", "0",
        "--state-out", str(directory / "candidate.state"), "--progress-interval", "1000000",
        harness.CAPTURE_FLAG,
    ]


def compile_command(source, tool, directory, layer, sources):
    command = list(source["command"])
    command[0] = tool["path"]
    command[command.index("--Mdir") + 1] = str(directory / "obj")
    command[command.index("-GLAYER_INDEX=4")] = f"-GLAYER_INDEX={layer}"
    paths = {Path(r["path"]).name: r["path"] for r in sources}
    for rec in source["source_closure"]:
        if rec["path"] in command:
            command[command.index(rec["path"])] = paths[Path(rec["path"]).name]
    return command


def load_monitor(rec, hidden, *, capture_only=False):
    with np.load(authenticate(rec), allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    local.finite_words(hidden, (896,))
    local.require(capture_only or np.array_equal(arrays["input_hidden"], hidden),
                  "actual parent differs from retained exact-monitor input; no stale monitor replay")
    for stage, count in local.SIZES.items():
        local.finite_words(arrays[f"stage{stage:02d}"], (count,))
    return arrays


def monitor_trace(arrays, coordinates):
    offsets, trace = {8: 0, 9: 0}, []
    for stage, position, index in coordinates:
        offset = offsets[stage] if stage in offsets else index
        trace.append((stage, index, int(arrays[f"stage{stage:02d}"][offset]), position))
        if stage in offsets:
            offsets[stage] += 1
    local.require(len(trace) == sum(local.SIZES.values()), "monitor trace extent mismatch")
    return trace


def actual_arrays(transaction, hidden, coordinates):
    rows = runtime.trace_rows(runtime.artifact(transaction["raw"]["trace"]))
    arrays = runtime.canonical_stages(rows, coordinates)
    final = runtime.hex_words(runtime.artifact(transaction["output"]), indexed=True)
    local.require(np.array_equal(final, arrays["stage18"]), "actual final/trace disagreement")
    arrays["input_hidden"] = hidden.copy()
    for kind, stage in (("k", 6), ("v", 7)):
        arrays["input_cache_" + kind] = np.empty((0, 128), dtype="<u2")
        arrays["output_cache_" + kind] = arrays[f"stage{stage:02d}"].reshape(1, 128).copy()
    return arrays


def recover(out, original_plan, historical_sources):
    """Derive operands from a completed capture; never invoke or restore RTL."""
    original_plan = original_plan.resolve()
    original = json.loads(original_plan.read_text())
    local.require(original["schema"] == SCHEMA and original["policy_id"] == policy.POLICY_ID
                  and original_plan == Path(original["root"]) / "freeze.json",
                  "original capture plan identity mismatch")
    directory = Path(original["transactions"]["5"]["directory"])
    original_manifest = record(directory / "manifest.json")
    manifest = runtime.document(original_manifest)
    local.require(manifest["node"] == [5, 0] and manifest["history"] == [9707]
                  and "derivation" not in manifest, "recovery requires original L5/P0 capture")
    sources = [record(Path(__file__).resolve()), record(Path(runtime.__file__).resolve())]
    changed_paths = {r["path"] for r in sources}
    bindings, historical = [], []
    for rec in original["bindings"]:
        if rec["path"] in changed_paths:
            saved = historical_sources / "project" / Path(rec["path"]).relative_to(ROOT)
            replacement = dict(rec, path=str(saved.resolve()))
            authenticate(replacement)
            historical.append({"original": rec, "preserved": replacement})
            bindings.append(replacement)
        else:
            authenticate(rec)
            bindings.append(rec)
    local.require({r["original"]["path"] for r in historical} == changed_paths,
                  "missing frozen producer source binding")
    context = runtime._trusted_context(json.loads(policy.CONTRACT.read_text()), 5)
    local.require(manifest["parent"] == original["actual_l4_parent"] == context["root_hidden"],
                  "retained recovery L4 producer mismatch")
    transaction = runtime.document(manifest["transaction"])
    execution = runtime.document(manifest["execution"])
    local.require(execution["executed"] is True and execution["natural_exit"] is True
                  and execution["returncode"] == 0 and execution["transaction"] == manifest["transaction"]
                  and transaction["raw"]["done_count"] == 1
                  and transaction["raw"]["trace_count"] == sum(local.SIZES.values())
                  and transaction["raw"]["final_count"] == 896,
                  "retained recovery requires completed original capture")
    hidden = runtime.hex_words(runtime.artifact(manifest["parent"]), indexed=True)
    arrays = actual_arrays(transaction, hidden, context["trace_coordinates"])
    local.validate_lineage(arrays, hidden, position=0, history=[9707])
    generated = runtime.document(manifest["compile"])["generated"]
    runtime.saved_state(
        runtime.artifact(transaction["output_state"]),
        runtime.artifact(generated["header"]).decode("ascii"),
        runtime.artifact(generated["slow"]).decode("ascii"),
        runtime.artifact(generated["symbols"]).decode("ascii"), 5, arrays)
    old_operands = runtime.artifact(manifest["actual_operands"])
    with np.load(io.BytesIO(old_operands), allow_pickle=False) as archive:
        differences = {name: int(np.count_nonzero(archive[name] != value))
                       for name, value in arrays.items()}
    out = fresh(out, BUILD)
    out.mkdir()
    plan = {
        "schema": "ace3-v3-retained-recovery-plan-v1", "root": str(out),
        "policy_id": policy.POLICY_ID, "scope": {"layers": [5], "position": 0, "history": [9707]},
        "original_plan": record(original_plan), "original_manifest": original_manifest,
        "original_admission": record(directory / "admission/result.json"),
        "historical_producer_sources": historical, "decoder_sources": sources,
        "checkpoint": original["checkpoint"], "transactions": {"5": {"directory": str(out)}},
        "bindings": bindings + sources, "decoder_rtl_invocations": 0,
        "numerical_status": "NOT_EVALUATED", "independent_review": "REQUIRED",
    }
    write(out / "freeze.json", plan)
    with (out / "actual_operands.npz").open("xb") as stream:
        np.savez(stream, **arrays)
    manifest = dict(manifest, actual_operands=record(out / "actual_operands.npz"))
    boundary = {
        "schema": "ace3-v3-retained-coordinate-decoding-v1", "freeze": record(out / "freeze.json"),
        "original_manifest": original_manifest, "decoder_sources": sources,
        "raw_trace": transaction["raw"]["trace"], "final": transaction["output"],
        "state": transaction["output_state"], "actual_operands": manifest["actual_operands"],
        "decoder_rtl_invocations": 0,
    }
    write(out / "vector_boundary_manifest.json", boundary)
    manifest["derivation"] = record(out / "vector_boundary_manifest.json")
    write(out / "manifest.json", manifest)
    runtime.validate_derivation(manifest)
    write(out / "recovery_result.json", {
        "status": "RETAINED_DECODING_CHECKED_NOT_ADMITTED", "numerical_status": "NOT_EVALUATED",
        "decoder_rtl_invocations": 0, "manifest_path": str(out / "manifest.json"),
        "changed_operand_coordinates": differences, "original_admission": plan["original_admission"],
        "failure_taxonomy": "trace_coordinate_assembly",
        "root_cause_hypothesis": "RoPE emission order was incorrectly used as canonical tensor order.",
        "regression": "Retained coordinate-keyed K producer/cache and saved-state identity agree.",
    })
    command = ["env", f"PYTHONPATH={ROOT}:{ROOT / 'ace3/model'}",
               sys.executable, "-B", "-m", "ace3.model.candidates.host_capture_v3",
               "admit-recovery", "--plan", str(out / "freeze.json"), "--layer", "5",
               "--trusted-manifest-sha256", "HOST_OBSERVED_DERIVED_MANIFEST_SHA256"]
    write(out / "host_action.json", {
        "argv": command, "command": shlex.join(command),
        "trust_boundary": "Parent authenticates derived manifest and original owner-observed capture; "
                          "replace the placeholder only with its independently observed digest.",
        "decoder_rtl_invocations": 0, "no_l6": True,
    })


def load_recovery(path):
    plan = json.loads(path.read_text())
    local.require(plan["schema"] == "ace3-v3-retained-recovery-plan-v1"
                  and plan["policy_id"] == policy.POLICY_ID
                  and plan["scope"] == {"layers": [5], "position": 0, "history": [9707]}
                  and path == Path(plan["root"]) / "freeze.json"
                  and plan["transactions"] == {"5": {"directory": plan["root"]}},
                  "retained recovery plan scope/path mismatch")
    for rec in (*plan["bindings"], plan["original_plan"], plan["original_manifest"],
                plan["original_admission"]):
        authenticate(rec)
    boundary = runtime.document(record(Path(plan["root"]) / "vector_boundary_manifest.json"))
    local.require(boundary["freeze"] == record(path), "recovery freeze/boundary mismatch")
    runtime.validate_derivation(runtime.document(record(Path(plan["root"]) / "manifest.json")))
    return plan


def tensors_from(checkpoint, canonical, layer):
    with safe_open(checkpoint, framework="np") as model:
        tensors = {name: model.get_tensor(name) for name in local.tensor_shapes(layer)}
    local.authenticate_tensors(tensors, canonical, layer)
    return tensors


def review(path, mission):
    value = json.loads(path.read_text())
    local.require(value["kind"] == "round_reviewed_handoff"
                  and value["producer_role"] == "reviewer" and value["mission_id"] == mission
                  and value["review"]["status"] == "done", "missing genuine accepted review")
    return record(path)


def reviewed_policy_source():
    frozen = json.loads((RUNTIME_SOFTWARE / "freeze.json").read_text())
    rec = runtime.unique(frozen["sources"], lambda r: r["path"] ==
                         "ace3/model/candidates/decoder_gate_policy_v3.py",
                         "reviewed runtime-wired v3 policy source")
    successor = dict(rec, path=str(ROOT / rec["path"]))
    authenticate(successor)
    return successor


def prepare(out):
    fresh(out, BUILD).mkdir()
    traversal = adapter()
    contract = json.loads(policy.CONTRACT.read_text())
    contexts = {layer: runtime._trusted_context(contract, layer) for layer in LAYERS}
    source = contexts[5]["source"]
    runtime.source_map(source["source_closure"])
    top = runtime.artifact(next(r for r in source["source_closure"]
                               if Path(r["path"]).name == runtime.TOP + ".sv")).decode("ascii")
    header = re.search(r"\bmodule\s+" + runtime.TOP + r"\s*#\(.*?\);", top, re.S)
    local.require(header is not None and header[0] == source["public_contract"],
                  "accepted public RTL contract mismatch")
    members = json.loads((SOFTWARE / "output_manifest.json").read_text())["artifacts"]

    def software_document(name):
        rec = runtime.unique(members, lambda r: r["path"] == str(SOFTWARE / name), name)
        return runtime.document(rec), rec

    software, software_rec = software_document("result.json")
    frozen, frozen_rec = software_document("freeze.json")
    local.require(software["status"] == "SUPPORTED_SOFTWARE_ONLY"
                  and software["policy_id"] == policy.POLICY_ID
                  and software["first_failure"] is None
                  and software["decoder_rtl_invocations"] == 0
                  and software["mandatory_local_operator_fp16"] ==
                  {"gates": 72, "coordinates": 89712, "failures": 0}
                  and software["mandatory_global_binary64_v1"] ==
                  [{"node": [l, 0, 18], "coordinates": 896, "failure_count": 0} for l in LAYERS]
                  and len(software["independent_lineage_state"]) == 4
                  and all(r["status"] == "PASS" for r in software["independent_lineage_state"]),
                  "accepted v3 software support not established")
    local.require(frozen["actual_input_root"] == contexts[5]["root_hidden"]
                  and frozen["scope"] == {"layers": list(LAYERS), "position": 0, "history": [9707]}
                  and frozen["input_state"] is None
                  and frozen["candidate_semantics"] == source["semantics"],
                  "software/arithmetic/actual L4 parent incompatibility")
    software_review = review(REVIEWS / "229c5f00e558/round-0002.json", "229c5f00e558")
    runtime_result = json.loads((RUNTIME_SOFTWARE / "result.json").read_text())
    local.require(runtime_result["status"] == "PASS_REGRESSIONS_ONLY"
                  and runtime_result["returncode"] == 0
                  and runtime_result["decoder_rtl_invocations"] == 0,
                  "reviewed runtime-wiring regression support missing")
    policy_successor = reviewed_policy_source()
    for rec in frozen["sources"]:
        authenticate(policy_successor if rec["path"] == str(Path(policy.__file__).resolve()) else rec)
    specification = runtime.document(frozen["operator_specification"])
    checkpoint = authenticate(specification["checkpoint"])
    hidden = runtime.hex_words(runtime.artifact(contexts[5]["root_hidden"]), indexed=True)
    local.finite_words(hidden, (896,))
    local.require(hashlib.sha256(hidden.tobytes()).hexdigest() ==
                  contexts[5]["root_hidden"]["semantic_sha256"], "L4 semantic identity mismatch")
    tools = {}
    for name, flag in (("verilator", "--version"), ("iverilog", "-V"),
                       ("make", "--version"), ("g++", "--version")):
        path = shutil.which(name)
        local.require(path is not None, f"host tool unavailable: {name}; no RTL conclusion")
        tools[name] = record(path)
        observe([tools[name]["path"], flag], out, name + "_version")
    verilator_bin = Path(tools["verilator"]["path"]).with_name("verilator_bin")
    local.require(verilator_bin.is_file(), "Verilator backend identity unavailable")
    tools["verilator_bin"] = record(verilator_bin)
    tools["python"] = record(sys.executable)
    public = out / "public_contract.sv"
    shutil.copyfile(authenticate(frozen["public_contract_source"]), public)
    public_compile = [tools["iverilog"]["path"], "-g2012", "-s", "frozen_contract",
                      "-o", str(out / "public_contract.vvp"), str(public)]
    observe(public_compile, out, "public_contract_compile")
    capture_dir = out / "capture_source"
    capture_dir.mkdir()
    capture_sources = []
    for rec in source["source_closure"]:
        name = Path(rec["path"]).name
        if name in (harness.HARNESS, "ace3_layer0_trace_capture_policy.h"):
            data = runtime.artifact(rec)
            with (capture_dir / name).open("xb") as stream:
                stream.write(harness.capture_source(data) if name == harness.HARNESS else data)
            capture_sources.append(record(capture_dir / name))
        else:
            capture_sources.append(rec)
    local.require(harness.compatible_sources(
        runtime.source_map(capture_sources), runtime.source_map(source["source_closure"]),
        runtime.artifact), "capture-only harness derivative missing")
    runtime_root = out / "runtime"
    plans, bindings = {}, [software_rec, frozen_rec, record(SOFTWARE / "output_manifest.json"),
        frozen["operator_specification"], specification["checkpoint"], contexts[5]["root_hidden"],
        record(public), record(policy.CONTRACT),
        software_review, policy_successor, record(RUNTIME_SOFTWARE / "freeze.json"),
        record(RUNTIME_SOFTWARE / "result.json"),
        review(REVIEWS / "fb06353c5f53/round-0005.json", "fb06353c5f53")]
    bindings.extend(source["source_closure"])
    bindings.extend(capture_sources)
    bindings.extend(frozen["input_bindings"])
    expected = hidden
    for layer in LAYERS:
        context = contexts[layer]
        tensors_from(checkpoint, context["canonical"], layer)
        provenance = runtime.unique(software["software_provenance"],
                                    lambda r: r["layer"] == layer, "software layer")
        monitor = provenance["actual_archive"]
        arrays = load_monitor(monitor, expected)
        expected = arrays["stage18"]
        directory = runtime_root / f"layer{layer:02d}"
        fresh(directory, out)
        plans[str(layer)] = {
            "directory": str(directory), "parameters": {"LAYER_INDEX": layer, "ACCURATE_SILU": 1},
            "compile_argv": compile_command(source, tools["verilator"], directory, layer,
                                            capture_sources),
            "simulation_argv": simulation_command(directory, layer),
            "exact_monitor": monitor, "canonical_tensors": context["canonical"],
            "input_state": None, "cache_slot": 0,
            "parent_rule": "accepted actual L4" if layer == 5 else
                           "preceding captured actual output with separately trusted v3 admission",
        }
        bindings.extend([monitor, *context["reference_bindings"].values()])
    (out / "independent_p0_rope.hex").write_bytes(runtime.p0_rope_bytes())
    bindings.append(record(out / "independent_p0_rope.hex"))
    # Bind the imported executable closure, including the existing adapter and native modules.
    module_files = {Path(m.__file__).resolve() for m in tuple(sys.modules.values())
                    if getattr(m, "__file__", None) and Path(m.__file__).is_file()}
    bindings.extend(record(p) for p in sorted(module_files)
                    if p.is_relative_to(ROOT) or p.suffix in (".so", ".pyd"))
    bindings.extend(tools.values())
    unique = {}
    for rec in bindings:
        signature = {key: rec[key] for key in ("path", "bytes", "sha256")}
        local.require(unique.setdefault(rec["path"], signature) == signature,
                      "conflicting frozen binding")
        authenticate(signature)
    plan = {
        "schema": SCHEMA, "status": "PREEXECUTION_READY_NOT_CAPTURED",
        "policy_id": policy.POLICY_ID, "root": str(out), "scope": frozen["scope"],
        "public_contract": source["public_contract"], "semantics": source["semantics"],
        "source_closure": capture_sources, "checkpoint": specification["checkpoint"],
        "accepted_source_closure": source["source_closure"],
        "capture_mode": "capture-only; exact source-derived monitor change, unchanged RTL and state ABI",
        "actual_l4_parent": contexts[5]["root_hidden"], "tools": tools,
        "transactions": plans, "bindings": list(unique.values()),
        "independent_p0_rope": record(out / "independent_p0_rope.hex"),
        "software_support": software_rec, "public_contract_compile": record(
            out / "public_contract_compile.process.json"),
        "policy_source_successor": {
            "original_software_freeze": frozen_rec,
            "later_reviewed_runtime_freeze": record(RUNTIME_SOFTWARE / "freeze.json"),
            "selected_policy_source": policy_successor, "review": software_review,
            "boundary": "Use the already reviewed runtime-wired policy, not its older software-only revision.",
        },
        "monitor_boundary": (
            "Retained values never gate capture or feed RTL. Only their transaction coordinates/counts "
            "gate capture; all actual values are recorded for separate v3 numerical admission."),
        "external_trust": "Only the invoking parent Host may supply observed manifest/admission digests.",
        "runtime_admission": "NOT_EVALUATED", "numerical_status": "NOT_EVALUATED",
        "decoder_rtl_invocations": 0, "independent_review": "REQUIRED",
    }
    write(out / "freeze.json", plan)
    sums = [record(out / "freeze.json"), *plan["bindings"]]
    with (out / "preexecution.sha256").open("x") as stream:
        for rec in sums:
            stream.write(f"{rec['sha256']}  {rec['path']}\n")
    argv = [tools["python"]["path"], "-B", "-m", "ace3.model.candidates.host_capture_v3",
            "capture", "--plan", str(out / "freeze.json"), "--layer", "5"]
    command = (
        "set -euC\ncd " + shlex.quote(str(ROOT)) + "\n"
        "export PYTHONDONTWRITEBYTECODE=1\n"
        "export PYTHONPATH=" + shlex.quote(str(ROOT) + ":" + str(ROOT / "ace3/model")) + "\n"
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\n"
        "test ! -e " + shlex.quote(str(runtime_root)) + "\n"
        "sha256sum --check " + shlex.quote(str(out / "preexecution.sha256")) + " > "
        + shlex.quote(str(out / "host_preexecution.log")) + " 2>&1\nexec " + shlex.join(argv) + "\n")
    with (out / "capture-layer05.command.sh").open("x") as stream:
        stream.write(command)
    write(out / "launch_action.json", {
        "argv": ["sh", str(out / "capture-layer05.command.sh")],
        "command_file": record(out / "capture-layer05.command.sh"),
        "plan": record(out / "freeze.json"), "preexecution_checks": record(out / "preexecution.sha256"),
        "status": "READY_FOR_PARENT_INSPECTION_AND_LAUNCH_NOT_INVOKED",
    })
    print(f"PREEXECUTION_READY_NOT_CAPTURED {out / 'launch_action.json'}")


def load_plan(path):
    plan = json.loads(path.read_text())
    local.require(plan["schema"] == SCHEMA and plan["policy_id"] == policy.POLICY_ID
                  and plan["scope"] in (
                      {"layers": list(LAYERS), "position": 0, "history": [9707]},
                      {"layers": list(REMAINING_LAYERS), "position": 0, "history": [9707]})
                  and path == Path(plan["root"]) / "freeze.json", "capture plan scope/path mismatch")
    if plan["scope"]["layers"] == list(REMAINING_LAYERS):
        from ace3.model.candidates.remaining_layers_v3 import validate_plan
        validate_plan(plan)
    for rec in plan["bindings"]:
        authenticate(rec)
    return plan


def reference_contract(plan):
    contract = json.loads(policy.CONTRACT.read_text())
    for key in ("reference_extension", "evaluator_extension"):
        if key in plan:
            contract[key] = plan[key]
    return contract


def admitted_parent(plan, layer, digest):
    if layer == 5:
        local.require(digest is None, "L5 must consume the accepted L4 root, not a supplied parent")
        return plan["actual_l4_parent"], plan["actual_l4_parent"]
    local.require(isinstance(digest, str) and re.fullmatch("[0-9a-f]{64}", digest),
                  "parent Host must supply preceding admission digest")
    path = Path(plan["transactions"][str(layer - 1)]["directory"]) / "admission/result.json"
    rec = record(path)
    local.require(rec["sha256"] == digest, "preceding admission differs from Host trust input")
    parent = runtime.document(rec)
    local.require(parent["status"] == parent["numerical_status"] == "PASS"
                  and parent["policy_id"] == policy.POLICY_ID
                  and parent["evidence_kind"] == "actual_rtl_numerical_evaluation"
                  and parent["runtime_admission"]["node"] == [layer - 1, 0]
                  and parent["runtime_admission"]["history"] == [9707]
                  and parent["runtime_admission"]["source_sha256"] ==
                  {Path(r["path"]).name: r["sha256"] for r in plan["source_closure"]},
                  "preceding actual output is not compatibly admitted")
    if layer > 9:
        local.require(parent.get("evaluator_extension") == plan.get("evaluator_extension")
                      and parent.get("evaluator_extension") is not None,
                      "preceding remaining evaluator lineage mismatch")
    return rec, parent["runtime_admission"]["output"]


def capture(plan, layer, parent_digest):
    transaction_plan = plan["transactions"][str(layer)]
    local.require(transaction_plan["input_state"] is None, "P0 cannot restore saved state")
    parent, parent_output = admitted_parent(plan, layer, parent_digest)
    hidden = runtime.hex_words(runtime.artifact(parent_output), indexed=True)
    monitor = load_monitor(transaction_plan["exact_monitor"], hidden, capture_only=True)
    context = runtime._trusted_context(reference_contract(plan), layer)
    directory = fresh(Path(transaction_plan["directory"]), Path(plan["root"]))
    directory.mkdir(parents=True)
    write(directory / "capture_invocation.json", {
        "argv": sys.argv, "plan": record(Path(plan["root"]) / "freeze.json"),
        "parent": parent, "external_parent_result_sha256": parent_digest,
    })
    traversal = adapter()
    phase = "vector_preparation"
    try:
        tensors_from(authenticate(plan["checkpoint"]), context["canonical"], layer)
        with safe_open(plan["checkpoint"]["path"], framework="np") as model:
            vectors = traversal.materialize_transaction_vectors(
                model, layer, 0, hidden, directory / "vectors")
        for tensor in vectors["tensors"]:
            metadata = tensor["checkpoint_tensor"]
            canonical = context["canonical"][metadata["name"]]
            local.require(metadata["sha256"] == canonical["sha256"]
                          and metadata["shape"] == canonical["shape"]
                          and metadata["dtype"] == {"float16": "F16", "int32": "I32"}[
                              canonical["dtype"]], "serialized canonical tensor mismatch")
        local.require(runtime.artifact(vectors["rope_coefficients"]) == runtime.p0_rope_bytes(),
                      "serialized P0 RoPE mismatch")
        vectors.update(traversal.materialize_runtime_vector_contract(
            layer, 0, monitor["stage18"], monitor_trace(monitor, context["trace_coordinates"]),
            directory / "vectors"))
        phase = "compile"
        compiled = observe(transaction_plan["compile_argv"], directory, "compile")
        binary_path = directory / ("obj/V" + runtime.TOP)
        binary = record(binary_path)
        compiled.update(tool=plan["tools"]["verilator"], binary=binary,
                        object_dir=str(directory / "obj"), generated={
                            key: record(Path(str(binary_path) + suffix)) for key, suffix in
                            (("header", ".h"), ("slow", "__Slow.cpp"), ("symbols", "__Syms.cpp"))})
        write(directory / "compile.json", compiled)
        write(directory / "prepared.json", {
            "binary": binary, "input_hidden": parent_output, "input_state": None,
            "vectors": vectors, "kv_policy": "own empty P0 prior KV",
            "exact_monitor": transaction_plan["exact_monitor"],
        })
        launch = {
            "binary": binary, "prepared": record(directory / "prepared.json"),
            "node": [layer, 0], "history": [9707], "parameters": transaction_plan["parameters"],
            "argv": transaction_plan["simulation_argv"], "input_hidden": parent_output,
            "input_state": None, "cache_slot": 0,
            "independent_p0_rope_sha256": plan["independent_p0_rope"]["sha256"],
        }
        write(directory / "launch.json", launch)
        execution = {}

        def observed_run(argv, log):
            capture_argv = [*argv, harness.CAPTURE_FLAG]
            local.require(capture_argv == launch["argv"]
                          and log == directory / "runtime/simulation.log",
                          "adapter changed frozen simulator command")
            authenticate(binary)
            execution.update(observe(capture_argv, log.parent, "simulation"))

        old = traversal.run_logged
        traversal.run_logged = observed_run
        phase = "simulation"
        try:
            transaction, _ = traversal.execute_transaction(
                binary_path, layer, 0, hidden, vectors, directory / "vectors",
                directory / "runtime", directory / "candidate.state", None)
        finally:
            traversal.run_logged = old
        write(directory / "transaction.json", transaction)
        execution.update(binary=binary, launch=record(directory / "launch.json"),
                         transaction=record(directory / "transaction.json"),
                         output_state=transaction["output_state"])
        write(directory / "execution.json", execution)
        phase = "raw_capture"
        arrays = actual_arrays(transaction, hidden, context["trace_coordinates"])
        with (directory / "actual_operands.npz").open("xb") as stream:
            np.savez(stream, **arrays)
        manifest = {
            "schema": runtime.SCHEMA, "policy_id": policy.POLICY_ID, "evidence_kind": "actual_rtl",
            "node": [layer, 0], "history": [9707], "public_contract": plan["public_contract"],
            "parameters": transaction_plan["parameters"], "semantics": plan["semantics"],
            "source_closure": plan["source_closure"], "parent": parent,
            "independent_p0_rope": plan["independent_p0_rope"],
            **{name: record(directory / (name + ".json")) for name in
               ("compile", "launch", "prepared", "execution", "transaction")},
            "actual_operands": record(directory / "actual_operands.npz"),
        }
        for key in ("reference_extension", "evaluator_extension"):
            if key in plan:
                manifest[key] = plan[key]
        write(directory / "manifest.json", manifest)
        write(directory / "capture_result.json", {
            "status": "CAPTURED_NOT_ADMITTED", "manifest_path": str(directory / "manifest.json"),
            "numerical_status": "NOT_EVALUATED", "external_trust_input": "NOT_SUPPLIED",
        })
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        write(directory / "capture_failure.json", {
            "status": "FAILED_OR_PARTIAL", "phase": phase, "error": str(error),
            "numerical_status": "NOT_EVALUATED", "failure_taxonomy": "capture_boundary",
            "root_cause_hypothesis": "The named capture phase did not satisfy its executable contract.",
            "regression": "Keep this attempt; no manifest/admission from failed or partial capture.",
        })
        raise


def admit(plan, layer, digest, out=None):
    local.require(isinstance(digest, str) and re.fullmatch("[0-9a-f]{64}", digest),
                  "separately supplied parent-Host manifest digest required")
    directory = Path(plan["transactions"][str(layer)]["directory"])
    receipt = record(directory / "manifest.json")
    local.require(receipt["sha256"] == digest, "manifest differs from external Host digest")
    if out is None:
        out = fresh(directory / "admission", directory)
    else:
        out = fresh(out, BUILD)
        local.require(out.parent == BUILD.resolve(),
                      "explicit admission output must be a fresh top-level build root")
    manifest = runtime.document(receipt)
    context = runtime._trusted_context(reference_contract(plan), layer)
    with np.load(io.BytesIO(runtime.artifact(manifest["actual_operands"])), allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    out.mkdir()
    write(out / "command.json", {"argv": sys.argv, "trusted_manifest_sha256": digest,
                                "plan": record(Path(plan["root"]) / "freeze.json"),
                                "manifest": receipt, "output_root": str(out)})
    tensors = tensors_from(authenticate(plan["checkpoint"]), context["canonical"], layer)
    evaluator, extension = policy, {}
    if layer in REMAINING_LAYERS:
        from ace3.model.candidates import remaining_decoder_gate_policy_v3 as evaluator
        extension = {key: plan.get(key) for key in ("reference_extension", "evaluator_extension")}
    result = evaluator.evaluate_actual_rtl_result(
        runtime_admission=receipt, trusted_runtime_manifest_sha256=digest, arrays=arrays,
        tensors=tensors, canonical_records=context["canonical"], expected_hidden=arrays["input_hidden"],
        trajectory=context["trajectory"], reference_binary64=context["binary64"],
        layer=layer, position=0, history=[9707], policy=policy.POLICY_ID,
        **extension,
        emit=lambda report: write(out / f"stage{report['stage']:02d}.json", report))
    references = result.pop("local_references", None)
    if references is not None:
        with (out / "local_references.npz").open("xb") as stream:
            np.savez(stream, **references)
        result["local_references"] = record(out / "local_references.npz")
    result["admission_invocation"] = record(out / "command.json")
    write(out / "result.json", result)
    return 0 if result["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "capture", "admit", "recover", "admit-recovery"))
    parser.add_argument("--out", type=Path,
                        help="fresh preparation/recovery path or top-level build admission root")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--layer", type=int, choices=(*LAYERS, *REMAINING_LAYERS))
    parser.add_argument("--parent-result-sha256")
    parser.add_argument("--trusted-manifest-sha256")
    parser.add_argument("--historical-sources", type=Path)
    args = parser.parse_args()
    if args.action == "recover":
        local.require(args.out is not None and args.plan is not None
                      and args.historical_sources is not None and args.layer is None
                      and args.parent_result_sha256 is None and args.trusted_manifest_sha256 is None,
                      "recover requires fresh --out, original --plan and --historical-sources")
        recover(args.out, args.plan, args.historical_sources)
        return 0
    local.require(args.historical_sources is None, "historical sources apply only to recovery")
    if args.action == "prepare":
        local.require(args.out is not None and args.plan is None and args.layer is None
                      and args.parent_result_sha256 is None and args.trusted_manifest_sha256 is None,
                      "prepare requires only a fresh --out")
        prepare(args.out)
        return 0
    local.require(args.plan is not None and args.layer in (*LAYERS, *REMAINING_LAYERS),
                  "capture/admit require --plan and --layer")
    if args.action == "admit-recovery":
        local.require(args.layer == 5 and args.parent_result_sha256 is None,
                      "retained recovery cannot advance to L6 or replace its parent")
        return admit(load_recovery(args.plan), 5, args.trusted_manifest_sha256, out=args.out)
    local.require(args.action != "capture" or args.out is None,
                  "capture cannot override its frozen output path")
    plan = load_plan(args.plan)
    local.require(args.layer in plan.get("continuation", plan["scope"])["layers"],
                  "retained ancestors are not executable continuation layers")
    if args.action == "capture":
        local.require(args.trusted_manifest_sha256 is None, "capture cannot accept a future digest")
        capture(plan, args.layer, args.parent_result_sha256)
        return 0
    local.require(args.parent_result_sha256 is None, "admission requires manifest trust, not parent trust")
    return admit(plan, args.layer, args.trusted_manifest_sha256, out=args.out)


if __name__ == "__main__":
    raise SystemExit(main())
