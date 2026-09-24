"""Bind a fresh capture plan to retained L5 admission; never capture or admit RTL."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import sys

from ace3.model.candidates import host_capture_v3 as host


LAYERS = (6, 7, 8)


def records(value):
    if isinstance(value, dict):
        if {"path", "sha256", "bytes"} <= value.keys():
            yield {key: value[key] for key in ("path", "sha256", "bytes")}
        else:
            for item in value.values():
                yield from records(item)
    elif isinstance(value, list):
        for item in value:
            yield from records(item)


def continuation_plan(original, recovery, parent_record, out):
    parent = host.runtime.document(parent_record)
    admitted = parent["runtime_admission"]
    host.local.require(
        original["schema"] == host.SCHEMA
        and original["scope"] == {"layers": [5, 6, 7, 8], "position": 0, "history": [9707]}
        and recovery["schema"] == "ace3-v3-retained-recovery-plan-v1"
        and recovery["scope"] == {"layers": [5], "position": 0, "history": [9707]}
        and recovery["policy_id"] == original["policy_id"] == host.policy.POLICY_ID,
        "incompatible original/recovery scope or policy")
    host.local.require(
        Path(parent_record["path"]) == Path(recovery["root"]) / "admission/result.json"
        and admitted["manifest"] == host.record(Path(recovery["root"]) / "manifest.json"),
        "L5 admission does not belong to the explicit recovery")
    host.local.require(admitted["saved_state"] == {
        "idle": True, "layer": 5, "state_restore_or_eval_performed": False,
        "status": "PASS", "valid_entries": 128,
    }, "L5-only saved-state evidence mismatch")
    manifest = host.runtime.document(admitted["manifest"])
    transaction = host.runtime.document(manifest["transaction"])
    host.local.require(admitted["output"] == transaction["output"]
                       and admitted["output_state"] == transaction["output_state"],
                       "L5 actual output/state receipt mismatch")
    hidden = host.runtime.hex_words(host.runtime.artifact(admitted["output"]), indexed=True)
    host.local.finite_words(hidden, (896,))
    host.local.require(hashlib.sha256(hidden.tobytes()).hexdigest()
                       == admitted["output"]["semantic_sha256"], "L5 hidden semantic mismatch")
    plan = copy.deepcopy(original)
    plan.update(
        root=str(out), status="PREEXECUTION_READY_NOT_CAPTURED",
        decoder_rtl_invocations=0, numerical_status="NOT_EVALUATED",
        runtime_admission="NOT_EVALUATED", independent_review="REQUIRED",
        continuation={
            "layers": list(LAYERS), "position": 0, "history": [9707],
            "parent_admission": parent_record, "actual_l5_hidden": admitted["output"],
            "l5_state_evidence_only": admitted["output_state"],
            "prior_kv": "each layer owns empty P0 prior KV; no state restore",
            "review_boundary": "normal independent acceptance of L5 and launch bindings before L6",
            "original_plan": recovery["original_plan"],
        },
    )
    # The supported parent lookup reads this existing admission, never a copied result.
    plan["transactions"]["5"]["directory"] = recovery["root"]
    plan["transactions"]["5"]["retained_parent_only"] = True
    host.admitted_parent(plan, 6, parent_record["sha256"])
    for layer in LAYERS:
        transaction_plan = plan["transactions"][str(layer)]
        host.local.require(transaction_plan["input_state"] is None
                           and transaction_plan["cache_slot"] == 0
                           and transaction_plan["parameters"] ==
                           {"LAYER_INDEX": layer, "ACCURATE_SILU": 1},
                           "continuation must preserve own empty P0 state and parameters")
        old_directory = Path(transaction_plan["directory"])
        directory = host.fresh(out / "runtime" / f"layer{layer:02d}", out)
        command = list(transaction_plan["compile_argv"])
        host.local.require(command[command.index("--Mdir") + 1] ==
                           str(old_directory / "obj"), "unexpected original compile path")
        command[command.index("--Mdir") + 1] = str(directory / "obj")
        transaction_plan.update(
            directory=str(directory), compile_argv=command,
            simulation_argv=host.simulation_command(directory, layer))
    return plan


def command_files(plan, out):
    environment = (
        "set -euC\ncd " + shlex.quote(str(host.ROOT)) + "\n"
        "export PYTHONDONTWRITEBYTECODE=1\n"
        "export PYTHONPATH=" + shlex.quote(str(host.ROOT) + ":" + str(host.ROOT / "ace3/model")) + "\n"
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\n")
    commands = []
    for layer in LAYERS:
        for action, variable, option in (
            ("capture", "HOST_PARENT_RESULT_SHA256", "--parent-result-sha256"),
            ("admit", "HOST_MANIFEST_SHA256", "--trusted-manifest-sha256"),
        ):
            argv = [plan["tools"]["python"]["path"], "-B", "-m",
                    "ace3.model.candidates.host_capture_v3", action,
                    "--plan", str(out / "freeze.json"), "--layer", str(layer)]
            command = environment + ': "${' + variable + ':?separately Host-observed digest required}"\n'
            command += ("sha256sum --check " + shlex.quote(str(out / "preexecution.sha256"))
                        + " > " + shlex.quote(str(out / f"{action}-layer{layer:02d}.preexecution.log"))
                        + " 2>&1\n")
            command += "exec " + shlex.join(argv) + " " + option + ' "$' + variable + '"\n'
            path = out / f"{action}-layer{layer:02d}.command.sh"
            with path.open("x") as stream:
                stream.write(command)
            commands.append(host.record(path))
    return commands


def prepare(out, parent_result, recovery_plan, historical_sources):
    host.fresh(out, host.BUILD).mkdir()
    host.write(out / "prepare.command.json", {"argv": sys.argv, "cwd": str(host.ROOT)})
    recovery_record = host.record(recovery_plan)
    recovery = host.runtime.document(recovery_record)
    original = host.runtime.document(recovery["original_plan"])
    parent_record = host.record(parent_result)
    plan = continuation_plan(original, recovery, parent_record, out)

    historical = {item["original"]["path"]: item
                  for item in recovery["historical_producer_sources"]}
    preserved = []
    bindings = []
    for rec in original["bindings"]:
        if rec["path"] in historical:
            item = historical[rec["path"]]
            host.local.require(rec == item["original"], "historical runtime producer mismatch")
            host.authenticate(item["preserved"])
            bindings.append(item["preserved"])
            preserved.append(item)
        else:
            bindings.append(rec)
    for rec in recovery["decoder_sources"]:
        archived = host.record(historical_sources / Path(rec["path"]).name)
        host.local.require((archived["sha256"], archived["bytes"]) ==
                           (rec["sha256"], rec["bytes"]), "historical admission producer mismatch")
        preserved.append({"original": rec, "preserved": archived})
        bindings.append(archived)
    plan["historical_producer_sources"] = preserved

    parent = host.runtime.document(parent_record)
    manifest = host.runtime.document(parent["runtime_admission"]["manifest"])
    bindings.extend([recovery_record, parent_record, recovery["original_plan"]])
    bindings.extend(records(recovery))
    bindings.extend(records(parent))
    bindings.extend(records(manifest))
    for key in ("compile", "launch", "prepared", "execution", "transaction", "derivation"):
        bindings.extend(records(host.runtime.document(manifest[key])))
    # Historical documents keep their original paths; only verified archived revisions
    # replace those paths in the new preexecution checksum list.
    old_producers = {item["original"]["path"] for item in preserved}
    historical_signatures = {
        (item["original"]["path"], item["original"]["sha256"], item["original"]["bytes"])
        for item in preserved}
    for rec in bindings:
        if rec["path"] in old_producers:
            host.local.require((rec["path"], rec["sha256"], rec["bytes"]) in historical_signatures,
                               "unrecognized historical helper revision")
    bindings = [rec for rec in bindings if rec["path"] not in old_producers]
    bindings.extend(host.record(module.__file__) for module in
                    (host, host.runtime, host.harness))
    bindings.append(host.record(__file__))
    bindings.append(host.review(host.REVIEWS / "54dc8dbc7037/round-0001.json", "54dc8dbc7037"))

    contract = json.loads(host.policy.CONTRACT.read_text())
    for layer in LAYERS:
        context = host.runtime._trusted_context(contract, layer)
        transaction = plan["transactions"][str(layer)]
        host.local.require(transaction["canonical_tensors"] == context["canonical"]
                           and plan["semantics"] == context["source"]["semantics"]
                           and plan["public_contract"] == context["source"]["public_contract"],
                           "canonical controls or public contract changed")
        expected_command = host.compile_command(
            context["source"], plan["tools"]["verilator"],
            Path(transaction["directory"]), layer, plan["source_closure"])
        host.local.require(transaction["compile_argv"] == expected_command,
                           "continuation compile command changed beyond output/layer")
        bindings.extend(context["reference_bindings"].values())
    host.local.require(host.harness.compatible_sources(
        host.runtime.source_map(plan["source_closure"]),
        host.runtime.source_map(plan["accepted_source_closure"]), host.runtime.artifact),
        "capture-only harness or RTL source changed")

    unique = {}
    for rec in bindings:
        signature = {key: rec[key] for key in ("path", "bytes", "sha256")}
        host.local.require(unique.setdefault(rec["path"], signature) == signature,
                           "conflicting continuation binding")
    for rec in unique.values():
        host.authenticate(rec)
    for name, flag in (("verilator", "--version"), ("iverilog", "-V"),
                       ("make", "--version"), ("g++", "--version")):
        path = shutil.which(name)
        host.local.require(path is not None and host.record(path) == plan["tools"][name],
                           f"host tool unavailable or changed: {name}; no RTL conclusion")
        host.observe([plan["tools"][name]["path"], flag], out, name + "_version")
    host.local.require(host.record(sys.executable) == plan["tools"]["python"],
                       "preparation interpreter differs from frozen runtime")

    public = host.runtime.unique(original["bindings"], lambda rec: rec["path"] ==
                                 str(Path(original["root"]) / "public_contract.sv"),
                                 "original public contract")
    shutil.copyfile(host.authenticate(public), out / "public_contract.sv")
    host.observe([plan["tools"]["iverilog"]["path"], "-g2012", "-s", "frozen_contract",
                  "-o", str(out / "public_contract.vvp"), str(out / "public_contract.sv")],
                 out, "public_contract_compile")
    plan["public_contract_compile"] = host.record(out / "public_contract_compile.process.json")
    unique[str(out / "public_contract.sv")] = host.record(out / "public_contract.sv")
    plan["bindings"] = list(unique.values())
    commands = command_files(plan, out)
    plan["continuation"]["command_files"] = commands
    host.write(out / "freeze.json", plan)
    with (out / "preexecution.sha256").open("x") as stream:
        for rec in [host.record(out / "freeze.json"), *plan["bindings"], *commands]:
            stream.write(f"{rec['sha256']}  {rec['path']}\n")
    host.write(out / "preparation_result.json", {
        "status": "PREEXECUTION_READY_NOT_CAPTURED", "decoder_rtl_invocations": 0,
        "l5_admission_replays": 0, "numerical_status": "NOT_EVALUATED",
        "independent_review": "REQUIRED", "plan": host.record(out / "freeze.json"),
        "parent_admission": parent_record, "public_contract_compiled": True,
        "boundary": "source/command binding only; not new L5 review or L6-L8 numerical admission",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parent-result", type=Path, required=True)
    parser.add_argument("--recovery-plan", type=Path, required=True)
    parser.add_argument("--historical-admission-sources", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.out, args.parent_result, args.recovery_plan, args.historical_admission_sources)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
