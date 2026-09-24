"""Bind P0 L9-L23 and missing original references; never execute candidate layers."""

import argparse
import ast
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import sys
from types import SimpleNamespace

import numpy as np
from safetensors import safe_open

from ace3.model.candidates import host_capture_v3 as host
from ace3.model.candidates import prepare_l6_l8_continuation_v3 as prior
from ace3.model.candidates import remaining_decoder_gate_policy_v3 as evaluator


LAYERS = tuple(range(9, 24))
SCOPE = {"layers": list(LAYERS), "position": 0, "history": [9707]}
SCHEMA = "ace3-v3-original-reference-suffix-p0-v1"
PARENT_ROOT = host.BUILD / "l6_l8_continuation_8d9993c9e84c_attempt001"
FP16_SOURCE_SHA256 = "7b90fcc34b9a00d614906e36df8002edbac0388755de8736f9d41d49648ae477"


def definitions(record, names, namespace):
    tree = ast.parse(host.runtime.artifact(record))
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    host.local.require({node.name for node in nodes} == set(names), "missing frozen oracle functions")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, record["path"], "exec"), namespace)


def original_context():
    return host.runtime._trusted_context(json.loads(host.policy.CONTRACT.read_text()), 8)


def reference_context(contract, layer):
    host.local.require(type(layer) is int and layer in LAYERS,
                       "unsupported remaining reference layer")
    rec = contract.get("reference_extension")
    host.local.require(isinstance(rec, dict), "missing original reference extension")
    extension = host.runtime.document(rec)
    base = original_context()
    host.local.require(
        extension["schema"] == SCHEMA and extension["scope"] == SCOPE
        and extension["policy_id"] == contract["policy_id"] == host.policy.POLICY_ID
        and extension["global_reference_policy"] == host.policy.legacy.binary64.REFERENCE_POLICY
        and extension["binary64_profile"] == host.policy.legacy.binary64.PROFILE_ID
        and extension["original_fp16_parent"] == base["fp16_final"]
        and extension["original_binary64_parent"] == base["reference_bindings"]["binary64"],
        "reference extension is not rooted in original independent L8")
    specification = host.runtime.document(base["reference_bindings"]["independent"])
    host.local.require(extension["checkpoint"] == specification["checkpoint"]
                       and set(extension["layers"]) == {str(l) for l in LAYERS},
                       "reference checkpoint/coverage mismatch")
    # Follow reference-only edges. Actual hidden, candidate KV and local references
    # are not permissible roots of either original trajectory.
    fp16_parent, binary64_parent = extension["original_fp16_parent"], extension["original_binary64_parent"]
    for index in LAYERS:
        item = extension["layers"][str(index)]
        host.local.require(item["input_fp16"] == fp16_parent
                           and item["input_binary64"] == binary64_parent
                           and item["prior_kv"] == "own empty P0"
                           and set(item["canonical"]) == set(host.local.tensor_shapes(index)),
                           "spliced/misindexed original reference chain")
        fp16_parent, binary64_parent = item["fp16"], item["binary64"]
    item = extension["layers"][str(layer)]
    with np.load(io.BytesIO(host.runtime.artifact(item["fp16"])), allow_pickle=False) as arrays:
        trajectory = {s: arrays[f"stage{s:02d}"].copy() for s in range(19)}
    for stage, count in host.local.SIZES.items():
        host.local.finite_words(trajectory[stage], (count,))
    binary64 = np.load(io.BytesIO(host.runtime.artifact(item["binary64"])), allow_pickle=False)
    host.local.require(binary64.dtype == np.dtype("<f8") and binary64.shape == (896,)
                       and np.all(np.isfinite(binary64)) and np.all(np.abs(binary64) <= 65504),
                       "invalid original global binary64 reference")
    return {**base, "canonical": item["canonical"], "trajectory": trajectory, "binary64": binary64,
            "reference_bindings": {"extension": rec, "independent": base["reference_bindings"]["independent"],
                                   "fp16": item["fp16"], "binary64": item["binary64"]}}


def validate_plan(plan):
    evaluator.validate_binding(plan.get("evaluator_extension"))
    host.local.require(plan["historical_policy_source"] == host.reviewed_policy_source(),
                       "historical reviewed policy source mismatch")
    host.local.require(plan["scope"] == SCOPE and plan["continuation"]["layers"] == list(LAYERS)
                       and plan["continuation"]["position"] == 0
                       and plan["continuation"]["history"] == [9707]
                       and set(plan["transactions"]) == {str(l) for l in range(8, 24)},
                       "remaining continuation scope mismatch")
    extension = host.runtime.document(plan["reference_extension"])
    host.local.require(extension["scope"] == SCOPE and extension["schema"] == SCHEMA,
                       "unsupported reference extension")
    parent = plan["continuation"]["parent_admission"]
    host.local.require(parent["path"] ==
                       str(PARENT_ROOT / "runtime/layer08/admission/result.json")
                       and plan["transactions"]["8"] == {
                           "directory": str(PARENT_ROOT / "runtime/layer08"),
                           "retained_parent_only": True},
                       "continuation must consume original admitted actual L8")
    _, hidden = host.admitted_parent(plan, 9, parent["sha256"])
    host.local.require(hidden == plan["continuation"]["actual_l8_hidden"],
                       "actual L8 hidden binding mismatch")
    base = original_context()
    host.local.require(plan["semantics"] == base["source"]["semantics"]
                       and plan["public_contract"] == base["source"]["public_contract"],
                       "arithmetic/public contract changed")
    for layer in LAYERS:
        item = plan["transactions"][str(layer)]
        directory = Path(plan["root"]) / "runtime" / f"layer{layer:02d}"
        reference = extension["layers"][str(layer)]
        host.local.require(
            item["directory"] == str(directory) and item["input_state"] is None
            and item["cache_slot"] == 0
            and item["parameters"] == {"LAYER_INDEX": layer, "ACCURATE_SILU": 1}
            and item["canonical_tensors"] == reference["canonical"]
            and item["exact_monitor"] == reference["fp16"]
            and item["compile_argv"] == host.compile_command(
                base["source"], plan["tools"]["verilator"], directory, layer, plan["source_closure"])
            and item["simulation_argv"] == host.simulation_command(directory, layer),
            "noncanonical remaining layer/control/state/command")


def bind_references(out, base, checkpoint):
    """Propagate only missing reference suffixes, starting at retained reference L8."""
    import torch
    import torch.nn.functional as functional

    specification_rec = base["reference_bindings"]["independent"]
    specification = host.runtime.document(specification_rec)
    sources = specification["sources"]
    single = host.runtime.unique(sources, lambda r: r["path"].endswith(
        "/official_single_decoder_layer.py"), "original mathematical helpers")
    dialogue = host.runtime.unique(sources, lambda r: r["path"].endswith(
        "/official_model24_dialogue.py"), "original binary64 recurrence")
    fp16_candidates = [r for r in prior.records(specification) if r["sha256"] == FP16_SOURCE_SHA256]
    host.local.require(bool(fp16_candidates), "original FP16 producer binding missing")
    fp16 = fp16_candidates[0]
    for rec in (single, dialogue, fp16):
        host.authenticate(rec)
    namespace = {
        "np": np, "torch": torch, "torch_functional": functional, "math": math,
        "AWQ_REVERSE_ORDER": (0, 4, 1, 5, 2, 6, 3, 7), "GROUP_SIZE": 128,
        "HEAD_DIM": 64, "HIDDEN_SIZE": 896, "QUERY_HEADS": 14, "KEY_VALUE_HEADS": 2,
        "_require": host.local.require, "require": host.local.require, "DiagnosticError": ValueError,
    }
    definitions(single, ("_torch_unpack", "_torch_linear", "_torch_rmsnorm"), namespace)
    definitions(fp16, ("propagated_reference",), namespace)
    definitions(dialogue, ("_reference_projection", "_reference_layer_step"), namespace)
    torch.set_num_threads(1)
    packages = {name: importlib.metadata.version(name) for name in ("numpy", "torch", "safetensors")}
    host.local.require(packages == specification["python_packages"], "original oracle package drift")
    reference_root = out / "references"
    reference_root.mkdir()
    inputs = {
        "schema": SCHEMA, "scope": SCOPE, "policy_id": host.policy.POLICY_ID,
        "global_reference_policy": host.policy.legacy.binary64.REFERENCE_POLICY,
        "binary64_profile": host.policy.legacy.binary64.PROFILE_ID,
        "original_fp16_parent": base["fp16_final"],
        "original_binary64_parent": base["reference_bindings"]["binary64"],
        "original_specification": specification_rec, "checkpoint": checkpoint,
        "generators": [single, dialogue, fp16, host.record(__file__)],
        "python_packages": packages, "python": host.record(sys.executable),
        "reference_only": True, "accepted_ancestor_oracle_replays": 0,
        "binary64_dequantization": "native unpack; float64 (q-z)*FP16_scale, no FP16 weight rounding",
        "fp16_policy": "w4a16_fp16_interstage; original propagated_reference unchanged",
    }
    host.write(reference_root / "inputs.json", inputs)
    h16 = torch.from_numpy(base["trajectory"][18].view("<f2").astype("<f8").reshape(1, 896))
    h64 = torch.from_numpy(base["binary64"].copy().reshape(1, 896))
    previous16, previous64 = inputs["original_fp16_parent"], inputs["original_binary64_parent"]
    layers = {}
    with safe_open(checkpoint["path"], framework="np") as model, torch.no_grad():
        for layer in LAYERS:
            tensors = {name: np.ascontiguousarray(model.get_tensor(name))
                       for name in host.local.tensor_shapes(layer)}
            canonical = {name: {"name": name, "dtype": str(value.dtype), "shape": list(value.shape),
                                "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
                         for name, value in tensors.items()}
            host.local.authenticate_tensors(tensors, canonical, layer)
            values, k16, v16 = namespace["propagated_reference"](
                h16, tensors, layer, reference_policy="w4a16_fp16_interstage",
                cached_k=None, cached_v=None, position_offset=0)
            arrays = {f"stage{s:02d}": np.asarray(values[0][s], dtype="<f2").view("<u2")
                      for s in range(19)}
            arrays["input_hidden"] = h16.numpy().reshape(-1).astype("<f2").view("<u2")
            for stage, count in host.local.SIZES.items():
                host.local.finite_words(arrays[f"stage{stage:02d}"], (count,))
            host.local.require(k16.shape == v16.shape == (1, 2, 64), "FP16 reference P0 KV mismatch")
            projections = {}
            for key, suffix in (("q", "self_attn.q_proj"), ("k", "self_attn.k_proj"),
                                ("v", "self_attn.v_proj"), ("o", "self_attn.o_proj"),
                                ("gate", "mlp.gate_proj"), ("up", "mlp.up_proj"),
                                ("down", "mlp.down_proj")):
                prefix = f"model.layers.{layer}.{suffix}"
                q = namespace["_torch_unpack"](tensors[prefix + ".qweight"]).to(torch.float64)
                z = namespace["_torch_unpack"](tensors[prefix + ".qzeros"]).to(torch.float64)
                scale = torch.from_numpy(tensors[prefix + ".scales"].astype("<f8"))
                weight = (q - z.repeat_interleave(128, dim=0)) * scale.repeat_interleave(128, dim=0)
                bias = torch.from_numpy(tensors[prefix + ".bias"].astype("<f8")) if key in ("q", "k", "v") else None
                projections[key] = SimpleNamespace(reference_weight=weight, reference_bias=bias)
            state = SimpleNamespace(
                input_norm=tensors[f"model.layers.{layer}.input_layernorm.weight"],
                post_attention_norm=tensors[f"model.layers.{layer}.post_attention_layernorm.weight"],
                projections=projections, reference_k=torch.empty((0, 2, 64), dtype=torch.float64),
                reference_v=torch.empty((0, 2, 64), dtype=torch.float64))
            h64 = namespace["_reference_layer_step"](state, h64, 0)
            binary64 = h64.numpy().reshape(-1).copy()
            host.local.require(binary64.dtype == np.dtype("<f8") and binary64.shape == (896,)
                               and np.all(np.isfinite(binary64)) and np.all(np.abs(binary64) <= 65504)
                               and state.reference_k.shape == state.reference_v.shape == (1, 2, 64),
                               "invalid original binary64 suffix")
            path16 = reference_root / f"layer{layer:02d}_fp16.npz"
            path64 = reference_root / f"layer{layer:02d}_binary64.npy"
            with path16.open("xb") as stream:
                np.savez(stream, **arrays)
            with path64.open("xb") as stream:
                np.save(stream, binary64, allow_pickle=False)
            layers[str(layer)] = {
                "input_fp16": previous16, "input_binary64": previous64, "prior_kv": "own empty P0",
                "canonical": canonical, "fp16": host.record(path16), "binary64": host.record(path64)}
            previous16, previous64 = layers[str(layer)]["fp16"], layers[str(layer)]["binary64"]
            h16 = torch.from_numpy(values[0][18].copy().reshape(1, 896))
    extension = {**inputs, "input_freeze": host.record(reference_root / "inputs.json"), "layers": layers}
    host.write(reference_root / "freeze.json", extension)
    return host.record(reference_root / "freeze.json"), extension


def command_files(plan, out):
    environment = ("set -euC\ncd " + shlex.quote(str(host.ROOT))
                   + "\nexport PYTHONDONTWRITEBYTECODE=1\nexport PYTHONPATH="
                   + shlex.quote(str(host.ROOT) + ":" + str(host.ROOT / "ace3/model"))
                   + "\nexport OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\n")
    records = []
    for layer in LAYERS:
        for action, variable, option in (
            ("capture", "HOST_PARENT_RESULT_SHA256", "--parent-result-sha256"),
            ("admit", "HOST_MANIFEST_SHA256", "--trusted-manifest-sha256")):
            argv = [plan["tools"]["python"]["path"], "-B", "-m",
                    "ace3.model.candidates.host_capture_v3", action,
                    "--plan", str(out / "freeze.json"), "--layer", str(layer)]
            text = environment + ': "${' + variable + ':?separately Host-observed digest required}"\n'
            text += "sha256sum --check " + shlex.quote(str(out / "preexecution.sha256"))
            text += " > " + shlex.quote(str(out / f"{action}-layer{layer:02d}.preexecution.log")) + " 2>&1\n"
            text += "exec " + shlex.join(argv) + " " + option + ' "$' + variable + '"\n'
            path = out / f"{action}-layer{layer:02d}.command.sh"
            with path.open("x") as stream:
                stream.write(text)
            records.append(host.record(path))
    return records


def prepare(out, historical_sources, reuse_reference_extension=None, reference_source_copies=None):
    host.local.require((reuse_reference_extension is None) == (reference_source_copies is None),
                       "reference reuse requires preserved generator source bindings")
    host.fresh(out, host.BUILD).mkdir()
    host.write(out / "prepare.command.json", {"argv": sys.argv, "cwd": str(host.ROOT)})
    original_rec = host.record(PARENT_ROOT / "freeze.json")
    original = host.runtime.document(original_rec)
    base = original_context()
    parent_rec = host.record(PARENT_ROOT / "runtime/layer08/admission/result.json")
    parent = host.runtime.document(parent_rec)
    admission = parent["runtime_admission"]
    host.local.require(admission["saved_state"] == {
        "idle": True, "layer": 8, "state_restore_or_eval_performed": False,
        "status": "PASS", "valid_entries": 128}, "accepted L8 own-layer state evidence mismatch")
    manifest = host.runtime.document(admission["manifest"])
    transaction = host.runtime.document(manifest["transaction"])
    host.local.require(admission["output"] == transaction["output"]
                       and admission["output_state"] == transaction["output_state"],
                       "original L8 actual output/state receipt mismatch")
    actual = host.runtime.hex_words(host.runtime.artifact(admission["output"]), indexed=True)
    host.local.finite_words(actual, (896,))
    host.local.require(hashlib.sha256(actual.tobytes()).hexdigest() ==
                       admission["output"]["semantic_sha256"], "actual L8 semantic mismatch")
    review = host.review(host.REVIEWS / "d986b19b9f95/round-0001.json", "d986b19b9f95")
    l5_review = host.review(host.REVIEWS / "34b40381decf/round-0002.json", "34b40381decf")
    preserved = json.loads(historical_sources.read_text())
    for item in preserved:
        old = host.runtime.unique(original["bindings"],
                                  lambda r: r == item["original"], "accepted producer revision")
        host.local.require((old["bytes"], old["sha256"]) ==
                           (item["preserved"]["bytes"], item["preserved"]["sha256"]),
                           "preserved producer revision differs")
        host.authenticate(item["preserved"])
    tools = original["tools"]
    for name, flag in (("verilator", "--version"), ("iverilog", "-V"),
                       ("make", "--version"), ("g++", "--version")):
        executable = shutil.which(name)
        host.local.require(executable is not None and host.record(executable) == tools[name],
                           f"unconfirmed/changed host tool: {name}; no RTL conclusion")
        host.observe([tools[name]["path"], flag], out, name + "_version")
    host.local.require(host.record(sys.executable) == tools["python"], "frozen Python mismatch")
    host.authenticate(tools["verilator_bin"])
    host.local.require(host.harness.compatible_sources(
        host.runtime.source_map(original["source_closure"]),
        host.runtime.source_map(base["source"]["source_closure"]), host.runtime.artifact),
        "accepted RTL/capture source incompatibility")
    public = out / "public_contract.sv"
    header = base["source"]["public_contract"]
    top = host.runtime.artifact(host.runtime.unique(original["source_closure"],
        lambda r: Path(r["path"]).name == host.runtime.TOP + ".sv", "exact public top")).decode("ascii")
    match = re.search(r"\bmodule\s+" + host.runtime.TOP + r"\s*#\(.*?\);", top, re.S)
    host.local.require(match is not None and match[0] == header, "public header drift")
    with public.open("x") as stream:
        stream.write(header + "\nendmodule\nmodule frozen_contract;\n")
        for layer in LAYERS:
            stream.write(f"{host.runtime.TOP} #(.LAYER_INDEX({layer}), .ACCURATE_SILU(1)) layer{layer}();\n")
        stream.write("endmodule\n")
    host.observe([tools["iverilog"]["path"], "-g2012", "-s", "frozen_contract",
                  "-o", str(out / "public_contract.vvp"), str(public)], out, "public_contract_compile")
    host.authenticate(original["checkpoint"])
    historical_policy = host.reviewed_policy_source()
    source_copies = []
    if reuse_reference_extension is None:
        extension_rec, extension = bind_references(out, base, original["checkpoint"])
    else:
        extension_rec = host.record(reuse_reference_extension)
        extension = host.runtime.document(extension_rec)
        source_copies = json.loads(reference_source_copies.read_text())
        host.local.require(extension["checkpoint"] == original["checkpoint"],
                           "reused original reference checkpoint mismatch")
        inputs = host.runtime.document(extension["input_freeze"])
        host.local.require(inputs == {k: v for k, v in extension.items()
                                     if k not in ("input_freeze", "layers")},
                           "reused reference input freeze mismatch")
        reference_context(dict(json.loads(host.policy.CONTRACT.read_text()),
                               reference_extension=extension_rec), 23)
    replacements = {}
    for item in source_copies:
        old, saved = item["original"], item["preserved"]
        host.local.require((old["bytes"], old["sha256"]) == (saved["bytes"], saved["sha256"]),
                           "preserved reference generator differs")
        host.authenticate(saved)
        replacements[(old["path"], old["sha256"], old["bytes"])] = saved
    reference_bindings = []
    for rec in prior.records(extension):
        selected = replacements.get((rec["path"], rec["sha256"], rec["bytes"]), rec)
        host.authenticate(selected)
        reference_bindings.append(selected)
    plan = {key: original[key] for key in (
        "schema", "policy_id", "semantics", "source_closure", "accepted_source_closure",
        "checkpoint", "tools", "independent_p0_rope", "monitor_boundary", "capture_mode", "external_trust")}
    plan.update(
        root=str(out), scope=SCOPE, public_contract=header, status="PREEXECUTION_READY_NOT_CAPTURED",
        decoder_rtl_invocations=0, numerical_status="NOT_EVALUATED", runtime_admission="NOT_EVALUATED",
        independent_review="REQUIRED", reference_extension=extension_rec,
        evaluator_extension=evaluator.binding(), historical_policy_source=historical_policy,
        reference_source_copies=source_copies,
        public_contract_compile=host.record(out / "public_contract_compile.process.json"),
        historical_producer_sources=preserved,
        continuation={
            **SCOPE, "parent_admission": parent_rec, "actual_l8_hidden": admission["output"],
            "l8_state_evidence_only": admission["output_state"],
            "prior_kv": "each layer owns empty P0 prior KV; no state restore",
            "accepted_review": review, "accepted_l5_review": l5_review,
            "capture_timeout_seconds": 7200, "launch_owner": "parent Host after independent review"},
        transactions={"8": {"directory": str(PARENT_ROOT / "runtime/layer08"),
                            "retained_parent_only": True}})
    host.admitted_parent(plan, 9, parent_rec["sha256"])
    for layer in LAYERS:
        directory = host.fresh(out / "runtime" / f"layer{layer:02d}", out)
        plan["transactions"][str(layer)] = {
            "directory": str(directory), "parameters": {"LAYER_INDEX": layer, "ACCURATE_SILU": 1},
            "compile_argv": host.compile_command(base["source"], tools["verilator"], directory,
                                                 layer, plan["source_closure"]),
            "simulation_argv": host.simulation_command(directory, layer),
            "canonical_tensors": extension["layers"][str(layer)]["canonical"],
            "exact_monitor": extension["layers"][str(layer)]["fp16"],
            "input_state": None, "cache_slot": 0,
            "parent_rule": "preceding admitted actual output only; own empty P0 KV"}
    host.adapter()
    bindings = [original_rec, parent_rec, review, l5_review, host.record(historical_sources),
                admission["output"], admission["output_state"], admission["manifest"], manifest["transaction"],
                host.record(public), extension_rec, host.record(host.policy.CONTRACT),
                *plan["source_closure"], *plan["accepted_source_closure"], *tools.values(),
                original["checkpoint"], plan["independent_p0_rope"]]
    bindings.extend(reference_bindings)
    bindings.extend(prior.records(plan["evaluator_extension"]))
    bindings.append(historical_policy)
    if reference_source_copies is not None:
        bindings.append(host.record(reference_source_copies))
    bindings.extend(item["preserved"] for item in preserved)
    modules = {Path(m.__file__).resolve() for m in tuple(sys.modules.values())
               if getattr(m, "__file__", None) and Path(m.__file__).is_file()}
    bindings.extend(host.record(p) for p in sorted(modules)
                    if p.is_relative_to(host.ROOT) or p.suffix in (".so", ".pyd"))
    unique = {}
    for rec in bindings:
        signature = {key: rec[key] for key in ("path", "bytes", "sha256")}
        host.local.require(unique.setdefault(rec["path"], signature) == signature,
                           "conflicting continuation binding")
    plan["bindings"] = list(unique.values())
    commands = command_files(plan, out)
    plan["continuation"]["command_files"] = commands
    host.write(out / "freeze.json", plan)
    with (out / "preexecution.sha256").open("x") as stream:
        for rec in [host.record(out / "freeze.json"), *plan["bindings"], *commands]:
            stream.write(f"{rec['sha256']}  {rec['path']}\n")
    host.write(out / "preparation_result.json", {
        "status": "PREEXECUTION_READY_NOT_CAPTURED", "plan": host.record(out / "freeze.json"),
        "decoder_rtl_invocations": 0, "accepted_ancestor_replays": 0,
        "new_original_reference_layers": list(LAYERS) if reuse_reference_extension is None else [],
        "reused_original_reference_layers": list(LAYERS) if reuse_reference_extension is not None else [],
        "public_contract_compiled_layers": list(LAYERS),
        "numerical_status": "NOT_EVALUATED", "independent_review": "REQUIRED",
        "boundary": "Original-reference and launch binding only; no candidate suffix or model admission"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--historical-sources", type=Path, required=True)
    parser.add_argument("--reuse-reference-extension", type=Path)
    parser.add_argument("--reference-source-copies", type=Path)
    args = parser.parse_args()
    prepare(args.out, args.historical_sources, args.reuse_reference_extension, args.reference_source_copies)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
