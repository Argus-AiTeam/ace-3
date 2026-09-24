#!/usr/bin/env python3
"""Read-only lineage admission; compile public tops without executing tail RTL."""

from __future__ import annotations

import argparse
import ast
import copy
import difflib
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_position2_tail_binding as binding

ROOT = binding.ROOT


def write_new(path: Path, value: object) -> None:
    with path.open("x", encoding="ascii") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def records(value: object):
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"} <= value.keys():
            yield value
        else:
            for item in value.values():
                yield from records(item)
    elif isinstance(value, list):
        for item in value:
            yield from records(item)


def audit_retained_layers(layers: dict, prepared_inputs: list, hidden_size: int) -> list:
    require = binding.require
    require(set(layers) == {21, 22, 23}, "incomplete retained layer ancestry")
    links = []
    for index in (21, 22, 23):
        layer = layers[index]
        require(layer["layer_index"] == index and layer["actual_output_fed_rtl_chain"]
                and (index == 21 or "earliest_material_mismatch" in layer)
                and layer.get("earliest_material_mismatch") is None
                and [row["position"] for row in layer["positions"]] == [0, 1, 2]
                and [row["stage"] for row in layer["independent_comparisons"]] == list(range(19))
                and all(row["failure_count"] == 0 for row in layer["independent_comparisons"]),
                f"incomplete retained layer{index} comparison evidence")
        for tx in layer["positions"]:
            binding.authenticate(tx["output"])
            bits = binding.head.load_terminal_bits(Path(tx["output"]["path"]))
            require(len(bits) == hidden_size
                    and all(0 <= word <= 0xffff and (word & 0x7c00) != 0x7c00
                            for word in bits),
                    f"incomplete or nonfinite retained layer{index} output")
            semantic = binding.head.sha256_bytes(binding.head.terminal_payload(bits))
            require(semantic == tx["output"]["semantic_sha256"], "hidden semantic digest mismatch")
            if index > 21:
                preceding = layers[index - 1]["positions"][tx["position"]]["output"]
                require(tx["input"]["sha256"] == preceding["semantic_sha256"],
                        f"broken layer{index - 1}->layer{index} hidden ancestry")
                links.append({"from_layer": index - 1, "to_layer": index,
                              "position": tx["position"], "parent_output": preceding,
                              "child_input": tx["input"], "child_output": tx["output"]})
    require(prepared_inputs == [tx["output"] for tx in layers[21]["positions"]],
            "prepared layer22 input differs from layer21")
    return links


def probe_rmsnorm_policy(source: str) -> dict:
    """Evaluate only source-extracted error expressions on synthetic FP16 pairs."""
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "fp16_policy_comparison"]
    binding.require(len(functions) == 1, "missing unique RMSNorm comparison")
    function = functions[0]
    relative = [node.value for node in ast.walk(function) if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "relative"
                        for target in node.targets)]
    predicates = [node.test for node in ast.walk(function) if isinstance(node, ast.If)
                  and any(isinstance(name, ast.Name) and name.id == "ABSOLUTE_TOLERANCE"
                          for name in ast.walk(node.test))]
    binding.require(len(relative) == len(predicates) == 1,
                    "unrecognized RMSNorm comparison expressions")
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in (
                        "ABSOLUTE_TOLERANCE", "RELATIVE_TOLERANCE", "MAX_ULP_DISTANCE"):
                    constants[target.id] = ast.literal_eval(node.value)
    binding.require(constants == {"ABSOLUTE_TOLERANCE": .125,
                                  "RELATIVE_TOLERANCE": .001, "MAX_ULP_DISTANCE": 1},
                    "RMSNorm tolerance drift")
    relative_code = compile(ast.Expression(relative[0]), "<source-relative>", "eval")
    predicate_code = compile(ast.Expression(predicates[0]), "<source-failure>", "eval")
    cases = []
    for name, reference, actual in (
        ("exact", 256, 256),
        ("inclusive_absolute", 1, 1.125),
        ("one_ulp_relative_pass", 1024, 1025),
        ("relative_equality_two_ulp", 1000, 1001),
        ("relative_pass_two_ulp", 1023.5, 1022.5),
        ("negative_relative_equality", -1000, -1001),
        ("all_thresholds_exceeded", 1000, 1002),
        ("zero_to_subnormal", 0, 2**-24),
    ):
        values = binding.np.asarray([reference, actual], dtype="<f2")
        bits = values.view("<u2")
        binding.require(binding.np.isfinite(values).all()
                        and values.tolist() == [reference, actual],
                        "synthetic policy control is not exactly finite FP16")
        def ordered(raw: int) -> int:
            return 0x8000 - (raw & 0x7fff) if raw & 0x8000 else 0x8000 + raw
        distance = abs(ordered(int(bits[0])) - ordered(int(bits[1])))
        error = abs(Fraction(actual) - Fraction(reference))
        ratio = error / max(abs(Fraction(reference)), Fraction(1, 2**14))
        oracle_failure = error > Fraction(1, 8) and (
            ratio >= Fraction(1, 1000) or distance > 1)
        operands = dict(constants, np=binding.np, token=0, index=0, distance=distance,
                        absolute=binding.np.asarray([[float(error)]]),
                        expected_values=binding.np.asarray([[float(reference)]]))
        operands["relative"] = eval(relative_code, {"__builtins__": {}}, operands)
        failure = bool(eval(predicate_code, {"__builtins__": {}}, operands))
        cases.append({"case": name, "reference_fp16": f"{int(bits[0]):04x}",
                      "actual_fp16": f"{int(bits[1]):04x}", "absolute_error": str(error),
                      "relative_error_exact": str(ratio), "ordered_fp16_ulp": distance,
                      "source_failure": failure, "required_policy_failure": oracle_failure})
    return {"relative_expression": ast.unparse(relative[0]),
            "failure_expression": ast.unparse(predicates[0]), "cases": cases,
            "required_policy_disagreements": sum(
                row["source_failure"] != row["required_policy_failure"] for row in cases),
            "boundary": "Synthetic finite-FP16 gate controls only; no model outputs or RTL."}


def audit(output: Path, mission: Path) -> None:
    require = binding.require
    load = binding.load
    record = binding.record
    require(output.parent == ROOT / "build", "output must be directly under build")
    require(not os.path.lexists(output), "attempt already exists")
    require(mission.name == "mission.json", "explicit mission contract required")
    contract = load(binding.CONTRACT)
    head_contract = load(ROOT / "ace3/contracts/streaming_tied_lm_head_topk.json")
    previous_root = ROOT / "build/position2_tail_admission_c9a75df8604d_attempt001"
    previous = load(previous_root / "result.json")
    review_path = mission.parent.parent / "c9a75df8604d/round-0001.json"
    review = load(review_path)
    require(review["producer_role"] == "reviewer"
            and review["mission_id"] == "c9a75df8604d"
            and review["review"]["status"] == "done",
            "missing reviewed predecessor blocker")
    legacy_path = ROOT / "build/position2_tail_binding_attempt002/runtime_pass_package.json"
    legacy = load(legacy_path)
    prior_freeze = load(previous_root / "frozen.json")
    binding.authenticate(prior_freeze["retained_package"])
    binding.authenticate(previous["frozen"])
    blocked_path = ROOT / "build/position2_tail_admission_05d5102ed6fd_attempt004/result.json"
    blocked = load(blocked_path)
    binding.authenticate(blocked["frozen"])
    require(blocked["classification"] == "source_compatibility_not_admitted"
            and blocked["runtime_tail_invocations"] == 0
            and blocked["authority_created"] is False, "attempt004 blocker identity changed")
    output.mkdir()
    namespaces = lambda: {
        name: os.path.lexists(output / name)
        for name in contract["admission"]["runtime_namespaces_must_be_absent"]
    }
    compiler = shutil.which("iverilog")
    require(compiler is not None,
            "compiler unconfirmed: inspect declared local containers before reporting unavailable")
    compiler = str(Path(compiler).resolve())
    sources = []
    paths = set(binding.source_paths())
    paths.add(Path(__file__).resolve())
    paths.add(ROOT / "ace3/model/tests/test_audit_position2_tail_lineage.py")
    paths.add(ROOT / "ace3/model/run_position2_tail.py")
    for path in sorted(paths):
        target = output / "sources" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(path.read_bytes())
        sources.append({"current": record(path), "frozen": record(target)})
    interfaces = []
    plans = []
    for key in ("final_rmsnorm", "tied_head"):
        section = contract[key]
        top = section["top"]
        text = (output / "sources/ace3/rtl" / f"{top}.sv").read_text()
        declaration = re.search(r"module\s+" + re.escape(top) + r"\s*#\(.*?\n\);",
                                text, re.S)
        require(declaration is not None, f"missing public declaration: {top}")
        interfaces.append({**section, "declaration": declaration.group(0)})
        plans.append([compiler, "-g2012", "-s", top,
                      *[f"-P{top}.{k}={v}" for k, v in section["parameters"].items()],
                      "-o", str(output / f"{top}.vvp"),
                      *[str(output / "sources" / name) for name in binding.RTL]])
    frozen = {
        "mission": record(mission), "objective": load(mission)["objective"],
        "reviewed_blocker": record(review_path), "prior_result": record(previous_root / "result.json"),
        "prior_freeze": record(previous_root / "frozen.json"),
        "attempt004_result": record(blocked_path), "attempt004_freeze": blocked["frozen"],
        "contract": contract, "contract_identity": record(binding.CONTRACT),
        "head_contract": head_contract, "sources": sources,
        "public_interfaces": interfaces, "compile_commands": plans,
        "compiler": record(Path(compiler)), "python_version": sys.version,
        "environment": {key: os.environ.get(key) for key in binding.BUILD_ENVIRONMENT},
        "compiler_version": subprocess.check_output(
            [compiler, "-V"], stderr=subprocess.STDOUT, text=True, timeout=30),
        "score_policy": "Identity/ancestry/source admissibility only; no numerical re-evaluation.",
        "runtime_tail_invocations": 0, "execution_authority": False,
        "namespace_presence_before": namespaces(),
    }
    write_new(output / "frozen.json", frozen)
    compiles = [binding.log_command(output, section["top"], argv)
                for section, argv in zip(interfaces, plans, strict=True)]

    parent, _ = binding.admitted_parent(contract)
    require(parent == previous["blocker"]["contract_parent"], "reviewed candidate identity changed")
    launch_freeze = load(binding.PARENT / "frozen.json")
    for key in ("predecessor", "predecessor_seal", "preparation_manifest",
                "preparation_seal", "review"):
        binding.authenticate(launch_freeze[key])
    manifest = load(Path(launch_freeze["preparation_manifest"]["path"]))
    layer22_path = Path(launch_freeze["predecessor"]["path"])
    execution22_path = layer22_path.parent / "execution.json"
    execution22 = load(execution22_path)
    binding.authenticate(execution22["predecessor"])
    layer21_path = Path(execution22["predecessor"]["path"])
    layer21_record = record(layer21_path)
    parent_seal = load(Path(manifest["parent_seal"]["path"]))
    require(layer21_record in list(records(parent_seal)),
            "layer21 result is not bound by accepted parent seal")
    for key in ("parent_result", "parent_seal", "parent_review", "layer21_kv_ancestry"):
        binding.authenticate_tree(manifest[key])
    review21 = load(Path(manifest["parent_review"]["path"]))
    review22 = load(Path(launch_freeze["review"]["path"]))
    require(review21["producer_role"] == "reviewer"
            and review21["review"]["status"] == "done",
            "layer21 independent acceptance missing")
    require(review22["producer_role"] == "reviewer"
            and review22["review"]["status"] == "continue"
            and review22["review"]["reason"].startswith("Independent review admits layer22:"),
            "layer22 explicit independent admission missing")
    require(execution22["preparation_seal"] == launch_freeze["preparation_seal"],
            "layer22 preparation ancestry mismatch")
    layers = {21: load(layer21_path), 22: load(layer22_path),
              23: load(Path(parent["result"]["path"]))}
    links = audit_retained_layers(layers, manifest["layer22_hidden"],
                                 contract["parent"]["hidden_size"])

    foreign = copy.deepcopy(legacy)
    foreign["parent"]["attempt_id"] = "model24_layer23_attempt001"
    foreign["parent"]["result"] = previous["requested_parent_authenticated_artifacts"][0]
    rejections = []
    for name, package, prefix in (
        ("foreign_parent", foreign, "foreign_parent"),
        ("stale_retained_package", legacy, "artifact drift:"),
    ):
        try:
            binding.admit(package, output, contract)
        except binding.evidence.AttemptError as error:
            require(str(error).startswith(prefix), f"unexpected {name} rejection: {error}")
            rejections.append({"case": name, "reason": str(error), "rejected": True})
        else:
            raise binding.evidence.AttemptError(f"unexpected admission: {name}")

    producer_drift = []
    for expected in manifest["sources"]:
        actual = record(Path(expected["path"]))
        if actual != expected:
            producer_drift.append({"expected": expected, "current": actual,
                                   "disposition": "Current-source semantic/ABI compatibility not admitted by historical reviews."})
    tail_drift = []
    for row in legacy["sources"]:
        binding.authenticate(row["frozen"])
        current = record(Path(row["current"]["path"]))
        if current != row["current"]:
            old = Path(row["frozen"]["path"]).read_text().splitlines(keepends=True)
            new = Path(current["path"]).read_text().splitlines(keepends=True)
            tail_drift.append({
                "expected": row["current"], "current": current,
                "diff": "".join(difflib.unified_diff(old, new, fromfile=row["frozen"]["path"],
                                                   tofile=current["path"])),
                "disposition": "Retained package is stale; no substitution, policy edit, or numerical PASS.",
            })
    require(tail_drift, "retained stale-source blocker no longer reproduced")
    runner = ROOT / "ace3/model/run_final_rmsnorm_from_layer23.py"
    retained_runner = next(row for row in legacy["sources"]
                           if row["current"]["path"] == str(runner))
    retained_text = Path(retained_runner["frozen"]["path"]).read_text()
    current_text = runner.read_text()
    retained_tree, current_tree = ast.parse(retained_text), ast.parse(current_text)
    for tree in (retained_tree, current_tree):
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.FunctionDef) and node.name == "fp16_policy_comparison")]
    comparison_only = ast.dump(retained_tree) == ast.dump(current_tree)
    for rows in (producer_drift, tail_drift):
        for row in rows:
            if row["current"]["path"] == str(runner) and comparison_only:
                row["disposition"] = (
                    "Only fp16_policy_comparison changed; no producer RTL/arithmetic/ABI "
                    "change is established by this drift. Comparison equivalence is rejected.")
    legacy_expression = "(absolute > 0.125) & (relative > 0.001) & (ulp > 1)"
    for relative_path, function_name, target_name in (
        ("ace3/model/prepare_position2_tail_binding.py", "rms_expectation", "failures"),
        ("ace3/model/run_position2_tail.py", "execute", "material"),
    ):
        tree = ast.parse((ROOT / relative_path).read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == function_name)
        expressions = [ast.unparse(node.value) for node in ast.walk(function)
                       if isinstance(node, ast.Assign) and any(
                           isinstance(target, ast.Name) and target.id == target_name
                           for target in node.targets)]
        denominators = [ast.unparse(node.value) for node in ast.walk(function)
                        if isinstance(node, ast.Assign) and any(
                            isinstance(target, ast.Name) and target.id == "relative"
                            for target in node.targets)]
        # execute() also computes head errors; bind only its mathematical RMSNorm gate.
        expressions = [expression for expression in expressions
                       if legacy_expression in expression]
        denominators = [expression for expression in denominators
                        if "mathematical" in expression]
        require(len(expressions) == len(denominators) == 1
                and legacy_expression in expressions[0]
                and denominators[0].endswith(", 1e-30)"),
                f"tail policy changed; reassess compatibility: {relative_path}")
    policies = {"retained": probe_rmsnorm_policy(retained_text),
                "current": probe_rmsnorm_policy(current_text)}
    require(policies["current"]["required_policy_disagreements"] == 0,
            "current RMSNorm source violates frozen synthetic policy controls")
    witnesses = [old["case"] for old, new in zip(
        policies["retained"]["cases"], policies["current"]["cases"], strict=True)
        if old["source_failure"] != new["source_failure"]]
    require(witnesses, "source policy incompatibility no longer reproduced")
    compatibility = {
        "retained_runner": retained_runner["frozen"], "current_runner": record(runner),
        "drift_confined_to_fp16_policy_comparison": comparison_only,
        "policies": policies, "non_equivalence_witnesses": witnesses,
        "disposition": "Not semantically equivalent. Preserve the current stricter gate; "
                       "restoring the retained predicate would weaken acceptance.",
        "remaining_incompatibility": {
            "contract": record(binding.CONTRACT),
            "retained_rule": contract["final_rmsnorm"]["mathematical_material_failure"],
            "preparation": record(ROOT / "ace3/model/prepare_position2_tail_binding.py"),
            "runtime": record(ROOT / "ace3/model/run_position2_tail.py"),
            "detail": "The retained tail contract, rms_expectation(), and execute() retain "
                      "the AND/> predicate; their relative denominator floor is 1e-30. "
                      "The current FP16-interstage comparison uses AND (>= OR >), floor "
                      "2^-14, and a different FP32/FP16-interstage reference recurrence. "
                      "These are distinct comparison boundaries, not an RTL arithmetic "
                      "repair. Unifying them needs explicit contract/reference mapping "
                      "and edits outside this mission's writable source paths.",
        },
        "scope": "No decoder arithmetic/ABI incompatibility inferred from this helper drift; "
                 "no retained numerical result relabeled or re-evaluated.",
    }
    checkpoint = record(binding.CHECKPOINT)
    require(checkpoint["sha256"] == contract["model"]["checkpoint_sha256"]
            == head_contract["model_binding"]["checkpoint_sha256"], "official checkpoint drift")
    tensor_records = {}
    with binding.CHECKPOINT.open("rb") as stream:
        header_bytes = int.from_bytes(stream.read(8), "little")
        header = json.loads(stream.read(header_bytes))
        for name, expected in legacy["tensors"].items():
            if name not in ("model.norm.weight", "model.embed_tokens.weight", "lm_head.weight"):
                continue
            tensor = header[name]
            start, end = tensor["data_offsets"]
            stream.seek(8 + header_bytes + start)
            hasher = hashlib.sha256()
            remaining = end - start
            while remaining:
                chunk = stream.read(min(4 * 1024 * 1024, remaining))
                require(bool(chunk), f"truncated tensor: {name}")
                hasher.update(chunk)
                remaining -= len(chunk)
            actual = {"dtype": tensor["dtype"], "shape": tensor["shape"],
                      "offset": 8 + header_bytes + start, "bytes": end - start,
                      "sha256": hasher.hexdigest()}
            require(actual == expected == previous["official_tensors"][name],
                    f"official tensor identity mismatch: {name}")
            tensor_records[name] = actual
    require(set(tensor_records) == {"model.norm.weight", "model.embed_tokens.weight", "lm_head.weight"},
            "missing required official tensors")
    require(tensor_records["lm_head.weight"]["sha256"]
            == tensor_records["model.embed_tokens.weight"]["sha256"]
            == head_contract["model_binding"]["tied_value_sha256"], "head is not tied")
    historical = ROOT / "build/position2_tail_runtime_attempt001"
    historical_result = load(historical / "result.json")
    require(load(historical / "frozen.json")["parent"] == parent
            and historical_result["runtime_tail_invocations"] == 1,
            "historical consumed runtime identity mismatch")
    for key in ("result", "seal"):
        binding.authenticate(previous["existing_contract_parent_runtime"][key])
    for row in sources:
        binding.authenticate_tree(row)
    require(not any(namespaces().values()), "runtime namespace created")
    result = {
        "schema": "ace3_position2_tail_lineage_admission_v1",
        "status": "BLOCKED", "mission_id": mission.parent.name,
        "classification": "source_compatibility_not_admitted",
        "parent_selection": "Resolved by live objective: active contract parent only.",
        "parent": parent, "layer21": layer21_record, "layer22": record(layer22_path),
        "layer22_execution": record(execution22_path),
        "ancestry_links": links, "preparation": launch_freeze["preparation_manifest"],
        "ancestry_reviews": [manifest["parent_review"], launch_freeze["review"], parent["review"]],
        "retained_comparison_boundary": "Historical 19-stage comparisons, not fresh numerical evaluation.",
        "layer21_schema_disposition": {
            "earliest_material_mismatch_present": "earliest_material_mismatch" in layers[21],
            "disposition": "No recorded mismatch; legacy absence is accepted only for layer21 after complete-output, 19-stage zero-failure, finite-output, ancestry, and independent-review checks.",
        },
        "rejected_parent": previous["blocker"]["objective_parent"],
        "regressions": rejections, "producer_source_drift": producer_drift,
        "source_compatibility": compatibility,
        "tail_source_drift": tail_drift, "checkpoint": checkpoint, "official_tensors": tensor_records,
        "historical_runtime": previous["existing_contract_parent_runtime"],
        "failure_taxonomy": "source_compatibility_not_admitted",
        "root_cause_hypothesis": "Comparison-policy drift, not demonstrated producer arithmetic drift: source-extracted finite-FP16 controls distinguish the retained and current predicates.",
        "regression": "Source-extracted predicates versus exact Fraction controls, plus stale-package and foreign-parent rejection.",
        "blocker": compatibility["remaining_incompatibility"],
        "next_state": "BLOCKED_NO_EXECUTION_AUTHORITY",
        "execution_authority": False, "authority_created": False, "unconsumed_authority_count": 0,
        "runtime_tail_invocations": 0, "compile_only_invocations": len(compiles),
        "compile_results": compiles, "namespace_presence_after": namespaces(),
        "frozen": record(output / "frozen.json"),
        "independent_review": "pending normal Host Reviewer",
        "claim_boundary": "No new RTL simulation, numerical PASS, tail selection, dialogue, or hardware evidence.",
    }
    write_new(output / "runtime_pass_package.json", result)
    write_new(output / "result.json", result)
    print(json.dumps({"status": result["status"], "parent": parent["attempt_id"],
                      "ancestry_links": len(links), "producer_source_drift": len(producer_drift),
                      "tail_source_drift": len(tail_drift),
                      "next_state": result["next_state"], "runtime_tail_invocations": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mission", required=True, type=Path)
    args = parser.parse_args()
    audit(args.output.absolute(), args.mission.absolute())
