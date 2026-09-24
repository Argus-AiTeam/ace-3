"""Non-admitting L9/P0 coordinate-62 entry/producer cuts with native L10-L13."""

import argparse
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import py_compile
import sys
import time
import traceback
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import torch.nn.functional as functional
from safetensors import safe_open

from ace3.model.candidates import diagnose_q24_s16_l12_coordinate62_producer_cone_v1 as producer


coordinate = producer.coordinate
upstream = producer.upstream
paired = producer.paired
prior = producer.prior
native = producer.native
ROOT = producer.ROOT
ID = "ace3-q24-s16-l9-coordinate62-entry-producer-cone-v1"
MODULE = "ace3.model.candidates.diagnose_q24_s16_l9_coordinate62_entry_producer_cone_v1"
TEST_MODULE = "tests.test_q24_s16_l9_coordinate62_entry_producer_cone_v1"
CONTRACT = ROOT / "ace3/contracts/candidates/q24_s16_l9_coordinate62_entry_producer_cone_v1.json"
REVIEWED = ROOT / "build/q24_s16_l12_coordinate62_producer_cone_eaee0c6ac5e4_attempt001"
REVIEWED_SHA = "54260b6d161471d9aaf3021c271cd7281e604e94fe0147a2e2a10cc27e86bd3e"
L8_REVIEW = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/dc98a86e7c02/round-0003.json")
L8_REVIEW_SHA = "9edfeb4b588ed9de86dc584d3619bb18c0fd04acda9557d8c310c25513fba76e"
require = prior.require
record = paired.record
write = paired.write


def plan():
    return producer.plan()[:8] + [
        ("inherited_native", ("inherited_native",)),
        ("mapped62", ("mapped62",)), ("mapped_all", ("mapped_all",)),
    ]


def origins():
    result = {}
    for name, module in sorted(list(sys.modules.items())):
        if name not in ("ace3", "tests") and not name.startswith(("ace3.", "tests.")):
            continue
        expected = ROOT.joinpath(*name.split("."))
        filename = getattr(module, "__file__", None)
        if filename is None:
            paths = list(getattr(module, "__path__", ()))
            require(paths and all(Path(p).resolve() == expected for p in paths),
                    f"namespace origin mismatch: {name}")
            result[name] = {"namespace_paths": paths}
        else:
            path = Path(filename).resolve()
            require(path.is_relative_to(ROOT)
                    and path in (expected.with_suffix(".py"), expected / "__init__.py"),
                    f"module origin mismatch: {name}")
            result[name] = record(path)
    require(MODULE in result and TEST_MODULE in result, "missing candidate/test origins")
    return result


def validate(out):
    require(Path.cwd() == ROOT and Path(sys.executable) == prior.PYTHON
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "use the published repository-bound command")
    require(Path(__file__).resolve() == ROOT.joinpath(*MODULE.split(".")).with_suffix(".py"),
            "entry source origin mismatch")
    tests = importlib.import_module(TEST_MODULE)
    source_origins = origins()
    compiled = []
    for index, path in enumerate((Path(__file__).resolve(), Path(tests.__file__).resolve())):
        py_compile.compile(str(path), cfile=str(out / f"compiled_{index}.pyc"), doraise=True)
        compiled.append(record(path))
    contract = json.loads(CONTRACT.read_text())
    require(CONTRACT.resolve().is_relative_to(ROOT)
            and contract["diagnostic_id"] == ID and contract["policy_id"] == prior.gates.POLICY_ID
            and contract["reviewed_result_sha256"] == REVIEWED_SHA
            and contract["coordinate_result_sha256"] == producer.REVIEWED_SHA
            and contract["upstream_result_sha256"] == coordinate.REVIEWED_SHA
            and contract["controls"] == [label for label, _ in plan()]
            and contract["normal_host_review"] == "REQUIRED" and contract["rtl_invocations"] == 0
            and all(contract[k] is False for k in
                    ("candidate_admitted", "policy_adopted", "successor_published")),
            "entry producer contract mismatch")
    with (out / "unittest.log").open("x") as log:
        suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
        count = suite.countTestCases()
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    write(out / "validation.json", {
        "cwd": str(ROOT), "executable": sys.executable, "PYTHONPATH": os.environ["PYTHONPATH"],
        "sys_path": sys.path, "origins": source_origins, "compiled": compiled,
        "contract": record(CONTRACT), "collected": count, "executed": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors),
        "skipped": len(result.skipped), "accepted_L0_L8_tests_executed": False,
    })
    require(count == tests.EXPECTED_TESTS and result.testsRun == count
            and result.wasSuccessful() and not result.skipped, "focused validation failed")
    return contract


def check_review(item):
    require(Path(item["path"]) == L8_REVIEW and item["sha256"] == L8_REVIEW_SHA
            and record(L8_REVIEW) == item, "accepted L8 review binding mismatch")
    review = json.loads(L8_REVIEW.read_text())
    require(review["kind"] == "round_reviewed_handoff"
            and review["producer_role"] == "reviewer" and review["mission_id"] == "dc98a86e7c02"
            and review["round"] == 3 and review["review"]["status"] == "done",
            "missing independent accepted L8 review")
    return review


def authenticate():
    data = producer.authenticate()
    bind = lambda item: upstream.bind_input(item, data["bound"])
    read = lambda item: json.loads(bind(item).read_text())
    archive = lambda item: native.load_state(item, bind)
    item = record(REVIEWED / "result.json")
    require(item["sha256"] == REVIEWED_SHA, "wrong reviewed L12 producer result")
    reviewed = read(item)
    require(reviewed["diagnostic_id"] == producer.ID and reviewed["status"] == "DIAGNOSED"
            and reviewed["native_retained_bitwise_reproduction"] is True
            and reviewed["original_L12_S18_bitwise_reproduction"] is True
            and reviewed["unchanged_baseline_gates"] is True
            and reviewed["original_global_reference_unchanged"] is True
            and reviewed["source_operand_state_KV_lineage_checks"] == "PASS"
            and reviewed["rtl_invocations"] == 0
            and all(reviewed[k] is False for k in
                    ("candidate_admitted", "policy_adopted", "successor_published")),
            "reviewed L12 producer scope mismatch")
    for item in reviewed["input_bindings"] + reviewed["artifacts"] + [reviewed["validation"]]:
        bind(item)
    for item in reviewed["origins_after_execution"].values():
        if "path" in item:
            bind(item)
    freeze = read(record(prior.INPUT / "freeze.json"))
    accepted = freeze["state_lineage"]["prior_residual"]
    check_review(accepted["review"])
    parent = read(accepted["receipt"])
    native.validate_parent_scope(parent)
    require(parent == accepted["parent"] and accepted["prior_layer_kv_consumed"] is False,
            "accepted L8 receipt changed")
    accepted_freeze = read(parent["arithmetic_lineage"])
    result = read(accepted["result"])
    require(result["status"] == "PASS" and result["candidate_admitted"] is True
            and result["first_failure"] is None and result["rtl_invocations"] == 0
            and result["policy_id"] == prior.gates.POLICY_ID
            and result["scope"] == accepted_freeze["contract"]["scope"]
            and result["scope"]["layers"] == list(range(3, 9)),
            "accepted L8 result scope mismatch")
    final = result["layers"][-1]
    require(final["layer"] == 8 and final["output_parent"] == parent["state"]
            and final["kv_state"] == parent["kv"]
            and final["input_parent"] == parent["state_lineage_parent"]
            and final["reports"] == parent["numerical_report"],
            "accepted L8 endpoint linkage mismatch")
    incoming = archive(parent["state_lineage_parent"])
    arrays = archive(final["actual_stages"])
    for stage in range(19):
        prior.retained.check_stage_state(stage, arrays, incoming)
    paired.same_arrays(archive(parent["kv"]),
                       {k: arrays["output_cache_" + k] for k in ("k", "v")})
    actual = archive(parent["state"])
    prior.retained.verify_parent(actual, prior.retained.state_from(arrays, "output", "stage18"))
    prior.retained.verify_parent(actual, data["layers"][9]["parent"])
    reports = read(parent["numerical_report"])
    require(len(reports) == 19 and all(r["status"] == "PASS" and r["node"] == [8, 0, s]
            for s, r in enumerate(reports)), "accepted L8 stage evidence changed")
    data["L8_review_binding"] = accepted["review"]
    data["layers"][13] = {
        "parent": data["actual"], "arrays": data["actual_arrays"], "locals": data["actual_locals"],
        "reports": data["actual_reports"], "trajectory": data["trajectory"],
        "reference": data["reference"],
    }
    with safe_open(str(bind(data["extension"]["checkpoint"])), framework="numpy") as model:
        for layer in range(9, 14):
            tensors = {k: np.ascontiguousarray(model.get_tensor(k))
                       for k in prior.local.tensor_shapes(layer)}
            prior.local.authenticate_tensors(tensors, data["extension"]["layers"][str(layer)]["canonical"], layer)
            data["layers"][layer]["tensors"] = tensors
    return data


def original_branches(data):
    extension, tensors = data["extension"], data["layers"][9]["tensors"]
    bind = lambda item: upstream.bind_input(item, data["bound"])
    specification = json.loads(bind(extension["original_specification"]).read_text())
    packages = {name: importlib.metadata.version(name) for name in ("numpy", "torch", "safetensors")}
    require(packages == extension["python_packages"] == specification["python_packages"],
            "independent oracle package mismatch")
    namespace = {
        "np": np, "torch": torch, "torch_functional": functional, "math": math,
        "AWQ_REVERSE_ORDER": (0, 4, 1, 5, 2, 6, 3, 7), "GROUP_SIZE": 128,
        "HEAD_DIM": 64, "HIDDEN_SIZE": 896, "QUERY_HEADS": 14, "KEY_VALUE_HEADS": 2,
    }
    for suffix, names in (
        ("/official_single_decoder_layer.py", ("_torch_unpack", "_torch_rmsnorm")),
        ("/official_model24_dialogue.py", ("_reference_projection", "_reference_layer_step")),
    ):
        matches = [r for r in extension["generators"] if r["path"].endswith(suffix)]
        require(len(matches) == 1 and matches[0] in specification["sources"],
                "unbound independent reference generator")
        bind(matches[0])
        paired.drift.references.definitions(matches[0], names, namespace)
    projections = {}
    for key, suffix in (("q", "self_attn.q_proj"), ("k", "self_attn.k_proj"),
                        ("v", "self_attn.v_proj"), ("o", "self_attn.o_proj"),
                        ("gate", "mlp.gate_proj"), ("up", "mlp.up_proj"), ("down", "mlp.down_proj")):
        prefix = "model.layers.9." + suffix
        q = namespace["_torch_unpack"](tensors[prefix + ".qweight"]).to(torch.float64)
        z = namespace["_torch_unpack"](tensors[prefix + ".qzeros"]).to(torch.float64)
        scales = torch.from_numpy(tensors[prefix + ".scales"].astype("<f8"))
        weight = (q - z.repeat_interleave(128, dim=0)) * scales.repeat_interleave(128, dim=0)
        bias = torch.from_numpy(tensors[prefix + ".bias"].astype("<f8")) if key in ("q", "k", "v") else None
        projections[key] = SimpleNamespace(reference_weight=weight, reference_bias=bias)
    state = SimpleNamespace(
        input_norm=tensors["model.layers.9.input_layernorm.weight"],
        post_attention_norm=tensors["model.layers.9.post_attention_layernorm.weight"],
        projections=projections, reference_k=torch.empty((0, 2, 64), dtype=torch.float64),
        reference_v=torch.empty((0, 2, 64), dtype=torch.float64))
    require(extension["layers"]["9"]["input_binary64"] == extension["original_binary64_parent"],
            "L9 reference not rooted in original-input L8")
    hidden = np.load(bind(extension["original_binary64_parent"]), allow_pickle=False)
    observed = paired.drift.capture(namespace, state, hidden)
    paired.drift.same_binary64(observed["s18"], data["layers"][9]["reference"])
    return observed


def execute_suffix(parent, data, observe):
    incoming = parent
    rows = []
    for layer in range(10, 14):
        item = data["layers"][layer]
        arrays, locals_, reports = upstream.execute_layer(
            item["tensors"], layer, incoming, item["trajectory"], item["reference"])
        rows.append(observe(layer, arrays, locals_, reports))
        incoming = prior.retained.state_from(arrays, "output", "stage18")
    return rows


def classify(rows):
    require([r["label"] for r in rows] == [label for label, _ in plan()],
            "incomplete or reordered L9 branch controls")
    by_name = {r["label"]: r for r in rows}
    require(by_name["actual"]["index62"]["accepted"] is False
            and by_name["actual"]["index62"]["actual_fp16_bits"] == "6630",
            "retained L13 baseline changed")
    require(by_name["mapped_all"]["index62"]["accepted"] is True
            and by_name["mapped_all"]["index62"]["actual_fp16_bits"] == "662f",
            "reviewed L9 full-parent endpoint changed")
    singles = [b for b in producer.BRANCHES if by_name["frozen_" + b]["index62"]["accepted"]]
    return {
        "classification": ("conditional_inherited_L8_branch_sufficiency" if singles == ["inherited"]
                           else "conditional_local_L9_branch_sufficiency"
                           if len(singles) == 1 and singles[0] in ("o", "down")
                           else "unresolved_under_tested_branch_mappings"),
        "sufficient_frozen_single_branches": singles,
        "sufficient_frozen_joint_branches": [label for label, parts in plan()[:8]
                                            if len(parts) > 1 and by_name[label]["index62"]["accepted"]],
        "inherited_native_rescue": by_name["inherited_native"]["index62"]["accepted"],
        "mapped_output_coordinate62_rescue": by_name["mapped62"]["index62"]["accepted"],
        "unique_upstream_producer_attributed": False,
        "missing_evidence": [
            "Frozen scalar cuts hold other branches at actual values. Sufficiency is conditional, "
            "not necessity, a uniquely responsible producer, or an independently realizable repair.",
            "Incoming L8 and L9-output controls use nearest-Q24 ties-even; O/down use binary64-to-"
            "FP16 RNE. Other mappings, other incoming coordinates and earlier L8 producers are not isolated.",
            "Inherited-native recomputes reachable L9 effects, while frozen O/down cuts do not "
            "recompute the L9 MLP. Exact S12/S18 addition checks do not diagnose a residual arithmetic defect.",
            "Numerically failing counterfactual suffixes are diagnostic only, never admitted lineage. "
            "Scalar rescue does not imply all gates pass, strict-FP16 state, new-token or full-model PASS.",
        ],
    }


def diagnose(out, log, contract):
    started = time.monotonic()
    torch.set_num_threads(1)
    data = authenticate()
    timing = {"authentication": time.monotonic() - started}
    artifacts, rows = [], []

    def document(name, value):
        write(out / name, value)
        item = record(out / name)
        artifacts.append(item)
        return item

    def save(name, arrays):
        item = prior.retained.save(out / name, arrays)
        artifacts.append(item)
        return item

    document("retained_authentication.json", data["authentication"])
    began = time.monotonic()
    original = original_branches(data)
    save("L9_original_branches.npz", original)
    timing["original_L9_branch_capture"] = time.monotonic() - began
    layer = data["layers"][9]
    mapped_input, mapped_output = paired.mapped_parent(original["input"]), paired.mapped_parent(original["s18"])
    document("L8_entry_coordinate62.json",
             coordinate.compare_parents(layer["parent"], mapped_input, original["input"])[62])
    began = time.monotonic()
    actual, locals_, reports = upstream.execute_layer(
        layer["tensors"], 9, layer["parent"], layer["trajectory"], layer["reference"])
    paired.same_arrays(actual, layer["arrays"])
    paired.same_arrays(locals_, layer["locals"])
    coordinate.check_reports(reports, layer["reports"])
    save("actual_L9.npz", actual)
    document("actual_L9_gates.json", reports)
    timing["native_L9_baseline_and_gates"] = time.monotonic() - began
    baseline = prior.retained.state_from(actual, "output", "stage18")
    document("L9_coordinate62_decomposition.json", {
        **paired.parts(layer["parent"], actual, original, 62),
        "S12_actual_vs_original": coordinate.compare_parents(
            prior.retained.state_from(actual, "scratch", "stage12"),
            paired.mapped_parent(original["residual"]), original["residual"])[62],
        "S18_actual_vs_original": coordinate.compare_parents(baseline, mapped_output, original["s18"])[62],
        "branch_definitions": contract["branch_definitions"],
        "original_values_hex": {k: float(v[62]).hex() for k, v in original.items()},
    })
    for label, parts in plan():
        began = time.monotonic()
        l9_statuses = None
        if label == "inherited_native":
            incoming = coordinate.intervene(layer["parent"], mapped_input, [62])
            generated, local_values, local_reports = upstream.execute_layer(
                layer["tensors"], 9, incoming, layer["trajectory"], layer["reference"])
            save(label + "_L9.npz", generated)
            save(label + "_L9_local_references.npz", local_values)
            document(label + "_L9_gates.json", local_reports)
            parent = prior.retained.state_from(generated, "output", "stage18")
            l9_statuses = [r["status"] for r in local_reports]
            kind = "recomputed_native_L9_from_complete_incoming_coordinate62"
        elif label in ("mapped62", "mapped_all"):
            parent = coordinate.intervene(baseline, mapped_output,
                                          [62] if label == "mapped62" else list(range(896)))
            kind = "exogenous_L9_output_mapping"
        else:
            incoming, o, down, scratch, parent = producer.frozen_parent(layer["parent"], actual, original, parts)
            operands = {"input_" + k: v for k, v in incoming.items()}
            operands.update({"scratch_" + k: v for k, v in scratch.items()})
            operands.update(o=o, down=down)
            save(label + "_L9_cut_operands.npz", operands)
            unchanged = np.arange(896) != 62
            require(all(np.array_equal(parent[k][unchanged], baseline[k][unchanged]) for k in parent),
                    "frozen L9 cut changed unselected coordinates")
            kind = "exogenous_frozen_scalar_branch_cut"
            if label == "actual":
                paired.same_arrays(parent, baseline)
                paired.same_arrays(scratch, prior.retained.state_from(actual, "scratch", "stage12"))
                l9_statuses = [r["status"] for r in reports]
        parent_record = save(label + "_L9_parent.npz", parent)

        def observe(suffix_layer, arrays, local_values, gate_reports):
            item = data["layers"][suffix_layer]
            if label == "actual":
                paired.same_arrays(arrays, item["arrays"])
                paired.same_arrays(local_values, item["locals"])
                coordinate.check_reports(gate_reports, item["reports"])
            elif label == "mapped_all":
                stem = coordinate.REVIEWED / f"cut09_mapped_layer{suffix_layer:02d}"
                array_record = data["bound"][str(stem.with_suffix(".npz"))]
                report_record = data["bound"][str(stem) + "_gates.json"]
                bind = lambda value: upstream.bind_input(value, data["bound"])
                paired.same_arrays(arrays, native.load_state(array_record, bind))
                coordinate.check_reports(gate_reports, json.loads(bind(report_record).read_text()))
            stem = f"{label}_L{suffix_layer}"
            vectors = save(stem + ".npz", arrays)
            gates = document(stem + "_gates.json", gate_reports)
            scalar = prior.measure(int(arrays["stage18"][62]), float(item["reference"][62]))
            failures = [r["index"] for r in gate_reports[18]["binary64_v1"]["failures"]]
            require(scalar["accepted"] == (62 not in failures), "scalar/full-vector gate disagreement")
            return {
                "layer": suffix_layer, "vectors": vectors, "gates": gates, "index62": scalar,
                "S18_failure_indices": failures, "mandatory_statuses": [r["status"] for r in gate_reports],
                "all_gates_pass": all(r["status"] == "PASS" for r in gate_reports),
                "source_operand_state_KV_RTZ_checks": "PASS",
                "coordinate62_output": coordinate.compare_parents(
                    prior.retained.state_from(arrays, "output", "stage18"),
                    paired.mapped_parent(item["reference"]), item["reference"])[62],
            }

        suffix = execute_suffix(parent, data, observe)
        if label == "actual":
            require(suffix[-1]["S18_failure_indices"] == [62], "retained L13 failure set changed")
        row = {
            "label": label, "kind": kind, "parts": list(parts), "parent": parent_record,
            "L9_operator_gate_statuses": l9_statuses,
            "L9_output_Q24_units62": int(parent["i"][62]),
            "L9_output_minus_actual_Q24_units62": int(parent["i"][62]) - int(baseline["i"][62]),
            "L9_output_FP16_bits62": f"{int(parent['h'][62]):04x}",
            "suffix": suffix, "index62": suffix[-1]["index62"],
            "all_L10_L13_gates_pass": all(r["all_gates_pass"] for r in suffix),
            "candidate_admitted": False, "seconds": time.monotonic() - began,
        }
        rows.append(row)
        log.write(json.dumps(row) + "\n")
        log.flush()
    document("interventions.json", rows)
    by_name = {r["label"]: r for r in rows}
    interactions = {}
    for label, parts in plan()[:8]:
        if len(parts) > 1:
            delta = by_name[label]["L9_output_minus_actual_Q24_units62"]
            delta -= sum(by_name["frozen_" + b]["L9_output_minus_actual_Q24_units62"] for b in parts)
            require(delta == 0, "frozen Q24 additive decomposition did not close")
            interactions[label] = delta
    document("frozen_Q24_additive_interactions.json", interactions)
    after = origins()
    validation = json.loads((out / "validation.json").read_text())
    require(after == validation["origins"] and record(CONTRACT) == validation["contract"],
            "source/module/contract changed during execution")
    for item in data["bound"].values():
        require(record(Path(item["path"])) == item, f"authenticated input changed: {item['path']}")
    check_review(data["L8_review_binding"])
    timing["native_controls_and_gates"] = sum(r["seconds"] for r in rows)
    timing["total"] = time.monotonic() - started
    return {
        "diagnostic_id": ID, "status": "DIAGNOSED", "node": [9, 0, 18], "index": 62,
        "candidate_admitted": False, "policy_adopted": False, "successor_published": False,
        "rtl_invocations": 0, "normal_host_review": "REQUIRED", "retained_status": "FAIL",
        "accepted_L0_L8_execution": False, "native_retained_bitwise_reproduction": True,
        "reviewed_cut9_mapped_suffix_bitwise_reproduction": True,
        "original_L9_S18_bitwise_reproduction": True, "unchanged_baseline_gates": True,
        "original_global_reference_unchanged": True, "source_operand_state_KV_lineage_checks": "PASS",
        "control_count": len(rows), "timing_seconds": timing, "origins_after_execution": after,
        "input_bindings": list(data["bound"].values()), "artifacts": artifacts,
        "metadata_only_external_review": data["L8_review_binding"],
        "all_L10_L13_gate_passing_controls": [r["label"] for r in rows if r["all_L10_L13_gates_pass"]],
        "claim_boundary": contract["claim_boundary"], **classify(rows),
    }


def output_path(value):
    out = value.resolve()
    require(out.parent == ROOT / "build"
            and out.name.startswith("q24_s16_l9_coordinate62_producer_cone_"),
            "output outside bounded build scope")
    require(not out.exists(), "output already exists")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    out = output_path(parser.parse_args().out)
    out.mkdir(exist_ok=False)
    with (out / "command.log").open("x") as log:
        log.write(json.dumps({
            "cwd": str(Path.cwd()), "executable": sys.executable, "argv": sys.argv,
            "PYTHONPATH": os.environ.get("PYTHONPATH"), "loadavg": os.getloadavg(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "torch_threads_before": torch.get_num_threads(), "rtl_invocations": 0,
        }) + "\n")
        log.flush()
        try:
            result = diagnose(out, log, validate(out))
            result["validation"] = record(out / "validation.json")
        except (ValueError, OSError, KeyError, TypeError, IndexError, ArithmeticError,
                RuntimeError, py_compile.PyCompileError) as exc:
            traceback.print_exc(file=log)
            result = {
                "diagnostic_id": ID, "status": "BLOCKED", "classification": "inconclusive",
                "missing_evidence": [f"{type(exc).__name__}: {exc}"], "candidate_admitted": False,
                "policy_adopted": False, "successor_published": False, "rtl_invocations": 0,
                "normal_host_review": "REQUIRED",
            }
        write(out / "result.json", result)
        summary = {k: result[k] for k in ("status", "classification", "missing_evidence")}
        summary["result"] = str(out / "result.json")
        log.write(json.dumps(summary) + "\n")
        print(json.dumps(summary))
    return 0 if result["status"] == "DIAGNOSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
