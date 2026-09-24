"""CPU-only L10/P0 coordinate-62 producer cuts, never an admitted repair.

Run from the isolated repository with command-local PYTHONPATH and:
/home/argustest/miniconda3/bin/python -B -m
ace3.model.candidates.diagnose_q24_s16_l10_coordinate62_producer_cone_v1
--out build/q24_s16_l10_coordinate62_producer_cone_<fresh-name>

The entry point compiles both sources and runs only its focused tests once.
Frozen cuts map incoming/S12/S18 to nearest Q24 ties-even and O/down to
FP16 RNE. Native recomputation is restricted to L10-L13; accepted earlier
layers are authenticated read-only. Original-input references never change.
"""

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

from ace3.model.candidates import diagnose_q24_s16_l9_coordinate62_entry_producer_cone_v1 as entry


producer = entry.producer
coordinate = entry.coordinate
upstream = entry.upstream
paired = entry.paired
prior = entry.prior
native = entry.native
ROOT = entry.ROOT
ID = "ace3-q24-s16-l10-coordinate62-producer-cone-v1"
MODULE = "ace3.model.candidates.diagnose_q24_s16_l10_coordinate62_producer_cone_v1"
TEST_MODULE = "tests.test_q24_s16_l10_coordinate62_producer_cone_v1"
require, record, write = entry.require, entry.record, entry.write
REVIEWED = (
    ("q24_s16_l9_coordinate62_entry_producer_cone_d0755047099e_attempt001",
     "569320a0f55ffdd30647decb676c06d5a83610163761ab5a1807ef648fb97d54",
     entry.ID),
    ("q24_s16_l8_coordinate62_producer_cone_174d3a4c0f95_attempt001",
     "8b15f1b056f92cf07c766fc4a2fc9e67ae8ba53a06be9026b0b3c899b9aecff8",
     "ace3-q24-s16-l8-coordinate62-producer-cone-v1"),
    ("q24_s16_l7_coordinate62_producer_cone_6b7120b2b23e_attempt001",
     "d049db823487bf591007ab0e2819adf8bb35a2420c86e0a196fd9fce9a4dcd3a",
     "ace3-q24-s16-l7-coordinate62-producer-cone-v1"),
)
BRANCH_DEFINITIONS = {
    "inherited": "L9 output / L10 input complete I/Z/H coordinate 62; nearest-Q24 original input",
    "o": "L10 S11 O contribution coordinate 62; original binary64 mapped FP16 RNE",
    "down": "L10 S17 down contribution coordinate 62; original binary64 mapped FP16 RNE",
    "scratch": "L10 S12 complete I/Z/H coordinate 62; nearest-Q24 original residual",
    "S12": "exact Q24 input + FP16 S11; H is the FP16 observation, not a new state seed",
    "S18": "exact Q24 S12 + FP16 S17",
    "inherited_native": "change incoming coordinate 62 then recompute native L10, including MLP",
    "mapped62": "exogenous nearest-Q24 original L10 output at coordinate 62 only",
    "mapped_all": "exogenous nearest-Q24 original L10 output at all 896 coordinates",
}


def plan():
    return producer.plan()


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
    with (out / "unittest.log").open("x") as log:
        suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
        count = suite.countTestCases()
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    write(out / "validation.json", {
        "cwd": str(ROOT), "executable": sys.executable, "PYTHONPATH": os.environ["PYTHONPATH"],
        "sys_path": sys.path, "origins": source_origins, "compiled": compiled,
        "collected": count, "executed": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped),
        "accepted_L0_L8_tests_executed": False,
    })
    require(count == tests.EXPECTED_TESTS and result.testsRun == count
            and result.wasSuccessful() and not result.skipped, "focused validation failed")


def authenticate():
    # This helper authenticates frozen ancestors and loads tensors, but dispatches no layer.
    data = entry.authenticate()
    bind = lambda item: upstream.bind_input(item, data["bound"])
    historical = []
    for directory, digest, diagnostic_id in REVIEWED:
        item = record(ROOT / "build" / directory / "result.json")
        require(item["sha256"] == digest, "wrong reviewed producer result")
        reviewed = json.loads(bind(item).read_text())
        require(reviewed["diagnostic_id"] == diagnostic_id and reviewed["status"] == "DIAGNOSED"
                and reviewed["native_retained_bitwise_reproduction"] is True
                and reviewed["unchanged_baseline_gates"] is True
                and reviewed["original_global_reference_unchanged"] is True
                and reviewed["source_operand_state_KV_lineage_checks"] == "PASS"
                and reviewed["rtl_invocations"] == 0
                and all(reviewed[k] is False for k in
                        ("candidate_admitted", "policy_adopted", "successor_published")),
                "reviewed producer scope mismatch")
        for binding in reviewed["input_bindings"] + reviewed["artifacts"] + [reviewed["validation"]]:
            bind(binding)
        for binding in reviewed["origins_after_execution"].values():
            if "path" in binding:
                bind(binding)
        historical.append({"result": item, "classification": reviewed["classification"],
                           "read_only": True, "execution_authorized_by_this_diagnostic": False})
    data["historical"] = historical
    paired.same_arrays(data["layers"][9]["output"], data["layers"][10]["parent"])
    return data


def original_branches(data):
    extension, tensors = data["extension"], data["layers"][10]["tensors"]
    bind = lambda item: upstream.bind_input(item, data["bound"])
    specification = json.loads(bind(extension["original_specification"]).read_text())
    require({name: importlib.metadata.version(name) for name in ("numpy", "torch", "safetensors")}
            == extension["python_packages"] == specification["python_packages"],
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
        prefix = "model.layers.10." + suffix
        q = namespace["_torch_unpack"](tensors[prefix + ".qweight"]).to(torch.float64)
        z = namespace["_torch_unpack"](tensors[prefix + ".qzeros"]).to(torch.float64)
        scales = torch.from_numpy(tensors[prefix + ".scales"].astype("<f8"))
        weight = (q - z.repeat_interleave(128, dim=0)) * scales.repeat_interleave(128, dim=0)
        bias = torch.from_numpy(tensors[prefix + ".bias"].astype("<f8")) if key in ("q", "k", "v") else None
        projections[key] = SimpleNamespace(reference_weight=weight, reference_bias=bias)
    state = SimpleNamespace(
        input_norm=tensors["model.layers.10.input_layernorm.weight"],
        post_attention_norm=tensors["model.layers.10.post_attention_layernorm.weight"],
        projections=projections, reference_k=torch.empty((0, 2, 64), dtype=torch.float64),
        reference_v=torch.empty((0, 2, 64), dtype=torch.float64))
    require(extension["layers"]["10"]["input_binary64"] == extension["layers"]["9"]["binary64"],
            "L10 original reference predecessor mismatch")
    hidden = np.load(bind(extension["layers"]["10"]["input_binary64"]), allow_pickle=False)
    paired.drift.same_binary64(hidden, data["layers"][9]["reference"])
    observed = paired.drift.capture(namespace, state, hidden)
    paired.drift.same_binary64(observed["s18"], data["layers"][10]["reference"])
    return observed


def execute_layer(layer, parent, data, dispatch):
    require(type(layer) is int and 10 <= layer <= 13, "native dispatch outside L10-L13")
    item = data["layers"][layer]
    dispatch.append(layer)
    return upstream.execute_layer(
        item["tensors"], layer, parent, item["trajectory"], item["reference"])


def execute_suffix(parent, data, observe, dispatch):
    rows = []
    for layer in range(11, 14):
        arrays, locals_, reports = execute_layer(layer, parent, data, dispatch)
        rows.append(observe(layer, arrays, locals_, reports))
        parent = prior.retained.state_from(arrays, "output", "stage18")
    return rows


def cut_parent(layer, original, label):
    controls = dict(plan())
    require(label in controls and label != "inherited_native", "unknown frozen L10 control")
    baseline = prior.retained.state_from(layer["arrays"], "output", "stage18")
    if label in ("mapped62", "mapped_all"):
        return {}, coordinate.intervene(
            baseline, paired.mapped_parent(original["s18"]),
            [62] if label == "mapped62" else list(range(896)))
    if label in ("scratch", "scratch_down"):
        scratch = coordinate.intervene(
            prior.retained.state_from(layer["arrays"], "scratch", "stage12"),
            paired.mapped_parent(original["residual"]), [62])
        down = layer["arrays"]["stage17"].copy()
        if label == "scratch_down":
            down[62] = native.candidate.rne(torch.from_numpy(original["s17"].copy()))[62]
        output = native.candidate.state.add(scratch, down)
        prior.retained.verify_parent(output, prior.retained.transition_reference(scratch, down))
        return {**{"scratch_" + k: v for k, v in scratch.items()}, "down": down}, output
    incoming, o, down, scratch, output = producer.frozen_parent(
        layer["parent"], layer["arrays"], original, controls[label])
    if label == "actual":
        paired.same_arrays(output, baseline)
        paired.same_arrays(scratch, prior.retained.state_from(layer["arrays"], "scratch", "stage12"))
    return {**{"input_" + k: v for k, v in incoming.items()},
            **{"scratch_" + k: v for k, v in scratch.items()}, "o": o, "down": down}, output


def classify(rows):
    require([r["label"] for r in rows] == [label for label, _ in plan()],
            "incomplete or reordered L10 branch controls")
    by_name = {r["label"]: r for r in rows}
    require(by_name["actual"]["index62"]["accepted"] is False
            and by_name["actual"]["index62"]["actual_fp16_bits"] == "6630",
            "retained L13 baseline changed")
    singles = [b for b in producer.BRANCHES if by_name["frozen_" + b]["index62"]["accepted"]]
    return {
        "classification": ("conditional_inherited_L9_branch_sufficiency" if singles == ["inherited"]
                           else "conditional_local_L10_branch_sufficiency"
                           if len(singles) == 1 and singles[0] in ("o", "down")
                           else "unresolved_under_tested_branch_mappings"),
        "sufficient_frozen_single_branches": singles,
        "sufficient_frozen_joint_branches": [label for label, parts in plan()[:8]
                                            if len(parts) > 1 and by_name[label]["index62"]["accepted"]],
        "inherited_native_rescue": by_name["inherited_native"]["index62"]["accepted"],
        "scratch_rescue": by_name["scratch"]["index62"]["accepted"],
        "mapped_output_coordinate62_rescue": by_name["mapped62"]["index62"]["accepted"],
        "mapped_all_rescue": by_name["mapped_all"]["index62"]["accepted"],
        "unique_upstream_producer_attributed": False,
        "missing_evidence": [
            "Frozen scalar branch sufficiency is conditional on the actual background, not necessity, "
            "a unique producer, or an independently realizable operator repair.",
            "Incoming/S12/S18 use nearest-Q24 ties-even and O/down use FP16 RNE. Other mappings and "
            "other inherited coordinates are not isolated. No native L0-L8 layer is executed.",
            "Inherited-native recomputes L10 downstream effects; frozen O/S12/down cuts do not. "
            "Exact residual closure is checked, not evidence of a residual arithmetic defect.",
            "Scalar rescue does not certify all gates or lineage admission. Q24 residual state is "
            "wider than FP16; no strict-FP16 W4A16, new-token, full-model or hardware PASS follows.",
            "Historical L7/L8 results are authenticated read-only, not retroactively authorized; "
            "independent Host Reviewer reproduction remains required.",
        ],
    }


def diagnose(out, log):
    started = time.monotonic()
    torch.set_num_threads(1)
    data = authenticate()
    timing = {"authentication": time.monotonic() - started}
    artifacts, rows, dispatch = [], [], []

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
    document("historical_read_only_inputs.json", data["historical"])
    began = time.monotonic()
    original = original_branches(data)
    save("L10_original_branches.npz", original)
    timing["original_L10_branch_capture"] = time.monotonic() - began
    layer = data["layers"][10]
    document("L9_L10_entry_coordinate62.json", coordinate.compare_parents(
        layer["parent"], paired.mapped_parent(original["input"]), original["input"])[62])
    began = time.monotonic()
    actual, locals_, reports = execute_layer(10, layer["parent"], data, dispatch)
    paired.same_arrays(actual, layer["arrays"])
    paired.same_arrays(locals_, layer["locals"])
    coordinate.check_reports(reports, layer["reports"])
    save("actual_L10.npz", actual)
    save("actual_L10_local_references.npz", locals_)
    document("actual_L10_gates.json", reports)
    timing["native_L10_baseline_and_gates"] = time.monotonic() - began
    baseline = prior.retained.state_from(actual, "output", "stage18")
    document("L10_coordinate62_decomposition.json", {
        **paired.parts(layer["parent"], actual, original, 62),
        "S12_actual_vs_original": coordinate.compare_parents(
            prior.retained.state_from(actual, "scratch", "stage12"),
            paired.mapped_parent(original["residual"]), original["residual"])[62],
        "S18_actual_vs_original": coordinate.compare_parents(
            baseline, paired.mapped_parent(original["s18"]), original["s18"])[62],
        "branch_definitions": BRANCH_DEFINITIONS,
        "original_values_hex": {k: float(v[62]).hex() for k, v in original.items()},
    })
    for label, parts in plan():
        began = time.monotonic()
        l10_statuses = None
        if label == "inherited_native":
            incoming = coordinate.intervene(
                layer["parent"], paired.mapped_parent(original["input"]), [62])
            generated, local_values, local_reports = execute_layer(10, incoming, data, dispatch)
            save(label + "_L10.npz", generated)
            save(label + "_L10_local_references.npz", local_values)
            document(label + "_L10_gates.json", local_reports)
            parent = prior.retained.state_from(generated, "output", "stage18")
            l10_statuses = [r["status"] for r in local_reports]
            kind = "recomputed_native_L10_from_complete_incoming_coordinate62"
        else:
            operands, parent = cut_parent(layer, original, label)
            if operands:
                save(label + "_L10_cut_operands.npz", operands)
            if label != "mapped_all":
                unchanged = np.arange(896) != 62
                require(all(np.array_equal(parent[k][unchanged], baseline[k][unchanged]) for k in parent),
                        "scalar L10 cut changed unselected coordinates")
            if label == "actual":
                l10_statuses = [r["status"] for r in reports]
            kind = "exogenous_L10_output_mapping" if label.startswith("mapped") else "frozen_scalar_branch_cut"
        parent_record = save(label + "_L10_parent.npz", parent)

        def observe(suffix_layer, arrays, local_values, gate_reports):
            item = data["layers"][suffix_layer]
            if label == "actual":
                paired.same_arrays(arrays, item["arrays"])
                paired.same_arrays(local_values, item["locals"])
                coordinate.check_reports(gate_reports, item["reports"])
            stem = f"{label}_L{suffix_layer}"
            vectors = save(stem + ".npz", arrays)
            local_record = save(stem + "_local_references.npz", local_values)
            gates = document(stem + "_gates.json", gate_reports)
            scalar = prior.measure(int(arrays["stage18"][62]), float(item["reference"][62]))
            failures = [r["index"] for r in gate_reports[18]["binary64_v1"]["failures"]]
            require(scalar["accepted"] == (62 not in failures), "scalar/full-vector gate disagreement")
            return {
                "layer": suffix_layer, "vectors": vectors, "local_references": local_record,
                "gates": gates, "index62": scalar, "S18_failure_indices": failures,
                "mandatory_statuses": [r["status"] for r in gate_reports],
                "all_gates_pass": all(r["status"] == "PASS" for r in gate_reports),
                "source_operand_state_KV_RTZ_checks": "PASS",
                "coordinate62_output": coordinate.compare_parents(
                    prior.retained.state_from(arrays, "output", "stage18"),
                    paired.mapped_parent(item["reference"]), item["reference"])[62],
            }

        suffix = execute_suffix(parent, data, observe, dispatch)
        if label == "actual":
            require(suffix[-1]["S18_failure_indices"] == [62], "retained L13 failure set changed")
        row = {
            "label": label, "kind": kind, "parts": list(parts), "parent": parent_record,
            "L10_operator_gate_statuses": l10_statuses,
            "L10_output_Q24_units62": int(parent["i"][62]),
            "L10_output_minus_actual_Q24_units62": int(parent["i"][62]) - int(baseline["i"][62]),
            "L10_output_FP16_bits62": f"{int(parent['h'][62]):04x}",
            "suffix": suffix, "index62": suffix[-1]["index62"],
            "all_L11_L13_gates_pass": all(r["all_gates_pass"] for r in suffix),
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
            delta = by_name[label]["L10_output_minus_actual_Q24_units62"]
            delta -= sum(by_name["frozen_" + b]["L10_output_minus_actual_Q24_units62"] for b in parts)
            require(delta == 0, "frozen Q24 additive decomposition did not close")
            interactions[label] = delta
    document("frozen_Q24_additive_interactions.json", interactions)
    expected = [10]
    for label, _ in plan():
        if label == "inherited_native":
            expected.append(10)
        expected.extend((11, 12, 13))
    require(dispatch == expected and not any(layer <= 8 for layer in dispatch),
            "native execution scope mismatch")
    document("execution_scope.json", {
        "guard": "execute_layer rejects every layer outside L10-L13 before delegate invocation",
        "native_layer_dispatches": dispatch, "native_L0_L8_invocations": 0,
        "accepted_L0_L8_evidence_read_only": True, "rtl_invocations": 0,
        "successor_published": False, "original_reference_layer_executions": [10],
    })
    after = origins()
    require(after == json.loads((out / "validation.json").read_text())["origins"],
            "source/module changed during execution")
    for item in data["bound"].values():
        require(record(Path(item["path"])) == item, f"authenticated input changed: {item['path']}")
    entry.check_review(data["L8_review_binding"])
    timing["native_controls_and_gates"] = sum(r["seconds"] for r in rows)
    timing["total"] = time.monotonic() - started
    return {
        "diagnostic_id": ID, "status": "DIAGNOSED", "node": [10, 0, 18], "index": 62,
        "candidate_admitted": False, "policy_adopted": False, "successor_published": False,
        "rtl_invocations": 0, "normal_host_review": "REQUIRED", "retained_status": "FAIL",
        "accepted_L0_L8_execution": False, "native_L0_L8_invocations": 0,
        "native_layer_dispatches": dispatch, "native_retained_bitwise_reproduction": True,
        "original_L10_S18_bitwise_reproduction": True, "unchanged_baseline_gates": True,
        "original_global_reference_unchanged": True, "source_operand_state_KV_lineage_checks": "PASS",
        "policy_id": prior.gates.POLICY_ID, "control_count": len(rows),
        "timing_seconds": timing, "origins_after_execution": after,
        "input_bindings": list(data["bound"].values()), "artifacts": artifacts,
        "metadata_only_external_review": data["L8_review_binding"],
        "all_L11_L13_gate_passing_controls": [r["label"] for r in rows if r["all_L11_L13_gates_pass"]],
        "admission_limit": "All interventions remain non-admitting diagnostic state; suffix-only "
                           "gate success does not certify the exogenous L10 branch edits.",
        **classify(rows),
    }


def output_path(value):
    out = value.resolve()
    require(out.parent == ROOT / "build"
            and out.name.startswith("q24_s16_l10_coordinate62_producer_cone_"),
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
            validate(out)
            result = diagnose(out, log)
            result["validation"] = record(out / "validation.json")
        except (ValueError, OSError, KeyError, TypeError, IndexError, ArithmeticError,
                RuntimeError, py_compile.PyCompileError) as exc:
            traceback.print_exc(file=log)
            result = {
                "diagnostic_id": ID, "status": "BLOCKED", "classification": "inconclusive",
                "missing_evidence": [f"{type(exc).__name__}: {exc}"],
                "candidate_admitted": False, "policy_adopted": False,
                "successor_published": False, "rtl_invocations": 0, "normal_host_review": "REQUIRED",
            }
        write(out / "result.json", result)
        summary = {k: result[k] for k in ("status", "classification", "missing_evidence")}
        summary["result"] = str(out / "result.json")
        log.write(json.dumps(summary) + "\n")
        print(json.dumps(summary))
    return 0 if result["status"] == "DIAGNOSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
