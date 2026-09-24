"""Non-admitting L11/P0 coordinate-62 CPU producer cuts; see the contract.

--check --out validates only, with no native dispatch. Omitting --check is
reserved for a separately selected diagnostic execution, never preflight.
"""

import argparse
from contextlib import contextmanager
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import py_compile
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as functional

from ace3.model.candidates import diagnose_q24_s16_l10_coordinate62_producer_cone_v2 as selected


legacy = selected.legacy
entry = selected.entry
producer, coordinate, upstream = entry.producer, entry.coordinate, entry.upstream
paired, prior, native = entry.paired, entry.prior, entry.native
require, record, write = entry.require, entry.record, entry.write
ROOT = Path("/home/argustest/ace3-argus")
ID = "ace3-q24-s16-l11-coordinate62-producer-cone-v1"
MODULE = "ace3.model.candidates.diagnose_q24_s16_l11_coordinate62_producer_cone_v1"
TEST_MODULE = "tests.test_q24_s16_l11_coordinate62_producer_cone_v1"
ANCESTOR_TEST = "tests.test_q24_s16_toward_zero_l3_l8_v1"
CONTRACT = ROOT / "ace3/contracts/candidates/q24_s16_l11_coordinate62_producer_cone_v1.json"
OUTPUT_PREFIX = "q24_s16_l11_coordinate62_producer_cone_"
NATIVE_LAYERS = (11, 12, 13)
EXPECTED_TESTS = 20
plan = producer.plan
cut_parent = legacy.cut_parent


def authenticate():
    data = selected.authenticate()
    paired.same_arrays(data["layers"][10]["output"], data["layers"][11]["parent"])
    extension = data["extension"]["layers"]
    for layer in NATIVE_LAYERS:
        require(extension[str(layer)]["input_binary64"]
                == extension[str(layer - 1)]["binary64"],
                f"L{layer} original reference predecessor mismatch")
    return data


def original_branches(data):
    extension, tensors = data["extension"], data["layers"][11]["tensors"]
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
        prefix = "model.layers.11." + suffix
        q = namespace["_torch_unpack"](tensors[prefix + ".qweight"]).to(torch.float64)
        z = namespace["_torch_unpack"](tensors[prefix + ".qzeros"]).to(torch.float64)
        scales = torch.from_numpy(tensors[prefix + ".scales"].astype("<f8"))
        weight = (q - z.repeat_interleave(128, dim=0)) * scales.repeat_interleave(128, dim=0)
        bias = torch.from_numpy(tensors[prefix + ".bias"].astype("<f8")) if key in ("q", "k", "v") else None
        projections[key] = SimpleNamespace(reference_weight=weight, reference_bias=bias)
    state = SimpleNamespace(
        input_norm=tensors["model.layers.11.input_layernorm.weight"],
        post_attention_norm=tensors["model.layers.11.post_attention_layernorm.weight"],
        projections=projections, reference_k=torch.empty((0, 2, 64), dtype=torch.float64),
        reference_v=torch.empty((0, 2, 64), dtype=torch.float64))
    require(extension["layers"]["11"]["input_binary64"] == extension["layers"]["10"]["binary64"],
            "L11 original reference predecessor mismatch")
    hidden = np.load(bind(extension["layers"]["11"]["input_binary64"]), allow_pickle=False)
    paired.drift.same_binary64(hidden, data["layers"][10]["reference"])
    observed = paired.drift.capture(namespace, state, hidden)
    paired.drift.same_binary64(observed["s18"], data["layers"][11]["reference"])
    return observed


def check_layer(layer):
    require(type(layer) is int and layer in NATIVE_LAYERS,
            "native execution outside L11-L13; L0-L8 forbidden")


@contextmanager
def guard_stages(dispatch):
    stages = native.stages

    def guarded(tensors, layer, parent, arrays):
        check_layer(layer)
        dispatch.append(layer)
        return stages(tensors, layer, parent, arrays)

    with patch.object(native, "stages", side_effect=guarded):
        yield


def execute_layer(layer, parent, data, dispatch):
    check_layer(layer)
    item = data["layers"][layer]
    dispatch.append(layer)
    return upstream.execute_layer(
        item["tensors"], layer, parent, item["trajectory"], item["reference"])


def execute_suffix(parent, data, observe, dispatch):
    rows = []
    for layer in (12, 13):
        arrays, locals_, reports = execute_layer(layer, parent, data, dispatch)
        rows.append(observe(layer, arrays, locals_, reports))
        parent = prior.retained.state_from(arrays, "output", "stage18")
    return rows


def expected_dispatches():
    dispatch = [11]
    for label, _ in plan():
        if label == "inherited_native":
            dispatch.append(11)
        dispatch.extend((12, 13))
    return dispatch


def classify(rows):
    require([r["label"] for r in rows] == [label for label, _ in plan()],
            "incomplete or reordered L11 branch controls")
    by_name = {r["label"]: r for r in rows}
    require(by_name["actual"]["index62"]["accepted"] is False
            and by_name["actual"]["index62"]["actual_fp16_bits"] == "6630",
            "retained L13 baseline changed")
    singles = [b for b in producer.BRANCHES if by_name["frozen_" + b]["index62"]["accepted"]]
    return {
        "classification": ("conditional_single_branch_sufficiency" if len(singles) == 1
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
            "Frozen scalar cuts on the actual background show conditional sufficiency, not "
            "necessity, a unique producer, or an independently realizable operator repair.",
            "Only inherited-native recomputes L11 nonlinear effects. Other input coordinates "
            "and mappings are not isolated. Accepted L0-L8 are read-only, never executed.",
            "Scalar rescue is separate from full suffix gates and lineage admission. Wide Q24 "
            "state is not strict-FP16-state W4A16; no new-token, full-model, hardware, "
            "performance-bottleneck, policy adoption or successor claim follows.",
        ],
    }


def source_context():
    require(entry.ROOT == ROOT and Path.cwd() == ROOT and Path(sys.executable) == prior.PYTHON
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "use the published repository-bound command")
    importlib.import_module(TEST_MODULE)
    importlib.import_module(ANCESTOR_TEST)
    importlib.import_module(legacy.TEST_MODULE)
    origins = legacy.origins()
    for name in (MODULE, TEST_MODULE, ANCESTOR_TEST):
        require(name in origins
                and origins[name]["path"] == str(ROOT.joinpath(*name.split(".")).with_suffix(".py")),
                f"missing or foreign source origin: {name}")
    return origins


def check_contract():
    contract = json.loads(CONTRACT.read_text())
    require(contract["diagnostic_id"] == ID and contract["policy_id"] == prior.gates.POLICY_ID
            and contract["selected_L9"] == list(selected.SELECTED)
            and contract["historical_L10"] == list(selected.HISTORICAL)
            and contract["controls"] == [label for label, _ in plan()]
            and contract["native_layers"] == list(NATIVE_LAYERS)
            and contract["native_layer_invocations"] == len(expected_dispatches())
            and contract["position"] == 0 and contract["coordinate"] == 62
            and contract["focused_tests"] == EXPECTED_TESTS
            and contract["preflight_native_invocations"] == contract["rtl_invocations"] == 0
            and contract["normal_host_review"] == "REQUIRED"
            and all(contract[key] is False for key in (
                "candidate_admitted", "policy_adopted", "successor_published",
                "accepted_L0_L8_execution", "scientific_result_claim")),
            "L11 producer contract mismatch")
    return contract


def validate(out):
    origins = source_context()
    check_contract()
    compiled = []
    for index, name in enumerate((MODULE, TEST_MODULE)):
        path = Path(origins[name]["path"])
        py_compile.compile(str(path), cfile=str(out / f"compiled_{index}.pyc"), doraise=True)
        compiled.append(record(path))
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[TEST_MODULE])
    count = suite.countTestCases()
    with (out / "unittest.log").open("x") as log, \
            patch.object(upstream, "execute_layer",
                         side_effect=AssertionError("native dispatch forbidden in preflight")) as execute, \
            patch.object(native, "stages",
                         side_effect=AssertionError("native stages forbidden in preflight")) as stages:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
        require(not execute.called and not stages.called, "native preflight dispatch attempted")
    evidence = {
        "diagnostic_id": ID, "cwd": str(ROOT), "executable": sys.executable,
        "PYTHONPATH": os.environ["PYTHONPATH"], "sys_path": sys.path,
        "origins": origins, "compiled": compiled, "contract": record(CONTRACT),
        "collected": count, "executed": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped),
        "native_layer_invocations": 0, "rtl_invocations": 0,
        "accepted_L0_L8_tests_executed": False, "scientific_result_claim": False,
        "candidate_admitted": False, "policy_adopted": False, "successor_published": False,
        "normal_host_review": "REQUIRED",
        "authentication_scope": "Read-only selected-parent source/operand/state/KV/lineage authentication",
        "selected_L9": record(ROOT / "build" / selected.SELECTED[0] / "result.json"),
        "historical_L10": record(ROOT / "build" / selected.HISTORICAL[0] / "result.json"),
    }
    write(out / "validation.json", evidence)
    require(count == EXPECTED_TESTS and result.testsRun == count
            and result.wasSuccessful() and not result.skipped, "focused validation failed")
    require(source_context() == origins, "source changed during validation")
    return evidence


def diagnose(out, log, validation):
    require(source_context() == validation["origins"] and record(CONTRACT) == validation["contract"],
            "source/contract changed before diagnostic")
    started = time.monotonic()
    torch.set_num_threads(1)
    data = authenticate()
    timing = {"authentication": time.monotonic() - started}
    artifacts, rows, dispatch, stages = [], [], [], []

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
    save("L11_original_branches.npz", original)
    timing["original_L11_branch_capture"] = time.monotonic() - began
    layer = data["layers"][11]
    with guard_stages(stages):
        began = time.monotonic()
        actual, locals_, reports = execute_layer(11, layer["parent"], data, dispatch)
        paired.same_arrays(actual, layer["arrays"])
        paired.same_arrays(locals_, layer["locals"])
        coordinate.check_reports(reports, layer["reports"])
        save("actual_L11.npz", actual)
        save("actual_L11_local_references.npz", locals_)
        document("actual_L11_gates.json", reports)
        timing["native_L11_baseline_and_gates"] = time.monotonic() - began
        baseline = prior.retained.state_from(actual, "output", "stage18")
        document("L11_coordinate62_decomposition.json", {
            **paired.parts(layer["parent"], actual, original, 62),
            "S12_actual_vs_original": coordinate.compare_parents(
                prior.retained.state_from(actual, "scratch", "stage12"),
                paired.mapped_parent(original["residual"]), original["residual"])[62],
            "S18_actual_vs_original": coordinate.compare_parents(
                baseline, paired.mapped_parent(original["s18"]), original["s18"])[62],
            "original_values_hex": {k: float(v[62]).hex() for k, v in original.items()},
        })
        for label, parts in plan():
            began = time.monotonic()
            statuses = None
            if label == "inherited_native":
                incoming = coordinate.intervene(
                    layer["parent"], paired.mapped_parent(original["input"]), [62])
                save(label + "_L11_input.npz", incoming)
                generated, local_values, local_reports = execute_layer(11, incoming, data, dispatch)
                save(label + "_L11.npz", generated)
                save(label + "_L11_local_references.npz", local_values)
                document(label + "_L11_gates.json", local_reports)
                parent = prior.retained.state_from(generated, "output", "stage18")
                statuses = [r["status"] for r in local_reports]
            else:
                operands, parent = cut_parent(layer, original, label)
                if operands:
                    save(label + "_L11_cut_operands.npz", operands)
                if label != "mapped_all":
                    unchanged = np.arange(896) != 62
                    require(all(np.array_equal(parent[k][unchanged], baseline[k][unchanged])
                                for k in parent), "scalar L11 cut changed unselected coordinates")
                if label == "actual":
                    statuses = [r["status"] for r in reports]
            parent_record = save(label + "_L11_parent.npz", parent)

            def observe(suffix_layer, arrays, local_values, gate_reports):
                item = data["layers"][suffix_layer]
                if label == "actual":
                    paired.same_arrays(arrays, item["arrays"])
                    paired.same_arrays(local_values, item["locals"])
                    coordinate.check_reports(gate_reports, item["reports"])
                stem = f"{label}_L{suffix_layer}"
                scalar = prior.measure(int(arrays["stage18"][62]), float(item["reference"][62]))
                failures = [r["index"] for r in gate_reports[18]["binary64_v1"]["failures"]]
                require(scalar["accepted"] == (62 not in failures), "scalar/full-vector disagreement")
                return {
                    "layer": suffix_layer, "vectors": save(stem + ".npz", arrays),
                    "local_references": save(stem + "_local_references.npz", local_values),
                    "gates": document(stem + "_gates.json", gate_reports),
                    "index62": scalar, "S18_failure_indices": failures,
                    "mandatory_statuses": [r["status"] for r in gate_reports],
                    "all_gates_pass": all(r["status"] == "PASS" for r in gate_reports),
                    "source_operand_state_KV_RTZ_checks": "PASS",
                }

            suffix = execute_suffix(parent, data, observe, dispatch)
            if label == "actual":
                require(suffix[-1]["S18_failure_indices"] == [62], "retained L13 failure set changed")
            row = {
                "label": label, "parts": list(parts), "parent": parent_record,
                "L11_operator_gate_statuses": statuses,
                "L11_output_minus_actual_Q24_units62": int(parent["i"][62]) - int(baseline["i"][62]),
                "suffix": suffix, "index62": suffix[-1]["index62"],
                "all_L12_L13_gates_pass": all(r["all_gates_pass"] for r in suffix),
                "candidate_admitted": False, "seconds": time.monotonic() - began,
            }
            rows.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
    require(dispatch == stages == expected_dispatches(), "native execution scope mismatch")
    document("interventions.json", rows)
    by_name = {r["label"]: r for r in rows}
    interactions = {}
    for label, parts in plan()[:8]:
        if len(parts) > 1:
            delta = by_name[label]["L11_output_minus_actual_Q24_units62"]
            delta -= sum(by_name["frozen_" + b]["L11_output_minus_actual_Q24_units62"] for b in parts)
            require(delta == 0, "frozen Q24 additive decomposition did not close")
            interactions[label] = delta
    document("frozen_Q24_additive_interactions.json", interactions)
    require(source_context() == validation["origins"] and record(CONTRACT) == validation["contract"],
            "source/contract changed during diagnostic")
    for item in data["bound"].values():
        require(record(Path(item["path"])) == item, f"authenticated input changed: {item['path']}")
    entry.check_review(data["L8_review_binding"])
    timing["native_controls_and_gates"] = sum(r["seconds"] for r in rows)
    timing["total"] = time.monotonic() - started
    return {
        "diagnostic_id": ID, "status": "DIAGNOSED", "node": [11, 0, 18], "index": 62,
        "candidate_admitted": False, "policy_adopted": False, "successor_published": False,
        "rtl_invocations": 0, "normal_host_review": "REQUIRED", "retained_status": "FAIL",
        "accepted_L0_L8_execution": False, "native_L0_L8_invocations": 0,
        "native_layer_dispatches": dispatch, "native_stage_dispatches": stages,
        "native_retained_bitwise_reproduction": True, "original_L11_S18_bitwise_reproduction": True,
        "unchanged_baseline_gates": True, "original_global_reference_unchanged": True,
        "source_operand_state_KV_lineage_checks": "PASS", "policy_id": prior.gates.POLICY_ID,
        "selected_L9_evidence": data["selected_L9_evidence"], "control_count": len(rows),
        "timing_seconds": timing, "origins_after_execution": validation["origins"],
        "input_bindings": list(data["bound"].values()), "artifacts": artifacts,
        "metadata_only_external_review": data["L8_review_binding"],
        "contract": validation["contract"], **classify(rows),
    }


def output_path(value):
    require(not value.is_symlink(), "output must not be a symlink")
    out = value.resolve()
    require(out.parent == ROOT / "build" and out.name.startswith(OUTPUT_PREFIX)
            and len(out.name) > len(OUTPUT_PREFIX), "output outside bounded build scope")
    require(not out.exists(), "output already exists")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = output_path(args.out)
    out.mkdir(exist_ok=False)
    validation = validate(out)
    if args.check:
        print(json.dumps({"status": "VALIDATED_SOFTWARE_ONLY", "validation": str(out / "validation.json")}))
        return 0
    with (out / "command.log").open("x") as log:
        log.write(json.dumps({"cwd": str(Path.cwd()), "executable": sys.executable,
                              "argv": sys.argv, "PYTHONPATH": os.environ["PYTHONPATH"],
                              "loadavg": os.getloadavg(),
                              "cpu_affinity": sorted(os.sched_getaffinity(0))}) + "\n")
        log.flush()
        result = diagnose(out, log, validation)
        result["validation"] = record(out / "validation.json")
        write(out / "result.json", result)
    print(json.dumps({"status": result["status"], "result": str(out / "result.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
