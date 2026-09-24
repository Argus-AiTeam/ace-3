"""Read-only accepted L8 producer reconstruction and non-admitting L9-L13 cuts."""

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

from ace3.model.candidates import diagnose_q24_s16_l9_coordinate62_entry_producer_cone_v1 as entry


producer, coordinate, upstream = entry.producer, entry.coordinate, entry.upstream
paired, prior, native = entry.paired, entry.prior, entry.native
ROOT = entry.ROOT
ID = "ace3-q24-s16-l8-coordinate62-producer-cone-v1"
MODULE = "ace3.model.candidates.diagnose_q24_s16_l8_coordinate62_producer_cone_v1"
TEST_MODULE = "tests.test_q24_s16_l8_coordinate62_producer_cone_v1"
CONTRACT = ROOT / "ace3/contracts/candidates/q24_s16_l8_coordinate62_producer_cone_v1.json"
REVIEWED = ROOT / "build/q24_s16_l9_coordinate62_entry_producer_cone_d0755047099e_attempt001"
REVIEWED_SHA = "569320a0f55ffdd30647decb676c06d5a83610163761ab5a1807ef648fb97d54"
require, record, write = prior.require, paired.record, paired.write


def plan():
    return producer.plan()[:8] + [
        ("scratch", ("scratch",)), ("scratch_down", ("scratch", "down")),
        ("mapped62", ("mapped62",)),
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
    require(CONTRACT.resolve().is_relative_to(ROOT), "foreign contract")
    contract = json.loads(CONTRACT.read_text())
    require(contract["diagnostic_id"] == ID and contract["policy_id"] == prior.gates.POLICY_ID
            and contract["reviewed_result_sha256"] == REVIEWED_SHA
            and contract["controls"] == [label for label, _ in plan()]
            and contract["normal_host_review"] == "REQUIRED" and contract["rtl_invocations"] == 0
            and all(contract[k] is False for k in
                    ("candidate_admitted", "policy_adopted", "successor_published")),
            "L8 diagnostic contract mismatch")
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


def authenticate():
    data = producer.authenticate()
    bind = lambda item: upstream.bind_input(item, data["bound"])
    read = lambda item: json.loads(bind(item).read_text())
    archive = lambda item: native.load_state(item, bind)
    for directory, digest, diagnostic_id in (
        (entry.REVIEWED, entry.REVIEWED_SHA, producer.ID),
        (REVIEWED, REVIEWED_SHA, entry.ID),
    ):
        item = record(directory / "result.json")
        require(item["sha256"] == digest, "wrong reviewed producer result")
        reviewed = read(item)
        require(reviewed["diagnostic_id"] == diagnostic_id and reviewed["status"] == "DIAGNOSED"
                and reviewed["native_retained_bitwise_reproduction"] is True
                and reviewed["unchanged_baseline_gates"] is True
                and reviewed["original_global_reference_unchanged"] is True
                and reviewed["source_operand_state_KV_lineage_checks"] == "PASS"
                and reviewed["rtl_invocations"] == 0
                and all(reviewed[k] is False for k in
                        ("candidate_admitted", "policy_adopted", "successor_published")),
                "reviewed producer scope mismatch")
        for item in reviewed["input_bindings"] + reviewed["artifacts"] + [reviewed["validation"]]:
            bind(item)
        for item in reviewed["origins_after_execution"].values():
            if "path" in item:
                bind(item)
    require(reviewed["classification"] == "conditional_inherited_L8_branch_sufficiency"
            and reviewed["original_L9_S18_bitwise_reproduction"] is True,
            "wrong reviewed L9 scientific parent")
    freeze = read(record(prior.INPUT / "freeze.json"))
    accepted = freeze["state_lineage"]["prior_residual"]
    require(accepted["review"] == reviewed["metadata_only_external_review"]
            and accepted["review"]["sha256"] == entry.L8_REVIEW_SHA
            and accepted["review"]["path"] == str(entry.L8_REVIEW),
            "accepted review identity differs from pinned reviewed repository evidence")
    parent = read(accepted["receipt"])
    native.validate_parent_scope(parent)
    require(parent == accepted["parent"] and accepted["prior_layer_kv_consumed"] is False,
            "accepted L8 receipt mismatch")
    accepted_freeze = read(parent["arithmetic_lineage"])
    accepted_result = read(accepted["result"])
    require(accepted_result["status"] == "PASS" and accepted_result["candidate_admitted"] is True
            and accepted_result["first_failure"] is None and accepted_result["rtl_invocations"] == 0
            and accepted_result["policy_id"] == prior.gates.POLICY_ID
            and accepted_result["scope"] == {"history": [9707], "layers": list(range(3, 9)), "position": 0}
            and accepted_freeze["global_reference_lineage"]["candidate_data_used"] is False,
            "accepted L3-L8 scope/reference mismatch")
    for source in accepted_freeze["sources"]:
        require(Path(source["original"]["path"]).resolve().is_relative_to(ROOT)
                and source["original"]["sha256"] == source["snapshot"]["sha256"],
                "accepted source origin/snapshot mismatch")
        bind(source["snapshot"])
    l7, l8 = accepted_result["layers"][-2:]
    require(l7["layer"] == 7 and l8["layer"] == 8
            and l7["output_parent"] == l8["input_parent"] == parent["state_lineage_parent"]
            and l8["output_parent"] == parent["state"]
            and l8["reports"] == parent["numerical_report"] and l8["kv_state"] == parent["kv"],
            "accepted L7/L8 endpoint linkage mismatch")
    for item in (l7, l8):
        incoming, arrays = archive(item["input_parent"]), archive(item["actual_stages"])
        for stage in range(19):
            prior.retained.check_stage_state(stage, arrays, incoming)
        output = archive(item["output_parent"])
        prior.retained.verify_parent(output, prior.retained.state_from(arrays, "output", "stage18"))
        paired.same_arrays(archive(item["kv_state"]),
                           {k: arrays["output_cache_" + k] for k in ("k", "v")})
        reports = read(item["reports"])
        require(len(reports) == 19 and all(
            r["node"] == [item["layer"], 0, s] and r["status"] == "PASS"
            and r["policy_id"] == prior.gates.POLICY_ID
            and r["residual_state_lineage"] == r["kv_lineage"] == "PASS"
            for s, r in enumerate(reports)), "accepted stage/state/KV gates changed")
        if item["layer"] == 8:
            data["L8"] = {"parent": incoming, "arrays": arrays, "output": output, "reports": reports}
    paired.same_arrays(data["L8"]["output"], data["layers"][9]["parent"])
    control = read(accepted_freeze["global_reference_lineage"]["control"])
    model_freeze = read(accepted_freeze["global_reference_lineage"]["model"])
    require(model_freeze["checkpoint"] == control["checkpoint"] == data["extension"]["checkpoint"]
            and model_freeze["reference_source"] == control["fp16_reference_root"],
            "original input/checkpoint lineage mismatch")
    read(control["binary64_reference_freeze"])
    read(control["binary64_array_binding_freeze"])
    refs = {}
    for layer in (7, 8):
        cases = [c for c in control["cases"] if c["layer"] == layer and c["position"] == 0]
        require(len(cases) == 1, "ambiguous L7/L8 original reference")
        refs[layer] = np.load(bind(cases[0]["binary64_reference_array"]), allow_pickle=False)
        paired.drift.same_binary64(refs[layer], refs[layer])
    paired.drift.same_binary64(refs[8], np.load(
        bind(data["extension"]["original_binary64_parent"]), allow_pickle=False))
    data["original7"], data["original8"] = refs[7], refs[8]
    data["layers"][13] = {
        "parent": data["actual"], "arrays": data["actual_arrays"], "locals": data["actual_locals"],
        "reports": data["actual_reports"], "trajectory": data["trajectory"], "reference": data["reference"],
    }
    with safe_open(str(bind(data["extension"]["checkpoint"])), framework="numpy") as model:
        for layer in range(8, 14):
            tensors = {k: np.ascontiguousarray(model.get_tensor(k))
                       for k in prior.local.tensor_shapes(layer)}
            canonical = ({r["name"]: r for r in model_freeze["checkpoint_tensors"]}
                         if layer == 8 else data["extension"]["layers"][str(layer)]["canonical"])
            prior.local.authenticate_tensors(tensors, canonical, layer)
            (data["L8"] if layer == 8 else data["layers"][layer])["tensors"] = tensors
    data["review_identity_basis"] = {
        "repository_result": record(REVIEWED / "result.json"),
        "external_metadata_read": False,
        "limit": "Accepted round-3 review identity is inherited from the pinned reviewed repository "
                 "result. External Host review metadata is not reopened or claimed freshly authenticated.",
    }
    return data


def original_branches(data):
    extension, tensors = data["extension"], data["L8"]["tensors"]
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
        prefix = "model.layers.8." + suffix
        q = namespace["_torch_unpack"](tensors[prefix + ".qweight"]).to(torch.float64)
        z = namespace["_torch_unpack"](tensors[prefix + ".qzeros"]).to(torch.float64)
        scales = torch.from_numpy(tensors[prefix + ".scales"].astype("<f8"))
        weight = (q - z.repeat_interleave(128, dim=0)) * scales.repeat_interleave(128, dim=0)
        bias = torch.from_numpy(tensors[prefix + ".bias"].astype("<f8")) if key in ("q", "k", "v") else None
        projections[key] = SimpleNamespace(reference_weight=weight, reference_bias=bias)
    state = SimpleNamespace(
        input_norm=tensors["model.layers.8.input_layernorm.weight"],
        post_attention_norm=tensors["model.layers.8.post_attention_layernorm.weight"],
        projections=projections, reference_k=torch.empty((0, 2, 64), dtype=torch.float64),
        reference_v=torch.empty((0, 2, 64), dtype=torch.float64))
    observed = paired.drift.capture(namespace, state, data["original7"])
    paired.drift.same_binary64(observed["s18"], data["original8"])
    return observed


def cut_parent(layer, original, label):
    controls = dict(plan())
    require(label in controls, "unknown L8 control")
    if label == "mapped62":
        return None, coordinate.intervene(layer["output"], paired.mapped_parent(original["s18"]), [62])
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
    operands = {"input_" + k: v for k, v in incoming.items()}
    operands.update({"scratch_" + k: v for k, v in scratch.items()})
    operands.update(o=o, down=down)
    if label == "actual":
        paired.same_arrays(output, layer["output"])
        paired.same_arrays(scratch, prior.retained.state_from(layer["arrays"], "scratch", "stage12"))
    return operands, output


def execute_suffix(parent, data, observe):
    incoming, rows = parent, []
    for layer in range(9, 14):
        item = data["layers"][layer]
        arrays, locals_, reports = upstream.execute_layer(
            item["tensors"], layer, incoming, item["trajectory"], item["reference"])
        rows.append(observe(layer, arrays, locals_, reports))
        incoming = prior.retained.state_from(arrays, "output", "stage18")
    return rows


def classify(rows):
    require([r["label"] for r in rows] == [label for label, _ in plan()],
            "incomplete or reordered L8 branch controls")
    by_name = {r["label"]: r for r in rows}
    require(by_name["actual"]["index62"]["accepted"] is False
            and by_name["actual"]["index62"]["actual_fp16_bits"] == "6630",
            "retained L13 baseline changed")
    singles = [b for b in producer.BRANCHES if by_name["frozen_" + b]["index62"]["accepted"]]
    return {
        "classification": ("conditional_inherited_L7_branch_sufficiency" if singles == ["inherited"]
                           else "conditional_local_L8_branch_sufficiency"
                           if len(singles) == 1 and singles[0] in ("o", "down")
                           else "unresolved_under_tested_branch_mappings"),
        "sufficient_frozen_single_branches": singles,
        "sufficient_frozen_joint_branches": [label for label, parts in plan()[:8]
                                            if len(parts) > 1 and by_name[label]["index62"]["accepted"]],
        "scratch_rescue": by_name["scratch"]["index62"]["accepted"],
        "scratch_down_rescue": by_name["scratch_down"]["index62"]["accepted"],
        "mapped_output_coordinate62_rescue": by_name["mapped62"]["index62"]["accepted"],
        "unique_upstream_producer_attributed": False,
        "missing_evidence": [
            "Frozen branch sufficiency is conditional on actual uncut coordinates/branches, not "
            "necessity or a unique producer. Nearest-Q24 input/S12/output and RNE-FP16 O/down are "
            "disclosed diagnostic mappings, not newly admitted arithmetic.",
            "Accepted native L0-L8 is not replayed. Frozen L8 O/down interventions do not recompute "
            "reachable L8 attention/MLP effects; inherited-native and other-coordinate effects are untested.",
            "Exact Q24 S12/S18 sums and signed decomposition do not establish a residual arithmetic "
            "defect. Scalar L13 rescue need not pass all unchanged suffix gates.",
            "All cuts remain non-admitting software diagnostics. No strict-FP16-state W4A16, "
            "new-token, full-model, hardware or performance-cause claim follows.",
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
    save("L8_original_branches.npz", original)
    timing["original_L8_branch_capture"] = time.monotonic() - began
    layer = data["L8"]
    document("L7_L8_coordinate62.json", {
        name: coordinate.compare_parents(actual, paired.mapped_parent(reference), reference)[62]
        for name, actual, reference in (
            ("L7_output_L8_input", layer["parent"], original["input"]),
            ("L8_S12", prior.retained.state_from(layer["arrays"], "scratch", "stage12"), original["residual"]),
            ("L8_output_L9_input", layer["output"], original["s18"]),
        )
    })
    document("L8_coordinate62_decomposition.json", {
        **paired.parts(layer["parent"], layer["arrays"], original, 62),
        "branch_definitions": contract["branch_definitions"],
        "original_values_hex": {k: float(v[62]).hex() for k, v in original.items()},
    })
    for label, parts in plan():
        began = time.monotonic()
        operands, parent = cut_parent(layer, original, label)
        unchanged = np.arange(896) != 62
        require(all(np.array_equal(parent[k][unchanged], layer["output"][k][unchanged]) for k in parent),
                "L8 intervention changed unselected coordinates")
        if operands is not None:
            save(label + "_L8_cut_operands.npz", operands)
        parent_record = save(label + "_L8_parent.npz", parent)

        def observe(suffix_layer, arrays, locals_, reports):
            item = data["layers"][suffix_layer]
            if label == "actual":
                paired.same_arrays(arrays, item["arrays"])
                paired.same_arrays(locals_, item["locals"])
                coordinate.check_reports(reports, item["reports"])
            elif label == "mapped62":
                stem = REVIEWED / f"inherited_native_L{suffix_layer}"
                bind = lambda r: upstream.bind_input(r, data["bound"])
                paired.same_arrays(arrays, native.load_state(
                    data["bound"][str(stem.with_suffix(".npz"))], bind))
                coordinate.check_reports(reports, json.loads(
                    bind(data["bound"][str(stem) + "_gates.json"]).read_text()))
            stem = f"{label}_L{suffix_layer}"
            vectors = save(stem + ".npz", arrays)
            local_record = save(stem + "_local_references.npz", locals_)
            gates = document(stem + "_gates.json", reports)
            scalar = prior.measure(int(arrays["stage18"][62]), float(item["reference"][62]))
            failures = [r["index"] for r in reports[18]["binary64_v1"]["failures"]]
            require(scalar["accepted"] == (62 not in failures), "scalar/full-vector gate disagreement")
            return {
                "layer": suffix_layer, "vectors": vectors, "local_references": local_record,
                "gates": gates, "index62": scalar, "S18_failure_indices": failures,
                "mandatory_statuses": [r["status"] for r in reports],
                "all_gates_pass": all(r["status"] == "PASS" for r in reports),
                "source_operand_state_KV_RTZ_checks": "PASS",
                "coordinate62_output": coordinate.compare_parents(
                    prior.retained.state_from(arrays, "output", "stage18"),
                    paired.mapped_parent(item["reference"]), item["reference"])[62],
            }

        suffix = execute_suffix(parent, data, observe)
        if label == "actual":
            require(suffix[-1]["S18_failure_indices"] == [62], "retained L13 failure set changed")
        rows.append({
            "label": label, "parts": list(parts), "parent": parent_record,
            "kind": "retained_actual" if label == "actual" else "exogenous_L8_branch_mapping",
            "L8_operator_gates": "retained_PASS_not_reexecuted" if label == "actual" else "not_claimed",
            "L8_output_Q24_units62": int(parent["i"][62]),
            "L8_output_minus_actual_Q24_units62": int(parent["i"][62]) - int(layer["output"]["i"][62]),
            "L8_output_FP16_bits62": f"{int(parent['h'][62]):04x}",
            "suffix": suffix, "index62": suffix[-1]["index62"],
            "all_L9_L13_gates_pass": all(r["all_gates_pass"] for r in suffix),
            "candidate_admitted": False, "seconds": time.monotonic() - began,
        })
        log.write(json.dumps(rows[-1]) + "\n")
        log.flush()
    document("interventions.json", rows)
    by_name = {r["label"]: r for r in rows}
    interactions = {}
    for label, parts in plan()[:8]:
        if len(parts) > 1:
            delta = by_name[label]["L8_output_minus_actual_Q24_units62"]
            delta -= sum(by_name["frozen_" + b]["L8_output_minus_actual_Q24_units62"] for b in parts)
            require(delta == 0, "frozen additive decomposition did not close")
            interactions[label] = delta
    document("frozen_Q24_additive_interactions.json", interactions)
    after = origins()
    validation = json.loads((out / "validation.json").read_text())
    require(after == validation["origins"] and record(CONTRACT) == validation["contract"],
            "source/module/contract changed during execution")
    for item in data["bound"].values():
        require(Path(item["path"]).resolve().is_relative_to(ROOT)
                and record(Path(item["path"])) == item, f"authenticated input changed: {item['path']}")
    timing["native_controls_and_gates"] = sum(r["seconds"] for r in rows)
    timing["total"] = time.monotonic() - started
    return {
        "diagnostic_id": ID, "status": "DIAGNOSED", "node": [8, 0, 18], "index": 62,
        "candidate_admitted": False, "policy_adopted": False, "successor_published": False,
        "rtl_invocations": 0, "normal_host_review": "REQUIRED", "retained_status": "FAIL",
        "accepted_L0_L8_native_execution": False, "native_retained_bitwise_reproduction": True,
        "reviewed_L9_inherited_native_suffix_bitwise_reproduction": True,
        "original_L8_S18_bitwise_reproduction": True, "unchanged_baseline_gates": True,
        "original_global_reference_unchanged": True, "source_operand_state_KV_lineage_checks": "PASS",
        "control_count": len(rows), "native_suffix_layer_evaluations": 5 * len(rows),
        "timing_seconds": timing, "origins_after_execution": after,
        "input_bindings": list(data["bound"].values()), "artifacts": artifacts,
        "review_identity_basis": data["review_identity_basis"],
        "all_L9_L13_gate_passing_controls": [r["label"] for r in rows if r["all_L9_L13_gates_pass"]],
        "claim_boundary": contract["claim_boundary"], **classify(rows),
    }


def output_path(value):
    out = value.resolve()
    require(out.parent == ROOT / "build"
            and out.name.startswith("q24_s16_l8_coordinate62_producer_cone_"),
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
