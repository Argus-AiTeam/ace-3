"""Non-admitting L9-L12 parent cuts with native CPU suffixes through L13/P0.

The adjacent contract publishes the repo-bound command and exact control scope.
The command compiles this module and its tests, runs the focused suite once,
then writes an exclusive evidence directory. No accepted L0-L8 replay occurs.
"""

import argparse
from fractions import Fraction
import importlib
import json
import os
from pathlib import Path
import py_compile
import sys
import time
import unittest

import numpy as np
import torch
from safetensors import safe_open

from ace3.model.candidates import diagnose_q24_s16_l13_native_paired_intervention_v1 as paired


prior = paired.prior
native = paired.native
ROOT = prior.ROOT
ID = "ace3-q24-s16-l13-upstream-parent-cut-v1"
MODULE = "ace3.model.candidates.diagnose_q24_s16_l13_upstream_parent_cut_v1"
TEST_MODULE = "tests.test_q24_s16_l13_upstream_parent_cut_v1"
CONTRACT = ROOT / "ace3/contracts/candidates/q24_s16_l13_upstream_parent_cut_v1.json"
CUTS = (9, 10, 11, 12)
REVIEWED = {
    "paired": ROOT / "build/q24_s16_l13_native_paired_intervention_6a99e9155b0e_attempt001",
    "drift": paired.REVIEWED,
}
require = prior.require
record = paired.record
write = paired.write


def bind_input(item, bound):
    signature = {key: item[key] for key in ("path", "bytes", "sha256")}
    path = Path(signature["path"])
    require(path.is_absolute() and path.resolve().is_relative_to(ROOT),
            f"input/source outside isolated repository: {path}")
    if str(path) in bound:
        require(bound[str(path)] == signature, f"conflicting binding: {path}")
    else:
        require(record(path) == signature, f"binding mismatch: {path}")
        bound[str(path)] = signature
    return path


def origins():
    result = {}
    for name, module in sorted(list(sys.modules.items())):
        if name not in ("ace3", "tests") and not name.startswith(("ace3.", "tests.")):
            continue
        path = getattr(module, "__file__", None)
        expected = ROOT.joinpath(*name.split("."))
        if path is None:
            locations = list(getattr(module, "__path__", ()))
            require(locations and all(Path(p).resolve() == expected for p in locations),
                    f"namespace origin mismatch: {name}")
            result[name] = {"namespace_paths": locations}
        else:
            path = Path(path).resolve()
            require(path in (expected.with_suffix(".py"), expected / "__init__.py"),
                    f"module origin mismatch: {name}")
            result[name] = record(path)
    require(MODULE in result and TEST_MODULE in result, "missing candidate/test source origins")
    return result


def classify(cuts):
    require([row["cut_after_layer"] for row in cuts] == list(CUTS),
            "missing or unordered full-parent cuts")
    for row in cuts:
        require(row["actual_index62"]["actual_fp16_bits"] == "6630"
                and row["actual_index62"]["accepted"] is False,
                "retained index-62 baseline changed")
    flips = [row["cut_after_layer"] for row in cuts
             if row["mapped_index62"]["actual_fp16_bits"] == "662f"
             and row["mapped_index62"]["accepted"] is True]
    return {
        "classification": ("conditional_full_parent_cut_sensitivity" if flips
                           else "no_6630_to_662f_flip_in_tested_cuts"),
        "cut_order": list(CUTS),
        "flipping_cuts_after_layer": flips,
        "first_flipping_cut_after_layer": flips[0] if flips else None,
        "unique_upstream_producer_attributed": False,
        "missing_evidence": [
            "Each cut replaces all 896 parent coordinates, combining inherited drift "
            "and the cut layer's effects. No paired individual upstream producer/stage "
            "interventions isolate those contributions or their nonlinear interactions.",
            "The first flip is first only in the declared ascending L9-L12 cut order. "
            "No cut before L9 is tested and no accepted L0-L8 computation is replayed.",
            "The binary64 parent is not exact Q24 state. Conclusions are conditional "
            "on nearest-Q24 ties-even mapping, not all mappings or an admitted repair.",
        ],
    }


def execute_layer(tensors, layer, parent, trajectory, binary64):
    require(type(layer) is int and 9 <= layer <= 13, "diagnostic layer outside L9-L13/P0")
    prior.retained.verify_parent(parent, parent)
    arrays, locals_, reports = {}, {}, []
    stages = []
    for stage in native.stages(tensors, layer, parent, arrays):
        require(stage == len(stages), "out-of-order native stage")
        stages.append(stage)
        expected = prior.retained.check_stage_state(stage, arrays, parent)
        if stage < 18 and stage != 12:
            expected = prior.local.local_reference(
                stage, {key: arrays[key].copy() for key in prior.local.OPERANDS[stage]},
                tensors, layer)
        if stage < 18:
            locals_[f"stage{stage:02d}"] = expected
        report = prior.gates.evaluate_decoder_stage(
            stage=stage, actual=arrays[f"stage{stage:02d}"],
            reference=trajectory[f"stage{stage:02d}"], policy=prior.gates.POLICY_ID,
            local_reference=expected, reference_binary64=binary64 if stage == 18 else None)
        require(report["status"] in ("PASS", "FAIL"), "missing mandatory gate evidence")
        report.update(node=[layer, 0, stage], residual_state_lineage="PASS", kv_lineage="PASS")
        reports.append(report)
    require(stages == list(range(19)), "incomplete native suffix layer")
    require(np.array_equal(prior.rtz_reference(arrays["s16_unrounded_binary64"]),
                           arrays["stage16"]), "S16 RTZ operand check failed")
    return arrays, locals_, reports


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
    require(contract["diagnostic_id"] == ID and contract["cuts_after_layer"] == list(CUTS)
            and contract["policy_id"] == prior.gates.POLICY_ID
            and contract["candidate_admitted"] is False and contract["policy_adopted"] is False
            and contract["successor_published"] is False and contract["rtl_invocations"] == 0
            and contract["normal_host_review"] == "REQUIRED", "diagnostic contract mismatch")
    with (out / "unittest.log").open("x") as log:
        suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
        count = suite.countTestCases()
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    report = {
        "cwd": str(ROOT), "executable": sys.executable, "PYTHONPATH": os.environ["PYTHONPATH"],
        "sys_path": sys.path, "origins": source_origins, "compiled": compiled,
        "contract": record(CONTRACT), "collected": count, "executed": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors),
        "skipped": len(result.skipped), "accepted_L0_L8_tests_executed": False,
    }
    write(out / "validation.json", report)
    require(count == tests.EXPECTED_TESTS and result.testsRun == count
            and result.wasSuccessful() and not result.skipped, "focused validation failed")
    return contract


def diagnose(out, log, contract):
    started = time.monotonic()
    torch.set_num_threads(1)
    authentication = prior.diagnose(prior.INPUT)
    bound = {item["path"]: item for item in authentication["input_bindings"]}
    bind = lambda item: bind_input(item, bound)
    read = lambda item: json.loads(bind(item).read_text())
    archive = lambda item: native.load_state(item, bind)
    reviewed = {}
    for name, directory in REVIEWED.items():
        item = record(directory / "result.json")
        require(item["sha256"] == contract["reviewed_result_sha256"][name],
                f"wrong reviewed {name} evidence")
        document = read(item)
        require(document["diagnostic_id"] == (paired.ID if name == "paired" else paired.drift.ID)
                and document["status"] == "DIAGNOSED"
                and document["candidate_admitted"] is False
                and document["policy_adopted"] is False
                and document["successor_published"] is False and document["rtl_invocations"] == 0,
                "reviewed evidence scope mismatch")
        for artifact in document["input_bindings"] + document["artifacts"] + [document["validation"]]:
            bind(artifact)
        reviewed[name] = document
    require(reviewed["drift"]["original_S18_bitwise_reproduction"] is True
            and reviewed["paired"]["native_retained_bitwise_reproduction"] is True
            and reviewed["paired"]["unchanged_baseline_gates"] is True,
            "missing reviewed reference/native reproduction")

    def reviewed_archive(name, filename):
        path = str(REVIEWED[name] / filename)
        matches = [item for item in reviewed[name]["artifacts"] if item["path"] == path]
        require(len(matches) == 1, f"missing reviewed artifact: {path}")
        return archive(matches[0])

    intermediates = reviewed_archive("drift", "reference_intermediates.npz")
    paired_vectors = reviewed_archive("paired", "native_vectors.npz")
    freeze_record = record(prior.INPUT / "freeze.json")
    freeze = read(freeze_record)
    extension = read(freeze["reference_extension"])
    retained = read(record(prior.INPUT / "result.json"))
    entries = {entry["layer"]: entry for entry in retained["layers"]}
    expected_parent = freeze["state_lineage"]["prior_residual"]["parent"]["state"]
    require(freeze["state_lineage"]["prior_kv"] == "own empty P0"
            and freeze["state_lineage"]["prior_residual"]["prior_layer_kv_consumed"] is False,
            "wrong accepted-parent KV boundary")
    expected64, expected16 = extension["original_binary64_parent"], extension["original_fp16_parent"]
    data = {}
    with safe_open(str(bind(extension["checkpoint"])), framework="numpy") as model:
        for layer in range(9, 14):
            item, entry = extension["layers"][str(layer)], entries[layer]
            require(item["input_binary64"] == expected64 and item["input_fp16"] == expected16
                    and item["prior_kv"] == "own empty P0",
                    f"spliced original-input reference at L{layer}")
            bind(item["input_binary64"])
            bind(item["input_fp16"])
            expected64, expected16 = item["binary64"], item["fp16"]
            reference = np.load(bind(item["binary64"]), allow_pickle=False)
            paired.drift.same_binary64(reference, reference)
            require(np.all(np.abs(reference) <= 65504), "out-of-range original reference")
            tensors = {key: np.ascontiguousarray(model.get_tensor(key))
                       for key in prior.local.tensor_shapes(layer)}
            prior.local.authenticate_tensors(tensors, item["canonical"], layer)
            require(entry["input_parent"] == expected_parent, f"spliced actual parent at L{layer}")
            parent = archive(entry["input_parent"])
            saved = archive(entry["actual_stages"])
            reports = read(entry["reports"])
            require(len(reports) == 19 and all(
                report["node"] == [layer, 0, stage]
                and report["policy_id"] == prior.gates.POLICY_ID
                and report["residual_state_lineage"] == report["kv_lineage"] == "PASS"
                for stage, report in enumerate(reports)), "retained node/gate/lineage mismatch")
            prior.retained.verify_parent(parent, parent)
            for stage in range(19):
                prior.retained.check_stage_state(stage, saved, parent)
            if layer < 13:
                receipt = read(record(prior.INPUT / f"layer{layer:02d}/software_parent.json"))
                require(receipt["state"] == entry["output_parent"]
                        and receipt["state_lineage_parent"] == entry["input_parent"]
                        and receipt["numerical_report"] == entry["reports"]
                        and receipt["kv"] == entry["kv_state"]
                        and receipt["arithmetic_lineage"] == freeze_record
                        and receipt["candidate_id"] == freeze["contract"]["candidate_id"]
                        and receipt["state_id"] == freeze["contract"]["state_id"]
                        and receipt["policy_id"] == prior.gates.POLICY_ID
                        and receipt["evidence_kind"] == "cpu_software_q24"
                        and receipt["history"] == [9707] and receipt["position"] == 0
                        and receipt["next_layer"] == layer + 1 and receipt["rtl_admissible"] is False
                        and receipt["normal_host_review"] == "REQUIRED"
                        and all(report["status"] == "PASS" for report in reports),
                        f"L{layer} retained receipt mismatch")
                prior.retained.verify_parent(archive(entry["output_parent"]),
                    prior.retained.state_from(saved, "output", "stage18"))
                paired.same_arrays(archive(entry["kv_state"]),
                                   {kind: saved["output_cache_" + kind] for kind in ("k", "v")})
                expected_parent = entry["output_parent"]
            data[layer] = {
                "tensors": tensors, "parent": parent, "saved": saved, "reports": reports,
                "locals": archive(entry["local_references"]),
                "trajectory": archive(item["fp16"]), "reference": reference,
            }
    paired.drift.same_binary64(intermediates["original_input"], data[12]["reference"])
    paired.drift.same_binary64(intermediates["original_s18"], data[13]["reference"])
    timing = {"authentication": time.monotonic() - started}
    artifacts, baseline, cut_results = [], {}, []

    def run_layer(label, layer, incoming, actual=False):
        began = time.monotonic()
        item = data[layer]
        arrays, locals_, reports = execute_layer(
            item["tensors"], layer, incoming, item["trajectory"], item["reference"])
        if actual:
            paired.same_arrays(arrays, item["saved"])
            paired.same_arrays(locals_, item["locals"])
            for fresh, saved in zip(reports, item["reports"], strict=True):
                for key in ("status", "local_operator_fp16" if fresh["stage"] < 18 else "binary64_v1"):
                    require(fresh[key] == saved[key], f"retained gate changed: L{layer}/{key}")
        stem = f"{label}_layer{layer:02d}"
        artifact = prior.retained.save(out / f"{stem}.npz", arrays)
        artifacts.append(artifact)
        report_path = out / f"{stem}_gates.json"
        write(report_path, reports)
        artifacts.append(record(report_path))
        elapsed = time.monotonic() - began
        timing[stem] = elapsed
        summary = {
            "layer": layer, "vectors": artifact, "gates": artifacts[-1],
            "mandatory_statuses": [r["status"] for r in reports],
            "S18_failure_indices": [row["index"] for row in reports[18]["binary64_v1"]["failures"]],
            "state_operand_KV_RTZ_checks": "PASS",
            "index62": prior.measure(int(arrays["stage18"][62]), float(item["reference"][62])),
        }
        log.write(json.dumps({"control": label, **summary, "seconds": elapsed}) + "\n")
        log.flush()
        return arrays, summary

    incoming = data[9]["parent"]
    for layer in range(9, 14):
        prior.retained.verify_parent(incoming, data[layer]["parent"])
        arrays, summary = run_layer("actual", layer, incoming, actual=True)
        baseline[layer] = {"arrays": arrays, "summary": summary}
        incoming = prior.retained.state_from(arrays, "output", "stage18")
    actual_rows = [prior.measure(int(baseline[13]["arrays"]["stage18"][i]),
                                float(data[13]["reference"][i])) for i in range(896)]
    require([i for i, row in enumerate(actual_rows) if not row["accepted"]] == [62],
            "retained full-vector failure set changed")
    full_vectors = {"actual": actual_rows}
    for cut in CUTS:
        original = data[cut]["reference"]
        mapped = paired.mapped_parent(original)
        errors = [Fraction(int(i), 1 << 24) - paired.drift.exact(value)
                  for i, value in zip(mapped["i"], original, strict=True)]
        parent_record = prior.retained.save(out / f"cut{cut:02d}_mapped_parent.npz", mapped)
        artifacts.append(parent_record)
        incoming, suffix = mapped, []
        for layer in range(cut + 1, 14):
            arrays, summary = run_layer(f"cut{cut:02d}_mapped", layer, incoming)
            suffix.append(summary)
            incoming = prior.retained.state_from(arrays, "output", "stage18")
        rows = [prior.measure(int(arrays["stage18"][i]), float(data[13]["reference"][i]))
                for i in range(896)]
        full_vectors[f"cut{cut:02d}_mapped"] = rows
        if cut == 12:
            for name, reproduced in (("actual", baseline[13]["arrays"]),
                                     ("original_Q24_RNE", arrays)):
                prefix = name + "__native__"
                paired.same_arrays(reproduced, {
                    key[len(prefix):]: value for key, value in paired_vectors.items()
                    if key.startswith(prefix)})
                require(rows[62] == reviewed["paired"]["selected_coordinate"]
                        ["original_Q24_RNE"]["native"]["gate"], "reviewed mapped endpoint changed")
        cut_results.append({
            "cut_after_layer": cut, "suffix_layers": list(range(cut + 1, 14)),
            "actual_parent": entries[cut]["output_parent"], "original_parent": extension["layers"][str(cut)]["binary64"],
            "mapped_parent": parent_record, "mapping_error_per_coordinate": list(map(str, errors)),
            "maximum_absolute_mapping_error": str(max(map(abs, errors))),
            "actual_suffix": [baseline[layer]["summary"] for layer in range(cut + 1, 14)],
            "mapped_suffix": suffix, "actual_index62": actual_rows[62], "mapped_index62": rows[62],
            "L13_S18_changed_indices": [i for i in range(896)
                                      if rows[i]["actual_fp16_bits"] != actual_rows[i]["actual_fp16_bits"]],
            "L13_S18_failure_indices": [i for i, row in enumerate(rows) if not row["accepted"]],
            "all_suffix_gates_pass": all(
                status == "PASS" for step in suffix for status in step["mandatory_statuses"]),
            "candidate_admitted": False,
        })
    write(out / "full_vector_outcomes.json", full_vectors)
    write(out / "cuts.json", cut_results)
    write(out / "retained_authentication.json", authentication)
    artifacts.extend(record(out / name) for name in
                     ("full_vector_outcomes.json", "cuts.json", "retained_authentication.json"))
    timing["total_diagnosis"] = time.monotonic() - started
    return {
        "diagnostic_id": ID, "status": "DIAGNOSED", "node": [13, 0, 18], "index": 62,
        "candidate_admitted": False, "policy_adopted": False, "successor_published": False,
        "rtl_invocations": 0, "normal_host_review": "REQUIRED", "retained_status": "FAIL",
        "native_retained_bitwise_reproduction": True, "unchanged_baseline_gates": True,
        "original_global_reference_unchanged": True, "reviewed_L12_cut_reproduced": True,
        "source_operand_state_KV_lineage_checks": "PASS",
        "material_set": {"cuts": list(CUTS), "coordinates_per_control": 896,
                         "actual_native_layers": 5, "mapped_native_layers": 10,
                         "actual_suffix_reuse": "one authenticated L9-L13 replay, shared suffixes"},
        "selected_coordinate": [{"cut_after_layer": r["cut_after_layer"],
                                 "actual": r["actual_index62"], "mapped": r["mapped_index62"],
                                 "all_suffix_gates_pass": r["all_suffix_gates_pass"]}
                                for r in cut_results],
        **classify(cut_results), "timing_seconds": timing, "input_bindings": list(bound.values()),
        "origins_after_execution": origins(), "artifacts": artifacts,
        "claim_boundary": contract["claim_boundary"],
    }


def output_path(value):
    out = value.resolve()
    require(out.parent == ROOT / "build"
            and out.name.startswith("q24_s16_l13_upstream_parent_cut_"),
            "output outside bounded build scope")
    require(not out.exists(), "output already exists")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(args.input.resolve() == prior.INPUT, "unselected retained input")
    out = output_path(args.out)
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
            contract = validate(out)
            result = diagnose(out, log, contract)
            result["validation"] = record(out / "validation.json")
        except (ValueError, OSError, KeyError, TypeError, IndexError, ArithmeticError,
                RuntimeError, py_compile.PyCompileError) as exc:
            result = {
                "diagnostic_id": ID, "status": "BLOCKED", "classification": "inconclusive",
                "missing_evidence": [f"{type(exc).__name__}: {exc}"],
                "candidate_admitted": False, "policy_adopted": False,
                "successor_published": False, "rtl_invocations": 0, "normal_host_review": "REQUIRED",
            }
        write(out / "result.json", result)
        summary = {key: result[key] for key in
                   ("status", "classification", "candidate_admitted", "missing_evidence")}
        summary["result"] = str(out / "result.json")
        log.write(json.dumps(summary) + "\n")
        print(json.dumps(summary), flush=True)
    return 0 if result["status"] == "DIAGNOSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
