"""One stdout-only full-stage17-vector suffix experiment, never admission."""

import argparse
from contextlib import contextmanager
from fractions import Fraction
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_mlp_stage17_hotspot_suffix_intervention_v1 as previous


ROOT, PYTHON = previous.ROOT, previous.PYTHON
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_mlp_stage17_full_vector_suffix_intervention_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / "ace3/model/candidates" / (NAME + ".py")
TEST = ROOT / "tests" / ("test_" + NAME + ".py")
MISSION = "f96d2447f5f5"
CONTROLS, BRANCHES, PAIR = previous.CONTROLS, previous.BRANCHES, previous.PAIR
COMPONENT = previous.COMPONENT
capture, retained, direct, hotspot = previous.capture, previous.retained, previous.direct, previous.hotspot
guards, parent, layer, native = previous.guards, previous.parent, previous.layer, previous.native
require, same, bound_bytes, exact, middle = (
    previous.require, previous.same, previous.bound_bytes, previous.exact, previous.middle)
ENVIRONMENT, FLAGS, COUNTS = previous.ENVIRONMENT, previous.FLAGS, previous.COUNTS
PARENT_ROOT = ROOT / "build/mlp17-hotspot-c6a656a6533d-attempt001"
PINS = {
    "stdout": {"path": str(PARENT_ROOT / "run/check.stdout"), "bytes": 8850162,
               "sha256": "b767030a71361288c1638b3cbb51c220fbdcb831889f668c6bc94a3eed359c34"},
    "capture": {"path": str(PARENT_ROOT / "run/capture.json"), "bytes": 19173,
                "sha256": "c79e47a166a1b761c0b5641429bcf51a42d7fe1f06ea3ee5f7d40a7fca013b3c"},
    "outer": {"path": str(PARENT_ROOT / "launcher.capture.json"), "bytes": 1425,
              "sha256": "a3a34764dc9be3a9c2e7126da1bba32c815160535b93e756ab0e76a7150942b9"},
    "review": {
        "path": "/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/c6a656a6533d/round-0001.json",
        "bytes": 689, "sha256": "3f7f741d478af69820504ab095c498cbf9753d5d0ad27b1dab2eb4063b94a2f8"},
}
BOUNDARY = (
    "One full 896-coordinate stage17 replacement, L23/P0, pair (34319,13), nine "
    "retained controls. Only S18 addition/state-lineage oracle, final RMSNorm and "
    "two tied-head rows execute. Original-input FP16/binary64 global references, "
    "thresholds, native S16 RTZ, G128 asymmetric INT4 native GEMM nibble ordering "
    "without qzero plus-one, FP16 scales/operator/KV boundaries and wider-than-FP16 "
    "Q24 residual state remain unchanged. No prefix/admission/reference/closed "
    "producer or intervention replay, strict-FP16-state W4A16, new-token, full-model "
    "admission, precision/scale change or hardware/GPU/RTL/ACE2 action. Stop after "
    "this observation and independent Host review; no next ranking or classifier."
)
PREREGISTRATION = {
    "replacement": "Replace only the working L23 stage17 vector with the retained "
                   "original-input FP16 stage17 vector for each fixed control.",
    "prediction": "Before suffix execution, negate the exact sum over all 896 "
                  "retained stage17 actual-minus-original-FP16 components, weighted "
                  "by the retained branch reference inverse-norm anchor, unchanged "
                  "FP16 norm weight and tied-head row difference. This includes "
                  "the unselected complement, not just retained hotspot coordinates.",
    "SUPPORTED": "Every integrity and existing S18 numerical gate passes and all "
                 "18 rows have strict nonzero full-vector and actual pair-margin "
                 "movement in the preregistered direction, with exact component closure.",
    "REJECTED": "Authenticated numerical failure, zero/opposite movement or component "
                "prediction mismatch rejects full-stage17 sufficiency. An obstruction "
                "outside this replacement remains; no unique causal repair is adopted.",
    "UNKNOWN": "Authentication, source/test/capture, state/KV/lineage, zero-prediction "
               "or execution-integrity defect.",
}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "independent oracle loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_members(directory, receipt):
    same(receipt["preflight"]["environment"], ENVIRONMENT, "capture environment identity")
    for result in receipt["results"]:
        same([p["path"] for p in result["files"]],
             [str(directory / (result["label"] + s)) for s in capture.SUFFIXES],
             "capture member paths")
        data = [bound_bytes(pin) for pin in result["files"]]
        same(data[0], (result["command"] + "\n").encode(), "command binding")
        same(json.loads(data[1]), result["argv"], "argv binding")
        same(json.loads(data[2]), receipt["preflight"], "environment binding")
        same(result["command"], capture.environment_command(ENVIRONMENT, result["argv"]),
             "command identity")
        require(result["exit_status"] == 0 and not result["timed_out"]
                and not result.get("launch_error") and data[4] == b"", "capture failed")
        same(data[5], b"COMMAND\n" + data[0] + b"ENVIRONMENT\n" + data[2]
             + b"\nSTDOUT\n" + data[3] + b"\nSTDERR\n" + data[4]
             + b"\nEXIT_STATUS=0\nTIMED_OUT=False\n", "whole-command binding")


def authenticate_suffix():
    review = json.loads(bound_bytes(PINS["review"]))
    same((review["mission_id"], review["producer_role"], review["review"]["status"]),
         ("c6a656a6533d", "reviewer", "done"), "independent parent review")
    inner = json.loads(bound_bytes(PINS["capture"]))
    outer = json.loads(bound_bytes(PINS["outer"]))
    require(inner["success"] is True and inner["failure"] is None
            and outer["success"] is True and outer["exit_status"] == 0
            and outer["timed_out"] is False, "parent capture failure")
    preflight = inner["preflight"]
    same((preflight["mission_id"], preflight["role"], preflight["uid"],
          preflight["cwd"], preflight["python"], preflight["environment"]),
         ("c6a656a6533d", "engineer", 1000, str(ROOT), PYTHON, ENVIRONMENT),
         "parent identity")
    same(preflight["command_budget"], {"compile": 1, "pytest": 1, "check": 1},
         "parent budget")
    same(preflight["model_or_service_calls_authorized"], 0, "parent service authority")
    same(preflight["independent_host_review"], "REQUIRED", "parent review authority")
    same(inner["sources_after"], preflight["sources"], "parent source drift")
    same(preflight["sources"], [capture.binding(previous.SOURCE), capture.binding(previous.TEST)],
         "parent source/test pins")
    for pins in (preflight["capture_implementation"], inner["capture_implementation_after"],
                 outer["capture_implementation_after"]):
        same(pins, capture.implementation_pins(), "parent shared sealer drift")
    bound_bytes(preflight["executable"])
    same([r["label"] for r in inner["results"]],
         ["branch", "ignored-build", "concurrency", "compile", "pytest", "check"],
         "parent validation census")
    verify_members(PARENT_ROOT / "run", inner)
    same(inner["results"][-1]["argv"], [PYTHON, "-B", "-m", previous.MODULE, "--check"],
         "parent diagnostic command")
    same(inner["results"][-1]["files"][3], PINS["stdout"], "parent stdout splice")
    same([p["path"] for p in outer["files"]],
         [str(PARENT_ROOT / ("launcher." + s)) for s in
          ("identity.json", "stdout", "stderr", "whole-command.log")], "outer paths")
    identity, output, error, whole = [bound_bytes(p) for p in outer["files"]]
    launch = json.loads(identity)
    same((launch["argv"], launch["environment"], launch["cwd"], launch["uid"]),
         ([PYTHON, "-B", str(previous.TEST), "--run", str(PARENT_ROOT / "run")],
          ENVIRONMENT, str(ROOT), 1000), "outer identity")
    same(launch["launcher"], preflight["sources"][1], "outer launcher source")
    same(launch["capture_implementation"], capture.implementation_pins(), "outer helper")
    same(launch["command"], capture.environment_command(ENVIRONMENT, launch["argv"]),
         "outer command")
    same(error, b"", "outer stderr")
    same(whole, b"IDENTITY\n" + identity + b"STDOUT\n" + output + b"\nSTDERR\n"
         + error + b"\nEXIT_STATUS=0\nTIMED_OUT=False\n", "outer whole command")
    same(json.loads(output), {"capture": PINS["capture"], "success": True, "failure": None},
         "outer/inner receipt splice")
    result = json.loads(bound_bytes(PINS["stdout"]))
    same((result["diagnostic_id"], result["version"], result["artifact_authentication"],
          result["status"]), (previous.NAME, 1, "AUTHENTICATED", "REJECTED"),
         "reviewed single-coordinate classification")
    same(result["flags"], FLAGS, "parent non-admission flags")
    same(result["dispatch_and_write_audit"], COUNTS, "parent suffix census")
    same(result["parent_pins"], {"hotspot": hotspot.PINS, "direct_hidden": direct.PINS},
         "parent lineage splice")
    for pin in (*result["diagnostic_sources"].values(), *result["authenticated_files"]):
        bound_bytes(pin)
    return result


def prepare(actual, reference):
    require(actual["stage17"].dtype.str == reference["stage17"].dtype.str == "<u2"
            and actual["stage17"].shape == reference["stage17"].shape == (896,),
            "stage17 operand shape/dtype")
    arrays = {k: v.copy() for k, v in actual.items()}
    arrays["stage17"] = reference["stage17"].copy()
    for value in arrays.values():
        value.flags.writeable = False
    protected(actual, arrays, reference)
    return arrays


def protected(actual, arrays, reference):
    for key, value in actual.items():
        if key in ("stage17", "stage18", "output_i", "output_z"):
            continue
        other = arrays[key]
        require(value.dtype == other.dtype and value.shape == other.shape
                and value.tobytes() == other.tobytes(), "protected operand/state/KV: " + key)
    value, other = reference["stage17"], arrays["stage17"]
    require(value.dtype == other.dtype and value.shape == other.shape
            and value.tobytes() == other.tobytes(), "full-vector replacement changed")
    for kind, stage in (("k", "05"), ("v", "03")):
        same(actual["output_cache_" + kind].shape, (1, 128), "P0 KV shape")
        require(np.array_equal(actual["output_cache_" + kind][0], actual["stage" + stage]),
                "retained KV lineage")


def preregister(evidence, archives, reference, operands):
    controls, originals = evidence["report"]["controls"], evidence["retained_50f"]["report"]["controls"]
    same([c["control"] for c in controls], list(CONTROLS), "hotspot control census")
    same([c["control"] for c in originals], list(CONTROLS), "direct-hidden control census")
    same(list(archives), list(CONTROLS), "archive census")
    same((operands[0].shape, operands[1].shape), ((896,), (2, 896)), "fixed operand shapes")
    same((operands[0].dtype.str, operands[1].dtype.str), ("<f2", "<f2"), "FP16 boundaries")
    gamma, weights = operands[0].view("<u2"), operands[1].view("<u2")
    rows = []
    for control, original in zip(controls, originals, strict=True):
        actual = archives[control["control"]]
        prepare(actual, reference)
        differences = [exact(a) - exact(b) for a, b in
                       zip(actual["stage17"], reference["stage17"], strict=True)]
        for branch in BRANCHES:
            table, source = middle(control)["branches"][branch], middle(original)["branches"][branch]
            same(source["residual_internal_reference"], "original_input_L23_fp16", "internal reference")
            same(source["hidden_reference"], "original_input_L23_" + branch, "global reference")
            same(source["binary64_internal_stages"], "NOT_RETAINED_NO_RECONSTRUCTION",
                 "internal binary64 reconstruction")
            cells = source["selected_coordinates"]
            indices = [c["coordinate"] for c in cells]
            require(indices and indices == sorted(set(indices))
                    and all(type(k) is int and 0 <= k < 896 for k in indices), "coordinate census")
            anchor = Fraction(cells[0]["reference_inverse_norm_anchor"])
            require(anchor > 0, "invalid retained reference anchor")
            factors = [exact(gamma[k]) * anchor * (exact(weights[0, k]) - exact(weights[1, k]))
                       for k in range(896)]
            components = [factor * delta for factor, delta in zip(factors, differences, strict=True)]
            ranking = table["per_component_coordinate_hotspots"][COMPONENT]["ranking"]
            same(sorted(r["coordinate"] for r in ranking), indices, "hotspot coordinate splice")
            ranked = {r["coordinate"]: r for r in ranking}
            for cell in cells:
                k = cell["coordinate"]
                same(Fraction(cell["reference_inverse_norm_anchor"]), anchor, "anchor splice")
                boundary = cell["retained_fp16_residual_account"]["boundaries"]["stage17"]
                same(exact(actual["stage17"][k]), Fraction(boundary["actual_exact"]), "actual operand splice")
                same(exact(reference["stage17"][k]), Fraction(boundary["reference_exact"]),
                     "original-input operand splice")
                same(differences[k], Fraction(cell["hidden_components"][COMPONENT]), "hidden closure")
                same(factors[k], Fraction(cell["weight_times_reference_anchor_times_row_difference"]),
                     "fixed-weight closure")
                same(components[k], Fraction(cell["weighted_components"][COMPONENT]), "component closure")
                same((Fraction(ranked[k]["signed"]), Fraction(ranked[k]["absolute"])),
                     (components[k], abs(components[k])), "hotspot mass splice")
            selected = sum((components[k] for k in indices), Fraction())
            same(selected, Fraction(source["direct_hidden_accounting"]["component_totals"][COMPONENT]["signed"]),
                 "retained selected-component total")
            prediction = -sum(components, Fraction())
            require(prediction != 0, "zero retained full-vector prediction")
            rows.append({"control": control["control"], "branch": branch,
                         "reference_inverse_norm_anchor": str(anchor),
                         "predicted_delta": str(prediction), "retained_selected_signed": str(selected),
                         "unselected_signed": str(-prediction - selected),
                         "factors": list(map(str, factors)), "components": list(map(str, components))})
    return {**PREREGISTRATION, "pair": list(PAIR), "component": COMPONENT,
            "coordinates": list(range(896)), "source_position": 0, "source_token_id": 9707, "rows": rows}


@contextmanager
def suffix_only(audit, active, operands):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("closed single-coordinate intervention replay forbidden")
    with previous.suffix_only(audit, active, operands), patch.object(previous, "check", forbidden):
        yield


def classify(rows, reports):
    same([(r["control"], r["branch"]) for r in rows],
         [(c, b) for c in CONTROLS for b in BRANCHES], "directional census")
    same(list(reports), list(CONTROLS), "numerical census")
    require(all(r["stage"] == 18 and r["status"] in ("PASS", "FAIL")
                and r["residual_state_lineage"] == r["kv_lineage"] == "PASS"
                for r in reports.values()), "S18 state/KV/lineage integrity")
    agreements = []
    for row in rows:
        p, m, delta = (Fraction(row[k]) for k in
                       ("predicted_delta", "mlp_full_vector_delta", "margin_delta"))
        require(p != 0, "zero retained full-vector prediction")
        agreement = p * m > 0 and p * delta > 0 and m == p
        same(row["direction_observed"], agreement, "direction result splice")
        agreements.append(agreement)
    return ("SUPPORTED" if all(agreements) and all(r["status"] == "PASS" for r in reports.values())
            else "REJECTED")


def capture_preflight():
    require(Path.cwd() == ROOT and sys.executable == PYTHON and os.getuid() == 1000
            and sys.dont_write_bytecode and not sys.flags.optimize, "source/account/Python gate")
    same({k: os.environ.get(k) for k in ENVIRONMENT}, ENVIRONMENT, "environment gate")
    out = Path(os.readlink("/proc/self/fd/1"))
    require(out.name == "check.stdout" and out.parent.name == "run"
            and out.parent.parent.parent == ROOT / "build" and out.resolve() == out,
            "byte-exact shared capture required")
    same(os.readlink("/proc/self/fd/2"), str(out.with_name("check.stderr")), "stderr capture")
    preflight = json.loads(out.with_name("check.environment.json").read_bytes())
    same(preflight["sources"], [capture.binding(SOURCE), capture.binding(TEST)], "source/test pins")
    same(preflight["environment"], ENVIRONMENT, "capture environment")
    same(preflight["capture_implementation"], capture.implementation_pins(), "shared sealer drift")
    same((preflight["mission_id"], preflight["role"], preflight["independent_host_review"],
          preflight["model_or_service_calls_authorized"]), (MISSION, "engineer", "REQUIRED", 0),
         "mission/review/service authority")
    same(preflight["command_budget"], {"compile": 1, "pytest": 1, "check": 1}, "command budget")
    bound_bytes(preflight["executable"])
    argv = [PYTHON, "-B", "-m", MODULE, "--check"]
    same(json.loads(out.with_name("check.argv.json").read_bytes()), argv, "check argv")
    same(out.with_name("check.command.txt").read_bytes(),
         (capture.environment_command(ENVIRONMENT, argv) + "\n").encode(), "check command")
    for label in ("compile", "pytest"):
        same(out.with_name(label + ".stderr").read_bytes(), b"", "validation stderr")
        require(out.with_name(label + ".whole-command.log").read_bytes().endswith(
            b"\nEXIT_STATUS=0\nTIMED_OUT=False\n"), "validation incomplete")
    suites = ET.fromstring(out.with_name("pytest.xml").read_bytes()).findall("testsuite")
    require(suites and sum(int(s.attrib["tests"]) for s in suites) >= 12
            and not any(int(s.attrib[k]) for s in suites for k in ("errors", "failures", "skipped")),
            "task-native tests failed or incomplete")
    return preflight, {"compiled": preflight["sources"],
                       "pytest": capture.binding(out.with_name("pytest.xml")),
                       "executed": sum(int(s.attrib["tests"]) for s in suites)}


def check():
    preflight, tests = capture_preflight()
    native.torch.set_num_threads(1)
    same(str(native.torch.tensor(0).device), "cpu", "CPU-only execution")
    oracle = load_module(TEST, NAME + "_tests")
    sources = parent.source_context()
    sources[MODULE], sources["task_tests"] = capture.binding(SOURCE), capture.binding(TEST)
    audit = dict.fromkeys(COUNTS, 0)
    began = time.monotonic()
    with guards.no_writes(audit), parent.no_dispatch(audit):
        single = authenticate_suffix()
        evidence, original = hotspot.authenticate(), direct.authenticate()
        same(evidence["retained_50f"], original, "reviewed direct-hidden splice")
        for pin in (*original["diagnostic_sources"].values(), *original["authenticated_files"],
                    *original["hidden_pins"]):
            bound_bytes(pin)
        result, baseline, references, files = retained.authenticate()
        summary = result["preflight"]
        for key, value in (("original_thresholds", summary["thresholds"]),
                           ("L23_reference_authority", summary["L23_original_reference"]),
                           ("final_reference_authority", summary["final_reference"])):
            same(single[key], value, "single-coordinate reference/threshold splice")
        tensors, trajectory, binary64, checkpoint = layer.load_inputs({"summary": summary})
        same(parent.preflight.bind_assets(checkpoint), summary["assets"], "source/asset identity")
        full = parent.load_operands(summary)
        operands = (full[0], full[1][list(PAIR)].copy())
        archives = {r["control"]: layer.archive(r["parent"]["terminal_archive"])
                    for r in result["controls"]}
        for value in (*tensors.values(), *operands, *trajectory.values(), binary64,
                      *references.values(), *(v for a in archives.values() for v in a.values())):
            value.flags.writeable = False
        plan = preregister(evidence, archives, trajectory, operands)
        authentication_seconds = time.monotonic() - began
        outputs, reports, timings, active = {}, {}, {}, {}
        with suffix_only(audit, active, operands):
            for control in CONTROLS:
                actual = archives[control]
                arrays = prepare(actual, trajectory)
                state = {"i": arrays["input_i"], "z": arrays["input_z"], "h": arrays["input_hidden"]}
                scratch = {"i": arrays["scratch_i"], "z": arrays["scratch_z"], "h": arrays["stage12"]}
                active.update(stage=18, arrays=arrays, input=state, scratch=scratch, tensors=tensors)
                start = time.monotonic()
                changed = native.state.add(scratch, arrays["stage17"])
                arrays.update(output_i=changed["i"], output_z=changed["z"], stage18=changed["h"])
                expected = layer.expected_stage(18, arrays, state, tensors)
                reports[control] = layer.stage_report(18, arrays, trajectory, binary64, expected)
                protected(actual, arrays, trajectory)
                stage_end = time.monotonic()
                active["stage"] = 19
                arrays["final_rmsnorm"], norm_account = parent.rmsnorm(arrays["stage18"], operands[0])
                active["stage"] = 20
                arrays["pair_logits"] = parent.logits(arrays["final_rmsnorm"], operands[1])
                final_end = time.monotonic()
                oracle.verify_suffix(actual, arrays, trajectory, operands, norm_account)
                outputs[control] = {key: arrays[key].tolist() for key in (
                    "stage17", "output_i", "output_z", "stage18", "final_rmsnorm", "pair_logits")}
                outputs[control]["norm_account"] = norm_account
                timings[control] = {"s18_and_gate": stage_end - start,
                                    "final_norm_and_two_rows": final_end - stage_end,
                                    "independent_oracle": time.monotonic() - final_end}
        rows = []
        for prediction in plan["rows"]:
            control, branch = prediction["control"], prediction["branch"]
            mlp = sum((Fraction(factor) * (exact(new) - exact(old)) for factor, new, old in
                       zip(prediction["factors"], outputs[control]["stage17"],
                           archives[control]["stage17"], strict=True)), Fraction())
            old = exact(baseline[control]["logits"][PAIR[0]]) - exact(baseline[control]["logits"][PAIR[1]])
            new = exact(outputs[control]["pair_logits"][0]) - exact(outputs[control]["pair_logits"][1])
            ref = references["logits_" + branch]
            ref_margin = (exact(ref[PAIR[0]]) - exact(ref[PAIR[1]]) if branch == "fp16" else
                          Fraction.from_float(float(ref[PAIR[0]])) - Fraction.from_float(float(ref[PAIR[1]])))
            comparator = [r for r in single["rows"] if (r["control"], r["branch"]) == (control, branch)]
            same(len(comparator), 1, "single-coordinate row census")
            same((comparator[0]["retained_margin"], comparator[0]["independent_original_reference_margin"]),
                 (str(old), str(ref_margin)), "single-coordinate baseline/reference splice")
            p, delta = Fraction(prediction["predicted_delta"]), new - old
            rows.append({"control": control, "branch": branch, "predicted_delta": str(p),
                         "mlp_full_vector_delta": str(mlp), "mlp_prediction_remainder": str(mlp - p),
                         "retained_margin": str(old), "intervened_margin": str(new),
                         "independent_original_reference_margin": str(ref_margin),
                         "retained_margin_error": str(old - ref_margin),
                         "intervened_margin_error": str(new - ref_margin), "margin_delta": str(delta),
                         "margin_prediction_remainder": str(delta - p),
                         "retained_single_coordinate_margin_delta": comparator[0]["margin_delta"],
                         "direction_observed": p * mlp > 0 and p * delta > 0 and mlp == p})
        decision = classify(rows, reports)
        same(audit, COUNTS, "suffix/forbidden dispatch census")
        for pin in (*sources.values(), *files, *original["hidden_pins"], *PINS.values(),
                    *hotspot.PINS.values(), *direct.PINS.values()):
            bound_bytes(pin)
    return {
        "diagnostic_id": NAME, "version": 1, "status": decision,
        "artifact_authentication": "AUTHENTICATED", "preregistration": plan,
        "parent_pins": {"hotspot": hotspot.PINS, "direct_hidden": direct.PINS, "single_coordinate": PINS},
        "diagnostic_sources": sources, "authenticated_files": files,
        "capture_preflight": preflight, "tests": tests, "flags": FLAGS,
        "dispatch_and_write_audit": audit, "stage_reports": reports, "rows": rows, "outputs": outputs,
        "original_thresholds": summary["thresholds"],
        "L23_reference_authority": summary["L23_original_reference"],
        "final_reference_authority": summary["final_reference"],
        "retained_controls_and_failure_gates": original["retained_controls_and_failure_gates"],
        "retained_lineage_separation": original["report"]["lineage_separation"],
        "retained_single_coordinate_classification": single["status"],
        "historical_failures_preserved": True, "original_global_reference_unchanged": True,
        "authentication_seconds": authentication_seconds, "timing_seconds": timings,
        "claim_boundary": BOUNDARY, "normal_host_review": "REQUIRED",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    try:
        result = check()
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, AssertionError,
            ArithmeticError, ET.ParseError) as error:
        print(json.dumps({"diagnostic_id": NAME, "status": "UNKNOWN", "flags": FLAGS,
                          "error_type": type(error).__name__, "error": str(error),
                          "normal_host_review": "REQUIRED"}, sort_keys=True, allow_nan=False))
        return 1
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    from importlib import import_module

    raise SystemExit(import_module(MODULE).main())
