"""One retained stage13-vector intervention; CPU suffix only, never admission."""

import argparse
from contextlib import contextmanager
from fractions import Fraction
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_mlp_stage17_full_vector_suffix_intervention_v1 as shared


ROOT, PYTHON = shared.ROOT, shared.PYTHON
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_stage13_rmsnorm_full_vector_suffix_intervention_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / "ace3/model/candidates" / (NAME + ".py")
TEST = ROOT / "tests" / ("test_" + NAME + ".py")
MISSION = "8de620770717"
CONTROLS, BRANCHES, PAIR = shared.CONTROLS, shared.BRANCHES, shared.PAIR
capture, retained, direct = shared.capture, shared.retained, shared.direct
guards, parent, layer, native = shared.guards, shared.parent, shared.layer, shared.native
require, same, bound_bytes, exact = shared.require, shared.same, shared.bound_bytes, shared.exact
ENVIRONMENT = shared.ENVIRONMENT
FLAGS = {
    **guards.FLAGS, "stage13_rmsnorm_replay": 0, "prefix_replay": 0,
    "closed_producer_replay": 0, "original_reference_replay": 0,
    "full_vector_stage13_intervention": True,
}
COUNTS = {"forbidden_calls": 0, "projections": 27, "local_oracles": 45,
          "s18_adds": 9, "final_norms": 9, "pair_heads": 9}
BOUNDARY = (
    "One L23/P0 full stage13 FP16-vector replacement, pair (34319,13), nine "
    "retained controls. Only stage14-stage18, final RMSNorm and two head rows "
    "execute. Original-input FP16/binary64 references, exact existing thresholds, "
    "native-S16-RTZ, G128 asymmetric INT4/native GEMM/no qzero plus-one, FP16 "
    "scales/operator/KV boundaries and wider-than-FP16 Q24 residual state are fixed. "
    "No prefix/admission/original-reference/closed-producer/source-bridge replay, "
    "strict-FP16-state W4A16, new-token/full-model admission, precision or scale "
    "expansion, hardware/GPU/RTL or ACE2 action. Terminate this full-vector lane "
    "after its one observation; independent Host review is required."
)
PREREGISTRATION = {
    "replacement": "All 896 working stage13 words become the authenticated retained "
                   "original-input FP16 stage13 words, without executing stage13.",
    "prediction": "Before any operator runs, negate the full retained stage17 "
                  "actual-minus-original-FP16 vector weighted by each retained "
                  "original-reference inverse-norm anchor, FP16 norm weights and "
                  "the fixed (34319 minus 13) head-row difference. The recomputed "
                  "stage17 weighted movement and actual FP16 pair-margin movement "
                  "must both have this strict nonzero sign in all 18 rows. "
                  "Magnitude equality is not predicted across nonlinear boundaries.",
    "SUPPORTED": "Every unchanged stage14-stage18 gate and independent final "
                 "RMSNorm/two-row arithmetic check passes, with both strict "
                 "directional contrasts in all rows: bounded stage13 sufficiency only.",
    "REJECTED": "Any numerical gate failure or zero/opposite prediction or movement "
                "rejects sufficiency and terminates the stage13 full-vector lane.",
    "UNKNOWN": "Missing/invalid retained binding, source/operand/state/KV/lineage "
               "or execution/capture integrity defect; terminate without replay.",
}


def stage13_binding(archive, pin):
    label = pin["path"] + "::stage13"
    require("stage13" in archive, "missing retained stage13 binding: " + label)
    words = archive["stage13"]
    require(words.dtype.str == "<u2" and words.shape == (896,),
            "invalid retained stage13 shape/dtype: " + label)
    require(np.isfinite(words.view("<f2")).all(), "nonfinite retained stage13: " + label)
    return {"archive": pin, "member": "stage13", "shape": [896], "dtype": "<u2"}


def prepare(actual, reference):
    stage13_binding(actual, {"path": "actual"})
    stage13_binding(reference, {"path": "original_input_fp16"})
    arrays = {k: v.copy() for k, v in actual.items()}
    arrays["stage13"] = reference["stage13"].copy()
    for value in arrays.values():
        value.flags.writeable = False
    protected(actual, arrays, reference)
    return arrays


def protected(actual, arrays, reference):
    changed = {f"stage{s:02d}" for s in range(13, 19)}
    changed.update(("output_i", "output_z", "s16_unrounded_binary64"))
    for key, value in actual.items():
        if key not in changed:
            other = arrays[key]
            require(value.dtype == other.dtype and value.shape == other.shape
                    and value.tobytes() == other.tobytes(),
                    "protected source/operand/state/KV changed: " + key)
    a, b = arrays["stage13"], reference["stage13"]
    require(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(),
            "full stage13 replacement changed")
    for kind, stage in (("k", "05"), ("v", "03")):
        same(actual["output_cache_" + kind].shape, (1, 128), "P0 KV shape")
        require(np.array_equal(actual["output_cache_" + kind][0], actual["stage" + stage]),
                "retained KV lineage")


def preregister(original, archives, reference, operands):
    controls = original["report"]["controls"]
    same([c["control"] for c in controls], list(CONTROLS), "retained control census")
    same(list(archives), list(CONTROLS), "archive census")
    same(tuple(v.shape for v in operands), ((896,), (2, 896)), "fixed operand shapes")
    same(tuple(v.dtype.str for v in operands), ("<f2", "<f2"), "FP16 operand boundaries")
    gamma, weights = operands[0].view("<u2"), operands[1].view("<u2")
    rows = []
    for control in controls:
        actual = archives[control["control"]]
        for branch in BRANCHES:
            source = shared.middle(control)["branches"][branch]
            same(source["residual_internal_reference"], "original_input_L23_fp16",
                 "internal reference reanchored")
            same(source["hidden_reference"], "original_input_L23_" + branch,
                 "global reference reanchored")
            same(source["binary64_internal_stages"], "NOT_RETAINED_NO_RECONSTRUCTION",
                 "binary64 internal reconstruction")
            cells = source["selected_coordinates"]
            indices = [c["coordinate"] for c in cells]
            require(indices and indices == sorted(set(indices))
                    and all(type(k) is int and 0 <= k < 896 for k in indices),
                    "retained coordinate census")
            anchor = Fraction(cells[0]["reference_inverse_norm_anchor"])
            require(anchor > 0, "invalid reference inverse-norm binding")
            factors = [anchor * exact(gamma[k]) * (exact(weights[0, k]) - exact(weights[1, k]))
                       for k in range(896)]
            components = [factors[k] * (exact(actual["stage17"][k])
                                       - exact(reference["stage17"][k])) for k in range(896)]
            for cell in cells:
                k = cell["coordinate"]
                same(Fraction(cell["reference_inverse_norm_anchor"]), anchor, "anchor splice")
                same(Fraction(cell["weight_times_reference_anchor_times_row_difference"]),
                     factors[k], "fixed row-factor splice")
                boundary = cell["retained_fp16_residual_account"]["boundaries"]["stage17"]
                same((exact(actual["stage17"][k]), exact(reference["stage17"][k])),
                     (Fraction(boundary["actual_exact"]), Fraction(boundary["reference_exact"])),
                     "retained stage17 operand splice")
                same(Fraction(cell["weighted_components"]["mlp_stage17"]), components[k],
                     "retained weighted component splice")
            same(sum((components[k] for k in indices), Fraction()),
                 Fraction(source["direct_hidden_accounting"]["component_totals"]["mlp_stage17"]["signed"]),
                 "retained selected component closure")
            rows.append({"control": control["control"], "branch": branch,
                         "reference_inverse_norm_anchor": str(anchor),
                         "predicted_delta": str(-sum(components, Fraction())),
                         "factors": list(map(str, factors))})
    return {**PREREGISTRATION, "pair": list(PAIR), "source_position": 0,
            "source_token_id": 9707, "coordinates": list(range(896)), "rows": rows}


@contextmanager
def suffix_only(audit, active, tensors, operands):
    raw_add = native.state.add

    def add(state, words):
        require(active.get("stage") == 18 and state is active["scratch"]
                and words is active["arrays"]["stage17"], "S18 state/operand scope")
        audit["s18_adds"] += 1
        return raw_add(state, words)

    with guards.suffix_only(audit, active, tensors, operands):
        projection, expected = native.projection, layer.expected_stage

        def projected(*args):
            require(active.get("stage") in (14, 15, 17), "projection outside stage13 suffix")
            return projection(*args)

        def oracle(stage, *args):
            require(14 <= stage <= 18, "oracle outside stage13 suffix")
            return expected(stage, *args)

        with patch.object(native.state, "add", add), patch.object(native, "projection", projected), \
                patch.object(layer, "expected_stage", oracle):
            yield


def compute_suffix(tensors, arrays, scratch, active):
    for stage in range(14, 19):
        active["stage"] = stage
        if stage in (14, 15, 17):
            name, source = {14: ("gate", 13), 15: ("up", 13), 17: ("down", 16)}[stage]
            word = native.projection(tensors, "model.layers.23.mlp." + name + "_proj",
                                     arrays[f"stage{source:02d}"])
        elif stage == 16:
            value = native.functional.silu(native.decoded(arrays["stage14"])) * native.decoded(arrays["stage15"])
            arrays["s16_unrounded_binary64"] = value.numpy().copy()
            word = native.toward_zero(arrays["s16_unrounded_binary64"])
        else:
            state = native.state.add(scratch, arrays["stage17"])
            arrays["output_i"], arrays["output_z"] = state["i"], state["z"]
            word = state["h"]
        arrays[f"stage{stage:02d}"] = word
        yield stage


def classify(rows, reports):
    same([(r["control"], r["branch"]) for r in rows],
         [(c, b) for c in CONTROLS for b in BRANCHES], "directional census")
    same(list(reports), list(CONTROLS), "gate control census")
    for gates in reports.values():
        same([g["stage"] for g in gates], list(range(14, 19)), "suffix gate census")
        require(all(g["status"] in ("PASS", "FAIL")
                    and g["residual_state_lineage"] == g["kv_lineage"] == "PASS"
                    for g in gates), "source/state/KV/lineage gate integrity")
    agreements = []
    for row in rows:
        p, m, d = (Fraction(row[k]) for k in ("predicted_delta", "mlp_vector_delta", "margin_delta"))
        agreement = p * m > 0 and p * d > 0
        same(row["direction_observed"], agreement, "directional result splice")
        agreements.append(agreement)
    return ("SUPPORTED" if all(agreements)
            and all(g["status"] == "PASS" for gates in reports.values() for g in gates)
            else "REJECTED")


def capture_preflight():
    require((Path.cwd(), sys.executable, os.getuid()) == (ROOT, PYTHON, 1000)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "source/account/interpreter gate")
    same({k: os.environ.get(k) for k in ENVIRONMENT}, ENVIRONMENT, "environment gate")
    out = Path(os.readlink("/proc/self/fd/1"))
    require(out.name == "check.stdout" and out.parent.name == "run"
            and out.parent.parent.parent == ROOT / "build" and out.resolve() == out,
            "byte-exact shared capture required")
    same(os.readlink("/proc/self/fd/2"), str(out.with_name("check.stderr")), "stderr capture")
    preflight = json.loads(out.with_name("check.environment.json").read_bytes())
    same(preflight["sources"], [capture.binding(SOURCE), capture.binding(TEST)], "source/test binding")
    same(preflight["environment"], ENVIRONMENT, "capture environment")
    same(preflight["capture_implementation"], capture.implementation_pins(), "capture implementation")
    same((preflight["mission_id"], preflight["role"], preflight["independent_host_review"],
          preflight["model_or_service_calls_authorized"]), (MISSION, "engineer", "REQUIRED", 0),
         "mission/role/review/service authority")
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
            "focused pytest incomplete")
    return preflight


def check():
    preflight = capture_preflight()
    native.torch.set_num_threads(1)
    same(str(native.torch.tensor(0).device), "cpu", "CPU-only execution")
    oracle = shared.load_module(TEST, NAME + "_tests")
    sources = {**parent.source_context(), MODULE: capture.binding(SOURCE),
               "task_tests": capture.binding(TEST)}
    audit = dict.fromkeys(COUNTS, 0)
    began = time.monotonic()
    with guards.no_writes(audit), parent.no_dispatch(audit):
        original = direct.authenticate()
        for pin in (*original["diagnostic_sources"].values(), *original["authenticated_files"],
                    *original["hidden_pins"]):
            bound_bytes(pin)
        result, baseline, references, files = retained.authenticate()
        summary = result["preflight"]
        reference_pin = summary["L23_original_reference"]["reference"]["fp16"]
        reference_archive = layer.archive(reference_pin)
        bindings = {"original_input_fp16": stage13_binding(reference_archive, reference_pin)}
        archives = {}
        for row in result["controls"]:
            pin = row["parent"]["terminal_archive"]
            archives[row["control"]] = layer.archive(pin)
            bindings[row["control"]] = stage13_binding(archives[row["control"]], pin)
        tensors, trajectory, binary64, checkpoint = layer.load_inputs({"summary": summary})
        same(parent.preflight.bind_assets(checkpoint), summary["assets"], "source/asset identity")
        same(trajectory["stage13"].tobytes(), reference_archive["stage13"].tobytes(),
             "retained original stage13 splice")
        full_operands = parent.load_operands(summary)
        operands = (full_operands[0], full_operands[1][list(PAIR)].copy())
        for value in (*tensors.values(), *operands, *trajectory.values(), binary64,
                      *references.values(), *(v for a in archives.values() for v in a.values())):
            value.flags.writeable = False
        plan = preregister(original, archives, trajectory, operands)
        authentication_seconds = time.monotonic() - began
        outputs, reports, timings, active = {}, {}, {}, {}
        with suffix_only(audit, active, tensors, operands):
            for control in CONTROLS:
                actual = archives[control]
                arrays = prepare(actual, trajectory)
                state = {"i": arrays["input_i"], "z": arrays["input_z"], "h": arrays["input_hidden"]}
                scratch = {"i": arrays["scratch_i"], "z": arrays["scratch_z"], "h": arrays["stage12"]}
                active.update(arrays=arrays, state=state, scratch=scratch)
                reports[control], timings[control] = [], {}
                start = time.monotonic()
                for stage in compute_suffix(tensors, arrays, scratch, active):
                    expected = layer.expected_stage(stage, arrays, state, tensors)
                    reports[control].append(layer.stage_report(stage, arrays, trajectory, binary64, expected))
                    now = time.monotonic()
                    timings[control][f"stage{stage:02d}_and_gate"] = now - start
                    start = now
                protected(actual, arrays, trajectory)
                active["stage"] = 19
                arrays["final_rmsnorm"], account = parent.rmsnorm(arrays["stage18"], operands[0])
                active["stage"] = 20
                arrays["pair_logits"] = parent.logits(arrays["final_rmsnorm"], operands[1])
                now = time.monotonic()
                timings[control]["final_norm_and_two_rows"] = now - start
                oracle.verify_final_suffix(actual, arrays, operands, account)
                timings[control]["independent_final_oracle"] = time.monotonic() - now
                outputs[control] = {k: arrays[k].tolist() for k in (
                    *(f"stage{s:02d}" for s in range(13, 19)), "output_i", "output_z",
                    "s16_unrounded_binary64", "final_rmsnorm", "pair_logits")}
                outputs[control]["norm_account"] = account
        rows = []
        for prediction in plan["rows"]:
            control, branch = prediction["control"], prediction["branch"]
            mlp = sum((Fraction(f) * (exact(new) - exact(old)) for f, new, old in
                       zip(prediction["factors"], outputs[control]["stage17"],
                           archives[control]["stage17"], strict=True)), Fraction())
            old = exact(baseline[control]["logits"][PAIR[0]]) - exact(baseline[control]["logits"][PAIR[1]])
            new = exact(outputs[control]["pair_logits"][0]) - exact(outputs[control]["pair_logits"][1])
            ref = references["logits_" + branch]
            ref_margin = (exact(ref[PAIR[0]]) - exact(ref[PAIR[1]]) if branch == "fp16" else
                          Fraction.from_float(float(ref[PAIR[0]])) - Fraction.from_float(float(ref[PAIR[1]])))
            p, delta = Fraction(prediction["predicted_delta"]), new - old
            rows.append({"control": control, "branch": branch, "predicted_delta": str(p),
                         "mlp_vector_delta": str(mlp), "margin_delta": str(delta),
                         "retained_margin": str(old), "intervened_margin": str(new),
                         "independent_original_reference_margin": str(ref_margin),
                         "retained_margin_error": str(old - ref_margin),
                         "intervened_margin_error": str(new - ref_margin),
                         "direction_observed": p * mlp > 0 and p * delta > 0})
        decision = classify(rows, reports)
        same(audit, COUNTS, "suffix dispatch census")
        for pin in (*sources.values(), *files, *original["hidden_pins"], *direct.PINS.values()):
            bound_bytes(pin)
    return {
        "diagnostic_id": NAME, "version": 1, "status": decision,
        "artifact_authentication": "AUTHENTICATED", "preregistration": plan,
        "stage13_bindings": bindings, "parent_pins": {"direct_hidden": direct.PINS},
        "diagnostic_sources": sources, "authenticated_files": files,
        "capture_preflight": preflight, "flags": FLAGS, "dispatch_and_write_audit": audit,
        "stage_reports": reports, "rows": rows, "outputs": outputs,
        "final_arithmetic_oracle": "PASS",
        "final_comparison_scope": "No new final-head admission threshold; exact arithmetic and directional contrasts only",
        "original_thresholds": summary["thresholds"],
        "L23_reference_authority": summary["L23_original_reference"],
        "final_reference_authority": summary["final_reference"],
        "retained_controls_and_failure_gates": original["retained_controls_and_failure_gates"],
        "retained_lineage_separation": original["report"]["lineage_separation"],
        "historical_failures_preserved": True, "original_global_reference_unchanged": True,
        "authentication_seconds": authentication_seconds, "timing_seconds": timings,
        "claim_boundary": BOUNDARY, "lane_terminated": True, "normal_host_review": "REQUIRED",
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
                          "lane_terminated": True, "normal_host_review": "REQUIRED"},
                         sort_keys=True, allow_nan=False))
        return 1
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    from importlib import import_module

    raise SystemExit(import_module(MODULE).main())
