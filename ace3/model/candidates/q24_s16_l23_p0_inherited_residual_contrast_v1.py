"""One non-admitting L23/P0 inherited-residual substitution at S12."""

import argparse
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time
import unittest

import numpy as np

from ace3.model.candidates import q24_s16_l23_p0_value_content_sufficiency_v1 as base


ROOT = base.ROOT
NAME = "q24_s16_l23_p0_inherited_residual_contrast_v1"
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
OUTPUT = ROOT / "build" / (NAME + "_attempt001")
CONTROLS = base.CONTROLS
final, layer, native = base.final, base.layer, base.native
require, same, record, read_bound = base.require, base.same, base.record, base.read_bound
STAGES = tuple(range(12, 19))
AFFECTED = frozenset(
    [f"stage{s:02d}" for s in STAGES]
    + ["scratch_i", "scratch_z", "output_i", "output_z", "s16_unrounded_binary64"])
PREDECESSOR = {
    "path": str(layer.preflight.HANDOFFS / "b0966ee0594c" / "round-0001.json"),
    "sha256": "619862e428ae412aadccec999c6c29e9bbb645759b03fe0489f518fb2f0ab15f",
    "bytes": 1091,
}
VALUE_RESULT = {
    "path": str(base.OUTPUT / "result.json"),
    "sha256": "2d22d249491843cd2e6b31dfb40cb8c2770438b778ddb4a61d47345654e61eed",
    "bytes": 24290175,
}
FLAGS = {
    **base.FLAGS, "stage07_native_operator_validated": False,
    "inherited_residual_native_lineage_validated": False,
    "attention_recomputed": False,
}
BOUNDARY = (
    "Conditional rescue test of nine fresh S12-S18 outputs only. Substitute the "
    "original-input FP16 inherited residual, exactly lifted to paired Q24 I/Z/H, "
    "only at the S12 residual-add operand. Retain actual input I/Z/H and S0-S11, "
    "including V/Q/K/probabilities/attention output and actual FP16 KV. Recompute "
    "S12-S18 and final RMSNorm/tied-head only. Final comparisons are diagnostic; "
    "there is no existing final-head admission threshold. Historical L0-L23 and "
    "value-intervention FAIL remain unchanged. Q24 state is wider than FP16; "
    "native INT4 and FP16 operator/KV boundaries remain unchanged. No original "
    "prefix, admission or reference replay; no unique-cause, strict-FP16-state "
    "W4A16, new-token, full-model, hardware, GPU or RTL claim."
)


def predecessor_review():
    review = json.loads(read_bound(PREDECESSOR))
    mission = "b0966ee0594c"
    directory = layer.preflight.HANDOFFS / mission
    latest = json.loads((directory / "latest.json").read_bytes())
    require(latest["kind"] == "handoff_ref"
            and latest["handoff"]["path"] == PREDECESSOR["path"],
            "value-intervention review identity changed")
    layer.preflight.parent.matrix.check_review(review, mission, 1)
    rows = [json.loads(line) for line in layer.preflight.BACKLOG.read_text().splitlines()
            if line.strip()]
    rows = [row for row in rows if row["id"] == mission]
    require(len(rows) == 1 and rows[0]["status"] == "done"
            and rows[0]["outcome"]["review_status"] == "done"
            and rows[0]["finished_ts"] >= review["created_at"],
            "value-intervention review lacks terminal corroboration")
    result = json.loads(read_bound(VALUE_RESULT))
    require(result["diagnostic_id"] == base.NAME and result["status"] == "FAIL"
            and result["output"] == str(base.OUTPUT),
            "reviewed value-intervention result changed")
    same([row["control"] for row in result["controls"]], list(CONTROLS),
         "value-intervention control identity changed")
    same(record(base.SOURCE), result["origins"]["value_sufficiency_source"],
         "reviewed shared helper source changed")
    return {"review": PREDECESSOR, "result": VALUE_RESULT}


def prepare(actual, reference):
    # Reuse the retained P0 source/probability/KV gates, not the value substitution.
    actual_parent, _ = base.prepare(actual, reference)
    layer.prior.retained.verify_parent(actual_parent, actual_parent)
    layer.prior.retained.check_stage_state(18, actual, actual_parent)
    words = reference["input_hidden"]
    layer.prior.local.finite_words(words, (896,))
    parent = native.state.lift(words)
    layer.prior.retained.verify_parent(parent, parent, embedding=words)
    arrays = {key: value.copy() for key, value in actual.items() if key not in AFFECTED}
    for key, value in parent.items():
        arrays["residual_operand_" + key] = value.copy()
    for value in (*arrays.values(), *parent.values()):
        value.flags.writeable = False
    return parent, arrays


def protected(actual, arrays, reference, parent):
    checks = {}
    for key in actual.keys() - AFFECTED:
        checks[key] = (actual[key].dtype == arrays[key].dtype
                       and actual[key].shape == arrays[key].shape
                       and actual[key].tobytes() == arrays[key].tobytes())
        require(checks[key], "protected field changed: " + key)
    layer.prior.retained.verify_parent(parent, parent, embedding=reference["input_hidden"])
    for key in ("i", "z", "h"):
        same(arrays["residual_operand_" + key].dtype.str, parent[key].dtype.str,
             "residual operand dtype changed")
        require(np.array_equal(arrays["residual_operand_" + key], parent[key]),
                "substituted residual operand changed")
    return checks


def suffix_stages(tensors, parent, arrays):
    prefix = "model.layers.23."
    scratch = None
    for stage in STAGES:
        if stage == 12:
            scratch = native.state.add(parent, arrays["stage11"])
            arrays["scratch_i"], arrays["scratch_z"] = scratch["i"], scratch["z"]
            word = scratch["h"]
        elif stage == 13:
            value = native.decoded(arrays["stage12"])
            gamma = native.torch.from_numpy(
                tensors[prefix + "post_attention_layernorm.weight"].astype("<f8"))
            word = native.rne(
                (value * native.torch.rsqrt(value.square().mean() + 1e-6)) * gamma)
        elif stage in (14, 15, 17):
            name, source = {14: ("gate_proj", 13), 15: ("up_proj", 13),
                            17: ("down_proj", 16)}[stage]
            word = native.projection(tensors, prefix + "mlp." + name,
                                     arrays[f"stage{source:02d}"])
        elif stage == 16:
            value = (native.functional.silu(native.decoded(arrays["stage14"]))
                     * native.decoded(arrays["stage15"]))
            arrays["s16_unrounded_binary64"] = value.numpy().copy()
            word = native.toward_zero(arrays["s16_unrounded_binary64"])
        else:
            require(scratch is not None, "missing intervened residual scratch")
            successor = native.state.add(scratch, arrays["stage17"])
            arrays["output_i"], arrays["output_z"] = successor["i"], successor["z"]
            word = successor["h"]
        arrays[f"stage{stage:02d}"] = word
        yield stage


def expected_stage(stage, arrays, parent, tensors):
    # The oracle checks the explicitly intervened operand; retained inputs stay raw.
    oracle_arrays = {**arrays, "input_hidden": parent["h"],
                     "input_i": parent["i"], "input_z": parent["z"]}
    return layer.expected_stage(stage, oracle_arrays, parent, tensors)


def scientific_status(rows):
    same([row["control"] for row in rows], list(CONTROLS), "incomplete/reordered controls")
    for row in rows:
        same([r["stage"] for r in row["reports"]], list(STAGES), "incomplete affected cone")
        require(all(r["status"] in ("PASS", "FAIL") for r in row["reports"]),
                "invalid acceptance result")
        require(row["preservation_checks"] and all(row["preservation_checks"].values()),
                "retained-state protection defect")
    return "PASS" if all(r["status"] == "PASS" for row in rows for r in row["reports"]) else "FAIL"


def focused_tests():
    spec = importlib.util.spec_from_file_location(NAME + "_tests", TEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    require(result.wasSuccessful() and result.testsRun == 17 and not result.skipped,
            "focused tests failed:\n" + output.getvalue())
    return {"executed": result.testsRun, "failures": 0, "errors": 0, "skipped": 0,
            "retained_control_dispatches": 0, "raw_output": output.getvalue(),
            "type_checks": "runtime dtype/shape and paired-state schema gates"}


def execute():
    require(not OUTPUT.exists() and not OUTPUT.is_symlink(), "occupied output; zero dispatch")
    require(OUTPUT.resolve() == OUTPUT and OUTPUT.parent == ROOT / "build"
            and Path(__file__).resolve() == SOURCE, "noncanonical source/output")
    ignored = final.subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "--quiet", "--", str(OUTPUT)],
        capture_output=True, check=False)
    require(ignored.returncode == 0, "output not confirmed ignored")
    native.torch.set_num_threads(1)
    require(str(native.torch.tensor(0).device) == "cpu", "non-CPU default device")
    origins = final.source_context()
    origins["inherited_residual_source"], origins["inherited_residual_test"] = record(SOURCE), record(TEST)
    OUTPUT.mkdir(exist_ok=False)
    audit = {"controls": [], "stage_dispatches": [], "final_rmsnorm_invocations": 0,
             "lm_head_invocations": 0, "top_k_invocations": 0, "forbidden_calls": 0,
             "file_write_opens": 0, "native_L0_L22_invocations": 0,
             "native_S0_S11_invocations": 0, "retained_evidence_writes": 0,
             "reference_producer_invocations": 0, "external_invocations": 0}
    rows, artifacts = [], []

    def save_json(name, payload):
        with (OUTPUT / name).open("x") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        return record(OUTPUT / name)

    try:
        with base.write_scope(OUTPUT, audit):
            artifacts.append(save_json("command.json", {
                "argv": sys.argv, "cwd": str(ROOT), "uid": os.getuid(),
                "executable": sys.executable, "origins": origins,
                "host_managed_controls": "unchanged role/model/account/budget/access/concurrency locks",
            }))
            compiled = []
            for pin in {p["path"]: p for p in origins.values()}.values():
                compile(read_bound(pin), pin["path"], "exec")
                compiled.append(pin)
            tests = focused_tests()
            tests["compiled"] = compiled
            artifacts.append(save_json("compile_tests.json", tests))
            predecessor = predecessor_review()
            started = time.monotonic()
            baseline, baseline_arrays, references, pins = base.retained.authenticate()
            summary = baseline["preflight"]
            tensors, trajectory, reference, checkpoint = layer.load_inputs({"summary": summary})
            operands = final.load_operands(summary)
            same(final.preflight.bind_assets(checkpoint), summary["assets"], "asset identity changed")
            require(summary["L23_original_reference"]["reference"]["prior_kv"] == "own empty P0",
                    "reference KV/source scope changed")
            same(summary["L23_original_reference"]["reference"]["fp16"],
                 summary["final_reference"]["reference"]["input_fp16"],
                 "reference input lineage splice")
            for tensor in (*tensors.values(), *trajectory.values(), *references.values()):
                tensor.flags.writeable = False
            prepared = []
            for row in baseline["controls"]:
                actual = base.archive(row["parent"]["terminal_archive"])
                parent, arrays = prepare(actual, trajectory)
                prepared.append((row, actual, parent, arrays))
            same([r["control"] for r, _, _, _ in prepared], list(CONTROLS),
                 "nine-control source order changed")
            authentication_seconds = time.monotonic() - started
            with final.no_dispatch(audit):
                for row, actual, parent, arrays in prepared:
                    label = row["control"]
                    audit["controls"].append(label)
                    reports, timings = [], {"native_seconds": 0.0, "oracle_seconds": 0.0}
                    stages = suffix_stages(tensors, parent, arrays)
                    for expected in STAGES:
                        began = time.monotonic()
                        audit["stage_dispatches"].append([label, 23, 0, expected])
                        stage = next(stages)
                        require(stage == expected, "out-of-cone dispatch")
                        timings["native_seconds"] += time.monotonic() - began
                        began = time.monotonic()
                        oracle = expected_stage(stage, arrays, parent, tensors)
                        report = layer.stage_report(stage, arrays, trajectory, reference, oracle)
                        report["residual_state_lineage"] = "PASS_explicit_S12_operand_intervention"
                        reports.append(report)
                        if oracle is not None:
                            arrays[f"local_reference_stage{stage:02d}"] = oracle
                        timings["oracle_seconds"] += time.monotonic() - began
                    require(next(stages, None) is None, "extra suffix stage")
                    checks = protected(actual, arrays, trajectory, parent)
                    require(np.array_equal(layer.prior.rtz_reference(arrays["s16_unrounded_binary64"]),
                                           arrays["stage16"]), "S16 RTZ boundary changed")
                    began = time.monotonic()
                    audit["final_rmsnorm_invocations"] += 1
                    arrays["final_rmsnorm"], details = final.rmsnorm(arrays["stage18"], operands[0])
                    timings["final_rmsnorm_seconds"] = time.monotonic() - began
                    began = time.monotonic()
                    audit["lm_head_invocations"] += 1
                    arrays["final_logits"] = final.logits(arrays["final_rmsnorm"], operands[1])
                    timings["lm_head_seconds"] = time.monotonic() - began
                    began = time.monotonic()
                    metrics = final.comparisons(
                        {"rmsnorm": arrays["final_rmsnorm"], "logits": arrays["final_logits"]}, references)
                    audit["top_k_invocations"] += 3
                    margins = base.margin_rows(
                        baseline_arrays[label]["logits"], arrays["final_logits"], references)
                    timings["final_comparison_seconds"] = time.monotonic() - began
                    payload = OUTPUT / f"{label}.npz"
                    with payload.open("xb") as stream:
                        np.savez(stream, **arrays)
                    pin = record(payload)
                    artifacts.append(pin)
                    entry = {
                        "control": label, "reports": reports, "final_comparisons": metrics,
                        "paired_final_margins": margins, "parent": row["parent"], "raw_payload": pin,
                        "preservation_checks": checks,
                        "substitution": {
                            "source": summary["L23_original_reference"]["reference"]["fp16"],
                            "field": "input_hidden", "shape": [896], "dtype": "<u2",
                            "boundary": "S12 inherited residual operand only, then own scratch at S18",
                            "representation": "exact FP16 lift to paired Q24 I/Z/H; no residual rounding",
                            "changed_fp16_components": int(np.count_nonzero(actual["input_hidden"] != parent["h"])),
                            "changed_q24_components": int(np.count_nonzero(actual["input_i"] != parent["i"])),
                            "retained_KV_unchanged": True, "retained_input_state_unchanged": True,
                            "attention_S0_S11_unchanged": True,
                        },
                        "rmsnorm_integer_details": details, "timing_seconds": timings,
                        "status": "PASS" if all(r["status"] == "PASS" for r in reports) else "FAIL",
                    }
                    rows.append(entry)
                    print(json.dumps({k: entry[k] for k in
                                      ("control", "status", "final_comparisons", "paired_final_margins",
                                       "preservation_checks", "substitution", "timing_seconds")}), flush=True)
            same(final.source_context(), {k: v for k, v in origins.items()
                                          if k not in ("inherited_residual_source", "inherited_residual_test")},
                 "source changed during execution")
            for pin in [*pins, PREDECESSOR, VALUE_RESULT, origins["inherited_residual_source"],
                        origins["inherited_residual_test"]]:
                read_bound(pin)
            status = scientific_status(rows)
            require(len(audit["stage_dispatches"]) == 63
                    and audit["final_rmsnorm_invocations"] == audit["lm_head_invocations"] == 9
                    and audit["top_k_invocations"] == 27
                    and audit["forbidden_calls"] == 0 and audit["file_write_opens"] == 11,
                    "dispatch/write census mismatch")
            result = {
                "diagnostic_id": NAME, "status": status, "controls": rows, "origins": origins,
                "predecessor_review": predecessor, "input_pins": pins, "preflight": summary,
                "thresholds": summary["thresholds"], "final_reference": summary["final_reference"],
                "audit": audit, "artifacts": artifacts, "output": str(OUTPUT),
                "expected_final_file_count": 12, "authentication_seconds": authentication_seconds,
                "flags": FLAGS, "claim_boundary": BOUNDARY, "normal_host_review": "REQUIRED",
                "decision": ("conditional all-nine inherited-residual rescue; not unique cause"
                             if status == "PASS" else
                             "reject all-nine inherited-residual rescue; no causal irrelevance claim"),
            }
            save_json("result.json", result)
            require(len(list(OUTPUT.iterdir())) == 12 and audit["file_write_opens"] == 12,
                    "final write census mismatch")
            print(json.dumps({"status": status, "output": str(OUTPUT), "audit": audit,
                              "decision": result["decision"], "flags": FLAGS,
                              "claim_boundary": BOUNDARY}), flush=True)
    except Exception as error:
        with base.write_scope(OUTPUT, audit):
            save_json("failure.json", {"status": "UNKNOWN", "error_type": type(error).__name__,
                                      "error": str(error), "audit": audit, "controls": rows,
                                      "flags": FLAGS, "automatic_replay_authorized": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    execute()


if __name__ == "__main__":
    from importlib import import_module

    import_module("ace3.model.candidates." + NAME).main()
