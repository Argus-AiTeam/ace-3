"""Q24 scratch-state RMSNorm contrast; CPU-only, nine controls, no admission.

The fixed fresh output must be provisioned by Manager before --check. Consumed
FP16-input contrast attempts are never used as writable execution roots.
"""

import argparse
from contextlib import contextmanager
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from ace3.model.candidates import q24_s16_l23_p0_value_content_stage13_rmsnorm_integer_contrast_v1 as base


retained, prior, layer, native = base.retained, base.prior, base.layer, base.native
require, pin = base.require, base.pin
ROOT = base.ROOT
SOURCE = Path(__file__).resolve()
NAME = SOURCE.stem
TEST = ROOT / f"tests/test_{NAME}.py"
MISSION = "9f7d34f11f40"
OUTPUT = ROOT / "build" / (NAME + "-" + MISSION + "-attempt001")
CONTROLS, PAIR, PREFIX = base.CONTROLS, base.PAIR, base.PREFIX
BOUNDARY = (
    "CPU-only width896/L23/P0/pair[34319,13], nine retained controls, non-admission. "
    "S13 uses authenticated scratch_i/scratch_z Q24 state and FP16 norm weights, "
    "not rounded S12 words or reference-selected output bits. Only S13-S18, final "
    "integer RMSNorm and two tied-head rows execute. S10-S12 gates are checked "
    "without replay. Original-input FP16/binary64 references are comparators only. "
    "Retained value-substitution ancestry is not native-V repair. Native S16 RTZ, "
    "G128 asymmetric INT4/GEMM nibble ordering/no qzero plus-one, FP16 scales, "
    "operator boundaries/KV, wider Q24 residuals and all existing gates remain. "
    "No prefix/admission/reference/closed-producer replay; no strict-FP16-state "
    "W4A16, new-token, full-model, hardware or scale-expansion claim."
)


def project_sources():
    return base.project_sources() | {str(SOURCE), str(TEST)}


@contextmanager
def guarded(out, audit):
    require(prior._active is None and sys.getprofile() is None, "nested execution guard")
    if not prior._hook_installed:
        sys.addaudithook(prior.audit_event)
        prior._hook_installed = True
    allowed = project_sources()
    forbidden = {
        "check", "main", "execute", "run", "diagnose", "stages", "_stages",
        "continuation_stages", "native_layer", "execute_layer", "execute_layers",
        "original_branches", "reference_arrays", "build_report", "intervene",
        "adjust", "nearest", "finite_table", "run_control", "guarded", "suffix_stages",
        "evaluate_actual_rtl_result", "evaluate_p0_transaction",
    }

    def profile(frame, event, arg):
        if event != "call":
            return
        file, name = frame.f_code.co_filename, frame.f_code.co_name
        if not file.startswith(str(ROOT) + "/"):
            return
        key = file.removeprefix(str(ROOT) + "/") + ":" + name
        audit["project_calls"][key] = audit["project_calls"].get(key, 0) + 1
        denied = file not in allowed or (file != str(SOURCE) and name in forbidden)
        if file == str(Path(native.__file__).resolve()):
            denied |= name not in {"projection", "decoded", "rne", "toward_zero", "<genexpr>"}
            if name == "projection":
                denied |= frame.f_locals["name"] not in {
                    PREFIX + "mlp." + s for s in ("gate_proj", "up_proj", "down_proj")}
        if name in {"expected_stage", "stage_report", "local_reference", "check_stage_state",
                    "evaluate_decoder_stage"}:
            denied |= frame.f_locals.get("stage") not in range(10, 19)
        if denied:
            audit["forbidden_events"].append(key)
            raise retained.bridge.BindingError("unapproved project dispatch: " + key)

    prior._active = (out, audit)
    sys.setprofile(profile)
    try:
        yield
    finally:
        sys.setprofile(None)
        prior._active = None


def stage13(integers, zeros, gamma):
    require(integers.shape == zeros.shape == gamma.shape == (896,)
            and integers.dtype.str == "<i8" and zeros.dtype.str == "|u1"
            and gamma.dtype.str == "<f2" and np.all(np.isfinite(gamma)),
            "Q24 stage13 entry width/dtype/weight")
    values = [int(value) for value in integers]
    for value, zero in zip(values, zeros, strict=True):
        prior.state.fp16_view(value, int(zero))
    norm = retained.fp16_adaptation_oracle
    weights = [norm.decode_f16_q24(int(word)) for word in gamma.view("<u2")]
    require(all(weight[1] for weight in weights), "invalid RMSNorm weight")
    mean = norm.round_div_even_unsigned(sum(x * x for x in values) + 896 * 281474977, 896)
    root = math.isqrt(mean)
    require(root > 0, "positive RMSNorm epsilon produced zero root")
    words = []
    for value, zero, weight in zip(values, zeros, weights, strict=True):
        product = value * weight[0]
        magnitude = norm.round_div_even_unsigned(abs(product), root)
        integer = -magnitude if product < 0 else magnitude
        sign = (int(value < 0) if value else int(zero)) ^ weight[3]
        word, saturated = norm.q24_to_f16(integer, zero_sign=sign)
        require(not saturated, "saturated Q24-state RMSNorm")
        words.append(word)
    return np.asarray(words, dtype="<u2"), {"mean_q48": mean, "root_q24": root}


def compute_suffix(raw, tensors, operands):
    """Output bits depend only on native state, weights and arithmetic."""
    prior.check_state(raw)
    arrays, timings = dict(raw), {}
    began = time.monotonic()
    arrays["stage13"], detail = stage13(
        raw["scratch_i"], raw["scratch_z"], tensors[PREFIX + "post_attention_layernorm.weight"])
    timings["stage13"] = time.monotonic() - began
    for stage, name, source in ((14, "gate_proj", 13), (15, "up_proj", 13),
                                (16, None, None), (17, "down_proj", 16), (18, None, None)):
        began = time.monotonic()
        if stage in (14, 15, 17):
            words = native.projection(tensors, PREFIX + "mlp." + name, arrays[f"stage{source:02d}"])
        elif stage == 16:
            value = (native.functional.silu(native.decoded(arrays["stage14"]))
                     * native.decoded(arrays["stage15"]))
            arrays["s16_unrounded_binary64"] = value.numpy().copy()
            words = native.toward_zero(arrays["s16_unrounded_binary64"])
        else:
            state = native.state.add(
                {"i": raw["scratch_i"], "z": raw["scratch_z"], "h": raw["stage12"]},
                arrays["stage17"])
            arrays["output_i"], arrays["output_z"] = state["i"], state["z"]
            words = state["h"]
        arrays[f"stage{stage:02d}"] = words
        timings[f"stage{stage}"] = time.monotonic() - began
    began = time.monotonic()
    arrays["final_rmsnorm"], final_detail = retained.suffix.rmsnorm(arrays["stage18"], operands[0])
    timings["final_rmsnorm"] = time.monotonic() - began
    began = time.monotonic()
    arrays["final_logits"] = retained.suffix.logits(arrays["final_rmsnorm"], operands[1])
    timings["pair_head"] = time.monotonic() - began
    prior.check_state(arrays)
    return arrays, detail, final_detail, timings


def evaluate_control(control, raw, old, classified, tensors, trajectory, reference,
                     refs, operands, oracle, audit):
    require(control == old["control"] == classified["control"], "retained control lineage splice")
    protected = {k: retained.digest(v) for k, v in raw.items()}
    arrays, detail, final_detail, timings = compute_suffix(raw, tensors, operands)
    audit["controls"].append(control)
    audit["stage_dispatches"].extend([[control, 23, 0, s] for s in range(13, 19)])
    audit["final_rmsnorm_invocations"] += 1
    audit["pair_head_invocations"] += 1
    began = time.monotonic()
    oracle.verify_norm(raw["scratch_i"], raw["scratch_z"],
                       tensors[PREFIX + "post_attention_layernorm.weight"], arrays["stage13"], detail)
    oracle.suffix_oracle.verify_suffix(
        arrays["stage18"], operands, arrays["final_rmsnorm"], final_detail, arrays["final_logits"])
    require(np.array_equal(layer.prior.rtz_reference(arrays["s16_unrounded_binary64"]),
                           arrays["stage16"]), "native S16 RTZ mismatch")
    reports = []
    parent = {"h": raw["input_hidden"], "i": raw["input_i"], "z": raw["input_z"]}
    for stage in range(10, 19):
        expected = layer.expected_stage(stage, arrays, parent, tensors)
        reports.append(layer.stage_report(stage, arrays, trajectory, reference, expected))
        if expected is not None:
            arrays[f"local_reference_stage{stage:02d}"] = expected
        audit["stage_gate_checks"].append([control, 23, 0, stage])
    before, after = prior.compare(raw["stage18"], reference), prior.compare(arrays["stage18"], reference)
    require(before["failure_count"] == old["failure_count"], "retained S18 comparator splice")
    require(after["status"] == reports[-1]["status"], "S18 gate comparator disagreement")
    margins = retained.paired_margins(arrays["final_logits"], raw["final_logits"],
                                     classified["paired_margins"], refs)
    require(protected == {k: retained.digest(v) for k, v in raw.items()}, "retained bytes changed")
    for key in tuple(f"stage{s:02d}" for s in range(13)) + (
            "input_hidden", "input_i", "input_z", "scratch_i", "scratch_z",
            "input_cache_k", "input_cache_v", "output_cache_k", "output_cache_v"):
        require(retained.digest(arrays[key]) == protected[key], "protected prefix/state/KV: " + key)
    timings["oracles_and_gates"] = time.monotonic() - began
    changed = {k: v for k, v in arrays.items() if k not in raw or v is not raw[k]}
    return {**raw, **{"q24_state_contrast_" + k: v for k, v in changed.items()}}, {
        "control": control, "stages": reports, "S18_before": before, "S18": after,
        "paired_margins": margins, "protected_arrays": protected, "protected_bytes_unchanged": True,
        "component_oracles": {"status": "PASS", "errors": []},
        "stage13_integer_oracle": {
            **detail, "epsilon_q48": 281474977, "width": 896,
            "semantics": "RNE mean Q48; floor integer sqrt; signed magnitude RNE division Q24; FP16 RNE",
            "input_fields": ["scratch_i", "scratch_z", PREFIX + "post_attention_layernorm.weight"],
            "reference_selected_output_words": False,
            "changed_words": int(np.count_nonzero(arrays["stage13"] != raw["stage13"]))},
        "final_rmsnorm_integer_details": final_detail, "timing_seconds": timings}


def dispatch_census(audit):
    def count(module, function):
        key = str(Path(module.__file__).resolve().relative_to(ROOT)) + ":" + function
        return audit["project_calls"].get(key, 0)
    require(audit["forbidden_events"] == [] and audit["controls"] == list(CONTROLS)
            and audit["stage_dispatches"] == [[c, 23, 0, s] for c in CONTROLS for s in range(13, 19)]
            and audit["stage_gate_checks"] == [[c, 23, 0, s] for c in CONTROLS for s in range(10, 19)]
            and audit["final_rmsnorm_invocations"] == audit["pair_head_invocations"] == 9,
            "ordered dispatch/gate census defect")
    own = str(SOURCE.relative_to(ROOT)) + ":stage13"
    require(audit["project_calls"].get(own, 0) == 9 and count(native, "projection") == 27
            and count(native, "toward_zero") == 9 and count(native.state, "add") == 9
            and count(retained.fp16_adaptation_oracle, "rmsnorm") == 9,
            "observed arithmetic dispatch census defect")
    require(audit["writes"] == [str(OUTPUT / (c + ".npz")) for c in CONTROLS], "write census defect")


def save(name, value):
    with (OUTPUT / name).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def check():
    require(Path.cwd() == ROOT and os.getuid() == 1000 and sys.dont_write_bytecode
            and not sys.flags.optimize, "isolated source/account/bytecode context")
    require(subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
            == "argus/full-projection", "isolated branch")
    require(subprocess.run(["git", "check-ignore", "--quiet", str(OUTPUT)]).returncode == 0,
            "output must be ignored")
    require(OUTPUT.resolve() == OUTPUT and OUTPUT.is_dir()
            and {p.name for p in OUTPUT.iterdir()} == {"command.sh"},
            "fresh Manager-provisioned output missing or occupied")
    command = (OUTPUT / "command.sh").read_text()
    require(command == f"python3 -B -m ace3.model.candidates.{NAME} --check\n", "command sidecar mismatch")
    mission_path = retained.HANDOFFS / MISSION / "mission.json"
    mission = json.loads(mission_path.read_bytes())
    require(mission["mission_id"] == MISSION and mission["scope"] == "bounded"
            and mission["node_key"] == "stage13-q24-state-coherent-execution"
            and mission["execution_workdir"] == str(ROOT) and mission["deps"] == []
            and "review:required" in mission["tags"], "current science owner/dependency preflight")
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec")
    oracle = retained.module_from(TEST, NAME + "_oracle")
    sources = [pin(Path(p)) for p in sorted(project_sources())]
    save("command.json", {
        "mission_id": MISSION, "mission": pin(mission_path), "command_bytes": command,
        "argv": sys.argv, "cwd": str(ROOT), "uid": os.getuid(), "pid": os.getpid(),
        "executable": sys.executable, "timestamp": time.time(), "science_budget": 1,
        "authorization_id": "", "action": "", "source_pins": sources,
        "claim": "exclusive command.json; fixed fresh root; no retries or alternate roots"})
    audit, rows = base.new_audit(), []
    result = {
        "mission_id": MISSION, "status": "UNKNOWN", "controls": rows, "audit": audit,
        "science_budget": 1, "science_invocations": 0, "non_admission": True,
        "normal_independent_review": "REQUIRED", "automatic_replay_authorized": False,
        "boundary": BOUNDARY, "flags": base.closure_source.FLAGS, "source_pins": sources}
    try:
        native.torch.set_num_threads(1)
        with guarded(OUTPUT, audit):
            closure, failure, bridge, ref, refs, operands, tensors, trajectory, records = base.inputs()
            records.extend([pin(mission_path), pin(OUTPUT / "command.sh")])
            result.update(input_pins=records, thresholds=closure["thresholds"],
                          historical_failure=closure["historical_failure"],
                          historical_science_status=closure["science_status"],
                          operand_pins=[retained.digest(v) for v in operands],
                          tensor_pins={k: retained.digest(v) for k, v in tensors.items()})
            prepared = []
            for document in (failure, closure, bridge):
                require([r["control"] for r in document["controls"]] == list(CONTROLS),
                        "retained control order")
            for row in failure["controls"]:
                with np.load(io.BytesIO(retained.selection.bound(row["raw_payload"])),
                             allow_pickle=False) as archive:
                    require(len(archive.files) == len(set(archive.files)), "duplicate archive keys")
                    arrays = {k: archive[k] for k in archive.files}
                for array in arrays.values():
                    array.flags.writeable = False
                prior.check_state(arrays)
                prepared.append(arrays)
            result["science_invocations"] = 1
            for control, raw, old, classified in zip(
                    CONTROLS, prepared, closure["controls"], bridge["controls"], strict=True):
                payload, row = evaluate_control(
                    control, raw, old, classified, tensors, trajectory, ref, refs, operands, oracle, audit)
                path = OUTPUT / (control + ".npz")
                with path.open("xb") as stream:
                    np.savez(stream, **payload)
                row["payload"] = pin(path)
                rows.append(row)
            for record in [*records, *sources]:
                retained.selection.bound(record)
            require(result["operand_pins"] == [retained.digest(v) for v in operands]
                    and result["tensor_pins"] == {k: retained.digest(v) for k, v in tensors.items()},
                    "authenticated operands changed")
            dispatch_census(audit)
            result["status"] = base.scientific_status(rows)
            result["protected_input_files_unchanged"] = True
    except Exception as error:
        # A consumed integrity/execution defect is UNKNOWN, never numerical rejection.
        result.update(status="UNKNOWN", error_type=type(error).__name__, error=str(error))
    result["evidence_write_census"] = sorted(p.name for p in OUTPUT.iterdir()) + ["result.json"]
    save("result.json", result)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 1 if result["status"] == "UNKNOWN" else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", required=True, action="store_true")
    parser.parse_args()
    return check()


if __name__ == "__main__":
    raise SystemExit(main())
