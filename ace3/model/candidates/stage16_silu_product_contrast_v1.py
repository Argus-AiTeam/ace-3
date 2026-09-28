"""Native accurate Q24 SiLU/product with the unchanged S16 FP16 RTZ boundary.

CPU-only width896/L23/P0/pair[34319,13], non-admission. The polynomial,
sigmoid and product rounding are the repository's accurate integer semantics;
only the final FP16 conversion uses native RTZ rather than accurate-path RNE.
References never enter construction. Q24 residuals remain wider than FP16.
This library cannot claim an output or authorize a retained-data invocation.
"""

from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time

import numpy as np

from ace3.model.candidates import stage17_q24_down_projection_contrast_v1 as parent


base, retained, prior = parent.base, parent.retained, parent.prior
native, layer, require, pin = parent.native, parent.layer, parent.require, parent.pin
ROOT, PREFIX, CONTROLS, PAIR = parent.ROOT, parent.PREFIX, parent.CONTROLS, parent.PAIR
SOURCE = Path(__file__).resolve()
TEST = ROOT / f"tests/test_{SOURCE.stem}.py"
SOURCE_PATHS = frozenset(
    (parent.project_sources() | {str(SOURCE), str(TEST)}) - {
        str(SOURCE.with_name("stage16_silu_product_execution_v1.py")),
        str(ROOT / "tests/test_stage16_silu_product_execution_v1.py"),
    })
PROTECTED = tuple(f"stage{s:02d}" for s in range(16)) + parent.PROTECTED[17:-1]
STAGE17_MISSION = "a564f7e7a160"
RESULT_PIN = {
    "path": str(ROOT / "build" /
                f"stage17_q24_down_projection_execution_v1-{STAGE17_MISSION}-attempt001/result.json"),
    "sha256": "ca96cd25bed8ea91f46a1a2b5c1155ada66f090f0f23cdd6de34d75e4551347b",
}
REVIEW_PIN = {
    "path": str(retained.HANDOFFS / STAGE17_MISSION / "round-0001.json"),
    "sha256": "a5028951cae206f79cb15179d8fba982b921f1da1e42be4ac397416da35b42ce",
}
BOUNDARY = (
    "Native accurate integer SiLU/product then S16 RTZ; S17 exact G128 INT4 "
    "down projection, S18 wider Q24 residual, final RMSNorm and two tied rows. "
    "CPU-only width896/L23/P0/pair[34319,13], nine controls, non-admission. "
    "Native nibble order, no qzero plus-one, FP16 scales/operators/KV. "
    "Original references are comparators only. No prefix/admission/reference/"
    "closed-producer replay, output patching, GPU/RTL/hardware, strict-FP16-state "
    "W4A16, new-token or full-model claim. Retained substitution ancestry is "
    "not native-V repair."
)


def project_sources():
    return set(SOURCE_PATHS)


def stage16(gate, up):
    for words in (gate, up):
        layer.prior.local.finite_words(words, (4864,))
    arithmetic = retained.fp16_adaptation_oracle
    values = []
    for g, u in zip(gate, up, strict=True):
        gi = arithmetic.decode_f16_q24(int(g))[0]
        ui = arithmetic.decode_f16_q24(int(u))[0]
        exponential = arithmetic._exp_negative_q24(abs(gi))
        negative = arithmetic.round_div_even_unsigned(
            exponential << 24, arithmetic.Q24_ONE + exponential)
        sigmoid = negative if gi < 0 else arithmetic.Q24_ONE - negative
        product = arithmetic.round_shift_even_signed(gi * ui * sigmoid, 48)
        value = product / (1 << 24)
        if product == 0 and (int(g) ^ int(u)) & 0x8000:
            value = -0.0
        values.append(value)
    unrounded = np.asarray(values, dtype="<f8")
    return native.toward_zero(unrounded), unrounded


def compute_suffix(raw, tensors, operands):
    parent.validate_state(raw)
    arrays, timings = dict(raw), {}
    began = time.monotonic()
    arrays["stage16"], arrays["s16_unrounded_binary64"] = stage16(
        raw["stage14"], raw["stage15"])
    timings["stage16"] = time.monotonic() - began
    began = time.monotonic()
    arrays["stage17"] = parent.projection(tensors, arrays["stage16"])
    timings["stage17"] = time.monotonic() - began
    began = time.monotonic()
    state = native.state.add(
        {"i": raw["scratch_i"], "z": raw["scratch_z"], "h": raw["stage12"]},
        arrays["stage17"])
    arrays["output_i"], arrays["output_z"], arrays["stage18"] = state["i"], state["z"], state["h"]
    timings["stage18"] = time.monotonic() - began
    began = time.monotonic()
    arrays["final_rmsnorm"], detail = retained.suffix.rmsnorm(arrays["stage18"], operands[0])
    timings["final_rmsnorm"] = time.monotonic() - began
    began = time.monotonic()
    arrays["final_logits"] = retained.suffix.logits(arrays["final_rmsnorm"], operands[1])
    timings["pair_head"] = time.monotonic() - began
    parent.validate_state(arrays)
    return arrays, detail, timings


def inputs():
    closure, bridge, reference, refs, operands, tensors, trajectory, records, prepared = parent.inputs()
    document = json.loads(parent.parent.historical_record(RESULT_PIN))
    review = json.loads(parent.parent.historical_record(REVIEW_PIN))
    require(review["mission_id"] == STAGE17_MISSION and review["producer_role"] == "reviewer"
            and review["review"]["status"] == "done", "independent Stage17 review")
    require(document["mission_id"] == STAGE17_MISSION
            and document["status"] == parent.parent.scientific_status(document["controls"])
            and document["status"] == "REJECTED"
            and type(document["science_invocations"]) is int
            and document["science_invocations"] == document["science_budget"] == 1
            and document["non_admission"] is True
            and document["audit"]["forbidden_events"] == []
            and document["protected_input_files_unchanged"] is True
            and document["thresholds"] == closure["thresholds"],
            "accepted Stage17 identity/budget/thresholds")
    require(document["operand_pins"] == [retained.digest(v) for v in operands]
            and document["tensor_pins"] == {k: retained.digest(v) for k, v in tensors.items()},
            "Stage17 operand lineage")
    require([row["control"] for row in document["controls"]] == list(CONTROLS),
            "Stage17 ordered controls")
    current = []
    for control, row, (original, raw) in zip(CONTROLS, document["controls"], prepared, strict=True):
        require(row["payload"]["path"] == str(Path(RESULT_PIN["path"]).parent / (control + ".npz")),
                "Stage17 payload splice")
        require(row["protected_arrays"] == {k: retained.digest(v) for k, v in raw.items()},
                "Stage14/15 to Stage17 lineage")
        actual = parent.archive(row["payload"])
        parent.validate_state(actual)
        for key in parent.PROTECTED:
            require(retained.digest(actual[key]) == retained.digest(raw[key]),
                    "Stage17 protected input: " + key)
        current.append((original, actual))
    records.extend([RESULT_PIN, REVIEW_PIN, *document["source_pins"], *document["input_pins"],
                    *(row["payload"] for row in document["controls"])])
    for record in records:
        parent.parent.historical_record(record)
    return closure, bridge, reference, refs, operands, tensors, trajectory, records, current


@contextmanager
def guarded(audit):
    require(prior._active is None and sys.getprofile() is None, "nested execution guard")
    if not prior._hook_installed:
        sys.addaudithook(prior.audit_event)
        prior._hook_installed = True
    forbidden = {
        "check", "main", "execute", "run", "diagnose", "stages", "_stages",
        "continuation_stages", "native_layer", "execute_layer", "execute_layers",
        "original_branches", "reference_arrays", "build_report", "intervene",
        "adjust", "nearest", "finite_table", "run_control", "guarded", "suffix_stages",
        "compute_suffix", "stage13", "evaluate_control", "evaluate",
        "evaluate_actual_rtl_result", "evaluate_p0_transaction",
    }
    native_source = str(Path(native.__file__).resolve())
    state_source = str(Path(native.state.__file__).resolve())

    def profile(frame, event, arg):
        if event != "call":
            return
        file, name = frame.f_code.co_filename, frame.f_code.co_name
        if not file.startswith(str(ROOT) + "/"):
            return
        key = file.removeprefix(str(ROOT) + "/") + ":" + name
        audit["project_calls"][key] = audit["project_calls"].get(key, 0) + 1
        denied = file not in SOURCE_PATHS or (file != str(SOURCE) and name in forbidden)
        if file == native_source:
            denied |= name not in {"decoded", "rne", "toward_zero", "<genexpr>"}
        if file == state_source and name == "project":
            denied |= frame.f_locals["prefix"] != PREFIX + "mlp.down_proj"
        if name in {"expected_stage", "stage_report", "local_reference", "check_stage_state",
                    "evaluate_decoder_stage"}:
            denied |= frame.f_locals.get("stage") not in range(10, 19)
        if denied:
            audit["forbidden_events"].append(key)
            raise retained.bridge.BindingError("unapproved project dispatch: " + key)

    prior._active = (None, audit)
    sys.setprofile(profile)
    try:
        yield
    finally:
        sys.setprofile(None)
        prior._active = None


def evaluate_control(control, raw, original, classified, tensors, trajectory,
                     reference, refs, operands, oracle, audit):
    require(control == classified["control"], "retained control lineage splice")
    protected = {k: retained.digest(v) for k, v in raw.items()}
    arrays, detail, timings = compute_suffix(raw, tensors, operands)
    audit["controls"].append(control)
    audit["stage_dispatches"].extend([[control, 23, 0, s] for s in (16, 17, 18)])
    audit["final_rmsnorm_invocations"] += 1
    audit["pair_head_invocations"] += 1
    began = time.monotonic()
    oracle.verify_stage16(raw["stage14"], raw["stage15"], arrays["stage16"],
                          arrays["s16_unrounded_binary64"])
    oracle.verify_projection(tensors, arrays["stage16"], arrays["stage17"])
    oracle.suffix_oracle.verify_suffix(
        arrays["stage18"], operands, arrays["final_rmsnorm"], detail, arrays["final_logits"])
    reports = []
    state = {"h": raw["input_hidden"], "i": raw["input_i"], "z": raw["input_z"]}
    for stage in range(10, 19):
        expected = layer.expected_stage(stage, arrays, state, tensors)
        reports.append(layer.stage_report(stage, arrays, trajectory, reference, expected))
        audit["stage_gate_checks"].append([control, 23, 0, stage])
    after = prior.compare(arrays["stage18"], reference)
    require(after["status"] == reports[-1]["status"], "S18 comparator disagreement")
    margins = retained.paired_margins(
        arrays["final_logits"], original["final_logits"], classified["paired_margins"], refs)
    require(protected == {k: retained.digest(v) for k, v in raw.items()}, "retained bytes changed")
    for key in PROTECTED:
        require(retained.digest(arrays[key]) == protected[key], "protected array: " + key)
    timings["oracles_and_gates"] = time.monotonic() - began
    return arrays, {
        "control": control, "stages": reports, "S18": after, "paired_margins": margins,
        "component_oracles": {"status": "PASS", "errors": []},
        "protected_arrays": protected, "protected_bytes_unchanged": True,
        "changed_words": {str(s): int(np.count_nonzero(arrays[f"stage{s}"] != raw[f"stage{s}"]))
                          for s in (16, 17, 18)},
        "final_rmsnorm_integer_details": detail, "timing_seconds": timings,
    }


def dispatch_census(audit):
    def count(module, name):
        key = str(Path(module.__file__).resolve().relative_to(ROOT)) + ":" + name
        return audit["project_calls"].get(key, 0)
    require(audit["forbidden_events"] == audit["writes"] == []
            and audit["controls"] == list(CONTROLS)
            and audit["stage_dispatches"] == [[c, 23, 0, s] for c in CONTROLS for s in (16, 17, 18)]
            and audit["stage_gate_checks"] == [[c, 23, 0, s] for c in CONTROLS for s in range(10, 19)]
            and audit["final_rmsnorm_invocations"] == audit["pair_head_invocations"] == 9,
            "ordered dispatch/gate/write census")
    require(count(native.state, "project") == count(native.state, "add") == 9
            and count(native, "toward_zero") == 9 and count(native, "projection") == 0
            and count(retained.fp16_adaptation_oracle, "rmsnorm") == 9,
            "observed arithmetic dispatch census")


def evaluate(oracle, *, audit=None):
    """In-memory retained entry, callable only by an independently authorized owner."""
    audit = base.new_audit() if audit is None else audit
    require(audit == base.new_audit(), "fresh evaluation audit required")
    audit["science_invocations"] = 0
    rows, payloads = [], {}
    sources = [pin(Path(p)) for p in sorted(SOURCE_PATHS)]
    with guarded(audit):
        closure, bridge, reference, refs, operands, tensors, trajectory, records, prepared = inputs()
        operand_pins = [retained.digest(v) for v in operands]
        tensor_pins = {k: retained.digest(v) for k, v in tensors.items()}
        reference_pins = [retained.digest(reference), *[retained.digest(refs[b])
                                                     for b in ("fp16", "binary64")]]
        audit["science_invocations"] = 1
        for control, (original, raw), classified in zip(
                CONTROLS, prepared, bridge["controls"], strict=True):
            payloads[control], row = evaluate_control(
                control, raw, original, classified, tensors, trajectory, reference,
                refs, operands, oracle, audit)
            rows.append(row)
        for record in records:
            parent.parent.historical_record(record)
        for record in sources:
            retained.selection.bound(record)
        require(operand_pins == [retained.digest(v) for v in operands]
                and tensor_pins == {k: retained.digest(v) for k, v in tensors.items()}
                and reference_pins == [retained.digest(reference), *[retained.digest(refs[b])
                                          for b in ("fp16", "binary64")]],
                "authenticated operand/reference mutation")
        dispatch_census(audit)
        status = parent.parent.scientific_status(rows)
    return {
        "schema_version": 1, "status": status, "controls": rows, "audit": audit,
        "non_admission": True, "normal_independent_review": "REQUIRED",
        "automatic_replay_authorized": False, "boundary": BOUNDARY,
        "source_pins": sources, "input_pins": records, "operand_pins": operand_pins,
        "tensor_pins": tensor_pins, "thresholds": closure["thresholds"],
        "historical_failure": closure["historical_failure"],
        "protected_input_files_unchanged": True, "flags": base.closure_source.FLAGS,
    }, payloads
