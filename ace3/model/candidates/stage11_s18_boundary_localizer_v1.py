"""Retained b497 S18 boundary substitution; final two-row CPU suffix only."""

import argparse
from bisect import bisect_left, bisect_right
from fractions import Fraction
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).resolve()
TEST = ROOT / "tests/test_stage11_s18_boundary_localizer_v1.py"
RETAINED = ROOT / "build/stage11-attention-output-suffix-authorized-b4978b898600-attempt001/run"
PINS = {
    "stdout": {"path": str(RETAINED / "check.stdout"), "bytes": 16628091,
               "sha256": "f2e71a908cc1f828a19429c0909e7afcfbf91bb0a20149a9e4b416af0cf8c8b7"},
    "receipt": {"path": str(RETAINED / "terminal-receipt.json"), "bytes": 959,
                "sha256": "5863ebb41575923082806cdea996c2552976a859a8da3ca987550060dd2ff32c"},
    "review": {
        "path": "/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/b4978b898600/round-0001.json",
        "bytes": 690, "sha256": "0a022f65bcc7bcf5d983c3a67480fbcec08ba0c68cd2536ab56946a98976d957"},
}
COMMON = (53, 56, 62, 190, 208, 241, 289, 490, 493, 494, 502, 648)
UNION = COMMON + (783, 827)
CONTROLS = ("frozen_inherited", "frozen_inherited_o", "frozen_inherited_down",
            "frozen_inherited_o_down", "scratch", "scratch_down", "mapped62",
            "mapped_all", "inherited_native")
PAIR = (34319, 13)
BOUNDARY = (
    "Isolated non-admission FP16 S18 boundary substitution, not a Q24 state "
    "projection or native-trajectory repair. Only final RMSNorm and tied-head "
    "rows 34319/13 execute. All retained Stage11/S12-S17/Q24/KV/lineage, "
    "thresholds, original-input references, account/role/model/budget/access "
    "bindings and historical failures remain unchanged. Native-S16-RTZ, "
    "G128 asymmetric packed INT4/native GEMM/no qzero plus-one and FP16 "
    "scales/operators/KV are unchanged. Q24 state is wider than FP16; no "
    "strict-FP16-state W4A16, new-token or full-model admission."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def pin(path):
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def bound(record):
    path = Path(record["path"])
    require(path.is_absolute() and path.resolve() == path, "noncanonical retained pin")
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == record["sha256"]
            and ("bytes" not in record or len(data) == record["bytes"]),
            "retained hash/size mismatch: " + str(path))
    return data


def scalar(word):
    require(type(word) is int and 0 <= word <= 65535 and word & 0x7c00 != 0x7c00,
            "invalid finite FP16 word")
    return Fraction.from_float(float(np.asarray(word, dtype="<u2").view("<f2")))


def finite_table():
    entries = sorted((scalar(w), w) for w in range(65536) if w & 0x7c00 != 0x7c00)
    return tuple(v for v, _ in entries), tuple(w for _, w in entries)


def nearest(word, reference, table):
    """Exact closed excess interval; ties use value then unsigned word."""
    values, words = table
    pos = bisect_left(values, reference)
    candidates = values[max(0, pos - 1):min(len(values), pos + 1)]
    floor = min(abs(v - reference) for v in candidates)
    radius = floor + Fraction(1, 8)
    lo, hi = bisect_left(values, reference - radius), bisect_right(values, reference + radius)
    require(lo < hi, "empty passable FP16 interval")
    actual = scalar(word)
    if abs(actual - reference) <= radius:
        return word, floor
    chosen = lo if actual < reference - radius else hi - 1
    chosen = bisect_left(values, values[chosen], lo, hi)
    return words[chosen], floor


def adjust(words, references, failures, table, evaluate):
    require(len(words) == len(references) == 896, "S18 width changed")
    expected = {row["index"]: row for row in failures}
    require(len(expected) == len(failures), "duplicate failure coordinate")
    adjusted, reports, observed = list(words), [], []
    for index, (word, reference) in enumerate(zip(words, references, strict=True)):
        require(np.isfinite(reference), "nonfinite original reference")
        gate = evaluate(actual_fp16_bits=word, reference_binary64_hex=float(reference).hex())
        if gate["accepted"]:
            continue
        observed.append(index)
        require(index in expected and gate == {k: v for k, v in expected[index].items()
                                               if k != "index"}, "retained scalar gate changed")
        selected, floor = nearest(word, Fraction.from_float(float(reference)), table)
        require(str(floor) == gate["q"], "independent representation-floor mismatch")
        after = evaluate(actual_fp16_bits=selected, reference_binary64_hex=float(reference).hex())
        require(after["accepted"] and selected != word, "substitution did not pass")
        adjusted[index] = selected
        reports.append({"index": index, "retained_word": word, "selected_word": selected,
                        "signed_delta": str(scalar(selected) - scalar(word)),
                        "retained_gate": gate, "adjusted_gate": after})
    require(observed == list(expected), "retained failure census changed")
    require([i for i, (a, b) in enumerate(zip(words, adjusted)) if a != b] == observed,
            "passing coordinate changed")
    return np.asarray(adjusted, dtype="<u2"), reports


def classify(rows):
    require(len(rows) == 18 and {(r["control"], r["branch"]) for r in rows}
            == {(c, b) for c in CONTROLS for b in ("fp16", "binary64")},
            "terminal contrast census changed")
    return "SUPPORTED" if all(Fraction(r["predicted_delta"]) * Fraction(r["margin_delta"]) > 0
                              for r in rows) else "REJECTED"


def named_assets(value):
    found = []
    if isinstance(value, dict):
        if "assets" in value:
            found.append(value["assets"])
        for child in value.values():
            found.extend(named_assets(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(named_assets(child))
    return found


def authenticate():
    result, receipt, review = (json.loads(bound(PINS[k])) for k in ("stdout", "receipt", "review"))
    require(receipt["candidate_result"] == PINS["stdout"] and receipt["status"] == "REJECTED"
            and receipt["scientific_check_invocations"] == 1 and receipt["failure"] is None
            and receipt["mission_id"] == review["mission_id"] == "b4978b898600"
            and review["producer_role"] == "reviewer" and review["review"]["status"] == "done",
            "retained certification mismatch")
    capture = json.loads(bound(receipt["capture"]))
    require(capture["success"] and capture["failure"] is None
            and capture["preflight"] == result["capture_preflight"], "retained capture mismatch")
    require(bound(receipt["candidate_stderr"]) == b"", "retained stderr is not empty")
    require(result["status"] == "REJECTED" and result["non_admission"]
            and result["lane_terminated"] and result["final_arithmetic_oracle"] == "PASS",
            "retained result identity mismatch")
    contract = result["frozen_contract"]
    require(contract["controls"] == list(CONTROLS)
            and contract["gates"] == dict.fromkeys(("source", "operand", "state", "kv", "lineage"), "PASS")
            and contract["source_identity"] == {"layer": 23, "position": 0, "token_id": 9707},
            "retained control/source/lineage mismatch")
    declaration = json.loads(bound(result["capture_preflight"]["execution_authorization"]["declaration"]))
    require(declaration["contract"] == contract, "retained declaration contract mismatch")
    records = [*contract["sources"].values(), *contract["inputs"].values(),
               *contract["references"].values(), *declaration["runtime_sources"].values()]
    for record in records:
        bound(record)
    for control in CONTROLS:
        stages = result["stage_reports"][control]
        require([s["stage"] for s in stages] == list(range(12, 19)), "stage report census changed")
        require(all(s["kv_lineage"] == s["residual_state_lineage"] == "PASS" for s in stages),
                "retained stage lineage gate changed")
        report = stages[-1]["binary64_v1"]
        require([f["index"] for f in report["failures"]]
                == list(COMMON if control == "mapped_all" else UNION)
                and report["failure_count"] == len(report["failures"]), "S18 failure union changed")
    return result, records


def check(audit):
    retained, records = authenticate()
    # Import arithmetic only after the original source/test census authenticates.
    from ace3.model.candidates import binary64_fp16_excess_v1 as policy
    from ace3.model.candidates import q24_s16_final_head_from_l23_coordinate62_suffix_execution_v1 as final

    require(policy.EXCESS_BUDGET == Fraction(1, 8), "S18 excess threshold changed")
    summary_pin = final.preflight.PINS["result"]
    summary = json.loads(bound(summary_pin))
    assets = named_assets(summary)
    require(assets and len({encoded(a) for a in assets}) == 1, "ambiguous retained operand assets")
    assets = assets[0]
    bound(assets["checkpoint"])
    operands = final.load_operands({"assets": assets})
    operands = (operands[0], operands[1][list(PAIR)].copy())
    for array in operands:
        array.flags.writeable = False
    reference_pins = retained["frozen_contract"]["references"]
    reference64 = np.load(io.BytesIO(bound(reference_pins["original_input_L23_binary64"])),
                          allow_pickle=False)
    require(reference64.shape == (896,) and reference64.dtype.str == "<f8", "S18 reference type changed")
    references = {b: np.load(io.BytesIO(bound(reference_pins["original_input_final_" + b])),
                             allow_pickle=False) for b in ("fp16", "binary64")}
    own_sources = {"candidate": pin(SOURCE), "test": pin(TEST)}
    spec = importlib.util.spec_from_file_location("stage11_s18_boundary_localizer_oracle", TEST)
    require(spec is not None and spec.loader is not None, "missing suffix oracle")
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    table = finite_table()
    original = encoded(retained)
    rows, controls = [], []
    with final.no_dispatch(audit):
        for control in CONTROLS:
            output = retained["outputs"][control]
            words, reports = adjust(output["stage18"], reference64,
                                    retained["stage_reports"][control][-1]["binary64_v1"]["failures"],
                                    table, policy.evaluate_layer_final_output)
            start = time.monotonic()
            audit["final_rmsnorm_invocations"] += 1
            norm, account = final.rmsnorm(words, operands[0])
            norm_end = time.monotonic()
            audit["tied_pair_head_invocations"] += 1
            logits = final.logits(norm, operands[1])
            head_end = time.monotonic()
            oracle.verify_suffix(words, operands, norm, account, logits)
            new = scalar(int(logits[0])) - scalar(int(logits[1]))
            old = scalar(output["pair_logits"][0]) - scalar(output["pair_logits"][1])
            for branch in ("fp16", "binary64"):
                parent = next(r for r in retained["rows"] if r["control"] == control and r["branch"] == branch)
                ref = references[branch]
                margin = (scalar(int(ref[PAIR[0]])) - scalar(int(ref[PAIR[1]])) if branch == "fp16" else (
                    Fraction.from_float(float(ref[PAIR[0]])) - Fraction.from_float(float(ref[PAIR[1]]))))
                require(str(margin) == parent["independent_original_reference_margin"]
                        and str(old) == parent["intervened_margin"], "terminal reference/operand splice")
                delta = new - Fraction(parent["retained_margin"])
                rows.append({**parent, "b497_margin": str(old), "adjusted_margin": str(new),
                             "adjusted_margin_error": str(new - margin), "margin_delta": str(delta),
                             "boundary_adjustment_delta": str(new - old),
                             "direction_observed": Fraction(parent["predicted_delta"]) * delta > 0})
            controls.append({"control": control, "changed_coordinates": [r["index"] for r in reports],
                             "substitutions": reports, "adjusted_stage18": words.tolist(),
                             "final_rmsnorm": norm.tolist(), "pair_logits": logits.tolist(),
                             "S18_binary64_status": "PASS", "independent_suffix_oracle": "PASS",
                             "timing_seconds": {"rmsnorm": norm_end - start,
                                                "pair_head": head_end - norm_end}})
    require(encoded(retained) == original, "retained bytes changed in memory")
    for record in [*records, *PINS.values(), summary_pin, *own_sources.values()]:
        bound(record)
    require(audit["final_rmsnorm_invocations"] == audit["tied_pair_head_invocations"] == 9,
            "suffix dispatch census changed")
    status = classify(rows)
    return {"status": status, "hypothesis": "S18 boundary substitutions remove opposite terminal response",
            "classification": "direction_restored" if status == "SUPPORTED" else "boundary_only_explanation_rejected",
            "controls": controls, "rows": rows, "failure_union": list(UNION),
            "common_failures": list(COMMON), "sources": own_sources, "retained": PINS,
            "retained_bytes_preserved": True, "frozen_contract": retained["frozen_contract"],
            "operand_assets": assets, "suffix_oracle": "PASS"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    audit = dict.fromkeys(("forbidden_calls", "admission_invocations", "prefix_invocations",
                           "reference_invocations", "producer_invocations", "service_invocations",
                           "final_rmsnorm_invocations", "tied_pair_head_invocations"), 0)

    def read_only(event, args):
        if ((event == "open" and (args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)))
                or event in ("subprocess.Popen", "os.system", "os.remove", "os.rename",
                             "os.mkdir", "socket.connect", "socket.bind")):
            audit["forbidden_calls"] += 1
            raise RuntimeError("retained-only diagnostic forbids writes and external dispatch")

    sys.dont_write_bytecode = True
    sys.addaudithook(read_only)
    try:
        result = check(audit)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, AssertionError) as error:
        result = {"status": "UNKNOWN", "error": type(error).__name__ + ": " + str(error)}
    result.update({"audit": audit, "boundary": BOUNDARY, "non_admission": True,
                   "normal_independent_review": "REQUIRED"})
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["status"] != "UNKNOWN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
