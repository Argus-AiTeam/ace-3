"""Read-only retained S18 obstruction/paired-margin diagnostic, not a producer.

Run with python -B -m ace3.model.candidates.
q24_s16_l23_p0_value_content_s18_obstruction_bridge_v1 (join the module name).
SUPPORTED concerns only the retained obstruction structure and exact accounting.
The original science UNKNOWN, source failure, and all admission limits remain.
No arrays, head dot products, candidate imports, or scientific checks execute.
"""

from contextlib import contextmanager
from fractions import Fraction as F
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "build/q24_s16_l23_p0_value_content_sufficiency_v1-8ee0b3aaeecb-attempt001"
CLOSURE = OUT / "closure-2eb443534561-round3-observations.json"
CLOSURE_SHA256 = "5682fa4a4e908d8e9d8ff6adba4678337d74b197213570fa2753796c8d9f8dc1"
CLOSURE_BYTES = 4284230
HISTORICAL_AUTHORITY = frozenset(
    Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/eff468ba33e5") / name
    for name in ("mission.json", "round-0001.json")
)
CONTROLS = (
    "frozen_inherited", "frozen_inherited_o", "frozen_inherited_down",
    "frozen_inherited_o_down", "scratch", "scratch_down", "mapped62",
    "mapped_all", "inherited_native",
)
PAIR = [34319, 13]
BUDGET = F(1, 8)
BOUNDARY = (
    "Descriptive retained-only association, not causal attribution or a repair. "
    "Original science UNKNOWN and historical failures remain unchanged. "
    "Native S16 RTZ; G128 asymmetric packed INT4, FP16 operators/scales/KV, "
    "wider-than-FP16 Q24 residual state. No admission, strict-FP16-state W4A16, "
    "new-token, full-model, native-V validation, or reference substitution claim. "
    "No prefix, admission, reference, closed-producer or full-head replay."
)


class BindingError(ValueError):
    """Retained evidence is absent, malformed, or inconsistent."""


def require(condition, message):
    if not condition:
        raise BindingError(message)


_guard = None
_hook_installed = False


def _audit(event, args):
    if _guard is None:
        return
    forbidden = event.startswith(("subprocess.", "socket.", "ctypes."))
    forbidden |= event in {
        "os.system", "os.exec", "os.posix_spawn", "os.fork", "os.forkpty",
        "os.remove", "os.rename", "os.rmdir", "os.mkdir", "os.link",
        "os.symlink", "os.truncate", "os.chmod", "os.chown", "os.utime",
    }
    if event == "open":
        mode, flags = args[1:3]
        forbidden |= bool(mode and any(c in mode for c in "wax+"))
        forbidden |= bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT |
                                   os.O_TRUNC | os.O_APPEND))
    if event == "import":
        forbidden |= args[0].split(".")[0] in {"ace3", "numpy", "scipy"}
    if forbidden:
        _guard["forbidden_events"].append(event)
        raise BindingError("forbidden retained-diagnostic operation: " + event)


@contextmanager
def readonly_guard():
    """Process-local audit guard, not a claim of an OS-wide sandbox."""
    global _guard, _hook_installed
    require(_guard is None, "nested diagnostic guard")
    if not _hook_installed:
        sys.addaudithook(_audit)
        _hook_installed = True
    audit = {"forbidden_events": [], "scope": "process-local Python audit"}
    _guard = audit
    try:
        yield audit
    finally:
        _guard = None


def authenticate(path, digest, size=None):
    path = Path(path)
    require(path.is_absolute() and (
        path.resolve().is_relative_to(ROOT) or path == path.resolve() in HISTORICAL_AUTHORITY),
        "pin path outside isolated worktree or exact retained authority")
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == digest, "sha256 mismatch: " + str(path))
    require(size is None or len(data) == size, "byte length mismatch: " + str(path))
    return data


def load_inputs():
    closure = json.loads(authenticate(CLOSURE, CLOSURE_SHA256, CLOSURE_BYTES))
    require(closure["closure_status"] == "COMPLETE_RETAINED_NUMERICAL_CLOSURE"
            and closure["mission_id"] == "2eb443534561"
            and closure["science_status"] == "UNKNOWN"
            and closure["science_invocations_this_round"] == 0,
            "closure identity/status")
    preserved = closure["preservation"]
    require(preserved["changed"] == []
            and preserved["files"] == len(preserved["sha256_before"]),
            "closure preservation census")
    inventory = preserved["sha256_before"]
    sizes = {}
    for pin in closure["pins"]:
        actual, declared = pin["actual"], pin["declared"]
        path = Path(declared["path"])
        if not path.is_absolute():
            path = ROOT / path
        require(pin["match"] is True and str(path) == actual["path"]
                and declared["sha256"] == actual["sha256"]
                and declared.get("bytes", actual["bytes"]) == actual["bytes"]
                and inventory[actual["path"]] == actual["sha256"],
                "closure pin declaration: " + pin["field"])
        sizes[actual["path"]] = actual["bytes"]
    # Authenticate bytes, never import historical source declarations or run checks.
    for path, digest in inventory.items():
        authenticate(path, digest, sizes.get(path))

    def document(path):
        return json.loads(authenticate(path, inventory[str(path)], sizes.get(str(path))))

    terminal = document(Path(closure["terminal_path"]))
    require(terminal == closure["terminal"] and terminal["mission_id"] == "8ee0b3aaeecb"
            and terminal["status"] == "UNKNOWN"
            and terminal["scientific_check_invocations"] == 1
            and terminal["non_admission"] is True
            and terminal["normal_independent_review"] == "REQUIRED",
            "terminal identity/status/budget")
    stdout_pin = terminal["candidate_result"]
    stdout = authenticate(stdout_pin["path"], stdout_pin["sha256"], stdout_pin["bytes"])
    failure, command, compile_tests = (
        document(OUT / name) for name in ("failure.json", "command.json", "compile_tests.json")
    )
    require(command["cwd"] == str(ROOT) and command["uid"] == 1000, "retained cwd/account")
    auth_pins = [pin for pin in command["manager_binding"]
                 if pin["path"].endswith(
                     "/stage11-science-8ee0b3aaeecb-attempt001/issued/execution-authorization.json")]
    require(len(auth_pins) == 1, "issued authorization binding")
    contract = document(Path(auth_pins[0]["path"]))["contract"]
    require(contract["controls"] == list(CONTROLS)
            and contract["thresholds"] == closure["thresholds"], "contract controls/thresholds")
    baseline_path = ROOT / "build/q24_s16_final_head_from_l23_coordinate62_suffix_execution_v1_attempt001/result.json"
    baseline = document(baseline_path)
    require(baseline["controls"] == contract["historical_failures"], "historical parent splice")
    require([r["control"] for r in failure["controls"]] == list(CONTROLS),
            "failure control identities/order")
    for row, old in zip(failure["controls"], contract["historical_failures"]):
        require(row["control"] == old["control"] and row["parent"] == old["parent"]
                and row["parent"]["terminal_archive"] == contract["inputs"][row["control"]]
                and row["substitution"]["source"] == contract["references"]["original_input_L23_fp16"],
                "parent/operand/lineage splice")
    history = {
        str(path): document(path)
        for name in (
            "closure-2eb443534561-observations.json",
            "closure-2eb443534561-round2-observations.json",
            "closure-2eb443534561-round2-corrected-observations.json",
            "closure-2eb443534561-round2-final-observations.json",
        )
        for path in (OUT / name,)
    }
    return closure, failure, command, compile_tests, stdout, history


def parse_envelope(stdout, failure, compile_tests, command):
    envelope = json.loads(stdout)
    require(isinstance(envelope, dict) and set(envelope) == {
        "status", "error_type", "error", "normal_host_review", "expected_binding",
        "events", "automatic_replay_authorized", "flags",
    }, "single JSON envelope fields")
    for key in ("status", "error_type", "error", "automatic_replay_authorized", "flags"):
        require(envelope[key] == failure[key], "envelope mismatch: " + key)
    require(envelope["status"] == "UNKNOWN"
            and envelope["automatic_replay_authorized"] is False
            and envelope["normal_host_review"] == "REQUIRED"
            and isinstance(envelope["expected_binding"], dict), "UNKNOWN/replay/review envelope")
    events = envelope["events"]
    require(isinstance(events, list) and len(events) == 10, "exactly ten ordered events")
    origins = {p["path"]: p for p in command["origins"].values()}
    require(len(compile_tests["compiled"]) == len(origins)
            and {p["path"]: p for p in compile_tests["compiled"]} == origins,
            "compile origin pins")
    entries = []
    for relative in (
        "ace3/model/candidates/q24_s16_l23_p0_value_content_sufficiency_v1.py",
        "tests/test_q24_s16_l23_p0_value_content_sufficiency_v1.py",
    ):
        matches = [p for p in command["manager_binding"] if p["path"] == str(ROOT / relative)]
        require(len(matches) == 1, "compile entry pin")
        entries.append(matches[0])
    require(events[0] == {**compile_tests, "phase": "compile_tests", "compiled": entries},
            "compile event mismatch")
    require(type(compile_tests["executed"]) is int and compile_tests["executed"] > 0
            and compile_tests["executed"] == compile_tests["loaded"]
            and all(compile_tests[k] == 0 for k in (
                "errors", "failures", "skipped", "unexpected_successes", "retained_control_dispatches")),
            "compile event outcome")
    controls = failure["controls"]
    require([r["control"] for r in controls] == list(CONTROLS)
            and failure["audit"]["controls"] == list(CONTROLS), "failure control order")
    fields = {"control", "paired_final_margins", "status", "substitution", "timing_seconds"}
    for label, emitted, retained in zip(CONTROLS, events[1:], controls):
        require(isinstance(emitted, dict) and set(emitted) == fields
                and emitted["control"] == label, "control identity/order/fields")
        require(emitted == {k: retained[k] for k in fields}, "control shared fields mismatch")
    return envelope


def s18_quantities(word, reference):
    actual = struct.unpack("<e", bytes.fromhex(word)[::-1])[0]
    ref = float.fromhex(reference)
    require(math.isfinite(actual) and math.isfinite(ref), "nonfinite S18 operand")
    nearest_bytes = struct.pack("<e", ref)
    nearest = struct.unpack("<e", nearest_bytes)[0]
    error = abs(F(actual) - F(ref))
    q = abs(F(nearest) - F(ref))
    excess = error - q
    return {
        "actual_fp16_bits": word, "reference_binary64_hex": reference,
        "nearest_fp16_bits": nearest_bytes[::-1].hex(), "actual_error": str(error),
        "q": str(q), "excess_error": str(excess), "excess_budget": str(BUDGET),
        "excess_over_budget": str(excess - BUDGET),
        "threshold_margin": str(BUDGET - excess), "accepted": excess <= BUDGET,
    }


def margin_bridge(row):
    a, r, s = (F(row[k]) for k in (
        "retained_actual_margin", "retained_reference_margin", "substituted_actual_margin"))
    values = {
        "retained_margin_change": a - r, "substituted_margin_change": s - r,
        "paired_intervention_delta": s - a, "strict_direction_product": (s - a) * (r - a),
    }
    require(row["pair"] == PAIR and row["reference_branch"] in ("fp16", "binary64"),
            "paired reference identity")
    for key, value in values.items():
        require(row[key] == str(value), "margin arithmetic: " + key)
    require(row["strict_direction_pass"] is (values["strict_direction_product"] > 0),
            "strict direction predicate")
    return {**row, "required_direction": str(r - a)}


def build_report(closure, failure):
    require([row["control"] for row in closure["controls"]] == list(CONTROLS)
            and [row["control"] for row in failure["controls"]] == list(CONTROLS),
            "S18 control identities/order")
    controls = []
    for retained, original in zip(closure["controls"], failure["controls"]):
        label = retained["control"]
        archived = original["reports"][-1]
        require(archived["stage"] == 18, "S18 report stage")
        gate = archived["binary64_v1"]
        rows = retained["S18"]
        require(gate["coordinates"] == 896 and len(rows) == len(gate["rows"]) == 896,
                "S18 width")
        failures = []
        for index, (row, old) in enumerate(zip(rows, gate["rows"])):
            new = {"index": index, **s18_quantities(
                row["actual_fp16_bits"], row["reference_binary64_hex"])}
            require(row == new, f"S18 exact arithmetic: {label}/{index}")
            require(all(old[k] == v for k, v in new.items()
                        if k not in {"excess_over_budget", "threshold_margin"}),
                    f"S18 archived row mismatch: {label}/{index}")
            if not row["accepted"]:
                failures.append(row)
        require(retained["failure_count"] == gate["failure_count"] == len(failures)
                and gate["failures"] == [r for r in gate["rows"] if not r["accepted"]]
                and gate["passed"] is (len(failures) == 0), "S18 failure family census")
        margins = retained["paired_margins"]
        require([m["reference_branch"] for m in margins] == ["fp16", "binary64"]
                and original["head_pair"] == PAIR
                and original["paired_final_margins"] == [
                    {k: v for k, v in m.items()
                     if k not in {"strict_direction_pass", "strict_direction_product"}}
                    for m in margins], "archived paired margin splice")
        margins = [margin_bridge(m) for m in margins]
        require(all(margins[0][k] == margins[1][k] for k in (
            "retained_actual_margin", "substituted_actual_margin", "paired_intervention_delta")),
            "paired actual margins differ by reference branch")
        controls.append({
            "control": label, "failure_coordinates": [r["index"] for r in failures],
            "failure_count": len(failures), "exact_failures": failures, "paired_margins": margins,
        })
    ordinary = [c for c in controls if c["control"] != "mapped_all"]
    mapped = next(c for c in controls if c["control"] == "mapped_all")
    sets = [set(c["failure_coordinates"]) for c in controls]
    core = set.intersection(*sets)
    ordinary_union = set.union(*(set(c["failure_coordinates"]) for c in ordinary))
    mapped_set = set(mapped["failure_coordinates"])
    ordinary_uniform = all(c["exact_failures"] == ordinary[0]["exact_failures"] for c in ordinary)
    source_rows = {c["control"]: c["S18"] for c in closure["controls"]}
    contrast = []
    for index in sorted(ordinary_union | mapped_set):
        a = source_rows[ordinary[0]["control"]][index]
        b = source_rows["mapped_all"][index]
        require(a["reference_binary64_hex"] == b["reference_binary64_hex"],
                "independent original S18 reference differs between controls")
        contrast.append({
            "index": index, "ordinary": a, "mapped_all": b,
            "mapped_minus_ordinary_excess": str(F(b["excess_error"]) - F(a["excess_error"])),
        })
    deltas = [F(c["paired_margins"][0]["paired_intervention_delta"]) for c in ordinary]
    mapped_delta = F(mapped["paired_margins"][0]["paired_intervention_delta"])
    supported = (ordinary_uniform and len(core) == 12 and len(ordinary_union) == 14
                 and mapped_set == core and ordinary_union - core == {783, 827}
                 and all(d == F(-3, 32) for d in deltas) and mapped_delta == F(-11, 128)
                 and all(not m["strict_direction_pass"] for c in controls for m in c["paired_margins"]))
    return {
        "status": "SUPPORTED" if supported else "REJECTED",
        "claim": "retained common-core/mapped_all obstruction structure and paired-margin accounting",
        "common_core": sorted(core),
        "ordinary_only": sorted(ordinary_union - mapped_set),
        "mapped_all_only": sorted(mapped_set - ordinary_union),
        "ordinary_exact_family_uniform": ordinary_uniform,
        "mapped_all_excess_contrast": contrast,
        "controls": controls,
        "margin_relation": {
            "ordinary_delta": str(deltas[0]), "mapped_all_delta": str(mapped_delta),
            "mapped_minus_ordinary": str(mapped_delta - deltas[0]),
            "strict_direction_passes": sum(m["strict_direction_pass"] for c in controls for m in c["paired_margins"]),
            "interpretation": "Co-occurrence only; S18 excess is not an additive head-margin contribution.",
        },
        "thresholds": closure["thresholds"],
        "historical_failure": closure["historical_failure"],
    }


def check():
    result = {
        "status": "UNKNOWN", "original_science_status": "UNKNOWN",
        "original_status_authenticated": False, "non_admission": True,
        "science_invocations_this_round": 0, "normal_independent_review": "REQUIRED",
        "boundary": BOUNDARY,
        "closure_pin": {"path": str(CLOSURE), "bytes": CLOSURE_BYTES, "sha256": CLOSURE_SHA256},
    }
    with readonly_guard() as audit:
        try:
            closure, failure, command, compile_tests, stdout, history = load_inputs()
            parse_envelope(stdout, failure, compile_tests, command)
            result["original_status_authenticated"] = True
            result.update(build_report(closure, failure))
            result["historical_checker_failures"] = history
            result["authenticated_retained_files"] = closure["preservation"]["files"]
            result["retained_pins"] = closure["pins"]
        except (BindingError, OSError, ValueError, KeyError, TypeError, OverflowError) as error:
            result.update(status="UNKNOWN", error_type=type(error).__name__, error=str(error))
    result["audit"] = audit
    return result


def main():
    require(len(sys.argv) == 1, "diagnostic accepts no dispatch or output arguments")
    print(json.dumps(check(), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
