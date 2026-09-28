"""One CPU S18 boundary counterfactual on the reviewed value-content closure.

Run with python -B -m ace3.model.candidates.
q24_s16_l23_p0_value_content_s18_family_counterfactual_v1 --execute
(join the module name). The fixed fresh output consumes this invocation even
on UNKNOWN. Retained Q24 I/Z/H are not projected or repaired: a separate FP16
diagnostic boundary feeds only final RMSNorm and tied-head rows [34319,13].
Direction is measured from the bridge's ORIGINAL retained margin, not from
the already substituted value-content margin. No result admits a trajectory.
"""

import argparse
from contextlib import contextmanager
from fractions import Fraction as F
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

import numpy as np
from safetensors import safe_open  # Load the CPU reader before installing guards.

from ace3.model import awq_bit_oracle, fp16_adaptation_oracle, streaming_lm_head_reference
from ace3.model.candidates import binary64_fp16_excess_v1 as policy
from ace3.model.candidates import q24_s16_l23_p0_value_content_s18_obstruction_bridge_v1 as bridge
from ace3.model.candidates import stage11_s18_boundary_localizer_v1 as selection
from ace3.model.candidates import stage11_s18_boundary_localizer_v2 as suffix


ROOT = bridge.ROOT
SOURCE = Path(__file__).resolve()
NAME = SOURCE.stem
TEST = ROOT / f"tests/test_{NAME}.py"
OUTPUT = ROOT / "build" / (NAME + "_attempt001")
MISSION = "9196b8c0777c"
HANDOFFS = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs")
COMMON = (53, 56, 62, 190, 208, 241, 289, 490, 493, 494, 502, 648)
CONTROLS, PAIR = bridge.CONTROLS, bridge.PAIR
BRIDGE = bridge.OUT / "obstruction-bridge-6af2b58b51f6-result2.json"
BRIDGE_HASH = "72165a85838fd4f0f2794cb8a8c04af2ce1e9892df107e9e9b49c82db3bc4bcd"
ORACLE = ROOT / "tests/test_stage11_s18_boundary_localizer_v2.py"
BOUNDARY = selection.BOUNDARY + (
    " Reference-informed boundary counterfactual only, not reference-free output "
    "repair. The reviewed 8ee0b3aaeecb UNKNOWN and all historical failures remain. "
    "No producer, native L0-L23, prefix, admission, reference, or full-head replay."
)
require = bridge.require
pin = selection.pin
_active = None
_hook_installed = False


def module_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "missing test/oracle module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(array):
    return {"shape": list(array.shape), "dtype": array.dtype.str,
            "bytes": array.nbytes, "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def audit_event(event, args):
    if _active is None:
        return
    out, audit = _active
    denied = event.startswith(("subprocess.", "socket.", "ctypes."))
    denied |= event in {
        "os.system", "os.exec", "os.posix_spawn", "os.fork", "os.forkpty",
        "os.remove", "os.rename", "os.rmdir", "os.mkdir", "os.link",
        "os.symlink", "os.truncate", "os.chmod", "os.chown", "os.utime",
    }
    if event == "import":
        denied |= args[0].split(".")[0] in {"torch", "cupy", "jax", "tensorflow"}
    if event == "open":
        mode, flags = args[1:3]
        writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC |
                                os.O_APPEND)) or bool(mode and any(c in mode for c in "wax+"))
        if writing:
            path = None if isinstance(args[0], int) else Path(args[0]).absolute()
            denied |= (path is None or path.parent != out or path.resolve() != path
                       or not flags & os.O_EXCL)
            if not denied:
                audit["writes"].append(str(path))
    if denied:
        audit["forbidden_events"].append(event)
        raise bridge.BindingError("forbidden experiment operation: " + event)


@contextmanager
def guarded(out, audit):
    global _active, _hook_installed
    require(_active is None and sys.getprofile() is None, "nested execution guard")
    if not _hook_installed:
        sys.addaudithook(audit_event)
        _hook_installed = True
    forbidden = {
        "check", "main", "execute", "run", "diagnose", "stages", "_stages",
        "continuation_stages", "native_layer", "execute_layer", "execute_layers",
        "original_branches", "reference_arrays", "build_report",
    }
    allowed = {str(Path(m.__file__).resolve()) for m in (
        bridge, selection, suffix, policy, fp16_adaptation_oracle,
        streaming_lm_head_reference, awq_bit_oracle)}
    allowed.update((str(SOURCE), str(TEST), str(ORACLE)))

    def profile(frame, event, arg):
        if event != "call":
            return
        filename, name = frame.f_code.co_filename, frame.f_code.co_name
        if not filename.startswith(str(ROOT) + "/"):
            return
        key = filename.removeprefix(str(ROOT) + "/") + ":" + name
        audit["project_calls"][key] = audit["project_calls"].get(key, 0) + 1
        if (filename not in allowed or (filename != str(SOURCE) and name in forbidden)
                or (name == "authenticate" and filename != str(Path(bridge.__file__).resolve()))):
            audit["forbidden_events"].append(key)
            raise bridge.BindingError("unapproved project dispatch: " + key)

    _active = (out, audit)
    sys.setprofile(profile)
    try:
        yield
    finally:
        sys.setprofile(None)
        _active = None


def new_audit():
    return {"scope": "process-local Python audit and project call profile; not OS sandbox",
            "forbidden_events": [], "writes": [], "project_calls": {},
            "controls": [], "final_rmsnorm_invocations": 0, "pair_head_invocations": 0}


def read_array(record, shape, dtype):
    value = np.load(io.BytesIO(selection.bound(record)), allow_pickle=False)
    require(value.shape == shape and value.dtype.str == dtype
            and np.all(np.isfinite(value.view("<f2") if dtype == "<u2" else value)),
            "reference shape/dtype/finite gate")
    value.flags.writeable = False
    return value


def inputs():
    require(policy.EXCESS_BUDGET == F(1, 8), "S18 excess threshold changed")
    report = json.loads(bridge.authenticate(BRIDGE, BRIDGE_HASH))
    require(report["status"] == "SUPPORTED" and report["non_admission"] is True
            and report["common_core"] == list(COMMON)
            and report["ordinary_only"] == [783, 827]
            and report["mapped_all_only"] == []
            and report["audit"]["forbidden_events"] == [], "reviewed bridge classification")
    reviews = []
    for mission, round_number in (("6af2b58b51f6", 1), ("2eb443534561", 3)):
        latest = json.loads((HANDOFFS / mission / "latest.json").read_bytes())
        path = HANDOFFS / mission / f"round-{round_number:04d}.json"
        require(latest["kind"] == "handoff_ref" and latest["handoff"]["path"] == str(path),
                "review frontier changed")
        review = json.loads(path.read_bytes())
        require(review["mission_id"] == mission and review["producer_role"] == "reviewer"
                and review["review"]["status"] == "done", "independent acceptance missing")
        reviews.extend((pin(path), pin(HANDOFFS / mission / "latest.json")))
    closure, failure, command, _, _, _ = bridge.load_inputs()
    require(report["closure_pin"]["sha256"] == bridge.CLOSURE_SHA256
            and report["thresholds"] == closure["thresholds"]
            and [r["control"] for r in report["controls"]] == list(CONTROLS),
            "bridge/closure lineage")
    inventory = closure["preservation"]["sha256_before"]
    baseline = ROOT / "build/q24_s16_final_head_from_l23_coordinate62_suffix_execution_v1_attempt001/result.json"
    summary = json.loads(bridge.authenticate(baseline, inventory[str(baseline)]))["preflight"]
    reference = summary["L23_original_reference"]["reference"]
    require(reference["prior_kv"] == "own empty P0"
            and reference["fp16"] == summary["final_reference"]["reference"]["input_fp16"],
            "original global reference lineage changed")
    ref64 = read_array(reference["binary64"], (896,), "<f8")
    refs = {}
    for binding in closure["final_reference_bindings"]:
        branch = binding["branch"]
        require(binding["match"] is True and binding["issued"]["sha256"]
                == inventory[binding["issued"]["path"]], "final branch pin")
        refs[branch] = read_array(binding["issued"], (151936,),
                                  "<u2" if branch == "fp16" else "<f8")
    require(set(refs) == {"fp16", "binary64"}, "original reference branches")
    operands = suffix.load_operands(summary["assets"])
    records = [pin(bridge.CLOSURE), pin(BRIDGE), *reviews,
               *({"path": p, "sha256": h} for p, h in inventory.items())]
    require(command["uid"] == os.getuid() == 1000, "retained account changed")
    return closure, failure, report, ref64, refs, operands, records


def intervene(words, reference, rows, control, table):
    require(words.shape == (896,) and words.dtype.str == "<u2"
            and len(rows) == 896, "S18 schema")
    coordinates = list(COMMON) + ([] if control == "mapped_all" else [783, 827])
    require(control in CONTROLS, "unknown control")
    changed = words.copy()
    before, after, selections = [], [], []
    for index in range(896):
        word, ref = int(words[index]), float(reference[index])
        measured = {"index": index, **bridge.s18_quantities(f"{word:04x}", ref.hex())}
        require(measured == rows[index], "retained S18/reference splice")
        before.append(measured)
        if index in coordinates:
            require(not measured["accepted"], "classified coordinate no longer fails")
            selected, floor = selection.nearest(word, F(ref), table)
            require(str(floor) == measured["q"], "representation floor differs")
            changed[index] = selected
            selections.append({"index": index, "retained_word": f"{word:04x}",
                               "selected_word": f"{selected:04x}",
                               "signed_delta": str(selection.scalar(selected) - selection.scalar(word)),
                               "selection_order": "minimum absolute change; value; unsigned word"})
        else:
            require(measured["accepted"], "unclassified S18 failure")
        gate = policy.evaluate_layer_final_output(
            actual_fp16_bits=int(changed[index]), reference_binary64_hex=ref.hex())
        after.append({"index": index, **gate})
    require(np.flatnonzero(words != changed).tolist() == coordinates,
            "coordinate write census differs")
    return changed, {"coordinates": coordinates, "selections": selections,
                     "before": before, "after": after,
                     "failure_count": sum(not r["accepted"] for r in after),
                     "status": "PASS" if all(r["accepted"] for r in after) else "FAIL"}


def paired_margins(logits, retained_logits, margins, references):
    require(logits.shape == retained_logits.shape == (2,)
            and logits.dtype.str == retained_logits.dtype.str == "<u2",
            "paired head schema")
    new = selection.scalar(int(logits[0])) - selection.scalar(int(logits[1]))
    old = selection.scalar(int(retained_logits[0])) - selection.scalar(int(retained_logits[1]))
    require([m["reference_branch"] for m in margins] == ["fp16", "binary64"],
            "paired branch order")
    result = []
    for row in margins:
        branch = row["reference_branch"]
        r = references[branch][PAIR]
        ref = (selection.scalar(int(r[0])) - selection.scalar(int(r[1]))
               if branch == "fp16" else F(float(r[0])) - F(float(r[1])))
        original = F(row["retained_actual_margin"])
        require(row["pair"] == PAIR and F(row["substituted_actual_margin"]) == old
                and F(row["retained_reference_margin"]) == ref
                and F(row["required_direction"]) == ref - original,
                "predeclared original-reference direction splice")
        product = (new - original) * (ref - original)
        result.append({**row, "value_content_margin": str(old),
                       "counterfactual_margin": str(new),
                       "boundary_only_delta": str(new - old),
                       "original_baseline_delta": str(new - original),
                       "counterfactual_reference_error": str(new - ref),
                       "direction_product": str(product), "direction_pass": product > 0})
    return result


def classify(rows):
    require([r["control"] for r in rows] == list(CONTROLS), "control order/census")
    for row in rows:
        require(row["S18"]["status"] in {"PASS", "FAIL"}
                and [m["reference_branch"] for m in row["paired_margins"]]
                == ["fp16", "binary64"], "incomplete numerical evidence")
    return "SUPPORTED" if all(r["S18"]["status"] == "PASS"
                             and all(m["direction_pass"] for m in r["paired_margins"])
                             for r in rows) else "REJECTED"


def run_control(control, raw, retained, classified, reference, references, operands, table, oracle, audit):
    require(control == retained["control"] == classified["control"], "control splice")
    protected = {key: digest(value) for key, value in raw.items()}
    words, gate = intervene(raw["stage18"], reference, retained["S18"], control, table)
    require(gate["coordinates"] == classified["failure_coordinates"], "bridge family changed")
    audit["controls"].append(control)
    start = time.monotonic()
    audit["final_rmsnorm_invocations"] += 1
    normalized, details = suffix.rmsnorm(words, operands[0])
    midpoint = time.monotonic()
    audit["pair_head_invocations"] += 1
    logits = suffix.logits(normalized, operands[1])
    end = time.monotonic()
    oracle.verify_suffix(words, operands, normalized, details, logits)
    margins = paired_margins(logits, raw["final_logits"], classified["paired_margins"], references)
    require(protected == {k: digest(v) for k, v in raw.items()}, "protected retained array changed")
    payload = {**raw, "counterfactual_stage18": words,
               "counterfactual_final_rmsnorm": normalized, "counterfactual_pair_logits": logits}
    return payload, {"control": control, "S18": gate, "paired_margins": margins,
                     "protected_arrays": protected, "protected_bytes_unchanged": True,
                     "independent_suffix_oracle": "PASS", "rmsnorm_integer_details": details,
                     "timing_seconds": {"rmsnorm": midpoint - start, "pair_head": end - midpoint}}


def save(out, name, value):
    with (out / name).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def execute():
    require(Path.cwd() == ROOT and os.getuid() == 1000 and sys.dont_write_bytecode
            and not sys.flags.optimize and SOURCE.parent == ROOT / "ace3/model/candidates",
            "isolated source/account/bytecode context")
    require(subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
            == "argus/full-projection", "isolated branch")
    require(subprocess.run(["git", "check-ignore", "--quiet", str(OUTPUT)]).returncode == 0,
            "output must be ignored")
    require(not OUTPUT.exists() and not OUTPUT.is_symlink(), "one-use output already occupied")
    OUTPUT.mkdir()  # Exclusive directory creation is the fixed one-use/concurrency claim.
    audit, rows = new_audit(), []
    result = {"mission_id": MISSION, "status": "UNKNOWN", "non_admission": True,
              "normal_independent_review": "REQUIRED", "automatic_replay_authorized": False,
              "boundary": BOUNDARY, "science_budget": 1, "science_invocations": 0,
              "controls": rows, "audit": audit}
    records = []
    try:
        tests = module_from(TEST, NAME + "_tests")
        oracle = module_from(ORACLE, NAME + "_oracle")
        sources = {str(SOURCE), str(TEST), str(ORACLE)}
        sources.update(str(Path(m.__file__).resolve()) for n, m in tuple(sys.modules.items())
                       if n.startswith("ace3.") and getattr(m, "__file__", None))
        source_pins = [pin(Path(p)) for p in sorted(sources)]
        save(OUTPUT, "command.json", {
            "argv": sys.argv, "cwd": str(ROOT), "uid": os.getuid(), "pid": os.getpid(),
            "executable": sys.executable, "timestamp": time.time(), "mission_id": MISSION,
            "sources": source_pins, "science_budget": 1,
            "host_controls": "Existing Manager locks/account/role/model/access unchanged",
            "claim": "fixed output mkdir, exclusive files, no retry or alternate output",
            "validation_command": "in-memory compile + unittest.loadTestsFromModule(focused test)"})
        compiled = []
        for path in (SOURCE, TEST):
            compile(path.read_bytes(), str(path), "exec")
            compiled.append(str(path))
        text = io.StringIO()
        checked = unittest.TextTestRunner(stream=text, verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromModule(tests))
        save(OUTPUT, "validation.json", {"compiled": compiled, "tests": checked.testsRun,
                                       "failures": len(checked.failures),
                                       "errors": len(checked.errors), "skips": len(checked.skipped),
                                       "log": text.getvalue()})
        require(checked.wasSuccessful() and checked.testsRun > 0 and not checked.skipped,
                "focused compilation/tests failed; no science")
        with guarded(OUTPUT, audit):
            closure, failure, report, ref, refs, operands, records = inputs()
            result.update({"input_pins": records, "source_pins": source_pins,
                           "thresholds": closure["thresholds"],
                           "historical_failure": closure["historical_failure"],
                           "historical_science_status": closure["science_status"]})
            table = selection.finite_table()
            prepared = []
            for row in failure["controls"]:
                with np.load(io.BytesIO(selection.bound(row["raw_payload"])), allow_pickle=False) as z:
                    arrays = {key: z[key] for key in z.files}
                for array in arrays.values():
                    array.flags.writeable = False
                prepared.append(arrays)
            require(len(prepared) == 9, "retained payload census")
            result["science_invocations"] = 1
            for control, raw, retained, classified in zip(
                    CONTROLS, prepared, closure["controls"], report["controls"], strict=True):
                payload, row = run_control(control, raw, retained, classified, ref, refs,
                                           operands, table, oracle, audit)
                path = OUTPUT / (control + ".npz")
                with path.open("xb") as stream:
                    np.savez(stream, **payload)
                row["payload"] = pin(path)
                rows.append(row)
            for record in [*records, *source_pins]:
                selection.bound(record)
            require(audit["forbidden_events"] == [] and audit["controls"] == list(CONTROLS)
                    and audit["final_rmsnorm_invocations"] == audit["pair_head_invocations"] == 9,
                    "execution census defect")
            require(audit["writes"] == [str(OUTPUT / (c + ".npz")) for c in CONTROLS],
                    "payload write census defect")
            result.update({"status": classify(rows), "protected_input_files_unchanged": True})
    except Exception as error:
        # A consumed defect is explicit UNKNOWN, never numerical rejection or replay.
        result.update({"status": "UNKNOWN", "error_type": type(error).__name__, "error": str(error)})
    result["evidence_write_census"] = sorted(p.name for p in OUTPUT.iterdir()) + ["result.json"]
    save(OUTPUT, "result.json", result)
    print(json.dumps({"status": result["status"], "output": str(OUTPUT),
                      "controls": len(rows), "error": result.get("error")}))
    return 1 if result["status"] == "UNKNOWN" else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", required=True, action="store_true")
    parser.parse_args()
    return execute()


if __name__ == "__main__":
    raise SystemExit(main())
