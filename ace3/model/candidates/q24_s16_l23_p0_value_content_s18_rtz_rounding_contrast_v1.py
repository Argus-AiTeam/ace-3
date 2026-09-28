"""One-use, non-admission Q24 S18 RTZ contrast; final norm/two-row head only.

Run the command.sh in the fixed attempt001 output directory. Compilation and
synthetic regressions precede the single retained-data invocation. References
only measure outputs: rtz_view accepts I/Z, never references or failed indices.
The retained RNE H and wider Q24 state remain separate, immutable operands.
UNKNOWN consumes the attempt, as do SUPPORTED and numerical REJECTED.
"""

import argparse
from contextlib import contextmanager
from fractions import Fraction as F
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

import numpy as np

from ace3.model.candidates import q24_s16_l23_p0_value_content_s18_family_counterfactual_v1 as retained
from ace3.model.candidates import residual_exact_grid_q24_runtime_v1 as state


ROOT = retained.ROOT
SOURCE = Path(__file__).resolve()
NAME = SOURCE.stem
TEST = ROOT / f"tests/test_{NAME}.py"
OUTPUT = ROOT / "build" / (NAME + "_attempt001")
MISSION = "9602adc7f96e"
CONTROLS, PAIR = retained.CONTROLS, retained.PAIR
FAMILY = retained.OUTPUT / "result.json"
FAMILY_HASH = "d46974e73ecb306c4371e59968572f16476ebf53876b28cacee231fbbf76dc82"
REVIEWS = {
    "9196b8c0777c": (1, "eea66a0edaa27ab6a7970b4d0a2307bca6f3772ffd251aafa5d6064c0816e3b4"),
    "6af2b58b51f6": (1, "9356793dbd23b3e753e73148111a20bdb07c887031c4bb044d87724fd0ca0cb3"),
    "2eb443534561": (3, "6f8c04031b9d6381de9a1554086339c0c7b87c00bf46afb45e9e6675ca65a8e8"),
}
BOUNDARY = (
    "CPU-only, width896/L23/P0, pair[34319,13], isolated non-admission contrast. "
    "All 896 S18 words derive from retained Q24 I/Z by exact RTZ, independent "
    "of references, intervals, failing indices, logits and margin targets. "
    "Only final RMSNorm and two tied-head rows are recomputed. Original global "
    "FP16/binary64 references are comparators only. The value-content operand "
    "substitution ancestry remains; this does not establish native-V repair. "
    "Native S16 RTZ, G128 asymmetric INT4/GEMM nibble order/no qzero plus-one, "
    "FP16 scales/operators/KV and wider Q24 state are unchanged. No strict-"
    "FP16-state W4A16, new-token/full-model admission, prefix/admission/reference/"
    "closed-producer/native L0-L17 replay, hardware or precision/scale expansion."
)
require, pin = retained.require, retained.pin
_active = None
_hook_installed = False


def rtz_word(integer, tag):
    """Integer bit truncation, including exact subnormals and canonical -0."""
    state._record(integer, tag)
    sign = 0x8000 if integer < 0 or tag else 0
    magnitude = abs(integer)
    if magnitude < 1024:
        return sign | magnitude
    shift = magnitude.bit_length() - 11
    return sign | ((shift + 1) << 10) | ((magnitude >> shift) - 1024)


def rtz_view(integers, tags):
    require(integers.shape == tags.shape == (896,)
            and integers.dtype.str == "<i8" and tags.dtype.str == "|u1",
            "Q24 I/Z width/dtype")
    return np.asarray([rtz_word(int(i), int(z)) for i, z in
                       zip(integers, tags, strict=True)], dtype="<u2")


def check_state(raw):
    for integer, zero, hidden in (
            ("input_i", "input_z", "input_hidden"),
            ("scratch_i", "scratch_z", "stage12"),
            ("output_i", "output_z", "stage18")):
        i, z, h = (raw[k] for k in (integer, zero, hidden))
        require(i.shape == z.shape == h.shape == (896,)
                and i.dtype.str == "<i8" and z.dtype.str == "|u1"
                and h.dtype.str == "<u2", "retained I/Z/H schema")
        require(all(state.fp16_view(int(a), int(b)) == int(c)
                    for a, b, c in zip(i, z, h, strict=True)), "retained I/Z/H disagreement")
    for kind, stage in (("k", "stage06"), ("v", "retained_stage07")):
        require(raw["input_cache_" + kind].shape == (0, 128)
                and raw["output_cache_" + kind].shape == (1, 128)
                and raw["output_cache_" + kind].dtype.str == "<u2"
                and np.array_equal(raw["output_cache_" + kind][0], raw[stage]),
                "retained P0 KV/source gate")
    require(np.array_equal(raw["stage03"], raw["retained_stage07"])
            and np.array_equal(raw["stage09"], np.full(14, 0x3c00, dtype="<u2")),
            "retained singleton/source lineage")


def compare(words, references):
    require(words.shape == references.shape == (896,) and words.dtype.str == "<u2"
            and references.dtype.str == "<f8", "S18 comparator schema")
    rows = [{"index": i, **retained.policy.evaluate_layer_final_output(
        actual_fp16_bits=int(w), reference_binary64_hex=float(r).hex())}
        for i, (w, r) in enumerate(zip(words, references, strict=True))]
    failures = sum(not r["accepted"] for r in rows)
    return {"status": "FAIL" if failures else "PASS", "failure_count": failures,
            "coordinates": rows, "excess_budget": "1/8"}


def run_control(control, raw, closure, bridge, reference, references, operands, oracle, audit):
    require(control == closure["control"] == bridge["control"], "control lineage splice")
    protected = {k: retained.digest(v) for k, v in raw.items()}
    check_state(raw)
    start = time.monotonic()
    words = rtz_view(raw["output_i"], raw["output_z"])
    oracle.verify_rtz(raw["output_i"], raw["output_z"], words)
    converted = time.monotonic()
    # Comparison occurs after all output bits have been fixed by I/Z.
    before = compare(raw["stage18"], reference)
    after = compare(words, reference)
    require(before["failure_count"] == closure["failure_count"],
            "closure S18 comparator splice")
    audit["controls"].append(control)
    audit["rtz_coordinates"] += len(words)
    audit["final_rmsnorm_invocations"] += 1
    begin_norm = time.monotonic()
    normalized, details = retained.suffix.rmsnorm(words, operands[0])
    begin_head = time.monotonic()
    audit["pair_head_invocations"] += 1
    logits = retained.suffix.logits(normalized, operands[1])
    end_head = time.monotonic()
    oracle.suffix_oracle.verify_suffix(words, operands, normalized, details, logits)
    margins = retained.paired_margins(
        logits, raw["final_logits"], bridge["paired_margins"], references)
    require(protected == {k: retained.digest(v) for k, v in raw.items()},
            "protected state/operand/KV bytes changed")
    return {**raw, "rtz_stage18": words, "rtz_final_rmsnorm": normalized,
            "rtz_pair_logits": logits}, {
        "control": control, "S18_before": before, "S18": after,
        "changed_coordinate_count": int(np.count_nonzero(words != raw["stage18"])),
        "paired_margins": margins, "protected_arrays": protected,
        "protected_bytes_unchanged": True, "independent_suffix_oracle": "PASS",
        "rtz_oracle": {"status": "PASS", "coordinates": 896,
                       "method": "independent exhaustive finite-half lattice floor of abs(I)/2^24",
                       "input_fields": ["output_i", "output_z"],
                       "rounding": "toward zero; canonical signed zero; no reference operands"},
        "rmsnorm_integer_details": details,
        "timing_seconds": {"rtz_and_oracle": converted - start,
                           "rmsnorm": begin_head - begin_norm, "pair_head": end_head - begin_head}}


def audit_event(event, args):
    if _active is None:
        return
    out, audit = _active
    denied = event.startswith(("subprocess.", "socket.", "ctypes."))
    denied |= event in {
        "os.system", "os.exec", "os.posix_spawn", "os.fork", "os.forkpty",
        "os.remove", "os.rename", "os.rmdir", "os.mkdir", "os.link", "os.symlink",
        "os.truncate", "os.chmod", "os.chown", "os.utime",
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
        raise retained.bridge.BindingError("forbidden experiment operation: " + event)


@contextmanager
def guarded(out, audit):
    global _active, _hook_installed
    require(_active is None and sys.getprofile() is None, "nested execution guard")
    if not _hook_installed:
        sys.addaudithook(audit_event)
        _hook_installed = True
    allowed = {
        str(Path(m.__file__).resolve()) for m in (
            retained, retained.bridge, retained.selection, retained.suffix,
            retained.policy, retained.fp16_adaptation_oracle,
            retained.streaming_lm_head_reference, retained.awq_bit_oracle, state)}
    allowed.update((str(SOURCE), str(TEST), str(retained.ORACLE)))
    forbidden = {
        "check", "main", "execute", "run", "diagnose", "stages", "_stages",
        "continuation_stages", "native_layer", "execute_layer", "execute_layers",
        "original_branches", "reference_arrays", "build_report", "intervene",
        "adjust", "nearest", "finite_table", "run_control", "guarded",
    }
    retained_helpers = {"inputs", "read_array", "paired_margins", "digest",
                        "<genexpr>", "<dictcomp>", "<listcomp>", "<setcomp>"}

    def profile(frame, event, arg):
        if event != "call":
            return
        filename, name = frame.f_code.co_filename, frame.f_code.co_name
        if not filename.startswith(str(ROOT) + "/"):
            return
        key = filename.removeprefix(str(ROOT) + "/") + ":" + name
        audit["project_calls"][key] = audit["project_calls"].get(key, 0) + 1
        if (filename not in allowed
                or (filename != str(SOURCE) and name in forbidden)
                or (filename == str(retained.SOURCE) and name not in retained_helpers)):
            audit["forbidden_events"].append(key)
            raise retained.bridge.BindingError("unapproved project dispatch: " + key)

    _active = (out, audit)
    sys.setprofile(profile)
    try:
        yield
    finally:
        sys.setprofile(None)
        _active = None


def new_audit():
    return {"scope": "process-local Python audit/project-call profile, not OS sandbox",
            "forbidden_events": [], "writes": [], "project_calls": {}, "controls": [],
            "rtz_coordinates": 0, "final_rmsnorm_invocations": 0, "pair_head_invocations": 0}


def inputs():
    family = json.loads(retained.bridge.authenticate(FAMILY, FAMILY_HASH))
    require(family["mission_id"] == "9196b8c0777c" and family["status"] == "REJECTED"
            and family["non_admission"] is True and family["science_invocations"] == 1
            and family["audit"]["forbidden_events"] == [], "rejected family identity/census")
    records = [pin(FAMILY)]
    for mission, (number, digest) in REVIEWS.items():
        path = retained.HANDOFFS / mission / f"round-{number:04d}.json"
        record = {"path": str(path), "sha256": digest}
        review = json.loads(retained.selection.bound(record))
        latest = retained.HANDOFFS / mission / "latest.json"
        require(json.loads(latest.read_bytes())["handoff"]["path"] == str(path)
                and review["mission_id"] == mission and review["producer_role"] == "reviewer"
                and review["review"]["status"] == "done", "independent dependency acceptance")
        records.extend((record, pin(latest)))
    for record in [*family["input_pins"], *family["source_pins"],
                   *(r["payload"] for r in family["controls"])]:
        retained.selection.bound(record)
        records.append(record)
    records.extend(pin(retained.OUTPUT / name) for name in ("command.json", "validation.json"))
    closure, failure, bridge, ref, refs, operands, bindings = retained.inputs()
    require(family["thresholds"] == closure["thresholds"], "unchanged scientific thresholds")
    return closure, failure, bridge, ref, refs, operands, [*records, *bindings]


def save(name, value):
    with (OUTPUT / name).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def execute():
    require(Path.cwd() == ROOT and os.getuid() == 1000 and sys.dont_write_bytecode
            and not sys.flags.optimize, "isolated source/account/bytecode context")
    require(subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
            == "argus/full-projection", "isolated branch")
    require(subprocess.run(["git", "check-ignore", "--quiet", str(OUTPUT)]).returncode == 0,
            "output must be ignored")
    require(OUTPUT.resolve() == OUTPUT and OUTPUT.is_dir()
            and {p.name for p in OUTPUT.iterdir()} == {"command.sh"},
            "one-use output occupied or command sidecar missing")
    command = (OUTPUT / "command.sh").read_text()
    require(command == f"python3 -B -m ace3.model.candidates.{NAME} --execute\n",
            "command sidecar mismatch")
    mission_path = retained.HANDOFFS / MISSION / "mission.json"
    mission = json.loads(mission_path.read_bytes())
    require(mission["mission_id"] == MISSION and mission["scope"] == "bounded"
            and mission["node_key"] == "value-content-s18-rtz-rounding-contrast"
            and mission["execution_workdir"] == str(ROOT)
            and "review:required" in mission["tags"] and mission["deps"] == [],
            "current native owner/dependency preflight")
    # Exclusive creation is the concurrency/one-use claim, before tests or science.
    save("command.json", {
        "mission_id": MISSION, "command_bytes": command, "argv": sys.argv,
        "cwd": str(ROOT), "uid": os.getuid(), "pid": os.getpid(),
        "executable": sys.executable, "timestamp": time.time(), "mission": pin(mission_path),
        "science_budget": 1, "authorization_id": "", "action": "",
        "host_controls": "Host-owned account/role/model/access and locks not changed",
        "claim": "exclusive command.json; fixed root, no retries or alternate outputs",
        "direction": "(new_margin-original_actual_margin)*(original_reference_margin-original_actual_margin)>0",
        "validation": "in-memory compile + focused unittest suite, once before science"})
    audit, rows = new_audit(), []
    result = {"mission_id": MISSION, "status": "UNKNOWN", "controls": rows, "audit": audit,
              "science_budget": 1, "science_invocations": 0, "non_admission": True,
              "normal_independent_review": "REQUIRED", "automatic_replay_authorized": False,
              "boundary": BOUNDARY}
    try:
        tests = retained.module_from(TEST, NAME + "_tests")
        sources = {str(SOURCE), str(TEST), str(retained.ORACLE)}
        sources.update(str(Path(m.__file__).resolve()) for n, m in tuple(sys.modules.items())
                       if n.startswith("ace3.") and getattr(m, "__file__", None))
        sources = [pin(Path(p)) for p in sorted(sources)]
        compiled = []
        for path in (SOURCE, TEST):
            compile(path.read_bytes(), str(path), "exec")
            compiled.append(str(path))
        text = io.StringIO()
        checked = unittest.TextTestRunner(stream=text, verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromModule(tests))
        save("validation.json", {
            "compiled": compiled, "tests": checked.testsRun, "failures": len(checked.failures),
            "errors": len(checked.errors), "skips": len(checked.skipped), "log": text.getvalue(),
            "source_pins": sources})
        require(checked.wasSuccessful() and checked.testsRun > 0 and not checked.skipped,
                "focused compile/tests failed; no science")
        with guarded(OUTPUT, audit):
            closure, failure, bridge, ref, refs, operands, records = inputs()
            records.extend((pin(mission_path), pin(OUTPUT / "command.sh")))
            result.update({"input_pins": records, "source_pins": sources,
                           "thresholds": closure["thresholds"],
                           "historical_failure": closure["historical_failure"],
                           "historical_science_status": closure["science_status"],
                           "historical_family_status": "REJECTED",
                           "operand_pins": [retained.digest(v) for v in operands]})
            prepared = []
            for row in failure["controls"]:
                with np.load(io.BytesIO(retained.selection.bound(row["raw_payload"])),
                             allow_pickle=False) as archive:
                    arrays = {k: archive[k] for k in archive.files}
                for array in arrays.values():
                    array.flags.writeable = False
                check_state(arrays)
                prepared.append(arrays)
            require(len(prepared) == 9, "retained payload census")
            result["science_invocations"] = 1
            for control, raw, old, classified in zip(
                    CONTROLS, prepared, closure["controls"], bridge["controls"], strict=True):
                payload, row = run_control(
                    control, raw, old, classified, ref, refs, operands, tests, audit)
                path = OUTPUT / (control + ".npz")
                with path.open("xb") as stream:
                    np.savez(stream, **payload)
                row["payload"] = pin(path)
                rows.append(row)
            for record in [*records, *sources]:
                retained.selection.bound(record)
            require(result["operand_pins"] == [retained.digest(v) for v in operands],
                    "final operands changed")
            require(audit["forbidden_events"] == [] and audit["controls"] == list(CONTROLS)
                    and audit["rtz_coordinates"] == 8064
                    and audit["final_rmsnorm_invocations"] == audit["pair_head_invocations"] == 9,
                    "dispatch census defect")
            require(audit["writes"] == [str(OUTPUT / (c + ".npz")) for c in CONTROLS],
                    "write census defect")
            require([r["control"] for r in rows] == list(CONTROLS), "result order")
            result["status"] = ("SUPPORTED" if all(
                r["S18"]["status"] == "PASS" and all(m["direction_pass"] for m in r["paired_margins"])
                for r in rows) else "REJECTED")
            result["protected_input_files_unchanged"] = True
    except Exception as error:
        # Consumed authentication/execution defects are visible and never retried.
        result.update({"status": "UNKNOWN", "error_type": type(error).__name__, "error": str(error)})
    result["evidence_write_census"] = sorted(p.name for p in OUTPUT.iterdir()) + ["result.json"]
    save("result.json", result)
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
