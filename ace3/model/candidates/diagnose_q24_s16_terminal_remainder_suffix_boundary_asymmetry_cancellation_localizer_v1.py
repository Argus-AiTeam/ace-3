"""Retained-only outer-pair suffix localization; never reconstruct lost operands."""

import argparse
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[3]
NAME = Path(__file__).stem
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
STORE = Path("/home/argustest/.argus-skill-ace3")
INPUT_IDENTITY = "50bea3d7df449900a07fd1fc1d049b95540f65ebff9b879baffb2a11f4bac407"
CONTROLS = (
    "frozen_inherited", "frozen_inherited_o", "frozen_inherited_down",
    "frozen_inherited_o_down", "scratch", "scratch_down", "mapped62",
    "mapped_all", "inherited_native",
)
POLARITIES = ("forward", "reverse")
PAIRS = ((319, 34319), (34319, 319))
BRANCHES = ("fp16", "binary64")
SPECS = (
    {
        "dose": "sixteenth", "suffix": "sixteenth_vector_dose_contrast_v1",
        "session": "d214d4cc-fbc5-4272-b906-c012f40c7f6d",
        "call": "call_ZW5adPPhqLooKOOTCxvrEPop", "mission": "453240259c8d",
        "receipt": "bc581f95e342d490e900b2827cfdb5047973849c2f7bdf7ea17d715d5f3a6e8f",
        "source": "0e2a2c3fedfb9f8e1df9a9aec5da17a20f0b570418108d82dab66f91b1e47a8b",
        "test": "0edd081c39b97770eaaeac69d2e454a81c0e036f9fa896323ca3ea61753e70e7",
        "review": "2c073e860319127a8df9478825adc5ddfd9a9fccd5539a54afe731e81ada6b53",
        "command": "c8578298fb9e62ebbc91f9e4e429c4383e3b886fb4da9bd688f07bbc24313083",
        "outcome": ("supported", 72, 72), "changed_words": 827,
        "comparators": ("eighth", "quarter", "half", "full"),
    },
    {
        "dose": "thirtysecond", "suffix": "thirtysecond_vector_dose_contrast_v1",
        "session": "94dd5886-094c-4234-937d-3864ece95a91",
        "call": "call_H1TDNvYkLzJ0aIJ4YO8D9LsE", "mission": "b5d8f57125c8",
        "receipt": "ad375f7730b93efc4a5b91cf88ebfeb03511b72a488dd22bb8ce4b5fdcc4c7d9",
        "source": "16264e95f1afb2cf7dcc327db293ab0406916b71e73d6c258d6b4878a52cb5ad",
        "test": "dc4b83a350d2e25fd4ac41ca063fe442ad8c06f96d3fdcef2a8946d2aa3833b9",
        "review": "927a04873d38749ce7e929e183df17e1e4c2523c7ec3801e1eba2851d3decead",
        "command": "efcdbd5b9213f4d075141fbbe7c03f1eccc526fd3aff1b8e626f5409700760ec",
        "outcome": ("rejected", 40, 40), "changed_words": 742,
        "comparators": ("sixteenth", "eighth", "quarter", "half", "full"),
    },
    {
        "dose": "three_sixtyfourths",
        "suffix": "three_sixtyfourths_vector_dose_bracket_v1",
        "session": "32dae28e-eb59-43e5-a3f1-cf7380230f35",
        "call": "call_c2VB09Msd2dcjtLxlHaSsu3E", "mission": "f3d86d60d275",
        "receipt": "f4984358e7c3c387ed93dfd8679afd4a1263edabb4065b6ccae340ac7352ef2e",
        "source": "73e0f4cb94dd93ffea0ff610a14ddc90cd16da4a7c0b728fff030362adf69f75",
        "test": "aaa8a9b702cb2da78089a0e1ea78499759259dd81db7e31969bedf3e73788de0",
        "review": "6669f0c0aa68a85267235cfeedf2c2c7939355cfe68f9a72663e9a490e7c63cd",
        "command": "888a184ec0d51a10caa113199454a7614167590779dd28e4c41d6234c3600279",
        "outcome": ("rejected", 72, 36), "changed_words": 795,
        "comparators": ("thirtysecond", "sixteenth", "eighth", "quarter", "half", "full"),
    },
)
PREREGISTRATION = {
    "hypothesis": "Forward saturation and reverse recovery are determined at suffix "
                  "operand conversion, before final-RMSNorm/head remainders.",
    "SUPPORTED": "Exact retained decomposition localizes both distinctions before "
                 "final-RMSNorm/head remainders with every closure and binding passing.",
    "REJECTED": "Exact closures localize the distinction only to final-RMSNorm "
                "scale/cancellation or head accumulation/RNE, or contradict the prediction.",
    "UNKNOWN": "Concrete missing/malformed retained bindings or integrity failure only. "
               "Counts and summary margins cannot supply absent exact components.",
    "termination": "Terminate this branch for every outcome; no successor or dose search.",
}
MISSING = {
    "operand_conversion": "Only changed_coordinate_counts retained; coordinate identities, "
                          "exact targets, operand words and conversion remainders absent.",
    "final_rmsnorm": "Per-control/polarity final-RMSNorm vectors, scales and exact "
                     "vector/scale/interaction/cancellation terms absent from capture.",
    "selected_head": "Per-row pre-conversion accumulations and accumulation/RNE "
                     "remainders absent from capture.",
    "original_input_bindings": "Only protected_input_identity retained; original-input "
                               "reference, source/operand/state/KV/lineage members absent.",
    "producer_environment": "Launcher command retained, but inherited producer "
                            "environment and complete native stdout/stderr not archived.",
}
FLAGS = {
    "CPU_only": True, "retained_only": True, "historical_failures_preserved": True,
    "original_global_reference_unchanged": True, "Q24_wider_than_FP16": True,
    "native_S16_RTZ_unchanged": True, "INT4_weights_unchanged": True,
    "FP16_operator_and_KV_boundaries_unchanged": True,
    "thresholds_unchanged": True, "candidate_admitted": False,
    "strict_FP16_state_claim": False, "new_token_claim": False,
    "full_model_claim": False, "precision_or_scale_expansion": False,
    "causal_boundary_identified": False, "successor_scheduled": False,
}
AUDIT_KEYS = (
    "prefix_dispatches", "admission_dispatches", "reference_producer_dispatches",
    "native_decoder_dispatches", "closed_producer_dispatches", "GPU_dispatches",
    "RTL_hardware_dispatches", "service_dispatches", "subprocess_dispatches",
    "evidence_writes", "outside_attempt_evidence_writes", "blocked_attempts",
)


class IntegrityError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise IntegrityError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def binding(path):
    path = Path(path)
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": digest(data)}


def read_bound(pin):
    path = Path(pin["path"])
    raw = path.read_bytes()
    require(digest(raw) == pin["sha256"], f"SHA256 mismatch: {path}")
    if "bytes" in pin:
        require(len(raw) == pin["bytes"], f"byte count mismatch: {path}")
    return raw


def unique_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON member: {key}")
            result[key] = value
        return result

    return json.loads(raw, object_pairs_hook=pairs)


@contextmanager
def retained_only(audit):
    active = True

    def refuse(label):
        audit["blocked_attempts"] += 1
        raise IntegrityError("retained-only guard: " + label)

    def hook(event, args):
        if not active:
            return
        if event == "open":
            _, mode, flags = args
            if (mode and any(c in mode for c in "wax+")) or (
                flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            ):
                refuse("evidence write")
        if event in {
            "os.mkdir", "os.remove", "os.rename", "os.rmdir", "os.link", "os.symlink",
            "os.truncate", "os.chmod", "os.chown", "os.utime", "os.chdir",
            "os.putenv", "os.unsetenv", "os.system", "os.fork", "os.forkpty",
        } or event.startswith(("subprocess.", "socket.", "ctypes.", "os.exec", "os.spawn")):
            refuse(event)
        if event == "import":
            top = args[0].split(".")[0]
            if top not in sys.stdlib_module_names:
                refuse("non-stdlib import " + args[0])

    def profile(frame, event, arg):
        if event == "call":
            module = frame.f_globals.get("__name__", "")
            if module.startswith("ace3.") and module not in {MODULE, __name__}:
                refuse("closed project computation " + module)

    previous = sys.getprofile()
    sys.addaudithook(hook)
    sys.setprofile(profile)
    try:
        yield
    finally:
        sys.setprofile(previous)
        active = False


def extract_receipt(raw, spec):
    events = [unique_json(line) for line in raw.splitlines()]
    matches = [e["data"] for e in events if e["type"] == "tool.execution_complete"
               and e["data"]["toolCallId"] == spec["call"]]
    require(len(matches) == 1 and matches[0]["success"] is True,
            "missing/duplicate/failed dose completion: " + spec["dose"])
    requests = [t for e in events if e["type"] == "assistant.message"
                for t in e["data"].get("toolRequests", [])
                if t.get("toolCallId") == spec["call"]]
    require(len(requests) == 1 and requests[0]["name"] == "bash",
            "missing/duplicate dose command: " + spec["dose"])
    command = requests[0]["arguments"]["command"]
    require(digest(command.encode()) == spec["command"], "retained command mismatch")
    text = matches[0]["result"]["content"]
    marker = '{"changed_coordinate_counts":'
    require(text.count(marker) == 1, "missing/duplicate retained summary")
    start = text.index(marker)
    _, end = json.JSONDecoder().raw_decode(text[start:])
    summary_bytes = text[start:start + end].encode()
    return unique_json(summary_bytes), {
        "command": command, "command_sha256": digest(command.encode()),
        "tool_content_sha256": digest(text.encode()),
        "summary_sha256": digest(summary_bytes),
        "producer_environment": {"status": "UNKNOWN", "reason": MISSING["producer_environment"]},
    }


def authenticate(spec):
    stem = "diagnose_q24_s16_terminal_remainder_" + spec["suffix"]
    paths = {
        "source": ROOT / f"ace3/model/candidates/{stem}.py",
        "test": ROOT / f"tests/test_{stem}.py",
        "review": STORE / f"projects/s-62150b05/handoffs/{spec['mission']}/round-0001.json",
        "receipt": STORE / f"copilot-home/session-state/{spec['session']}/events.jsonl",
    }
    pins = {key: {"path": str(path), "sha256": spec[key]} for key, path in paths.items()}
    data = {key: read_bound(pin) for key, pin in pins.items()}
    review = unique_json(data["review"])
    require((review["kind"], review["mission_id"], review["producer_role"],
             review["round"], review["review"]["status"]) ==
            ("round_reviewed_handoff", spec["mission"], "reviewer", 1, "done"),
            "independent review binding failed: " + spec["dose"])
    summary, capture = extract_receipt(data["receipt"], spec)
    compiled = summary["tests"]["compiled"]
    require(compiled == [
        {**pins[key], "bytes": len(data[key])} for key in ("source", "test")
    ], "retained compiled source/test binding failed")
    return summary, {"pins": pins, "capture": capture, "review_status": "done"}


def exact(value):
    require(isinstance(value, str), "exact retained rational must be a string")
    return Fraction(value)


def inspect_summary(spec, summary):
    require((summary["status"], summary["directional_agreements"],
             summary["strict_dose_agreements"]) == spec["outcome"], "historical outcome drift")
    require(summary["protected_input_identity"] == INPUT_IDENTITY, "protected identity drift")
    require((summary["native_exit"], summary["stdout_json_documents"],
             summary["contrast_count"], summary["retained_common_component"],
             summary["normal_host_review"]) == (0, 1, 72, "UNKNOWN", "REQUIRED"),
            "retained execution/scope census drift")
    require([summary["tests"][k] for k in ("executed", "errors", "failures", "skipped")]
            == [14, 0, 0, 0], "retained test census drift")
    require(summary["dispatch_and_write_audit"] == {
        "final_rmsnorm_invocations": 27, "selected_row_head_invocations": 27,
        "forbidden_calls": 0,
    }, "retained dispatch census drift")
    if spec["dose"] == "three_sixtyfourths":
        require([summary[k] for k in ("reverse_zero_obstructions",
                                      "recovered_reverse_zero_contrasts")] == [32, 32],
                "historical reverse obstruction drift")
        require(all(v == (k in {"historical_failures_preserved",
                               "original_global_reference_unchanged"})
                    for k, v in summary["flags"].items()), "retained non-admission flag drift")
    fields = ["control", "polarity", "left_id", "right_id", "branch",
              "predicted_delta", "observed_delta",
              *("same_polarity_" + d + "_dose_delta" for d in spec["comparators"]),
              "retained_margin_change", "intervened_margin_change"]
    require(summary["contrast_fields"] == fields, "retained contrast schema drift")
    require(len(summary["contrast_values"]) == 36, "retained row count drift")
    require(all(len(row) == len(fields) for row in summary["contrast_values"]),
            "malformed retained row")
    rows = [dict(zip(fields, row, strict=True)) for row in summary["contrast_values"]]
    require([tuple(row[k] for k in fields[:5]) for row in rows] == [
        (c, p, left, right, "binary64") for c in CONTROLS for p in POLARITIES
        for left, right in PAIRS
    ], "retained row identity/order drift")
    require(summary["changed_coordinate_counts"] == {
        p: {c: spec["changed_words"] for c in CONTROLS} for p in POLARITIES
    }, "retained changed-word census drift")
    for row in rows:
        values = {k: exact(row[k]) for k in fields[5:]}
        require(values["observed_delta"] ==
                values["intervened_margin_change"] - values["retained_margin_change"],
                "retained binary64 margin closure failed")
    for left, right in zip(rows[::2], rows[1::2], strict=True):
        require(all(exact(left[k]) == -exact(right[k]) for k in fields[5:]),
                "retained ordered-pair reversal failed")
    return rows


def localize(summaries):
    require(len(summaries) == 3, "three retained dose receipts required")
    rows = [inspect_summary(spec, summary) for spec, summary in zip(SPECS, summaries, strict=True)]
    for index in range(36):
        require(len({dose[index]["retained_margin_change"] for dose in rows}) == 1,
                "original-reference margin-change binding drift")
        for newer, older in ((1, 0), (2, 0), (2, 1)):
            key = "same_polarity_" + SPECS[older]["dose"] + "_dose_delta"
            require(exact(rows[newer][index][key]) == exact(rows[older][index]["observed_delta"]),
                    "neighboring retained dose comparator drift")
    accounts = []
    for spec, dose_rows in zip(SPECS, rows, strict=True):
        for row in dose_rows:
            for branch in BRANCHES:
                margin = (
                    {"status": "retained", **{k: row[k] for k in (
                        "predicted_delta", "observed_delta", "retained_margin_change",
                        "intervened_margin_change")}, "closure_residual": "0"}
                    if branch == "binary64" else
                    {"status": "UNKNOWN", "reason": "FP16-reference rows excluded by retained launcher."}
                )
                accounts.append({
                    "dose": spec["dose"], **{k: row[k] for k in
                                           ("control", "polarity", "left_id", "right_id")},
                    "branch": branch, "margin": margin,
                    "retained_changed_word_count": spec["changed_words"],
                    "components": {k: {"status": "UNKNOWN", "reason": reason}
                                   for k, reason in MISSING.items()},
                })
    return {
        "classification": "UNKNOWN", "reason": "Exact decomposition bindings absent from "
        "the authenticated lossy dose captures; no causal localization is inferred.",
        "accounts": accounts, "retained_binary64_margin_closures": 108,
        "retained_ordered_pair_reversals": 54, "exact_decomposition_closures": 0,
        "missing_bindings": MISSING,
        "retained_summaries": dict(zip((s["dose"] for s in SPECS), summaries, strict=True)),
    }


def bind_command_sidecar(raw, environment, argv):
    tokens = shlex.split(raw.decode())
    count = len(environment)
    require(tokens[count:] == argv, "check sidecar argv mismatch")
    assignments = [token.split("=", 1) for token in tokens[:count]]
    require(len(assignments) == count and all(len(pair) == 2 for pair in assignments),
            "check sidecar environment assignments malformed")
    require(len({pair[0] for pair in assignments}) == count and
            dict(assignments) == environment, "check sidecar environment mismatch")
    return digest(raw)


def execution_identity(attempt):
    preflight_path = attempt / "preflight.json"
    preflight = unique_json(preflight_path.read_bytes())
    require(preflight["argv"] == [sys.executable, "-B", "-m", MODULE, "--check",
                                 "--attempt", str(attempt)], "check argv preflight mismatch")
    require(sys.argv[1:] == preflight["argv"][4:], "actual check argv mismatch")
    require(preflight["cwd"] == str(ROOT) == os.getcwd(), "check cwd mismatch")
    require(preflight["uid"] == os.getuid(), "check account mismatch")
    require(preflight["environment"] == dict(os.environ), "check environment mismatch")
    command = " ".join(f"{key}={shlex.quote(value)}"
                       for key, value in sorted(preflight["environment"].items()))
    command += " " + shlex.join(preflight["argv"])
    require(preflight["command"] == command, "check command identity mismatch")
    command_sha256 = bind_command_sidecar(
        (attempt / "check.command.txt").read_bytes(),
        preflight["environment"], preflight["argv"])
    for key, path in (("source", SOURCE), ("test", TEST), ("python", Path(sys.executable))):
        require(preflight[key] == binding(path), "current execution binding mismatch: " + key)
    require(preflight["validation"]["tests"] > 0 and
            all(preflight["validation"][k] == 0 for k in ("errors", "failures", "skipped")),
            "fresh task-native validation missing/failing/skipped")
    require([r["label"] for r in preflight["commands"]] == ["compile", "pytest"],
            "compile/pytest command receipts missing")
    expected_argv = [
        [sys.executable, "-B", "-m", "py_compile", str(SOURCE), str(TEST)],
        [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "--basetemp", str(attempt / "pytest-tmp"),
         "--junitxml", str(attempt / "pytest.xml"), str(TEST)],
    ]
    require([r["argv"] for r in preflight["commands"]] == expected_argv,
            "task-native validation command mismatch")
    for command in preflight["commands"]:
        require(command["exit_status"] == 0 and command["timed_out"] is False,
                "validation command failed")
        for pin in command["files"]:
            require(Path(pin["path"]).is_relative_to(attempt), "validation capture outside attempt")
            read_bound(pin)
    report_pin = preflight["pytest_report"]
    require(report_pin["path"] == str(attempt / "pytest.xml"), "pytest report path mismatch")
    report = ET.fromstring(read_bound(report_pin))
    suites = list(report.iter("testsuite"))
    require(bool(suites) and {
        key: sum(int(s.attrib[key]) for s in suites)
        for key in ("tests", "errors", "failures", "skipped")
    } == preflight["validation"], "pytest result/capture mismatch")
    return {
        "preflight": binding(preflight_path), **preflight,
        "command_sha256": command_sha256,
        "canonical_command_sha256": digest(command.encode()),
        "environment_sha256": digest(json.dumps(preflight["environment"],
                                                sort_keys=True).encode()),
    }


def check(attempt):
    audit = dict.fromkeys(AUDIT_KEYS, 0)
    result = {
        "diagnostic_id": NAME, "version": 1, "classification": "UNKNOWN", "status": "UNKNOWN",
        "lane_terminated": True, "normal_host_review": "REQUIRED",
        "flags": FLAGS, "preregistration": PREREGISTRATION,
        "dispatch_and_write_audit": audit,
    }
    try:
        with retained_only(audit):
            require(attempt.is_relative_to(ROOT / "build") and attempt.resolve() == attempt,
                    "canonical ignored-build attempt required")
            result["execution_identity"] = execution_identity(attempt)
            summaries, receipts = [], []
            for spec in SPECS:
                summary, receipt = authenticate(spec)
                summaries.append(summary)
                receipts.append(receipt)
            result["authenticated_receipts"] = receipts
            result.update(localize(summaries))
    except (IntegrityError, OSError, KeyError, TypeError, ValueError,
            ZeroDivisionError, ET.ParseError) as error:
        result["integrity_failure"] = {"type": type(error).__name__, "message": str(error)}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    args = parser.parse_args(argv)
    result = check(args.attempt)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return int("integrity_failure" in result)


if __name__ == "__main__":
    raise SystemExit(main())
