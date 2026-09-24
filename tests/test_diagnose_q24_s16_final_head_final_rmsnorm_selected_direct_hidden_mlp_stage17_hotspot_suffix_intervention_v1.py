"""Independent scalar oracles, mutation tests and one byte-sealed task launcher."""

import copy
from fractions import Fraction
import json
import os
from pathlib import Path
import struct
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_mlp_stage17_hotspot_suffix_intervention_v1 as diagnostic


capture_helper = diagnostic.capture


def scalar(bits):
    bits = int(bits)
    exponent, mantissa = (bits >> 10) & 31, bits & 1023
    assert exponent != 31
    value = (Fraction(mantissa, 1 << 24) if exponent == 0 else
             Fraction(1024 + mantissa) * Fraction(2) ** (exponent - 25))
    return -value if bits & 32768 else value


def round_word(value, negative_zero=False):
    sign = 32768 if value < 0 or value == 0 and negative_zero else 0
    value = abs(value)
    low, high = 0, 31743
    assert value <= scalar(high)
    while low < high:
        mid = (low + high) // 2
        if scalar(mid) < value:
            low = mid + 1
        else:
            high = mid
    choices = (low,) if low == 0 else (low - 1, low)
    return sign | min(choices, key=lambda k: (abs(scalar(k) - value), k & 1))


def verify_suffix(actual, arrays, reference, coordinate, operands):
    expected = actual["stage17"].copy()
    expected[coordinate] = reference["stage17"][coordinate]
    assert arrays["stage17"].tobytes() == expected.tobytes()
    for k, word in enumerate(expected):
        increment = scalar(word) * (1 << 24)
        assert increment.denominator == 1
        total = int(actual["scratch_i"][k]) + increment.numerator
        zero = int(total == 0 and actual["scratch_i"][k] == 0
                   and actual["scratch_z"][k] == 1 and int(word) == 32768)
        assert int(arrays["output_i"][k]) == total
        assert int(arrays["output_z"][k]) == zero
        assert int(arrays["stage18"][k]) == round_word(Fraction(total, 1 << 24), bool(zero))
        if k != coordinate:
            for key in ("output_i", "output_z", "stage18"):
                assert arrays[key][k] == actual[key][k]
    for row, weights in enumerate(operands[1].view("<u2")):
        total = sum((scalar(a) * scalar(b) for a, b in
                     zip(arrays["final_rmsnorm"], weights, strict=True)), Fraction())
        assert int(arrays["pair_logits"][row]) == round_word(total)


@pytest.fixture
def synthetic():
    controls, originals = [], []
    for control in diagnostic.CONTROLS:
        branches, raw = {}, {}
        for branch in diagnostic.BRANCHES:
            cells = [{"coordinate": k, "weighted_components": {"mlp_stage17": "-2"},
                      "hidden_components": {"mlp_stage17": "-1"},
                      "weight_times_reference_anchor_times_row_difference": "2"}
                     for k in (2, 7)]
            ranking = [{"coordinate": k, "signed": "-2", "absolute": "2"} for k in (2, 7)]
            branches[branch] = {"per_component_coordinate_hotspots": {
                "mlp_stage17": {"ranking": ranking, "maximum_ties": ranking}}}
            raw[branch] = {"selected_coordinates": cells,
                           "residual_internal_reference": "original_input_L23_fp16",
                           "hidden_reference": "original_input_L23_" + branch,
                           "binary64_internal_stages": "NOT_RETAINED_NO_RECONSTRUCTION"}
        controls.append({"control": control, "pairs": [{"left_id": 34319, "right_id": 13,
                                                       "branches": branches}]})
        originals.append({"control": control, "pairs": [{"left_id": 34319, "right_id": 13,
                                                        "branches": raw}]})
    return {"report": {"controls": controls}, "retained_50f": {"report": {"controls": originals}}}


def test_exact_mass_and_tie_before_execution(synthetic, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("selection must not run operators")
    monkeypatch.setattr(diagnostic.native.state, "add", forbidden)
    plan = diagnostic.preregister(synthetic)
    assert plan["coordinate"] == 2
    assert plan["selection_masses"] == {"2": "36", "7": "36"}
    assert [r["predicted_delta"] for r in plan["rows"]] == ["2"] * 18
    moved = copy.deepcopy(synthetic)
    moved["report"]["controls"].reverse()
    with pytest.raises(ValueError, match="census"):
        diagnostic.preregister(moved)


@pytest.mark.parametrize("defect", ["mass", "duplicate", "reference", "zero"])
def test_parent_prediction_mutations_unknown(synthetic, defect):
    table = synthetic["report"]["controls"][0]["pairs"][0]["branches"]["fp16"]
    cells = synthetic["retained_50f"]["report"]["controls"][0]["pairs"][0]["branches"]["fp16"]
    if defect == "mass":
        table["per_component_coordinate_hotspots"]["mlp_stage17"]["ranking"][0]["absolute"] = "3"
    elif defect == "duplicate":
        cells["selected_coordinates"].append(copy.deepcopy(cells["selected_coordinates"][0]))
    elif defect == "reference":
        cells["hidden_reference"] = "local_actual_input"
    else:
        for k in range(2):
            cells["selected_coordinates"][k]["weighted_components"]["mlp_stage17"] = "0"
            cells["selected_coordinates"][k]["hidden_components"]["mlp_stage17"] = "0"
            row = table["per_component_coordinate_hotspots"]["mlp_stage17"]["ranking"][k]
            row.update(signed="0", absolute="0")
    with pytest.raises(ValueError):
        diagnostic.preregister(synthetic)


def fixture_arrays():
    words = np.full(896, 15360, dtype="<u2")
    actual = {"stage17": words.copy(), "stage18": words.copy(),
              "scratch_i": np.zeros(896, dtype="<i8"), "scratch_z": np.zeros(896, dtype="u1"),
              "stage12": np.zeros(896, dtype="<u2"),
              "output_i": np.full(896, 1 << 24, dtype="<i8"), "output_z": np.zeros(896, dtype="u1"),
              "stage03": np.zeros(128, dtype="<u2"), "stage05": np.zeros(128, dtype="<u2"),
              "output_cache_k": np.zeros((1, 128), dtype="<u2"),
              "output_cache_v": np.zeros((1, 128), dtype="<u2")}
    reference = {"stage17": words.copy()}
    reference["stage17"][2] = 16384
    return actual, reference


@pytest.mark.parametrize("key", ["stage03", "scratch_i", "output_cache_k", "stage17"])
def test_one_operand_and_protected_state(key):
    actual, reference = fixture_arrays()
    arrays = diagnostic.prepare(actual, reference, 2)
    assert np.flatnonzero(arrays["stage17"] != actual["stage17"]).tolist() == [2]
    assert not any(v.flags.writeable for v in arrays.values())
    bad = {k: v.copy() for k, v in arrays.items()}
    bad[key].flat[0] += 1
    with pytest.raises(ValueError):
        diagnostic.protected(actual, bad, reference, 2)


def directional_rows(delta="1", mlp="1"):
    return [{"control": c, "branch": b, "predicted_delta": "1",
             "mlp_maxima_delta": mlp, "margin_delta": delta,
             "direction_observed": Fraction(delta) > 0 and mlp == "1"}
            for c in diagnostic.CONTROLS for b in diagnostic.BRANCHES]


@pytest.mark.parametrize("delta,mlp,gate,expected", [
    ("1", "1", "PASS", "SUPPORTED"), ("0", "1", "PASS", "REJECTED"),
    ("-1", "1", "PASS", "REJECTED"), ("1", "-1", "PASS", "REJECTED"),
    ("1", "1", "FAIL", "REJECTED"), ("1", "2", "PASS", "REJECTED"),
])
def test_outcome_distinction(delta, mlp, gate, expected):
    reports = {c: {"stage": 18, "status": gate} for c in diagnostic.CONTROLS}
    assert diagnostic.classify(directional_rows(delta, mlp), reports) == expected


def test_zero_prediction_and_incomplete_census_unknown():
    rows = directional_rows()
    reports = {c: {"stage": 18, "status": "PASS"} for c in diagnostic.CONTROLS}
    rows[0]["predicted_delta"] = "0"
    with pytest.raises(ValueError, match="zero"):
        diagnostic.classify(rows, reports)
    with pytest.raises(ValueError, match="census"):
        diagnostic.classify(rows[:-1], reports)


def test_independent_bit_rounding_and_suffix_oracles():
    for word in (0, 1, 1023, 1024, 15360, 31743, 32768, 48128):
        assert float(scalar(word)) == struct.unpack("<e", struct.pack("<H", word))[0]
        assert round_word(scalar(word), word == 32768) == word
    assert round_word((scalar(15360) + scalar(15361)) / 2) == 15360
    actual, reference = fixture_arrays()
    arrays = diagnostic.prepare(actual, reference, 2)
    scratch = {"i": arrays["scratch_i"], "z": arrays["scratch_z"], "h": arrays["stage12"]}
    changed = diagnostic.native.state.add(scratch, arrays["stage17"])
    arrays.update(output_i=changed["i"], output_z=changed["z"], stage18=changed["h"])
    operands = (np.ones(896, dtype="<f2"), np.full((2, 896), 0.125, dtype="<f2"))
    arrays["final_rmsnorm"], _ = diagnostic.parent.rmsnorm(arrays["stage18"], operands[0])
    arrays["pair_logits"] = diagnostic.parent.logits(arrays["final_rmsnorm"], operands[1])
    verify_suffix(actual, arrays, reference, 2, operands)
    ones, _ = diagnostic.parent.rmsnorm(np.full(896, 15360, dtype="<u2"), operands[0])
    assert np.all(ones == 15360)
    arrays["output_i"][2] += 1
    with pytest.raises(AssertionError):
        verify_suffix(actual, arrays, reference, 2, operands)


def test_forbidden_operators_and_writes(tmp_path):
    audit = dict.fromkeys(diagnostic.COUNTS, 0)
    with diagnostic.suffix_only(audit, {}, (None, None)):
        for call in (
            lambda: diagnostic.native.projection(None, None, None),
            lambda: diagnostic.direct.report(None),
            lambda: diagnostic.guards.check(),
            lambda: (tmp_path / "forbidden").write_text("no"),
        ):
            with pytest.raises(RuntimeError):
                call()
        with pytest.raises(ValueError, match="S18"):
            diagnostic.native.state.add({}, None)
    assert audit["forbidden_calls"] == 4
    assert not (tmp_path / "forbidden").exists()


def test_missing_capture_and_authentication_unknown(monkeypatch, capsys):
    def fail():
        raise ValueError("authentication defect")
    monkeypatch.setattr(diagnostic, "check", fail)
    assert diagnostic.main(["--check"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "UNKNOWN" and result["flags"] == diagnostic.FLAGS


def test_uncaptured_check_refused(monkeypatch):
    monkeypatch.setattr(os, "readlink", lambda path: "/dev/null")
    with pytest.raises(ValueError, match="capture"):
        diagnostic.capture_preflight()


def verify_members(directory, receipt):
    for result in receipt["results"]:
        paths = [Path(pin["path"]) for pin in result["files"]]
        assert paths == [directory / (result["label"] + suffix) for suffix in capture_helper.SUFFIXES]
        data = [diagnostic.bound_bytes(pin) for pin in result["files"]]
        assert data[0] == (result["command"] + "\n").encode()
        assert json.loads(data[1]) == result["argv"]
        assert json.loads(data[2]) == receipt["preflight"]
        assert data[4] == b"" and result["exit_status"] == 0 and not result["timed_out"]
        assert data[5] == (b"COMMAND\n" + data[0] + b"ENVIRONMENT\n" + data[2]
                           + b"\nSTDOUT\n" + data[3] + b"\nSTDERR\n" + data[4]
                           + b"\nEXIT_STATUS=0\nTIMED_OUT=False\n")


def capture_run(directory):
    root, python = diagnostic.ROOT, diagnostic.PYTHON
    if (Path.cwd(), os.getuid(), sys.executable) != (root, 1000, python):
        raise RuntimeError("account/workdir/interpreter gate")
    directory = Path(directory).absolute()
    if (directory.parent.parent != root / "build" or directory.name != "run"
            or directory.resolve() != directory):
        raise RuntimeError("fresh build/<attempt>/run required")
    with capture_helper.same_scope(root, diagnostic.MODULE):
        directory.mkdir(exist_ok=False)
        sources = [capture_helper.binding(diagnostic.SOURCE), capture_helper.binding(diagnostic.TEST)]
        preflight = {
            "cwd": str(root), "uid": os.getuid(), "python": python, "python_version": sys.version,
            "environment": diagnostic.ENVIRONMENT, "sources": sources,
            "executable": capture_helper.binding(Path(python).resolve()),
            "independent_host_review": "REQUIRED", "scope": diagnostic.BOUNDARY,
            "model_or_service_calls_authorized": 0, "mission_id": "c6a656a6533d",
            "role": "engineer", "attempt": str(directory.parent),
            "command_budget": {"compile": 1, "pytest": 1, "check": 1},
            "capture_implementation": capture_helper.implementation_pins(),
        }
        results, failure = [], None

        def run(label, argv):
            return capture_helper.run_command(directory, label, argv, preflight, results)

        try:
            assert run("branch", ["git", "branch", "--show-current"]) == b"argus/full-projection\n"
            run("ignored-build", ["git", "check-ignore", str(directory)])
            probe = (
                "import json,os,subprocess; "
                "rows=subprocess.check_output(['ps','-eo','pid=,args='],text=True).splitlines(); "
                f"module={diagnostic.MODULE!r}; test={str(diagnostic.TEST)!r}; "
                "matches=[r for r in rows if int(r.split(None,1)[0]) != os.getpid() "
                "and ((' -m '+module+' ') in r or (' -m pytest ' in r and test in r))]; print(json.dumps(matches))"
            )
            assert json.loads(run("concurrency", [python, "-B", "-c", probe])) == []
            code = ("from pathlib import Path; "
                    f"paths={[str(diagnostic.SOURCE), str(diagnostic.TEST)]!r}; "
                    "[compile(Path(p).read_bytes(),p,'exec') for p in paths]; print('compiled 2 Python files')")
            run("compile", [python, "-B", "-c", code])
            run("pytest", [python, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                           "--basetemp", str(directory / "pytest-tmp"),
                           "--junitxml", str(directory / "pytest.xml"), str(diagnostic.TEST)])
            suites = ET.fromstring((directory / "pytest.xml").read_bytes()).findall("testsuite")
            assert suites and sum(int(s.attrib["tests"]) for s in suites) >= 12
            assert not any(int(s.attrib[k]) for s in suites for k in ("errors", "failures", "skipped"))
            result = json.loads(run("check", [python, "-B", "-m", diagnostic.MODULE, "--check"]))
            assert result["artifact_authentication"] == "AUTHENTICATED"
            assert result["status"] in ("SUPPORTED", "REJECTED")
            assert result["dispatch_and_write_audit"] == diagnostic.COUNTS
            assert result["flags"] == diagnostic.FLAGS
            assert [capture_helper.binding(diagnostic.SOURCE), capture_helper.binding(diagnostic.TEST)] == sources
            assert capture_helper.implementation_pins() == preflight["capture_implementation"]
            verify_members(directory, {"results": results, "preflight": preflight})
        except (OSError, ValueError, KeyError, RuntimeError, AssertionError, ET.ParseError) as error:
            failure = {"type": type(error).__name__, "message": str(error)}
        receipt = {
            "preflight": preflight, "results": results, "failure": failure, "success": failure is None,
            "sources_after": [capture_helper.binding(diagnostic.SOURCE), capture_helper.binding(diagnostic.TEST)],
            "capture_implementation_after": capture_helper.implementation_pins(),
        }
        pin = capture_helper.save(directory, "capture.json", capture_helper.encoded(receipt))
        print(json.dumps({"capture": pin, "success": failure is None, "failure": failure}))
        return int(failure is not None)


def launch(directory):
    directory = Path(directory).absolute()
    if (directory.parent != diagnostic.ROOT / "build" or directory.resolve() != directory
            or (Path.cwd(), os.getuid(), sys.executable) !=
            (diagnostic.ROOT, 1000, diagnostic.PYTHON)):
        raise RuntimeError("fresh isolated build directory/account/interpreter required")
    directory.mkdir(exist_ok=False)
    argv = [diagnostic.PYTHON, "-B", str(diagnostic.TEST), "--run", str(directory / "run")]
    identity = {
        "argv": argv, "command": capture_helper.environment_command(diagnostic.ENVIRONMENT, argv),
        "cwd": str(diagnostic.ROOT), "uid": os.getuid(), "environment": diagnostic.ENVIRONMENT,
        "launcher": capture_helper.binding(diagnostic.TEST), "invocation_argv": sys.orig_argv,
    }
    output, outer = capture_helper.seal_launcher(directory, identity)
    if outer["success"]:
        receipt = json.loads(output)
        assert receipt["success"]
        inner = json.loads(diagnostic.bound_bytes(receipt["capture"]))
        verify_members(directory / "run", inner)
        for pin in outer["files"]:
            diagnostic.bound_bytes(pin)
        assert outer["capture_implementation_after"] == capture_helper.implementation_pins()
    print(output.decode(), end="")
    print(json.dumps(outer))
    return outer["exit_status"] or (0 if outer["success"] else 1)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("--launch", "--run"):
        raise SystemExit("usage: test module --launch BUILD_ATTEMPT | --run BUILD_ATTEMPT/run")
    raise SystemExit(launch(sys.argv[2]) if sys.argv[1] == "--launch" else capture_run(sys.argv[2]))
