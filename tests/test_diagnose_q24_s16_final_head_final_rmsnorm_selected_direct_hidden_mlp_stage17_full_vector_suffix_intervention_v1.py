"""Full-vector independent scalar oracles, mutation checks and shared byte capture."""

import copy
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_mlp_stage17_full_vector_suffix_intervention_v1 as diagnostic


capture_helper = diagnostic.capture
scalar_oracle = diagnostic.load_module(diagnostic.previous.TEST, diagnostic.NAME + "_scalar_oracle")
scalar, round_word = scalar_oracle.scalar, scalar_oracle.round_word


def verify_suffix(actual, arrays, reference, operands, norm_account):
    assert arrays["stage17"].tobytes() == reference["stage17"].tobytes()
    for k, word in enumerate(reference["stage17"]):
        increment = scalar(word) * (1 << 24)
        assert increment.denominator == 1
        total = int(actual["scratch_i"][k]) + increment.numerator
        zero = int(total == 0 and actual["scratch_i"][k] == 0
                   and actual["scratch_z"][k] == 1 and int(word) == 32768)
        assert int(arrays["output_i"][k]) == total
        assert int(arrays["output_z"][k]) == zero
        assert int(arrays["stage18"][k]) == round_word(Fraction(total, 1 << 24), bool(zero))
        if word == actual["stage17"][k]:
            for key in ("output_i", "output_z", "stage18"):
                assert arrays[key][k] == actual[key][k]
    decoded = [scalar(word) * (1 << 24) for word in arrays["stage18"]]
    mean = round(sum((v * v for v in decoded), Fraction()) / 896 + 281474977)
    root = math.isqrt(mean)
    assert norm_account == {"mean_q48": mean, "root_q24": root}
    for k, weight in enumerate(operands[0].view("<u2")):
        product = decoded[k] * scalar(weight) * (1 << 24)
        result = round(product / root)
        negative_zero = bool((int(arrays["stage18"][k]) ^ int(weight)) & 32768)
        assert int(arrays["final_rmsnorm"][k]) == round_word(
            Fraction(result, 1 << 24), negative_zero)
    for row, weights in enumerate(operands[1].view("<u2")):
        total = sum((scalar(a) * scalar(b) for a, b in
                     zip(arrays["final_rmsnorm"], weights, strict=True)), Fraction())
        assert int(arrays["pair_logits"][row]) == round_word(total)


def fixture_arrays():
    actual, reference = scalar_oracle.fixture_arrays()
    reference["stage17"][:] = 16384
    reference["stage17"][3] = actual["stage17"][3]
    return actual, reference


@pytest.fixture
def synthetic():
    evidence = {"report": {"controls": []}, "retained_50f": {"report": {"controls": []}}}
    archives = {}
    actual, reference = fixture_arrays()
    operands = (np.ones(896, dtype="<f2"),
                np.array([np.ones(896), np.zeros(896)], dtype="<f2"))
    for control in diagnostic.CONTROLS:
        archives[control] = copy.deepcopy(actual)
        branches, raw = {}, {}
        for branch in diagnostic.BRANCHES:
            cells = [{
                "coordinate": k, "reference_inverse_norm_anchor": "1",
                "hidden_components": {"mlp_stage17": "-1"},
                "weighted_components": {"mlp_stage17": "-1"},
                "weight_times_reference_anchor_times_row_difference": "1",
                "retained_fp16_residual_account": {"boundaries": {
                    "stage17": {"actual_exact": "1", "reference_exact": "2"}}},
            } for k in (2, 7)]
            branches[branch] = {"per_component_coordinate_hotspots": {"mlp_stage17": {
                "ranking": [{"coordinate": k, "signed": "-1", "absolute": "1"} for k in (2, 7)]}}}
            raw[branch] = {
                "selected_coordinates": cells, "residual_internal_reference": "original_input_L23_fp16",
                "hidden_reference": "original_input_L23_" + branch,
                "binary64_internal_stages": "NOT_RETAINED_NO_RECONSTRUCTION",
                "direct_hidden_accounting": {"component_totals": {"mlp_stage17": {"signed": "-2"}}},
            }
        for target, values in ((evidence["report"]["controls"], branches),
                               (evidence["retained_50f"]["report"]["controls"], raw)):
            target.append({"control": control, "pairs": [{"left_id": 34319, "right_id": 13,
                                                         "branches": values}]})
    return evidence, archives, reference, operands


def test_preregister_full_vector_not_selected_union(synthetic, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("prediction must precede operators")
    monkeypatch.setattr(diagnostic.native.state, "add", forbidden)
    plan = diagnostic.preregister(*synthetic)
    assert plan["coordinates"] == list(range(896))
    assert len(plan["rows"]) == 18
    for row in plan["rows"]:
        assert row["predicted_delta"] == "895"
        assert row["retained_selected_signed"] == "-2" and row["unselected_signed"] == "-893"
        assert len(row["components"]) == len(row["factors"]) == 896


@pytest.mark.parametrize("defect", ["mass", "duplicate", "reference", "anchor", "operand", "zero"])
def test_preregistration_defects_unknown(synthetic, defect):
    evidence, archives, reference, operands = synthetic
    source = diagnostic.middle(evidence["retained_50f"]["report"]["controls"][0])["branches"]["fp16"]
    if defect == "mass":
        source["selected_coordinates"][0]["weighted_components"]["mlp_stage17"] = "3"
    elif defect == "duplicate":
        source["selected_coordinates"].append(copy.deepcopy(source["selected_coordinates"][0]))
    elif defect == "reference":
        source["hidden_reference"] = "local_actual"
    elif defect == "anchor":
        source["selected_coordinates"][1]["reference_inverse_norm_anchor"] = "2"
    elif defect == "operand":
        reference["stage17"][2] = 15360
    else:
        # The unselected complement can cancel the selected prediction exactly.
        reference["stage17"][:] = 15360
        reference["stage17"][[2, 7]] = 16384
        for arrays in archives.values():
            arrays["stage17"][10] = 16896
    with pytest.raises(ValueError):
        diagnostic.preregister(evidence, archives, reference, operands)


@pytest.mark.parametrize("key", ["stage03", "scratch_i", "output_cache_k", "stage17"])
def test_full_operand_and_protected_state(key):
    actual, reference = fixture_arrays()
    arrays = diagnostic.prepare(actual, reference)
    assert np.count_nonzero(arrays["stage17"] != actual["stage17"]) == 895
    assert not any(v.flags.writeable for v in arrays.values())
    assert actual["stage17"][2] == 15360
    bad = {k: v.copy() for k, v in arrays.items()}
    bad[key].flat[0] += 1
    with pytest.raises(ValueError):
        diagnostic.protected(actual, bad, reference)


def reports(gate="PASS"):
    return {c: {"stage": 18, "status": gate, "residual_state_lineage": "PASS", "kv_lineage": "PASS"}
            for c in diagnostic.CONTROLS}


def directional_rows(delta="1", mlp="1", prediction="1"):
    p, m, d = map(Fraction, (prediction, mlp, delta))
    return [{"control": c, "branch": b, "predicted_delta": prediction,
             "mlp_full_vector_delta": mlp, "margin_delta": delta,
             "direction_observed": p * m > 0 and p * d > 0 and p == m}
            for c in diagnostic.CONTROLS for b in diagnostic.BRANCHES]


@pytest.mark.parametrize("delta,mlp,prediction,gate,expected", [
    ("1", "1", "1", "PASS", "SUPPORTED"), ("-1", "-1", "-1", "PASS", "SUPPORTED"),
    ("0", "1", "1", "PASS", "REJECTED"), ("-1", "1", "1", "PASS", "REJECTED"),
    ("1", "-1", "1", "PASS", "REJECTED"), ("1", "2", "1", "PASS", "REJECTED"),
    ("1", "1", "1", "FAIL", "REJECTED"),
])
def test_outcome_distinction(delta, mlp, prediction, gate, expected):
    assert diagnostic.classify(directional_rows(delta, mlp, prediction), reports(gate)) == expected


@pytest.mark.parametrize("defect", ["zero", "census", "lineage", "kv", "direction"])
def test_classification_integrity_unknown(defect):
    rows, gates = directional_rows(), reports()
    if defect == "zero":
        rows[0]["predicted_delta"] = "0"
    elif defect == "census":
        rows.pop()
    elif defect in ("lineage", "kv"):
        gates[diagnostic.CONTROLS[0]]["kv_lineage" if defect == "kv" else "residual_state_lineage"] = "FAIL"
    else:
        rows[0]["direction_observed"] = False
    with pytest.raises(ValueError):
        diagnostic.classify(rows, gates)


def test_independent_s18_norm_and_two_row_oracles():
    actual, reference = fixture_arrays()
    arrays = diagnostic.prepare(actual, reference)
    scratch = {"i": arrays["scratch_i"], "z": arrays["scratch_z"], "h": arrays["stage12"]}
    changed = diagnostic.native.state.add(scratch, arrays["stage17"])
    arrays.update(output_i=changed["i"], output_z=changed["z"], stage18=changed["h"])
    operands = (np.ones(896, dtype="<f2"), np.full((2, 896), 0.125, dtype="<f2"))
    arrays["final_rmsnorm"], account = diagnostic.parent.rmsnorm(arrays["stage18"], operands[0])
    arrays["pair_logits"] = diagnostic.parent.logits(arrays["final_rmsnorm"], operands[1])
    verify_suffix(actual, arrays, reference, operands, account)
    for key in ("output_i", "output_z", "stage18", "final_rmsnorm", "pair_logits"):
        bad = {k: v.copy() for k, v in arrays.items()}
        bad[key].flat[0] += 1
        with pytest.raises(AssertionError):
            verify_suffix(actual, bad, reference, operands, account)


def test_forbidden_operators_and_writes(tmp_path):
    audit = dict.fromkeys(diagnostic.COUNTS, 0)
    with diagnostic.suffix_only(audit, {}, (None, None)):
        for call in (
            lambda: diagnostic.native.projection(None, None, None),
            lambda: diagnostic.previous.check(),
            lambda: diagnostic.direct.report(None),
            lambda: diagnostic.guards.check(),
            lambda: (tmp_path / "forbidden").write_text("no"),
        ):
            with pytest.raises(RuntimeError):
                call()
        with pytest.raises(ValueError, match="S18"):
            diagnostic.native.state.add({}, None)
    assert audit["forbidden_calls"] == 5
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


@pytest.mark.parametrize("member", ["stdout", "capture", "outer", "review"])
def test_parent_byte_tampering_refused(member, monkeypatch):
    pins = copy.deepcopy(diagnostic.PINS)
    pins[member]["sha256"] = "0" * 64
    monkeypatch.setattr(diagnostic, "PINS", pins)
    with pytest.raises(ValueError):
        diagnostic.authenticate_suffix()


def test_parent_bytes_authenticate_without_replay(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("retained authentication must not execute the parent")
    monkeypatch.setattr(diagnostic.previous, "check", forbidden)
    monkeypatch.setattr(diagnostic.native.state, "add", forbidden)
    result = diagnostic.authenticate_suffix()
    assert result["status"] == "REJECTED" and result["artifact_authentication"] == "AUTHENTICATED"


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
            "model_or_service_calls_authorized": 0, "mission_id": diagnostic.MISSION,
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
            diagnostic.verify_members(directory, {"results": results, "preflight": preflight})
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
        diagnostic.verify_members(directory / "run", inner)
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
