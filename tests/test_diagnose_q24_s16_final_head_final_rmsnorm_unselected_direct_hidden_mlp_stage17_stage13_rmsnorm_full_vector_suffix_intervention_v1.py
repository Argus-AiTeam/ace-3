"""Stage13-only mutation guards, independent scalar suffix oracle and byte capture."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_stage13_rmsnorm_full_vector_suffix_intervention_v1 as diagnostic


shared_tests = diagnostic.shared.load_module(diagnostic.shared.TEST, diagnostic.NAME + "_scalar_helpers")
scalar, round_word = shared_tests.scalar, shared_tests.round_word
capture = diagnostic.capture


def verify_final_suffix(actual, arrays, operands, account):
    for k, word in enumerate(arrays["stage17"]):
        increment = scalar(word) * (1 << 24)
        assert increment.denominator == 1
        total = int(actual["scratch_i"][k]) + increment.numerator
        zero = int(total == 0 and actual["scratch_i"][k] == 0
                   and actual["scratch_z"][k] == 1 and int(word) == 32768)
        assert int(arrays["output_i"][k]) == total
        assert int(arrays["output_z"][k]) == zero
        assert int(arrays["stage18"][k]) == round_word(Fraction(total, 1 << 24), bool(zero))
    decoded = [scalar(word) * (1 << 24) for word in arrays["stage18"]]
    mean = round(sum((v * v for v in decoded), Fraction()) / 896 + 281474977)
    root = math.isqrt(mean)
    assert account == {"mean_q48": mean, "root_q24": root}
    for k, weight in enumerate(operands[0].view("<u2")):
        result = round(decoded[k] * scalar(weight) * (1 << 24) / root)
        negative_zero = bool((int(arrays["stage18"][k]) ^ int(weight)) & 32768)
        assert int(arrays["final_rmsnorm"][k]) == round_word(Fraction(result, 1 << 24), negative_zero)
    for row, weights in enumerate(operands[1].view("<u2")):
        total = sum((scalar(a) * scalar(b) for a, b in
                     zip(arrays["final_rmsnorm"], weights, strict=True)), Fraction())
        assert int(arrays["pair_logits"][row]) == round_word(total)


def fixture_arrays():
    actual, reference = shared_tests.fixture_arrays()
    actual["input_i"] = np.zeros(896, dtype="<i8")
    actual["stage13"] = np.full(896, 15360, dtype="<u2")
    reference["stage13"] = np.full(896, 16384, dtype="<u2")
    return actual, reference


def test_whole_vector_only_and_immutable():
    actual, reference = fixture_arrays()
    arrays = diagnostic.prepare(actual, reference)
    assert np.count_nonzero(arrays["stage13"] != actual["stage13"]) == 896
    assert arrays["stage17"].tobytes() == actual["stage17"].tobytes()
    assert not any(v.flags.writeable for v in arrays.values())
    assert np.all(actual["stage13"] == 15360)


@pytest.mark.parametrize("key", ["stage03", "stage05", "stage12", "scratch_i",
                                 "scratch_z", "input_i", "output_cache_k", "stage13"])
def test_protected_source_state_and_kv(key):
    actual, reference = fixture_arrays()
    arrays = {k: v.copy() for k, v in diagnostic.prepare(actual, reference).items()}
    arrays[key].flat[0] += 1
    with pytest.raises(ValueError):
        diagnostic.protected(actual, arrays, reference)


@pytest.mark.parametrize("defect", ["absent", "shape", "dtype", "nonfinite"])
def test_exact_missing_or_invalid_stage13_binding(defect):
    actual, _ = fixture_arrays()
    if defect == "absent":
        del actual["stage13"]
    elif defect == "shape":
        actual["stage13"] = actual["stage13"][:-1]
    elif defect == "dtype":
        actual["stage13"] = actual["stage13"].astype("<i4")
    else:
        actual["stage13"][0] = 31744
    with pytest.raises(ValueError, match="archive.npz::stage13"):
        diagnostic.stage13_binding(actual, {"path": "archive.npz"})


def rows(delta="1", mlp="2", prediction="1"):
    p, m, d = map(Fraction, (prediction, mlp, delta))
    return [{"control": c, "branch": b, "predicted_delta": prediction,
             "mlp_vector_delta": mlp, "margin_delta": delta,
             "direction_observed": p * m > 0 and p * d > 0}
            for c in diagnostic.CONTROLS for b in diagnostic.BRANCHES]


def gates():
    return {c: [{"stage": s, "status": "PASS", "residual_state_lineage": "PASS",
                 "kv_lineage": "PASS"} for s in range(14, 19)] for c in diagnostic.CONTROLS}


@pytest.mark.parametrize("delta,mlp,prediction,expected", [
    ("1", "2", "1", "SUPPORTED"), ("-1", "-2", "-1", "SUPPORTED"),
    ("0", "1", "1", "REJECTED"), ("-1", "1", "1", "REJECTED"),
    ("1", "0", "1", "REJECTED"), ("1", "-1", "1", "REJECTED"),
    ("1", "1", "0", "REJECTED"),
])
def test_numerical_direction_not_integrity_failure(delta, mlp, prediction, expected):
    assert diagnostic.classify(rows(delta, mlp, prediction), gates()) == expected


@pytest.mark.parametrize("stage", range(14, 19))
def test_every_existing_gate_is_decisive(stage):
    reports = gates()
    reports[diagnostic.CONTROLS[0]][stage - 14]["status"] = "FAIL"
    assert diagnostic.classify(rows(), reports) == "REJECTED"


@pytest.mark.parametrize("defect", ["row", "stage", "kv", "state", "direction"])
def test_integrity_defects_are_unknown(defect):
    reports, contrasts = gates(), rows()
    first = reports[diagnostic.CONTROLS[0]][0]
    if defect == "row":
        contrasts.pop()
    elif defect == "stage":
        first["stage"] = 13
    elif defect == "kv":
        first["kv_lineage"] = "FAIL"
    elif defect == "state":
        first["residual_state_lineage"] = "FAIL"
    else:
        contrasts[0]["direction_observed"] = False
    with pytest.raises(ValueError):
        diagnostic.classify(contrasts, reports)


def test_preregistration_uses_retained_full_vector_before_dispatch(monkeypatch):
    evidence, archives, reference, operands = shared_tests.synthetic.__wrapped__()
    original = evidence["retained_50f"]
    def forbidden(*args, **kwargs):
        raise AssertionError("no operator during preregistration")
    monkeypatch.setattr(diagnostic.native, "projection", forbidden)
    monkeypatch.setattr(diagnostic.parent, "rmsnorm", forbidden)
    plan = diagnostic.preregister(original, archives, reference, operands)
    assert plan["coordinates"] == list(range(896)) and len(plan["rows"]) == 18
    assert all(row["predicted_delta"] == "895" for row in plan["rows"])
    original["report"]["controls"][0]["pairs"][0]["branches"]["fp16"]["hidden_reference"] = "local"
    with pytest.raises(ValueError, match="reanchored"):
        diagnostic.preregister(original, archives, reference, operands)


def test_independent_final_oracle_and_mutations():
    actual, reference = fixture_arrays()
    arrays = diagnostic.prepare(actual, reference)
    scratch = {"i": arrays["scratch_i"], "z": arrays["scratch_z"], "h": arrays["stage12"]}
    changed = diagnostic.native.state.add(scratch, arrays["stage17"])
    arrays.update(output_i=changed["i"], output_z=changed["z"], stage18=changed["h"])
    operands = (np.ones(896, dtype="<f2"), np.full((2, 896), 0.125, dtype="<f2"))
    arrays["final_rmsnorm"], account = diagnostic.parent.rmsnorm(arrays["stage18"], operands[0])
    arrays["pair_logits"] = diagnostic.parent.logits(arrays["final_rmsnorm"], operands[1])
    verify_final_suffix(actual, arrays, operands, account)
    for key in ("output_i", "output_z", "stage18", "final_rmsnorm", "pair_logits"):
        bad = {k: v.copy() for k, v in arrays.items()}
        bad[key].flat[0] += 1
        with pytest.raises(AssertionError):
            verify_final_suffix(actual, bad, operands, account)


def test_out_of_suffix_and_closed_producer_dispatch_forbidden(tmp_path):
    audit = dict.fromkeys(diagnostic.COUNTS, 0)
    active = {"stage": 13}
    with diagnostic.suffix_only(audit, active, {}, (None, None)):
        with pytest.raises(ValueError, match="outside"):
            diagnostic.native.projection({}, "model.layers.23.mlp.gate_proj", None)
        with pytest.raises(ValueError, match="outside"):
            diagnostic.layer.expected_stage(13, {}, {}, {})
        with pytest.raises(ValueError, match="scope"):
            diagnostic.native.state.add({}, None)
        with pytest.raises(RuntimeError):
            diagnostic.shared.check()
        with pytest.raises(RuntimeError):
            (tmp_path / "forbidden").write_text("forbidden")
    assert not (tmp_path / "forbidden").exists()


def test_missing_capture_reports_unknown(monkeypatch, capsys):
    def missing():
        raise ValueError("missing retained stage13 binding: archive.npz::stage13")
    monkeypatch.setattr(diagnostic, "check", missing)
    assert diagnostic.main(["--check"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "UNKNOWN" and result["lane_terminated"]
    assert result["flags"] == diagnostic.FLAGS and "archive.npz::stage13" in result["error"]


def capture_run(directory):
    root, python = diagnostic.ROOT, diagnostic.PYTHON
    if (Path.cwd(), os.getuid(), sys.executable) != (root, 1000, python):
        raise RuntimeError("account/workdir/interpreter gate")
    directory = Path(directory).absolute()
    if directory.parent.parent != root / "build" or directory.name != "run" or directory.resolve() != directory:
        raise RuntimeError("fresh isolated build/<attempt>/run required")
    with capture.same_scope(root, diagnostic.MODULE):
        directory.mkdir(exist_ok=False)
        sources = [capture.binding(diagnostic.SOURCE), capture.binding(diagnostic.TEST)]
        preflight = {
            "cwd": str(root), "uid": os.getuid(), "python": python, "python_version": sys.version,
            "environment": diagnostic.ENVIRONMENT, "sources": sources,
            "executable": capture.binding(Path(python).resolve()),
            "independent_host_review": "REQUIRED", "scope": diagnostic.BOUNDARY,
            "model_or_service_calls_authorized": 0, "mission_id": diagnostic.MISSION,
            "role": "engineer", "attempt": str(directory.parent),
            "command_budget": {"compile": 1, "pytest": 1, "check": 1},
            "capture_implementation": capture.implementation_pins(),
        }
        results, failure = [], None
        def run(label, argv):
            return capture.run_command(directory, label, argv, preflight, results, timeout=90)
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
            code = (
                "import py_compile; "
                f"paths={[str(diagnostic.SOURCE), str(diagnostic.TEST)]!r}; "
                f"out={str(directory)!r}; "
                "[py_compile.compile(p,cfile=out+'/compiled-'+str(i)+'.pyc',doraise=True) "
                "for i,p in enumerate(paths)]; print('py_compile: 2 files')"
            )
            run("compile", [python, "-B", "-c", code])
            run("pytest", [python, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                           "--basetemp", str(directory / "pytest-tmp"),
                           "--junitxml", str(directory / "pytest.xml"), str(diagnostic.TEST)])
            result = json.loads(run("check", [python, "-B", "-m", diagnostic.MODULE, "--check"]))
            assert result["artifact_authentication"] == "AUTHENTICATED"
            assert result["status"] in ("SUPPORTED", "REJECTED")
            assert result["dispatch_and_write_audit"] == diagnostic.COUNTS
            assert result["flags"] == diagnostic.FLAGS and result["lane_terminated"]
            assert [capture.binding(diagnostic.SOURCE), capture.binding(diagnostic.TEST)] == sources
            assert capture.implementation_pins() == preflight["capture_implementation"]
            diagnostic.shared.verify_members(directory, {"results": results, "preflight": preflight})
        except (OSError, ValueError, KeyError, RuntimeError, AssertionError, ET.ParseError) as error:
            failure = {"type": type(error).__name__, "message": str(error)}
        receipt = {
            "preflight": preflight, "results": results, "failure": failure, "success": failure is None,
            "sources_after": [capture.binding(diagnostic.SOURCE), capture.binding(diagnostic.TEST)],
            "capture_implementation_after": capture.implementation_pins(),
        }
        pin = capture.save(directory, "capture.json", capture.encoded(receipt))
        print(json.dumps({"capture": pin, "success": failure is None, "failure": failure}))
        return int(failure is not None)


def launch(directory):
    directory = Path(directory).absolute()
    if (directory.parent != diagnostic.ROOT / "build" or directory.resolve() != directory
            or (Path.cwd(), os.getuid(), sys.executable) !=
            (diagnostic.ROOT, 1000, diagnostic.PYTHON)):
        raise RuntimeError("fresh isolated build/account/interpreter required")
    directory.mkdir(exist_ok=False)
    argv = [diagnostic.PYTHON, "-B", str(diagnostic.TEST), "--run", str(directory / "run")]
    identity = {
        "argv": argv, "command": capture.environment_command(
            dict(sorted(diagnostic.ENVIRONMENT.items())), argv),
        "cwd": str(diagnostic.ROOT), "uid": os.getuid(), "environment": diagnostic.ENVIRONMENT,
        "launcher": capture.binding(diagnostic.TEST), "invocation_argv": sys.orig_argv,
        "capture_implementation": capture.implementation_pins(),
    }
    output, outer = capture.seal_launcher(directory, identity, timeout=110)
    capture.verify_launcher(directory, identity, capture.binding(directory / "launcher.capture.json"),
                            require_success=False)
    if outer["success"]:
        inner = json.loads(diagnostic.bound_bytes(json.loads(output)["capture"]))
        diagnostic.shared.verify_members(directory / "run", inner)
    print(output.decode(), end="")
    print(json.dumps(outer))
    return outer["exit_status"] or (0 if outer["success"] else 1)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("--launch", "--run"):
        raise SystemExit("usage: test module --launch BUILD_ATTEMPT | --run BUILD_ATTEMPT/run")
    raise SystemExit(launch(sys.argv[2]) if sys.argv[1] == "--launch" else capture_run(sys.argv[2]))
