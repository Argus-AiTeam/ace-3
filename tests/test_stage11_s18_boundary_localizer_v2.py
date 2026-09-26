"""Independent exact suffix oracle and cold-process retained-path regressions."""

from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from ace3.model.candidates import stage11_s18_boundary_localizer_v2 as d


def scalar(word):
    word = int(word)
    sign, exponent, mantissa = word >> 15, (word >> 10) & 31, word & 1023
    assert exponent != 31
    value = Fraction(mantissa if exponent == 0 else 1024 + mantissa)
    value *= Fraction(2) ** (-24 if exponent == 0 else exponent - 25)
    return -value if sign else value


def round_word(value, negative_zero=False):
    if not value:
        return 32768 if negative_zero else 0
    negative, value = value < 0, abs(value)
    lo, hi = 0, 31743
    while lo < hi:
        mid = (lo + hi) // 2
        if scalar(mid) < value:
            lo = mid + 1
        else:
            hi = mid
    word = min({lo, max(0, lo - 1)}, key=lambda w: (abs(scalar(w) - value), w & 1, w))
    return word | (32768 if negative else 0)


def verify_suffix(words, operands, norm, account, logits):
    decoded = [scalar(w) * (1 << 24) for w in words]
    mean = round(sum((v * v for v in decoded), Fraction()) / 896 + 281474977)
    root = math.isqrt(mean)
    assert account == {"mean_q48": mean, "root_q24": root}
    for i, weight in enumerate(operands[0].view("<u2")):
        q24 = round(decoded[i] * scalar(weight) * (1 << 24) / root)
        assert int(norm[i]) == round_word(Fraction(q24, 1 << 24),
                                         bool((int(words[i]) ^ int(weight)) & 32768))
    for row, weights in enumerate(operands[1].view("<u2")):
        value = sum((scalar(a) * scalar(b) for a, b in zip(norm, weights, strict=True)), Fraction())
        assert int(logits[row]) == round_word(value)


def synthetic_guarded_path(directory):
    """Use real checkpoint loading, policy, imports and arithmetic; no retained science."""
    from safetensors.numpy import save_file
    from ace3.model.candidates import binary64_fp16_excess_v1 as policy

    tensors = {"model.norm.weight": np.ones(896, dtype="<f2"),
               "lm_head.weight": np.zeros((151936, 896), dtype="<f2")}
    tensors["lm_head.weight"][34319] = 1 / 1024
    checkpoint = directory / "synthetic.safetensors"
    save_file(tensors, str(checkpoint))
    assets = {"checkpoint": d.retained.pin(checkpoint), "tensors": {
        name: {"shape": list(a.shape), "sha256": hashlib.sha256(a.tobytes()).hexdigest()}
        for name, a in tensors.items()}}
    del tensors
    refs = {}
    for name, array in (
            ("original_input_L23_binary64", np.ones(896, dtype="<f8")),
            ("original_input_final_fp16", np.zeros(151936, dtype="<u2")),
            ("original_input_final_binary64", np.zeros(151936, dtype="<f8"))):
        path = directory / (name + ".npy")
        np.save(path, array, allow_pickle=False)
        refs[name] = d.retained.pin(path)
    summary = directory / "summary.json"
    summary.write_text(json.dumps({"assets": assets}))
    d.SUMMARY = d.retained.pin(summary)
    d.operand_assets = lambda summary: assets
    result = {"frozen_contract": {"references": refs}, "outputs": {}, "stage_reports": {}, "rows": []}
    for control in d.retained.CONTROLS:
        coordinates = d.retained.COMMON if control == "mapped_all" else d.retained.UNION
        words = [0x3c00] * 896
        for index in coordinates:
            words[index] = 0x4000
        failures = [{"index": i, **policy.evaluate_layer_final_output(
            actual_fp16_bits=words[i], reference_binary64_hex=float(1).hex())} for i in coordinates]
        result["outputs"][control] = {"stage18": words, "pair_logits": [0, 0]}
        result["stage_reports"][control] = [{"binary64_v1": {"failures": failures}}]
        result["rows"].extend({"control": control, "branch": branch, "predicted_delta": "1",
                               "retained_margin": "0", "intervened_margin": "0",
                               "independent_original_reference_margin": "0"}
                              for branch in ("fp16", "binary64"))
    original = d.retained.encoded(result)
    d.retained.authenticate = lambda: (result, list(refs.values()))
    # Synthetic transport only; the separate live-binding test authenticates b497.
    d.retained.PINS = {}
    report = d.guarded_check()
    assert report["status"] == "SUPPORTED", report
    assert report["audit"] == {**d.new_audit(), "final_rmsnorm_invocations": 9,
                               "tied_pair_head_invocations": 9}
    assert d.retained.encoded(result) == original
    assert len(report["rows"]) == 18
    for control in report["controls"]:
        expected = d.retained.COMMON if control["control"] == "mapped_all" else d.retained.UNION
        assert control["changed_coordinates"] == list(expected)
        assert control["S18_binary64_status"] == control["independent_suffix_oracle"] == "PASS"
    assert not any(name == "torch" or name == "dill" or
                   "q24_s16_final_head_from_l23_coordinate62_suffix_execution" in name
                   for name in sys.modules)
    print(json.dumps({"status": report["status"], "audit": report["audit"]}))


def test_cold_real_path_nine_suffixes(tmp_path):
    script = (
        "import importlib.util; from pathlib import Path; "
        f"s=importlib.util.spec_from_file_location('regression', {str(d.TEST)!r}); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        f"m.synthetic_guarded_path(Path({str(tmp_path)!r}))"
    )
    completed = subprocess.run([sys.executable, "-B", "-c", script], cwd=d.ROOT,
                               text=True, capture_output=True, timeout=90,
                               env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stderr == ""
    assert json.loads(completed.stdout)["audit"] == {
        **d.new_audit(), "final_rmsnorm_invocations": 9, "tied_pair_head_invocations": 9}


def test_live_bindings_without_scientific_replay():
    result, records = d.retained.authenticate()
    assert result["status"] == "REJECTED" and records
    summary = json.loads(d.retained.bound(d.SUMMARY))
    assets = d.operand_assets(summary)
    operands = d.load_operands(assets)
    assert [a.shape for a in operands] == [(896,), (2, 896)]
    summary["checkpoint"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint identity"):
        d.operand_assets(summary)


@pytest.mark.parametrize("event,args", [
    ("open", ("/dev/null", "r+", os.O_RDWR)),
    ("open", ("retained.json", "w", os.O_WRONLY | os.O_CREAT)),
    ("subprocess.Popen", ()), ("os.system", ()), ("os.remove", ()),
    ("os.rename", ()), ("os.mkdir", ()), ("socket.connect", ()), ("socket.bind", ()),
])
def test_readonly_guard_remains_strict(event, args):
    audit = d.new_audit()
    with pytest.raises(RuntimeError, match=event.replace(".", r"\.")):
        d.ReadOnly(audit)(event, args)
    assert audit == {**d.new_audit(), "forbidden_calls": 1}


def test_no_dispatch_refuses_prior_check_and_services():
    audit = d.new_audit()
    with d.no_dispatch(audit):
        for call in (d.retained.check, d.retained.main, subprocess.Popen, os.system):
            with pytest.raises(RuntimeError, match="dispatch forbidden"):
                call(None)
    assert audit == {**d.new_audit(), "forbidden_calls": 4}


def test_readonly_read_and_independent_corruption():
    audit = d.new_audit()
    d.ReadOnly(audit)("open", ("retained.json", "r", os.O_RDONLY))
    assert audit == d.new_audit()
    words = np.full(896, 0x3c00, dtype="<u2")
    words[0] = 0x8000
    operands = (np.ones(896, dtype="<f2"), np.full((2, 896), 1 / 1024, dtype="<f2"))
    norm, account = d.rmsnorm(words, operands[0])
    scores = d.logits(norm, operands[1])
    verify_suffix(words, operands, norm, account, scores)
    scores[0] += 1
    with pytest.raises(AssertionError):
        verify_suffix(words, operands, norm, account, scores)
