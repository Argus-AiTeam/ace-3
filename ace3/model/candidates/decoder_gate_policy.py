"""Explicit decoder gate applicability; the binary64-v1 profile is unchanged."""

from fractions import Fraction
from pathlib import Path

from ace3.model.candidates import binary64_fp16_excess_v1 as binary64
from ace3.model.fp16_adaptation_oracle import decode_f16_q24

POLICY_ID = "ace3-w4a16-layer-final-binary64-authority-v2"
CONTRACT = Path(__file__).resolve().parents[2] / "contracts/candidates/decoder_gate_policy_v2.json"


def fp16_is_mandatory(stage, policy=None):
    if policy not in (None, POLICY_ID):
        raise ValueError(f"unknown decoder gate policy: {policy}")
    if type(stage) is not int or not 0 <= stage <= 18:
        raise ValueError("policy applies only to decoder stages S0-S18")
    return not (policy == POLICY_ID and stage == 18)


def compare_fp16(actual, reference):
    """The original exact FP16 predicate, including its strict relative branch."""
    import numpy as np

    if (actual.dtype != np.dtype("<u2") or reference.dtype != np.dtype("<u2")
            or actual.ndim != 1 or actual.shape != reference.shape or not actual.size):
        raise ValueError("FP16 stage shape/dtype mismatch")
    if np.any((actual & 0x7c00) == 0x7c00) or np.any((reference & 0x7c00) == 0x7c00):
        raise binary64.InvalidOperand("nonfinite FP16 stage operand")
    failures = []
    for index, (a, r) in enumerate(zip(actual, reference, strict=True)):
        av = Fraction(decode_f16_q24(int(a))[0], 1 << 24)
        rv = Fraction(decode_f16_q24(int(r))[0], 1 << 24)
        error = abs(av - rv)
        relative = error / max(abs(rv), Fraction(1, 1 << 14))
        def ordered(word):
            return 0x8000 - (int(word) & 0x7fff) if word & 0x8000 else 0x8000 + int(word)
        ulp = abs(ordered(a) - ordered(r))
        if not (error <= Fraction(1, 8) or (relative < Fraction(1, 1000) and ulp <= 1)):
            failures.append({
                "index": index, "actual_bits": f"{int(a):04x}", "reference_bits": f"{int(r):04x}",
                "actual": float(av), "reference": float(rv), "absolute_error": str(error),
                "relative_error": str(relative), "ulp": ulp})
    return {"comparisons": int(actual.size), "finite_comparisons": int(actual.size),
            "failure_count": len(failures), "failures": failures,
            "passed": not failures}


def evaluate_decoder_stage(*, stage, actual, reference, policy=None, reference_binary64=None):
    mandatory = fp16_is_mandatory(stage, policy)
    fp16 = compare_fp16(actual, reference)
    report = {
        "stage": stage, "policy_id": policy, "fp16": fp16,
        "fp16_role": "mandatory" if mandatory else "layer-final-trajectory-conformance-diagnostic",
        "status": "FAIL" if mandatory and not fp16["passed"] else "PASS"}
    if stage != 18:
        if reference_binary64 is not None:
            raise ValueError("binary64 layer-final profile cannot apply to internal stages")
        return report
    if reference_binary64 is None:
        if policy == POLICY_ID:
            report.update(status="BLOCKED", reason="missing required original binary64 reference")
        return report
    import numpy as np

    if reference_binary64.dtype != np.dtype("<f8") or reference_binary64.shape != actual.shape:
        raise ValueError("binary64 stage shape/dtype mismatch")
    rows = [
        {"index": index, **binary64.evaluate_layer_final_output(
            actual_fp16_bits=int(bits), reference_binary64_hex=float(r).hex())}
        for index, (bits, r) in enumerate(zip(actual, reference_binary64, strict=True))]
    failures = [row for row in rows if not row["accepted"]]
    report["binary64_v1"] = {
        "profile_id": binary64.PROFILE_ID, "role": "mandatory", "rows": rows,
        "coordinates": len(rows), "failure_count": len(failures), "failures": failures,
        "passed": not failures}
    if failures:
        report["status"] = "FAIL"
    return report
