"""Opt-in v3 numerical evaluation, separate from immutable legacy/v2 sources."""

import json
from pathlib import Path
import struct

from ace3.model.candidates import decoder_gate_policy as legacy
from ace3.model.candidates import local_operator_reference_v3 as local
from ace3.model.candidates import runtime_admission_v3 as runtime


POLICY_ID = "ace3-w4a16-local-operator-global-binary64-authority-v3"
CONTRACT = Path(__file__).resolve().parents[2] / "contracts/candidates/decoder_gate_policy_v3.json"


def evaluate_decoder_stage(*, stage, actual, reference, policy,
                           local_reference=None, reference_binary64=None):
    if policy != POLICY_ID:
        raise ValueError("v3 requires explicit policy opt-in; use legacy evaluator for legacy/v2")
    legacy.fp16_is_mandatory(stage)
    diagnostic = legacy.compare_fp16(actual, reference)
    report = {"stage": stage, "policy_id": policy, "fp16": diagnostic,
              "fp16_role": "independent-whole-FP16-trajectory-diagnostic",
              "status": "PASS", "claim_scope": "numerical_only_requires_source_and_state_admission"}
    if stage < 18:
        if reference_binary64 is not None:
            raise ValueError("global binary64 profile cannot apply to internal stages")
        if local_reference is None:
            report.update(status="BLOCKED", reason="missing independent local operator reference")
            return report
        report["local_operator_fp16"] = {
            **legacy.compare_fp16(actual, local_reference), "role": "mandatory"}
        if not report["local_operator_fp16"]["passed"]:
            report["status"] = "FAIL"
    else:
        if local_reference is not None:
            raise ValueError("S18 must not be re-anchored to a local reference")
        result = legacy.evaluate_decoder_stage(
            stage=18, actual=actual, reference=reference, policy=legacy.POLICY_ID,
            reference_binary64=reference_binary64)
        report["status"] = result["status"]
        if "binary64_v1" in result:
            report["binary64_v1"] = result["binary64_v1"]
        if "reason" in result:
            report["reason"] = result["reason"]
    return report


def evaluate_p0_transaction(*, arrays, tensors, canonical_records, expected_hidden,
                            trajectory, reference_binary64, layer, position, history,
                            policy, emit):
    if policy != POLICY_ID:
        raise ValueError("explicit v3 transaction policy required")
    local.require(type(layer) is int and 5 <= layer <= 8, "bounded layer scope is L5-L8")
    local.authenticate_tensors(tensors, canonical_records, layer)
    lineage = local.validate_lineage(arrays, expected_hidden, position=position, history=history)
    local.require(set(trajectory) == set(range(19)), "missing original trajectory references")
    reports, references = [], {}
    for stage in range(19):
        expected = None
        if stage < 18:
            inputs = {name: arrays[name] for name in local.OPERANDS[stage]}
            expected = local.local_reference(stage, inputs, tensors, layer)
            references[f"stage{stage:02d}"] = expected
        report = evaluate_decoder_stage(
            stage=stage, actual=arrays[f"stage{stage:02d}"], reference=trajectory[stage],
            policy=policy, local_reference=expected,
            reference_binary64=reference_binary64 if stage == 18 else None)
        report.update(node=[layer, position, stage],
                      actual_operands=list(local.OPERANDS[stage]) if stage < 18 else None)
        emit(report)
        reports.append(report)
        if report["status"] != "PASS":
            break
    return {"status": reports[-1]["status"], "lineage": lineage,
            "reports": reports, "local_references": references}


def evaluate_actual_rtl_result(*, runtime_admission, trusted_runtime_manifest_sha256=None, **kwargs):
    """Numerical wiring for decoded actual results; never run or restore RTL.

    The owning controller supplies its independently authenticated manifest
    digest separately. A result's self-reported PASS or self-supplied digest is
    not an authority. Invalid evidence blocks before any numerical gate runs.
    """
    try:
        admission = runtime.validate_runtime(
            receipt=runtime_admission,
            trusted_manifest_sha256=trusted_runtime_manifest_sha256,
            contract=json.loads(CONTRACT.read_text()),
            **{key: value for key, value in kwargs.items() if key != "emit"})
    except (OSError, ValueError, KeyError, TypeError, IndexError, struct.error) as exc:
        return {"status": "BLOCKED", "numerical_status": "NOT_EVALUATED",
                "evidence_kind": "rtl_runtime_not_admitted", "policy_id": POLICY_ID,
                "failure_taxonomy": "runtime_provenance",
                "reason": f"{type(exc).__name__}: {exc}"}
    result = evaluate_p0_transaction(**kwargs)
    result["evidence_kind"] = "actual_rtl_numerical_evaluation"
    result["runtime_admission"] = admission
    result["numerical_status"] = result["status"]
    result["policy_id"] = POLICY_ID
    result["normal_host_review"] = "REQUIRED"
    result["no_full_model_or_generation_admission"] = True
    return result
