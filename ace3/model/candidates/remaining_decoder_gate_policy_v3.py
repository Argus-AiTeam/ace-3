"""Source-bound L9-L23/P0 extension; historical L5-L8 policy bytes stay unchanged."""

import json
import struct

from ace3.model.candidates import decoder_gate_policy_v3 as base
from ace3.model.candidates import local_operator_reference_v3 as local
from ace3.model.candidates import runtime_admission_v3 as runtime
from ace3.model.candidates.run_single_round_residual_rtl import retained_record as record


EXTENSION_ID = "ace3-v3-remaining-decoder-p0-evaluator-v1"
LAYERS = tuple(range(9, 24))


def binding():
    return {
        "id": EXTENSION_ID, "policy_id": base.POLICY_ID,
        "scope": {"layers": list(LAYERS), "position": 0, "history": [9707]},
        "sources": [record(path) for path in (
            __file__, base.__file__, local.__file__, runtime.__file__, base.CONTRACT)],
    }


def validate_binding(value):
    local.require(value == binding(), "missing/mismatched remaining evaluator extension")


def evaluate_p0_transaction(*, arrays, tensors, canonical_records, expected_hidden,
                            trajectory, reference_binary64, layer, position, history,
                            policy, emit):
    local.require(policy == base.POLICY_ID, "explicit v3 transaction policy required")
    local.require(type(layer) is int and layer in LAYERS, "bounded layer scope is L9-L23")
    local.authenticate_tensors(tensors, canonical_records, layer)
    lineage = local.validate_lineage(arrays, expected_hidden, position=position, history=history)
    local.require(set(trajectory) == set(range(19)), "missing original trajectory references")
    reports, references = [], {}
    # Keep the historical transaction implementation byte-identical; share its
    # numerical predicates and independent operators, not its L5-L8 scope guard.
    for stage in range(19):
        expected = None
        if stage < 18:
            operands = {name: arrays[name] for name in local.OPERANDS[stage]}
            expected = local.local_reference(stage, operands, tensors, layer)
            references[f"stage{stage:02d}"] = expected
        report = base.evaluate_decoder_stage(
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


def evaluate_actual_rtl_result(*, runtime_admission, trusted_runtime_manifest_sha256=None,
                              reference_extension=None, evaluator_extension=None, **kwargs):
    try:
        validate_binding(evaluator_extension)
        contract = dict(json.loads(base.CONTRACT.read_text()),
                        reference_extension=reference_extension,
                        evaluator_extension=evaluator_extension)
        local.require(type(kwargs.get("layer")) is int and kwargs["layer"] in LAYERS,
                      "bounded layer scope is L9-L23")
        admission = runtime.validate_runtime(
            receipt=runtime_admission, trusted_manifest_sha256=trusted_runtime_manifest_sha256,
            contract=contract, **{key: value for key, value in kwargs.items() if key != "emit"})
    except (OSError, ValueError, KeyError, TypeError, IndexError, struct.error) as exc:
        return {"status": "BLOCKED", "numerical_status": "NOT_EVALUATED",
                "evidence_kind": "rtl_runtime_not_admitted", "policy_id": base.POLICY_ID,
                "failure_taxonomy": "runtime_provenance",
                "reason": f"{type(exc).__name__}: {exc}"}
    result = evaluate_p0_transaction(**kwargs)
    result.update(evidence_kind="actual_rtl_numerical_evaluation", runtime_admission=admission,
                  numerical_status=result["status"], policy_id=base.POLICY_ID,
                  evaluator_extension=evaluator_extension, normal_host_review="REQUIRED",
                  no_full_model_or_generation_admission=True)
    return result
