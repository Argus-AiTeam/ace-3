"""Static authority/specification checks only; no model or arithmetic imports."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PACKET = ROOT / (
    "ace3/contracts/candidates/"
    "residual_exact_grid_q24_decoder_adoption_c39ca7b7722a_attempt001.json"
)
SECTIONS = {
    "schema_version", "attempt_id", "mission_id", "kind", "specification",
    "authority", "identities", "precision", "scope", "state", "references",
    "public_interface", "status", "validation", "bindings",
}
BINDINGS = {
    "mission", "manager_wait", "manager_directive", "primitive_freeze",
    "primitive_result", "primitive_review", "isolated_authority",
    "isolated_authority_review", "frontier_review", "semantics",
    "source_change_plan", "primitive_arithmetic", "primitive_state",
    "primitive_policy", "v3_policy", "binary64_profile", "accepted_decoder_source",
}
FIELDS = {
    "authority": "manager_decision manager_wait_resolved_at objective_fingerprint operator_objective disposition independent_review runtime_launch_authorized production_policy_adopted",
    "identities": "arithmetic_id primitive_policy_id primitive_state_id decoder_policy_id decoder_state_id decoder_public_abi decoder_runtime_schema base_reference_policy local_reference_id global_profile original_independent_freeze_sha256 claim_label model_id",
    "precision": "weight_bits group_size asymmetric gemm_nibble_order qzero_plus_one scales operator_activations kv state_width fraction_bits zero_tag_bits addition_width unchanged_strict_w4a16 other_precision_modes_supported",
    "scope": "specified_layers positions token_history hidden_size execution_cone excluded",
    "state": "origin root_artifact admitted_q24_parent root_production fp16_only_nonroot_seed reference_derived_carry primitive_fixture_as_model_parent payload_bytes record_bytes record_order tag_rule rounding finite_view_integer_limit_exclusive save_boundary kv_lineage",
    "references": "local_stages local_gate s12_operands s13_input s18 negative_excess_clamping global_candidate_reseeding s18_exact_state_transition legacy_fp16_trajectory missing_mandatory_evidence",
    "public_interface": "module base_header_binding sideband_specification_binding sideband_section parameters compatibility_aliases literal_candidate_header compile_status first_runtime_prerequisite",
    "status": "decoder_model_runtime simulator_or_oracle L0_L9_numerical_correctness downstream_model_scopes state_and_kv_admission hardware",
    "validation": "command python_version policy numerical_inputs score",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def unique_fields(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON field: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def load_json(data: str | bytes) -> dict:
    result = json.loads(
        data, object_pairs_hook=unique_fields, parse_constant=reject_constant,
    )
    require(type(result) is dict, "JSON root must be an object")
    return result


def validate_schema(packet: dict) -> None:
    require(set(packet) == SECTIONS, "unknown or missing packet section")
    require(type(packet["schema_version"]) is int and packet["schema_version"] == 1,
            "unsupported schema")
    require(packet["attempt_id"] == "q24_decoder_adoption_c39ca7b7722a_attempt001",
            "wrong attempt identity")
    require(packet["mission_id"] == "c39ca7b7722a", "wrong mission")
    require(packet["kind"] == "non_executing_decoder_adoption_specification",
            "wrong artifact kind")
    for section in SECTIONS - {
        "schema_version", "attempt_id", "mission_id", "kind", "specification",
    }:
        require(type(packet[section]) is dict, f"{section} must be an object")
    for section, fields in FIELDS.items():
        require(set(packet[section]) == set(fields.split()),
                f"unknown or missing fields: {section}")
    require(packet["specification"] ==
            "ace3/contracts/candidates/residual_exact_grid_q24_decoder_adoption_c39ca7b7722a_attempt001.md",
            "wrong normative specification")
    require(set(packet["bindings"]) == BINDINGS, "missing or unknown binding")
    for name, record in packet["bindings"].items():
        require(type(record) is dict and set(record) == {"path", "sha256"},
                f"invalid binding schema: {name}")
        require(type(record["path"]) is str and bool(record["path"]),
                f"invalid binding path: {name}")
        digest = record["sha256"]
        require(type(digest) is str and len(digest) == 64
                and all(c in "0123456789abcdef" for c in digest),
                f"invalid binding digest: {name}")
    authority = packet["authority"]
    require(authority["runtime_launch_authorized"] is False
            and authority["production_policy_adopted"] is False,
            "specification must not grant adoption or runtime")
    require(authority["independent_review"] == "PENDING_NORMAL_HOST_REVIEWER",
            "Engineer must not mint review acceptance")
    require(authority["manager_decision"] == "authorize_bounded_semantics_specification"
            and authority["disposition"] ==
            "decoder_model_root_specification_selected_not_runtime_adopted",
            "authority disposition drift")
    scope = packet["scope"]
    require(scope["specified_layers"] == list(range(10))
            and scope["positions"] == [0] and scope["token_history"] == [9707]
            and scope["execution_cone"] == [], "scope expansion")
    require(scope["excluded"] == [
        "FP16-only non-root seeding", "invented carry", "accepted FP16 replay",
        "failed-state forwarding", "L10-L23", "later positions", "final RMSNorm",
        "lm_head", "top-k", "tokenizer", "dialogue", "synthesis", "timing/PPA",
        "FPGA", "hardware",
    ], "exclusions changed")
    state = packet["state"]
    require(state["origin"] == "authenticated_model_root_embedding_only"
            and state["root_artifact"] is None
            and state["admitted_q24_parent"] is None
            and state["root_production"] == "NO_EXECUTION", "invented root or parent")
    for key in ("fp16_only_nonroot_seed", "reference_derived_carry",
                "primitive_fixture_as_model_parent"):
        require(state[key] is False, f"forbidden state source: {key}")
    require(state["payload_bytes"] == 8064 and state["record_bytes"] == 9
            and scope["hidden_size"] == 896, "state geometry")
    require(state["record_order"] ==
            "coordinate_ascending_signed_int64_little_endian_then_u8_zero_tag"
            and state["tag_rule"] ==
            "0 or 1; nonzero integer requires 0; exact cancellation gives +0; only -0 plus -0 gives -0"
            and state["rounding"] == "RNE16_gradual_underflow_no_saturation"
            and state["finite_view_integer_limit_exclusive"] == 1099243192320
            and state["save_boundary"] == "completed_idle_admitted_whole_layer_only"
            and state["kv_lineage"] == "separate_own_layer_fp16_causal_state",
            "state semantics changed")
    precision = packet["precision"]
    require(precision["state_width"] == 64 and precision["fraction_bits"] == 24
            and precision["zero_tag_bits"] == 1 and precision["addition_width"] == 65,
            "residual precision changed")
    require(precision["weight_bits"] == 4 and precision["group_size"] == 128
            and precision["asymmetric"] is True
            and precision["gemm_nibble_order"] == [0, 4, 1, 5, 2, 6, 3, 7]
            and precision["qzero_plus_one"] is False
            and all(precision[k] == "FP16" for k in ("scales", "operator_activations", "kv"))
            and precision["unchanged_strict_w4a16"] is False
            and precision["other_precision_modes_supported"] == [], "precision drift")
    require(packet["status"] == {
        "decoder_model_runtime": "NO_EXECUTION",
        "simulator_or_oracle": "NO_EXECUTION",
        "L0_L9_numerical_correctness": "NOT_EVALUATED",
        "downstream_model_scopes": "NOT_EVALUATED",
        "state_and_kv_admission": "NOT_EVALUATED",
        "hardware": "NOT_EVALUATED",
    }, "invalid result boundary")
    refs = packet["references"]
    require(refs["local_stages"] == list(range(18))
            and refs["s12_operands"] == ["authenticated_I", "authenticated_Z", "actual_S11_O"]
            and refs["s13_input"] == "actual_rounded_S12_FP16"
            and refs["negative_excess_clamping"] is False
            and refs["global_candidate_reseeding"] is False
            and refs["missing_mandatory_evidence"] == "BLOCKED_NOT_EVALUATED",
            "reference boundary changed")
    require(refs["local_gate"] ==
            "finite AND (abs_error <= 1/8 OR (relative_error < 1/1000 AND ordered_FP16_ULP <= 1)); denominator=max(abs(independent_local_FP16_reference),2^-14)",
            "local threshold drift")
    require(refs["s18"] ==
            "ORIGINAL-input independent binary64 r; finite abs(r)<=65504; exact abs(actual_FP16-r)-min_finite_FP16_h(abs(h-r))<=1/8",
            "global threshold drift")
    ids = packet["identities"]
    require(ids["decoder_policy_id"] ==
            "ace3-decoder-residual-exact-grid-q24-local-global-policy-v1"
            and ids["decoder_state_id"] == "ace3-decoder-residual-exact-grid-q24-state-v1",
            "decoder must not reuse isolated policy/state identity")
    require(ids["local_reference_id"] ==
            "canonical-awq-q24-residual-operator-fp16-boundary-p0-v1"
            and ids["decoder_public_abi"] == "ace3-decoder-q24-public-abi-v1"
            and ids["decoder_runtime_schema"] == "ace3-q24-actual-runtime-v1",
            "prospective decoder identity changed")
    require(packet["public_interface"]["literal_candidate_header"] is None
            and packet["public_interface"]["compile_status"] == "NO_EXECUTION"
            and packet["public_interface"]["compatibility_aliases"] == [],
            "unsupported compiled ABI claim")
    require(packet["public_interface"]["module"] == "ace3_decoder_token_engine_q24_v1"
            and packet["public_interface"]["base_header_binding"] == "accepted_decoder_source"
            and packet["public_interface"]["sideband_specification_binding"] == "source_change_plan"
            and packet["public_interface"]["parameters"] == {
                "LAYER_INDEX": 0, "ACCURATE_SILU": "(LAYER_INDEX >= 3)",
                "RESIDUAL_WIDTH": 64, "RESIDUAL_FRAC_BITS": 24,
            }, "prospective public contract changed")
    require(packet["validation"]["numerical_inputs"] == [], "unexpected numerical inputs")


def validate_bindings(packet: dict) -> int:
    records = {}
    for name, binding in packet["bindings"].items():
        data = (ROOT / binding["path"]).read_bytes()
        require(hashlib.sha256(data).hexdigest() == binding["sha256"],
                f"source identity mismatch: {name}")
        if binding["path"].endswith(".json"):
            records[name] = load_json(data)
    wait, directive = records["manager_wait"], records["manager_directive"]
    require(wait["active"] is False and wait["operator_action_required"] is False,
            "Manager wait is unresolved")
    require(wait["objective_fingerprint"] == directive["objective_sha256"]
            == packet["authority"]["objective_fingerprint"], "authority objective mismatch")
    require(wait["manager_resolution"]["resolved_at"]
            == packet["authority"]["manager_wait_resolved_at"]
            > directive["set_at"], "authority chronology mismatch")
    require(wait["manager_resolution"]["target_stage"] == "rtl",
            "unexpected Manager stage")
    require(records["mission"]["mission_id"] == packet["mission_id"],
            "bound mission identity mismatch")
    for key in ("primitive_review", "isolated_authority_review", "frontier_review"):
        review = records[key]
        require(review["kind"] == "round_reviewed_handoff"
                and review["producer_role"] == "reviewer"
                and review["review"]["status"] == "done", f"not a genuine done receipt: {key}")
    require(records["primitive_freeze"]["scope"] == "isolated_primitive_reference_codec"
            and records["primitive_result"]["attempt_id"]
            == "q24_primitive_a8b9021c084b_attempt003"
            and records["primitive_result"]["decoder_controller_model_tokenizer_token_kv"]
            == "NOT_EVALUATED", "primitive scope changed")
    ids = packet["identities"]
    for key, field, expected in (
        ("primitive_arithmetic", "arithmetic_id", "arithmetic_id"),
        ("primitive_policy", "policy_id", "primitive_policy_id"),
        ("primitive_state", "state_id", "primitive_state_id"),
        ("v3_policy", "policy_id", "base_reference_policy"),
        ("v3_policy", "numerical_profile", "global_profile"),
        ("v3_policy", "trusted_independent_freeze_sha256", "original_independent_freeze_sha256"),
    ):
        require(records[key][field] == ids[expected], f"policy identity mismatch: {expected}")
    require("isolated primitive" in records["primitive_policy"]["adoption"],
            "isolated policy adoption changed")
    return len(records)
