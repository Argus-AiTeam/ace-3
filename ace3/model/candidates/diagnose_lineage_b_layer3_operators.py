"""Retained lineage-B admission with an exact down-projection error telescope."""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ace3.model.candidates import diagnose_lineage_b_layer3 as prior

PARENT = ROOT / "build/lineage_b_layer3_repair_f6846b6185e4_attempt001"
VARIANTS = (
    "independent_binary64_gated",
    "actual_residual1_ideal_remaining_mlp",
    "actual_norm2_ideal_remaining_mlp",
    "own_norm2_projection_fp16_rne_ideal_silu",
    "actual_gate_up_ideal_silu",
    "actual_gate_up_ideal_silu_fp16_rne",
    "actual_gated_exact_down",
)


def exact_dot(values, weights):
    return sum((prior.exact(a) * prior.exact(b)
                for a, b in zip(values, weights, strict=True)), Fraction())


def diagnose(first, outputs, frozen, backend):
    import numpy as np
    import torch
    import torch.nn.functional as functional

    layer, position, index = (first[k] for k in ("layer", "position", "index"))
    prior.require(layer == 3 and position == 0, "unexpected first-failure cone")
    item = outputs[layer, position]
    receipt = prior.load(item["receipt"]["path"])
    trace_binding = receipt["transaction"]["raw"]["trace"]
    trace = Path(trace_binding["path"])
    current_trace = prior.record(trace)
    prior.require(all(current_trace[k] == trace_binding[k] for k in ("path", "bytes", "sha256")),
                  "retained trace binding drift")
    stages = {}
    for text in trace.read_text().splitlines():
        word = int(text, 16)
        pos, stage, channel = (word >> 40) & 0xFFFF, (word >> 32) & 0xFF, (word >> 16) & 0xFFFF
        prior.require(pos == position, "foreign trace position")
        if 11 <= stage <= 18:
            prior.require((stage, channel) not in stages, "duplicate trace coordinate")
            stages[stage, channel] = word & 0xFFFF
    sizes = {s: 4864 if s in (14, 15, 16) else 896 for s in range(11, 19)}
    prior.require(set(stages) == {(s, i) for s, n in sizes.items() for i in range(n)},
                  "incomplete MLP/residual trace")
    actual = {s: np.array([prior.half(stages[s, i]) for i in range(n)])
              for s, n in sizes.items()}
    prior.require(all(np.isfinite(a).all() for a in actual.values()), "nonfinite trace")
    state = backend.reference_layer(Path(frozen["checkpoint"]["path"]), layer, None)
    reference_input = np.load(outputs[layer - 1, position]["reference"]["path"], allow_pickle=False)
    reference_final = np.load(item["reference"]["path"], allow_pickle=False)
    captured = {}
    original = backend.dialogue._reference_projection

    def capture(activation, projection):
        result = original(activation, projection)
        name = next(k for k, v in state.projections.items() if v is projection)
        captured[name] = (activation.detach().numpy().reshape(-1).copy(),
                          result.detach().numpy().reshape(-1).copy())
        return result

    backend.dialogue._reference_projection = capture
    try:
        with torch.no_grad():
            reproduced = backend.dialogue._reference_layer_step(
                state, torch.from_numpy(reference_input.reshape(1, 896)), position
            ).numpy().reshape(-1)
    finally:
        backend.dialogue._reference_projection = original
    prior.require(np.array_equal(reproduced, reference_final), "independent recurrence drift")

    def silu_product(gate, up):
        return (functional.silu(gate) * up).numpy().reshape(-1)

    def project(norm):
        return tuple(original(norm, state.projections[name]) for name in ("gate", "up"))

    def rounded(values):
        return np.array([prior.half(prior.rne(float(x))) for x in values.reshape(-1)])

    own_residual = torch.from_numpy(actual[12].reshape(1, 896))
    ideal_norm = backend.dialogue._torch_rmsnorm(own_residual, state.post_attention_norm)
    own_norm = torch.from_numpy(actual[13].reshape(1, 896))
    ideal_gate, ideal_up = project(own_norm)
    gate_up = tuple(torch.from_numpy(actual[s]) for s in (14, 15))
    ideal_silu = silu_product(*gate_up)
    variants = (
        captured["down"][0],
        silu_product(*project(ideal_norm)),
        silu_product(ideal_gate, ideal_up),
        silu_product(*(torch.from_numpy(rounded(x.numpy())) for x in (ideal_gate, ideal_up))),
        ideal_silu,
        rounded(ideal_silu),
        actual[16],
    )
    weight = state.projections["down"].reference_weight[:, index].numpy()
    dots = [exact_dot(v, weight) for v in variants]
    ref_down = prior.exact(captured["down"][1][index])
    actual_down = prior.exact(actual[17][index])
    terms = {"binary64_dot_rounding": dots[0] - ref_down}
    terms.update({name: after - before for name, before, after
                  in zip(VARIANTS[1:], dots[:-1], dots[1:], strict=True)})
    terms["down_output_rounding_or_defect"] = actual_down - dots[-1]
    prior.require(sum(terms.values()) == actual_down - ref_down,
                  "down error telescope does not close")
    exact_down = dots[-1]
    # FP16 operands and native FP16 scales are dyadic; retain an exact rational dot.
    prior.require(prior.exact(float(exact_down)) == exact_down,
                  "binary64 cannot exactly hold this dot; do not double-round the oracle")
    down_rne = prior.rne(float(exact_down))
    down_agrees = down_rne == stages[17, index]
    res1 = prior.exact(actual[12][index])
    final_sum = res1 + actual_down
    prior.require(prior.rne(float(final_sum)) == stages[18, index], "residual RNE defect")
    prior.require(stages[18, index] == int(backend.engine.load_hidden_bits(
        Path(item["actual"]["path"]))[index]), "final/trace mismatch")

    contributions = [(i, prior.exact(a - b) * prior.exact(w)) for i, (a, b, w)
                     in enumerate(zip(actual[16], variants[5], weight, strict=True))]
    contributors = sorted(contributions, key=lambda x: (-abs(x[1]), x[0]))[:16]
    local = []
    for channel, contribution in contributors:
        projections = {}
        for name, stage in (("gate", 14), ("up", 15)):
            dot = exact_dot(actual[13], state.projections[name].reference_weight[:, channel].numpy())
            prior.require(prior.exact(float(dot)) == dot, "inexact projection oracle conversion")
            bits = prior.rne(float(dot))
            projections[name] = {
                "exact_dot": str(dot), "rne_bits": f"{bits:04x}",
                "actual_bits": f"{stages[stage, channel]:04x}",
                "own_operand_rne_agrees": bits == stages[stage, channel],
            }
        local.append({
            "channel": channel, "weighted_silu_difference": str(contribution),
            "gate": actual[14][channel], "up": actual[15][channel],
            "actual_gated": actual[16][channel], "ideal_silu_product": ideal_silu[channel],
            "ideal_gated_rne": variants[5][channel], "projections": projections,
        })
    fused_sum = res1 + exact_down
    prior.require(prior.exact(float(fused_sum)) == fused_sum, "inexact fused diagnostic conversion")
    fused_bits = prior.rne(float(fused_sum))
    return {
        "scope": "Conditional software operator diagnosis only, no replacement of actual/reference history",
        "coordinate": {k: first[k] for k in ("layer", "position", "index")},
        "independent_reference_bit_exact_elements": len(reproduced),
        "operator_variants": [
            {"name": name, "exact_down": str(dot), "decimal_display_only": float(dot)}
            for name, dot in zip(VARIANTS, dots, strict=True)
        ],
        "down_error_terms_exact": {k: str(v) for k, v in terms.items()},
        "down_error_terms_decimal_display_only": {k: float(v) for k, v in terms.items()},
        "actual_down": float(actual_down), "reference_down": float(ref_down),
        "exact_own_operand_down": str(exact_down), "own_operand_down_rne_bits": f"{down_rne:04x}",
        "own_operand_down_rne_agrees": down_agrees,
        "final_residual_exact_sum": str(final_sum), "final_residual_rne_agrees": True,
        "silu_own_operand_ideal_rne_differences": int(np.count_nonzero(actual[16] != variants[5])),
        "largest_silu_weighted_differences": local,
        "fused_down_residual_counterfactual": {
            "exact_sum": str(fused_sum), "fp16_bits": f"{fused_bits:04x}",
            **prior.independent_scalar(fused_bits, float(reference_final[index])),
            "scope": "Single scalar sensitivity only; not RTL evidence or an admitted repair",
        },
        "rtl_repair_justified": False,
        "remaining_question": "Which general upstream arithmetic treatment improves the independent "
                              "trajectory while preserving the unchanged FP16 interfaces and gates?",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    prior.require(output.is_relative_to(ROOT / "build"), "evidence must remain under build")
    output.mkdir()
    started = time.monotonic()
    parent = prior.load(PARENT / "result.json")
    old = prior.load(prior.ADMISSION / "freeze.json")
    prior.require(prior.record(PARENT / "freeze.json") == parent["freeze"], "parent freeze drift")
    for binding in old["sources_and_inputs"]:
        current = prior.record(binding["path"])
        prior.require(all(current[k] == binding[k] for k in ("path", "bytes", "sha256")),
                      f"bound source/input drift: {binding['path']}")
    parent_freeze = prior.load(PARENT / "freeze.json")
    prior.require(prior.record(prior.ADMISSION / "freeze.json") ==
                  parent_freeze["parent_admission_freeze"], "admission freeze drift")
    prior.require(prior.record(prior.__file__) == parent_freeze["source"], "parent source drift")
    frozen = prior.load(old["lineage_B_freeze"]["path"])
    prior.require(frozen["token_history"] == [9707, 1879, 0], "foreign token roots")
    outputs = {(v["layer"], v["position"]): v for v in old["outputs"]}
    prior.require(list(outputs) == [(l, p) for l in range(9) for p in range(3)], "scope/order mismatch")
    sys.path.insert(0, str(prior.ARCHIVE / "ace3/model"))
    import continuous_kv_rtl_backend as backend
    import numpy as np
    import torch

    bindings = {v["path"] for v in old["sources_and_inputs"]}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).resolve().is_relative_to(prior.ARCHIVE):
            prior.require(str(Path(filename).resolve()) in bindings, "unbound reference import")
    prior.write(output / "freeze.json", {
        "mission": "1cbfde5e5d07", "attempt": output.name, "source": prior.record(__file__),
        "parent_result": prior.record(PARENT / "result.json"), "parent_freeze": parent["freeze"],
        "admission_freeze": prior.record(prior.ADMISSION / "freeze.json"),
        "profile": prior.profile.PROFILE_ID, "reference_policy": prior.profile.REFERENCE_POLICY,
        "policy": "finite valid operands; abs(reference)<=65504; exact excess_error<=1/8",
        "inputs": old["outputs"], "lineage_B_freeze": old["lineage_B_freeze"],
        "public_interfaces": frozen["public_interfaces"], "parameters": frozen["parameters"],
        "retained_compile_commands": frozen["compile_commands"], "retained_tools": frozen["tools"],
        "python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
        "order": "layer, position, scalar ascending; stop at first rejection",
        "diagnostic_selection": "First failure, position-zero layer3 MLP; all stage13-16 elements; "
                                "16 largest weighted own-input SiLU differences, ties by index",
        "diagnostic_variants": VARIANTS, "command": sys.argv, "new_RTL_simulations": 0,
    })
    first, count, complete = None, 0, 0
    with (output / "scalar_results.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "layer", "position", "index", "actual_fp16_bits", "reference_binary64_hex",
            "nearest_fp16_bits", "actual_error", "q", "excess_error", "accepted",
        ])
        writer.writeheader()
        for (layer, position), item in outputs.items():
            actual = backend.engine.load_hidden_bits(Path(item["actual"]["path"]))
            reference = np.load(item["reference"]["path"], allow_pickle=False)
            prior.require(actual.shape == reference.shape == (896,) and reference.dtype == np.float64,
                          "invalid array shape/dtype")
            for index, (bits, ref) in enumerate(zip(actual, reference, strict=True)):
                measured = prior.profile.evaluate_layer_final_output(
                    actual_fp16_bits=int(bits), reference_binary64_hex=float(ref).hex())
                oracle = prior.independent_scalar(int(bits), float(ref))
                prior.require(all(measured[k] == v for k, v in oracle.items()), "scalar oracle disagreement")
                row = {"layer": layer, "position": position, "index": index,
                       "actual_fp16_bits": f"{int(bits):04x}",
                       "reference_binary64_hex": float(ref).hex(), **oracle}
                writer.writerow(row)
                count += 1
                if not measured["accepted"]:
                    first = row
                    break
            if first:
                break
            complete += 1
    diagnosis = diagnose(first, outputs, frozen, backend) if first else None
    if first:
        prior.require(count == (first["layer"] * 3 + first["position"]) * 896 + first["index"] + 1,
                      "earliest-failure stop defect")
        prior.write(output / "operator_diagnosis.json", diagnosis)
    result = {
        "status": "RETAINED_OUTPUT_BINARY64_V1_FAIL" if first else "RETAINED_OUTPUT_BINARY64_V1_PASS",
        "admitted": first is None, "first_failure": first,
        "complete_passing_outputs": complete, "scalar_count": count,
        "independent_scalar_oracle_agreement_count": count,
        "freeze": prior.record(output / "freeze.json"),
        "scalars": prior.record(output / "scalar_results.csv"),
        "diagnosis": prior.record(output / "operator_diagnosis.json") if diagnosis else None,
        "failure_taxonomy": "retained_layer_final_excess_error" if first else None,
        "root_cause_hypothesis": "Upstream MLP operand trajectory and interstage rounding; "
                                 "operator telescope distinguishes local down rounding from its inputs.",
        "regression": "Independent scalar oracle, earliest-stop identity, exact error telescope, "
                      "source-bound independent binary64 layer reproduction and own-input rational dots",
        "rtl_arithmetic_changed": False, "new_RTL_simulations": 0,
        "lineage_A_actual_or_state_imported": False, "position3_executed": False,
        "tail_executed": False, "full_driver_admitted": False,
        "normal_host_review": "PENDING", "elapsed_seconds": time.monotonic() - started,
    }
    prior.write(output / "result.json", result)
    print(prior.json.dumps(result, sort_keys=True))
    if diagnosis:
        print(prior.json.dumps({k: diagnosis[k] for k in (
            "down_error_terms_decimal_display_only", "own_operand_down_rne_agrees",
            "silu_own_operand_ideal_rne_differences", "fused_down_residual_counterfactual")}, sort_keys=True))
    return 1 if first else 0


if __name__ == "__main__":
    raise SystemExit(main())
