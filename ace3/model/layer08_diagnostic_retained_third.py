#!/usr/bin/env python3
"""Exact retained originalA accounting; no trajectory intervention or RTL replay."""

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import time
from fractions import Fraction

sys.dont_write_bytecode = True

import numpy as np

import diagnose_lineage_a_position3_boundary as boundary
from retained_causal_repair_campaign import projection_contributions, records
from retained_layer00_stage18_producer_split_diagnostic import record, require

BUILD = boundary.BUILD
PREFIX = boundary.PREFIX
SHARED = BUILD / "retained_l2p0s16_90efa10b0762_attempt002"
QPRODUCER = (BUILD / "model24_selected_token_position3_continuations"
             / "lineage_a_q_producer_968f6ec61614_attempt002")
PRIOR = BUILD / "lineage_a_position3_boundary_e4fa3091ffbd_attempt001"
SUCCESSOR = (BUILD / "model24_tokenizer_host_integration_attempt001"
             / "position2_successor_attempt001")
REVIEWS = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs")


def rational(n, denominator=1 << 48):
    return str(Fraction(n, denominator))


def half(value):
    return int.from_bytes(struct.pack("<e", value), "little")


def floating(bits):
    boundary.units(bits)
    return struct.unpack("<e", int(bits).to_bytes(2, "little"))[0]


def dot_accounting(q_actual, k_actual, q_reference, k_reference):
    require(len(q_actual) == len(k_actual) == len(q_reference) == len(k_reference) == 64,
            "attention dot must have exactly 64 dimensions")
    terms = []
    for i, (qa, ka, qr, kr) in enumerate(zip(
            q_actual, k_actual, q_reference, k_reference, strict=True)):
        a, b, r, s = map(boundary.units, (qa, ka, qr, kr))
        terms.append(dict(
            dimension=i, actual_q=f"{qa:04x}", reference_q=f"{qr:04x}",
            actual_k=f"{ka:04x}", reference_k=f"{kr:04x}",
            actual_q48=a*b, reference_q48=r*s,
            q_delta_q48=(a-r)*s, k_delta_q48=r*(b-s),
            interaction_q48=(a-r)*(b-s)))
    sums = {key: sum(t[key] for t in terms) for key in (
        "actual_q48", "reference_q48", "q_delta_q48", "k_delta_q48", "interaction_q48")}
    require(sums["actual_q48"] - sums["reference_q48"]
            == sums["q_delta_q48"] + sums["k_delta_q48"] + sums["interaction_q48"],
            "score contribution identity")
    for label, q, k in (("actual", q_actual, k_actual),
                         ("reference", q_reference, k_reference)):
        exact = boundary.rounded_dot(sums[label + "_q48"])
        independent = half(math.fsum(floating(a)*floating(b)
                                     for a, b in zip(q, k, strict=True)) / 8)
        require(exact == independent, "integer dot / independent fsum disagreement")
        sums[label + "_bits"] = f"{exact:04x}"
    return dict(terms=terms, totals=sums,
                score_units={k: rational(v, 1 << 51) for k, v in sums.items()
                             if k.endswith("_q48")})


def project(activation, tensors, channel):
    terms = projection_contributions(activation, tensors, channel, boundary)
    bias = boundary.units(tensors.get("bias", [0] * len(tensors["scales"]))[channel])
    total = sum(t["contribution_q48"] for t in terms) + (bias << 24)
    exact = boundary.rounded_dot(total << 3)
    independent = half(math.fsum(
        floating(int(t["activation"], 16)) * (t["qweight"]-t["qzero"])
        * floating(int(t["scale"], 16)) for t in terms) + bias / 2**24)
    require(exact == independent, "native AWQ exact projection / fsum disagreement")
    return exact, total, terms


def run(out):
    started = time.monotonic()
    require(out.parent == BUILD and out.name.startswith("retained_")
            and "third" in out.name, "output outside writable diagnostic namespace")
    require(out.is_dir() and (out / "command.sh").is_file(), "missing command sidecar")
    require(not (out / "frozen.json").exists(), "attempt already used")
    inputs = boundary.Inputs()

    def load_root(path):
        return inputs.load(path, root=True)

    def load(path):
        return inputs.load(path)

    def read(path):
        return inputs.read(path)

    def write(name, value):
        boundary.write(out / name, value)

    def checked_review(mission, round_number):
        receipt = load_root(REVIEWS / mission / f"round-{round_number:04d}.json")
        require(receipt["kind"] == "round_reviewed_handoff"
                and receipt["producer_role"] == "reviewer"
                and receipt["mission_id"] == mission
                and receipt["review"]["status"] == "done", "genuine review missing")
        return receipt

    review_receipts = {m: checked_review(m, r) for m, r in (
        ("1693a83a95c6", 3), ("1f9852a2d4f5", 2),
        ("968f6ec61614", 1), ("90efa10b0762", 1))}
    for directory in (boundary.TAIL, boundary.LAYER0, PREFIX, QPRODUCER):
        load_root(directory / "seal.json")
    load_root(SHARED / "artifacts.json")
    shared_freeze = load(SHARED / "frozen.json")
    shared_result = load(SHARED / "result.json")
    for mission, rec in shared_freeze["reviews"].items():
        inputs.register(rec)
        receipt = load(rec["path"])
        require(receipt["producer_role"] == "reviewer" and receipt["mission_id"] == mission
                and receipt["review"]["status"] == "done", "RMSNorm arm review missing")
        review_receipts[mission] = receipt
    load_root(PRIOR / "frozen.json")
    prior_result = load_root(PRIOR / "result.json")
    qfreeze = load(QPRODUCER / "frozen.json")
    qresult = load(QPRODUCER / "result.json")
    norm = load(QPRODUCER / "stage00.json")
    frozen = load(PREFIX / "frozen.json")
    p8 = load(PREFIX / "layer08/prepared.json")
    p7 = load(PREFIX / "layer07/prepared.json")
    r7 = load(PREFIX / "layer07/result.json")
    r8 = load(PREFIX / "layer08/result.json")
    tail = load(boundary.TAIL / "result.json")
    successor_freeze = load_root(SUCCESSOR / "freeze.json")
    successor_result = load_root(SUCCESSOR / "result.json")
    inputs.register(successor_result)
    load_root(SUCCESSOR / "manifest.json")
    successor = load(SUCCESSOR / "integration.json")
    require(tail["natural_exit"] and tail["runtime_tail_invocations"] == 1
            and tail["selected_token_id"] == 358
            and all(tail[k] == 0 for k in (
                "final_rmsnorm_material_mismatches", "logit_mismatches",
                "selected_accumulator_mismatches", "topk_mismatches")),
            "accepted tail outcome changed")
    require(successor["host_result"]["resulting_token_history"] == boundary.HISTORY
            and successor["selection"]["selected_token_id"] == 358
            and successor["source_lineage"]["tail_attempt"] == str(boundary.TAIL),
            "token358 successor history changed")
    # Read only lineage records, not the 151936-logit array or any simulator.
    selected = load(boundary.TAIL / "selected_token.json")
    require(selected["actual_token_id"] == selected["expected_token_id"] == 358
            and selected["matches"], "selected token binding")
    require(successor["source_lineage"]["selected_token"] == inputs.records[
        str(boundary.TAIL / "selected_token.json")], "successor selected-token identity")
    read(selected["parent_layer23"]["path"])
    read(selected["tail_input"]["path"])
    read(successor["source_lineage"]["layer23"]["terminal"]["path"])
    require(p8["input_hidden"] == r7["output_hidden"]
            and p8["binary"] == p8["kv_parent"]["live_binary"]
            and p8["kv_parent"]["valid_positions"] == [0, 1, 2]
            and p8["kv_parent"]["layer_index"] == 8, "hidden/state ABI mismatch")
    require("/model24_persistent_kv_selected_token_corrected_q_attempt005/"
            in p8["kv_parent"]["state"]["path"], "not originalA historical state")
    read(p8["binary"]["path"])
    read(p8["kv_parent"]["state"]["path"])
    actual = boundary.decode_trace(read(PREFIX / "layer08/position003/raw/trace.hex"), 3)
    actual7 = boundary.decode_trace(read(PREFIX / "layer07/position003/raw/trace.hex"), 3)
    incoming_bytes = read(p8["vectors"]["input"]["path"])
    require(incoming_bytes == read(r7["output_hidden"]["path"]), "hidden bytes differ")
    incoming_rows = incoming_bytes.decode("ascii").splitlines()
    require(len(incoming_rows) == 896 and all(
        len(row) == 10 and int(row[:6], 16) == i for i, row in enumerate(incoming_rows)),
        "hidden framing")
    incoming = [int(row[-4:], 16) for row in incoming_rows]
    require(incoming == actual7[18], "hidden does not match residual trace")

    def reference(prepared, stage):
        rows = read(prepared["independent_stages"][str(stage)]["path"]).splitlines()
        require(all(len(row) == 4 for row in rows), "reference FP16 framing")
        return [int(row, 16) for row in rows]

    ref = {s: reference(p8, s) for s in (0, 1, 2, 3, 4, 5, 6, 7, 8)}
    ref7 = {s: reference(p7, s) for s in (12, 17, 18)}
    history = [boundary.decode_trace(read(rec["path"]), p)
               for p, rec in enumerate(p8["kv_parent"]["actual_kv_traces"])]
    require(len(history) == 3, "historical K/V count")
    actual_k = [x[6] for x in history] + [actual[6]]
    parent = load(p8["independent_parent"]["path"])
    require(parent["history"] == boundary.HISTORY[:3] and parent["layer"] == 8,
            "reference K history")
    rk = np.load(io.BytesIO(read(PREFIX / "layer08/independent/position003_own_k.npy")),
                 allow_pickle=False)
    pk = np.load(io.BytesIO(read(parent["own_cache"]["k"]["path"])), allow_pickle=False)
    require(rk.dtype == np.dtype("<u2") and rk.shape == (4, 2, 64)
            and np.array_equal(rk[:3], pk), "reference cache append identity")
    reference_k = rk.reshape(4, 128).tolist()
    require(actual[5] == actual[6] and actual[3] == actual[7]
            and ref[5] == ref[6] == reference_k[3] and ref[3] == ref[7],
            "current K/V cache copy mismatch")
    tensors = {}
    for item in p8["vectors"]["tensors"]:
        meta = item["checkpoint_tensor"]
        if "self_attn.q_proj." not in meta["name"]:
            continue
        words = [int(row, 16) for row in read(item["serialized"]["path"]).splitlines()]
        payload = b"".join(x.to_bytes(2 if meta["dtype"] == "F16" else 4, "little")
                           for x in words)
        require(len(payload) == meta["bytes"]
                and hashlib.sha256(payload).hexdigest() == meta["sha256"],
                "official native AWQ tensor content mismatch")
        require(any(t["name"] == meta["name"] and t["sha256"] == meta["sha256"]
                    for t in parent["checkpoint_tensor_hashes"]), "reference tensor identity")
        tensors[meta["name"].rsplit(".", 1)[1]] = words
    require(set(tensors) == {"qweight", "qzeros", "scales", "bias"}, "Q tensors incomplete")
    earliest_arms, rejections, spines = {}, {}, {}
    for arm, dirname in (
            ("quotient_only", "retained_quotient_only_0fe3b4cb1657_attempt003"),
            ("root_only", "retained_root_only_371d2a404cad_attempt002"),
            ("combined_mathematical", "retained_combined_mathematical_rmsnorm_08e70e80a3ea_attempt002")):
        load_root(BUILD / dirname / "artifacts.json")
        rejected = load(BUILD / dirname / "result.json")
        first = rejected["first_material_failure"]
        require(rejected["status"] == "CANDIDATE_REJECTED"
                and rejected["frozen_control_bit_differences"] == 0
                and (first["layer"], first["position"], first["stage"]) == (2, 0, 16)
                and [x["index"] for x in first["comparison"]["failures"]] == [813],
                "reviewed candidate rejection changed")
        rejections[arm] = first
        earliest_arms[arm] = load(SHARED / f"{arm}_earliest_producer.json")
        spines[arm] = load(SHARED / f"{arm}_contribution_spines.json")
    require(all(f["comparison_policy"] == boundary.POLICY for f in (
        frozen, qfreeze, shared_freeze)), "numerical policy drift")
    require(shared_freeze["public_port_contract"] == frozen["public_interfaces"]["decoder"],
            "ambiguous_objective: retained public decoder contracts differ")
    compile_argv = list(shared_freeze["compile_argv"])
    compile_argv[compile_argv.index("-o") + 1] = str(out / "public_contracts.vvp")
    compile_argv = [x.replace("ace3_decoder_layer0_token_engine.LAYER_INDEX=0",
                              "ace3_decoder_layer0_token_engine.LAYER_INDEX=8")
                    for x in compile_argv]
    for path in compile_argv:
        if path.endswith(".sv"):
            read(path)
    sources = [record(Path(__file__))]
    for name in ("diagnose_lineage_a_position3_boundary.py", "retained_causal_repair_campaign.py",
                 "retained_layer00_stage18_producer_split_diagnostic.py"):
        sources.append(record(Path(__file__).with_name(name)))
    write("frozen.json", dict(
        objective="Full-head originalA L8/P3/S8 producer accounting and rejected-arm cone join",
        candidate_set=[], interventions=[], token_history=boundary.HISTORY,
        public_interfaces=frozen["public_interfaces"], profile=frozen["profile"],
        comparison_policy=boundary.POLICY, sources=sources,
        consumed=list(inputs.records.values()), reviews=review_receipts,
        compile_argv=compile_argv, command=record(out / "command.sh"),
        compile_scope="Public decoder ABI only, retained source set; not originalA numerical execution",
        python=sys.version, numpy=np.__version__,
        method="Exact signed algebra on retained operands; independent struct/floating fsum cross-check; "
               "full 64-channel Q projection and 896 residual terms; no substituted output trajectories.",
        score_policy="Original FP16 interstage gate; binary64 admission not evaluated",
        stop_rule="Any identity, own-input or independent arithmetic disagreement stops this diagnostic",
        RTL_invocations=0, tail_replays=0, token358_executions=0))
    with (out / "compile.log").open("x") as log:
        compiled = subprocess.run(compile_argv, stdout=log, stderr=subprocess.STDOUT,
                                  check=False, timeout=60)
    write("compile.json", dict(argv=compile_argv, returncode=compiled.returncode,
                              numerical_conclusion=None))
    require(compiled.returncode == 0, "public contract compile failed; no RTL conclusion")

    scores, failures = [], []
    for index, (a, r) in enumerate(zip(actual[8], ref[8], strict=True)):
        head, position = divmod(index, 4)
        kv_head = head // 7
        row = dot_accounting(actual[4][head*64:(head+1)*64],
                             actual_k[position][kv_head*64:(kv_head+1)*64],
                             ref[4][head*64:(head+1)*64],
                             reference_k[position][kv_head*64:(kv_head+1)*64])
        require(row["totals"]["actual_bits"] == f"{a:04x}"
                and row["totals"]["reference_bits"] == f"{r:04x}",
                "same-operand score reconstruction")
        row.update(index=index, head=head, key_position=position,
                   accepted=boundary.accepts(a, r))
        if not row["accepted"]:
            failures.append(index)
        scores.append(row)
    require(failures == [13, 15] == qresult["original_stage08_failure_indices"]
            == prior_result["earliest_failure"]["failure_indices"], "original failure regression")
    write("scores.json", scores)
    channels = norm["channels"]
    same_hidden = [x["same_hidden_independent"]["bits"] for x in channels]
    same_hidden = [int(x, 16) for x in same_hidden]
    require([int(x["actual"]["bits"], 16) for x in channels] == actual[0]
            and [int(x["reference"]["bits"], 16) for x in channels] == ref[0]
            and [int(x["hidden_actual"]["bits"], 16) for x in channels] == incoming
            and [int(x["hidden_reference"]["bits"], 16) for x in channels] == ref7[18],
            "retained RMSNorm operand identity")
    qrows, contribution_rows = [], []
    for c in range(192, 256):
        a, at, terms = project(actual[0], tensors, c)
        r, rt, _ = project(ref[0], tensors, c)
        require(a == actual[1][c] and r == ref[1][c], "Q same-input projection mismatch")
        norm_delta = hidden_delta = 0
        for i, term in enumerate(terms):
            w = term["coefficient_q24"]
            n = (boundary.units(actual[0][i])-boundary.units(same_hidden[i])) * w
            h = (boundary.units(same_hidden[i])-boundary.units(ref[0][i])) * w
            norm_delta += n
            hidden_delta += h
            contribution_rows.append(dict(
                q_channel=c, **term, reference_activation=f"{ref[0][i]:04x}",
                same_hidden_independent=f"{same_hidden[i]:04x}",
                norm_delta_q48=n, hidden_delta_q48=h))
        require(norm_delta + hidden_delta == at-rt, "full Q projection telescoping identity")
        qrows.append(dict(channel=c, actual=f"{a:04x}", reference=f"{r:04x}",
                          actual_total_q48=at, reference_total_q48=rt,
                          norm_delta_q48=norm_delta, hidden_delta_q48=hidden_delta))
    write("q_projection.json", qrows)
    write("q_contributions.json", contribution_rows)
    residuals = []
    for i in range(896):
        values = [actual7[s][i] for s in (12, 17, 18)] + [ref7[s][i] for s in (12, 17, 18)]
        ap, ad, ah, rp, rd, rh = map(boundary.units, values)
        require(half((ap+ad)/2**24) == actual7[18][i]
                and half((rp+rd)/2**24) == ref7[18][i], "L7 residual RNE reconstruction")
        rounding = (ah-ap-ad)-(rh-rp-rd)
        require(ah-rh == ap-rp+ad-rd+rounding, "residual contribution identity")
        residuals.append(dict(
            index=i, actual=[f"{v:04x}" for v in values[:3]],
            reference=[f"{v:04x}" for v in values[3:]],
            residual_input_delta_q24=ap-rp, down_projection_delta_q24=ad-rd,
            rounding_delta_q24=rounding, final_delta_q24=ah-rh))
    write("layer07_residual.json", residuals)
    shared = {}
    for arm, earliest in earliest_arms.items():
        row = earliest["reviewed_decomposition"][261]
        require(row["activation"] == "2118" and row["weight"] == "2eb1"
                and row["outputs"]["frozen_Q48"] == "2cc4"
                and row["outputs"]["retained_reference"] == "2cc3", "shared boundary changed")
        shared[arm] = dict(witness=row, changed_indices=earliest["changed_indices"],
                           spine_roots=spines[arm]["roots"],
                           spine_nodes=list(spines[arm]["nodes"]))
    write("shared_boundary.json", dict(
        arms=shared, rejections=rejections,
        structural_link="L0/P0 RMSNorm can affect historical V/K, attention/residuals and later Q. "
                        "Reviewed contribution spines connect 261 to L2/P0/S16/813.",
        numerical_link_to_originalA_witnesses="NOT_ESTABLISHED: all three candidate-native software "
        "trajectories stop at L2/P0/S16/813, before L8/P3. A dependency path is not a measured repair "
        "effect. Do not continue a rejected arm or splice its values into originalA."))
    witness_summary = []
    for index in failures:
        row = scores[index]
        ranked = sorted(row["terms"], key=lambda t: (-abs(t["q_delta_q48"]), t["dimension"]))
        witness_summary.append(dict(
            index=index, actual=row["totals"]["actual_bits"],
            reference=row["totals"]["reference_bits"], exact=row["score_units"],
            largest_signed_Q_terms=[dict(channel=192+t["dimension"], **t) for t in ranked[:8]]))
    result = dict(
        status="DIAGNOSED_NO_JUSTIFIED_GENERAL_REPAIR",
        token_history=boundary.HISTORY, accepted_tail_consumed_not_reset=True,
        successor_integration=record(SUCCESSOR / "integration.json"),
        earliest_material_failure=prior_result["earliest_failure"],
        earliest_local_observed_divergence=dict(layer=8, position=3, stage=0,
            different_norm_coordinates=sum(a != r for a, r in zip(actual[0], ref[0], strict=True)),
            boundary="Inherited from L7 residual, not the earliest divergence in the full historical cone"),
        full_head_Q_channels=64, exact_projection_terms=len(contribution_rows),
        norm_operator_nonzero_contribution_channels=sum(r["norm_delta_q48"] != 0 for r in qrows),
        exact_score_reconstructions=len(scores), witnesses=witness_summary,
        candidate_rejections=rejections, candidate=None,
        cause_disposition=dict(
            score="Exact on actual and independent operands for all 56 scores; no score-core repair supported",
            Q="Full failing-head AWQ projection reconstructed; incoming hidden versus norm-operator "
              "contributions separately measured, not reference replacements",
            K="Historical originalA serialized-state/binary and independent K prefix bound; "
              "current post-RoPE K cache copy exact; signed K/interaction terms measured",
            V="Not an input of this layer's stage08; current V cache copy exact. Earlier V/softmax "
              "remain possible contributors through inherited hidden and historical K",
            RoPE="Prior independently reviewed four-channel own-input reconstruction retained; "
                 "no full-head RoPE defect exclusion is claimed by this diagnostic",
            residual="All 896 L7 residual pairs reconstruct; exact stage12, stage17 and rounding deltas retained"),
        scientific_blocker="No measured valid candidate-native link from the shared 261 intervention "
        "to originalA L8/P3. Every reviewed RMSNorm arm fails earlier at L2/P0/S16/813. Correct local "
        "projection/residual/score rounding plus inherited drift does not specify a general repair; "
        "a coordinate patch, unchanged arm, Q injection or A/B splice is not justified.",
        retained_reference_provenance_boundary=shared_result["proof_boundary"],
        failure_taxonomy="independently_propagated_FP16_trajectory_drift",
        root_cause_hypothesis="Distributed inherited Q operand differences cross score rounding cells; "
                              "their unique upstream cause remains unlocalized.",
        regression="Preserve failures13/15, rejected L2/P0/S16/813, exact own-input identities, "
                   "originalA history and both unchanged numerical policies",
        RTL_invocations=0, tail_replays=0, token358_executions=0,
        third_token=False, binary64_admission=False, numerical_policy_changes=0,
        review="Pending normal independent Host Reviewer, not self-approved",
        elapsed_seconds=time.monotonic()-started)
    write("result.json", result)
    write("artifacts.json", [record(p) for p in sorted(out.iterdir())
                             if p.is_file() and p.suffix not in (".log", ".vvp")])
    print(json.dumps({k: result[k] for k in (
        "status", "exact_score_reconstructions", "full_head_Q_channels",
        "exact_projection_terms", "norm_operator_nonzero_contribution_channels",
        "RTL_invocations", "elapsed_seconds")}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.out.resolve())
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        path = args.out / "failure.json"
        if not path.exists():
            boundary.write(path, dict(
                failure_taxonomy="retained_diagnostic_binding_or_evaluator_failure",
                root_cause_hypothesis=str(error),
                regression="Resolve the exact reported binding or evaluator issue in a fresh attempt",
                RTL_correctness_conclusion=None))
        raise


if __name__ == "__main__":
    main()
