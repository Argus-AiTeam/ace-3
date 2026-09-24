"""Screen retained new-association software outputs; never admit an RTL trajectory."""

import argparse
import csv
from fractions import Fraction
import json
from pathlib import Path
import shutil
import struct
import sys
import time

import numpy as np

from ace3.model.candidates import binary64_fp16_excess_v1 as profile
from ace3.model.candidates.fp16_reference_binary64_control import load, record, require, write

ROOT = Path(__file__).resolve().parents[3]
SOFTWARE = ROOT / "build/general_b_hidden_drift_l0_s18_state_choice_8b12d6a4aea7_attempt001"
CONTROL = ROOT / "build/fp16_reference_binary64_control_attempt002"
HANDOFFS = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    require(out.parent == ROOT / "build" and out.name.startswith("o_down_then_hidden_"),
            "output outside isolated candidate namespace")
    out.mkdir(exist_ok=False)
    shutil.copyfile(__file__, out / "source.py")
    bindings = {}
    started = time.monotonic()

    def bind(item):
        path = item["path"]
        if path not in bindings:
            bindings[path] = record(path)
        require(all(bindings[path][key] == item[key] for key in ("bytes", "sha256")),
                f"input binding mismatch: {path}")
        return Path(path)

    def read(item):
        return load(bind(item))

    reviews = []
    for mission, round_name in (("8b12d6a4aea7", "round-0001.json"),
                                ("0106645d010b", "round-0002.json"),
                                ("77a1655e196f", "round-0001.json")):
        item = record(HANDOFFS / mission / round_name)
        review = read(item)
        require(review["kind"] == "round_reviewed_handoff"
                and review["producer_role"] == "reviewer"
                and review["mission_id"] == mission
                and review["review"]["status"] == "done", f"missing reviewed parent: {mission}")
        reviews.append(item)
    manifest = read(record(SOFTWARE / "output_manifest.json"))
    artifacts = {item["path"]: item for item in manifest["artifacts"]}
    sf = read(artifacts[str(SOFTWARE / "freeze.json")])
    inherited = {item["path"]: item for item in sf["bindings"]}
    bind(sf["source"])
    positive = read(artifacts[str(SOFTWARE / "o_down_then_hidden/result.json")])
    require(positive["supported_choice"]
            and positive["prior_474_gate_prefix_preserved"]
            and positive["first_failure"] is None, "software parent scope changed")

    cr = read(record(CONTROL / "result.json"))
    cf = read(cr["freeze"])
    control_bindings = {item["path"]: item for item in cf["bindings"]}
    for relative in (*profile.SOURCE_PATHS,
                     "ace3/model/candidates/fp16_reference_binary64_control.py"):
        bind(control_bindings[str(ROOT / relative)])
    for key in ("binary64_reference_freeze", "binary64_array_binding_freeze"):
        read(cf[key])
    require(cf["profile_id"] == profile.PROFILE_ID and cf["excess_budget"] == "1/8",
            "approved profile differs")
    require(cf["checkpoint"] == inherited[cf["checkpoint"]["path"]],
            "candidate and reference checkpoints differ")
    bind(cf["checkpoint"])
    require([item["token"] for item in cf["embeddings"][:3]] == [9707, 1879, 0],
            "fixed-history reference tokens differ")
    csv_path = bind(cf["binary64_csv"])
    with csv_path.open(newline="", encoding="ascii") as stream:
        csv_reference = {}
        for row in csv.DictReader(stream):
            key = tuple(int(row[k]) for k in ("layer", "position", "index"))
            require(key not in csv_reference, f"duplicate reference coordinate: {key}")
            csv_reference[key] = row["reference_binary64_hex"]

    cases = []
    for case in cf["cases"]:
        layer, position = case["layer"], case["position"]
        if layer >= 8:
            continue
        require(0 <= layer < 8 and 0 <= position < 3, "unexpected reference selector")
        suffix = "state" if layer == 0 else "stages"
        path = SOFTWARE / "o_down_then_hidden" / f"layer{layer:02d}_position{position}_{suffix}.npz"
        item = artifacts[str(path)]
        with np.load(bind(item), allow_pickle=False) as arrays:
            actual = arrays["stage18"].copy()
            incoming = arrays["input_hidden"].copy() if layer == 0 else None
        require(actual.shape == (896,) and actual.dtype == np.dtype("<u2"),
                f"invalid candidate FP16 array: {path}")
        if layer == 0:
            embedding = bind(cf["embeddings"][position]["input"])
            words = [int(line.strip()[-4:], 16) for line in embedding.read_text("ascii").splitlines()]
            require(np.array_equal(incoming, np.asarray(words, dtype="<u2")),
                    f"candidate/reference embedding mismatch P{position}")
        ref_item = case["binary64_reference_array"]
        reference = np.load(bind(ref_item), allow_pickle=False)
        require(reference.shape == (896,) and reference.dtype == np.dtype("<f8"),
                "invalid independent binary64 array")
        for index, value in enumerate(reference):
            require(float(value).hex() == csv_reference[layer, position, index],
                    f"reference array/CSV mismatch: {layer}/{position}/{index}")
        cases.append((layer, position, actual, reference, item, ref_item))
    require([(layer, position) for layer, position, *_ in cases]
            == [(layer, position) for layer in range(8) for position in range(3)],
            "retained reference coverage differs")
    write(out / "freeze.json", {
        "association": "RNE16(RNE16(attention_o + mlp_down) + incoming_hidden)",
        "candidate_kind": "reviewed retained SOFTWARE outputs, not RTL-produced states",
        "unchanged_stage12_and_mlp": True,
        "source": record(__file__), "source_snapshot": record(out / "source.py"),
        "runtime": {"python": record(sys.executable), "version": sys.version,
                    "numpy": np.__version__, "command": sys.argv},
        "reviews": reviews, "bindings": list(bindings.values()),
        "profile_id": profile.PROFILE_ID, "excess_budget": "1/8",
        "candidate_outputs": [item for *_, item, ref_item in cases],
        "reference_outputs": [ref_item for *_, item, ref_item in cases],
        "order": "layer, position, index; all retained available outputs, no model replay",
        "scope": "L0-7/P0-2 layer-final outputs only; fixed [9707,1879,0] causal prefix",
        "missing_reference_scope": "P3 absent from this authenticated reference set",
        "no_admission": True, "running_rtl_attempt002_untouched": True,
    })

    failures, per_output = [], []
    measured = 0
    with (out / "all_coordinates.jsonl").open("x", encoding="ascii") as stream:
        for layer, position, actual, reference, _, _ in cases:
            rejected = 0
            for index, (bits, ref) in enumerate(zip(actual, reference)):
                bits, ref = int(bits), float(ref)
                result = profile.evaluate_layer_final_output(
                    actual_fp16_bits=bits, reference_binary64_hex=ref.hex())
                a = Fraction.from_float(struct.unpack(">e", bits.to_bytes(2, "big"))[0])
                nearest_bytes = struct.pack(">e", ref)
                nearest = Fraction.from_float(struct.unpack(">e", nearest_bytes)[0])
                r = Fraction.from_float(ref)
                q, error = abs(nearest - r), abs(a - r)
                require(result["nearest_fp16_bits"] == nearest_bytes.hex()
                        and Fraction(result["q"]) == q
                        and Fraction(result["actual_error"]) == error
                        and Fraction(result["excess_error"]) == error - q
                        and result["accepted"] == (error - q <= Fraction(1, 8)),
                        f"independent scalar oracle disagreement: {layer}/{position}/{index}")
                result.update(layer=layer, position=position, index=index,
                              legacy_absolute_accepted=error <= Fraction(1, 8))
                stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
                measured += 1
                if not result["accepted"]:
                    rejected += 1
                    failures.append(result)
            per_output.append({"layer": layer, "position": position, "coordinates": 896,
                               "v1_failures": rejected})
    write(out / "failures.json", failures)
    result = {
        "status": "FAIL_NEW_CANDIDATE_BINARY64_V1" if failures else "PASS_AVAILABLE_OUTPUT_SCREEN_ONLY",
        "profile_id": profile.PROFILE_ID, "candidate_kind": "SOFTWARE",
        "outputs": len(cases), "coordinates": measured, "failed_coordinates": len(failures),
        "per_output": per_output, "first_failure": failures[0] if failures else None,
        "failure_taxonomy": "layer_final_binary64_v1_excess_error" if failures else None,
        "root_cause_hypothesis": (
            "The uniform association's propagated FP16 outputs can still exceed the independent "
            "binary64 reference's quantization-aware budget; no unique upstream producer established."
        ) if failures else None,
        "regression": "Same frozen candidate/reference arrays, all 21504 coordinates, exact v1 "
                      "plus independent struct/Fraction check; no tolerance or reference changes.",
        "positive_8b12_scope_preserved": True,
        "L8_suffix": "NOT_EXECUTED; eligibility screen only",
        "P3_binary64": "UNBOUND in the retained reviewed reference set; not PASS",
        "rtl_attempt002": "Already running; not cancelled, replayed, modified, or admitted here",
        "rtl_or_model_replays": 0, "policy_adoption": False, "numerical_admission": False,
        "normal_host_review": "PENDING", "elapsed_seconds": time.monotonic() - started,
        "freeze": record(out / "freeze.json"),
    }
    write(out / "result.json", result)
    write(out / "output_manifest.json", {
        "artifacts": [record(path) for path in sorted(out.iterdir()) if path.is_file()]})
    print(json.dumps({key: result[key] for key in
                      ("status", "outputs", "coordinates", "failed_coordinates", "first_failure",
                       "elapsed_seconds")}, sort_keys=True))


if __name__ == "__main__":
    main()
