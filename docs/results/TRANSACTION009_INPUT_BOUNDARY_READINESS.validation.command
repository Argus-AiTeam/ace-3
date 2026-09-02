#!/usr/bin/env bash
set -euo pipefail
cd /home/argustest/ace3-argus
python3 - <<'PY'
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path("/home/argustest/ace3-argus")
RUNTIME = ROOT / (
    "build/model24_selected_token_position3_runs/"
    "ace3-position3-fresh-r11-20260831t215500z"
)
STATE = RUNTIME / (
    "transaction2-receipt-adoption-r4/state-generations/"
    "generation-0000000009"
)
NOTE = ROOT / "docs/results/TRANSACTION009_INPUT_BOUNDARY_READINESS.md"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


expected = {
    RUNTIME / "transaction2-receipt-adoption-r4/authoritative-state.json":
        (813, "e73aa8883540356c2b035e32a7c09cbed2626dc076e578932018d04d9e1ce8b2"),
    STATE / "generation-manifest.json":
        (8346, "440a55cd06462d35f46be95b2f78e79b6e304130b29d3967422bc86b3ce89041"),
    STATE / "ledger.json":
        (5396, "208fa40f308708d01ba67cc496f9899ef9148ead20e6da49c46e4f2dd1ff40bf"),
    RUNTIME / "invocation.json":
        (73186, "9fb5418fe06c492633399ea79791a08ae25507f3e74b0cf0c003bdb79b43919d"),
    ROOT / "build/cursor9_next_position_runtime_readiness/readiness.json":
        (7925, "7ed4bdaba4106368505d4e251a5e87edd61be735d4eeb98d775301172b27d659"),
    STATE / "checkpoints/transaction-000.json":
        (11814, "9fabdd9a91f1331220ca4e2c5682cc17e8be18fed444799f97cd3c66276cbdd0"),
    STATE / "checkpoints/transaction-001.json":
        (24153, "6019478026ea51b2cefd34e4a2e261edca8cb1f02034c1c5ed8fb20e0c3250a7"),
    RUNTIME / "transactions/transaction-000/selected-token-embedding.hex":
        (4480, "fcfad3b827ab02406ee259fff2572f7c7ff07c1e51df3477ffa09f3d1d20a58a"),
    ROOT / "build/model24_rtl_cascade/checkpoint/model.safetensors":
        (730652248, "c50d807b7bed7ff314308972e0f4bcf4e5a70bc60ad88fc7df53940831ed0c1b"),
    RUNTIME / "shared/lm-head/obj_dir/Vace3_streaming_tied_lm_head_topk":
        (145184, "8168495a9164cfd3376e8ddacc3d62ffc49f3824ccc916d07aa06d891e6a7e97"),
    ROOT / "build/host_dialogue_audit_20260829T182819Z/tokenizer/tokenizer.json":
        (7031645, "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"),
    ROOT / "build/host_dialogue_audit_20260829T182819Z/tokenizer/tokenizer_config.json":
        (7305, "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583"),
    ROOT / "ace3/contracts/model24_execution.json":
        (18465, "084c28f413a219b095b06e41cdc9bf3c1138a583cc5d0a2d00b94133cf9ba9e6"),
    STATE / "checkpoints/transaction-008.json":
        (2595, "f4d4bb424bc1e52f2f56b1434e626d7ec7bbe29ad8ea8340c0c48deaeb63a44b"),
    RUNTIME / "transactions/transaction-008/position004.state":
        (249565, "65f4d2b9cb29a757948ea9c22f69555d14285123036531e739dfcfc840511a55"),
    ROOT / (
        "build/model24_selected_token_position2_runs/"
        "ace3-position2-fresh-v10-20260831t110456z/traversal/"
        "layer08/vectors/manifest.json"
    ): (17600, "2d83dafa9b4f1e3a843633b9265abbd6dc1584012f980333f1698784ae1c35fc"),
    ROOT / (
        "build/model24_selected_token_position2_runs/"
        "ace3-position2-fresh-v10-20260831t110456z/traversal/"
        "layer08/position003.state"
    ): (249565, "66fd0083cc752685da450ae00079b8431b9ee1e8b5fe600c01322726a37a0265"),
    ROOT / (
        "build/argus-audit/tx008-r5/authority-transition-gate-r1/"
        "authority-transition.json"
    ): (7849, "263f693b59681932568cb0f71bea2d465fda42cbca3137e33f59e53c2d99848f"),
}

branch = subprocess.run(
    ["git", "branch", "--show-current"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
assert branch == "argus/full-projection", branch
assert subprocess.run(
    ["git", "rev-parse", "--show-toplevel"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip() == str(ROOT)

for path, (size, digest) in expected.items():
    assert path.is_file(), path
    assert path.stat().st_size == size, (path, path.stat().st_size, size)
    assert sha256(path) == digest, path

pointer = load(RUNTIME / "transaction2-receipt-adoption-r4/authoritative-state.json")
manifest = load(STATE / "generation-manifest.json")
ledger = load(STATE / "ledger.json")
tx0 = load(STATE / "checkpoints/transaction-000.json")
tx1 = load(STATE / "checkpoints/transaction-001.json")
tx8 = load(STATE / "checkpoints/transaction-008.json")
readiness = load(ROOT / "build/cursor9_next_position_runtime_readiness/readiness.json")
fixture = load(ROOT / (
    "build/model24_selected_token_position2_runs/"
    "ace3-position2-fresh-v10-20260831t110456z/traversal/"
    "layer08/vectors/manifest.json"
))
gate = load(ROOT / (
    "build/argus-audit/tx008-r5/authority-transition-gate-r1/"
    "authority-transition.json"
))

assert pointer["status"] == "COMMITTED"
assert pointer["generation"] == 9
assert pointer["generation_manifest"]["sha256"] == expected[STATE / "generation-manifest.json"][1]
assert manifest["generation"] == 9
assert manifest["parent_generation"] == 8
assert manifest["status"] == "PREPARED"
assert manifest["transactions009_025_executed"] is False
assert manifest["activity_counters"]["transactions009_025_execution"] == 0
assert ledger["state_generation"] == 9
assert ledger["status"] == "IN_PROGRESS"
assert ledger["completed_transaction_count"] == 9
assert ledger["next_transaction_index"] == 9
assert ledger["transaction_count"] == 26
assert ledger["authority_consumption_status"] == "CONSUMED_ONCE"
assert [Path(item["path"]).name for item in ledger["completed_receipts"]] == [
    f"transaction-{index:03d}.json" for index in range(9)
]

assert tx0["status"] == "COMPLETE"
assert tx0["transaction_index"] == 0
assert tx0["result"]["selected_token_id"] == 2
assert tx0["result"]["selected_logit_f16_bits"] == 19479
embedding = tx0["output_semantics"]["selected_token_embedding"]
assert embedding == {
    "checkpoint": {
        "bytes": 730652248,
        "path": str(ROOT / "build/model24_rtl_cascade/checkpoint/model.safetensors"),
        "sha256": "c50d807b7bed7ff314308972e0f4bcf4e5a70bc60ad88fc7df53940831ed0c1b",
    },
    "dtype": "FP16",
    "elements": 896,
    "selected_logit_f16_bits": 19479,
    "semantic_sha256": "8367b1f56e896acd2d99b64c3f0bd73f3090b8310ec7b294614074836a8af06a",
    "tensor": "model.embed_tokens.weight",
    "token_id": 2,
}
tx1_serialized = json.dumps(tx1, sort_keys=True)
assert tx1["status"] == "COMPLETE"
assert tx1["transaction_index"] == 1
assert embedding["semantic_sha256"] in tx1_serialized
assert expected[RUNTIME / "transactions/transaction-000/selected-token-embedding.hex"][1] in tx1_serialized

assert tx8["status"] == "COMPLETE"
assert tx8["transaction_index"] == 8
assert tx8["layer_index"] == 7
assert tx8["result"]["natural_rtl_terminal"] is True
assert tx8["result"]["exact_integer_oracle_match"] is True
assert tx8["result"]["output_hidden_elements"] == 896
assert tx8["result"]["output_state"]["sha256"] == expected[
    RUNTIME / "transactions/transaction-008/position004.state"
][1]

assert fixture["kind"] == "ace3_position2_live_transaction_vectors"
assert fixture["layer_index"] == 8
assert fixture["position"] == 2
assert len(fixture["tensors"]) == 26
for item in [fixture["input"], fixture["rope_coefficients"]] + [
    tensor["serialized"] for tensor in fixture["tensors"]
]:
    path = Path(item["path"])
    assert path.is_file(), path
    assert path.stat().st_size == item["bytes"], path
    assert sha256(path) == item["sha256"], path

assert readiness["data_bindings_ready"] is True
assert readiness["status"] == "NOT_READY"
assert readiness["launchable"] is False
assert readiness["cursor"]["generation"] == 9
assert readiness["cursor"]["checkpoint"] == 8
assert readiness["cursor"]["next_transaction_index"] == 9
assert readiness["selected_token"]["token_id"] == 2
assert readiness["embedding_feedback"]["semantics"] == embedding
assert readiness["next_layer_inputs"] == {
    "checkpoint008_hidden_kv_state": {
        "bytes": 249565,
        "path": str(RUNTIME / "transactions/transaction-008/position004.state"),
        "sha256": "65f4d2b9cb29a757948ea9c22f69555d14285123036531e739dfcfc840511a55",
    },
    "layer8_fixture_manifest": {
        "bytes": 17600,
        "path": str(ROOT / (
            "build/model24_selected_token_position2_runs/"
            "ace3-position2-fresh-v10-20260831t110456z/traversal/"
            "layer08/vectors/manifest.json"
        )),
        "sha256": "2d83dafa9b4f1e3a843633b9265abbd6dc1584012f980333f1698784ae1c35fc",
    },
    "layer8_position2_kv_parent": {
        "bytes": 249565,
        "path": str(ROOT / (
            "build/model24_selected_token_position2_runs/"
            "ace3-position2-fresh-v10-20260831t110456z/traversal/"
            "layer08/position003.state"
        )),
        "sha256": "66fd0083cc752685da450ae00079b8431b9ee1e8b5fe600c01322726a37a0265",
    },
    "layer_index": 8,
    "required_result": {
        "exact_integer_oracle_match": True,
        "natural_rtl_terminal": True,
        "output_hidden_elements": 896,
        "output_state_position": 4,
    },
    "template_input_binding_sha256":
        "dec51a69a7afd42f780a77b975b1e716046f22c3d84e975f2a60e30105a9267b",
    "transaction_index": 9,
    "transaction_position": 3,
}

assert gate["status"] == "PASS"
assert gate["manager_prohibition"]["status"] == "ACTIVE_UNCLEARED"
assert gate["manager_prohibition"]["clearance_record_present"] is False
assert gate["transaction009_transition"] == {
    "manager_clearance_required": True,
    "prohibition_preserved": True,
    "status": "NOT_READY",
}
assert gate["transaction009_025_artifacts"] == {
    "created_by_this_gate": False,
    "named_authority_execution_or_publication_paths_found": [],
    "status": "ABSENT",
}

assert not (RUNTIME / "transactions/transaction-009").exists()
assert not (STATE / "checkpoints/transaction-009.json").exists()
future_name = re.compile(
    r"transaction-?(?:0?9|0?1[0-9]|0?2[0-5])(?:\D|$)",
    re.IGNORECASE,
)
future_named_paths = sorted(
    str(path.relative_to(ROOT))
    for path in (ROOT / "build").rglob("*")
    if future_name.search(path.name)
)
assert future_named_paths == [], future_named_paths

note = NOTE.read_text(encoding="utf-8")
for required_text in (
    "**Input status: READY_INPUTS. Authority status: NOT_READY (fail closed).**",
    "Exactly one transaction: index `9`; indices `10` through `25` are excluded.",
    "Exactly one decoder layer: layer `8`.",
    "zero transaction009-025 authority, launcher,",
):
    assert required_text in note, required_text

print("PASS transaction009 input-boundary readiness")
print(f"branch={branch}")
print(f"head={subprocess.run(['git', 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True).stdout.strip()}")
print(f"note_sha256={sha256(NOTE)}")
print("input_status=READY_INPUTS")
print("authority_status=NOT_READY")
print("cursor_generation=9")
print("checkpoint_index=8")
print("next_transaction_index=9")
print("would_be_bounds=transaction009/layer8/position3->state4")
print("fixture_serialized_files_verified=28")
print("transaction009_025_named_paths=0")
print("transaction009_025_artifacts_created_by_this_task=0")
PY
