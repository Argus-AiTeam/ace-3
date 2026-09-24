"""Ordered actual-output handoff; execution and admission are distinct callbacks.

No callback result alone is a certificate. The admission callback must be the
source-bound independent local/global/state evaluator selected by the reviewed
runtime plan. Fixture callbacks demonstrate routing, not model execution.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib

from ace3.model.candidates import residual_exact_grid_q24_runtime_v1 as codec


@dataclass(frozen=True)
class Parent:
    text: str
    sha256: str
    metadata: Mapping[str, object]
    hidden: tuple[int, ...]

    def records(self) -> tuple[tuple[int, int], ...]:
        return codec.decode_decoder(
            self.text, trusted_metadata=self.metadata,
            trusted_sha256=self.sha256, hidden=self.hidden)


@dataclass(frozen=True)
class LayerOutput:
    parent: Parent
    layer: int
    input_sha256: str
    prior_kv_count: int
    completed: bool
    idle: bool
    fault_code: int
    evidence_kind: str


def run_root_through_l8(
    root: Parent,
    execute: Callable[[int, Parent], LayerOutput],
    admit: Callable[[Parent, LayerOutput], Mapping[str, object]],
    persist: Callable[[LayerOutput, Mapping[str, object]], None],
) -> Parent:
    """Stop before dispatching a consumer of any unadmitted or mismatched state."""
    root.records()
    if root.metadata["next_layer"] != 0 or root.metadata["position"] != 0:
        raise ValueError("Q24 route requires an authenticated P0 model-root producer")
    if root.metadata["model_id"] != "Qwen/Qwen2.5-0.5B-Instruct-AWQ":
        raise ValueError("Q24 route model identity mismatch")
    current = root
    for layer in range(9):
        output = execute(layer, current)
        if (type(output.layer) is not int or output.layer != layer or
                output.input_sha256 != current.sha256):
            raise ValueError(f"L{layer}: execution did not consume its immediate paired parent")
        if (output.completed is not True or output.idle is not True or
                type(output.fault_code) is not int or output.fault_code != 0 or
                type(output.prior_kv_count) is not int or output.prior_kv_count != 0):
            raise ValueError(f"L{layer}: incomplete/faulted transaction or nonempty P0 KV")
        if output.evidence_kind != "actual_decoder_rtl":
            raise ValueError(f"L{layer}: non-RTL output cannot seed the actual decoder route")
        output.parent.records()
        metadata = output.parent.metadata
        if metadata["next_layer"] != layer + 1:
            raise ValueError(f"L{layer}: invalid successor layer")
        for key in ("model_id", "root_id", "token_history_id", "slot", "position"):
            if metadata[key] != root.metadata[key]:
                raise ValueError(f"L{layer}: incompatible {key}")
        if (metadata["producer_id"] == current.metadata["producer_id"] or
                metadata["kv_lineage_id"] == current.metadata["kv_lineage_id"]):
            raise ValueError(f"L{layer}: reused producer or cross-layer KV identity")
        report = admit(current, output)
        for key in ("local_operator_fp16", "binary64_v1", "residual_state_lineage",
                    "kv_lineage", "source_runtime"):
            if report.get(key) != "PASS":
                raise ValueError(f"L{layer}: missing or failed mandatory admission: {key}")
        if (report.get("input_sha256") != current.sha256 or
                report.get("output_sha256") != output.parent.sha256):
            raise ValueError(f"L{layer}: admission is not bound to these actual operands")
        persist(output, report)
        current = output.parent
    return current


def paired_parent(records: Sequence[tuple[int, int]], metadata: Mapping[str, object],
                  hidden: Sequence[int]) -> Parent:
    """Serialize produced state; this does not authenticate or admit its producer."""
    text = codec.encode_decoder(records, metadata, hidden=hidden)
    return Parent(text, hashlib.sha256(text.encode("utf-8")).hexdigest(),
                  dict(metadata), tuple(hidden))
