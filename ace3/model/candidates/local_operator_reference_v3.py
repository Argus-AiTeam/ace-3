"""Independent mathematical P0 operators; never import candidate arithmetic."""

import hashlib
import math

import numpy as np


SIZES = {s: (128 if s in (2, 3, 5, 6, 7) else
             14 if s in (8, 9) else 4864 if s in (14, 15, 16) else 896)
         for s in range(19)}
OPERANDS = {
    0: ("input_hidden",), 1: ("stage00",), 2: ("stage00",), 3: ("stage00",),
    4: ("stage01",), 5: ("stage02",), 6: ("stage05",), 7: ("stage03",),
    8: ("stage04", "stage06"), 9: ("stage08",), 10: ("stage09", "stage07"),
    11: ("stage10",), 12: ("input_hidden", "stage11"), 13: ("stage12",),
    14: ("stage13",), 15: ("stage13",), 16: ("stage14", "stage15"),
    17: ("stage16",)}
PROJECTIONS = {
    1: "self_attn.q_proj", 2: "self_attn.k_proj", 3: "self_attn.v_proj",
    11: "self_attn.o_proj", 14: "mlp.gate_proj", 15: "mlp.up_proj",
    17: "mlp.down_proj"}


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def finite_words(array, shape):
    require(isinstance(array, np.ndarray) and array.dtype == np.dtype("<u2")
            and array.shape == shape, f"FP16 shape/dtype mismatch: expected {shape}")
    require(not np.any((array & 0x7c00) == 0x7c00), "nonfinite FP16 operand")
    return array.view("<f2").astype(np.float64)


def rounded(values):
    values = np.asarray(values, dtype=np.float64)
    require(np.all(np.isfinite(values)), "nonfinite independent operator reference")
    with np.errstate(over="raise", invalid="raise"):
        words = values.astype("<f2").view("<u2")
    finite_words(words, values.shape)
    return words


def tensor_shapes(layer):
    prefix = f"model.layers.{layer}."
    shapes = {prefix + name + ".weight": ((896,), "float16") for name in
              ("input_layernorm", "post_attention_layernorm")}
    for stage, name in PROJECTIONS.items():
        inputs, outputs = (4864 if stage == 17 else 896), SIZES[stage]
        for suffix, shape, dtype in (
                ("qweight", (inputs, outputs // 8), "int32"),
                ("qzeros", (inputs // 128, outputs // 8), "int32"),
                ("scales", (inputs // 128, outputs), "float16")):
            shapes[prefix + name + "." + suffix] = (shape, dtype)
        if stage in (1, 2, 3):
            shapes[prefix + name + ".bias"] = ((outputs,), "float16")
    return shapes


def authenticate_tensors(tensors, canonical_records, layer):
    """Canonical records come from the pinned independent model freeze."""
    shapes = tensor_shapes(layer)
    require(set(tensors) == set(shapes), "wrong canonical tensor selection")
    for name, (shape, dtype) in shapes.items():
        tensor, record = tensors[name], canonical_records[name]
        require(tensor.shape == shape and str(tensor.dtype) == dtype
                and record["shape"] == list(shape) and record["dtype"] == dtype,
                f"canonical tensor shape/dtype mismatch: {name}")
        require(hashlib.sha256(tensor.tobytes()).hexdigest() == record["sha256"],
                f"canonical tensor data mismatch: {name}")
        if dtype == "float16":
            require(np.all(np.isfinite(tensor)), f"nonfinite canonical tensor: {name}")


def validate_lineage(arrays, expected_hidden, *, position, history):
    require(type(position) is int and position == 0 and history == [9707],
            "unsupported token/position/history; no P0 fallback")
    finite_words(expected_hidden, (896,))
    for stage, count in SIZES.items():
        finite_words(arrays[f"stage{stage:02d}"], (count,))
    finite_words(arrays["input_hidden"], (896,))
    require(np.array_equal(arrays["input_hidden"], expected_hidden),
            "producer-to-consumer hidden identity mismatch")
    for kind, stage in (("k", 6), ("v", 7)):
        finite_words(arrays[f"input_cache_{kind}"], (0, 128))
        finite_words(arrays[f"output_cache_{kind}"], (1, 128))
        require(np.array_equal(arrays[f"output_cache_{kind}"][0], arrays[f"stage{stage:02d}"]),
                "own-layer KV write/read identity mismatch")
    for consumer, producer in ((6, 5), (7, 3)):
        require(np.array_equal(arrays[f"stage{consumer:02d}"], arrays[f"stage{producer:02d}"]),
                "cache producer identity mismatch")
    # Three finite binary16 addends have an exact binary64 sum.
    total = sum(finite_words(arrays[name], (896,))
                for name in ("input_hidden", "stage11", "stage17"))
    require(np.array_equal(rounded(total), arrays["stage18"]),
            "single-round residual ownership/arithmetic lineage mismatch")
    return {"status": "PASS", "position": 0, "history": history,
            "hidden_identity": "exact", "prior_kv": "own empty P0",
            "cache_producer_identity": "exact", "single_round_residual": "exact"}


def local_reference(stage, operands, tensors, layer):
    """Only declared input data is visible, never the operator's candidate output."""
    require(type(stage) is int and stage in OPERANDS, "local reference scope is S0-S17")
    require(set(operands) == set(OPERANDS[stage]), "wrong operator operand identity")
    x = {name: finite_words(value, (896,) if name == "input_hidden"
                           else (SIZES[int(name[5:])],))
         for name, value in operands.items()}
    first = x[OPERANDS[stage][0]]
    prefix = f"model.layers.{layer}."
    if stage in (0, 13):
        name = "input_layernorm" if stage == 0 else "post_attention_layernorm"
        gamma = tensors[prefix + name + ".weight"].astype(np.float64)
        return rounded(first * (1.0 / math.sqrt(float(np.mean(first * first)) + 1e-6)) * gamma)
    if stage in PROJECTIONS:
        name = prefix + PROJECTIONS[stage]
        weights, zeros = tensors[name + ".qweight"], tensors[name + ".qzeros"]
        scales = tensors[name + ".scales"].astype(np.float64)
        # Unpack the official GEMM lanes directly, independently of AWQ helpers.
        shifts = np.asarray([0, 16, 4, 20, 8, 24, 12, 28], dtype=np.uint32)
        q = ((weights.astype(np.uint32)[..., None] >> shifts) & 15).reshape(len(first), -1)
        z = ((zeros.astype(np.uint32)[..., None] >> shifts) & 15).reshape(len(first) // 128, -1)
        weight = (q.astype(np.float64) - np.repeat(z, 128, axis=0)) * np.repeat(scales, 128, axis=0)
        result = np.einsum("i,ij->j", first, weight, optimize=False)
        if stage in (1, 2, 3):
            result += tensors[name + ".bias"].astype(np.float64)
        return rounded(result)
    if stage in (4, 5, 6, 7):
        return operands[OPERANDS[stage][0]].copy()
    if stage == 8:
        key = np.repeat(x["stage06"].reshape(2, 64), 7, axis=0)
        return rounded(np.sum(first.reshape(14, 64) * key, axis=1) / 8)
    if stage == 9:
        return rounded(np.ones(14))
    if stage == 10:
        value = np.repeat(x["stage07"].reshape(2, 64), 7, axis=0)
        return rounded((first[:, None] * value).reshape(-1))
    if stage == 12:
        return rounded(first + x["stage11"])
    if stage == 16:
        sigmoid = np.empty_like(first)
        positive = first >= 0
        sigmoid[positive] = 1 / (1 + np.exp(-first[positive]))
        exponential = np.exp(first[~positive])
        sigmoid[~positive] = exponential / (1 + exponential)
        return rounded(first * sigmoid * x["stage15"])
    raise ValueError("unimplemented local operator")
