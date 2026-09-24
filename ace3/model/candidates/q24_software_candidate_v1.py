"""CPU-only P0 candidate. No reference arrays or decoder/runtime ABI imports."""

import numpy as np

from ace3.model import awq_bit_oracle as awq
from ace3.model import fp16_adaptation_oracle as fixed


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def units(words):
    require(words.dtype == np.dtype("<u2") and words.ndim == 1,
            "candidate requires FP16 word vectors")
    require(not np.any((words & 0x7c00) == 0x7c00), "nonfinite candidate operand")
    return [fixed.decode_f16_q24(int(word))[0] for word in words]


def view(integer, tag):
    require(type(integer) is int and -(1 << 63) <= integer < (1 << 63),
            "Q24 signed 64-bit overflow")
    require(type(tag) is int and tag in (0, 1) and (integer == 0 or tag == 0),
            "invalid Q24 zero tag")
    word, saturated = fixed.q24_to_f16(integer, bool(tag))
    require(not saturated, "Q24 FP16 view overflow")
    return word


def lift(hidden):
    integers = np.asarray(units(hidden), dtype="<i8")
    return {"i": integers, "z": (hidden == 0x8000).astype("u1"), "h": hidden.copy()}


def add(state, words):
    increments = units(words)
    require(set(state) == {"i", "z", "h"}, "missing paired Q24 state")
    require(state["i"].dtype == np.dtype("<i8") and state["z"].dtype == np.dtype("u1")
            and all(a.shape == words.shape for a in state.values()), "Q24 state shape/dtype")
    result, tags, hidden = [], [], []
    for integer, tag, old, word, increment in zip(
            state["i"], state["z"], state["h"], words, increments, strict=True):
        integer, tag, word = int(integer), int(tag), int(word)
        require(view(integer, tag) == int(old), "Q24 input/view disagreement")
        total = integer + increment
        zero = int(total == 0 and integer == 0 and tag == 1 and word == 0x8000)
        hidden.append(view(total, zero))
        result.append(total)
        tags.append(zero)
    return {"i": np.asarray(result, dtype="<i8"), "z": np.asarray(tags, dtype="u1"),
            "h": np.asarray(hidden, dtype="<u2")}


def checked(outputs):
    require(all(not invalid and not saturated for _, invalid, saturated in outputs),
            "candidate operator invalid/overflow")
    return np.asarray([word for word, _, _ in outputs], dtype="<u2")


def project(tensors, prefix, activation, *, single_bias=False):
    """Native integer AWQ grouping from the retained single-round CPU screen."""
    weights = tensors[prefix + ".qweight"].view("<u4")
    zeros = tensors[prefix + ".qzeros"].view("<u4")
    count, columns = weights.shape
    outputs = columns * 8
    scales = np.asarray(units(tensors[prefix + ".scales"].reshape(-1).view("<u2")),
                        dtype="<i8").reshape(count // 128, outputs)
    activations = np.asarray(units(activation), dtype="<i8").reshape(count // 128, 128)
    totals = [0] * outputs
    for group in range(count // 128):
        dots = np.empty(outputs, dtype="<i8")
        for lane, nibble in enumerate(awq.AWQ_REVERSE_ORDER):
            quantized = ((weights[group * 128:(group + 1) * 128] >>
                          (4 * nibble)) & 15).astype("<i8")
            zero = ((zeros[group] >> (4 * nibble)) & 15).astype("<i8")
            dots[lane::8] = activations[group] @ (quantized - zero)
        for channel in range(outputs):
            totals[channel] += int(dots[channel]) * int(scales[group, channel])
    bias = tensors.get(prefix + ".bias")
    bias_words = None if bias is None else bias.view("<u2")
    bias_units = None if bias is None else units(bias_words)
    result = []
    for channel, total in enumerate(totals):
        if single_bias and bias_units is not None:
            total += bias_units[channel] << 24
        word, saturated = awq.q47_48_to_f16(total)
        require(not saturated, "native AWQ projection overflow")
        if single_bias and word == 0 and total < 0:
            word = 0x8000
        if not single_bias and bias_words is not None:
            word, invalid, saturated = fixed.residual_add(word, int(bias_words[channel]))
            require(not invalid and not saturated, "native AWQ bias invalid/overflow")
        result.append(word)
    return np.asarray(result, dtype="<u2")


def stages(tensors, layer, parent, arrays):
    """Yield each actual stage before dispatching its consumer; P0 only."""
    require(type(layer) is int and 0 <= layer <= 8, "CPU candidate scope is L0-L8")
    require(set(parent) == {"i", "z", "h"}, "FP16-only parent forbidden")
    prefix = f"model.layers.{layer}."
    arrays["input_hidden"] = parent["h"].copy()
    for key in ("i", "z"):
        arrays["input_" + key] = parent[key].copy()
    for kind in ("k", "v"):
        arrays["input_cache_" + kind] = np.empty((0, 128), dtype="<u2")
    scratch = None
    for stage in range(19):
        if stage in (0, 13):
            hidden = parent["h"] if stage == 0 else arrays["stage12"]
            name = "input_layernorm" if stage == 0 else "post_attention_layernorm"
            weight = tensors[prefix + name + ".weight"].view("<u2")
            word = checked(fixed.rmsnorm(hidden.tolist(), weight.tolist())[0])
        elif stage in (1, 2, 3, 11, 14, 15, 17):
            name, source = {
                1: ("self_attn.q_proj", 0), 2: ("self_attn.k_proj", 0),
                3: ("self_attn.v_proj", 0), 11: ("self_attn.o_proj", 10),
                14: ("mlp.gate_proj", 13), 15: ("mlp.up_proj", 13),
                17: ("mlp.down_proj", 16)}[stage]
            word = project(tensors, prefix + name, arrays[f"stage{source:02d}"],
                           single_bias=stage in (1, 2))
        elif stage in (4, 5, 6, 7):
            word = arrays[f"stage{ {4: 1, 5: 2, 6: 5, 7: 3}[stage]:02d}"].copy()
            if stage in (6, 7):
                arrays["output_cache_" + ("k" if stage == 6 else "v")] = word.reshape(1, 128).copy()
        elif stage == 8:
            q, k = units(arrays["stage04"]), units(arrays["stage06"])
            scores = []
            for head in range(14):
                total = sum(q[head * 64 + i] * k[(head // 7) * 64 + i] for i in range(64))
                scores.append(view(fixed.round_shift_even_signed(total, 27), 0))
            word = np.asarray(scores, dtype="<u2")
        elif stage == 9:
            word = np.full(14, 0x3c00, dtype="<u2")
        elif stage == 10:
            word = np.repeat(arrays["output_cache_v"].reshape(2, 64), 7, axis=0).reshape(-1)
        elif stage == 12:
            scratch = add(parent, arrays["stage11"])
            arrays["scratch_i"], arrays["scratch_z"] = scratch["i"], scratch["z"]
            word = scratch["h"]
        elif stage == 16:
            word = checked([fixed.silu_gate_exp(int(g), int(u)) for g, u in
                            zip(arrays["stage14"], arrays["stage15"], strict=True)])
        else:
            require(scratch is not None, "missing same-layer Q24 scratch state")
            successor = add(scratch, arrays["stage17"])
            arrays["output_i"], arrays["output_z"] = successor["i"], successor["z"]
            word = successor["h"]
        arrays[f"stage{stage:02d}"] = word
        yield stage
