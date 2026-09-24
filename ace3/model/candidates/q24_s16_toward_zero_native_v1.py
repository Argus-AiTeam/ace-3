"""CPU-native canonical operators with an isolated S16 FP16 RTZ boundary.

This implementation never imports a local/global reference or retained output.
Q24 residual arithmetic reuses the candidate integer implementation, not its
independent rational verifier. All non-residual compute is CPU binary64.
"""

import math
import struct

import numpy as np
import torch
import torch.nn.functional as functional

from ace3.model.candidates import q24_software_candidate_v1 as state


require = state.require
lift = state.lift


def decoded(words):
    require(isinstance(words, np.ndarray) and words.dtype == np.dtype("<u2")
            and words.ndim == 1, "native operand must be an FP16 word vector")
    require(not np.any((words & 0x7c00) == 0x7c00), "nonfinite native operand")
    return torch.from_numpy(words.view("<f2").astype("<f8"))


def rne(values):
    require(values.device.type == "cpu" and values.dtype == torch.float64
            and bool(torch.isfinite(values).all()), "invalid native binary64 result")
    # Torch's CPU double->half cast can double round through float32.
    packed = b"".join(struct.pack("<e", float(value)) for value in values.numpy())
    words = np.frombuffer(packed, dtype="<u2").copy()
    decoded(words)
    return words


def toward_zero(values):
    """Truncate the binary significand directly, including gradual underflow."""
    require(isinstance(values, np.ndarray) and values.dtype == np.dtype("<f8")
            and values.ndim == 1 and np.all(np.isfinite(values))
            and np.all(np.abs(values) <= 65504), "invalid FP16-range RTZ operand")
    words = []
    for value in values:
        value = float(value)
        sign = 0x8000 if math.copysign(1.0, value) < 0 else 0
        magnitude = abs(value)
        if magnitude < 2 ** -14:
            word = int(math.ldexp(magnitude, 24))
        else:
            significand, exponent = math.frexp(magnitude)
            word = ((exponent + 14) << 10) | (int(significand * 2048) - 1024)
        words.append(sign | word)
    return np.asarray(words, dtype="<u2")


def projection(tensors, name, activation):
    # Construct output lanes independently of the NumPy reference unpacker.
    packed = torch.from_numpy(tensors[name + ".qweight"].astype("<i8"))
    zero = torch.from_numpy(tensors[name + ".qzeros"].astype("<i8"))
    lanes, zero_lanes = [], []
    for nibble in (0, 4, 1, 5, 2, 6, 3, 7):
        lanes.append((packed >> (4 * nibble)) & 15)
        zero_lanes.append((zero >> (4 * nibble)) & 15)
    quantized = torch.stack(lanes, dim=-1).reshape(packed.shape[0], -1)
    zeros = torch.stack(zero_lanes, dim=-1).reshape(zero.shape[0], -1)
    scales = torch.from_numpy(tensors[name + ".scales"].astype("<f8"))
    weight = (quantized - zeros.repeat_interleave(128, dim=0)).to(torch.float64)
    weight *= scales.repeat_interleave(128, dim=0)
    value = torch.mv(weight.T, decoded(activation))
    if name + ".bias" in tensors:
        value += torch.from_numpy(tensors[name + ".bias"].astype("<f8"))
    return rne(value)


def stages(tensors, layer, parent, arrays):
    require(type(layer) is int and layer in range(3), "native scope is L0-L2/P0")
    yield from _stages(tensors, layer, parent, arrays)


def continuation_stages(tensors, layer, parent, arrays):
    require(type(layer) is int and layer in range(3, 9), "continuation scope is L3-L8/P0")
    yield from _stages(tensors, layer, parent, arrays)


def _stages(tensors, layer, parent, arrays):
    require(set(parent) == {"i", "z", "h"}, "paired Q24 parent required")
    prefix = f"model.layers.{layer}."
    arrays.update(input_hidden=parent["h"].copy(), input_i=parent["i"].copy(),
                  input_z=parent["z"].copy(),
                  input_cache_k=np.empty((0, 128), dtype="<u2"),
                  input_cache_v=np.empty((0, 128), dtype="<u2"))
    scratch = None
    for stage in range(19):
        if stage in (0, 13):
            hidden = parent["h"] if stage == 0 else arrays["stage12"]
            norm = "input_layernorm" if stage == 0 else "post_attention_layernorm"
            x = decoded(hidden)
            gamma = torch.from_numpy(tensors[prefix + norm + ".weight"].astype("<f8"))
            word = rne((x * torch.rsqrt(x.square().mean() + 1e-6)) * gamma)
        elif stage in (1, 2, 3, 11, 14, 15, 17):
            name, source = {
                1: ("self_attn.q_proj", 0), 2: ("self_attn.k_proj", 0),
                3: ("self_attn.v_proj", 0), 11: ("self_attn.o_proj", 10),
                14: ("mlp.gate_proj", 13), 15: ("mlp.up_proj", 13),
                17: ("mlp.down_proj", 16)}[stage]
            word = projection(tensors, prefix + name, arrays[f"stage{source:02d}"])
        elif stage in (4, 5, 6, 7):
            source = {4: 1, 5: 2, 6: 5, 7: 3}[stage]
            word = arrays[f"stage{source:02d}"].copy()
            if stage in (6, 7):
                arrays["output_cache_" + ("k" if stage == 6 else "v")] = word.reshape(1, 128).copy()
        elif stage == 8:
            q = decoded(arrays["stage04"]).reshape(14, 64)
            k = decoded(arrays["stage06"]).reshape(2, 64).repeat_interleave(7, dim=0)
            word = rne((q * k).sum(dim=1) / 8)
        elif stage == 9:
            word = rne(torch.softmax(decoded(arrays["stage08"]).reshape(14, 1), dim=1).reshape(-1))
        elif stage == 10:
            v = decoded(arrays["stage07"]).reshape(2, 64).repeat_interleave(7, dim=0)
            word = rne((decoded(arrays["stage09"])[:, None] * v).reshape(-1))
        elif stage == 12:
            scratch = state.add(parent, arrays["stage11"])
            arrays["scratch_i"], arrays["scratch_z"] = scratch["i"], scratch["z"]
            word = scratch["h"]
        elif stage == 16:
            value = functional.silu(decoded(arrays["stage14"])) * decoded(arrays["stage15"])
            arrays["s16_unrounded_binary64"] = value.numpy().copy()
            word = toward_zero(arrays["s16_unrounded_binary64"])
        else:
            require(scratch is not None, "missing own-layer residual scratch")
            successor = state.add(scratch, arrays["stage17"])
            arrays["output_i"], arrays["output_z"] = successor["i"], successor["z"]
            word = successor["h"]
        arrays[f"stage{stage:02d}"] = word
        yield stage
