# ACE-3: bounded software continuation preview

**SOFTWARE REFERENCE PREVIEW - NOT AN RTL-GENERATED CONTINUATION**

Date: 2026-09-06. This completed CPU-software demonstration uses one fixed
plain-text seed, `Hello world`, with the pinned AWQ checkpoint and the
mathematical FP16 stage policy identified below. It is separate from the
[RTL component highlights](ACE3_ENGINEERING_HIGHLIGHTS_20260906.md).
The [numeric summary](ACE3_SOFTWARE_CONTINUATION_PREVIEW_20260906.json) includes
all token IDs and explicit text-hash conventions.

## Complete bounded output

The following is the **entire decoded text: the seed plus all 32 generated
tokens**, not a selected excerpt:

```text
Hello world! I'm a beginner in Python and I'm trying to create a program that can convert a string to a list of words. How can I achieve this?
```

The run reached its **32-generated-token cap without EOS**. This is the complete
record of the bounded run, not an EOS-terminated response or a claim that the
model finished answering. The seed was plain text, not a chat-formatted request.

## Resume and generation boundary

The preview resumed **its own software K/V for all 24 layers**, with three
positions already processed and token `358` pending. The starting history,
including that pending token, was `[9707,1879,0,358]`: two seed tokens and two
previously selected generated tokens.

It computed **30 additional generated tokens**, bringing the total to **32**.
No prefill was replayed and no RTL state was used. At the bounded end, 33
positions had been processed and the final selected token, `30`, remained
pending; it was included in the decoded text, not fed back for another step.

Selection was greedy across the full vocabulary: exact Q48 dot products,
one FP16 round-to-nearest-even per logit, and the lowest token ID for equal
rounded finite logits. This demonstration does not establish deployment of a
continuous RTL driver.

## Model and numerical policy

| Binding | Identity |
| --- | --- |
| Model | `Qwen/Qwen2.5-0.5B-Instruct-AWQ` |
| Revision | `db09cd27ead7fee40cdee309693cf83601b9c899` |
| Checkpoint SHA256 | `c50d807b7bed7ff314308972e0f4bcf4e5a70bc60ad88fc7df53940831ed0c1b` |
| Numerical-policy document SHA256 | `19cc511f564037c61a0d50849f23364e9db6a4fb6dc755e438c20c48173b1a18` |
| Tokenizer SHA256 | `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539` |

The custom software reference follows the mathematical FP16 stage policy,
including bias before output FP16 rounding. Final RMSNorm uses float32
mean-square/rsqrt with epsilon `1e-6`, rounds the normalized activation to FP16,
and multiplies official FP16 normalization weights with FP16 rounding.

This is **not a stock Hugging Face generation claim**, full-model numerical
certification, or proof that RTL produces this 32-token continuation.

## Computation time and text identity

The **new CPU computation for the 30 additional tokens** took
**319.94626382 seconds**, using **two CPU threads**. This duration excludes the
earlier work that produced the two starting generated tokens. It is a host-side
measurement for this resumed fixture, not cold-start/end-to-end timing, RTL or
hardware latency, or a general quality/performance benchmark.

Both hashes below cover the complete decoded text shown above:

| UTF-8 byte convention | Bytes | SHA256 |
| --- | ---: | --- |
| Text with **no trailing newline** | 142 | `5b91602ad19a5a3e2696e275cbb4d7ddb7908175246f09278bd0a6eb96775b18` |
| Text followed by **exactly one LF byte (`0x0a`)** | 143 | `8f5a8c54f96bd3d6bdc64daabfcbd02126a36c264d6bfc3b4a9c44bf80a3dfd1` |

Only this scoped example, positive aggregate facts, token IDs, and binding
hashes are published. Runtime source/reproduction packaging remains separate.
No model weights, K/V tensors, execution archives, or internal reports are
included. This single capped software example does not establish general
dialogue quality, accepted RTL multi-token execution, or hardware performance.
