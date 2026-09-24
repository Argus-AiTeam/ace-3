# Transaction009 input-boundary readiness

**Input status: READY_INPUTS. Authority status: NOT_READY (fail closed).**
This artifact only assesses the would-be transaction009 input boundary. It is
not authority, a launcher, execution or consumption evidence, or publication
permission.

## Authoritative cursor9 boundary

The committed pointer selects generation 9, whose ledger has nine completed
receipts (`transaction-000` through `transaction-008`) and
`next_transaction_index: 9`.

| Binding | Absolute path | SHA256 |
|---|---|---|
| Authoritative cursor9 pointer | `/home/argustest/ace3-argus/build/model24_selected_token_position3_runs/ace3-position3-fresh-r11-20260831t215500z/transaction2-receipt-adoption-r4/authoritative-state.json` | `e73aa8883540356c2b035e32a7c09cbed2626dc076e578932018d04d9e1ce8b2` |
| Generation9 manifest | `/home/argustest/ace3-argus/build/model24_selected_token_position3_runs/ace3-position3-fresh-r11-20260831t215500z/transaction2-receipt-adoption-r4/state-generations/generation-0000000009/generation-manifest.json` | `440a55cd06462d35f46be95b2f78e79b6e304130b29d3967422bc86b3ce89041` |
| Generation9 ledger | `/home/argustest/ace3-argus/build/model24_selected_token_position3_runs/ace3-position3-fresh-r11-20260831t215500z/transaction2-receipt-adoption-r4/state-generations/generation-0000000009/ledger.json` | `208fa40f308708d01ba67cc496f9899ef9148ead20e6da49c46e4f2dd1ff40bf` |
| Invocation bound by the ledger | `/home/argustest/ace3-argus/build/model24_selected_token_position3_runs/ace3-position3-fresh-r11-20260831t215500z/invocation.json` | `9fb5418fe06c492633399ea79791a08ae25507f3e74b0cf0c003bdb79b43919d` |
| Cursor9 runtime-readiness record | `/home/argustest/ace3-argus/build/cursor9_next_position_runtime_readiness/readiness.json` | `7ed4bdaba4106368505d4e251a5e87edd61be735d4eeb98d775301172b27d659` |

## Selected token and FP16 embedding feedback

Cursor9 retains selected token ID `2` (tokenizer piece `#`) and FP16 logit bits
`19479`. The selected row is 896 FP16 elements from
`model.embed_tokens.weight`, with semantic SHA256
`8367b1f56e896acd2d99b64c3f0bd73f3090b8310ec7b294614074836a8af06a`.
The layer-0 receipt is also bound because it is the first receipt that consumed
that feedback.

| Binding | Absolute path | SHA256 |
|---|---|---|
| Selected-token `lm_head` receipt (`transaction-000`) | `/home/argustest/ace3-argus/build/model24_selected_token_position3_runs/ace3-position3-fresh-r11-20260831t215500z/transaction2-receipt-adoption-r4/state-generations/generation-0000000009/checkpoints/transaction-000.json` | `9fabdd9a91f1331220ca4e2c5682cc17e8be18fed444799f97cd3c66276cbdd0` |
| Layer-0 embedding-consumption receipt (`transaction-001`) | `/home/argustest/ace3-argus/build/model24_selected_token_position3_runs/ace3-position3-fresh-r11-20260831t215500z/transaction2-receipt-adoption-r4/state-generations/generation-0000000009/checkpoints/transaction-001.json` | `6019478026ea51b2cefd34e4a2e261edca8cb1f02034c1c5ed8fb20e0c3250a7` |
| Serialized selected-token embedding | `/home/argustest/ace3-argus/build/model24_selected_token_position3_runs/ace3-position3-fresh-r11-20260831t215500z/transactions/transaction-000/selected-token-embedding.hex` | `fcfad3b827ab02406ee259fff2572f7c7ff07c1e51df3477ffa09f3d1d20a58a` |
| Official checkpoint | `/home/argustest/ace3-argus/build/model24_rtl_cascade/checkpoint/model.safetensors` | `c50d807b7bed7ff314308972e0f4bcf4e5a70bc60ad88fc7df53940831ed0c1b` |
| `lm_head` RTL binary that produced the selection | `/home/argustest/ace3-argus/build/model24_selected_token_position3_runs/ace3-position3-fresh-r11-20260831t215500z/shared/lm-head/obj_dir/Vace3_streaming_tied_lm_head_topk` | `8168495a9164cfd3376e8ddacc3d62ffc49f3824ccc916d07aa06d891e6a7e97` |
| Official tokenizer | `/home/argustest/ace3-argus/build/host_dialogue_audit_20260829T182819Z/tokenizer/tokenizer.json` | `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539` |
| Official tokenizer config | `/home/argustest/ace3-argus/build/host_dialogue_audit_20260829T182819Z/tokenizer/tokenizer_config.json` | `5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583` |
| Tokenizer/model execution contract | `/home/argustest/ace3-argus/ace3/contracts/model24_execution.json` | `084c28f413a219b095b06e41cdc9bf3c1138a583cc5d0a2d00b94133cf9ba9e6` |

## Checkpoint008 parent and layer8 fixture

Checkpoint008 is the complete layer-7 receipt. Its state binds the 896-element
hidden output and the current position's layer-7 FP16 K/V state. The layer-8
fixture manifest binds exactly 26 official checkpoint tensors. The separate
layer-8 `position003.state` is the authenticated prior-position FP16 K/V parent
for layer 8.

| Binding | Absolute path | SHA256 |
|---|---|---|
| Complete checkpoint008 / layer-7 receipt | `/home/argustest/ace3-argus/build/model24_selected_token_position3_runs/ace3-position3-fresh-r11-20260831t215500z/transaction2-receipt-adoption-r4/state-generations/generation-0000000009/checkpoints/transaction-008.json` | `f4d4bb424bc1e52f2f56b1434e626d7ec7bbe29ad8ea8340c0c48deaeb63a44b` |
| Layer-7 hidden/current K/V output state | `/home/argustest/ace3-argus/build/model24_selected_token_position3_runs/ace3-position3-fresh-r11-20260831t215500z/transactions/transaction-008/position004.state` | `65f4d2b9cb29a757948ea9c22f69555d14285123036531e739dfcfc840511a55` |
| Layer-8 fixture manifest | `/home/argustest/ace3-argus/build/model24_selected_token_position2_runs/ace3-position2-fresh-v10-20260831t110456z/traversal/layer08/vectors/manifest.json` | `2d83dafa9b4f1e3a843633b9265abbd6dc1584012f980333f1698784ae1c35fc` |
| Layer-8 prior-position FP16 K/V parent | `/home/argustest/ace3-argus/build/model24_selected_token_position2_runs/ace3-position2-fresh-v10-20260831t110456z/traversal/layer08/position003.state` | `66fd0083cc752685da450ae00079b8431b9ee1e8b5fe600c01322726a37a0265` |

The intended input binding SHA256 is
`dec51a69a7afd42f780a77b975b1e716046f22c3d84e975f2a60e30105a9267b`.

## Exact would-be traversal bounds

- Exactly one transaction: index `9`; indices `10` through `25` are excluded.
- Exactly one decoder layer: layer `8`.
- Exactly one token step: zero-based transaction position `3`, consuming the
  layer-8 `position003.state` K/V parent and producing state position `4`.
- Exactly 896 hidden outputs, one natural RTL terminal, and an exact integer
  oracle match would be required for a successful result.
- Native asymmetric packed INT4 AWQ W4A16 G128 remains fixed: no qzero
  plus-one adjustment, FP16 scales/activations, and FP16 K/V.

## Fail-closed decision

All data inputs above are present and hash-bound, so the bounded data verdict
is **READY_INPUTS**. Transaction009 itself remains **NOT_READY**:
`build/argus-audit/tx008-r5/authority-transition-gate-r1/authority-transition.json`
(SHA256
`263f693b59681932568cb0f71bea2d465fda42cbca3137e33f59e53c2d99848f`)
records the Manager prohibition as `ACTIVE_UNCLEARED`, requires Manager
clearance, and reports an empty
`named_authority_execution_or_publication_paths_found` list.

This readiness task created zero transaction009-025 authority, launcher,
execution, consumption, or publication artifacts. It did not run model,
oracle, RTL compile/simulation, synthesis, PPA, FPGA, latency, or dialogue
work.
