# Standalone final RMSNorm

`ace3_final_rmsnorm` fixes the accepted FP16 RMSNorm arithmetic boundary to the
Qwen2.5 0.5B final hidden width of 896 elements. Its streamed weight input is
bound to `model.norm.weight` from the authenticated official AWQ checkpoint.

The public module has no parameters. One accepted start is followed by exactly
896 activation/weight input handshakes and exactly 896 indexed output
handshakes. Arithmetic and exceptional behavior are frozen in
`ace3/contracts/final_rmsnorm.json`.

Run the claim-bearing acceptance with a caller-supplied checkpoint path:

    make final-rmsnorm-acceptance \
      FINAL_RMSNORM_CHECKPOINT=/path/to/model.safetensors \
      FINAL_RMSNORM_RUN_DIR=build/final-rmsnorm/attempt-01

The runner authenticates the official revision and checkpoint digest, writes
all generated vectors and simulator evidence below the ignored run directory,
checks malformed and tampered inputs, covers 896 four-state Icarus outputs, and
performs 3,584 exact Verilator comparisons. Simulator capture is oracle-free;
the comparator opens oracle artifacts only after validating natural completion
and raw-output integrity.

This bounded result covers only the standalone final normalization transaction.
