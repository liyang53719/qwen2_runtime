# Current Full Block Baseline

This repository snapshot captures the current reduced-dimension full-block HDL baseline for qwen2 runtime.

## Decisive files under +qwen2_runtime

- `+qwen2_runtime/defaultRuntimeConfig.m`
  Base runtime struct. `defaultHDLConfig` extends this, so the HDL baseline depends on its field layout.
- `+qwen2_runtime/defaultHDLConfig.m`
  Defines the HDL-oriented configuration, including `BlockKernel`, safe math knobs, and floating-point execution mode.
- `+qwen2_runtime/+hdl/block_entry.m`
  HDL-facing block entry that normalizes top-level inputs before dispatching into the block kernel.
- `+qwen2_runtime/+hdl/block_kernel.m`
  The main architectural baseline: input RMSNorm -> attention -> residual -> post-attention RMSNorm -> gated MLP -> residual.
- `+qwen2_runtime/+hdl/rmsnorm_step.m`
  HDL-safe RMSNorm approximation used on both norm sites.
- `+qwen2_runtime/+hdl/attention_step.m`
  Full attention baseline with RoPE, KV cache update, repeated KV heads, streaming head accumulation, and output projection.
- `+qwen2_runtime/+hdl/gated_mlp_step.m`
  Full gated MLP baseline including the HDL-safe SiLU approximation path.
- `+qwen2_runtime/+hdl/linear_step.m`
  Common floating-point linear helper used by attention and MLP.
- `+qwen2_runtime/+hdl/qwen2_runtime_hdl_block_entry.m`
  HDL codegen wrapper targeted by the full-block generator.
- `+qwen2_runtime/+hdl/block_entry_baseline_args.m`
  Reduced synthetic argument builder that makes the full-block baseline reproducible without loading full model weights.
- `+qwen2_runtime/+hdl/generate_block_full_baseline.m`
  The generator entrypoint for the current baseline. This is the command path that now completes HDL code generation successfully.

## Supporting files outside +qwen2_runtime

- `qwen2_runtime_hdl_block_entry.m`
  Thin root-level wrapper used as the top function for MATLAB HDL Coder.
- `qwen2_runtime_hdl_block_entry_tb.m`
  MATLAB stimulus used to generate the auto testbench for the reduced full-block baseline.

## Reproduction

Run this from the repository root in MATLAB:

```matlab
addpath(pwd);
qwen2_runtime.hdl.generate_block_full_baseline();
```

Expected result:

- HDL code generation succeeds.
- The generated Verilog top is emitted under `codegen/hdl_block_full_baseline/.../hdlsrc/`.
- The flow currently uses native floating-point HDL and therefore emits many warnings, but no blocking errors.

## Current boundary of the baseline

- This is a proof baseline for the real block sequence, not yet a hardware-efficient fixed-point or DDR-fed implementation.
- The next iteration should focus on reducing port pressure, replacing floating-point-heavy kernels, and introducing hardware-oriented streaming and memory interfaces.

## Follow-on hardware subpath

- The first validated hardware-oriented increment now lives in `CURRENT_GATED_MLP_HARDWARE_BASELINE.md`.
- That path keeps the full-block baseline unchanged, but adds a separate fixed-point gated MLP codegen flow as the first proven hardware submodule.