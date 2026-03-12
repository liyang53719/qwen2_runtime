# Current Fixed-Point Subpaths

This repository snapshot extends the earlier fixed-point gated MLP work with two more validated hardware-oriented subpaths and one required configuration fix for the full-block baseline.

## Scope of this increment

- Keep the existing full-block float HDL baseline valid.
- Add a standalone fixed-point RMSNorm baseline.
- Add a combined fixed-point block tail baseline:
  post-attention RMSNorm -> gated MLP -> residual add.
- Replace HDL-unsafe string mode checks with a boolean configuration flag so the float proof baseline still codegens cleanly.

## Decisive files

- `+qwen2_runtime/defaultHDLConfig.m`
  Adds `UseFixedPointHDL=false` to the float proof baseline.
- `+qwen2_runtime/defaultHardwareHDLConfig.m`
  Sets `UseFixedPointHDL=true` for the hardware-oriented subpaths.
- `+qwen2_runtime/+hdl/linear_step.m`
  Uses the boolean flag instead of HDL-unsafe string checks when selecting the fixed-point path.
- `+qwen2_runtime/+hdl/gated_mlp_step.m`
  Uses the boolean flag for fixed-point mode detection.
- `+qwen2_runtime/+hdl/rmsnorm_step.m`
  Adds a fixed-point RMSNorm execution path while preserving the float baseline path.
- `+qwen2_runtime/+hdl/attention_step.m`
  Propagates `cfg` through the common linear helper so attention no longer bypasses runtime numeric configuration.
- `+qwen2_runtime/+hdl/rmsnorm_entry_hardware_args.m`
  Reduced fixed-point RMSNorm argument builder.
- `+qwen2_runtime/+hdl/generate_rmsnorm_hardware_baseline.m`
  Standalone RTL generation entry for the fixed-point RMSNorm subpath.
- `+qwen2_runtime/+hdl/block_mlp_tail_step.m`
  Combined fixed-point post-attention tail subchain.
- `+qwen2_runtime/+hdl/block_mlp_tail_entry_hardware_args.m`
  Reduced fixed-point argument builder for the combined tail baseline.
- `+qwen2_runtime/+hdl/generate_block_mlp_tail_hardware_baseline.m`
  Standalone RTL generation entry for the combined tail baseline.
- `qwen2_runtime_hdl_rmsnorm_entry.m`
  Root HDL Coder top for the fixed-point RMSNorm baseline.
- `qwen2_runtime_hdl_rmsnorm_entry_tb.m`
  MATLAB stimulus for the fixed-point RMSNorm baseline.
- `qwen2_runtime_hdl_block_mlp_tail_entry.m`
  Root HDL Coder top for the combined tail baseline.
- `qwen2_runtime_hdl_block_mlp_tail_entry_tb.m`
  MATLAB stimulus for the combined tail baseline.

## Verified generation flows

Run these from the repository root in MATLAB:

```matlab
addpath(pwd);
qwen2_runtime.hdl.generate_block_full_baseline();
qwen2_runtime.hdl.generate_rmsnorm_hardware_baseline();
qwen2_runtime.hdl.generate_block_mlp_tail_hardware_baseline();
```

Expected result:

- The full-block float proof baseline still codegens successfully.
- The fixed-point RMSNorm baseline codegens successfully.
- The fixed-point block-tail baseline codegens successfully.

## Current boundary

- Attention is still not converted into the same fixed-point hardware flow.
- The full-block mainline remains a float proof baseline.
- The fixed-point subpaths now cover the normalization and MLP-heavy post-attention tail, but not the attention core itself.