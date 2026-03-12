# Current Gated MLP Hardware Baseline

This repository snapshot also contains the first hardware-oriented increment built on top of the pushed full-block baseline: a fixed-point gated MLP HDL path.

## Purpose

- Keep the existing full-block float HDL baseline intact.
- Prove a smaller fixed-point hardware submodule before converting the whole block.
- Reuse the existing HDL tile MAC kernel instead of introducing a separate linear implementation style.

## Decisive files

- `+qwen2_runtime/defaultHDLConfig.m`
  Adds numeric configuration fields used by the fixed-point linear and MLP path.
- `+qwen2_runtime/defaultHardwareHDLConfig.m`
  Switches the HDL path to fixed-point mode for the gated MLP baseline.
- `+qwen2_runtime/+hdl/linear_step.m`
  Extends the common HDL linear helper so it can run either float or fixed-point based on config.
- `+qwen2_runtime/+hdl/gated_mlp_step.m`
  Routes gate/up/down projections through the config-driven linear helper and uses an HDL-safe SiLU approximation in fixed-point mode.
- `+qwen2_runtime/+hdl/gated_mlp_entry_hardware_args.m`
  Builds reduced fixed-point inputs and weights directly as `fi` objects at the entry boundary.
- `+qwen2_runtime/+hdl/qwen2_runtime_hdl_gated_mlp_entry.m`
  Package-level wrapper used for HDL codegen.
- `qwen2_runtime_hdl_gated_mlp_entry.m`
  Root-level HDL Coder top function.
- `qwen2_runtime_hdl_gated_mlp_entry_tb.m`
  MATLAB testbench stimulus matching the fixed-point entry boundary.
- `+qwen2_runtime/+hdl/generate_gated_mlp_hardware_baseline.m`
  Reproducible RTL generation entrypoint for this hardware subpath.

## What is new

- The repository now has a separate hardware config for fixed-point HDL experiments.
- The HDL linear helper can execute in fixed-point mode using the existing tile MAC kernel.
- The gated MLP path can generate RTL without relying on internal float-to-fixed casts.
- Input tensors and weights for the hardware baseline are constructed as `fi` values before entering the kernel, which avoids the HDL Coder cast failures seen in earlier attempts.

## Correct RTL generation flow

Run this from the repository root in MATLAB:

```matlab
addpath(pwd);
info = qwen2_runtime.hdl.generate_gated_mlp_hardware_baseline();
disp(info);
```

Expected result:

- HDL code generation succeeds.
- Verilog and HDL testbench are emitted under:
  `codegen/hdl_gated_mlp_hardware_baseline/qwen2_runtime_hdl_gated_mlp_entry/.../hdlsrc/`
- The generated top corresponds to the fixed-point gated MLP submodule, not the full transformer block.

## Current boundary

- This is only the gated MLP hardware baseline.
- RMSNorm and attention are not yet converted into the same hardware-oriented fixed-point flow.
- The full-block mainline remains the earlier float proof baseline until more subpaths are integrated.