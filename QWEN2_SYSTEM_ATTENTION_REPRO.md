# Qwen2 System Attention Reproduction

This document records the current system-level token-step attention path, how to reproduce the numerical checks, and how to attempt real-dimension RTL generation.

## Current Scope

- Quantization error is accepted as-is for now. This branch does not include further precision-tuning work.
- The active system path is external-KV token-step attention with explicit SRAM-facing boundaries.
- Three system layers now exist:
  - Array-style external KV contract:
    - `qwen2_runtime.hdl.attention_token_step_sram_contract_step`
  - Multi-cycle handshake controller:
    - `qwen2_runtime.hdl.attention_token_controller_sram_handshake_step`
  - Block-0 token-step system top:
    - `qwen2_runtime.hdl.block0_token_system_step`

## Key Files

- `+qwen2_runtime/+hdl/attention_token_step_sram_step.m`
- `+qwen2_runtime/+hdl/attention_token_step_sram_contract_step.m`
- `+qwen2_runtime/+hdl/attention_token_step_sram_handshake_step.m`
- `+qwen2_runtime/+hdl/attention_token_controller_sram_step.m`
- `+qwen2_runtime/+hdl/attention_token_controller_sram_handshake_step.m`
- `+qwen2_runtime/+hdl/block0_token_system_step.m`
- `+qwen2_runtime/+hdl/try_real_block0_forward_with_system_attention.m`
- `+qwen2_runtime/+hdl/validate_block0_system_nonempty_cache.m`

## Reproduce Current Numerical Checks

Run the real first-token block-0 attempt:

```matlab
addpath(pwd);
qwen2_runtime.hdl.try_real_block0_forward_with_system_attention(151644, 8);
```

Current observed output:

- `real_block0_attention max abs diff : 4.01519`
- `real_block0_attention mean abs     : 0.233031`
- `real_block0_block max abs diff     : 9.46539`
- `real_block0_block mean abs         : 0.764593`

Run the non-empty-cache block-0 check on a short real prompt:

```matlab
addpath(pwd);
qwen2_runtime.hdl.validate_block0_system_nonempty_cache([151644 9707 25], 2, 8);
```

This drives the new block-0 system top with an externally supplied prior KV cache and compares the output against `qwen2_runtime.layer.block`.

Current observed output:

- `target token index           : 2`
- `cache valid length           : 1`
- `nonempty block0 max abs diff : 3.74982`
- `nonempty block0 mean abs     : 0.560314`
- `out_valid                    : 1`
- `write_addr                   : 2`
- `next_valid_len               : 2`

## Reproduce Handshake-Level Smoke Checks

Handshake attention top smoke:

```matlab
addpath(pwd);
args = qwen2_runtime.hdl.attention_token_step_sram_handshake_args(8, false);
[attn_out, out_valid, busy, read_req, read_addr, write_req, write_addr, shift_enable, write_key_token, write_value_token, next_valid_len] = ...
    qwen2_runtime.hdl.attention_token_step_sram_handshake_step(args{:});
```

Current observed output:

- `size(attn_out)  = [1536 1]`
- `out_valid       = 1`
- `busy            = 0`
- `read_req        = 0`
- `read_addr       = 1`
- `write_req       = 1`
- `write_addr      = 1`
- `next_valid_len  = 1`

## Attempt Real-Dimension RTL Generation

Generate real-dimension handshake attention controller RTL:

```matlab
addpath(pwd);
info = qwen2_runtime.hdl.generate_attention_token_controller_sram_handshake_baseline('qwen_params.mat', 8);
disp(info.OutputDir);
```

Generate real-dimension block-0 system RTL:

```matlab
addpath(pwd);
info = qwen2_runtime.hdl.generate_block0_token_system_baseline('qwen_params.mat', 8);
disp(info.OutputDir);
```

Expected output directories:

- `codegen/hdl_attention_token_controller_sram_handshake/qwen2_runtime_hdl_attention_token_controller_sram_handshake_entry`
- `codegen/hdl_block0_token_system/qwen2_runtime_hdl_block0_token_system_entry`

Current observed status:

- Array-style real-dimension attention baseline creates `hdlsrc/interface/`, but that directory is empty, so no RTL source was emitted.
- Handshake controller real-dimension attempt creates `hdlsrc/` and an HDL conformance report, but `hdlsrc/interface/` is empty.
- Block-0 real-dimension attempt creates `hdlsrc/html/`, `hdlsrc/interface/`, and `hdlsrc/html/report.mldatx`, but `hdlsrc/interface/` is empty.
- The handshake controller attempt currently fails HDL conformance on variable-size matrix usage in internal cache/score/value tensors; this is the active blocker after resolving the earlier fixed-point typing issues.

## Notes

- Root-level `qwen2_runtime_hdl_*.m` wrappers are required in this repo for HDL Coder entry points.
- The real-dimension RTL attempt uses true Qwen dimensions and actual block-0 weights from `qwen_params.mat`.
- If the block-0 top proves too heavy for a given machine or license configuration, the handshake controller RTL should still be attempted first because it isolates the external-memory system boundary.
- At the current state, the real-dimension attempts reach report/conformance generation, but they do not yet emit Verilog/VHDL artifacts in `hdlsrc/interface/`.