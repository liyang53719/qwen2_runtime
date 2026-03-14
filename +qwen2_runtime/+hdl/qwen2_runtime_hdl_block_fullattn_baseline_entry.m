function [block_out, out_valid] = qwen2_runtime_hdl_block_fullattn_baseline_entry(start, input_vec, attn_mix_vec, residual_seed)
%QWEN2_RUNTIME_HDL_BLOCK_FULLATTN_BASELINE_ENTRY Wrapper for full-attention block baseline.

    [block_out, out_valid] = qwen2_runtime.hdl.block_fullattn_baseline_step(start, input_vec, attn_mix_vec, residual_seed);
end
