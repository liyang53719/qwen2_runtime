function [h_out, key_out, value_out] = qwen2_runtime_hdl_block_entry(h_in, key_in, value_in, cache_valid_len, weights, hyperParameters, freqs_cis, runtimeCfg)
%QWEN2_RUNTIME_HDL_BLOCK_ENTRY Thin top-level wrapper for HDL codegen.

    [h_out, key_out, value_out] = qwen2_runtime.hdl.block_entry(h_in, key_in, value_in, cache_valid_len, weights, hyperParameters, freqs_cis, runtimeCfg);
end
