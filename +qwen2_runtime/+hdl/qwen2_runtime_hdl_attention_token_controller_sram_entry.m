function [attn_proj_out, key_cache_out, value_cache_out, next_valid_len, read_enable, read_addr, write_enable, write_addr, shift_enable] = qwen2_runtime_hdl_attention_token_controller_sram_entry(h_token, key_cache_in, value_cache_in, cache_valid_len, rope_position, weights, freqs_cis, hyperParameters, cfg)
%QWEN2_RUNTIME_HDL_ATTENTION_TOKEN_CONTROLLER_SRAM_ENTRY Wrapper for token-step attention controller.

    [attn_proj_out, key_cache_out, value_cache_out, next_valid_len, read_enable, read_addr, write_enable, write_addr, shift_enable] = ...
        qwen2_runtime.hdl.attention_token_controller_sram_step( ...
            h_token, key_cache_in, value_cache_in, cache_valid_len, rope_position, weights, freqs_cis, hyperParameters, cfg);
end