function [attn_out, key_cache_out, value_cache_out, next_valid_len, read_enable, read_addr, write_enable, write_addr, shift_enable, write_key_token, write_value_token] = qwen2_runtime_hdl_attention_token_step_sram_contract_entry(q_token, k_token, v_token, key_cache_in, value_cache_in, cache_valid_len, rope_position, freqs_cis, hyperParameters, cfg)
%QWEN2_RUNTIME_HDL_ATTENTION_TOKEN_STEP_SRAM_CONTRACT_ENTRY Wrapper for token-step SRAM contract kernel.

    [attn_out, key_cache_out, value_cache_out, next_valid_len, read_enable, read_addr, write_enable, write_addr, shift_enable, write_key_token, write_value_token] = ...
        qwen2_runtime.hdl.attention_token_step_sram_contract_step( ...
            q_token, k_token, v_token, key_cache_in, value_cache_in, cache_valid_len, rope_position, freqs_cis, hyperParameters, cfg);
end