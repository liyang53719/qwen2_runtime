function [attn_proj_out, key_cache_out, value_cache_out, next_valid_len, read_enable, read_addr, write_enable, write_addr, shift_enable] = attention_token_controller_sram_step(h_token, key_cache_in, value_cache_in, cache_valid_len, rope_position, weights, freqs_cis, hyperParameters, cfg)
%ATTENTION_TOKEN_CONTROLLER_SRAM_STEP Token-step attention controller with real q/k/v/o projections.

    [q_token, k_token, v_token] = qwen2_runtime.hdl.attention_token_qkv_project_step(h_token, weights, hyperParameters, cfg);

    [attn_flat, key_cache_out, value_cache_out, next_valid_len, read_enable, read_addr, write_enable, write_addr, shift_enable] = ...
        qwen2_runtime.hdl.attention_token_step_sram_contract_step( ...
            q_token, k_token, v_token, key_cache_in, value_cache_in, cache_valid_len, rope_position, freqs_cis, hyperParameters, cfg);

    attn_proj_out = qwen2_runtime.hdl.attention_token_o_project_step(attn_flat, weights, hyperParameters, cfg);
end