function [attn_out, out_valid, busy, read_req, read_addr, write_req, write_addr, shift_enable, write_key_token, write_value_token, next_valid_len] = qwen2_runtime_hdl_attention_token_step_sram_handshake_entry(start, q_token, k_token, v_token, cache_valid_len, rope_position, read_key_data, read_value_data, read_data_valid, freqs_cis, hyperParameters, cfg)
%QWEN2_RUNTIME_HDL_ATTENTION_TOKEN_STEP_SRAM_HANDSHAKE_ENTRY Root wrapper for handshake attention top.

    [attn_out, out_valid, busy, read_req, read_addr, write_req, write_addr, shift_enable, write_key_token, write_value_token, next_valid_len] = ...
        qwen2_runtime.hdl.attention_token_step_sram_handshake_step(start, q_token, k_token, v_token, cache_valid_len, rope_position, read_key_data, read_value_data, read_data_valid, freqs_cis, hyperParameters, cfg);
end