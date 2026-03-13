function [attn_proj_out, out_valid, busy, read_req, read_addr, write_req, write_addr, shift_enable, write_key_token, write_value_token, next_valid_len] = qwen2_runtime_hdl_attention_token_controller_sram_handshake_entry(start, h_token, cache_valid_len, rope_position, read_key_data, read_value_data, read_data_valid, weights, freqs_cis, hyperParameters, cfg)
%QWEN2_RUNTIME_HDL_ATTENTION_TOKEN_CONTROLLER_SRAM_HANDSHAKE_ENTRY Wrapper for handshake attention controller.

    [attn_proj_out, out_valid, busy, read_req, read_addr, write_req, write_addr, shift_enable, write_key_token, write_value_token, next_valid_len] = ...
        qwen2_runtime.hdl.attention_token_controller_sram_handshake_step(start, h_token, cache_valid_len, rope_position, read_key_data, read_value_data, read_data_valid, weights, freqs_cis, hyperParameters, cfg);
end