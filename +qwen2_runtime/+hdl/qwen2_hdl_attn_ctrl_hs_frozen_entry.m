function [attn_proj_out, out_valid, busy, read_req, read_addr, write_req, write_addr, shift_enable, write_key_token, write_value_token, next_valid_len] = qwen2_hdl_attn_ctrl_hs_frozen_entry(start, h_token, cache_valid_len, rope_position, read_key_data, read_value_data, read_data_valid)
%QWEN2_HDL_ATTN_CTRL_HS_FROZEN_ENTRY HDL entry with compile-time frozen constants.

    [weights, freqs_cis, hyperParameters, cfg] = coder.const(@qwen2_runtime.hdl.load_handshake_controller_codegen_constants);
    [attn_proj_out, out_valid, busy, read_req, read_addr, write_req, write_addr, shift_enable, write_key_token, write_value_token, next_valid_len] = ...
        qwen2_runtime.hdl.attention_token_controller_sram_handshake_step(start, h_token, cache_valid_len, rope_position, read_key_data, read_value_data, read_data_valid, weights, freqs_cis, hyperParameters, cfg);
end