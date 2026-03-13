function [block_out, out_valid, busy, read_req, read_addr, write_req, write_addr, shift_enable, write_key_token, write_value_token, next_valid_len] = qwen2_runtime_hdl_block0_token_system_entry(start, h_token, cache_valid_len, rope_position, read_key_data, read_value_data, read_data_valid, weights, hyperParameters, freqs_cis, cfg)
%QWEN2_RUNTIME_HDL_BLOCK0_TOKEN_SYSTEM_ENTRY Root wrapper for block-0 system top.

    [block_out, out_valid, busy, read_req, read_addr, write_req, write_addr, shift_enable, write_key_token, write_value_token, next_valid_len] = ...
        qwen2_runtime.hdl.block0_token_system_step(start, h_token, cache_valid_len, rope_position, read_key_data, read_value_data, read_data_valid, weights, hyperParameters, freqs_cis, cfg);
end