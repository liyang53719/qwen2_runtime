function [score_out, output_valid] = qwen2_runtime_hdl_attention_score_mac_entry(start, query_val, key_val, score_seed, row_last, scale)
%QWEN2_RUNTIME_HDL_ATTENTION_SCORE_MAC_ENTRY Wrapper for sequential score PE.

    [score_out, output_valid] = qwen2_runtime.hdl.attention_score_mac_step(start, query_val, key_val, score_seed, row_last, scale);
end
