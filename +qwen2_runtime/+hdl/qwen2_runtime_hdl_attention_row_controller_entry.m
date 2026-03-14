function [score_out, out_valid] = qwen2_runtime_hdl_attention_row_controller_entry(start, query_vec, key_vec, score_seed, scale)
%QWEN2_RUNTIME_HDL_ATTENTION_ROW_CONTROLLER_ENTRY Wrapper for attention row controller.

    [score_out, out_valid] = qwen2_runtime.hdl.attention_row_controller_step(start, query_vec, key_vec, score_seed, scale);
end
