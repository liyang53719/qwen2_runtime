function [head_out, out_valid] = qwen2_runtime_hdl_attention_head_controller_entry(start, score_vec, value_mat, max_seed, sum_seed)
%QWEN2_RUNTIME_HDL_ATTENTION_HEAD_CONTROLLER_ENTRY Wrapper for attention head controller HDL codegen.

    [head_out, out_valid] = qwen2_runtime.hdl.attention_head_controller_step(start, score_vec, value_mat, max_seed, sum_seed);
end