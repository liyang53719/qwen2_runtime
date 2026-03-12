function [attn_out, out_valid] = qwen2_runtime_hdl_attention_multihead_controller_entry(start, score_mat, value_tensor, max_seed, sum_seed)
%QWEN2_RUNTIME_HDL_ATTENTION_MULTIHEAD_CONTROLLER_ENTRY Thin top-level wrapper for HDL codegen.

    [attn_out, out_valid] = qwen2_runtime.hdl.qwen2_runtime_hdl_attention_multihead_controller_entry(start, score_mat, value_tensor, max_seed, sum_seed);
end