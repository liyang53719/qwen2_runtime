function [block_out, out_valid] = qwen2_runtime_hdl_block_skeleton_entry(start, input_vec, score_mat, value_tensor, residual_seed)
%QWEN2_RUNTIME_HDL_BLOCK_SKELETON_ENTRY Wrapper for block skeleton HDL codegen.

    [block_out, out_valid] = qwen2_runtime.hdl.block_skeleton_step( ...
        start, input_vec, score_mat, value_tensor, residual_seed);
end
