function [block_out, out_valid] = qwen2_runtime_hdl_block_skeleton_streaming_entry(start, input_vec, score_token, value_token, token_valid, token_last, residual_seed)
%QWEN2_RUNTIME_HDL_BLOCK_SKELETON_STREAMING_ENTRY Streaming wrapper for block skeleton HDL codegen.

    [block_out, out_valid] = qwen2_runtime.hdl.block_skeleton_streaming_step( ...
        start, input_vec, score_token, value_token, token_valid, token_last, residual_seed);
end