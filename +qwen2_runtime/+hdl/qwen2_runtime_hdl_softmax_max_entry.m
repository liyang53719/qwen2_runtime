function [max_out, output_valid] = qwen2_runtime_hdl_softmax_max_entry(start, score_val, max_seed, row_last)
%QWEN2_RUNTIME_HDL_SOFTMAX_MAX_ENTRY Wrapper for max tracker.

    [max_out, output_valid] = qwen2_runtime.hdl.softmax_max_step(start, score_val, max_seed, row_last);
end
