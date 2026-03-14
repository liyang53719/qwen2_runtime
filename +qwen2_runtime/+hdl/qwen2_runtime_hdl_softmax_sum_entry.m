function [sum_out, output_valid] = qwen2_runtime_hdl_softmax_sum_entry(start, exp_val, sum_seed, row_last)
%QWEN2_RUNTIME_HDL_SOFTMAX_SUM_ENTRY Wrapper for denominator accumulator.

    [sum_out, output_valid] = qwen2_runtime.hdl.softmax_sum_step(start, exp_val, sum_seed, row_last);
end
