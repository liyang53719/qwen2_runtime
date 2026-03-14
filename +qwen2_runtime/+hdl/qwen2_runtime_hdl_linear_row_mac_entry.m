function [acc_out, output_valid] = qwen2_runtime_hdl_linear_row_mac_entry(start, x_val, w_val, acc_seed, row_last)
%QWEN2_RUNTIME_HDL_LINEAR_ROW_MAC_ENTRY Wrapper for sequential MAC engine.

    [acc_out, output_valid] = qwen2_runtime.hdl.linear_row_mac_step(start, x_val, w_val, acc_seed, row_last);
end
