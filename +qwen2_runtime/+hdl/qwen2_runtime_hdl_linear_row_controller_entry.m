function [acc_out, out_valid] = qwen2_runtime_hdl_linear_row_controller_entry(start, x_vec, w_row, acc_seed)
%QWEN2_RUNTIME_HDL_LINEAR_ROW_CONTROLLER_ENTRY Wrapper for row controller.

    [acc_out, out_valid] = qwen2_runtime.hdl.linear_row_controller_step(start, x_vec, w_row, acc_seed);
end
