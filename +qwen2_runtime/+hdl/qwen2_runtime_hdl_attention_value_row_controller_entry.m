function [value_acc_out, out_valid] = qwen2_runtime_hdl_attention_value_row_controller_entry(start, weight_vec, value_vec, acc_seed)
%QWEN2_RUNTIME_HDL_ATTENTION_VALUE_ROW_CONTROLLER_ENTRY Wrapper for value controller.

    [value_acc_out, out_valid] = qwen2_runtime.hdl.attention_value_row_controller_step(start, weight_vec, value_vec, acc_seed);
end
