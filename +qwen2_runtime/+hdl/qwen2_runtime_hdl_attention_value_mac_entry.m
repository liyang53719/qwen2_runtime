function [value_acc_out, output_valid] = qwen2_runtime_hdl_attention_value_mac_entry(start, weight_val, value_val, acc_seed, row_last)
%QWEN2_RUNTIME_HDL_ATTENTION_VALUE_MAC_ENTRY Wrapper for value accumulation PE.

    [value_acc_out, output_valid] = qwen2_runtime.hdl.attention_value_mac_step(start, weight_val, value_val, acc_seed, row_last);
end
