function weight_out = qwen2_runtime_hdl_softmax_normalize_half_entry(exp_val, recip_val)
%QWEN2_RUNTIME_HDL_SOFTMAX_NORMALIZE_HALF_ENTRY Wrapper for half normalize primitive.

    weight_out = qwen2_runtime.hdl.softmax_normalize_half_step(exp_val, recip_val);
end
