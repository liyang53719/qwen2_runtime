function weight_out = qwen2_runtime_hdl_softmax_normalize_entry(exp_val, denom_recip)
%QWEN2_RUNTIME_HDL_SOFTMAX_NORMALIZE_ENTRY Wrapper for normalized weight multiply.

    weight_out = qwen2_runtime.hdl.softmax_normalize_step(exp_val, denom_recip);
end
