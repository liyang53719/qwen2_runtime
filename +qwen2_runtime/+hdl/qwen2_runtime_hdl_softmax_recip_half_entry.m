function recip_out = qwen2_runtime_hdl_softmax_recip_half_entry(denom_val)
%QWEN2_RUNTIME_HDL_SOFTMAX_RECIP_HALF_ENTRY Wrapper for half reciprocal primitive.

    recip_out = qwen2_runtime.hdl.softmax_recip_half_step(denom_val);
end
