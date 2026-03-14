function recip_out = qwen2_runtime_hdl_softmax_recip_lookup_entry(denom_val)
%QWEN2_RUNTIME_HDL_SOFTMAX_RECIP_LOOKUP_ENTRY Wrapper for reciprocal lookup.

    recip_out = qwen2_runtime.hdl.softmax_recip_lookup_step(denom_val);
end
