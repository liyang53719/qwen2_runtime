function exp_out = qwen2_runtime_hdl_softmax_exp_half_entry(score_val, max_val)
%QWEN2_RUNTIME_HDL_SOFTMAX_EXP_HALF_ENTRY Wrapper for half exp primitive.

    exp_out = qwen2_runtime.hdl.softmax_exp_half_step(score_val, max_val);
end
