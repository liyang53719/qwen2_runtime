function score = qwen2_runtime_hdl_attention_score_entry(query_vec, key_vec, scale)
%QWEN2_RUNTIME_HDL_ATTENTION_SCORE_ENTRY Wrapper for score HDL codegen.

    score = qwen2_runtime.hdl.attention_score_step(query_vec, key_vec, scale);
end
