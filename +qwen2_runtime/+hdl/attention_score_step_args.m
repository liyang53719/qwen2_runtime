function args = attention_score_step_args()
%ATTENTION_SCORE_STEP_ARGS Representative args for score HDL codegen.

    vecLen = 128;
    query_vec = fi(zeros(vecLen, 1), true, 16, 14);
    key_vec = fi(zeros(vecLen, 1), true, 16, 14);
    scale = coder.Constant(fi(single(0.0883883476483184), true, 16, 14));
    args = {query_vec, key_vec, scale};
end
