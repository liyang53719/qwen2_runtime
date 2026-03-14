function args = attention_row_controller_step_args()
%ATTENTION_ROW_CONTROLLER_STEP_ARGS Representative args for attention row controller.

    vecLen = 128;
    start = false;
    query_vec = fi(zeros(vecLen, 1), true, 16, 14);
    key_vec = fi(zeros(vecLen, 1), true, 16, 14);
    score_seed = fi(0, true, 32, 14);
    scale = coder.Constant(fi(single(0.0883883476483184), true, 16, 14));
    args = {start, query_vec, key_vec, score_seed, scale};
end
