function args = attention_weighted_value_controller_step_args()
%ATTENTION_WEIGHTED_VALUE_CONTROLLER_STEP_ARGS Representative args for assembled attention datapath.

    vecLen = 16;
    start = false;
    score_vec = fi(zeros(vecLen, 1), true, 16, 14);
    value_vec = fi(zeros(vecLen, 1), true, 16, 14);
    max_seed = fi(-8, true, 16, 14);
    sum_seed = fi(0, true, 32, 14);
    value_seed = fi(0, true, 32, 14);
    args = {start, score_vec, value_vec, max_seed, sum_seed, value_seed};
end
