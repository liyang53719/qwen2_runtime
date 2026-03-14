function args = softmax_exp_step_args()
%SOFTMAX_EXP_STEP_ARGS Representative args for exp approximation.

    score_val = fi(0, true, 16, 14);
    max_val = fi(0, true, 16, 14);
    args = {score_val, max_val};
end
