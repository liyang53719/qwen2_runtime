function args = softmax_exp_half_step_args()
%SOFTMAX_EXP_HALF_STEP_ARGS Representative args for half exp primitive.

    score_val = half(0);
    max_val = half(0);
    args = {score_val, max_val};
end
