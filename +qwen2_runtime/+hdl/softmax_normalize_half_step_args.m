function args = softmax_normalize_half_step_args()
%SOFTMAX_NORMALIZE_HALF_STEP_ARGS Representative args for half normalize primitive.

    exp_val = half(1);
    recip_val = half(1);
    args = {exp_val, recip_val};
end
