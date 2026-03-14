function args = softmax_normalize_step_args()
%SOFTMAX_NORMALIZE_STEP_ARGS Representative args for normalization multiply.

    exp_val = fi(0, true, 16, 14);
    denom_recip = fi(1, true, 16, 14);
    args = {exp_val, denom_recip};
end
