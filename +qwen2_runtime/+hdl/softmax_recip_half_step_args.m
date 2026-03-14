function args = softmax_recip_half_step_args()
%SOFTMAX_RECIP_HALF_STEP_ARGS Representative args for half reciprocal primitive.

    denom_val = half(1);
    args = {denom_val};
end
