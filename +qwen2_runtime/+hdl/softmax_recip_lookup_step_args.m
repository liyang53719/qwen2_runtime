function args = softmax_recip_lookup_step_args()
%SOFTMAX_RECIP_LOOKUP_STEP_ARGS Representative args for reciprocal lookup.

    denom_val = fi(1, true, 16, 14);
    args = {denom_val};
end
