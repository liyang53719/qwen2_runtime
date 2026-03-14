function args = softmax_sum_step_args()
%SOFTMAX_SUM_STEP_ARGS Representative args for denominator accumulator.

    start = false;
    exp_val = fi(0, true, 16, 14);
    sum_seed = fi(0, true, 32, 14);
    row_last = false;
    args = {start, exp_val, sum_seed, row_last};
end
