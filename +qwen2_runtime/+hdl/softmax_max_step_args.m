function args = softmax_max_step_args()
%SOFTMAX_MAX_STEP_ARGS Representative args for sequential max tracker.

    start = false;
    score_val = fi(0, true, 16, 14);
    max_seed = fi(-8.0, true, 16, 14);
    row_last = false;
    args = {start, score_val, max_seed, row_last};
end
