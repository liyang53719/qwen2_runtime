function args = attention_multihead_controller_step_args()
%ATTENTION_MULTIHEAD_CONTROLLER_STEP_ARGS Representative args for multi-head controller.

    cacheLen = 16;
    laneCount = 4;
    numHeads = 2;
    start = false;
    score_mat = fi(zeros(cacheLen, numHeads), true, 16, 14);
    value_tensor = fi(zeros(cacheLen, laneCount, numHeads), true, 16, 14);
    max_seed = fi(-8, true, 16, 14);
    sum_seed = fi(0, true, 32, 14);
    args = {start, score_mat, value_tensor, max_seed, sum_seed};
end
