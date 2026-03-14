function args = attention_head_controller_step_args()
%ATTENTION_HEAD_CONTROLLER_STEP_ARGS Representative args for head controller.

    cacheLen = 16;
    laneCount = 4;
    start = false;
    score_vec = fi(zeros(cacheLen, 1), true, 16, 14);
    value_mat = fi(zeros(cacheLen, laneCount), true, 16, 14);
    max_seed = fi(-8, true, 16, 14);
    sum_seed = fi(0, true, 32, 14);
    args = {start, score_vec, value_mat, max_seed, sum_seed};
end
