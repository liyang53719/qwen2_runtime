function args = block_skeleton_step_args()
%BLOCK_SKELETON_STEP_ARGS Representative args for block skeleton.

    hiddenSize = 8;
    cacheLen = 16;
    laneCount = 4;
    numHeads = 2;
    start = false;
    input_vec = fi(zeros(hiddenSize, 1), true, 32, 14);
    score_mat = fi(zeros(cacheLen, numHeads), true, 16, 14);
    value_tensor = fi(zeros(cacheLen, laneCount, numHeads), true, 16, 14);
    residual_seed = fi(zeros(hiddenSize, 1), true, 32, 14);
    args = {start, input_vec, score_mat, value_tensor, residual_seed};
end
