function args = block_fullattn_baseline_step_args()
%BLOCK_FULLATTN_BASELINE_STEP_ARGS Representative args for full-attention block baseline.

    hiddenSize = 8;
    start = false;
    input_vec = fi(zeros(hiddenSize, 1), true, 32, 14);
    attn_mix_vec = fi(zeros(hiddenSize, 1), true, 32, 14);
    residual_seed = fi(zeros(hiddenSize, 1), true, 32, 14);
    args = {start, input_vec, attn_mix_vec, residual_seed};
end
