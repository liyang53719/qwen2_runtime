function args = attention_score_mac_step_args()
%ATTENTION_SCORE_MAC_STEP_ARGS Representative args for sequential attention score PE.

    start = false;
    query_val = fi(0, true, 16, 14);
    key_val = fi(0, true, 16, 14);
    score_seed = fi(0, true, 32, 14);
    row_last = false;
    scale = coder.Constant(fi(single(0.0883883476483184), true, 16, 14));
    args = {start, query_val, key_val, score_seed, row_last, scale};
end
