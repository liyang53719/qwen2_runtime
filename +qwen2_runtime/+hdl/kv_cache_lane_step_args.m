function args = kv_cache_lane_step_args()
%KV_CACHE_LANE_STEP_ARGS Representative args for single-lane cache update.

    maxCacheLen = 16;
    cache_word_in = fi(zeros(maxCacheLen, 1), true, 16, 14);
    cache_valid_len = coder.Constant(uint8(0));
    token_word = fi(0, true, 16, 14);
    args = {cache_word_in, cache_valid_len, token_word};
end
