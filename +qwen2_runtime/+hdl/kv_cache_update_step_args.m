function args = kv_cache_update_step_args()
%KV_CACHE_UPDATE_STEP_ARGS Representative args for cache update HDL codegen.

    headDim = 128;
    numKVHeads = 2;
    maxCacheLen = 16;
    batchSize = 1;

    key_cache_in = fi(zeros(headDim, numKVHeads, maxCacheLen, batchSize), true, 16, 14);
    value_cache_in = fi(zeros(headDim, numKVHeads, maxCacheLen, batchSize), true, 16, 14);
    cache_valid_len = coder.Constant(uint8(0));
    key_token = fi(zeros(headDim, numKVHeads, 1, batchSize), true, 16, 14);
    value_token = fi(zeros(headDim, numKVHeads, 1, batchSize), true, 16, 14);
    args = {key_cache_in, value_cache_in, cache_valid_len, key_token, value_token};
end
