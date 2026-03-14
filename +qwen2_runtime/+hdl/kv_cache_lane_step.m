function [cache_word_out, next_valid_len] = kv_cache_lane_step(cache_word_in, cache_valid_len, token_word)
%KV_CACHE_LANE_STEP Single-lane KV cache update to avoid huge IO fanout.

    cache_word_out = fi(cache_word_in, true, 16, 14);
    token_word = fi(token_word, true, 16, 14);
    maxLen = uint8(length(cache_word_in));
    next_valid_len = cache_valid_len;

    if cache_valid_len < maxLen
        insertIdx = cache_valid_len + 1;
        cache_word_out(double(insertIdx)) = token_word;
        next_valid_len = insertIdx;
    else
        for pos = uint8(1):maxLen-1
            cache_word_out(double(pos)) = cache_word_in(double(pos + 1));
        end
        cache_word_out(double(maxLen)) = token_word;
        next_valid_len = maxLen;
    end
end
