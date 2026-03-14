function [key_cache_out, value_cache_out, next_valid_len] = kv_cache_update_step(key_cache_in, value_cache_in, cache_valid_len, key_token, value_token)
%KV_CACHE_UPDATE_STEP Fixed-size KV cache update kernel.

    key_cache_out = fi(key_cache_in, true, 16, 14);
    value_cache_out = fi(value_cache_in, true, 16, 14);
    key_token = fi(key_token, true, 16, 14);
    value_token = fi(value_token, true, 16, 14);
    maxLenIndex = size(key_cache_in, 3);
    maxLen = cast(maxLenIndex, 'like', cache_valid_len);
    next_valid_len = cache_valid_len;

    if cache_valid_len < maxLen
        insertIdx = cache_valid_len + cast(1, 'like', cache_valid_len);
        for d = 1:size(key_cache_in, 1)
            for h = 1:size(key_cache_in, 2)
                key_cache_out(d, h, double(insertIdx), 1) = key_token(d, h, 1, 1);
                value_cache_out(d, h, double(insertIdx), 1) = value_token(d, h, 1, 1);
            end
        end
        next_valid_len = insertIdx;
    else
        for pos = 1:maxLenIndex-1
            for d = 1:size(key_cache_in, 1)
                for h = 1:size(key_cache_in, 2)
                    key_cache_out(d, h, pos, 1) = key_cache_in(d, h, pos + 1, 1);
                    value_cache_out(d, h, pos, 1) = value_cache_in(d, h, pos + 1, 1);
                end
            end
        end
        for d = 1:size(key_cache_in, 1)
            for h = 1:size(key_cache_in, 2)
                key_cache_out(d, h, maxLenIndex, 1) = key_token(d, h, 1, 1);
                value_cache_out(d, h, maxLenIndex, 1) = value_token(d, h, 1, 1);
            end
        end
        next_valid_len = maxLen;
    end
end
