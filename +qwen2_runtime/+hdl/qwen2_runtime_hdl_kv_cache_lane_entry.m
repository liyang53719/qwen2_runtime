function [cache_word_out, next_valid_len] = qwen2_runtime_hdl_kv_cache_lane_entry(cache_word_in, cache_valid_len, token_word)
%QWEN2_RUNTIME_HDL_KV_CACHE_LANE_ENTRY Wrapper for single-lane cache update.

    [cache_word_out, next_valid_len] = qwen2_runtime.hdl.kv_cache_lane_step(cache_word_in, cache_valid_len, token_word);
end
