function [key_cache_out, value_cache_out, next_valid_len] = qwen2_runtime_hdl_kv_cache_entry(key_cache_in, value_cache_in, cache_valid_len, key_token, value_token)
%QWEN2_RUNTIME_HDL_KV_CACHE_ENTRY Wrapper for KV cache HDL codegen.

    [key_cache_out, value_cache_out, next_valid_len] = qwen2_runtime.hdl.kv_cache_update_step(key_cache_in, value_cache_in, cache_valid_len, key_token, value_token);
end
