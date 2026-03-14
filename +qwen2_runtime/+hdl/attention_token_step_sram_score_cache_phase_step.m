function [score_cache_out, fill_valid] = attention_token_step_sram_score_cache_phase_step(start, q_rot_reg, head_idx, key_cache_reg, kv_head_idx, scale_fix, active_len, cfg)
%ATTENTION_TOKEN_STEP_SRAM_SCORE_CACHE_PHASE_STEP Wrap query/key assembly for score-cache fill.

    headDim = size(q_rot_reg, 1);
    maxCacheLen = size(key_cache_reg, 3);
    F = fimath(q_rot_reg);

    query_vec = fi(q_rot_reg(:, double(head_idx)), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
    key_mat = reshape(key_cache_reg(:, kv_head_idx, :, 1), [headDim, maxCacheLen]);
    [score_cache_out, fill_valid] = qwen2_runtime.hdl.attention_token_step_sram_score_cache_fill_step( ...
        start, query_vec, key_mat, active_len, scale_fix);
end