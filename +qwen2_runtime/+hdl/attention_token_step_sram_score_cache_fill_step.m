function [score_cache_out, fill_valid] = attention_token_step_sram_score_cache_fill_step(start, query_vec, key_mat, active_len, scale)
%ATTENTION_TOKEN_STEP_SRAM_SCORE_CACHE_FILL_STEP Build a fixed-length score cache via the row controller.

    F = scoreCacheFimath();
    maxCacheLen = uint16(size(key_mat, 2));
    persistent token_idx running score_cache_reg score_start_pending
    if isempty(token_idx)
        token_idx = uint16(1);
        running = false;
        score_cache_reg = fi(zeros(double(maxCacheLen), 1), true, 16, 14, F);
        score_start_pending = false;
    end

    score_cache_out = score_cache_reg;
    fill_valid = false;

    if start
        token_idx = uint16(1);
        running = true;
        score_cache_reg(:) = fi(0, true, 16, 14, F);
        score_start_pending = true;
    end

    if ~running
        return;
    end

    key_vec = fi(key_mat(:, double(token_idx)), true, 16, 14, F);
    score_seed = fi(0, true, 32, 14);
    [score_now, score_valid] = qwen2_runtime.hdl.attention_row_controller_step(score_start_pending, query_vec, key_vec, score_seed, scale);
    score_start_pending = false;

    if score_valid
        score_cache_reg(double(token_idx)) = fi(score_now, true, 16, 14, F);
        if token_idx >= active_len
            running = false;
            fill_valid = true;
        else
            token_idx = token_idx + uint16(1);
            score_start_pending = true;
        end
    end

    score_cache_out = score_cache_reg;
end

function F = scoreCacheFimath()
    F = fimath( ...
        'RoundingMethod', 'Floor', ...
        'OverflowAction', 'Saturate', ...
        'ProductMode', 'SpecifyPrecision', ...
        'ProductWordLength', 24, ...
        'ProductFractionLength', 14, ...
        'SumMode', 'SpecifyPrecision', ...
        'SumWordLength', 16, ...
        'SumFractionLength', 14);
end