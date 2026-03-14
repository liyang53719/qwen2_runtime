function [sum_out, recip_out, exp_cache_out, scan_valid] = attention_token_step_sram_sum_scan_step(start, score_cache_in, active_len, max_reg, sum_seed)
%ATTENTION_TOKEN_STEP_SRAM_SUM_SCAN_STEP Scan a fixed-length score cache for exp/sum and cache exp values.

    Fsum = fimath(sum_seed);
    Fmax = fimath(max_reg);
    maxCacheLen = uint16(length(score_cache_in));
    persistent token_idx running sum_reg exp_cache_reg recip_reg
    if isempty(token_idx)
        token_idx = uint16(1);
        running = false;
        sum_reg = fi(sum_seed, true, 32, 14, Fsum);
        exp_cache_reg = fi(zeros(double(maxCacheLen), 1), true, 16, 14, Fmax);
        recip_reg = fi(1, true, 16, 14, Fmax);
    end

    sum_out = sum_reg;
    recip_out = recip_reg;
    exp_cache_out = exp_cache_reg;
    scan_valid = false;

    if start
        token_idx = uint16(1);
        running = true;
        sum_reg = fi(sum_seed, true, 32, 14, Fsum);
        exp_cache_reg(:) = fi(0, true, 16, 14, Fmax);
        recip_reg = fi(1, true, 16, 14, Fmax);
    end

    if ~running
        return;
    end

    token_last = (token_idx >= active_len);
    score_now = fi(score_cache_in(double(token_idx)), true, 16, 14, Fmax);
    exp_now = qwen2_runtime.hdl.softmax_exp_step(score_now, max_reg);
    exp_cache_reg(double(token_idx)) = fi(exp_now, true, 16, 14, Fmax);
    [sum_reg, scan_valid] = qwen2_runtime.hdl.softmax_sum_step(token_idx == uint16(1), exp_now, sum_seed, token_last);
    sum_out = sum_reg;

    if scan_valid
        denom_safe = fi(sum_reg, true, 32, 14, Fsum);
        if denom_safe == fi(0, true, 32, 14, Fsum)
            denom_safe = fi(1, true, 32, 14, Fsum);
        end
        recip_reg = qwen2_runtime.hdl.softmax_recip_lookup_step(fi(denom_safe, true, 16, 14, Fmax));
        recip_out = recip_reg;
        exp_cache_out = exp_cache_reg;
        running = false;
    else
        token_idx = token_idx + uint16(1);
    end
end