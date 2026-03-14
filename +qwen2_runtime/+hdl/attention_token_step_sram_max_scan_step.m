function [max_out, scan_valid] = attention_token_step_sram_max_scan_step(start, score_cache_in, active_len, max_seed)
%ATTENTION_TOKEN_STEP_SRAM_MAX_SCAN_STEP Scan a fixed-length score cache to find the row max.

    F = fimath(max_seed);
    persistent token_idx running max_reg
    if isempty(token_idx)
        token_idx = uint16(1);
        running = false;
        max_reg = fi(max_seed, true, 16, 14, F);
    end

    max_out = max_reg;
    scan_valid = false;

    if start
        token_idx = uint16(1);
        running = true;
        max_reg = fi(max_seed, true, 16, 14, F);
    end

    if ~running
        return;
    end

    token_last = (token_idx >= active_len);
    score_now = fi(score_cache_in(double(token_idx)), true, 16, 14, F);
    [max_reg, scan_valid] = qwen2_runtime.hdl.softmax_max_step(token_idx == uint16(1), score_now, max_seed, token_last);
    max_out = max_reg;
    if scan_valid
        running = false;
    else
        token_idx = token_idx + uint16(1);
    end
end