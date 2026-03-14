function weight_vec = attention_token_step_sram_weight_finalize_step(exp_vec, recip_reg, active_len)
%ATTENTION_TOKEN_STEP_SRAM_WEIGHT_FINALIZE_STEP Finalize normalized weights for the value phase.

    F = fimath(recip_reg);
    maxCacheLen = size(exp_vec, 1);
    weight_vec = fi(zeros(maxCacheLen, 1), true, 16, 14, F);
    recip_fix = fi(recip_reg, true, 16, 14, F);
    active_len_u16 = uint16(active_len);
    for idx = 1:maxCacheLen
        if uint16(idx) <= active_len_u16
            exp_fix = fi(exp_vec(idx), true, 16, 14, F);
            weight_vec(idx) = qwen2_runtime.hdl.softmax_normalize_step(exp_fix, recip_fix);
        else
            weight_vec(idx) = fi(0, true, 16, 14, F);
        end
    end
end