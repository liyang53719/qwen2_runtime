function [sum_out, recip_out, exp_out, score_valid, sum_valid] = attention_token_step_sram_sum_phase_step(score_start, query_vec, key_vec, scale, max_reg, token_first, token_last, sum_seed)
%ATTENTION_TOKEN_STEP_SRAM_SUM_PHASE_STEP Stream one score row into the exp/sum phase.

    score_seed = fi(0, true, 32, 14);
    [score_now, score_valid] = qwen2_runtime.hdl.attention_row_controller_step(score_start, query_vec, key_vec, score_seed, scale);
    Fsum = fimath(sum_seed);
    Fmax = fimath(max_reg);
    sum_out = fi(sum_seed, true, 32, 14, Fsum);
    recip_out = fi(1, true, 16, 14, Fmax);
    exp_out = fi(0, true, 16, 14, Fmax);
    sum_valid = false;
    if score_valid
        score_now_max = fi(score_now, true, 16, 14, Fmax);
        exp_val = qwen2_runtime.hdl.softmax_exp_step(score_now_max, max_reg);
        exp_out = fi(exp_val, true, 16, 14, Fmax);
        [sum_out, sum_valid] = qwen2_runtime.hdl.softmax_sum_step(token_first, exp_val, sum_seed, token_last);
        if sum_valid
            denom_safe = fi(sum_out, true, 32, 14, Fsum);
            if denom_safe == fi(0, true, 32, 14, Fsum)
                denom_safe = fi(1, true, 32, 14, Fsum);
            end
            recip_out = qwen2_runtime.hdl.softmax_recip_lookup_step(fi(denom_safe, true, 16, 14, Fmax));
        end
    end
end