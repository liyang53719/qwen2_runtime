function [lane_acc_out, score_valid, lane_valid] = attention_token_step_sram_value_phase_step(score_start, query_vec, key_vec, scale, max_reg, recip_reg, value_val, token_first, token_last, value_seed)
%ATTENTION_TOKEN_STEP_SRAM_VALUE_PHASE_STEP Stream one score row into the weighted-value phase.

    score_seed = fi(0, true, 32, 14);
    [score_now, score_valid] = qwen2_runtime.hdl.attention_row_controller_step(score_start, query_vec, key_vec, score_seed, scale);
    Fval = fimath(value_seed);
    Fmax = fimath(max_reg);
    lane_acc_out = fi(value_seed, true, 32, 14, Fval);
    lane_valid = false;
    if score_valid
        score_now_max = fi(score_now, true, 16, 14, Fmax);
        exp_val = qwen2_runtime.hdl.softmax_exp_step(score_now_max, max_reg);
        weight_val = qwen2_runtime.hdl.softmax_normalize_step(fi(exp_val, true, 16, 14, Fmax), fi(recip_reg, true, 16, 14, Fmax));
        [lane_acc_out, lane_valid] = qwen2_runtime.hdl.attention_value_mac_step(token_first, weight_val, value_val, value_seed, token_last);
    end
end