function [max_out, score_valid, max_valid] = attention_token_step_sram_max_phase_step(score_start, query_vec, key_vec, scale, token_first, token_last, max_seed)
%ATTENTION_TOKEN_STEP_SRAM_MAX_PHASE_STEP Stream one score row into the max phase.

    score_seed = fi(0, true, 32, 14);
    [score_now, score_valid] = qwen2_runtime.hdl.attention_row_controller_step(score_start, query_vec, key_vec, score_seed, scale);
    max_out = fi(max_seed, true, 16, 14);
    max_valid = false;
    if score_valid
        score_now_max = fi(score_now, true, 16, 14);
        [max_out, max_valid] = qwen2_runtime.hdl.softmax_max_step(token_first, score_now_max, max_seed, token_last);
    end
end