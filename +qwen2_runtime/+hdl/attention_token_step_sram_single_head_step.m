function [head_acc_out, head_valid] = attention_token_step_sram_single_head_step(start, query_vec, key_mat, value_mat, active_len, scale_fix, cfg)
%ATTENTION_TOKEN_STEP_SRAM_SINGLE_HEAD_STEP Run all four attention phases for one head.

    maxCacheLen = size(key_mat, 2);
    Fmax = maxPathFimath();
    Fsum = sumPathFimath();
    persistent phase running max_reg sum_reg recip_reg score_cache_reg exp_cache_reg weight_cache_reg phase_start_pending
    if isempty(phase)
        phase = uint8(0);
        running = false;
        max_reg = fi(-8, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, Fmax);
        sum_reg = fi(0, true, cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, Fsum);
        recip_reg = fi(1, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, Fmax);
        score_cache_reg = fi(zeros(maxCacheLen, 1), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, Fmax);
        exp_cache_reg = fi(zeros(maxCacheLen, 1), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, Fmax);
        weight_cache_reg = fi(zeros(maxCacheLen, 1), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, Fmax);
        phase_start_pending = false;
    end

    head_acc_out = fi(zeros(size(value_mat, 1), 1), true, cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, attentionFimath(cfg));
    head_valid = false;

    if start
        phase = uint8(1);
        running = true;
        max_reg = fi(-8, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, Fmax);
        sum_reg = fi(0, true, cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, Fsum);
        recip_reg = fi(1, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, Fmax);
        score_cache_reg(:) = fi(0, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, Fmax);
        exp_cache_reg(:) = fi(0, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, Fmax);
        weight_cache_reg(:) = fi(0, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, Fmax);
        phase_start_pending = true;
    end

    if ~running
        return;
    end

    if phase == uint8(1)
        [score_cache_reg, fill_valid] = qwen2_runtime.hdl.attention_token_step_sram_score_cache_fill_step( ...
            phase_start_pending, query_vec, key_mat, active_len, scale_fix);
        phase_start_pending = false;
        if fill_valid
            phase = uint8(2);
            phase_start_pending = true;
        end
    elseif phase == uint8(2)
        max_seed = fi(-8, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, Fmax);
        [max_reg, max_valid] = qwen2_runtime.hdl.attention_token_step_sram_max_scan_step( ...
            phase_start_pending, score_cache_reg, active_len, max_seed);
        phase_start_pending = false;
        if max_valid
            phase = uint8(3);
            phase_start_pending = true;
        end
    elseif phase == uint8(3)
        sum_seed = fi(0, true, cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, Fsum);
        [sum_reg, recip_reg, exp_cache_reg, sum_valid] = qwen2_runtime.hdl.attention_token_step_sram_sum_scan_step( ...
            phase_start_pending, score_cache_reg, active_len, max_reg, sum_seed);
        phase_start_pending = false;
        if sum_valid
            weight_cache_reg = qwen2_runtime.hdl.attention_token_step_sram_weight_finalize_step(exp_cache_reg, recip_reg, active_len);
            phase = uint8(4);
            phase_start_pending = true;
        end
    else
        value_seed = fi(0, true, cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, valuePathFimath());
        [head_acc_out, head_done] = qwen2_runtime.hdl.attention_token_step_sram_head_value_scan_step( ...
            phase_start_pending, weight_cache_reg, value_mat, value_seed, cfg);
        phase_start_pending = false;
        if head_done
            running = false;
            phase = uint8(0);
            head_valid = true;
        end
    end
end

function F = attentionFimath(cfg)
    F = fimath( ...
        'RoundingMethod', 'Floor', ...
        'OverflowAction', 'Saturate', ...
        'ProductMode', 'SpecifyPrecision', ...
        'ProductWordLength', 24, ...
        'ProductFractionLength', cfg.HDLLinearAccumFractionLength, ...
        'SumMode', 'SpecifyPrecision', ...
        'SumWordLength', cfg.HDLLinearAccumWordLength, ...
        'SumFractionLength', cfg.HDLLinearAccumFractionLength);
end

function F = maxPathFimath()
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

function F = sumPathFimath()
    F = fimath( ...
        'RoundingMethod', 'Floor', ...
        'OverflowAction', 'Saturate', ...
        'ProductMode', 'SpecifyPrecision', ...
        'ProductWordLength', 24, ...
        'ProductFractionLength', 14, ...
        'SumMode', 'SpecifyPrecision', ...
        'SumWordLength', 32, ...
        'SumFractionLength', 14);
end

function F = valuePathFimath()
    F = fimath( ...
        'RoundingMethod', 'Floor', ...
        'OverflowAction', 'Saturate', ...
        'ProductMode', 'SpecifyPrecision', ...
        'ProductWordLength', 24, ...
        'ProductFractionLength', 14, ...
        'SumMode', 'SpecifyPrecision', ...
        'SumWordLength', 32, ...
        'SumFractionLength', 14);
end