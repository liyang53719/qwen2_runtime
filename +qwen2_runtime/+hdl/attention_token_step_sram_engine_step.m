function [attn_out, out_valid, busy, next_valid_len] = attention_token_step_sram_engine_step(start, q_token, k_token, v_token, key_cache_in, value_cache_in, cache_valid_len, rope_position, freqs_cis, ~, cfg)
%ATTENTION_TOKEN_STEP_SRAM_ENGINE_STEP Incremental token-step attention engine.

    headDim = 128;
    numHeads = 12;
    numKVHeads = 2;
    maxCacheLen = 256;
    F = attentionFimath(cfg);
    Fmax = maxPathFimath();
    Fsum = sumPathFimath();
    Fval = valuePathFimath();

    persistent phase running head_idx lane_idx token_idx q_rot_reg key_cache_reg value_cache_reg next_len_reg attn_reg valid_reg active_len_reg scale_reg max_reg sum_reg recip_reg
    if isempty(phase)
        phase = uint8(0);
        running = false;
        head_idx = uint8(1);
        lane_idx = uint16(1);
        token_idx = uint16(1);
        q_rot_reg = fi(zeros(headDim, numHeads), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
        key_cache_reg = fi(zeros(headDim, numKVHeads, maxCacheLen, 1), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
        value_cache_reg = fi(zeros(headDim, numKVHeads, maxCacheLen, 1), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
        next_len_reg = uint16(0);
        attn_reg = fi(zeros(headDim * numHeads, 1), true, cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, F);
        valid_reg = false;
        active_len_reg = uint16(0);
        scale_reg = fi(headDimScale(headDim, cfg), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
        max_reg = fi(-8, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, Fmax);
        sum_reg = fi(0, true, cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, Fsum);
        recip_reg = fi(1, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, Fmax);
    end

    attn_out = attn_reg;
    out_valid = valid_reg;
    busy = running;
    next_valid_len = next_len_reg;
    valid_reg = false;

    if start
        q_fix = reshape(projectedTokenLike(q_token, cfg), [headDim, numHeads]);
        k_fix = reshape(projectedTokenLike(k_token, cfg), [headDim, numKVHeads]);
        v_fix = reshape(projectedTokenLike(v_token, cfg), [headDim, numKVHeads]);
        [q_rot_reg, k_rot] = qwen2_runtime.hdl.attention_rope_single_token_step(q_fix, k_fix, rope_position, freqs_cis, cfg);

        k_token_4d = reshape(k_rot, [headDim, numKVHeads, 1, 1]);
        v_token_4d = reshape(v_fix, [headDim, numKVHeads, 1, 1]);
        [key_cache_reg, value_cache_reg, next_valid_len_raw] = qwen2_runtime.hdl.kv_cache_update_step( ...
            key_cache_in, value_cache_in, double(cache_valid_len), k_token_4d, v_token_4d);

        next_len_reg = cast(next_valid_len_raw, 'like', uint16(0));
        next_valid_len = next_len_reg;
        active_len_reg = uint16(min(double(next_valid_len_raw), maxCacheLen));
        attn_reg(:) = fi(0, true, cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, F);
        phase = uint8(1);
        running = true;
        head_idx = uint8(1);
        lane_idx = uint16(1);
        token_idx = uint16(1);
        scale_reg = fi(headDimScale(headDim, cfg), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
        max_reg = fi(-8, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, Fmax);
        sum_reg = fi(0, true, cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, Fsum);
        recip_reg = fi(1, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, Fmax);
    end

    if ~running
        return;
    end

    kv_head_idx = kvHeadIndex(head_idx, numHeads, numKVHeads);
    scale_fix = fi(scale_reg, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
    query_vec = fi(q_rot_reg(:, double(head_idx)), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
    key_vec = fi(key_cache_reg(:, kv_head_idx, double(token_idx), 1), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
    score_now = qwen2_runtime.hdl.attention_score_step(query_vec, key_vec, scale_fix);
    score_now_max = fi(score_now, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, Fmax);
    token_last = (token_idx >= active_len_reg);

    if phase == uint8(1)
        max_seed = fi(-8, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, Fmax);
        [max_reg, max_valid] = qwen2_runtime.hdl.softmax_max_step(token_idx == uint16(1), score_now_max, max_seed, token_last);
        if max_valid
            phase = uint8(2);
            token_idx = uint16(1);
            sum_reg = fi(0, true, cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, Fsum);
        else
            token_idx = token_idx + uint16(1);
        end
    elseif phase == uint8(2)
        exp_val = qwen2_runtime.hdl.softmax_exp_step(score_now_max, max_reg);
        sum_seed = fi(0, true, cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, Fsum);
        [sum_reg, sum_valid] = qwen2_runtime.hdl.softmax_sum_step(token_idx == uint16(1), exp_val, sum_seed, token_last);
        if sum_valid
            denom_safe = fi(sum_reg, true, cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, Fsum);
            if denom_safe == fi(0, true, cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, Fsum)
                denom_safe = fi(1, true, cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, Fsum);
            end
            recip_reg = qwen2_runtime.hdl.softmax_recip_lookup_step(fi(denom_safe, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, Fmax));
            phase = uint8(3);
            lane_idx = uint16(1);
            token_idx = uint16(1);
        else
            token_idx = token_idx + uint16(1);
        end
    else
        exp_val = qwen2_runtime.hdl.softmax_exp_step(score_now_max, max_reg);
        weight_val = qwen2_runtime.hdl.softmax_normalize_step(fi(exp_val, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, Fmax), fi(recip_reg, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, Fmax));
        value_val = fi(value_cache_reg(double(lane_idx), kv_head_idx, double(token_idx), 1), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, Fval);
        value_seed = fi(0, true, cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, Fval);
        [lane_acc, lane_valid] = qwen2_runtime.hdl.attention_value_mac_step(token_idx == uint16(1), weight_val, value_val, value_seed, token_last);
        if lane_valid
            flat_idx = (double(head_idx) - 1) * headDim + double(lane_idx);
            attn_reg(flat_idx) = fi(lane_acc, true, cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, F);
            if lane_idx >= uint16(headDim)
                if head_idx >= uint8(numHeads)
                    attn_out = attn_reg;
                    out_valid = true;
                    running = false;
                    phase = uint8(0);
                else
                    head_idx = head_idx + uint8(1);
                    lane_idx = uint16(1);
                    token_idx = uint16(1);
                    phase = uint8(1);
                    max_reg = fi(-8, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, Fmax);
                    sum_reg = fi(0, true, cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, Fsum);
                end
            else
                lane_idx = lane_idx + uint16(1);
                token_idx = uint16(1);
            end
        else
            token_idx = token_idx + uint16(1);
        end
    end

    busy = running;
    next_valid_len = next_len_reg;
end

function idx = kvHeadIndex(headIdx, numHeads, numKVHeads)
    coder.inline('always');

    if numHeads == 12 && numKVHeads == 2
        if headIdx <= uint8(6)
            idx = 1;
        else
            idx = 2;
        end
    else
        headsPerKV = uint8(numHeads / numKVHeads);
        if headIdx <= headsPerKV
            idx = 1;
        else
            idx = numKVHeads;
        end
    end
end

function token = projectedTokenLike(tokenIn, cfg)
    F = attentionFimath(cfg);
    token = fi(tokenIn, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
end

function scale = headDimScale(headDim, cfg)
    if isfield(cfg, 'EnableHDLMathSafeMode') && logical(cfg.EnableHDLMathSafeMode)
        switch double(headDim)
            case 128
                scale = single(0.0883883476483184);
            case 64
                scale = single(0.125);
            otherwise
                scale = single(1.0 / sqrt(double(headDim)));
        end
    else
        scale = single(1.0 / sqrt(double(headDim)));
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