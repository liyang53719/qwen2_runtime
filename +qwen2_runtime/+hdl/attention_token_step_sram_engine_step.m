function [attn_out, out_valid, busy, next_valid_len] = attention_token_step_sram_engine_step(start, q_token, k_token, v_token, key_cache_in, value_cache_in, cache_valid_len, rope_position, freqs_cis, ~, cfg)
%ATTENTION_TOKEN_STEP_SRAM_ENGINE_STEP Incremental token-step attention engine.

    headDim = 128;
    numHeads = 12;
    numKVHeads = 2;
    maxCacheLen = 256;
    F = attentionFimath(cfg);

    persistent phase running head_idx token_idx q_rot_reg score_mat_reg value_tensor_reg key_cache_reg value_cache_reg next_len_reg attn_reg valid_reg controller_start_pending active_len_reg scale_reg
    if isempty(phase)
        phase = uint8(0);
        running = false;
        head_idx = uint8(1);
        token_idx = uint16(1);
        q_rot_reg = fi(zeros(headDim, numHeads), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
        score_mat_reg = fi(zeros(maxCacheLen, numHeads), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
        value_tensor_reg = fi(zeros(maxCacheLen, headDim, numHeads), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
        key_cache_reg = fi(zeros(headDim, numKVHeads, maxCacheLen, 1), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
        value_cache_reg = fi(zeros(headDim, numKVHeads, maxCacheLen, 1), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
        next_len_reg = uint16(0);
        attn_reg = fi(zeros(headDim * numHeads, 1), true, cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, F);
        valid_reg = false;
        controller_start_pending = false;
        active_len_reg = uint16(0);
        scale_reg = fi(headDimScale(headDim, cfg), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
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
        score_mat_reg(:) = fi(-8, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
        value_tensor_reg(:) = fi(0, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
        attn_reg(:) = fi(0, true, cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, F);
        phase = uint8(1);
        running = true;
        head_idx = uint8(1);
        token_idx = uint16(1);
        controller_start_pending = false;
        scale_reg = fi(headDimScale(headDim, cfg), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
    end

    if ~running
        return;
    end

    if phase == uint8(1)
        if active_len_reg == uint16(0)
            phase = uint8(2);
            controller_start_pending = true;
        else
            kv_head_idx = kvHeadIndex(head_idx, numHeads, numKVHeads);
            scale_fix = fi(scale_reg, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
            query_vec = fi(q_rot_reg(:, double(head_idx)), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
            key_vec = fi(key_cache_reg(:, kv_head_idx, double(token_idx), 1), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
            value_vec = fi(value_cache_reg(:, kv_head_idx, double(token_idx), 1), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
            score_now = qwen2_runtime.hdl.attention_score_step(query_vec, key_vec, scale_fix);
            score_mat_reg(double(token_idx), double(head_idx)) = fi(score_now, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
            value_tensor_reg(double(token_idx), :, double(head_idx)) = reshape(value_vec, [1, headDim]);

            if token_idx >= active_len_reg
                if head_idx >= uint8(numHeads)
                    phase = uint8(2);
                    controller_start_pending = true;
                else
                    head_idx = head_idx + uint8(1);
                    token_idx = uint16(1);
                end
            else
                token_idx = token_idx + uint16(1);
            end
        end
    else
        max_seed = fi(-8, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
        sum_seed = fi(0, true, cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, F);
        [attn_candidate, controller_valid] = qwen2_runtime.hdl.attention_multihead_controller_step( ...
            controller_start_pending, score_mat_reg, value_tensor_reg, max_seed, sum_seed);
        controller_start_pending = false;
        if controller_valid
            attn_reg = reshape(fi(attn_candidate, true, cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, F), [headDim * numHeads, 1]);
            attn_out = attn_reg;
            out_valid = true;
            running = false;
            phase = uint8(0);
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