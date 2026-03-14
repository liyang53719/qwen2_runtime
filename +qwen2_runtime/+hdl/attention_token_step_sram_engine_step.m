function [attn_out, out_valid, busy, next_valid_len] = attention_token_step_sram_engine_step(start, q_token, k_token, v_token, key_cache_in, value_cache_in, cache_valid_len, rope_position, freqs_cis, ~, cfg)
%ATTENTION_TOKEN_STEP_SRAM_ENGINE_STEP Incremental token-step attention engine.

    headDim = 128;
    numHeads = 12;
    numKVHeads = 2;
    maxCacheLen = 256;
    F = attentionFimath(cfg);
    persistent running head_idx q_rot_reg key_cache_reg value_cache_reg next_len_reg attn_reg active_len_reg scale_reg head_start_pending
    if isempty(running)
        running = false;
        head_idx = uint8(1);
        q_rot_reg = fi(zeros(headDim, numHeads), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
        key_cache_reg = fi(zeros(headDim, numKVHeads, maxCacheLen, 1), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
        value_cache_reg = fi(zeros(headDim, numKVHeads, maxCacheLen, 1), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
        next_len_reg = uint16(0);
        attn_reg = fi(zeros(headDim * numHeads, 1), true, cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, F);
        active_len_reg = uint16(0);
        scale_reg = fi(headDimScale(headDim, cfg), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
        head_start_pending = false;
    end

    attn_out = attn_reg;
    out_valid = false;
    busy = running;
    next_valid_len = next_len_reg;

    if start
        q_fix = projectedTokenLike(q_token, cfg);
        k_fix = projectedTokenLike(k_token, cfg);
        v_fix = projectedTokenLike(v_token, cfg);
        [q_rot_reg, k_rot] = qwen2_runtime.hdl.attention_rope_single_token_step(q_fix, k_fix, rope_position, freqs_cis, cfg);

        [key_cache_reg, value_cache_reg, next_valid_len_raw] = qwen2_runtime.hdl.kv_cache_update_step( ...
            key_cache_in, value_cache_in, double(cache_valid_len), k_rot, v_fix);

        next_len_reg = cast(next_valid_len_raw, 'like', uint16(0));
        next_valid_len = next_len_reg;
        active_len_reg = uint16(min(double(next_valid_len_raw), maxCacheLen));
        attn_reg(:) = fi(0, true, cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, F);
        running = true;
        head_idx = uint8(1);
        scale_reg = fi(headDimScale(headDim, cfg), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
        head_start_pending = true;
    end

    if ~running
        return;
    end

    kv_head_idx = kvHeadIndex(head_idx, numHeads, numKVHeads);
    scale_fix = fi(scale_reg, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);

    query_vec = fi(q_rot_reg(:, double(head_idx)), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
    key_mat = cacheHeadSliceToMatrix(key_cache_reg, kv_head_idx, headDim, maxCacheLen, cfg);
    value_mat = cacheHeadSliceToMatrix(value_cache_reg, kv_head_idx, headDim, maxCacheLen, cfg);
    [head_acc, head_valid] = qwen2_runtime.hdl.attention_token_step_sram_single_head_step( ...
        head_start_pending, query_vec, key_mat, value_mat, active_len_reg, scale_fix, cfg);
    head_start_pending = false;
    if head_valid
        flat_idx = (double(head_idx) - 1) * headDim + 1;
        attn_reg(flat_idx:flat_idx + headDim - 1) = fi(head_acc, true, cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, F);
        if head_idx >= uint8(numHeads)
            attn_out = attn_reg;
            out_valid = true;
            running = false;
        else
            head_idx = head_idx + uint8(1);
            head_start_pending = true;
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

function matrix = cacheHeadSliceToMatrix(cacheTensor, headIndex, headDim, maxCacheLen, cfg)
    F = attentionFimath(cfg);
    matrix = fi(zeros(headDim, maxCacheLen), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
    for pos = 1:maxCacheLen
        for dim = 1:headDim
            matrix(dim, pos) = cacheTensor(dim, headIndex, pos, 1);
        end
    end
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
