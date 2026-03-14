function [attn_out, key_cache_out, value_cache_out, next_valid_len] = attention_token_step_sram_step(q_token, k_token, v_token, key_cache_in, value_cache_in, cache_valid_len, rope_position, freqs_cis, hyperParameters, cfg)
%ATTENTION_TOKEN_STEP_SRAM_STEP Token-step attention core with external KV SRAM state.

    headDim = coder.const(double(hyperParameters.HeadDim));
    numHeads = coder.const(double(hyperParameters.NumHeads));
    numKVHeads = coder.const(double(hyperParameters.NumKVHeads));

    q_token = reshape(projectedTokenLike(q_token, cfg), [headDim, numHeads]);
    k_token = reshape(projectedTokenLike(k_token, cfg), [headDim, numKVHeads]);
    v_token = reshape(projectedTokenLike(v_token, cfg), [headDim, numKVHeads]);

    [q_rot, k_rot] = fixedPointRoPESinglePos(q_token, k_token, rope_position, freqs_cis, cfg);

    k_token_4d = reshape(k_rot, [headDim, numKVHeads, 1, 1]);
    v_token_4d = reshape(v_token, [headDim, numKVHeads, 1, 1]);
    [key_cache_out, value_cache_out, next_valid_len_raw] = qwen2_runtime.hdl.kv_cache_update_step( ...
        key_cache_in, value_cache_in, double(cache_valid_len), k_token_4d, v_token_4d);
    next_valid_len = cast(next_valid_len_raw, 'like', cache_valid_len);

    repeatedKeys = repeatKVHeads(key_cache_out, numHeads, numKVHeads);
    repeatedValues = repeatKVHeads(value_cache_out, numHeads, numKVHeads);
    scale = headDimScale(headDim, cfg);

    attn_heads = fixedPointAttentionMultihead(q_rot, repeatedKeys(:, :, :, 1), repeatedValues(:, :, :, 1), double(next_valid_len_raw), scale, cfg);
    attn_out = reshape(attn_heads, [headDim * numHeads, 1]);
end

function token = projectedTokenLike(tokenIn, cfg)
    F = attentionFimath(cfg);
    token = fi(tokenIn, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
end

function [xq_rot, xk_rot] = fixedPointRoPESinglePos(xq, xk, rope_position, freqs_cis, cfg)
    half = size(xq, 1) / 2;
    if mod(size(xq, 1), 2) ~= 0
        error('RoPE:InvalidHeadDim', 'headDim must be even.');
    end

    F = attentionFimath(cfg);
    pos = double(rope_position);
    if pos < 1
        error('RoPE:InvalidPosition', 'rope_position must be >= 1.');
    end

    cosTheta = fi(reshape(freqs_cis.Cos(:, pos), [half, 1]), true, xq.WordLength, xq.FractionLength, F);
    sinTheta = fi(reshape(freqs_cis.Sin(:, pos), [half, 1]), true, xq.WordLength, xq.FractionLength, F);

    xq_rot = rotateTokenPair(xq, cosTheta, sinTheta, F);
    xk_rot = rotateTokenPair(xk, cosTheta, sinTheta, F);
end

function x_rot = rotateTokenPair(x, cosTheta, sinTheta, F)
    half = size(x, 1) / 2;
    headCount = size(x, 2);
    x_rot = x;
    for h = 1:headCount
        realPart = x(1:half, h);
        imagPart = x(half+1:end, h);
        x_rot(1:half, h) = fi(realPart .* cosTheta - imagPart .* sinTheta, true, x.WordLength, x.FractionLength, F);
        x_rot(half+1:end, h) = fi(realPart .* sinTheta + imagPart .* cosTheta, true, x.WordLength, x.FractionLength, F);
    end
end

function out = fixedPointAttentionMultihead(query_heads, key_bank, value_bank, totalLen, scale, cfg)
    headDim = size(query_heads, 1);
    numHeads = size(query_heads, 2);
    maxCacheLen = coder.const(size(key_bank, 3));
    activeLen = min(totalLen, maxCacheLen);
    F = attentionFimath(cfg);
    out = fi(zeros(headDim, numHeads), true, cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, F);
    if activeLen <= 0
        return;
    end

    score_mat = fi(zeros(maxCacheLen, numHeads), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
    value_tensor = fi(zeros(maxCacheLen, headDim, numHeads), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
    score_mat(:) = fi(-8, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
    scale_fix = fi(scale, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);

    for h = 1:numHeads
        query_vec = fi(query_heads(:, h), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
        for t = 1:maxCacheLen
            if t <= activeLen
                key_vec = fi(key_bank(:, h, t), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
                value_vec = fi(value_bank(:, h, t), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
                score_mat(t, h) = fi(qwen2_runtime.hdl.attention_score_step(query_vec, key_vec, scale_fix), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
                value_tensor(t, :, h) = reshape(value_vec, [1, headDim]);
            end
        end
    end

    max_seed = fi(-8, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
    sum_seed = fi(0, true, cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, F);
    totalCycles = maxCacheLen * (3 * headDim * numHeads) + 16;
    done = false;
    for cyc = 1:totalCycles
        if ~done
            [attn_candidate, out_valid] = qwen2_runtime.hdl.attention_multihead_controller_step( ...
                cyc == 1, score_mat, value_tensor, max_seed, sum_seed);
            if out_valid
                out = attn_candidate;
                done = true;
            end
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

function Y = repeatKVHeads(X, numHeads, numKVHeads)
    nRep = numHeads / numKVHeads;
    if nRep == 1
        Y = X;
        return;
    end

    [headDim, ~, seqLen, batchSize] = size(X);
    Y = zeros(headDim, numHeads, seqLen, batchSize, 'like', X);
    for kv = 1:numKVHeads
        base = (kv - 1) * nRep;
        for rep = 1:nRep
            Y(:, base + rep, :, :) = X(:, kv, :, :);
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