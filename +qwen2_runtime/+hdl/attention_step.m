function [X_out, key_cache_out, value_cache_out] = attention_step(X, key_cache_in, value_cache_in, cache_valid_len, weights, hyperParameters, freqs_cis, cfg)
%ATTENTION_STEP HDL-friendly attention with fixed-size KV cache.

    [hiddenSize, seqLen, batchSize] = size(X);
    numHeads = hyperParameters.NumHeads;
    numKVHeads = hyperParameters.NumKVHeads;
    headDim = hyperParameters.HeadDim;
    useFixedCore = useFixedPointAttentionCore(cfg);

    X2 = reshape(X, hiddenSize, []);
    xq_proj = qwen2_runtime.hdl.linear_step(weights.q_proj, X2, cfg);
    xk_proj = qwen2_runtime.hdl.linear_step(weights.k_proj, X2, cfg);
    xv_proj = qwen2_runtime.hdl.linear_step(weights.v_proj, X2, cfg);

    xq = xq_proj;
    xk = xk_proj;
    xv = xv_proj;

    xq = reshape(xq, [headDim, numHeads, seqLen, batchSize]);
    xk = reshape(xk, [headDim, numKVHeads, seqLen, batchSize]);
    xv = reshape(xv, [headDim, numKVHeads, seqLen, batchSize]);

    xq = applyBiasLike(xq, reshape(weights.q_bias, size(xq)));
    xk = applyBiasLike(xk, reshape(weights.k_bias, size(xk)));
    xv = applyBiasLike(xv, reshape(weights.v_bias, size(xv)));

    if useFixedCore
        [xq, xk] = fixedPointRoPE(xq, xk, freqs_cis);
    else
        [xq, xk] = transformer.layer.RoPE(xq, xk, freqs_cis);
    end

    [key_cache_out, value_cache_out, totalLen] = updateKVCache(key_cache_in, value_cache_in, cache_valid_len, xk, xv);
    repeatedKeys = repeatKVHeads(key_cache_out, numHeads, numKVHeads);
    repeatedValues = repeatKVHeads(value_cache_out, numHeads, numKVHeads);

    if useFixedCore
        attn_output = fi(zeros(headDim, numHeads, seqLen, batchSize), true, ...
            cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, attentionFimath(cfg));
    else
        attn_output = zeros(headDim, numHeads, seqLen, batchSize, 'single');
    end
    scale = headDimScale(headDim, cfg);
    maxCacheLen = size(repeatedKeys, 3);

    for b = 1:batchSize
        if useFixedCore
            attn_output(:, :, 1, b) = fixedPointAttentionMultihead( ...
                xq(:, :, 1, b), repeatedKeys(:, :, :, b), repeatedValues(:, :, :, b), totalLen, scale);
        else
            for h = 1:numHeads
                queryVec = xq(:, h, 1, b);
                attn_output(:, h, 1, b) = streamingAttentionHead(queryVec, repeatedKeys(:, h, :, b), repeatedValues(:, h, :, b), totalLen, maxCacheLen, scale, cfg);
            end
        end
    end

    attn_output_cat = reshape(attn_output, headDim * numHeads, seqLen * batchSize);
    X_out = qwen2_runtime.hdl.linear_step(weights.o_proj, attn_output_cat, cfg);
    X_out = reshape(X_out, hiddenSize, seqLen, batchSize);
    X_out = applyBiasLike(X_out, reshape(weights.o_bias, size(X_out)));
end

function out = fixedPointAttentionMultihead(query_heads, key_bank, value_bank, totalLen, scale)
    headDim = size(query_heads, 1);
    numHeads = size(query_heads, 2);
    maxCacheLen = size(key_bank, 3);
    activeLen = min(totalLen, maxCacheLen);
    F = localFimath(query_heads);
    out = fi(zeros(headDim, numHeads), true, 32, 14, F);
    if activeLen <= 0
        return;
    end

    score_pad = fi(-8, true, 16, 14, F);
    score_mat = fi(zeros(maxCacheLen, numHeads), true, 16, 14, F);
    value_tensor = fi(zeros(maxCacheLen, headDim, numHeads), true, 16, 14, F);
    score_mat(:) = score_pad;
    scale_fix = fi(scale, true, 16, 14, F);
    for h = 1:numHeads
        query_vec = fi(query_heads(:, h), true, 16, 14, F);
        for t = 1:maxCacheLen
            if t <= activeLen
                key_vec = fi(key_bank(:, h, t), true, 16, 14, F);
                value_vec = fi(value_bank(:, h, t), true, 16, 14, F);
                score_mat(t, h) = fi(qwen2_runtime.hdl.attention_score_step(query_vec, key_vec, scale_fix), true, 16, 14, F);
                value_tensor(t, :, h) = reshape(value_vec, 1, headDim);
            end
        end
    end

    max_seed = fi(-8, true, 16, 14, F);
    sum_seed = fi(0, true, 32, 14, F);
    totalCycles = maxCacheLen * (3 * headDim * numHeads) + 16;
    attn_fix = fi(zeros(headDim, numHeads), true, 32, 14, F);
    done = false;
    for cyc = 1:totalCycles
        if ~done
            [attn_candidate, out_valid] = qwen2_runtime.hdl.attention_multihead_controller_step( ...
                cyc == 1, score_mat, value_tensor, max_seed, sum_seed);
            if out_valid
                attn_fix = attn_candidate;
                done = true;
            end
        end
    end
    out = attn_fix;
end

function [xq_rot, xk_rot] = fixedPointRoPE(xq, xk, freqs_cis)
    headDim = size(xq, 1);
    if mod(headDim, 2) ~= 0
        error('RoPE:InvalidHeadDim', 'headDim must be even.');
    end

    half = headDim / 2;
    seqLen = size(xq, 3);
    F = localFimath(xq);

    [cosTheta, sinTheta] = unpackRoPETables(freqs_cis, half, seqLen, F, xq.WordLength, xq.FractionLength);

    xq_rot = xq;
    xk_rot = xk;

    xq_r = xq(1:half, :, :, :);
    xq_i = xq(half+1:end, :, :, :);
    xk_r = xk(1:half, :, :, :);
    xk_i = xk(half+1:end, :, :, :);

    xq_rot(1:half, :, :, :) = fi(xq_r .* cosTheta - xq_i .* sinTheta, true, xq.WordLength, xq.FractionLength, F);
    xq_rot(half+1:end, :, :, :) = fi(xq_r .* sinTheta + xq_i .* cosTheta, true, xq.WordLength, xq.FractionLength, F);

    xk_rot(1:half, :, :, :) = fi(xk_r .* cosTheta - xk_i .* sinTheta, true, xk.WordLength, xk.FractionLength, F);
    xk_rot(half+1:end, :, :, :) = fi(xk_r .* sinTheta + xk_i .* cosTheta, true, xk.WordLength, xk.FractionLength, F);
end

function [cosTheta, sinTheta] = unpackRoPETables(freqs_cis, half, seqLen, F, wordLength, fractionLength)
    if isstruct(freqs_cis)
        cosTheta = reshape(freqs_cis.Cos(:, 1:seqLen), [half, 1, seqLen, 1]);
        sinTheta = reshape(freqs_cis.Sin(:, 1:seqLen), [half, 1, seqLen, 1]);
        return;
    end

    cosTheta = fi(reshape(real(freqs_cis(:, 1:seqLen)), [half, 1, seqLen, 1]), true, wordLength, fractionLength, F);
    sinTheta = fi(reshape(imag(freqs_cis(:, 1:seqLen)), [half, 1, seqLen, 1]), true, wordLength, fractionLength, F);
end

function out = streamingAttentionHead(queryVec, keyBank, valueBank, totalLen, maxCacheLen, scale, cfg)
    headDim = size(queryVec, 1);
    maxScore = cfg.HDLSoftmaxNegInit;
    for t = 1:maxCacheLen
        if t <= totalLen
            keyVec = keyBank(:, 1, t, 1);
            score = dotProduct(queryVec, keyVec) * scale;
            if score > maxScore
                maxScore = score;
            end
        end
    end

    denom = single(0);
    out = zeros(headDim, 1, 'single');
    for t = 1:maxCacheLen
        if t <= totalLen
            keyVec = keyBank(:, 1, t, 1);
            valueVec = valueBank(:, 1, t, 1);
            score = dotProduct(queryVec, keyVec) * scale;
            weight = safeExp(score - maxScore, cfg);
            denom = denom + weight;
            out = out + valueVec * weight;
        end
    end

    if denom < cfg.HDLMinDenominator
        denom = cfg.HDLMinDenominator;
    end
    out = out / denom;
end

function scale = headDimScale(headDim, cfg)
    if isfield(cfg, 'EnableHDLMathSafeMode') && logical(cfg.EnableHDLMathSafeMode)
        scale = reciprocalSqrtInteger(headDim);
    else
        scale = single(1.0 / sqrt(double(headDim)));
    end
end

function y = safeExp(x, cfg)
    if isfield(cfg, 'EnableHDLMathSafeMode') && logical(cfg.EnableHDLMathSafeMode)
        y = approxExpNeg(clampLower(x, cfg.HDLExpNegLimit));
        return;
    end
    y = exp(x);
end

function y = reciprocalSqrtInteger(headDim)
    switch double(headDim)
        case 128
            y = single(0.0883883476483184);
        case 64
            y = single(0.125);
        otherwise
            y = single(1.0 / sqrt(double(headDim)));
    end
end

function value = dotProduct(a, b)
    numelA = size(a, 1);
    value = single(0);
    for i = 1:numelA
        value = value + a(i) * b(i);
    end
end

function y = approxExpNeg(x)
    if x <= single(-8.0)
        y = single(0);
    elseif x <= single(-4.0)
        y = linearInterp(x, single(-8.0), single(-4.0), single(0.00033546), single(0.01831564));
    elseif x <= single(-2.0)
        y = linearInterp(x, single(-4.0), single(-2.0), single(0.01831564), single(0.13533528));
    elseif x <= single(-1.0)
        y = linearInterp(x, single(-2.0), single(-1.0), single(0.13533528), single(0.36787945));
    elseif x <= single(0.0)
        y = linearInterp(x, single(-1.0), single(0.0), single(0.36787945), single(1.0));
    else
        y = single(1.0) + x;
    end
end

function y = linearInterp(x, x0, x1, y0, y1)
    slope = (y1 - y0) / (x1 - x0);
    y = y0 + (x - x0) * slope;
end

function y = clampLower(x, lowerBound)
    y = x;
    if y < lowerBound
        y = lowerBound;
    end
end

function tf = useFixedPointAttentionCore(cfg)
    tf = false;
    if ~isstruct(cfg)
        return;
    end
    if isfield(cfg, 'UseFixedPointHDL')
        tf = logical(cfg.UseFixedPointHDL);
        return;
    end
    if isfield(cfg, 'HDLNumericMode')
        tf = isequal(cfg.HDLNumericMode, 'fixed');
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

function F = localFimath(value)
    if isa(value, 'embedded.fi')
        F = fimath(value);
    else
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
end

function Y = applyBiasLike(X, bias)
    if isa(X, 'embedded.fi')
        Y = X + fi(bias, true, X.WordLength, X.FractionLength, fimath(X));
    else
        Y = X + cast(bias, 'like', X);
    end
end

function [key_cache_out, value_cache_out, totalLen] = updateKVCache(key_cache_in, value_cache_in, cache_valid_len, xk, xv)
    key_cache_out = key_cache_in;
    value_cache_out = value_cache_in;
    maxLen = size(key_cache_in, 3);
    totalLen = cache_valid_len + size(xk, 3);
    if totalLen < 0
        totalLen = 0;
    end
    if totalLen > maxLen
        totalLen = maxLen;
    end

    if cache_valid_len < maxLen
        insertIdx = cache_valid_len + 1;
        key_cache_out(:, :, insertIdx, :) = xk(:, :, 1, :);
        value_cache_out(:, :, insertIdx, :) = xv(:, :, 1, :);
    else
        key_cache_out(:, :, 1:end-1, :) = key_cache_in(:, :, 2:end, :);
        value_cache_out(:, :, 1:end-1, :) = value_cache_in(:, :, 2:end, :);
        key_cache_out(:, :, end, :) = xk(:, :, 1, :);
        value_cache_out(:, :, end, :) = xv(:, :, 1, :);
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
