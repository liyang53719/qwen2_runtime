function [attn_out, key_cache_out, value_cache_out, next_valid_len, read_enable, read_addr, write_enable, write_addr, shift_enable, write_key_token, write_value_token] = attention_token_step_sram_contract_step(q_token, k_token, v_token, key_cache_in, value_cache_in, cache_valid_len, rope_position, freqs_cis, hyperParameters, cfg)
%ATTENTION_TOKEN_STEP_SRAM_CONTRACT_STEP Token-step attention with explicit KV SRAM contract signals.

    headDim = hyperParameters.HeadDim;
    numKVHeads = hyperParameters.NumKVHeads;
    maxCacheLen = size(key_cache_in, 3);
    historyLen = min(double(cache_valid_len), maxCacheLen);

    read_enable = false(maxCacheLen, 1);
    read_addr = uint16(zeros(maxCacheLen, 1));
    for idx = 1:historyLen
        read_enable(idx) = true;
        read_addr(idx) = uint16(idx);
    end

    shift_enable = double(cache_valid_len) >= maxCacheLen;
    write_enable = true;
    if shift_enable
        write_addr = uint16(maxCacheLen);
    else
        write_addr = uint16(double(cache_valid_len) + 1);
    end

    [~, k_rot] = fixedPointRoPESinglePos(q_token, k_token, rope_position, freqs_cis, cfg);
    write_key_token = reshape(projectedTokenLike(k_rot, cfg), [headDim, numKVHeads]);
    write_value_token = reshape(projectedTokenLike(v_token, cfg), [headDim, numKVHeads]);

    [attn_out, key_cache_out, value_cache_out, next_valid_len] = qwen2_runtime.hdl.attention_token_step_sram_step( ...
        q_token, k_token, v_token, key_cache_in, value_cache_in, cache_valid_len, rope_position, freqs_cis, hyperParameters, cfg);
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
    cosTheta = fi(reshape(freqs_cis.Cos(:, pos), [half, 1]), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
    sinTheta = fi(reshape(freqs_cis.Sin(:, pos), [half, 1]), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);

    xq_rot = rotateTokenPair(xq, cosTheta, sinTheta, F, cfg);
    xk_rot = rotateTokenPair(xk, cosTheta, sinTheta, F, cfg);
end

function x_rot = rotateTokenPair(x, cosTheta, sinTheta, F, cfg)
    half = size(x, 1) / 2;
    headCount = size(x, 2);
    x_rot = fi(zeros(size(x)), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
    for h = 1:headCount
        realPart = fi(x(1:half, h), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
        imagPart = fi(x(half+1:end, h), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
        x_rot(1:half, h) = fi(realPart .* cosTheta - imagPart .* sinTheta, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
        x_rot(half+1:end, h) = fi(realPart .* sinTheta + imagPart .* cosTheta, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
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