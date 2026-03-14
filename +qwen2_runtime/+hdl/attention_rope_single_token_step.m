function [xq_rot, xk_rot] = attention_rope_single_token_step(xq, xk, rope_position, freqs_cis, cfg)
%ATTENTION_ROPE_SINGLE_TOKEN_STEP Apply fixed-point RoPE to one token of q/k heads.

    half = size(xq, 1) / 2;
    F = localFimath(cfg);
    pos = double(rope_position);
    cosTheta = fi(freqs_cis.Cos(:, pos), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
    sinTheta = fi(freqs_cis.Sin(:, pos), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
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

function F = localFimath(cfg)
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