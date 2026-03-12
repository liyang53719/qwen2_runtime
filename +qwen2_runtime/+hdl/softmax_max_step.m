function [max_out, output_valid] = softmax_max_step(start, score_val, max_seed, row_last)
%SOFTMAX_MAX_STEP Sequential max tracker for attention scores.

    coder.inline('never');

    F = maxFimath();
    persistent max_reg
    if isempty(max_reg)
        max_reg = fi(-8.0, true, 16, 14, F);
    end

    if start
        max_reg = fi(max_seed, true, 16, 14, F);
    end

    score_fix = fi(score_val, true, 16, 14, F);
    if score_fix > max_reg
        max_reg = score_fix;
    end

    max_out = max_reg;
    output_valid = row_last;
end

function F = maxFimath()
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
