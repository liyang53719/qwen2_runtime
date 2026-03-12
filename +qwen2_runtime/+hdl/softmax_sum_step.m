function [sum_out, output_valid] = softmax_sum_step(start, exp_val, sum_seed, row_last)
%SOFTMAX_SUM_STEP Sequential sum accumulator for softmax denominator.

    coder.inline('never');

    F = sumFimath();
    persistent sum_reg
    if isempty(sum_reg)
        sum_reg = fi(0, true, 32, 14, F);
    end

    if start
        sum_reg = fi(sum_seed, true, 32, 14, F);
    end

    exp_fix = fi(exp_val, true, 16, 14, F);
    increment = fi(exp_fix, true, 32, 14, F);
    sum_reg = fi(sum_reg + increment, true, 32, 14, F);
    sum_out = sum_reg;
    output_valid = row_last;
end

function F = sumFimath()
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
