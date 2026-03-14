function [score_out, output_valid] = attention_score_mac_step(start, query_val, key_val, score_seed, row_last, scale)
%ATTENTION_SCORE_MAC_STEP Sequential fixed-point attention score PE.

    F = scoreFimath();
    persistent acc_reg
    if isempty(acc_reg)
        acc_reg = fi(0, true, 32, 14, F);
    end

    if start
        acc_reg = fi(score_seed, true, 32, 14, F);
    end

    qVal = fi(query_val, true, 16, 14, F);
    kVal = fi(key_val, true, 16, 14, F);
    product = fi(qVal * kVal, true, 32, 14, F);
    acc_reg = acc_reg + product;

    scaleVal = fi(scale, true, 16, 14, F);
    if row_last
        score_out = fi(acc_reg * scaleVal, true, 32, 14, F);
    else
        score_out = acc_reg;
    end
    output_valid = row_last;
end

function F = scoreFimath()
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
