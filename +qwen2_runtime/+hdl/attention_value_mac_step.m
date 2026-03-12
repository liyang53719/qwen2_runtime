function [value_acc_out, output_valid] = attention_value_mac_step(start, weight_val, value_val, acc_seed, row_last)
%ATTENTION_VALUE_MAC_STEP Sequential weighted value accumulation PE.

    coder.inline('never');

    F = valueFimath();
    persistent acc_reg
    if isempty(acc_reg)
        acc_reg = fi(0, true, 32, 14, F);
    end

    if start
        acc_reg = fi(acc_seed, true, 32, 14, F);
    end

    wVal = fi(weight_val, true, 16, 14, F);
    vVal = fi(value_val, true, 16, 14, F);
    product = fi(wVal * vVal, true, 32, 14, F);
    acc_reg = acc_reg + product;
    value_acc_out = acc_reg;
    output_valid = row_last;
end

function F = valueFimath()
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
