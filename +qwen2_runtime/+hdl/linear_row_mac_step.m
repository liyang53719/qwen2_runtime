function [acc_out, output_valid] = linear_row_mac_step(start, x_val, w_val, acc_seed, row_last)
%LINEAR_ROW_MAC_STEP Sequential fixed-point MAC engine.

    F = macFimath();
    persistent acc_reg
    if isempty(acc_reg)
        acc_reg = fi(0, true, 32, 14, F);
    end

    if start
        acc_reg = fi(acc_seed, true, 32, 14, F);
    end

    x_fix = fi(x_val, true, 16, 14, F);
    w_fix = fi(w_val, true, 16, 14, F);
    acc_reg = acc_reg + w_fix * x_fix;

    acc_out = acc_reg;
    output_valid = row_last;
end

function F = macFimath()
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
