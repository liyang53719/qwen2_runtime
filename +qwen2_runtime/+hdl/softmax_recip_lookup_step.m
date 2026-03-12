function recip_out = softmax_recip_lookup_step(denom_val)
%SOFTMAX_RECIP_LOOKUP_STEP Piecewise reciprocal approximation for denominator.

    coder.inline('never');

    F = recipFimath();
    denom_fix = fi(denom_val, true, 16, 14, F);
    if denom_fix <= fi(0.25, true, 16, 14, F)
        recip_out = fi(4.0, true, 16, 14, F);
    elseif denom_fix <= fi(0.5, true, 16, 14, F)
        recip_out = fi(2.0, true, 16, 14, F);
    elseif denom_fix <= fi(1.0, true, 16, 14, F)
        recip_out = fi(1.0, true, 16, 14, F);
    elseif denom_fix <= fi(2.0, true, 16, 14, F)
        recip_out = fi(0.5, true, 16, 14, F);
    else
        recip_out = fi(0.25, true, 16, 14, F);
    end
end

function F = recipFimath()
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
