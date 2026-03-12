function exp_out = softmax_exp_step(score_val, max_val)
%SOFTMAX_EXP_STEP Approximate exp(score-max) for HDL.

    coder.inline('never');

    F = expFimath();
    delta = fi(score_val, true, 16, 14, F) - fi(max_val, true, 16, 14, F);
    exp_out = approxExpNeg(delta);
end

function F = expFimath()
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

function y = approxExpNeg(x)
    F = expFimath();
    if x <= fi(-8.0, true, 16, 14, F)
        y = fi(0, true, 16, 14, F);
    elseif x <= fi(-4.0, true, 16, 14, F)
        y = linearInterp(x, fi(-8.0, true, 16, 14, F), fi(-4.0, true, 16, 14, F), fi(0.00033546, true, 16, 14, F), fi(0.01831564, true, 16, 14, F));
    elseif x <= fi(-2.0, true, 16, 14, F)
        y = linearInterp(x, fi(-4.0, true, 16, 14, F), fi(-2.0, true, 16, 14, F), fi(0.01831564, true, 16, 14, F), fi(0.13533528, true, 16, 14, F));
    elseif x <= fi(-1.0, true, 16, 14, F)
        y = linearInterp(x, fi(-2.0, true, 16, 14, F), fi(-1.0, true, 16, 14, F), fi(0.13533528, true, 16, 14, F), fi(0.36787945, true, 16, 14, F));
    elseif x <= fi(0.0, true, 16, 14, F)
        y = linearInterp(x, fi(-1.0, true, 16, 14, F), fi(0.0, true, 16, 14, F), fi(0.36787945, true, 16, 14, F), fi(1.0, true, 16, 14, F));
    else
        y = fi(1.0, true, 16, 14, F);
    end
end

function y = linearInterp(x, x0, x1, y0, y1)
    F = fimath( ...
        'RoundingMethod', 'Floor', ...
        'OverflowAction', 'Saturate', ...
        'ProductMode', 'SpecifyPrecision', ...
        'ProductWordLength', 24, ...
        'ProductFractionLength', 14, ...
        'SumMode', 'SpecifyPrecision', ...
        'SumWordLength', 16, ...
        'SumFractionLength', 14);
    slope = fi((y1 - y0) / (x1 - x0), true, 16, 14, F);
    delta = fi(x - x0, true, 16, 14, F);
    interp = fi(delta * slope, true, 16, 14, F);
    y = fi(y0 + interp, true, 16, 14, F);
end
