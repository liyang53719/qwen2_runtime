function out_vec = residual_add_step(residual_vec, update_vec)
%RESIDUAL_ADD_STEP Fixed-point residual add primitive.

    coder.inline('never');

    F = resFimath();
    len = size(residual_vec, 1);
    out_vec = fi(zeros(size(residual_vec)), true, 32, 14, F);
    for i = 1:len
        a = fi(residual_vec(i), true, 32, 14, F);
        b = fi(update_vec(i), true, 32, 14, F);
        out_vec(i) = fi(a + b, true, 32, 14, F);
    end
end

function F = resFimath()
    F = fimath('RoundingMethod', 'Floor', 'OverflowAction', 'Saturate', 'ProductMode', 'SpecifyPrecision', 'ProductWordLength', 24, 'ProductFractionLength', 14, 'SumMode', 'SpecifyPrecision', 'SumWordLength', 32, 'SumFractionLength', 14);
end
