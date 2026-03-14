function recip_out = softmax_recip_half_step(denom_val)
%SOFTMAX_RECIP_HALF_STEP Native half-precision reciprocal.

    coder.inline('never');

    denom_half = half(denom_val);
    if denom_half == half(0)
        denom_half = half(1);
    end
    recip_out = half(1) ./ denom_half;
end
