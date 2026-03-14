function weight_out = softmax_normalize_half_step(exp_val, recip_val)
%SOFTMAX_NORMALIZE_HALF_STEP Native half-precision normalization multiply.

    coder.inline('never');

    weight_out = half(exp_val) .* half(recip_val);
end
