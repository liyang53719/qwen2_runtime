function weight_out = softmax_normalize_single_step(exp_val, recip_val)
%SOFTMAX_NORMALIZE_SINGLE_STEP Single-precision normalize multiply.

    coder.inline('never');

    weight_out = single(exp_val) .* single(recip_val);
end
