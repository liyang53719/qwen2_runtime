function recip_out = softmax_recip_single_step(denom_val)
%SOFTMAX_RECIP_SINGLE_STEP Single-precision reciprocal baseline-sensitive path.

    coder.inline('never');

    denom = single(denom_val);
    if denom == 0
        denom = single(1);
    end
    recip_out = single(1) ./ denom;
end
