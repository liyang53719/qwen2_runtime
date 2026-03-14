function exp_out = softmax_exp_half_step(score_val, max_val)
%SOFTMAX_EXP_HALF_STEP Native half-precision exp(score-max).

    coder.inline('never');

    delta = half(score_val) - half(max_val);
    exp_out = half(exp(single(delta)));
end
