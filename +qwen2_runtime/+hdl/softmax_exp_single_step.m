function exp_out = softmax_exp_single_step(score_val, max_val)
%SOFTMAX_EXP_SINGLE_STEP Single-precision exp(score-max) baseline-sensitive path.

    coder.inline('never');

    exp_out = single(exp(single(score_val) - single(max_val)));
end
