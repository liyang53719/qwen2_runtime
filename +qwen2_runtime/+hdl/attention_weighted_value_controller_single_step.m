function out4 = attention_weighted_value_controller_single_step(scores, valuesMat)
%ATTENTION_WEIGHTED_VALUE_CONTROLLER_SINGLE_STEP Single-precision sensitive reference path.

    coder.inline('never');

    cacheLen = size(scores, 1);
    laneCount = size(valuesMat, 2);
    scoreMax = max(single(scores));
    expVals = zeros(cacheLen, 1, 'single');
    for t = 1:cacheLen
        expVals(t) = qwen2_runtime.hdl.softmax_exp_single_step(single(scores(t)), scoreMax);
    end
    recip = qwen2_runtime.hdl.softmax_recip_single_step(sum(expVals));

    out4 = zeros(laneCount, 1, 'single');
    for lane = 1:laneCount
        acc = single(0);
        for t = 1:cacheLen
            w = qwen2_runtime.hdl.softmax_normalize_single_step(expVals(t), recip);
            acc = acc + w * single(valuesMat(t, lane));
        end
        out4(lane) = acc;
    end
end
