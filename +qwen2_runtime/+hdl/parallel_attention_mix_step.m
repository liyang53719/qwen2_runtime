function attn_mix = parallel_attention_mix_step(score_mat, value_tensor)
%PARALLEL_ATTENTION_MIX_STEP Compute all head/lane weighted values in one call.

    coder.inline('never');

    F16 = localFimath16();
    F32 = localFimath32();

    cacheLen = size(score_mat, 1);
    numHeads = size(score_mat, 2);
    laneCount = size(value_tensor, 2);

    attn_mix = fi(zeros(numHeads * laneCount, 1), true, 32, 14, F32);
    exp_vals = fi(zeros(cacheLen, 1), true, 16, 14, F16);

    for head_idx = 1:numHeads
        score_max = fi(score_mat(1, head_idx), true, 16, 14, F16);
        for token_idx = 2:cacheLen
            score_now = fi(score_mat(token_idx, head_idx), true, 16, 14, F16);
            if score_now > score_max
                score_max = score_now;
            end
        end

        exp_sum = fi(0, true, 32, 14, F32);
        for token_idx = 1:cacheLen
            score_now = fi(score_mat(token_idx, head_idx), true, 16, 14, F16);
            exp_now = qwen2_runtime.hdl.softmax_exp_step(score_now, score_max);
            exp_vals(token_idx) = fi(exp_now, true, 16, 14, F16);
            exp_sum = fi(exp_sum + fi(exp_now, true, 32, 14, F32), true, 32, 14, F32);
        end

        denom_safe = exp_sum;
        if denom_safe == fi(0, true, 32, 14, F32)
            denom_safe = fi(1, true, 32, 14, F32);
        end
        denom_recip = qwen2_runtime.hdl.softmax_recip_lookup_step(fi(denom_safe, true, 16, 14, F16));

        for lane_idx = 1:laneCount
            lane_acc = fi(0, true, 32, 14, F32);
            for token_idx = 1:cacheLen
                weight_now = qwen2_runtime.hdl.softmax_normalize_step(exp_vals(token_idx), fi(denom_recip, true, 16, 14, F16));
                value_now = fi(value_tensor(token_idx, lane_idx, head_idx), true, 16, 14, F16);
                lane_acc = fi(lane_acc + fi(weight_now * value_now, true, 32, 14, F32), true, 32, 14, F32);
            end

            out_idx = (head_idx - 1) * laneCount + lane_idx;
            attn_mix(out_idx) = lane_acc;
        end
    end
end

function F = localFimath16()
    F = fimath('RoundingMethod', 'Floor', 'OverflowAction', 'Saturate', ...
        'ProductMode', 'SpecifyPrecision', 'ProductWordLength', 24, 'ProductFractionLength', 14, ...
        'SumMode', 'SpecifyPrecision', 'SumWordLength', 16, 'SumFractionLength', 14);
end

function F = localFimath32()
    F = fimath('RoundingMethod', 'Floor', 'OverflowAction', 'Saturate', ...
        'ProductMode', 'SpecifyPrecision', 'ProductWordLength', 24, 'ProductFractionLength', 14, ...
        'SumMode', 'SpecifyPrecision', 'SumWordLength', 32, 'SumFractionLength', 14);
end