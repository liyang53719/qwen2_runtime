function [value_out, out_valid] = attention_weighted_value_controller_step(start, score_vec, value_vec, max_seed, sum_seed, value_seed)
%ATTENTION_WEIGHTED_VALUE_CONTROLLER_STEP Approximate softmax-weighted value accumulation.

    coder.inline('never');

    Fmax = maxFimath();
    Fsum = sumFimath();
    Fval = valueFimath();
    persistent idx phase max_reg sum_reg value_reg valid_reg running
    if isempty(idx)
        idx = uint8(1);
        phase = uint8(0);
        max_reg = fi(-8, true, 16, 14, Fmax);
        sum_reg = fi(0, true, 32, 14, Fsum);
        value_reg = fi(0, true, 32, 14, Fval);
        valid_reg = false;
        running = false;
    end

    vecLen = uint8(length(score_vec));
    value_out = value_reg;
    out_valid = valid_reg;
    valid_reg = false;

    if start
        idx = uint8(1);
        phase = uint8(0);
        max_reg = fi(max_seed, true, 16, 14, Fmax);
        sum_reg = fi(sum_seed, true, 32, 14, Fsum);
        value_reg = fi(value_seed, true, 32, 14, Fval);
        running = true;
    end

    if ~running
        return;
    end

    score_val = fi(score_vec(idx), true, 16, 14, Fmax);
    value_val = fi(value_vec(idx), true, 16, 14, Fval);
    lastElem = (idx == vecLen);
    max_seed_local = fi(max_seed, true, 16, 14, Fmax);
    sum_seed_local = fi(sum_seed, true, 32, 14, Fsum);
    value_seed_local = fi(value_seed, true, 32, 14, Fval);

    if phase == uint8(0)
        [max_reg, max_valid] = qwen2_runtime.hdl.softmax_max_step(idx == uint8(1), score_val, max_seed_local, lastElem);
        if max_valid
            phase = uint8(1);
            idx = uint8(1);
        else
            idx = idx + uint8(1);
        end
    elseif phase == uint8(1)
        exp_val = qwen2_runtime.hdl.softmax_exp_step(score_val, max_reg);
        [sum_reg, sum_valid] = qwen2_runtime.hdl.softmax_sum_step(idx == uint8(1), exp_val, sum_seed_local, lastElem);
        if sum_valid
            phase = uint8(2);
            idx = uint8(1);
        else
            idx = idx + uint8(1);
        end
    else
        exp_val = qwen2_runtime.hdl.softmax_exp_step(score_val, max_reg);
        denomSafe = fi(sum_reg, true, 32, 14, Fsum);
        if denomSafe == fi(0, true, 32, 14, Fsum)
            denomSafe = fi(1, true, 32, 14, Fsum);
        end
        denom_recip = qwen2_runtime.hdl.softmax_recip_lookup_step(fi(denomSafe, true, 16, 14));
        weight_val = qwen2_runtime.hdl.softmax_normalize_step(fi(exp_val, true, 16, 14), fi(denom_recip, true, 16, 14));
        [value_reg, value_valid] = qwen2_runtime.hdl.attention_value_mac_step(idx == uint8(1), weight_val, value_val, value_seed_local, lastElem);
        value_out = value_reg;
        if value_valid
            out_valid = true;
            valid_reg = true;
            running = false;
        else
            idx = idx + uint8(1);
        end
    end
end

function F = maxFimath()
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

function F = sumFimath()
    F = fimath( ...
        'RoundingMethod', 'Floor', ...
        'OverflowAction', 'Saturate', ...
        'ProductMode', 'SpecifyPrecision', ...
        'ProductWordLength', 24, ...
        'ProductFractionLength', 14, ...
        'SumMode', 'SpecifyPrecision', ...
        'SumWordLength', 32, ...
        'SumFractionLength', 14);
end

function F = valueFimath()
    F = fimath( ...
        'RoundingMethod', 'Floor', ...
        'OverflowAction', 'Saturate', ...
        'ProductMode', 'SpecifyPrecision', ...
        'ProductWordLength', 24, ...
        'ProductFractionLength', 14, ...
        'SumMode', 'SpecifyPrecision', ...
        'SumWordLength', 32, ...
        'SumFractionLength', 14);
end
