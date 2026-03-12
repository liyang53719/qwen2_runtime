function [head_out, out_valid] = attention_head_controller_step(start, score_vec, value_mat, max_seed, sum_seed)
%ATTENTION_HEAD_CONTROLLER_STEP Sequential controller for one attention head.

    coder.inline('never');

    F = headFimath();
    persistent lane_idx running head_reg valid_reg lane_start_pending
    if isempty(lane_idx)
        lane_idx = uint8(1);
        running = false;
        head_reg = fi(zeros(size(value_mat, 2), 1), true, 32, 14, F);
        valid_reg = false;
        lane_start_pending = false;
    end

    laneCount = uint8(size(value_mat, 2));
    head_out = head_reg;
    out_valid = valid_reg;
    valid_reg = false;

    if start
        lane_idx = uint8(1);
        running = true;
        head_reg = fi(zeros(size(value_mat, 2), 1), true, 32, 14, F);
        lane_start_pending = true;
    end

    if ~running
        return;
    end

    value_seed = fi(0, true, 32, 14, F);
    lane_vec = value_mat(:, lane_idx);
    [lane_out, lane_valid] = qwen2_runtime.hdl.attention_weighted_value_controller_step( ...
        lane_start_pending, score_vec, lane_vec, max_seed, sum_seed, value_seed);
    lane_start_pending = false;

    if lane_valid
        head_reg(lane_idx) = fi(lane_out, true, 32, 14, F);
        head_out = head_reg;
        if lane_idx == laneCount
            out_valid = true;
            valid_reg = true;
            running = false;
        else
            lane_idx = lane_idx + uint8(1);
            lane_start_pending = true;
        end
    end
end

function F = headFimath()
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