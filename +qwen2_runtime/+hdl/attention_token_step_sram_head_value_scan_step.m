function [head_acc_out, head_valid] = attention_token_step_sram_head_value_scan_step(start, weight_vec, value_mat, value_seed, cfg)
%ATTENTION_TOKEN_STEP_SRAM_HEAD_VALUE_SCAN_STEP Scan all lanes for one head outside the top engine.

    headDim = uint16(size(value_mat, 1));
    F = valuePathFimath();
    persistent lane_idx running head_acc_reg row_start_pending
    if isempty(lane_idx)
        lane_idx = uint16(1);
        running = false;
        head_acc_reg = fi(zeros(double(headDim), 1), true, cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, F);
        row_start_pending = false;
    end

    head_acc_out = head_acc_reg;
    head_valid = false;

    if start
        lane_idx = uint16(1);
        running = true;
        head_acc_reg(:) = fi(0, true, cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, F);
        row_start_pending = true;
    end

    if ~running
        return;
    end

    value_vec = reshape(value_mat(double(lane_idx), :), [size(value_mat, 2), 1]);
    [lane_acc, lane_valid] = qwen2_runtime.hdl.attention_value_row_controller_step(row_start_pending, weight_vec, value_vec, value_seed);
    row_start_pending = false;
    if lane_valid
        head_acc_reg(double(lane_idx)) = fi(lane_acc, true, cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, F);
        if lane_idx >= headDim
            running = false;
            head_valid = true;
        else
            lane_idx = lane_idx + uint16(1);
            row_start_pending = true;
        end
    end

    head_acc_out = head_acc_reg;
end

function F = valuePathFimath()
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