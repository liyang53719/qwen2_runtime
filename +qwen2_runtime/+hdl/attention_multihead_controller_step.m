function [attn_out, out_valid] = attention_multihead_controller_step(start, score_mat, value_tensor, max_seed, sum_seed)
%ATTENTION_MULTIHEAD_CONTROLLER_STEP Sequential controller for all attention heads.

    coder.inline('never');

    F = attnFimath();
    numHeads = uint8(size(score_mat, 2));
    laneCount = size(value_tensor, 2);
    persistent head_idx running attn_reg valid_reg head_start_pending
    if isempty(head_idx)
        head_idx = uint8(1);
        running = false;
        attn_reg = fi(zeros(laneCount, double(numHeads)), true, 32, 14, F);
        valid_reg = false;
        head_start_pending = false;
    end

    attn_out = attn_reg;
    out_valid = valid_reg;
    valid_reg = false;

    if start
        head_idx = uint8(1);
        running = true;
        attn_reg = fi(zeros(laneCount, double(numHeads)), true, 32, 14, F);
        head_start_pending = true;
    end

    if ~running
        return;
    end

    score_vec = score_mat(:, head_idx);
    value_mat = value_tensor(:, :, head_idx);
    [head_out, head_valid] = qwen2_runtime.hdl.attention_head_controller_step( ...
        head_start_pending, score_vec, value_mat, max_seed, sum_seed);
    head_start_pending = false;

    if head_valid
        attn_reg(:, head_idx) = fi(head_out, true, 32, 14, F);
        attn_out = attn_reg;
        if head_idx == numHeads
            out_valid = true;
            valid_reg = true;
            running = false;
        else
            head_idx = head_idx + uint8(1);
            head_start_pending = true;
        end
    end
end

function F = attnFimath()
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