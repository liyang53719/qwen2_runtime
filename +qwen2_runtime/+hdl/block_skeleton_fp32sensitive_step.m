function [block_out, out_valid] = block_skeleton_fp32sensitive_step(start, input_vec, score_mat, value_tensor, residual_seed)
%BLOCK_SKELETON_FP32SENSITIVE_STEP Block skeleton with FP32-sensitive attention path.

    coder.inline('never');

    F = localFimath();
    persistent phase attn_buf valid_reg running
    if isempty(phase)
        phase = uint8(0);
        attn_buf = fi(zeros(size(input_vec)), true, 32, 14, F);
        valid_reg = false;
        running = false;
    end

    block_out = attn_buf;
    out_valid = valid_reg;
    valid_reg = false;

    if start
        phase = uint8(0);
        attn_buf = fi(zeros(size(input_vec)), true, 32, 14, F);
        running = true;
    end

    if ~running
        return;
    end

    if phase == uint8(0)
        [attn_mat, attn_valid] = attention_multihead_fp32sensitive_controller(start, score_mat, value_tensor);
        attn_buf = flatten_two_heads(attn_mat, F);
        block_out = attn_buf;
        if attn_valid
            phase = uint8(1);
        end
    else
        input_norm = fi(input_vec, true, 32, 14, F);
        block_out = qwen2_runtime.hdl.residual_add_step(input_norm, attn_buf);
        out_valid = true;
        valid_reg = true;
        running = false;
    end
end

function [attn_out, out_valid] = attention_multihead_fp32sensitive_controller(start, score_mat, value_tensor)
    F = localFimath();
    persistent head_idx phase_cycle out_buf valid_reg running
    if isempty(head_idx)
        head_idx = uint8(1);
        phase_cycle = uint8(0);
        out_buf = fi(zeros(4, 2), true, 32, 14, F);
        valid_reg = false;
        running = false;
    end

    numHeads = uint8(size(score_mat, 2));
    cacheLen = uint8(size(score_mat, 1));
    laneCount = uint8(size(value_tensor, 2));
    totalHeadCycles = uint8(1); %#ok<NASGU>

    attn_out = out_buf;
    out_valid = valid_reg;
    valid_reg = false;

    if start
        head_idx = uint8(1);
        phase_cycle = uint8(0);
        out_buf = fi(zeros(size(out_buf)), true, 32, 14, F);
        running = true;
    end

    if ~running
        return;
    end

    if head_idx == uint8(1)
        score_vec = score_mat(:, 1);
        value_mat = value_tensor(:, :, 1);
    else
        score_vec = score_mat(:, 2);
        value_mat = value_tensor(:, :, 2);
    end

    head_out = attention_head_fp32sensitive_controller(score_vec, value_mat, cacheLen, laneCount);
    out_buf(:, head_idx) = fi(head_out, true, 32, 14, F);
    attn_out = out_buf;

    if head_idx == numHeads
        out_valid = true;
        valid_reg = true;
        running = false;
    else
        head_idx = head_idx + uint8(1);
    end
end

function head_out = attention_head_fp32sensitive_controller(score_vec, value_mat, cacheLen, laneCount)
    scores = zeros(cacheLen, 1, 'single');
    values = zeros(cacheLen, laneCount, 'single');
    for t = 1:cacheLen
        scores(t) = single(score_vec(t));
        for lane = 1:laneCount
            values(t, lane) = single(value_mat(t, lane));
        end
    end
    head_out = qwen2_runtime.hdl.attention_weighted_value_controller_single_step(scores, values);
end

function vec = flatten_two_heads(attn_mat, F)
    vec = fi(zeros(8, 1), true, 32, 14, F);
    vec(1) = cast32(attn_mat(1, 1), F);
    vec(2) = cast32(attn_mat(2, 1), F);
    vec(3) = cast32(attn_mat(3, 1), F);
    vec(4) = cast32(attn_mat(4, 1), F);
    vec(5) = cast32(attn_mat(1, 2), F);
    vec(6) = cast32(attn_mat(2, 2), F);
    vec(7) = cast32(attn_mat(3, 2), F);
    vec(8) = cast32(attn_mat(4, 2), F);
end

function y = cast32(x, F)
    y = fi(double(x), true, 32, 14, F);
end

function F = localFimath()
    F = fimath('RoundingMethod', 'Floor', 'OverflowAction', 'Saturate', 'ProductMode', 'SpecifyPrecision', 'ProductWordLength', 24, 'ProductFractionLength', 14, 'SumMode', 'SpecifyPrecision', 'SumWordLength', 32, 'SumFractionLength', 14);
end
