function [block_out, out_valid] = block_skeleton_streaming_step(start, input_vec, score_token, value_token, token_valid, token_last, residual_seed)
%BLOCK_SKELETON_STREAMING_STEP Buffered streaming variant of the block skeleton.

    coder.inline('never');

    F16 = localFimath16();
    F32 = localFimath32();

    persistent score_buf_flat value_buf_flat write_idx input_reg residual_reg out_reg valid_reg pending_compute
    if isempty(write_idx)
        score_buf_flat = fi(zeros(32, 1), true, 16, 14, F16);
        value_buf_flat = fi(zeros(128, 1), true, 16, 14, F16);
        write_idx = uint8(1);
        input_reg = fi(zeros(8, 1), true, 32, 14, F32);
        residual_reg = fi(zeros(8, 1), true, 32, 14, F32);
        out_reg = fi(zeros(8, 1), true, 32, 14, F32);
        valid_reg = false;
        pending_compute = false;
    end

    block_out = out_reg;
    out_valid = valid_reg;
    valid_reg = false;

    if start
        score_buf_flat(:) = fi(0, true, 16, 14, F16);
        value_buf_flat(:) = fi(0, true, 16, 14, F16);
        write_idx = uint8(1);
        input_reg = fi(input_vec, true, 32, 14, F32);
        residual_reg = fi(residual_seed, true, 32, 14, F32);
        pending_compute = false;
    end

    if pending_compute
        score_mat = fi(zeros(16, 2), true, 16, 14, F16);
        value_tensor = fi(zeros(16, 4, 2), true, 16, 14, F16);
        for tokenIdx = 1:16
            score_mat(tokenIdx, 1) = score_buf_flat(tokenIdx);
            score_mat(tokenIdx, 2) = score_buf_flat(16 + tokenIdx);
            for laneIdx = 1:4
                value_tensor(tokenIdx, laneIdx, 1) = value_buf_flat((laneIdx - 1) * 16 + tokenIdx);
                value_tensor(tokenIdx, laneIdx, 2) = value_buf_flat(64 + (laneIdx - 1) * 16 + tokenIdx);
            end
        end

        attn_mix = qwen2_runtime.hdl.parallel_attention_mix_step(score_mat, value_tensor);
        tmp = qwen2_runtime.hdl.residual_add_step(input_reg, attn_mix);
        out_reg = qwen2_runtime.hdl.residual_add_step(tmp, residual_reg);
        valid_reg = true;
        write_idx = uint8(1);
        pending_compute = false;
    elseif token_valid
        score_buf_flat(write_idx) = fi(score_token(1), true, 16, 14, F16);
        score_buf_flat(16 + write_idx) = fi(score_token(2), true, 16, 14, F16);

        value_buf_flat(write_idx) = fi(value_token(1), true, 16, 14, F16);
        value_buf_flat(16 + write_idx) = fi(value_token(2), true, 16, 14, F16);
        value_buf_flat(32 + write_idx) = fi(value_token(3), true, 16, 14, F16);
        value_buf_flat(48 + write_idx) = fi(value_token(4), true, 16, 14, F16);
        value_buf_flat(64 + write_idx) = fi(value_token(5), true, 16, 14, F16);
        value_buf_flat(80 + write_idx) = fi(value_token(6), true, 16, 14, F16);
        value_buf_flat(96 + write_idx) = fi(value_token(7), true, 16, 14, F16);
        value_buf_flat(112 + write_idx) = fi(value_token(8), true, 16, 14, F16);

        if token_last || write_idx == uint8(16)
            pending_compute = true;
        else
            write_idx = write_idx + uint8(1);
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