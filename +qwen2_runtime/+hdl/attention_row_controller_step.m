function [score_out, out_valid] = attention_row_controller_step(start, query_vec, key_vec, score_seed, scale)
%ATTENTION_ROW_CONTROLLER_STEP Sequential controller for attention score row.

    F = scoreFimath();
    persistent idx running score_reg valid_reg
    if isempty(idx)
        idx = uint8(1);
        running = false;
        score_reg = fi(0, true, 32, 14, F);
        valid_reg = false;
    end

    vecLen = uint8(length(query_vec));
    score_out = score_reg;
    out_valid = valid_reg;
    valid_reg = false;

    if start
        idx = uint8(1);
        running = true;
        score_reg = fi(score_seed, true, 32, 14, F);
    end

    if running
        row_last = (idx == vecLen);
        [score_reg, local_valid] = qwen2_runtime.hdl.attention_score_mac_step(idx == uint8(1), query_vec(idx), key_vec(idx), score_seed, row_last, scale);
        score_out = score_reg;
        if local_valid
            out_valid = true;
            valid_reg = true;
            running = false;
        else
            idx = idx + uint8(1);
        end
    end
end

function F = scoreFimath()
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
