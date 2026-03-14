function [value_acc_out, out_valid] = attention_value_row_controller_step(start, weight_vec, value_vec, acc_seed)
%ATTENTION_VALUE_ROW_CONTROLLER_STEP Sequential controller for value accumulation.

    F = valueFimath();
    persistent idx running acc_reg valid_reg prev_running
    if isempty(idx)
        idx = uint8(1);
        running = false;
        acc_reg = fi(0, true, 32, 14, F);
        valid_reg = false;
        prev_running = false;
    end

    vecLen = uint8(length(weight_vec));
    value_acc_out = acc_reg;
    out_valid = valid_reg;
    valid_reg = false;

    if start
        idx = uint8(1);
        running = true;
        acc_reg = fi(acc_seed, true, 32, 14, F);
        prev_running = false;
    end

    if running
        row_last = (idx == vecLen);
        mac_start = ~prev_running;
        [acc_reg, local_valid] = qwen2_runtime.hdl.attention_value_mac_step(mac_start, weight_vec(idx), value_vec(idx), acc_seed, row_last);
        value_acc_out = acc_reg;
        if local_valid
            out_valid = true;
            valid_reg = true;
            running = false;
            prev_running = false;
        else
            idx = idx + uint8(1);
            prev_running = true;
        end
    end
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
