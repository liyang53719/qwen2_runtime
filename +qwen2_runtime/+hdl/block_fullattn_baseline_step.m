function [block_out, out_valid] = block_fullattn_baseline_step(start, input_vec, attn_mix_vec, residual_seed)
%BLOCK_FULLATTN_BASELINE_STEP Full-attention-output block baseline.

    coder.inline('never');

    F = localFimath();
    persistent running valid_reg out_reg
    if isempty(running)
        running = false;
        valid_reg = false;
        out_reg = fi(zeros(size(input_vec)), true, 32, 14, F);
    end

    block_out = out_reg;
    out_valid = valid_reg;
    valid_reg = false;

    if start
        running = true;
        out_reg = fi(zeros(size(input_vec)), true, 32, 14, F);
    end

    if ~running
        return;
    end

    input_fix = fi(input_vec, true, 32, 14, F);
    attn_fix = fi(attn_mix_vec, true, 32, 14, F);
    residual_fix = fi(residual_seed, true, 32, 14, F);

    tmp = qwen2_runtime.hdl.residual_add_step(input_fix, attn_fix);
    out_reg = qwen2_runtime.hdl.residual_add_step(tmp, residual_fix);
    block_out = out_reg;
    out_valid = true;
    valid_reg = true;
    running = false;
end

function F = localFimath()
    F = fimath('RoundingMethod', 'Floor', 'OverflowAction', 'Saturate', 'ProductMode', 'SpecifyPrecision', 'ProductWordLength', 24, 'ProductFractionLength', 14, 'SumMode', 'SpecifyPrecision', 'SumWordLength', 32, 'SumFractionLength', 14);
end
