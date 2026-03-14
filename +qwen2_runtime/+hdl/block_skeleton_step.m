function [block_out, out_valid] = block_skeleton_step(start, input_vec, score_mat, value_tensor, residual_seed)
%BLOCK_SKELETON_STEP HDL-safe block skeleton using sequential attention controllers.

    coder.inline('never');

    F32 = fimath('RoundingMethod', 'Floor', 'OverflowAction', 'Saturate', ...
        'ProductMode', 'SpecifyPrecision', 'ProductWordLength', 24, 'ProductFractionLength', 14, ...
        'SumMode', 'SpecifyPrecision', 'SumWordLength', 32, 'SumFractionLength', 14);

    persistent out_reg valid_reg
    if isempty(out_reg)
        out_reg = fi(zeros(8, 1), true, 32, 14, F32);
        valid_reg = false;
    end

    block_out = out_reg;
    out_valid = valid_reg;
    valid_reg = false;

    if start
        input_fix = fi(input_vec, true, 32, 14, F32);
        attn_buf = qwen2_runtime.hdl.parallel_attention_mix_step(score_mat, value_tensor);
        resid_fix = fi(residual_seed, true, 32, 14, F32);
        tmp = qwen2_runtime.hdl.residual_add_step(input_fix, attn_buf);
        out_reg = qwen2_runtime.hdl.residual_add_step(tmp, resid_fix);
        valid_reg = true;
    end
end
