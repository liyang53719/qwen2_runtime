function h_out = block_mlp_tail_step(h_in, norm_weight, mlp_weights, cfg)
%BLOCK_MLP_TAIL_STEP Fixed-point post-attention tail: RMSNorm -> gated MLP -> residual add.

    h_post = qwen2_runtime.hdl.rmsnorm_step(h_in, norm_weight, single(1.0e-6), cfg);
    h_ffn = qwen2_runtime.hdl.gated_mlp_step(h_post, mlp_weights, cfg);
    h_out = addTensor(h_in, h_ffn, cfg);
end

function Y = addTensor(A, B, cfg)
    F = tailFimath(cfg);
    accumWL = cfg.HDLLinearAccumWordLength;
    accumFL = cfg.HDLLinearAccumFractionLength;
    [hiddenSize, seqLen, batchSize] = size(A);
    Y = fi(zeros(hiddenSize, seqLen, batchSize), true, accumWL, accumFL, F);
    for b = 1:batchSize
        for s = 1:seqLen
            for i = 1:hiddenSize
                aVal = fi(A(i, s, b), true, accumWL, accumFL, F);
                bVal = fi(B(i, s, b), true, accumWL, accumFL, F);
                Y(i, s, b) = fi(aVal + bVal, true, accumWL, accumFL, F);
            end
        end
    end
end

function F = tailFimath(cfg)
    F = fimath( ...
        'RoundingMethod', 'Floor', ...
        'OverflowAction', 'Saturate', ...
        'ProductMode', 'SpecifyPrecision', ...
        'ProductWordLength', 24, ...
        'ProductFractionLength', cfg.HDLLinearAccumFractionLength, ...
        'SumMode', 'SpecifyPrecision', ...
        'SumWordLength', cfg.HDLLinearAccumWordLength, ...
        'SumFractionLength', cfg.HDLLinearAccumFractionLength);
end