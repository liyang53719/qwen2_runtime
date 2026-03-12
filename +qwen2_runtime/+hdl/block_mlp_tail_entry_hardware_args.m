function args = block_mlp_tail_entry_hardware_args()
%BLOCK_MLP_TAIL_ENTRY_HARDWARE_ARGS Reduced arguments for fixed-point block tail baseline.

    hiddenSize = 8;
    seqLen = 1;
    batchSize = 1;
    mlpSize = 16;
    cfg = qwen2_runtime.defaultHardwareHDLConfig();
    F = tailFimath(cfg);

    h_in = fi(reshape(single(linspace(-0.25, 0.25, hiddenSize)), hiddenSize, seqLen, batchSize), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
    norm_weight = fi(reshape(single(linspace(1.05, 0.95, hiddenSize)), hiddenSize, 1), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);

    mlp_weights = struct();
    mlp_weights.gate_proj = fi(makeMatrix(mlpSize, hiddenSize, 0.075, 0.4), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
    mlp_weights.up_proj = fi(makeMatrix(mlpSize, hiddenSize, 0.075, 1.1), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
    mlp_weights.down_proj = fi(makeMatrix(hiddenSize, mlpSize, 0.075, 1.8), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);

    args = {h_in, norm_weight, coder.Constant(mlp_weights), coder.Constant(cfg)};
end

function W = makeMatrix(rows, cols, scale, phase)
    values = sin(single(1:(rows * cols)) + single(phase));
    W = reshape(single(scale) .* single(values), rows, cols);
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