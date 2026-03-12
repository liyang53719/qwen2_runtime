function args = rmsnorm_entry_hardware_args()
%RMSNORM_ENTRY_HARDWARE_ARGS Reduced arguments for fixed-point RMSNorm hardware baseline.

    hiddenSize = 8;
    seqLen = 1;
    batchSize = 1;
    cfg = qwen2_runtime.defaultHardwareHDLConfig();
    F = rmsFimath(cfg);

    X = fi(reshape(single(linspace(-0.25, 0.25, hiddenSize)), hiddenSize, seqLen, batchSize), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
    weight = fi(reshape(single(linspace(0.95, 1.05, hiddenSize)), hiddenSize, 1), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
    epsilon = coder.Constant(single(1.0e-6));

    args = {X, weight, epsilon, coder.Constant(cfg)};
end

function F = rmsFimath(cfg)
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