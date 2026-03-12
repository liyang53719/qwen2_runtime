function qwen2_runtime_hdl_rmsnorm_entry_tb
%QWEN2_RUNTIME_HDL_RMSNORM_ENTRY_TB MATLAB stimulus for fixed-point RMSNorm HDL baseline.

    hiddenSize = 8;
    seqLen = 1;
    batchSize = 1;
    cfg = qwen2_runtime.defaultHardwareHDLConfig(); %#ok<NASGU>
    F = rmsFimath(cfg);

    X = fi(reshape(single(linspace(-0.25, 0.25, hiddenSize)), hiddenSize, seqLen, batchSize), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F); %#ok<NASGU>
    weight = fi(reshape(single(linspace(0.95, 1.05, hiddenSize)), hiddenSize, 1), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F); %#ok<NASGU>
    epsilon = single(1.0e-6); %#ok<NASGU>

    clear qwen2_runtime_hdl_rmsnorm_entry

    Y = qwen2_runtime_hdl_rmsnorm_entry(X, weight, epsilon, cfg); %#ok<NASGU>
    final_Y = Y; %#ok<NASGU>
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