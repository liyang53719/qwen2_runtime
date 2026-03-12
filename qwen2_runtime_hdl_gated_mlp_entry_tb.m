function qwen2_runtime_hdl_gated_mlp_entry_tb
%QWEN2_RUNTIME_HDL_GATED_MLP_ENTRY_TB MATLAB stimulus for fixed-point gated MLP HDL baseline.

    hiddenSize = 8;
    seqLen = 1;
    batchSize = 1;
    mlpSize = 16;
    cfg = qwen2_runtime.defaultHardwareHDLConfig(); %#ok<NASGU>
    F = mlpFimath(cfg);

    X = fi(reshape(single(linspace(-0.25, 0.25, hiddenSize)), hiddenSize, seqLen, batchSize), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F); %#ok<NASGU>
    weights = struct();
    weights.gate_proj = fi(makeMatrix(mlpSize, hiddenSize, 0.075, 0.4), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F); %#ok<NASGU>
    weights.up_proj = fi(makeMatrix(mlpSize, hiddenSize, 0.075, 1.1), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F); %#ok<NASGU>
    weights.down_proj = fi(makeMatrix(hiddenSize, mlpSize, 0.075, 1.8), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F); %#ok<NASGU>

    clear qwen2_runtime_hdl_gated_mlp_entry

    Z = qwen2_runtime_hdl_gated_mlp_entry(X, weights, cfg); %#ok<NASGU>
    final_Z = Z; %#ok<NASGU>
end

function W = makeMatrix(rows, cols, scale, phase)
    values = sin(single(1:(rows * cols)) + single(phase));
    W = reshape(single(scale) .* single(values), rows, cols);
end

function F = mlpFimath(cfg)
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