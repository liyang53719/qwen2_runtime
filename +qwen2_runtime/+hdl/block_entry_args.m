function args = block_entry_args(matParamsFile)
%BLOCK_ENTRY_ARGS Representative argument set for HDL codegen of one block.

    parameters = qwen2_runtime.load(matParamsFile, 'PrepareDynamicInt8', false, 'ConvertDLArrayToSingle', true);
    plan = qwen2_runtime.prepareForHDL(parameters, qwen2_runtime.defaultHDLConfig());
    hp = plan.Hyperparameters;

    seqLen = 1;
    batchSize = 1;
    hiddenSize = hp.HiddenSize;
    numKVHeads = hp.NumKVHeads;
    headDim = hp.HeadDim;
    maxCacheLen = 2;

    h_in = zeros(hiddenSize, seqLen, batchSize, 'single');
    key_in = zeros(headDim, numKVHeads, maxCacheLen, batchSize, 'single');
    value_in = zeros(headDim, numKVHeads, maxCacheLen, batchSize, 'single');
    cache_valid_len = 0;

    freqsFull = transformer.layer.precomputeFreqsCis(headDim, seqLen + 8, hp.RopeTheta);
    freqs_cis = complex(single(real(freqsFull(:, 1:seqLen))), single(imag(freqsFull(:, 1:seqLen))));

    args = { ...
        h_in, ...
        key_in, ...
        value_in, ...
        coder.Constant(cache_valid_len), ...
        coder.Constant(plan.LayerWeights{1}), ...
        coder.Constant(hp), ...
        coder.Constant(freqs_cis), ...
        coder.Constant(plan.RuntimeConfig)};
end
