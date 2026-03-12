function args = block_entry_baseline_args()
%BLOCK_ENTRY_BASELINE_ARGS Representative reduced-dimension args for full block HDL baseline.

    hiddenSize = 8;
    seqLen = 1;
    batchSize = 1;
    numHeads = 2;
    numKVHeads = 2;
    headDim = 4;
    mlpSize = 16;
    maxCacheLen = 4;

    h_in = reshape(single(linspace(-0.25, 0.25, hiddenSize)), hiddenSize, seqLen, batchSize);
    key_in = zeros(headDim, numKVHeads, maxCacheLen, batchSize, 'single');
    value_in = zeros(headDim, numKVHeads, maxCacheLen, batchSize, 'single');
    cache_valid_len = 0;

    weights = struct();
    weights.input_layernorm = reshape(single(linspace(0.95, 1.05, hiddenSize)), hiddenSize, 1);
    weights.post_attention_layernorm = reshape(single(linspace(1.05, 0.95, hiddenSize)), hiddenSize, 1);
    weights.self_attn_q_proj = makeMatrix(hiddenSize, hiddenSize, 0.125, 0.0);
    weights.self_attn_k_proj = makeMatrix(hiddenSize, hiddenSize, 0.125, 0.7);
    weights.self_attn_v_proj = makeMatrix(hiddenSize, hiddenSize, 0.125, 1.3);
    weights.self_attn_o_proj = makeMatrix(hiddenSize, hiddenSize, 0.125, 2.1);
    weights.mlp_gate_proj = makeMatrix(mlpSize, hiddenSize, 0.075, 0.4);
    weights.mlp_up_proj = makeMatrix(mlpSize, hiddenSize, 0.075, 1.1);
    weights.mlp_down_proj = makeMatrix(hiddenSize, mlpSize, 0.075, 1.8);

    hyperParameters = struct();
    hyperParameters.HiddenSize = hiddenSize;
    hyperParameters.NumHeads = numHeads;
    hyperParameters.NumKVHeads = numKVHeads;
    hyperParameters.HeadDim = headDim;
    hyperParameters.RopeTheta = 10000.0;

    freqsFull = transformer.layer.precomputeFreqsCis(headDim, seqLen + 8, hyperParameters.RopeTheta);
    freqs_cis = complex(single(real(freqsFull(:, 1:seqLen))), single(imag(freqsFull(:, 1:seqLen))));

    runtimeCfg = qwen2_runtime.defaultHDLConfig();

    args = { ...
        h_in, ...
        key_in, ...
        value_in, ...
        coder.Constant(cache_valid_len), ...
        coder.Constant(weights), ...
        coder.Constant(hyperParameters), ...
        coder.Constant(freqs_cis), ...
        coder.Constant(runtimeCfg)};
end

function W = makeMatrix(rows, cols, scale, phase)
    values = sin(single(1:(rows * cols)) + single(phase));
    W = reshape(single(scale) .* single(values), rows, cols);
end