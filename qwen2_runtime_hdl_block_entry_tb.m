function qwen2_runtime_hdl_block_entry_tb
%QWEN2_RUNTIME_HDL_BLOCK_ENTRY_TB MATLAB stimulus for full block HDL baseline generation.

    hiddenSize = 8;
    seqLen = 1;
    batchSize = 1;
    numHeads = 2;
    numKVHeads = 2;
    headDim = 4;
    mlpSize = 16;
    maxCacheLen = 4;

    h_in = reshape(single(linspace(-0.25, 0.25, hiddenSize)), hiddenSize, seqLen, batchSize); %#ok<NASGU>
    key_in = zeros(headDim, numKVHeads, maxCacheLen, batchSize, 'single'); %#ok<NASGU>
    value_in = zeros(headDim, numKVHeads, maxCacheLen, batchSize, 'single'); %#ok<NASGU>
    cache_valid_len = 0; %#ok<NASGU>

    weights = struct();
    weights.input_layernorm = reshape(single(linspace(0.95, 1.05, hiddenSize)), hiddenSize, 1); %#ok<NASGU>
    weights.post_attention_layernorm = reshape(single(linspace(1.05, 0.95, hiddenSize)), hiddenSize, 1); %#ok<NASGU>
    weights.self_attn_q_proj = makeMatrix(hiddenSize, hiddenSize, 0.125, 0.0); %#ok<NASGU>
    weights.self_attn_k_proj = makeMatrix(hiddenSize, hiddenSize, 0.125, 0.7); %#ok<NASGU>
    weights.self_attn_v_proj = makeMatrix(hiddenSize, hiddenSize, 0.125, 1.3); %#ok<NASGU>
    weights.self_attn_o_proj = makeMatrix(hiddenSize, hiddenSize, 0.125, 2.1); %#ok<NASGU>
    weights.mlp_gate_proj = makeMatrix(mlpSize, hiddenSize, 0.075, 0.4); %#ok<NASGU>
    weights.mlp_up_proj = makeMatrix(mlpSize, hiddenSize, 0.075, 1.1); %#ok<NASGU>
    weights.mlp_down_proj = makeMatrix(hiddenSize, mlpSize, 0.075, 1.8); %#ok<NASGU>

    hyperParameters = struct();
    hyperParameters.HiddenSize = hiddenSize; %#ok<NASGU>
    hyperParameters.NumHeads = numHeads; %#ok<NASGU>
    hyperParameters.NumKVHeads = numKVHeads; %#ok<NASGU>
    hyperParameters.HeadDim = headDim; %#ok<NASGU>
    hyperParameters.RopeTheta = 10000.0; %#ok<NASGU>

    freqsFull = transformer.layer.precomputeFreqsCis(headDim, seqLen + 8, hyperParameters.RopeTheta);
    freqs_cis = complex(single(real(freqsFull(:, 1:seqLen))), single(imag(freqsFull(:, 1:seqLen)))); %#ok<NASGU>

    runtimeCfg = qwen2_runtime.defaultHDLConfig(); %#ok<NASGU>

    clear qwen2_runtime_hdl_block_entry

    [h_out, key_out, value_out] = qwen2_runtime_hdl_block_entry( ...
        h_in, key_in, value_in, cache_valid_len, weights, hyperParameters, freqs_cis, runtimeCfg); %#ok<NASGU>

    final_h_out = h_out; %#ok<NASGU>
    final_key_out = key_out; %#ok<NASGU>
    final_value_out = value_out; %#ok<NASGU>
end

function W = makeMatrix(rows, cols, scale, phase)
    values = sin(single(1:(rows * cols)) + single(phase));
    W = reshape(single(scale) .* single(values), rows, cols);
end