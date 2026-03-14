function result = diagnose_single_sensitive_gap()
%DIAGNOSE_SINGLE_SENSITIVE_GAP Check if single-precision sensitive path closes the runtime gap.

    projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    paramsFile = fullfile(projectRoot, 'qwen_params.mat');
    tokenizerPath = fullfile(projectRoot, 'qwen_model');

    promptText = "The process of photosynthesis is how plants convert light energy into chemical energy. Chlorophyll absorbs sunlight and uses it to convert carbon dioxide and water into glucose.";
    prompt = sprintf("Summarize this: %s Summary:", promptText);

    params = qwen2.load(paramsFile);
    cfg = qwen2_runtime.defaultRuntimeConfig();
    cfg.LinearMode = 'float';
    cfg.TraceTensors = true;
    params.RuntimeConfig = cfg;

    tokenizer = qwen2.tokenizer.QwenTokenizer(tokenizerPath);
    inputIds = double(tokenizer.encode(prompt));
    if size(inputIds, 1) > size(inputIds, 2)
        inputIds = inputIds';
    end

    [~, ~, dbg] = qwen2_runtime.model(inputIds, params, [], struct('RuntimeConfig', cfg));
    hp = params.Hyperparameters;
    headDim = hp.HeadDim;
    seqLen = numel(inputIds);
    cacheLen = min(16, seqLen);
    tokenPos = cacheLen;
    attnTrace = dbg.TensorTrace.blocks{1}.attention;
    weights = params.Weights.h0;

    q = single(attnTrace.q_proj);
    k = single(attnTrace.k_proj);
    v = single(attnTrace.v_proj);
    q = addBias(q, weights, 'self_attn_q_bias');
    k = addBias(k, weights, 'self_attn_k_bias');
    v = addBias(v, weights, 'self_attn_v_bias');

    q = reshape(q, headDim, hp.NumHeads, seqLen, 1);
    k = reshape(k, headDim, hp.NumKVHeads, seqLen, 1);
    v = reshape(v, headDim, hp.NumKVHeads, seqLen, 1);
    freqs = transformer.layer.precomputeFreqsCis(headDim, seqLen, hp.RopeTheta);
    freqs = complex(single(real(freqs)), single(imag(freqs)));
    [q, k] = transformer.layer.RoPE(q, k, freqs(:, 1:seqLen));
    kRep = repeatKVHeads(k, hp.NumHeads / hp.NumKVHeads);
    vRep = repeatKVHeads(v, hp.NumHeads / hp.NumKVHeads);

    runtimeMix = single([attnTrace.attn_mix(1:4, tokenPos, 1); attnTrace.attn_mix(headDim+(1:4), tokenPos, 1)]);
    singleMix = zeros(8, 1, 'single');
    for h = 1:2
        qv = q(:, h, tokenPos, 1);
        scores = zeros(cacheLen, 1, 'single');
        values4 = zeros(cacheLen, 4, 'single');
        for t = 1:cacheLen
            scores(t) = single(sum(qv .* kRep(:, h, t, 1)) / sqrt(headDim));
            for lane = 1:4
                values4(t, lane) = single(vRep(lane, h, t, 1));
            end
        end
        singleMix((h-1)*4 + (1:4)) = qwen2_runtime.hdl.attention_weighted_value_controller_single_step(scores, values4);
    end

    result = struct();
    result.RuntimeMix = runtimeMix;
    result.SingleSensitiveMix = singleMix;
    result.MaxAbsDiff = max(abs(singleMix - runtimeMix));
    result.MeanAbsDiff = mean(abs(singleMix - runtimeMix));
    fprintf('single_sensitive_vs_runtime max=%g mean=%g\n', result.MaxAbsDiff, result.MeanAbsDiff);
end

function X = addBias(X, weights, name)
    if isfield(weights, name)
        X = X + reshape(single(weights.(name)), [], 1, 1);
    end
end

function Y = repeatKVHeads(X, nRep)
    if nRep == 1
        Y = X;
        return;
    end
    [headDim, numKVHeads, seqLen, batchSize] = size(X);
    Y = zeros(headDim, numKVHeads * nRep, seqLen, batchSize, 'like', X);
    for kv = 1:numKVHeads
        base = (kv - 1) * nRep;
        for rep = 1:nRep
            Y(:, base + rep, :, :) = X(:, kv, :, :);
        end
    end
end
