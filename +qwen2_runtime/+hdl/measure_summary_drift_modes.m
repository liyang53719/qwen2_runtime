function result = measure_summary_drift_modes()
%MEASURE_SUMMARY_DRIFT_MODES Compare approximate vs half-sensitive softmax in hybrid generation.

    result = struct();
    result.Approx = runHybrid("approx");
    result.Half = runHybrid("half");
end

function out = runHybrid(mode)
    projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    paramsFile = fullfile(projectRoot, 'qwen_params.mat');
    tokenizerPath = fullfile(projectRoot, 'qwen_model');

    promptText = "The process of photosynthesis is how plants convert light energy into chemical energy. Chlorophyll absorbs sunlight and uses it to convert carbon dioxide and water into glucose.";
    prompt = sprintf("Summarize this: %s Summary:", promptText);
    maxNewTokens = 30;

    params = qwen2.load(paramsFile);
    cfg = qwen2_runtime.defaultRuntimeConfig();
    cfg.LinearMode = 'float';
    params.RuntimeConfig = cfg;
    tokenizer = qwen2.tokenizer.QwenTokenizer(tokenizerPath);

    inputIds = double(tokenizer.encode(prompt));
    if size(inputIds, 1) > size(inputIds, 2)
        inputIds = inputIds';
    end

    refIds = referenceGenerateIds(params, inputIds, maxNewTokens);
    hybridIds = hybridGenerateIds(params, inputIds, maxNewTokens, mode);

    out = struct();
    out.Mode = mode;
    out.ReferenceGeneratedIds = refIds(numel(inputIds)+1:end);
    out.HybridGeneratedIds = hybridIds(numel(inputIds)+1:end);
    out.ReferenceText = string(tokenizer.decode(out.ReferenceGeneratedIds));
    out.HybridText = string(tokenizer.decode(out.HybridGeneratedIds));
    out.TokenIdsMatch = isequal(refIds, hybridIds);
    out.TextMatch = strcmp(out.ReferenceText, out.HybridText);
    fprintf('mode=%s ids=%d text=%d\n', mode, out.TokenIdsMatch, out.TextMatch);
    fprintf('reference drift summary (%s): %s\n', mode, out.ReferenceText);
    fprintf('hybrid drift summary    (%s): %s\n', mode, out.HybridText);
end

function ids = referenceGenerateIds(params, inputIds, maxNewTokens)
    [logits, state] = qwen2_runtime.model(inputIds, params, []);
    nextId = greedyNext(logits);
    ids = [inputIds, nextId];
    for i = 1:maxNewTokens
        if isStop(nextId)
            break;
        end
        [logits, state] = qwen2_runtime.model(nextId, params, state);
        nextId = greedyNext(logits);
        ids = [ids, nextId]; %#ok<AGROW>
    end
end

function ids = hybridGenerateIds(params, inputIds, maxNewTokens, mode)
    [logits, state] = hybridModel(inputIds, params, [], mode);
    nextId = greedyNext(logits);
    ids = [inputIds, nextId];
    for i = 1:maxNewTokens
        if isStop(nextId)
            break;
        end
        [logits, state] = hybridModel(nextId, params, state, mode);
        nextId = greedyNext(logits);
        ids = [ids, nextId]; %#ok<AGROW>
    end
end

function [logits, layerStates] = hybridModel(X, params, layerStates, mode)
    if isempty(layerStates)
        layerStates = cell(params.Hyperparameters.NumLayers, 1);
    end
    hp = params.Hyperparameters;
    weights = params.Weights;
    numLayers = hp.NumLayers;
    headDim = hp.HeadDim;
    ropeTheta = hp.RopeTheta;

    if ismatrix(X) && size(X, 1) == 1
        [~, seqLen] = size(X);
        batchSize = 1;
    else
        [~, seqLen, batchSize] = size(X);
        X = reshape(X, 1, []);
    end

    idx = double(X) + 1;
    Z = single(weights.embed_tokens(:, idx));
    Z = reshape(Z, [], seqLen, batchSize);

    startPos = 1;
    if ~isempty(layerStates) && ~isempty(layerStates{1})
        startPos = size(layerStates{1}.keys, 3) + 1;
    end
    maxSeqLen = startPos + seqLen + 128;
    freqs = transformer.layer.precomputeFreqsCis(headDim, maxSeqLen, ropeTheta);
    freqs = complex(single(real(freqs)), single(imag(freqs)));
    currentFreqs = freqs(:, startPos:startPos+seqLen-1);

    for i = 1:numLayers
        layerName = sprintf('h%d', i-1);
        if seqLen == 1
            [Z, newState] = hybridBlockDecode(Z, layerStates{i}, weights.(layerName), hp, currentFreqs, mode);
        else
            [Z, newState] = qwen2_runtime.layer.block(Z, layerStates{i}, weights.(layerName), hp, currentFreqs, params.RuntimeConfig);
        end
        layerStates{i} = newState;
    end

    Z = transformer.layer.rmsNormalization(Z, single(weights.norm), 1e-6);
    Zflat = reshape(Z, size(Z, 1), []);
    logits = single(weights.lm_head) * Zflat;
    logits = reshape(logits, size(weights.lm_head, 1), seqLen, batchSize);
end

function [h, present] = hybridBlockDecode(h, past, weights, hp, freqs, mode)
    resid = h;
    hNorm = transformer.layer.rmsNormalization(h, single(weights.input_layernorm), 1e-6);

    attnWeights.q_proj = weights.self_attn_q_proj;
    attnWeights.k_proj = weights.self_attn_k_proj;
    attnWeights.v_proj = weights.self_attn_v_proj;
    attnWeights.o_proj = weights.self_attn_o_proj;
    if isfield(weights, 'self_attn_q_bias'), attnWeights.q_bias = single(weights.self_attn_q_bias); end
    if isfield(weights, 'self_attn_k_bias'), attnWeights.k_bias = single(weights.self_attn_k_bias); end
    if isfield(weights, 'self_attn_v_bias'), attnWeights.v_bias = single(weights.self_attn_v_bias); end
    if isfield(weights, 'self_attn_o_bias'), attnWeights.o_bias = single(weights.self_attn_o_bias); end

    traceCfg = struct('LinearMode', 'float', 'TraceTensors', true);
    [hAttn, present, attnTrace] = qwen2_runtime.layer.attentionGQA(hNorm, past, attnWeights, freqs, hp, traceCfg);
    hAfterAttn = resid + hAttn;
    approxVec = buildApproxVec(resid, attnTrace, weights, hp, freqs, present, mode);
    hAfterAttn(1:8, 1, 1) = approxVec;

    resid = hAfterAttn;
    hPost = transformer.layer.rmsNormalization(hAfterAttn, single(weights.post_attention_layernorm), 1e-6);
    ffnWeights.gate_proj = weights.mlp_gate_proj;
    ffnWeights.up_proj = weights.mlp_up_proj;
    ffnWeights.down_proj = weights.mlp_down_proj;
    [hFfn, ~] = qwen2_runtime.layer.gatedMLP(hPost, ffnWeights, traceCfg);
    h = resid + hFfn;
end

function approxVec = buildApproxVec(resid, attnTrace, weights, hp, freqs, present, mode)
    headDim = hp.HeadDim;
    totalLen = size(present.keys, 3);
    cacheLen = min(16, totalLen);
    startIdx = totalLen - cacheLen + 1;

    q = single(attnTrace.q_proj);
    k = single(attnTrace.k_proj);
    v = single(attnTrace.v_proj);
    q = addBias(q, weights, 'self_attn_q_bias');
    k = addBias(k, weights, 'self_attn_k_bias');
    v = addBias(v, weights, 'self_attn_v_bias');

    q = reshape(q, headDim, hp.NumHeads, 1, 1);
    kCur = reshape(k, headDim, hp.NumKVHeads, 1, 1);
    vCur = reshape(v, headDim, hp.NumKVHeads, 1, 1);
    [q, kCur] = transformer.layer.RoPE(q, kCur, freqs);

    keys = present.keys;
    values = present.values;
    keys(:, :, end, :) = kCur;
    values(:, :, end, :) = vCur;
    keys = keys(:, :, startIdx:end, :);
    values = values(:, :, startIdx:end, :);
    keys = repeatKVHeads(keys, hp.NumHeads / hp.NumKVHeads);
    values = repeatKVHeads(values, hp.NumHeads / hp.NumKVHeads);

    approxVec = zeros(8, 1, 'single');
    for h = 1:2
        qv = q(:, h, 1, 1);
        scores = zeros(cacheLen, 1, 'single');
        for t = 1:cacheLen
            scores(t) = single(sum(qv .* keys(:, h, t, 1)) / sqrt(headDim));
        end
        scoreMax = max(scores);
        expVals = zeros(cacheLen, 1, 'single');
        for t = 1:cacheLen
            if mode == "half"
                expVals(t) = double(qwen2_runtime.hdl.softmax_exp_half_step(half(scores(t)), half(scoreMax)));
            else
                expVals(t) = double(qwen2_runtime.hdl.softmax_exp_step(fi(scores(t), true, 16, 14), fi(scoreMax, true, 16, 14)));
            end
        end
        if mode == "half"
            approxVec((h-1)*4 + (1:4)) = runHalfWeightedValue(scores, values(:, h, 1:cacheLen, 1), cacheLen);
        else
            recip = double(qwen2_runtime.hdl.softmax_recip_lookup_step(fi(sum(expVals), true, 16, 14)));
            for lane = 1:4
                vals = squeeze(values(lane, h, 1:cacheLen, 1));
                acc = 0;
                for t = 1:cacheLen
                    w = double(qwen2_runtime.hdl.softmax_normalize_step(fi(expVals(t), true, 16, 14), fi(recip, true, 16, 14)));
                    acc = acc + w * double(vals(t));
                end
                approxVec((h-1)*4 + lane) = single(acc);
            end
        end
    end

    F32 = fimath('RoundingMethod', 'Floor', 'OverflowAction', 'Saturate', 'ProductMode', 'SpecifyPrecision', 'ProductWordLength', 24, 'ProductFractionLength', 14, 'SumMode', 'SpecifyPrecision', 'SumWordLength', 32, 'SumFractionLength', 14);
    clear qwen2_runtime.hdl.block_skeleton_step
    inputVec = fi(single(extractScalar(resid(1:8, 1, 1))), true, 32, 14, F32);
    residualSeed = fi(zeros(8, 1), true, 32, 14, F32);
    scoreMat = fi(zeros(cacheLen, 2), true, 16, 14);
    valueTensor = fi(zeros(cacheLen, 4, 2), true, 16, 14);
    for h = 1:2
        qv = q(:, h, 1, 1);
        for t = 1:cacheLen
            scoreMat(t, h) = fi(single(extractScalar(sum(qv .* keys(:, h, t, 1)) / sqrt(headDim))), true, 16, 14);
            for lane = 1:4
                valueTensor(t, lane, h) = fi(single(extractScalar(values(lane, h, t, 1))), true, 16, 14);
            end
        end
    end
    totalCycles = 2 * 3 * cacheLen * 4 + 1;
    for cyc = 1:totalCycles
        [dut, ~] = qwen2_runtime.hdl.block_skeleton_step(cyc == 1, inputVec, scoreMat, valueTensor, residualSeed);
    end
    approxVec = single(dut);
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

function nextId = greedyNext(logits)
    [~, idx] = max(logits(:, end, 1), [], 1);
    nextId = double(idx) - 1;
end

function tf = isStop(token)
    tf = token == 151643 || token == 151645;
end

function x = extractScalar(x)
    if isa(x, 'dlarray')
        x = extractdata(x);
    end
    x = single(x);
end

function out4 = runHalfWeightedValue(scores, valuesHead, cacheLen)
    clear qwen2_runtime.hdl.softmax_exp_half_step qwen2_runtime.hdl.softmax_recip_half_step qwen2_runtime.hdl.softmax_normalize_half_step
    out4 = zeros(4, 1, 'single');
    scoreProxy = half(zeros(cacheLen, 1));
    scoreMax = half(-8);
    for t = 1:cacheLen
        scoreProxy(t) = half(scores(t));
        if scoreProxy(t) > scoreMax
            scoreMax = scoreProxy(t);
        end
    end
    expVals = half(zeros(cacheLen, 1));
    expSum = half(0);
    for t = 1:cacheLen
        expVals(t) = qwen2_runtime.hdl.softmax_exp_half_step(scoreProxy(t), scoreMax);
        expSum = expSum + expVals(t);
    end
    recip = qwen2_runtime.hdl.softmax_recip_half_step(expSum);
    for lane = 1:4
        acc = half(0);
        for t = 1:cacheLen
            valueNow = half(single(extractScalar(valuesHead(lane, 1, t, 1))));
            weightNow = qwen2_runtime.hdl.softmax_normalize_half_step(expVals(t), recip);
            acc = acc + weightNow .* valueNow;
        end
        out4(lane) = single(acc);
    end
end
