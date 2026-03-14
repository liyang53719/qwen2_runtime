function result = validate_block_skeleton_fp32sensitive_vs_runtime_trace()
%VALIDATE_BLOCK_SKELETON_FP32SENSITIVE_VS_RUNTIME_TRACE Compare FP32-sensitive skeleton against runtime trace.

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
    blockTrace = dbg.TensorTrace.blocks{1};
    attnTrace = blockTrace.attention;
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

    F16 = fimath('RoundingMethod', 'Floor', 'OverflowAction', 'Saturate', 'ProductMode', 'SpecifyPrecision', 'ProductWordLength', 24, 'ProductFractionLength', 14, 'SumMode', 'SpecifyPrecision', 'SumWordLength', 16, 'SumFractionLength', 14);
    F32 = fimath('RoundingMethod', 'Floor', 'OverflowAction', 'Saturate', 'ProductMode', 'SpecifyPrecision', 'ProductWordLength', 24, 'ProductFractionLength', 14, 'SumMode', 'SpecifyPrecision', 'SumWordLength', 32, 'SumFractionLength', 14);
    scoreMat = fi(zeros(cacheLen, 2), true, 16, 14, F16);
    valueTensor = fi(zeros(cacheLen, 4, 2), true, 16, 14, F16);
    for h = 1:2
        qv = q(:, h, tokenPos, 1);
        for t = 1:cacheLen
            scoreVal = single(extractScalar(sum(qv .* kRep(:, h, t, 1)) / sqrt(headDim)));
            scoreMat(t, h) = fi(scoreVal, true, 16, 14, F16);
            for lane = 1:4
                valueTensor(t, lane, h) = fi(single(extractScalar(vRep(lane, h, t, 1))), true, 16, 14, F16);
            end
        end
    end

    exactAttn = single([attnTrace.attn_mix(1:4, tokenPos, 1); attnTrace.attn_mix(headDim+(1:4), tokenPos, 1)]);
    expected = qwen2_runtime.hdl.residual_add_step(fi(single(blockTrace.input_norm(1:8, tokenPos, 1)), true, 32, 14, F32), fi(exactAttn, true, 32, 14, F32));

    inputVec = fi(single(blockTrace.input_norm(1:8, tokenPos, 1)), true, 32, 14, F32);
    residualSeed = fi(zeros(8, 1), true, 32, 14, F32);
    dut = directFp32SensitiveBlock(inputVec, scoreMat, valueTensor, F32);
    valid = true;

    diff = abs(double(storedInteger(dut)) - double(storedInteger(expected)));
    result = struct();
    result.Valid = valid;
    result.ExactMatch = isequal(storedInteger(dut), storedInteger(expected));
    result.MaxAbsIntDiff = max(diff(:));
    result.MeanAbsIntDiff = mean(diff(:));
    result.DUT = dut;
    result.Reference = expected;
    fprintf('fp32_sensitive_block_vs_runtime exact   : %d\n', result.ExactMatch);
    fprintf('fp32_sensitive_block_vs_runtime max int : %.6g\n', result.MaxAbsIntDiff);
    fprintf('fp32_sensitive_block_vs_runtime mean int: %.6g\n', result.MeanAbsIntDiff);
end

function x = extractScalar(x)
    if isa(x, 'dlarray')
        x = extractdata(x);
    end
    x = single(x);
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

function dut = directFp32SensitiveBlock(inputVec, scoreMat, valueTensor, F32)
    attnMat = zeros(4, 2, 'single');
    for h = 1:2
        scores = zeros(size(scoreMat, 1), 1, 'single');
        values = zeros(size(valueTensor, 1), 4, 'single');
        for t = 1:size(scoreMat, 1)
            scores(t) = single(scoreMat(t, h));
            for lane = 1:4
                values(t, lane) = single(valueTensor(t, lane, h));
            end
        end
        attnMat(:, h) = qwen2_runtime.hdl.attention_weighted_value_controller_single_step(scores, values);
    end
    attnVec = zeros(8, 1, 'single');
    attnVec(1:4) = attnMat(:, 1);
    attnVec(5:8) = attnMat(:, 2);
    dut = qwen2_runtime.hdl.residual_add_step(inputVec, fi(attnVec, true, 32, 14, F32));
end
