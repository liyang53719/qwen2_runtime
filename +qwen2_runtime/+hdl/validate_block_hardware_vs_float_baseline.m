function result = validate_block_hardware_vs_float_baseline()
%VALIDATE_BLOCK_HARDWARE_VS_FLOAT_BASELINE Compare reduced block hardware baseline against float baseline.

    floatArgs = qwen2_runtime.hdl.block_entry_baseline_args();
    hwArgs = qwen2_runtime.hdl.block_entry_hardware_args();

    floatIn = unwrapArgs(floatArgs);
    hwIn = unwrapArgs(hwArgs);

    past = struct();
    past.keys = [];
    past.values = [];
    [hFloat, presentFloat] = qwen2_runtime.layer.block( ...
        floatIn{1}, past, floatIn{5}, floatIn{6}, floatIn{7}, floatIn{8});
    [hHw, keyHw, valueHw] = qwen2_runtime.hdl.block_entry( ...
        hwIn{1}, hwIn{2}, hwIn{3}, hwIn{4}, hwIn{5}, hwIn{6}, hwIn{7}, hwIn{8});

    keyFloat = padRuntimeCacheLikeHardware(presentFloat.keys, keyHw);
    valueFloat = padRuntimeCacheLikeHardware(presentFloat.values, valueHw);

    hStats = compareNumeric(hFloat, hHw);
    keyStats = compareNumeric(keyFloat, keyHw);
    valueStats = compareNumeric(valueFloat, valueHw);
    ropeStats = compareRoPEOnly(floatIn, hwIn);

    result = struct();
    result.BlockOutput = hStats;
    result.KeyCache = keyStats;
    result.ValueCache = valueStats;
    result.RoPEOnly = ropeStats;

    fprintf('block_h_out max abs    : %.6g\n', hStats.MaxAbs);
    fprintf('block_h_out mean abs   : %.6g\n', hStats.MeanAbs);
    fprintf('block_h_out max rel    : %.6g\n', hStats.MaxRel);
    fprintf('key_cache max abs      : %.6g\n', keyStats.MaxAbs);
    fprintf('value_cache max abs    : %.6g\n', valueStats.MaxAbs);
    fprintf('rope_q max abs         : %.6g\n', ropeStats.Query.MaxAbs);
    fprintf('rope_k max abs         : %.6g\n', ropeStats.Key.MaxAbs);
end

function argsOut = unwrapArgs(argsIn)
    argsOut = argsIn;
    for idx = 1:numel(argsIn)
        if isa(argsIn{idx}, 'coder.Constant')
            argsOut{idx} = argsIn{idx}.Value;
        end
    end
end

function stats = compareNumeric(reference, dut)
    ref = double(reference);
    act = double(dut);
    absDiff = abs(act - ref);
    denom = max(abs(ref), 1.0e-12);

    stats = struct();
    stats.MaxAbs = max(absDiff(:));
    stats.MeanAbs = mean(absDiff(:));
    stats.RMSE = sqrt(mean((act(:) - ref(:)).^2));
    stats.MaxRel = max((absDiff(:) ./ denom(:)));
    stats.Reference = reference;
    stats.DUT = dut;
end

function padded = padRuntimeCacheLikeHardware(runtimeCache, hardwareCache)
    padded = zeros(size(hardwareCache), 'like', double(hardwareCache));
    validLen = min(size(runtimeCache, 3), size(hardwareCache, 3));
    padded(:, :, 1:validLen, :) = double(runtimeCache(:, :, 1:validLen, :));
end

function ropeStats = compareRoPEOnly(floatIn, hwIn)
    hIn = hwIn{1};
    weights = hwIn{5};
    hyper = hwIn{6};
    freqsFixed = hwIn{7};
    freqsFloat = floatIn{7};
    cfg = hwIn{8};

    hiddenSize = hyper.HiddenSize;
    seqLen = size(hIn, 2);
    batchSize = size(hIn, 3);
    headDim = hyper.HeadDim;
    numHeads = hyper.NumHeads;
    numKVHeads = hyper.NumKVHeads;

    hNorm = qwen2_runtime.hdl.rmsnorm_step(hIn, weights.input_layernorm, single(1e-6), cfg);
    X2 = reshape(hNorm, hiddenSize, []);
    xq = qwen2_runtime.hdl.linear_step(weights.self_attn_q_proj, X2, cfg);
    xk = qwen2_runtime.hdl.linear_step(weights.self_attn_k_proj, X2, cfg);

    xq = reshape(xq, [headDim, numHeads, seqLen, batchSize]);
    xk = reshape(xk, [headDim, numKVHeads, seqLen, batchSize]);

    [xqFixed, xkFixed] = applyRoPEFixed(xq, xk, freqsFixed);
    [xqFloat, xkFloat] = applyRoPEFloatReference(xq, xk, freqsFloat);

    ropeStats = struct();
    ropeStats.Query = compareNumeric(xqFloat, xqFixed);
    ropeStats.Key = compareNumeric(xkFloat, xkFixed);
end

function [xqOut, xkOut] = applyRoPEFixed(xq, xk, freqs)
    half = size(xq, 1) / 2;
    seqLen = size(xq, 3);
    cosTheta = reshape(freqs.Cos(:, 1:seqLen), [half, 1, seqLen, 1]);
    sinTheta = reshape(freqs.Sin(:, 1:seqLen), [half, 1, seqLen, 1]);

    xqOut = xq;
    xkOut = xk;

    xq_r = xq(1:half, :, :, :);
    xq_i = xq(half+1:end, :, :, :);
    xk_r = xk(1:half, :, :, :);
    xk_i = xk(half+1:end, :, :, :);

    xqOut(1:half, :, :, :) = xq_r .* cosTheta - xq_i .* sinTheta;
    xqOut(half+1:end, :, :, :) = xq_r .* sinTheta + xq_i .* cosTheta;
    xkOut(1:half, :, :, :) = xk_r .* cosTheta - xk_i .* sinTheta;
    xkOut(half+1:end, :, :, :) = xk_r .* sinTheta + xk_i .* cosTheta;
end

function [xqOut, xkOut] = applyRoPEFloatReference(xq, xk, freqs)
    F = fimath(xq);

    xqSingle = single(xq);
    xkSingle = single(xk);
    [xqOutSingle, xkOutSingle] = transformer.layer.RoPE(xqSingle, xkSingle, freqs);

    xqOut = fi(xqOutSingle, true, xq.WordLength, xq.FractionLength, F);
    xkOut = fi(xkOutSingle, true, xk.WordLength, xk.FractionLength, F);
end