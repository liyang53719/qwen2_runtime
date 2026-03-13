function result = validate_block0_system_nonempty_cache(inputIds, targetIndex, maxCacheLen)
%VALIDATE_BLOCK0_SYSTEM_NONEMPTY_CACHE Validate block-0 system top with a non-empty external KV cache.

    if nargin < 1 || isempty(inputIds)
        inputIds = [151644 9707 25];
    end
    if nargin < 2
        targetIndex = 2;
    end
    if nargin < 3
        maxCacheLen = 8;
    end

    projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    params = qwen2_runtime.load(fullfile(projectRoot, 'qwen_params.mat'), 'PrepareDynamicInt8', false, 'ConvertDLArrayToSingle', true);
    hp = params.Hyperparameters;
    layer0 = params.Weights.h0;
    cfgFloat = qwen2_runtime.defaultRuntimeConfig();
    cfgFloat.LinearMode = 'float';
    cfgSys = qwen2_runtime.defaultHDLConfig();
    cfgSys.HDLMaxCacheLength = maxCacheLen;
    cfgSys.UseExternalWeightMemory = true;
    cfgSys.UseExternalKVMemory = true;

    freqsFull = transformer.layer.precomputeFreqsCis(hp.HeadDim, max(numel(inputIds), maxCacheLen) + 8, hp.RopeTheta);
    past = struct('keys', [], 'values', []);
    for idx = 1:targetIndex-1
        hiddenPrev = reshape(single(params.Weights.embed_tokens(:, double(inputIds(idx)) + 1)), [hp.HiddenSize, 1, 1]);
        freqPrev = complex(single(real(freqsFull(:, idx))), single(imag(freqsFull(:, idx))));
        [~, past] = qwen2_runtime.layer.block(hiddenPrev, past, layer0, hp, freqPrev, cfgFloat);
    end

    hiddenTarget = reshape(single(params.Weights.embed_tokens(:, double(inputIds(targetIndex)) + 1)), [hp.HiddenSize, 1]);
    freqTarget = complex(single(real(freqsFull(:, targetIndex))), single(imag(freqsFull(:, targetIndex))));
    [blockRef, ~, ~] = qwen2_runtime.layer.block(reshape(hiddenTarget, [hp.HiddenSize, 1, 1]), past, layer0, hp, freqTarget, cfgFloat);

    freqsSys = struct();
    freqsSys.Cos = fi(single(real(freqsFull(:, 1:maxCacheLen + 8))), true, cfgSys.HDLLinearInputWordLength, cfgSys.HDLLinearInputFractionLength);
    freqsSys.Sin = fi(single(imag(freqsFull(:, 1:maxCacheLen + 8))), true, cfgSys.HDLLinearInputWordLength, cfgSys.HDLLinearInputFractionLength);
    readKeyZero = fi(zeros(hp.HeadDim, hp.NumKVHeads, 1), true, cfgSys.HDLLinearInputWordLength, cfgSys.HDLLinearInputFractionLength);
    readValueZero = fi(zeros(hp.HeadDim, hp.NumKVHeads, 1), true, cfgSys.HDLLinearInputWordLength, cfgSys.HDLLinearInputFractionLength);

    cacheLen = min(size(past.keys, 3), maxCacheLen);
    readValid = false;
    readKey = readKeyZero;
    readValue = readValueZero;
    pendingAddr = 0;
    maxCycles = 4 * max(cacheLen, 1) + 16;
    blockOut = fi(zeros(hp.HiddenSize, 1), true, cfgSys.HDLLinearInputWordLength, cfgSys.HDLLinearInputFractionLength);
    outValid = false;
    for cyc = 1:maxCycles
        start = (cyc == 1);
        [blockOut, outValid, busy, readReq, readAddr, writeReq, writeAddr, shiftEnable, writeKeyToken, writeValueToken, nextValidLen] = ...
            qwen2_runtime.hdl.block0_token_system_step( ...
                start, fi(hiddenTarget, true, cfgSys.HDLLinearInputWordLength, cfgSys.HDLLinearInputFractionLength), uint16(cacheLen), uint16(targetIndex), readKey, readValue, readValid, layer0, hp, freqsSys, cfgSys);
        if outValid
            break;
        end

        readValid = false;
        readKey = readKeyZero;
        readValue = readValueZero;
        if pendingAddr >= 1 && pendingAddr <= cacheLen
            readKey = fi(single(past.keys(:, :, pendingAddr, 1)), true, cfgSys.HDLLinearInputWordLength, cfgSys.HDLLinearInputFractionLength);
            readValue = fi(single(past.values(:, :, pendingAddr, 1)), true, cfgSys.HDLLinearInputWordLength, cfgSys.HDLLinearInputFractionLength);
            readValid = true;
        end
        pendingAddr = 0;
        if busy && readReq
            pendingAddr = double(readAddr);
        end
    end

    diff = abs(single(blockOut) - single(blockRef(:)));
    result = struct();
    result.InputIds = inputIds;
    result.TargetIndex = targetIndex;
    result.CacheLength = cacheLen;
    result.OutValid = logical(outValid);
    result.BlockMaxAbsDiff = max(diff(:));
    result.BlockMeanAbsDiff = mean(diff(:));
    result.WriteReq = logical(writeReq);
    result.WriteAddr = double(writeAddr);
    result.ShiftEnable = logical(shiftEnable);
    result.NextValidLen = double(nextValidLen);
    result.WriteKeyTokenSize = size(writeKeyToken);
    result.WriteValueTokenSize = size(writeValueToken);

    fprintf('block0_nonempty_cache target index   : %d\n', result.TargetIndex);
    fprintf('block0_nonempty_cache cache length   : %d\n', result.CacheLength);
    fprintf('block0_nonempty_cache max abs diff   : %.6g\n', result.BlockMaxAbsDiff);
    fprintf('block0_nonempty_cache mean abs diff  : %.6g\n', result.BlockMeanAbsDiff);
    fprintf('block0_nonempty_cache out valid      : %d\n', result.OutValid);
    fprintf('block0_nonempty_cache write addr     : %d\n', result.WriteAddr);
    fprintf('block0_nonempty_cache next valid len : %d\n', result.NextValidLen);
end