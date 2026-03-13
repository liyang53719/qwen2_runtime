function result = try_real_block0_forward_with_system_attention(inputIds, maxCacheLen)
%TRY_REAL_BLOCK0_FORWARD_WITH_SYSTEM_ATTENTION Attempt one real block-0 forward pass with the system attention controller.

    if nargin < 1 || isempty(inputIds)
        inputIds = 151644;
    end
    if nargin < 2
        maxCacheLen = 8;
    end

    projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    paramsFile = fullfile(projectRoot, 'qwen_params.mat');
    params = qwen2_runtime.load(paramsFile, 'PrepareDynamicInt8', false, 'ConvertDLArrayToSingle', true);

    hp = params.Hyperparameters;
    layer0 = params.Weights.h0;
    tokenId = double(inputIds(1));
    hiddenIn = reshape(single(params.Weights.embed_tokens(:, tokenId + 1)), [hp.HiddenSize, 1, 1]);

    cfgFloat = qwen2_runtime.defaultRuntimeConfig();
    cfgFloat.LinearMode = 'float';
    cfgSys = qwen2_runtime.defaultHDLConfig();
    cfgSys.UseExternalWeightMemory = true;
    cfgSys.UseExternalKVMemory = true;
    cfgSys.SystemAttentionKernel = 'qwen2_runtime.hdl.attention_token_step_sram_step';
    cfgSys.SystemKVInterfaceKernel = 'qwen2_runtime.hdl.attention_token_step_sram_contract_step';
    cfgSys.SystemAttentionControllerKernel = 'qwen2_runtime.hdl.attention_token_controller_sram_step';

    freqsFull = transformer.layer.precomputeFreqsCis(hp.HeadDim, double(maxCacheLen) + 8, hp.RopeTheta);
    freqsRef = complex(single(real(freqsFull(:, 1))), single(imag(freqsFull(:, 1))));
    freqsSys = struct();
    freqsSys.Cos = fi(single(real(freqsFull(:, 1:double(maxCacheLen) + 8))), true, cfgSys.HDLLinearInputWordLength, cfgSys.HDLLinearInputFractionLength);
    freqsSys.Sin = fi(single(imag(freqsFull(:, 1:double(maxCacheLen) + 8))), true, cfgSys.HDLLinearInputWordLength, cfgSys.HDLLinearInputFractionLength);

    hiddenNorm = transformer.layer.rmsNormalization(hiddenIn, single(layer0.input_layernorm), 1e-6);
    hiddenNormFix = fi(single(hiddenNorm), true, cfgSys.HDLLinearInputWordLength, cfgSys.HDLLinearInputFractionLength);
    keyCacheIn = fi(zeros(hp.HeadDim, hp.NumKVHeads, maxCacheLen, 1), true, cfgSys.HDLLinearInputWordLength, cfgSys.HDLLinearInputFractionLength);
    valueCacheIn = fi(zeros(hp.HeadDim, hp.NumKVHeads, maxCacheLen, 1), true, cfgSys.HDLLinearInputWordLength, cfgSys.HDLLinearInputFractionLength);

    attnWeights = struct();
    attnWeights.q_proj = layer0.self_attn_q_proj;
    attnWeights.k_proj = layer0.self_attn_k_proj;
    attnWeights.v_proj = layer0.self_attn_v_proj;
    attnWeights.o_proj = layer0.self_attn_o_proj;
    if isfield(layer0, 'self_attn_q_bias'), attnWeights.q_bias = single(layer0.self_attn_q_bias); end
    if isfield(layer0, 'self_attn_k_bias'), attnWeights.k_bias = single(layer0.self_attn_k_bias); end
    if isfield(layer0, 'self_attn_v_bias'), attnWeights.v_bias = single(layer0.self_attn_v_bias); end
    if isfield(layer0, 'self_attn_o_bias'), attnWeights.o_bias = single(layer0.self_attn_o_bias); end

    [attnProj, keyCacheOut, valueCacheOut, nextValidLen, readEnable, readAddr, writeEnable, writeAddr, shiftEnable] = ...
        qwen2_runtime.hdl.attention_token_controller_sram_step( ...
            hiddenNormFix, keyCacheIn, valueCacheIn, uint16(0), uint16(1), attnWeights, freqsSys, hp, cfgSys);

    attnProjFloat = reshape(single(attnProj), [hp.HiddenSize, 1, 1]);
    hiddenAfterAttn = single(hiddenIn) + attnProjFloat;
    postNorm = transformer.layer.rmsNormalization(hiddenAfterAttn, single(layer0.post_attention_layernorm), 1e-6);

    mlpWeights = struct();
    mlpWeights.gate_proj = layer0.mlp_gate_proj;
    mlpWeights.up_proj = layer0.mlp_up_proj;
    mlpWeights.down_proj = layer0.mlp_down_proj;
    [ffnOut, ~] = qwen2_runtime.layer.gatedMLP(postNorm, mlpWeights, cfgFloat);
    blockOut = hiddenAfterAttn + single(ffnOut);

    past = struct('keys', [], 'values', []);
    [attnRef, ~, ~] = qwen2_runtime.layer.attentionGQA(hiddenNorm, past, attnWeights, freqsRef, hp, cfgFloat);
    [blockRef, presentRef, ~] = qwen2_runtime.layer.block(hiddenIn, past, layer0, hp, freqsRef, cfgFloat);

    attnDiff = abs(single(attnProjFloat) - single(attnRef));
    blockDiff = abs(single(blockOut) - single(blockRef));

    result = struct();
    result.InputTokenId = tokenId;
    result.AttentionMaxAbsDiff = max(attnDiff(:));
    result.AttentionMeanAbsDiff = mean(attnDiff(:));
    result.BlockMaxAbsDiff = max(blockDiff(:));
    result.BlockMeanAbsDiff = mean(blockDiff(:));
    result.NextValidLen = double(nextValidLen);
    result.ReadCount = sum(readEnable);
    result.LastReadAddr = double(max(readAddr));
    result.WriteEnable = logical(writeEnable);
    result.WriteAddr = double(writeAddr);
    result.ShiftEnable = logical(shiftEnable);
    result.KeyCacheOutSize = size(keyCacheOut);
    result.ValueCacheOutSize = size(valueCacheOut);
    result.ReferenceCacheSize = size(presentRef.keys);

    fprintf('real_block0_attention max abs diff : %.6g\n', result.AttentionMaxAbsDiff);
    fprintf('real_block0_attention mean abs     : %.6g\n', result.AttentionMeanAbsDiff);
    fprintf('real_block0_block max abs diff     : %.6g\n', result.BlockMaxAbsDiff);
    fprintf('real_block0_block mean abs         : %.6g\n', result.BlockMeanAbsDiff);
    fprintf('real_block0_next_valid_len         : %d\n', result.NextValidLen);
    fprintf('real_block0_read_count             : %d\n', result.ReadCount);
    fprintf('real_block0_write_addr             : %d\n', result.WriteAddr);
    fprintf('real_block0_shift_enable           : %d\n', result.ShiftEnable);
end