function info = generate_attention_token_controller_sram_handshake_baseline(matParamsFile, maxCacheLen)
%GENERATE_ATTENTION_TOKEN_CONTROLLER_SRAM_HANDSHAKE_BASELINE Generate real-dimension RTL for handshake attention controller.

    if nargin < 1 || strlength(string(matParamsFile)) == 0
        matParamsFile = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'qwen_params.mat');
    end
    if nargin < 2
        maxCacheLen = 8;
    end

    projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    cfg = coder.config('hdl');
    cfg.TargetLanguage = 'Verilog';
    cfg.GenerateHDLTestBench = false;
    cfg.GenerateCosimTestBench = false;
    cfg.SynthesizeGeneratedCode = false;
    cfg.Workflow = 'Generic ASIC/FPGA';
    cfg.InstantiateFunctions = false;

    outDir = fullfile(projectRoot, 'codegen', 'hdl_attention_token_controller_sram_handshake', 'qwen2_runtime_hdl_attention_token_controller_sram_handshake_entry');
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    params = qwen2_runtime.load(matParamsFile, 'PrepareDynamicInt8', false, 'ConvertDLArrayToSingle', true);
    hp = params.Hyperparameters;
    cfgRun = qwen2_runtime.defaultHardwareHDLConfig();
    cfgRun.HDLMaxCacheLength = maxCacheLen;
    layer0 = params.Weights.h0;
    tokenId = 151644;
    h_token = fi(single(params.Weights.embed_tokens(:, tokenId + 1)), true, cfgRun.HDLLinearInputWordLength, cfgRun.HDLLinearInputFractionLength);
    read_key_data = fi(zeros(hp.HeadDim, hp.NumKVHeads, 1), true, cfgRun.HDLLinearInputWordLength, cfgRun.HDLLinearInputFractionLength);
    read_value_data = fi(zeros(hp.HeadDim, hp.NumKVHeads, 1), true, cfgRun.HDLLinearInputWordLength, cfgRun.HDLLinearInputFractionLength);
    freqsFull = transformer.layer.precomputeFreqsCis(hp.HeadDim, double(maxCacheLen) + 8, hp.RopeTheta);
    freqs_cis = struct();
    freqs_cis.Cos = fi(single(real(freqsFull(:, 1:double(maxCacheLen) + 8))), true, cfgRun.HDLLinearInputWordLength, cfgRun.HDLLinearInputFractionLength);
    freqs_cis.Sin = fi(single(imag(freqsFull(:, 1:double(maxCacheLen) + 8))), true, cfgRun.HDLLinearInputWordLength, cfgRun.HDLLinearInputFractionLength);
    controllerWeights = struct();
    controllerWeights.q_proj = layer0.self_attn_q_proj;
    controllerWeights.k_proj = layer0.self_attn_k_proj;
    controllerWeights.v_proj = layer0.self_attn_v_proj;
    controllerWeights.o_proj = layer0.self_attn_o_proj;
    if isfield(layer0, 'self_attn_q_bias'), controllerWeights.q_bias = layer0.self_attn_q_bias; end
    if isfield(layer0, 'self_attn_k_bias'), controllerWeights.k_bias = layer0.self_attn_k_bias; end
    if isfield(layer0, 'self_attn_v_bias'), controllerWeights.v_bias = layer0.self_attn_v_bias; end
    if isfield(layer0, 'self_attn_o_bias'), controllerWeights.o_bias = layer0.self_attn_o_bias; end
    controllerArgs = {false, h_token, uint16(0), uint16(1), read_key_data, read_value_data, false, coder.Constant(controllerWeights), coder.Constant(freqs_cis), coder.Constant(hp), coder.Constant(cfgRun)};
    codegen('-config', cfg, 'qwen2_runtime_hdl_attention_token_controller_sram_handshake_entry', '-args', controllerArgs, '-d', outDir, '-I', projectRoot);

    info = struct();
    info.OutputDir = outDir;
    info.MaxCacheLength = maxCacheLen;
end