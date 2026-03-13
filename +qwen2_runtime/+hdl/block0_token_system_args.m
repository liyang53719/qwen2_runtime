function args = block0_token_system_args(matParamsFile, maxCacheLen, forCodegen)
%BLOCK0_TOKEN_SYSTEM_ARGS Real-dimension args for block-0 system top.

    if nargin == 0 || strlength(string(matParamsFile)) == 0
        matParamsFile = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'qwen_params.mat');
    end
    if nargin < 2
        maxCacheLen = 8;
    end
    if nargin < 3
        forCodegen = false;
    end

    params = qwen2_runtime.load(matParamsFile, 'PrepareDynamicInt8', false, 'ConvertDLArrayToSingle', true);
    if forCodegen
        cfg = qwen2_runtime.defaultHardwareHDLConfig();
    else
        cfg = qwen2_runtime.defaultHDLConfig();
    end
    cfg.HDLMaxCacheLength = maxCacheLen;
    hp = params.Hyperparameters;
    weights = params.Weights.h0;

    tokenId = 151644;
    h_token = single(params.Weights.embed_tokens(:, tokenId + 1));
    h_token = reshape(h_token, [hp.HiddenSize, 1]);

    freqsFull = transformer.layer.precomputeFreqsCis(hp.HeadDim, double(maxCacheLen) + 8, hp.RopeTheta);
    freqs_cis = struct();
    freqs_cis.Cos = fi(single(real(freqsFull(:, 1:double(maxCacheLen) + 8))), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength);
    freqs_cis.Sin = fi(single(imag(freqsFull(:, 1:double(maxCacheLen) + 8))), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength);

    read_key_data = fi(zeros(hp.HeadDim, hp.NumKVHeads, 1), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength);
    read_value_data = fi(zeros(hp.HeadDim, hp.NumKVHeads, 1), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength);
    h_token_fix = fi(h_token, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength);
    start = false;
    cache_valid_len = uint16(0);
    rope_position = uint16(1);
    read_data_valid = false;

    if forCodegen
        weightsArg = coder.Constant(weights);
        hpArg = coder.Constant(hp);
        freqsArg = coder.Constant(freqs_cis);
        cfgArg = coder.Constant(cfg);
    else
        weightsArg = weights;
        hpArg = hp;
        freqsArg = freqs_cis;
        cfgArg = cfg;
    end

    args = {start, h_token_fix, cache_valid_len, rope_position, read_key_data, read_value_data, read_data_valid, weightsArg, hpArg, freqsArg, cfgArg};
end