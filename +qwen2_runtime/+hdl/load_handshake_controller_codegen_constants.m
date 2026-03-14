function [weights, freqs_cis, hyperParameters, cfg] = load_handshake_controller_codegen_constants()
%LOAD_HANDSHAKE_CONTROLLER_CODEGEN_CONSTANTS Compile-time constants for frozen handshake controller entry.

    compact = coder.load('qwen_handshake_controller_params_v73.mat');

    cfg = qwen2_runtime.defaultHardwareHDLConfig();
    if isfield(compact, 'FreqsMaxCacheLength')
        cfg.HDLMaxCacheLength = double(compact.FreqsMaxCacheLength);
    end

    hyperParameters = compact.Hyperparameters;
    weights = compact.ControllerWeights;

    expectedCols = double(cfg.HDLMaxCacheLength) + 8;
    ntInput = numerictype(true, 16, 14);
    freqs_cis = struct();
    freqs_cis.Cos = fi(single(compact.FreqsCos(:, 1:expectedCols)), ntInput);
    freqs_cis.Sin = fi(single(compact.FreqsSin(:, 1:expectedCols)), ntInput);
end