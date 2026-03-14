function args = attention_token_step_sram_handshake_args(maxCacheLen, forCodegen)
%ATTENTION_TOKEN_STEP_SRAM_HANDSHAKE_ARGS Representative args for handshake attention top.

    if nargin == 0
        maxCacheLen = qwen2_runtime.defaultHardwareHDLConfig().HDLMaxCacheLength;
    end
    if nargin < 2
        forCodegen = false;
    end

    baseArgs = qwen2_runtime.hdl.attention_token_step_sram_args(maxCacheLen, forCodegen);
    headDim = size(baseArgs{2}, 1);
    numKVHeads = size(baseArgs{2}, 2);
    F = fimath(baseArgs{1});
    start = false;
    read_key_data = fi(zeros(headDim, numKVHeads, 1), true, baseArgs{1}.WordLength, baseArgs{1}.FractionLength, F);
    read_value_data = fi(zeros(headDim, numKVHeads, 1), true, baseArgs{1}.WordLength, baseArgs{1}.FractionLength, F);
    read_data_valid = false;

    args = {start, baseArgs{1}, baseArgs{2}, baseArgs{3}, baseArgs{6}, baseArgs{7}, read_key_data, read_value_data, read_data_valid, baseArgs{8}, baseArgs{9}, baseArgs{10}};
end