function [h_out, key_out, value_out] = block_entry(h_in, key_in, value_in, cache_valid_len, weights, hyperParameters, freqs_cis, runtimeCfg)
%BLOCK_ENTRY HDL-oriented single block entry point.

    arguments
        h_in {mustBeNumeric}
        key_in {mustBeNumeric}
        value_in {mustBeNumeric}
        cache_valid_len (1,1) double
        weights struct
        hyperParameters struct
        freqs_cis
        runtimeCfg struct
    end

    if useFixedPointBlockEntry(runtimeCfg)
        [h_out, key_out, value_out] = qwen2_runtime.hdl.block_kernel(h_in, key_in, value_in, cache_valid_len, weights, hyperParameters, freqs_cis, runtimeCfg);
    else
        [h_out, key_out, value_out] = qwen2_runtime.hdl.block_kernel(single(h_in), single(key_in), single(value_in), cache_valid_len, weights, hyperParameters, freqs_cis, runtimeCfg);
    end
end

function tf = useFixedPointBlockEntry(runtimeCfg)
    tf = false;
    if ~isstruct(runtimeCfg)
        return;
    end
    if isfield(runtimeCfg, 'UseFixedPointHDL')
        tf = logical(runtimeCfg.UseFixedPointHDL);
        return;
    end
    if isfield(runtimeCfg, 'HDLNumericMode')
        tf = isequal(runtimeCfg.HDLNumericMode, 'fixed');
    end
end
