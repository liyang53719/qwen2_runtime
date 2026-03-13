function [attn_proj_out, key_cache_out, value_cache_out, next_valid_len, read_enable, read_addr, write_enable, write_addr, shift_enable] = attention_token_controller_sram_step(h_token, key_cache_in, value_cache_in, cache_valid_len, rope_position, weights, freqs_cis, hyperParameters, cfg)
%ATTENTION_TOKEN_CONTROLLER_SRAM_STEP Token-step attention controller with real q/k/v/o projections.

    hiddenSize = hyperParameters.HiddenSize;
    headDim = hyperParameters.HeadDim;
    numHeads = hyperParameters.NumHeads;
    numKVHeads = hyperParameters.NumKVHeads;

    h_vec = reshape(h_token, [hiddenSize, 1]);
    q_proj = qwen2_runtime.hdl.linear_step(weights.q_proj, h_vec, cfg);
    k_proj = qwen2_runtime.hdl.linear_step(weights.k_proj, h_vec, cfg);
    v_proj = qwen2_runtime.hdl.linear_step(weights.v_proj, h_vec, cfg);

    q_proj = addBiasIfPresent(q_proj, weights, 'q_bias');
    k_proj = addBiasIfPresent(k_proj, weights, 'k_bias');
    v_proj = addBiasIfPresent(v_proj, weights, 'v_bias');

    q_token = reshape(q_proj, [headDim, numHeads]);
    k_token = reshape(k_proj, [headDim, numKVHeads]);
    v_token = reshape(v_proj, [headDim, numKVHeads]);

    [attn_flat, key_cache_out, value_cache_out, next_valid_len, read_enable, read_addr, write_enable, write_addr, shift_enable] = ...
        qwen2_runtime.hdl.attention_token_step_sram_contract_step( ...
            q_token, k_token, v_token, key_cache_in, value_cache_in, cache_valid_len, rope_position, freqs_cis, hyperParameters, cfg);

    attn_proj_out = qwen2_runtime.hdl.linear_step(weights.o_proj, reshape(attn_flat, [hiddenSize, 1]), cfg);
    attn_proj_out = addBiasIfPresent(attn_proj_out, weights, 'o_bias');
end

function value = addBiasIfPresent(value, weights, fieldName)
    if isfield(weights, fieldName)
        value = value + reshape(weights.(fieldName), size(value));
    end
end