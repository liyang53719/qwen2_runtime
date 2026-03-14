function [h_out, key_cache_out, value_cache_out] = block_kernel(h_in, key_cache_in, value_cache_in, cache_valid_len, weights, hyperParameters, freqs_cis, cfg)
%BLOCK_KERNEL HDL-focused block kernel with explicit cache interface.

    resid = h_in;
    h_norm = qwen2_runtime.hdl.rmsnorm_step(h_in, getNumeric(weights.input_layernorm), single(1e-6), cfg);
    hiddenSize = hyperParameters.HiddenSize;
    kvSize = hyperParameters.HeadDim * hyperParameters.NumKVHeads;

    attnWeights.q_proj = weights.self_attn_q_proj;
    attnWeights.k_proj = weights.self_attn_k_proj;
    attnWeights.v_proj = weights.self_attn_v_proj;
    attnWeights.o_proj = weights.self_attn_o_proj;
    attnWeights.q_bias = zeros(hiddenSize, 1, 'like', h_in);
    attnWeights.k_bias = zeros(kvSize, 1, 'like', h_in);
    attnWeights.v_bias = zeros(kvSize, 1, 'like', h_in);
    attnWeights.o_bias = zeros(hiddenSize, 1, 'like', h_in);
    if isfield(weights, 'self_attn_q_bias'), attnWeights.q_bias = getNumeric(weights.self_attn_q_bias); end
    if isfield(weights, 'self_attn_k_bias'), attnWeights.k_bias = getNumeric(weights.self_attn_k_bias); end
    if isfield(weights, 'self_attn_v_bias'), attnWeights.v_bias = getNumeric(weights.self_attn_v_bias); end
    if isfield(weights, 'self_attn_o_bias'), attnWeights.o_bias = getNumeric(weights.self_attn_o_bias); end

    [h_attn, key_cache_out, value_cache_out] = qwen2_runtime.hdl.attention_step(h_norm, key_cache_in, value_cache_in, cache_valid_len, attnWeights, hyperParameters, freqs_cis, cfg);
    h_mid = resid + h_attn;

    resid = h_mid;
    h_post = qwen2_runtime.hdl.rmsnorm_step(h_mid, getNumeric(weights.post_attention_layernorm), single(1e-6), cfg);

    ffnWeights.gate_proj = weights.mlp_gate_proj;
    ffnWeights.up_proj = weights.mlp_up_proj;
    ffnWeights.down_proj = weights.mlp_down_proj;
    h_ffn = qwen2_runtime.hdl.gated_mlp_step(h_post, ffnWeights, cfg);
    h_out = resid + h_ffn;
end

function value = getNumeric(value)
    if isa(value, 'dlarray')
        value = extractdata(value);
    end
    if ~isa(value, 'embedded.fi')
        value = single(value);
    end
end
