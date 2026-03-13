function [block_out, out_valid, busy, read_req, read_addr, write_req, write_addr, shift_enable, write_key_token, write_value_token, next_valid_len] = block0_token_system_step(start, h_token, cache_valid_len, rope_position, read_key_data, read_value_data, read_data_valid, weights, hyperParameters, freqs_cis, cfg)
%BLOCK0_TOKEN_SYSTEM_STEP Block-0 token-step system top with external KV handshake.

    hiddenSize = hyperParameters.HiddenSize;
    persistent resid_reg h_norm_reg
    if isempty(resid_reg)
        resid_reg = initRuntimeVector(hiddenSize, h_token, cfg);
        h_norm_reg = initRuntimeVector(hiddenSize, h_token, cfg);
    end

    if start
        h_start = reshape(toRuntimeValue(h_token, cfg), [hiddenSize, 1]);
        resid_reg = h_start;
        h_norm_reg = reshape(qwen2_runtime.hdl.rmsnorm_step(reshape(h_start, [hiddenSize, 1, 1]), toRuntimeValue(weights.input_layernorm, cfg), single(1.0e-6), cfg), [hiddenSize, 1]);
    end

    attnWeights = struct();
    attnWeights.q_proj = weights.self_attn_q_proj;
    attnWeights.k_proj = weights.self_attn_k_proj;
    attnWeights.v_proj = weights.self_attn_v_proj;
    attnWeights.o_proj = weights.self_attn_o_proj;
    if isfield(weights, 'self_attn_q_bias'), attnWeights.q_bias = weights.self_attn_q_bias; end
    if isfield(weights, 'self_attn_k_bias'), attnWeights.k_bias = weights.self_attn_k_bias; end
    if isfield(weights, 'self_attn_v_bias'), attnWeights.v_bias = weights.self_attn_v_bias; end
    if isfield(weights, 'self_attn_o_bias'), attnWeights.o_bias = weights.self_attn_o_bias; end

    [attn_proj_out, attn_valid, busy, read_req, read_addr, write_req, write_addr, shift_enable, write_key_token, write_value_token, next_valid_len] = ...
        qwen2_runtime.hdl.attention_token_controller_sram_handshake_step( ...
            start, h_norm_reg, cache_valid_len, rope_position, read_key_data, read_value_data, read_data_valid, attnWeights, freqs_cis, hyperParameters, cfg);

    block_out = zeros(hiddenSize, 1, 'like', h_token);
    out_valid = false;
    if attn_valid
        h_attn = reshape(toRuntimeValue(attn_proj_out, cfg), [hiddenSize, 1]);
        h_mid = runtimeResidualAdd(resid_reg, h_attn, cfg);
        h_post = qwen2_runtime.hdl.rmsnorm_step(reshape(h_mid, [hiddenSize, 1, 1]), toRuntimeValue(weights.post_attention_layernorm, cfg), single(1.0e-6), cfg);
        mlpWeights = struct();
        mlpWeights.gate_proj = weights.mlp_gate_proj;
        mlpWeights.up_proj = weights.mlp_up_proj;
        mlpWeights.down_proj = weights.mlp_down_proj;
        h_ffn = qwen2_runtime.hdl.gated_mlp_step(h_post, mlpWeights, cfg);
        block_out = runtimeResidualAdd(h_mid, reshape(toRuntimeValue(h_ffn, cfg), [hiddenSize, 1]), cfg);
        out_valid = true;
    end
end

function value = toRuntimeValue(value, cfg)
    if isa(value, 'dlarray')
        value = extractdata(value);
    end
    if isFixedPointMode(cfg)
        if ~isa(value, 'embedded.fi')
            value = fi(single(value), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength);
        end
    else
        value = single(value);
    end
end

function value = initRuntimeVector(hiddenSize, prototype, cfg)
    if isFixedPointMode(cfg) && isa(prototype, 'embedded.fi')
        value = fi(zeros(hiddenSize, 1), true, prototype.WordLength, prototype.FractionLength, fimath(prototype));
    elseif isFixedPointMode(cfg)
        value = fi(zeros(hiddenSize, 1), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength);
    else
        value = zeros(hiddenSize, 1, 'single');
    end
end

function out = runtimeResidualAdd(a, b, cfg)
    if isFixedPointMode(cfg)
        out = qwen2_runtime.hdl.residual_add_step(a, b);
    else
        out = single(a) + single(b);
    end
end

function tf = isFixedPointMode(cfg)
    tf = false;
    if ~isstruct(cfg)
        return;
    end
    if isfield(cfg, 'UseFixedPointHDL')
        tf = logical(cfg.UseFixedPointHDL);
        return;
    end
    if isfield(cfg, 'HDLNumericMode')
        tf = isequal(cfg.HDLNumericMode, 'fixed');
    end
end