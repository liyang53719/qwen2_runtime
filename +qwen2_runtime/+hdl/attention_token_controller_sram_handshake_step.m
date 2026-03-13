function [attn_proj_out, out_valid, busy, read_req, read_addr, write_req, write_addr, shift_enable, write_key_token, write_value_token, next_valid_len] = attention_token_controller_sram_handshake_step(start, h_token, cache_valid_len, rope_position, read_key_data, read_value_data, read_data_valid, weights, freqs_cis, hyperParameters, cfg)
%ATTENTION_TOKEN_CONTROLLER_SRAM_HANDSHAKE_STEP Token-step controller with multi-cycle external KV handshake.

    hiddenSize = hyperParameters.HiddenSize;
    headDim = hyperParameters.HeadDim;
    numHeads = hyperParameters.NumHeads;
    numKVHeads = hyperParameters.NumKVHeads;

    persistent q_token_reg k_token_reg v_token_reg
    if isempty(q_token_reg)
        q_token_reg = initTokenBuffer(headDim, numHeads, cfg, h_token);
        k_token_reg = initTokenBuffer(headDim, numKVHeads, cfg, h_token);
        v_token_reg = initTokenBuffer(headDim, numKVHeads, cfg, h_token);
    end

    attn_proj_out = initVectorBuffer(hiddenSize, cfg, h_token);
    if start
        h_vec = reshape(h_token, [hiddenSize, 1]);
        q_proj = qwen2_runtime.hdl.linear_step(weights.q_proj, h_vec, cfg);
        k_proj = qwen2_runtime.hdl.linear_step(weights.k_proj, h_vec, cfg);
        v_proj = qwen2_runtime.hdl.linear_step(weights.v_proj, h_vec, cfg);

        q_proj = addBiasIfPresent(q_proj, weights, 'q_bias');
        k_proj = addBiasIfPresent(k_proj, weights, 'k_bias');
        v_proj = addBiasIfPresent(v_proj, weights, 'v_bias');
        q_proj = castTokenLikeInput(q_proj, cfg);
        k_proj = castTokenLikeInput(k_proj, cfg);
        v_proj = castTokenLikeInput(v_proj, cfg);

        q_token_reg = reshape(q_proj, [headDim, numHeads]);
        k_token_reg = reshape(k_proj, [headDim, numKVHeads]);
        v_token_reg = reshape(v_proj, [headDim, numKVHeads]);
    end

    [attn_flat, attn_valid, busy, read_req, read_addr, write_req, write_addr, shift_enable, write_key_token, write_value_token, next_valid_len] = ...
        qwen2_runtime.hdl.attention_token_step_sram_handshake_step( ...
            start, q_token_reg, k_token_reg, v_token_reg, cache_valid_len, rope_position, read_key_data, read_value_data, read_data_valid, freqs_cis, hyperParameters, cfg);

    out_valid = false;
    if attn_valid
        attn_proj_out = qwen2_runtime.hdl.linear_step(weights.o_proj, reshape(attn_flat, [hiddenSize, 1]), cfg);
        attn_proj_out = addBiasIfPresent(attn_proj_out, weights, 'o_bias');
        out_valid = true;
    end
end

function value = addBiasIfPresent(value, weights, fieldName)
    if isfield(weights, fieldName)
        value = value + reshape(weights.(fieldName), size(value));
    end
end

function value = initTokenBuffer(headDim, headCount, cfg, prototype)
    if isFixedPointMode(cfg)
        F = controllerFimath(cfg);
        if isa(prototype, 'embedded.fi')
            value = fi(zeros(headDim, headCount), true, prototype.WordLength, prototype.FractionLength, F);
        else
            value = fi(zeros(headDim, headCount), true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, F);
        end
    else
        value = zeros(headDim, headCount, 'single');
    end
end

function value = initVectorBuffer(hiddenSize, cfg, ~)
    if isFixedPointMode(cfg)
        value = fi(zeros(hiddenSize, 1), true, cfg.HDLLinearAccumWordLength, cfg.HDLLinearAccumFractionLength, controllerFimath(cfg));
    else
        value = zeros(hiddenSize, 1, 'single');
    end
end

function value = castTokenLikeInput(value, cfg)
    if isFixedPointMode(cfg)
        value = fi(value, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, controllerFimath(cfg));
    else
        value = single(value);
    end
end

function F = controllerFimath(cfg)
    F = fimath( ...
        'RoundingMethod', 'Floor', ...
        'OverflowAction', 'Saturate', ...
        'ProductMode', 'SpecifyPrecision', ...
        'ProductWordLength', 24, ...
        'ProductFractionLength', cfg.HDLLinearAccumFractionLength, ...
        'SumMode', 'SpecifyPrecision', ...
        'SumWordLength', cfg.HDLLinearAccumWordLength, ...
        'SumFractionLength', cfg.HDLLinearAccumFractionLength);
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