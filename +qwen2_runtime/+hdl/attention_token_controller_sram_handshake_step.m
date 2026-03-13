function [attn_proj_out, out_valid, busy, read_req, read_addr, write_req, write_addr, shift_enable, write_key_token, write_value_token, next_valid_len] = attention_token_controller_sram_handshake_step(start, h_token, cache_valid_len, rope_position, read_key_data, read_value_data, read_data_valid, weights, freqs_cis, hyperParameters, cfg)
%ATTENTION_TOKEN_CONTROLLER_SRAM_HANDSHAKE_STEP Token-step controller with multi-cycle external KV handshake.

    hiddenSize = 1536;
    headDim = 128;
    numHeads = 12;
    numKVHeads = 2;

    persistent q_token_reg k_token_reg v_token_reg
    if isempty(q_token_reg)
        q_token_reg = initTokenBuffer(headDim, numHeads, cfg, h_token);
        k_token_reg = initTokenBuffer(headDim, numKVHeads, cfg, h_token);
        v_token_reg = initTokenBuffer(headDim, numKVHeads, cfg, h_token);
    end

    attn_proj_out = initVectorBuffer(hiddenSize, cfg, h_token);
    if start
        [q_token_reg, k_token_reg, v_token_reg] = qwen2_runtime.hdl.attention_token_qkv_project_step(h_token, weights, hyperParameters, cfg);
    end

    [attn_flat, attn_valid, busy, read_req, read_addr, write_req, write_addr, shift_enable, write_key_token, write_value_token, next_valid_len] = ...
        qwen2_runtime.hdl.attention_token_step_sram_handshake_step( ...
            start, q_token_reg, k_token_reg, v_token_reg, cache_valid_len, rope_position, read_key_data, read_value_data, read_data_valid, freqs_cis, hyperParameters, cfg);

    out_valid = false;
    if attn_valid
        attn_proj_out = qwen2_runtime.hdl.attention_token_o_project_step(attn_flat, weights, hyperParameters, cfg);
        out_valid = true;
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