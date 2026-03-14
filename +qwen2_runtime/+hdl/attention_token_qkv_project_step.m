function [q_token, k_token, v_token] = attention_token_qkv_project_step(h_token, weights, hyperParameters, cfg)
%ATTENTION_TOKEN_QKV_PROJECT_STEP Project one token into q/k/v token matrices.

    headDim = double(hyperParameters.HeadDim);
    numHeads = double(hyperParameters.NumHeads);
    numKVHeads = double(hyperParameters.NumKVHeads);

    q_proj = qwen2_runtime.hdl.linear_step(weights.q_proj, h_token, cfg);
    k_proj = qwen2_runtime.hdl.linear_step(weights.k_proj, h_token, cfg);
    v_proj = qwen2_runtime.hdl.linear_step(weights.v_proj, h_token, cfg);

    q_proj = addBiasIfPresent(q_proj, weights, 'q_bias');
    k_proj = addBiasIfPresent(k_proj, weights, 'k_bias');
    v_proj = addBiasIfPresent(v_proj, weights, 'v_bias');

    q_proj = castTokenLikeInput(q_proj, cfg);
    k_proj = castTokenLikeInput(k_proj, cfg);
    v_proj = castTokenLikeInput(v_proj, cfg);

    q_token = columnVectorToMatrix(q_proj, headDim, numHeads);
    k_token = columnVectorToMatrix(k_proj, headDim, numKVHeads);
    v_token = columnVectorToMatrix(v_proj, headDim, numKVHeads);
end

function value = addBiasIfPresent(value, weights, fieldName)
    if isfield(weights, fieldName)
        value = value + weights.(fieldName);
    end
end

function matrix = columnVectorToMatrix(vector, rowCount, colCount)
    matrix = zeros(rowCount, colCount, 'like', vector);
    for col = 1:colCount
        base = (col - 1) * rowCount;
        for row = 1:rowCount
            matrix(row, col) = vector(base + row);
        end
    end
end

function value = castTokenLikeInput(value, cfg)
    if isFixedPointMode(cfg)
        value = fi(value, true, cfg.HDLLinearInputWordLength, cfg.HDLLinearInputFractionLength, localFimath(cfg));
    else
        value = single(value);
    end
end

function F = localFimath(cfg)
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
    tf = isstruct(cfg) && logical(cfg.UseFixedPointHDL);
end