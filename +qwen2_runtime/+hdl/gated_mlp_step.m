function Z = gated_mlp_step(X, weights, cfg)
%GATED_MLP_STEP HDL-focused MLP step.

    [hiddenSize, seqLen, batchSize] = size(X);
    X2 = reshape(X, hiddenSize, []);

    gate = qwen2_runtime.hdl.linear_step(weights.gate_proj, X2, cfg);
    up = qwen2_runtime.hdl.linear_step(weights.up_proj, X2, cfg);
    intermediate = safeSilu(gate, cfg) .* up;
    Z = qwen2_runtime.hdl.linear_step(weights.down_proj, intermediate, cfg);
    Z = reshape(Z, [], seqLen, batchSize);
end

function Y = safeSilu(X, cfg)
    if isFixedPointMode(cfg)
        Y = approxSilu(X, cfg);
        return;
    end

    if isfield(cfg, 'EnableHDLMathSafeMode') && logical(cfg.EnableHDLMathSafeMode)
        Y = approxSilu(X, cfg);
    else
        Y = transformer.layer.silu(X);
    end
end

function Y = approxSilu(X, cfg)
    Y = zeros(size(X), 'like', X);
    lower = castLike(cfg.HDLExpNegLimit, X);
    upper = castLike(8.0, X);
    for i = 1:numel(X)
        x = X(i);
        if x < lower
            x = lower;
        elseif x > upper
            x = upper;
        end

        sigma = approxSigmoid(x);
        Y(i) = x * sigma;
    end
end

function y = approxSigmoid(x)
    if x <= castLike(-4.0, x)
        y = castLike(0.0, x);
    elseif x < castLike(4.0, x)
        y = castLike(0.5, x) + castLike(0.125, x) * x;
    else
        y = castLike(1.0, x);
    end

    if y < castLike(0.0, x)
        y = castLike(0.0, x);
    elseif y > castLike(1.0, x)
        y = castLike(1.0, x);
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

function y = castLike(value, prototype)
    if isa(prototype, 'embedded.fi')
        y = fi(value, true, prototype.WordLength, prototype.FractionLength, fimath(prototype));
    else
        y = cast(value, 'like', prototype);
    end
end
