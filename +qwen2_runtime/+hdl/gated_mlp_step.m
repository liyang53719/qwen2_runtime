function Z = gated_mlp_step(X, weights, cfg)
%GATED_MLP_STEP HDL-focused MLP step.

    [hiddenSize, seqLen, batchSize] = size(X);
    X2 = reshape(X, hiddenSize, []);

    gate = qwen2_runtime.hdl.linear_step(weights.gate_proj, X2);
    up = qwen2_runtime.hdl.linear_step(weights.up_proj, X2);
    intermediate = safeSilu(gate, cfg) .* up;
    Z = qwen2_runtime.hdl.linear_step(weights.down_proj, intermediate);
    Z = reshape(Z, [], seqLen, batchSize);
end

function Y = safeSilu(X, cfg)
    if isfield(cfg, 'EnableHDLMathSafeMode') && logical(cfg.EnableHDLMathSafeMode)
        Y = approxSilu(X, cfg);
    else
        Y = transformer.layer.silu(X);
    end
end

function Y = approxSilu(X, cfg)
    Y = zeros(size(X), 'like', X);
    lower = cfg.HDLExpNegLimit;
    upper = single(8.0);
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
    if x <= single(-4.0)
        y = single(0.0);
    elseif x < single(4.0)
        y = single(0.5) + single(0.125) * x;
    else
        y = single(1.0);
    end

    if y < single(0.0)
        y = single(0.0);
    elseif y > single(1.0)
        y = single(1.0);
    end
end
