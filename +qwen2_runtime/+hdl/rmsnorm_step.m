function Y = rmsnorm_step(X, weight, epsilon, cfg)
%RMSNORM_STEP HDL-friendly RMSNorm approximation.

    [hiddenSize, seqLen, batchSize] = size(X);
    Y = zeros(hiddenSize, seqLen, batchSize, 'single');

    for b = 1:batchSize
        for s = 1:seqLen
            sumSquares = single(0);
            for i = 1:hiddenSize
                value = X(i, s, b);
                sumSquares = sumSquares + value * value;
            end
            meanSquare = sumSquares / single(hiddenSize);
            variance = meanSquare + epsilon;
            invStd = reciprocalSqrtApprox(variance, cfg);
            for i = 1:hiddenSize
                Y(i, s, b) = X(i, s, b) * invStd * weight(i);
            end
        end
    end
end

function y = reciprocalSqrtApprox(x, cfg)
    if x < single(1.0e-6)
        x = single(1.0e-6);
    end

    y = initialRsqrtGuess(x);
    iterations = 3;
    if isfield(cfg, 'HDLInvSqrtIterations')
        iterations = double(cfg.HDLInvSqrtIterations);
    end

    halfx = single(0.5) * x;
    for i = 1:iterations
        y = y * (single(1.5) - halfx * y * y);
    end
end

function y = initialRsqrtGuess(x)
    if x < single(0.25)
        y = single(2.0);
    elseif x < single(1.0)
        y = single(1.0);
    elseif x < single(4.0)
        y = single(0.5);
    else
        y = single(0.25);
    end
end
