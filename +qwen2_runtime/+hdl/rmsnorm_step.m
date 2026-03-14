function Y = rmsnorm_step(X, weight, epsilon, cfg)
%RMSNORM_STEP HDL-friendly RMSNorm approximation.

    if nargin < 4
        cfg = [];
    end

    if useFixedPointRmsnorm(cfg)
        Y = fixedRmsnorm(X, weight, epsilon, cfg);
        return;
    end

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

function Y = fixedRmsnorm(X, weight, epsilon, cfg)
    F = rmsFimath(cfg);
    inputWL = cfg.HDLLinearInputWordLength;
    inputFL = cfg.HDLLinearInputFractionLength;
    accumWL = cfg.HDLLinearAccumWordLength;
    accumFL = cfg.HDLLinearAccumFractionLength;

    X_fix = toFixedInput(X, inputWL, inputFL, F);
    weight_fix = toFixedInput(weight, inputWL, inputFL, F);
    [hiddenSize, seqLen, batchSize] = size(X_fix);
    Y = fi(zeros(hiddenSize, seqLen, batchSize), true, accumWL, accumFL, F);

    invHidden = fi(single(1.0 / double(hiddenSize)), true, inputWL, inputFL, F);
    epsilon_fix = fi(single(epsilon), true, accumWL, accumFL, F);
    for b = 1:batchSize
        for s = 1:seqLen
            sumSquares = fi(0, true, accumWL, accumFL, F);
            for i = 1:hiddenSize
                value = fi(X_fix(i, s, b), true, inputWL, inputFL, F);
                sumSquares = fi(sumSquares + value * value, true, accumWL, accumFL, F);
            end
            meanSquare = fi(sumSquares * invHidden, true, accumWL, accumFL, F);
            variance = fi(meanSquare + epsilon_fix, true, accumWL, accumFL, F);
            invStd = reciprocalSqrtApproxFixed(variance, cfg, F);
            for i = 1:hiddenSize
                xVal = fi(X_fix(i, s, b), true, inputWL, inputFL, F);
                wVal = fi(weight_fix(i), true, inputWL, inputFL, F);
                Y(i, s, b) = fi(xVal * invStd * wVal, true, accumWL, accumFL, F);
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

function y = reciprocalSqrtApproxFixed(x, cfg, F)
    minDenom = fi(single(1.0e-6), true, 32, 14, F);
    if isfield(cfg, 'HDLMinDenominator')
        minDenom = fi(single(cfg.HDLMinDenominator), true, 32, 14, F);
    end

    xFix = fi(x, true, 32, 14, F);
    if xFix < minDenom
        xFix = minDenom;
    end

    y = initialRsqrtGuessFixed(xFix, F);
    iterations = 3;
    if isfield(cfg, 'HDLInvSqrtIterations')
        iterations = double(cfg.HDLInvSqrtIterations);
    end

    halfx = fi(castLike(0.5, xFix) * xFix, true, 32, 14, F);
    for i = 1:iterations
        y = fi(y * (castLike(1.5, xFix) - halfx * y * y), true, 32, 14, F);
    end
end

function y = initialRsqrtGuessFixed(x, F)
    if x < fi(0.25, true, 32, 14, F)
        y = fi(2.0, true, 32, 14, F);
    elseif x < fi(1.0, true, 32, 14, F)
        y = fi(1.0, true, 32, 14, F);
    elseif x < fi(4.0, true, 32, 14, F)
        y = fi(0.5, true, 32, 14, F);
    else
        y = fi(0.25, true, 32, 14, F);
    end
end

function tf = useFixedPointRmsnorm(cfg)
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

function F = rmsFimath(cfg)
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

function value_fix = toFixedInput(value, wordLength, fractionLength, F)
    if isa(value, 'embedded.fi')
        value_fix = value;
    else
        value_fix = fi(value, true, wordLength, fractionLength, F);
    end
end

function y = castLike(value, prototype)
    y = fi(value, true, prototype.WordLength, prototype.FractionLength, fimath(prototype));
end
