function Y = linear_step(W, X, cfg)
%LINEAR_STEP HDL-only floating-point linear kernel.

    if nargin < 3
        cfg = [];
    end

    W_data = getNumeric(W);
    X_data = getNumeric(X);
    if useFixedPointLinear(cfg)
        Y = fixedLinearMatrix(W_data, X_data, cfg);
        return;
    end

    Y = single(W_data) * single(X_data);
end

function data = getNumeric(data)
    if isstruct(data)
        if isfield(data, 'Float')
            data = data.Float;
        elseif isfield(data, 'Q')
            data = single(data.Q) .* single(data.Scale);
        else
            error('qwen2_runtime:UnsupportedHDLWeight', 'Unsupported HDL weight struct.');
        end
    end
    if isa(data, 'dlarray')
        data = extractdata(data);
    end
    if ~isa(data, 'embedded.fi')
        data = single(data);
    end
end

function tf = useFixedPointLinear(cfg)
    tf = false;
    if isempty(cfg) || ~isstruct(cfg)
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

function Y = fixedLinearMatrix(W_data, X_data, cfg)
    F = linearFimath(cfg);
    inputWL = cfg.HDLLinearInputWordLength;
    inputFL = cfg.HDLLinearInputFractionLength;
    accumWL = cfg.HDLLinearAccumWordLength;
    accumFL = cfg.HDLLinearAccumFractionLength;

    W_fix = toFixedInput(W_data, inputWL, inputFL, F);
    X_fix = toFixedInput(X_data, inputWL, inputFL, F);
    [outDim, inDim] = size(W_fix);
    numCols = size(X_fix, 2);
    Y = fi(zeros(outDim, numCols), true, accumWL, accumFL, F);
    zeroSeed = fi(zeros(outDim, 1), true, accumWL, accumFL, F);

    for col = 1:numCols
        x_col = reshape(X_fix(:, col), inDim, 1);
        Y(:, col) = qwen2_runtime.hdl.linear_tile_step(x_col, W_fix, zeroSeed);
    end
end

function F = linearFimath(cfg)
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
