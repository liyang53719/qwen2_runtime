function Y = linear_step(W, X)
%LINEAR_STEP HDL-only floating-point linear kernel.

    W_data = getNumeric(W);
    X_data = getNumeric(X);
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
    data = single(data);
end
