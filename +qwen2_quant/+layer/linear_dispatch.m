function Y = linear_dispatch(W, X, cfg, meta)
%LINEAR_DISPATCH Dispatch linear execution by weight kind.

    if nargin < 3 || ~isstruct(cfg)
        cfg = struct('LinearMode', 'float');
    end
    if nargin < 4 || ~isstruct(meta)
        meta = struct();
    end

    mode = "float";
    if isfield(cfg, 'LinearMode')
        mode = lower(string(cfg.LinearMode));
    end

    if mode ~= "mixed_dynamic"
        [Y, ~] = qwen2_quant.layer.quantized_matmul(W, X, cfg);
        return;
    end

    kind = classifyWeightKind(W);
    opName = getMetaString(meta, 'OpName', 'linear');
    layerIdx = getMetaNumber(meta, 'LayerIndex', -1);
    if isfield(cfg, 'TracePrecision') && cfg.TracePrecision
        try
            qwen2_quant.internal.precision_trace('log', [opName '.input'], getNumeric2D(X));
        catch
        end
    end

    switch kind
        case "float"
            qwen2_quant.internal.execution_trace('log', opName, layerIdx, kind, 'float');
            Y = applyFloatLinear(W, X);
        case {"q8_0", "q4_0", "q4_k", "q6_k"}
            qwen2_quant.internal.execution_trace('log', opName, layerIdx, kind, 'integer');
            Y = applyGgufIntegerLinear(W, X, kind, cfg, meta);
        case {"gptq_int4", "awq_int4"}
            qwen2_quant.internal.execution_trace('log', opName, layerIdx, kind, 'integer');
            gptqCfg = cfg;
            gptqCfg.LinearMode = 'gptq_int4_quant_sim';
            [Y, ~] = qwen2_quant.layer.quantized_matmul(W, X, gptqCfg);
        otherwise
            qwen2_quant.internal.execution_trace('log', opName, layerIdx, kind, 'float');
            Y = applyFloatLinear(W, X);
    end

    if isfield(cfg, 'TracePrecision') && cfg.TracePrecision
        qwen2_quant.internal.precision_trace('log', [opName '.output'], Y);
    end
end

function kind = classifyWeightKind(W)
    if isa(W, 'qwen2_quant.internal.quantized_weight')
        qt = upper(string(W.QuantType));
        switch qt
            case "Q8_0"
                kind = "q8_0";
            case "Q4_0"
                kind = "q4_0";
            case "Q4_K"
                kind = "q4_k";
            case "Q6_K"
                kind = "q6_k";
            otherwise
                kind = "float";
        end
        return;
    end

    if isstruct(W) && isfield(W, 'QuantType')
        qt = upper(string(W.QuantType));
        if qt == "GPTQ_INT4"
            kind = "gptq_int4";
            return;
        end
        if qt == "AWQ_INT4"
            kind = "awq_int4";
            return;
        end
    end

    kind = "float";
end

function Y = applyFloatLinear(W, X)
    X2 = getNumeric2D(X);
    Wf = getFloatWeight(W);
    Y = Wf * X2;
end

function Y = applyGgufIntegerLinear(W, X, kind, cfg, meta)
    X2 = getNumeric2D(X);
    meta = configureGgufActivationGroups(meta, kind, size(X2, 1));
    [X_q, X_scale] = quantizeActivationSymmetric(X2, cfg, meta);

    switch kind
        case "q8_0"
            [W_q, W_scale] = W.get_q8_0_components();
            Y = q8_0_dynamic_matmul(W, W_q, W_scale, X_q, X_scale);
        case "q4_0"
            [W_q, W_delta] = W.get_q4_0_components();
            Y = q4_0_dynamic_matmul(W, W_q, W_delta, X_q, X_scale);
        case "q4_k"
            comp = W.get_q4_k_components();
            Y = q4_k_dynamic_matmul(W, comp, X_q, X_scale);
        case "q6_k"
            comp = W.get_q6_k_components();
            Y = q6_k_dynamic_matmul(W, comp, X_q, X_scale);
        otherwise
            Y = applyFloatLinear(W, X2);
    end
end

function [q, scale] = quantizeActivationSymmetric(X, cfg, meta)
    groupSize = 0;
    if isfield(meta, 'DisableGroupedActivationQuantization') && logical(meta.DisableGroupedActivationQuantization)
        groupSize = 0;
    elseif isfield(meta, 'GroupSize')
        candidate = double(meta.GroupSize);
        if candidate > 0 && mod(size(X, 1), candidate) == 0
            groupSize = candidate;
        end
    elseif isfield(cfg, 'ActivationGroupSize')
        candidate = double(cfg.ActivationGroupSize);
        if candidate > 0 && mod(size(X, 1), candidate) == 0
            groupSize = candidate;
        end
    end

    if groupSize > 0
        [q, scale] = quantizeGroupedSymmetric(X, groupSize);
        return;
    end

    maxAbs = max(abs(X), [], 1);
    scale = max(single(maxAbs / 127.0), eps('single'));
    q = int8(max(min(round(X ./ scale), 127), -127));
end

function meta = configureGgufActivationGroups(meta, kind, numRows)
    if isfield(meta, 'DisableGroupedActivationQuantization') && logical(meta.DisableGroupedActivationQuantization)
        return;
    end

    switch kind
        case {"q8_0", "q4_0"}
            groupSize = 32;
        case {"q4_k", "q6_k"}
            groupSize = 32;
        otherwise
            groupSize = 0;
    end

    if groupSize > 0 && mod(numRows, groupSize) == 0
        meta.GroupSize = groupSize;
    end
end

function [q, scale] = quantizeGroupedSymmetric(X, groupSize)
    [numRows, nCols] = size(X);
    numGroups = numRows / groupSize;
    q = zeros(numRows, nCols, 'int8');
    scale = zeros(numGroups, nCols, 'single');
    for g = 1:numGroups
        rows = (g-1) * groupSize + (1:groupSize);
        chunk = X(rows, :);
        maxAbs = max(abs(chunk), [], 1);
        scale_g = max(single(maxAbs / 127.0), eps('single'));
        q(rows, :) = int8(max(min(round(chunk ./ scale_g), 127), -127));
        scale(g, :) = scale_g;
    end
end

function Y = q8_0_dynamic_matmul(W, W_q_blocks, W_scales, X_q, X_scale)
    [M, K, blocksPerRow] = getWeightShape32(W);
    [~, N] = size(X_q);
    expectedBlocks = M * blocksPerRow;
    W_q_grid = reshape(W_q_blocks(:, 1:expectedBlocks), 32, blocksPerRow, M);
    W_s_grid = reshape(W_scales(1:expectedBlocks), blocksPerRow, M);
    Y = zeros(M, N, 'single');

    for b = 1:blocksPerRow
        idxStart = (b-1) * 32 + 1;
        idxEnd = min(b * 32, K);
        validLen = idxEnd - idxStart + 1;
        wChunk = int32(permute(W_q_grid(1:validLen, b, :), [3, 1, 2]));
        xChunk = int32(X_q(idxStart:idxEnd, :));
        acc = intMatmulAccInt32(wChunk, xChunk);
        scaleProd = single(W_s_grid(b, :).') .* activationScaleSlice(X_scale, idxStart, validLen, K);
        Y = Y + single(acc) .* scaleProd;
    end
end

function Y = q4_0_dynamic_matmul(W, W_q_blocks, W_deltas, X_q, X_scale)
    [M, K, blocksPerRow] = getWeightShape32(W);
    [~, N] = size(X_q);
    expectedBlocks = M * blocksPerRow;
    W_q_grid = reshape(W_q_blocks(:, 1:expectedBlocks), 32, blocksPerRow, M);
    W_d_grid = reshape(W_deltas(1:expectedBlocks), blocksPerRow, M);
    Y = zeros(M, N, 'single');

    for b = 1:blocksPerRow
        idxStart = (b-1) * 32 + 1;
        idxEnd = min(b * 32, K);
        validLen = idxEnd - idxStart + 1;
        wChunk = int32(permute(W_q_grid(1:validLen, b, :), [3, 1, 2]));
        xChunk = int32(X_q(idxStart:idxEnd, :));
        acc = intMatmulAccInt32(wChunk, xChunk);
        scaleProd = single(W_d_grid(b, :).') .* activationScaleSlice(X_scale, idxStart, validLen, K);
        Y = Y + single(acc) .* scaleProd;
    end
end

function Y = q6_k_dynamic_matmul(W, comp, X_q, X_scale)
    [M, K, blocksPerRow] = getWeightShape256(W);
    [~, N] = size(X_q);
    expectedBlocks = M * blocksPerRow;
    ql = comp.ql(:, 1:expectedBlocks);
    qh = comp.qh(:, 1:expectedBlocks);
    sc = single(comp.scales(:, 1:expectedBlocks));
    d = single(comp.d(1:expectedBlocks));
    Y = zeros(M, N, 'single');

    for b = 1:blocksPerRow
        idxBlocks = (0:M-1) * blocksPerRow + b;
        ql_b = ql(:, idxBlocks);
        qh_b = qh(:, idxBlocks);
        sc_b = sc(:, idxBlocks);
        d_b = reshape(d(idxBlocks), M, 1);

        idxStart = (b-1) * 256 + 1;
        idxEnd = min(b * 256, K);
        validLen = idxEnd - idxStart + 1;
        xChunk = int32(X_q(idxStart:idxEnd, :));

        ql_ptr = 1;
        qh_ptr = 1;
        sc_ptr = 1;
        xOffset = 0;
        for nHalf = 1:2
            ql64 = ql_b(ql_ptr:ql_ptr+63, :);
            qh32 = qh_b(qh_ptr:qh_ptr+31, :);
            for l = 0:31
                qh_v = qh32(l+1, :);
                q1 = int32(single(bitand(ql64(l+1, :), uint8(15))) + single(bitshift(bitand(qh_v, uint8(3)), 4)) - 32);
                q2 = int32(single(bitand(ql64(l+33, :), uint8(15))) + single(bitshift(bitand(bitshift(qh_v, -2), uint8(3)), 4)) - 32);
                q3 = int32(single(bitshift(ql64(l+1, :), -4)) + single(bitshift(bitand(bitshift(qh_v, -4), uint8(3)), 4)) - 32);
                q4 = int32(single(bitshift(ql64(l+33, :), -4)) + single(bitshift(bitand(bitshift(qh_v, -6), uint8(3)), 4)) - 32);

                is = sc_ptr + floor(l / 16);
                Y = accumulateIntRowChunk(Y, q1, d_b .* reshape(sc_b(is + 0, :).', M, 1), xChunk, X_scale, idxStart + xOffset + l, xOffset + l + 1, validLen, K);
                Y = accumulateIntRowChunk(Y, q2, d_b .* reshape(sc_b(is + 2, :).', M, 1), xChunk, X_scale, idxStart + xOffset + l + 32, xOffset + l + 33, validLen, K);
                Y = accumulateIntRowChunk(Y, q3, d_b .* reshape(sc_b(is + 4, :).', M, 1), xChunk, X_scale, idxStart + xOffset + l + 64, xOffset + l + 65, validLen, K);
                Y = accumulateIntRowChunk(Y, q4, d_b .* reshape(sc_b(is + 6, :).', M, 1), xChunk, X_scale, idxStart + xOffset + l + 96, xOffset + l + 97, validLen, K);
            end

            xOffset = xOffset + 128;
            ql_ptr = ql_ptr + 64;
            qh_ptr = qh_ptr + 32;
            sc_ptr = sc_ptr + 8;
        end
    end
end

function Y = q4_k_dynamic_matmul(W, comp, X_q, X_scale)
    [M, K, blocksPerRow] = getWeightShape256(W);
    [~, N] = size(X_q);
    expectedBlocks = M * blocksPerRow;
    d = single(comp.d(1:expectedBlocks));
    dmin = single(comp.dmin(1:expectedBlocks));
    sc = single(comp.scales(:, 1:expectedBlocks));
    mn = single(comp.mins(:, 1:expectedBlocks));
    qs = comp.qs(:, 1:expectedBlocks);
    Y = zeros(M, N, 'single');

    for b = 1:blocksPerRow
        idxBlocks = (0:M-1) * blocksPerRow + b;
        d_b = reshape(d(idxBlocks), M, 1);
        dmin_b = reshape(dmin(idxBlocks), M, 1);
        sc_b = sc(:, idxBlocks);
        mn_b = mn(:, idxBlocks);
        qs_b = qs(:, idxBlocks);

        idxStart = (b-1) * 256 + 1;
        idxEnd = min(b * 256, K);
        validLen = idxEnd - idxStart + 1;
        xChunk = int32(X_q(idxStart:idxEnd, :));

        for seg = 0:3
            q32 = qs_b(seg*32 + (1:32), :);
            ql = int32(single(bitand(q32, uint8(15))).');
            qh = int32(single(bitshift(q32, -4)).');

            is1 = seg*2 + 1;
            is2 = seg*2 + 2;
            gain1 = d_b .* reshape(sc_b(is1, :).', M, 1);
            off1 = dmin_b .* reshape(mn_b(is1, :).', M, 1);
            gain2 = d_b .* reshape(sc_b(is2, :).', M, 1);
            off2 = dmin_b .* reshape(mn_b(is2, :).', M, 1);

            base = seg * 64;
            scaleX1 = activationScaleSlice(X_scale, idxStart + base, min(32, max(validLen - base, 0)), K);
            scaleX2 = activationScaleSlice(X_scale, idxStart + base + 32, min(32, max(validLen - (base + 32), 0)), K);
            Y = accumulateQ4KSegment(Y, ql, gain1, off1, xChunk, scaleX1, base + 1, validLen);
            Y = accumulateQ4KSegment(Y, qh, gain2, off2, xChunk, scaleX2, base + 33, validLen);
        end
    end
end

function Y = accumulateQ4KSegment(Y, qVals, gain, offset, xChunk, scaleX, chunkStart, validLen)
    lastPos = min(chunkStart + 31, validLen);
    if lastPos < chunkStart
        return;
    end
    localLen = lastPos - chunkStart + 1;
    cols = chunkStart:lastPos;
    xLocal = xChunk(cols, :);
    qLocal = qVals(:, 1:localLen);
    accQ = intMatmulAccInt32(int32(qLocal), xLocal);
    sumX = sum(xLocal, 1, 'native');
    Y = Y + single(accQ) .* (gain * scaleX);
    Y = Y - (offset * single(sumX)) .* scaleX;
end

function Y = accumulateIntRowChunk(Y, qRow, gain, xChunk, scale, globalRowPos, localRowPos, validLen, totalRows)
    if localRowPos > validLen
        return;
    end
    scaleX = activationScaleSlice(scale, globalRowPos, 1, totalRows);
    acc = int32(qRow(:)) .* int32(xChunk(localRowPos, :));
    Y = Y + single(acc) .* (gain * scaleX);
end

function acc = intMatmulAccInt32(W_q, X_q)
    [m, k_w] = size(W_q);
    [k_x, n] = size(X_q);
    if k_w ~= k_x
        error('qwen2_quant:linear_dispatch:ShapeMismatch', ...
            'Inner dimensions must agree: size(W,2)=%d, size(X,1)=%d', k_w, k_x);
    end

    acc = zeros(m, n, 'int32');
    for kk = 1:k_w
        acc = acc + int32(W_q(:, kk)) .* int32(X_q(kk, :));
    end
end

function s = activationScaleSlice(scale, idxStart, validLen, totalRows)
    if validLen <= 0 || idxStart > totalRows
        s = single(1);
        return;
    end
    if isscalar(scale)
        s = single(scale);
        return;
    end
    if isrow(scale)
        s = single(scale);
        return;
    end

    groups = size(scale, 1);
    groupSize = totalRows / groups;
    g = floor((idxStart - 1) / groupSize) + 1;
    maxG = floor((idxStart + validLen - 2) / groupSize) + 1;
    if g ~= maxG
        error('qwen2_quant:linear_dispatch:GroupCrossing', ...
            'Activation group crosses GGUF block boundary. groupSize=%d idxStart=%d len=%d', groupSize, idxStart, validLen);
    end
    s = single(scale(g, :));
end

function [M, K, blocksPerRow] = getWeightShape32(W)
    dims = W.Dims;
    if W.NeedsTranspose
        M = dims(2);
        K = dims(1);
    else
        M = dims(1);
        K = dims(2);
    end
    blocksPerRow = ceil(K / 32);
end

function [M, K, blocksPerRow] = getWeightShape256(W)
    dims = W.Dims;
    if W.NeedsTranspose
        M = dims(2);
        K = dims(1);
    else
        M = dims(1);
        K = dims(2);
    end
    blocksPerRow = ceil(K / 256);
end

function data = getFloatWeight(W)
    if isa(W, 'qwen2_quant.internal.quantized_weight')
        data = single(W.dequantize());
    elseif isa(W, 'dlarray')
        data = single(extractdata(W));
    else
        data = single(W);
    end
end

function data = getNumeric2D(X)
    if isa(X, 'dlarray')
        data = single(extractdata(X));
    elseif isnumeric(X)
        data = single(X);
    else
        error('qwen2_quant:linear_dispatch:UnsupportedActivation', ...
            'Unsupported activation type: %s', class(X));
    end

    if ndims(data) > 2
        data = reshape(data, size(data, 1), []);
    end
end

function value = getMetaString(meta, fieldName, defaultValue)
    value = defaultValue;
    if isfield(meta, fieldName)
        value = char(string(meta.(fieldName)));
    end
end

function value = getMetaNumber(meta, fieldName, defaultValue)
    value = defaultValue;
    if isfield(meta, fieldName)
        value = double(meta.(fieldName));
    end
end