function info = export_handshake_controller_codegen_params(sourceMatFile, v73MatFile, compactMatFile)
%EXPORT_HANDSHAKE_CONTROLLER_CODEGEN_PARAMS Export reusable parameter assets for handshake controller iteration.

    projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    if nargin < 1 || strlength(string(sourceMatFile)) == 0
        sourceMatFile = fullfile(projectRoot, 'qwen_params.mat');
    end
    if nargin < 2 || strlength(string(v73MatFile)) == 0
        v73MatFile = fullfile(projectRoot, 'qwen_params_v73.mat');
    end
    if nargin < 3 || strlength(string(compactMatFile)) == 0
        compactMatFile = fullfile(projectRoot, 'qwen_handshake_controller_params_v73.mat');
    end

    sourceMatFile = char(string(sourceMatFile));
    v73MatFile = char(string(v73MatFile));
    compactMatFile = char(string(compactMatFile));

    exportTimer = tic;
    fprintf('[handshake_params] source=%s\n', sourceMatFile);

    if i_isReusableV73(v73MatFile)
        fprintf('[handshake_params] reusing existing v7.3 copy %s\n', v73MatFile);
        compactSourceFile = v73MatFile;
    else
        if exist(v73MatFile, 'file') == 2
            delete(v73MatFile);
            fprintf('[handshake_params] removed invalid v7.3 copy %s\n', v73MatFile);
        end
        raw = load(sourceMatFile);
        fprintf('[handshake_params] full source loaded in %.3fs\n', toc(exportTimer));
        save(v73MatFile, '-struct', 'raw', '-v7.3');
        fprintf('[handshake_params] wrote v7.3 copy %s in %.3fs\n', v73MatFile, toc(exportTimer));
        compactSourceFile = v73MatFile;
    end

    compact = i_buildCompactParams(compactSourceFile);

    save(compactMatFile, '-struct', 'compact', '-v7');
    fprintf('[handshake_params] wrote compact params %s in %.3fs\n', compactMatFile, toc(exportTimer));

    info = struct();
    info.SourceMatFile = sourceMatFile;
    info.V73MatFile = v73MatFile;
    info.CompactMatFile = compactMatFile;
    info.TotalElapsedSeconds = toc(exportTimer);
end

function value = i_forceColumn(value)
    if isvector(value) && size(value, 2) > 1
        value = value(:);
    end
end

function compact = i_buildCompactParams(sourceMatFile)
    compact = struct();
    compact.TokenId = uint32(151644);

    hpFields = {'NumLayers', 'HiddenSize', 'NumHeads', 'NumKVHeads', 'HeadDim', 'VocabSize', 'RopeTheta'};
    layerFields = {'layer_0_self_attn_q_proj', 'layer_0_self_attn_k_proj', 'layer_0_self_attn_v_proj', 'layer_0_self_attn_o_proj', ...
        'layer_0_self_attn_q_bias', 'layer_0_self_attn_k_bias', 'layer_0_self_attn_v_bias'};
    loaded = load(sourceMatFile, hpFields{:}, layerFields{:});

    compact.Hyperparameters = struct();
    for idx = 1:numel(hpFields)
        fieldName = hpFields{idx};
        compact.Hyperparameters.(fieldName) = double(loaded.(fieldName));
    end

    matObj = matfile(sourceMatFile);
    compact.HToken = single(matObj.embed_tokens(:, double(compact.TokenId) + 1));

    weights = struct();
    weights.q_proj = single(loaded.layer_0_self_attn_q_proj);
    weights.k_proj = single(loaded.layer_0_self_attn_k_proj);
    weights.v_proj = single(loaded.layer_0_self_attn_v_proj);
    weights.o_proj = single(loaded.layer_0_self_attn_o_proj);
    if isfield(loaded, 'layer_0_self_attn_q_bias')
        weights.q_bias = i_forceColumn(single(loaded.layer_0_self_attn_q_bias));
    end
    if isfield(loaded, 'layer_0_self_attn_k_bias')
        weights.k_bias = i_forceColumn(single(loaded.layer_0_self_attn_k_bias));
    end
    if isfield(loaded, 'layer_0_self_attn_v_bias')
        weights.v_bias = i_forceColumn(single(loaded.layer_0_self_attn_v_bias));
    end
    compact.ControllerWeights = weights;
end

function tf = i_isReusableV73(v73MatFile)
    tf = false;
    if exist(v73MatFile, 'file') ~= 2
        return;
    end
    try
        probe = whos('-file', v73MatFile);
        names = string({probe.name});
        tf = any(names == "embed_tokens") && any(names == "HeadDim") && any(names == "layer_0_self_attn_q_proj");
    catch
        tf = false;
    end
end