function result = validate_full_network(inputIds)
%VALIDATE_FULL_NETWORK Compare qwen2 and qwen2_runtime whole-network outputs.

    if nargin < 1
        inputIds = double([151644 872 374 264 11133 429 15836 13 151645]);
    end

    projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    paramsFile = fullfile(projectRoot, 'qwen_params.mat');

    params = qwen2.load(paramsFile);
    params.RuntimeConfig = qwen2_runtime.defaultRuntimeConfig();
    params.RuntimeConfig.LinearMode = 'float';

    ids = double(inputIds);
    if size(ids, 1) > size(ids, 2)
        ids = ids';
    end

    [refLogits, refStates] = qwen2.model(ids, params, []);
    [dutLogits, dutStates] = qwen2_runtime.model(ids, params, [], struct('RuntimeConfig', struct('LinearMode', 'float')));

    diff = abs(single(dutLogits) - single(refLogits));
    result = struct();
    result.InputIds = ids;
    result.MaxAbsDiff = max(diff(:));
    result.MeanAbsDiff = mean(diff(:));
    result.NextTokenRef = nextToken(refLogits);
    result.NextTokenDUT = nextToken(dutLogits);
    result.NextTokenMatch = isequal(result.NextTokenRef, result.NextTokenDUT);
    result.LayerStateCountMatch = numel(dutStates) == numel(refStates);

    fprintf('full_network max diff : %.6g\n', result.MaxAbsDiff);
    fprintf('full_network mean diff: %.6g\n', result.MeanAbsDiff);
    fprintf('next token ref/dut   : %d / %d\n', result.NextTokenRef, result.NextTokenDUT);
    fprintf('next token match     : %d\n', result.NextTokenMatch);
end

function token = nextToken(logits)
    [~, idx] = max(logits(:, end, 1), [], 1);
    token = double(idx) - 1;
end
