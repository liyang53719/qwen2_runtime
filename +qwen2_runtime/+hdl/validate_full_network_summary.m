function result = validate_full_network_summary()
%VALIDATE_FULL_NETWORK_SUMMARY Compare full-network text generation outputs.

    projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    paramsFile = fullfile(projectRoot, 'qwen_params.mat');
    tokenizerPath = fullfile(projectRoot, 'qwen_model');

    promptText = "The process of photosynthesis is how plants convert light energy into chemical energy. Chlorophyll absorbs sunlight and uses it to convert carbon dioxide and water into glucose.";
    promptTemplate = "Summarize this: %s Summary:";
    maxNewTokens = 30;
    topK = 1;

    params = qwen2.load(paramsFile);
    params.RuntimeConfig = qwen2_runtime.defaultRuntimeConfig();
    params.RuntimeConfig.LinearMode = 'float';
    tokenizer = qwen2.tokenizer.QwenTokenizer(tokenizerPath);

    mdlRef = struct('Parameters', params, 'Tokenizer', tokenizer);
    mdlDut = struct('Parameters', params, 'Tokenizer', tokenizer);

    prompt = sprintf(promptTemplate, promptText);
    promptIds = double(tokenizer.encode(prompt));
    if size(promptIds, 1) > size(promptIds, 2)
        promptIds = promptIds';
    end

    refIds = generateIdsReference(mdlRef, promptIds, maxNewTokens, topK);
    dutIds = qwen2_runtime.generate(mdlDut, promptIds, 'MaxNewTokens', maxNewTokens, 'TopK', topK);

    refGenIds = refIds(numel(promptIds)+1:end);
    dutGenIds = dutIds(numel(promptIds)+1:end);
    refText = string(tokenizer.decode(refGenIds));
    dutText = string(tokenizer.decode(dutGenIds));

    result = struct();
    result.PromptText = promptText;
    result.PromptTemplate = promptTemplate;
    result.PromptIds = promptIds;
    result.ReferenceIds = refIds;
    result.RuntimeIds = dutIds;
    result.ReferenceGeneratedIds = refGenIds;
    result.RuntimeGeneratedIds = dutGenIds;
    result.ReferenceText = refText;
    result.RuntimeText = dutText;
    result.TokenIdsMatch = isequal(refIds, dutIds);
    result.TextMatch = strcmp(refText, dutText);

    fprintf('summary token ids match: %d\n', result.TokenIdsMatch);
    fprintf('summary text match     : %d\n', result.TextMatch);
    fprintf('reference summary      : %s\n', refText);
    fprintf('runtime summary        : %s\n', dutText);
end

function ids = generateIdsReference(mdl, inputIds, maxNewTokens, topK)
    params = mdl.Parameters;
    tokenizer = mdl.Tokenizer;

    [logits, state] = qwen2.model(inputIds, params);
    nextId = sampleGreedy(logits, topK);
    ids = [inputIds, nextId];

    for i = 1:maxNewTokens
        if nextId == tokenizer.EosTokenId || nextId == tokenizer.PadTokenId || nextId == 151645
            break;
        end
        [logits, state] = qwen2.model(nextId, params, state);
        nextId = sampleGreedy(logits, topK);
        ids = [ids, nextId]; %#ok<AGROW>
    end
end

function nextId = sampleGreedy(logits, topK)
    lastLogits = logits(:, end, 1);
    [~, idx] = sort(lastLogits, 'descend');
    if topK == 1
        nextId = double(idx(1) - 1);
    else
        nextId = double(idx(1) - 1);
    end
end
