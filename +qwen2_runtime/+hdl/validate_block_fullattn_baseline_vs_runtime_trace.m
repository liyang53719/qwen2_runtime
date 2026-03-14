function result = validate_block_fullattn_baseline_vs_runtime_trace()
%VALIDATE_BLOCK_FULLATTN_BASELINE_VS_RUNTIME_TRACE Compare full-attention block baseline against runtime block trace.

    projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    paramsFile = fullfile(projectRoot, 'qwen_params.mat');
    tokenizerPath = fullfile(projectRoot, 'qwen_model');

    promptText = "The process of photosynthesis is how plants convert light energy into chemical energy. Chlorophyll absorbs sunlight and uses it to convert carbon dioxide and water into glucose.";
    prompt = sprintf("Summarize this: %s Summary:", promptText);

    params = qwen2.load(paramsFile);
    cfg = qwen2_runtime.defaultRuntimeConfig();
    cfg.LinearMode = 'float';
    cfg.TraceTensors = true;
    params.RuntimeConfig = cfg;

    tokenizer = qwen2.tokenizer.QwenTokenizer(tokenizerPath);
    inputIds = double(tokenizer.encode(prompt));
    if size(inputIds, 1) > size(inputIds, 2)
        inputIds = inputIds';
    end

    [~, ~, dbg] = qwen2_runtime.model(inputIds, params, [], struct('RuntimeConfig', cfg));
    blockTrace = dbg.TensorTrace.blocks{1};
    tokenPos = min(16, numel(inputIds));

    F32 = fimath('RoundingMethod', 'Floor', 'OverflowAction', 'Saturate', 'ProductMode', 'SpecifyPrecision', 'ProductWordLength', 24, 'ProductFractionLength', 14, 'SumMode', 'SpecifyPrecision', 'SumWordLength', 32, 'SumFractionLength', 14);
    inputVec = fi(single(blockTrace.input_norm(1:8, tokenPos, 1)), true, 32, 14, F32);
    attnVec = fi(single(blockTrace.attention.attn_mix(1:8, tokenPos, 1)), true, 32, 14, F32);
    residualSeed = fi(zeros(8, 1), true, 32, 14, F32);
    reference = qwen2_runtime.hdl.residual_add_step(inputVec, attnVec);

    clear qwen2_runtime.hdl.block_fullattn_baseline_step
    [dut, valid] = qwen2_runtime.hdl.block_fullattn_baseline_step(true, inputVec, attnVec, residualSeed);

    diff = abs(double(storedInteger(dut)) - double(storedInteger(reference)));
    result = struct();
    result.Valid = valid;
    result.ExactMatch = isequal(storedInteger(dut), storedInteger(reference)) && valid;
    result.MaxAbsIntDiff = max(diff(:));
    result.MeanAbsIntDiff = mean(diff(:));
    result.DUT = dut;
    result.Reference = reference;
    fprintf('fullattn_block_vs_runtime exact   : %d\n', result.ExactMatch);
    fprintf('fullattn_block_vs_runtime max int : %.6g\n', result.MaxAbsIntDiff);
    fprintf('fullattn_block_vs_runtime mean int: %.6g\n', result.MeanAbsIntDiff);
end
