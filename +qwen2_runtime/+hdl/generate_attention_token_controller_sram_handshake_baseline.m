function info = generate_attention_token_controller_sram_handshake_baseline(matParamsFile, maxCacheLen, timeoutSeconds, runMode)
%GENERATE_ATTENTION_TOKEN_CONTROLLER_SRAM_HANDSHAKE_BASELINE Generate real-dimension RTL for handshake attention controller.

    if nargin < 1 || strlength(string(matParamsFile)) == 0
        matParamsFile = resolvePreferredParamsFile();
    end
    if nargin < 2
        maxCacheLen = qwen2_runtime.defaultHardwareHDLConfig().HDLMaxCacheLength;
    end
    if nargin < 3 || isempty(timeoutSeconds)
        timeoutSeconds = 20;
    end
    if nargin < 4 || strlength(string(runMode)) == 0
        runMode = "direct";
    else
        runMode = string(runMode);
    end

    projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    overallTimer = tic;
    fprintf('[handshake_codegen] start mat=%s cache=%d timeout=%.2fs\n', char(string(matParamsFile)), double(maxCacheLen), double(timeoutSeconds));

    outDir = fullfile(projectRoot, 'codegen', 'hdl_attention_token_controller_sram_handshake', 'qwen2_runtime_hdl_attention_token_controller_sram_handshake_entry');
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
    fprintf('[handshake_codegen] output directory ready in %.3fs\n', toc(overallTimer));

    [controllerPayload, prepInfo] = buildControllerCodegenPayload(matParamsFile, maxCacheLen);
    fprintf('[handshake_codegen] args ready in %.3fs (cache_hit=%d load=%.3fs freqs=%.3fs wrap=%.3fs)\n', ...
        prepInfo.TotalSeconds, prepInfo.CacheHit, prepInfo.LoadSeconds, prepInfo.FreqSeconds, prepInfo.WrapSeconds);

    screenerInfo = runControllerScreener();
    fprintf('[handshake_codegen] screener finished in %.3fs\n', screenerInfo.ElapsedSeconds);
    runControllerStructuralDiagnostics(projectRoot, outDir);

    if prepInfo.TotalSeconds >= timeoutSeconds
        info = struct();
        info.OutputDir = outDir;
        info.MaxCacheLength = maxCacheLen;
        info.TimeoutSeconds = timeoutSeconds;
        info.TimedOut = true;
        info.UsedAsyncTimeout = false;
        info.TimeoutEnforced = false;
        info.ArgPrep = prepInfo;
        info.Screener = screenerInfo;
        info.ConformanceReport = findConformanceReport(outDir);
        info.CodegenElapsedSeconds = 0;
        info.TotalElapsedSeconds = toc(overallTimer);
        fprintf('[handshake_codegen] timeout consumed by arg preparation, skipping codegen\n');
        return;
    end

    codegenTimer = tic;
    usedAsyncTimeout = false;
    timedOut = false;
    if runMode ~= "direct"
        fprintf('[handshake_codegen] legacy runMode=%s requested; forcing direct MATLAB logging\n', char(runMode));
    end

    if timeoutSeconds > 0
        remainingSeconds = max(double(timeoutSeconds) - prepInfo.TotalSeconds, 0);
        fprintf('[handshake_codegen] timeout budget remaining before direct codegen: %.3fs\n', remainingSeconds);
        fprintf('[handshake_codegen] direct MATLAB codegen cannot be force-stopped in-process; timeout is advisory once codegen starts\n');
    end

    fprintf('[handshake_codegen] direct mode codegen start\n');
    cfg = makeHdlCodegenConfig();
    fprintf('[handshake_codegen] configured HDL coder in %.3fs\n', toc(overallTimer));
    codegenError = [];
    try
        runControllerCodegen(cfg, outDir, projectRoot, buildControllerArgs(controllerPayload));
    catch err
        codegenError = err;
    end

    codegenElapsed = toc(codegenTimer);
    fprintf('[handshake_codegen] codegen phase finished in %.3fs timed_out=%d\n', codegenElapsed, timedOut);

    conformanceReport = findConformanceReport(outDir);
    printConformanceDiagnostics(conformanceReport);

    info = struct();
    info.OutputDir = outDir;
    info.MaxCacheLength = maxCacheLen;
    info.TimeoutSeconds = timeoutSeconds;
    info.TimedOut = timedOut;
    info.UsedAsyncTimeout = usedAsyncTimeout;
    info.TimeoutEnforced = false;
    info.ArgPrep = prepInfo;
    info.Screener = screenerInfo;
    info.ConformanceReport = conformanceReport;
    info.CodegenElapsedSeconds = codegenElapsed;
    info.TotalElapsedSeconds = toc(overallTimer);
    fprintf('[handshake_codegen] total elapsed %.3fs\n', info.TotalElapsedSeconds);

    if ~isempty(codegenError)
        rethrow(codegenError);
    end
end

function [payload, info] = buildControllerCodegenPayload(matParamsFile, maxCacheLen)
    persistent cachedKey cachedPayload

    buildTimer = tic;
    cacheKey = sprintf('%s|%d', char(string(matParamsFile)), double(maxCacheLen));
    if ~isempty(cachedKey) && strcmp(cachedKey, cacheKey)
        payload = cachedPayload;
        loadSeconds = 0;
        freqSeconds = 0;
        cacheHit = true;
    else
        cacheHit = false;
        loadTimer = tic;
        raw = loadControllerRaw(matParamsFile);
        hp = raw.Hyperparameters;
        cfgRun = qwen2_runtime.defaultHardwareHDLConfig();
        cfgRun.HDLMaxCacheLength = maxCacheLen;
        h_token = fi(single(raw.EmbedToken), true, cfgRun.HDLLinearInputWordLength, cfgRun.HDLLinearInputFractionLength);
        read_key_data = fi(zeros(hp.HeadDim, hp.NumKVHeads), true, cfgRun.HDLLinearInputWordLength, cfgRun.HDLLinearInputFractionLength);
        read_value_data = fi(zeros(hp.HeadDim, hp.NumKVHeads), true, cfgRun.HDLLinearInputWordLength, cfgRun.HDLLinearInputFractionLength);
        loadSeconds = toc(loadTimer);

        freqTimer = tic;
        freqs_cis = buildFreqsCis(raw, cfgRun, maxCacheLen);
        freqSeconds = toc(freqTimer);

        controllerWeights = struct();
        controllerWeights.q_proj = raw.ControllerWeights.q_proj;
        controllerWeights.k_proj = raw.ControllerWeights.k_proj;
        controllerWeights.v_proj = raw.ControllerWeights.v_proj;
        controllerWeights.o_proj = raw.ControllerWeights.o_proj;
        if isfield(raw.ControllerWeights, 'q_bias'), controllerWeights.q_bias = raw.ControllerWeights.q_bias; end
        if isfield(raw.ControllerWeights, 'k_bias'), controllerWeights.k_bias = raw.ControllerWeights.k_bias; end
        if isfield(raw.ControllerWeights, 'v_bias'), controllerWeights.v_bias = raw.ControllerWeights.v_bias; end
        if isfield(raw.ControllerWeights, 'o_bias'), controllerWeights.o_bias = raw.ControllerWeights.o_bias; end

        payload = struct();
        payload.HToken = h_token;
        payload.ReadKeyData = read_key_data;
        payload.ReadValueData = read_value_data;
        payload.ControllerWeights = controllerWeights;
        payload.FreqsCis = freqs_cis;
        payload.HyperParameters = hp;
        payload.RuntimeConfig = cfgRun;
        cachedKey = cacheKey;
        cachedPayload = payload;
    end

    wrapTimer = tic;
    info = struct();
    info.CacheHit = cacheHit;
    info.LoadSeconds = loadSeconds;
    info.FreqSeconds = freqSeconds;
    info.WrapSeconds = toc(wrapTimer);
    info.TotalSeconds = toc(buildTimer);
end

function outDir = runControllerCodegen(cfg, outDir, projectRoot, controllerArgs)
    codegen('-config', cfg, 'qwen2_hdl_attn_hs_entry', '-args', controllerArgs, '-d', outDir, '-I', projectRoot);
end

function cfg = makeHdlCodegenConfig()
    cfg = coder.config('hdl');
    cfg.TargetLanguage = 'Verilog';
    cfg.GenerateHDLTestBench = false;
    cfg.GenerateCosimTestBench = false;
    cfg.SynthesizeGeneratedCode = false;
    cfg.Workflow = 'Generic ASIC/FPGA';
    cfg.InstantiateFunctions = false;
end

function screenerInfo = runControllerScreener()
    screenerTimer = tic;
    fprintf('[handshake_codegen] running coder.screener on short handshake entry\n');

    screenerInfo = struct();
    screenerInfo.Result = coder.screener('qwen2_hdl_attn_hs_entry');
    screenerInfo.ElapsedSeconds = toc(screenerTimer);
    disp(screenerInfo.Result);
end

function runControllerStructuralDiagnostics(projectRoot, outDir)
    fprintf('[handshake_codegen] running native structural diagnostics on handshake engine\n');

    engineScreener = coder.screener('qwen2_runtime.hdl.attention_token_step_sram_engine_step');
    disp(engineScreener.Messages);
    disp(engineScreener.Files);

    engineFile = fullfile(projectRoot, '+qwen2_runtime', '+hdl', 'attention_token_step_sram_engine_step.m');
    analyzerMessages = checkcode(engineFile, '-cyc');
    disp(analyzerMessages);

    reportPath = findConformanceReport(outDir);
    printPriorConformanceSourceContext(reportPath, engineFile);
end

function controllerArgs = buildControllerArgs(payload)
    controllerArgs = {false, payload.HToken, uint16(0), uint16(1), payload.ReadKeyData, payload.ReadValueData, false, ...
        coder.Constant(payload.ControllerWeights), coder.Constant(payload.FreqsCis), ...
        coder.Constant(payload.HyperParameters), coder.Constant(payload.RuntimeConfig)};
end

function raw = loadControllerRaw(matParamsFile)
    if contains(string(matParamsFile), "qwen_handshake_controller_params")
        compact = load(matParamsFile);
        raw = struct();
        raw.Hyperparameters = compact.Hyperparameters;
        raw.EmbedToken = single(compact.HToken);
        raw.ControllerWeights = compact.ControllerWeights;
        if isfield(compact, 'FreqsMaxCacheLength')
            raw.FreqsMaxCacheLength = double(compact.FreqsMaxCacheLength);
        end
        if isfield(compact, 'FreqsCos')
            raw.FreqsCos = single(compact.FreqsCos);
        end
        if isfield(compact, 'FreqsSin')
            raw.FreqsSin = single(compact.FreqsSin);
        end
        return;
    end

    probe = whos('-file', matParamsFile);
    names = string({probe.name});
    if any(names == "ControllerWeights") && any(names == "Hyperparameters") && any(names == "HToken")
        compact = load(matParamsFile, 'ControllerWeights', 'Hyperparameters', 'HToken');
        raw = struct();
        raw.Hyperparameters = compact.Hyperparameters;
        raw.EmbedToken = single(compact.HToken);
        raw.ControllerWeights = compact.ControllerWeights;
        return;
    end

    hpFields = {'NumLayers', 'HiddenSize', 'NumHeads', 'NumKVHeads', 'HeadDim', 'VocabSize', 'RopeTheta'};
    layerFields = {'layer_0_self_attn_q_proj', 'layer_0_self_attn_k_proj', 'layer_0_self_attn_v_proj', 'layer_0_self_attn_o_proj', ...
        'layer_0_self_attn_q_bias', 'layer_0_self_attn_k_bias', 'layer_0_self_attn_v_bias'};
    loaded = load(matParamsFile, hpFields{:}, layerFields{:});

    raw = struct();
    raw.Hyperparameters = struct();
    for idx = 1:numel(hpFields)
        fieldName = hpFields{idx};
        raw.Hyperparameters.(fieldName) = double(loaded.(fieldName));
    end
    if ~isfield(raw.Hyperparameters, 'RopeTheta')
        raw.Hyperparameters.RopeTheta = 10000.0;
    end

    try
        matObj = matfile(matParamsFile);
        raw.EmbedToken = single(matObj.embed_tokens(:, 151644 + 1));
    catch
        embedLoaded = load(matParamsFile, 'embed_tokens');
        raw.EmbedToken = single(embedLoaded.embed_tokens(:, 151644 + 1));
    end

    raw.ControllerWeights = struct();
    raw.ControllerWeights.q_proj = single(loaded.layer_0_self_attn_q_proj);
    raw.ControllerWeights.k_proj = single(loaded.layer_0_self_attn_k_proj);
    raw.ControllerWeights.v_proj = single(loaded.layer_0_self_attn_v_proj);
    raw.ControllerWeights.o_proj = single(loaded.layer_0_self_attn_o_proj);
    if isfield(loaded, 'layer_0_self_attn_q_bias')
        raw.ControllerWeights.q_bias = forceColumn(single(loaded.layer_0_self_attn_q_bias));
    end
    if isfield(loaded, 'layer_0_self_attn_k_bias')
        raw.ControllerWeights.k_bias = forceColumn(single(loaded.layer_0_self_attn_k_bias));
    end
    if isfield(loaded, 'layer_0_self_attn_v_bias')
        raw.ControllerWeights.v_bias = forceColumn(single(loaded.layer_0_self_attn_v_bias));
    end
end

function value = forceColumn(value)
    if isvector(value) && size(value, 2) > 1
        value = value(:);
    end
end

function reportPath = findConformanceReport(outDir)
    reportPath = '';
    matches = findFilesRecursive(outDir, '*_hdl_conformance_report.html');
    if isempty(matches)
        return;
    end

    newestTime = -inf;
    for idx = 1:numel(matches)
        fileInfo = dir(matches{idx});
        if isempty(fileInfo)
            continue;
        end
        if fileInfo.datenum > newestTime
            newestTime = fileInfo.datenum;
            reportPath = matches{idx};
        end
    end
end

function printConformanceDiagnostics(reportPath)
    if strlength(string(reportPath)) == 0 || exist(reportPath, 'file') ~= 2
        fprintf('[handshake_codegen] no HDL conformance report found under output directory\n');
        return;
    end

    fprintf('[handshake_codegen] HDL conformance report: %s\n', reportPath);
    summaryLines = extractConformanceSummary(reportPath);
    for idx = 1:numel(summaryLines)
        fprintf('[handshake_codegen] %s\n', summaryLines{idx});
    end
end

function printPriorConformanceSourceContext(reportPath, engineFile)
    if strlength(string(reportPath)) == 0 || exist(reportPath, 'file') ~= 2
        return;
    end

    html = fileread(reportPath);
    token = regexp(html, 'attention_token_step_sram_engine_step.*?line\s+(\d+),\s+column\s+(\d+)', 'tokens', 'once');
    if isempty(token)
        return;
    end

    lineNumber = str2double(token{1});
    startLine = max(1, lineNumber - 12);
    endLine = lineNumber + 24;
    fprintf('[handshake_codegen] previous conformance source context for attention_token_step_sram_engine_step:%d\n', lineNumber);
    dbtype(engineFile, sprintf('%d:%d', startLine, endLine));
end

function summaryLines = extractConformanceSummary(reportPath)
    html = fileread(reportPath);
    text = regexprep(html, '<[^>]+>', sprintf('\n'));
    text = strrep(text, '&nbsp;', ' ');
    text = strrep(text, '&gt;', '>');
    text = strrep(text, '&lt;', '<');
    text = regexprep(text, '[ \t]+', ' ');
    rawLines = regexp(text, '\r\n|\n|\r', 'split');
    rawLines = rawLines(~cellfun(@isempty, rawLines));

    patterns = {
        'HDL Conformance check complete', ...
        'unsupported unbounded loop structure', ...
        'MATLAB HDL Coder failed', ...
        'attention_token_step_sram_engine_step'};
    summaryLines = {};
    for idx = 1:numel(rawLines)
        line = strtrim(rawLines{idx});
        if isempty(line)
            continue;
        end
        for patIdx = 1:numel(patterns)
            if contains(line, patterns{patIdx})
                summaryLines{end+1} = line; %#ok<AGROW>
                break;
            end
        end
    end

    if isempty(summaryLines)
        summaryLines = {'Conformance report generated, but no known summary lines were extracted.'};
        return;
    end

    [~, uniqueIdx] = unique(summaryLines, 'stable');
    summaryLines = summaryLines(sort(uniqueIdx));
end

function matches = findFilesRecursive(rootDir, pattern)
    matches = {};
    if exist(rootDir, 'dir') ~= 7
        return;
    end

    entries = dir(rootDir);
    for idx = 1:numel(entries)
        entry = entries(idx);
        if entry.isdir
            if strcmp(entry.name, '.') || strcmp(entry.name, '..')
                continue;
            end
            childMatches = findFilesRecursive(fullfile(rootDir, entry.name), pattern);
            if ~isempty(childMatches)
                matches = [matches, childMatches]; %#ok<AGROW>
            end
            continue;
        end

        if ~isempty(regexp(entry.name, wildcardPatternToRegexp(pattern), 'once'))
            matches{end+1} = fullfile(rootDir, entry.name); %#ok<AGROW>
        end
    end
end

function expr = wildcardPatternToRegexp(pattern)
    expr = regexptranslate('wildcard', pattern);
end

function freqs_cis = buildFreqsCis(raw, cfgRun, maxCacheLen)
    expectedCols = double(maxCacheLen) + 8;
    if isfield(raw, 'FreqsMaxCacheLength') && isfield(raw, 'FreqsCos') && isfield(raw, 'FreqsSin') && ...
            double(raw.FreqsMaxCacheLength) == double(maxCacheLen) && size(raw.FreqsCos, 2) >= expectedCols && size(raw.FreqsSin, 2) >= expectedCols
        freqs_cis = struct();
        freqs_cis.Cos = fi(single(raw.FreqsCos(:, 1:expectedCols)), true, cfgRun.HDLLinearInputWordLength, cfgRun.HDLLinearInputFractionLength);
        freqs_cis.Sin = fi(single(raw.FreqsSin(:, 1:expectedCols)), true, cfgRun.HDLLinearInputWordLength, cfgRun.HDLLinearInputFractionLength);
        return;
    end

    freqsFull = transformer.layer.precomputeFreqsCis(raw.Hyperparameters.HeadDim, expectedCols, raw.Hyperparameters.RopeTheta);
    freqs_cis = struct();
    freqs_cis.Cos = fi(single(real(freqsFull(:, 1:expectedCols))), true, cfgRun.HDLLinearInputWordLength, cfgRun.HDLLinearInputFractionLength);
    freqs_cis.Sin = fi(single(imag(freqsFull(:, 1:expectedCols))), true, cfgRun.HDLLinearInputWordLength, cfgRun.HDLLinearInputFractionLength);
end

function matParamsFile = resolvePreferredParamsFile()
    projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    compactMatFile = fullfile(projectRoot, 'qwen_handshake_controller_params_v73.mat');
    if exist(compactMatFile, 'file') == 2
        matParamsFile = compactMatFile;
        return;
    end
    matParamsFile = fullfile(projectRoot, 'qwen_params.mat');
end