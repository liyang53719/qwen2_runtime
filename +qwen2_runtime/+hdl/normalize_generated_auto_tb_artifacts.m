function info = normalize_generated_auto_tb_artifacts(outDir, designName)
%NORMALIZE_GENERATED_AUTO_TB_ARTIFACTS Normalize HDL Coder auto-TB outputs for a design.

    hdlsrcDir = fullfile(outDir, designName, 'hdlsrc');
    if ~exist(hdlsrcDir, 'dir')
        error('normalize_generated_auto_tb_artifacts:MissingHDLSrc', ...
            'HDL source directory not found: %s', hdlsrcDir);
    end

    stimulusFiles = {
        'start.dat'
        'score_token_0.dat'
        'score_token_1.dat'
        'value_token_0.dat'
        'value_token_1.dat'
        'value_token_2.dat'
        'value_token_3.dat'
        'value_token_4.dat'
        'value_token_5.dat'
        'value_token_6.dat'
        'value_token_7.dat'
        'token_valid.dat'
        'token_last.dat'
    };

    expectedFiles = {
        'out_valid_expected.dat'
        'block_out_0_0_expected.dat'
        'block_out_0_1_expected.dat'
        'block_out_0_2_expected.dat'
        'block_out_0_3_expected.dat'
        'block_out_0_4_expected.dat'
        'block_out_0_5_expected.dat'
        'block_out_0_6_expected.dat'
        'block_out_0_7_expected.dat'
    };

    [stimulusCount, stimulusCollapsed] = normalizeStimulusFiles(hdlsrcDir, stimulusFiles);
    [expectedCount, expectedCollapsed] = normalizeExpectedFiles(hdlsrcDir, expectedFiles, stimulusCount);

    if stimulusCount ~= expectedCount
        error('normalize_generated_auto_tb_artifacts:CountMismatch', ...
            'Stimulus and expected sample counts differ after normalization: stimulus=%d expected=%d', ...
            stimulusCount, expectedCount);
    end

    tbPath = fullfile(hdlsrcDir, sprintf('%s_tb.v', designName));
    updateGeneratedTb(tbPath, stimulusCount, designName);

    info = struct();
    info.HDLSrcDir = hdlsrcDir;
    info.SampleCount = stimulusCount;
    info.CollapsedFiles = [stimulusFiles(stimulusCollapsed); expectedFiles(expectedCollapsed)];
end


function [sampleCount, collapsedFlags] = normalizeStimulusFiles(hdlsrcDir, fileNames)
    rawCounts = zeros(numel(fileNames), 1);
    duplicatePairs = false(numel(fileNames), 1);
    collapsedFlags = false(numel(fileNames), 1);
    cachedLines = cell(numel(fileNames), 1);

    for idx = 1:numel(fileNames)
        filePath = fullfile(hdlsrcDir, fileNames{idx});
        lines = readNonEmptyLines(filePath);
        if isempty(lines)
            error('normalize_generated_auto_tb_artifacts:EmptyFile', 'Auto TB data file is empty: %s', filePath);
        end
        cachedLines{idx} = lines;
        rawCounts(idx) = numel(lines);
        duplicatePairs(idx) = hasDuplicatePairs(lines);
    end

    uniqueCounts = unique(rawCounts);
    if numel(uniqueCounts) ~= 1
        error('normalize_generated_auto_tb_artifacts:StimulusCountMismatch', ...
            'Stimulus files have inconsistent sample counts: %s', mat2str(rawCounts.'));
    end

    sampleCount = uniqueCounts;
    if all(duplicatePairs)
        sampleCount = sampleCount / 2;
        for idx = 1:numel(fileNames)
            filePath = fullfile(hdlsrcDir, fileNames{idx});
            lines = cachedLines{idx}(1:2:end);
            writeLines(filePath, lines);
            collapsedFlags(idx) = true;
        end
    end
end


function [sampleCount, collapsedFlags] = normalizeExpectedFiles(hdlsrcDir, fileNames, stimulusCount)
    rawCounts = zeros(numel(fileNames), 1);
    duplicatePairs = false(numel(fileNames), 1);
    collapsedFlags = false(numel(fileNames), 1);
    cachedLines = cell(numel(fileNames), 1);

    for idx = 1:numel(fileNames)
        filePath = fullfile(hdlsrcDir, fileNames{idx});
        lines = readNonEmptyLines(filePath);
        if isempty(lines)
            error('normalize_generated_auto_tb_artifacts:EmptyFile', 'Auto TB data file is empty: %s', filePath);
        end
        cachedLines{idx} = lines;
        rawCounts(idx) = numel(lines);
        duplicatePairs(idx) = hasDuplicatePairs(lines);
    end

    uniqueCounts = unique(rawCounts);
    if numel(uniqueCounts) ~= 1
        error('normalize_generated_auto_tb_artifacts:ExpectedCountMismatch', ...
            'Expected files have inconsistent sample counts: %s', mat2str(rawCounts.'));
    end

    sampleCount = uniqueCounts;
    if sampleCount == 2 * stimulusCount && all(duplicatePairs)
        sampleCount = stimulusCount;
        for idx = 1:numel(fileNames)
            filePath = fullfile(hdlsrcDir, fileNames{idx});
            lines = cachedLines{idx}(1:2:end);
            writeLines(filePath, lines);
            collapsedFlags(idx) = true;
        end
    end
end


function updateGeneratedTb(tbPath, sampleCount, designName)
    wrapCount = sampleCount - 1;
    text = fileread(tbPath);
    widthTokens = regexp(text, 'reg \[(\d+):(\d+)\] start_addr', 'tokens', 'once');
    if isempty(widthTokens)
        widthInfo = regexp(text, '(?<=start_addr >= )(?<width>\d+)''b[01]+', 'names', 'once');
        if isempty(widthInfo)
            error('normalize_generated_auto_tb_artifacts:NoBitWidth', ...
                'Unable to infer counter width from generated TB: %s', tbPath);
        end
        counterWidth = str2double(widthInfo.width);
    else
        counterWidth = str2double(widthTokens{1}) + 1;
    end
    wrapCountBits = sprintf("%d'b%s", counterWidth, dec2bin(wrapCount, counterWidth));

    text = regexprep(text, '(?<=start_addr >= )\d+''b[01]+', wrapCountBits);
    text = regexprep(text, '(?<=start_addr != )\d+''b[01]+', wrapCountBits);
    text = regexprep(text, '(?<=block_out_0_addr >= )\d+''b[01]+', wrapCountBits);
    text = regexprep(text, '(?<=block_out_0_addr != )\d+''b[01]+', wrapCountBits);
    text = injectFsdbDump(text, designName);
    text = strrep(text, '$stop;', '$finish;');

    fid = fopen(tbPath, 'w');
    if fid == -1
        error('normalize_generated_auto_tb_artifacts:OpenFailed', 'Unable to open generated TB for writing: %s', tbPath);
    end
    cleaner = onCleanup(@() fclose(fid));
    fwrite(fid, text);
end


function tf = hasDuplicatePairs(lines)
    tf = false;
    if mod(numel(lines), 2) == 0
        oddLines = lines(1:2:end);
        evenLines = lines(2:2:end);
        tf = all(strcmp(oddLines, evenLines));
    end
end


function text = injectFsdbDump(text, designName)
    if ~contains(text, '$fsdbDumpfile')
        tbName = sprintf('%s_tb', designName);
        dutInstance = sprintf('u_%s', designName);
        fsdbName = sprintf('%s_tb.fsdb', designName);
        marker = sprintf('module %s;', tbName);
        injection = sprintf([ ...
            'module %s;\n\n' ...
            '  initial\n' ...
            '    begin : fsdb_dump\n' ...
            '      $fsdbDumpfile("%s");\n' ...
            '      $fsdbDumpvars(0, %s);\n' ...
            '      $fsdbDumpvars(0, %s.%s);\n' ...
            '    end\n\n' ...
            '  initial\n' ...
            '    begin : sim_timeout\n' ...
            '      # (100000);\n' ...
            '      $display("ERROR: auto TB timeout");\n' ...
            '      $finish;\n' ...
            '    end'], tbName, fsdbName, tbName, tbName, dutInstance);
        text = strrep(text, marker, injection);
    end
end


function lines = readNonEmptyLines(filePath)
    rawText = fileread(filePath);
    rawLines = regexp(rawText, '\r\n|\n|\r', 'split');
    if ~isempty(rawLines) && isempty(rawLines{end})
        rawLines(end) = [];
    end
    keepMask = ~cellfun(@isempty, rawLines);
    lines = rawLines(keepMask);
end


function writeLines(filePath, lines)
    fid = fopen(filePath, 'w');
    if fid == -1
        error('normalize_generated_auto_tb_artifacts:OpenFailed', 'Unable to open data file for writing: %s', filePath);
    end
    cleaner = onCleanup(@() fclose(fid));
    fprintf(fid, '%s\n', lines{:});
end