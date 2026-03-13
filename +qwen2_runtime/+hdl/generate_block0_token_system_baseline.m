function info = generate_block0_token_system_baseline(matParamsFile, maxCacheLen)
%GENERATE_BLOCK0_TOKEN_SYSTEM_BASELINE Generate real-dimension RTL for the block-0 system top.

    if nargin < 1 || strlength(string(matParamsFile)) == 0
        matParamsFile = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'qwen_params.mat');
    end
    if nargin < 2
        maxCacheLen = 8;
    end

    projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    cfg = coder.config('hdl');
    cfg.TargetLanguage = 'Verilog';
    cfg.GenerateHDLTestBench = false;
    cfg.GenerateCosimTestBench = false;
    cfg.SynthesizeGeneratedCode = false;
    cfg.Workflow = 'Generic ASIC/FPGA';
    cfg.InstantiateFunctions = false;

    outDir = fullfile(projectRoot, 'codegen', 'hdl_block0_token_system', 'qwen2_runtime_hdl_block0_token_system_entry');
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    args = qwen2_runtime.hdl.block0_token_system_args(matParamsFile, maxCacheLen, true);
    codegen('-config', cfg, 'qwen2_runtime_hdl_block0_token_system_entry', '-args', args, '-d', outDir, '-I', projectRoot);

    info = struct();
    info.OutputDir = outDir;
    info.MaxCacheLength = maxCacheLen;
end