function info = generate_attention_token_step_sram_baseline(maxCacheLen)
%GENERATE_ATTENTION_TOKEN_STEP_SRAM_BASELINE Generate HDL for external-KV token-step attention top.

    if nargin == 0
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

    outDir = fullfile(projectRoot, 'codegen', 'hdl_attention_token_step_sram_baseline', 'qwen2_runtime_hdl_attention_token_step_sram_entry');
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    args = qwen2_runtime.hdl.attention_token_step_sram_args(maxCacheLen, true);
    codegen('-config', cfg, 'qwen2_runtime_hdl_attention_token_step_sram_entry', '-args', args, '-d', outDir, '-I', projectRoot);

    info = struct();
    info.OutputDir = outDir;
    info.MaxCacheLength = maxCacheLen;
end