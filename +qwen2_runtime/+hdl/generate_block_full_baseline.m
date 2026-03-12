function info = generate_block_full_baseline()
%GENERATE_BLOCK_FULL_BASELINE Generate reduced-dimension full block RTL baseline.

    projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    cfg = coder.config('hdl');
    cfg.TargetLanguage = 'Verilog';
    cfg.TreatRealsInGeneratedCodeAs = 'Warning';
    cfg.FloatingPointLibrary = 'NativeFloatingPoint';
    cfg.TreatIOThresholdAs = 'Warning';
    cfg.IOThreshold = 10000;
    cfg.GenerateHDLTestBench = true;
    cfg.TestBenchName = 'qwen2_runtime_hdl_block_entry_tb';
    cfg.GenerateCosimTestBench = false;
    cfg.SynthesizeGeneratedCode = false;
    cfg.Workflow = 'Generic ASIC/FPGA';
    cfg.InstantiateFunctions = false;

    outDir = fullfile(projectRoot, 'codegen', 'hdl_block_full_baseline', 'qwen2_runtime_hdl_block_entry');
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    args = qwen2_runtime.hdl.block_entry_baseline_args();
    codegen('-config', cfg, 'qwen2_runtime_hdl_block_entry', '-args', args, '-d', outDir, '-I', projectRoot);

    info = struct();
    info.OutputDir = outDir;
end