function info = generate_block_streaming_baseline()
%GENERATE_BLOCK_STREAMING_BASELINE Generate RTL baseline for the streaming block entry.

    projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    cfg = coder.config('hdl');
    cfg.TargetLanguage = 'Verilog';
    cfg.GenerateHDLTestBench = true;
    cfg.TestBenchName = 'qwen2_runtime_hdl_block_skeleton_streaming_entry_tb';
    cfg.GenerateCosimTestBench = false;
    cfg.SynthesizeGeneratedCode = false;
    cfg.Workflow = 'Generic ASIC/FPGA';
    cfg.InstantiateFunctions = false;

    outDir = fullfile(projectRoot, 'codegen', 'hdl_block_streaming_baseline', 'qwen2_runtime_hdl_block_skeleton_streaming_entry');
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    args = qwen2_runtime.hdl.block_skeleton_streaming_step_args();
    codegen('-config', cfg, 'qwen2_runtime_hdl_block_skeleton_streaming_entry', '-args', args, '-d', outDir, '-I', projectRoot);
    qwen2_runtime.hdl.normalize_generated_auto_tb_artifacts(outDir, 'qwen2_runtime_hdl_block_skeleton_streaming_entry');

    info = struct();
    info.OutputDir = outDir;
end