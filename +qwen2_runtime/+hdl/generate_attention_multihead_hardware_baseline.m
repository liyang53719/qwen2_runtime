function info = generate_attention_multihead_hardware_baseline()
%GENERATE_ATTENTION_MULTIHEAD_HARDWARE_BASELINE Generate a fixed-point attention-core controller baseline.

    projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    cfg = coder.config('hdl');
    cfg.TargetLanguage = 'Verilog';
    cfg.GenerateHDLTestBench = true;
    cfg.TestBenchName = 'qwen2_runtime_hdl_attention_multihead_controller_entry_tb';
    cfg.GenerateCosimTestBench = false;
    cfg.SynthesizeGeneratedCode = false;
    cfg.Workflow = 'Generic ASIC/FPGA';
    cfg.InstantiateFunctions = false;

    outDir = fullfile(projectRoot, 'codegen', 'hdl_attention_multihead_hardware_baseline', 'qwen2_runtime_hdl_attention_multihead_controller_entry');
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    args = qwen2_runtime.hdl.attention_multihead_controller_step_args();
    codegen('-config', cfg, 'qwen2_runtime_hdl_attention_multihead_controller_entry', '-args', args, '-d', outDir, '-I', projectRoot);

    info = struct();
    info.OutputDir = outDir;
end