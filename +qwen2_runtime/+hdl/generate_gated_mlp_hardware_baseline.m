function info = generate_gated_mlp_hardware_baseline()
%GENERATE_GATED_MLP_HARDWARE_BASELINE Generate a fixed-point gated MLP HDL baseline.

    projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    cfg = coder.config('hdl');
    cfg.TargetLanguage = 'Verilog';
    cfg.GenerateHDLTestBench = true;
    cfg.TestBenchName = 'qwen2_runtime_hdl_gated_mlp_entry_tb';
    cfg.GenerateCosimTestBench = false;
    cfg.SynthesizeGeneratedCode = false;
    cfg.Workflow = 'Generic ASIC/FPGA';
    cfg.InstantiateFunctions = false;

    outDir = fullfile(projectRoot, 'codegen', 'hdl_gated_mlp_hardware_baseline', 'qwen2_runtime_hdl_gated_mlp_entry');
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    args = qwen2_runtime.hdl.gated_mlp_entry_hardware_args();
    codegen('-config', cfg, 'qwen2_runtime_hdl_gated_mlp_entry', '-args', args, '-d', outDir, '-I', projectRoot);

    info = struct();
    info.OutputDir = outDir;
end