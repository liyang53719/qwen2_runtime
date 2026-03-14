function info = generate_block_baseline()
%GENERATE_BLOCK_BASELINE Generate hierarchical block-level RTL baseline.

    projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    cfg = coder.config('hdl');
    cfg.TargetLanguage = 'Verilog';
    cfg.GenerateHDLTestBench = true;
    cfg.TestBenchName = 'qwen2_runtime_hdl_block_skeleton_entry_tb';
    cfg.GenerateCosimTestBench = false;
    cfg.SynthesizeGeneratedCode = false;
    cfg.Workflow = 'Generic ASIC/FPGA';
    cfg.InstantiateFunctions = false;

    outDir = fullfile(projectRoot, 'codegen', 'hdl_block_baseline', 'qwen2_runtime_hdl_block_skeleton_entry');
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
    args = qwen2_runtime.hdl.block_skeleton_step_args();
    codegen('-config', cfg, 'qwen2_runtime_hdl_block_skeleton_entry', '-args', args, '-d', outDir, '-I', projectRoot);
    qwen2_runtime.hdl.normalize_auto_tb_artifacts(outDir);

    vectorInfo = qwen2_runtime.hdl.write_block_baseline_vectors(fullfile(projectRoot, 'artifacts', 'block_baseline_vectors'));
    tbInfo = qwen2_runtime.hdl.write_block_baseline_tb(vectorInfo.OutputDir);

    info = struct();
    info.OutputDir = outDir;
    info.VectorDir = vectorInfo.OutputDir;
    info.TestbenchPath = tbInfo.TestbenchPath;
end
