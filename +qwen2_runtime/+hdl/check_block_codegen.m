function info = check_block_codegen(matParamsFile)
%CHECK_BLOCK_CODEGEN Run MATLAB codegen/HDL readiness checks for one block.

    projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    args = qwen2_runtime.hdl.block_entry_args(matParamsFile);

    info = struct();
    info.Screener = coder.screener('qwen2_runtime_hdl_block_entry');

    cfg = coder.config('hdl');
    cfg.TargetLanguage = 'Verilog';
    cfg.GenerateHDLTestBench = false;
    cfg.SynthesizeGeneratedCode = false;
    cfg.GenerateCosimTestBench = false;
    cfg.Workflow = 'Generic ASIC/FPGA';
    cfg.FloatingPointLibrary = 'NativeFloatingPoint';
    cfg.FloatingPointTargetConfiguration = hdlcoder.createFloatingPointTargetConfig('NativeFloatingPoint');
    cfg.InstantiateFunctions = true;
    cfg.ShareMultipliers = false;

    outDir = fullfile(projectRoot, 'codegen', 'hdl', 'qwen2_runtime_block_entry');
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    codegen('-config', cfg, 'qwen2_runtime_hdl_block_entry', '-args', args, '-d', outDir, '-I', projectRoot);
    info.OutputDir = outDir;
end
