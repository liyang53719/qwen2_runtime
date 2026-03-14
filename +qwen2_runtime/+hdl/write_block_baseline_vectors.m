function info = write_block_baseline_vectors(outputDir)
%WRITE_BLOCK_BASELINE_VECTORS Emit deterministic test vectors for RTL TB.

    if nargin < 1 || strlength(string(outputDir)) == 0
        outputDir = fullfile(pwd, 'artifacts', 'block_baseline_vectors');
    end
    outputDir = char(outputDir);
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    F16 = fimath('RoundingMethod', 'Floor', 'OverflowAction', 'Saturate', 'ProductMode', 'SpecifyPrecision', 'ProductWordLength', 24, 'ProductFractionLength', 14, 'SumMode', 'SpecifyPrecision', 'SumWordLength', 16, 'SumFractionLength', 14);
    F32 = fimath('RoundingMethod', 'Floor', 'OverflowAction', 'Saturate', 'ProductMode', 'SpecifyPrecision', 'ProductWordLength', 24, 'ProductFractionLength', 14, 'SumMode', 'SpecifyPrecision', 'SumWordLength', 32, 'SumFractionLength', 14);

    input_vec = fi(linspace(-0.25, 0.25, 8)', true, 32, 14, F32);
    score_mat = fi(reshape(linspace(-4, -1, 32), 16, 2), true, 16, 14, F16);
    value_tensor = fi(reshape(sin(1:128) / 4, 16, 4, 2), true, 16, 14, F16);
    residual_seed = fi(zeros(8, 1), true, 32, 14, F32);

    totalCycles = 8;
    clear qwen2_runtime.hdl.block_skeleton_step
    for cyc = 1:totalCycles
        [dut, valid] = qwen2_runtime.hdl.block_skeleton_step(cyc == 1, input_vec, score_mat, value_tensor, residual_seed);
    end

    expected = storedInteger(dut(:));

    writematrix(double(storedInteger(input_vec(:))), fullfile(outputDir, 'input_vec_int.txt'), 'Delimiter', 'space');
    writematrix(double(storedInteger(score_mat(:))), fullfile(outputDir, 'score_mat_int.txt'), 'Delimiter', 'space');
    writematrix(double(storedInteger(value_tensor(:))), fullfile(outputDir, 'value_tensor_int.txt'), 'Delimiter', 'space');
    writematrix(double(storedInteger(residual_seed(:))), fullfile(outputDir, 'residual_seed_int.txt'), 'Delimiter', 'space');
    writematrix(double(expected(:)), fullfile(outputDir, 'expected_block_out_int.txt'), 'Delimiter', 'space');

    meta = struct();
    meta.totalCycles = totalCycles;
    meta.valid = logical(valid);
    meta.inputWordLength = 32;
    meta.inputFracLength = 14;
    meta.scoreWordLength = 16;
    meta.scoreFracLength = 14;
    meta.valueWordLength = 16;
    meta.valueFracLength = 14;
    meta.expectedWordLength = 32;
    meta.expectedFracLength = 14;
    save(fullfile(outputDir, 'block_baseline_vectors.mat'), 'meta');

    info = struct();
    info.OutputDir = outputDir;
    info.TotalCycles = totalCycles;
    info.Expected = expected;
end
