function info = write_block_baseline_tb(outputDir)
%WRITE_BLOCK_BASELINE_TB Emit a runnable Verilog TB for block baseline review.

    if nargin < 1 || strlength(string(outputDir)) == 0
        outputDir = fullfile(pwd, 'artifacts', 'block_baseline_vectors');
    end
    outputDir = char(outputDir);
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    vecInfo = qwen2_runtime.hdl.write_block_baseline_vectors(outputDir);
    expected = double(vecInfo.Expected(:));
    totalCycles = vecInfo.TotalCycles;

    F16 = fimath('RoundingMethod', 'Floor', 'OverflowAction', 'Saturate', 'ProductMode', 'SpecifyPrecision', 'ProductWordLength', 24, 'ProductFractionLength', 14, 'SumMode', 'SpecifyPrecision', 'SumWordLength', 16, 'SumFractionLength', 14);
    F32 = fimath('RoundingMethod', 'Floor', 'OverflowAction', 'Saturate', 'ProductMode', 'SpecifyPrecision', 'ProductWordLength', 24, 'ProductFractionLength', 14, 'SumMode', 'SpecifyPrecision', 'SumWordLength', 32, 'SumFractionLength', 14);
    inputVec = double(storedInteger(fi(linspace(-0.25, 0.25, 8)', true, 32, 14, F32)));
    scoreMat = double(storedInteger(fi(reshape(linspace(-4, -1, 32), 16, 2), true, 16, 14, F16)));
    valueTensor = double(storedInteger(fi(reshape(sin(1:128) / 4, 16, 4, 2), true, 16, 14, F16)));
    residualSeed = double(storedInteger(fi(zeros(8, 1), true, 32, 14, F32)));

    tbPath = fullfile(outputDir, 'tb_qwen2_runtime_hdl_block_skeleton_entry.v');
    fid = fopen(tbPath, 'w');
    fprintf(fid, '`timescale 1ns/1ps\n');
    fprintf(fid, 'module tb_qwen2_runtime_hdl_block_skeleton_entry;\n');
    fprintf(fid, '  reg clk;\n  reg reset;\n  reg clk_enable;\n  reg start;\n');
    fprintf(fid, '  integer cycle_count;\n  reg saw_valid;\n');
    for i = 0:7
        fprintf(fid, '  reg signed [31:0] input_vec_%d;\n', i);
    end
    for i = 0:31
        fprintf(fid, '  reg signed [15:0] score_mat_%d;\n', i);
    end
    for i = 0:127
        fprintf(fid, '  reg signed [15:0] value_tensor_%d;\n', i);
    end
    for i = 0:7
        fprintf(fid, '  reg signed [31:0] residual_seed_%d;\n', i);
    end
    fprintf(fid, '  wire ce_out;\n');
    for i = 0:7
        fprintf(fid, '  wire signed [31:0] block_out_%d;\n', i);
    end
    fprintf(fid, '  wire out_valid;\n\n');

    fprintf(fid, '  qwen2_runtime_hdl_block_skeleton_entry dut (\n');
    fprintf(fid, '    .clk(clk), .reset(reset), .clk_enable(clk_enable), .start(start),\n');
    for i = 0:7
        fprintf(fid, '    .input_vec_%d(input_vec_%d),\n', i, i);
    end
    for i = 0:31
        fprintf(fid, '    .score_mat_%d(score_mat_%d),\n', i, i);
    end
    for i = 0:127
        fprintf(fid, '    .value_tensor_%d(value_tensor_%d),\n', i, i);
    end
    for i = 0:7
        fprintf(fid, '    .residual_seed_%d(residual_seed_%d),\n', i, i);
    end
    fprintf(fid, '    .ce_out(ce_out),\n');
    for i = 0:6
        fprintf(fid, '    .block_out_%d(block_out_%d),\n', i, i);
    end
    fprintf(fid, '    .block_out_7(block_out_7),\n');
    fprintf(fid, '    .out_valid(out_valid));\n\n');

    fprintf(fid, '  initial clk = 1''b0;\n');
    fprintf(fid, '  always #5 clk = ~clk;\n\n');

    fprintf(fid, '  initial begin\n');
    fprintf(fid, '    $dumpfile("tb_qwen2_runtime_hdl_block_skeleton_entry.vcd");\n');
    fprintf(fid, '    $dumpvars(0, tb_qwen2_runtime_hdl_block_skeleton_entry);\n');
    fprintf(fid, '  end\n\n');

    fprintf(fid, '  initial begin\n');
    fprintf(fid, '    reset = 1''b1; clk_enable = 1''b1; start = 1''b0;\n');
    for i = 0:7
        fprintf(fid, '    input_vec_%d = %s;\n', i, verilogSignedLiteral(32, inputVec(i+1)));
    end
    for i = 0:31
        fprintf(fid, '    score_mat_%d = %s;\n', i, verilogSignedLiteral(16, scoreMat(i+1)));
    end
    flatValue = valueTensor(:);
    for i = 0:127
        fprintf(fid, '    value_tensor_%d = %s;\n', i, verilogSignedLiteral(16, flatValue(i+1)));
    end
    for i = 0:7
        fprintf(fid, '    residual_seed_%d = %s;\n', i, verilogSignedLiteral(32, residualSeed(i+1)));
    end
    fprintf(fid, '    repeat (2) @(posedge clk);\n');
    fprintf(fid, '    reset = 1''b0;\n');
    fprintf(fid, '    @(posedge clk); start = 1''b1;\n');
    fprintf(fid, '    @(posedge clk); start = 1''b0;\n');
    fprintf(fid, '    cycle_count = 0; saw_valid = 1''b0;\n');
    fprintf(fid, '    while ((cycle_count < %d) && !saw_valid) begin\n', totalCycles + 20);
    fprintf(fid, '      @(posedge clk);\n');
    fprintf(fid, '      cycle_count = cycle_count + 1;\n');
    fprintf(fid, '      if (out_valid) begin\n');
    fprintf(fid, '        saw_valid = 1''b1;\n');
    fprintf(fid, '      end\n');
    fprintf(fid, '    end\n');
    fprintf(fid, '    if (!saw_valid) begin\n');
    fprintf(fid, '      $display("TB_FAIL: out_valid was not asserted");\n');
    fprintf(fid, '      $finish;\n');
    fprintf(fid, '    end\n');
    for i = 0:7
        fprintf(fid, '    if (block_out_%d !== %s) begin $display("TB_FAIL: block_out_%d mismatch got=%%0d exp=%d", block_out_%d); $finish; end\n', i, verilogSignedLiteral(32, expected(i+1)), i, expected(i+1), i);
    end
    fprintf(fid, '    $display("TB_PASS: block baseline outputs match MATLAB vectors");\n');
    fprintf(fid, '    $finish;\n');
    fprintf(fid, '  end\n');
    fprintf(fid, 'endmodule\n');
    fclose(fid);

    info = struct();
    info.TestbenchPath = tbPath;
end

function text = verilogSignedLiteral(width, value)
    value = double(value);
    if value < 0
        text = sprintf('-%d''sd%d', width, abs(round(value)));
    else
        text = sprintf('%d''sd%d', width, round(value));
    end
end
