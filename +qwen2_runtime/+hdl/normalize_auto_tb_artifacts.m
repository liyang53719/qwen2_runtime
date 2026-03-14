function info = normalize_auto_tb_artifacts(outDir)
%NORMALIZE_AUTO_TB_ARTIFACTS Collapse duplicated HDL Coder TB samples for block baseline.

    info = qwen2_runtime.hdl.normalize_generated_auto_tb_artifacts(outDir, 'qwen2_runtime_hdl_block_skeleton_entry');
end