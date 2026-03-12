function h_out = qwen2_runtime_hdl_block_mlp_tail_entry(h_in, norm_weight, mlp_weights, cfg)
%QWEN2_RUNTIME_HDL_BLOCK_MLP_TAIL_ENTRY Package-level wrapper for block tail HDL codegen.

    h_out = qwen2_runtime.hdl.block_mlp_tail_step(h_in, norm_weight, mlp_weights, cfg);
end