function h_out = qwen2_runtime_hdl_block_mlp_tail_entry(h_in, norm_weight, mlp_weights, cfg)
%QWEN2_RUNTIME_HDL_BLOCK_MLP_TAIL_ENTRY Thin top-level wrapper for HDL codegen.

    h_out = qwen2_runtime.hdl.qwen2_runtime_hdl_block_mlp_tail_entry(h_in, norm_weight, mlp_weights, cfg);
end