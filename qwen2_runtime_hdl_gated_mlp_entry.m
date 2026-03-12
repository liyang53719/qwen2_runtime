function Z = qwen2_runtime_hdl_gated_mlp_entry(X, weights, cfg)
%QWEN2_RUNTIME_HDL_GATED_MLP_ENTRY Thin top-level wrapper for HDL codegen.

    Z = qwen2_runtime.hdl.qwen2_runtime_hdl_gated_mlp_entry(X, weights, cfg);
end