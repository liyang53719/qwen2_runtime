function Z = qwen2_runtime_hdl_gated_mlp_entry(X, weights, cfg)
%QWEN2_RUNTIME_HDL_GATED_MLP_ENTRY Package-level wrapper for gated MLP HDL codegen.

    Z = qwen2_runtime.hdl.gated_mlp_step(X, weights, cfg);
end