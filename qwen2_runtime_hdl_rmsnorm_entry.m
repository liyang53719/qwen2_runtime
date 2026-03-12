function Y = qwen2_runtime_hdl_rmsnorm_entry(X, weight, epsilon, cfg)
%QWEN2_RUNTIME_HDL_RMSNORM_ENTRY Thin top-level wrapper for HDL codegen.

    Y = qwen2_runtime.hdl.qwen2_runtime_hdl_rmsnorm_entry(X, weight, epsilon, cfg);
end