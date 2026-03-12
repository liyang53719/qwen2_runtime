function Y = qwen2_runtime_hdl_rmsnorm_entry(X, weight, epsilon, cfg)
%QWEN2_RUNTIME_HDL_RMSNORM_ENTRY Package-level wrapper for RMSNorm HDL codegen.

    Y = qwen2_runtime.hdl.rmsnorm_step(X, weight, epsilon, cfg);
end