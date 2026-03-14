function out_vec = qwen2_runtime_hdl_residual_add_entry(residual_vec, update_vec)
%QWEN2_RUNTIME_HDL_RESIDUAL_ADD_ENTRY Wrapper for residual add.

    out_vec = qwen2_runtime.hdl.residual_add_step(residual_vec, update_vec);
end
