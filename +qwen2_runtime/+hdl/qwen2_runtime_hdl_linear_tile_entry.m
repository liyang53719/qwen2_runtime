function acc_out = qwen2_runtime_hdl_linear_tile_entry(x_tile, w_tile, acc_in)
%QWEN2_RUNTIME_HDL_LINEAR_TILE_ENTRY Wrapper for tile MAC HDL codegen.

    acc_out = qwen2_runtime.hdl.linear_tile_step(x_tile, w_tile, acc_in);
end
