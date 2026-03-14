function acc_out = qwen2_runtime_hdl_linear_tile_serial_entry(x_tile, w_tile, acc_in)
%QWEN2_RUNTIME_HDL_LINEAR_TILE_SERIAL_ENTRY Wrapper for serial tile MAC.

    acc_out = qwen2_runtime.hdl.linear_tile_step_serial(x_tile, w_tile, acc_in);
end
