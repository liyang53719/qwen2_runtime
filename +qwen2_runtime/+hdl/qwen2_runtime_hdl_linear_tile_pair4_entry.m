function acc_out = qwen2_runtime_hdl_linear_tile_pair4_entry(x_tile, w_tile, acc_in)
%QWEN2_RUNTIME_HDL_LINEAR_TILE_PAIR4_ENTRY Wrapper for 4-lane tile MAC.

    acc_out = qwen2_runtime.hdl.linear_tile_step_pair4(x_tile, w_tile, acc_in);
end
