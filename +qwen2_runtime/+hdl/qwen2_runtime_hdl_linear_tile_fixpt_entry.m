function acc_out = qwen2_runtime_hdl_linear_tile_fixpt_entry(x_tile, w_tile, acc_in)
%QWEN2_RUNTIME_HDL_LINEAR_TILE_FIXPT_ENTRY Fixed-point tile MAC wrapper.

    acc_out = qwen2_runtime.hdl.linear_tile_step(x_tile, w_tile, acc_in);
end
