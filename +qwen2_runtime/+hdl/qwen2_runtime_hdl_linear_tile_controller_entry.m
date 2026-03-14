function [acc_out, tile_done] = qwen2_runtime_hdl_linear_tile_controller_entry(start, x_tile, w_tile, acc_seed)
%QWEN2_RUNTIME_HDL_LINEAR_TILE_CONTROLLER_ENTRY Wrapper for tile controller.

    [acc_out, tile_done] = qwen2_runtime.hdl.linear_tile_controller_step(start, x_tile, w_tile, acc_seed);
end
