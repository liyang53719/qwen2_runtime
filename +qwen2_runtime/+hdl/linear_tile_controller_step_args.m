function args = linear_tile_controller_step_args()
%LINEAR_TILE_CONTROLLER_STEP_ARGS Representative args for tile controller HDL codegen.

    inTile = 16;
    outTile = 8;
    start = false;
    x_tile = fi(zeros(inTile, 1), true, 16, 14);
    w_tile = fi(zeros(outTile, inTile), true, 16, 14);
    acc_seed = fi(zeros(outTile, 1), true, 32, 14);
    args = {start, x_tile, w_tile, acc_seed};
end
