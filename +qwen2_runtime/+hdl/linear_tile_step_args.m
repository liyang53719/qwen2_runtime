function args = linear_tile_step_args()
%LINEAR_TILE_STEP_ARGS Representative args for tile MAC HDL codegen.

    inTile = 16;
    outTile = 8;
    x_tile = fi(zeros(inTile, 1), true, 16, 14);
    w_tile = fi(zeros(outTile, inTile), true, 16, 14);
    acc_in = fi(zeros(outTile, 1), true, 32, 14);
    args = {x_tile, w_tile, acc_in};
end
