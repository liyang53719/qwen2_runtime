function acc_out = linear_tile_step_serial(x_tile, w_tile, acc_in)
%LINEAR_TILE_STEP_SERIAL Fixed-point serial MAC tile.

    F = macFimath();
    [outTile, inTile] = size(w_tile);
    acc_out = fi(zeros(outTile, 1), true, 32, 14, F);

    for o = 1:outTile
        partial = fi(acc_in(o), true, 32, 14, F);
        for i = 1:inTile
            xVal = fi(x_tile(i), true, 16, 14, F);
            wVal = fi(w_tile(o, i), true, 16, 14, F);
            partial = partial + wVal * xVal;
        end
        acc_out(o) = partial;
    end
end

function F = macFimath()
    F = fimath( ...
        'RoundingMethod', 'Floor', ...
        'OverflowAction', 'Saturate', ...
        'ProductMode', 'SpecifyPrecision', ...
        'ProductWordLength', 24, ...
        'ProductFractionLength', 14, ...
        'SumMode', 'SpecifyPrecision', ...
        'SumWordLength', 32, ...
        'SumFractionLength', 14);
end
