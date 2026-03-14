function acc_out = linear_tile_step_pair4(x_tile, w_tile, acc_in)
%LINEAR_TILE_STEP_PAIR4 Fixed-point tile with 4-lane accumulation.

    F = macFimath();
    [outTile, inTile] = size(w_tile);
    acc_out = fi(zeros(outTile, 1), true, 32, 14, F);
    laneCount = 4;

    for o = 1:outTile
        partial = fi(acc_in(o), true, 32, 14, F);
        for base = 1:laneCount:inTile
            laneAcc = fi(0, true, 32, 14, F);
            for lane = 0:laneCount-1
                idx = base + lane;
                if idx <= inTile
                    xVal = fi(x_tile(idx), true, 16, 14, F);
                    wVal = fi(w_tile(o, idx), true, 16, 14, F);
                    laneAcc = laneAcc + wVal * xVal;
                end
            end
            partial = partial + laneAcc;
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
