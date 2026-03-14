function [acc_out, tile_done] = linear_tile_controller_step(start, x_tile, w_tile, acc_seed)
%LINEAR_TILE_CONTROLLER_STEP Reuse one MAC engine across all rows of a tile.

    F = macFimath();
    persistent row_idx col_idx running acc_buf
    if isempty(row_idx)
        row_idx = uint8(1);
        col_idx = uint8(1);
        running = false;
        acc_buf = fi(zeros(8, 1), true, 32, 14, F);
    end

    outTile = uint8(size(w_tile, 1));
    inTile = uint8(size(w_tile, 2));
    acc_out = acc_buf;
    tile_done = false;

    if start
        row_idx = uint8(1);
        col_idx = uint8(1);
        running = true;
        acc_buf = fi(acc_seed, true, 32, 14, F);
    end

    if running
        partial = fi(acc_buf(row_idx), true, 32, 14, F);
        xVal = fi(x_tile(col_idx), true, 16, 14, F);
        wVal = fi(w_tile(row_idx, col_idx), true, 16, 14, F);
        product = fi(wVal * xVal, true, 32, 14, F);
        partial = partial + product;
        acc_buf(row_idx) = partial;
        acc_out = acc_buf;

        if col_idx == inTile
            col_idx = uint8(1);
            if row_idx == outTile
                tile_done = true;
                running = false;
            else
                row_idx = row_idx + uint8(1);
            end
        else
            col_idx = col_idx + uint8(1);
        end
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
