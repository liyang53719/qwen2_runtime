function args = linear_row_mac_step_args()
%LINEAR_ROW_MAC_STEP_ARGS Representative args for sequential MAC HDL codegen.

    start = false;
    x_val = fi(0, true, 16, 14);
    w_val = fi(0, true, 16, 14);
    acc_seed = fi(0, true, 32, 14);
    row_last = false;
    args = {start, x_val, w_val, acc_seed, row_last};
end
