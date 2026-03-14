function args = attention_value_mac_step_args()
%ATTENTION_VALUE_MAC_STEP_ARGS Representative args for weighted value PE.

    start = false;
    weight_val = fi(0, true, 16, 14);
    value_val = fi(0, true, 16, 14);
    acc_seed = fi(0, true, 32, 14);
    row_last = false;
    args = {start, weight_val, value_val, acc_seed, row_last};
end
