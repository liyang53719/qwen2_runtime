function args = attention_value_row_controller_step_args()
%ATTENTION_VALUE_ROW_CONTROLLER_STEP_ARGS Representative args for value controller.

    vecLen = 16;
    start = false;
    weight_vec = fi(zeros(vecLen, 1), true, 16, 14);
    value_vec = fi(zeros(vecLen, 1), true, 16, 14);
    acc_seed = fi(0, true, 32, 14);
    args = {start, weight_vec, value_vec, acc_seed};
end
