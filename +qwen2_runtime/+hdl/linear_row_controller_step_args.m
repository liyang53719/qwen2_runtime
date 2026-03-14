function args = linear_row_controller_step_args()
%LINEAR_ROW_CONTROLLER_STEP_ARGS Representative args for row controller HDL codegen.

    vecLen = 16;
    start = false;
    x_vec = fi(zeros(vecLen, 1), true, 16, 14);
    w_row = fi(zeros(vecLen, 1), true, 16, 14);
    acc_seed = fi(0, true, 32, 14);
    args = {start, x_vec, w_row, acc_seed};
end
