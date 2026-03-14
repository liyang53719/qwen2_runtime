function args = residual_add_step_args()
%RESIDUAL_ADD_STEP_ARGS Representative args for residual add.

    len = 8;
    residual_vec = fi(zeros(len, 1), true, 32, 14);
    update_vec = fi(zeros(len, 1), true, 32, 14);
    args = {residual_vec, update_vec};
end
