function args = flatten_two_heads_step_args()
%FLATTEN_TWO_HEADS_STEP_ARGS Representative args for two-head flattener.

    head0 = fi(zeros(4, 1), true, 32, 14);
    head1 = fi(zeros(4, 1), true, 32, 14);
    args = {head0, head1};
end
