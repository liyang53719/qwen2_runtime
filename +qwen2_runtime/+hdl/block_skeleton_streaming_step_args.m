function args = block_skeleton_streaming_step_args()
%BLOCK_SKELETON_STREAMING_STEP_ARGS Representative args for streaming block skeleton.

    F16 = fimath('RoundingMethod', 'Floor', 'OverflowAction', 'Saturate', ...
        'ProductMode', 'SpecifyPrecision', 'ProductWordLength', 24, 'ProductFractionLength', 14, ...
        'SumMode', 'SpecifyPrecision', 'SumWordLength', 16, 'SumFractionLength', 14);
    F32 = fimath('RoundingMethod', 'Floor', 'OverflowAction', 'Saturate', ...
        'ProductMode', 'SpecifyPrecision', 'ProductWordLength', 24, 'ProductFractionLength', 14, ...
        'SumMode', 'SpecifyPrecision', 'SumWordLength', 32, 'SumFractionLength', 14);

    start = false;
    input_vec = fi(zeros(8, 1), true, 32, 14, F32);
    score_token = fi(zeros(2, 1), true, 16, 14, F16);
    value_token = fi(zeros(8, 1), true, 16, 14, F16);
    token_valid = false;
    token_last = false;
    residual_seed = fi(zeros(8, 1), true, 32, 14, F32);
    args = {start, input_vec, score_token, value_token, token_valid, token_last, residual_seed};
end