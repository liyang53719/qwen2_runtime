function report_readiness(matParamsFile)
%REPORT_READINESS Print HDL-oriented readiness information.

    parameters = qwen2_runtime.load(matParamsFile);
    plan = qwen2_runtime.prepareForHDL(parameters, qwen2_runtime.defaultHDLConfig());

    fprintf('HDL block kernel   : %s\n', plan.BlockKernel);
    fprintf('RTL layer reuse    : %s\n', string(~plan.UnrollLayersForRTL));
    fprintf('Num layers         : %d\n', plan.NumLayers);
    fprintf('Hidden size        : %d\n', plan.Hyperparameters.HiddenSize);
    fprintf('Head dim           : %d\n', plan.Hyperparameters.HeadDim);
    fprintf('Num heads          : %d\n', plan.Hyperparameters.NumHeads);
    fprintf('Num KV heads       : %d\n', plan.Hyperparameters.NumKVHeads);
end
