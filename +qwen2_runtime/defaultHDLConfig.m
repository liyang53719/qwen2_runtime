function cfg = defaultHDLConfig()
%DEFAULTHDLCONFIG HDL-oriented configuration with reusable block kernel.

    cfg = qwen2_runtime.defaultRuntimeConfig();
    cfg.LinearMode = 'float';
    cfg.MlpGateLinearMode = 'float';
    cfg.MlpUpLinearMode = 'float';
    cfg.MlpDownLinearMode = 'float';
    cfg.TailLayerStart = 24;
    cfg.ForceFloatLayers = 24:27;
    cfg.UnrollLayersForRTL = false;
    cfg.BlockKernel = 'qwen2_runtime.hdl.block_kernel';
    cfg.EnableHDLMathSafeMode = true;
    cfg.HDLExpNegLimit = single(-8.0);
    cfg.HDLSoftmaxNegInit = single(-1.0e4);
    cfg.HDLMinDenominator = single(1.0e-6);
    cfg.HDLInvSqrtIterations = 3;
end
