function cfg = defaultHardwareHDLConfig()
%DEFAULTHARDWAREHDLCONFIG Hardware-oriented HDL config starting from fixed-point kernels.

    cfg = qwen2_runtime.defaultHDLConfig();
    cfg.HDLNumericMode = 'fixed';
    cfg.LinearMode = 'fixed';
    cfg.MlpGateLinearMode = 'fixed';
    cfg.MlpUpLinearMode = 'fixed';
    cfg.MlpDownLinearMode = 'fixed';
    cfg.ForceFloatLayers = -1;
end