function cfg = defaultHardwareHDLConfig()
%DEFAULTHARDWAREHDLCONFIG Hardware-oriented HDL config starting from fixed-point kernels.

    cfg = qwen2_runtime.defaultHDLConfig();
    cfg.HDLNumericMode = 'fixed';
    cfg.UseFixedPointHDL = true;
    cfg.LinearMode = 'fixed';
    cfg.MlpGateLinearMode = 'fixed';
    cfg.MlpUpLinearMode = 'fixed';
    cfg.MlpDownLinearMode = 'fixed';
    cfg.BlockKernel = 'qwen2_runtime.hdl.block_entry';
    cfg.ForceFloatLayers = -1;
    cfg.UseExternalWeightMemory = true;
    cfg.UseExternalKVMemory = true;
    cfg.SystemAttentionKernel = 'qwen2_runtime.hdl.attention_token_step_sram_step';
    cfg.SystemKVInterfaceKernel = 'qwen2_runtime.hdl.attention_token_step_sram_contract_step';
    cfg.SystemAttentionControllerKernel = 'qwen2_runtime.hdl.attention_token_controller_sram_step';
    cfg.SystemAttentionHandshakeKernel = 'qwen2_runtime.hdl.attention_token_controller_sram_handshake_step';
    cfg.SystemBlockKernel = 'qwen2_runtime.hdl.block0_token_system_step';
end