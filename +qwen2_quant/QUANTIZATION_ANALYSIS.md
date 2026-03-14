# Qwen2 Quant 推理精度分析（用于 Verilog 对比）

## 1) 结论：当前实现是否“仅权重量化”

是。当前 `+qwen2_quant` 代码路径属于 **权重存储量化**，推理计算主体仍为浮点：

- 权重在算子内先 `dequantize()`，再参与矩阵乘法。
- 激活未做量化约束，注意力/MLP/Norm/Softmax 全是浮点流程。
- `Q4_0/Q8_0` 反量化函数输出 `single`。

可定位证据：

- `model` 中嵌入、norm、lm_head 都先反量化再算：
  - [model.m](model.m#L55)
  - [model.m](model.m#L106)
  - [model.m](model.m#L116)
- `quantized_matmul` 中反量化后直接 `single` 乘法：
  - [+layer/quantized_matmul.m](+layer/quantized_matmul.m#L19-L23)
- `q8_0/q4_0` 反量化输出 `single`：
  - [+internal/+dequant/q8_0.m](+internal/+dequant/q8_0.m#L59)
  - [+internal/+dequant/q4_0.m](+internal/+dequant/q4_0.m#L77)

## 2) 与 llama.cpp 常见实现对齐的精度建议

为便于与硬件整数 MAC 对比，推荐使用如下“算子级精度约束”：

| 算子 | 建议输入 | MAC/累加 | 输出 | 备注 |
|---|---|---|---|---|
| Linear (Q/K/V/O, MLP) | int8 激活 + int8 权重 | int32 累加 | single（或再量化） | 当前已实现 `int8_int32_sim` 模式 |
| RMSNorm | single | single | single | 常保留浮点，硬件可后续定点化 |
| RoPE | single | single | single | 复数旋转通常用浮点参考 |
| QK^T / AV | single | single | single | 若硬件定点注意力，需单独定标策略 |
| Softmax | single | single | single | 参考模型建议保留浮点 |
| Residual Add | single | single | single | 可选插入量化节点做截断仿真 |

## 3) 已落地的执行改造（第一版）

已在本仓库中实现：

1. **运行时量化配置**（默认不改行为）
   - 新增 [+internal/runtime_config.m](+internal/runtime_config.m)
   - 支持 `LinearMode = 'float' | 'int8_int32_sim'`

2. **算子精度追踪器**
   - 新增 [+internal/precision_trace.m](+internal/precision_trace.m)
   - 记录每个节点：`Class/IsInteger/Size/Min/Max/AbsMax`

3. **线性层整数 MAC 仿真路径**
   - 更新 [+layer/quantized_matmul.m](+layer/quantized_matmul.m)
   - `int8_int32_sim` 模式下：
     - 激活/权重对称量化到 int8
     - 使用整数域累加（以精确整数值形式计算）
     - 再按 scale 回到 single

4. **全链路追踪接入**
   - 更新 [model.m](model.m)
   - 更新 [+layer/block.m](+layer/block.m)
   - 更新 [+layer/attentionGQA.m](+layer/attentionGQA.m)
   - 更新 [+layer/gatedMLP.m](+layer/gatedMLP.m)

## 4) 使用方法（当前版本）

```matlab
opts.RuntimeConfig = struct(...
    'LinearMode', 'int8_int32_sim', ...
    'TracePrecision', true);

[logits, state, dbg] = qwen2_quant.model(inputIds, mdl.Parameters, [], opts);
trace = dbg.PrecisionTrace;
save('qwen2_quant_precision_trace.mat', 'trace');
```

## 5) 下一步建议（第二版）

为更贴近特定 Verilog 实现，建议继续做：

1. 将 `int8_int32_sim` 从 **per-tensor scale** 升级为 **per-channel/per-group scale**。
2. 对 Residual/RMSNorm 前后插入可配置截断（例如 int16 或 bf16 仿真）。
3. 增加中间张量导出白名单（只导出硬件关心节点，控制数据量）。
4. 输出与硬件对齐的十六进制定点格式（含 scale/zero-point 元数据）。
