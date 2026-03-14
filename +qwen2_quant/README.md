# Qwen2 Quantization Support - Quick Start Guide

## 📁 File Structure

The quantization support is implemented in the `+qwen2_quant/` package:

```
+qwen2_quant/
├── load_gguf.m                     # Main GGUF loader
├── model.m                         # Quantization-aware forward pass
├── generate.m                      # Text generation
├── +internal/
│   ├── quantized_weight.m          # Quantized weight container class
│   ├── gguf_reader.m               # GGUF binary file parser
│   └── +dequant/
│       ├── q8_0.m                  # Q8_0 dequantization
│       ├── q4_0.m                  # Q4_0 dequantization
│       └── mxfp4.m                 # MXFP4 dequantization
└── +layer/
    ├── quantized_matmul.m          # Quantization-aware matrix multiply
    ├── block.m                     # Transformer block
    ├── attentionGQA.m              # Grouped Query Attention
    └── gatedMLP.m                  # Gated MLP

Test Files:
- TestQwen2QuantSummarize.m         # Main test script
- generateSummary_Qwen2_quant.m     # Quantized generation function
```

## 🚀 Quick Start

### Step 1: Download GGUF Models

```bash
# Install huggingface-cli
pip install huggingface_hub

# Optional: use mirror endpoint
export HF_ENDPOINT=https://hf-mirror.com

# Download Q8_0 model (best quality, ~1.9GB)
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  qwen2.5-1.5b-instruct-q8_0.gguf \
  --local-dir qwen_gguf

# Download Q4_0 model (good balance, ~1GB)
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  qwen2.5-1.5b-instruct-q4_0.gguf \
  --local-dir qwen_gguf

# Download Q4_K_M model (recommended 4-bit quality)
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --local-dir qwen_gguf
```

### Step 2: Run Test Script

In MATLAB:

```matlab
% Run the test script
TestQwen2QuantSummarize

% Or use directly
mdl.Parameters = qwen2_quant.load_gguf('qwen_gguf/qwen2.5-1.5b-instruct-q8_0.gguf');
mdl.Tokenizer = qwen2.tokenizer.QwenTokenizer('qwen_model');

summary = generateSummary_Qwen2_quant(mdl, 'Your text here', ...
    'MaxNewTokens', 50, 'TopK', 1);
```

## 📊 Supported Quantization Formats

| Format | Bit Width | File Size | Quality | Use Case |
|--------|-----------|-----------|---------|----------|
| **Q8_0** | 8-bit | ~1.9 GB | Excellent | Best quality, slight compression |
| **Q4_0** | 4-bit | ~1 GB | Good | Best balance of quality/size |
| **Q4_K_M** | 4-bit (K-quant) | ~1 GB | Better than Q4_0 | Recommended 4-bit GGUF |
| **MXFP4** | 4-bit | ~1 GB | Experimental | Research/special cases |

## 🎯 Key Features

1. **Memory Efficient**: Keeps weights quantized until needed
2. **Runtime Dequantization**: Dequantizes on-the-fly during inference
3. **Compatible**: Uses same tokenizer as regular `+qwen2` package
4. **Modular**: Clean separation from original `+qwen2` code

## 🧪 Precision Trace & Int8 Simulation

You can run linear layers in an integer-MAC simulation mode and collect
per-operator precision stats for hardware comparison.

```matlab
opts.RuntimeConfig = struct(...
  'LinearMode', 'int8_int32_sim', ...
  'TracePrecision', true);

[logits, state, dbg] = qwen2_quant.model(inputIds, mdl.Parameters, [], opts);
trace = dbg.PrecisionTrace;
save('qwen2_quant_precision_trace.mat', 'trace');
```

`LinearMode` options:
- `float` (default): current float matmul path
- `int8_int32_sim`: int8 weight/activation quantization + int32-style accumulation simulation

See `QUANTIZATION_ANALYSIS.md` for detailed operator-level precision notes.

## 🔧 API Usage

### GPTQ / AWQ Branches (Parallel to GGUF)

```matlab
% 1) Download once (uses HF endpoint mirror by default)
gptqPath = qwen2_quant.download("gptq");
awqPath  = qwen2_quant.download("awq");

% 2) Build parameter structs for quant branches
mdl = struct();
mdl.Parameters = qwen2_quant.load_hf_quant("Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4", ...
  'LocalFilesOnly', true, 'HFEndpoint', "https://hf-mirror.com");

% 3) Inference through the same summary API
summary = generateSummary_Qwen2_quant(mdl, "Your text here", ...
  'PromptTemplate', "Summarize this: %s Summary:", ...
  'MaxNewTokens', 50, 'TopK', 1);
```

Notes:
- GPTQ/AWQ branch runs through Python Transformers backend from inside `+qwen2_quant`.
- AWQ branch generally requires GPU runtime.

### Load GGUF Model

```matlab
% Keep quantized (saves memory)
params = qwen2_quant.load_gguf('model.gguf', 'DequantizeNow', false);

% Or dequantize immediately (faster inference)
params = qwen2_quant.load_gguf('model.gguf', 'DequantizeNow', true);
```

### Generate Text

```matlab
mdl.Parameters = qwen2_quant.load_gguf('model.gguf');
mdl.Tokenizer = qwen2.tokenizer.QwenTokenizer('qwen_model');

% Greedy decoding
summary = generateSummary_Qwen2_quant(mdl, "Summarize this text", ...
    'MaxNewTokens', 50, 'TopK', 1);

% With prompt template
summary = generateSummary_Qwen2_quant(mdl, "Raw text here", ...
    'PromptTemplate', "Summarize: %s", ...
    'MaxNewTokens', 30);
```

## 🔍 Implementation Details

### Quantization Formats

**Q8_0 (8-bit)**:
- Block size: 32 elements
- Format: 1× float16 delta + 32× int8 values
- Formula: `weight[i] = delta × qvalue[i]`

**Q4_0 (4-bit)**:
- Block size: 32 elements  
- Format: 1× float16 delta + 16 bytes (packed)
- Formula: `weight[i] = delta × (qvalue[i] - 8)`

**MXFP4 (Microsoft Microscaling)**:
- Block size: 32 elements
- Format: E2M1 (1 sign, 2 exponent, 1 mantissa)
- Shared exponent per block

### Runtime vs Load-Time Dequantization

**Runtime Dequantization** (default):
- ✅ Low memory (~1-2GB)
- ❌ Slower inference (~10-20% overhead)
- Best for memory-constrained systems

**Load-Time Dequantization** (`DequantizeNow=true`):
- ❌ High memory (~6GB)
- ✅ Faster inference (no dequant overhead)
- Best for speed-critical applications

## 🧪 Testing

The test script `TestQwen2QuantSummarize.m` performs:

1. ✅ Load GGUF model
2. ✅ Generate text summary
3. ✅ Compare with full precision (if available)
4. ✅ Benchmark multiple formats
5. ✅ Report memory usage

## 📈 Expected Performance

On typical hardware:
- **Load Time**: Q8_0 ~8s, Q4_0 ~5s
- **Inference**: ~0.1-0.5s per token
- **Memory**: Q8_0 ~1.9GB, Q4_0 ~1.1GB
- **Quality**: Q8_0 ≈ FP32, Q4_0 ~99% of FP32

## 🐛 Troubleshooting

**Error: "GGUF file not found"**
- Download the GGUF file first (see Step 1)

**Error: "Tokenizer not found"**
- Ensure `qwen_model/` folder exists with tokenizer files
- Run `python tools/prepare_qwen.py` to download

**Memory Error**
- Use smaller quantization (Q4_0 instead of Q8_0)
- Keep `DequantizeNow=false`

**Slow Inference**
- Try `DequantizeNow=true` (uses more memory)
- Use Q8_0 for better speed/quality balance

## 📚 References

- GGUF Specification: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- Qwen2 Model: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- Implementation Plan: GGUF_IMPLEMENTATION_PLAN.md
