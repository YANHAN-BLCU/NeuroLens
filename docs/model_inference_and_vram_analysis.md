# 微调后模型调用和显存分析

## 1. 微调后模型的调用方式

### 1.1 模型加载流程

```python
# 步骤 1: 应用 Delta 权重到基础模型
model = load_delta_weights(
    base_model_path="/root/autodl-tmp/neurolens/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct",
    delta_weights_path="outputs/tsft_finetuning/model/delta_weights.pt",
    device="cuda:0"
)
# 返回：微调后的完整模型（原始模型 + delta 权重）
```

### 1.2 Delta 权重应用原理

```
原始模型权重 = Base Model State Dict
Delta 权重 = 只包含修改的层参数

应用过程：
for layer_name in delta_state.keys():
    original_state[layer_name] = original_state[layer_name] + delta[layer_name]

结果：得到微调后完整模型（不需要保存完整模型文件）
```

**核心优势**：
- 原始模型: ~16GB（LLaMA-3-8B）
- Delta 权重: ~1-2MB（只有修改的层）
- 节省存储空间 99.9%

### 1.3 推理调用（隐藏态提取）

```python
# 批量推理 + 提取隐藏态
with torch.inference_mode():
    gen_outputs = model.generate(
        input_ids=inputs["input_ids"],              # Shape: (batch_size, seq_len)
        attention_mask=inputs["attention_mask"],    # Shape: (batch_size, seq_len)
        max_new_tokens=2048,                        # 最大生成长度
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.05,
        output_hidden_states=True,                  # 关键：启用隐藏态输出
        return_dict_in_generate=True,
        use_cache=True,                             # KV 缓存加速
    )

# 输出：
# - gen_outputs.sequences: 生成的 token 序列
# - gen_outputs.hidden_states: 所有层的隐藏态（用于后续分析）
```

---

## 2. 显存占用分析（RTX PRO 6000 - 90GB）

### 2.1 各阶段显存占用

#### **阶段 A: 模型加载**
```
LLaMA-3-8B 模型权重:
- 参数量: 8B (80亿参数)
- FP16/BF16 精度: 16GB
- 优化器状态(训练时): 不适用（推理模式）
- KV 缓存(推理): 动态，取决于 batch_size 和 seq_len

基础显存占用: ~16-18GB
```

#### **阶段 B: 批量推理（batch_size=N, max_new_tokens=2048）**

**主要显存消费者**：
1. **模型权重**: 16GB（固定）
2. **KV 缓存**: batch_size × max_seq_len × num_layers × hidden_dim × 2
3. **激活值**: 推理过程中的中间激活
4. **隐藏态**：输出时保存所有层的隐藏态

**具体计算**：
```
对于 LLaMA-3-8B：
- 层数: 32 层
- 隐藏维度: 4096
- 每个 token 的 KV 缓存: 4096 × 2 × 2 bytes/fp16 = 32KB (单层)

单样本 KV 缓存（最坏情况，max_seq_len=16384+2048）：
= batch_size × (16384+2048) × 32 × 4096 × 2 × 2 / 1024^3 GB

示例计算（batch_size=8）：
= 8 × 18432 × 32 × 4096 × 2 × 2 / 1024^3
= ~97GB （仅 KV 缓存，不包括激活值）
```

### 2.2 不同 batch_size 的显存需求

| Batch Size | KV Cache (估算) | 激活值 + 其他 | 总计 | 可行性 |
|-----------|-----------------|-------------|------|--------|
| 1 | ~12GB | 2GB | **14-16GB** | ✅ 充足 |
| 2 | ~24GB | 2GB | **26-28GB** | ✅ 充足 |
| 4 | ~48GB | 3GB | **51-54GB** | ✅ 充足 |
| **6** | ~72GB | 4GB | **76-80GB** | ✅ 充足（推荐） |
| 8 | ~96GB | 5GB | **101-106GB** | ❌ 超额 |
| 16 | ~192GB | 8GB | **200+GB** | ❌ 严重超额 |

### 2.3 显存优化策略

#### **策略 1: 减少 max_new_tokens**
```python
# 当前设置: 2048
# 建议: 256-512（在推理效果满足的前提下）

显存节省: 2048/512 = 4 倍减少
预期效果: batch_size 可从 6 增加到 8
```

#### **策略 2: 更小的输入长度**
```python
# 当前最大: 16384
# 建议: 8192（多数安全数据集样本在 3000-5000 内）

显存节省: 约 50%
预期效果: batch_size 可从 6 增加到 12
```

#### **策略 3: 使用 Flash Attention**
```python
# 需要修改模型加载
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    attn_implementation="flash_attention_2",  # 需要安装 flash_attn
    torch_dtype=torch.bfloat16,
)

显存节省: 30-50%（注意力机制优化）
```

#### **策略 4: 梯度检查点（推理时不适用）**
```
# 仅用于训练，推理不需要
```

---

## 3. 推荐配置（RTX PRO 6000）

### 方案 A: 保守方案（最稳定）
```bash
python scripts/apply_delta_extract_and_label.py \
    --base_model "..." \
    --delta_weights "..." \
    --classifier "..." \
    --batch_size 4 \
    --max_new_tokens 512
    # 预期显存占用: ~50-55GB
    # 推荐度: ⭐⭐⭐⭐⭐ 最稳定
```

### 方案 B: 平衡方案（推荐）
```bash
python scripts/apply_delta_extract_and_label.py \
    --base_model "..." \
    --delta_weights "..." \
    --classifier "..." \
    --batch_size 6 \
    --max_new_tokens 1024
    # 预期显存占用: ~75-80GB
    # 性能提升: batch_size +50% vs 方案A
    # 推荐度: ⭐⭐⭐⭐⭐ 最优平衡
```

### 方案 C: 激进方案（风险）
```bash
python scripts/apply_delta_extract_and_label.py \
    --base_model "..." \
    --delta_weights "..." \
    --classifier "..." \
    --batch_size 8 \
    --max_new_tokens 512
    # 预期显存占用: ~90-95GB
    # 风险: 接近 90GB 上限，可能 OOM
    # 推荐度: ⭐⭐ 仅在充足监控下使用
```

---

## 4. 隐藏态提取的具体过程

### 4.1 单个批次的流程

```python
# 输入
batch_size = 6
seq_len = 512  # 平均输入长度

# 1. Tokenize 和 padding
inputs = tokenizer(
    batch_texts,
    return_tensors="pt",
    padding=True,
    truncation=False,
)
# inputs["input_ids"]: (6, 512)
# 显存: ~6KB（相对于模型权重可忽略）

# 2. 模型推理 (最耗显存)
with torch.inference_mode():
    gen_outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        output_hidden_states=True,
        use_cache=True,
    )
# 序列长度: 512 + 1024 = 1536
# 显存峰值: ~76GB （包含所有中间激活和隐藏态）

# 3. 隐藏态提取和 CPU 转移
hidden_states = gen_outputs.hidden_states  # Tuple of (32) layers
# 每层: (batch=6, seq=1536, hidden=4096) → FP16 = 192MB/层
# 总计: 32 × 192MB = 6GB（在 GPU 上）

# 4. 转换到 CPU/NumPy
for layer_idx in range(num_layers):
    hs = hidden_states[layer_idx][b, pos, :].float().cpu().numpy()
    # 逐个移到 CPU，释放 GPU 显存
# 显存恢复: 76GB → 16GB

# 5. 删除中间变量
del gen_outputs, hidden_states
# 显存: 16GB（仅模型权重）
```

### 4.2 隐藏态输出格式

```python
# 返回格式
hidden_states: np.ndarray  # Shape: (batch_size, num_layers, hidden_dim)
                           # = (6, 32, 4096)
                           # = 3.1MB / batch

outputs: List[str]  # 生成的文本
indices: List[int]  # 原始数据集索引
```

---

## 5. 标注阶段（Qwen3Guard）的显存占用

### 5.1 模型加载
```python
# Guard 模型加载（新增）
guard_model = AutoModelForCausalLM.from_pretrained(
    classifier_path,  # Qwen3Guard-Gen-8B
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
# 显存: ~16GB（与 LLaMA 相同）
# 总显存占用: 16GB (LLaMA) + 16GB (Guard) = 32GB
```

### 5.2 批量标注（优化后）
```python
# 批量处理 16 条样本
batch_prompts = prompts[0:16]      # 字符串
batch_responses = responses[0:16]   # 字符串

# Tokenize 后：
texts = [tokenizer.apply_chat_template(msg, tokenize=False) for msg in msgs]
inputs = tokenizer(texts, padding=True, ...)
# 输入形状: (16, ~300-500)  # prompt+response 通常很短

# Guard 推理
with torch.no_grad():
    generated_ids = guard_model.generate(
        **inputs,
        max_new_tokens=128,  # Guard 输出很短（只有 "Safe/Unsafe/Controversial"）
    )
# 显存占用: ~30-35GB（远低于提取阶段的 76GB）

# 标签提取
labels = [extract_label(decoded_text) for decoded_text in outputs]
# 输出：16 个标签 (Safe/Unsafe/Controversial)
```

**关键优势**：
- Guard 模型较小且输出短
- 无需保存隐藏态
- 显存占用低，批量大小可更大

### 5.3 推荐的标注 batch_size

```python
def run_labeling(...):
    # 当前代码中的批量大小
    labels = moderate_batch(
        prompts,
        responses,
        guard_model,
        guard_tokenizer,
        batch_size=16  # 可增加至 32（显存充足）
    )
```

**建议更新**：
```python
# 对于 90GB 显存，可安全使用
batch_size=32  # 显存占用: ~35GB
```

---

## 6. 完整推理流程的显存时间表

### 执行流程（假设处理 10,000 样本）

```
时间点          操作                          显存占用     耗时
────────────────────────────────────────────────────────────────
T=0             启动脚本                      1GB         -
T=1             加载 LLaMA-3-8B               16GB        3s
T=2             加载 Delta 权重 + 应用         16GB        5s
T=3-5           数据加载和预处理              18GB        10s
───────────────────────────────────────────────────────────────
T=5             ** 开始隐藏态提取 **          16GB
T=5-60          批处理推理 (batch_size=6)     76GB        55min
                - 1667 批 × 2s/batch
T=60            隐藏态提取完成                16GB
───────────────────────────────────────────────────────────────
T=60            加载 Qwen3Guard               32GB        3s
T=61-80         批量标注 (batch_size=32)      35GB        20min
                - 312 批 × 4s/batch
T=80            标注完成                      32GB
───────────────────────────────────────────────────────────────
T=80-81         生成报告                      18GB        1min
T=81            ** 总耗时: 81 分钟 **         --          --
```

**峰值显存**: 76GB （在提取隐藏态时达到）

---

## 7. 进一步优化建议

### 7.1 短期优化（实施难度：低）

| 优化项 | 方法 | 显存节省 | 耗时收益 |
|--------|------|---------|---------|
| 减少生成长度 | `max_new_tokens: 2048→512` | 20% | 30% 加速 |
| 减少输入长度 | `max_length: 16384→8192` | 25% | 20% 加速 |
| 增加标注 batch | `batch_size: 16→32` | 0% | 50% 加速 |
| **推荐 batch_size** | **6（当前）→ 8** | -10% | 33% 加速 |

### 7.2 中期优化（实施难度：中）

```python
# Flash Attention（需要额外依赖）
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    attn_implementation="flash_attention_2",  # pip install flash-attn
    torch_dtype=torch.bfloat16,
)
# 显存节省: 35-40%
# batch_size 可从 6 增加到 10
```

### 7.3 长期优化（实施难度：高）

```python
# 1. 模型量化（INT8/INT4）
# 2. 多 GPU 分布式推理
# 3. 分层卸载（layer offloading）
```

---

## 8. 故障排除

### 问题：OOM（Out of Memory）

```
症状: RuntimeError: CUDA out of memory
原因: batch_size 过大或 max_new_tokens 过长

解决方案:
1. 减少 batch_size: 8 → 6 → 4
2. 减少 max_new_tokens: 2048 → 1024 → 512
3. 减少输入长度: 16384 → 8192 → 4096
4. 检查是否有其他 GPU 进程占用显存
```

### 问题：推理速度慢

```
症状: 单个批次耗时 > 10 秒
原因: batch_size 过小或序列过长

解决方案:
1. 增加 batch_size（在显存允许范围内）
2. 启用 use_cache=True（已默认启用）
3. 检查 GPU 频率是否被限制
4. 使用 --skip_labeling 跳过标注，测试提取速度
```

---

## 总结

### 核心要点
1. **模型调用**: Delta 权重 + 基础模型 → 微调后完整模型
2. **显存瓶颈**: KV 缓存（取决于 batch_size × seq_len）
3. **推荐配置**: batch_size=6, max_new_tokens=1024（75-80GB 显存）
4. **可能优化**: batch_size 可增至 8（但接近 90GB 上限）

### 性能预期（RTX PRO 6000 + 优化配置）
- **总耗时**: 60-80 分钟 / 10,000 样本
- **吞吐量**: 125-167 样本/分钟
- **显存峰值**: 76-80GB

### 是否可增加 batch_size？
✅ **可以，但有限制**：
- 安全范围: batch_size ≤ 6
- 冒险范围: batch_size = 8（接近上限，需谨慎）
- 过大: batch_size ≥ 10（几乎必然 OOM）
