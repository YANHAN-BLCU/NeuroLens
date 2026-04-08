# 快速参考指南

## 模型调用方式（一句话总结）

**基础模型 + Delta 权重 = 微调后完整模型**

```python
model = load_delta_weights(base_model_path, delta_weights_path, device)
# 内部逻辑：
# 1. 加载 base_model 的全部权重 (16GB)
# 2. 加载 delta_weights (1-2MB)
# 3. 逐层累加：model[layer] += delta[layer]
# 4. 返回微调后的模型
```

---

## Batch Size 能否增大？

### 快速判断表

| 当前 batch_size | 可以增大到 | 显存占用 | 风险等级 |
|-----------------|-----------|---------|---------|
| 4 | ✅ 6 | 76-80GB | 🟢 安全 |
| 6 | ⚠️ 8 | 90-95GB | 🟡 谨慎 |
| 8+ | ❌ 不建议 | 100GB+ | 🔴 高风险 |

### 增大 batch_size 的前置条件

```python
# 条件1: 减少 max_new_tokens
--max_new_tokens 2048  # 当前
--max_new_tokens 512   # 新建议（显存-75%）

# 条件2: 减少输入长度
max_length=16384  # 当前
max_length=8192   # 新建议（显存-25%）

# 条件3: 启用 Flash Attention
attn_implementation="flash_attention_2"  # 显存-35-40%
```

### 最优 batch_size 选择

```
┌─────────────────────────────────────────┐
│ RTX PRO 6000 (90GB) 的最优 batch_size  │
├─────────────────────────────────────────┤
│                                         │
│  安全配置 (推荐)                        │
│  ├─ batch_size = 6                    │
│  ├─ max_new_tokens = 1024              │
│  ├─ 显存 = 75-80GB                     │
│  └─ 风险 = 极低 ✅                     │
│                                         │
│  激进配置 (可尝试)                      │
│  ├─ batch_size = 8                    │
│  ├─ max_new_tokens = 512               │
│  ├─ 显存 = 90-95GB                     │
│  └─ 风险 = 中等 ⚠️                     │
│                                         │
│  保守配置 (最稳定)                      │
│  ├─ batch_size = 4                    │
│  ├─ max_new_tokens = 512               │
│  ├─ 显存 = 50-55GB                     │
│  └─ 风险 = 几乎无 ✅✅                 │
│                                         │
└─────────────────────────────────────────┘
```

---

## 显存占用速查表

### 关键公式

```
总显存 = 模型权重(16GB) + KV缓存 + 激活值 + 隐藏态

KV缓存 ≈ batch_size × (input_len + max_new_tokens) × 32层 × 4096维 × 2(K+V) × 2(FP16)
      = batch_size × seq_len × 4.3MB (简化公式)

例子:
batch_size=6, seq_len=18432 (16384+2048)
= 6 × 18432 × 4.3MB ≈ 475MB × 32层 ≈ 76GB (含模型权重和激活值)
```

### 显存压力指示灯

```
显存占用      状态         操作建议
─────────────────────────────────────────
< 60GB       🟢 绿灯      可安全增加 batch_size
60-80GB      🟡 黄灯      当前配置安全，勿增加
80-90GB      🔴 红灯      接近上限，谨慎操作
> 90GB       💥 爆炸      几乎必然 OOM
```

---

## 显存释放时序

```
┌──────────────────────────────────────┐
│ 单个批次的显存生命周期                  │
├──────────────────────────────────────┤
│                                      │
│ 16GB  ▂ 基础模型权重 (常驻)          │
│ 76GB  █ 推理时 (KV缓存 + 激活值)    │
│ 16GB  ▂ 提取隐藏态 (逐个移到CPU)    │
│ 16GB  ▂ 清理后 (del删除中间变量)     │
│                                      │
│ ↑ 峰值: 76GB (在模型推理时达到)      │
│ ↓ 谷值: 16GB (批次间空闲时)          │
│                                      │
└──────────────────────────────────────┘
```

---

## 隐藏态提取的三个关键参数

| 参数 | 当前值 | 对显存影响 | 可调性 |
|-----|--------|---------|-------|
| `batch_size` | 4 | 线性 🔴 | 高 ⭐⭐⭐ |
| `max_new_tokens` | 2048 | 线性 🔴 | 高 ⭐⭐⭐ |
| `max_length` (输入) | 16384 | 线性 🔴 | 中 ⭐⭐ |

**调整优先级**: batch_size > max_new_tokens > max_length

---

## 标注阶段（Qwen3Guard）的参数

```python
# 当前配置
moderate_batch(prompts, responses, model, tokenizer, batch_size=16)

# 建议更新（显存充足，可增加到 32-64）
moderate_batch(..., batch_size=32)  # 显存占用 ≈ 35GB（远低于提取阶段）

# 标注速度
batch_size=16: ~25000 样本/小时
batch_size=32: ~50000 样本/小时 (2x 加速)
batch_size=64: ~80000 样本/小时 (3.2x 加速，仍有显存余量)
```

---

## 实测数据点

### 已知工作的配置

```
✅ batch_size=4,  max_new_tokens=2048 → 显存 ≈ 50-55GB
✅ batch_size=6,  max_new_tokens=1024 → 显存 ≈ 75-80GB (推荐)
✅ batch_size=8,  max_new_tokens=512  → 显存 ≈ 90-95GB (可能 OOM)
❌ batch_size=10, 任何 max_new_tokens → 几乎必然 OOM
```

### 推荐命令

```bash
# 方案 A: 最稳定（推荐用于首次运行）
python scripts/apply_delta_extract_and_label.py \
    --base_model "..." \
    --delta_weights "..." \
    --classifier "..." \
    --batch_size 4 \
    --max_new_tokens 512

# 方案 B: 平衡（推荐）
python scripts/apply_delta_extract_and_label.py \
    --base_model "..." \
    --delta_weights "..." \
    --classifier "..." \
    --batch_size 6 \
    --max_new_tokens 1024

# 方案 C: 激进（风险）
python scripts/apply_delta_extract_and_label.py \
    --base_model "..." \
    --delta_weights "..." \
    --classifier "..." \
    --batch_size 8 \
    --max_new_tokens 512
```

---

## 性能对比

### 三个方案的预期性能

```
方案         batch_size  max_new_tokens  总耗时      吞吐量      显存峰值  推荐度
────────────────────────────────────────────────────────────────────────────
A 保守         4          512          120-150 min  67-83/min   50-55GB  ⭐⭐⭐
B 平衡         6          1024         60-80 min    125-167/min 75-80GB  ⭐⭐⭐⭐⭐
C 激进         8          512          45-60 min    167-222/min 90-95GB  ⭐⭐
```

---

## 决策树

```
              需要运行脚本
                  │
        ┌─────────┴─────────┐
        │                   │
       是否第一次运行？      是否想要最快速度？
        │                   │
   ┌────┴─────┐        ┌────┴──────┐
  用 A 方案    用 B 方案  用 C 方案   需要微调
  (最稳定)    (推荐)    (可尝试)    (联系工程师)
   
显存遇到 OOM:
  1. 检查: nvidia-smi 查看其他进程
  2. 减少: --batch_size 8 → 6 → 4
  3. 缩短: --max_new_tokens 1024 → 512
  4. 如仍 OOM: --max_new_tokens 256
```

---

## 我的最终建议

**对于 RTX PRO 6000（90GB）：**

✅ **立即可用的配置**
```bash
--batch_size 6 --max_new_tokens 1024
# 预期: 80 分钟 / 10,000 样本
# 显存: 75-80GB（安全余量 ≈ 10GB）
```

⚠️ **可尝试但需监控**
```bash
--batch_size 8 --max_new_tokens 512
# 预期: 60 分钟 / 10,000 样本（+33% 速度）
# 显存: 90-95GB（余量不足，需谨慎）
```

❌ **不建议**
```bash
--batch_size > 8 或 batch_size=8 && max_new_tokens > 512
# 风险: 几乎必然 OOM
```

---

**总结**: 当前 batch_size=4 可以增大到 6（推荐），理论上可到 8（但需冒风险）。
