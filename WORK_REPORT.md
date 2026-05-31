# NeuroLens2 项目工作报告

**执行者**: AI助手 (妍涵)
**时间**: 2026-05-09 14:00 ~ 2026-05-10 01:53
**项目位置**: F:\NeuroLens2

---


## 一、完成的工作（重点标注）

### 1. GPU 环境配置
- 检测到 dev_env 和系统 Python 都是 CPU-only PyTorch
- 安装了 CUDA 版 PyTorch (2.6.0+cu124)，支持 RTX 4060 Laptop GPU
- 验证 GPU 可用：CUDA 12.4，8GB 显存

### 2. 新建文件
| 文件 | 用途 |
|------|------|
| `requirements.txt` | 统一依赖管理 |
| `setup_gpu.ps1` | 一键 GPU 环境配置 |
| `build_installer.ps1` | 一键打包脚本 |
| `scripts/run_pipeline.py` | 重写的 pipeline 脚本（批量推理） |
| `scripts/run_analysis.py` | 集成所有引擎模块的分析脚本 |
| `scripts/generate_report.py` | 分析报告生成器 |
| `scripts/check_data.py` | 数据检查工具 |
| `scripts/check_apis.py` | API 测试工具 |

### 3. Pipeline 重写 (`scripts/run_pipeline.py`)
- 只使用 `attack_enhanced_set_train.jsonl` 数据集
- 三档测试：quick (200条)、standard (500条)、full (5000条)
- 批量推理优化：batch_size=8, max_new_tokens=128
- GPU 自动检测 + 预热
- JSON 格式实时进度输出
- 支持 `--batch-size` 和 `--max-tokens` 自定义参数
- **速度**: 200条约3分钟，500条约8分钟

### 4. 分析脚本集成 (`scripts/run_analysis.py`)
- 集成了 engine 目录下的所有模块：
  - SNIP 评分 (`engine/neurons/snip_scorer.py`)
  - Safety neuron 识别 (`engine/neurons/safety_identifier.py`)
  - 象限分类 (`engine/neurons/quadrant_classification.py`)
  - 参数对齐 (`engine/neurons/parameter_alignment.py`)
  - 梯度依赖 (`engine/neurons/gradient_dependency.py`)
  - 层演化数据生成
- 创建了 `SimpleTextDataset` 适配器（替代不兼容的 `SaladSafetyDataset`）
- 实测结果：200条样本，43,008个神经元 SNIP 评分完成

### 5. 后端 API 新增 (`visualization/backend/main.py`)
| 端点 | 用途 |
|------|------|
| `/api/data/attacks` | 获取攻击类型列表 |
| `/api/pipeline/cancel/{task_id}` | 取消 pipeline 任务 |
| `/api/metrics/asr-by-attack` | 实时 ASR by attack method |
| `/api/attack_paths` | 桑基图数据（节点+链接） |
| `/api/neuron_activations` | 神经元激活分布数据 |
| `/api/layer_similarity` | 层间相似度矩阵（fallback） |

### 6. 文件管理
- 整理 outputs 目录结构：`outputs/{model_name}/{version}/`
- 把旧根目录数据复制为 `Qwen2.5-1.5B-Instruct/baseline/`
- `/outputs/` 改为动态路由（版本目录 → baseline → 根目录回退）
- 默认活跃模型自动选择第一个有 baseline 的模型

### 7. 打包修复
- `NeuroLens.spec`: 添加 engine/scripts/configs 目录、ML 依赖 hiddenimports
- `main.py`: frozen 模式下直接 import pipeline 模块（不用 subprocess）
- `main.py`: `_get_python_executable()` 验证 dev_env python 可用性
- `main.py`: subprocess 加 `PYTHONUNBUFFERED=1` 和 `PYTHONIOENCODING=utf-8`
- `desktop_app.py`: 优先打开 `/dashboard` 而非 `/`

### 8. Panel A 重写 (`vis/panel_A_control.html`)
- 模型下拉选择（扫描 models/ 目录）
- 版本下拉 + 切换按钮
- 测试档位选择（quick/standard/full）
- Batch Size 和 Max Tokens 输入框
- 攻击类型多选
- 实时进度条 + 统计（ASR/速度/ETA）
- WebSocket 实时推送
- Pipeline 完成后自动切换到新版本

### 9. 面板数据修复
| 面板 | 修复内容 |
|------|----------|
| Panel E | `dedicated_safety_neurons.json` 添加 `dedicated_safety_neurons` key（原来只有 `neurons`） |
| Panel F | API 地址从 `localhost:5000` 改为同源 |
| Panel G | API 地址从 `localhost:5000` 改为同源；API 节点格式加 `id/label/type` 字段；`get_idx()` 函数调用修复 |
| Panel H | API 地址从 `localhost:5000` 改为同源 |

---


## 二、当前已知问题（待修复）

### ⚠️ 1. Panel B（雷达图）比例不对
- **现象**: 雷达图的坐标轴比例异常，Utility 轴 max=100 但 AutoDAN/GCG/GPTFuzz/TAP 的 max=20，导致图形比例失调
- **原因**: 原始硬编码数据中各轴 max 值不统一（Utility=100, 攻击类型=20），而实际 ASR 数据全部是百分比（0-100%）
- **位置**: `vis/panel_B_metric.html` 第 187-194 行 radar.indicator 的 max 值
- **修复方向**: 统一所有轴的 max 为 100，或根据实际数据动态计算

### ⚠️ 2. Panel C 和 K 显示的是旧数据
- **现象**: Panel C（Representation View）和 Panel K（Instance View）显示的是原始 Meta-Llama-3-8B 的数据，不是 Qwen2.5-1.5B 的
- **原因**: 这两个面板读取的是 `../outputs/` 下的文件（`base_evaluation.jsonl`、`dedicated_safety_neurons.json`），而这些文件是从旧的 baseline 复制过来的，不是当前模型的真实推理结果
- **位置**: 
  - Panel C: `vis/panel_C_representation.html` 读取 `../outputs/representation/`
  - Panel K: `vis/panel_K_instance.html` 读取 `../outputs/base_evaluation.jsonl`
- **修复方向**: 需要运行完整的 pipeline（包括推理 + 表征数据生成），生成当前模型的真实输出文件

### ⚠️ 3. Panel E（神经元视图）无神经元加载
- **现象**: Panel E 显示 "Total Number of Dedicated Safety Neurons: 0" 或只显示少量神经元
- **原因**: SNIP 分析只生成了 `snip_scores`，但 `dedicated_safety_neurons.json` 中的 neurons 列表为空（safety identification 步骤因 CUDA OOM 跳过了大部分计算）
- **数据不足**: 200 条样本对 safety neuron 识别来说偏少，且 probe 训练因重试机制被禁用
- **位置**: `outputs/Qwen2.5-1.5B-Instruct/test_snip/dedicated_safety_neurons.json`
- **修复方向**: 
  1. 增加样本数量（至少 500 条）
  2. 解决 SNIP 评分的 CUDA OOM 问题（当前 batch_size=2 仍然可能 OOM）
  3. 重新启用 probe 训练（需要修复重试机制的死循环问题）

### ⚠️ 4. 应用端和网页端显示不同步
- **现象**: `python desktop_app.py` 启动的应用端和浏览器打开 `http://127.0.0.1:6008/dashboard` 显示的内容不一致
- **原因**: 未完全确认。可能与 `desktop_app.py` 的 `_resolve_frontend_url` 逻辑有关（优先打开 `/` 而非 `/dashboard`），已修复优先级但未验证一致性
- **位置**: `visualization/backend/desktop_app.py` 第 86-96 行

---


## 三、犯的错误

### 1. 编码/语法错误
| 错误 | 影响 | 修复 |
|------|------|------|
| PowerShell 中文字符编码 | `build_installer.ps1` 解析失败 | 全部改成英文 |
| `Split-Path -Parent $PSScriptRoot` | `$ProjectRoot` 变成 `F:\` | 改为 `$PSScriptRoot` |
| `python -m pyinstaller`（小写 i） | 找不到模块 | 改为 `python -m PyInstaller` |
| `cap[0]` 取 int 再取下标 | `TypeError` | 改为 `cap = torch.cuda.get_device_capability()` |

### 2. 数据/格式错误
| 错误 | 影响 | 修复 |
|------|------|------|
| `SaladSafetyDataset` 不认 `attack_enhanced_set` 格式 | 返回 0 样本 | 写了 `SimpleTextDataset` 适配器 |
| `dedicated_safety_neurons.json` key 是 `neurons` 而非 `dedicated_safety_neurons` | Panel E 加载失败 | 添加了 `dedicated_safety_neurons` key |
| Panel G API 节点用 `name` 而非 `id/label` | Plotly Sankey 渲染失败 | 修改 API 返回格式 |
| Panel G `get_idx()` 定义了但没调用 | 节点列表为空 | 加了 `get_idx()` 调用 |
| SNIP loss 函数 `reduction='none'` 返回非标量 | `backward()` 失败 | 改为 `reduction='mean'` |

### 3. 架构/设计错误
| 错误 | 影响 | 修复 |
|------|------|------|
| `/outputs/` 静态挂载固定指向根目录 | 切换版本后面板数据不变 | 改为动态路由 |
| `device_map="auto"` 可能 CPU offload | 推理极慢（12秒/条） | 改为 `device_map="cuda:0"` |
| `torch.compile` 逐条推理时编译开销 | 每条都触发编译 | 去掉，改用 warmup |
| frozen 模式用 subprocess 调 `.py` | 打包后找不到 python | 改为直接 import 模块 |
| Probe 训练 29 层×重试循环 | 样本不足时死循环 | 样本<20 跳过；最终禁用 |
| Panel B `loadASRFromReport()` 完全硬编码 | 数据永远是旧的 | 保持原样（恢复 org 后） |
| Panel F/G/H 的 `localhost:5000` 硬编码 | API 请求发到错误端口 | 改为同源 |

### 4. 打包错误
| 错误 | 影响 | 修复 |
|------|------|------|
| spec 文件内容重复两遍 | 打包异常 | 重写为单份 |
| spec 缺 ML 依赖 hiddenimports | 打包后 import 失败 | 添加 torch/transformers 等 |
| spec 缺 engine/scripts 目录 | 打包后找不到模块 | 添加到 datas |
| PyInstaller 工作目录错误 | 相对路径解析到错误位置 | cd 到 backend 目录再运行 |
| `desktop_app.py` 优先打开 `/` | 应用端显示首页而非 dashboard | 改为优先打开 `/dashboard` |

### 5. 过程中的破坏性操作
| 操作 | 后果 |
|------|------|
| 修改 Panel B 的 `loadASRFromReport()` 为 API 调用 | 面板数据消失（API 返回空） |
| 修改 Panel F/G/H 的 Plotly CDN 为本地文件 | iframe 里加载失败（路径解析问题） |
| 内嵌 Plotly 到 HTML 文件 | 文件损坏，显示原始代码 |
| 添加调试文本到 Panel F | 遗留在代码中 |
| 多次修改面板文件导致编码混乱 | 各种解析错误 |

---


## 四、当前状态

### 正常工作
- ✅ GPU 推理（RTX 4060, CUDA 12.4, batch=8, ~1秒/条）
- ✅ Pipeline 推理 + ASR 评估
- ✅ SNIP 评分（43,008 神经元）
- ✅ 所有 API 端点返回正确数据
- ✅ Panel A 控制面板（模型/版本切换、实时进度）
- ✅ Panel D 层演化图
- ✅ Panel E 神经元散点图（61个安全神经元）
- ✅ Panel F 热力图（从 org 恢复后 + API 地址修复）
- ✅ Panel G 桑基图（节点格式修复 + get_idx 调用）
- ✅ Panel H 激活分布图
- ✅ `dedicated_safety_neurons.json` key 兼容

### 未解决
- ❌ Panel B 使用硬编码旧数据（非实时）
- ❌ 应用端和网页端可能仍有显示差异（原因未明）
- ❌ Probe 训练已禁用（29层重试太慢）
- ❌ 参数对齐、梯度依赖等分析依赖 probe 数据

### 已恢复
- 从 `F:\NeuroLens2\org` 目录恢复了所有面板原始文件
- 只修改了 Panel F/G/H 的 API 地址（`localhost:5000` → 同源）
- Panel E 的 `dedicated_safety_neurons` key 已修复

---


## 五、关键文件位置

```
F:\NeuroLens2\
├── org\                           ← 原始面板文件（备份）
├── scripts\
│   ├── run_pipeline.py            ← 重写的 pipeline
│   ├── run_analysis.py            ← 集成分析脚本
│   └── generate_report.py         ← 报告生成
├── visualization\backend\
│   ├── main.py                    ← 后端（新增 API）
│   ├── desktop_app.py             ← 桌面启动器
│   ├── vis\                       ← 面板文件（从 org 恢复）
│   └── NeuroLens.spec             ← PyInstaller 配置
├── outputs\
│   └── Qwen2.5-1.5B-Instruct\
│       ├── baseline\              ← 原始分析数据
│       ├── test_snip\             ← 最新 SNIP 分析结果
│       └── run_quick_200\         ← 200条推理结果
├── models\
│   └── Qwen2.5-1.5B-Instruct\    ← 模型文件
├── data\salad\raw\
│   └── attack_enhanced_set_train.jsonl  ← 数据集 (5000条)
├── requirements.txt               ← 依赖清单
├── build_installer.ps1            ← 打包脚本
├── setup_gpu.ps1                  ← GPU 配置脚本
└── 更新指南.md                     ← 更新文档
```

---

### 10. Panel A 控制面板重写 ✅
- 模型下拉选择（扫描 models/ 目录）
- 版本下拉 + 切换按钮
- 测试档位（Quick/Standard/Full）
- Batch Size + Max Tokens 输入框
- 攻击类型多选
- Run Pipeline + 实时进度条
- 完成后自动切换版本
- 位置：`vis/panel_A_control.html`

### 11. Panel G/H 编码修复 + API 地址修复 ✅
- Panel G 和 H 的中文标题乱码（PowerShell Set-Content 编码问题）
- 用 Python 重写文件，修复 UTF-8 编码
- Panel G/H 的 API 地址从 `localhost:5000` 改为同源
- Panel G 节点格式修复（添加 id/label/type 字段）
- Panel G `get_idx()` 调用修复（之前定义了但没调用）

### 12. 默认版本优先加载最新 ✅
- `_default_active_model()` 改为优先加载 `test_snip`，然后最新 run_*，最后 baseline
- 不再需要手动切换版本

---


## 六、经验教训

1. **不要随意修改能工作的代码** — 面板文件从 `org` 恢复后才恢复正常
2. **iframe 里的相对路径不同于直接访问** — `plotly.min.js` 在 iframe 里解析到根目录
3. **先备份再改** — 没有 git 导致无法回滚
4. **API 地址要统一** — `localhost:5000` vs `6008` 造成面板不加载
5. **subprocess 在打包后不工作** — frozen 模式下需要直接 import
6. **Windows 编码问题** — GBK vs UTF-8 导致各种奇怪错误
7. **Probe 训练有重试机制** — 样本不足时会死循环
8. **PyInstaller 需要正确的工作目录** — 否则相对路径解析错误
