# NeuroLens 项目结构说明

## 顶层目录概览

```
neurolens/
├── engine/           核心库 — 模型推理、神经元分析、探针、微调算法
├── scripts/          研究脚本 — 按功能分为 8 个子目录
├── visualization/    可视化系统 — React 前端 + FastAPI 后端
├── configs/          运行时配置（YAML、环境变量）
├── docker/           Docker 构建文件
├── docs/             项目文档（中英文）
├── data/             数据集（SALAD、Alpaca、Utility 基准）
├── outputs/          实验输出产物（神经元分析、探针、ASR、微调模型）
├── ms_models/        预训练模型缓存（Llama-3-8B、Llama-Guard-3-8B）
└── logs/             评估日志
```

---

## 1. engine/ — 核心库

项目核心算法实现，被 scripts/ 和 visualization/ 共同调用。

```
engine/
├── __init__.py
├── models.py                  模型加载（4-bit 量化）、推理、安全审核
├── server.py                  FastAPI 推理服务端点
│
├── assessment/                评估管线
│   ├── evaluate.py            批量越狱评估（SALAD-Bench）
│   ├── report.py              结果聚合与统计报告
│   └── utility_evaluator.py   通用任务效用评估（ARC、HellaSwag 等）
│
├── neurons/                   神经元分析（项目核心）
│   ├── snip_scorer.py         SNIP 重要性评分 I = |w ⊙ ∇L|
│   ├── safety_identifier.py   安全神经元识别 S(q)
│   ├── utility_identifier.py  效用神经元识别 U(p)
│   ├── activation_projection.py  激活投影 A_i^k
│   ├── parameter_alignment.py    参数对齐 S_i^k
│   ├── quadrant_classification.py 象限分类 (S+A+, S-A+, S+A-, S-A-)
│   ├── gradient_dependency.py  梯度依赖分析
│   ├── salad_safety_dataset.py SALAD 数据集处理
│   └── data_loaders.py        通用数据加载器
│
├── probes/                    线性探针
│   ├── linear_probe.py        基础探针实现
│   └── linear_probe_balanced.py 平衡准确率探针
│
└── fine_tuning/               安全微调方法
    ├── tsft.py                TSFT（目标安全微调）
    ├── dataset_builder.py     训练数据构建
    ├── refusal_templates.py   安全拒绝模板
    └── salad_taxonomy.py      SALAD 攻击分类体系
```

---

## 2. scripts/ — 研究脚本

所有可执行脚本按功能分类为 8 个子目录。

### 2.1 scripts/pipeline/ — 核心研究管线

端到端实验流程的主要入口。

| 脚本 | 功能 |
|------|------|
| `run_neurobreak_pipeline.py` | 完整 NeuroBreak 管线（一键运行所有阶段） |
| `run_vatsft_pipeline.py` | VA-TSFT（脆弱感知目标安全微调）管线 |
| `run_safety_identifier_salad.py` | Phase 1: 识别安全神经元 S(q) |
| `run_utility_identifier.py` | Phase 1: 识别效用神经元 U(p) |
| `run_activation_projection.py` | Phase 2: 激活投影分析 |
| `run_parameter_alignment.py` | Phase 2: 参数对齐分析 |
| `run_quadrant_classification.py` | Phase 3: 象限分类 |
| `run_gradient_dependency.py` | Phase 3: 梯度依赖分析 |
| `run_gradient_correlation.py` | 梯度相关性分析 |
| `run_snip_scorer.py` | SNIP 评分计算 |
| `compute_dedicated_safety_neurons.py` | 计算专用安全神经元 D(p,q) = S(q) \ U(p) |

### 2.2 scripts/evaluation/ — 评估脚本

| 脚本 | 功能 |
|------|------|
| `evaluate_salad_pipeline.py` | SALAD-Bench 全套评估 |
| `evaluate_tsft.py` | TSFT 微调前后对比评估 |
| `evaluate_utility.py` | 效用保持度评估 |
| `run_evaluate_asr.py` | ASR（攻击成功率）评估 |
| `run_evaluate_utility.py` | 效用指标评估 |
| `run_evaluate_report.py` | 评估报告生成 |

### 2.3 scripts/data/ — 数据准备

| 脚本 | 功能 |
|------|------|
| `download_models.py` | 下载 Llama-3-8B、Llama-Guard-3-8B |
| `download_salad.py` | 下载 SALAD-Bench 数据集 |
| `download_alpaca.py` | 下载 Alpaca 指令数据集 |
| `download_utility_datasets.py` | 下载 ARC、HellaSwag 等基准数据集 |
| `extract_hidden_states.py` | 提取并缓存模型隐藏状态 |
| `extract_salad_hidden_states.py` | SALAD 专用隐藏状态提取 |
| `extract_toxic_vectors.py` | 从线性探针提取毒性向量 w_toxic |
| `extract_salad_safety_samples.py` | 提取 SALAD 安全样本 |
| `preprocess_activation_dataset.py` | 激活投影数据预处理 |
| `preprocess_gradient_dataset.py` | 梯度依赖数据预处理 |

### 2.4 scripts/finetuning/ — 微调相关

| 脚本 | 功能 |
|------|------|
| `run_tsft_finetuning.py` | 执行 TSFT 微调训练 |
| `run_build_refusal_dataset.py` | 构建拒绝训练数据集 |
| `apply_delta_and_evaluate.py` | 应用 delta 权重并评估 |
| `apply_delta_extract_and_label.py` | 应用 delta 权重 + 提取隐藏状态 + 标注 |

### 2.5 scripts/probes/ — 探针训练

| 脚本 | 功能 |
|------|------|
| `linear_probe_balanced.py` | 平衡线性探针训练 |
| `offline_snip_compute.py` | 离线 SNIP 分数计算 |
| `offline_snip_select.py` | 离线 SNIP 神经元选择 |
| `select_neurons_by_threshold.py` | 按阈值选择神经元 |

### 2.6 scripts/reporting/ — 报告生成

| 脚本 | 功能 |
|------|------|
| `generate_activation_projection_report.py` | 激活投影分析报告 |
| `generate_quadrant_classification_report.py` | 象限分类报告 |
| `generate_probe_report.py` | 探针训练报告 |
| `generate_asr_report.py` | ASR 评估报告 |
| `generate_layer_evolution.py` | 层间演化分析 |
| `generate_gradient_correlation_report.py` | 梯度相关性报告 |
| `generate_refusal_templates_report.py` | 拒绝模板报告 |
| `generate_salad_category_tables.py` | SALAD 分类统计表 |

### 2.7 scripts/analysis/ — 数据分析

| 脚本 | 功能 |
|------|------|
| `analyze_model_outputs.py` | 分析模型输出结果 |
| `analyze_separability.py` | 层间可分性分析 |
| `analyze_dataset.py` | 数据集统计分析 |
| `analyze_fixed_corrected.py` | 修正后结果分析 |
| `analyze_samples_by_count.py` | 按样本数量分析 |
| `check_activation_projection_data.py` | 验证激活投影数据 |
| `check_dataset_structure.py` | 验证数据集结构 |
| `split_quadrant_neurons.py` | 按象限拆分神经元 |

### 2.8 scripts/tools/ — 工具脚本

| 脚本 | 功能 |
|------|------|
| `check_gpu.py` | CUDA/GPU 环境诊断 |
| `check_models.py` | 模型路径与大小检查 |
| `test_models.py` | 模型推理快速测试 |
| `fix_false_positives.py` | 修复安全分类器误报 |
| `reevaluate_with_fixed_guard.py` | 使用修正后的 Guard 重新评估 |
| `label_outputs_qwen3guard.py` | 使用 Qwen3Guard 重新标注 |
| `mine_refusal_phrases.py` | 挖掘拒绝短语模式 |
| `add_asr_fields.py` | 为评估日志添加 ASR 字段 |
| `rerun_empty_outputs.py` | 重跑空输出的样本 |
| `prepare_finetuning_eval_log.py` | 准备微调评估日志 |
| `extract_dedicated_neurons_for_instance_view.py` | 为可视化提取专用神经元数据 |

---

## 3. visualization/ — 可视化系统

主界面为纯 HTML Dashboard，由 FastAPI 直接托管（port 6008）。React SPA 作为独立开发产物保留。

### 启动方式

```bash
cd visualization/backend
uvicorn main:app --host 0.0.0.0 --port 6008
# 访问 http://localhost:6008/
```

### 目录结构

```
visualization/
├── backend/                    FastAPI 服务（主入口）
│   ├── main.py                 API 端点 + 静态文件托管 + subprocess task runner
│   ├── index.html              Dashboard 主页面（9 个 panel 的 iframe 容器）
│   ├── vis/                    各 panel HTML 文件（与 frontend/vis/ 保持同步）
│   │   ├── panel_A_control.html    控制面板 — 微调/Pipeline/ASR 触发，接入真实后端
│   │   ├── panel_B_metric.html     评估指标雷达图（硬编码 ASR 数据）
│   │   ├── panel_C_representation.html  表征散点图（fetch outputs/representation/）
│   │   ├── panel_D_layer.html      层间演化 + 梯度依赖（fetch outputs/layer_evolution/）
│   │   ├── panel_E_neuron.html     神经元连接图（D3，fetch outputs/quadrant_classification/）
│   │   ├── panel_F_heatmap.html    跨层相似度热力图（fetch :5000/api/layer_similarity）
│   │   ├── panel_G_sankey.html     样本溯源桑基图（fetch :5000/api/attack_paths）
│   │   ├── panel_H_violin.html     神经元激活小提琴图（fetch :5000/api/neuron_activations）
│   │   └── panel_K_instance.html   样本详情表格（fetch outputs/dedicated_safety_neurons.json）
│   └── requirements.txt
│
└── frontend/                   React 18 + TypeScript SPA（独立开发，port 6006）
    ├── index.html              Dashboard 主页面副本（与 backend/index.html 同源）
    ├── vis/                    panel HTML 源文件（编辑此处，cp 同步到 backend/vis/）
    ├── src/                    React 组件（独立 SPA，未集成到主 dashboard）
    │   ├── components/         ControlPanel, MetricView, RepresentationView,
    │   │                       LayerView, NeuronView, InstanceView 等
    │   ├── services/api.ts     API 客户端
    │   └── store/index.ts      Zustand 全局状态
    └── vite.config.ts          dev: port 6006, proxy /api → :6008
```

### Panel 数据依赖与 Fallback

所有 panel（除 A、B 外）在无真实 outputs 时自动使用内嵌默认数据渲染，不显示空白或错误。

| Panel | 真实数据来源 | Fallback 策略 |
|-------|-------------|---------------|
| C | `outputs/representation/representation_layer_N_mode.json` | 内嵌确定性散点 |
| D | `outputs/layer_evolution/streamgraph_data.json` + gradient_dependency | 内嵌 sin 曲线数据 |
| E | `outputs/quadrant_classification/`, `parameter_alignment/`, `gradient_dependency/`, `dedicated_safety_neurons.json` | `_default*()` 函数生成 |
| F | `localhost:5000/api/layer_similarity` | 内嵌 33×33 衰减矩阵 |
| G | `localhost:5000/api/attack_paths` | 内嵌桑基图 nodes/links |
| H | `localhost:5000/api/neuron_activations` | 内嵌 sin 波形激活值 |
| K | `outputs/dedicated_safety_neurons.json`, `outputs/base_evaluation.jsonl` | 空神经元字典，已有样本数据 |

### Task Runner API（Panel A 接入）

`main.py` 中基于 `subprocess` + `threading` 实现真实脚本调度：

| 端点 | 功能 |
|------|------|
| `POST /api/tasks/finetune` | 启动 `scripts/finetuning/run_tsft_finetuning.py` |
| `POST /api/tasks/pipeline` | 启动 `scripts/pipeline/run_neurobreak_pipeline.py` |
| `POST /api/tasks/asr` | 启动 `scripts/evaluation/run_evaluate_asr.py` |
| `GET /api/tasks/{id}?log_offset=N` | 增量日志轮询（Panel A 每 1.5s 拉取一次） |
| `DELETE /api/tasks/{id}` | terminate 进程 |

---

## 4. configs/ — 配置文件

```
configs/
└── runtime/
    ├── salad.yaml         SALAD 评估参数（攻击类型、采样数、阈值）
    └── .env               环境变量（ModelScope Token、CUDA 设备等）
```

---

## 5. 数据与输出目录

### data/ — 输入数据集

| 子目录 | 内容 | 规模 |
|--------|------|------|
| `salad/raw/` | SALAD-Bench 越狱攻击数据集 | ~30K 样本 |
| `salad/processed/` | 预处理后的评估集 | — |
| `alpaca/` | Alpaca 指令数据集（安全基准） | 52K 样本 |
| `utility/arc/` | ARC Challenge | 26K+ 样本 |
| `utility/hellaswag/` | HellaSwag 常识推理 | ~20K 样本 |
| `utility/openbookqa/` | OpenBookQA | ~10K 样本 |
| `utility/super_glue/` | BoolQ, RTE | ~10K 样本 |
| `utility/winogrande/` | Winogrande 共指消解 | ~12K 样本 |
| `utility/wikitext/` | WikiText 语言建模 | ~5K 样本 |

### outputs/ — 实验产物

| 子目录 | 内容 | 大小 |
|--------|------|------|
| `tsft_finetuning/` | TSFT 微调模型与检查点 | 46G |
| `vatsft_pipeline/` | VA-TSFT 管线输出 | 41G |
| `hidden_states/` | 缓存的隐藏状态 | 13G |
| `neurons/` | 神经元分析结果（JSON） | 61M |
| `linear_probes/` | 探针模型权重 | 2.8M |
| `toxic_vectors/` | 毒性向量（NPZ） | 1.1M |
| `asr/` | ASR 评估结果 | 16K |
| `asr_baseline.jsonl` | 基线评估详细日志 | 22M |

### ms_models/ — 预训练模型

| 模型 | 路径 | 用途 |
|------|------|------|
| Meta-Llama-3-8B-Instruct | `LLM-Research/Meta-Llama-3-8B-Instruct/` | 目标 LLM |
| Llama-Guard-3-8B | `LLM-Research/Llama-Guard-3-8B/` | 安全分类器 |

---

## 6. 典型工作流

```
Phase 1: 神经元识别
  scripts/pipeline/run_safety_identifier_salad.py   → outputs/neurons/safety_neurons.json
  scripts/pipeline/run_utility_identifier.py        → outputs/neurons/utility_neurons.json

Phase 2: 激活与对齐分析
  scripts/pipeline/run_activation_projection.py     → outputs/neurons/activation_projection.json
  scripts/pipeline/run_parameter_alignment.py       → outputs/neurons/parameter_alignment.json

Phase 3: 分类与综合
  scripts/pipeline/run_quadrant_classification.py   → outputs/neurons/quadrant_classification.json
  scripts/pipeline/compute_dedicated_safety_neurons.py → outputs/neurons/dedicated_safety_neurons.json

Phase 4（可选）: 微调
  scripts/finetuning/run_tsft_finetuning.py         → outputs/tsft_finetuning/

Phase 5: 评估
  scripts/evaluation/evaluate_salad_pipeline.py     → outputs/asr/
  scripts/evaluation/evaluate_utility.py            → outputs/utility/

一键运行:
  scripts/pipeline/run_neurobreak_pipeline.py       → 自动串联 Phase 1-5
```

---

## 7. 本次重构中删除的文件

### 已删除的重复脚本

| 文件 | 删除原因 | 替代方案 |
|------|----------|----------|
| `scripts/snip_scorer.py` | engine/neurons/snip_scorer.py 的简化副本（255 行 vs 1342 行） | `engine/neurons/snip_scorer.py` |
| `scripts/extract_toxicity_vectors.py` | 与 extract_toxic_vectors.py 功能重复（旧版） | `scripts/data/extract_toxic_vectors.py` |
| `scripts/eval_utility.py` | 与 evaluate_utility.py 功能重复 | `scripts/evaluation/evaluate_utility.py` |
| `scripts/calculate_asr.py` | 与 run_evaluate_asr.py 功能重复（基于标签的简化版） | `scripts/evaluation/run_evaluate_asr.py` |
| `scripts/evaluate_results.py` | 与 evaluate_salad_pipeline.py 功能重复 | `scripts/evaluation/evaluate_salad_pipeline.py` |

### 已删除的临时/测试脚本

| 文件 | 删除原因 |
|------|----------|
| `scripts/quick_test.py` | 临时快速测试，功能被 test_models.py 覆盖 |
| `scripts/quick_test_cached.py` | 同上（缓存版本） |
| `scripts/simple_test.py` | 已从 git 中删除 |
| `scripts/start_server.py` | 已从 git 中删除，engine/server.py 已替代 |

### 已删除的过时 Shell/PowerShell 脚本

| 文件 | 删除原因 |
|------|----------|
| `scripts/run_salad_evaluation.ps1` | Windows PowerShell，Linux 环境无用 |
| `scripts/run_salad_evaluation.sh` | 已过时 |
| `scripts/run_snip_scorer.ps1` | Windows PowerShell |
| `scripts/sync_to_container.ps1` | Windows PowerShell |
| `scripts/sync_to_container.sh` | 已过时 |
| `scripts/sync_to_docker.ps1` | Windows PowerShell |
| `scripts/sync_to_docker.sh` | 已过时 |
| `scripts/verify_salad.ps1` | Windows PowerShell |
| `scripts/download_alpaca_docker.ps1` | Windows PowerShell |
| `scripts/remove_models.ps1` | Windows PowerShell |
| `scripts/remove_models.sh` | 危险性删除脚本 |
| `scripts/remove_models_from_container.ps1` | Windows PowerShell |
| `scripts/run_container.ps1` | Windows PowerShell |
| `scripts/run_snip_scorer_example.ps1` | Windows PowerShell |

### 已删除的过时训练脚本

| 文件 | 删除原因 |
|------|----------|
| `scripts/train_linear_probe_labels.py` | 已从 git 中删除 |
| `scripts/train_probes_balanced.py` | 已从 git 中删除 |
| `scripts/train_probes_balanced .py` | 带空格的文件名，已从 git 中删除 |

### 已删除的过时文档

| 文件 | 删除原因 |
|------|----------|
| `scripts/README_REMOVE_MODELS.md` | 关联的脚本已删除 |
| `scripts/README_SALAD_EVALUATION.md` | 关联的脚本已删除 |

---

## 8. outputs/ 中可能需要清理的冗余目录

以下输出目录存在命名重复，建议后续统一：

| 目录 | 大小 | 说明 |
|------|------|------|
| `outputs/toxic_vectors/` | 1.1M | 新版毒性向量（NPZ 格式） |
| `outputs/toxicity_vectors/` | 676K | 旧版毒性向量（JSON/NPY 格式），建议后续移除 |
| `outputs/simulated_eval/` | 28M | 模拟评估数据，如不再需要可清理 |
| `outputs/tmp_refusal_templates/` | 16K | 临时文件，可清理 |
