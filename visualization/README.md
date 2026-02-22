# NeuroLens Visualization

NeuroLens 系统的期刊级可视化前端和后端服务。

## 项目结构

```
visualization/
├── frontend/                 # React 前端
│   ├── src/
│   │   ├── components/      # React 组件
│   │   │   ├── ControlPanel.tsx
│   │   │   ├── MetricView.tsx
│   │   │   ├── RepresentationView.tsx
│   │   │   ├── LayerView.tsx
│   │   │   ├── NeuronView.tsx
│   │   │   └── InstanceView.tsx
│   │   ├── services/       # API 服务
│   │   │   └── api.ts
│   │   ├── store/          # 状态管理
│   │   │   └── index.ts
│   │   ├── types/          # TypeScript 类型定义
│   │   │   └── index.ts
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   └── index.css
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── index.html
│
├── backend/                 # FastAPI 后端
│   ├── main.py             # 主服务文件
│   ├── requirements.txt    # Python 依赖
│   ├── api/                # API 路由
│   └── data/               # 数据文件
│
└── logs/                    # 运行日志
```

## 技术栈

### 前端
- React 18 + TypeScript
- Material UI 5 (组件库)
- ECharts 5 (统计图表)
- Plotly.js (科学图表)
- Cytoscape.js (网络图)
- Zustand (状态管理)
- Vite (构建工具)

### 后端
- FastAPI (Python Web 框架)
- NumPy (数据处理)
- Pydantic (数据验证)

## 快速开始

### 前端开发

```bash
cd visualization/frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

访问 http://localhost:5173

### 后端服务

```bash
cd visualization/backend

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt

# 启动服务器
python main.py
```

服务运行在 http://localhost:8000

## 视图说明

### 1. Control Panel (控制面板)
- 模型选择
- 攻击类型配置
- 检测阈值设置
- 微调方法选择

### 2. Metric View (指标视图)
- 雷达图：微调前后指标对比
- 柱状图：各攻击类型 ASR 对比
- 时间序列：训练过程指标变化

### 3. Representation View (表征视图)
- PCA/t-SNE 散点图
- 按层查看隐藏状态分布
- 安全/毒性样本区分

### 4. Layer View (层视图)
- 桑基图：样本流动
- 折线图：层间演化
- 统计表格

### 5. Neuron View (神经元视图)
- 网络图：神经元连接
- 象限分类可视化
- 神经元详情面板

### 6. Instance View (实例视图)
- 实例列表展示
- Prompt 和输出查看
- 筛选和搜索

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/api/metrics` | 获取评估指标 |
| GET | `/api/representation` | 获取表征数据 |
| GET | `/api/layers/evolution` | 获取层演化数据 |
| GET | `/api/neurons/quadrants` | 获取象限分类 |
| GET | `/api/instances` | 获取实例列表 |
| POST | `/api/pipeline/run` | 运行推理管道 |
| POST | `/api/finetune` | 启动微调任务 |
| POST | `/api/intervene` | 神经元干预 |

## 期刊级输出

- 支持 SVG/PNG 导出 (300 DPI)
- 字体配置: Times New Roman
- 色盲友好配色方案
- 矢量格式保证印刷质量

## 许可证

MIT License

