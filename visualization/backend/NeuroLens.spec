# -*- mode: python ; coding: utf-8 -*-

# NeuroLens PyInstaller 配置文件
# 包含 Home Dashboard、Help Dashboard、APISafety、Engine、Scripts 等模块
#
# 输出目录（由下方 DISTPATH / WORKPATH 控制）：
#   {项目根目录}/NeuroLens.exe
#   {项目根目录}/_internal/

import os

PROJECT_ROOT = os.path.normpath(os.path.join(SPECPATH, '..', '..'))
DISTPATH = PROJECT_ROOT
WORKPATH = os.path.join(PROJECT_ROOT, 'build')

_NER_MODEL_DIR = os.path.join(SPECPATH, 'Desensization-dashboard', 'models', 'chinese-ner-per-addr-rbt3')
if not os.path.isdir(_NER_MODEL_DIR):
    raise SystemExit(f"未找到脱敏 NER 模型目录：{_NER_MODEL_DIR}")
_has_ner_weights = any(
    os.path.isfile(os.path.join(_NER_MODEL_DIR, name))
    for name in ('model.safetensors', 'pytorch_model.bin')
)
if not _has_ner_weights:
    raise SystemExit(
        f"脱敏 NER 模型目录缺少权重文件：{_NER_MODEL_DIR}\n"
        "至少需要 model.safetensors 或 pytorch_model.bin。"
    )

a = Analysis(
    ['desktop_app.py'],
    pathex=[os.path.join(SPECPATH, 'Desensization-dashboard')],
    binaries=[],
    datas=[
        # 后端 Python 代码
        ('main.py', '.'),                 # 主模块（包含 FastAPI app）
        ('index.html', '.'),              # 可视化页面
        ('neurolens.png', '.'),           # 徽标图片
        ('vis', 'vis'),                   # 可视化静态资源

        # 前端 React 应用
        ('apisafety-dashboard\\dist', 'apisafety'),
        ('home-dashboard\\dist', 'home-dashboard\\dist'),
        ('help-dashboard\\dist', 'help-dashboard\\dist'),
        ('Desensization-dashboard\\dist', 'Desensization-dashboard\\dist'),
        ('Desensization-dashboard\\ner_service.py', 'Desensization-dashboard'),
        ('Desensization-dashboard\\models\\chinese-ner-per-addr-rbt3', 'Desensization-dashboard\\models\\chinese-ner-per-addr-rbt3'),
        ('frozen_patches\\scipy_stats_distn_infrastructure.py', 'frozen_patches'),

        # 资源文件
        ('..\\..\\pic', 'pic'),

        # 版本文件
        ('..\\..\\VERSION', '.'),

        # Pipeline 脚本 (被 main.py subprocess 调用)
        ('..\\..\\scripts', 'scripts'),

        # 引擎模块 (神经元分析、探针、评估等)
        ('..\\..\\engine', 'engine'),

        # 配置文件
        ('..\\..\\configs', 'configs'),

        # 仓库根 models/ 由 Inno Setup 单独打入安装包；此处打包 _internal 内 chinese-ner-per-addr-rbt3 全目录
    ],
    hiddenimports=[
        # Web 框架
        'webview',                        # Web 视图库
        'uvicorn',                        # ASGI 服务器
        'uvicorn.loops',                  # Uvicorn 循环
        'uvicorn.loops.auto',             # Uvicorn 自动循环
        'uvicorn.config',                 # Uvicorn 配置
        'uvicorn.protocols',              # Uvicorn 协议
        'uvicorn.protocols.http',         # Uvicorn HTTP 协议
        'uvicorn.protocols.http.auto',    # Uvicorn HTTP 自动协议
        'uvicorn.protocols.websockets',   # Uvicorn WebSocket 协议
        'uvicorn.protocols.websockets.auto', # Uvicorn WebSocket 自动协议
        'uvicorn.lifespan',               # Uvicorn 生命周期
        'uvicorn.lifespan.on',            # Uvicorn 生命周期事件
        'fastapi',                        # FastAPI 框架
        'pydantic',                       # 数据验证
        'starlette',                      # Starlette 框架
        'starlette.middleware',           # Starlette 中间件
        'starlette.middleware.cors',      # CORS 中间件
        'starlette.responses',            # Starlette 响应
        'starlette.staticfiles',          # Starlette 静态文件
        'jinja2',                         # 模板引擎
        'fastapi.staticfiles',            # FastAPI 静态文件
        'httpx',                          # HTTP 客户端
        'requests',                       # HTTP 请求

        # ML 框架 (Pipeline 脚本需要)
        'torch',                          # PyTorch
        'torch.nn',
        'torch.cuda',
        'torch.backends',
        'torch.backends.cudnn',
        'torch.backends.cuda',
        'transformers',                   # HuggingFace Transformers
        'transformers.models',
        'transformers.models.auto',
        'transformers.generation',        # generate() 需要
        'accelerate',                     # 模型加速加载 (device_map)
        'tokenizers',                     # 分词器 (Rust 实现)
        'safetensors',                    # 安全张量格式
        'transformers.pipelines',         # NER pipeline
        'transformers.pipelines.token_classification',
        'transformers.models.bert.modeling_bert',
        'transformers.models.bert.tokenization_bert',
        'transformers.models.auto.modeling_auto',

        # 数值计算
        'numpy',                          # 数值计算

        # 数据处理 (engine 模块需要)
        'datasets',                       # HuggingFace Datasets
        'pandas',                         # 数据分析
        'scipy',                          # 科学计算
        'sklearn',                        # 机器学习

        # 量化 (可选，8B 模型需要)
        'bitsandbytes',                   # 4-bit 量化
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[os.path.join(SPECPATH, 'pyi_rth_torch_dynamo.py')],
    excludes=[
        # 排除不需要的测试模块
        'matplotlib.tests',
        'pandas.tests',
        'scipy.tests',

        'tensorflow',
        'tensorflow_intel',
        'keras',
        'jupyter',
        'notebook',
        'IPython',
        'pytest',
        'PIL',                            # Pillow (除非需要图片处理)
        'cv2',                            # OpenCV (除非需要)
        'tkinter',                        # GUI (用 webview)
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, compress_level=9)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='NeuroLens',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,                       # True = 显示控制台 (调试用)，False = 无控制台
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='..\\..\\pic\\icon.ico',
)

_COLLECT_NAME = '.'

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=_COLLECT_NAME,
)
