#!/usr/bin/env python3
"""
训练LLAMA每层线性探针分类器 - 使用 labels/ 目录中的标签和 hidden_states/ 目录中的隐藏态
- 标签目录: outputs/data_set_output/labels/
- 隐藏态目录: outputs/hidden_states/
- 自动按文件名前缀匹配，确保标签与隐藏态严格一一对应
- 每层训练100轮（无早停），实时保存每层每轮指标
- 支持 K 折交叉验证，减少波动并提供更稳定的性能估计
"""

import json, os, sys, time, platform, signal, argparse, re, traceback
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import copy, warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm


# ========================= 配置 =========================

class Config:
    labels_dir      = r'outputs/data_set_output/labels'
    hidden_states_dir = r'outputs/hidden_states'

    output_dir  = r'outputs/linear_probes'
    layers_dir  = os.path.join(output_dir, 'layers')

    hidden_dim   = 4096
    num_layers   = 32
    batch_size   = 128
    lr           = 1e-3
    num_epochs   = 100
    weight_decay = 1e-4
    seed         = 42
    device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_folds       = 5       # K 折交叉验证的折数


def _compute_label_name(hs_name):
    """从隐藏态文件名推导对应的标签文件名。"""
    # base_set_hidden_states_0_4999.hs.npy -> base_set_outputs_0_4999.jsonl
    # attack_enhanced_hidden_states.hs.npy -> attack_enhanced_outputs.jsonl
    name = re.sub(r'_hidden_states', '_outputs', hs_name)
    name = re.sub(r'\.hs\.npy$', '.jsonl', name)
    return name


def _build_datasets(config):
    """
    遍历 hidden_states 目录，按命名匹配找到对应标签文件。

    严格对齐机制：
    1. 加载 .idx.npy，获取每个隐藏态对应的 original_index
    2. 读取标签 JSONL，建立 original_index → label 的映射
    3. 用 original_index 从映射中取标签，而非依赖 JSONL 行号
    4. 验证 idx.npy 与 JSONL 的 original_index 是否一致，报告不一致

    仅保留 Safe 和 Unsafe 样本，Controversial 样本会被过滤。
    """
    import glob as _glob

    hs_files = sorted(_glob.glob(os.path.join(config.hidden_states_dir, '*.hs.npy')))
    datasets  = []

    for hs_path in hs_files:
        hs_basename = os.path.basename(hs_path)

        label_name = _compute_label_name(hs_basename)
        label_path = os.path.join(config.labels_dir, label_name)

        if not os.path.exists(label_path):
            print(f'[跳过] 隐藏态 {hs_basename} 没有对应的标签文件: {label_path}', flush=True)
            continue

        idx_path = hs_path.replace('.hs.npy', '.idx.npy')
        if not os.path.exists(idx_path):
            print(f'[跳过] 隐藏态 {hs_basename} 没有对应的索引文件: {idx_path}', flush=True)
            continue

        # --- 步骤1：加载隐藏态（mmap 只读）和 idx.npy（每个隐藏态对应的 original_index） ---
        hs = np.load(hs_path, mmap_mode='r')  # shape: (n_hs, 32, 4096)
        original_indices = np.load(idx_path, mmap_mode='r')  # shape: (n_hs,)
        n_hs = original_indices.shape[0]

        # --- 步骤2：读取标签 JSONL，建立 original_index → label 的映射 ---
        # 支持两种字段名：'label'（新格式）和 '3-category'（原始格式）
        oi_to_label = {}
        oi_to_line  = {}          # 用于对齐验证
        skip_count  = 0
        with open(label_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                oi = data.get('original_index')
                if oi is None:
                    continue
                try:
                    oi = int(oi)
                except (ValueError, TypeError):
                    continue

                # 优先用 label 字段（人工/规则标注），fallback 到 3-category 字段
                label_val = data.get('label', '')
                if not label_val or label_val == 'Controversial':
                    # 跳过 Controversial 或无法判断的样本
                    skip_count += 1
                    continue

                if label_val == 'Unsafe':
                    oi_to_label[oi] = 1
                elif label_val == 'Safe':
                    oi_to_label[oi] = 0
                else:
                    skip_count += 1
                    continue

                oi_to_line[oi] = label_val  # 用于验证报告

        # --- 步骤3：验证 idx.npy 与 JSONL 的 original_index 对齐情况 ---
        mismatches = 0
        missing_in_label = []
        n_jsonl = len(oi_to_line) + skip_count  # 估算原始 JSONL 行数
        for pos in range(min(n_hs, n_jsonl)):
            oi = int(original_indices[pos])
            if oi not in oi_to_line:
                mismatches += 1
                if len(missing_in_label) < 5:
                    missing_in_label.append((pos, oi))

        if mismatches > 0:
            print(f'[警告] {label_name}: idx.npy 与 JSONL 不对齐！', flush=True)
            print(f'       idx.npy 样本数={n_hs}, JSONL 样本数≈{n_jsonl}, 不一致={mismatches}', flush=True)
            if missing_in_label:
                print(f'       idx.npy 中有但 JSONL 缺失的 original_index（前5个）:', flush=True)
                for pos, oi in missing_in_label:
                    print(f'         位置{pos}: original_index={oi}', flush=True)

        # --- 步骤4：用 original_index 严格取标签，跳过无标签的隐藏态 ---
        hs_list, labels_list = [], []
        skipped_no_label = 0
        for pos in range(n_hs):
            oi = int(original_indices[pos])
            if oi not in oi_to_label:
                skipped_no_label += 1
                continue
            hs_list.append(pos)
            labels_list.append(oi_to_label[oi])

        if not hs_list:
            print(f'[跳过] {label_name}: 无有效样本（全部标签缺失或为 Controversial）', flush=True)
            continue

        n_valid = len(labels_list)
        n_safe  = int(np.sum(np.array(labels_list) == 0))
        n_harm  = int(np.sum(np.array(labels_list) == 1))

        if mismatches > 0 or skipped_no_label > 0:
            print(f'[对齐] {label_name}: 总hs={n_hs}, 有效={n_valid}(safe={n_safe}, harm={n_harm}), '
                  f'idx不一致={mismatches}, 缺标签={skipped_no_label}, 跳过Controversial={skip_count}', flush=True)
        else:
            print(f'[加载] {label_name}: {n_valid} 个样本, safe={n_safe}, harmful={n_harm} '
                  f'(idx.npy 与 JSONL 完全对齐)', flush=True)

        hs_arr = np.array(hs_list, dtype=np.int64)
        labels_arr = np.array(labels_list, dtype=np.int64)
        datasets.append((hs, labels_arr, hs_arr, n_valid))

    return datasets


# ========================= 日志系统 =========================

class Logger:
    def __init__(self, log_path):
        self._file   = open(log_path, 'a', encoding='utf-8', buffering=1)
        self._stdout = sys.__stdout__

    def write(self, msg):
        if msg.strip():
            ts   = time.strftime('%m-%d %H:%M:%S')
            self._file.write(f'[{ts}] {msg}\n')
            self._file.flush()
            self._stdout.write(msg + '\n')
            self._stdout.flush()

    def flush(self):
        pass

    def close(self):
        self._file.close()


# ========================= 守护进程 =========================

def daemonize(stdout_path, stderr_path):
    try:
        pid = os.fork()
        if pid > 0:
            print(f"[daemon] 守护进程已启动，子进程 PID={pid}", flush=True)
            sys.exit(0)
    except OSError as e:
        sys.stderr.write(f"[daemon] 第一次 fork 失败: {e}\n")
        sys.stderr.flush()
        sys.exit(1)

    os.setsid()

    try:
        pid = os.fork()
        if pid > 0:
            print(f"[daemon] 孙子进程 PID={pid}，子进程退出", flush=True)
            sys.exit(0)
    except OSError as e:
        sys.stderr.write(f"[daemon] 第二次 fork 失败: {e}\n")
        sys.stderr.flush()
        sys.exit(1)

    for _, fd_num, target in [("stdout", 1, stdout_path), ("stderr", 2, stderr_path)]:
        try:
            log_fd = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o666)
            os.dup2(log_fd, fd_num)
            os.close(log_fd)
        except OSError:
            pass

    signal.signal(signal.SIGHUP, signal.SIG_IGN)


# ========================= 数据处理 =========================

def stratified_kfold_indices(all_labels, n_folds=5, seed=42):
    """
    生成分层 K 折交叉验证的索引。
    确保每折中有害/安全样本比例与整体一致。
    返回: [(train_idx, val_idx), ...] 共 n_folds 个折
    """
    harmful_idx = np.where(all_labels == 1)[0]
    safe_pool   = np.where(all_labels == 0)[0]
    
    if len(safe_pool) < len(harmful_idx):
        raise ValueError(f'安全样本({len(safe_pool)})少于有害样本({len(harmful_idx)})，无法平衡')
    
    safe_sampled = np.random.RandomState(seed).choice(safe_pool, size=len(harmful_idx), replace=False)
    
    labels_balanced = np.concatenate([
        np.zeros(len(safe_sampled)),  # 安全样本标记为 0
        np.ones(len(harmful_idx))     # 有害样本标记为 1
    ])
    indices_balanced = np.concatenate([safe_sampled, harmful_idx])
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in skf.split(indices_balanced, labels_balanced):
        folds.append((indices_balanced[train_idx], indices_balanced[val_idx]))
    
    return folds


# ========================= Dataset & Model =========================

class HS(Dataset):
    """
    多文件分片 Dataset，不拼接 mmap，避免 OOM。

    参数：
        hs_chunks      : list of mmap (N_i, 32, 4096)，各文件的只读内存映射
        all_labels     : ndarray (N_total,) 全局标签
        pos_to_chunk   : ndarray (N_total,) 第 i 个样本属于第几个 chunk
        pos_to_in_idx  : ndarray (N_total,) 第 i 个样本在对应 chunk 的行号
        idx            : ndarray  本次使用的样本索引（全局索引）
        layer          : int  提取哪一层（1-32）
    """

    def __init__(self, hs_chunks, all_labels, pos_to_chunk, pos_to_in_idx, idx, layer):
        self.hs_chunks       = hs_chunks
        self.all_labels      = all_labels
        self.pos_to_chunk    = pos_to_chunk
        self.pos_to_in_idx   = pos_to_in_idx
        self.idx             = np.asarray(idx, dtype=np.int64)
        self.layer           = layer

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        g  = int(self.idx[i])              # 全局索引
        ck = int(self.pos_to_chunk[g])     # 属于哪个 chunk
        row = int(self.pos_to_in_idx[g])   # 在 chunk 内的行号
        layer_idx = self.layer - 1
        return (
            torch.tensor(self.hs_chunks[ck][row, layer_idx, :], dtype=torch.float32),
            torch.tensor(self.all_labels[g], dtype=torch.long),
        )


class Probe(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 2)

    def forward(self, x):
        return self.fc(x)

    def tox_vec(self):
        return self.fc.weight[1] - self.fc.weight[0]


# ========================= 训练/验证 =========================

def train_epoch(model, loader, optim, criterion, device):
    model.train()
    preds, lbls, losses = [], [], 0.0
    for h, y in loader:
        h, y = h.to(device), y.to(device)
        optim.zero_grad()
        out  = model(h)
        loss = criterion(out, y)
        loss.backward()
        optim.step()
        losses += loss.item()
        preds.extend(torch.argmax(out, 1).cpu().numpy())
        lbls.extend(y.cpu().numpy())
    losses = losses / len(lbls)
    preds  = np.array(preds)
    lbls   = np.array(lbls)
    acc    = accuracy_score(lbls, preds)
    s_acc  = accuracy_score(lbls[lbls==0], preds[lbls==0]) if (lbls==0).sum() else 0.0
    h_acc  = accuracy_score(lbls[lbls==1], preds[lbls==1]) if (lbls==1).sum() else 0.0
    return losses, acc, s_acc, h_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    preds, lbls, losses = [], [], 0.0
    for h, y in loader:
        h, y = h.to(device), y.to(device)
        out  = model(h)
        losses += criterion(out, y).item()
        preds.extend(torch.argmax(out, 1).cpu().numpy())
        lbls.extend(y.cpu().numpy())
    acc   = accuracy_score(lbls, preds)
    pa    = np.array(preds)
    la    = np.array(lbls)
    s_acc = accuracy_score(la[la==0], pa[la==0]) if (la==0).sum() else 0.0
    h_acc = accuracy_score(la[la==1], pa[la==1]) if (la==1).sum() else 0.0
    return losses / len(lbls), acc, s_acc, h_acc


def train_fold(config, hs_chunks, all_labels, pos_to_chunk, pos_to_in_idx, train_idx, val_idx, layer, fold_idx=None):
    """训练单个 fold，返回训练好的模型和指标。"""
    layer_str = f'L{layer:02d}'
    if fold_idx is not None:
        layer_str += f'-F{fold_idx+1}'

    train_ds = HS(hs_chunks, all_labels, pos_to_chunk, pos_to_in_idx, train_idx, layer)
    val_ds   = HS(hs_chunks, all_labels, pos_to_chunk, pos_to_in_idx, val_idx,   layer)
    _num_workers = 0  # 禁用多进程，避免 Linux/CUDA 下 DataLoader 死锁
    _pin_memory  = config.device.type == 'cuda'
    train_ld = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                          num_workers=_num_workers, pin_memory=_pin_memory)
    val_ld   = DataLoader(val_ds,   batch_size=config.batch_size, shuffle=False,
                          num_workers=_num_workers, pin_memory=_pin_memory)

    model     = Probe(config.hidden_dim).to(config.device)
    criterion = nn.CrossEntropyLoss()
    optim     = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=config.num_epochs)

    best_val_acc, best_epoch, best_state = -1.0, 0, None
    all_metrics = []

    for epoch in range(config.num_epochs):
        try:
            train_loss, train_acc, train_s_acc, train_h_acc = train_epoch(model, train_ld, optim, criterion, config.device)
            val_loss,   val_acc,  val_s_acc,  val_h_acc  = evaluate(model, val_ld, criterion, config.device)
            scheduler.step()
        except Exception as e:
            raise RuntimeError(f'Fold {fold_idx} Epoch {epoch} 训练失败: {e}\n{traceback.format_exc()}') from e

        em = {
            'epoch':        epoch,
            'train_loss':   float(train_loss),
            'train_acc':    float(train_acc),
            'train_s_acc':  float(train_s_acc),
            'train_h_acc':  float(train_h_acc),
            'val_loss':     float(val_loss),
            'val_acc':      float(val_acc),
            'val_s_acc':    float(val_s_acc),
            'val_h_acc':    float(val_h_acc),
        }
        all_metrics.append(em)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            best_state   = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return {
        'model': model,
        'best_epoch': best_epoch,
        'best_metrics': all_metrics[best_epoch],
        'all_metrics': all_metrics,
        'val_acc': best_val_acc,
    }


def _save_layer_partial(layer_dir, layer, completed_folds, all_fold_results,
                        all_fold_models, total_folds, config):
    """每完成一个 fold 就保存中间结果，支持中断后继续。"""
    # 计算已完成 folds 的平均指标
    val_accs  = [r['val_acc']   for r in all_fold_results]
    val_s_accs = [r['val_s_acc'] for r in all_fold_results]
    val_h_accs = [r['val_h_acc'] for r in all_fold_results]

    avg_metrics = {
        'avg_val_acc':   float(np.mean(val_accs)),
        'avg_val_s_acc': float(np.mean(val_s_accs)),
        'avg_val_h_acc': float(np.mean(val_h_accs)),
        'std_val_acc':   float(np.std(val_accs)) if len(val_accs) > 1 else 0.0,
        'std_val_s_acc': float(np.std(val_s_accs)) if len(val_s_accs) > 1 else 0.0,
        'std_val_h_acc': float(np.std(val_h_accs)) if len(val_h_accs) > 1 else 0.0,
    }

    best_fold_idx = int(np.argmax(val_accs))
    best_result   = all_fold_results[best_fold_idx]
    best_model    = all_fold_models[best_fold_idx]

    # 保存中间 metrics（标注进度）
    metrics_data = {
        'layer': layer,
        'n_folds': total_folds,
        'completed_folds': completed_folds + 1,
        'in_progress': True,
        'best_fold': best_fold_idx,
        'avg_metrics': avg_metrics,
        'all_folds': all_fold_results,
    }
    metrics_path = os.path.join(layer_dir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as mf:
        json.dump(metrics_data, mf, indent=2, ensure_ascii=False)

    # 保存当前最佳模型权重（仅在有完整结果时覆盖最终模型）
    # 如果有完整结果，主循环会覆盖这个文件


def train_layer_cv(config, hs_chunks, all_labels, pos_to_chunk, pos_to_in_idx, cv_folds, layer):
    """
    使用 K 折交叉验证训练单层。
    每个 fold 独立训练，最终返回平均指标和最佳 fold 的模型。
    每个 fold 完成后立即保存，支持断点续训。
    """
    layer_dir = os.path.join(config.layers_dir, f'layer{layer:02d}')
    os.makedirs(layer_dir, exist_ok=True)

    all_fold_results = []
    all_fold_models = []          # 同时保留每个 fold 的最佳模型，避免重复训练

    try:
        fold_pbar = tqdm(cv_folds, desc=f'L{layer:02d} 折', ncols=30, leave=False)
    except Exception as e:
        print(f'[错误] L{layer:02d} 初始化 tqdm 失败: {e}', flush=True)
        raise

    for fold_idx, (train_idx, val_idx) in enumerate(fold_pbar):
        print(f'L{layer:02d} F{fold_idx+1}/{len(cv_folds)} 开始...', flush=True)
        try:
            result = train_fold(config, hs_chunks, all_labels, pos_to_chunk, pos_to_in_idx, train_idx, val_idx, layer, fold_idx)
        except Exception as e:
            print(f'[错误] L{layer:02d} F{fold_idx+1} 训练崩溃: {e}', flush=True)
            raise
        all_fold_models.append(result['model'])
        all_fold_results.append({
            'fold': fold_idx,
            'best_epoch': result['best_epoch'],
            **result['best_metrics'],
        })
        fold_pbar.set_postfix_str(f'F{fold_idx+1} val_acc={result["val_acc"]:.4f}')
        print(f'L{layer:02d} F{fold_idx+1}/{len(cv_folds)} 完成 val_acc={result["val_acc"]:.4f}', flush=True)

        # 每个 fold 完成后立即保存中间结果，防止中断丢失
        _save_layer_partial(layer_dir, layer, fold_idx, all_fold_results,
                           all_fold_models, len(cv_folds), config)

    avg_metrics = {
        'avg_val_acc':  float(np.mean([r['val_acc'] for r in all_fold_results])),
        'avg_val_s_acc': float(np.mean([r['val_s_acc'] for r in all_fold_results])),
        'avg_val_h_acc': float(np.mean([r['val_h_acc'] for r in all_fold_results])),
        'std_val_acc':  float(np.std([r['val_acc'] for r in all_fold_results])),
        'std_val_s_acc': float(np.std([r['val_s_acc'] for r in all_fold_results])),
        'std_val_h_acc': float(np.std([r['val_h_acc'] for r in all_fold_results])),
    }

    best_fold_idx = int(np.argmax([r['val_acc'] for r in all_fold_results]))
    best_result_metrics = all_fold_results[best_fold_idx]
    best_model = all_fold_models[best_fold_idx]

    metrics_data = {
        'layer': layer,
        'n_folds': len(cv_folds),
        'completed_folds': len(cv_folds),
        'in_progress': False,
        'best_fold': best_fold_idx,
        'avg_metrics': avg_metrics,
        'all_folds': all_fold_results,
    }

    metrics_path = os.path.join(layer_dir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as mf:
        json.dump(metrics_data, mf, indent=2, ensure_ascii=False)

    w_path = os.path.join(layer_dir, f'layer{layer:02d}.pt')
    torch.save({
        'layer':           layer,
        'best_fold':       best_fold_idx,
        'weight':          best_model.fc.weight.data.cpu().numpy(),
        'bias':            best_model.fc.bias.data.cpu().numpy(),
        'toxicity_vector': best_model.tox_vec().detach().cpu().numpy(),
        **best_result_metrics,
        'cv_avg_val_acc':  avg_metrics['avg_val_acc'],
        'cv_avg_val_s_acc': avg_metrics['avg_val_s_acc'],
        'cv_avg_val_h_acc': avg_metrics['avg_val_h_acc'],
        'cv_std_val_acc':  avg_metrics['std_val_acc'],
    }, w_path)

    save_readme_cv(layer_dir, config, layer, best_fold_idx,
                    {'best_epoch': best_result_metrics['best_epoch'], 'best_metrics': best_result_metrics},
                    avg_metrics, all_fold_results)

    return {
        'layer': layer,
        'best_fold': best_fold_idx,
        'best_epoch': best_result_metrics['best_epoch'],
        **best_result_metrics,
        **{f'cv_{k}': v for k, v in avg_metrics.items()},
    }


def save_readme_cv(layer_dir, config, layer, best_fold_idx, best_result, avg_metrics, all_fold_results):
    """保存带有交叉验证结果的 README。"""
    readme_path = os.path.join(layer_dir, 'README.md')
    m = best_result['best_metrics']

    with open(readme_path, 'w', encoding='utf-8') as rf:
        rf.write(f"# Linear Probe — Layer {layer:02d} (Cross-Validation)\n\n")
        rf.write(f"## Training Configuration\n\n")
        rf.write(f"| Parameter       | Value         |\n")
        rf.write(f"|-----------------|---------------|\n")
        rf.write(f"| hidden_dim      | {config.hidden_dim}        |\n")
        rf.write(f"| batch_size      | {config.batch_size}        |\n")
        rf.write(f"| learning_rate   | {config.lr}     |\n")
        rf.write(f"| weight_decay    | {config.weight_decay}     |\n")
        rf.write(f"| num_epochs      | {config.num_epochs}        |\n")
        rf.write(f"| n_folds         | {len(all_fold_results)}           |\n")
        rf.write(f"| seed            | {config.seed}        |\n")
        rf.write(f"| device          | {config.device}   |\n")
        rf.write(f"| best_fold       | {best_fold_idx + 1} / {len(all_fold_results)}         |\n\n")

        rf.write(f"## Cross-Validation Results (Mean ± Std)\n\n")
        rf.write(f"| Metric       | Mean ± Std          |\n")
        rf.write(f"|--------------|---------------------|\n")
        rf.write(f"| Val Acc      | {avg_metrics['avg_val_acc']*100:.2f}% ± {avg_metrics['std_val_acc']*100:.2f}%    |\n")
        rf.write(f"| Val Safe Acc | {avg_metrics['avg_val_s_acc']*100:.2f}% ± {avg_metrics['std_val_s_acc']*100:.2f}%    |\n")
        rf.write(f"| Val Harm Acc | {avg_metrics['avg_val_h_acc']*100:.2f}% ± {avg_metrics['std_val_h_acc']*100:.2f}%    |\n\n")

        rf.write(f"## Per-Fold Results\n\n")
        rf.write(f"| Fold | Val Acc | Val Safe Acc | Val Harm Acc |\n")
        rf.write(f"|------|---------|--------------|--------------|\n")
        for fold_res in all_fold_results:
            rf.write(f"| {fold_res['fold']+1} | {fold_res['val_acc']*100:.2f}% | {fold_res['val_s_acc']*100:.2f}% | {fold_res['val_h_acc']*100:.2f}% |\n")
        rf.write(f"\n")

        rf.write(f"## Best Fold Performance (Fold {best_fold_idx + 1}, Epoch {best_result['best_epoch'] + 1})\n\n")
        rf.write(f"| Metric       | Train           | Validation      |\n")
        rf.write(f"|--------------|-----------------|-----------------|\n")
        rf.write(f"| Loss         | {m['train_loss']:.6f}       | {m['val_loss']:.6f}       |\n")
        rf.write(f"| Accuracy     | {m['train_acc']*100:.2f}%          | {m['val_acc']*100:.2f}%          |\n")
        rf.write(f"| Safe Acc     | {m['train_s_acc']*100:.2f}%          | {m['val_s_acc']*100:.2f}%          |\n")
        rf.write(f"| Harmful Acc  | {m['train_h_acc']*100:.2f}%          | {m['val_h_acc']*100:.2f}%          |\n\n")
        rf.write(f"## Data Sources\n\n")
        rf.write(f"- Labels: `{config.labels_dir}`\n")
        rf.write(f"- Hidden states: `{config.hidden_states_dir}`\n\n")
        rf.write(f"## Output Files\n\n")
        rf.write(f"- `layer{layer:02d}.pt` — PyTorch state dict of the best fold model\n")
        rf.write(f"- `metrics.json` — all folds metrics + CV statistics\n")
        rf.write(f"- `README.md` — this file\n")


# ========================= 主流程 =========================

def main():
    parser = argparse.ArgumentParser(description='线性探针训练（labels + hidden_states 配对）')
    parser.add_argument('--daemon', action='store_true',
                        help='以后台守护进程模式运行，SSH 断开后训练继续执行')
    parser.add_argument('--labels_dir', type=str, default=None,
                        help='标签 jsonl 文件目录（默认: outputs/data_set_output/labels）')
    parser.add_argument('--hs_dir', type=str, default=None,
                        help='隐藏态 .hs.npy 文件目录（默认: outputs/hidden_states）')
    parser.add_argument('--resume', action='store_true',
                        help='跳过已完成的层，从上次中断处继续')
    parser.add_argument('--k_folds', '--n_folds', dest='k_folds', type=int, default=5,
                        help='交叉验证的折数（默认: 5）')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数（默认: 100）')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批次大小（默认: 128）')
    parser.add_argument('--start_layer', type=int, default=1,
                        help='起始层编号（默认: 1，从模型的第 N 层开始训练）')
    parser.add_argument('--end_layer', type=int, default=None,
                        help='终止层编号（默认: 32，从模型的第 N 层截止训练）')
    args = parser.parse_args()

    if args.labels_dir:
        Config.labels_dir = args.labels_dir
    if args.hs_dir:
        Config.hidden_states_dir = args.hs_dir
    if args.k_folds:
        Config.n_folds = args.k_folds
    if args.epochs:
        Config.num_epochs = args.epochs
    if args.batch_size:
        Config.batch_size = args.batch_size

    start_layer = args.start_layer
    end_layer = args.end_layer if args.end_layer is not None else Config.num_layers
    if start_layer < 1 or end_layer > Config.num_layers or start_layer > end_layer:
        print(f'错误: 层级范围无效 (1-{Config.num_layers})，请检查 --start_layer 和 --end_layer', flush=True)
        return

    output_dir = Config.output_dir
    os.makedirs(Config.output_dir, exist_ok=True)
    os.makedirs(Config.layers_dir, exist_ok=True)

    log_file    = os.path.join(output_dir, 'train.log')
    stdout_file = os.path.join(output_dir, 'daemon_stdout.log')
    stderr_file = os.path.join(output_dir, 'daemon_stderr.log')

    if args.daemon:
        daemonize(stdout_path=stdout_file, stderr_path=stderr_file)

    sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
    sys.stderr.reconfigure(line_buffering=True, encoding='utf-8')

    logger = Logger(log_file)
    sys.stdout = logger
    sys.stderr = logger

    print('=' * 60, flush=True)
    print('线性探针训练  |  标签目录:', Config.labels_dir, flush=True)
    print('              |  隐藏态目录:', Config.hidden_states_dir, flush=True)
    print('              |  层范围:', start_layer, '-', end_layer, flush=True)
    print('              |  每层', Config.num_epochs, '轮', flush=True)
    print('              |  模式: K折交叉验证', flush=True)
    print('              |  折数:', Config.n_folds, flush=True)
    print('=' * 60, flush=True)

    datasets = _build_datasets(Config)
    if not datasets:
        print('错误: 未找到任何标签/隐藏态配对文件', flush=True)
        return

    # datasets 中的每个元素: (hs_mmap, labels_arr, pos_arr, n_valid)
    # hs_mmap: shape (N_i, 32, 4096) 的 mmap
    # labels_arr: shape (N_i,) 的 ndarray
    # pos_arr: shape (N_i,) 的 ndarray，记录每个有效样本在 hs_mmap 中的行号
    #
    # 不做拼接，改用分片映射：
    #   pos_to_chunk  [N_total] 第 i 个样本属于第几个 chunk
    #   pos_to_in_idx [N_total] 第 i 个样本在对应 chunk 的行号
    # 训练时由 HS.__getitem__ 按需从 mmap 中取数，避免 OOM
    total = sum(d[3] for d in datasets)
    pos_to_chunk  = np.empty(total, dtype=np.int32)
    pos_to_in_idx = np.empty(total, dtype=np.int32)
    all_labels_list = []
    hs_chunks = []
    offset = 0
    for ck_idx, (hs_mmap, labels_arr, pos_arr, n_valid) in enumerate(datasets):
        pos_to_chunk[offset:offset + n_valid]  = ck_idx
        pos_to_in_idx[offset:offset + n_valid] = pos_arr[:n_valid]
        all_labels_list.append(labels_arr)
        hs_chunks.append(hs_mmap)
        offset += n_valid

    all_labels = np.concatenate(all_labels_list)  # shape (total,)

    # === 平衡采样：先找所有有害样本，再等量抽取安全样本 ===
    harmful_global_idx = np.where(all_labels == 1)[0]
    safe_pool_global_idx = np.where(all_labels == 0)[0]
    n_harm = len(harmful_global_idx)

    rng = np.random.RandomState(Config.seed)
    safe_balanced_global_idx = rng.choice(safe_pool_global_idx, size=n_harm, replace=False)

    balanced_global_idx = np.concatenate([safe_balanced_global_idx, harmful_global_idx])
    balanced_labels = all_labels[balanced_global_idx]

    pos_to_chunk  = pos_to_chunk[balanced_global_idx]
    pos_to_in_idx = pos_to_in_idx[balanced_global_idx]

    n_bal_safe = int(np.sum(balanced_labels == 0))
    n_bal_harm = int(np.sum(balanced_labels == 1))

    print(f'样本总数={n_harm*2}  安全={n_bal_safe}  有害={n_bal_harm}', flush=True)
    print(f'\n使用 {Config.n_folds} 折交叉验证模式', flush=True)
    print(f'每折: 训练约 {n_bal_harm*2*(Config.n_folds-1)//Config.n_folds} 样本, 验证约 {n_bal_harm*2//Config.n_folds} 样本', flush=True)
    print('', flush=True)

    cv_folds = stratified_kfold_indices(balanced_labels, Config.n_folds, Config.seed)

    layer_results = []

    print(f'\n开始训练层 {start_layer}-{end_layer}，共 {end_layer - start_layer + 1} 层，每层 {Config.n_folds} 折', flush=True)

    try:
        for layer in range(start_layer, end_layer + 1):
            if args.resume:
                w_path = os.path.join(Config.layers_dir, f'layer{layer:02d}', f'layer{layer:02d}.pt')
                if os.path.exists(w_path):
                    print(f'[跳过] layer{layer:02d} 已存在，跳过', flush=True)
                    continue

            print(f'\n>>> 开始训练 L{layer:02d} ({layer - start_layer + 1}/{end_layer - start_layer + 1})', flush=True)
            start_time = time.time()

            try:
                result = train_layer_cv(Config, hs_chunks, balanced_labels,
                                        pos_to_chunk, pos_to_in_idx,
                                        cv_folds, layer)
            except Exception as e:
                elapsed = time.time() - start_time
                print(f'\n[错误] L{layer:02d} 训练失败 (用时 {elapsed:.0f}s): {e}', flush=True)
                print(traceback.format_exc(), flush=True)
                # 保存当前进度
                if layer_results:
                    with open(os.path.join(output_dir, 'all_layers_summary.json'), 'w') as sf:
                        json.dump(layer_results, sf, indent=2, ensure_ascii=False)
                raise

            layer_results.append({
                'layer': layer,
                'best_fold': result['best_fold'],
                'best_epoch': result['best_epoch'],
                **{k: v for k, v in result.items() if k not in ['layer', 'best_fold', 'best_epoch']},
            })
            elapsed = time.time() - start_time
            print(f'L{layer:02d} 完成 | 最优折={result["best_fold"]+1} 轮={result["best_epoch"]+1} | '
                  f'CV均值={result["cv_avg_val_acc"]*100:.2f}%±{result["cv_std_val_acc"]*100:.2f}% | '
                  f'安全={result["cv_avg_val_s_acc"]*100:.2f}% 有害={result["cv_avg_val_h_acc"]*100:.2f}%'
                  f' (用时 {elapsed:.0f}s)', flush=True)

            with open(os.path.join(output_dir, 'all_layers_summary.json'), 'w') as sf:
                json.dump(layer_results, sf, indent=2, ensure_ascii=False)

    except KeyboardInterrupt:
        print('\n[中断] 用户手动中断训练，已保存当前进度', flush=True)
        if layer_results:
            with open(os.path.join(output_dir, 'all_layers_summary.json'), 'w') as sf:
                json.dump(layer_results, sf, indent=2, ensure_ascii=False)
        raise

    print(f'\n全部完成  |  日志: {log_file}', flush=True)
    logger.close()


if __name__ == '__main__':
    # 全局异常处理器必须在 daemonize 之前设置，确保子进程崩溃时能输出错误
    _log_lock_file = None  # 延迟初始化

    def _global_excepthook(exc_type, exc_value, exc_tb):
        """捕获所有未处理异常，强制输出到 stderr 和日志文件，防止静默崩溃。"""
        import traceback
        tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        msg = f'\n[崩溃] 进程异常退出\n{tb_str}\n'

        # 写入到原始 stderr（daemon 模式下这会被重定向到 daemon_stderr.log）
        sys.__stderr__.write(msg)
        sys.__stderr__.flush()

        # 同时写入到 train.log（如果有的话）
        try:
            log_path = os.path.join(Config.output_dir, 'train.log')
            with open(log_path, 'a', encoding='utf-8') as lf:
                lf.write(msg)
                lf.flush()
        except Exception:
            pass

        # 写入独立的崩溃日志
        try:
            crash_log = os.path.join(Config.output_dir, 'crash.log')
            with open(crash_log, 'a', encoding='utf-8') as cf:
                cf.write(msg)
                cf.flush()
        except Exception:
            pass

    sys.excepthook = _global_excepthook
    main()
