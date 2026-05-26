"""
通用实验可视化脚本 (适用于实验1-5)
====================================
为每个对比实验生成一致的可视化图表集合

每个实验生成的图表类型（固定6张）：
1. 训练曲线对比 (Training Curves Comparison)
2. 损失曲线对比 (Loss Curves Comparison)
3. 准确率曲线对比 (Accuracy Curves Comparison)
4. 学习率调度对比 (Learning Rate Schedule Comparison)
5. 混淆矩阵对比 (Confusion Matrix Comparison)
6. 性能指标雷达图 (Performance Metrics Radar Chart)

所有图表符合顶级学术会议/期刊出版标准：
- 300dpi分辨率
- 清晰的标题、坐标轴标签及单位
- 规范的图例和数据标注
- 专业的配色方案
"""

import json
import csv
import shutil
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

# 全局样式设置
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'axes.linewidth': 1.0,
    'grid.alpha': 0.3,
})

# 学术配色方案
COLORS = {
    'model_a': '#2166AC',      # 蓝色 - 模型A
    'model_b': '#D6604D',      # 红色 - 模型B
    'train': '#2166AC',        # 训练
    'val': '#D6604D',          # 验证
    'stage1': '#67A9CF',       # Stage1
    'stage2': '#EF8A62',       # Stage2
    'positive': '#1A9850',     # 正面指标
    'negative': '#D73027',     # 负面指标
}

OUTPUT_DIR = Path("experiments/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(csv_path: str) -> dict:
    """加载训练曲线CSV文件"""
    epochs = []
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    lr = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            train_loss.append(float(row['train_loss']))
            train_acc.append(float(row['train_acc']))
            val_loss.append(float(row['val_loss']))
            val_acc.append(float(row['val_acc']))
            lr.append(float(row['lr']))
    
    return {
        'epochs': epochs,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'lr': lr,
    }


def load_metrics(json_path: str) -> dict:
    """加载thesis_metrics.json"""
    with open(json_path, 'r') as f:
        return json.load(f)


def merge_stages(stage1: dict, stage2: dict) -> dict:
    """合并Stage1和Stage2数据"""
    if not stage1:
        return stage2
    if not stage2:
        return stage1
    
    max_epoch = max(stage1['epochs'])
    return {
        'epochs': stage1['epochs'] + [e + max_epoch for e in stage2['epochs']],
        'train_loss': stage1['train_loss'] + stage2['train_loss'],
        'train_acc': stage1['train_acc'] + stage2['train_acc'],
        'val_loss': stage1['val_loss'] + stage2['val_loss'],
        'val_acc': stage1['val_acc'] + stage2['val_acc'],
        'lr': stage1['lr'] + stage2['lr'],
    }


def fig1_training_curves_comparison(
    model_a_data: dict,
    model_b_data: dict,
    model_a_name: str,
    model_b_name: str,
    output_path: str
):
    """
    图1: 训练曲线对比 (Training Curves Comparison)
    
    包含：
    - 左图：训练损失与验证损失
    - 右图：训练准确率与验证准确率
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 左图：损失曲线
    ax1.plot(model_a_data['epochs'], model_a_data['train_loss'], 
             color=COLORS['model_a'], linestyle='-', linewidth=1.5, 
             label=f'{model_a_name} 训练', alpha=0.8)
    ax1.plot(model_a_data['epochs'], model_a_data['val_loss'], 
             color=COLORS['model_a'], linestyle='--', linewidth=1.5, 
             label=f'{model_a_name} 验证', alpha=0.8)
    ax1.plot(model_b_data['epochs'], model_b_data['train_loss'], 
             color=COLORS['model_b'], linestyle='-', linewidth=1.5, 
             label=f'{model_b_name} 训练', alpha=0.8)
    ax1.plot(model_b_data['epochs'], model_b_data['val_loss'], 
             color=COLORS['model_b'], linestyle='--', linewidth=1.5, 
             label=f'{model_b_name} 验证', alpha=0.8)
    
    ax1.set_xlabel('轮次 (Epoch)')
    ax1.set_ylabel('损失 (Loss)')
    ax1.set_title('训练与验证损失对比')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, linestyle=':', alpha=0.5)
    
    # 右图：准确率曲线
    ax2.plot(model_a_data['epochs'], model_a_data['train_acc'], 
             color=COLORS['model_a'], linestyle='-', linewidth=1.5, 
             label=f'{model_a_name} 训练', alpha=0.8)
    ax2.plot(model_a_data['epochs'], model_a_data['val_acc'], 
             color=COLORS['model_a'], linestyle='--', linewidth=1.5, 
             label=f'{model_a_name} 验证', alpha=0.8)
    ax2.plot(model_b_data['epochs'], model_b_data['train_acc'], 
             color=COLORS['model_b'], linestyle='-', linewidth=1.5, 
             label=f'{model_b_name} 训练', alpha=0.8)
    ax2.plot(model_b_data['epochs'], model_b_data['val_acc'], 
             color=COLORS['model_b'], linestyle='--', linewidth=1.5, 
             label=f'{model_b_name} 验证', alpha=0.8)
    
    ax2.set_xlabel('轮次 (Epoch)')
    ax2.set_ylabel('准确率 (%)')
    ax2.set_title('训练与验证准确率对比')
    ax2.legend(loc='lower right', framealpha=0.9)
    ax2.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  [OK] Fig1: {output_path}")


def fig2_loss_curves_detailed(
    model_a_data: dict,
    model_b_data: dict,
    model_a_name: str,
    model_b_name: str,
    output_path: str
):
    """
    图2: 损失曲线详细对比 (Loss Curves Detailed)
    
    包含：
    - 训练损失对比
    - 验证损失对比
    - 训练-验证Gap对比
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # 训练损失对比
    axes[0].plot(model_a_data['epochs'], model_a_data['train_loss'], 
                 color=COLORS['model_a'], linestyle='-', linewidth=1.5, 
                 label=model_a_name, alpha=0.8)
    axes[0].plot(model_b_data['epochs'], model_b_data['train_loss'], 
                 color=COLORS['model_b'], linestyle='-', linewidth=1.5, 
                 label=model_b_name, alpha=0.8)
    axes[0].set_xlabel('轮次 (Epoch)')
    axes[0].set_ylabel('训练损失')
    axes[0].set_title('训练损失对比')
    axes[0].legend(framealpha=0.9)
    axes[0].grid(True, linestyle=':', alpha=0.5)
    
    # 验证损失对比
    axes[1].plot(model_a_data['epochs'], model_a_data['val_loss'], 
                 color=COLORS['model_a'], linestyle='-', linewidth=1.5, 
                 label=model_a_name, alpha=0.8)
    axes[1].plot(model_b_data['epochs'], model_b_data['val_loss'], 
                 color=COLORS['model_b'], linestyle='-', linewidth=1.5, 
                 label=model_b_name, alpha=0.8)
    axes[1].set_xlabel('轮次 (Epoch)')
    axes[1].set_ylabel('验证损失')
    axes[1].set_title('验证损失对比')
    axes[1].legend(framealpha=0.9)
    axes[1].grid(True, linestyle=':', alpha=0.5)
    
    # Train-Val Gap
    gap_a = [t - v for t, v in zip(model_a_data['train_loss'], model_a_data['val_loss'])]
    gap_b = [t - v for t, v in zip(model_b_data['train_loss'], model_b_data['val_loss'])]
    
    axes[2].plot(model_a_data['epochs'], gap_a, 
                 color=COLORS['model_a'], linestyle='-', linewidth=1.5, 
                 label=model_a_name, alpha=0.8)
    axes[2].plot(model_b_data['epochs'], gap_b, 
                 color=COLORS['model_b'], linestyle='-', linewidth=1.5, 
                 label=model_b_name, alpha=0.8)
    axes[2].axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    axes[2].set_xlabel('轮次 (Epoch)')
    axes[2].set_ylabel('训练-验证差距')
    axes[2].set_title('过拟合差距 (训练-验证损失)')
    axes[2].legend(framealpha=0.9)
    axes[2].grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  [OK] Fig2: {output_path}")


def fig3_accuracy_curves_detailed(
    model_a_data: dict,
    model_b_data: dict,
    model_a_name: str,
    model_b_name: str,
    output_path: str
):
    """
    图3: 准确率曲线详细对比 (Accuracy Curves Detailed)
    
    包含：
    - 训练准确率对比
    - 验证准确率对比
    - 准确率差值对比
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # 训练准确率对比
    axes[0].plot(model_a_data['epochs'], model_a_data['train_acc'], 
                 color=COLORS['model_a'], linestyle='-', linewidth=1.5, 
                 label=model_a_name, alpha=0.8)
    axes[0].plot(model_b_data['epochs'], model_b_data['train_acc'], 
                 color=COLORS['model_b'], linestyle='-', linewidth=1.5, 
                 label=model_b_name, alpha=0.8)
    axes[0].set_xlabel('轮次 (Epoch)')
    axes[0].set_ylabel('训练准确率 (%)')
    axes[0].set_title('训练准确率对比')
    axes[0].legend(framealpha=0.9)
    axes[0].grid(True, linestyle=':', alpha=0.5)
    
    # 验证准确率对比
    axes[1].plot(model_a_data['epochs'], model_a_data['val_acc'], 
                 color=COLORS['model_a'], linestyle='-', linewidth=1.5, 
                 label=model_a_name, alpha=0.8)
    axes[1].plot(model_b_data['epochs'], model_b_data['val_acc'], 
                 color=COLORS['model_b'], linestyle='-', linewidth=1.5, 
                 label=model_b_name, alpha=0.8)
    axes[1].set_xlabel('轮次 (Epoch)')
    axes[1].set_ylabel('验证准确率 (%)')
    axes[1].set_title('验证准确率对比')
    axes[1].legend(framealpha=0.9)
    axes[1].grid(True, linestyle=':', alpha=0.5)
    
    # 标注最佳点
    best_a = max(model_a_data['val_acc'])
    best_b = max(model_b_data['val_acc'])
    best_a_idx = model_a_data['val_acc'].index(best_a)
    best_b_idx = model_b_data['val_acc'].index(best_b)
    
    axes[1].annotate(f'最优: {best_a:.1f}%', 
                    xy=(model_a_data['epochs'][best_a_idx], best_a),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=8, color=COLORS['model_a'],
                    arrowprops=dict(arrowstyle='->', color=COLORS['model_a']))
    axes[1].annotate(f'最优: {best_b:.1f}%', 
                    xy=(model_b_data['epochs'][best_b_idx], best_b),
                    xytext=(10, -15), textcoords='offset points',
                    fontsize=8, color=COLORS['model_b'],
                    arrowprops=dict(arrowstyle='->', color=COLORS['model_b']))
    
    # 准确率差值 (需要对齐epoch)
    min_epochs = min(len(model_a_data['val_acc']), len(model_b_data['val_acc']))
    diff = [b - a for a, b in zip(model_a_data['val_acc'][:min_epochs], model_b_data['val_acc'][:min_epochs])]
    axes[2].plot(model_b_data['epochs'][:min_epochs], diff, 
                 color=COLORS['positive'] if diff[-1] > 0 else COLORS['negative'], 
                 linestyle='-', linewidth=1.5, alpha=0.8)
    axes[2].axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    axes[2].fill_between(model_b_data['epochs'][:min_epochs], 0, diff, alpha=0.2,
                        color=COLORS['positive'] if diff[-1] > 0 else COLORS['negative'])
    axes[2].set_xlabel('轮次 (Epoch)')
    axes[2].set_ylabel(f'准确率差距 ({model_b_name} - {model_a_name})')
    axes[2].set_title('验证准确率差异')
    axes[2].grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  [OK] Fig3: {output_path}")


def fig4_learning_rate_schedule(
    model_a_data: dict,
    model_b_data: dict,
    model_a_name: str,
    model_b_name: str,
    output_path: str
):
    """
    图4: 学习率调度对比 (Learning Rate Schedule Comparison)
    
    包含：
    - 学习率变化曲线
    - 学习率与验证准确率的关系
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 学习率变化曲线
    ax1.semilogy(model_a_data['epochs'], model_a_data['lr'], 
                 color=COLORS['model_a'], linestyle='-', linewidth=1.5, 
                 label=model_a_name, alpha=0.8)
    ax1.semilogy(model_b_data['epochs'], model_b_data['lr'], 
                 color=COLORS['model_b'], linestyle='-', linewidth=1.5, 
                 label=model_b_name, alpha=0.8)
    ax1.set_xlabel('轮次 (Epoch)')
    ax1.set_ylabel('学习率 (对数刻度)')
    ax1.set_title('学习率调度对比')
    ax1.legend(framealpha=0.9)
    ax1.grid(True, linestyle=':', alpha=0.5)
    
    # 学习率 vs 验证准确率
    ax2.plot(model_a_data['lr'], model_a_data['val_acc'], 
             color=COLORS['model_a'], linestyle='-', linewidth=1.5, 
             label=model_a_name, alpha=0.8, marker='o', markersize=3)
    ax2.plot(model_b_data['lr'], model_b_data['val_acc'], 
             color=COLORS['model_b'], linestyle='-', linewidth=1.5, 
             label=model_b_name, alpha=0.8, marker='s', markersize=3)
    ax2.set_xlabel('学习率 (对数刻度)')
    ax2.set_ylabel('验证准确率 (%)')
    ax2.set_title('学习率与验证准确率关系')
    ax2.set_xscale('log')
    ax2.legend(framealpha=0.9)
    ax2.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  [OK] Fig4: {output_path}")


def fig5_confusion_matrix_comparison(
    cm_a: np.ndarray,
    cm_b: np.ndarray,
    model_a_name: str,
    model_b_name: str,
    output_path: str,
    class_names: list = None,
    normalize: bool = True
):
    """
    图5: 混淆矩阵对比 (Confusion Matrix Comparison)
    """
    if normalize:
        cm_a = cm_a.astype('float') / cm_a.sum(axis=1)[:, np.newaxis]
        cm_b = cm_b.astype('float') / cm_b.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, cm, title in [(ax1, cm_a, model_a_name), (ax2, cm_b, model_b_name)]:
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f'{title} - 混淆矩阵', pad=15, fontsize=11)
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('百分比 (%)', rotation=-90, va="bottom")
        
        ax.set_xlabel('预测标签', fontsize=10)
        ax.set_ylabel('真实标签', fontsize=10)
        
        # 类别太多，只显示部分刻度
        n_classes = cm.shape[0]
        tick_step = max(1, n_classes // 20)
        tick_indices = list(range(0, n_classes, tick_step))
        
        if class_names:
            ax.set_xticks(tick_indices)
            ax.set_xticklabels([class_names[i] for i in tick_indices], rotation=45, ha='right')
            ax.set_yticks(tick_indices)
            ax.set_yticklabels([class_names[i] for i in tick_indices])
        else:
            ax.set_xticks(tick_indices)
            ax.set_yticks(tick_indices)
        
        # 添加数值标注
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if i in tick_indices and j in tick_indices:
                    ax.text(j, i, f'{cm[i, j]:.2f}',
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black",
                           fontsize=6)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  [OK] Fig5: {output_path}")


def fig6_performance_radar_chart(
    metrics_a: dict,
    metrics_b: dict,
    model_a_name: str,
    model_b_name: str,
    output_path: str
):
    """
    图6: 性能指标雷达图 (Performance Metrics Radar Chart)
    
    包含：Top-1 Acc, Top-5 Acc, F1-Score, Inverse Inference Time, Convergence Speed
    """
    # 归一化指标 (0-1范围)
    categories = ['Top-1 准确率', 'Top-5 准确率', 'F1分数', '推理速度\n(1/时间)', '收敛速度\n(1/轮次)']
    N = len(categories)
    
    # 提取并归一化
    top1_a = metrics_a.get('top1_acc', 0) / 100
    top1_b = metrics_b.get('top1_acc', 0) / 100
    
    top5_a = metrics_a.get('top5_acc', 0) / 100
    top5_b = metrics_b.get('top5_acc', 0) / 100
    
    f1_a = metrics_a.get('macro_f1', 0)
    f1_b = metrics_b.get('macro_f1', 0)
    
    # 速度指标 (取倒数，越小越好 -> 越大越好)
    time_a = metrics_a.get('avg_inference_time_ms', 1)
    time_b = metrics_b.get('avg_inference_time_ms', 1)
    max_time = max(time_a, time_b)
    speed_a = 1 / (time_a / max_time + 0.01)
    speed_b = 1 / (time_b / max_time + 0.01)
    max_speed = max(speed_a, speed_b)
    speed_a /= max_speed
    speed_b /= max_speed
    
    # 收敛速度 (取倒数)
    conv_a = metrics_a.get('convergence_epoch', 50)
    conv_b = metrics_b.get('convergence_epoch', 50)
    max_conv = max(conv_a, conv_b)
    conv_speed_a = 1 / (conv_a / max_conv + 0.01)
    conv_speed_b = 1 / (conv_b / max_conv + 0.01)
    max_conv_speed = max(conv_speed_a, conv_speed_b)
    conv_speed_a /= max_conv_speed
    conv_speed_b /= max_conv_speed
    
    values_a = [top1_a, top5_a, f1_a, speed_a, conv_speed_a]
    values_b = [top1_b, top5_b, f1_b, speed_b, conv_speed_b]
    
    # 计算角度
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    values_a += values_a[:1]
    values_b += values_b[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
    
    ax.plot(angles, values_a, 'o-', linewidth=2, color=COLORS['model_a'], label=model_a_name)
    ax.fill(angles, values_a, alpha=0.15, color=COLORS['model_a'])
    
    ax.plot(angles, values_b, 's-', linewidth=2, color=COLORS['model_b'], label=model_b_name)
    ax.fill(angles, values_b, alpha=0.15, color=COLORS['model_b'])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), framealpha=0.9, title='模型')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  [OK] Fig6: {output_path}")


def generate_all_figures_for_experiment(
    exp_name: str,
    model_a_dir: str,
    model_b_dir: str,
    model_a_name: str,
    model_b_name: str,
    class_names: list = None
):
    """
    为单个实验生成所有6张可视化图表
    
    参数：
    - exp_name: 实验名称 (如 "exp1", "exp2", etc.)
    - model_a_dir: 模型A的结果目录
    - model_b_dir: 模型B的结果目录
    - model_a_name: 模型A名称
    - model_b_name: 模型B名称
    - class_names: 类别名称列表 (可选)
    """
    print(f"\n{'='*60}")
    print(f"生成 {exp_name} 可视化图表")
    print(f"{'='*60}")
    
    exp_output_dir = OUTPUT_DIR / exp_name
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    try:
        stage1_a = load_csv(f"{model_a_dir}/curves/stage1.csv")
    except FileNotFoundError:
        stage1_a = None
    
    try:
        stage2_a = load_csv(f"{model_a_dir}/curves/stage2.csv")
    except FileNotFoundError:
        stage2_a = None
    
    try:
        stage1_b = load_csv(f"{model_b_dir}/curves/stage1.csv")
    except FileNotFoundError:
        stage1_b = None
    
    try:
        stage2_b = load_csv(f"{model_b_dir}/curves/stage2.csv")
    except FileNotFoundError:
        stage2_b = None
    
    # 合并阶段
    data_a = merge_stages(stage1_a, stage2_a)
    data_b = merge_stages(stage1_b, stage2_b)
    
    # 加载metrics
    try:
        metrics_a = load_metrics(f"{model_a_dir}/thesis_metrics.json")
    except FileNotFoundError:
        metrics_a = None
    
    try:
        metrics_b = load_metrics(f"{model_b_dir}/thesis_metrics.json")
    except FileNotFoundError:
        metrics_b = None
    
    # 生成图1: 训练曲线对比
    if data_a and data_b:
        fig1_training_curves_comparison(
            data_a, data_b, model_a_name, model_b_name,
            str(exp_output_dir / "fig1_training_curves_comparison.png")
        )
    
    # 生成图2: 损失曲线详细对比
    if data_a and data_b:
        fig2_loss_curves_detailed(
            data_a, data_b, model_a_name, model_b_name,
            str(exp_output_dir / "fig2_loss_curves_detailed.png")
        )
    
    # 生成图3: 准确率曲线详细对比
    if data_a and data_b:
        fig3_accuracy_curves_detailed(
            data_a, data_b, model_a_name, model_b_name,
            str(exp_output_dir / "fig3_accuracy_curves_detailed.png")
        )
    
    # 生成图4: 学习率调度对比
    if data_a and data_b:
        fig4_learning_rate_schedule(
            data_a, data_b, model_a_name, model_b_name,
            str(exp_output_dir / "fig4_learning_rate_schedule.png")
        )
    
    # 生成图5: 混淆矩阵对比 (从训练结果复制)
    cm_a_path = Path(f"{model_a_dir}/visualization/confusion_matrix.png")
    cm_b_path = Path(f"{model_b_dir}/visualization/confusion_matrix.png")
    
    if cm_a_path.exists() and cm_b_path.exists():
        # 复制混淆矩阵到输出目录
        fig5_output = str(exp_output_dir / "fig5_confusion_matrix_comparison.png")
        
        # 创建对比图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        import matplotlib.image as mpimg
        
        # 读取并显示混淆矩阵
        cm_a_img = mpimg.imread(str(cm_a_path))
        cm_b_img = mpimg.imread(str(cm_b_path))
        
        ax1.imshow(cm_a_img)
        ax1.set_title(f'{model_a_name} - 混淆矩阵', pad=15, fontsize=11)
        ax1.axis('off')
        
        ax2.imshow(cm_b_img)
        ax2.set_title(f'{model_b_name} - 混淆矩阵', pad=15, fontsize=11)
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(fig5_output, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  [OK] Fig5: {fig5_output}")
    else:
        print(f"  [WARN] 图5: 混淆矩阵文件不存在，跳过")
    
    # 生成图6: 性能指标雷达图
    if metrics_a and metrics_b:
        fig6_performance_radar_chart(
            metrics_a, metrics_b, model_a_name, model_b_name,
            str(exp_output_dir / "fig6_performance_radar.png")
        )
    
    print(f"\n{exp_name} 图表生成完成！")


def main():
    """主函数：为所有实验生成可视化图表"""
    print("="*60)
    print("开始生成所有实验可视化图表")
    print("="*60)
    
    # 实验1: ResNet50 vs ConvNeXt-Tiny (单一背景)
    generate_all_figures_for_experiment(
        exp_name="exp1",
        model_a_dir="experiments/exp1_resnet50_single/20260421_110057/20260421_110057",
        model_b_dir="experiments/exp1_convnext_single/20260422_020821/20260422_020821",
        model_a_name="ResNet50",
        model_b_name="ConvNeXt-Tiny"
    )
    
    # 实验4: 预训练 vs 从头训练
    generate_all_figures_for_experiment(
        exp_name="exp4",
        model_a_dir="experiments/exp4_resnet50_scratch/20260421_120004/20260421_120004",
        model_b_dir="experiments/exp4_resnet50_pretrained/20260421_120004",
        model_a_name="From Scratch",
        model_b_name="Pretrained"
    )
    
    # 实验5: 两阶段 vs 单阶段
    generate_all_figures_for_experiment(
        exp_name="exp5",
        model_a_dir="experiments/exp5_resnet50_singlestage/20260421_124431/20260421_124431",
        model_b_dir="experiments/exp5_resnet50_twostage/20260421_124431/20260421_124431",
        model_a_name="Single-Stage",
        model_b_name="Two-Stage"
    )
    
    print("\n" + "="*60)
    print("所有实验可视化图表生成完成！")
    print(f"输出目录: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
