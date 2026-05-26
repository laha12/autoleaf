import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import os
import re
import json

# ============================================================
# 全局配置：学术风格 + 中文支持
# ============================================================
plt.rcParams['font.family'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = '#2C2C2C'
plt.rcParams['axes.labelcolor'] = '#2C2C2C'
plt.rcParams['xtick.color'] = '#2C2C2C'
plt.rcParams['ytick.color'] = '#2C2C2C'
plt.rcParams['text.color'] = '#2C2C2C'

OUTPUT_DIR = r'c:\Users\86185\Desktop\autoleaf\experiments\figures\exp3'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 数据解析函数
# ============================================================

def parse_training_log(log_path):
    """从 training.log 解析训练曲线数据"""
    epochs = []
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    val_top5 = []
    lr_values = []

    pattern = re.compile(
        r'Time\s+[\d.]+s\s*\|\s*TrainLoss\s+([\d.]+)\s*\|\s*TrainAcc\s+([\d.]+)%\s*\|\s*ValLoss\s+([\d.]+)\s*\|\s*ValAcc\s+([\d.]+)%\s*\|\s*ValTop5\s+([\d.]+)%'
    )
    lr_pattern = re.compile(r'Current\s+Stage\d+\s+learning\s+rate:\s+([\d.e+-]+)')

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = len(epochs) + 1
                epochs.append(epoch)
                train_loss.append(float(match.group(1)))
                train_acc.append(float(match.group(2)))
                val_loss.append(float(match.group(3)))
                val_acc.append(float(match.group(4)))
                val_top5.append(float(match.group(5)))
            
            lr_match = lr_pattern.search(line)
            if lr_match:
                lr_values.append(float(lr_match.group(1)))

    return {
        'epochs': epochs,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'val_top5': val_top5,
        'lr_values': lr_values,
    }


def parse_thesis_metrics(metrics_path):
    """从 thesis_metrics.json 读取论文指标"""
    with open(metrics_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return {
        'Top-1 Acc (%)': round(data['top1_acc'], 2),
        'Top-5 Acc (%)': round(data['top5_acc'], 2),
        '参数量 (M)': round(data['total_params_m'], 2),
        'FLOPs (G)': round(data['flops_g'], 2),
        '训练时间 (s)': round(data['training_time_s'], 1),
        '推理速度 (ms)': round(data['avg_inference_time_ms'], 2),
        '收敛轮数': data['convergence_epoch'],
        'Macro-P': round(data['macro_precision'], 3),
        'Macro-R': round(data['macro_recall'], 3),
        'Macro-F1': round(data['macro_f1'], 3),
        '最终损失': round(data['final_loss'], 3),
        'stability': data.get('stability', {}),
    }


# ============================================================
# 数据加载
# ============================================================

# 实验路径配置
YOLO_DIR = r'c:\Users\86185\Desktop\autoleaf\experiments\exp3_resnet50_yolo\20260427_154646\20260427_154646'
CENTER_DIR = r'c:\Users\86185\Desktop\autoleaf\experiments\exp3_resnet50_center\20260427_154500\20260427_154500'

# 解析YOLO ROI数据
yolo_log = parse_training_log(os.path.join(YOLO_DIR, 'training.log'))
yolo_metrics = parse_thesis_metrics(os.path.join(YOLO_DIR, 'thesis_metrics.json'))

# 解析Center ROI数据
center_log = parse_training_log(os.path.join(CENTER_DIR, 'training.log'))
center_metrics = parse_thesis_metrics(os.path.join(CENTER_DIR, 'thesis_metrics.json'))

# 统一配色方案
COLOR_YOLO = '#2E86AB'    # 深蓝色：YOLO ROI提取
COLOR_CENTER = '#A23B72'  # 紫红色：中心裁剪
COLOR_GRID = '#E8E8E8'
COLOR_TEXT = '#2C2C2C'

print(f"YOLO ROI: {len(yolo_log['epochs'])} epochs, Top-1={yolo_metrics['Top-1 Acc (%)']}%, 收敛={yolo_metrics['收敛轮数']}轮")
print(f"Center ROI: {len(center_log['epochs'])} epochs, Top-1={center_metrics['Top-1 Acc (%)']}%, 收敛={center_metrics['收敛轮数']}轮")

# 从日志中动态推断Stage分界点
def infer_stage1_end(log_path):
    """从training.log推断Stage1实际结束轮次"""
    stage1_end = None
    with open(log_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if 'Stage1 training done' in line:
                stage1_end = i
                break
    if stage1_end is None:
        return None
    
    # 统计Stage1 training done之前的训练轮次数
    epoch_count = 0
    with open(log_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= stage1_end:
                break
            if re.search(r'Time\s+[\d.]+s\s*\|\s*TrainLoss', line):
                epoch_count += 1
    return epoch_count

yolo_stage1_epochs = infer_stage1_end(os.path.join(YOLO_DIR, 'training.log'))
center_stage1_epochs = infer_stage1_end(os.path.join(CENTER_DIR, 'training.log'))

print(f"YOLO Stage1实际轮次: {yolo_stage1_epochs}")
print(f"Center Stage1实际轮次: {center_stage1_epochs}")

# ============================================================
# 图1：训练曲线对比 + 局部放大（收敛阶段细节）
# ============================================================
fig, ax = plt.subplots(figsize=(11, 7))

# 主图：完整训练曲线
ax.plot(yolo_log['epochs'], yolo_log['val_acc'], color=COLOR_YOLO, linewidth=2.0,
        marker='s', markersize=4, markevery=2, label='YOLOv8 ROI提取', zorder=3)
ax.plot(center_log['epochs'], center_log['val_acc'], color=COLOR_CENTER, linewidth=2.0,
        marker='o', markersize=4, markevery=2, label='中心裁剪', zorder=3)

# Stage分界线
ax.axvline(x=yolo_stage1_epochs + 0.5, color=COLOR_YOLO, linestyle='--', linewidth=1.0, alpha=0.4)
ax.axvline(x=center_stage1_epochs + 0.5, color=COLOR_CENTER, linestyle='--', linewidth=1.0, alpha=0.4)

# 标注Stage2区域
ax.text(yolo_stage1_epochs + 1, 2, 'Stage 2', fontsize=8, color=COLOR_YOLO, rotation=90, va='bottom', alpha=0.7)
ax.text(center_stage1_epochs + 1, 2, 'Stage 2', fontsize=8, color=COLOR_CENTER, rotation=90, va='bottom', alpha=0.7)

# 主图配置
ax.set_xlabel('训练轮次 (Epoch)', fontsize=13, fontweight='medium')
ax.set_ylabel('验证准确率 (%)', fontsize=13, fontweight='medium')
ax.set_title('验证准确率收敛曲线对比', fontsize=14, fontweight='bold', pad=12)
ax.set_xlim(1, max(len(yolo_log['epochs']), len(center_log['epochs'])))
ax.set_ylim(0, 100)
ax.grid(True, linestyle='-', alpha=0.3, color=COLOR_GRID)
ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='#CCCCCC', fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_convergence_curves.png'), format='png', dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_convergence_curves.svg'), format='svg', bbox_inches='tight')
plt.close(fig)
print('[OK] 图1 已保存: 训练曲线对比')

# ============================================================
# 图2：综合性能对比柱状图（精度 + 效率）
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左图：精度指标
ax1 = axes[0]
categories_acc = ['Top-1\n准确率 (%)', 'Top-5\n准确率 (%)', 'Macro-F1\n(×100)']
yolo_acc_vals = [yolo_metrics['Top-1 Acc (%)'], yolo_metrics['Top-5 Acc (%)'], yolo_metrics['Macro-F1'] * 100]
center_acc_vals = [center_metrics['Top-1 Acc (%)'], center_metrics['Top-5 Acc (%)'], center_metrics['Macro-F1'] * 100]

x = np.arange(len(categories_acc))
width = 0.32
bars1 = ax1.bar(x - width/2, yolo_acc_vals, width, label='YOLOv8 ROI提取', 
                color=COLOR_YOLO, edgecolor='white', linewidth=0.8)
bars2 = ax1.bar(x + width/2, center_acc_vals, width, label='中心裁剪', 
                color=COLOR_CENTER, edgecolor='white', linewidth=0.8)

ax1.set_ylabel('准确率 / F1 (%)', fontsize=12, fontweight='medium')
ax1.set_title('(a) 分类精度对比', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(categories_acc, fontsize=11)
ax1.set_ylim(0, 105)
ax1.legend(frameon=True, fancybox=False, edgecolor='#CCCCCC', fontsize=11)
ax1.grid(True, axis='y', linestyle='-', alpha=0.3, color=COLOR_GRID)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha='center', va='bottom', fontsize=10, color=COLOR_TEXT, fontweight='bold')
for bar in bars2:
    height = bar.get_height()
    ax1.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha='center', va='bottom', fontsize=10, color=COLOR_TEXT, fontweight='bold')

# 右图：效率指标
ax2 = axes[1]
categories_eff = ['训练时间\n(s)', '收敛轮数', '推理速度\n(ms)']
yolo_eff_vals = [yolo_metrics['训练时间 (s)'], yolo_metrics['收敛轮数'], yolo_metrics['推理速度 (ms)']]
center_eff_vals = [center_metrics['训练时间 (s)'], center_metrics['收敛轮数'], center_metrics['推理速度 (ms)']]

x2 = np.arange(len(categories_eff))
bars3 = ax2.bar(x2 - width/2, yolo_eff_vals, width, label='YOLOv8 ROI提取', 
                color=COLOR_YOLO, edgecolor='white', linewidth=0.8)
bars4 = ax2.bar(x2 + width/2, center_eff_vals, width, label='中心裁剪', 
                color=COLOR_CENTER, edgecolor='white', linewidth=0.8)

ax2.set_ylabel('数值', fontsize=12, fontweight='medium')
ax2.set_title('(b) 训练效率对比', fontsize=13, fontweight='bold')
ax2.set_xticks(x2)
ax2.set_xticklabels(categories_eff, fontsize=11)
ax2.legend(frameon=True, fancybox=False, edgecolor='#CCCCCC', fontsize=11)
ax2.grid(True, axis='y', linestyle='-', alpha=0.3, color=COLOR_GRID)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

for bar in bars3:
    height = bar.get_height()
    ax2.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha='center', va='bottom', fontsize=10, color=COLOR_TEXT)
for bar in bars4:
    height = bar.get_height()
    ax2.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha='center', va='bottom', fontsize=10, color=COLOR_TEXT)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig2_performance_comparison.png'), format='png', dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUTPUT_DIR, 'fig2_performance_comparison.svg'), format='svg', bbox_inches='tight')
plt.close(fig)
print('[OK] 图2 已保存: 综合性能对比')

# ============================================================
# 图3：多维度能力评估雷达图
# ============================================================
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, polar=True)

categories_radar = ['Top-1 准确率', 'Top-5 准确率', 'Macro-F1', '收敛速度', '训练效率']
N = len(categories_radar)

# 构建原始值矩阵
raw_matrix = np.array([
    [yolo_metrics['Top-1 Acc (%)'], yolo_metrics['Top-5 Acc (%)'],
     yolo_metrics['Macro-F1'], 1/yolo_metrics['收敛轮数'], 1/yolo_metrics['训练时间 (s)']],
    [center_metrics['Top-1 Acc (%)'], center_metrics['Top-5 Acc (%)'],
     center_metrics['Macro-F1'], 1/center_metrics['收敛轮数'], 1/center_metrics['训练时间 (s)']],
])

# Min-Max 归一化
norm_matrix = np.zeros_like(raw_matrix)
for j in range(raw_matrix.shape[1]):
    col = raw_matrix[:, j]
    min_v, max_v = col.min(), col.max()
    if max_v > min_v:
        norm_matrix[:, j] = (col - min_v) / (max_v - min_v)
    else:
        norm_matrix[:, j] = 0.5

yolo_norm = norm_matrix[0].tolist() + [norm_matrix[0][0]]
center_norm = norm_matrix[1].tolist() + [norm_matrix[1][0]]

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

ax.plot(angles, yolo_norm, color=COLOR_YOLO, linewidth=2.5, linestyle='solid', label='YOLOv8 ROI提取', zorder=3)
ax.fill(angles, yolo_norm, color=COLOR_YOLO, alpha=0.15, zorder=2)
ax.plot(angles, center_norm, color=COLOR_CENTER, linewidth=2.5, linestyle='solid', label='中心裁剪', zorder=3)
ax.fill(angles, center_norm, color=COLOR_CENTER, alpha=0.15, zorder=2)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories_radar, fontsize=12, color=COLOR_TEXT)
ax.set_ylim(0, 1.15)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9, color='#888888')
ax.grid(True, linestyle='-', alpha=0.4, color=COLOR_GRID)

# 外侧标注原始值
label_data = [
    (f"{yolo_metrics['Top-1 Acc (%)']:.2f}%", f"{center_metrics['Top-1 Acc (%)']:.2f}%"),
    (f"{yolo_metrics['Top-5 Acc (%)']:.2f}%", f"{center_metrics['Top-5 Acc (%)']:.2f}%"),
    (f"{yolo_metrics['Macro-F1']:.3f}", f"{center_metrics['Macro-F1']:.3f}"),
    (f"{yolo_metrics['收敛轮数']} 轮", f"{center_metrics['收敛轮数']} 轮"),
    (f"{yolo_metrics['训练时间 (s)']:.0f}s", f"{center_metrics['训练时间 (s)']:.0f}s"),
]

for i, (y_val, c_val) in enumerate(label_data):
    angle = angles[i]
    ax.text(angle, 1.08, f'YOLO: {y_val}\nCenter: {c_val}',
           ha='center', va='center', fontsize=9, color=COLOR_TEXT,
           transform=ax.transData)

ax.set_title('多维度综合能力评估', fontsize=15, fontweight='bold', y=1.08, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), frameon=True, fancybox=False, edgecolor='#CCCCCC', fontsize=11)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig3_radar_comparison.png'), format='png', dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUTPUT_DIR, 'fig3_radar_comparison.svg'), format='svg', bbox_inches='tight')
plt.close(fig)
print('[OK] 图3 已保存: 雷达图')

# ============================================================
# 图4：过拟合程度对比（Train-Val Gap）
# ============================================================
fig, ax = plt.subplots(figsize=(11, 6.5))

# 计算Train-Val准确率差距
yolo_gap = [t - v for t, v in zip(yolo_log['train_acc'], yolo_log['val_acc'])]
center_gap = [t - v for t, v in zip(center_log['train_acc'], center_log['val_acc'])]

ax.plot(yolo_log['epochs'], yolo_gap, color=COLOR_YOLO, linewidth=2.0,
        marker='s', markersize=4, markevery=2, label='YOLOv8 ROI提取', zorder=3)
ax.plot(center_log['epochs'], center_gap, color=COLOR_CENTER, linewidth=2.0,
        marker='o', markersize=4, markevery=2, label='中心裁剪', zorder=3)

# 零线
ax.axhline(y=0, color='#666666', linestyle='-', linewidth=0.8, alpha=0.5)

# Stage分界线
ax.axvline(x=yolo_stage1_epochs + 0.5, color=COLOR_YOLO, linestyle='--', linewidth=1.0, alpha=0.4)
ax.axvline(x=center_stage1_epochs + 0.5, color=COLOR_CENTER, linestyle='--', linewidth=1.0, alpha=0.4)

# Stage2标注
ax.text(yolo_stage1_epochs + 1, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05, 
        'Stage 2', fontsize=8, color=COLOR_YOLO, rotation=90, va='bottom', alpha=0.7)
ax.text(center_stage1_epochs + 1, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05, 
        'Stage 2', fontsize=8, color=COLOR_CENTER, rotation=90, va='bottom', alpha=0.7)

ax.set_xlabel('训练轮次 (Epoch)', fontsize=13, fontweight='medium')
ax.set_ylabel('Train-Val 准确率差距 (%)', fontsize=13, fontweight='medium')
ax.set_title('过拟合程度对比 (Train-Val Gap)', fontsize=14, fontweight='bold', pad=12)
ax.set_xlim(1, max(len(yolo_log['epochs']), len(center_log['epochs'])))
ax.grid(True, linestyle='-', alpha=0.3, color=COLOR_GRID)
ax.legend(frameon=True, fancybox=False, edgecolor='#CCCCCC', fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig4_overfitting_gap.png'), format='png', dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUTPUT_DIR, 'fig4_overfitting_gap.svg'), format='svg', bbox_inches='tight')
plt.close(fig)
print('[OK] 图4 已保存: 过拟合对比')

# ============================================================
# 图5：训练损失收敛曲线
# ============================================================
fig, ax = plt.subplots(figsize=(11, 6.5))

ax.plot(yolo_log['epochs'], yolo_log['train_loss'], color=COLOR_YOLO, linewidth=2.0,
        marker='s', markersize=4, markevery=2, label='YOLOv8 ROI提取 (Train)', zorder=3)
ax.plot(yolo_log['epochs'], yolo_log['val_loss'], color=COLOR_YOLO, linewidth=1.5,
        linestyle='--', marker='s', markersize=3, markevery=2, label='YOLOv8 ROI提取 (Val)', alpha=0.7, zorder=2)
ax.plot(center_log['epochs'], center_log['train_loss'], color=COLOR_CENTER, linewidth=2.0,
        marker='o', markersize=4, markevery=2, label='中心裁剪 (Train)', zorder=3)
ax.plot(center_log['epochs'], center_log['val_loss'], color=COLOR_CENTER, linewidth=1.5,
        linestyle='--', marker='o', markersize=3, markevery=2, label='中心裁剪 (Val)', alpha=0.7, zorder=2)

# Stage分界线
ax.axvline(x=yolo_stage1_epochs + 0.5, color=COLOR_YOLO, linestyle='--', linewidth=1.0, alpha=0.4)
ax.axvline(x=center_stage1_epochs + 0.5, color=COLOR_CENTER, linestyle='--', linewidth=1.0, alpha=0.4)

# Stage2标注
ax.text(yolo_stage1_epochs + 1, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05, 
        'Stage 2', fontsize=8, color=COLOR_YOLO, rotation=90, va='bottom', alpha=0.7)
ax.text(center_stage1_epochs + 1, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05, 
        'Stage 2', fontsize=8, color=COLOR_CENTER, rotation=90, va='bottom', alpha=0.7)

ax.set_xlabel('训练轮次 (Epoch)', fontsize=13, fontweight='medium')
ax.set_ylabel('损失值 (Loss)', fontsize=13, fontweight='medium')
ax.set_title('训练损失收敛曲线对比', fontsize=14, fontweight='bold', pad=12)
ax.set_xlim(1, max(len(yolo_log['epochs']), len(center_log['epochs'])))
ax.grid(True, linestyle='-', alpha=0.3, color=COLOR_GRID)
ax.legend(frameon=True, fancybox=False, edgecolor='#CCCCCC', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig5_loss_curves.png'), format='png', dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUTPUT_DIR, 'fig5_loss_curves.svg'), format='svg', bbox_inches='tight')
plt.close(fig)
print('[OK] 图5 已保存: 损失曲线')

# ============================================================
# 图6：Top-5 准确率对比
# ============================================================
fig, ax = plt.subplots(figsize=(11, 6.5))

ax.plot(yolo_log['epochs'], yolo_log['val_top5'], color=COLOR_YOLO, linewidth=2.0,
        marker='s', markersize=4, markevery=2, label='YOLOv8 ROI提取', zorder=3)
ax.plot(center_log['epochs'], center_log['val_top5'], color=COLOR_CENTER, linewidth=2.0,
        marker='o', markersize=4, markevery=2, label='中心裁剪', zorder=3)

# Stage分界线
ax.axvline(x=yolo_stage1_epochs + 0.5, color=COLOR_YOLO, linestyle='--', linewidth=1.0, alpha=0.4)
ax.axvline(x=center_stage1_epochs + 0.5, color=COLOR_CENTER, linestyle='--', linewidth=1.0, alpha=0.4)

# Stage2标注
ax.text(yolo_stage1_epochs + 1, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05, 
        'Stage 2', fontsize=8, color=COLOR_YOLO, rotation=90, va='bottom', alpha=0.7)
ax.text(center_stage1_epochs + 1, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05, 
        'Stage 2', fontsize=8, color=COLOR_CENTER, rotation=90, va='bottom', alpha=0.7)

ax.set_xlabel('训练轮次 (Epoch)', fontsize=13, fontweight='medium')
ax.set_ylabel('Top-5 验证准确率 (%)', fontsize=13, fontweight='medium')
ax.set_title('Top-5 准确率收敛曲线对比', fontsize=14, fontweight='bold', pad=12)
ax.set_xlim(1, max(len(yolo_log['epochs']), len(center_log['epochs'])))
ax.set_ylim(0, 100)
ax.grid(True, linestyle='-', alpha=0.3, color=COLOR_GRID)
ax.legend(frameon=True, fancybox=False, edgecolor='#CCCCCC', fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig6_top5_curves.png'), format='png', dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUTPUT_DIR, 'fig6_top5_curves.svg'), format='svg', bbox_inches='tight')
plt.close(fig)
print('[OK] 图6 已保存: Top-5曲线')

print('\n' + '='*60)
print('所有图表已生成完成！')
print(f'输出目录: {OUTPUT_DIR}')
print('='*60)
