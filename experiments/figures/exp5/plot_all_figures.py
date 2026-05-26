import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import os
import re
import json

# ============================================================
# 全局配置：中文字体 + 学术风格
# ============================================================
plt.rcParams['font.family'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.labelcolor'] = '#333333'
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'
plt.rcParams['text.color'] = '#333333'

OUTPUT_DIR = r'c:\Users\86185\Desktop\autoleaf\experiments\figures\exp5'
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

    pattern = re.compile(
        r'Time\s+[\d.]+s\s*\|\s*TrainLoss\s+([\d.]+)\s*\|\s*TrainAcc\s+([\d.]+)%\s*\|\s*ValLoss\s+([\d.]+)\s*\|\s*ValAcc\s+([\d.]+)%\s*\|\s*ValTop5\s+([\d.]+)%'
    )

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

    return {
        'epochs': epochs,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'val_top5': val_top5,
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
EXP1_TWOSTAGE_DIR = r'c:\Users\86185\Desktop\autoleaf\experiments\exp1_resnet50_single\20260422_075325\20260422_075325'
EXP5_SINGLESTAGE_DIR = r'c:\Users\86185\Desktop\autoleaf\experiments\exp5_resnet50_singlestage\20260422_091047\20260422_091047'

# 解析两阶段训练数据 (exp1)
twostage_log = parse_training_log(os.path.join(EXP1_TWOSTAGE_DIR, 'training.log'))
twostage_metrics = parse_thesis_metrics(os.path.join(EXP1_TWOSTAGE_DIR, 'thesis_metrics.json'))

# 解析单阶段训练数据 (exp5)
singlestage_log = parse_training_log(os.path.join(EXP5_SINGLESTAGE_DIR, 'training.log'))
singlestage_metrics = parse_thesis_metrics(os.path.join(EXP5_SINGLESTAGE_DIR, 'thesis_metrics.json'))

# 统一配色
COLOR_TWOSTAGE = '#4E79A7'   # 蓝色：两阶段训练
COLOR_SINGLESTAGE = '#59A14F'  # 绿色：单阶段训练
COLOR_GRID = '#E0E0E0'
COLOR_TEXT = '#333333'

print(f"两阶段训练: {len(twostage_log['epochs'])} epochs, Top-1={twostage_metrics['Top-1 Acc (%)']}%")
print(f"单阶段训练: {len(singlestage_log['epochs'])} epochs, Top-1={singlestage_metrics['Top-1 Acc (%)']}%")

# 获取Stage分界点（从日志中推断）
twostage_stage1_epochs = 10

# ============================================================
# 图1：训练曲线对比 (验证准确率 + 损失) — 带阶段分隔
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

max_epochs = max(len(twostage_log['epochs']), len(singlestage_log['epochs']))

ax1 = axes[0]
ax1.plot(twostage_log['epochs'], twostage_log['val_acc'], color=COLOR_TWOSTAGE, linewidth=1.5,
         marker='s', markersize=3, markevery=2, label='两阶段训练')
ax1.plot(singlestage_log['epochs'], singlestage_log['val_acc'], color=COLOR_SINGLESTAGE, linewidth=1.5,
         marker='o', markersize=3, markevery=2, label='单阶段训练')
ax1.axvline(x=twostage_stage1_epochs + 0.5, color=COLOR_TWOSTAGE, linestyle='--', linewidth=0.8, alpha=0.5)
ax1.text(twostage_stage1_epochs + 0.7, 3, 'Stage 2', fontsize=8, color=COLOR_TWOSTAGE, rotation=90, va='bottom')
ax1.set_xlabel('训练轮次 (Epoch)', fontsize=12)
ax1.set_ylabel('验证准确率 (%)', fontsize=12)
ax1.set_title('(a) 验证准确率收敛曲线', fontsize=13, fontweight='bold')
ax1.set_xlim(1, max_epochs)
ax1.set_ylim(0, 100)
ax1.grid(True, linestyle='-', alpha=0.4, color=COLOR_GRID)
ax1.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='#CCCCCC')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax2 = axes[1]
ax2.plot(twostage_log['epochs'], twostage_log['val_loss'], color=COLOR_TWOSTAGE, linewidth=1.5,
         marker='s', markersize=3, markevery=2, label='两阶段训练')
ax2.plot(singlestage_log['epochs'], singlestage_log['val_loss'], color=COLOR_SINGLESTAGE, linewidth=1.5,
         marker='o', markersize=3, markevery=2, label='单阶段训练')
ax2.axvline(x=twostage_stage1_epochs + 0.5, color=COLOR_TWOSTAGE, linestyle='--', linewidth=0.8, alpha=0.5)
ax2.text(twostage_stage1_epochs + 0.7, 0.2, 'Stage 2', fontsize=8, color=COLOR_TWOSTAGE, rotation=90, va='bottom')
ax2.set_xlabel('训练轮次 (Epoch)', fontsize=12)
ax2.set_ylabel('验证损失 (Validation Loss)', fontsize=12)
ax2.set_title('(b) 验证损失收敛曲线', fontsize=13, fontweight='bold')
ax2.set_xlim(1, max_epochs)
ax2.set_ylim(0, max(max(twostage_log['val_loss']), max(singlestage_log['val_loss'])) * 1.1)
ax2.grid(True, linestyle='-', alpha=0.4, color=COLOR_GRID)
ax2.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='#CCCCCC')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig1_training_curves_comparison.png'), format='png', dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUTPUT_DIR, 'fig1_training_curves_comparison.svg'), format='svg', bbox_inches='tight')
plt.close(fig)
print('图1 已保存')

# ============================================================
# 图2：局部放大折线图 (收敛阶段细节)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6.5))

ax.plot(twostage_log['epochs'], twostage_log['val_acc'], color=COLOR_TWOSTAGE, linewidth=1.5,
        marker='s', markersize=3, markevery=1, label='两阶段训练')
ax.plot(singlestage_log['epochs'], singlestage_log['val_acc'], color=COLOR_SINGLESTAGE, linewidth=1.5,
        marker='o', markersize=3, markevery=1, label='单阶段训练')
ax.axvline(x=twostage_stage1_epochs + 0.5, color=COLOR_TWOSTAGE, linestyle='--', linewidth=0.8, alpha=0.5)

# 使用 inset_axes 创建子图，放大收敛阶段
ax_inset = inset_axes(ax, width="45%", height="45%", loc='lower right',
                       bbox_to_anchor=(0.12, 0.12, 0.85, 0.85),
                       bbox_transform=ax.transAxes)

# 动态确定放大范围
zoom_start = twostage_stage1_epochs + 5
zoom_start_idx_two = min(zoom_start - 1, len(twostage_log['epochs']) - 1)
zoom_start_idx_single = min(zoom_start - 1, len(singlestage_log['epochs']) - 1)

ax_inset.plot(twostage_log['epochs'][zoom_start_idx_two:], twostage_log['val_acc'][zoom_start_idx_two:],
              color=COLOR_TWOSTAGE, linewidth=2.0, marker='s', markersize=4, markevery=1)
ax_inset.plot(singlestage_log['epochs'][zoom_start_idx_single:], singlestage_log['val_acc'][zoom_start_idx_single:],
              color=COLOR_SINGLESTAGE, linewidth=2.0, marker='o', markersize=4, markevery=1)

# 动态标注最终值
twostage_final = twostage_metrics['Top-1 Acc (%)']
singlestage_final = singlestage_metrics['Top-1 Acc (%)']
ax_inset.axhline(y=twostage_final, color=COLOR_TWOSTAGE, linestyle=':', linewidth=1.0, alpha=0.7)
ax_inset.axhline(y=singlestage_final, color=COLOR_SINGLESTAGE, linestyle=':', linewidth=1.0, alpha=0.7)
ax_inset.text(max_epochs - 1, twostage_final + 1, f'{twostage_final:.2f}%', fontsize=9, color=COLOR_TWOSTAGE, ha='right')
ax_inset.text(max_epochs - 1, singlestage_final + 1, f'{singlestage_final:.2f}%', fontsize=9, color=COLOR_SINGLESTAGE, ha='right')

ax_inset.set_xlim(zoom_start, max_epochs)
ax_inset.set_ylim(min(twostage_final, singlestage_final) - 5, max(twostage_final, singlestage_final) + 5)
ax_inset.set_xlabel('Epoch', fontsize=9)
ax_inset.set_ylabel('Val Acc (%)', fontsize=9)
ax_inset.set_title('收敛阶段放大', fontsize=10, fontweight='bold', pad=5)
ax_inset.grid(True, linestyle='-', alpha=0.3, color=COLOR_GRID)
ax_inset.spines['top'].set_visible(False)
ax_inset.spines['right'].set_visible(False)
for spine in ax_inset.spines.values():
    spine.set_color('#AAAAAA')
    spine.set_linewidth(1.2)

ax.set_xlabel('训练轮次 (Epoch)', fontsize=12)
ax.set_ylabel('验证准确率 (%)', fontsize=12)
ax.set_title('验证准确率收敛曲线与局部放大', fontsize=13, fontweight='bold')
ax.set_xlim(1, max_epochs)
ax.set_ylim(0, 100)
ax.grid(True, linestyle='-', alpha=0.4, color=COLOR_GRID)
ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='#CCCCCC')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig2_zoomed_convergence.png'), format='png', dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUTPUT_DIR, 'fig2_zoomed_convergence.svg'), format='svg', bbox_inches='tight')
plt.close(fig)
print('图2 已保存')

# ============================================================
# 图3：纵向分组柱状图 (综合性能指标对比)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# 左图：精度指标
ax3 = axes[0]
categories_acc = ['Top-1\n准确率 (%)', 'Top-5\n准确率 (%)', 'Macro-F1\n(×100)']
twostage_acc_vals = [twostage_metrics['Top-1 Acc (%)'], twostage_metrics['Top-5 Acc (%)'], twostage_metrics['Macro-F1'] * 100]
singlestage_acc_vals = [singlestage_metrics['Top-1 Acc (%)'], singlestage_metrics['Top-5 Acc (%)'], singlestage_metrics['Macro-F1'] * 100]

x = np.arange(len(categories_acc))
width = 0.35
bars1 = ax3.bar(x - width/2, twostage_acc_vals, width, label='两阶段训练', color=COLOR_TWOSTAGE, edgecolor='white', linewidth=0.5)
bars2 = ax3.bar(x + width/2, singlestage_acc_vals, width, label='单阶段训练', color=COLOR_SINGLESTAGE, edgecolor='white', linewidth=0.5)

ax3.set_ylabel('准确率 / F1 (%)', fontsize=12)
ax3.set_title('(a) 分类精度对比', fontsize=13, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(categories_acc, fontsize=10)
ax3.set_ylim(0, 105)
ax3.legend(frameon=True, fancybox=False, edgecolor='#CCCCCC')
ax3.grid(True, axis='y', linestyle='-', alpha=0.4, color=COLOR_GRID)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

for bar in bars1:
    height = bar.get_height()
    ax3.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=10, color=COLOR_TEXT, fontweight='bold')
for bar in bars2:
    height = bar.get_height()
    ax3.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=10, color=COLOR_TEXT, fontweight='bold')

# 右图：效率与收敛指标 - 使用对数坐标
ax4 = axes[1]
categories_eff = ['训练时间\n(s)', '收敛轮数', '推理速度\n(ms)']
twostage_eff_vals = [twostage_metrics['训练时间 (s)'], twostage_metrics['收敛轮数'], twostage_metrics['推理速度 (ms)']]
singlestage_eff_vals = [singlestage_metrics['训练时间 (s)'], singlestage_metrics['收敛轮数'], singlestage_metrics['推理速度 (ms)']]

x2 = np.arange(len(categories_eff))
bars3 = ax4.bar(x2 - width/2, twostage_eff_vals, width, label='两阶段训练', color=COLOR_TWOSTAGE, edgecolor='white', linewidth=0.5)
bars4 = ax4.bar(x2 + width/2, singlestage_eff_vals, width, label='单阶段训练', color=COLOR_SINGLESTAGE, edgecolor='white', linewidth=0.5)

ax4.set_ylabel('数值 (对数坐标)', fontsize=12)
ax4.set_title('(b) 训练效率对比', fontsize=13, fontweight='bold')
ax4.set_xticks(x2)
ax4.set_xticklabels(categories_eff, fontsize=10)
ax4.set_yscale('log')
ax4.set_ylim(1, max(max(twostage_eff_vals), max(singlestage_eff_vals)) * 1.5)
ax4.legend(frameon=True, fancybox=False, edgecolor='#CCCCCC')
ax4.grid(True, axis='y', linestyle='-', alpha=0.4, color=COLOR_GRID)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

for bar in bars3:
    height = bar.get_height()
    ax4.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9, color=COLOR_TEXT)
for bar in bars4:
    height = bar.get_height()
    ax4.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9, color=COLOR_TEXT)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig3_performance_bars.png'), format='png', dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUTPUT_DIR, 'fig3_performance_bars.svg'), format='svg', bbox_inches='tight')
plt.close(fig)
print('图3 已保存')

# ============================================================
# 图4：雷达图 (多维度综合能力评估)
# ============================================================
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, polar=True)

categories_radar = ['Top-1 准确率', 'Top-5 准确率', 'Macro-F1', '收敛速度', '训练效率']
N = len(categories_radar)

# 构建原始值矩阵 (行=模型, 列=维度)
# 维度: [Top-1, Top-5, Macro-F1, 1/收敛epoch, 1/训练时间]
raw_matrix = np.array([
    [twostage_metrics['Top-1 Acc (%)'], twostage_metrics['Top-5 Acc (%)'],
     twostage_metrics['Macro-F1'], 1/twostage_metrics['收敛轮数'], 1/twostage_metrics['训练时间 (s)']],
    [singlestage_metrics['Top-1 Acc (%)'], singlestage_metrics['Top-5 Acc (%)'],
     singlestage_metrics['Macro-F1'], 1/singlestage_metrics['收敛轮数'], 1/singlestage_metrics['训练时间 (s)']],
])

# 对每个维度做 min-max 归一化
norm_matrix = np.zeros_like(raw_matrix)
for j in range(raw_matrix.shape[1]):
    col = raw_matrix[:, j]
    min_v, max_v = col.min(), col.max()
    if max_v > min_v:
        norm_matrix[:, j] = (col - min_v) / (max_v - min_v)
    else:
        norm_matrix[:, j] = 0.5

twostage_norm = norm_matrix[0].tolist() + [norm_matrix[0][0]]
singlestage_norm = norm_matrix[1].tolist() + [norm_matrix[1][0]]

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

ax.plot(angles, twostage_norm, color=COLOR_TWOSTAGE, linewidth=2.5, linestyle='solid', label='两阶段训练')
ax.fill(angles, twostage_norm, color=COLOR_TWOSTAGE, alpha=0.20)
ax.plot(angles, singlestage_norm, color=COLOR_SINGLESTAGE, linewidth=2.5, linestyle='solid', label='单阶段训练')
ax.fill(angles, singlestage_norm, color=COLOR_SINGLESTAGE, alpha=0.20)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories_radar, fontsize=12, color=COLOR_TEXT)
ax.set_ylim(0, 1.15)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9, color='#888888')
ax.grid(True, linestyle='-', alpha=0.4, color=COLOR_GRID)

# 在雷达图外侧添加原始值标注
label_data = [
    (f"{twostage_metrics['Top-1 Acc (%)']:.2f}%", f"{singlestage_metrics['Top-1 Acc (%)']:.2f}%"),
    (f"{twostage_metrics['Top-5 Acc (%)']:.2f}%", f"{singlestage_metrics['Top-5 Acc (%)']:.2f}%"),
    (f"{twostage_metrics['Macro-F1']:.3f}", f"{singlestage_metrics['Macro-F1']:.3f}"),
    (f"{twostage_metrics['收敛轮数']} ep", f"{singlestage_metrics['收敛轮数']} ep"),
    (f"{twostage_metrics['训练时间 (s)']:.0f}s", f"{singlestage_metrics['训练时间 (s)']:.0f}s"),
]

for i, (t_val, s_val) in enumerate(label_data):
    angle = angles[i]
    ax.text(angle, 1.08, f'T: {t_val}\nS: {s_val}',
           ha='center', va='center', fontsize=9, color=COLOR_TEXT,
           transform=ax.transData)

ax.set_title('多维度综合能力评估', fontsize=15, fontweight='bold', y=1.08, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), frameon=True, fancybox=False, edgecolor='#CCCCCC', fontsize=11)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig4_radar_comparison.png'), format='png', dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUTPUT_DIR, 'fig4_radar_comparison.svg'), format='svg', bbox_inches='tight')
plt.close(fig)
print('图4 已保存')

# ============================================================
# 图5：训练/验证准确率差距对比 (过拟合分析)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

# 两阶段训练的 train-val gap
twostage_gap = [t - v for t, v in zip(twostage_log['train_acc'], twostage_log['val_acc'])]
singlestage_gap = [t - v for t, v in zip(singlestage_log['train_acc'], singlestage_log['val_acc'])]

ax.plot(twostage_log['epochs'], twostage_gap, color=COLOR_TWOSTAGE, linewidth=1.5,
        marker='s', markersize=3, markevery=2, label='两阶段训练')
ax.plot(singlestage_log['epochs'], singlestage_gap, color=COLOR_SINGLESTAGE, linewidth=1.5,
        marker='o', markersize=3, markevery=2, label='单阶段训练')

ax.axhline(y=0, color='#888888', linestyle='-', linewidth=0.5)
ax.set_xlabel('训练轮次 (Epoch)', fontsize=12)
ax.set_ylabel('Train-Val 准确率差距 (%)', fontsize=12)
ax.set_title('过拟合程度对比 (Train-Val Gap)', fontsize=13, fontweight='bold')
ax.set_xlim(1, max_epochs)
ax.grid(True, linestyle='-', alpha=0.4, color=COLOR_GRID)
ax.legend(frameon=True, fancybox=False, edgecolor='#CCCCCC')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig5_overfitting_gap.png'), format='png', dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUTPUT_DIR, 'fig5_overfitting_gap.svg'), format='svg', bbox_inches='tight')
plt.close(fig)
print('图5 已保存')

# ============================================================
# 图6：Top-5 准确率曲线对比
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(twostage_log['epochs'], twostage_log['val_top5'], color=COLOR_TWOSTAGE, linewidth=1.5,
        marker='s', markersize=3, markevery=2, label='两阶段训练')
ax.plot(singlestage_log['epochs'], singlestage_log['val_top5'], color=COLOR_SINGLESTAGE, linewidth=1.5,
        marker='o', markersize=3, markevery=2, label='单阶段训练')

ax.axvline(x=twostage_stage1_epochs + 0.5, color=COLOR_TWOSTAGE, linestyle='--', linewidth=0.8, alpha=0.5)
ax.text(twostage_stage1_epochs + 0.7, 5, 'Stage 2', fontsize=8, color=COLOR_TWOSTAGE, rotation=90, va='bottom')

ax.set_xlabel('训练轮次 (Epoch)', fontsize=12)
ax.set_ylabel('Top-5 验证准确率 (%)', fontsize=12)
ax.set_title('Top-5 准确率收敛曲线对比', fontsize=13, fontweight='bold')
ax.set_xlim(1, max_epochs)
ax.set_ylim(0, 100)
ax.grid(True, linestyle='-', alpha=0.4, color=COLOR_GRID)
ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='#CCCCCC')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig6_top5_accuracy.png'), format='png', dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUTPUT_DIR, 'fig6_top5_accuracy.svg'), format='svg', bbox_inches='tight')
plt.close(fig)
print('图6 已保存')

# ============================================================
# 图7：训练损失曲线对比
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(twostage_log['epochs'], twostage_log['train_loss'], color=COLOR_TWOSTAGE, linewidth=1.5,
        marker='s', markersize=3, markevery=2, label='两阶段训练')
ax.plot(singlestage_log['epochs'], singlestage_log['train_loss'], color=COLOR_SINGLESTAGE, linewidth=1.5,
        marker='o', markersize=3, markevery=2, label='单阶段训练')

ax.axvline(x=twostage_stage1_epochs + 0.5, color=COLOR_TWOSTAGE, linestyle='--', linewidth=0.8, alpha=0.5)
ax.text(twostage_stage1_epochs + 0.7, 0.5, 'Stage 2', fontsize=8, color=COLOR_TWOSTAGE, rotation=90, va='bottom')

ax.set_xlabel('训练轮次 (Epoch)', fontsize=12)
ax.set_ylabel('训练损失 (Train Loss)', fontsize=12)
ax.set_title('训练损失收敛曲线对比', fontsize=13, fontweight='bold')
ax.set_xlim(1, max_epochs)
ax.set_ylim(0, max(max(twostage_log['train_loss']), max(singlestage_log['train_loss'])) * 1.1)
ax.grid(True, linestyle='-', alpha=0.4, color=COLOR_GRID)
ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='#CCCCCC')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig7_train_loss.png'), format='png', dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUTPUT_DIR, 'fig7_train_loss.svg'), format='svg', bbox_inches='tight')
plt.close(fig)
print('图7 已保存')

# ============================================================
# 图8：稳定性指标对比 (柱状图)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5.5))

stability_categories = ['整体方差', '局部方差', '最大波动(%)', '后期方差']
twostage_stab = twostage_metrics.get('stability', {})
singlestage_stab = singlestage_metrics.get('stability', {})

twostage_stab_vals = [
    twostage_stab.get('overall_variance', 0),
    twostage_stab.get('local_variance', 0),
    twostage_stab.get('max_fluctuation', 0),
    twostage_stab.get('late_stage_variance', 0),
]
singlestage_stab_vals = [
    singlestage_stab.get('overall_variance', 0),
    singlestage_stab.get('local_variance', 0),
    singlestage_stab.get('max_fluctuation', 0),
    singlestage_stab.get('late_stage_variance', 0),
]

x = np.arange(len(stability_categories))
bars1 = ax.bar(x - width/2, twostage_stab_vals, width, label='两阶段训练', color=COLOR_TWOSTAGE, edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x + width/2, singlestage_stab_vals, width, label='单阶段训练', color=COLOR_SINGLESTAGE, edgecolor='white', linewidth=0.5)

ax.set_ylabel('稳定性指标值', fontsize=12)
ax.set_title('训练稳定性对比', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(stability_categories, fontsize=10)
ax.set_yscale('log')
ax.set_ylim(0.1, max(max(twostage_stab_vals), max(singlestage_stab_vals)) * 2)
ax.legend(frameon=True, fancybox=False, edgecolor='#CCCCCC')
ax.grid(True, axis='y', linestyle='-', alpha=0.4, color=COLOR_GRID)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for bar in bars1:
    height = bar.get_height()
    if height > 0:
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, color=COLOR_TEXT)
for bar in bars2:
    height = bar.get_height()
    if height > 0:
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, color=COLOR_TEXT)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig8_stability_comparison.png'), format='png', dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUTPUT_DIR, 'fig8_stability_comparison.svg'), format='svg', bbox_inches='tight')
plt.close(fig)
print('图8 已保存')

print('\n所有图表已保存到:', OUTPUT_DIR)
