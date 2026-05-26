"""
实验结果分析与可视化工具
======================
读取实验运行器生成的结果数据，生成：
1. 对比表格（Markdown格式）
2. 训练曲线对比图
3. 混淆矩阵对比图
4. 综合分析报告
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional


class ExperimentAnalyzer:
    """实验结果分析器"""
    
    def __init__(self, experiments_dir: str = "experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.results_file = self.experiments_dir / "experiment_results.json"
        self.output_dir = self.experiments_dir / "analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_results(self) -> List[Dict]:
        """加载实验结果"""
        if not self.results_file.exists():
            raise FileNotFoundError(f"实验结果文件不存在: {self.results_file}")
        
        with open(self.results_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def generate_comparison_tables(self) -> str:
        """
        生成对比表格
        
        Returns:
            Markdown格式的对比表格
        """
        results = self.load_results()
        
        # 实验1: ResNet50 vs ConvNeXt-Tiny on single
        exp1_results = [r for r in results if r["exp_id"].startswith("exp1_")]
        exp1_table = self._format_model_comparison(exp1_results, "单一背景数据集")
        
        # 实验2: ResNet50 vs ConvNeXt-Tiny on complex
        exp2_results = [r for r in results if r["exp_id"].startswith("exp2_")]
        exp2_table = self._format_model_comparison(exp2_results, "复杂背景数据集")
        
        # 实验3: ROI方案对比
        exp3_results = [r for r in results if r["exp_id"].startswith("exp3_")]
        exp3_table = self._format_roi_comparison(exp3_results)
        
        # 实验4: 预训练 vs 从头训练
        exp4_results = [r for r in results if r["exp_id"].startswith("exp4_")]
        exp4_table = self._format_pretrained_comparison(exp4_results)
        
        # 实验5: 两阶段 vs 单阶段
        exp5_results = [r for r in results if r["exp_id"].startswith("exp5_")]
        exp5_table = self._format_training_strategy_comparison(exp5_results)
        
        report = f"""# 叶片识别对比实验分析报告

## 实验1: 单一背景数据集上的模型对比

{exp1_table}

## 实验2: 复杂背景数据集上的模型对比

{exp2_table}

## 实验3: ROI提取方案对比

{exp3_table}

## 实验4: 预训练权重影响

{exp4_table}

## 实验5: 训练策略对比

{exp5_table}
"""
        
        # 保存报告
        report_file = self.output_dir / "comparison_report.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)
        
        return report
    
    def _format_model_comparison(self, results: List[Dict], dataset_name: str) -> str:
        """格式化模型对比表格"""
        table = f"| 模型 | 最终验证准确率 (%) | 最佳验证准确率 (%) | 训练时间 (s) | 状态 |\n"
        table += f"|------|-------------------|-------------------|-------------|------|\n"
        
        for r in results:
            model = r["exp_id"].split("_")[1]
            metrics = r.get("metrics", {})
            stage2 = metrics.get("stage2", {})
            best_acc = stage2.get("best_val_acc", "N/A")
            final_acc = stage2.get("final_val_acc", "N/A")
            time = f"{r.get('elapsed_time', 0):.1f}" if r.get("status") == "success" else "N/A"
            status = r.get("status", "unknown")
            
            table += f"| {model} | {final_acc:.2f} | {best_acc:.2f} | {time} | {status} |\n"
        
        return table
    
    def _format_roi_comparison(self, results: List[Dict]) -> str:
        """格式化ROI方案对比表格"""
        table = "| ROI方案 | 最终验证准确率 (%) | 最佳验证准确率 (%) | 训练时间 (s) | 状态 |\n"
        table += "|---------|-------------------|-------------------|-------------|------|\n"
        
        for r in results:
            roi_type = "center" if "center" in r["exp_id"] else "yolo"
            roi_name = "中心裁剪" if roi_type == "center" else "YOLOv8"
            metrics = r.get("metrics", {})
            stage2 = metrics.get("stage2", {})
            best_acc = stage2.get("best_val_acc", "N/A")
            final_acc = stage2.get("final_val_acc", "N/A")
            time = f"{r.get('elapsed_time', 0):.1f}" if r.get("status") == "success" else "N/A"
            status = r.get("status", "unknown")
            
            table += f"| {roi_name} | {final_acc:.2f} | {best_acc:.2f} | {time} | {status} |\n"
        
        return table
    
    def _format_pretrained_comparison(self, results: List[Dict]) -> str:
        """格式化预训练对比表格"""
        table = "| 训练方式 | 最终验证准确率 (%) | 最佳验证准确率 (%) | 收敛轮数 | 训练时间 (s) | 状态 |\n"
        table += "|---------|-------------------|-------------------|---------|-------------|------|\n"
        
        for r in results:
            train_type = "pretrained" if "pretrained" in r["exp_id"] else "scratch"
            type_name = "预训练+微调" if train_type == "pretrained" else "从头训练"
            metrics = r.get("metrics", {})
            stage2 = metrics.get("stage2", {})
            best_acc = stage2.get("best_val_acc", "N/A")
            final_acc = stage2.get("final_val_acc", "N/A")
            time = f"{r.get('elapsed_time', 0):.1f}" if r.get("status") == "success" else "N/A"
            status = r.get("status", "unknown")
            
            table += f"| {type_name} | {final_acc:.2f} | {best_acc:.2f} | 待统计 | {time} | {status} |\n"
        
        return table
    
    def _format_training_strategy_comparison(self, results: List[Dict]) -> str:
        """格式化训练策略对比表格"""
        table = "| 训练策略 | Stage1准确率 (%) | Stage2准确率 (%) | 最终验证准确率 (%) | 训练时间 (s) | 状态 |\n"
        table += "|---------|-----------------|-----------------|-------------------|-------------|------|\n"
        
        for r in results:
            strategy = "twostage" if "twostage" in r["exp_id"] else "singlestage"
            strategy_name = "两阶段训练" if strategy == "twostage" else "单阶段训练"
            metrics = r.get("metrics", {})
            stage1 = metrics.get("stage1", {})
            stage2 = metrics.get("stage2", {})
            s1_acc = stage1.get("best_val_acc", "N/A")
            s2_acc = stage2.get("best_val_acc", "N/A")
            time = f"{r.get('elapsed_time', 0):.1f}" if r.get("status") == "success" else "N/A"
            status = r.get("status", "unknown")
            
            table += f"| {strategy_name} | {s1_acc:.2f} | {s2_acc:.2f} | {s2_acc:.2f} | {time} | {status} |\n"
        
        return table
    
    def plot_training_curves_comparison(self, exp_group: str):
        """
        绘制训练曲线对比图
        
        Args:
            exp_group: 实验组ID（如exp1, exp2等）
        """
        results = self.load_results()
        exp_results = [r for r in results if r["exp_id"].startswith(exp_group)]
        
        if len(exp_results) == 0:
            print(f"未找到实验组 {exp_group} 的结果")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        
        for idx, result in enumerate(exp_results):
            exp_id = result["exp_id"]
            curve_dir = self.experiments_dir / exp_id / "curves"
            
            if not curve_dir.exists():
                continue
            
            # 读取所有stage的CSV
            all_epochs = []
            all_train_loss = []
            all_val_loss = []
            all_train_acc = []
            all_val_acc = []
            epoch_offset = 0
            
            for stage_csv in sorted(curve_dir.glob("stage*.csv")):
                try:
                    df = pd.read_csv(stage_csv)
                    if df.empty:
                        continue
                    
                    df["global_epoch"] = df["epoch"] + epoch_offset
                    all_epochs.extend(df["global_epoch"].tolist())
                    all_train_loss.extend(df["train_loss"].tolist())
                    all_val_loss.extend(df["val_loss"].tolist())
                    all_train_acc.extend(df["train_acc"].tolist())
                    all_val_acc.extend(df["val_acc"].tolist())
                    epoch_offset = int(df["global_epoch"].max())
                except Exception as e:
                    print(f"读取 {stage_csv} 失败: {e}")
            
            if not all_epochs:
                continue
            
            color = colors[idx % len(colors)]
            label = exp_id.replace(f"{exp_group}_", "")
            
            axes[0, 0].plot(all_epochs, all_train_loss, label=f"{label} (train)", color=color, linestyle="-")
            axes[0, 0].plot(all_epochs, all_val_loss, label=f"{label} (val)", color=color, linestyle="--")
            axes[0, 1].plot(all_epochs, all_train_acc, label=f"{label} (train)", color=color, linestyle="-")
            axes[0, 1].plot(all_epochs, all_val_acc, label=f"{label} (val)", color=color, linestyle="--")
        
        # 设置子图标题和标签
        axes[0, 0].set_title("Loss Comparison")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend(frameon=False, fontsize=8)
        axes[0, 0].grid(alpha=0.3)
        
        axes[0, 1].set_title("Accuracy Comparison")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy (%)")
        axes[0, 1].legend(frameon=False, fontsize=8)
        axes[0, 1].grid(alpha=0.3)
        
        # 隐藏空子图
        axes[1, 0].axis("off")
        axes[1, 1].axis("off")
        
        plt.tight_layout()
        
        # 保存图片
        output_file = self.output_dir / f"{exp_group}_training_curves.png"
        plt.savefig(output_file, bbox_inches="tight")
        plt.close()
        
        print(f"训练曲线对比图已保存至: {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="实验结果分析工具")
    parser.add_argument(
        "--experiments_dir",
        type=str,
        default="experiments",
        help="实验结果目录（默认: experiments）"
    )
    parser.add_argument(
        "--action",
        type=str,
        default="all",
        choices=["all", "report", "plot"],
        help="执行的操作（默认: all）"
    )
    parser.add_argument(
        "--exp_group",
        type=str,
        help="要绘制曲线对比的实验组ID（如exp1）"
    )
    
    args = parser.parse_args()
    
    analyzer = ExperimentAnalyzer(experiments_dir=args.experiments_dir)
    
    if args.action in ["all", "report"]:
        report = analyzer.generate_comparison_tables()
        print("\n" + report)
    
    if args.action in ["all", "plot"]:
        if args.exp_group:
            analyzer.plot_training_curves_comparison(args.exp_group)
        else:
            # 默认绘制所有实验组的对比图
            for exp_group in ["exp1", "exp2", "exp3", "exp4", "exp5"]:
                analyzer.plot_training_curves_comparison(exp_group)


if __name__ == "__main__":
    main()
