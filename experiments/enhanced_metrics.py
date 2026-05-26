"""指标计算工具"""

import json
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score


class MetricsCalculator:
    """实验指标计算器"""
    
    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def calculate_topk_accuracy(self, dataloader, k: int = 5) -> Dict[str, float]:
        """计算Top-1和Top-K准确率"""
        correct_top1 = 0
        correct_topk = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            
            # Top-1准确率
            _, predicted_top1 = outputs.max(1)
            correct_top1 += predicted_top1.eq(labels).sum().item()
            
            # Top-K准确率
            _, predicted_topk = outputs.topk(k, 1, largest=True, sorted=True)
            correct_topk += predicted_topk.eq(labels.view(-1, 1).expand_as(predicted_topk)).sum().item()
            
            total += labels.size(0)
            
            all_preds.extend(predicted_top1.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        return {
            "top1_acc": correct_top1 / total * 100,
            f"top{k}_acc": correct_topk / total * 100,
            "all_preds": all_preds,
            "all_labels": all_labels
        }
    
    @torch.no_grad()
    def measure_inference_speed(
        self,
        input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        """测量模型推理速度"""
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # 预热
        for _ in range(warmup_runs):
            _ = self.model(dummy_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 测量推理时间
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = self.model(dummy_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)
        
        # 计算统计信息
        avg_time = np.mean(times) * 1000  # 转换为毫秒
        std_time = np.std(times) * 1000
        
        return {
            "avg_inference_time_ms": avg_time,
            "std_inference_time_ms": std_time,
            "fps": 1000 / avg_time,
            "batch_size": input_shape[0]
        }
    
    def count_parameters(self) -> Dict[str, float]:
        """统计模型参数量"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "total_params_m": total_params / 1e6,
            "trainable_params_m": trainable_params / 1e6
        }
    
    def calculate_flops(self, input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224)) -> float:
        """计算FLOPs（需要安装thop库）"""
        try:
            from thop import profile
            dummy_input = torch.randn(input_shape).to(self.device)
            flops, params = profile(self.model, inputs=(dummy_input,))
            return flops / 1e9  # 转换为G
        except ImportError:
            print("警告: thop库未安装，跳过FLOPs计算")
            print("安装命令: pip install thop")
            return 0.0
    
    def calculate_confusion_matrix_metrics(
        self, all_preds: List[int], all_labels: List[int],
        class_names: Optional[List[str]] = None
    ) -> Dict:
        """计算混淆矩阵及相关指标"""
        # 计算混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        
        # 计算精确率、召回率、F1分数
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None
        )
        
        # 宏平均和微平均
        precision_macro = precision_recall_fscore_support(
            all_labels, all_preds, average='macro'
        )
        precision_micro = precision_recall_fscore_support(
            all_labels, all_preds, average='micro'
        )
        
        results = {
            "confusion_matrix": cm.tolist(),
            "per_class_precision": precision.tolist(),
            "per_class_recall": recall.tolist(),
            "per_class_f1": f1.tolist(),
            "per_class_support": support.tolist(),
            "macro_precision": precision_macro[0],
            "macro_recall": precision_macro[1],
            "macro_f1": precision_macro[2],
            "micro_precision": precision_micro[0],
            "micro_recall": precision_micro[1],
            "micro_f1": precision_micro[2]
        }
        
        if class_names:
            results["class_names"] = class_names
        
        return results
    
    def calculate_convergence_epoch(self, val_acc_curve: List[float], threshold: float = 0.95) -> int:
        """计算收敛轮数（达到最佳准确率95%的轮数）"""
        best_acc = max(val_acc_curve)
        target_acc = best_acc * threshold
        
        for epoch, acc in enumerate(val_acc_curve):
            if acc >= target_acc:
                return epoch + 1
        
        return len(val_acc_curve)
    
    def calculate_training_stability(self, val_acc_curve: List[float], window_size: int = 5) -> Dict[str, float]:
        """计算训练稳定性指标"""
        acc_array = np.array(val_acc_curve)
        
        # 整体方差
        overall_variance = np.var(acc_array)
        
        # 滑动窗口方差（衡量局部波动）
        if len(acc_array) >= window_size:
            window_vars = []
            for i in range(len(acc_array) - window_size + 1):
                window = acc_array[i:i+window_size]
                window_vars.append(np.var(window))
            local_variance = np.mean(window_vars)
        else:
            local_variance = overall_variance
        
        # 最大单轮波动
        max_fluctuation = np.max(np.abs(np.diff(acc_array)))
        
        # 后期稳定性（最后10轮）
        late_stage_var = np.var(acc_array[-10:]) if len(acc_array) >= 10 else overall_variance
        
        return {
            "overall_variance": overall_variance,
            "local_variance": local_variance,
            "max_fluctuation": max_fluctuation,
            "late_stage_variance": late_stage_var
        }


def calculate_roi_extraction_success_rate(
    yolo_predictions: List[Dict], ground_truth_boxes: List[Dict],
    iou_threshold: float = 0.5
) -> float:
    """计算ROI提取成功率"""
    def calculate_iou(box1: List[float], box2: List[float]) -> float:
        """计算两个边界框的IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    success_count = 0
    total = len(ground_truth_boxes)
    
    for gt_box in ground_truth_boxes:
        # 找到IoU最大的预测框
        max_iou = 0
        for pred in yolo_predictions:
            iou = calculate_iou(gt_box['bbox'], pred['bbox'])
            max_iou = max(max_iou, iou)
        
        if max_iou >= iou_threshold:
            success_count += 1
    
    return (success_count / total * 100) if total > 0 else 0.0


def generate_thesis_tables(experiment_results: Dict, output_file: str = "thesis_tables.md") -> str:
    """生成可直接用于论文的表格"""
    tables = "# 论文实验结果表格\n\n"
    
    # 表4.1 单一背景数据集模型对比
    if "exp1" in experiment_results:
        exp1 = experiment_results["exp1"]
        tables += "## 表4.1 单一背景数据集上的模型对比\n\n"
        tables += "| 模型 | Top-1 Acc (%) | Top-5 Acc (%) | 参数量 (M) | FLOPs (G) | 训练时间 (h) | 推理速度 (ms) |\n"
        tables += "|------|---------------|---------------|-----------|-----------|-------------|--------------|\n"
        
        for model_name, metrics in exp1.items():
            tables += f"| {model_name} "
            tables += f"| {metrics.get('top1_acc', '待补充'):.2f} "
            tables += f"| {metrics.get('top5_acc', '待补充'):.2f} "
            tables += f"| {metrics.get('params_m', '待补充'):.2f} "
            tables += f"| {metrics.get('flops_g', '待补充'):.2f} "
            tables += f"| {metrics.get('training_time_h', '待补充'):.2f} "
            tables += f"| {metrics.get('inference_time_ms', '待补充'):.2f} |\n"
        
        tables += "\n"
    
    # 表4.2 复杂背景ROI方案对比
    if "exp3" in experiment_results:
        exp3 = experiment_results["exp3"]
        tables += "## 表4.2 复杂背景数据集上的ROI方案对比\n\n"
        tables += "| ROI方案 | Top-1 Acc (%) | Top-5 Acc (%) | ROI提取成功率 (%) | 推理速度 (ms) |\n"
        tables += "|---------|---------------|---------------|------------------|--------------|\n"
        
        for roi_name, metrics in exp3.items():
            tables += f"| {roi_name} "
            tables += f"| {metrics.get('top1_acc', '待补充'):.2f} "
            tables += f"| {metrics.get('top5_acc', '待补充'):.2f} "
            tables += f"| {metrics.get('roi_success_rate', '待补充'):.2f} "
            tables += f"| {metrics.get('inference_time_ms', '待补充'):.2f} |\n"
        
        tables += "\n"
    
    # 表4.3 预训练权重影响
    if "exp4" in experiment_results:
        exp4 = experiment_results["exp4"]
        tables += "## 表4.3 预训练权重对模型性能的影响\n\n"
        tables += "| 训练方式 | Top-1 Acc (%) | Top-5 Acc (%) | 收敛轮数 | 最终Loss |\n"
        tables += "|---------|---------------|---------------|---------|---------|\n"
        
        for train_type, metrics in exp4.items():
            tables += f"| {train_type} "
            tables += f"| {metrics.get('top1_acc', '待补充'):.2f} "
            tables += f"| {metrics.get('top5_acc', '待补充'):.2f} "
            tables += f"| {metrics.get('convergence_epoch', '待补充')} "
            tables += f"| {metrics.get('final_loss', '待补充'):.4f} |\n"
        
        tables += "\n"
    
    # 表4.4 两阶段训练策略对比
    if "exp5" in experiment_results:
        exp5 = experiment_results["exp5"]
        tables += "## 表4.4 两阶段训练策略的效果对比\n\n"
        tables += "| 训练策略 | Stage1 Acc (%) | Stage2 Acc (%) | 最终 Acc (%) | 训练稳定性 |\n"
        tables += "|---------|----------------|----------------|-------------|-----------|\n"
        
        for strategy, metrics in exp5.items():
            tables += f"| {strategy} "
            tables += f"| {metrics.get('stage1_acc', '待补充'):.2f} "
            tables += f"| {metrics.get('stage2_acc', '待补充'):.2f} "
            tables += f"| {metrics.get('final_acc', '待补充'):.2f} "
            tables += f"| {metrics.get('stability_score', '待补充'):.4f} |\n"
        
        tables += "\n"
    
    # 保存表格
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(tables)
    
    print(f"论文表格已保存至: {output_file}")
    return tables


if __name__ == "__main__":
    # 示例用法
    print("本模块提供以下功能：")
    print("1. MetricsCalculator类 - 计算模型各项指标")
    print("2. calculate_roi_extraction_success_rate - 计算ROI提取成功率")
    print("3. generate_thesis_tables - 生成论文表格")
    print("\n使用示例：")
    print("from enhanced_metrics import MetricsCalculator, generate_thesis_tables")
    print("calculator = MetricsCalculator(model)")
    print("metrics = calculator.calculate_topk_accuracy(dataloader)")
