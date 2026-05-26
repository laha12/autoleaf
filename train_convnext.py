import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
import pandas as pd
from pathlib import Path
from torch.cuda.amp import GradScaler
from models.convnext import convnext_tiny as create_model
from datasets.dataloader import get_data_loaders, get_test_loader
from engine.trainer import train_stage
from engine.evaluator import evaluate
from configs.config import load_config
from utils.get_cuda import get_cuda
from utils.logger import setup_logger
from utils.forconvnext import get_params_groups, create_lr_scheduler
from utils.train_visualizer import plot_training_curves, plot_confusion_matrix
from utils.thesis_metrics import collect_all_metrics
# 获取CUDA设备信息
USE_CUDA, DEVICE = get_cuda()


def get_num_classes_from_loader(loader):
    dataset = loader.dataset
    if hasattr(dataset, "classes"):
        return len(dataset.classes)
    if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "classes"):
        return len(dataset.dataset.classes)
    raise ValueError("无法从数据加载器中获取类别数")


def get_class_names_from_loader(loader):
    dataset = loader.dataset
    if hasattr(dataset, "classes"):
        return dataset.classes
    if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "classes"):
        return dataset.dataset.classes
    raise ValueError("无法从数据加载器中获取类别名")


def load_backbone_weights(model, weights_path, device):
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model_state = model.state_dict()
    filtered_state = {}
    for k, v in checkpoint.items():
        key = k[7:] if k.startswith("module.") else k
        if key.startswith("head."):
            continue
        if key in model_state and model_state[key].shape == v.shape:
            filtered_state[key] = v
    model.load_state_dict(filtered_state, strict=False)
    print(f"[INFO] 加载骨干权重参数数量: {len(filtered_state)}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", default="single", choices=["single", "complex_center", "complex_yolo", "complex_raw"])
    parser.add_argument("--pretrained_weights", default="", help="用于微调的预训练权重路径（实验3、4需要）")
    parser.add_argument("--config", default="configs/convnext_tiny.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logger(cfg)  # 复用日志配置
    
    # 记录训练开始时间
    run_start_time = time.time()

    train_loader, val_loader = get_data_loaders(
        cfg.dataset.input_size,
        cfg.dataset.batch_size,
        USE_CUDA,
        cfg.dataset.limit_train_per_class,
        cfg.dataset.limit_val_per_class,
        dataset_type=args.dataset_type,
    )

    
    num_classes = get_num_classes_from_loader(train_loader)
    class_names = get_class_names_from_loader(val_loader)
    print(f"[INFO] 数据集实际类别数: {num_classes}")
    if hasattr(cfg, "model") and hasattr(cfg.model, "num_classes") and cfg.model.num_classes != num_classes:
        print(f"[WARNING] 配置中的 model.num_classes={cfg.model.num_classes} 与数据集类别数 {num_classes} 不一致，已自动采用数据集类别数。")
    model = create_model(num_classes=num_classes).to(DEVICE)
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    config_name = Path(args.config).stem
    run_tag = f"{config_name}_{run_stamp}"
    run_root = Path("results") / config_name / run_stamp
    curve_dir = run_root / "curves"
    vis_dir = run_root / "visualization"
    save_dir = run_root / "checkpoints"
    curve_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    stage1_csv = curve_dir / "stage1.csv"
    stage2_csv = curve_dir / "stage2.csv"

    # 混合精度 Scaler
    scaler = GradScaler()
    
    # 加载预训练权重
    weights_to_load = args.pretrained_weights if args.pretrained_weights else cfg.model.pretrained_weights
    if weights_to_load != "":
        assert Path(weights_to_load).exists(), \
            f"预训练权重文件: {weights_to_load} 不存在"
        
        if args.pretrained_weights:
            print(f"[INFO] 加载微调预训练权重: {weights_to_load}")
            load_backbone_weights(model, weights_to_load, DEVICE)
        else:
            # 如果是官方预训练权重，需要剔除 head
            weights_dict = torch.load(weights_to_load, map_location=DEVICE)["model"]
            # 删除分类头权重
            for k in list(weights_dict.keys()):
                if "head" in k:
                    del weights_dict[k]
            # 加载权重
            model.load_state_dict(weights_dict, strict=False)
            print("ConvNeXt预训练权重加载完成（已剔除分类头）")

    # 损失函数
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.criterion.label_smoothing)

    # ===================== Stage1 =====================
    # 冻结除head外的所有参数
    for name, param in model.named_parameters():
        if "head" not in name:  # ConvNeXt的分类层为head，替换ResNet的fc
            param.requires_grad = False
        else:
            print(f"Stage1 训练参数: {name}")

    # 初始化优化器
    optimizer = optim.AdamW(
        [p for name, p in model.named_parameters() if "head" in name],
        lr=cfg.optimizer.lr_stage1,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        eps=cfg.optimizer.eps,
        weight_decay=cfg.optimizer.weight_decay1
    )

    warmup_epochs_s1 = getattr(cfg.scheduler, 'warmup_epochs_stage1', 3)
    warmup_lr_init_s1 = getattr(cfg.scheduler, 'warmup_lr_init_stage1', cfg.optimizer.lr_stage1 * 0.01)
    min_lr_s1 = getattr(cfg.scheduler.reduce, 'min_lr', 1e-6)

    if warmup_epochs_s1 > 0:
        scheduler = create_lr_scheduler(
            optimizer,
            len(train_loader),
            cfg.train.epochs_stage1,
            warmup=True,
            warmup_epochs=warmup_epochs_s1,
            warmup_factor=warmup_lr_init_s1 / cfg.optimizer.lr_stage1,
            end_factor=min_lr_s1 / cfg.optimizer.lr_stage1
        )
        print(f"[INFO] Stage1 使用 Cosine Warmup (warmup_epochs={warmup_epochs_s1})")
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=cfg.scheduler.reduce.mode,
            factor=cfg.scheduler.reduce.factor,
            patience=cfg.scheduler.reduce.patience,
            min_lr=cfg.scheduler.reduce.min_lr
        )
        print("[INFO] Stage1 使用 ReduceLROnPlateau")

    # 训练Stage1
    model = train_stage(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        scheduler,
        "reduce",
        scaler,
        cfg,
        cfg.train.epochs_stage1,
        "Stage1",
        DEVICE,
        metrics_csv_path=stage1_csv
    )

    # =====================  Stage2 =====================
    # 解冻所有参数（和train.py一致）
    for param in model.parameters():
        param.requires_grad = True
    print("Stage2 解冻所有参数，开始全量训练")

    # 初始化优化器
    pg = get_params_groups(model, weight_decay=cfg.optimizer.weight_decay2)
    optimizer = optim.AdamW(
        pg,
        lr=cfg.optimizer.lr_stage2,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        eps=cfg.optimizer.eps,
        weight_decay=cfg.optimizer.weight_decay2
    )
    scheduler = create_lr_scheduler(optimizer, 
                                    len(train_loader), 
                                    cfg.train.epochs_stage2,
                                    warmup=True, 
                                    warmup_epochs=cfg.scheduler.cosin.warmup_epochs,
                                    warmup_factor=cfg.scheduler.cosin.warmup_lr_init / cfg.optimizer.lr_stage2,
                                    end_factor=cfg.scheduler.cosin.min_lr / cfg.optimizer.lr_stage2)

    # 训练Stage2
    model = train_stage(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        scheduler,
        "cosine",
        scaler,
        cfg,
        cfg.train.epochs_stage2,
        "Stage2",
        DEVICE,
        metrics_csv_path=stage2_csv
    )
    
    # ===================== 保存模型=====================
    # 创建results目录
    # 生成带时间戳的文件名
    save_path = save_dir / f"convnext_tiny_best_{run_stamp}.pth"
    # 保存模型权重
    torch.save(model.state_dict(), save_path)
    print(f"ConvNeXt模型已保存至：{save_path}")
    plot_training_curves([stage1_csv, stage2_csv], vis_dir / "curves.png", f"ConvNeXt {args.dataset_type}")
    plot_confusion_matrix(model, val_loader, class_names, DEVICE, vis_dir / "confusion_matrix.png")
    
    # ===================== 计算论文所需指标 =====================
    # 从CSV文件加载训练历史
    training_history = {"val_acc": []}
    for csv_file in [stage1_csv, stage2_csv]:
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            if not df.empty and "val_acc" in df.columns:
                training_history["val_acc"].extend(df["val_acc"].tolist())
    
    # 收集所有论文指标
    thesis_metrics = collect_all_metrics(
        model=model,
        val_loader=val_loader,
        criterion=criterion,
        device=DEVICE,
        class_names=class_names,
        training_history=training_history,
        output_dir=run_root
    )
    
    # 添加训练时间
    thesis_metrics["training_time_s"] = time.time() - run_start_time
    thesis_metrics["training_time_h"] = thesis_metrics["training_time_s"] / 3600
    
    # 保存完整指标
    metrics_file = run_root / "thesis_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(thesis_metrics, f, indent=2, ensure_ascii=False)
    print(f"\n论文指标已保存至: {metrics_file}")
    
    test_loader = get_test_loader(
        cfg.dataset.input_size,
        cfg.dataset.batch_size,
        USE_CUDA,
        dataset_type=args.dataset_type
    )
    if test_loader is not None:
        test_result = evaluate(model, test_loader, criterion, DEVICE, k=5)
        print(f"[INFO] Independent TestLoss {test_result['loss']:.4f} | TestAcc {test_result['top1_acc']:.2f}% | TestTop5 {test_result['top5_acc']:.2f}%")
    print(f"可视化已保存至：{vis_dir}")


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    if USE_CUDA:
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True  # 固定卷积算法
        torch.backends.cudnn.benchmark = False    # 关闭自动优化卷积算法
    main()
