import torch
import torch.nn as nn
import torch.optim as optim
import time  
import json
import pandas as pd
from pathlib import Path  
from models.resnet50 import build_resnet50
from datasets.dataloader import get_data_loaders, get_test_loader
from engine.trainer import train_stage
from engine.evaluator import evaluate
from configs.config import load_config
from utils.get_cuda import get_cuda
from utils.logger import setup_logger
from utils.train_visualizer import plot_training_curves, plot_confusion_matrix
from utils.forconvnext import create_lr_scheduler
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
        if key.startswith("fc."):
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
    parser.add_argument("--config", default="configs/resnet50.yaml")
    parser.add_argument("--no_imagenet_pretrain", action="store_true", help="不使用ImageNet预训练权重（实验4从头训练需要）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logger(cfg)
    
    # 记录训练开始时间
    run_start_time = time.time()
    
    # 获取数据加载器
    train_loader, val_loader = get_data_loaders(
        cfg.dataset.input_size,
        cfg.dataset.batch_size,
        USE_CUDA,
        cfg.dataset.limit_train_per_class,
        cfg.dataset.limit_val_per_class,
        dataset_type=args.dataset_type
    )

    num_classes = get_num_classes_from_loader(train_loader)
    class_names = get_class_names_from_loader(val_loader)
    print(f"[INFO] 数据集实际类别数: {num_classes}")
    if hasattr(cfg, "model") and hasattr(cfg.model, "num_classes") and cfg.model.num_classes != num_classes:
        print(f"[WARNING] 配置中的 model.num_classes={cfg.model.num_classes} 与数据集类别数 {num_classes} 不一致，已自动采用数据集类别数。")
    use_imagenet_pretrained = (not bool(args.pretrained_weights)) and (not args.no_imagenet_pretrain)
    model = build_resnet50(
        num_classes,
        pretrained=use_imagenet_pretrained
    ).to(DEVICE)
    print(f"[INFO] ResNet50 ImageNet pretrained: {use_imagenet_pretrained}")
    
    if args.pretrained_weights:
        print(f"[INFO] 加载预训练权重进行微调: {args.pretrained_weights}")
        load_backbone_weights(model, args.pretrained_weights, DEVICE)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.criterion.label_smoothing)

    # ===================== Stage1：仅训练全连接层 =====================
    # 冻结除fc层外的所有参数
    for name, param in model.named_parameters():
        if not name.startswith("fc"):
            param.requires_grad = False

    optimizer = optim.Adam(
        model.fc.parameters(),
        lr=cfg.optimizer.lr_stage1,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        eps=cfg.optimizer.eps,
        weight_decay=cfg.optimizer.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=cfg.scheduler.mode,
        factor=cfg.scheduler.factor,
        patience=cfg.scheduler.patience,
        min_lr=cfg.scheduler.min_lr
    )

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

    # 训练Stage1（仅当epochs_stage1 > 0时执行）
    if cfg.train.epochs_stage1 > 0:
        optimizer = optim.Adam(
            model.fc.parameters(),
            lr=cfg.optimizer.lr_stage1,
            betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
            eps=cfg.optimizer.eps,
            weight_decay=cfg.optimizer.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=cfg.scheduler.mode,
            factor=cfg.scheduler.factor,
            patience=cfg.scheduler.patience,
            min_lr=cfg.scheduler.min_lr
        )
        
        model = train_stage(
            model,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            scheduler,
            "reduce",
            None,
            cfg,
            cfg.train.epochs_stage1,
            "Stage1",
            DEVICE,
            metrics_csv_path=stage1_csv
        )
    else:
        print("[INFO] 跳过Stage1（单阶段训练模式）")

    # ===================== Stage2：训练所有参数 =====================
    # 解冻所有参数
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.optimizer.lr_stage2,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        eps=cfg.optimizer.eps,
        weight_decay=cfg.optimizer.weight_decay
    )
    warmup_epochs = cfg.scheduler.cosin.warmup_epochs if hasattr(cfg.scheduler, "cosin") else 3
    warmup_lr_init = cfg.scheduler.cosin.warmup_lr_init if hasattr(cfg.scheduler, "cosin") else 1e-8
    min_lr = cfg.scheduler.cosin.min_lr if hasattr(cfg.scheduler, "cosin") else 1e-8
    scheduler = create_lr_scheduler(
        optimizer,
        len(train_loader),
        cfg.train.epochs_stage2,
        warmup=True,
        warmup_epochs=warmup_epochs,
        warmup_factor=warmup_lr_init / cfg.optimizer.lr_stage2,
        end_factor=min_lr / cfg.optimizer.lr_stage2,
    )

    # 训练Stage2
    model = train_stage(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        scheduler,
        "cosine",
        None,
        cfg,
        cfg.train.epochs_stage2,
        "Stage2",
        DEVICE,
        metrics_csv_path=stage2_csv
    )
    
    # ===================== 保存模型 =====================
    # 创建results目录（不存在则创建，已存在不报错）
    # 生成带时间戳的模型文件名，避免覆盖
    save_path = save_dir / f"resnet50_best_{run_stamp}.pth"
    # 保存模型权重
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存至：{save_path}")
    plot_training_curves([stage1_csv, stage2_csv], vis_dir / "curves.png", f"ResNet50 {args.dataset_type}")
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
    # 设置随机种子，保证实验可复现
    torch.manual_seed(42)
    if USE_CUDA:
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True  # 固定卷积算法
        torch.backends.cudnn.benchmark = False     # 关闭自动优化卷积算法
    main()
