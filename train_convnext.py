import torch
import torch.nn as nn
import torch.optim as optim
import time
from pathlib import Path
from torch.cuda.amp import GradScaler
from models.convnext import convnext_tiny as create_model
from datasets.dataloader import get_data_loaders
from engine.trainer import train_stage
from configs.config import load_config
from utils.get_cuda import get_cuda
from utils.logger import setup_logger
from utils.forconvnext import get_params_groups, create_lr_scheduler
# 获取CUDA设备信息
USE_CUDA, DEVICE = get_cuda()


def main():
    
    cfg = load_config("configs/convnext_tiny.yaml")  # 对应ConvNeXt的配置文件
    setup_logger(cfg)  # 复用日志配置

    
    train_loader, val_loader = get_data_loaders(
        cfg.dataset.input_size,
        cfg.dataset.batch_size,
        USE_CUDA,
        cfg.dataset.limit_train_per_class,
        cfg.dataset.limit_val_per_class
    )

    
    model = create_model(num_classes=cfg.model.num_classes).to(DEVICE)

    # 混合精度 Scaler
    scaler = GradScaler()
    
    # 加载预训练权重
    if cfg.model.pretrained_weights != "":
        assert Path(cfg.model.pretrained_weights).exists(), \
            f"预训练权重文件: {cfg.model.pretrained_weights} 不存在"
        weights_dict = torch.load(cfg.model.pretrained_weights, map_location=DEVICE)["model"]
        # 删除分类头权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        # 加载权重
        model.load_state_dict(weights_dict, strict=False)
        print("ConvNeXt预训练权重加载完成（已剔除分类头）")

    # 损失函数
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.criterion.label_smoothing)

    # ===================== Stage1：仅训练分类头（head） =====================
    # 冻结除head外的所有参数
    for name, param in model.named_parameters():
        if "head" not in name:  # ConvNeXt的分类层为head，替换ResNet的fc
            param.requires_grad = False
        else:
            print(f"Stage1 训练参数: {name}")

    # 初始化优化器
    optimizer = optim.AdamW(
        # 筛选出仅head的可训练参数
        [p for name, p in model.named_parameters() if "head" in name],
        lr=cfg.optimizer.lr_stage1,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        eps=cfg.optimizer.eps,
        weight_decay=cfg.optimizer.weight_decay1
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode=cfg.scheduler.reduce.mode,
            factor=cfg.scheduler.reduce.factor,
            patience=cfg.scheduler.reduce.patience,
            min_lr=cfg.scheduler.reduce.min_lr
            )

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
        DEVICE
    )

    # ===================== 5. Stage2：训练所有参数 =====================
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
        DEVICE
    )
    
    # ===================== 保存模型=====================
    # 创建results目录
    save_dir = Path("results")
    save_dir.mkdir(exist_ok=True)
    # 生成带时间戳的文件名
    save_path = save_dir / f"convnext_tiny_best_{time.strftime('%Y%m%d_%H%M%S')}.pth"
    # 保存模型权重
    torch.save(model.state_dict(), save_path)
    print(f"ConvNeXt模型已保存至：{save_path}")


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    if USE_CUDA:
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True  # 固定卷积算法
        torch.backends.cudnn.benchmark = False    # 关闭自动优化卷积算法
    main()