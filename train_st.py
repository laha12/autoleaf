import torch
import torch.nn as nn
import torch.optim as optim
import time
import timm
from pathlib import Path
from torch.cuda.amp import GradScaler
from datasets.dataloader import get_data_loaders
from engine.trainer import train_stage  # 需修改 train_stage 支持混合精度
from configs.config import load_config
from utils.get_cuda import get_cuda
from utils.logger import setup_logger
from utils.forconvnext import get_params_groups, create_lr_scheduler  # 可复用于 Swin 的参数分组
USE_CUDA, DEVICE = get_cuda()

def main():
    cfg = load_config("configs/swin_tiny.yaml")  # 需新建 Swin 配置文件
    setup_logger(cfg)

    # 数据加载
    train_loader, val_loader = get_data_loaders(
        cfg.dataset.input_size,
        cfg.dataset.batch_size,
        USE_CUDA,
        cfg.dataset.limit_train_per_class,
        cfg.dataset.limit_val_per_class
    )
    n_iter_per_epoch = len(train_loader)  # 每个 epoch 的迭代数

    # 创建 Swin Transformer 模型
    model = timm.create_model(
        cfg.model.model_name,  # 如 'swin_tiny_patch4_window7_224'
        pretrained=True,
        num_classes=cfg.model.num_classes
    ).to(DEVICE)

    # 损失函数
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.criterion.label_smoothing)

    # 混合精度 Scaler
    scaler = GradScaler()


    # ===================== Stage 1：仅训练分类头（head） =====================
    for name, param in model.named_parameters():
        param.requires_grad = "head" in name  # 仅解冻 head
        if param.requires_grad:
            print(f"Stage 1 训练参数: {name}")

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.optimizer.lr_stage1,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        eps=cfg.optimizer.eps,
        weight_decay=cfg.optimizer.weight_decay1
    )
    cfg.train.epochs_current_stage = cfg.train.epochs_stage1
    scheduler = create_lr_scheduler(
        optimizer,
        num_step=n_iter_per_epoch,                # 每个 epoch 的步数
        epochs=cfg.train.epochs_stage1,           # 当前 stage 的总 epochs
        warmup=True,
        warmup_epochs=cfg.scheduler.warmup_epochs, # 从 cfg 取预热 epochs
        warmup_factor=cfg.scheduler.warmup_lr_init / cfg.optimizer.lr_stage1, # 计算 warmup_factor
        end_factor=cfg.scheduler.min_lr / cfg.optimizer.lr_stage1              # 计算 end_factor
    )

    model = train_stage(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        criterion,
        scheduler, 
        scaler, 
        cfg, 
        cfg.train.epochs_stage1, 
        "Stage 1", 
        DEVICE
    )


    # ===================== Stage 2：训练 head + norm + 最后一个 stage（layers[3]） =====================
    for name, param in model.named_parameters():
        param.requires_grad = any(k in name for k in ["head", "norm", "layers.3"])
        if param.requires_grad:
            print(f"Stage 2 训练参数: {name}")

    pg = get_params_groups(model, weight_decay=cfg.optimizer.weight_decay2)
    optimizer = optim.AdamW(
        pg, lr=cfg.optimizer.lr_stage2,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        eps=cfg.optimizer.eps, weight_decay=cfg.optimizer.weight_decay2
    )
    cfg.train.epochs_current_stage = cfg.train.epochs_stage2
    scheduler = create_lr_scheduler(
        optimizer,
        num_step=n_iter_per_epoch,                # 每个 epoch 的步数
        epochs=cfg.train.epochs_stage2,           # 当前 stage 的总 epochs
        warmup=True,
        warmup_epochs=cfg.scheduler.warmup_epochs, # 从 cfg 取预热 epochs
        warmup_factor=cfg.scheduler.warmup_lr_init / cfg.optimizer.lr_stage2, # 计算 warmup_factor
        end_factor=cfg.scheduler.min_lr / cfg.optimizer.lr_stage2              # 计算 end_factor
    )

    model = train_stage(
        model, train_loader, val_loader, optimizer, criterion,
        scheduler, scaler, cfg, cfg.train.epochs_stage2, "Stage 2", DEVICE
    )


    # ===================== Stage 3：训练 head + norm + layers[2] + layers[3] =====================
    for name, param in model.named_parameters():
        param.requires_grad = any(k in name for k in ["head", "norm", "layers.2", "layers.3"])
        if param.requires_grad:
            print(f"Stage 3 训练参数: {name}")

    pg = get_params_groups(model, weight_decay=cfg.optimizer.weight_decay3)
    optimizer = optim.AdamW(
        pg, lr=cfg.optimizer.lr_stage3,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        eps=cfg.optimizer.eps, weight_decay=cfg.optimizer.weight_decay3
    )
    cfg.train.epochs_current_stage = cfg.train.epochs_stage3
    scheduler = create_lr_scheduler(
        optimizer,
        num_step=n_iter_per_epoch,                # 每个 epoch 的步数
        epochs=cfg.train.epochs_stage3,           # 当前 stage 的总 epochs
        warmup=True,
        warmup_epochs=cfg.scheduler.warmup_epochs, # 从 cfg 取预热 epochs
        warmup_factor=cfg.scheduler.warmup_lr_init / cfg.optimizer.lr_stage3, # 计算 warmup_factor
        end_factor=cfg.scheduler.min_lr / cfg.optimizer.lr_stage3              # 计算 end_factor
    )

    model = train_stage(
        model, train_loader, val_loader, optimizer, criterion,
        scheduler, scaler, cfg, cfg.train.epochs_stage3, "Stage 3", DEVICE
    )


    # ===================== Stage 4：全量训练 =====================
    for param in model.parameters():
        param.requires_grad = True
    print("Stage 4 解冻所有参数，开始全量训练")

    pg = get_params_groups(model, weight_decay=cfg.optimizer.weight_decay4)
    optimizer = optim.AdamW(
        pg, lr=cfg.optimizer.lr_stage4,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        eps=cfg.optimizer.eps, weight_decay=cfg.optimizer.weight_decay4
    )
    cfg.train.epochs_current_stage = cfg.train.epochs_stage4
    scheduler = create_lr_scheduler(
        optimizer,
        num_step=n_iter_per_epoch,                # 每个 epoch 的步数
        epochs=cfg.train.epochs_stage4,           # 当前 stage 的总 epochs
        warmup=True,
        warmup_epochs=cfg.scheduler.warmup_epochs, # 从 cfg 取预热 epochs
        warmup_factor=cfg.scheduler.warmup_lr_init / cfg.optimizer.lr_stage4, # 计算 warmup_factor
        end_factor=cfg.scheduler.min_lr / cfg.optimizer.lr_stage4              # 计算 end_factor
    )

    model = train_stage(
        model, train_loader, val_loader, optimizer, criterion,
        scheduler, scaler, cfg, cfg.train.epochs_stage4, "Stage 4", DEVICE
    )


    # ===================== 保存模型 =====================
    save_dir = Path("results")
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / f"swin_tiny_best_{time.strftime('%Y%m%d_%H%M%S')}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Swin Transformer 模型已保存至：{save_path}")


if __name__ == "__main__":
    torch.manual_seed(42)
    if USE_CUDA:
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    main()