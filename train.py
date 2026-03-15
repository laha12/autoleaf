import torch
import torch.nn as nn
import torch.optim as optim
import time  # 新增：用于生成时间戳文件名
from pathlib import Path  # 修正：Pathlib→pathlib，impot→import
from models.resnet50 import build_resnet50
from datasets.dataloader import get_data_loaders
from engine.trainer import train_stage
from configs.config import load_config
from utils.get_cuda import get_cuda
from utils.logger import setup_logger

# 获取CUDA设备信息
USE_CUDA, DEVICE = get_cuda()


def main():
    # 加载配置文件（已做数值类型转换，避免eps等参数为字符串）
    cfg = load_config("configs/resnet50.yaml")
    setup_logger(cfg)
    # 获取数据加载器
    train_loader, val_loader = get_data_loaders(
        cfg.dataset.input_size,
        cfg.dataset.batch_size,
        USE_CUDA,
        cfg.dataset.limit_train_per_class,
        cfg.dataset.limit_val_per_class
    )

    # 构建模型并移至指定设备（CPU/GPU）
    model = build_resnet50(cfg.model.num_classes).to(DEVICE)
    
    # 损失函数（交叉熵损失，适配分类任务）
    criterion = nn.CrossEntropyLoss()

    # ===================== Stage1：仅训练全连接层 =====================
    # 冻结除fc层外的所有参数
    for name, param in model.named_parameters():
        if not name.startswith("fc"):
            param.requires_grad = False

    # 初始化优化器（仅优化fc层参数）
    optimizer = optim.Adam(
        model.fc.parameters(),
        lr=cfg.optimizer.lr_stage1,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        eps=cfg.optimizer.eps,
        weight_decay=cfg.optimizer.weight_decay
    )

    # 训练Stage1
    model = train_stage(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        cfg,
        cfg.train.epochs_stage1,
        "Stage1",
        DEVICE
    )

    # ===================== Stage2：训练所有参数 =====================
    # 解冻所有参数
    for param in model.parameters():
        param.requires_grad = True

    # 初始化优化器（优化所有参数，学习率更低）
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.optimizer.lr_stage2,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        eps=cfg.optimizer.eps,
        weight_decay=cfg.optimizer.weight_decay
    )

    # 训练Stage2
    model = train_stage(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        cfg,
        cfg.train.epochs_stage2,
        "Stage2",
        DEVICE
    )
    
    # ===================== 保存模型 =====================
    # 创建results目录（不存在则创建，已存在不报错）
    save_dir = Path("results")
    save_dir.mkdir(exist_ok=True)
    # 生成带时间戳的模型文件名，避免覆盖
    save_path = save_dir / f"resnet50_best_{time.strftime('%Y%m%d_%H%M%S')}.pth"
    # 保存模型权重
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存至：{save_path}")


if __name__ == "__main__":
    # 设置随机种子，保证实验可复现
    torch.manual_seed(42)
    if USE_CUDA:
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True  # 固定卷积算法
        torch.backends.cudnn.benchmark = False     # 关闭自动优化卷积算法
    main()