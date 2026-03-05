import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import argparse
from pathlib import Path
# 导入你的自定义8层网络
from models.eight_layers4res import CustomMLP
import sys

# -------------------------- 1. 基础配置与日志 --------------------------
parser = argparse.ArgumentParser(description='ResNet50 Real-time Feature Extraction + Custom 8-Layer CNN Training')
# 基础参数
parser.add_argument('--input_size', type=int, default=224, help='ResNet input size (default: 224)')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
parser.add_argument('--epochs', type=int, default=30, help='Training epochs (default: 30)')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
# 数据与保存配置
parser.add_argument('--data_root', type=str, default='dataset', help='Dataset root path')
parser.add_argument('--log_dir', type=str, default='logs', help='Log save dir')
parser.add_argument('--model_save_path', type=str, default='custom_cnn_best.pth', help='Best model save path')
# 训练策略
parser.add_argument('--patience', type=int, default=5, help='Early stopping patience (default: 5)')
parser.add_argument('--min_delta', type=float, default=1e-3, help='Min delta for early stopping')
# 性能优化
parser.add_argument('--num_workers', type=int, default=8, help='Dataloader num workers (default: 8)')
parser.add_argument('--pin_memory', type=bool, default=True, help='Dataloader pin memory (default: True)')
args = parser.parse_args()

# 设备配置
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
NUM_CLASSES = 185

# 日志保存（同第一部分逻辑）
def setup_logger():
    log_path = Path(args.log_dir)
    log_path.mkdir(exist_ok=True)
    log_file = log_path / f"custom_cnn_realtime_train_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    class Logger:
        def __init__(self, file_path):
            self.console = sys.stdout
            self.file = open(file_path, 'w', encoding='utf-8')
        def write(self, message):
            self.console.write(message)
            self.file.write(message)
            self.file.flush()
        def flush(self):
            self.console.flush()
            self.file.flush()
    sys.stdout = Logger(log_file)
    print(f"[INFO] Using device: {DEVICE}")
    print(f"[INFO] Log saved to {log_file}")
    print("\n[INFO] Training Config:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

# -------------------------- 2. ResNet50实时特征提取器 --------------------------
class ResNetFeatureExtractor:
    def __init__(self):
        # 加载预训练ResNet50，移除最后一层全连接层
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # 提取avgpool后的特征（2048维）
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])
        self.feature_extractor.eval()  # 推理模式，固定参数不训练
        self.feature_extractor = self.feature_extractor.to(DEVICE)
        
        # 数据预处理（与ResNet训练时一致）
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @torch.no_grad()  # 特征提取阶段禁用梯度
    def extract_feat(self, images):
        """单次batch图像特征提取"""
        images = images.to(DEVICE, non_blocking=True)
        feats = self.feature_extractor(images)
        feats = feats.view(feats.size(0), -1)  # 展平为 (batch, 2048)
        return feats

# -------------------------- 3. 训练与评估函数 --------------------------
def evaluate(model, val_loader, feature_extractor, criterion):
    """评估函数：实时提取特征并评估"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            # 实时提取特征
            feats = feature_extractor.extract_feat(images)
            labels = labels.to(DEVICE, non_blocking=True)
            
            # 自定义CNN前向传播
            outputs, _ = model(feats)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    avg_loss = total_loss / total
    acc = 100 * correct / total
    return avg_loss, acc

def train_custom_cnn():
    # 1. 初始化特征提取器
    feature_extractor = ResNetFeatureExtractor()
    
    # 2. 加载原始图像数据集（3w样本）
    print("[INFO] Loading image datasets (3w samples)...")
    train_dataset = datasets.ImageFolder(
        os.path.join(args.data_root, 'train'),
        transform=feature_extractor.train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(args.data_root, 'test'),
        transform=feature_extractor.val_transform
    )
    
    # 3. 构建数据加载器（优化3w样本加载速度）
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers if USE_CUDA else 0,
        pin_memory=args.pin_memory,
        drop_last=True  # 避免最后一个不完整batch影响训练
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,  # 验证时batch可翻倍，加快速度
        shuffle=False,
        num_workers=args.num_workers if USE_CUDA else 0,
        pin_memory=args.pin_memory,
        drop_last=False
    )
    
    # 4. 初始化自定义8层网络
    model = CustomMLP(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Custom 8-layer CNN initialized (params: {total_params/1e6:.2f}M)")
    
    # 5. 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.004
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )
    
    # 6. 训练主逻辑（实时提取特征）
    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    
    print("\n[INFO] Starting custom CNN training with real-time ResNet features...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        epoch_start = time.time()
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            # 实时提取ResNet特征（无梯度）
            feats = feature_extractor.extract_feat(images)
            labels = labels.to(DEVICE, non_blocking=True)
            
            # 自定义CNN训练
            optimizer.zero_grad()
            outputs, _ = model(feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # 打印batch进度（可选）
            if (batch_idx + 1) % 100 == 0:
                print(f"Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        # 训练指标
        avg_train_loss = running_loss / len(train_loader)
        # 验证（实时提取特征）
        val_loss, val_acc = evaluate(model, val_loader, feature_extractor, criterion)
        # 学习率调度
        scheduler.step(val_acc)
        epoch_time = time.time() - epoch_start
        
        # 打印日志
        print(f"\nEpoch [{epoch+1}/{args.epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Time: {epoch_time:.2f}s, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最优模型
        if val_acc > best_val_acc + args.min_delta:
            best_val_acc = val_acc
            best_state = model.state_dict()
            patience_counter = 0
            model_save_path = Path(args.model_save_path)
            model_save_path.parent.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_state,
                'best_val_acc': best_val_acc,
                'optimizer_state_dict': optimizer.state_dict()
            }, args.model_save_path)
            print(f"[INFO] Updated best model (val acc: {best_val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"[INFO] Early stopping counter: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print("[INFO] Early stopping triggered")
                break
    
    # 最终评估
    model.load_state_dict(best_state)
    final_val_loss, final_val_acc = evaluate(model, val_loader, feature_extractor, criterion)
    print(f"\n[FINAL] Best Val Loss: {final_val_loss:.4f}, Best Val Acc: {final_val_acc:.2f}%")
    print(f"Best model saved to {args.model_save_path}")

# -------------------------- 4. 主函数 --------------------------
def main():
    setup_logger()
    train_custom_cnn()

if __name__ == '__main__':
    # 固定随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    if USE_CUDA:
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True  # 开启benchmark加速3w样本训练
    
    main()