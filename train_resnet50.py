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
from torch.utils.data import Subset, DataLoader
import argparse
from models.cbam import CBAM


# -------------------------- 核心优化1：常量与参数更规范 --------------------------
parser = argparse.ArgumentParser(description='ResNet50 + Softmax Training (Stage1: Only FC, Stage2: All Layers)')
# 基础参数
parser.add_argument('--input_size', type=int, default=224, help='Input image size (default: 224)')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
parser.add_argument('--epochs_stage1', type=int, default=10, help='Epochs for stage1 (only FC layer)')
parser.add_argument('--epochs_stage2', type=int, default=30, help='Epochs for stage2 (all layers)')
parser.add_argument('--lr_stage1', type=float, default=0.001, help='Learning rate for stage1 (default: 0.001)')
parser.add_argument('--lr_stage2', type=float, default=0.0002, help='Learning rate for stage2 (default: 0.0001)')
# Adam优化器专属参数
parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1 parameter (default: 0.9)')
parser.add_argument('--beta2', type=float, default=0.999, help='Adam beta2 parameter (default: 0.999)')
parser.add_argument('--eps', type=float, default=1e-8, help='Adam epsilon parameter (default: 1e-8)')
# 数据限制
parser.add_argument('--limit_train_per_class', type=int, default=0, help='Limit samples per class for training (0=unlimited)')
parser.add_argument('--limit_val_per_class', type=int, default=0, help='Limit samples per class for validation (0=unlimited)')
# 训练策略
parser.add_argument('--patience', type=int, default=5, help='Early stopping patience (default: 5)')
parser.add_argument('--min_delta', type=float, default=5e-3, help='Min delta for early stopping (default: 5e-3)')
parser.add_argument('--grad_clip', type=float, default=1.0, help='Max norm for gradient clipping (default: 1.0)')
args = parser.parse_args()

# 设备配置
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print(f'[INFO] Using device: {DEVICE}')

# 固定参数
NUM_CLASSES = 185
# 从参数读取可变配置
INPUT_SIZE = args.input_size
BATCH_SIZE = args.batch_size
EPOCHS_STAGE1 = args.epochs_stage1
EPOCHS_STAGE2 = args.epochs_stage2
LR_STAGE1 = args.lr_stage1
LR_STAGE2 = args.lr_stage2
PATIENCE = args.patience
MIN_DELTA = args.min_delta
GRAD_CLIP = args.grad_clip
# Adam参数
BETA1 = args.beta1
BETA2 = args.beta2
EPS = args.eps

# -------------------------- 核心优化2：数据加载鲁棒性提升 --------------------------
def subset_by_class_limit(dataset, limit):
    if not limit or limit <= 0:
        return dataset
    indices = []
    counts = {}
    for idx in range(len(dataset)):
        _, cls = dataset[idx]
        c = counts.get(cls, 0)
        if c < limit:
            indices.append(idx)
            counts[cls] = c + 1
    print(f'[INFO] Subset dataset to {len(indices)} samples (limit {limit} per class)')
    return Subset(dataset, indices)

def get_data_loaders():
    print('[INFO] Reading Training and Testing Dataset')
    traindir = os.path.join('dataset', 'train')
    testdir = os.path.join('dataset', 'test')
    
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3,scale=(0.02,0.1))
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    for dir_path in [traindir, testdir]:
        if not os.path.exists(dir_path):
            print(f"[ERROR] Dataset directory not found: {dir_path}")
            return None, None
        if len(os.listdir(dir_path)) == 0:
            print(f"[ERROR] Dataset directory is empty: {dir_path}")
            return None, None

    try:
        data_train = datasets.ImageFolder(traindir, train_transforms)
        data_test = datasets.ImageFolder(testdir, test_transforms)
        print(f'[INFO] Loaded training samples: {len(data_train)}, validation samples: {len(data_test)}')
    except Exception as e:
        print(f"[ERROR] Failed to load datasets: {str(e)}")
        return None, None
    
    data_train = subset_by_class_limit(data_train, args.limit_train_per_class)
    data_test = subset_by_class_limit(data_test, args.limit_val_per_class)

    num_workers = 8 if USE_CUDA else 0
    train_loader = DataLoader(
        data_train, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        data_test, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    return train_loader, val_loader

# -------------------------- 核心优化3：评估函数（返回损失+准确率） --------------------------
def evaluate(model, loader, criterion):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            
            outputs = model(images)  # ResNet50只有output，无额外features
            loss = criterion(outputs, labels)
            
            loss_sum += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    total = max(total, 1)
    avg_loss = loss_sum / total
    acc = 100 * correct / total
    return avg_loss, acc

# -------------------------- 核心优化4：分阶段训练函数（保存最高准确率模型） --------------------------
def train_stage(model, train_loader, val_loader, criterion, optimizer, num_epochs, stage_name, is_finetune=False):
    """
    分阶段训练函数，保存验证准确率最高的模型
    :param model: 模型实例
    :param train_loader: 训练数据加载器
    :param val_loader: 验证数据加载器
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param num_epochs: 训练轮数
    :param stage_name: 阶段名称（用于保存模型命名）
    :param is_finetune: 是否为微调阶段（全层训练）
    """
    print(f'\n[INFO] Starting {stage_name} Training ({"finetune all layers" if is_finetune else "only FC layer"})')
    best_val_acc = 0.0  # 跟踪最高验证准确率
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',  # 改为基于准确率（max）调度学习率
        factor=0.5, 
        patience=1, 
        min_lr=1e-6
    )
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start = time.time()
        
        # 进度条
        try:
            from tqdm import tqdm
            train_iter = tqdm(train_loader, desc=f'{stage_name} Epoch {epoch+1}/{num_epochs}')
        except ImportError:
            train_iter = train_loader
            print(f'\n{stage_name} Epoch {epoch+1}/{num_epochs}')
        
        # 训练批次
        for i, (images, labels) in enumerate(train_iter):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            
            # 统计训练指标
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i + 1) % 50 == 0 and not isinstance(train_iter, tqdm):
                batch_acc = 100 * correct / total
                print(f'{stage_name} Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Batch Acc: {batch_acc:.2f}%')
        
        # 本轮训练指标
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / max(total, 1)
        epoch_time = time.time() - epoch_start
        print(f'\n[TRAIN] {stage_name} Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%, Time: {epoch_time:.2f}s')
        
        # 验证
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f'[VAL] {stage_name} Epoch [{epoch+1}/{num_epochs}] Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
        
        # 学习率调度（基于验证准确率）
        scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f'[INFO] Current {stage_name} learning rate: {current_lr:.6f}')
        
        # 核心：保存验证准确率最高的模型
        if val_acc > best_val_acc + MIN_DELTA:  # 允许微小波动
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
            print(f'[INFO] Updated best {stage_name} model (val acc improved to {best_val_acc:.2f}%)')
        else:
            patience_counter += 1
            print(f'[INFO] {stage_name} Early stopping counter: {patience_counter}/{PATIENCE}')
            if patience_counter >= PATIENCE:
                print(f'[INFO] Early stopping at {stage_name} epoch {epoch+1}')
                break

    # 保存最优模型（基于最高验证准确率）
    save_path = f'resnet50_{stage_name.lower()}_best_acc.pth'
    if best_state is not None:
        torch.save({
            'epoch': epoch,
            'model_state_dict': best_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'best_val_loss': best_val_loss,
            'stage': stage_name
        }, save_path)
        print(f'[INFO] Best {stage_name} model (val acc: {best_val_acc:.2f}%) saved to {save_path}')
    else:
        torch.save(model.state_dict(), save_path)
        print(f'[INFO] Current {stage_name} model saved to {save_path} (no improvement observed)')
    
    # 加载最优模型并输出最终指标
    if best_state is not None:
        model.load_state_dict(best_state)
    final_val_loss, final_val_acc = evaluate(model, val_loader, criterion)
    print(f'[FINAL] Best {stage_name} Val Loss: {final_val_loss:.4f}, Best Val Acc: {final_val_acc:.2f}%')
    
    return final_val_acc, model

# -------------------------- 核心优化5：ResNet50模型构建 --------------------------

# 修改build_resnet50函数，给每个残差块添加CBAM
# 修正：用函数工厂生成独立的forward
def build_resnet50(num_classes, pretrained=True):
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2) if pretrained else models.resnet50()
    
    # 定义函数工厂，为每个block生成独立的forward
    def create_forward(block, cbam):
        original_forward = block.forward
        def new_forward(x):
            identity = x
            out = block.conv1(x)
            out = block.bn1(out)
            out = block.relu(out)
            out = block.conv2(out)
            out = block.bn2(out)
            out = block.relu(out)
            out = block.conv3(out)
            out = block.bn3(out)
            out = cbam(out)  # 绑定当前block的cbam
            if block.downsample is not None:
                identity = block.downsample(x)
            out += identity
            out = block.relu(out)
            return out
        return new_forward
    
    # 循环中调用函数工厂，为每个block生成独立forward
    for name, module in model.named_children():
        if name in ['layer1', 'layer2', 'layer3', 'layer4']:
            for idx, block in enumerate(module):
                cbam = CBAM(block.bn3.num_features)
                block.cbam = cbam
                block.forward = create_forward(block, cbam)  # 绑定独立cbam
    
    # FC层部分保持不变
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(1024, num_classes)
    )
    model = model.to(DEVICE)
    return model

# -------------------------- 核心优化6：主函数（分两轮训练） --------------------------
def main():
    # 1. 数据加载
    train_loader, val_loader = get_data_loaders()
    if train_loader is None or val_loader is None:
        print('[ERROR] Data loading failed, exiting...')
        return

    # 2. 初始化ResNet50模型
    print('[INFO] Initializing ResNet50 model with pretrained weights...')
    model = build_resnet50(NUM_CLASSES, pretrained=True)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'[INFO] ResNet50 initialized with {total_params/1e6:.2f}M parameters')

    # 3. 损失函数
    criterion = nn.CrossEntropyLoss()

    # ======================== 第一轮训练：仅训练最后一层 ========================
    # 冻结除FC层外的所有参数
    for name, param in model.named_parameters():
        # 只有FC层或CBAM的参数可训练，其余冻结
        if name.startswith('fc') or "cbam" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    # 优化器仅优化FC层（替换为Adam）
    optimizer_stage1 = optim.Adam(
        [
        {'params': model.fc.parameters(), 'lr': LR_STAGE1},
        {'params': [p for name, p in model.named_parameters() if 'cbam' in name], 'lr': LR_STAGE1 * 2}  # CBAM用更高LR
        ], 
        betas=(BETA1, BETA2),
        eps=EPS,
        weight_decay=0.005)
    # 训练第一轮
    stage1_acc, model = train_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_stage1,
        num_epochs=EPOCHS_STAGE1,
        stage_name='Stage1',
        is_finetune=False
    )

    # ======================== 第二轮训练：所有层训练 ========================
    # 解冻所有参数
    for param in model.parameters():
        param.requires_grad = True
    # 优化器优化所有参数（替换为Adam，更小的学习率）
    # Stage2 优化器：分层设置学习率
    optimizer_stage2 = optim.Adam([
        # ResNet主干：低学习率微调
        {'params': [p for name, p in model.named_parameters() if not (name.startswith('fc') or 'cbam' in name)], 'lr': LR_STAGE2},
        # FC层：中等学习率
        {'params': model.fc.parameters(), 'lr': LR_STAGE2 * 5},
        # CBAM层：高学习率（快速收敛）
        {'params': [p for name, p in model.named_parameters() if 'cbam' in name], 'lr': LR_STAGE2 * 10}
    ], betas=(BETA1, BETA2), eps=EPS, weight_decay=0.003)
    # 训练第二轮
    stage2_acc, model = train_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_stage2,
        num_epochs=EPOCHS_STAGE2,
        stage_name='Stage2',
        is_finetune=True
    )

    # 结果汇总
    print('\n=========================================')
    print('          PERFORMANCE SUMMARY            ')
    print('=========================================')
    print(f'Method                  | Accuracy')
    print('-----------------------------------------')
    print(f'ResNet50 Stage1 (FC)    | {stage1_acc:.2f}%')
    print(f'ResNet50 Stage2 (All)   | {stage2_acc:.2f}%')
    print('=========================================')

    # 释放显存
    if USE_CUDA:
        torch.cuda.empty_cache()

if __name__ == '__main__':
    # 固定随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    if USE_CUDA:
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    main()