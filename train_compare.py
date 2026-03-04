import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Subset, DataLoader
import argparse
from models.custom_cnn import custom_cnn
import joblib  # 新增：用于保存sklearn模型

# -------------------------- 核心优化1：常量与参数更规范 --------------------------
parser = argparse.ArgumentParser(description='CNN + Softmax/SVM Training for Leaf Classification')
# 基础参数
parser.add_argument('--input_size', type=int, default=224, help='Input image size (default: 224)')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.0001)')
# 数据限制
parser.add_argument('--limit_train_per_class', type=int, default=2, help='Limit samples per class for training (0=unlimited)')
parser.add_argument('--limit_val_per_class', type=int, default=1, help='Limit samples per class for validation (0=unlimited)')
# 训练策略
parser.add_argument('--patience', type=int, default=3, help='Early stopping patience (default: 3)')
parser.add_argument('--min_delta', type=float, default=1e-3, help='Min delta for early stopping (default: 1e-3)')
parser.add_argument('--grad_clip', type=float, default=1.0, help='Max norm for gradient clipping (default: 1.0)')
args = parser.parse_args()

# 设备配置（优化：显式打印设备信息）
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print(f'[INFO] Using device: {DEVICE}')

# 固定参数
NUM_CLASSES = 185
# 从参数读取可变配置
INPUT_SIZE = args.input_size
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.epochs
LEARNING_RATE = args.lr
PATIENCE = args.patience
MIN_DELTA = args.min_delta
GRAD_CLIP = args.grad_clip

# -------------------------- 核心优化2：数据加载鲁棒性提升 --------------------------
def subset_by_class_limit(dataset, limit):
    if not limit or limit <= 0:
        return dataset
    indices = []
    counts = {}
    # 优化：兼容ImageFolder的target获取（避免samples属性依赖）
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
    
    # 优化：数据增强更全面，且避免尺寸失真
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.8, 1.0)),  # 随机裁剪+缩放，提升泛化
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),  # 随机旋转，增加数据多样性
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2)  # 随机擦除，缓解过拟合
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # 优化：更严格的路径检查
    for dir_path in [traindir, testdir]:
        if not os.path.exists(dir_path):
            print(f"[ERROR] Dataset directory not found: {dir_path}")
            return None, None
        if len(os.listdir(dir_path)) == 0:
            print(f"[ERROR] Dataset directory is empty: {dir_path}")
            return None, None

    try:
        # 
        data_train = datasets.ImageFolder(traindir, train_transforms)
        data_test = datasets.ImageFolder(testdir, test_transforms)
        print(f'[INFO] Loaded training samples: {len(data_train)}, validation samples: {len(data_test)}')
    except Exception as e:
        print(f"[ERROR] Failed to load datasets: {str(e)}")
        return None, None
    
    # 按类别限制样本
    data_train = subset_by_class_limit(data_train, args.limit_train_per_class)
    data_test = subset_by_class_limit(data_test, args.limit_val_per_class)

    # 优化：DataLoader参数更鲁棒（避免num_workers导致的问题）
    num_workers = 8 if USE_CUDA else 0  # CPU模式下关闭多进程，避免报错
    train_loader = DataLoader(
        data_train, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=True  # 丢弃最后不完整的batch，避免批次尺寸不一致
    )
    val_loader = DataLoader(
        data_test, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    return train_loader, val_loader

# -------------------------- 核心优化3：评估函数更精准 --------------------------
def evaluate(model, loader, criterion):
    model.eval()  # 严格保证验证时关闭训练模式
    loss_sum = 0.0
    correct = 0
    total = 0
    # 优化：使用torch.no_grad()上下文，彻底禁用梯度
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)  # 非阻塞传输，提升速度
            labels = labels.to(DEVICE, non_blocking=True)
            
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            
            # 优化：损失计算基于batch size加权
            # 单个批次的平均损失 * 单个批次样本数目
            loss_sum += loss.item() * labels.size(0)
            # 只要索引
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # .item()：将张量转为 Python 标量
            correct += (predicted == labels).sum().item()
    
    # 优化：避免除以0
    total = max(total, 1)
    avg_loss = loss_sum / total
    acc = 100 * correct / total
    return avg_loss, acc

# -------------------------- 核心优化4：训练逻辑修复与优化 --------------------------
def train_softmax(model, train_loader, val_loader, criterion, optimizer, epochs=NUM_EPOCHS):
    print('\n[INFO] Starting CNN + Softmax Training')
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    # 优化：学习率调度器参数更合理
    # mode=min 表示监控损失值 损失值越小越好
    # factor学习率衰减因子，新 lr = 旧 lr × factor
    # patience 指标连续多少个 epoch 无改善后，才降低学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=1, 
        min_lr=1e-6  # 设置最小学习率，避免学习率过低
    )
    
    for epoch in range(epochs):
        model.train()  # 每轮训练前重置为训练模式
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start = time.time()
        
        # 优化：使用tqdm显示进度条（需确保安装tqdm，若未安装可注释）
        try:
            from tqdm import tqdm
            train_iter = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        except ImportError:
            train_iter = train_loader
            print(f'\nEpoch {epoch+1}/{epochs}')
        
        for i, (images, labels) in enumerate(train_iter):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            
            # 核心：清零梯度（必须在每个batch前）
            optimizer.zero_grad()
            
            # 前向传播
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播 + 梯度裁剪（解决梯度爆炸）
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            
            # 参数更新
            optimizer.step()
            
            # 统计指标
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 优化：日志打印频率更合理（避免刷屏）
            if (i + 1) % 50 == 0 and not isinstance(train_iter, tqdm):
                batch_acc = 100 * correct / total
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Batch Acc: {batch_acc:.2f}%')
        
        # 计算本轮训练指标
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / max(total, 1)
        epoch_time = time.time() - epoch_start
        print(f'\n[TRAIN] Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%, Time: {epoch_time:.2f}s')
        
        # 验证
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f'[VAL] Epoch [{epoch+1}/{epochs}] Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 优化：早停逻辑更严谨（加入MIN_DELTA判断）
        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict()
            print(f'[INFO] Updated best model (val loss improved to {best_val_loss:.4f})')
        else:
            patience_counter += 1
            print(f'[INFO] Early stopping counter: {patience_counter}/{PATIENCE}')
            if patience_counter >= PATIENCE:
                print(f'[INFO] Early stopping at epoch {epoch+1}')
                break

    print('[INFO] CNN + Softmax Training Finished')
    
    # 优化：模型保存逻辑（优先保存最优模型）
    save_path = 'leaf_model_softmax.pth'
    if best_state is not None:
        torch.save({
            'epoch': epoch,
            'model_state_dict': best_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }, save_path)
        print(f'[INFO] Best model saved to {save_path} (val loss: {best_val_loss:.4f})')
    else:
        torch.save(model.state_dict(), save_path)
        print(f'[INFO] Current model saved to {save_path} (no improvement observed)')
    
    # 最终验证（加载最优模型）
    if best_state is not None:
        model.load_state_dict(best_state)
    final_val_loss, final_val_acc = evaluate(model, val_loader, criterion)
    print(f'[FINAL] Best Val Loss: {final_val_loss:.4f}, Best Val Acc: {final_val_acc:.2f}%')
    
    return final_val_acc

# -------------------------- 核心优化5：特征提取更高效 --------------------------
def extract_features(model, loader):
    model.eval()
    features_list = []
    labels_list = []
    
    print('[INFO] Extracting features...')
    with torch.no_grad():
        # 优化：使用tqdm显示特征提取进度
        try:
            from tqdm import tqdm
            loader_iter = tqdm(loader, desc='Extracting features')
        except ImportError:
            loader_iter = loader
        
        for images, labels in loader_iter:
            images = images.to(DEVICE, non_blocking=True)
            
            _, features = model(images)
            
            # 优化：更安全的特征展平
            features = features.view(features.size(0), -1)
            
            # 优化：分批转移到CPU，避免显存溢出
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())
    
    # 合并特征
    X = np.concatenate(features_list, axis=0)
    y = np.concatenate(labels_list, axis=0)
    print(f'[INFO] Extracted features - X shape: {X.shape}, y shape: {y.shape}')
    
    return X, y

# -------------------------- 核心优化6：SVM训练更稳定（新增保存逻辑） --------------------------
# def train_svm(model, train_loader, val_loader):
#     print('\n[INFO] Starting CNN + SVM Training (Optimized)')
    
#     # 1. 提取特征（保留原有逻辑）
#     X_train, y_train = extract_features(model, train_loader)
#     X_test, y_test = extract_features(model, val_loader)
    
#     # 2. 核心优化：特征降维（PCA）- 解决高维特征问题
#     from sklearn.decomposition import PCA
#     # 保留95%的方差，自动确定降维维度（可手动设n_components=256/512）
#     pca = PCA(n_components=0.95, random_state=42)
#     print(f'[INFO] Reducing feature dimension (original: {X_train.shape[1]})...')
#     X_train = pca.fit_transform(X_train)
#     X_test = pca.transform(X_test)
#     print(f'[INFO] Reduced feature dimension: {X_train.shape[1]}')
    
#     # 3. 特征标准化（保留，但优化计算）
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
    
#     # 4. 优化SVM参数（核心提速）
#     print('[INFO] Fitting SVM (optimized params)...')
#     try:
#         clf = SVC(
#             C=10,                      # 可后续调参，先固定
#             kernel='rbf',              # 保留RBF核
#             gamma='scale',             # 替代硬编码gamma：自动计算1/(n_features * X.var())
#             cache_size=5000,           # 增大缓存（根据内存调整，建议4000-8000）
#             tol=1e-2,                  # 放宽收敛阈值（从1e-3→1e-2，减少迭代）
#             max_iter=10000,            # 限制最大迭代（避免无限循环）
#             verbose=False,
#             n_jobs=-1,                 # 关键：启用所有CPU核心并行计算
#             random_state=42
#         )
        
#         clf.fit(X_train, y_train)
#     except Exception as e:
#         print(f'[ERROR] SVM training failed: {str(e)}')
#         return 0.0
    
#     # 5. 预测与评估（保留）
#     print('[INFO] Evaluating SVM...')
#     y_pred = clf.predict(X_test)
#     acc = accuracy_score(y_test, y_pred) * 100
    
#     # ========== 新增：保存SVM相关模型 ==========
#     svm_save_dir = 'saved_svm_models'
#     if not os.path.exists(svm_save_dir):
#         os.makedirs(svm_save_dir)
    
#     # 保存SVM模型
#     svm_path = os.path.join(svm_save_dir, 'svm_classifier.pkl')
#     joblib.dump(clf, svm_path)
#     print(f'[INFO] SVM classifier saved to {svm_path}')
    
#     # 保存PCA降维器（预测时需要用相同的PCA处理特征）
#     pca_path = os.path.join(svm_save_dir, 'pca_transformer.pkl')
#     joblib.dump(pca, pca_path)
#     print(f'[INFO] PCA transformer saved to {pca_path}')
    
#     # 保存标准化器（预测时需要用相同的Scaler处理特征）
#     scaler_path = os.path.join(svm_save_dir, 'standard_scaler.pkl')
#     joblib.dump(scaler, scaler_path)
#     print(f'[INFO] Standard scaler saved to {scaler_path}')
    
#     # 保存SVM训练的元信息
#     svm_meta = {
#         'accuracy': acc,
#         'feature_dimension': X_train.shape[1],
#         'train_samples': len(X_train),
#         'test_samples': len(X_test),
#         'pca_n_components': pca.n_components_,
#         'svm_params': clf.get_params()
#     }
#     meta_path = os.path.join(svm_save_dir, 'svm_meta.json')
#     import json
#     with open(meta_path, 'w') as f:
#         json.dump(svm_meta, f, indent=4)
#     print(f'[INFO] SVM metadata saved to {meta_path}')
    
#     return acc

# -------------------------- 核心优化7：主函数逻辑更清晰 --------------------------
def main():
    # 1. 数据加载
    train_loader, val_loader = get_data_loaders()
    if train_loader is None or val_loader is None:
        print('[ERROR] Data loading failed, exiting...')
        return

    # 2. 模型初始化
    print('[INFO] Initializing model...')
    model = custom_cnn(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    # 打印模型参数量（可选）
    total_params = sum(p.numel() for p in model.parameters())
    print(f'[INFO] Model initialized with {total_params/1e6:.2f}M parameters')

    # 3. 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    # SGD优化器
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=0.004,
        nesterov=True  # 启用Nesterov动量，提升收敛速度
    )

    # 4. 训练CNN + Softmax
    softmax_acc = train_softmax(model, train_loader, val_loader, criterion, optimizer)

    # 5. 训练CNN + SVM
    # svm_acc = train_svm(model, train_loader, val_loader)

    # 6. 结果对比
    print('\n=========================================')
    print('          PERFORMANCE COMPARISON         ')
    print('=========================================')
    print(f'Method              | Accuracy')
    print('-----------------------------------------')
    print(f'CNN + Softmax       | {softmax_acc:.2f}%')
    # print(f'CNN + SVM           | {svm_acc:.2f}%')
    print('=========================================')

    # 优化：释放显存
    if USE_CUDA:
        torch.cuda.empty_cache()

if __name__ == '__main__':
    # 优化：设置随机种子，保证可复现
    torch.manual_seed(42)
    np.random.seed(42)
    if USE_CUDA:
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # 关闭benchmark，保证可复现
    
    main()