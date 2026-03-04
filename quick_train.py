import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from sklearn.svm import SVC
from models.custom_cnn import custom_cnn
import os

# --- 快速测试配置 ---
NUM_CLASSES = 185
BATCH_SIZE = 16 # 本地小 batch
NUM_EPOCHS = 2  # 只跑两轮
LIMIT_SAMPLES_PER_CLASS = 2 # 每个类只取2张图
INPUT_SIZE = 224

def get_tiny_loaders():
    traindir = os.path.join('dataset', 'train')
    testdir = os.path.join('dataset', 'test')
    
    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    full_train_ds = datasets.ImageFolder(traindir, transform)
    # --- 按类别限制采样 ---
    train_indices = []
    class_counts = {}
    # ImageFolder.samples: list of (path, class_index)
    for idx, (_, class_id) in enumerate(full_train_ds.samples):
        cnt = class_counts.get(class_id, 0)
        if cnt < LIMIT_SAMPLES_PER_CLASS:
            train_indices.append(idx)
            class_counts[class_id] = cnt + 1
    tiny_train = Subset(full_train_ds, train_indices)
    print(f"[DEBUG] 训练采样: {len(train_indices)} 张, 覆盖 {len(class_counts)} 类, 每类最多 {LIMIT_SAMPLES_PER_CLASS} 张")

    # 测试集：每类取 1 张用于快速验证
    full_test_ds = datasets.ImageFolder(testdir, transform)
    test_indices = []
    test_counts = {}
    for idx, (_, class_id) in enumerate(full_test_ds.samples):
        if test_counts.get(class_id, 0) < 1:
            test_indices.append(idx)
            test_counts[class_id] = test_counts.get(class_id, 0) + 1
    tiny_test = Subset(full_test_ds, test_indices)
    print(f"[DEBUG] 测试采样: {len(test_indices)} 张, 覆盖 {len(test_counts)} 类, 每类最多 1 张")
    
    train_loader = DataLoader(tiny_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(tiny_test, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader
def quick_test():
    print("🚀 [START] 开始本地小批量全流程测试...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_tiny_loaders()
    
    # 1. CNN + Softmax 快速跑通
    model = custom_cnn(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    print("\n[Step 1/3] 正在测试 CNN 训练循环...")
    model.train()
    for epoch in range(NUM_EPOCHS):
        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
                _, preds = outputs.max(1)
                acc = (preds == labels).float().mean().item() * 100
                print(f"  - Epoch {epoch+1} Step {i+1}: Loss {loss.item():.4f}, Acc {acc:.2f}%")
        print(f"  - Epoch {epoch+1} 完成, 最后一个 batch Loss: {loss.item():.4f}")

    # 2. 特征提取测试
    print("\n[Step 2/3] 正在测试特征提取...")
    model.eval()
    all_features, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in train_loader:
            _, features = model(imgs.to(device))
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    X = np.concatenate(all_features)
    y = np.concatenate(all_labels)
    print(f"  - 成功提取特征，维度: {X.shape}")

    # 3. SVM 测试
    print("\n[Step 3/3] 正在测试 SVM 拟合...")
    clf = SVC(C=10, kernel='rbf', gamma=0.1) # 快速测试参数
    clf.fit(X, y)
    print("  - SVM 训练完成！")
    
    print("\n✅ [SUCCESS] 本地全流程测试通过！可以放心上传至华为云进行完整训练。")

if __name__ == "__main__":
    quick_test()
