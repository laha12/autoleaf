import torch.nn as nn

class CustomCNN(nn.Module):
    def __init__(self, num_classes=185):
        super(CustomCNN, self).__init__()
        
        # 1. C1: 卷积层（适配 224x224 输入）
        # 输入 224x224 -> 输出 55x55
        self.conv1 = nn.Conv2d(3, 32, kernel_size=11, stride=4, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        
        # 2. L2: 局部对比度归一化层
        self.lrn1 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1.0)
        
        # 3. S3: 子采样层 (MaxPooling)
        # 55x55 -> 27x27
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # 4. C4: 卷积层 - 保持尺寸不变
        # 27x27 -> 27x27
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        
        # 5. L5: 局部对比度归一化层
        self.lrn2 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1.0)
        
        # 6. S6: 子采样层 (MaxPooling)
        # 27x27 -> 13x13
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # 展平维度: 64通道 * 13 * 13（输入 224x224）
        self.flatten_dim = 64 * 13 * 13 
        
        # 7. F7: 全连接层
        self.fc1 = nn.Linear(self.flatten_dim, 1024)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        
        # 8. F8: 分类输出层
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Layer 1-3: C1 -> L2 -> S3
        x = self.pool1(self.lrn1(self.relu1(self.conv1(x))))
        
        # Layer 4-6: C4 -> L5 -> S6
        x = self.pool2(self.lrn2(self.relu2(self.conv2(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Layer 7: F7 + Dropout
        x = self.dropout(self.relu3(self.fc1(x)))
        
        # 用于 SVM 的特征向量
        features = x
        
        # Layer 8: F8 输出
        out = self.classifier(x)
        
        return out, features

def custom_cnn(num_classes=185):
    return CustomCNN(num_classes=num_classes)
