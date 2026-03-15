from torch import nn

class CustomMLP(nn.Module):
    def __init__(self, num_classes=185):
        super(CustomMLP, self).__init__()
        
        # 核心修正：2048维特征 → 8×16×16的2D特征图（8通道，16×16）
        # 验证：8 * 16 * 16 = 2048，维度完全匹配
        self.feature_reshape = lambda x: x.view(-1, 8, 16, 16)
        
        # 1. C1: 卷积层（适配 8×16×16 输入，输出 8×8）
        # 计算验证：输出尺寸 = floor((16 + 2*4 - 7)/2) +1 = floor(17/2)+1=8+1？→ 调整为 kernel=7, stride=2, padding=3
        # 最终：(16 + 2*3 -7)/2 +1 = (15)/2 +1=7+1=8 → 输出 8×8
        self.conv1 = nn.Conv2d(8, 32, kernel_size=7, stride=2, padding=3)  # 替换原11/4/2，保证输出8×8
        self.relu1 = nn.ReLU(inplace=True)
        
        # 2. L2: 局部对比度归一化层（保留原参数）
        self.lrn1 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1.0)

        # 3. S3: 子采样层 (MaxPooling) → 8×8 → 4×4（kernel=2, stride=2）
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 4. C4: 卷积层 - 4×4 → 4×4（padding=2 适配kernel=5，尺寸不变）
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        
        # 5. L5: 局部对比度归一化层（保留原参数）
        self.lrn2 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1.0)
        
        # 6. S6: 子采样层 (MaxPooling) → 4×4 → 2×2（kernel=2, stride=2）
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 展平维度: 64通道 * 2 * 2 = 256（修正原1×1的错误）
        self.flatten_dim = 64 * 2 * 2 
        
        # 7. F7: 全连接层（适配新的展平维度256）
        self.fc1 = nn.Linear(self.flatten_dim, 1024)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        
        # 8. F8: 分类输出层（保留原参数）
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Step1: 2048维一维特征 → 8×16×16的2D特征图
        x = self.feature_reshape(x)  # 输出 shape: [batch_size, 8, 16, 16]
        
        # Layer 1-3: C1 -> L2 -> S3 → 8×16×16 → 32×8×8 → 32×4×4
        x = self.pool1(self.lrn1(self.relu1(self.conv1(x))))
        
        # Layer 4-6: C4 -> L5 -> S6 → 32×4×4 → 64×4×4 → 64×2×2
        x = self.pool2(self.lrn2(self.relu2(self.conv2(x))))
        
        # Flatten → [batch_size, 64*2*2=256]
        x = x.view(x.size(0), -1)
        
        # Layer 7: F7 + Dropout → 256 → 1024
        x = self.dropout(self.relu3(self.fc1(x)))
        
        # 用于 SVM 的特征向量
        features = x
        
        # Layer 8: F8 输出 → 1024 → num_classes
        out = self.classifier(x)
        
        return out, features

# 验证维度是否匹配
if __name__ == "__main__":
    import torch
    # 构造2048维输入（batch_size=2，模拟ResNet输出）
    input_feat = torch.randn(2, 2048)
    model = CustomMLP(num_classes=185)
    out, features = model(input_feat)
    # print("输入特征维度:", input_feat.shape)       # torch.Size([2, 2048])
    # print("C1层输出维度:", model.relu1(model.conv1(model.feature_reshape(input_feat))).shape)  # torch.Size([2, 32, 8, 8])
    # print("S3层输出维度:", model.pool1(model.lrn1(model.relu1(model.conv1(model.feature_reshape(input_feat))))).shape)  # torch.Size([2, 32, 4, 4])
    # print("S6层输出维度:", model.pool2(model.lrn2(model.relu2(model.conv2(model.pool1(model.lrn1(model.relu1(model.conv1(model.feature_reshape(input_feat)))))))).shape)  # torch.Size([2, 64, 2, 2])
    # print("最终输出维度:", out.shape)               # torch.Size([2, 185])
    # print("特征向量维度:", features.shape)         # torch.Size([2, 1024])