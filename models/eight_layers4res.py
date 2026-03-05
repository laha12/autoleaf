import torch.nn as nn

class CustomMLP(nn.Module):
    def __init__(self, num_classes=185, resnet_feature_dim=2048):
        super(CustomMLP, self).__init__()
        
        # 适配ResNet 2048维输入的8层结构设计（全连接层替代卷积层）
        # 层1: 2048 -> 1024
        self.fc1 = nn.Linear(resnet_feature_dim, 1024)
        self.relu1 = nn.ReLU(inplace=True)
        # 层2: 局部响应归一化（适配一维特征，用nn.LayerNorm替代原LRN）
        self.norm1 = nn.LayerNorm(1024)
        # 层3: Dropout（子采样/正则化）
        self.drop1 = nn.Dropout(p=0.2)
        
        # 层4: 1024 -> 512
        self.fc2 = nn.Linear(1024, 512)
        self.relu2 = nn.ReLU(inplace=True)
        # 层5: 归一化
        self.norm2 = nn.LayerNorm(512)
        # 层6: Dropout（子采样/正则化）
        self.drop2 = nn.Dropout(p=0.2)
        
        # 层7: 512 -> 1024（特征映射回目标维度）
        self.fc3 = nn.Linear(512, 1024)
        self.relu3 = nn.ReLU(inplace=True)
        self.drop3 = nn.Dropout(p=0.5)
        
        # 层8: 分类输出层（最终分类头）
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        # 输入x: (batch, 2048) （ResNet提取的一维特征）
        # Layer 1-3
        x = self.drop1(self.norm1(self.relu1(self.fc1(x))))
        # Layer 4-6
        x = self.drop2(self.norm2(self.relu2(self.fc2(x))))
        # Layer 7
        x = self.drop3(self.relu3(self.fc3(x)))
        
        # 最终特征（维度: batch, 1024），可用于SVM
        features = x
        # Layer 8: 分类输出
        out = self.classifier(x)
        
        return out, features

def custom_mlp(num_classes=185):
    return CustomMLP(num_classes=num_classes)