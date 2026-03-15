import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


def build_resnet50(num_classes, pretrained=True):

    if pretrained:
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    else:
        model = models.resnet50()

    in_features = model.fc.in_features

    model.fc = nn.Linear(in_features, num_classes)

    return model