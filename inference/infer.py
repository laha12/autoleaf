import argparse
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from configs.config import load_config
from models.convnext import convnext_tiny
from models.resnet50 import build_resnet50
from utils.get_cuda import get_cuda


_, DEVICE = get_cuda()


DEFAULT_CONFIG = {
    "resnet50": "configs/resnet50.yaml",
    "convnext": "configs/convnext_tiny.yaml",
    "swintransformer": "configs/swin_tiny.yaml",
}

DEFAULT_WEIGHTS = {
    "resnet50": "results/resnet50_best.pth",
    "convnext": "results/convnext_tiny_best.pth",
    "swintransformer": "results/swin_tiny_best.pth",
}


def normalize_model_name(model_name):
    return model_name.lower().replace("_", "").replace("-", "")


def build_model(model_name, cfg):
    normalized_name = normalize_model_name(model_name)
    num_classes = cfg.model.num_classes
    if normalized_name == "resnet50":
        return build_resnet50(num_classes, False)
    if normalized_name == "convnext":
        return convnext_tiny(num_classes=num_classes)
    if normalized_name == "swintransformer":
        try:
            import timm
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "当前环境缺少 timm，无法加载 Swin Transformer。请先安装 timm。"
            ) from exc
        swin_model_name = getattr(cfg.model, "model_name", "swin_tiny_patch4_window7_224")
        return timm.create_model(swin_model_name, pretrained=False, num_classes=num_classes)
    raise ValueError(f"不支持的模型类型: {model_name}")


def build_transform(input_size):
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )


def load_model(model_name, weights_path, cfg):
    model = build_model(model_name, cfg)
    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    return model


def predict(image_path, model, transform):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output, 1)
    return pred.item()


def evaluate_test_accuracy(test_dir, model, transform, batch_size, use_cuda):
    dataset = ImageFolder(test_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 if use_cuda else 0,
        pin_memory=use_cuda,
    )
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    accuracy = 100.0 * correct / total if total else 0.0
    return accuracy, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--config", default="")
    parser.add_argument("--weights", default="")
    parser.add_argument("--image", default="")
    parser.add_argument("--test-dir", default="dataset/test")
    parser.add_argument("--eval-test", action="store_true")
    args = parser.parse_args()

    normalized_model_name = normalize_model_name(args.model)
    if normalized_model_name not in DEFAULT_CONFIG:
        raise ValueError(f"不支持的模型类型: {args.model}")

    config_path = args.config or DEFAULT_CONFIG[normalized_model_name]
    weights_path = args.weights or DEFAULT_WEIGHTS[normalized_model_name]

    cfg = load_config(config_path)
    input_size = cfg.dataset.input_size
    batch_size = cfg.dataset.batch_size

    transform = build_transform(input_size)
    model = load_model(args.model, weights_path, cfg)

    if args.image:
        pred = predict(args.image, model, transform)
        print(f"预测类别ID: {pred}")

    if args.eval_test:
        test_path = Path(args.test_dir)
        if not test_path.exists():
            raise FileNotFoundError(f"测试集目录不存在: {test_path}")
        use_cuda, _ = get_cuda()
        test_acc, sample_count = evaluate_test_accuracy(
            str(test_path), model, transform, batch_size, use_cuda
        )
        print(f"Test Accuracy: {test_acc:.2f}% ({sample_count} samples)")

    if not args.image and not args.eval_test:
        print("请传入 --image 做单图预测，或传入 --eval-test 评估 test 集准确率")


if __name__ == "__main__":
    main()
