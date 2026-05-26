import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset, DataLoader


def is_complex_dataset(dataset_type: str) -> bool:
    """判断是否为复杂背景数据集"""
    return dataset_type in ["complex_center", "complex_yolo", "complex_raw"]


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
    print(f"[INFO] Subset dataset to {len(indices)} samples (limit {limit} per class)")
    return Subset(dataset, indices)


def normalize_class_limit_map(class_limit_map):
    if not class_limit_map:
        return {}
    if hasattr(class_limit_map, "__dict__"):
        items = class_limit_map.__dict__.items()
    elif isinstance(class_limit_map, dict):
        items = class_limit_map.items()
    else:
        return {}
    normalized = {}
    for k, v in items:
        try:
            limit = int(v)
        except Exception:
            continue
        if limit > 0:
            normalized[str(k)] = limit
    return normalized


def subset_by_class_name_limit(dataset, class_limit_map):
    limits = normalize_class_limit_map(class_limit_map)
    if not limits:
        return dataset
    if not hasattr(dataset, "classes") or not hasattr(dataset, "targets"):
        print("[WARNING] Dataset 不支持按类别名限量，已跳过")
        return dataset

    indices = []
    counts = {}
    for idx, cls_idx in enumerate(dataset.targets):
        class_name = dataset.classes[cls_idx]
        class_limit = limits.get(class_name)
        if class_limit is None:
            indices.append(idx)
            continue
        current = counts.get(class_name, 0)
        if current < class_limit:
            indices.append(idx)
            counts[class_name] = current + 1
    print(f"[INFO] Class-specific subset dataset to {len(indices)} samples")
    for class_name, class_limit in limits.items():
        kept = counts.get(class_name, 0)
        print(f"[INFO] Class cap - {class_name}: kept {kept}, cap {class_limit}")
    return Subset(dataset, indices)


def get_data_loaders(
    input_size,
    batch_size,
    cuda,
    limit_train_per_class,
    limit_val_per_class,
    dataset_type="single",
    train_class_limit_map=None,
):
    """
    获取数据加载器
    dataset_type:
        - 'single' (单一背景, Leafsnap)
        - 'complex_center' (复杂背景, 中心裁剪轻量方案)
        - 'complex_yolo' (复杂背景, YOLOv8n ROI 方案)
        - 'complex_raw' (复杂背景, 不做ROI提取，直接resize)
    """
    print(f"[INFO] Reading Dataset for type: {dataset_type}")
    
    if dataset_type == "single":
        base_dir = "dataset/single_bg/processed"
    elif dataset_type == "complex_center":
        base_dir = "dataset/complex_bg/processed/center"
    elif dataset_type == "complex_yolo":
        base_dir = "dataset/complex_bg/processed/yolo"
    elif dataset_type == "complex_raw":
        base_dir = "dataset/complex_bg/processed/raw"
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    traindir = os.path.join(base_dir, "train")
    valdir = os.path.join(base_dir, "val")
    testdir = os.path.join(base_dir, "test")
    evaldir = valdir if os.path.exists(valdir) and len(os.listdir(valdir)) > 0 else testdir

    # 训练集基础数据增强
    if dataset_type == "single":
        train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        train_transforms = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(20),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    test_transforms = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    for dir_path in [traindir, evaldir]:
        if not os.path.exists(dir_path):
            print(f"[ERROR] Dataset directory not found: {dir_path}")
            return None, None
        if len(os.listdir(dir_path)) == 0:
            print(f"[ERROR] Dataset directory is empty: {dir_path}")
            return None, None

    try:
        data_train = datasets.ImageFolder(traindir, train_transforms)
        data_test = datasets.ImageFolder(evaldir, test_transforms)
        eval_name = "validation" if evaldir == valdir else "test(fallback)"
        print(
            f"[INFO] Loaded training samples: {len(data_train)}, {eval_name} samples: {len(data_test)}"
        )
    except Exception as e:
        print(f"[ERROR] Failed to load datasets: {str(e)}")
        return None, None

    data_train = subset_by_class_name_limit(data_train, train_class_limit_map)
    data_train = subset_by_class_limit(data_train, limit_train_per_class)
    data_test = subset_by_class_limit(data_test, limit_val_per_class)

    num_workers = 4 if cuda else 0

    train_loader = DataLoader(
        data_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        data_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_loader(input_size, batch_size, cuda, dataset_type="single"):
    if dataset_type == "single":
        base_dir = "dataset/single_bg/processed"
    elif dataset_type == "complex_center":
        base_dir = "dataset/complex_bg/processed/center"
    elif dataset_type == "complex_yolo":
        base_dir = "dataset/complex_bg/processed/yolo"
    elif dataset_type == "complex_raw":
        base_dir = "dataset/complex_bg/processed/raw"
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    testdir = os.path.join(base_dir, "test")
    if not os.path.exists(testdir) or len(os.listdir(testdir)) == 0:
        print(f"[WARNING] Test dataset directory not found or empty: {testdir}")
        return None

    test_transforms = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    num_workers = 4 if cuda else 0
    test_dataset = datasets.ImageFolder(testdir, test_transforms)
    print(f"[INFO] Loaded independent test samples: {len(test_dataset)}")
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
