import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset, DataLoader


# -------------------------- 数据加载、可小批量加载测试逻辑 --------------------------
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


def get_data_loaders(input_size, batch_size, cuda, limit_train_per_class, limit_val_per_class):
    print("[INFO] Reading Training and Testing Dataset")
    traindir = os.path.join("dataset", "train")
    testdir = os.path.join("dataset", "test")

    # 训练集数据增强
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

    test_transforms = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

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
        print(
            f"[INFO] Loaded training samples: {len(data_train)}, validation samples: {len(data_test)}"
        )
    except Exception as e:
        print(f"[ERROR] Failed to load datasets: {str(e)}")
        return None, None

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
