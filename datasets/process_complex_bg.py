import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from utils.img_process import hybrid_roi_extraction
import argparse
import random
import shutil

# 配置参数
RAW_COMPLEX_DIR = Path("dataset/complex_bg/raw")
CSV_SAVE_DIR = Path("dataset/csv")
CSV_SAVE_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_SIZE = (224, 224)
RANDOM_STATE = 42


def add_gaussian_noise(image, intensity=0.15):
    """添加高斯噪声"""
    noise = np.random.normal(0, intensity * 255, image.shape)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_random_occlusion(image, num_occlusions=3, max_size=60):
    """添加随机遮挡块"""
    h, w = image.shape[:2]
    result = image.copy()

    for _ in range(num_occlusions):
        x = np.random.randint(0, max(1, w - max_size))
        y = np.random.randint(0, max(1, h - max_size))
        size = np.random.randint(20, max_size)
        color = np.random.randint(0, 255, 3).tolist()
        cv2.rectangle(result, (x, y), (x + size, y + size), color, -1)

    return result


def add_brightness_change(image, factor_range=(0.6, 1.4)):
    """随机亮度变化"""
    factor = np.random.uniform(*factor_range)
    adjusted = image.astype(np.float32) * factor
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def add_compound_noise(image, noise_level=0.15, occlusion_prob=0.6):
    """综合噪声：高斯噪声 + 随机遮挡 + 亮度变化"""
    result = image.copy()

    result = add_gaussian_noise(result, intensity=noise_level)

    if np.random.random() < occlusion_prob:
        result = add_random_occlusion(result)

    result = add_brightness_change(result)

    return result


def load_and_split_data(exclude_species=None):
    if exclude_species is None:
        exclude_species = []
    elif isinstance(exclude_species, str):
        exclude_species = [s.strip() for s in exclude_species.split(",")]

    all_data = []

    for specie_dir in RAW_COMPLEX_DIR.iterdir():
        if not specie_dir.is_dir():
            continue
        if specie_dir.name.lower().endswith("yolo"):
            continue

        specie_name = specie_dir.name

        if specie_name in exclude_species:
            print(f"[INFO] 跳过排除类别: {specie_name}")
            continue

        for img_file in sorted(list(specie_dir.glob("*.jpg")) + list(specie_dir.glob("*.png"))):
            all_data.append({
                "image_path": str(img_file),
                "species": specie_name
            })

    if not all_data:
        raise RuntimeError(f"未在 {RAW_COMPLEX_DIR} 找到可用图片")

    df = pd.DataFrame(all_data)
    species_counts = df["species"].value_counts().sort_index()
    print(f"[INFO] 类别数: {len(species_counts)}")
    for specie, cnt in species_counts.items():
        print(f"[INFO] {specie}: {cnt}张")

    train_list = []
    val_list = []
    test_list = []

    rng = random.Random(RANDOM_STATE)

    for specie_name, group in df.groupby("species", sort=True):
        indices = list(group.index)
        rng.shuffle(indices)

        n = len(indices)
        n_test = max(1, int(n * 0.1))
        n_val = max(1, int(n * 0.2))
        n_train = n - n_test - n_val

        if n_train < 1:
            n_val -= 1
            n_train += 1

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        train_list.append(group.loc[train_idx])
        val_list.append(group.loc[val_idx])
        test_list.append(group.loc[test_idx])

    train_df = pd.concat(train_list, ignore_index=True)
    val_df = pd.concat(val_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)

    overlap = (
        set(train_df["image_path"]) & set(val_df["image_path"])
        | set(train_df["image_path"]) & set(test_df["image_path"])
        | set(val_df["image_path"]) & set(test_df["image_path"])
    )
    if overlap:
        raise RuntimeError(f"检测到重复样本: {len(overlap)}张")

    manifest = pd.concat([
        train_df.assign(split="train"),
        val_df.assign(split="val"),
        test_df.assign(split="test"),
    ], ignore_index=True)
    manifest.to_csv(CSV_SAVE_DIR / "complex_bg_split_manifest.csv", index=False, encoding="utf-8-sig")

    return train_df, val_df, test_df


def process_and_save(df, save_dir, csv_name, use_yolo=False, no_roi=False, add_noise=False, noise_level=0.15):
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    processed_paths = []
    processed_species = []
    count = 1
    failed_count = 0
    noise_count = 0

    for idx, row in df.iterrows():
        img_path = row["image_path"]
        specie = row["species"]

        try:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"[ERROR] 无法读取 {img_path}: {e}")
            continue

        if no_roi:
            roi = image
        else:
            roi = hybrid_roi_extraction(image, image_path=img_path, use_yolo=use_yolo)
            if roi is None or roi.size == 0:
                print(f"[WARNING] ROI提取失败，使用原图: {img_path}")
                roi = image
                failed_count += 1

        # 如果是测试集且需要加噪声，在ROI提取之后添加
        if add_noise:
            roi = add_compound_noise(roi, noise_level=noise_level)
            noise_count += 1

        roi_resized = cv2.resize(roi, IMAGE_SIZE)

        specie_dir = save_dir / specie.lower().replace(" ", "_")
        specie_dir.mkdir(parents=True, exist_ok=True)

        save_path = specie_dir / f"{count}.jpg"
        cv2.imwrite(str(save_path), cv2.cvtColor(roi_resized, cv2.COLOR_RGB2BGR))

        processed_paths.append(str(save_path))
        processed_species.append(specie)
        count += 1

        if idx > 0 and idx % 50 == 0:
            print(f"[INFO] {save_dir.name} - 已处理 {idx} 张")

    df_out = pd.DataFrame({
        "image_paths": processed_paths,
        "species": processed_species
    })
    df_out.to_csv(CSV_SAVE_DIR / csv_name, index=False)
    noise_info = f" (加噪声: {noise_count} 张)" if add_noise else ""
    print(f"[INFO] {save_dir.name} 完成: {len(processed_paths)} 张 (失败: {failed_count} 张){noise_info}")
    return df_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="复杂背景数据预处理")
    parser.add_argument("--roi_mode", default="center", choices=["center", "yolo", "no_roi"], help="ROI方案: center(中心裁剪)、yolo(YOLOv8 ROI提取)、no_roi(不做ROI提取，直接resize)")
    parser.add_argument("--exclude_species", type=str, default=None, help="排除的类别名，如: '苹果叶'")
    parser.add_argument("--test_noise", action="store_true", help="测试集添加噪声干扰")
    parser.add_argument("--noise_level", type=float, default=0.0, help="噪声强度 (0.1-0.3)")
    args = parser.parse_args()

    use_yolo = args.roi_mode == "yolo"
    no_roi = args.roi_mode == "no_roi"

    if use_yolo:
        PROCESSED_DIR = Path("dataset/complex_bg/processed/yolo")
        prefix = "complex_bg_yolo"
        print("[INFO] 使用YOLOv8 ROI提取")
    elif no_roi:
        PROCESSED_DIR = Path("dataset/complex_bg/processed/raw")
        prefix = "complex_bg_raw"
        print("[INFO] 不做ROI提取，直接resize原始图像")
    else:
        PROCESSED_DIR = Path("dataset/complex_bg/processed/center")
        prefix = "complex_bg_center"
        print("[INFO] 使用中心裁剪ROI提取")

    if args.test_noise:
        print(f"[INFO] 测试集将添加噪声干扰 (强度: {args.noise_level})")

    train_df, val_df, test_df = load_and_split_data(exclude_species=args.exclude_species)
    print(f"[INFO] 训练集: {len(train_df)}, 验证集: {len(val_df)}, 测试集: {len(test_df)}")

    # 训练集和验证集不加噪声
    process_and_save(train_df, PROCESSED_DIR / "train", f"{prefix}_train.csv", use_yolo=use_yolo, no_roi=no_roi)
    process_and_save(val_df, PROCESSED_DIR / "val", f"{prefix}_val.csv", use_yolo=use_yolo, no_roi=no_roi)
    # 测试集根据参数决定是否加噪声
    process_and_save(test_df, PROCESSED_DIR / "test", f"{prefix}_test.csv", use_yolo=use_yolo, no_roi=no_roi, add_noise=args.test_noise, noise_level=args.noise_level)

    print(f"[DONE] 完成! 输出目录: {PROCESSED_DIR}")
