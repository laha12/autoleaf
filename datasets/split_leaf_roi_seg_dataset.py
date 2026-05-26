import argparse
import random
import re
import shutil
from pathlib import Path

import pandas as pd


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def slugify(text):
    text = text.strip().lower()
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text if text else "class"


def normalize_name(text):
    text = text.strip().lower()
    text = text.replace(" ", "").replace("_", "").replace("-", "")
    text = text.replace("原图", "").replace("images", "")
    return text

# 获取叶片类别名
def yolo_base_name(folder_name):
    name = folder_name.strip()
    lower_name = name.lower()
    if lower_name.endswith("yolo"):
        name = name[: len(name) - 4]
    return name.strip(" _-")


def find_image_for_label(label_path, image_dir):
    for ext in IMAGE_EXTS:
        candidate = image_dir / f"{label_path.stem}{ext}"
        if candidate.exists():
            return candidate
        candidate_upper = image_dir / f"{label_path.stem}{ext.upper()}"
        if candidate_upper.exists():
            return candidate_upper
    return None


def split_counts(n, train_ratio, val_ratio):
    if n <= 0:
        return 0, 0, 0
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val
    if n_train <= 0:
        n_train = 1
    if n >= 3 and n_val <= 0:
        n_val = 1
    if n >= 3 and n_test <= 0:
        n_test = 1
    while n_train + n_val + n_test > n:
        if n_train >= n_val and n_train >= n_test and n_train > 1:
            n_train -= 1
        elif n_val >= n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break
    while n_train + n_val + n_test < n:
        n_train += 1
    return n_train, n_val, n_test


def prepare_dirs(output_root):
    for split in ["train", "val", "test"]:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

# 找到label所对应img
def collect_samples(input_root):
    samples_by_class = {}
    missing_images = []
    image_dir_map = {}
    for d in input_root.iterdir():
        if d.is_dir():
            key = normalize_name(d.name)
            image_dir_map[key] = d

    for yolo_dir in input_root.iterdir():
        if not yolo_dir.is_dir():
            continue
        if not yolo_dir.name.lower().endswith("yolo"):
            continue

        base_name = yolo_base_name(yolo_dir.name)
        image_dir = image_dir_map.get(normalize_name(base_name))
        if image_dir is None:
            missing_images.append(f"[MISSING_IMAGE_DIR] {yolo_dir}")
            continue

        class_name = base_name
        for label_path in yolo_dir.glob("*.txt"):
            if label_path.name.lower() == "classes.txt":
                continue
            image_path = find_image_for_label(label_path, image_dir)
            if image_path is None:
                missing_images.append(str(label_path))
                continue
            samples_by_class.setdefault(class_name, []).append(
                {"class_name": class_name, "image_path": image_path, "label_path": label_path}
            )
    return samples_by_class, missing_images


def copy_sample(sample, split, output_root, counters):
    class_slug = slugify(sample["class_name"])
    stem_slug = slugify(sample["label_path"].stem)
    counters.setdefault(split, 0)
    counters[split] += 1
    unique_stem = f"{class_slug}__{stem_slug}__{counters[split]:06d}"
    dst_img = output_root / "images" / split / f"{unique_stem}{sample['image_path'].suffix.lower()}"
    dst_label = output_root / "labels" / split / f"{unique_stem}.txt"
    shutil.copy2(sample["image_path"], dst_img)
    shutil.copy2(sample["label_path"], dst_label)
    return str(dst_img), str(dst_label), unique_stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="dataset/complex_bg/raw")
    parser.add_argument("--output-root", default="dataset/leaf_roi_seg")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    if not input_root.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_root}")
    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train_ratio 与 val_ratio 需 >0，且 train_ratio + val_ratio < 1")

    random.seed(args.seed)
    prepare_dirs(output_root)

    samples_by_class, missing_images = collect_samples(input_root)
    if not samples_by_class:
        raise RuntimeError("未找到可用的 yolo 标注样本（txt+同名图片）")

    manifest_rows = []
    counters = {}
    source_to_split = {}

    for class_name, samples in samples_by_class.items():
        random.shuffle(samples)
        n_train, n_val, n_test = split_counts(len(samples), args.train_ratio, args.val_ratio)
        split_plan = {
            "train": samples[:n_train],
            "val": samples[n_train:n_train + n_val],
            "test": samples[n_train + n_val:n_train + n_val + n_test],
        }
        for split, split_samples in split_plan.items():
            for sample in split_samples:
                source_key = str(sample["image_path"].resolve())
                if source_key in source_to_split:
                    raise RuntimeError(f"检测到样本重复划分: {source_key}")
                source_to_split[source_key] = split
                dst_img, dst_label, uid = copy_sample(sample, split, output_root, counters)
                manifest_rows.append(
                    {
                        "uid": uid,
                        "split": split,
                        "class_name": class_name,
                        "src_image": str(sample["image_path"]),
                        "src_label": str(sample["label_path"]),
                        "dst_image": dst_img,
                        "dst_label": dst_label,
                    }
                )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = output_root / "split_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    print(f"[INFO] 输入根目录: {input_root}")
    print(f"[INFO] 输出根目录: {output_root}")
    print(f"[INFO] 样本总数: {len(manifest_rows)}")
    print(f"[INFO] train: {(manifest_df['split'] == 'train').sum()}")
    print(f"[INFO] val: {(manifest_df['split'] == 'val').sum()}")
    print(f"[INFO] test: {(manifest_df['split'] == 'test').sum()}")
    print(f"[INFO] 类别数: {manifest_df['class_name'].nunique()}")
    print(f"[INFO] 清单文件: {manifest_path}")
    if missing_images:
        missing_path = output_root / "missing_image_for_label.txt"
        missing_path.write_text("\n".join(missing_images), encoding="utf-8")
        print(f"[WARNING] 有 {len(missing_images)} 个 txt 未找到同名图片，已记录到: {missing_path}")


if __name__ == "__main__":
    main()
