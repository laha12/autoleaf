from pathlib import Path  # 处理文件/文件夹路径
from utils.get_cuda import get_cuda
from configs.config import load_config
import yaml  # 读写yaml配置文件
from ultralytics import YOLO  # 导入YOLO模型


def validate_dataset(root: Path):
    """
    检查数据集文件夹结构是否正确
    """
    # 必须存在的4个文件夹
    required = [
        root / "images" / "train",
        root / "images" / "val",
        root / "labels" / "train",
        root / "labels" / "val",
    ]
    # 遍历检查每个文件夹是否存在
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"缺少目录: {p}")

    # 检查训练集和验证集是否有图片
    train_images = list((root / "images" / "train").glob("*"))
    val_images = list((root / "images" / "val").glob("*"))
    if len(train_images) == 0 or len(val_images) == 0:
        raise RuntimeError("train/val 图片为空，无法训练")


def write_data_yaml(dataset_root: Path, yaml_path: Path):
    """
    自动生成YOLO训练需要的data.yaml配置文件
    """
    # yaml文件内容（数据集路径、类别等）
    data = {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "leaf"},
    }
    
    # 创建yaml所在文件夹
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    # 写入yaml文件
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def normalize_labels_to_single_class(dataset_root: Path):
    label_root = dataset_root / "labels"
    if not label_root.exists():
        return 0
    changed = 0
    for txt in label_root.rglob("*.txt"):
        raw = txt.read_text(encoding="utf-8", errors="ignore").splitlines()
        new_lines = []
        file_changed = False
        for line in raw:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            if parts[0] != "0":
                parts[0] = "0"
                file_changed = True
            new_lines.append(" ".join(parts))
        if file_changed:
            txt.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")
            changed += 1
    return changed


def to_plain_dict(cfg_obj):
    if hasattr(cfg_obj, "__dict__"):
        return {k: v for k, v in cfg_obj.__dict__.items() if not k.startswith("_")}
    if isinstance(cfg_obj, dict):
        return dict(cfg_obj)
    return {}


def build_train_kwargs(train_cfg, stage_cfg):
    train_dict = to_plain_dict(train_cfg)
    stage_dict = to_plain_dict(stage_cfg)
    kwargs = {
        "imgsz": train_dict["imgsz"],
        "batch": train_dict["batch"],
        "workers": train_dict["workers"],
        "project": train_dict["project"],
        "save_period": train_dict["save_period"],
        "seed": train_dict["seed"],
        "single_cls": train_dict.get("single_cls", True),
        "epochs": stage_dict["epochs"],
        "patience": stage_dict["patience"],
        "freeze": stage_dict["freeze"],
        "lr0": stage_dict["lr0"],
    }
    optional_stage_keys = ["weight_decay", "lrf", "optimizer", "momentum", "warmup_epochs"]
    optional_train_keys = [
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "flipud",
        "fliplr",
        "mosaic",
        "mixup",
        "close_mosaic",
        "cos_lr",
        "amp",
    ]
    for k in optional_stage_keys:
        if k in stage_dict:
            kwargs[k] = stage_dict[k]
    for k in optional_train_keys:
        if k in train_dict:
            kwargs[k] = train_dict[k]
    return kwargs

def main():

    cfg = load_config("configs/yolov8.yaml")

    dataset_root = Path(cfg.dataset.root)
    data_yaml = Path(cfg.dataset.data_yaml)
    model_weights = Path(cfg.model.weights)
    task = cfg.model.task
    train_cfg = cfg.train
    stage1 = cfg.stage1
    stage2 = cfg.stage2

    validate_dataset(dataset_root)
    if getattr(train_cfg, "single_cls", True) and getattr(train_cfg, "normalize_label_ids", True):
        changed = normalize_labels_to_single_class(dataset_root)
        print(f"[INFO] 单类标签归一化完成，改写文件数: {changed}")
    write_data_yaml(dataset_root, data_yaml)
    _,device = get_cuda()


    stage1_name = f"{train_cfg.name}_stage1"
    model_stage1 = YOLO(model_weights)
    stage1_kwargs = build_train_kwargs(train_cfg, stage1)
    result_stage1 = model_stage1.train(
        data=str(data_yaml.resolve()),
        device=device,
        name=stage1_name,
        task=task,
        **stage1_kwargs,
    )

    best_stage1 = Path(result_stage1.save_dir) / "weights" / "best.pt"
    if not best_stage1.exists():
        raise FileNotFoundError(f"阶段1训练完成但未找到 best.pt: {best_stage1}")

    stage2_name = f"{train_cfg.name}_stage2"
    model_stage2 = YOLO(str(best_stage1))
    stage2_kwargs = build_train_kwargs(train_cfg, stage2)
    result_stage2 = model_stage2.train(
        data=str(data_yaml.resolve()),
        device=device,
        name=stage2_name,
        task=task,
        **stage2_kwargs,
    )

    best_path = Path(result_stage2.save_dir) / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"阶段2训练完成但未找到 best.pt: {best_path}")

    target = Path(train_cfg.copy_best_to)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(best_path.read_bytes())

    print(f"[INFO] data.yaml: {data_yaml.resolve()}")
    print(f"[INFO] 阶段1 best权重: {best_stage1.resolve()}")
    print(f"[INFO] best权重: {best_path.resolve()}")
    print(f"[INFO] 已复制到: {target.resolve()}")


if __name__ == "__main__":
    main()
