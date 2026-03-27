import cv2
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from utils.img_process import hybrid_roi_extraction  # 复用现有ROI提取逻辑

# 配置参数
RAW_COMPLEX_DIR = Path("dataset/complex_bg/raw")
PROCESSED_COMPLEX_DIR = Path("dataset/complex_bg/processed")
CSV_SAVE_DIR = Path("dataset/csv")
CSV_SAVE_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_SIZE = (224, 224)
TEST_SIZE = 0.2  # 测试集占20%
VAL_SIZE = 0.25  # 验证集占训练集的25%
RANDOM_STATE = 7

# 读取自建复杂背景原始数据
def load_complex_bg_raw():
    image_paths = []
    species_list = []
    # 遍历raw目录下的类别文件夹
    for specie_dir in RAW_COMPLEX_DIR.iterdir():
        if not specie_dir.is_dir():
            continue
        specie_name = specie_dir.name  # 类别名（比如class_A）
        # 遍历该类别下的所有图片
        for img_file in list(specie_dir.glob("*.jpg")) + list(specie_dir.glob("*.png")):
            image_paths.append(str(img_file))
            species_list.append(specie_name)
    # 转为DataFrame
    df = pd.DataFrame({
        "image_path": image_paths,
        "species": species_list
    })
    # 分层划分训练/验证/测试集
    train_val_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["species"]
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=train_val_df["species"]
    )
    return train_df, val_df, test_df

# 预处理并保存复杂背景数据（复用YOLO+兜底裁剪）
def process_and_save_complex_bg(df, save_dir, csv_name):
    save_dir.mkdir(parents=True, exist_ok=True)
    processed_paths = []
    processed_species = []
    count = 1
    for idx, row in df.iterrows():
        img_path = row["image_path"]
        specie = row["species"]
        # 读取图片（RGB）
        try:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"[ERROR] 无法读取复杂背景图片 {img_path}: {e}")
            continue
        # 提取ROI（无segment，直接用hybrid_roi_extraction）
        roi = hybrid_roi_extraction(image, image_path=img_path)
        if roi is None or roi.size == 0:
            print(f"[DEBUG] 复杂背景图片 {img_path} ROI提取失败，跳过")
            continue
        # 调整尺寸到224x224
        roi_resized = cv2.resize(roi, IMAGE_SIZE)
        # 按类别保存
        specie_dir = save_dir / specie.lower().replace(" ", "_")
        specie_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"{count}.jpg"
        save_path = specie_dir / file_name
        # 保存
        cv2.imwrite(str(save_path), cv2.cvtColor(roi_resized, cv2.COLOR_RGB2BGR))
        # 记录信息
        processed_paths.append(str(save_path))
        processed_species.append(specie)
        count += 1
        # 进度打印
        if idx > 0 and idx % 50 == 0:
            print(f"[INFO] 复杂背景 {save_dir.name} 集 - 处理了 {idx} 张图片")
    # 保存CSV
    df_processed = pd.DataFrame({
        "image_paths": processed_paths,
        "species": processed_species
    })
    df_processed.to_csv(CSV_SAVE_DIR / csv_name, index=False)
    print(f"[INFO] 复杂背景 {save_dir.name} 集最终有效样本数: {len(processed_paths)}")
    return df_processed

# 主流程
if __name__ == "__main__":
    # 加载原始数据并划分
    train_df, val_df, test_df = load_complex_bg_raw()
    print(f"[INFO] 复杂背景原始数据 - 训练集: {len(train_df)}, 验证集: {len(val_df)}, 测试集: {len(test_df)}")
    # 处理并保存各子集
    process_and_save_complex_bg(train_df, PROCESSED_COMPLEX_DIR / "train", "complex_bg_train.csv")
    process_and_save_complex_bg(val_df, PROCESSED_COMPLEX_DIR / "val", "complex_bg_val.csv")
    process_and_save_complex_bg(test_df, PROCESSED_COMPLEX_DIR / "test", "complex_bg_test.csv")
    print("[DONE] 复杂背景数据处理完成")