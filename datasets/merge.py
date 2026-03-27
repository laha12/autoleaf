import pandas as pd
from pathlib import Path

# 配置
CSV_DIR = Path("dataset/csv")
# leafsnap标签文件
SINGLE_TRAIN_CSV = CSV_DIR / "leafsnap-dataset-train-images.csv"
SINGLE_VAL_CSV = CSV_DIR / "leafsnap-dataset-val-images.csv"
SINGLE_TEST_CSV = CSV_DIR / "leafsnap-dataset-test-images.csv"
# 复杂背景标签文件
COMPLEX_TRAIN_CSV = CSV_DIR / "complex_bg_train.csv"
COMPLEX_VAL_CSV = CSV_DIR / "complex_bg_val.csv"
COMPLEX_TEST_CSV = CSV_DIR / "complex_bg_test.csv"

# 1. 加载所有标签
def load_all_labels():
    # 加载单一背景标签
    single_train = pd.read_csv(SINGLE_TRAIN_CSV)
    single_train["bg_type"] = "single"  # 标记背景类型
    single_val = pd.read_csv(SINGLE_VAL_CSV)
    single_val["bg_type"] = "single"
    single_test = pd.read_csv(SINGLE_TEST_CSV)
    single_test["bg_type"] = "single"
    
    # 加载复杂背景标签
    complex_train = pd.read_csv(COMPLEX_TRAIN_CSV)
    complex_train["bg_type"] = "complex"
    complex_val = pd.read_csv(COMPLEX_VAL_CSV)
    complex_val["bg_type"] = "complex"
    complex_test = pd.read_csv(COMPLEX_TEST_CSV)
    complex_test["bg_type"] = "complex"
    
    # 合并所有数据
    all_train = pd.concat([single_train, complex_train], ignore_index=True)
    all_val = pd.concat([single_val, complex_val], ignore_index=True)
    all_test = pd.concat([single_test, complex_test], ignore_index=True)
    
    # 生成类别映射（统一所有类别）
    all_species = list(set(all_train["species"]) | set(all_val["species"]) | set(all_test["species"]))
    all_species.sort()
    species2idx = {specie: idx for idx, specie in enumerate(all_species)}
    
    # 增加类别索引列
    all_train["species_idx"] = all_train["species"].map(species2idx)
    all_val["species_idx"] = all_val["species"].map(species2idx)
    all_test["species_idx"] = all_test["species"].map(species2idx)
    
    # 保存类别映射
    pd.DataFrame({
        "species": all_species,
        "idx": range(len(all_species))
    }).to_csv(CSV_DIR / "species_mapping.csv", index=False)
    
    # 保存合并后的标签
    all_train.to_csv(CSV_DIR / "combined_train.csv", index=False)
    all_val.to_csv(CSV_DIR / "combined_val.csv", index=False)
    all_test.to_csv(CSV_DIR / "combined_test.csv", index=False)
    
    print(f"[INFO] 合并后总类别数: {len(all_species)} (原leafsnap 185类 + 自建新增 {len(all_species)-185} 类)")
    print(f"[INFO] 合并后训练集: {len(all_train)} (单一背景 {len(single_train)} + 复杂背景 {len(complex_train)})")
    print(f"[INFO] 合并后验证集: {len(all_val)} (单一背景 {len(single_val)} + 复杂背景 {len(complex_val)})")
    print(f"[INFO] 合并后测试集: {len(all_test)} (单一背景 {len(single_test)} + 复杂背景 {len(complex_test)})")

if __name__ == "__main__":
    load_all_labels()