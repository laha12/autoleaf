import cv2
import pandas as pd
from utils.img_process import load_image_and_preprocess
from pathlib import Path
from sklearn.model_selection import train_test_split

# GLOBAL CONSTANTS
DATA_FILE = "C:\\Users\\86185\\Desktop\\Deep-Leafsnap\\dataset\\single_bg\\raw\\leafsnap-dataset-images.csv"
NUM_CLASSES = 185
bad_lab_species = {'Abies concolor', 'Abies nordmanniana', 'Picea pungens', 'Picea orientalis',
                   'Picea abies', 'Cedrus libani', 'Cedrus atlantica', 'Cedrus deodara',
                   'Juniperus virginiana', 'Tsuga canadensis', 'Larix decidua', 'Pseudolarix amabilis'}

# 读取数据并剔除不良样本
columns = ['file_id', 'image_path', 'segmented_path', 'species', 'source']
data = pd.read_csv(DATA_FILE, names=columns, header=1)

# 剔除在实验室采集的不良样本
bad_indices = []
for i in range(len(data)):
    if ((data.iloc[i]['species'] in bad_lab_species) and (data.iloc[i]['source'].lower() == 'lab')):
        bad_indices.append(i)
data.drop(data.index[bad_indices], inplace=True)

# ===================== 核心修改：拆分训练/验证/测试集 =====================
# 第一步：先拆分出测试集（20%），剩余80%为训练+验证集（保持类别分布）
train_val_df, test_df = train_test_split(
    data, 
    test_size=0.20,       
    random_state=7, 
    stratify=data['species']  
)

# 第二步：从训练+验证集中拆分出验证集
train_df, val_df = train_test_split(
    train_val_df, 
    test_size=0.25,       # 验证集占训练+验证集的25%
    random_state=7, 
    stratify=train_val_df['species']  # 继续分层
)

# ===================== 构造各数据集的图像/标签列表 =====================
# 训练集
images_train_original = train_df['image_path'].tolist()
images_train_segmented = train_df['segmented_path'].tolist()
images_train = {'original': images_train_original, 'segmented': images_train_segmented}
species_train = train_df['species'].tolist()
species_classes_train = sorted(set(species_train))

# 验证集
images_val_original = val_df['image_path'].tolist()
images_val_segmented = val_df['segmented_path'].tolist()
images_val = {'original': images_val_original, 'segmented': images_val_segmented}
species_val = val_df['species'].tolist()
species_classes_val = sorted(set(species_val))

# 测试集
images_test_original = test_df['image_path'].tolist()
images_test_segmented = test_df['segmented_path'].tolist()
images_test = {'original': images_test_original, 'segmented': images_test_segmented}
species_test = test_df['species'].tolist()
species_classes_test = sorted(set(species_test))

# 打印数据集规模
print(f'\n[INFO]  Training Samples  : {len(images_train["original"]):5d}')
print(f'\tValidation Samples: {len(images_val["original"]):5d}')
print(f'\tTesting Samples   : {len(images_test["original"]):5d}')

print('[INFO] Processing Images')

# 复用图片保存函数
def save_images(images, species, directory='train', csv_name='temp.csv'):
    cropped_images = []
    image_species = []
    image_paths = []
    count = 1
    base_write_dir = Path('dataset/single_bg/processed') / directory
    base_write_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path("dataset/csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    for index in range(len(images['original'])):
        image = load_image_and_preprocess(images['original'][index], images['segmented'][index])
        if image is not None:
            # 按物种创建目录
            specie_dir = base_write_dir / species[index].lower().replace(' ', '_')
            specie_dir.mkdir(parents=True, exist_ok=True)

            # 保存图片
            file_name = f"{count}.jpg"
            file_path = specie_dir / file_name
            image_to_write = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(file_path), image_to_write)
            
            # 记录信息
            image_paths.append(str(file_path))
            cropped_images.append(image)
            image_species.append(species[index])
            count += 1
        else:
            print(f'[DEBUG] 图像 {images["original"][index]} 处理失败，跳过')
            continue

        # 每处理1000张打印进度
        if index > 0 and index % 1000 == 0:
            print(f'[INFO] {directory} set - Processed {index:5d} images')

    # 打印最终有效样本数
    print(f'[INFO] Final Number of {directory} Samples: {len(image_paths)}')
    # 保存图片路径和标签到CSV
    raw_data = {'image_paths': image_paths, 'species': image_species}
    df = pd.DataFrame(raw_data, columns=['image_paths', 'species'])
    df.to_csv(csv_path / csv_name, index=False)  # 新增index=False，避免多余索引列

# 分别处理训练/验证/测试集
# save_images(images_train, species_train, directory='train',
#             csv_name='leafsnap-dataset-train-images.csv')
# save_images(images_val, species_val, directory='val',  # 验证集目录为val
#             csv_name='leafsnap-dataset-val-images.csv')
save_images(images_test, species_test, directory='test',
            csv_name='leafsnap-dataset-test-images.csv')

print('\n[DONE]')