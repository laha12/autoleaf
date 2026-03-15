import cv2
import pandas as pd
from utils.img_process import load_image_and_preprocess
from pathlib import Path
from sklearn.model_selection import train_test_split

# GLOBAL CONSTANTS
DATA_FILE = 'leafsnap-dataset-images.csv'
NUM_CLASSES = 185
bad_lab_species = {'Abies concolor', 'Abies nordmanniana', 'Picea pungens', 'Picea orientalis',
                       'Picea abies', 'Cedrus libani', 'Cedrus atlantica', 'Cedrus deodara',
                       'Juniperus virginiana', 'Tsuga canadensis', 'Larix decidua', 'Pseudolarix amabilis'}

columns = ['file_id', 'image_path', 'segmented_path', 'species', 'source']
data = pd.read_csv(DATA_FILE, names=columns, header=1)

# 剔除在实验室采集的不良样本
bad_indices = []
for i in range(len(data)):
    if ((data.iloc[i]['species'] in bad_lab_species) and (data.iloc[i]['source'].lower() == 'lab')):
        bad_indices.append(i)
data.drop(data.index[bad_indices], inplace=True)

# 划分训练集和测试集 并确保训练集和测试集的类别分布与原始数据一致
train_df, test_df = train_test_split(data, test_size=0.20, random_state=7, stratify=data['species'])

images_train_original = train_df['image_path'].tolist()
images_train_segmented = train_df['segmented_path'].tolist()
images_train = {'original': images_train_original, 'segmented': images_train_segmented}
species_train = train_df['species'].tolist()
species_classes_train = sorted(set(species_train))

images_test_original = test_df['image_path'].tolist()
images_test_segmented = test_df['segmented_path'].tolist()
images_test = {'original': images_test_original, 'segmented': images_test_segmented}
species_test = test_df['species'].tolist()
species_classes_test = sorted(set(species_test))

print(f'\n[INFO]  Training Samples : {len(images_train["original"]):5d}')
print(f'\tTesting Samples  : {len(images_test["original"]):5d}')

print('[INFO] Processing Images')


def save_images(images, species, directory='train', csv_name='temp.csv'):
    cropped_images = []
    image_species = []
    image_paths = []
    count = 1
    base_write_dir = Path('dataset') / directory
    base_write_dir.mkdir(parents=True, exist_ok=True)

    for index in range(len(images['original'])):
        image = load_image_and_preprocess(images['original'][index], images['segmented'][index])
        if image is not None:
            specie_dir = base_write_dir / species[index].lower().replace(' ', '_')
            specie_dir.mkdir(parents=True, exist_ok=True)

            file_name = f"{count}.jpg"
            file_path = specie_dir / file_name
            image_to_write = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(file_path), image_to_write)
            image_paths.append(str(file_path))
            cropped_images.append(image)
            image_species.append(species[index])
            count += 1
        else:
            print(f'[DEBUG] 图像 {images["original"][index]} 处理失败，跳过')
            continue

        if index > 0 and index % 1000 == 0:
            print('[INFO] Processed {:5d} images'.format(index))

    print(f'[INFO] Final Number of {directory} Samples: {len(image_paths)}')
    # 处理后所有图片的路径和对应的标签
    raw_data = {'image_paths': image_paths,
                'species': image_species}
    df = pd.DataFrame(raw_data, columns = ['image_paths', 'species'])
    df.to_csv(csv_name)

save_images(images_train, species_train, directory='train',
            csv_name='leafsnap-dataset-train-images.csv')
save_images(images_test, species_test, directory='test',
            csv_name='leafsnap-dataset-test-images.csv')

print('\n[DONE]')
