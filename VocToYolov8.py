import os
import shutil
import random
import xml.etree.ElementTree as ET
from tqdm import tqdm

# VOC格式数据集路径
voc_data_path = 'MyDataset'
voc_annotations_path = os.path.join(voc_data_path, 'Annotations')
voc_images_path = os.path.join(voc_data_path, 'JPEGImages')

# YOLO格式数据集保存路径
yolo_data_path = 'dataset/test-YOLO'
yolo_images_path = os.path.join(yolo_data_path, 'images')
yolo_labels_path = os.path.join(yolo_data_path, 'labels')

# 创建YOLO格式数据集目录
os.makedirs(yolo_images_path, exist_ok=True)
os.makedirs(yolo_labels_path, exist_ok=True)

# 类别映射 (可以根据自己的数据集进行调整)
class_mapping = {
    'person': 0,
}


def convert_voc_to_yolo(voc_annotation_file, yolo_label_file):
    tree = ET.parse(voc_annotation_file)
    root = tree.getroot()

    size = root.find('size')
    width = float(size.find('width').text)
    height = float(size.find('height').text)

    # 添加有效性检查
    if width == 0 or height == 0:
        print(f"警告：{voc_annotation_file} 的 width 或 height 为 0，跳过此文件。")
        return

    with open(yolo_label_file, 'w') as f:
        for obj in root.findall('object'):
            cls = obj.find('name').text
            if cls not in class_mapping:
                continue
            cls_id = class_mapping[cls]
            xmlbox = obj.find('bndbox')
            xmin = float(xmlbox.find('xmin').text)
            ymin = float(xmlbox.find('ymin').text)
            xmax = float(xmlbox.find('xmax').text)
            ymax = float(xmlbox.find('ymax').text)



            x_center = (xmin + xmax) / 2.0 / width
            y_center = (ymin + ymax) / 2.0 / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height

            f.write(f"{cls_id} {x_center} {y_center} {w} {h}\n")


# 遍历VOC数据集的Annotations目录，进行转换
print("开始VOC到YOLO格式转换...")
for voc_annotation in tqdm(os.listdir(voc_annotations_path)):
    if voc_annotation.endswith('.xml'):
        voc_annotation_file = os.path.join(voc_annotations_path, voc_annotation)
        image_id = os.path.splitext(voc_annotation)[0]
        voc_image_file = os.path.join(voc_images_path, f"{image_id}.jpg")
        yolo_label_file = os.path.join(yolo_labels_path, f"{image_id}.txt")
        yolo_image_file = os.path.join(yolo_images_path, f"{image_id}.jpg")

        convert_voc_to_yolo(voc_annotation_file, yolo_label_file)
        if os.path.exists(voc_image_file):
            shutil.copy(voc_image_file, yolo_image_file)

print("VOC到YOLO格式转换完成！")

# 划分数据集
train_images_path = os.path.join(yolo_data_path, 'train', 'images')
train_labels_path = os.path.join(yolo_data_path, 'train', 'labels')
val_images_path = os.path.join(yolo_data_path, 'val', 'images')
val_labels_path = os.path.join(yolo_data_path, 'val', 'labels')
test_images_path = os.path.join(yolo_data_path, 'test', 'images')
test_labels_path = os.path.join(yolo_data_path, 'test', 'labels')

os.makedirs(train_images_path, exist_ok=True)
os.makedirs(train_labels_path, exist_ok=True)
os.makedirs(val_images_path, exist_ok=True)
os.makedirs(val_labels_path, exist_ok=True)
os.makedirs(test_images_path, exist_ok=True)
os.makedirs(test_labels_path, exist_ok=True)

# 获取所有图片文件名（不包含扩展名）
image_files = [f[:-4] for f in os.listdir(yolo_images_path) if f.endswith('.jpg')]

# 随机打乱文件顺序
random.shuffle(image_files)

# 划分数据集比例
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

train_count = int(train_ratio * len(image_files))
val_count = int(val_ratio * len(image_files))
test_count = len(image_files) - train_count - val_count

train_files = image_files[:train_count]
val_files = image_files[train_count:train_count + val_count]
test_files = image_files[train_count + val_count:]


# 移动文件到相应的目录
def move_files(files, src_images_path, src_labels_path, dst_images_path, dst_labels_path):
    for file in tqdm(files):
        src_image_file = os.path.join(src_images_path, f"{file}.jpg")
        src_label_file = os.path.join(src_labels_path, f"{file}.txt")
        dst_image_file = os.path.join(dst_images_path, f"{file}.jpg")
        dst_label_file = os.path.join(dst_labels_path, f"{file}.txt")

        if os.path.exists(src_image_file) and os.path.exists(src_label_file):
            shutil.move(src_image_file, dst_image_file)
            shutil.move(src_label_file, dst_label_file)


# 移动训练集文件
print("移动训练集文件...")
move_files(train_files, yolo_images_path, yolo_labels_path, train_images_path, train_labels_path)
# 移动验证集文件
print("移动验证集文件...")
move_files(val_files, yolo_images_path, yolo_labels_path, val_images_path, val_labels_path)
# 移动测试集文件
print("移动测试集文件...")
move_files(test_files, yolo_images_path, yolo_labels_path, test_images_path, test_labels_path)

print("数据集划分完成！")

# 删除原始的 images 和 labels 文件夹
shutil.rmtree(yolo_images_path)
shutil.rmtree(yolo_labels_path)

print("原始 images 和 labels 文件夹删除完成！")
