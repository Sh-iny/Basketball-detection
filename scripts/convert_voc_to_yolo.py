"""
VOC格式转YOLO格式脚本
将pp数据集从VOC格式转换为YOLO格式
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import random

def convert_voc_to_yolo(voc_path, output_path, class_names, train_ratio=0.9):
    """
    将VOC格式数据集转换为YOLO格式
    """
    voc_path = Path(voc_path)
    output_path = Path(output_path)

    # 创建输出目录
    for split in ['train', 'valid']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    # 读取图片列表
    images_dir = voc_path / 'JPEGImages'
    annotations_dir = voc_path / 'Annotations'

    image_files = list(images_dir.glob('*.jpeg')) + list(images_dir.glob('*.jpg'))
    print(f"找到 {len(image_files)} 张图片")

    # 随机划分训练集和验证集
    random.shuffle(image_files)
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    valid_files = image_files[split_idx:]

    print(f"训练集: {len(train_files)} 张")
    print(f"验证集: {len(valid_files)} 张")

    # 类别映射
    class_to_id = {name: i for i, name in enumerate(class_names)}

    def process_files(files, split):
        for img_path in files:
            # 解析XML标注
            xml_path = annotations_dir / (img_path.stem + '.xml')
            if not xml_path.exists():
                print(f"警告: 找不到标注文件 {xml_path}")
                continue

            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 获取图片尺寸
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)

            # 转换标注
            yolo_labels = []
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name not in class_to_id:
                    continue

                class_id = class_to_id[class_name]
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)

                # 转换为YOLO格式 (x_center, y_center, w, h) 归一化
                x_center = (xmin + xmax) / 2 / width
                y_center = (ymin + ymax) / 2 / height
                w = (xmax - xmin) / width
                h = (ymax - ymin) / height

                yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

            # 复制图片
            dst_img = output_path / split / 'images' / img_path.name
            shutil.copy(img_path, dst_img)

            # 保存YOLO标注
            label_path = output_path / split / 'labels' / (img_path.stem + '.txt')
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_labels))

    process_files(train_files, 'train')
    process_files(valid_files, 'valid')

    # 创建data.yaml
    yaml_content = f"""path: {output_path.absolute()}
train: train/images
val: valid/images

nc: {len(class_names)}
names: {class_names}
"""
    with open(output_path / 'data.yaml', 'w') as f:
        f.write(yaml_content)

    print(f"\n转换完成！")
    print(f"输出目录: {output_path}")


if __name__ == '__main__':
    convert_voc_to_yolo(
        voc_path='/data/sui/basketball/datasets/pp/basketball/basketball',
        output_path='/data/sui/basketball/datasets/ball_dataset3',
        class_names=['basketball'],  # 匹配XML中的类别名
        train_ratio=0.9
    )
