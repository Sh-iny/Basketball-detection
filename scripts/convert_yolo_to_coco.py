"""
YOLO格式转COCO格式脚本
将ball_dataset2从YOLO格式转换为COCO格式，供RF-DETR训练使用
"""

import json
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def convert_yolo_to_coco(yolo_dataset_path, output_path, class_names):
    """
    将YOLO格式数据集转换为COCO格式

    Args:
        yolo_dataset_path: YOLO数据集路径
        output_path: 输出COCO数据集路径
        class_names: 类别名称列表
    """
    yolo_path = Path(yolo_dataset_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # 处理每个数据集分割
    for split in ['train', 'valid', 'test']:
        print(f"\n处理 {split} 集...")

        images_dir = yolo_path / split / 'images'
        labels_dir = yolo_path / split / 'labels'

        if not images_dir.exists():
            print(f"  跳过 {split}：目录不存在")
            continue

        # 创建输出目录
        out_images_dir = output_path / split
        out_images_dir.mkdir(parents=True, exist_ok=True)

        # COCO格式结构
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [{"id": i, "name": name} for i, name in enumerate(class_names)]
        }

        annotation_id = 1
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))

        for img_id, img_path in enumerate(tqdm(image_files, desc=f"  转换 {split}")):
            # 获取图像信息
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"  警告：无法读取图像 {img_path}: {e}")
                continue

            # 添加图像信息
            coco_data["images"].append({
                "id": img_id,
                "file_name": img_path.name,
                "width": width,
                "height": height
            })

            # 创建软链接到输出目录
            out_img_path = out_images_dir / img_path.name
            if not out_img_path.exists():
                os.symlink(img_path.absolute(), out_img_path)

            # 读取YOLO标注
            label_path = labels_dir / (img_path.stem + '.txt')
            if not label_path.exists():
                continue

            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    bbox_width = float(parts[3])
                    bbox_height = float(parts[4])

                    # YOLO格式转COCO格式 (x_center, y_center, w, h) -> (x, y, w, h)
                    x = (x_center - bbox_width / 2) * width
                    y = (y_center - bbox_height / 2) * height
                    w = bbox_width * width
                    h = bbox_height * height

                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": img_id,
                        "category_id": class_id,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0
                    })
                    annotation_id += 1

        # 保存COCO格式标注
        # RF-DETR期望的文件名格式
        if split == 'valid':
            json_name = 'val.json'
        else:
            json_name = f'{split}.json'

        json_path = output_path / json_name
        with open(json_path, 'w') as f:
            json.dump(coco_data, f)

        print(f"  完成：{len(coco_data['images'])} 张图像，{len(coco_data['annotations'])} 个标注")
        print(f"  保存到：{json_path}")

    # 创建data.yaml
    data_yaml = f"""# RF-DETR 数据集配置文件
# COCO格式数据集 - 篮球检测

path: {output_path.absolute()}
train: train
val: valid
test: test

nc: {len(class_names)}

names:
"""
    for i, name in enumerate(class_names):
        data_yaml += f"  {i}: {name}\n"

    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(data_yaml)

    print(f"\n配置文件保存到：{yaml_path}")


if __name__ == '__main__':
    # 转换ball_dataset2
    convert_yolo_to_coco(
        yolo_dataset_path='/data/sui/basketball/datasets/ball_dataset2',
        output_path='/data/sui/basketball/datasets/ball_dataset2_coco',
        class_names=['ball']
    )

    print("\n转换完成！")
