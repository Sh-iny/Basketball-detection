"""
合并ball和rim数据集
将两个数据集合并为一个统一的数据集，用于训练YOLOv11模型
"""

import os
import shutil
from pathlib import Path
import yaml


def process_dataset(source_dir, output_dir, class_mapping, prefix):
    """
    处理单个数据集，复制图片并转换标签

    Args:
        source_dir: 源数据集目录
        output_dir: 输出目录
        class_mapping: 类别ID映射字典
        prefix: 文件名前缀
    """
    images_dir = source_dir / 'images'
    labels_dir = source_dir / 'labels'

    if not images_dir.exists():
        print(f"  警告: {images_dir} 不存在，跳过")
        return

    image_files = list(images_dir.glob('*.*'))
    print(f"  处理 {prefix} 数据集: {len(image_files)} 张图片")

    for img_file in image_files:
        # 复制图片，添加前缀避免文件名冲突
        new_img_name = f"{prefix}_{img_file.name}"
        shutil.copy2(img_file, output_dir / 'images' / new_img_name)

        # 处理标签文件
        label_file = labels_dir / f"{img_file.stem}.txt"
        if label_file.exists():
            new_label_name = f"{prefix}_{img_file.stem}.txt"
            convert_label_file(label_file, output_dir / 'labels' / new_label_name, class_mapping)


def convert_label_file(input_label, output_label, class_mapping):
    """
    转换标签文件的类别ID

    Args:
        input_label: 输入标签文件路径
        output_label: 输出标签文件路径
        class_mapping: 类别ID映射字典
    """
    with open(input_label, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            old_class_id = int(parts[0])
            new_class_id = class_mapping.get(old_class_id, old_class_id)
            new_line = f"{new_class_id} {' '.join(parts[1:])}\n"
            new_lines.append(new_line)

    with open(output_label, 'w') as f:
        f.writelines(new_lines)


def merge_datasets(ball_dir, rim_dir, output_dir):
    """
    合并ball和rim数据集

    Args:
        ball_dir: ball数据集路径
        rim_dir: rim数据集路径
        output_dir: 输出合并后的数据集路径
    """
    print("开始合并数据集...")

    # 创建输出目录结构
    output_path = Path(output_dir)
    for split in ['train', 'valid', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    # 新的类别映射
    # ball数据集的5个类别 + rim数据集的1个类别
    new_classes = {
        'ball': 0,      # 合并所有球类为一个类别
        'rim': 1        # 篮筐
    }

    # ball数据集中的类别ID映射到新的ball类别(0)
    ball_class_mapping = {
        0: 0,  # Basketball -> ball
        1: 0,  # Blue-Basketball -> ball
        2: 0,  # Red-Basketball -> ball
        3: 0,  # Volleyball -> ball
        4: 0   # football -> ball
    }

    # rim数据集中的类别ID映射到新的rim类别(1)
    rim_class_mapping = {
        0: 1   # '3' -> rim
    }

    # 处理每个数据集分割
    for split in ['train', 'valid', 'test']:
        print(f"\n处理 {split} 集...")

        # 处理ball数据集
        process_dataset(
            source_dir=Path(ball_dir) / split,
            output_dir=output_path / split,
            class_mapping=ball_class_mapping,
            prefix='ball'
        )

        # 处理rim数据集
        process_dataset(
            source_dir=Path(rim_dir) / split,
            output_dir=output_path / split,
            class_mapping=rim_class_mapping,
            prefix='rim'
        )

    # 创建新的data.yaml文件
    data_yaml = {
        'path': str(output_path.absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 2,
        'names': ['ball', 'rim']
    }

    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"\n数据集合并完成！")
    print(f"输出路径: {output_path}")
    print(f"配置文件: {yaml_path}")


def main():
    """主函数"""
    # 数据集路径
    ball_dir = '/data/sui/basketball/ball'
    rim_dir = '/data/sui/basketball/rim'
    output_dir = '/data/sui/basketball/merged_dataset'

    # 合并数据集
    merge_datasets(ball_dir, rim_dir, output_dir)

    # 统计信息
    print("\n=== 数据集统计 ===")
    output_path = Path(output_dir)
    for split in ['train', 'valid', 'test']:
        img_count = len(list((output_path / split / 'images').glob('*.*')))
        label_count = len(list((output_path / split / 'labels').glob('*.txt')))
        print(f"{split:6s}: {img_count:4d} 张图片, {label_count:4d} 个标签文件")


if __name__ == '__main__':
    main()

