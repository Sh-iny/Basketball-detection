"""
数据集合并脚本
将 ball_classify_basket 数据集合并到 Basketball Computer Vision Model

类别对应:
  ball_classify_basket        ->  Basketball Computer Vision Model
  0: ball                     ->  0: ball
  1: basket                   ->  2: rim
  2: people                   ->  1: human
"""

import os
import shutil
import random
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Paths
source_train_images = Path(r"e:\Code\Basketball\dataset\ball_classify_basket.v1i.yolo26\train\images")
source_train_labels = Path(r"e:\Code\Basketball\dataset\ball_classify_basket.v1i.yolo26\train\labels")

target_train_images = Path(r"e:\Code\Basketball\Basketball Computer Vision Model\train\images")
target_train_labels = Path(r"e:\Code\Basketball\Basketball Computer Vision Model\train\labels")
target_valid_images = Path(r"e:\Code\Basketball\Basketball Computer Vision Model\valid\images")
target_valid_labels = Path(r"e:\Code\Basketball\Basketball Computer Vision Model\valid\labels")

# Source classes: ['ball', 'basket', 'people'] -> [0, 1, 2]
# Target classes: ['ball', 'human', 'rim'] -> [0, 1, 2]
# Mapping: 0->0, 1->2, 2->1
class_mapping = {0: 0, 1: 2, 2: 1}

print("=" * 60)
print("篮球数据集合并脚本")
print("=" * 60)
print()

# Step 1: Count existing files
print("步骤 1: 统计现有数据集")
existing_train_images = list(target_train_images.glob("*.jpg")) + list(target_train_images.glob("*.png"))
existing_valid_images = list(target_valid_images.glob("*.jpg")) + list(target_valid_images.glob("*.png"))
print(f"  现有 train 图片数: {len(existing_train_images)}")
print(f"  现有 valid 图片数: {len(existing_valid_images)}")
print()

# Step 2: Copy source files to train with class mapping
print("步骤 2: 合并源数据集到 train")
source_image_files = list(source_train_images.glob("*.jpg")) + list(source_train_images.glob("*.png"))
print(f"  源数据集图片数: {len(source_image_files)}")

copied_count = 0
for img_path in source_image_files:
    # Copy image
    shutil.copy2(img_path, target_train_images / img_path.name)

    # Process label file
    label_path = source_train_labels / (img_path.stem + ".txt")
    target_label_path = target_train_labels / (img_path.stem + ".txt")

    if label_path.exists():
        new_lines = []
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    if class_id in class_mapping:
                        new_class_id = class_mapping[class_id]
                        parts[0] = str(new_class_id)
                        new_lines.append(' '.join(parts))

        with open(target_label_path, 'w', encoding='utf-8') as f:
            for line in new_lines:
                f.write(line + '\n')

    copied_count += 1
    if copied_count % 100 == 0:
        print(f"    已复制 {copied_count}/{len(source_image_files)} 张图片...")

print(f"  已复制 {copied_count} 张图片到 train")
print()

# Step 3: Split 20% from train to valid
print("步骤 3: 从 train 分割 20% 到 valid")
all_train_images = list(target_train_images.glob("*.jpg")) + list(target_train_images.glob("*.png"))
print(f"  合并后 train 图片总数: {len(all_train_images)}")

# Random shuffle
random.shuffle(all_train_images)

# Calculate 20%
split_count = int(len(all_train_images) * 0.2)
print(f"  将分割 {split_count} 张图片 (20%) 到 valid")

moved_count = 0
for img_path in all_train_images[:split_count]:
    # Move image
    shutil.move(str(img_path), target_valid_images / img_path.name)

    # Move label
    label_path = target_train_labels / (img_path.stem + ".txt")
    if label_path.exists():
        shutil.move(str(label_path), target_valid_labels / label_path.name)

    moved_count += 1
    if moved_count % 100 == 0:
        print(f"    已移动 {moved_count}/{split_count} 张图片...")

print(f"  已移动 {moved_count} 张图片到 valid")
print()

# Step 4: Final statistics
print("=" * 60)
print("最终数据集统计")
print("=" * 60)
final_train_images = list(target_train_images.glob("*.jpg")) + list(target_train_images.glob("*.png"))
final_valid_images = list(target_valid_images.glob("*.jpg")) + list(target_valid_images.glob("*.png"))
total = len(final_train_images) + len(final_valid_images)

print(f"  Train 图片数: {len(final_train_images)} ({len(final_train_images)/total*100:.1f}%)")
print(f"  Valid 图片数: {len(final_valid_images)} ({len(final_valid_images)/total*100:.1f}%)")
print(f"  总计: {total} 张图片")
print()
print("完成！类别映射:")
print("  ball (0) -> ball (0)")
print("  basket (1) -> rim (2)")
print("  people (2) -> human (1)")
print("=" * 60)
