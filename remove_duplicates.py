"""
检测并删除 train 和 valid 之间的重复图片
保留 train 中的版本，删除 valid 中的重复项
"""

import shutil
from pathlib import Path

# Paths
root = Path(r"e:\Code\Basketball\Basketball Computer Vision Model")
train_images = root / "train" / "images"
train_labels = root / "train" / "labels"
valid_images = root / "valid" / "images"
valid_labels = root / "valid" / "labels"

print("=" * 60)
print("检测并删除重复图片")
print("=" * 60)
print()

# Get all image filenames from both sets
train_img_files = {f.name for f in train_images.glob("*.jpg")} | {f.name for f in train_images.glob("*.png")}
valid_img_files = {f.name for f in valid_images.glob("*.jpg")} | {f.name for f in valid_images.glob("*.png")}

print(f"Train 图片数: {len(train_img_files)}")
print(f"Valid 图片数: {len(valid_img_files)}")
print()

# Find duplicates (files that exist in both train and valid)
duplicates = train_img_files & valid_img_files

if duplicates:
    print(f"发现 {len(duplicates)} 个重复文件:")
    for name in sorted(duplicates):
        print(f"  - {name}")
    print()

    # Remove duplicates from valid (keep in train)
    deleted_count = 0
    for name in duplicates:
        # Remove image
        img_path = valid_images / name
        if img_path.exists():
            img_path.unlink()
            deleted_count += 1

        # Remove corresponding label
        label_path = valid_labels / (Path(name).stem + ".txt")
        if label_path.exists():
            label_path.unlink()
            deleted_count += 1

    print(f"已从 valid 中删除 {len(duplicates)} 个重复文件")
    print()
else:
    print("未发现重复文件")
    print()

# Final statistics
final_train_images = list(train_images.glob("*.jpg")) + list(train_images.glob("*.png"))
final_valid_images = list(valid_images.glob("*.jpg")) + list(valid_images.glob("*.png"))
total = len(final_train_images) + len(final_valid_images)

print("=" * 60)
print("清理后数据集统计")
print("=" * 60)
print(f"Train 图片数: {len(final_train_images)} ({len(final_train_images)/total*100:.1f}%)")
print(f"Valid 图片数: {len(final_valid_images)} ({len(final_valid_images)/total*100:.1f}%)")
print(f"总计: {total} 张图片")
print("=" * 60)
