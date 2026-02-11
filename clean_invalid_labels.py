"""
清理无效标注文件脚本
检测并删除没有对应图片的标注文件
"""

import os
from pathlib import Path

# 数据集根目录
DATASET_ROOT = Path("e:/Code/Basketball/merged_basketball_dataset")

# 支持的图片格式
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def find_invalid_labels(images_dir: Path, labels_dir: Path) -> list[Path]:
    """找出没有对应图片的标注文件"""
    invalid_labels = []

    if not labels_dir.exists():
        return invalid_labels

    # 获取所有图片文件名（不含扩展名）
    image_stems = set()
    if images_dir.exists():
        for img_file in images_dir.iterdir():
            if img_file.suffix.lower() in IMAGE_EXTENSIONS:
                image_stems.add(img_file.stem)

    # 检查每个标注文件是否有对应图片
    for label_file in labels_dir.iterdir():
        if label_file.suffix.lower() == '.txt':
            if label_file.stem not in image_stems:
                invalid_labels.append(label_file)

    return invalid_labels

def main():
    # 处理 train, valid, test 三个子集
    subsets = ['train', 'valid', 'test']

    all_invalid = []

    for subset in subsets:
        images_dir = DATASET_ROOT / subset / "images"
        labels_dir = DATASET_ROOT / subset / "labels"

        invalid = find_invalid_labels(images_dir, labels_dir)

        if invalid:
            print(f"\n[{subset}] 发现 {len(invalid)} 个无效标注文件:")
            for f in invalid:
                print(f"  - {f.name}")
            all_invalid.extend(invalid)

    if not all_invalid:
        print("未发现无效标注文件，数据集完整。")
        return

    print(f"\n共发现 {len(all_invalid)} 个无效标注文件")

    # 确认删除
    confirm = input("\n是否删除这些文件? (y/n): ").strip().lower()

    if confirm == 'y':
        for f in all_invalid:
            f.unlink()
            print(f"已删除: {f}")
        print(f"\n完成，共删除 {len(all_invalid)} 个文件")
    else:
        print("已取消删除操作")

if __name__ == "__main__":
    main()