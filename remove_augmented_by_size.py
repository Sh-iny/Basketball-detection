"""
移除数据增强图片脚本（按文件大小判断）
对于同一张原图的多个增强版本，保留文件最大的那张（通常是原图），删除其余的
"""

import re
from pathlib import Path
from collections import defaultdict

# 数据集根目录
DATASET_ROOT = Path("e:/Code/Basketball/br/basketball_and_hoop2.v4i.yolo26")

# 支持的图片格式
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def extract_original_name(filename: str) -> str:
    """
    从文件名中提取原图名称
    例如:
      -100_jpg.rf.xxx.jpg -> -100
      9_png_jpg.rf.xxx.jpg -> 9
      camcourt2_xxx_png.rf.xxx.jpg -> camcourt2_xxx
    """
    # 匹配 .rf.{hash} 部分，提取前面的原图名
    match = re.match(r'^(.+?)\.rf\.[a-f0-9]+', filename)
    if match:
        name = match.group(1)
        # 去掉常见的格式后缀
        name = re.sub(r'(_png)?_jpg$', '', name)
        name = re.sub(r'_png$', '', name)
        return name
    return filename


def find_duplicates(images_dir: Path) -> dict[str, list[Path]]:
    """找出同一原图的所有增强版本"""
    groups = defaultdict(list)

    if not images_dir.exists():
        return groups

    for img_file in images_dir.iterdir():
        if img_file.suffix.lower() in IMAGE_EXTENSIONS:
            original_name = extract_original_name(img_file.name)
            groups[original_name].append(img_file)

    return groups


def process_subset(subset_dir: Path) -> tuple[list[Path], list[Path]]:
    """处理一个子集，返回要删除的图片和标注列表"""
    images_dir = subset_dir / "images"
    labels_dir = subset_dir / "labels"

    images_to_delete = []
    labels_to_delete = []

    groups = find_duplicates(images_dir)

    for original_name, files in groups.items():
        if len(files) > 1:
            # 按文件大小降序排序，保留最大的
            files_sorted = sorted(files, key=lambda f: f.stat().st_size, reverse=True)
            keep = files_sorted[0]

            for img_file in files_sorted[1:]:
                images_to_delete.append(img_file)
                label_file = labels_dir / (img_file.stem + ".txt")
                if label_file.exists():
                    labels_to_delete.append(label_file)

    return images_to_delete, labels_to_delete


def main():
    subsets = ['train', 'valid', 'test']

    all_images = []
    all_labels = []

    for subset in subsets:
        subset_dir = DATASET_ROOT / subset
        if not subset_dir.exists():
            continue

        images, labels = process_subset(subset_dir)

        if images:
            print(f"[{subset}] 发现 {len(images)} 张增强图片需要删除")
            all_images.extend(images)
            all_labels.extend(labels)

    if not all_images:
        print("未发现需要删除的增强图片")
        return

    print(f"\n总计:")
    print(f"  - 图片: {len(all_images)} 个")
    print(f"  - 标注: {len(all_labels)} 个")

    confirm = input("\n是否删除这些文件? (y/n): ").strip().lower()

    if confirm == 'y':
        for f in all_images:
            f.unlink()
        for f in all_labels:
            f.unlink()
        print(f"\n完成，共删除 {len(all_images)} 张图片和 {len(all_labels)} 个标注")
    else:
        print("已取消删除操作")


if __name__ == "__main__":
    main()
