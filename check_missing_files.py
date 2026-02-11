"""
检查训练目录中图片和标注文件的数量是否一致
"""
from pathlib import Path

def check_files(dataset_dir, split="train"):
    """检查指定 split 目录中图片和标注文件的一致性"""
    images_dir = dataset_dir / split / "images"
    labels_dir = dataset_dir / split / "labels"

    # 读取所有图片文件
    image_files = set()
    for img_path in images_dir.glob("*.jpg"):
        image_files.add(img_path.stem)

    # 读取所有标签文件
    label_files = set()
    for label_path in labels_dir.glob("*.txt"):
        label_files.add(label_path.stem)

    # 找出缺失的文件
    images_without_labels = image_files - label_files
    labels_without_images = label_files - image_files

    return {
        "total_images": len(image_files),
        "total_labels": len(label_files),
        "images_without_labels": images_without_labels,
        "labels_without_images": labels_without_images
    }

def main():
    dataset_dir = Path("merged_basketball_dataset")

    print("=" * 70)
    print("检查文件一致性")
    print("=" * 70)

    # 检查 train 目录
    train_stats = check_files(dataset_dir, "train")
    print(f"\n📊 Train 目录:")
    print(f"   - 图片文件: {train_stats['total_images']} 个")
    print(f"   - 标注文件: {train_stats['total_labels']} 个")
    print(f"   - 无标注的图片: {len(train_stats['images_without_labels'])} 个")
    print(f"   - 无图片的标注: {len(train_stats['labels_without_images'])} 个")

    # 显示前10个无标注的图片
    if train_stats['images_without_labels']:
        print(f"\n   无标注的图片（前10个）:")
        for i, img_name in enumerate(list(train_stats['images_without_labels'])[:10]):
            print(f"     {i+1}. {img_name}.jpg")

    # 检查 valid 目录
    valid_stats = check_files(dataset_dir, "valid")
    print(f"\n📊 Valid 目录:")
    print(f"   - 图片文件: {valid_stats['total_images']} 个")
    print(f"   - 标注文件: {valid_stats['total_labels']} 个")
    print(f"   - 无标注的图片: {len(valid_stats['images_without_labels'])} 个")
    print(f"   - 无图片的标注: {len(valid_stats['labels_without_images'])} 个")

    # 检查 test 目录
    test_stats = check_files(dataset_dir, "test")
    print(f"\n📊 Test 目录:")
    print(f"   - 图片文件: {test_stats['total_images']} 个")
    print(f"   - 标注文件: {test_stats['total_labels']} 个")
    print(f"   - 无标注的图片: {len(test_stats['images_without_labels'])} 个")
    print(f"   - 无图片的标注: {len(test_stats['labels_without_images'])} 个")

if __name__ == "__main__":
    main()
