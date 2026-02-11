"""
检查合并后的数据集中 train 和 val 目录是否有重复文件
"""
from pathlib import Path
from collections import defaultdict

def check_duplicates(dataset_dir):
    """检查 train 和 val 目录中的重复文件"""
    train_images = dataset_dir / "train" / "images"
    val_images = dataset_dir / "valid" / "images"

    # 提取所有基础文件名
    train_files = set()
    val_files = set()

    # 读取 train 目录
    if train_images.exists():
        for img_path in train_images.glob("*.jpg"):
            # 提取基础文件名（去掉数据集前缀）
            # 格式: {dataset_name}_{original_name}
            parts = img_path.name.split('_', 1)
            if len(parts) == 2:
                base_name = parts[1]
                train_files.add(base_name)

    # 读取 val 目录
    if val_images.exists():
        for img_path in val_images.glob("*.jpg"):
            parts = img_path.name.split('_', 1)
            if len(parts) == 2:
                base_name = parts[1]
                val_files.add(base_name)

    # 找出交集
    duplicates = train_files & val_files

    return duplicates, len(train_files), len(val_files)

def main():
    dataset_dir = Path("merged_basketball_dataset")

    print("=" * 70)
    print("检查重复文件")
    print("=" * 70)

    duplicates, train_count, val_count = check_duplicates(dataset_dir)

    print(f"\n📊 统计:")
    print(f"   - train 目录文件数: {train_count}")
    print(f"   - val 目录文件数: {val_count}")
    print(f"   - 重复文件数: {len(duplicates)}")

    if duplicates:
        print(f"\n⚠️  发现重复文件:")
        for i, dup in enumerate(list(duplicates)[:10]):
            print(f"   {i+1}. {dup}")
        if len(duplicates) > 10:
            print(f"   ... 还有 {len(duplicates) - 10} 个重复文件")
    else:
        print("\n✅ 未发现重复文件！")

    print(f"\n📁 检查目录:")
    print(f"   {dataset_dir}")

if __name__ == "__main__":
    main()
