"""
计算train目录中文件的总数
"""
from pathlib import Path

def count_train_files(dataset_dir):
    """
    计算train目录中文件的总数
    
    Args:
        dataset_dir: 数据集根目录路径
    """
    print(f"=" * 80)
    print(f"计算train目录文件数量: {dataset_dir}")
    print(f"=" * 80)
    
    dataset_path = Path(dataset_dir)
    train_images_dir = dataset_path / "train" / "images"
    train_labels_dir = dataset_path / "train" / "labels"
    
    if not train_images_dir.exists():
        print(f"❌ train图片目录不存在: {train_images_dir}")
        return 0
    
    if not train_labels_dir.exists():
        print(f"❌ train标签目录不存在: {train_labels_dir}")
        return 0
    
    # 统计图片文件
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(train_images_dir.glob(f"*{ext}"))
    
    # 统计标注文件
    label_files = list(train_labels_dir.glob("*.txt"))
    
    total_images = len(image_files)
    total_labels = len(label_files)
    
    print(f"📊 train目录文件统计:")
    print(f"   - 图片文件: {total_images}")
    print(f"   - 标注文件: {total_labels}")
    print(f"   - 总计: {total_images + total_labels}")
    
    if total_images != total_labels:
        print(f"\n❌ 图片和标注文件数量不一致！")
    else:
        print(f"\n✅ 图片和标注文件数量一致")
    
    return total_images

if __name__ == "__main__":
    dataset_dir = "merged_basketball_dataset"
    count_train_files(dataset_dir)
