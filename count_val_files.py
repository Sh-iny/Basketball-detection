"""
计算val目录中文件的总数
"""
from pathlib import Path

def count_val_files(dataset_dir):
    """
    计算val目录中文件的总数
    
    Args:
        dataset_dir: 数据集根目录路径
    """
    print(f"=" * 80)
    print(f"计算val目录文件数量: {dataset_dir}")
    print(f"=" * 80)
    
    dataset_path = Path(dataset_dir)
    val_images_dir = dataset_path / "valid" / "images"
    val_labels_dir = dataset_path / "valid" / "labels"
    
    if not val_images_dir.exists():
        print(f"❌ val图片目录不存在: {val_images_dir}")
        return 0
    
    if not val_labels_dir.exists():
        print(f"❌ val标签目录不存在: {val_labels_dir}")
        return 0
    
    # 统计图片文件
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(val_images_dir.glob(f"*{ext}"))
    
    # 统计标注文件
    label_files = list(val_labels_dir.glob("*.txt"))
    
    total_images = len(image_files)
    total_labels = len(label_files)
    
    print(f"📊 val目录文件统计:")
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
    count_val_files(dataset_dir)
