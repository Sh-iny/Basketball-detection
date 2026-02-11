"""
检查数据集目录中图片和标注文件的数量是否一致
"""
import os
from pathlib import Path

def check_file_consistency(dataset_dir):
    """
    检查数据集目录中图片和标注文件的数量是否一致
    
    Args:
        dataset_dir: 数据集目录路径
    """
    print(f"=" * 80)
    print(f"检查文件一致性: {dataset_dir}")
    print(f"=" * 80)
    
    dataset_path = Path(dataset_dir)
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"
    
    if not images_dir.exists():
        print(f"❌ 图片目录不存在: {images_dir}")
        return
    
    if not labels_dir.exists():
        print(f"❌ 标签目录不存在: {labels_dir}")
        return
    
    # 获取所有图片和标签文件
    image_files = set()
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.update(images_dir.glob(f"*{ext}"))
    
    label_files = set(labels_dir.glob("*.txt"))
    
    # 转换为文件名（不含扩展名）
    image_basenames = {f.stem for f in image_files}
    label_basenames = {f.stem for f in label_files}
    
    # 统计数量
    total_images = len(image_files)
    total_labels = len(label_files)
    
    print(f"📊 文件统计:")
    print(f"   - 图片文件: {total_images}")
    print(f"   - 标注文件: {total_labels}")
    
    if total_images != total_labels:
        print(f"\n❌ 数量不一致！")
        
        # 找出缺失的文件
        images_without_labels = image_basenames - label_basenames
        labels_without_images = label_basenames - image_basenames
        
        if images_without_labels:
            print(f"\n⚠️  缺少标注文件的图片 ({len(images_without_labels)}):")
            for basename in sorted(images_without_labels)[:10]:  # 只显示前10个
                print(f"   - {basename}")
            if len(images_without_labels) > 10:
                print(f"   ... 还有 {len(images_without_labels) - 10} 个文件")
        
        if labels_without_images:
            print(f"\n⚠️  缺少图片文件的标注 ({len(labels_without_images)}):")
            for basename in sorted(labels_without_images)[:10]:  # 只显示前10个
                print(f"   - {basename}")
            if len(labels_without_images) > 10:
                print(f"   ... 还有 {len(labels_without_images) - 10} 个文件")
    else:
        print(f"\n✅ 数量一致！")
    
    print(f"\n=" * 80)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("用法: python check_file_consistency.py <dataset_dir>")
        sys.exit(1)
    
    dataset_dir = sys.argv[1]
    check_file_consistency(dataset_dir)
