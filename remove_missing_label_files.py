"""
移除数据集中缺少标注文件的图片文件
"""
import os
from pathlib import Path

def remove_missing_label_files(dataset_dir):
    """
    移除数据集中缺少标注文件的图片文件
    
    Args:
        dataset_dir: 数据集目录路径
    """
    print(f"=" * 80)
    print(f"移除缺少标注文件的图片: {dataset_dir}")
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
    image_basenames = {f.stem: f for f in image_files}
    label_basenames = {f.stem for f in label_files}
    
    # 找出缺少标注的图片
    images_without_labels = [f for stem, f in image_basenames.items() if stem not in label_basenames]
    
    if not images_without_labels:
        print(f"✅ 没有缺少标注文件的图片")
        return
    
    print(f"📊 发现 {len(images_without_labels)} 个缺少标注文件的图片:")
    
    # 移除这些图片
    removed_count = 0
    for img_path in images_without_labels:
        try:
            img_path.unlink()
            removed_count += 1
            print(f"   🗑️  移除: {img_path.name}")
        except Exception as e:
            print(f"   ❌ 移除失败: {img_path.name} - {e}")
    
    print(f"\n✅ 完成！")
    print(f"   - 检查的图片数量: {len(image_files)}")
    print(f"   - 发现缺少标注的图片: {len(images_without_labels)}")
    print(f"   - 成功移除: {removed_count}")
    
    # 再次检查一致性
    print(f"\n🔍 移除后检查:")
    remaining_images = set()
    for ext in ['.jpg', '.jpeg', '.png']:
        remaining_images.update(images_dir.glob(f"*{ext}"))
    
    remaining_labels = set(labels_dir.glob("*.txt"))
    
    print(f"   - 剩余图片: {len(remaining_images)}")
    print(f"   - 剩余标注: {len(remaining_labels)}")
    
    if len(remaining_images) == len(remaining_labels):
        print(f"   ✅ 数量一致！")
    else:
        print(f"   ❌ 数量仍然不一致！")
    
    print(f"\n=" * 80)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("用法: python remove_missing_label_files.py <dataset_dir>")
        sys.exit(1)
    
    dataset_dir = sys.argv[1]
    remove_missing_label_files(dataset_dir)
