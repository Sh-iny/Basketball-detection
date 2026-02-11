"""
将融合数据集中train的20%文件移动到val中
"""
import os
import random
from pathlib import Path

def move_train_to_val(dataset_dir, percentage=20):
    """
    将train目录中指定百分比的文件移动到val目录
    
    Args:
        dataset_dir: 数据集根目录路径
        percentage: 要移动的文件百分比
    """
    print(f"=" * 80)
    print(f"将train目录 {percentage}% 文件移动到val: {dataset_dir}")
    print(f"=" * 80)
    
    dataset_path = Path(dataset_dir)
    
    # 源目录
    train_images_dir = dataset_path / "train" / "images"
    train_labels_dir = dataset_path / "train" / "labels"
    
    # 目标目录
    val_images_dir = dataset_path / "valid" / "images"
    val_labels_dir = dataset_path / "valid" / "labels"
    
    # 检查目录是否存在
    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        if not dir_path.exists():
            print(f"❌ 目录不存在: {dir_path}")
            return
    
    # 获取所有图片文件
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(train_images_dir.glob(f"*{ext}"))
    
    total_images = len(image_files)
    move_count = int(total_images * percentage / 100)
    
    print(f"📊 统计信息:")
    print(f"   - train目录总文件数: {total_images}")
    print(f"   - 计划移动文件数: {move_count}")
    
    if move_count == 0:
        print(f"❌ 没有文件需要移动")
        return
    
    # 随机选择文件
    random.seed(42)  # 设置种子以保证结果可复现
    selected_images = random.sample(image_files, move_count)
    
    print(f"\n🚚 开始移动文件...")
    
    moved_count = 0
    for img_path in selected_images:
        try:
            # 构建对应的标注文件路径
            img_stem = img_path.stem
            label_path = train_labels_dir / f"{img_stem}.txt"
            
            if not label_path.exists():
                print(f"⚠️  标注文件不存在，跳过: {label_path.name}")
                continue
            
            # 构建目标路径
            dst_img_path = val_images_dir / img_path.name
            dst_label_path = val_labels_dir / label_path.name
            
            # 移动文件
            img_path.rename(dst_img_path)
            label_path.rename(dst_label_path)
            
            moved_count += 1
            if moved_count % 50 == 0:
                print(f"   已移动 {moved_count}/{move_count} 个文件...")
                
        except Exception as e:
            print(f"❌ 移动失败: {img_path.name} - {e}")
    
    print(f"\n✅ 移动完成！")
    print(f"   - 成功移动: {moved_count} 个文件")
    print(f"   - 失败: {move_count - moved_count} 个文件")
    
    # 验证移动后的数量
    print(f"\n🔍 验证移动后文件数量:")
    
    # 统计移动后的train目录
    after_train_images = []
    for ext in ['.jpg', '.jpeg', '.png']:
        after_train_images.extend(train_images_dir.glob(f"*{ext}"))
    after_train_labels = list(train_labels_dir.glob("*.txt"))
    
    # 统计移动后的val目录
    after_val_images = []
    for ext in ['.jpg', '.jpeg', '.png']:
        after_val_images.extend(val_images_dir.glob(f"*{ext}"))
    after_val_labels = list(val_labels_dir.glob("*.txt"))
    
    print(f"   📁 train目录:")
    print(f"      - 图片文件: {len(after_train_images)}")
    print(f"      - 标注文件: {len(after_train_labels)}")
    
    print(f"   📁 val目录:")
    print(f"      - 图片文件: {len(after_val_images)}")
    print(f"      - 标注文件: {len(after_val_labels)}")
    
    # 验证一致性
    train_consistent = len(after_train_images) == len(after_train_labels)
    val_consistent = len(after_val_images) == len(after_val_labels)
    
    if train_consistent and val_consistent:
        print(f"\n✅ 移动后所有目录文件数量一致！")
    else:
        print(f"\n❌ 移动后文件数量不一致！")
        if not train_consistent:
            print(f"   - train目录: 图片和标注数量不匹配")
        if not val_consistent:
            print(f"   - val目录: 图片和标注数量不匹配")

if __name__ == "__main__":
    dataset_dir = "merged_basketball_dataset"
    move_train_to_val(dataset_dir, 20)
