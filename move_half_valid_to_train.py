"""
将 Basketball Computer Vision Model1/valid 中的一半数据移动到 train 目录
"""
import shutil
from pathlib import Path
import random

# 设置随机种子以保证可重复性
random.seed(42)

# 定义路径
base_dir = Path("Basketball_Computer_Vision_Model")
valid_images_dir = base_dir / "valid" / "images"
valid_labels_dir = base_dir / "valid" / "labels"
train_images_dir = base_dir / "train" / "images"
train_labels_dir = base_dir / "train" / "labels"

# 确保目标目录存在
train_images_dir.mkdir(parents=True, exist_ok=True)
train_labels_dir.mkdir(parents=True, exist_ok=True)

# 获取所有有效的图片文件（有对应标签文件的）
valid_images = sorted(valid_images_dir.glob("*.jpg"))

# 过滤出同时有标签文件的图片
valid_pairs = []
for img_path in valid_images:
    label_path = valid_labels_dir / (img_path.stem + ".txt")
    if label_path.exists():
        valid_pairs.append((img_path, label_path))

print(f"找到 {len(valid_pairs)} 对有效的图片-标签文件")

# 计算要移动的数量（一半）
num_to_move = len(valid_pairs) // 2
print(f"将移动 {num_to_move} 对文件到 train 目录")

# 随机选择要移动的文件
pairs_to_move = random.sample(valid_pairs, num_to_move)

# 执行移动
moved_count = 0
for img_path, label_path in pairs_to_move:
    # 移动图片
    dest_img = train_images_dir / img_path.name
    shutil.move(str(img_path), str(dest_img))

    # 移动标签
    dest_label = train_labels_dir / label_path.name
    shutil.move(str(label_path), str(dest_label))

    moved_count += 1
    if moved_count % 100 == 0:
        print(f"已移动 {moved_count}/{num_to_move} 对文件")

print(f"\n完成！成功移动了 {moved_count} 对文件从 valid 到 train")

# 统计最终数量
remaining_valid_images = len(list(valid_images_dir.glob("*.jpg")))
remaining_valid_labels = len(list(valid_labels_dir.glob("*.txt")))
new_train_images = len(list(train_images_dir.glob("*.jpg")))
new_train_labels = len(list(train_labels_dir.glob("*.txt")))

print(f"\n移动后统计:")
print(f"  valid/images: {remaining_valid_images} 个文件")
print(f"  valid/labels: {remaining_valid_labels} 个文件")
print(f"  train/images: {new_train_images} 个文件")
print(f"  train/labels: {new_train_labels} 个文件")
