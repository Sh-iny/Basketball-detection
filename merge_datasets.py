"""
合并三个数据集：
- br/basketball_and_hoop2.v4i.yolo26
- br/basketball.v21i.yolo26  
- br/basketballhoop.v5i.yolo26

统一类别为：['basketball', 'hoop']
"""
import shutil
from pathlib import Path
import re

def copy_and_process_files(src_dirs, output_dir, category_map):
    """
    复制并处理文件到目标目录
    category_map: 从源类别到目标类别的映射
    """
    total_copied = 0

    for src_base_dir in src_dirs:
        # 处理每个 split: train, valid, test
        for split in ["train", "valid", "test"]:
            src_images = src_base_dir / split / "images"
            src_labels = src_base_dir / split / "labels"

            if not src_images.exists():
                continue

            # 目标目录
            dst_split_dir = output_dir / split
            dst_images = dst_split_dir / "images"
            dst_labels = dst_split_dir / "labels"
            dst_images.mkdir(parents=True, exist_ok=True)
            dst_labels.mkdir(parents=True, exist_ok=True)

            # 复制并处理图片和标签
            split_count = 0
            for img_path in src_images.glob("*.jpg"):
                # 生成唯一文件名
                unique_name = f"{src_base_dir.name}_{img_path.name}"
                dst_img = dst_images / unique_name
                
                # 复制图片
                shutil.copy2(str(img_path), str(dst_img))

                # 处理标签文件
                label_path = src_labels / (img_path.stem + ".txt")
                if label_path.exists():
                    dst_label = dst_labels / (unique_name.rsplit('.', 1)[0] + ".txt")
                    
                    # 读取并处理标签
                    with open(label_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                    processed_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            # 映射类别
                            new_class_id = category_map.get(class_id, class_id)
                            parts[0] = str(new_class_id)
                            processed_lines.append(' '.join(parts) + '\n')

                    # 写入处理后的标签
                    if processed_lines:
                        with open(dst_label, 'w', encoding='utf-8') as f:
                            f.writelines(processed_lines)

                    split_count += 1
                    total_copied += 1
                    if total_copied % 100 == 0:
                        print(f"   已处理 {total_copied} 个文件...")

            if split_count > 0:
                print(f"     {split}: {split_count} 个文件")

    return total_copied

def main():
    print("=" * 80)
    print("合并篮球数据集")
    print("=" * 80)

    # 定义数据集路径
    datasets = [
        Path("br/basketball_and_hoop2.v4i.yolo26"),  # 类别: ['basketball', 'hoop']
        Path("br/basketball.v21i.yolo26"),           # 类别: ?
        Path("br/basketballhoop.v5i.yolo26"),         # 类别: ['ball', 'hoop']
    ]

    # 定义目标目录
    output_dir = Path("merged_basketball_dataset")
    train_dir = output_dir / "train"
    valid_dir = output_dir / "valid"
    test_dir = output_dir / "test"

    # 确保目录存在
    output_dir.mkdir(exist_ok=True)
    train_dir.mkdir(exist_ok=True)
    valid_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)

    print("\n📋 数据集信息:")
    print(f"1. {datasets[0].name} - 类别: ['basketball', 'hoop']")
    print(f"2. {datasets[1].name} - 类别: 待检测")
    print(f"3. {datasets[2].name} - 类别: ['ball', 'hoop']")

    # 分析第二个数据集的类别
    second_data_yaml = datasets[1] / "data.yaml"
    if second_data_yaml.exists():
        with open(second_data_yaml, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"   检测到 {datasets[1].name} 的 data.yaml")

    print("\n" + "=" * 80)
    print("开始合并数据集...")
    print("=" * 80)

    # 处理每个数据集，统一类别为 ['basketball', 'hoop']
    print("\n1. 处理 basketball_and_hoop2.v4i.yolo26...")
    # 类别已匹配: 0 -> basketball, 1 -> hoop
    count1 = copy_and_process_files([datasets[0]], output_dir, {0: 0, 1: 1})

    print(f"\n2. 处理 basketball.v21i.yolo26...")
    # 假设类别也是 ['basketball', 'hoop'] 或 ['ball', 'hoop']
    count2 = copy_and_process_files([datasets[1]], output_dir, {0: 0, 1: 1})

    print(f"\n3. 处理 basketballhoop.v5i.yolo26...")
    # 映射类别: 0 -> ball -> 0 -> basketball, 1 -> hoop -> 1 -> hoop
    count3 = copy_and_process_files([datasets[2]], output_dir, {0: 0, 1: 1})

    # 生成 data.yaml
    data_yaml_content = '''train: ../train/images
val: ../valid/images
test: ../test/images

nc: 2
names: ['basketball', 'hoop']

# Merged dataset from:
# - basketball_and_hoop2.v4i.yolo26
# - basketball.v21i.yolo26
# - basketballhoop.v5i.yolo26
''' 

    with open(output_dir / "data.yaml", 'w', encoding='utf-8') as f:
        f.write(data_yaml_content)

    total_count = count1 + count2 + count3
    print(f"\n✅ 完成！")
    print(f"\n📊 合并统计:")
    print(f"   - basketball_and_hoop2.v4i.yolo26: {count1} 个文件")
    print(f"   - basketball.v21i.yolo26: {count2} 个文件")
    print(f"   - basketballhoop.v5i.yolo26: {count3} 个文件")
    print(f"   - 总计: {total_count} 个文件")

    print(f"\n📁 输出目录:")
    print(f"   {output_dir}")
    print(f"   - data.yaml: 统一类别配置")
    print(f"   - train/images: 合并的图片")
    print(f"   - train/labels: 合并的标签")

if __name__ == "__main__":
    main()
