"""
删除 BasketballAndHoop.v2i.yolo26 中的数据增强图片，只保留原图

文件名格式: {prefix}_jpg.rf.{hash}.jpg
同一个 prefix 的多张图片是同一原图的增强版本
"""
from pathlib import Path
from collections import defaultdict
import re

def analyze_files(base_dir: Path):
    """分析文件，按前缀分组"""
    images_dir = base_dir / "train" / "images"
    labels_dir = base_dir / "train" / "labels"

    # 获取所有图片文件
    image_files = list(images_dir.glob("*.jpg"))

    # 按前缀分组: {prefix}_jpg.rf.{hash}
    groups = defaultdict(list)
    for img_path in image_files:
        name = img_path.stem  # 去掉 .jpg
        # 匹配格式: {prefix}_jpg.rf.{hash}
        match = re.match(r'^(.+)_jpg\.rf\.[a-f0-9]+$', name)
        if match:
            prefix = match.group(1)
            groups[prefix].append(img_path)

    return groups, labels_dir

def main():
    base_dir = Path("br/BasketballAndHoop.v2i.yolo26")

    print("=" * 70)
    print("分析数据增强图片")
    print("=" * 70)

    groups, labels_dir = analyze_files(base_dir)

    # 统计信息
    total_groups = len(groups)
    total_images = sum(len(imgs) for imgs in groups.values())
    augmented_count = sum(len(imgs) - 1 for imgs in groups.values())

    print(f"\n📊 分析结果:")
    print(f"   - 原图组数: {total_groups} 组")
    print(f"   - 总图片数: {total_images} 张")
    print(f"   - 增强图片数: {augmented_count} 张")
    print(f"   - 保留后将剩余: {total_groups} 张")

    # 显示一些示例
    print(f"\n📋 分组示例 (显示前5组):")
    for i, (prefix, imgs) in enumerate(list(groups.items())[:5]):
        sorted_imgs = sorted(imgs, key=lambda x: x.name)
        print(f"\n   组 {i+1}: {prefix}")
        for j, img in enumerate(sorted_imgs[:5]):
            marker = "  [保留]" if j == 0 else "  [删除]"
            print(f"      {img.name[:40]}...{marker}")
        if len(imgs) > 5:
            print(f"      ... 还有 {len(imgs)-5} 张")

    if len(groups) > 5:
        print(f"\n   ... 还有 {len(groups) - 5} 组 ...")

    # 询问确认
    print("\n" + "=" * 70)
    print("⚠️  即将执行删除操作")
    print("=" * 70)
    print(f"将删除 {augmented_count} 张数据增强图片")
    print(f"保留 {total_groups} 张原图（每组保留一张）")
    print(f"同时会删除对应的标签文件 (.txt)")
    print("\n操作不可撤销，请确认:")

    while True:
        choice = input("\n请输入 'yes' 确认删除，或 'no' 取消: ").strip().lower()
        if choice == 'yes':
            break
        elif choice == 'no':
            print("\n已取消操作")
            return
        else:
            print("请输入 'yes' 或 'no'")

    # 执行删除
    print("\n" + "=" * 70)
    print("开始删除数据增强图片...")
    print("=" * 70)

    deleted_count = 0
    for prefix, imgs in groups.items():
        # 按文件名排序，保留第一张
        sorted_imgs = sorted(imgs, key=lambda x: x.name)
        for img_path in sorted_imgs[1:]:
            # 删除图片
            img_path.unlink()

            # 删除对应的标签文件
            label_path = labels_dir / (img_path.stem + ".txt")
            if label_path.exists():
                label_path.unlink()

            deleted_count += 1
            if deleted_count % 100 == 0:
                print(f"   已删除 {deleted_count}/{augmented_count} 张...")

    print(f"\n✅ 完成！共删除 {deleted_count} 张图片及其标签")

    # 最终统计
    remaining_images = len(list((base_dir / "train" / "images").glob("*.jpg")))
    remaining_labels = len(list((base_dir / "train" / "labels").glob("*.txt")))
    print(f"\n📊 最终统计:")
    print(f"   - train/images: {remaining_images} 张")
    print(f"   - train/labels: {remaining_labels} 个")

if __name__ == "__main__":
    main()
