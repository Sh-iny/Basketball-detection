"""
RT-DETR目标检测训练脚本 - 篮球检测
使用COCO格式数据集训练RT-DETR模型
"""

from ultralytics import RTDETR
import torch
from rfdetr import RFDETRNano
from pathlib import Path


def train_rtdetr_detector():
    """训练RT-DETR篮球目标检测模型"""

    # 检查CUDA是否可用
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # 数据集配置文件路径
    data_yaml = "/data/sui/basketball/datasets/coco_dataset/data.yaml"

    # 验证数据集配置文件是否存在
    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"数据集配置文件不存在: {data_yaml}")

    # 加载RT-DETR预训练模型
    # 可选: rtdetr-l.pt (large), rtdetr-x.pt (xlarge)
    print("加载RT-DETR-N预训练模型...")
    model = RFDETRNano()

    print("开始训练模型...")

    # 训练模型（RF-DETR的API）
    results = model.train(
        # 数据集配置
        dataset_dir="/data/sui/basketball/datasets/ball_dataset2_coco",  # 篮球数据集（COCO格式）

        # 基本训练参数
        epochs=150,              # 训练轮数
        batch_size=4,           # 批次大小
        image_size=640,          # 输入图像大小

        # 设备和性能
        device="cuda",           # 使用GPU
        num_workers=0,           # 数据加载线程数

        # 项目设置
        output_dir='/data/sui/basketball/trained_models/rtdetr_nano',
        resume="",               # 恢复训练的检查点路径（空字符串表示不恢复）
    )

    print("\n训练完成！")
    print(f"最佳模型保存在: {results.save_dir}/weights/best.pt")
    print(f"最后模型保存在: {results.save_dir}/weights/last.pt")

    return results


if __name__ == '__main__':
    train_rtdetr_detector()
