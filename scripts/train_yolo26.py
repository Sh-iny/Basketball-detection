"""
YOLO目标检测训练脚本 - 篮球检测
用于训练检测篮球、人和篮筐的YOLO模型
"""

from ultralytics import YOLO
import torch
import yaml
from pathlib import Path


def train_basketball_detector():
    """训练篮球目标检测模型"""

    # 检查CUDA是否可用
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 数据集配置文件路径
    data_yaml = "datasets/merged_basketball_dataset/data.yaml"

    # 验证数据集配置文件是否存在
    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"数据集配置文件不存在: {data_yaml}")

    # 加载模型
    # 如果要继续训练，加载last.pt；如果从头开始，加载预训练模型
    # checkpoint_path = '/data/sui/basketball/basketball_detect2/weights/best.pt'
    # if Path(checkpoint_path).exists():
    #     print(f"加载检查点: {checkpoint_path}")
    #     model = YOLO(checkpoint_path)
    # else:
    print("加载预训练模型: yolo26s.pt")
    model = YOLO('yolo26s.pt')

    print("开始训练模型...")

    # 训练模型
    results = model.train(
        data=data_yaml,           # 数据集配置文件
        epochs=200,               # 训练轮数
        patience=50,             # 早停耐心值（设置为epochs相同或0禁用早停）
        imgsz=640,                # 输入图像大小
        batch=128,                 # 批次大小
        device="0,1,2,3,4,5,6,7", # 使用的设备
        workers=8,                # 数据加载的工作进程数
        project='.',     # 项目保存路径
        name='BR', # 实验名称
        exist_ok=False,            # 允许覆盖已存在的项目
        # pretrained=True,          # 使用预训练权重
        # optimizer='auto',         # 优化器(auto, SGD, Adam, AdamW)
        # verbose=True,             # 详细输出
        # deterministic=True,       # 确定性训练
        # single_cls=False,         # 是否为单类检测
        # rect=False,               # 矩形训练
        # cos_lr=False,             # 余弦学习率调度
        # close_mosaic=10,          # 最后N个epoch关闭mosaic增强
        # resume=False,            # 不使用resume，而是作为预训练模型继续训练
        # amp=True,                 # 自动混合精度训练
        # fraction=1.0,             # 使用的数据集比例
        # profile=False,            # 性能分析
        # 数据增强参数
        # hsv_h=0.015,              # HSV色调增强
        # hsv_s=0.7,                # HSV饱和度增强
        # hsv_v=0.4,                # HSV明度增强
        # degrees=0.0,              # 旋转角度
        # translate=0.1,            # 平移
        # scale=0.5,                # 缩放
        # shear=0.0,                # 剪切
        # perspective=0.0,          # 透视变换
        # flipud=0.0,               # 上下翻转概率
        # fliplr=0.5,               # 左右翻转概率
        # mosaic=1.0,               # mosaic增强概率
        # mixup=0.0,                # mixup增强概率
        # copy_paste=0.0,           # copy-paste增强概率
    )

    print("\n训练完成!")
    print(f"最佳模型保存在: {results.save_dir}")

    return results


if __name__ == '__main__':
    train_basketball_detector()
