"""
训练专门的篮球检测模型
使用 ball_dataset 数据集，专注于提高球的检测率
"""

from ultralytics import YOLO
import torch

def train_ball_detector():
    """训练专门的球类检测模型"""

    # 检查GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 加载预训练模型
    model = YOLO('yolo11s.pt')  # 使用 YOLOv11s 作为基础

    # 训练参数
    results = model.train(
        data='datasets/ball_dataset/ball/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=device,

        # 优化参数 - 提高召回率
        conf=0.15,  # 降低置信度阈值
        iou=0.5,    # IoU阈值

        # 数据增强
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,

        # 训练设置
        patience=20,
        save=True,
        save_period=10,
        cache=False,
        workers=8,
        project='trained_models',
        name='ball_detector_yolo11s',
        exist_ok=True,

        # 优化器
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,

        # 损失权重 - 提高小目标检测
        box=7.5,
        cls=0.5,
        dfl=1.5,
    )

    print("\n训练完成!")
    print(f"最佳模型: trained_models/ball_detector_yolo11s/weights/best.pt")

    return results


if __name__ == '__main__':
    train_ball_detector()
