"""
训练专门的篮筐检测模型
专注于提高篮筐检测的稳定性和精度
"""

from ultralytics import YOLO
import torch

def train_rim_detector():
    """训练篮筐检测模型"""

    # 检查GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 加载预训练模型 - 使用更小的模型
    model = YOLO('pretrained/yolo26n.pt')

    # 训练参数 - 优化稳定性
    results = model.train(
        data='datasets/rim_dataset/rim/data.yaml',
        epochs=100,
        imgsz=640,         # 中等分辨率即可
        batch=16,
        device=device,

        # 优化参数 - 提高精度
        conf=0.25,
        iou=0.7,

        # 数据增强 - 适应篮筐场景
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=5.0,       # 轻微旋转
        translate=0.05,
        scale=0.3,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,

        # 训练设置
        patience=30,
        save=True,
        save_period=10,
        cache=False,
        workers=8,
        project='.',
        name='trained_models/rim_detector',
        exist_ok=True,

        # 优化器
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,

        # 损失权重 - 平衡
        box=7.5,
        cls=0.5,
        dfl=1.5,
    )

    print("\n篮筐检测模型训练完成!")
    print(f"模型路径: trained_models/rim_detector/weights/best.pt")

    return results


if __name__ == '__main__':
    train_rim_detector()
