"""
继续训练篮球检测模型
从已有的best.pt继续训练100轮
"""

from ultralytics import YOLO
import torch

def continue_train_ball_detector():
    """继续训练篮球检测模型"""

    # 检查GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 加载已训练的模型继续训练
    model = YOLO('trained_models/ball_detector_v3/weights/best.pt')

    # 继续训练参数
    results = model.train(
        data='datasets/ball_dataset2/data.yaml',
        epochs=600,            # 再训练100轮
        imgsz=640,
        batch=64,
        device="0,1,2,3,4,5,6,7",

        # 优化参数
        conf=0.001,
        iou=0.5,

        # 数据增强
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,

        # 训练设置
        patience=0,            # 关闭早停，训练到200轮
        save=True,
        save_period=10,
        cache=False,
        workers=8,
        project='.',
        name='ball_detector_v4',  # 新的输出目录
        exist_ok=True,
        resume=False,          # 不使用resume，而是作为新训练

        # 优化器
        # optimizer='AdamW',
        # lr0=0.0005,            # 降低学习率
        # lrf=0.01,
        # momentum=0.937,
        # weight_decay=0.0005,

        # 损失权重
        box=7.5,
        cls=0.5,
        dfl=1.5,
    )

    print("\n篮球检测模型继续训练完成!")
    print(f"模型路径: trained_models/ball_detector_v2/weights/best.pt")

    return results


if __name__ == '__main__':
    continue_train_ball_detector()
