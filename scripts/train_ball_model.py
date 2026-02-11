"""
训练专门的篮球检测模型
专注于提高篮球的检测召回率
"""

from ultralytics import YOLO
import torch

def train_ball_detector():
    """训练篮球检测模型"""

    # 检查GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 加载预训练模型
    model = YOLO('pretrained/yolo26m.pt')

    # 训练参数 - 优化小目标检测
    results = model.train(
        data='datasets/ball_dataset3/data.yaml',
        epochs=100,
        imgsz=1280,        # 高分辨率，更好检测小目标
        batch=8,           # 适应高分辨率
        device="0,1,2,3,4,5,6,7", # 使用所有可用GPU

        # # 优化参数 - 提高召回率
        # conf=0.001,        # 极低置信度训练
        # iou=0.5,

        # # 数据增强 - 适应篮球场景
        # hsv_h=0.015,
        # hsv_s=0.7,
        # hsv_v=0.4,
        # degrees=15.0,      # 旋转增强
        # translate=0.1,
        # scale=0.5,
        # shear=0.0,
        # perspective=0.0,
        # flipud=0.0,
        # fliplr=0.5,
        # mosaic=1.0,
        # mixup=0.15,

        # # 训练设置
        # patience=0,
        # save=True,
        # save_period=10,
        # cache=False,
        workers=8,
        project='.',
        name='ball_detector_v6',
        exist_ok=True,

        # # 优化器
        # optimizer='AdamW',
        # lr0=0.001,
        # lrf=0.01,
        # momentum=0.937,
        # weight_decay=0.0005,

        # # 损失权重 - 强化小目标
        # box=7.5,
        # cls=0.5,
        # dfl=1.5,
    )

    print("\n篮球检测模型训练完成!")
    print(f"模型路径: trained_models/ball_detector/weights/best.pt")

    return results


if __name__ == '__main__':
    train_ball_detector()
