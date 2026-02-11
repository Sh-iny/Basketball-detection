"""
训练YOLOv11s模型
使用合并后的ball+rim数据集训练
"""

from ultralytics import YOLO
import torch

def main():
    """主函数"""

    # 检查CUDA是否可用
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # 加载YOLOv11s预训练模型
    print("\n加载YOLOv11s预训练模型...")
    model = YOLO('yolo11s.pt')

    # 训练参数
    print("\n开始训练...")
    results = model.train(
        # 数据集配置
        data='/data/sui/basketball/merged_dataset/data.yaml',

        # 训练参数
        epochs=100,              # 训练轮数
        batch=16,                # 批次大小（根据GPU显存调整）
        imgsz=640,               # 输入图像大小

        # 优化器参数
        optimizer='AdamW',       # 优化器
        lr0=0.001,              # 初始学习率
        lrf=0.01,               # 最终学习率（lr0 * lrf）
        momentum=0.937,         # SGD动量/Adam beta1
        weight_decay=0.0005,    # 权重衰减

        # 数据增强
        hsv_h=0.015,            # 色调增强
        hsv_s=0.7,              # 饱和度增强
        hsv_v=0.4,              # 明度增强
        degrees=0.0,            # 旋转角度
        translate=0.1,          # 平移
        scale=0.5,              # 缩放
        shear=0.0,              # 剪切
        perspective=0.0,        # 透视变换
        flipud=0.0,             # 上下翻转概率
        fliplr=0.5,             # 左右翻转概率
        mosaic=1.0,             # Mosaic增强概率
        mixup=0.0,              # Mixup增强概率
        copy_paste=0.0,         # Copy-paste增强概率

        # 训练设置
        device='0,1,2,3,4,5,6,7',  # 使用8个GPU
        workers=8,              # 数据加载线程数
        project='runs/train',   # 项目目录
        name='yolo11s_ball_rim',  # 实验名称
        exist_ok=False,         # 是否覆盖已存在的实验
        pretrained=True,        # 使用预训练权重
        verbose=True,           # 详细输出
        seed=0,                 # 随机种子
        deterministic=True,     # 确定性训练

        # 验证设置
        val=True,               # 训练时进行验证
        save=True,              # 保存检查点
        save_period=10,         # 每N个epoch保存一次
        cache=False,            # 是否缓存图像到内存
        rect=False,             # 矩形训练
        resume=False,           # 恢复训练
        amp=True,               # 自动混合精度训练
        fraction=1.0,           # 使用数据集的比例
        profile=False,          # 性能分析
        freeze=None,            # 冻结层数

        # 早停
        patience=50,            # 早停耐心值（epoch）

        # 其他
        plots=True,             # 保存训练图表
        overlap_mask=True,      # 训练时mask重叠
        mask_ratio=4,           # mask下采样比例
        dropout=0.0,            # Dropout概率
        box=7.5,                # box loss权重
        cls=0.5,                # cls loss权重
        dfl=1.5,                # dfl loss权重
    )

    print("\n训练完成！")
    print(f"最佳模型: {results.save_dir}/weights/best.pt")
    print(f"最后模型: {results.save_dir}/weights/last.pt")


if __name__ == '__main__':
    main()
