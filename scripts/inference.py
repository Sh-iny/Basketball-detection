"""
YOLO目标检测推理脚本 - 篮球检测
用于使用训练好的模型进行预测
"""

from ultralytics import YOLO
import cv2
from pathlib import Path
import argparse


def predict_image(model_path, image_path, conf_threshold=0.25, save_dir='runs/predict'):
    """
    对单张图像进行预测

    Args:
        model_path: 模型权重文件路径
        image_path: 输入图像路径
        conf_threshold: 置信度阈值
        save_dir: 结果保存目录
    """
    # 加载训练好的模型
    model = YOLO(model_path)

    # 进行预测
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        save=True,
        project=save_dir,
        name='basketball_results',
        exist_ok=True
    )

    return results


def predict_video(model_path, video_path, conf_threshold=0.25, save_dir='runs/predict'):
    """
    对视频进行预测

    Args:
        model_path: 模型权重文件路径
        video_path: 输入视频路径
        conf_threshold: 置信度阈值
        save_dir: 结果保存目录
    """
    # 加载训练好的模型
    model = YOLO(model_path)

    # 进行预测，输出为mp4格式
    results = model.predict(
        source=video_path,
        conf=conf_threshold,
        save=True,
        project=save_dir,
        name='basketball_video_results',
        exist_ok=True,
        stream=True,  # 流式处理视频
        vid_stride=1,  # 每帧都处理
        save_txt=False,  # 不保存txt标注
        save_conf=False,  # 不保存置信度
    )

    return results


def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(description='篮球目标检测推理脚本')
    parser.add_argument('--model', type=str, required=True,
                        help='模型权重文件路径 (例如: runs/train/basketball_detect/weights/best.pt)')
    parser.add_argument('--source', type=str, required=True,
                        help='输入源 (图像路径、视频路径或摄像头ID)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='置信度阈值 (默认: 0.25)')
    parser.add_argument('--save-dir', type=str, default='runs/predict',
                        help='结果保存目录 (默认: runs/predict)')

    args = parser.parse_args()

    # 检查模型文件是否存在
    if not Path(args.model).exists():
        raise FileNotFoundError(f"模型文件不存在: {args.model}")

    # 判断输入源类型
    source_path = Path(args.source)

    if source_path.exists():
        if source_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            print(f"检测图像: {args.source}")
            results = predict_image(args.model, args.source, args.conf, args.save_dir)
        elif source_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            print(f"检测视频: {args.source}")
            results = predict_video(args.model, args.source, args.conf, args.save_dir)
        else:
            print(f"不支持的文件格式: {source_path.suffix}")
            return
    else:
        # 可能是摄像头ID或URL
        print(f"检测实时流: {args.source}")
        model = YOLO(args.model)
        results = model.predict(source=args.source, conf=args.conf, save=True,
                               project=args.save_dir, name='basketball_stream', exist_ok=True)

    print("检测完成!")


if __name__ == '__main__':
    main()
