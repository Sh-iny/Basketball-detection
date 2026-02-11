"""
视频目标检测脚本 - 输出MP4格式
使用训练好的YOLO模型对视频进行检测，并输出为mp4格式
"""

from ultralytics import YOLO
import cv2
from pathlib import Path
import argparse


def predict_video_to_mp4(model_path, video_path, output_path=None, conf_threshold=0.5):
    """
    对视频进行预测并输出为mp4格式

    Args:
        model_path: 模型权重文件路径
        video_path: 输入视频路径
        output_path: 输出视频路径（可选，默认自动生成）
        conf_threshold: 置信度阈值
    """
    # 加载模型
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)

    # 打开输入视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"视频信息: {width}x{height} @ {fps}fps, 总帧数: {total_frames}")

    # 设置输出路径
    if output_path is None:
        video_name = Path(video_path).stem
        output_path = f"runs/predict/{video_name}_detected.mp4"

    # 创建输出目录
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 创建视频写入器 - 使用mp4v编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"开始处理视频...")
    print(f"输出路径: {output_path}")

    frame_count = 0

    # 逐帧处理
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 使用模型进行预测
        results = model.predict(frame, conf=conf_threshold, verbose=False)

        # 在帧上绘制检测结果
        annotated_frame = results[0].plot()

        # 写入输出视频
        out.write(annotated_frame)

        # 显示进度
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"处理进度: {frame_count}/{total_frames} ({progress:.1f}%)")

    # 释放资源
    cap.release()
    out.release()

    print(f"\n处理完成!")
    print(f"输出文件: {output_path}")
    print(f"处理帧数: {frame_count}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='视频目标检测 - 输出MP4格式')
    parser.add_argument('--model', type=str, required=True,
                        help='模型权重文件路径')
    parser.add_argument('--video', type=str, required=True,
                        help='输入视频路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出视频路径（可选，默认自动生成）')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='置信度阈值（默认: 0.5）')

    args = parser.parse_args()

    # 检查文件是否存在
    if not Path(args.model).exists():
        raise FileNotFoundError(f"模型文件不存在: {args.model}")

    if not Path(args.video).exists():
        raise FileNotFoundError(f"视频文件不存在: {args.video}")

    # 执行预测
    predict_video_to_mp4(
        model_path=args.model,
        video_path=args.video,
        output_path=args.output,
        conf_threshold=args.conf
    )


if __name__ == '__main__':
    main()
