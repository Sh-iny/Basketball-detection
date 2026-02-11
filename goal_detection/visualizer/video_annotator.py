"""
视频标注器模块
用于在视频帧上绘制检测结果和统计信息
"""

import cv2


class VideoAnnotator:
    """视频标注器"""

    def __init__(self, output_path, fps, width, height):
        """
        初始化标注器

        Args:
            output_path: 输出视频路径
            fps: 帧率
            width: 视频宽度
            height: 视频高度
        """
        self.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    def write(self, frame):
        """
        写入帧

        Args:
            frame: 视频帧
        """
        self.writer.write(frame)

    def release(self):
        """释放资源"""
        self.writer.release()
