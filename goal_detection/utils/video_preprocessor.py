"""
视频预处理模块 - 篮球进球检测专用
针对进球检测场景优化，提高检测稳定性
"""

import cv2
import numpy as np


class VideoPreprocessor:
    """视频帧预处理器 - 进球检测专用"""

    def __init__(self, config=None):
        """
        初始化预处理器

        Args:
            config: 配置字典，可选
        """
        # 默认配置
        self.normalize_brightness = True
        self.brightness_target = 128
        self.brightness_strength = 0.3
        self.target_size = 640  # letterbox目标尺寸
        self.use_letterbox = False  # 是否使用letterbox（YOLO内部会做，这里可选）

        # 从配置加载
        if config and 'preprocessing' in config:
            prep_config = config['preprocessing']
            self.normalize_brightness = prep_config.get('normalize_brightness', True)
            self.brightness_target = prep_config.get('brightness_target', 128)
            self.brightness_strength = prep_config.get('brightness_strength', 0.3)
            self.target_size = prep_config.get('target_size', 640)
            self.use_letterbox = prep_config.get('use_letterbox', False)

        # letterbox变换信息（用于坐标还原）
        self.scale = 1.0
        self.pad = (0, 0)

    def process_frame(self, frame):
        """
        处理单帧

        Args:
            frame: 输入帧

        Returns:
            处理后的帧
        """
        # 亮度归一化
        if self.normalize_brightness:
            frame = self._light_normalize(frame)

        # letterbox缩放（保持宽高比）
        if self.use_letterbox:
            frame = self._letterbox(frame)

        return frame

    def _letterbox(self, frame, color=(114, 114, 114)):
        """
        Letterbox缩放，保持宽高比，不拉伸变形

        Args:
            frame: 输入帧
            color: 填充颜色（灰色）

        Returns:
            处理后的帧
        """
        h, w = frame.shape[:2]
        target = self.target_size

        # 计算缩放比例（取较小值保证图像完整）
        scale = min(target / h, target / w)
        self.scale = scale

        # 新尺寸
        new_w, new_h = int(w * scale), int(h * scale)

        # 缩放
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 计算填充
        dw = (target - new_w) // 2
        dh = (target - new_h) // 2
        self.pad = (dw, dh)

        # 添加边框填充
        frame = cv2.copyMakeBorder(
            frame, dh, target - new_h - dh, dw, target - new_w - dw,
            cv2.BORDER_CONSTANT, value=color
        )

        return frame

    def restore_bbox(self, bbox):
        """
        将letterbox后的坐标还原到原始图像坐标

        Args:
            bbox: [x1, y1, x2, y2] letterbox后的坐标

        Returns:
            原始图像坐标
        """
        if not self.use_letterbox:
            return bbox

        dw, dh = self.pad
        x1 = (bbox[0] - dw) / self.scale
        y1 = (bbox[1] - dh) / self.scale
        x2 = (bbox[2] - dw) / self.scale
        y2 = (bbox[3] - dh) / self.scale

        return [x1, y1, x2, y2]

    def _light_normalize(self, frame):
        """轻度亮度归一化，减少帧间亮度差异"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        current_brightness = np.mean(l)
        diff = self.brightness_target - current_brightness
        adjustment = diff * self.brightness_strength

        l = np.clip(l.astype(np.float32) + adjustment, 0, 255).astype(np.uint8)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


class DetectionSmoother:
    """
    检测结果平滑器（简单版本，默认禁用）
    """

    def __init__(self, confirm_frames=2, lost_tolerance=3):
        self.confirm_frames = confirm_frames
        self.lost_tolerance = lost_tolerance
        self.detection_history = []
        self.confirmed = False
        self.lost_count = 0
        self.last_valid_detection = None

    def update(self, ball_detections):
        """直接返回原始检测结果（不做平滑）"""
        return ball_detections

    def reset(self):
        """重置状态"""
        self.detection_history = []
        self.confirmed = False
        self.lost_count = 0
        self.last_valid_detection = None
