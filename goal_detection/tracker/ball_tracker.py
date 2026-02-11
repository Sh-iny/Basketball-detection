"""
篮球跟踪器模块
用于跟踪篮球的运动轨迹和状态
"""

from collections import deque
import numpy as np
from ..utils.geometry import bbox_center, calculate_velocity


class BallTracker:
    """篮球轨迹跟踪器"""

    def __init__(self, track_id, max_history=30):
        """
        初始化跟踪器

        Args:
            track_id: 跟踪ID
            max_history: 最大历史记录长度
        """
        self.track_id = track_id
        self.positions = deque(maxlen=max_history)
        self.bboxes = deque(maxlen=max_history)
        self.radii = deque(maxlen=max_history)  # 记录球的半径
        self.frame_ids = deque(maxlen=max_history)
        self.is_active = True
        self.lost_frames = 0  # 跟踪丢失的帧数
        self.max_lost_frames = 10  # 最大允许丢失帧数

    def update(self, bbox, frame_id):
        """
        更新跟踪信息

        Args:
            bbox: 边界框 [x1, y1, x2, y2]
            frame_id: 帧ID
        """
        center = bbox_center(bbox)
        self.positions.append(center)
        self.bboxes.append(bbox)
        
        # 计算球的半径（基于边界框的平均边长的一半）
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        radius = (width + height) / 4  # 平均边长的一半
        self.radii.append(radius)
        
        self.frame_ids.append(frame_id)

    def get_velocity(self):
        """
        获取当前速度

        Returns:
            tuple: (vx, vy, speed) - x方向速度、y方向速度、总速度
        """
        if len(self.positions) < 2:
            return 0, 0, 0
        return calculate_velocity(self.positions[-2], self.positions[-1])

    def get_trajectory(self, length=None):
        """
        获取轨迹点

        Args:
            length: 轨迹长度，None表示全部

        Returns:
            list: 轨迹点列表
        """
        if length is None:
            return list(self.positions)
        return list(self.positions)[-length:]

    @property
    def current_position(self):
        """当前位置"""
        return self.positions[-1] if self.positions else None

    @property
    def previous_position(self):
        """前一帧位置"""
        return self.positions[-2] if len(self.positions) >= 2 else None

    @property
    def current_bbox(self):
        """当前边界框"""
        return self.bboxes[-1] if self.bboxes else None

    @property
    def current_radius(self):
        """当前半径"""
        return self.radii[-1] if self.radii else None

    def get_average_radius(self, window=5):
        """
        获取最近几帧的平均半径

        Args:
            window: 窗口大小

        Returns:
            float: 平均半径
        """
        if len(self.radii) < 1:
            return 0
        recent_radii = list(self.radii)[-window:]
        return sum(recent_radii) / len(recent_radii)

    def predict_position(self, frames_ahead=1):
        """
        基于历史轨迹预测未来位置

        Args:
            frames_ahead: 预测未来多少帧

        Returns:
            tuple: 预测的(x, y)位置，如果无法预测则返回None
        """
        if len(self.positions) < 3:
            return None

        # 使用最近3个点进行线性预测
        recent_positions = list(self.positions)[-3:]

        # 计算平均速度
        vx_sum, vy_sum = 0, 0
        for i in range(len(recent_positions) - 1):
            vx, vy, _ = calculate_velocity(recent_positions[i], recent_positions[i+1])
            vx_sum += vx
            vy_sum += vy

        avg_vx = vx_sum / (len(recent_positions) - 1)
        avg_vy = vy_sum / (len(recent_positions) - 1)

        # 预测位置
        last_pos = recent_positions[-1]
        predicted_x = last_pos[0] + avg_vx * frames_ahead
        predicted_y = last_pos[1] + avg_vy * frames_ahead

        return (predicted_x, predicted_y)

    def get_recent_velocities(self, window=3):
        """
        获取最近几帧的速度数据

        Args:
            window: 窗口大小

        Returns:
            list: 速度数据列表，每个元素为 (vx, vy, speed)
        """
        if len(self.positions) < window + 1:
            return []
        velocities = []
        for i in range(len(self.positions) - window, len(self.positions) - 1):
            vx, vy, speed = calculate_velocity(self.positions[i], self.positions[i+1])
            velocities.append((vx, vy, speed))
        return velocities

    def get_velocity_change(self, window=2):
        """
        获取水平方向速度变化

        Args:
            window: 检测窗口大小

        Returns:
            float: 水平速度变化值
        """
        velocities = self.get_recent_velocities(window)
        if len(velocities) < 2:
            return 0
        # 计算水平速度变化
        vx_change = abs(velocities[-1][0] - velocities[-2][0])
        return vx_change

    def mark_lost(self):
        """标记跟踪丢失一帧"""
        self.lost_frames += 1
        if self.lost_frames > self.max_lost_frames:
            self.is_active = False

    def mark_found(self):
        """标记重新找到目标"""
        self.lost_frames = 0

    def clear_trajectory(self):
        """清除轨迹历史"""
        self.positions.clear()
        self.bboxes.clear()
        self.frame_ids.clear()
