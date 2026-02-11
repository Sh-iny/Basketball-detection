"""
进球检测器模块 - 单球追踪版
只追踪一个球，当球消失后重新出现时自动补全轨迹判断进球
支持颜色直方图变化检测
"""

import cv2
import numpy as np
from collections import deque
from ..utils.geometry import bbox_center


class GoalDetector:
    """进球检测器 - 单球追踪"""

    def __init__(self, config):
        self.config = config
        self.goal_events = []
        self.last_goal_frame = -1

        # 单球追踪状态
        self.last_ball_frame = -1
        self.last_ball_y = None
        self.last_above_rim_y = None  # 记录球在篮筐上方的最后Y坐标
        self.last_above_rim_frame = -1
        self.ball_disappeared = False

        # 球和篮筐半径记录
        self.ball_radius_history = deque(maxlen=10)  # 球半径历史
        self.rim_radius_history = deque(maxlen=10)  # 篮筐半径历史

    def check_goal(self, ball_tracker, rim_bbox, frame_id, frame=None):
        cooldown = self.config['goal_detection']['cooldown_frames']
        if frame_id - self.last_goal_frame < cooldown:
            return False

        rim_center_y = (rim_bbox[1] + rim_bbox[3]) / 2
        rim_center_x = (rim_bbox[0] + rim_bbox[2]) / 2
        rim_width = rim_bbox[2] - rim_bbox[0]
        rim_height = rim_bbox[3] - rim_bbox[1]
        
        # 计算篮筐半径（基于边界框的宽度的一半）
        rim_radius = rim_width / 2
        self.rim_radius_history.append(rim_radius)
        
        # 计算平均篮筐半径
        avg_rim_radius = sum(self.rim_radius_history) / len(self.rim_radius_history) if self.rim_radius_history else rim_radius

        # 当前帧有球
        if ball_tracker and ball_tracker.current_position:
            ball_x = ball_tracker.current_position[0]
            ball_y = ball_tracker.current_position[1]

            # 检查球是否在篮筐水平范围内
            near_rim_x = abs(ball_x - rim_center_x) < rim_width * 1.5

            # 记录球在篮筐上方且水平接近的位置
            if ball_y < rim_center_y and near_rim_x:
                self.last_above_rim_y = ball_y
                self.last_above_rim_frame = frame_id
                self.last_above_rim_x = ball_x

            # 检查进球
            if self.last_above_rim_y is not None:
                frames_since_above = frame_id - self.last_above_rim_frame
                below_rim = ball_y > rim_center_y + rim_height
                still_near_x = abs(ball_x - rim_center_x) < rim_width * 2

                # 球从上方到下方，间隔合理，且水平位置接近
                # 加入颜色直方图变化作为辅助验证
                position_valid = below_rim and still_near_x and 3 <= frames_since_above <= 35

                if position_valid:
                    # 检测水平速度变化
                    velocity_change_valid = False
                    velocity_change = 0
                    collision_detected = False
                    
                    if ball_tracker:
                        # 获取球的半径
                        ball_radius = ball_tracker.current_radius
                        if not ball_radius:
                            # 如果没有当前半径，使用平均半径
                            ball_radius = ball_tracker.get_average_radius()
                        
                        # 计算球和篮筐之间的距离
                        ball_x = ball_tracker.current_position[0]
                        ball_y = ball_tracker.current_position[1]
                        distance_to_rim = np.sqrt((ball_x - rim_center_x)**2 + (ball_y - rim_center_y)**2)
                        
                        # 水平碰撞的定义：篮球中心点到篮筐边界的水平距离等于篮球半径
                        if ball_radius:
                            # 计算篮筐的左右边界
                            rim_left = rim_bbox[0]
                            rim_right = rim_bbox[2]
                            
                            # 计算篮球中心点到篮筐左右边界的水平距离
                            distance_to_left = abs(ball_x - rim_left)
                            distance_to_right = abs(ball_x - rim_right)
                            
                            # 水平碰撞检测：当篮球中心点到篮筐边界的水平距离等于篮球半径时
                            # 考虑到检测误差，使用一个小的容差范围
                            horizontal_collision = (abs(distance_to_left - ball_radius) < ball_radius * 0.3 or 
                                                   abs(distance_to_right - ball_radius) < ball_radius * 0.3)
                            
                            if horizontal_collision:
                                collision_detected = True
                                
                                # 检查当前水平速度
                                vx, vy, _ = ball_tracker.get_velocity()
                                current_horizontal_velocity = abs(vx)
                                
                                # 获取配置参数
                                min_horizontal_velocity = self.config['goal_detection'].get('min_horizontal_velocity_for_change_detection', 0.5)
                                
                                # 只有当水平速度达到阈值时才启动速度变化检测
                                if current_horizontal_velocity >= min_horizontal_velocity:
                                    # 获取速度变化检测参数
                                    velocity_change_threshold = self.config['goal_detection'].get('horizontal_velocity_change_threshold', 0.3)
                                    velocity_change_window = self.config['goal_detection'].get('velocity_change_window', 3)
                                    
                                    velocity_change = ball_tracker.get_velocity_change(window=velocity_change_window)
                                    # 速度变化合理（球被篮筐阻挡）
                                    velocity_change_valid = velocity_change > velocity_change_threshold

                    # 速度变化信息
                    velocity_info = ""
                    if velocity_change > 0.1:
                        velocity_info = f", 水平速度变化: {velocity_change:.2f}"
                    
                    # 碰撞检测信息
                    collision_info = ""
                    if collision_detected:
                        collision_info = ", 检测到碰撞"

                    # 综合验证：位置变化 + 速度变化检测
                    # 只使用速度变化检测的结果作为判据
                    valid = velocity_change_valid
                    
                    # 如果没有检测到速度变化，但球的位置变化符合进球特征，也考虑为进球
                    if not valid and position_valid:
                        # 检查球的垂直速度
                        if ball_tracker:
                            vx, vy, _ = ball_tracker.get_velocity()
                            # 如果球有明显的向下速度，也考虑为进球
                            if vy > 1.0:  # 向下速度大于1.0像素/帧
                                valid = True

                    if valid:
                        print(f"\n[进球检测] @ 帧 {frame_id}")
                        print(f"  - 上方Y: {self.last_above_rim_y:.1f}, 下方Y: {ball_y:.1f}")
                        print(f"  - 篮筐Y: {rim_center_y:.1f}, 间隔: {frames_since_above}帧{velocity_info}{collision_info}")

                        self._record_goal_event(ball_tracker, rim_bbox, frame_id)
                        self.last_goal_frame = frame_id
                        self.last_above_rim_y = None
                        return True

            self.last_ball_frame = frame_id
            self.last_ball_y = ball_y

        return False

    def _record_goal_event(self, ball_tracker, rim_bbox, frame_id):
        """
        记录进球事件

        Args:
            ball_tracker: 篮球跟踪器
            rim_bbox: 篮筐边界框
            frame_id: 帧ID
        """
        # 计算篮筐半径
        rim_width = rim_bbox[2] - rim_bbox[0]
        rim_radius = rim_width / 2
        
        # 获取球的半径
        ball_radius = ball_tracker.current_radius
        if not ball_radius:
            ball_radius = ball_tracker.get_average_radius()
        
        event = {
            'frame_id': frame_id,
            'timestamp': frame_id / 30.0,
            'ball_position': ball_tracker.current_position,
            'rim_position': bbox_center(rim_bbox),
            'ball_velocity': ball_tracker.get_velocity(),
            'ball_radius': ball_radius,
            'rim_radius': rim_radius,
            'trajectory': ball_tracker.get_trajectory(30)
        }
        self.goal_events.append(event)
        print(f"\n{'='*60}")
        print(f"🏀 进球 #{len(self.goal_events)} 已确认！")
        print(f"  - 球半径: {ball_radius:.1f}px")
        print(f"  - 篮筐半径: {rim_radius:.1f}px")
        print(f"{'='*60}\n")

    def get_goal_events(self):
        """获取所有进球事件"""
        return self.goal_events
