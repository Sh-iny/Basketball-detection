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
        
        # 篮筐位置历史（用于检测失败时）
        self.rim_bbox_history = deque(maxlen=5)  # 最近5次篮筐边界框
        self.rim_center_history = deque(maxlen=5)  # 最近5次篮筐中心点

    def check_goal(self, ball_tracker, rim_bbox, frame_id, frame=None, fps=30):
        cooldown = self.config['goal_detection']['cooldown_frames']
        if frame_id - self.last_goal_frame < cooldown:
            return False

        # 处理篮筐检测失败的情况
        if rim_bbox is None:
            # 使用历史篮筐位置
            if self.rim_bbox_history:
                rim_bbox = self.rim_bbox_history[-1]  # 使用最近的篮筐位置
                print(f"[进球检测] 帧 {frame_id}: 使用历史篮筐位置")
            else:
                # 没有历史位置，无法检测进球
                return False
        else:
            # 更新篮筐位置历史
            self.rim_bbox_history.append(rim_bbox)
            rim_center = ((rim_bbox[0] + rim_bbox[2]) / 2, (rim_bbox[1] + rim_bbox[3]) / 2)
            self.rim_center_history.append(rim_center)

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
            # 使用篮筐宽度的1.5倍作为水平范围，确保篮球确实在篮筐附近
            max_horizontal_offset = rim_width * 1.5
            near_rim_x = abs(ball_x - rim_center_x) < max_horizontal_offset

            # 记录球在篮筐上方且水平接近的位置
            # 只有当篮球在篮筐水平范围内时，才记录其在篮筐上方的位置
            # 这样可以避免远处的篮球被误检为进球
            # 球的中心点应该在篮筐顶部的内部靠里一点点，可以是篮球半径的一个比例
            # 计算篮筐的水平边界
            rim_left = rim_center_x - rim_width / 2
            rim_right = rim_center_x + rim_width / 2
            
            # 检查球是否在篮筐水平范围内，使用篮球半径的比例作为容错
            # 这样可以确保球确实在篮筐内部或非常接近
            # 球的中心点应该在篮筐顶部的内部靠里一点点，可以是篮球半径的一个比例
            ball_radius = ball_tracker.get_average_radius() if ball_tracker else 0
            
            # 计算篮筐的实际水平范围，考虑篮球的大小
            # 只有当篮球的中心在篮筐内部时才考虑为可能的进球
            # 使用配置文件中的有效水平范围比例，确保球确实在篮筐内部
            effective_rim_width_ratio = self.config['goal_detection'].get('effective_rim_width_ratio', 0.8)
            effective_rim_width = rim_width * effective_rim_width_ratio
            effective_rim_left = rim_center_x - effective_rim_width / 2
            effective_rim_right = rim_center_x + effective_rim_width / 2
            
            # 球的中心必须在篮筐的有效水平范围内
            in_rim_horizontal = effective_rim_left < ball_x < effective_rim_right
            
            # 球的Y坐标小于篮筐顶部的Y坐标
            if ball_y < rim_bbox[1] and in_rim_horizontal:
                self.last_above_rim_y = ball_y
                self.last_above_rim_frame = frame_id
                self.last_above_rim_x = ball_x

            # 检查进球
            if self.last_above_rim_y is not None:
                frames_since_above = frame_id - self.last_above_rim_frame
                
                # 获取篮筐下部区间范围参数
                rim_bottom_offset_ratio = self.config['goal_detection'].get('rim_bottom_offset_ratio', 0.0)
                max_horizontal_offset_ratio = self.config['goal_detection'].get('max_horizontal_offset_ratio', 2.0)
                
                # 计算篮筐底部位置，考虑垂直偏移
                rim_bottom = rim_bbox[3] + rim_bottom_offset_ratio * rim_height
                # 球的Y坐标大于篮筐底部的Y坐标
                below_rim = ball_y > rim_bottom
                
                # 篮球在篮筐下方可以超出水平范围，但有一定限制
                # 使用篮筐半径的一定比例作为水平范围限制
                # 这样既允许篮球在网内移动，又避免检测到太远的误检
                max_horizontal_offset = rim_width * max_horizontal_offset_ratio  # 篮筐宽度的倍数作为最大水平偏移
                still_near_x = abs(ball_x - rim_center_x) < max_horizontal_offset
                
                # 篮球只需要在篮筐顶部的内部（上方时在水平范围内）
                # 篮筐下方是网，篮球可以在合理范围内超出水平范围
                position_valid = below_rim and still_near_x and 3 <= frames_since_above <= 35
                
                # 增加穿越条件判断：球的轨迹要从篮筐检测框内部穿过篮筐检测框的左/下/右的边到外边
                if position_valid and ball_tracker:
                    # 获取球的轨迹
                    trajectory = ball_tracker.get_trajectory()
                    if len(trajectory) >= 2:
                        # 检查轨迹是否从篮筐内部穿过篮筐的左/下/右边到外边
                        crossed_rim = self._check_rim_crossing(trajectory, rim_bbox)
                        if not crossed_rim:
                            position_valid = False
                
                # 严格进球检测（如果启用）
                if position_valid and self.config['goal_detection'].get('strict_goal_detection', False):
                    strict_valid = True
                    
                    # 只检查垂直方向穿透（篮球在篮筐下方时可以超出水平范围）
                    vertical_threshold = self.config['goal_detection'].get('vertical_penetration_threshold', 0.3)
                    # 球必须穿透篮筐足够深度
                    vertical_penetration = (ball_y - rim_center_y) / rim_height
                    if vertical_penetration < vertical_threshold:
                        strict_valid = False
                    
                    # 如果严格检测失败，位置验证也失败
                    if not strict_valid:
                        position_valid = False

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
                                
                                # 获取配置参数（相对于篮筐宽度/帧）
                                min_horizontal_velocity_ratio = self.config['goal_detection'].get('min_horizontal_velocity_for_change_detection', 0.02)  # 改为相对于篮筐宽度的比例
                                
                                # 考虑帧率的影响，标准化到30fps
                                fps_normalization = 30.0 / fps
                                
                                # 将相对速度阈值转换为像素/帧，并考虑帧率
                                min_horizontal_velocity = min_horizontal_velocity_ratio * rim_width * fps_normalization
                                
                                # 只有当水平速度达到阈值时才启动速度变化检测
                                if current_horizontal_velocity >= min_horizontal_velocity:
                                    # 获取速度变化检测参数（相对于篮筐宽度/帧）
                                    velocity_change_threshold_ratio = self.config['goal_detection'].get('horizontal_velocity_change_threshold', 0.01)  # 改为相对于篮筐宽度的比例
                                    
                                    # 将相对速度变化阈值转换为像素/帧，并考虑帧率
                                    velocity_change_threshold = velocity_change_threshold_ratio * rim_width * fps_normalization
                                    
                                    # 修改：使用位置检测的时间范围来计算速度变化
                                    # 从球在篮筐上方的最后一个点到当前帧
                                    velocity_change_window = frames_since_above
                                    if velocity_change_window < 2:
                                        velocity_change_window = 2
                                    
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
    
    def _check_rim_crossing(self, trajectory, rim_bbox):
        """
        检查球的轨迹是否从篮筐内部穿过篮筐的左/下/右边到外边
        
        Args:
            trajectory: 球的轨迹点列表 [(x1, y1), (x2, y2), ...]
            rim_bbox: 篮筐边界框 [x1, y1, x2, y2]
            
        Returns:
            bool: 如果球的轨迹从篮筐内部穿过篮筐的左/下/右边到外边，返回True；否则返回False
        """
        rim_x1, rim_y1, rim_x2, rim_y2 = rim_bbox
        
        # 检查轨迹中是否有在篮筐内部的点
        inside_points = []
        outside_points = []
        
        for point in trajectory:
            x, y = point
            # 检查点是否在篮筐内部
            if rim_x1 < x < rim_x2 and rim_y1 < y < rim_y2:
                inside_points.append(point)
            else:
                outside_points.append(point)
        
        # 如果没有内部点或外部点，返回False
        if not inside_points or not outside_points:
            return False
        
        # 检查是否有从内部到外部的穿越
        # 遍历轨迹，检查相邻点是否有从内部到外部的穿越
        for i in range(len(trajectory) - 1):
            prev_x, prev_y = trajectory[i]
            curr_x, curr_y = trajectory[i+1]
            
            # 检查前一点是否在篮筐内部，当前点是否在篮筐外部
            prev_inside = rim_x1 < prev_x < rim_x2 and rim_y1 < prev_y < rim_y2
            curr_outside = not (rim_x1 < curr_x < rim_x2 and rim_y1 < curr_y < rim_y2)
            
            if prev_inside and curr_outside:
                # 检查穿越的是左/下/右边
                # 计算两点之间的线段与篮筐边界的交点
                crossed_left = self._line_segment_intersects(prev_x, prev_y, curr_x, curr_y, rim_x1, rim_y1, rim_x1, rim_y2)
                crossed_bottom = self._line_segment_intersects(prev_x, prev_y, curr_x, curr_y, rim_x1, rim_y2, rim_x2, rim_y2)
                crossed_right = self._line_segment_intersects(prev_x, prev_y, curr_x, curr_y, rim_x2, rim_y1, rim_x2, rim_y2)
                
                if crossed_left or crossed_bottom or crossed_right:
                    return True
        
        return False
    
    def _line_segment_intersects(self, x1, y1, x2, y2, x3, y3, x4, y4):
        """
        检查线段 (x1,y1)-(x2,y2) 是否与线段 (x3,y3)-(x4,y4) 相交
        
        Args:
            x1, y1: 第一条线段的起点
            x2, y2: 第一条线段的终点
            x3, y3: 第二条线段的起点
            x4, y4: 第二条线段的终点
            
        Returns:
            bool: 如果两条线段相交，返回True；否则返回False
        """
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        A = (x1, y1)
        B = (x2, y2)
        C = (x3, y3)
        D = (x4, y4)
        
        return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))
