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
                
                # 修改：完全基于轨迹进行位置和穿越检测
                if ball_tracker:
                    # 获取球的轨迹
                    trajectory = ball_tracker.get_trajectory()
                    if len(trajectory) >= 2:
                        # 基于轨迹进行位置检测：检查轨迹是否有从篮筐上方到下方的变化
                        position_valid = self._check_position_change(trajectory, rim_bbox, frames_since_above, frame_id)
                        
                        # 基于轨迹进行穿越检测：检查轨迹是否从篮筐内部穿过篮筐的左/下/右的边到外边
                        if position_valid:
                            crossed_rim = self._check_rim_crossing(trajectory, rim_bbox)
                            if not crossed_rim:
                                position_valid = False
                        
                        # 详细调试输出
                        if 990 <= frame_id <= 1020:
                            print(f"[轨迹分析] 帧 {frame_id}")
                            print(f"轨迹长度: {len(trajectory)}")
                            print(f"位置检测结果: {'有效' if position_valid else '无效'}")
                            if position_valid:
                                print(f"穿越检测结果: {'有效' if crossed_rim else '无效'}")
                            
                            # 输出轨迹的前几个点和后几个点
                            print(f"轨迹前3点: {[(round(p[0],1), round(p[1],1)) for p in trajectory[:3]]}")
                            print(f"轨迹后3点: {[(round(p[0],1), round(p[1],1)) for p in trajectory[-3:]]}")
                            
                            # 输出篮筐位置和尺寸
                            rim_x1, rim_y1, rim_x2, rim_y2 = rim_bbox
                            rim_center_x = (rim_x1 + rim_x2) / 2
                            rim_center_y = (rim_y1 + rim_y2) / 2
                            rim_width = rim_x2 - rim_x1
                            rim_height = rim_y2 - rim_y1
                            print(f"篮筐位置: [{rim_x1:.1f}, {rim_y1:.1f}, {rim_x2:.1f}, {rim_y2:.1f}]")
                            print(f"篮筐中心: ({rim_center_x:.1f}, {rim_center_y:.1f})")
                            print(f"篮筐尺寸: 宽={rim_width:.1f}, 高={rim_height:.1f}")
                            
                            # 输出球的当前位置
                            ball_x = ball_tracker.current_position[0]
                            ball_y = ball_tracker.current_position[1]
                            print(f"球的当前位置: ({ball_x:.1f}, {ball_y:.1f})")
                            print(f"球在篮筐上方: {'是' if ball_y < rim_y1 else '否'}")
                            print(f"球在篮筐下方: {'是' if ball_y > (rim_y2 + rim_height * self.config['goal_detection'].get('rim_bottom_offset_ratio', 0.0)) else '否'}")
                            print(f"球在篮筐水平范围内: {'是' if abs(ball_x - rim_center_x) < rim_width * 1.5 else '否'}")
                            print(f"帧数间隔: {frames_since_above}")
                else:
                    # 如果没有轨迹，使用原有的位置检测逻辑
                    position_valid = below_rim and still_near_x and 3 <= frames_since_above <= 35
                
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
    
    def _check_position_change(self, trajectory, rim_bbox, frames_since_above, frame_id):
        """
        基于轨迹检查球的位置是否从篮筐上方变化到篮筐下方
        
        Args:
            trajectory: 球的轨迹点列表 [(x1, y1), (x2, y2), ...]
            rim_bbox: 篮筐边界框 [x1, y1, x2, y2]
            frames_since_above: 从球在篮筐上方到当前帧的间隔帧数（仅用于调试）
            frame_id: 当前帧ID（用于调试）
            
        Returns:
            bool: 如果球的轨迹显示从篮筐上方变化到篮筐下方，返回True；否则返回False
        """
        rim_x1, rim_y1, rim_x2, rim_y2 = rim_bbox
        rim_center_x = (rim_x1 + rim_x2) / 2
        rim_center_y = (rim_y1 + rim_y2) / 2
        rim_width = rim_x2 - rim_x1
        rim_height = rim_y2 - rim_y1
        
        # 获取篮筐底部位置，考虑垂直偏移
        rim_bottom_offset_ratio = self.config['goal_detection'].get('rim_bottom_offset_ratio', 0.0)
        rim_bottom = rim_y2 + rim_bottom_offset_ratio * rim_height
        
        # 检查轨迹中是否有在篮筐上方的点
        above_rim_points = []
        # 检查轨迹中是否有在篮筐下方的点
        below_rim_points = []
        # 检查轨迹中是否有在篮筐内部的点
        inside_rim_points = []
        
        for point in trajectory:
            x, y = point
            # 篮筐上方：球的Y坐标小于篮筐顶部的Y坐标
            if y < rim_y1:
                above_rim_points.append(point)
            # 篮筐下方：球的Y坐标大于篮筐底部的Y坐标
            if y > rim_bottom:
                below_rim_points.append(point)
            # 篮筐内部：球的中心点在篮筐边界内
            if rim_x1 < x < rim_x2 and rim_y1 < y < rim_y2:
                inside_rim_points.append(point)
        
        # 详细调试输出
        if 990 <= frame_id <= 1020:
            print(f"[位置检测分析] 帧 {frame_id}")
            print(f"篮筐上方点数量: {len(above_rim_points)}")
            print(f"篮筐下方点数量: {len(below_rim_points)}")
            print(f"篮筐内部点数量: {len(inside_rim_points)}")
            if above_rim_points:
                print(f"最后一个上方点: ({above_rim_points[-1][0]:.1f}, {above_rim_points[-1][1]:.1f})")
            if below_rim_points:
                print(f"最后一个下方点: ({below_rim_points[-1][0]:.1f}, {below_rim_points[-1][1]:.1f})")
            if inside_rim_points:
                print(f"最后一个内部点: ({inside_rim_points[-1][0]:.1f}, {inside_rim_points[-1][1]:.1f})")
        
        # 检查轨迹中是否有从上方到下方的连续变化
        # 情况1：有明确的上方点和下方点
        if above_rim_points and below_rim_points:
            # 检查上方点是否在篮筐水平范围内
            effective_rim_width_ratio = self.config['goal_detection'].get('effective_rim_width_ratio', 0.8)
            effective_rim_width = rim_width * effective_rim_width_ratio
            effective_rim_left = rim_center_x - effective_rim_width / 2
            effective_rim_right = rim_center_x + effective_rim_width / 2
            
            # 检查是否有上方点在篮筐水平范围内
            valid_above_points = [p for p in above_rim_points if effective_rim_left < p[0] < effective_rim_right]
            if not valid_above_points:
                return False
            
            # 检查下方点是否在篮筐水平范围内（允许更大的误差）
            max_horizontal_offset_ratio = self.config['goal_detection'].get('max_horizontal_offset_ratio', 2.0)
            max_horizontal_offset = rim_width * max_horizontal_offset_ratio
            valid_below_points = [p for p in below_rim_points if abs(p[0] - rim_center_x) < max_horizontal_offset]
            if not valid_below_points:
                return False
            
            # 检查上方点是否在下方点之前（时间顺序正确）
            # 轨迹是按时间顺序排列的，所以最后一个上方点应该在最后一个下方点之前
            # 使用坐标比较而不是对象比较，避免浮点数精度问题
            last_above_idx = -1
            for i, p in enumerate(trajectory):
                if any(abs(p[0] - ap[0]) < 1 and abs(p[1] - ap[1]) < 1 for ap in above_rim_points):
                    last_above_idx = i
            
            first_below_idx = -1
            for i, p in enumerate(trajectory):
                if any(abs(p[0] - bp[0]) < 1 and abs(p[1] - bp[1]) < 1 for bp in below_rim_points):
                    first_below_idx = i
                    break
            
            # 允许上方点和下方点之间有一定间隔（最多20帧）
            if last_above_idx != -1 and first_below_idx != -1 and first_below_idx - last_above_idx <= 20:
                return True
        
        # 情况2：有上方点和内部点（球穿过篮筐但未完全到达下方）
        elif above_rim_points and inside_rim_points:
            # 检查上方点是否在篮筐水平范围内
            effective_rim_width_ratio = self.config['goal_detection'].get('effective_rim_width_ratio', 0.8)
            effective_rim_width = rim_width * effective_rim_width_ratio
            effective_rim_left = rim_center_x - effective_rim_width / 2
            effective_rim_right = rim_center_x + effective_rim_width / 2
            
            # 详细调试输出
            if 990 <= frame_id <= 1020:
                print(f"[水平范围检查] 帧 {frame_id}")
                print(f"篮筐有效水平范围: [{effective_rim_left:.1f}, {effective_rim_right:.1f}]")
                print(f"最后一个上方点: ({above_rim_points[-1][0]:.1f}, {above_rim_points[-1][1]:.1f})")
                print(f"最后一个内部点: ({inside_rim_points[-1][0]:.1f}, {inside_rim_points[-1][1]:.1f})")
                
                # 检查最后一个上方点是否在篮筐水平范围内
                last_above_x = above_rim_points[-1][0]
                in_horizontal_range = effective_rim_left < last_above_x < effective_rim_right
                print(f"最后一个上方点是否在篮筐水平范围内: {'是' if in_horizontal_range else '否'}")
                
                # 检查最后一个内部点是否在篮筐水平范围内
                last_inside_x = inside_rim_points[-1][0]
                in_rim_width = abs(last_inside_x - rim_center_x) < rim_width
                print(f"最后一个内部点是否在篮筐水平范围内: {'是' if in_rim_width else '否'}")
            
            # 直接使用最后一个上方点和内部点进行检查，不依赖于轨迹中的索引
            last_above = above_rim_points[-1]
            last_inside = inside_rim_points[-1]
            
            # 详细调试输出
            if 990 <= frame_id <= 1020:
                print(f"[位置检测修复] 帧 {frame_id}")
                print(f"最后一个上方点X坐标: {last_above[0]:.1f}")
                print(f"篮筐右边界: {rim_x2:.1f}")
                print(f"最后一个上方点是否在篮筐右边界附近: {'是' if abs(last_above[0] - rim_x2) < 10 else '否'}")
            
            # 修复：当最后一个上方点在篮筐右边界附近（10像素内），并且球最终进入了篮筐内部时，也视为有效
            if not (effective_rim_left < last_above[0] < effective_rim_right):
                # 检查最后一个上方点是否在篮筐右边界附近
                if not (abs(last_above[0] - rim_x2) < 10):
                    return False
            
            # 检查最后一个内部点是否在篮筐水平范围内
            if not (abs(last_inside[0] - rim_center_x) < rim_width):
                return False
            
            # 直接返回True，因为轨迹是按时间顺序排列的，最后一个上方点一定在最后一个内部点之前
            # 并且我们已经确认了球从篮筐上方移动到了篮筐内部
            return True
        
        # 其他情况：位置变化不明显或时间顺序不正确
        return False
    
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
