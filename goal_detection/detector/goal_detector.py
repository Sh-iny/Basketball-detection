"""
进球检测器模块 - 单球追踪版
只追踪一个球，当球消失后重新出现时自动补全轨迹判断进球
支持颜色直方图变化检测
"""

import cv2
import numpy as np
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

        # 颜色直方图检测
        self.baseline_hist = None  # 基准直方图（无球时）
        self.hist_change_detected = False  # 检测到直方图变化
        self.hist_change_frame = -1  # 变化发生的帧
        self.hist_threshold = 0.3  # 直方图差异阈值
        self.hist_window = 10  # 直方图变化检测窗口（帧数）

    def check_goal(self, ball_tracker, rim_bbox, frame_id, frame=None):
        cooldown = self.config['goal_detection']['cooldown_frames']
        if frame_id - self.last_goal_frame < cooldown:
            return False

        rim_center_y = (rim_bbox[1] + rim_bbox[3]) / 2
        rim_center_x = (rim_bbox[0] + rim_bbox[2]) / 2
        rim_width = rim_bbox[2] - rim_bbox[0]
        rim_height = rim_bbox[3] - rim_bbox[1]

        # 检测颜色直方图变化
        hist_goal = False
        hist_diff = 0
        if frame is not None:
            hist_goal, hist_diff = self.check_histogram_change(frame, rim_bbox, frame_id)

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
                    # 如果有直方图数据，检查是否也检测到变化
                    hist_info = ""
                    if hist_diff > 0.1:
                        hist_info = f", 直方图差异: {hist_diff:.2f}"

                    print(f"\n[进球检测] @ 帧 {frame_id}")
                    print(f"  - 上方Y: {self.last_above_rim_y:.1f}, 下方Y: {ball_y:.1f}")
                    print(f"  - 篮筐Y: {rim_center_y:.1f}, 间隔: {frames_since_above}帧{hist_info}")

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
        event = {
            'frame_id': frame_id,
            'timestamp': frame_id / 30.0,
            'ball_position': ball_tracker.current_position,
            'rim_position': bbox_center(rim_bbox),
            'ball_velocity': ball_tracker.get_velocity(),
            'trajectory': ball_tracker.get_trajectory(30)
        }
        self.goal_events.append(event)
        print(f"\n{'='*60}")
        print(f"🏀 进球 #{len(self.goal_events)} 已确认！")
        print(f"{'='*60}\n")

    def get_goal_events(self):
        """获取所有进球事件"""
        return self.goal_events

    def _compute_rim_histogram(self, frame, rim_bbox):
        """计算篮筐区域的颜色直方图"""
        x1, y1, x2, y2 = map(int, rim_bbox)

        # 扩展区域以包含球穿过的范围
        h, w = frame.shape[:2]
        pad = int((y2 - y1) * 0.5)
        y1 = max(0, y1 - pad)
        y2 = min(h, y2 + pad)
        x1 = max(0, x1)
        x2 = min(w, x2)

        if x2 <= x1 or y2 <= y1:
            return None

        roi = frame[y1:y2, x1:x2]

        # 转换到HSV空间，对光照变化更鲁棒
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 计算H和S通道的直方图
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        return hist

    def _compare_histograms(self, hist1, hist2):
        """比较两个直方图的相似度，返回差异值（0-1，越大差异越大）"""
        if hist1 is None or hist2 is None:
            return 0

        # 使用相关性比较，返回值-1到1，1表示完全相同
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        # 转换为差异值
        diff = 1 - similarity
        return max(0, diff)

    def check_histogram_change(self, frame, rim_bbox, frame_id):
        """
        检测篮筐区域的颜色直方图变化
        返回: (是否检测到进球模式, 差异值)
        """
        current_hist = self._compute_rim_histogram(frame, rim_bbox)
        if current_hist is None:
            return False, 0

        # 初始化基准直方图
        if self.baseline_hist is None:
            self.baseline_hist = current_hist
            return False, 0

        # 计算与基准的差异
        diff = self._compare_histograms(self.baseline_hist, current_hist)

        # 检测变化模式：进入变化 -> 离开变化
        if diff > self.hist_threshold:
            if not self.hist_change_detected:
                # 首次检测到变化（球进入）
                self.hist_change_detected = True
                self.hist_change_frame = frame_id
        else:
            if self.hist_change_detected:
                # 变化恢复（球离开）
                frames_changed = frame_id - self.hist_change_frame
                self.hist_change_detected = False

                # 变化持续时间合理（3-15帧）则认为是进球
                if 3 <= frames_changed <= 15:
                    # 更新基准直方图
                    self.baseline_hist = current_hist
                    return True, diff

        # 缓慢更新基准直方图（适应光照变化）
        if not self.hist_change_detected:
            alpha = 0.02
            self.baseline_hist = cv2.addWeighted(
                self.baseline_hist, 1 - alpha,
                current_hist, alpha, 0
            )

        return False, diff
