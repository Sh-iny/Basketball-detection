"""
光流跟踪器模块
使用Lucas-Kanade方法实现篮球跟踪
在目标检测不稳定时增强跟踪稳定性
"""

import cv2
import numpy as np
from collections import deque


class OpticalFlowTracker:
    """
    基于Lucas-Kanade方法的光流跟踪器
    用于跟踪单个篮球目标
    """
    
    def __init__(self, bbox, frame, track_id=0):
        """
        初始化光流跟踪器
        
        Args:
            bbox: 初始边界框 [x1, y1, x2, y2]
            frame: 初始帧（灰度图像）
            track_id: 跟踪ID
        """
        self.track_id = track_id
        self.bbox = np.array(bbox, dtype=np.float32)
        self.prev_frame = frame.copy()
        
        # 光流参数
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # 特征点参数
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=5,
            blockSize=7
        )
        
        # 跟踪状态
        self.prev_points = None
        self.current_points = None
        self.good_old = None  # 用于可视化的旧特征点
        self.good_new = None  # 用于可视化的新特征点
        self.is_active = True
        self.lost_frames = 0
        self.max_lost_frames = 15
        
        # 轨迹历史
        self.position_history = deque(maxlen=30)
        self.bbox_history = deque(maxlen=30)
        
        # 初始化特征点
        self._detect_features(frame)
        
        # 记录初始位置
        center = self._get_bbox_center(self.bbox)
        self.position_history.append(center)
        self.bbox_history.append(self.bbox.copy())
    
    def _detect_features(self, frame):
        """
        在篮球区域内检测角点特征
        
        Args:
            frame: 当前帧（灰度图像）
        """
        x1, y1, x2, y2 = self.bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            self.prev_points = None
            return
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            self.prev_points = None
            return
        
        points = cv2.goodFeaturesToTrack(roi, mask=None, **self.feature_params)
        
        if points is not None and len(points) > 0:
            self.prev_points = points + np.array([x1, y1], dtype=np.float32)
        else:
            self.prev_points = None
    
    def _get_bbox_center(self, bbox):
        """
        计算边界框中心
        
        Args:
            bbox: 边界框 [x1, y1, x2, y2]
        
        Returns:
            tuple: (cx, cy)
        """
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    
    def _get_bbox_size(self, bbox):
        """
        计算边界框大小
        
        Args:
            bbox: 边界框 [x1, y1, x2, y2]
        
        Returns:
            tuple: (width, height)
        """
        return (bbox[2] - bbox[0], bbox[3] - bbox[1])
    
    def update(self, frame, detection=None):
        """
        更新跟踪器
        
        Args:
            frame: 当前帧（灰度图像）
            detection: 检测结果（可选），包含 'bbox' 和 'confidence'
        
        Returns:
            bool: 是否成功更新
        """
        if detection is not None:
            new_bbox = np.array(detection['bbox'], dtype=np.float32)
            self.bbox = new_bbox
            self._detect_features(frame)
            self.lost_frames = 0
            
            center = self._get_bbox_center(self.bbox)
            self.position_history.append(center)
            self.bbox_history.append(self.bbox.copy())
            
            self.prev_frame = frame.copy()
            return True
        
        if self.prev_points is None or len(self.prev_points) == 0:
            self._detect_features(frame)
            self.prev_frame = frame.copy()
            self.lost_frames += 1
            if self.lost_frames > self.max_lost_frames:
                self.is_active = False
            return False
        
        self.current_points, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_frame, frame, self.prev_points, None, **self.lk_params
        )
        
        if self.current_points is None or len(self.current_points) == 0:
            self.lost_frames += 1
            if self.lost_frames > self.max_lost_frames:
                self.is_active = False
            self.prev_frame = frame.copy()
            return False
        
        good_new = self.current_points[status == 1]
        good_old = self.prev_points[status == 1]
        
        # 保存用于可视化
        self.good_new = good_new
        self.good_old = good_old
        
        if len(good_new) < 3:
            self.lost_frames += 1
            if self.lost_frames > self.max_lost_frames:
                self.is_active = False
            self.prev_frame = frame.copy()
            return False
        
        movement = good_new - good_old
        avg_movement = np.mean(movement, axis=0)
        
        self.bbox[:2] += avg_movement
        self.bbox[2:] += avg_movement
        
        center = self._get_bbox_center(self.bbox)
        self.position_history.append(center)
        self.bbox_history.append(self.bbox.copy())
        
        self.prev_points = good_new.reshape(-1, 1, 2)
        self.prev_frame = frame.copy()
        self.lost_frames = 0
        
        return True
    
    def get_state(self):
        """
        获取当前跟踪状态
        
        Returns:
            dict: 包含边界框、中心位置、跟踪ID等信息
        """
        center = self._get_bbox_center(self.bbox)
        width, height = self._get_bbox_size(self.bbox)
        
        return {
            'bbox': self.bbox.copy(),
            'center': center,
            'width': width,
            'height': height,
            'track_id': self.track_id,
            'is_active': self.is_active,
            'lost_frames': self.lost_frames,
            'num_features': len(self.prev_points) if self.prev_points is not None else 0
        }
    
    def get_position(self):
        """
        获取当前位置
        
        Returns:
            tuple: (cx, cy)
        """
        return self._get_bbox_center(self.bbox)
    
    def get_bbox(self):
        """
        获取当前边界框
        
        Returns:
            np.ndarray: [x1, y1, x2, y2]
        """
        return self.bbox.copy()
    
    def get_trajectory(self, length=None):
        """
        获取轨迹历史
        
        Args:
            length: 轨迹长度，None表示全部
        
        Returns:
            list: 轨迹点列表
        """
        if length is None:
            return list(self.position_history)
        return list(self.position_history)[-length:]
    
    def get_velocity(self):
        """
        获取当前速度
        
        Returns:
            tuple: (vx, vy, speed)
        """
        if len(self.position_history) < 2:
            return (0, 0, 0)
        
        prev_pos = self.position_history[-2]
        curr_pos = self.position_history[-1]
        
        vx = curr_pos[0] - prev_pos[0]
        vy = curr_pos[1] - prev_pos[1]
        speed = np.sqrt(vx**2 + vy**2)
        
        return (vx, vy, speed)
    
    def draw_optical_flow(self, frame, color=(0, 255, 0), draw_lines=True, draw_points=True):
        """
        在帧上绘制光流可视化
        
        Args:
            frame: 当前帧（BGR图像）
            color: 绘制颜色
            draw_lines: 是否绘制光流线
            draw_points: 是否绘制特征点
        
        Returns:
            frame: 绘制后的帧
        """
        if self.good_new is None or self.good_old is None:
            return frame
        
        if len(self.good_new) == 0 or len(self.good_old) == 0:
            return frame
        
        mask = np.zeros_like(frame)
        
        for i, (new, old) in enumerate(zip(self.good_new, self.good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            
            if draw_points:
                cv2.circle(frame, (int(a), int(b)), 3, color, -1)
            
            if draw_lines:
                cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color, 2)
        
        if draw_lines:
            frame = cv2.add(frame, mask)
        
        return frame


class BasketballOpticalFlowTracker:
    """
    篮球光流跟踪器
    管理多个篮球目标的光流跟踪
    只跟踪YOLO检测到的篮球，不主动创建新跟踪器
    """
    
    def __init__(self, max_lost_frames=15, min_features=3, iou_threshold=0.3):
        """
        初始化篮球光流跟踪器
        
        Args:
            max_lost_frames: 最大丢失帧数
            min_features: 最小特征点数量
            iou_threshold: IoU匹配阈值
        """
        self.trackers = {}
        self.next_id = 0
        self.max_lost_frames = max_lost_frames
        self.min_features = min_features
        self.iou_threshold = iou_threshold
        self.frame_count = 0
        self.best_tracker_id = None  # 记录最佳跟踪器ID
    
    def update(self, frame, detections):
        """
        更新所有跟踪器
        
        Args:
            frame: 当前帧（BGR图像）
            detections: 检测结果列表，每个元素包含 'bbox' 和 'confidence'
        
        Returns:
            dict: 跟踪结果字典
        """
        self.frame_count += 1
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        matched_trackers = set()
        
        # 按置信度排序检测结果，优先匹配高置信度的检测
        sorted_detections = sorted(detections, key=lambda x: x.get('confidence', 0.5), reverse=True)
        
        # 如果有新的检测结果，优先使用检测结果
        for det in sorted_detections:
            det_bbox = np.array(det['bbox'])
            best_match_id = None
            best_iou = self.iou_threshold
            
            for track_id, tracker in self.trackers.items():
                if track_id in matched_trackers:
                    continue
                
                iou = self._calculate_iou(det_bbox, tracker.get_bbox())
                if iou > best_iou:
                    best_iou = iou
                    best_match_id = track_id
            
            if best_match_id is not None:
                # 更新匹配的跟踪器
                self.trackers[best_match_id].update(gray_frame, det)
                matched_trackers.add(best_match_id)
            else:
                # 没有匹配的跟踪器，创建新的跟踪器
                # 只保留1个跟踪器，确保只跟踪真正的篮球
                if len(self.trackers) == 0:
                    new_tracker = OpticalFlowTracker(
                        det_bbox, gray_frame, track_id=self.next_id
                    )
                    self.trackers[self.next_id] = new_tracker
                    matched_trackers.add(self.next_id)
                    self.best_tracker_id = self.next_id
                    self.next_id += 1
                else:
                    # 如果已有跟踪器但IoU不匹配，检查距离是否足够近
                    # 如果检测结果与现有跟踪器距离很近，更新现有跟踪器
                    det_center = ((det_bbox[0] + det_bbox[2]) / 2, (det_bbox[1] + det_bbox[3]) / 2)
                    
                    closest_id = None
                    min_dist = 100  # 最大距离阈值100像素
                    
                    for track_id, tracker in self.trackers.items():
                        if track_id in matched_trackers:
                            continue
                        
                        tracker_center = tracker.get_position()
                        dist = np.sqrt((det_center[0] - tracker_center[0])**2 + 
                                      (det_center[1] - tracker_center[1])**2)
                        
                        if dist < min_dist:
                            min_dist = dist
                            closest_id = track_id
                    
                    if closest_id is not None:
                        # 更新最近的跟踪器
                        self.trackers[closest_id].update(gray_frame, det)
                        matched_trackers.add(closest_id)
        
        # 更新未匹配的跟踪器（使用光流预测）
        for track_id, tracker in list(self.trackers.items()):
            if track_id not in matched_trackers:
                tracker.update(gray_frame)
                if not tracker.is_active or tracker.lost_frames > self.max_lost_frames:
                    del self.trackers[track_id]
                    if self.best_tracker_id == track_id:
                        self.best_tracker_id = None
        
        # 更新最佳跟踪器ID
        self._update_best_tracker_id()
        
        return self.get_ball_trackers()
    
    def init_tracker(self, gray_frame, bbox):
        """
        使用给定的边界框初始化光流跟踪器
        
        Args:
            gray_frame: 灰度图像
            bbox: 边界框 [x1, y1, x2, y2]
        """
        # 清除现有跟踪器
        self.trackers.clear()
        self.best_tracker_id = None
        
        # 创建新跟踪器
        new_tracker = OpticalFlowTracker(
            np.array(bbox, dtype=np.float32), gray_frame, track_id=self.next_id
        )
        self.trackers[self.next_id] = new_tracker
        self.best_tracker_id = self.next_id
        self.next_id += 1
    
    def _update_best_tracker_id(self):
        """
        更新最佳跟踪器ID
        选择特征点最多且丢失帧数最少的跟踪器
        """
        if not self.trackers:
            self.best_tracker_id = None
            return
        
        best_id = None
        best_score = -1
        
        for track_id, tracker in self.trackers.items():
            if not tracker.is_active:
                continue
            
            # 计算综合得分：特征点数量 - 丢失帧数 * 2
            num_features = len(tracker.prev_points) if tracker.prev_points is not None else 0
            score = num_features - tracker.lost_frames * 2
            
            if score > best_score:
                best_score = score
                best_id = track_id
        
        self.best_tracker_id = best_id
    
    def _calculate_iou(self, bbox1, bbox2):
        """
        计算两个边界框的IoU
        
        Args:
            bbox1: 边界框1 [x1, y1, x2, y2]
            bbox2: 边界框2 [x1, y1, x2, y2]
        
        Returns:
            float: IoU值
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_ball_trackers(self):
        """
        获取所有活跃的篮球跟踪器
        
        Returns:
            dict: 跟踪器字典，key为track_id，value为跟踪状态
        """
        result = {}
        for track_id, tracker in self.trackers.items():
            if tracker.is_active:
                state = tracker.get_state()
                result[track_id] = {
                    'bbox': state['bbox'],
                    'center': state['center'],
                    'track_id': state['track_id'],
                    'confidence': 1.0 - (state['lost_frames'] / self.max_lost_frames),
                    'num_features': state['num_features'],
                    'is_best': track_id == self.best_tracker_id
                }
        return result
    
    def get_best_tracker(self):
        """
        获取最佳跟踪器
        
        Returns:
            dict: 最佳跟踪器状态，如果没有活跃跟踪器则返回None
        """
        if self.best_tracker_id is None or self.best_tracker_id not in self.trackers:
            return None
        
        tracker = self.trackers[self.best_tracker_id]
        if not tracker.is_active:
            return None
        
        state = tracker.get_state()
        return {
            'bbox': state['bbox'],
            'center': state['center'],
            'track_id': state['track_id'],
            'confidence': 1.0 - (state['lost_frames'] / self.max_lost_frames),
            'num_features': state['num_features']
        }
    
    def draw_all_optical_flow(self, frame, draw_best_only=True):
        """
        在帧上绘制所有光流可视化
        
        Args:
            frame: 当前帧（BGR图像）
            draw_best_only: 是否只绘制最佳跟踪器的光流
        
        Returns:
            frame: 绘制后的帧
        """
        colors = [
            (0, 255, 0),    # 绿色 - 最佳跟踪器
            (255, 0, 0),    # 蓝色
            (0, 0, 255),    # 红色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 紫色
        ]
        
        if draw_best_only and self.best_tracker_id is not None:
            if self.best_tracker_id in self.trackers:
                tracker = self.trackers[self.best_tracker_id]
                frame = tracker.draw_optical_flow(frame, color=colors[0])
        else:
            color_idx = 0
            for track_id, tracker in self.trackers.items():
                if tracker.is_active:
                    color = colors[0] if track_id == self.best_tracker_id else colors[color_idx % len(colors)]
                    frame = tracker.draw_optical_flow(frame, color=color)
                    color_idx += 1
        
        return frame
    
    def clear(self):
        """
        清除所有跟踪器
        """
        self.trackers.clear()
        self.next_id = 0
        self.best_tracker_id = None
