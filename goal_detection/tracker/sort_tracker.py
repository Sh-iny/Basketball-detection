"""
SORT跟踪器模块
使用Kalman Filter + Hungarian Algorithm实现的SORT算法
用于篮球和其他目标的实时跟踪
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from collections import deque


class KalmanBoxTracker:
    """
    使用Kalman滤波器跟踪单个目标
    """
    count = 0

    def __init__(self, bbox):
        """
        初始化Kalman滤波器

        Args:
            bbox: 边界框 [x1, y1, x2, y2]
        """
        # 初始化Kalman滤波器
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]])

        # 调整Kalman滤波器参数以适应篮球运动特点
        # 篮球运动速度快，轨迹变化大，但需要平衡预测激进性
        self.kf.R[2:, 2:] *= 2.0  # 减少测量噪声，提高测量精度
        self.kf.P[4:, 4:] *= 300.0  # 调整初始状态协方差
        self.kf.P *= 3.0  # 调整初始状态协方差
        self.kf.Q[-1, -1] *= 0.01  # 减少过程噪声，使预测更加保守
        self.kf.Q[4:, 4:] *= 0.05  # 减少速度过程噪声，使预测更加稳定

        # 初始化状态
        self.kf.x = self._convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.class_id = None
        self.confidence = 0.0

    def update(self, bbox):
        """
        使用新的检测结果更新跟踪器

        Args:
            bbox: 新的边界框 [x1, y1, x2, y2]
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        # 只使用前4维作为观测值
        z = self._convert_bbox_to_z(bbox)
        self.kf.update(z[:4])

    def predict(self):
        """
        预测目标的下一个位置

        Returns:
            预测的边界框 [x1, y1, x2, y2]
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self._convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        获取当前目标状态

        Returns:
            当前边界框 [x1, y1, x2, y2]
        """
        return self._convert_x_to_bbox(self.kf.x)

    def _convert_bbox_to_z(self, bbox):
        """
        将边界框转换为Kalman滤波器状态

        Args:
            bbox: 边界框 [x1, y1, x2, y2]

        Returns:
            状态向量 [cx, cy, s, r, 0, 0, 0]
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h)
        return np.array([cx, cy, s, r, 0, 0, 0]).reshape((7, 1))

    def _convert_x_to_bbox(self, x, score=None):
        """
        将Kalman滤波器状态转换为边界框

        Args:
            x: 状态向量
            score: 置信度分数

        Returns:
            边界框 [x1, y1, x2, y2, score]
        """
        # 确保x是一维数组
        x = np.squeeze(x)
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        cx = x[0]
        cy = x[1]
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        if score is None:
            return np.array([x1, y1, x2, y2])
        else:
            return np.array([x1, y1, x2, y2, score])


def iou_batch(bb_test, bb_gt):
    """
    计算批量边界框的IoU

    Args:
        bb_test: 测试边界框 [N, 4]
        bb_gt:  ground truth边界框 [M, 4]

    Returns:
        IoU矩阵 [N, M]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h

    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o


class SORT:
    """
    SORT跟踪器类
    使用Kalman Filter + Hungarian Algorithm进行多目标跟踪
    """

    def __init__(self, max_age=15, min_hits=1, iou_threshold=0.15):
        """
        初始化SORT跟踪器

        Args:
            max_age: 目标最大生命周期
            min_hits: 最小命中次数
            iou_threshold: IoU阈值
        """
        # 调整参数以适应篮球跟踪场景
        self.max_age = max_age  # 延长目标生命周期，适应篮球暂时遮挡
        self.min_hits = min_hits  # 减少最小命中次数，更快开始跟踪
        self.iou_threshold = iou_threshold  # 降低IoU阈值，提高匹配成功率
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        更新跟踪器

        Args:
            dets: 检测结果 [N, 5]，格式为 [x1, y1, x2, y2, confidence]

        Returns:
            跟踪结果 [N, 7]，格式为 [x1, y1, x2, y2, id, class_id, confidence]
        """
        self.frame_count += 1

        # 预测所有跟踪器的下一个位置
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # 匹配检测结果和跟踪器
        if len(dets) > 0:
            matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(dets, trks)

            # 更新匹配的跟踪器
            for m in matched:
                self.trackers[m[1]].update(dets[m[0], :4])

            # 为未匹配的检测结果创建新的跟踪器
            for i in unmatched_dets:
                trk = KalmanBoxTracker(dets[i, :4])
                trk.confidence = dets[i, 4]
                if dets.shape[1] > 5:
                    trk.class_id = int(dets[i, 5])
                self.trackers.append(trk)

        # 过滤跟踪器
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if (trk.time_since_update < self.max_age) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id, trk.class_id, trk.confidence])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 7))

    def _associate_detections_to_trackers(self, detections, trackers):
        """
        关联检测结果和跟踪器

        Args:
            detections: 检测结果 [N, 5]
            trackers: 跟踪器预测结果 [M, 5]

        Returns:
            matched: 匹配对
            unmatched_dets: 未匹配的检测结果
            unmatched_trks: 未匹配的跟踪器
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

        iou_matrix = iou_batch(detections[:, :4], trackers[:, :4])

        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > self.iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                row_indices, col_indices = linear_sum_assignment(-iou_matrix)
                matched_indices = np.array(list(zip(row_indices, col_indices)))
                unmatched_detections = []
                for d, det in enumerate(detections):
                    if d not in matched_indices[:, 0]:
                        unmatched_detections.append(d)
                unmatched_trackers = []
                for t, trk in enumerate(trackers):
                    if t not in matched_indices[:, 1]:
                        unmatched_trackers.append(t)
                matches = []
                for m in matched_indices:
                    if iou_matrix[m[0], m[1]] < self.iou_threshold:
                        unmatched_detections.append(m[0])
                        unmatched_trackers.append(m[1])
                    else:
                        matches.append(m.reshape(1, 2))
                if len(matches) == 0:
                    matched_indices = np.empty((0, 2), dtype=int)
                else:
                    matched_indices = np.concatenate(matches, axis=0)
        else:
            matched_indices = np.empty((0, 2), dtype=int)

        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)

        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        return matched_indices, unmatched_detections, unmatched_trackers


class BasketballSORTTracker:
    """
    篮球SORT跟踪器
    专为篮球跟踪场景设计的SORT跟踪器
    """

    def __init__(self, max_age=5, min_hits=3, iou_threshold=0.3):
        """
        初始化篮球SORT跟踪器

        Args:
            max_age: 目标最大生命周期
            min_hits: 最小命中次数
            iou_threshold: IoU阈值
        """
        self.sort_tracker = SORT(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
        self.ball_trackers = {}
        self.frame_count = 0
        self.radius_history = {}

    def update(self, detections):
        """
        更新跟踪器

        Args:
            detections: 检测结果列表
        """
        self.frame_count += 1

        # 转换检测结果格式
        dets = []
        for det in detections:
            bbox = det['bbox']
            confidence = det.get('confidence', 0.5)
            class_id = det.get('class_id', 0)
            dets.append([bbox[0], bbox[1], bbox[2], bbox[3], confidence, class_id])

        dets = np.array(dets) if dets else np.empty((0, 6))

        # 更新SORT跟踪器
        trackers = self.sort_tracker.update(dets)

        # 保存当前的篮球跟踪器，用于后续更新
        current_track_ids = set()
        
        # 更新篮球跟踪器
        for trk in trackers:
            x1, y1, x2, y2, track_id, class_id, confidence = trk
            bbox = [x1, y1, x2, y2]
            # 只跟踪篮球
            if class_id == 0:
                track_id_int = int(track_id)
                current_track_ids.add(track_id_int)
                
                # 更新或创建篮球跟踪器
                self.ball_trackers[track_id_int] = {
                    'bbox': bbox,
                    'track_id': track_id_int,
                    'class_id': int(class_id),
                    'confidence': confidence,
                    'age': 0
                }

                # 计算半径
                width = x2 - x1
                height = y2 - y1
                radius = min(width, height) / 2
                if track_id_int not in self.radius_history:
                    self.radius_history[track_id_int] = deque(maxlen=10)
                self.radius_history[track_id_int].append(radius)
        
        # 移除不再活跃的篮球跟踪器
        inactive_track_ids = [track_id for track_id in self.ball_trackers if track_id not in current_track_ids]
        for track_id in inactive_track_ids:
            del self.ball_trackers[track_id]
            if track_id in self.radius_history:
                del self.radius_history[track_id]

    def get_ball_trackers(self):
        """
        获取篮球跟踪器

        Returns:
            篮球跟踪器字典
        """
        return self.ball_trackers

    def get_ball_radius(self, track_id):
        """
        获取篮球半径

        Args:
            track_id: 跟踪ID

        Returns:
            篮球半径
        """
        if track_id in self.radius_history and len(self.radius_history[track_id]) > 0:
            return np.mean(self.radius_history[track_id])
        return None

    def get_ball_position(self, track_id):
        """
        获取篮球位置

        Args:
            track_id: 跟踪ID

        Returns:
            篮球位置 (cx, cy)
        """
        if track_id in self.ball_trackers:
            bbox = self.ball_trackers[track_id]['bbox']
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            return (cx, cy)
        return None

    def get_ball_velocity(self, track_id):
        """
        获取篮球速度

        Args:
            track_id: 跟踪ID

        Returns:
            篮球速度 (vx, vy, speed)
        """
        # 简单实现，后续可根据历史位置计算速度
        return 0, 0, 0

    def clear(self):
        """
        清除跟踪器
        """
        self.ball_trackers = {}
        self.radius_history = {}
