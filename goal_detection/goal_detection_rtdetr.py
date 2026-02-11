"""
篮球进球检测系统 - RT-DETR版本
使用RF-DETR进行篮球检测，YOLO进行篮筐检测
"""

from rfdetr import RFDETRNano
from ultralytics import YOLO
import cv2
import yaml
import numpy as np
from pathlib import Path
import argparse
import json
import pandas as pd

from tracker.ball_tracker import BallTracker
from detector.goal_detector import GoalDetector
from visualizer.video_annotator import VideoAnnotator
from utils.video_preprocessor import VideoPreprocessor


class SimpleTracker:
    """简单的位置追踪器"""
    def __init__(self, bbox):
        self.bbox = bbox
        x1, y1, x2, y2 = bbox
        self.current_position = ((x1 + x2) / 2, (y1 + y2) / 2)
        self.track_id = 0

    def get_velocity(self):
        return (0, 0)

    def get_trajectory(self, n):
        return [self.current_position]


class RTDETRGoalDetectionSystem:
    """RT-DETR篮球进球检测系统"""

    def __init__(self, ball_model_path, rim_model_path, config_path, debug=False):
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        print(f"已加载配置: {config_path}")

        # 加载RT-DETR篮球检测模型
        self.ball_model = RFDETRNano(pretrain=ball_model_path)
        self.ball_model.optimize_for_inference()
        print(f"已加载RT-DETR篮球检测模型: {ball_model_path}")

        # 加载YOLO篮筐检测模型
        self.rim_model = YOLO(rim_model_path)
        print(f"已加载YOLO篮筐检测模型: {rim_model_path}")

        # 篮筐检测缓存
        self.rim_cache = None
        self.rim_update_interval = self.config['models']['rim_model'].get('update_interval', 5)
        self.rim_smooth_window = self.config['models']['rim_model'].get('smooth_window', 3)
        self.rim_history = []

        # 初始化组件
        self.goal_detector = GoalDetector(self.config)
        self.ball_trackers = {}
        self.annotator = None
        self.debug = debug

        # 篮筐模型类别
        self.rim_class_names = self.rim_model.names
        print(f"篮筐模型类别: {self.rim_class_names}")

        # 调试统计
        if self.debug:
            self.debug_stats = {
                'ball_detected_frames': 0,
                'rim_detected_frames': 0,
                'both_detected_frames': 0,
                'total_frames': 0
            }

        # 视频预处理器
        self.preprocessor = VideoPreprocessor(config=self.config)

        # 静止球过滤
        self.static_ball_positions = {}  # {位置key: 静止帧数}
        self.static_threshold = 30  # 静止超过30帧（约1秒）认为是误检
        self.position_tolerance = 20  # 位置容差（像素）

        # 跳跃检测过滤
        self.position_history = []  # 最近3帧的位置 [(cx, cy), ...]
        self.jump_threshold = 150  # 跳跃距离阈值（像素）

    def process_video(self, video_path, output_path=None):
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if output_path:
            self.annotator = VideoAnnotator(output_path, fps, width, height)

        print(f"\n处理视频: {video_path}")
        print(f"分辨率: {width}x{height}, FPS: {fps}, 总帧数: {total_frames}")

        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 篮筐检测（低频）
            if frame_id % self.rim_update_interval == 0:
                rim_results = self.rim_model.predict(
                    frame,
                    conf=self.config['models']['rim_model']['confidence'],
                    imgsz=self.config['models']['rim_model']['imgsz'],
                    verbose=False
                )
                self.rim_cache = self._parse_rim_detections(rim_results[0])

            # RT-DETR篮球检测
            ball_conf = self.config['models']['ball_model']['confidence']
            detections = self.ball_model.predict(frame, threshold=ball_conf)
            ball_detections = self._parse_rtdetr_detections(detections)

            # 合并检测结果
            all_detections = {
                'ball': ball_detections,
                'rim': self.rim_cache if self.rim_cache else [],
                'people': []
            }

            # 调试统计
            if self.debug:
                self.debug_stats['total_frames'] += 1
                if all_detections['ball']:
                    self.debug_stats['ball_detected_frames'] += 1
                if all_detections['rim']:
                    self.debug_stats['rim_detected_frames'] += 1
                if all_detections['ball'] and all_detections['rim']:
                    self.debug_stats['both_detected_frames'] += 1

            # 更新跟踪器
            self._update_trackers(all_detections, frame_id)

            # 进球检测
            goal_detected = self._detect_goals(all_detections, frame_id)

            # 可视化
            if self.annotator:
                frame = self._annotate_frame(frame, all_detections, goal_detected, frame_id)
                self.annotator.write(frame)

            frame_id += 1
            if frame_id % 30 == 0:
                print(f"处理进度: {frame_id}/{total_frames} ({frame_id/total_frames*100:.1f}%)")

        cap.release()
        if self.annotator:
            self.annotator.release()

        self._save_statistics(output_path)

        print(f"\n处理完成!")
        print(f"总进球数: {len(self.goal_detector.goal_events)}")

        if self.debug:
            print(f"\n=== 调试统计 ===")
            print(f"总帧数: {self.debug_stats['total_frames']}")
            print(f"检测到球的帧数: {self.debug_stats['ball_detected_frames']} "
                  f"({self.debug_stats['ball_detected_frames']/self.debug_stats['total_frames']*100:.1f}%)")
            print(f"检测到篮筐的帧数: {self.debug_stats['rim_detected_frames']} "
                  f"({self.debug_stats['rim_detected_frames']/self.debug_stats['total_frames']*100:.1f}%)")
            print(f"同时检测到球和篮筐的帧数: {self.debug_stats['both_detected_frames']} "
                  f"({self.debug_stats['both_detected_frames']/self.debug_stats['total_frames']*100:.1f}%)")

    def _parse_rtdetr_detections(self, detections):
        """解析RT-DETR检测结果（带尺寸和静止过滤）"""
        ball_detections = []

        if len(detections) == 0:
            return ball_detections

        for i in range(len(detections)):
            bbox = detections.xyxy[i]
            conf = detections.confidence[i]

            # 计算尺寸
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]

            # 尺寸过滤：篮球通常是15-80像素的小目标
            if w < 15 or w > 80 or h < 15 or h > 80:
                continue

            # 宽高比过滤：篮球应该接近正方形
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue

            # 静止球过滤
            cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            if self._is_static_ball(cx, cy):
                continue

            detection = {
                'bbox': bbox,
                'class_id': 0,
                'class_name': 'ball',
                'confidence': float(conf),
                'track_id': None
            }
            ball_detections.append(detection)

        return ball_detections

    def _is_static_ball(self, cx, cy):
        """检查是否是静止球（长时间不动的误检）"""
        # 生成位置key（量化到网格）
        grid_x = int(cx / self.position_tolerance)
        grid_y = int(cy / self.position_tolerance)
        pos_key = (grid_x, grid_y)

        # 更新静止计数
        if pos_key in self.static_ball_positions:
            self.static_ball_positions[pos_key] += 1
        else:
            self.static_ball_positions[pos_key] = 1

        # 清理旧的位置记录（只保留最近出现的）
        keys_to_remove = []
        for key in self.static_ball_positions:
            if key != pos_key:
                self.static_ball_positions[key] -= 1
                if self.static_ball_positions[key] <= 0:
                    keys_to_remove.append(key)
        for key in keys_to_remove:
            del self.static_ball_positions[key]

        # 判断是否静止
        return self.static_ball_positions[pos_key] > self.static_threshold

    def _parse_rim_detections(self, result):
        """解析篮筐检测结果"""
        rim_detections = []

        if result.boxes is None or len(result.boxes) == 0:
            return rim_detections

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        rim_indices = []
        for i, cls_id in enumerate(classes):
            class_name = self.rim_class_names.get(int(cls_id), 'unknown')
            if class_name == 'rim':
                rim_indices.append(i)

        if len(rim_indices) > 0:
            rim_confs = confs[rim_indices]
            max_conf_idx = rim_indices[np.argmax(rim_confs)]

            box = boxes[max_conf_idx]
            cls_id = int(classes[max_conf_idx])
            conf = confs[max_conf_idx]

            smoothed_box = self._smooth_rim_position(box)

            detection = {
                'bbox': smoothed_box,
                'class_id': cls_id,
                'class_name': 'rim',
                'confidence': float(conf),
                'track_id': None
            }
            rim_detections.append(detection)

        return rim_detections

    def _smooth_rim_position(self, new_rim_bbox):
        """平滑篮筐位置"""
        self.rim_history.append(new_rim_bbox)
        if len(self.rim_history) > self.rim_smooth_window:
            self.rim_history.pop(0)
        if len(self.rim_history) > 0:
            return np.mean(self.rim_history, axis=0)
        return new_rim_bbox

    def _update_trackers(self, detections, frame_id):
        """更新跟踪器"""
        if detections['ball']:
            best_ball = max(detections['ball'], key=lambda x: x.get('confidence', 0))
            if 0 not in self.ball_trackers:
                self.ball_trackers[0] = BallTracker(0)
            self.ball_trackers[0].update(best_ball['bbox'], frame_id)

    def _detect_goals(self, detections, frame_id):
        """进球检测"""
        if not detections['rim']:
            return False

        rim_det = max(detections['rim'], key=lambda x: x['confidence'])
        rim_bbox = rim_det['bbox']

        ball_tracker = None
        if detections['ball']:
            best_ball = max(detections['ball'], key=lambda x: x['confidence'])
            ball_tracker = SimpleTracker(best_ball['bbox'])

        goal_detected = self.goal_detector.check_goal(ball_tracker, rim_bbox, frame_id)

        if goal_detected:
            for tracker in self.ball_trackers.values():
                tracker.clear_trajectory()

        return goal_detected

    def _annotate_frame(self, frame, detections, goal_detected, frame_id):
        """标注帧"""
        colors = self.config['visualization']['colors']

        # 绘制篮球
        for ball_det in detections['ball']:
            bbox = ball_det['bbox']
            if hasattr(bbox, 'astype'):
                bbox = bbox.astype(int)
            else:
                bbox = [int(x) for x in bbox]
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colors['ball'], 2)
            conf = ball_det.get('confidence', 0.5)
            label = f"Ball: {conf:.2f}"
            cv2.putText(frame, label, (bbox[0], bbox[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['ball'], 2)

        # 绘制篮筐
        for rim_det in detections['rim']:
            bbox = rim_det['bbox'].astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colors['rim'], 2)
            cv2.putText(frame, "Rim", (bbox[0], bbox[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['rim'], 2)

        # 绘制轨迹
        if self.config['visualization']['draw_trajectory']:
            traj_length = self.config['visualization']['trajectory_length']
            for tracker in self.ball_trackers.values():
                trajectory = tracker.get_trajectory(traj_length)
                if len(trajectory) > 1:
                    points = np.array([(int(p[0]), int(p[1])) for p in trajectory], dtype=np.int32)
                    cv2.polylines(frame, [points], False, colors['trajectory'], 2)

        # 进球高亮
        if goal_detected or self._is_in_goal_highlight_period(frame_id):
            cv2.putText(frame, "GOAL!", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 3, colors['goal_event'], 5)

        # 进球计数器
        goal_count = len(self.goal_detector.goal_events)
        goal_text = f"GOALS: {goal_count}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        (text_width, text_height), baseline = cv2.getTextSize(goal_text, font, font_scale, thickness)

        padding = 10
        bg_x1 = frame.shape[1] - text_width - padding * 2 - 10
        bg_y1 = 10
        bg_x2 = frame.shape[1] - 10
        bg_y2 = text_height + padding * 2 + 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        text_x = frame.shape[1] - text_width - padding - 10
        text_y = text_height + padding + 10
        cv2.putText(frame, goal_text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)

        # 帧数信息
        cv2.putText(frame, f"Frame: {frame_id}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    def _is_in_goal_highlight_period(self, frame_id):
        if not self.goal_detector.goal_events:
            return False
        last_goal_frame = self.goal_detector.goal_events[-1]['frame_id']
        duration = self.config['visualization']['goal_highlight_duration']
        return frame_id - last_goal_frame < duration

    def _save_statistics(self, output_path):
        if output_path:
            output_dir = Path(output_path).parent
        else:
            output_dir = Path(self.config['output']['output_dir'])

        output_dir.mkdir(parents=True, exist_ok=True)

        if self.config['output']['save_events_json']:
            json_path = output_dir / 'goal_events.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.goal_detector.goal_events, f, indent=2, default=str)
            print(f"事件记录已保存: {json_path}")

        if self.config['output']['save_statistics'] and self.goal_detector.goal_events:
            df = pd.DataFrame([
                {
                    'goal_number': i+1,
                    'frame_id': event['frame_id'],
                    'timestamp': event['timestamp'],
                    'ball_x': event['ball_position'][0],
                    'ball_y': event['ball_position'][1]
                }
                for i, event in enumerate(self.goal_detector.goal_events)
            ])
            csv_path = output_dir / 'goal_statistics.csv'
            df.to_csv(csv_path, index=False)
            print(f"统计信息已保存: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='篮球进球检测系统（RT-DETR）')
    parser.add_argument('--ball-model', type=str, required=True, help='RT-DETR篮球检测模型路径')
    parser.add_argument('--rim-model', type=str, required=True, help='YOLO篮筐检测模型路径')
    parser.add_argument('--video', type=str, required=True, help='输入视频路径')
    parser.add_argument('--config', type=str,
                       default='goal_detection/config/goal_detection_config_v5.yaml',
                       help='配置文件路径')
    parser.add_argument('--output', type=str, help='输出视频路径')
    parser.add_argument('--debug', action='store_true', help='开启调试模式')

    args = parser.parse_args()

    system = RTDETRGoalDetectionSystem(
        args.ball_model,
        args.rim_model,
        args.config,
        debug=args.debug
    )

    system.process_video(args.video, args.output)


if __name__ == '__main__':
    main()
