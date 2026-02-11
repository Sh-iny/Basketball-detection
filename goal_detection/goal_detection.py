"""
篮球进球检测系统 - 主程序
基于YOLO目标检测和BoT-SORT跟踪实现进球检测
"""

from ultralytics import YOLO
import cv2
import yaml
import numpy as np
from pathlib import Path
import argparse
import json
import pandas as pd

from .detector.goal_detector import GoalDetector
from .visualizer.video_annotator import VideoAnnotator
from .utils.video_preprocessor import VideoPreprocessor, DetectionSmoother
from .tracker.ball_tracker import BallTracker


class SimpleTracker:
    """简单的位置追踪器，用于传递球的位置信息"""
    def __init__(self, bbox):
        self.bbox = bbox
        x1, y1, x2, y2 = bbox
        self.current_position = ((x1 + x2) / 2, (y1 + y2) / 2)
        self.track_id = 0

    def get_velocity(self):
        return (0, 0)

    def get_trajectory(self, n):
        return [self.current_position]


class BasketballGoalDetectionSystem:
    """篮球进球检测系统"""

    def __init__(self, model_path, config_path, debug=False):
        """
        初始化系统（单YOLO模型）

        Args:
            model_path: 检测模型路径（同时检测篮球和篮筐）
            config_path: 配置文件路径
            debug: 是否开启调试模式
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        print(f"已加载配置: {config_path}")

        # 加载单模型
        self.model = YOLO(model_path)
        print(f"已加载检测模型: {model_path}")

        # 篮筐检测缓存
        self.rim_cache = None
        self.rim_update_interval = self.config['models'].get('rim_update_interval', 5)
        self.rim_smooth_window = self.config['models'].get('rim_smooth_window', 3)
        self.rim_history = []  # 用于平滑

        # 初始化组件
        self.goal_detector = GoalDetector(self.config)
        self.ball_trackers = {}
        self.annotator = None
        self.debug = debug

        # 类别映射
        self.class_names = self.model.names
        print(f"模型类别: {self.class_names}")

        # 调试统计
        if self.debug:
            self.debug_stats = {
                'ball_detected_frames': 0,
                'rim_detected_frames': 0,
                'both_detected_frames': 0,
                'total_frames': 0
            }

        # 视频预处理器和检测平滑器
        self.preprocessor = VideoPreprocessor(config=self.config)
        self.detection_smoother = DetectionSmoother(
            confirm_frames=2,
            lost_tolerance=3
        )

        # 静止球过滤 - 调整为更宽松的参数
        self.static_ball_positions = {}  # {位置key: 静止帧数}
        self.static_threshold = 60  # 静止超过60帧（约2秒）认为是误检
        self.position_tolerance = 30  # 位置容差（像素）

        # 跳跃检测过滤 - 调整为更宽松的参数
        self.position_history = []  # 最近3帧的位置 [(cx, cy), ...]
        self.jump_threshold = 200  # 跳跃距离阈值（像素）

    def process_video(self, video_path, output_path=None):
        """
        处理视频（使用与 predict_video_mp4.py 类似的直接检测方法）

        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径
        """
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 初始化视频写入器
        if output_path:
            self.annotator = VideoAnnotator(output_path, fps, width, height)

        print(f"\n处理视频: {video_path}")
        print(f"分辨率: {width}x{height}, FPS: {fps}, 总帧数: {total_frames}")

        # 重置检测平滑器
        self.detection_smoother.reset()

        # 使用 predict_video_mp4.py 类似的直接检测方法
        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 直接使用原始帧进行检测（与 predict_video_mp4.py 一致）
            # 不进行预处理，保持原始帧质量

            # 单模型检测（与 predict_video_mp4.py 完全一致）
            results = self.model.predict(
                frame,  # 直接使用原始帧
                conf=0.5,  # 与 predict_video_mp4.py 一致
                verbose=False
                # 不指定 imgsz，使用视频原始尺寸
            )
            
            # 解析检测结果
            detections = self._parse_detections(results[0], frame_id)
            
            # 篮筐检测缓存处理（保留以提高性能）
            if frame_id % self.rim_update_interval == 0:
                self.rim_cache = detections['rim']
            else:
                # 使用缓存的篮筐检测结果
                detections['rim'] = self.rim_cache if self.rim_cache else []

            # 调试统计
            if self.debug:
                self.debug_stats['total_frames'] += 1
                if detections['ball']:
                    self.debug_stats['ball_detected_frames'] += 1
                if detections['rim']:
                    self.debug_stats['rim_detected_frames'] += 1
                if detections['ball'] and detections['rim']:
                    self.debug_stats['both_detected_frames'] += 1

            # 更新跟踪器
            self._update_trackers(detections, frame_id)

            # 进球检测
            goal_detected = self._detect_goals(detections, frame_id, frame)

            # 可视化（直接使用原始帧，不需要坐标还原）
            if self.annotator:
                # 直接使用模型的plot方法（与 predict_video_mp4.py 类似）
                annotated_frame = results[0].plot()
                
                # 绘制轨迹（保留轨迹功能）
                if self.config['visualization']['draw_trajectory']:
                    traj_length = self.config['visualization']['trajectory_length']
                    colors = self.config['visualization']['colors']
                    for tracker in self.ball_trackers.values():
                        trajectory = tracker.get_trajectory(traj_length)
                        if len(trajectory) > 1:
                            # 轨迹点已经是原始尺寸，直接绘制
                            points = np.array(trajectory, dtype=np.int32)
                            cv2.polylines(annotated_frame, [points], False, colors['trajectory'], 2)
                
                # 叠加进球检测结果
                if goal_detected or self._is_in_goal_highlight_period(frame_id):
                    cv2.putText(annotated_frame, "GOAL!", (50, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
                # 叠加进球计数
                goal_count = len(self.goal_detector.goal_events)
                goal_text = f"GOALS: {goal_count}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                thickness = 3
                (text_width, text_height), baseline = cv2.getTextSize(goal_text, font, font_scale, thickness)
                padding = 10
                bg_x1 = width - text_width - padding * 2 - 10
                bg_y1 = 10
                bg_x2 = width - 10
                bg_y2 = text_height + padding * 2 + 10
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
                text_x = width - text_width - padding - 10
                text_y = text_height + padding + 10
                cv2.putText(annotated_frame, goal_text, (text_x, text_y),
                           font, font_scale, (0, 255, 0), thickness)
                self.annotator.write(annotated_frame)

            frame_id += 1
            if frame_id % 30 == 0:
                print(f"处理进度: {frame_id}/{total_frames} ({frame_id/total_frames*100:.1f}%)")

        cap.release()
        if self.annotator:
            self.annotator.release()

        # 保存统计信息
        self._save_statistics(output_path)

        print(f"\n处理完成!")
        print(f"总进球数: {len(self.goal_detector.goal_events)}")

        # 输出调试统计
        if self.debug:
            print(f"\n=== 调试统计 ===")
            print(f"总帧数: {self.debug_stats['total_frames']}")
            print(f"检测到球的帧数: {self.debug_stats['ball_detected_frames']} "
                  f"({self.debug_stats['ball_detected_frames']/self.debug_stats['total_frames']*100:.1f}%)")
            print(f"检测到篮筐的帧数: {self.debug_stats['rim_detected_frames']} "
                  f"({self.debug_stats['rim_detected_frames']/self.debug_stats['total_frames']*100:.1f}%)")
            print(f"同时检测到球和篮筐的帧数: {self.debug_stats['both_detected_frames']} "
                  f"({self.debug_stats['both_detected_frames']/self.debug_stats['total_frames']*100:.1f}%)")

    def _parse_detections(self, result, frame_id):
        """
        解析单模型检测结果

        Args:
            result: YOLO检测结果
            frame_id: 帧ID

        Returns:
            dict: 检测结果字典
        """
        detections = {'ball': [], 'rim': [], 'people': []}

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        for i, (box, cls, conf) in enumerate(zip(boxes, classes, confs)):
            cls_id = int(cls)
            track_id = None  # 直接使用predict，没有track_id

            class_name = self.class_names.get(cls_id, 'unknown')
            
            detection = {
                'bbox': box,
                'class_id': cls_id,
                'class_name': class_name,
                'confidence': float(conf),
                'track_id': track_id
            }

            # 根据类别名称分类
            if class_name in ['ball', 'basketball']:
                # 应用静止球过滤
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                if not self._is_static_ball(cx, cy) and not self._is_jump_detection(cx, cy):
                    detections['ball'].append(detection)
            elif class_name in ['rim', 'hoop']:
                detections['rim'].append(detection)
            elif class_name == 'human':
                detections['people'].append(detection)

        return detections

    def _parse_ball_detections(self, result, frame_id):
        """
        解析篮球检测结果

        Args:
            result: YOLO检测结果
            frame_id: 帧ID

        Returns:
            list: 篮球检测结果列表
        """
        ball_detections = []

        if result.boxes is None or len(result.boxes) == 0:
            return ball_detections

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        for i, (box, cls, conf) in enumerate(zip(boxes, classes, confs)):
            cls_id = int(cls)
            track_id = None  # 直接使用predict，没有track_id

            # 获取类别名称
            class_name = self.class_names.get(cls_id, 'unknown')
            
            # 只处理篮球类别
            if class_name not in ['ball', 'basketball']:
                continue

            detection = {
                'bbox': box,
                'class_id': cls_id,
                'class_name': class_name,
                'confidence': float(conf),
                'track_id': track_id
            }

            # 静止球过滤
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            if self._is_static_ball(cx, cy):
                continue

            # 跳跃检测过滤
            if self._is_jump_detection(cx, cy):
                continue

            ball_detections.append(detection)

        return ball_detections

    def _is_static_ball(self, cx, cy):
        """检查是否是静止球（长时间不动的误检）"""
        grid_x = int(cx / self.position_tolerance)
        grid_y = int(cy / self.position_tolerance)
        pos_key = (grid_x, grid_y)

        if pos_key in self.static_ball_positions:
            self.static_ball_positions[pos_key] += 1
        else:
            self.static_ball_positions[pos_key] = 1

        # 清理旧位置
        keys_to_remove = []
        for key in self.static_ball_positions:
            if key != pos_key:
                self.static_ball_positions[key] -= 1
                if self.static_ball_positions[key] <= 0:
                    keys_to_remove.append(key)
        for key in keys_to_remove:
            del self.static_ball_positions[key]

        return self.static_ball_positions[pos_key] > self.static_threshold

    def _is_jump_detection(self, cx, cy):
        """
        检查是否是跳跃误检
        如果当前位置突然跳到很远的地方，认为是误检
        """
        current_pos = (cx, cy)

        # 历史不足，无法判断
        if len(self.position_history) < 1:
            self.position_history.append(current_pos)
            return False

        # 计算与上一帧的距离
        prev_pos = self.position_history[-1]
        dist_to_prev = np.sqrt((cx - prev_pos[0])**2 + (cy - prev_pos[1])**2)

        # 如果跳跃距离超过阈值，认为是误检
        if dist_to_prev > self.jump_threshold:
            # 不更新历史，保持原位置
            return True

        # 更新历史
        self.position_history.append(current_pos)
        if len(self.position_history) > 3:
            self.position_history.pop(0)

        return False

    def _parse_rim_detections(self, result):
        """
        解析篮筐检测结果（只提取rim/hoop类别，取置信度最高的）

        Args:
            result: YOLO检测结果

        Returns:
            list: 篮筐检测结果列表（最多1个）
        """
        rim_detections = []

        if result.boxes is None or len(result.boxes) == 0:
            return rim_detections

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        # 只提取rim/hoop类别的检测结果
        rim_indices = []
        for i, cls_id in enumerate(classes):
            class_name = self.class_names.get(int(cls_id), 'unknown')
            if class_name in ['rim', 'hoop']:
                rim_indices.append(i)

        # 如果有rim检测结果，取置信度最高的
        if len(rim_indices) > 0:
            rim_confs = confs[rim_indices]
            max_conf_idx = rim_indices[np.argmax(rim_confs)]

            box = boxes[max_conf_idx]
            cls_id = int(classes[max_conf_idx])
            conf = confs[max_conf_idx]
            class_name = self.class_names.get(cls_id, 'unknown')

            # 平滑篮筐位置
            smoothed_box = self._smooth_rim_position(box)

            detection = {
                'bbox': smoothed_box,
                'class_id': cls_id,
                'class_name': class_name,
                'confidence': float(conf),
                'track_id': None  # 篮筐不需要跟踪ID
            }

            rim_detections.append(detection)

        return rim_detections

    def _smooth_rim_position(self, new_rim_bbox):
        """
        平滑篮筐位置（避免抖动）

        Args:
            new_rim_bbox: 新的篮筐边界框

        Returns:
            np.ndarray: 平滑后的边界框
        """
        # 添加到历史记录
        self.rim_history.append(new_rim_bbox)

        # 保持窗口大小
        if len(self.rim_history) > self.rim_smooth_window:
            self.rim_history.pop(0)

        # 计算平均位置
        if len(self.rim_history) > 0:
            smoothed_bbox = np.mean(self.rim_history, axis=0)
            return smoothed_bbox
        else:
            return new_rim_bbox

    def _update_trackers(self, detections, frame_id):
        """
        更新篮球跟踪器

        Args:
            detections: 检测结果
            frame_id: 帧ID
        """
        # 单球模式：只维护一个tracker，忽略track_id变化
        if detections['ball']:
            best_ball = max(detections['ball'], key=lambda x: x.get('confidence', 0))

            # 使用固定的track_id=0
            if 0 not in self.ball_trackers:
                self.ball_trackers[0] = BallTracker(0)

            self.ball_trackers[0].update(best_ball['bbox'], frame_id)

    def _restore_detections(self, detections):
        """
        还原检测坐标到原始图像尺寸（用于可视化）
        """
        restored = {'ball': [], 'rim': [], 'people': []}

        for ball_det in detections['ball']:
            bbox = self.preprocessor.restore_bbox(ball_det['bbox'])
            restored['ball'].append({
                **ball_det,
                'bbox': np.array(bbox)
            })

        for rim_det in detections['rim']:
            bbox = self.preprocessor.restore_bbox(rim_det['bbox'])
            restored['rim'].append({
                **rim_det,
                'bbox': np.array(bbox)
            })

        return restored

    def _detect_goals(self, detections, frame_id, frame=None):
        """
        执行进球检测（单球追踪模式）

        Args:
            detections: 检测结果
            frame_id: 帧ID
            frame: 原始帧（用于颜色直方图检测）

        Returns:
            bool: 是否检测到进球
        """
        # 必须检测到篮筐
        if not detections['rim']:
            return False

        # 使用置信度最高的篮筐
        rim_det = max(detections['rim'], key=lambda x: x['confidence'])
        rim_bbox = rim_det['bbox']

        # 单球追踪：选择置信度最高的球，直接使用检测结果
        ball_tracker = None
        if detections['ball']:
            best_ball = max(detections['ball'], key=lambda x: x['confidence'])
            # 创建一个简单的tracker对象来传递位置信息
            ball_tracker = SimpleTracker(best_ball['bbox'])

        # 调用进球检测
        goal_detected = self.goal_detector.check_goal(ball_tracker, rim_bbox, frame_id, frame)

        # 进球后清除轨迹和重置平滑器
        if goal_detected:
            for tracker in self.ball_trackers.values():
                tracker.clear_trajectory()
            self.detection_smoother.reset()

        return goal_detected

    def _annotate_frame(self, frame, detections, goal_detected, frame_id):
        """
        标注帧

        Args:
            frame: 视频帧
            detections: 检测结果
            goal_detected: 是否检测到进球
            frame_id: 帧ID

        Returns:
            标注后的帧
        """
        colors = self.config['visualization']['colors']

        # 绘制篮球检测框
        for ball_det in detections['ball']:
            bbox = ball_det['bbox']
            if hasattr(bbox, 'astype'):
                bbox = bbox.astype(int)
            else:
                bbox = [int(x) for x in bbox]
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                         colors['ball'], 2)
            track_id = ball_det.get('track_id', 0)
            conf = ball_det.get('confidence', 0.5)
            label = f"Ball {track_id}: {conf:.2f}"
            cv2.putText(frame, label, (bbox[0], bbox[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['ball'], 2)

        # 绘制篮筐检测框
        for rim_det in detections['rim']:
            bbox = rim_det['bbox'].astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                         colors['rim'], 2)
            cv2.putText(frame, "Rim", (bbox[0], bbox[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['rim'], 2)

        # 绘制轨迹
        if self.config['visualization']['draw_trajectory']:
            traj_length = self.config['visualization']['trajectory_length']
            for tracker in self.ball_trackers.values():
                trajectory = tracker.get_trajectory(traj_length)
                if len(trajectory) > 1:
                    # 还原轨迹坐标到原始尺寸
                    restored_traj = []
                    for pt in trajectory:
                        restored_pt = self.preprocessor.restore_bbox([pt[0], pt[1], pt[0], pt[1]])
                        restored_traj.append((int(restored_pt[0]), int(restored_pt[1])))
                    points = np.array(restored_traj, dtype=np.int32)
                    cv2.polylines(frame, [points], False, colors['trajectory'], 2)

        # 进球高亮
        if goal_detected or self._is_in_goal_highlight_period(frame_id):
            cv2.putText(frame, "GOAL!", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 3, colors['goal_event'], 5)

        # 进球计数器 - 大号显示在右上角
        goal_count = len(self.goal_detector.goal_events)
        goal_text = f"GOALS: {goal_count}"

        # 获取文本大小以便绘制背景
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        (text_width, text_height), baseline = cv2.getTextSize(goal_text, font, font_scale, thickness)

        # 绘制半透明背景
        padding = 10
        bg_x1 = frame.shape[1] - text_width - padding * 2 - 10
        bg_y1 = 10
        bg_x2 = frame.shape[1] - 10
        bg_y2 = text_height + padding * 2 + 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # 绘制进球计数文字（绿色）
        text_x = frame.shape[1] - text_width - padding - 10
        text_y = text_height + padding + 10
        cv2.putText(frame, goal_text, (text_x, text_y),
                   font, font_scale, (0, 255, 0), thickness)

        # 帧数信息 - 左上角小字
        frame_text = f"Frame: {frame_id}"
        cv2.putText(frame, frame_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    def _is_in_goal_highlight_period(self, frame_id):
        """
        检查是否在进球高亮期

        Args:
            frame_id: 帧ID

        Returns:
            bool: 是否在高亮期
        """
        if not self.goal_detector.goal_events:
            return False
        last_goal_frame = self.goal_detector.goal_events[-1]['frame_id']
        duration = self.config['visualization']['goal_highlight_duration']
        return frame_id - last_goal_frame < duration

    def _save_statistics(self, output_path):
        """
        保存统计信息

        Args:
            output_path: 输出路径
        """
        if output_path:
            output_dir = Path(output_path).parent
        else:
            output_dir = Path(self.config['output']['output_dir'])

        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存JSON
        if self.config['output']['save_events_json']:
            json_path = output_dir / 'goal_events.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.goal_detector.goal_events, f, indent=2, default=str)
            print(f"事件记录已保存: {json_path}")

        # 保存CSV
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
    """主函数"""
    parser = argparse.ArgumentParser(description='篮球进球检测系统（单YOLO）')
    parser.add_argument('--model', type=str, required=True,
                       help='检测模型路径（同时检测篮球和篮筐）')
    parser.add_argument('--video', type=str, required=True,
                       help='输入视频路径')
    parser.add_argument('--config', type=str,
                       default='goal_detection/config/goal_detection_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--output', type=str,
                       help='输出视频路径')
    parser.add_argument('--debug', action='store_true',
                       help='开启调试模式，输出详细检测统计')

    args = parser.parse_args()

    # 创建系统实例
    system = BasketballGoalDetectionSystem(
        args.model,
        args.config,
        debug=args.debug
    )

    # 处理视频
    system.process_video(args.video, args.output)


if __name__ == '__main__':
    main()

