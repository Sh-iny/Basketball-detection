#!/usr/bin/env python3
"""
比较SORT跟踪器和原始BallTracker的性能
"""

import os
import sys
from pathlib import Path
import time

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from goal_detection.goal_detection import BasketballGoalDetectionSystem


def test_tracker(tracker_type):
    """
    测试指定类型的跟踪器

    Args:
        tracker_type: 跟踪器类型 ('sort' 或 'original')

    Returns:
        测试结果字典
    """
    # 测试参数
    model_path = "models/BR2/weights/best.pt"
    video_path = "test/basketball2.mp4"
    config_path = "goal_detection/config/goal_detection_config.yaml"
    output_path = f"runs/test_{tracker_type}_tracker/output.mp4"
    
    # 检查文件是否存在
    if not Path(model_path).exists():
        print(f"错误: 模型文件不存在: {model_path}")
        return None
    
    if not Path(video_path).exists():
        print(f"错误: 视频文件不存在: {video_path}")
        return None
    
    if not Path(config_path).exists():
        print(f"错误: 配置文件不存在: {config_path}")
        return None
    
    print(f"\n{'=' * 60}")
    print(f"测试 {tracker_type.upper()} 跟踪器")
    print(f"{'=' * 60}")
    print(f"模型路径: {model_path}")
    print(f"视频路径: {video_path}")
    print(f"配置路径: {config_path}")
    print(f"输出路径: {output_path}")
    print(f"{'=' * 60}")
    
    try:
        # 创建输出目录
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 初始化系统
        system = BasketballGoalDetectionSystem(
            model_path=model_path,
            config_path=config_path,
            debug=True,
            tracker_type=tracker_type
        )
        
        # 记录开始时间
        start_time = time.time()
        
        # 处理视频
        print("\n开始处理视频...")
        system.process_video(video_path, output_path)
        
        # 记录结束时间
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n{'=' * 60}")
        print(f"测试完成！")
        print(f"{'=' * 60}")
        print(f"输出视频: {output_path}")
        print(f"处理时间: {processing_time:.2f} 秒")
        print(f"{'=' * 60}")
        
        # 返回测试结果
        return {
            'tracker_type': tracker_type,
            'processing_time': processing_time,
            'output_path': output_path,
            'debug_stats': system.debug_stats if hasattr(system, 'debug_stats') else None
        }
        
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_trackers():
    """
    比较两种跟踪器的性能
    """
    print("篮球跟踪器性能对比测试")
    print("目标: 比较SORT跟踪器和原始BallTracker的性能差异")
    print()
    
    # 测试SORT跟踪器
    sort_result = test_tracker('sort')
    
    # 测试原始BallTracker
    original_result = test_tracker('original')
    
    # 比较结果
    print(f"\n{'=' * 80}")
    print("跟踪器性能对比结果")
    print(f"{'=' * 80}")
    
    if sort_result and original_result:
        # 提取统计信息
        sort_stats = sort_result['debug_stats']
        original_stats = original_result['debug_stats']
        
        print(f"{'指标':<30} {'SORT跟踪器':<20} {'原始BallTracker':<20}")
        print(f"{'=' * 80}")
        
        # 处理时间
        print(f"{'处理时间 (秒)':<30} {sort_result['processing_time']:<20.2f} {original_result['processing_time']:<20.2f}")
        
        # 检测率
        if sort_stats and original_stats:
            total_frames = sort_stats.get('total_frames', 0)
            if total_frames > 0:
                sort_ball_rate = (sort_stats.get('ball_detected_frames', 0) / total_frames) * 100
                original_ball_rate = (original_stats.get('ball_detected_frames', 0) / total_frames) * 100
                
                print(f"{'球检测率 (%)':<30} {sort_ball_rate:<20.1f} {original_ball_rate:<20.1f}")
                print(f"{'篮筐检测率 (%)':<30} {(sort_stats.get('rim_detected_frames', 0) / total_frames) * 100:<20.1f} {(original_stats.get('rim_detected_frames', 0) / total_frames) * 100:<20.1f}")
                print(f"{'同时检测率 (%)':<30} {(sort_stats.get('both_detected_frames', 0) / total_frames) * 100:<20.1f} {(original_stats.get('both_detected_frames', 0) / total_frames) * 100:<20.1f}")
        
        print(f"{'=' * 80}")
        
        # 分析结果
        if sort_result['processing_time'] < original_result['processing_time']:
            print("✅ SORT跟踪器处理速度更快")
        else:
            print("✅ 原始BallTracker处理速度更快")
        
        if sort_stats and original_stats:
            sort_ball_rate = (sort_stats.get('ball_detected_frames', 0) / sort_stats.get('total_frames', 1)) * 100
            original_ball_rate = (original_stats.get('ball_detected_frames', 0) / original_stats.get('total_frames', 1)) * 100
            
            if sort_ball_rate > original_ball_rate:
                print("✅ SORT跟踪器球检测率更高")
            else:
                print("✅ 原始BallTracker球检测率更高")
        
        print(f"{'=' * 80}")
        print("对比完成！")
        print("查看输出视频以直观比较跟踪效果：")
        print(f"- SORT跟踪器: {sort_result['output_path']}")
        print(f"- 原始BallTracker: {original_result['output_path']}")
    else:
        print("测试结果不完整，无法进行比较")


if __name__ == "__main__":
    compare_trackers()
