#!/usr/bin/env python3
"""
测试修改后的进球检测系统性能
验证是否与predict_video_mp4.py的检测率一致
支持多种跟踪器测试：SORT、Optical Flow、Original
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from goal_detection.goal_detection import BasketballGoalDetectionSystem


def test_goal_detection_system(tracker_type='sort', debug=False):
    """
    测试修改后的进球检测系统
    
    Args:
        tracker_type: 跟踪器类型 ('sort', 'optical_flow', 'original')
        debug: 是否开启调试模式
    """
    model_path = "models/BR2/weights/best.pt"
    video_path = "test/basketball3.mp4"
    config_path = "goal_detection/config/goal_detection_config.yaml"
    output_path = f"runs/test_goal_detection/output3_{tracker_type}.mp4"
    
    if not Path(model_path).exists():
        print(f"错误: 模型文件不存在: {model_path}")
        return False
    
    if not Path(video_path).exists():
        print(f"错误: 视频文件不存在: {video_path}")
        return False
    
    if not Path(config_path).exists():
        print(f"错误: 配置文件不存在: {config_path}")
        return False
    
    tracker_name = {
        'sort': 'SORT (Kalman + Hungarian)',
        'optical_flow': 'Optical Flow (Lucas-Kanade)',
        'original': 'Original BallTracker'
    }
    
    print("=" * 60)
    print("测试篮球进球检测系统")
    print("=" * 60)
    print(f"模型路径: {model_path}")
    print(f"视频路径: {video_path}")
    print(f"配置路径: {config_path}")
    print(f"输出路径: {output_path}")
    print(f"跟踪器类型: {tracker_name.get(tracker_type, tracker_type)}")
    print("=" * 60)
    
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        system = BasketballGoalDetectionSystem(
            model_path=model_path,
            config_path=config_path,
            debug=debug,
            tracker_type=tracker_type
        )
        
        print("\n开始处理视频...")
        system.process_video(video_path, output_path)
        
        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)
        print(f"输出视频: {output_path}")
        print("检测统计信息已在控制台输出")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_all_trackers():
    """
    比较所有跟踪器的性能
    """
    print("\n" + "=" * 60)
    print("比较所有跟踪器性能")
    print("=" * 60)
    
    trackers = ['sort', 'optical_flow', 'original']
    results = {}
    
    for tracker_type in trackers:
        print(f"\n{'='*60}")
        print(f"测试跟踪器: {tracker_type}")
        print(f"{'='*60}")
        
        success = test_goal_detection_system(tracker_type)
        results[tracker_type] = success
        
        if success:
            print(f"✓ {tracker_type} 测试成功")
        else:
            print(f"✗ {tracker_type} 测试失败")
    
    print("\n" + "=" * 60)
    print("所有跟踪器测试结果汇总")
    print("=" * 60)
    for tracker_type, success in results.items():
        status = "✓ 成功" if success else "✗ 失败"
        print(f"{tracker_type}: {status}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='测试篮球进球检测系统')
    parser.add_argument('--tracker', type=str, default='sort',
                        choices=['sort', 'optical_flow', 'original', 'all'],
                        help='跟踪器类型: sort, optical_flow, original, 或 all (比较所有)')
    parser.add_argument('--debug', action='store_true',
                        help='开启调试模式，输出详细检测统计')
    
    args = parser.parse_args()
    
    print("篮球进球检测系统性能测试")
    print("目标: 验证修改后的进球检测系统性能")
    print()
    
    if args.tracker == 'all':
        compare_all_trackers()
    else:
        test_goal_detection_system(args.tracker, args.debug)
    
    print("\n测试完成！")
    print("重点关注:")
    print("1. 球的检测率")
    print("2. 篮筐的检测率")
    print("3. 检测的稳定性")
    print("4. 视频输出质量")
