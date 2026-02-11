#!/usr/bin/env python3
"""
测试修改后的进球检测系统性能
验证是否与predict_video_mp4.py的检测率一致
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from goal_detection.goal_detection import BasketballGoalDetectionSystem


def test_goal_detection_system():
    """
    测试修改后的进球检测系统
    """
    # 测试参数
    model_path = "models/BR2/weights/best.pt"
    video_path = "output/basketball2.mp4"  # 使用实际存在的测试视频
    config_path = "goal_detection/config/goal_detection_config.yaml"
    output_path = "runs/test_goal_detection/output.mp4"
    
    # 检查文件是否存在
    if not Path(model_path).exists():
        print(f"错误: 模型文件不存在: {model_path}")
        return False
    
    if not Path(video_path).exists():
        print(f"错误: 视频文件不存在: {video_path}")
        return False
    
    if not Path(config_path).exists():
        print(f"错误: 配置文件不存在: {config_path}")
        return False
    
    print("=" * 60)
    print("测试修改后的进球检测系统")
    print("=" * 60)
    print(f"模型路径: {model_path}")
    print(f"视频路径: {video_path}")
    print(f"配置路径: {config_path}")
    print(f"输出路径: {output_path}")
    print("=" * 60)
    
    try:
        # 创建输出目录
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 初始化系统
        system = BasketballGoalDetectionSystem(
            model_path=model_path,
            config_path=config_path,
            debug=True  # 开启调试模式以获取详细统计
        )
        
        # 处理视频
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


def test_predict_video_mp4():
    """
    测试原始的predict_video_mp4.py脚本
    """
    print("\n" + "=" * 60)
    print("测试原始的predict_video_mp4.py脚本")
    print("=" * 60)
    
    # 测试参数
    model_path = "models/BR2/weights/best.pt"
    video_path = "output/basketball2.mp4"  # 使用实际存在的测试视频
    
    # 检查文件是否存在
    if not Path(model_path).exists():
        print(f"错误: 模型文件不存在: {model_path}")
        return False
    
    if not Path(video_path).exists():
        print(f"错误: 视频文件不存在: {video_path}")
        return False
    
    # 运行predict_video_mp4.py
    try:
        print(f"运行: python scripts/predict_video_mp4.py --model {model_path} --video {video_path}")
        os.system(f"python scripts/predict_video_mp4.py --model {model_path} --video {video_path}")
        print("测试完成！")
        return True
    except Exception as e:
        print(f"测试过程中出错: {e}")
        return False


if __name__ == "__main__":
    print("篮球进球检测系统性能测试")
    print("目标: 验证修改后的系统与predict_video_mp4.py的检测率一致")
    print()
    
    # 先测试原始脚本
    test_predict_video_mp4()
    
    # 再测试修改后的系统
    test_goal_detection_system()
    
    print("\n测试完成，请比较两个系统的检测结果")
    print("重点关注:")
    print("1. 球的检测率")
    print("2. 篮筐的检测率")
    print("3. 检测的稳定性")
    print("4. 视频输出质量")
