"""
测试单模型架构是否正常运行
"""
import sys
import os
from pathlib import Path

# 解决OpenMP冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

# 直接导入，因为goal_detection是一个包
from goal_detection.goal_detection import BasketballGoalDetectionSystem

def test_single_model():
    """
    测试单模型架构
    """
    print("=" * 80)
    print("测试单模型架构")
    print("=" * 80)
    
    # 模型路径
    model_path = "models/BR2/weights/best.pt"
    config_path = "goal_detection/config/goal_detection_config.yaml"
    
    # 检查文件是否存在
    model_path_obj = Path(model_path)
    config_path_obj = Path(config_path)
    
    if not model_path_obj.exists():
        print(f"❌ 模型文件不存在: {model_path}")
        return False
    
    if not config_path_obj.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    print(f"✅ 模型文件存在: {model_path}")
    print(f"✅ 配置文件存在: {config_path}")
    
    try:
        # 创建系统实例
        system = BasketballGoalDetectionSystem(
            model_path=model_path,
            config_path=config_path,
            debug=True
        )
        print("✅ 成功创建系统实例")
        
        # 测试视频路径（如果存在）
        test_video = "output/basketball2.mp4"
        test_video_obj = Path(test_video)
        
        if test_video_obj.exists():
            print(f"✅ 测试视频存在: {test_video}")
            print("\n开始处理测试视频...")
            
            # 处理视频（只处理前100帧）
            system.process_video(test_video, output_path="test_output_single_model.mp4")
            print("✅ 视频处理完成")
        else:
            print(f"⚠️  测试视频不存在: {test_video}")
            print("✅ 系统初始化成功，但未进行视频处理测试")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_single_model()
    print("\n" + "=" * 80)
    if success:
        print("✅ 单模型架构测试成功！")
    else:
        print("❌ 单模型架构测试失败！")
    print("=" * 80)
