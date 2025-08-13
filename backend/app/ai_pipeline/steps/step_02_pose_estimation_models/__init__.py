"""
Pose Estimation Step 패키지
"""

# 기존 step.py에서 PoseEstimationStep import (기존 기능 보존)
try:
    from .step import PoseEstimationStep as OriginalPoseEstimationStep
    ORIGINAL_STEP_AVAILABLE = True
except ImportError as e:
    ORIGINAL_STEP_AVAILABLE = False
    print(f"⚠️ 기존 step.py import 실패: {e}")

# 새로운 모듈화된 버전 import
try:
    from .step_modularized import PoseEstimationStep as ModularizedPoseEstimationStep
    MODULARIZED_STEP_AVAILABLE = True
except ImportError as e:
    MODULARIZED_STEP_AVAILABLE = False
    print(f"⚠️ 모듈화된 step_modularized.py import 실패: {e}")

# PoseEstimationModelLoader import
try:
    from .pose_estimation_model_loader import PoseEstimationModelLoader
    MODEL_LOADER_AVAILABLE = True
except ImportError as e:
    MODEL_LOADER_AVAILABLE = False
    print(f"⚠️ PoseEstimationModelLoader import 실패: {e}")

# 기본적으로 모듈화된 버전을 사용하되, 실패시 기존 버전 사용
if MODULARIZED_STEP_AVAILABLE:
    PoseEstimationStep = ModularizedPoseEstimationStep
    print("✅ 모듈화된 PoseEstimationStep 사용")
elif ORIGINAL_STEP_AVAILABLE:
    PoseEstimationStep = OriginalPoseEstimationStep
    print("✅ 기존 PoseEstimationStep 사용")
else:
    PoseEstimationStep = None
    print("❌ PoseEstimationStep을 사용할 수 없습니다")

# 팩토리 함수들
try:
    from .step_modularized import create_pose_estimation_step, create_pose_estimation_step_sync
    FACTORY_FUNCTIONS_AVAILABLE = True
except ImportError:
    try:
        from .step import create_pose_estimation_step, create_pose_estimation_step_sync
        FACTORY_FUNCTIONS_AVAILABLE = True
    except ImportError:
        FACTORY_FUNCTIONS_AVAILABLE = False
        print("⚠️ 팩토리 함수들을 사용할 수 없습니다")

__all__ = [
    'PoseEstimationStep',
    'PoseEstimationModelLoader',
    'create_pose_estimation_step',
    'create_pose_estimation_step_sync',
    'ORIGINAL_STEP_AVAILABLE',
    'MODULARIZED_STEP_AVAILABLE',
    'MODEL_LOADER_AVAILABLE',
    'FACTORY_FUNCTIONS_AVAILABLE'
]
