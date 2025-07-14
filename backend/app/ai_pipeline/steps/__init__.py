
"""
AI Pipeline Steps - 순환 참조 방지
"""

# 직접 import 대신 지연 로딩 사용
__all__ = [
    'HumanParsingStep',
    'PoseEstimationStep', 
    'ClothSegmentationStep',
    'GeometricMatchingStep',
    'ClothWarpingStep',
    'VirtualFittingStep',
    'PostProcessingStep',
    'QualityAssessmentStep'
]

# 지연 import 함수들
def get_all_steps():
    """모든 step 클래스들을 지연 로딩으로 가져오기"""
    try:
        from .step_01_human_parsing import HumanParsingStep
        from .step_02_pose_estimation import PoseEstimationStep
        from .step_03_cloth_segmentation import ClothSegmentationStep
        from .step_04_geometric_matching import GeometricMatchingStep
        from .step_05_cloth_warping import ClothWarpingStep
        from .step_06_virtual_fitting import VirtualFittingStep
        from .step_07_post_processing import PostProcessingStep
        from .step_08_quality_assessment import QualityAssessmentStep
        
        return {
            'HumanParsingStep': HumanParsingStep,
            'PoseEstimationStep': PoseEstimationStep,
            'ClothSegmentationStep': ClothSegmentationStep,
            'GeometricMatchingStep': GeometricMatchingStep,
            'ClothWarpingStep': ClothWarpingStep,
            'VirtualFittingStep': VirtualFittingStep,
            'PostProcessingStep': PostProcessingStep,
            'QualityAssessmentStep': QualityAssessmentStep
        }
    except ImportError as e:
        print(f"⚠️ Step 클래스 import 실패: {e}")
        return {}