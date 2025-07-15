
"""
AI Pipeline 모듈
"""

# 순환 참조 방지를 위해 필요한 것만 import
__version__ = "1.0.0"
__all__ = []

# 지연 import를 위한 함수들
def get_human_parsing_step():
    from .steps.step_01_human_parsing import HumanParsingStep
    return HumanParsingStep

def get_pose_estimation_step():
    from .steps.step_02_pose_estimation import PoseEstimationStep
    return PoseEstimationStep

def get_cloth_segmentation_step():
    from .steps.step_03_cloth_segmentation import ClothSegmentationStep
    return ClothSegmentationStep

def get_geometric_matching_step():
    from .steps.step_04_geometric_matching import GeometricMatchingStep
    return GeometricMatchingStep

def get_cloth_warping_step():
    from .steps.step_05_cloth_warping import ClothWarpingStep
    return ClothWarpingStep

def get_virtual_fitting_step():
    from .steps.step_06_virtual_fitting import VirtualFittingStep
    return VirtualFittingStep

def get_post_processing_step():
    from .steps.step_07_post_processing import PostProcessingStep
    return PostProcessingStep

def get_quality_assessment_step():
    from .steps.step_08_quality_assessment import QualityAssessmentStep
    return QualityAssessmentStep
