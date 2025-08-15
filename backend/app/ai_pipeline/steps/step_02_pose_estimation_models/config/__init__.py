"""
Pose Estimation 설정 패키지
"""
from .types import *
from .constants import *
from .config import *
from dataclasses import dataclass
from typing import Dict, Any
from enum import Enum

# QualityLevel enum 추가
class QualityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"

# KEYPOINTS 상수 추가
COCO_17_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

OPENPOSE_18_KEYPOINTS = [
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "right_hip",
    "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
    "right_eye", "left_eye", "right_ear", "left_ear"
]

# SKELETON_CONNECTIONS 추가
SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13)
]

# KEYPOINT_COLORS 추가
KEYPOINT_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0), (64, 0, 64)
]

__all__ = [
    'PoseModel',
    'PoseQuality', 
    'EnhancedPoseConfig',
    'PoseResult',
    'COCO_17_KEYPOINTS',
    'OPENPOSE_18_KEYPOINTS',
    'SKELETON_CONNECTIONS',
    'KEYPOINT_COLORS',
    'PoseEstimationModel',  # 추가
    'EnhancedPoseEstimationConfig',  # 추가
    'QualityLevel',  # 추가
    'KEYPOINTS',  # 추가
    'VISUALIZATION_COLORS'  # 추가
]

# KEYPOINTS와 VISUALIZATION_COLORS 별칭 추가
KEYPOINTS = COCO_17_KEYPOINTS
VISUALIZATION_COLORS = KEYPOINT_COLORS

# PoseEstimationModel 클래스 추가 (import 호환성을 위해)
@dataclass
class PoseEstimationModel:
    """포즈 추정 모델 설정"""
    
    # 기본 설정
    model_type: str = "openpose"
    input_size: tuple = (512, 512)
    output_size: tuple = (512, 512)
    
    # 모델 파라미터
    num_keypoints: int = 17
    feature_dim: int = 256
    num_layers: int = 8
    
    # 추론 설정
    batch_size: int = 1
    use_mps: bool = True
    memory_efficient: bool = True
    
    def __post_init__(self):
        """초기화 후 검증"""
        if self.num_keypoints <= 0:
            raise ValueError("num_keypoints는 양수여야 합니다")
        if self.feature_dim <= 0:
            raise ValueError("feature_dim은 양수여야 합니다")
        if self.num_layers <= 0:
            raise ValueError("num_layers는 양수여야 합니다")

# EnhancedPoseEstimationConfig 클래스 추가 (import 호환성을 위해)
@dataclass
class EnhancedPoseEstimationConfig:
    """향상된 포즈 추정 설정"""
    
    # 기본 설정
    model_type: str = "openpose"
    input_size: tuple = (512, 512)
    output_size: tuple = (512, 512)
    
    # 모델 파라미터
    num_keypoints: int = 17
    feature_dim: int = 256
    num_layers: int = 8
    
    # 추론 설정
    batch_size: int = 1
    use_mps: bool = True
    memory_efficient: bool = True
    
    # 품질 설정
    confidence_threshold: float = 0.7
    quality_threshold: float = 0.8
    
    # 고급 설정
    use_attention: bool = True
    use_ensemble: bool = True
    ensemble_size: int = 3
    
    # 후처리 설정
    smoothing_factor: float = 0.8
    interpolation_threshold: float = 0.3
    use_temporal_smoothing: bool = True
    
    def __post_init__(self):
        """초기화 후 검증"""
        if self.num_keypoints <= 0:
            raise ValueError("num_keypoints는 양수여야 합니다")
        if self.feature_dim <= 0:
            raise ValueError("feature_dim은 양수여야 합니다")
        if self.num_layers <= 0:
            raise ValueError("num_layers는 양수여야 합니다")
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'model_type': self.model_type,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'num_keypoints': self.num_keypoints,
            'feature_dim': self.feature_dim,
            'num_layers': self.num_layers,
            'batch_size': self.batch_size,
            'use_mps': self.use_mps,
            'memory_efficient': self.memory_efficient,
            'confidence_threshold': self.confidence_threshold,
            'quality_threshold': self.quality_threshold,
            'use_attention': self.use_attention,
            'use_ensemble': self.use_ensemble,
            'ensemble_size': self.ensemble_size,
            'smoothing_factor': self.smoothing_factor,
            'interpolation_threshold': self.interpolation_threshold,
            'use_temporal_smoothing': self.use_temporal_smoothing
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EnhancedPoseEstimationConfig':
        """딕셔너리에서 생성"""
        return cls(**config_dict)
    
    def validate(self) -> bool:
        """설정 유효성 검증"""
        try:
            self.__post_init__()
            return True
        except Exception:
            return False
