#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 05: Cloth Warping - Configuration
=======================================================

Cloth Warping Step의 모든 설정과 상수를 정의합니다.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Union
from enum import Enum

class WarpingMethod(Enum):
    """옷감 변형 방법"""
    TPS = "tps"
    RAFT = "raft"
    VITON_HD = "viton_hd"
    VGG = "vgg"
    DENSENET = "densenet"
    HR_VITON = "hr_viton"
    ACGPN = "acgpn"
    STYLEGAN = "stylegan"

class QualityLevel(Enum):
    """품질 레벨"""
    EXCELLENT = "excellent"     # 90-100점
    GOOD = "good"              # 75-89점  
    ACCEPTABLE = "acceptable"   # 60-74점
    POOR = "poor"              # 40-59점
    VERY_POOR = "very_poor"    # 0-39점

class FabricType(Enum):
    """직물 타입"""
    COTTON = "cotton"
    SILK = "silk"
    WOOL = "wool"
    POLYESTER = "polyester"
    DENIM = "denim"
    LEATHER = "leather"

@dataclass
class EnhancedClothWarpingConfig:
    """Enhanced Cloth Warping 설정"""
    input_size: tuple = (768, 1024)  # TPS 입력 크기
    warping_strength: float = 1.0
    enable_multi_stage: bool = True
    enable_depth_estimation: bool = True
    enable_quality_enhancement: bool = True
    enable_physics_simulation: bool = True
    device: str = "auto"
    
    # 고급 설정
    tps_control_points: int = 25
    raft_iterations: int = 12
    quality_assessment_enabled: bool = True
    fabric_type: str = "cotton"
    
    # 성능 설정
    batch_size: int = 1
    use_fp16: bool = False
    memory_efficient: bool = True
    
    # 모델 설정
    enable_ensemble: bool = True
    ensemble_models: List[str] = field(default_factory=lambda: ['tps', 'raft', 'viton_hd'])
    ensemble_method: str = 'weighted_average'
    ensemble_confidence_threshold: float = 0.8
    
    # 품질 설정
    quality_level: QualityLevel = QualityLevel.EXCELLENT
    confidence_threshold: float = 0.7
    quality_threshold: float = 0.6
    
    # 물리 시뮬레이션 설정
    physics_simulation_enabled: bool = True
    gravity_effect: bool = True
    wind_effect: bool = False
    wind_strength: float = 0.01
    
    # 전처리/후처리 설정
    auto_preprocessing: bool = True
    auto_postprocessing: bool = True
    strict_data_validation: bool = True

@dataclass
class WarpingResult:
    """옷감 변형 결과"""
    warped_cloth: Optional[Any] = None
    transformation_matrix: Optional[Any] = None
    quality_score: float = 0.0
    confidence_score: float = 0.0
    processing_time: float = 0.0
    method_used: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

# 상수 정의
DEFAULT_INPUT_SIZE = (768, 1024)
DEFAULT_TPS_CONTROL_POINTS = 25
DEFAULT_RAFT_ITERATIONS = 12
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_QUALITY_THRESHOLD = 0.6

# 모델 경로 상수
MODEL_PATHS = {
    'tps': 'models/tps_model.pth',
    'raft': 'models/raft_model.pth',
    'viton_hd': 'models/viton_hd_model.pth',
    'vgg': 'models/vgg_model.pth',
    'densenet': 'models/densenet_model.pth',
    'hr_viton': 'models/hr_viton_model.pth',
    'acgpn': 'models/acgpn_model.pth',
    'stylegan': 'models/stylegan_model.pth'
}

# 품질 레벨별 설정
QUALITY_LEVEL_CONFIGS = {
    QualityLevel.EXCELLENT: {
        'confidence_threshold': 0.9,
        'quality_threshold': 0.85,
        'enable_ensemble': True,
        'enable_physics_simulation': True
    },
    QualityLevel.GOOD: {
        'confidence_threshold': 0.8,
        'quality_threshold': 0.75,
        'enable_ensemble': True,
        'enable_physics_simulation': True
    },
    QualityLevel.ACCEPTABLE: {
        'confidence_threshold': 0.7,
        'quality_threshold': 0.6,
        'enable_ensemble': False,
        'enable_physics_simulation': False
    },
    QualityLevel.POOR: {
        'confidence_threshold': 0.5,
        'quality_threshold': 0.4,
        'enable_ensemble': False,
        'enable_physics_simulation': False
    },
    QualityLevel.VERY_POOR: {
        'confidence_threshold': 0.3,
        'quality_threshold': 0.2,
        'enable_ensemble': False,
        'enable_physics_simulation': False
    }
}
