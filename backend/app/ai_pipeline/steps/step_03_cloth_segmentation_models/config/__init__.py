#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 03: 의류 세그멘테이션 - Config Package
=====================================================================

설정 관련 클래스들을 포함하는 패키지

Author: MyCloset AI Team  
Date: 2025-08-01
Version: 1.0
"""

# types.py에서 클래스들 export
try:
    from .types import (
        SegmentationMethod, ClothCategory, QualityLevel,
        SegmentationModel, SegmentationQuality, ClothSegmentationConfig, SegmentationResult
    )
except ImportError:
    SegmentationMethod = None
    ClothCategory = None
    QualityLevel = None
    SegmentationModel = None
    SegmentationQuality = None
    ClothSegmentationConfig = None
    SegmentationResult = None

# config.py에서 클래스들 export
try:
    from .config import *
except ImportError:
    pass

# EnhancedClothSegmentationConfig 클래스 추가 (import 호환성을 위해)
from dataclasses import dataclass
from typing import Dict, Any
from enum import Enum

# QualityLevel enum 추가 (types.py에 없을 경우)
if QualityLevel is None:
    class QualityLevel(Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        ULTRA = "ultra"

# CLOTHING_TYPES 상수 추가
CLOTHING_TYPES = [
    "shirt", "pants", "dress", "skirt", "jacket", "coat",
    "sweater", "hoodie", "t-shirt", "jeans", "shorts", "blouse"
]

# VISUALIZATION_COLORS 추가
VISUALIZATION_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
]

# ClothSegmentationModel 클래스 추가 (import 호환성을 위해)
@dataclass
class ClothSegmentationModel:
    """의류 세그멘테이션 모델 설정"""
    
    # 기본 설정
    model_type: str = "sam"
    input_size: tuple = (1024, 1024)
    output_size: tuple = (1024, 1024)
    
    # 모델 파라미터
    feature_dim: int = 256
    num_layers: int = 12
    attention_heads: int = 8
    
    # 추론 설정
    batch_size: int = 1
    use_mps: bool = True
    memory_efficient: bool = True
    
    def __post_init__(self):
        """초기화 후 검증"""
        if self.feature_dim <= 0:
            raise ValueError("feature_dim은 양수여야 합니다")
        if self.num_layers <= 0:
            raise ValueError("num_layers는 양수여야 합니다")
        if self.attention_heads <= 0:
            raise ValueError("attention_heads는 양수여야 합니다")

@dataclass
class EnhancedClothSegmentationConfig:
    """향상된 의류 세그멘테이션 설정"""
    
    # 기본 설정
    model_type: str = "sam"
    input_size: tuple = (1024, 1024)
    output_size: tuple = (1024, 1024)
    
    # 모델 파라미터
    feature_dim: int = 256
    num_layers: int = 12
    attention_heads: int = 8
    
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
        if self.feature_dim <= 0:
            raise ValueError("feature_dim은 양수여야 합니다")
        if self.num_layers <= 0:
            raise ValueError("num_layers는 양수여야 합니다")
        if self.attention_heads <= 0:
            raise ValueError("attention_heads는 양수여야 합니다")

__all__ = [
    'SegmentationMethod',
    'ClothCategory', 
    'QualityLevel',
    'SegmentationModel',
    'SegmentationQuality',
    'ClothSegmentationConfig',
    'SegmentationResult',
    'EnhancedClothSegmentationConfig',  # 추가
    'ClothSegmentationModel',  # 추가
    'CLOTHING_TYPES',  # 추가
    'VISUALIZATION_COLORS'  # 추가
]
