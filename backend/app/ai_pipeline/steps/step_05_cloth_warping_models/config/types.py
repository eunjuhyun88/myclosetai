#!/usr/bin/env python3
"""
Cloth Warping Types - 의류 변형을 위한 데이터 타입 정의
"""

from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class WarpingConfig:
    """의류 변형 설정"""
    device: str = "cpu"
    batch_size: int = 1
    image_size: Tuple[int, int] = (512, 512)
    quality_mode: str = "high"
    memory_efficient: bool = True

@dataclass
class WarpingResult:
    """의류 변형 결과"""
    success: bool
    warped_image: Optional[np.ndarray] = None
    transformation_matrix: Optional[np.ndarray] = None
    quality_score: float = 0.0
    error_message: Optional[str] = None

@dataclass
class WarpingModel:
    """의류 변형 모델 정보"""
    name: str
    version: str
    supported_formats: List[str]
    device_requirements: Dict[str, Any]
    memory_usage: int  # MB

@dataclass
class WarpingParameters:
    """의류 변형 파라미터"""
    scale_x: float = 1.0
    scale_y: float = 1.0
    rotation: float = 0.0
    translation_x: float = 0.0
    translation_y: float = 0.0
    shear_x: float = 0.0
    shear_y: float = 0.0

# 기본 설정값들
DEFAULT_WARPING_CONFIG = WarpingConfig()
DEFAULT_WARPING_PARAMETERS = WarpingParameters()

# 지원하는 이미지 포맷
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# 품질 모드
QUALITY_MODES = ['low', 'medium', 'high', 'ultra']

# 디바이스 타입
DEVICE_TYPES = ['cpu', 'cuda', 'mps', 'auto']
