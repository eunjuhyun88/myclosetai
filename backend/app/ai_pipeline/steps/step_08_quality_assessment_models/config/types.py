#!/usr/bin/env python3
"""
Quality Assessment Types - 품질 평가를 위한 데이터 타입 정의
"""

from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class QualityConfig:
    """품질 평가 설정"""
    device: str = "cpu"
    batch_size: int = 1
    image_size: Tuple[int, int] = (512, 512)
    quality_threshold: float = 0.7
    enable_detailed_analysis: bool = True

@dataclass
class QualityResult:
    """품질 평가 결과"""
    success: bool
    overall_score: float = 0.0
    detailed_scores: Dict[str, float] = None
    quality_level: str = "unknown"
    recommendations: List[str] = None
    error_message: Optional[str] = None

@dataclass
class QualityMetrics:
    """품질 지표들"""
    sharpness: float = 0.0
    contrast: float = 0.0
    brightness: float = 0.0
    noise_level: float = 0.0
    color_balance: float = 0.0
    composition: float = 0.0

@dataclass
class AssessmentModel:
    """품질 평가 모델 정보"""
    name: str
    version: str
    supported_metrics: List[str]
    device_requirements: Dict[str, Any]
    accuracy_score: float

# 기본 설정값들
DEFAULT_QUALITY_CONFIG = QualityConfig()
DEFAULT_QUALITY_METRICS = QualityMetrics()

# 품질 레벨
QUALITY_LEVELS = ['excellent', 'good', 'fair', 'poor', 'very_poor']

# 지원하는 품질 지표
SUPPORTED_METRICS = [
    'sharpness', 'contrast', 'brightness', 'noise_level',
    'color_balance', 'composition', 'resolution', 'artifacts'
]

# 디바이스 타입
DEVICE_TYPES = ['cpu', 'cuda', 'mps', 'auto']
