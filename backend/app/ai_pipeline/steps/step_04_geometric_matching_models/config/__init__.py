"""
기하학적 매칭 설정 모듈
"""

from .config import GeometricMatchingConfig, ProcessingStatus
from dataclasses import dataclass
from typing import Dict, Any, List

__all__ = ['GeometricMatchingConfig', 'ProcessingStatus']

# EnhancedGeometricMatchingConfig 클래스 추가 (import 호환성을 위해)
@dataclass
class EnhancedGeometricMatchingConfig:
    """향상된 기하학적 매칭 설정"""
    
    # 기본 설정
    model_type: str = "gmm"
    input_size: tuple = (512, 512)
    output_size: tuple = (512, 512)
    
    # 모델 파라미터
    num_keypoints: int = 68
    feature_dim: int = 256
    num_layers: int = 6
    
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
    ensemble_size: int = 4
    
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
            'ensemble_size': self.ensemble_size
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EnhancedGeometricMatchingConfig':
        """딕셔너리에서 생성"""
        return cls(**config_dict)
    
    def validate(self) -> bool:
        """설정 유효성 검증"""
        try:
            self.__post_init__()
            return True
        except Exception:
            return False

# GeometricMatchingModel 클래스 추가 (import 호환성을 위해)
@dataclass
class GeometricMatchingModel:
    """기하학적 매칭 모델 설정"""
    
    # 기본 설정
    model_type: str = "gmm"
    input_size: tuple = (512, 512)
    output_size: tuple = (512, 512)
    
    # 모델 파라미터
    num_keypoints: int = 68
    feature_dim: int = 256
    num_layers: int = 6
    
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
    ensemble_size: int = 4
    
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
            'ensemble_size': self.ensemble_size
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GeometricMatchingModel':
        """딕셔너리에서 생성"""
        return cls(**config_dict)
    
    def validate(self) -> bool:
        """설정 유효성 검증"""
        try:
            self.__post_init__()
            return True
        except Exception:
            return False

# QualityLevel 클래스 추가 (import 호환성을 위해)
@dataclass
class QualityLevel:
    """품질 수준 설정"""
    
    # 품질 수준 상수
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA_HIGH = "ultra_high"
    
    # 품질 수준별 설정
    LEVELS = {
        LOW: {
            'confidence_threshold': 0.5,
            'quality_threshold': 0.6,
            'ensemble_size': 2,
            'use_attention': False
        },
        MEDIUM: {
            'confidence_threshold': 0.6,
            'quality_threshold': 0.7,
            'ensemble_size': 3,
            'use_attention': True
        },
        HIGH: {
            'confidence_threshold': 0.7,
            'quality_threshold': 0.8,
            'ensemble_size': 4,
            'use_attention': True
        },
        ULTRA_HIGH: {
            'confidence_threshold': 0.8,
            'quality_threshold': 0.9,
            'ensemble_size': 5,
            'use_attention': True
        }
    }
    
    @classmethod
    def get_config(cls, level: str) -> Dict[str, Any]:
        """품질 수준별 설정 반환"""
        return cls.LEVELS.get(level, cls.LEVELS[cls.MEDIUM])
    
    @classmethod
    def get_available_levels(cls) -> List[str]:
        """사용 가능한 품질 수준 반환"""
        return list(cls.LEVELS.keys())
    
    @classmethod
    def is_valid_level(cls, level: str) -> bool:
        """품질 수준 유효성 검사"""
        return level in cls.LEVELS

# MATCHING_TYPES 상수 추가 (import 호환성을 위해)
MATCHING_TYPES = {
    'gmm': {
        'name': 'Gaussian Mixture Model',
        'description': '가우시안 혼합 모델 기반 기하학적 매칭',
        'complexity': 'medium',
        'accuracy': 'high',
        'speed': 'fast'
    },
    'tps': {
        'name': 'Thin Plate Spline',
        'description': '얇은 판 스플라인 기반 변형 매칭',
        'complexity': 'high',
        'accuracy': 'very_high',
        'speed': 'medium'
    },
    'affine': {
        'name': 'Affine Transformation',
        'description': '아핀 변환 기반 매칭',
        'complexity': 'low',
        'accuracy': 'medium',
        'speed': 'very_fast'
    },
    'projective': {
        'name': 'Projective Transformation',
        'description': '투영 변환 기반 매칭',
        'complexity': 'medium',
        'accuracy': 'high',
        'speed': 'fast'
    },
    'elastic': {
        'name': 'Elastic Deformation',
        'description': '탄성 변형 기반 매칭',
        'complexity': 'very_high',
        'accuracy': 'very_high',
        'speed': 'slow'
    }
}

# VISUALIZATION_COLORS 상수 추가 (import 호환성을 위해)
VISUALIZATION_COLORS = {
    'primary': '#FF6B6B',      # 빨간색 (주요 영역)
    'secondary': '#4ECDC4',    # 청록색 (보조 영역)
    'accent': '#45B7D1',       # 파란색 (강조)
    'success': '#96CEB4',      # 초록색 (성공)
    'warning': '#FFEAA7',      # 노란색 (경고)
    'error': '#DDA0DD',        # 보라색 (오류)
    'info': '#F8BBD9',         # 분홍색 (정보)
    'neutral': '#D3D3D3',      # 회색 (중립)
    'background': '#FFFFFF',   # 흰색 (배경)
    'text': '#2C3E50',         # 진한 파란색 (텍스트)
    'border': '#BDC3C7',       # 연한 회색 (테두리)
    'highlight': '#E74C3C',    # 진한 빨간색 (하이라이트)
    'shadow': '#34495E',       # 진한 회색 (그림자)
    'transparent': 'rgba(0,0,0,0)',  # 투명
    'gradient_start': '#667eea',      # 그라데이션 시작
    'gradient_end': '#764ba2'         # 그라데이션 끝
}
