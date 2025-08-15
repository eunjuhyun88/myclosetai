"""
Quality Assessment 설정 패키지
"""
from .types import *
from .constants import *
from .config import *
from dataclasses import dataclass
from typing import Dict, Any

__all__ = [
    'QualityAssessmentConfig',
    'AssessmentModel',
    'AssessmentQuality',
    'AssessmentResult',
    'ASSESSMENT_CONSTANTS',
    'MODEL_CONFIGS'
]

# EnhancedQualityAssessmentConfig 클래스 추가 (import 호환성을 위해)
@dataclass
class EnhancedQualityAssessmentConfig:
    """향상된 품질 평가 설정"""
    
    # 기본 설정
    model_type: str = "clip"
    input_size: tuple = (224, 224)
    output_size: tuple = (1,)
    
    # 모델 파라미터
    num_channels: int = 3
    feature_dim: int = 512
    num_layers: int = 12
    
    # 추론 설정
    batch_size: int = 1
    use_mps: bool = True
    memory_efficient: bool = True
    
    # 품질 설정
    quality_threshold: float = 0.8
    confidence_threshold: float = 0.7
    
    # 고급 설정
    use_attention: bool = True
    use_ensemble: bool = True
    ensemble_size: int = 4
    use_multimodal: bool = True
    
    def __post_init__(self):
        """초기화 후 검증"""
        if self.num_channels <= 0:
            raise ValueError("num_channels는 양수여야 합니다")
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
            'num_channels': self.num_channels,
            'feature_dim': self.feature_dim,
            'num_layers': self.num_layers,
            'batch_size': self.batch_size,
            'use_mps': self.use_mps,
            'memory_efficient': self.memory_efficient,
            'quality_threshold': self.quality_threshold,
            'confidence_threshold': self.confidence_threshold,
            'use_attention': self.use_attention,
            'use_ensemble': self.use_ensemble,
            'ensemble_size': self.ensemble_size,
            'use_multimodal': self.use_multimodal
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EnhancedQualityAssessmentConfig':
        """딕셔너리에서 생성"""
        return cls(**config_dict)
    
    def validate(self) -> bool:
        """설정 유효성 검증"""
        try:
            self.__post_init__()
            return True
        except Exception:
            return False

# ASSESSMENT_TYPES 상수 추가 (import 호환성을 위해)
ASSESSMENT_TYPES = {
    'aesthetic': {
        'name': 'Aesthetic Assessment',
        'description': '미학적 품질 평가',
        'metrics': ['composition', 'color_balance', 'visual_appeal'],
        'weight': 0.3
    },
    'technical': {
        'name': 'Technical Assessment',
        'description': '기술적 품질 평가',
        'metrics': ['sharpness', 'noise_level', 'resolution'],
        'weight': 0.4
    },
    'semantic': {
        'name': 'Semantic Assessment',
        'description': '의미적 품질 평가',
        'metrics': ['clothing_fit', 'pose_naturalness', 'style_consistency'],
        'weight': 0.3
    },
    'comprehensive': {
        'name': 'Comprehensive Assessment',
        'description': '종합적 품질 평가',
        'metrics': ['all_above'],
        'weight': 1.0
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
