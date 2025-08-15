"""
Cloth Warping 설정 패키지
"""
from .types import *
from .constants import *
from .config import *
from dataclasses import dataclass
from typing import Dict, Any

__all__ = [
    'ClothWarpingConfig',
    'WarpingModel',
    'WarpingQuality',
    'WarpingResult',
    'WARPING_CONSTANTS',
    'MODEL_CONFIGS'
]

# ClothWarpingModel 클래스 추가 (import 호환성을 위해)
@dataclass
class ClothWarpingModel:
    """의류 워핑 모델 설정"""
    
    # 기본 설정
    model_type: str = "realvisxl"
    input_size: tuple = (1024, 1024)
    output_size: tuple = (1024, 1024)
    
    # 모델 파라미터
    num_channels: int = 3
    latent_dim: int = 512
    num_layers: int = 12
    
    # 추론 설정
    batch_size: int = 1
    use_mps: bool = True
    memory_efficient: bool = True
    
    # 품질 설정
    quality_threshold: float = 0.8
    detail_level: str = "high"
    
    # 고급 설정
    use_attention: bool = True
    use_style_transfer: bool = True
    use_detail_enhancement: bool = True
    
    def __post_init__(self):
        """초기화 후 검증"""
        if self.num_channels <= 0:
            raise ValueError("num_channels는 양수여야 합니다")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim은 양수여야 합니다")
        if self.num_layers <= 0:
            raise ValueError("num_layers는 양수여야 합니다")
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'model_type': self.model_type,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'num_channels': self.num_channels,
            'latent_dim': self.latent_dim,
            'num_layers': self.num_layers,
            'batch_size': self.batch_size,
            'use_mps': self.use_mps,
            'memory_efficient': self.memory_efficient,
            'quality_threshold': self.quality_threshold,
            'detail_level': self.detail_level,
            'use_attention': self.use_attention,
            'use_style_transfer': self.use_style_transfer,
            'use_detail_enhancement': self.use_detail_enhancement
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ClothWarpingModel':
        """딕셔너리에서 생성"""
        return cls(**config_dict)
    
    def validate(self) -> bool:
        """설정 유효성 검증"""
        try:
            self.__post_init__()
            return True
        except Exception:
            return False

# WARPING_TYPES 상수 추가 (import 호환성을 위해)
WARPING_TYPES = {
    'geometric': {
        'name': 'Geometric Warping',
        'description': '기하학적 변형 기반 의류 워핑',
        'supported_models': ['gmm', 'tps', 'affine'],
        'complexity': 'medium'
    },
    'deformable': {
        'name': 'Deformable Warping',
        'description': '가변형 컨볼루션 기반 의류 워핑',
        'supported_models': ['deformable_conv', 'flow_net'],
        'complexity': 'high'
    },
    'neural': {
        'name': 'Neural Warping',
        'description': '신경망 기반 의류 워핑',
        'supported_models': ['realvisxl', 'stable_diffusion'],
        'complexity': 'very_high'
    },
    'hybrid': {
        'name': 'Hybrid Warping',
        'description': '여러 방법을 결합한 의류 워핑',
        'supported_models': ['geometric+neural', 'deformable+flow'],
        'complexity': 'extreme'
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
