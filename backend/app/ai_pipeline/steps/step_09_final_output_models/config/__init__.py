"""
Final Output 설정 패키지
"""

from .types import *
from .constants import *
from .config import *

# EnhancedFinalOutputConfig 클래스 추가 (import 호환성을 위해)
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class FinalOutputModel:
    """최종 출력 모델 설정"""
    
    # 기본 설정
    output_format: str = "png"
    output_quality: int = 95
    output_resolution: tuple = (1024, 1024)
    
    # 모델 파라미터
    feature_dim: int = 512
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
class EnhancedFinalOutputConfig:
    """향상된 최종 출력 설정"""
    
    # 기본 설정
    output_format: str = "png"
    output_quality: int = 95
    output_resolution: tuple = (1024, 1024)
    
    # 모델 파라미터
    feature_dim: int = 512
    num_layers: int = 12
    attention_heads: int = 8
    
    # 추론 설정
    batch_size: int = 1
    use_mps: bool = True
    memory_efficient: bool = True
    
    # 품질 설정
    confidence_threshold: float = 0.8
    quality_threshold: float = 0.9
    
    # 고급 설정
    use_attention: bool = True
    use_ensemble: bool = True
    ensemble_size: int = 3
    
    # 후처리 설정
    smoothing_factor: float = 0.9
    interpolation_threshold: float = 0.5
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
    'OutputQuality',
    'IntegrationMethod',
    'OutputFormat',
    'ModelConfig',
    'IntegrationConfig',
    'QualityConfig',
    'OutputConfig',
    'PerformanceConfig',
    'FinalOutputConfig',
    'DEFAULT_CONFIG',
    'HIGH_QUALITY_CONFIG',
    'ULTRA_QUALITY_CONFIG',
    'QualityLevel',
    'ConfidenceLevel',
    'OutputStatus',
    'QualityMetrics',
    'ConfidenceMetrics',
    'OutputMetadata',
    'StepResult',
    'TransformerOutput',
    'CrossModalOutput',
    'GeneratorOutput',
    'IntegratedOutput',
    'FinalOutputResult',
    'ModelParameters',
    'QualityParameters',
    'OutputParameters',
    'ValidationResult',
    'PerformanceMetrics',
    'DEFAULT_D_MODEL',
    'DEFAULT_NUM_LAYERS',
    'DEFAULT_NUM_HEADS',
    'DEFAULT_D_FF',
    'DEFAULT_DROPOUT',
    'QUALITY_THRESHOLDS',
    'CONFIDENCE_THRESHOLDS',
    'PSNR_THRESHOLDS',
    'SSIM_THRESHOLDS',
    'LPIPS_THRESHOLDS',
    'RESOLUTION_OPTIONS',
    'COMPRESSION_QUALITY_OPTIONS',
    'MEMORY_LIMITS',
    'BATCH_SIZE_OPTIONS',
    'SUPPORTED_IMAGE_FORMATS',
    'SUPPORTED_METADATA_FORMATS',
    'LOG_LEVELS',
    'LOG_FORMAT',
    'ERROR_CODES',
    'ERROR_MESSAGES',
    'EnhancedFinalOutputConfig',  # 추가
    'FinalOutputModel'  # 추가
]
