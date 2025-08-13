#!/usr/bin/env python3
"""
🔥 Final Output 설정 파일
================================================================================

✅ 모델 파라미터 설정
✅ 시스템 설정
✅ 품질 기준 설정
✅ 출력 옵션 설정

Author: MyCloset AI Team
Date: 2025-08-13
Version: 1.0
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum

class OutputQuality(Enum):
    """출력 품질 등급"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"

class IntegrationMethod(Enum):
    """통합 방법"""
    TRANSFORMER = "transformer"
    ATTENTION = "attention"
    ENSEMBLE = "ensemble"
    HYBRID = "hybrid"

class OutputFormat(Enum):
    """출력 형식"""
    IMAGE = "image"
    METADATA = "metadata"
    BOTH = "both"

@dataclass
class ModelConfig:
    """모델 설정"""
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    output_size: int = 64
    num_channels: int = 3

@dataclass
class IntegrationConfig:
    """통합 설정"""
    method: IntegrationMethod = IntegrationMethod.HYBRID
    enable_cross_modal: bool = True
    enable_quality_assessment: bool = True
    enable_output_refinement: bool = True
    modality_weights: Dict[str, float] = field(default_factory=lambda: {
        'image': 0.4,
        'text': 0.3,
        'metadata': 0.3
    })

@dataclass
class QualityConfig:
    """품질 설정"""
    min_quality_threshold: float = 0.7
    quality_assessment_method: str = "transformer"
    enable_auto_refinement: bool = True
    refinement_iterations: int = 3
    quality_metrics: List[str] = field(default_factory=lambda: [
        'psnr', 'ssim', 'lpips', 'fid'
    ])

@dataclass
class OutputConfig:
    """출력 설정"""
    format: OutputFormat = OutputFormat.BOTH
    enable_metadata: bool = True
    enable_confidence_score: bool = True
    enable_quality_report: bool = True
    output_resolution: tuple = (512, 512)
    compression_quality: int = 95

@dataclass
class PerformanceConfig:
    """성능 설정"""
    enable_mixed_precision: bool = True
    enable_gradient_checkpointing: bool = False
    batch_size: int = 1
    max_memory_usage: str = "8GB"
    enable_parallel_processing: bool = True

@dataclass
class FinalOutputConfig:
    """최종 출력 설정"""
    model: ModelConfig = field(default_factory=ModelConfig)
    integration: IntegrationConfig = field(default_factory=IntegrationConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # 시스템 설정
    enable_logging: bool = True
    log_level: str = "INFO"
    save_intermediate_results: bool = False
    enable_progress_tracking: bool = True

# 기본 설정
DEFAULT_CONFIG = FinalOutputConfig()

# 고품질 설정
HIGH_QUALITY_CONFIG = FinalOutputConfig(
    model=ModelConfig(
        d_model=768,
        num_layers=6,
        num_heads=12,
        d_ff=3072,
        dropout=0.05
    ),
    quality=QualityConfig(
        min_quality_threshold=0.8,
        refinement_iterations=5
    ),
    output=OutputConfig(
        output_resolution=(1024, 1024),
        compression_quality=100
    )
)

# 초고품질 설정
ULTRA_QUALITY_CONFIG = FinalOutputConfig(
    model=ModelConfig(
        d_model=1024,
        num_layers=8,
        num_heads=16,
        d_ff=4096,
        dropout=0.05
    ),
    quality=QualityConfig(
        min_quality_threshold=0.9,
        refinement_iterations=10
    ),
    output=OutputConfig(
        output_resolution=(2048, 2048),
        compression_quality=100
    ),
    performance=PerformanceConfig(
        enable_gradient_checkpointing=True,
        max_memory_usage="16GB"
    )
)
