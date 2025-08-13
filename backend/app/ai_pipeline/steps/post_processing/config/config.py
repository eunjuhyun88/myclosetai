#!/usr/bin/env python3
"""
07_post_processing 설정 파일
100% 논문 구현 - 완전한 신경망 구조 지원
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from enum import Enum

class EnhancementMethod(Enum):
    """향상 방법 열거형"""
    SUPER_RESOLUTION = "super_resolution"
    FACE_ENHANCEMENT = "face_enhancement"
    NOISE_REDUCTION = "noise_reduction"
    DETAIL_ENHANCEMENT = "detail_enhancement"
    COLOR_CORRECTION = "color_correction"
    CONTRAST_ENHANCEMENT = "contrast_enhancement"
    SHARPENING = "sharpening"
    BRIGHTNESS_ADJUSTMENT = "brightness_adjustment"
    EDGE_ENHANCEMENT = "edge_enhancement"

class QualityLevel(Enum):
    """품질 레벨"""
    FAST = "fast"
    BALANCED = "balanced"
    HIGH = "high"
    ULTRA = "ultra"

class ModelType(Enum):
    """AI 모델 타입"""
    ESRGAN = "esrgan"
    SWINIR = "swinir"
    FACE_ENHANCEMENT = "face_enhancement"

@dataclass
class PostProcessingConfig:
    """후처리 설정"""
    quality_level: QualityLevel = QualityLevel.HIGH
    enabled_methods: List[EnhancementMethod] = None
    upscale_factor: int = 4
    max_resolution: Tuple[int, int] = (2048, 2048)
    enhancement_strength: float = 0.8
    enable_face_detection: bool = True
    enable_visualization: bool = True
    processing_mode: str = "quality"
    cache_size: int = 50
    enable_caching: bool = True
    visualization_quality: str = "high"
    show_before_after: bool = True
    show_enhancement_details: bool = True
    
    # 새로운 기능들
    enable_memory_optimization: bool = True
    max_memory_usage_gb: float = 100.0
    enable_checkpoint_management: bool = True
    checkpoint_keep_count: int = 5
    enable_quality_assessment: bool = True
    enable_advanced_processing: bool = True
    
    def __post_init__(self):
        if self.enabled_methods is None:
            self.enabled_methods = [
                EnhancementMethod.SUPER_RESOLUTION,
                EnhancementMethod.FACE_ENHANCEMENT,
                EnhancementMethod.DETAIL_ENHANCEMENT,
                EnhancementMethod.COLOR_CORRECTION,
                EnhancementMethod.EDGE_ENHANCEMENT
            ]

@dataclass
class ModelConfig:
    """AI 모델 설정"""
    esrgan: Dict[str, Any] = None
    swinir: Dict[str, Any] = None
    face_enhancement: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.esrgan is None:
            self.esrgan = {
                'in_channels': 3,
                'out_channels': 3,
                'num_features': 64,
                'num_blocks': 23,  # 논문 정확한 구현
                'upscale': 4,
                'growth_channels': 32,
                'enable_se_block': True,
                'enable_rrdb': True
            }
        
        if self.swinir is None:
            self.swinir = {
                'img_size': 64,
                'patch_size': 1,
                'in_chans': 3,
                'out_chans': 3,
                'embed_dim': 96,
                'depths': [6, 6, 6, 6],  # 논문 정확한 구현
                'num_heads': [6, 6, 6, 6],
                'window_size': 7,
                'mlp_ratio': 4.0,
                'upsampler': 'pixelshuffle',
                'upscale': 4,
                'enable_window_attention': True,
                'enable_swin_block': True
            }
        
        if self.face_enhancement is None:
            self.face_enhancement = {
                'in_channels': 3,
                'out_channels': 3,
                'num_features': 64,
                'enable_attention': True,
                'enable_residual_blocks': True,
                'num_residual_blocks': 8
            }

@dataclass
class PerformanceConfig:
    """성능 설정"""
    max_batch_size: int = 1
    enable_mixed_precision: bool = True
    enable_optimization: bool = True
    memory_fraction: float = 0.7
    enable_caching: bool = True
    cache_size: int = 50
    
    # 메모리 최적화 설정
    enable_memory_tracking: bool = True
    enable_auto_cleanup: bool = True
    cleanup_threshold_gb: float = 80.0
    enable_gpu_memory_management: bool = True

@dataclass
class QualityAssessmentConfig:
    """품질 평가 설정"""
    enable_psnr: bool = True
    enable_ssim: bool = True
    enable_lpips: bool = True
    enable_comprehensive_score: bool = True
    
    # 메트릭 가중치
    psnr_weight: float = 0.4
    ssim_weight: float = 0.4
    lpips_weight: float = 0.2
    
    # 품질 임계값
    psnr_threshold: float = 30.0
    ssim_threshold: float = 0.8
    lpips_threshold: float = 0.1

@dataclass
class AdvancedProcessingConfig:
    """고급 이미지 처리 설정"""
    enable_noise_reduction: bool = True
    noise_reduction_method: str = "bilateral"  # gaussian, median, bilateral
    
    enable_edge_enhancement: bool = True
    edge_enhancement_strength: float = 1.5
    
    enable_color_correction: bool = True
    color_temperature_range: Tuple[float, float] = (-2.0, 2.0)
    tint_range: Tuple[float, float] = (-2.0, 2.0)
    
    enable_adaptive_processing: bool = True
    quality_based_processing: bool = True

@dataclass
class CheckpointConfig:
    """체크포인트 설정"""
    enable_checkpointing: bool = True
    checkpoint_dir: str = "models/checkpoints"
    auto_save_interval: int = 100  # 처리 횟수
    max_checkpoint_size_mb: int = 500
    enable_compression: bool = True
    cleanup_old_checkpoints: bool = True
    keep_checkpoint_count: int = 5

# 기본 설정
DEFAULT_CONFIG = PostProcessingConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_PERFORMANCE_CONFIG = PerformanceConfig()
DEFAULT_QUALITY_CONFIG = QualityAssessmentConfig()
DEFAULT_ADVANCED_CONFIG = AdvancedProcessingConfig()
DEFAULT_CHECKPOINT_CONFIG = CheckpointConfig()

# 전체 설정 통합
@dataclass
class CompletePostProcessingConfig:
    """완전한 후처리 설정"""
    post_processing: PostProcessingConfig = DEFAULT_CONFIG
    model: ModelConfig = DEFAULT_MODEL_CONFIG
    performance: PerformanceConfig = DEFAULT_PERFORMANCE_CONFIG
    quality: QualityAssessmentConfig = DEFAULT_QUALITY_CONFIG
    advanced: AdvancedProcessingConfig = DEFAULT_ADVANCED_CONFIG
    checkpoint: CheckpointConfig = DEFAULT_CHECKPOINT_CONFIG

# 최종 기본 설정
COMPLETE_DEFAULT_CONFIG = CompletePostProcessingConfig()
