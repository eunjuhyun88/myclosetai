"""
Virtual Fitting 설정 파일
가상 피팅을 위한 모든 설정과 상수를 정의합니다.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent
AI_MODELS_DIR = PROJECT_ROOT / "backend" / "ai_models" / "step_06_virtual_fitting"

class FittingQuality(Enum):
    """피팅 품질 레벨"""
    LOW = "low"
    BALANCED = "balanced"
    HIGH = "high"
    ULTRA = "ultra"

class FittingModel(Enum):
    """사용 가능한 피팅 모델"""
    HR_VITON = "hr_viton"
    OOTD = "ootd"
    VITON_HD = "viton_hd"
    HYBRID = "hybrid"

@dataclass
class ModelConfig:
    """개별 모델 설정"""
    name: str
    checkpoint_path: str
    model_type: str
    input_size: tuple
    output_size: tuple
    device: str = "auto"
    precision: str = "fp16"
    memory_optimization: bool = True

@dataclass
class FittingConfig:
    """피팅 설정"""
    quality_level: FittingQuality = FittingQuality.HIGH
    model_type: FittingModel = FittingModel.HYBRID
    enable_ensemble: bool = True
    enable_post_processing: bool = True
    enable_quality_assessment: bool = True
    
    # 해상도 설정
    input_resolution: tuple = (1024, 1024)
    output_resolution: tuple = (1024, 1024)
    
    # 품질별 설정
    quality_settings: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.quality_settings is None:
            self.quality_settings = {
                FittingQuality.LOW.value: {
                    "input_resolution": (512, 512),
                    "output_resolution": (512, 512),
                    "enable_ensemble": False,
                    "enable_post_processing": False
                },
                FittingQuality.BALANCED.value: {
                    "input_resolution": (768, 768),
                    "output_resolution": (768, 768),
                    "enable_ensemble": True,
                    "enable_post_processing": True
                },
                FittingQuality.HIGH.value: {
                    "input_resolution": (1024, 1024),
                    "output_resolution": (1024, 1024),
                    "enable_ensemble": True,
                    "enable_post_processing": True
                },
                FittingQuality.ULTRA.value: {
                    "input_resolution": (2048, 2048),
                    "output_resolution": (2048, 2048),
                    "enable_ensemble": True,
                    "enable_post_processing": True
                }
            }

# 모델 설정
MODEL_CONFIGS = {
    "hr_viton": ModelConfig(
        name="HR-VITON",
        checkpoint_path=str(AI_MODELS_DIR / "hrviton_final.pth"),
        model_type="hr_viton",
        input_size=(1024, 1024),
        output_size=(1024, 1024),
        precision="fp16"
    ),
    "ootd": ModelConfig(
        name="OOTD",
        checkpoint_path=str(AI_MODELS_DIR / "ootdiffusion"),
        model_type="diffusion",
        input_size=(1024, 1024),
        output_size=(1024, 1024),
        precision="fp16"
    ),
    "viton_hd": ModelConfig(
        name="VITON-HD",
        checkpoint_path=str(AI_MODELS_DIR / "viton_hd_2.1gb.pth"),
        model_type="viton_hd",
        input_size=(1024, 1024),
        output_size=(1024, 1024),
        precision="fp16"
    )
}

# 상수 정의
FITTING_CONSTANTS = {
    "MAX_IMAGE_SIZE": 4096,
    "MIN_IMAGE_SIZE": 256,
    "DEFAULT_BATCH_SIZE": 1,
    "MAX_BATCH_SIZE": 4,
    "DEFAULT_CONFIDENCE_THRESHOLD": 0.7,
    "MIN_CONFIDENCE_THRESHOLD": 0.3,
    "MAX_CONFIDENCE_THRESHOLD": 0.95,
    "DEFAULT_BLENDING_ALPHA": 0.8,
    "MIN_BLENDING_ALPHA": 0.5,
    "MAX_BLENDING_ALPHA": 1.0,
    "DEFAULT_WARPING_STRENGTH": 0.7,
    "MIN_WARPING_STRENGTH": 0.3,
    "MAX_WARPING_STRENGTH": 1.0
}

# 품질 평가 임계값
QUALITY_THRESHOLDS = {
    "excellent": 0.9,
    "good": 0.8,
    "fair": 0.7,
    "poor": 0.6,
    "unacceptable": 0.5
}

# 메모리 최적화 설정
MEMORY_OPTIMIZATION = {
    "enable_gradient_checkpointing": True,
    "enable_mixed_precision": True,
    "enable_memory_efficient_attention": True,
    "max_memory_usage": "80%",
    "enable_model_sharding": False
}

# 디버깅 설정
DEBUG_CONFIG = {
    "save_intermediate_results": False,
    "save_debug_images": False,
    "log_model_parameters": False,
    "profile_memory_usage": False,
    "enable_verbose_logging": False
}

def get_fitting_config(quality_level: str = "high", model_type: str = "hybrid") -> FittingConfig:
    """피팅 설정을 가져옵니다."""
    return FittingConfig(
        quality_level=FittingQuality(quality_level),
        model_type=FittingModel(model_type)
    )

def get_model_config(model_name: str) -> Optional[ModelConfig]:
    """모델 설정을 가져옵니다."""
    return MODEL_CONFIGS.get(model_name)

def validate_config(config: FittingConfig) -> bool:
    """설정 유효성을 검증합니다."""
    if not os.path.exists(AI_MODELS_DIR):
        return False
    
    # 모델 파일 존재 여부 확인
    for model_name, model_config in MODEL_CONFIGS.items():
        if not os.path.exists(model_config.checkpoint_path):
            return False
    
    return True
